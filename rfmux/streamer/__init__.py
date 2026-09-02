"""
CRS Streaming Protocol

Unified API for CRS packet streaming, including:
- Packet structures (ReadoutPacket, PFBPacket, Timestamp)
- High-performance C++ packet receiver
- Socket utilities for multicast configuration
- Protocol constants
"""

# Aliased: `from .socket import ...` below rebinds the name `socket` in this
# namespace to the rfmux submodule, shadowing the stdlib one.
import datetime as _datetime
import socket as _socket
import time as _time
from typing import NamedTuple as _NamedTuple

# Import C++ packet receiver and structures, and ensure version parity
_PY_API_VERSION = 1  # must match _SO_API_VERSION in bindings.cpp
try:
    from ._receiver import _SO_API_VERSION
except ModuleNotFoundError as e:
    import textwrap
    raise ModuleNotFoundError(textwrap.dedent(
        '''
        rfmux recently integrated a c++ extension for faster packet processing.
        This extension requires a compile/install step that was not previously
        necessary.

        Try `pip install -e .` from the repository root, or see README.md for
        details.
        ''')) from e
except ImportError:
    _SO_API_VERSION = 0

if _SO_API_VERSION != _PY_API_VERSION:
    import textwrap
    raise ImportError(textwrap.dedent(
    f'''
    C++ fastpath: API version mismatch; {_SO_API_VERSION=}, {_PY_API_VERSION=}

    You probably need to recompile the _receiver extension. You can do this
    with something like

        pip install -e . --force-reinstall

    from the repository root. _receiver is a fairly recent addition to rfmux;
    please see README.md for details.
    '''))

from ._receiver import (
	# Packet classes
	ReadoutPacket,
	PFBPacket,
	Timestamp,
	TimestampSource,
	Packet,

	# Receivers
	ReadoutPacketReceiver,
	PFBPacketReceiver,
	PacketReceiver,

	# Queues and stats
	PacketQueue,
	PacketQueueStats,
	PacketReceiverStats,

	# Socket utilities (from C++)
	ip_mreq_source,

	# Constants
	MULTICAST_GROUP,
	READOUT_PACKET_MAGIC,
	PFB_PACKET_MAGIC,
	STREAMER_PORT,
	PFB_STREAMER_PORT,
	PFB_PACKET_SIZE,
	LONG_PACKET_SIZE,
	SHORT_PACKET_SIZE,
	LONG_PACKET_CHANNELS,
	SHORT_PACKET_CHANNELS,
	LONG_PACKET_VERSION,
	SHORT_PACKET_VERSION,
	PFBPACKET_NSAMP_MAX,
	SS_PER_SECOND,
)

# Import socket utilities
from .socket import (
	get_multicast_socket,
	get_local_ip,
)


def resolve_host(hostname):
	"""Map a CRS hostname to the address its packets actually arrive on.

	A mock CRS created from a serial number alone gets the synthesized
	hostname ``rfmux0000.local`` (see ``CRS.tuber_hostname``), which
	resolves nowhere — the mock streamer sends to loopback.  Callers
	that opened a socket on the unmapped name simply received nothing.

	It lives next to ``get_multicast_socket``, which is the only thing
	the answer is ever used for.
	"""
	if not hostname:
		return hostname
	# get_multicast_socket strips the port itself; do it here too so the
	# comparison sees a bare hostname.
	host = hostname.split(":")[0] if ":" in hostname else hostname
	if host == "rfmux0000.local":
		return "127.0.0.1"
	return host


def ts_to_seconds(ts):
	"""Convert a packet Timestamp to seconds-of-day, or None if not recent.

	``recent`` is the firmware's own flag for "this timestamp has been
	disciplined"; an undisciplined one is not a small error but a
	meaningless number, so callers get None and must decide rather than
	silently accumulating garbage into a time axis.

	Note this wraps at midnight — consumers accumulating elapsed time
	should clamp per-packet deltas rather than subtracting endpoints.
	"""
	if not ts.recent:
		return None
	return ts.h * 3600 + ts.m * 60 + ts.s + ts.ss / SS_PER_SECOND


def ts_day_epoch(ts):
	"""Seconds since 1970 of the UTC midnight of the day a packet
	Timestamp names (two-digit year, day of year), or None if not recent.

	Packet clocks are UTC when disciplined; the test source counts from
	whatever the mock was started with, which is still a valid date.
	"""
	if not ts.recent:
		return None
	day = _datetime.datetime(2000 + int(ts.y), 1, 1,
	                         tzinfo=_datetime.timezone.utc)
	day += _datetime.timedelta(days=int(ts.d) - 1)
	return day.timestamp()


def epoch_to_utc(epoch):
	"""ISO 8601 with microseconds, e.g. 2026-09-02T16:14:05.123456Z."""
	dt = _datetime.datetime.fromtimestamp(float(epoch), tz=_datetime.timezone.utc)
	return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


class MulticastCheck(_NamedTuple):
	"""Whether multicast works here, and which step failed if not."""

	ok: bool
	steps: list
	hint: str

	def report(self) -> str:
		"""A few lines naming each step and what to do about a failure."""
		lines = []
		for name, ok, detail in self.steps:
			mark = "ok  " if ok else "FAIL"
			lines.append(f"  [{mark}] {name}: {detail}")
		if self.hint:
			lines.append("")
			lines.append(self.hint)
		return "\n".join(lines)


_MULTICAST_PROBE = b"rfmux-multicast-probe"


def check_multicast_loopback(*, group: str | None = None,
                             timeout: float = 0.5) -> MulticastCheck:
	"""Can a multicast packet reach a listener on this machine?

	Real hardware always multicasts, so the mock does too — what you
	debug in simulation is then the transport you run in the lab.
	Multicast is also the part people most often have trouble with: an
	interface that does not advertise MULTICAST, a container with no
	route for the group, a firewall. So when it does not work this says
	which step failed rather than only that it did.

	Checked in the order a packet meets them: aim the sender at
	loopback, confirm that aim actually took (if it silently did not,
	packets leave on the machine's real NIC, onto whatever network the
	lab runs on), join the group as a receiver, then send and receive.

	Uses an ephemeral port, so it neither disturbs nor consumes a live
	stream on the streamer port.
	"""
	group = group or MULTICAST_GROUP
	steps = []
	sender = None
	receiver = None

	# An ephemeral port, so a running stream on 9876 is untouched.
	scratch = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
	scratch.bind(("", 0))
	port = scratch.getsockname()[1]
	scratch.close()

	try:
		sender = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
		sender.setsockopt(_socket.IPPROTO_IP, _socket.IP_MULTICAST_LOOP, 1)
		# TTL 0 means "never leaves this host". The mock must not put
		# simulated detectors on the group real boards stream to.
		sender.setsockopt(_socket.IPPROTO_IP, _socket.IP_MULTICAST_TTL, 0)
		try:
			sender.setsockopt(_socket.IPPROTO_IP, _socket.IP_MULTICAST_IF,
			                  _socket.inet_aton("127.0.0.1"))
		except OSError as exc:
			steps.append(("interface", False,
			              f"could not point multicast at loopback ({exc})"))
			return MulticastCheck(False, steps, _MULTICAST_HINT.format(
				group=group))

		try:
			raw = sender.getsockopt(_socket.IPPROTO_IP,
			                        _socket.IP_MULTICAST_IF, 4)
			chosen = _socket.inet_ntoa(raw)
		except OSError:
			chosen = None   # not readable everywhere; not a failure

		if chosen is None:
			steps.append(("interface", True,
			              "set to loopback (not verifiable on this platform)"))
		elif chosen in ("127.0.0.1", "0.0.0.0"):
			# 0.0.0.0 means "let the kernel route it", which for this
			# group usually means the real NIC -- worth saying plainly.
			steps.append(("interface", chosen == "127.0.0.1",
			              f"multicast interface reads back as {chosen}"))
			if chosen != "127.0.0.1":
				return MulticastCheck(False, steps,
				                      _MULTICAST_ESCAPE_HINT.format(group=group))
		else:
			steps.append(("interface", False,
			              f"multicast interface reads back as {chosen}, "
			              f"not loopback"))
			return MulticastCheck(False, steps,
			                      _MULTICAST_ESCAPE_HINT.format(group=group))

		try:
			receiver = get_multicast_socket("127.0.0.1", port=port)
			steps.append(("join", True, f"joined {group} on loopback"))
		except OSError as exc:
			steps.append(("join", False,
			              f"could not join {group} on loopback "
			              f"({exc.strerror or exc})"))
			return MulticastCheck(False, steps,
			                      _MULTICAST_HINT.format(group=group))

		try:
			for _ in range(3):
				sender.sendto(_MULTICAST_PROBE, (group, port))
		except OSError as exc:
			steps.append(("send", False,
			              f"sending to {group} failed ({exc.strerror or exc})"))
			return MulticastCheck(False, steps,
			                      _MULTICAST_HINT.format(group=group))
		steps.append(("send", True, f"sent to {group}:{port}"))

		receiver.settimeout(timeout)
		deadline = _time.monotonic() + timeout
		while _time.monotonic() < deadline:
			try:
				data = receiver.recv(4096)
			except (TimeoutError, OSError):
				break
			if data == _MULTICAST_PROBE:
				steps.append(("receive", True, "probe came back"))
				return MulticastCheck(True, steps, "")
		steps.append(("receive", False,
		              f"nothing came back within {timeout:.1f}s"))
		return MulticastCheck(False, steps, _MULTICAST_HINT.format(group=group))
	finally:
		for sock in (sender, receiver):
			if sock is not None:
				try:
					sock.close()
				except OSError:
					pass


_MULTICAST_HINT = """\
  Multicast is not reaching a listener on this machine. Usual causes,
  in the order worth checking:

    * the loopback interface does not advertise MULTICAST
        ip link show lo            (look for MULTICAST in the flags)
        sudo ip link set lo multicast on
    * no route for the group, common in containers and VMs
        ip route get {group}
        sudo ip route add {group}/32 dev lo
    * a firewall dropping it
        sudo iptables -L -n | grep -i {group}
    * a network namespace or sandbox without multicast at all

  Real hardware always multicasts, so this is worth fixing before
  trusting a board on this machine -- the mock can work around it, a
  CRS cannot."""


_MULTICAST_ESCAPE_HINT = """\
  The multicast interface is NOT loopback, so mock packets would leave
  this machine on its real network -- onto the same group {group} that
  real CRS boards stream to. Refusing to multicast for that reason.

  Check which interface the kernel picks:
        ip route get {group}"""


def _is_loopback(host: str) -> bool:
	"""True if *host* names this machine, port suffix and all."""
	if not host:
		return False
	name = host.split(':')[0] if ':' in host else host
	return name in ("127.0.0.1", "localhost", "::1")


def _port_has_receiver(port: int) -> str | None:
	"""Why a plain bind to *port* failed, or None if nobody is receiving.

	No socket options on purpose: the streamer's readers set
	``SO_REUSEPORT``, so a probe that also set it would bind happily
	alongside them and report the port free -- exactly the condition
	being looked for.
	"""
	probe = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
	try:
		probe.bind(("", port))
	except OSError as exc:
		return exc.strerror or str(exc)
	finally:
		probe.close()
	return None


def find_competing_receiver(host: str = "127.0.0.1", *,
                            port: int | None = None) -> str | None:
	"""Another receiver that would take our packets, or None.

	Reported only where it actually costs data, which is the loopback
	(mock) case, because the two transports behave differently:

	The mock sends UNICAST to 127.0.0.1 -- ``sendto((self.host, port))``
	in rfmux/mock/udp_streamer.py, whatever its "(multicast)" log line
	says. The kernel hands each unicast datagram to exactly ONE of the
	sockets bound with ``SO_REUSEPORT``, picked by hash of the sender's
	4-tuple. Measured over 20 sender ports: 6 to the first listener, 14
	to the second, never once split. So a second receiver does not share
	the stream, it takes all of it, and the loser sees nothing at all --
	no packets, no loss, no error.

	A real board MULTICASTS to the group, and every socket joined to it
	gets its own copy: measured 200 of 200 to both of two listeners.
	Multiple readers are fine and expected there, so this returns None
	for a non-loopback host rather than crying wolf on every launch.
	"""
	if not _is_loopback(host):
		return None
	if port is None:
		port = STREAMER_PORT
	why = _port_has_receiver(port)
	if why is None:
		return None
	return (f"Another process is already receiving on port {port} ({why}). "
	        f"The mock stream is unicast, so the kernel gives every packet "
	        f"to just one receiver -- this one may see nothing.")


def find_streamer_conflict(host: str = "127.0.0.1", *,
                           port: int | None = None,
                           timeout: float = 0.05) -> str | None:
	"""Describe whatever is already using the streamer port, or None if free.

	For callers about to stand up a *second* source of packets — chiefly
	``create_mock_crs()`` in a demo — where doing so silently corrupts the
	data. Mock streamers send to a fixed port, so two simulations reach one
	receiver interleaved: no exception, no dropped connection, just samples
	from two unrelated detectors in one trace.

	Two probes, because neither sees everything:

	1. A PLAIN bind, no socket options, which fails if anyone is RECEIVING.
	   The streamer's readers set ``SO_REUSEPORT``, so a probe that also set
	   it would bind happily alongside them and report the port free — which
	   is exactly the condition being looked for. Costs microseconds and
	   consumes nothing.
	2. A short read, which sees anyone SENDING even with no receiver
	   attached — a mock server outliving the kernel that made it, say. This
	   one does consume a datagram; at the default 596 Hz that is one packet
	   in six hundred.

	``timeout`` bounds only the second probe. Packets arrive every ~1.9 ms at
	the slowest decimation, so the 50 ms default has ~26x of margin.

	Checks the slow readout port, which every stream uses; pass ``port`` for
	the PFB one, which is only busy when PFB streaming is switched on.

	This is advice about the machine, not proof: a real board multicasting to
	this host trips probe 2 as well. That is still the right answer for the
	question being asked, since adding a simulation alongside real data
	interleaves it just the same — but say "something is streaming", not
	"Periscope is running".
	"""
	if port is None:
		port = STREAMER_PORT

	why = _port_has_receiver(port)
	if why is not None:
		return f"another process is already receiving on port {port} ({why})"

	try:
		with get_multicast_socket(host, port=port) as sock:
			sock.settimeout(timeout)
			try:
				sock.recv(65535)
			except (_socket.timeout, TimeoutError):
				return None
			return f"packets are already arriving on port {port}"
	except OSError as exc:
		return f"port {port} could not be probed ({exc.strerror or exc})"


# Backwards compatibility aliases
DfmuxPacket = ReadoutPacket
STREAMER_MAGIC = READOUT_PACKET_MAGIC
STREAMER_HOST = MULTICAST_GROUP
STREAMER_TIMEOUT = 60  # seconds

__all__ = [
	# Packet classes
	'ReadoutPacket',
	'PFBPacket',
	'Timestamp',
	'TimestampSource',
	'Packet',

	# Receivers
	'ReadoutPacketReceiver',
	'PFBPacketReceiver',
	'PacketReceiver',

	# Queues
	'PacketQueue',
	'PacketQueueStats',
	'PacketReceiverStats',

	# Socket utilities
	'get_multicast_socket',
	'get_local_ip',
	'ip_mreq_source',
	'find_streamer_conflict',
	'find_competing_receiver',
	'check_multicast_loopback',
	'MulticastCheck',

	# Timestamp helpers
	'ts_to_seconds',

	# Constants
	'MULTICAST_GROUP',
	'READOUT_PACKET_MAGIC',
	'PFB_PACKET_MAGIC',
	'STREAMER_PORT',
	'PFB_STREAMER_PORT',
	'ts_day_epoch',
	'epoch_to_utc',
	'LONG_PACKET_SIZE',
	'SHORT_PACKET_SIZE',
	'LONG_PACKET_CHANNELS',
	'SHORT_PACKET_CHANNELS',
	'LONG_PACKET_VERSION',
	'SHORT_PACKET_VERSION',
	'PFBPACKET_NSAMP_MAX',
	'SS_PER_SECOND',

	# Backwards compatibility
	'DfmuxPacket',
	'STREAMER_MAGIC',
	'STREAMER_HOST',
	'STREAMER_TIMEOUT',
]

__version__ = '1.0.0'
