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
import socket as _socket

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

#: Sample rate of ONE streamed PFB channel, in Hz.
#:
#: Not the same as core.transferfunctions.PFB_SAMPLING_FREQ, which is the
#: internal PFB bin rate -- this is half of it, and it is the number that
#: matters to anything consuming the fast stream.
PFB_SAMPLE_RATE = 625e6 / 512  # ~1.2207 MHz

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

	This rule had three copies with two behaviours: py_get_samples and
	trigger_capture mapped it, py_run_pfb_streamer did not.  It lives
	here now, next to ``get_multicast_socket``, which is the only thing
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

	probe = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
	try:
		probe.bind(("", port))
	except OSError as exc:
		return (f"another process is already receiving on port {port} "
		        f"({exc.strerror or exc})")
	finally:
		probe.close()

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

	# Timestamp helpers
	'ts_to_seconds',

	# Constants
	'MULTICAST_GROUP',
	'READOUT_PACKET_MAGIC',
	'PFB_PACKET_MAGIC',
	'STREAMER_PORT',
	'PFB_STREAMER_PORT',
	'LONG_PACKET_SIZE',
	'SHORT_PACKET_SIZE',
	'LONG_PACKET_CHANNELS',
	'SHORT_PACKET_CHANNELS',
	'LONG_PACKET_VERSION',
	'SHORT_PACKET_VERSION',
	'PFBPACKET_NSAMP_MAX',
	'SS_PER_SECOND',
	'PFB_SAMPLE_RATE',

	# Backwards compatibility
	'DfmuxPacket',
	'STREAMER_MAGIC',
	'STREAMER_HOST',
	'STREAMER_TIMEOUT',
]

__version__ = '1.0.0'
