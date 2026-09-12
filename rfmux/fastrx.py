"""fastrx: zero-copy consumers for the fastrxd channel-stream daemon.

A separate capture path from rfmux.streamer's socket-based receiver: this one
reads packets straight out of the NIC via AF_XDP, through a daemon
(fastrxd, run via "rfmux fastrxd") that a client attaches to over a Unix
socket. The compiled extension (_fastrx) is built alongside _receiver from
rfmux/streamer/ (see rfmux/streamer/CMakeLists.txt).

The API is synchronous; every blocking call releases the GIL.
"""

import os

from .streamer._fastrx import (
    ABI_VERSION,
    MAX_CHANNELS,
    MAX_CLIENTS,
    MAX_SAMPLES,
    NUM_MODULES,
    SOCKET_DIR,
    PacketFile,
    record_stride,
)
from .streamer._fastrx import PacketCapture as _PacketCapture
from .streamer._fastrx import PacketWriter as _PacketWriter

__all__ = [
    "PacketCapture",
    "PacketWriter",
    "PacketFile",
    "record_stride",
    "get_samples",
    "resolve_socket",
    "daemon_path",
    "running_interfaces",
    "start_command",
    "NUM_MODULES",
    "MAX_CHANNELS",
    "MAX_SAMPLES",
    "ABI_VERSION",
    "MAX_CLIENTS",
    "SOCKET_DIR",
]


def daemon_path() -> str:
    """The fastrxd binary: a sibling of the extension, wherever this
    install put it."""
    from .streamer import _fastrx
    return os.path.join(os.path.dirname(os.path.abspath(_fastrx.__file__)),
                        "fastrxd")


def running_interfaces() -> list[str]:
    """The interfaces a fastrxd is serving, one socket each in SOCKET_DIR."""
    try:
        return sorted(os.listdir(SOCKET_DIR))
    except OSError:
        return []


def start_command(interface: str) -> str:
    """The command that starts fastrxd on *interface*."""
    return f"sudo {daemon_path()} -i {interface}"


def resolve_socket(interface: str | None = None,
                   socket: str | None = None) -> str:
    """Turn an interface name (or nothing at all) into a fastrxd socket path.

    "socket" is for a daemon started with --socket-path, whose path cannot be
    derived from anything.
    """

    if interface is not None and socket is not None:
        raise ValueError("give either interface= or socket=, not both")
    if socket is not None:
        return socket

    if interface is None:
        # Nothing specified. One socket means there is no choice to make, so
        # requiring a name would be ceremony; two or more is a real ambiguity,
        # and picking one would silently attach to the wrong NIC -- which
        # presents as a transmitter that is not sending rather than a mistake.
        try:
            found = sorted(os.listdir(SOCKET_DIR))
        except OSError:
            found = []
        if len(found) != 1:
            raise ValueError(
                f"no fastrxd sockets in {SOCKET_DIR}; is fastrxd running?"
                if not found else
                "several fastrxd instances are running; name one with "
                f"interface=: {', '.join(found)}"
            )
        interface = found[0]

    if "/" in interface:
        raise ValueError(f"not an interface name: {interface!r}")

    return os.path.join(SOCKET_DIR, interface)


class PacketCapture(_PacketCapture):
    """One fastrxd client slot, with its own hot and cold thread.

    The extension takes a socket path; this adds the convenience of naming the
    interface instead, or of naming nothing when there is only one fastrxd.

        PacketCapture() # when only one fastrxd is running
        PacketCapture(interface="enp9s0f0np0")
        PacketCapture(socket="/tmp/fastrxd.sock")

    The channel count and module are chosen per capture() call:

        d = c.capture(1024, channels=16, module=2)   # d["i"] is (1024, 16)
    """

    def __init__(
        self,
        *,
        interface: str | None = None,
        socket: str | None = None,
    ):
        super().__init__(resolve_socket(interface, socket))


class PacketWriter(_PacketWriter):
    """Records the stream to disk (for readback using PacketFile).

        with PacketWriter("run.fastrx", channels=200, n_packets=1_000_000) as w:
            w.wait()
        # stop() (via __exit__) flushes, finalizes and closes.

    n_packets (defaults to None) bounds the recording.

    channels records the module's first that many channels of every packet,
    1..MAX_CHANNELS (None: all of them).  A pipe carries MAX_SAMPLES
    consecutive channels, so channels=200 records pipes 1 and 2: all of the
    first and 72 of the second.  It is fixed for the whole file, so every
    record has the same stride and PacketFile can return strided views
    instead of parsing; it is not inferred from what happens to be
    streaming.

    If the disk falls behind, records are dropped and counted in .overruns
    rather than ever blocking the packet path.

    A pipe the layout needs that a packet lacks (not streaming, or a
    transmitter reconfiguration mid-run) is zero-filled rather than
    dropped, counted in .dropouts; each record's pipe_snapshot
    (PacketFile.headers()) says which blocks are real.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        channels: int | None = None,
        n_packets: int | None = None,
        interface: str | None = None,
        socket: str | None = None,
        ring_mb: int = 256,
        queue_depth: int = 32,
    ):
        if channels is not None and not 1 <= channels <= MAX_CHANNELS:
            raise ValueError(
                f"channels must be in 1..{MAX_CHANNELS}, got {channels}")

        if n_packets is not None and n_packets <= 0:
            raise ValueError(
                f"n_packets must be positive (or None), got {n_packets}")

        super().__init__(
            resolve_socket(interface, socket),
            os.fspath(path),
            channels=channels,
            n_packets=n_packets or 0,
            ring_bytes=ring_mb << 20,
            queue_depth=queue_depth,
        )


def get_samples(n_packets: int, channels: int, module: int,
                timeout: float = 5.0, **kwargs):
    """Grab the next n_packets from one module.

    channels keeps the module's first that many channels (1..MAX_CHANNELS),
    so the returned arrays are (n_packets, channels).

    A short-lived PacketCapture, for callers who want one grab and no lifetime
    to manage.  Everything expensive is per-connection rather than per-packet,
    so code doing this repeatedly should keep a PacketCapture and call
    capture() on it instead:

        c = fastrx.PacketCapture(interface="enp9s0f0np0")
        while True:
            d = c.capture(1024, channels=MAX_CHANNELS, module=1)
    """

    if n_packets <= 0:
        raise ValueError(f"n_packets must be positive, got {n_packets}")
    with PacketCapture(**kwargs) as c:
        return c.capture(n_packets, channels, module, timeout)
