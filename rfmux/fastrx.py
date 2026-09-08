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
    MAX_CLIENTS,
    MAX_SAMPLES,
    NUM_PIPELINES,
    SOCKET_DIR,
    PacketFile,
)
from .streamer._fastrx import PacketCapture as _PacketCapture
from .streamer._fastrx import PacketWriter as _PacketWriter

__all__ = [
    "PacketCapture",
    "PacketWriter",
    "PacketFile",
    "get_samples",
    "resolve_socket",
    "NUM_PIPELINES",
    "MAX_SAMPLES",
    "ABI_VERSION",
    "MAX_CLIENTS",
    "SOCKET_DIR",
]


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

    The pipeline is chosen per capture() call, not here.
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

        with PacketWriter("run.fastrx", pipes=[1], n_packets=1_000_000) as w:
            w.wait()
        # stop() (via __exit__) flushes, finalizes and closes.

    n_packets (defaults to None) bounds the recording.

    pipes selects which pipelines to record (1-indexed) and is required.

    If the disk falls behind, records are dropped and counted in .overruns
    rather than ever blocking the packet path.

    A recorded pipe that goes missing mid-stream (transmitter
    reconfiguration) is zero-filled rather than dropped, counted in
    .dropouts; each record's pipe_snapshot (PacketFile.headers()) says
    which blocks are real.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        pipes: list[int],
        n_packets: int | None = None,
        interface: str | None = None,
        socket: str | None = None,
        ring_mb: int = 256,
        queue_depth: int = 32,
    ):
        mask = 0
        pipes = list(pipes)
        if not pipes:
            raise ValueError("pipes must name at least one pipeline")
        for p in pipes:
            if not 1 <= p <= NUM_PIPELINES:
                raise ValueError(
                    f"pipe must be in 1..{NUM_PIPELINES}, got {p}")
            mask |= 1 << (p - 1)

        if n_packets is not None and n_packets <= 0:
            raise ValueError(
                f"n_packets must be positive (or None), got {n_packets}")

        super().__init__(
            resolve_socket(interface, socket),
            os.fspath(path),
            pipe_mask=mask,
            n_packets=n_packets or 0,
            ring_bytes=ring_mb << 20,
            queue_depth=queue_depth,
        )


def get_samples(n_packets: int, pipe: int = 1, timeout: float = 5.0, **kwargs):
    """Grab the next `n_packets` from one pipe, then tear everything down.

    A short-lived PacketCapture, for callers who want one grab and no lifetime
    to manage.  Everything expensive is per-connection rather than per-packet,
    so code doing this repeatedly should keep a PacketCapture and call
    capture() on it instead:

        c = fastrx.PacketCapture(interface="enp9s0f0np0")
        while True:
            d = c.capture(1024)
    """

    if n_packets <= 0:
        raise ValueError(f"n_packets must be positive, got {n_packets}")
    with PacketCapture(**kwargs) as c:
        return c.capture(n_packets, pipe, timeout)
