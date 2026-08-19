"""fastrx: zero-copy consumer for the fastrxd channel-stream daemon.

A separate capture path from rfmux.streamer's socket-based receiver: this one
reads packets straight out of the NIC via AF_XDP, through a daemon
(fastrxd, run via "rfmux fastrx run") that a client attaches to over a Unix
socket. The compiled extension (_fastrx) is built alongside _receiver from
rfmux/streamer/ (see rfmux/streamer/CMakeLists.txt).
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
from .streamer._fastrx import Consumer as _Consumer

__all__ = [
    "Consumer",
    "PacketFile",
    "get_samples",
    "NUM_PIPELINES",
    "MAX_SAMPLES",
    "ABI_VERSION",
    "MAX_CLIENTS",
    "SOCKET_DIR",
]


class Consumer(_Consumer):
    """One fastrxd client slot, with its own hot and cold thread.

    The extension takes a socket path; this adds the convenience of naming the
    interface instead, or of naming nothing when there is only one fastrxd.

        Consumer(pipe=1) # when only one fastrxd is running
        Consumer(pipe=1, interface="enp9s0f0np0")
        Consumer(pipe=1, socket="/tmp/fastrxd.sock")

    "socket" is for a daemon started with --socket-path, whose path cannot be
    derived from anything.
    """

    @staticmethod
    def resolve_socket(interface: str | None = None,
                       socket: str | None = None) -> str:

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

    def __init__(
        self,
        pipe: int = 1,
        *,
        interface: str | None = None,
        socket: str | None = None,
    ):
        super().__init__(self.resolve_socket(interface, socket))


def get_samples(n_packets: int, pipe: int = 1, timeout: float = 5.0, **kwargs):
    """Grab the next `n_packets` from one pipe, then tear everything down.

    A short-lived Consumer, for callers who want one grab and no lifetime to
    manage.  Everything expensive is per-connection rather than per-packet, so
    code doing this repeatedly should keep a Consumer and call capture() on it
    instead:

        c = fastrx.Consumer(interface="enp9s0f0np0")
        while True:
            d = c.capture(1024)
    """

    if n_packets <= 0:
        raise ValueError(f"n_packets must be positive, got {n_packets}")
    with Consumer(**kwargs) as c:
        return c.capture(n_packets, pipe, timeout)
