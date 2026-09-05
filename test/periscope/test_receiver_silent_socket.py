"""
A silent socket must not wedge Periscope's receive thread -- nor spin it.

``receive_batch``'s own ``timeout_ms`` cannot be relied on: recvmmsg runs
with MSG_WAITFORONE and the kernel only consults that timeout BETWEEN
datagrams, so on a blocking socket that receives nothing the call blocks
forever. The thread then never reaches its queue-discovery loop, and
Periscope draws nothing while reporting "0 packets received" with no
error — which is exactly what a user saw, and what took a while to
explain.

The rescue is SO_RCVTIMEO on a socket left blocking. ``settimeout`` is
not it: that makes the fd non-blocking, so an empty socket returns EAGAIN
in microseconds and the receive loop spins, retaking the GIL every call.
"""

import socket
import struct
import sys
import threading
import time

import pytest

import rfmux.streamer as streamer

linux_only = pytest.mark.skipif(
    sys.platform != "linux",
    reason="recvmmsg with MSG_WAITFORONE is the Linux path; elsewhere the "
           "receiver waits in select() with its own timeout, so a silent "
           "socket neither blocks nor needs the rescue")


def _one_receive(rx, timeout_s=8.0):
    """Seconds one receive_batch on *rx* took, or None if it never returned."""
    done = threading.Event()

    def call():
        try:
            rx.receive_batch(batch_size=2048, timeout_ms=50)
        except Exception:
            pass
        done.set()

    t0 = time.perf_counter()
    threading.Thread(target=call, daemon=True).start()
    if not done.wait(timeout=timeout_s):
        return None
    return time.perf_counter() - t0


def _receiver_on(sock):
    return streamer.ReadoutPacketReceiver(sock, reorder_window=256,
                                          queue_max_size=1000,
                                          flush_threshold=16)


def _silent_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    return sock


@linux_only
def test_receive_batch_alone_blocks_on_a_silent_socket():
    """Documents WHY the timeout is needed — not a bug to fix here.

    If this starts returning, recvmmsg's timeout semantics changed and
    the socket timeout could in principle go away.
    """
    sock = _silent_socket()
    try:
        assert _one_receive(_receiver_on(sock), timeout_s=3.0) is None, \
            "recvmmsg honoured timeout_ms on a silent socket — the " \
            "workaround in UDPReceiver may no longer be needed"
    finally:
        sock.close()


@linux_only
def test_receive_timeout_bounds_a_silent_receive():
    """SO_RCVTIMEO makes an empty receive wait the timeout, not spin or hang."""
    sock = _silent_socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO,
                    struct.pack("ll", 0, 500_000))
    try:
        took = _one_receive(_receiver_on(sock))
        assert took is not None, "SO_RCVTIMEO did not unblock receive_batch"
        assert 0.3 < took < 3.0, f"empty receive took {took:.4f} s"
    finally:
        sock.close()


@linux_only
def test_periscope_receiver_waits_out_a_silent_socket(monkeypatch):
    """The receiver Periscope actually builds must carry the timeout.

    Built on an ephemeral port: the real socket is UDP 9876, and a silent
    extra listener on the streaming port must never be left lying around
    in this suite.  The competing-receiver probe binds 9876 too, so it is
    stubbed out.
    """
    pytest.importorskip("PyQt6")
    from rfmux.tools.periscope.tasks import UDPReceiver

    monkeypatch.setattr(streamer, "get_multicast_socket",
                        lambda host, *a, **kw: _silent_socket())
    monkeypatch.setattr(streamer, "find_competing_receiver",
                        lambda host, *a, **kw: None)
    rx = UDPReceiver("127.0.0.1", 1)
    try:
        assert rx.sock.getsockname()[1] != streamer.STREAMER_PORT, \
            "the stub socket did not take: this test is listening on 9876"
        assert rx.sock.gettimeout() is None, \
            "settimeout() makes the fd non-blocking and the receive loop spin"
        took = _one_receive(rx.receiver)
        assert took is not None, \
            "UDPReceiver's socket has no receive timeout: a stream that " \
            "never arrives will wedge the thread and show 0 packets forever"
        assert 0.3 < took < 3.0, \
            f"empty receive took {took:.4f} s: the thread spins on a " \
            f"silent socket instead of waiting out the timeout"
    finally:
        rx.sock.close()
