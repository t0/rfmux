"""
A silent socket must not wedge Periscope's receive thread.

``receive_batch``'s own ``timeout_ms`` cannot be relied on: recvmmsg runs
with MSG_WAITFORONE and the kernel only consults that timeout BETWEEN
datagrams, so on a socket that receives nothing the call blocks forever.
The thread then never reaches its queue-discovery loop, and Periscope
draws nothing while reporting "0 packets received" with no error — which
is exactly what a user saw, and what took a while to explain.
"""

import socket
import sys
import threading
import time

import pytest

import rfmux.streamer as streamer


def _blocking_call_returns(sock, timeout_s=8.0):
    """True if one receive_batch on *sock* returns within timeout_s."""
    rx = streamer.ReadoutPacketReceiver(sock, reorder_window=256,
                                        queue_max_size=1000,
                                        flush_threshold=16)
    done = threading.Event()

    def call():
        try:
            rx.receive_batch(batch_size=2048, timeout_ms=50)
        except Exception:
            pass
        done.set()

    threading.Thread(target=call, daemon=True).start()
    return done.wait(timeout=timeout_s)


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="recvmmsg with MSG_WAITFORONE is the Linux path; elsewhere the "
           "call returns on its own and there is no blocking to document")
def test_receive_batch_alone_blocks_on_a_silent_socket():
    """Documents WHY the timeout below is needed — not a bug to fix here.

    Linux only. The receiver reaches recvmmsg there, and the kernel
    consults timeout_ms only BETWEEN datagrams, so a silent socket
    blocks forever. macOS and Windows return without help, which is why
    they skip rather than assert the opposite: the socket timeout is
    still set on every platform, it simply has nothing to rescue.

    If this starts returning ON LINUX, recvmmsg's timeout semantics
    changed and the socket timeout could in principle go away.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    try:
        assert not _blocking_call_returns(sock, timeout_s=3.0), \
            "recvmmsg honoured timeout_ms on a silent socket — the " \
            "workaround in UDPReceiver may no longer be needed"
    finally:
        sock.close()


def test_socket_timeout_lets_the_call_return():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(0.5)
    try:
        assert _blocking_call_returns(sock), \
            "a socket timeout did not unblock receive_batch"
    finally:
        sock.close()


def test_periscope_receiver_sets_a_socket_timeout():
    """The receiver Periscope actually builds must carry the timeout."""
    pytest.importorskip("PyQt6")
    from rfmux.tools.periscope.tasks import UDPReceiver

    rx = UDPReceiver("127.0.0.1", 1)
    try:
        assert rx.sock.gettimeout() is not None, \
            "UDPReceiver's socket has no timeout: a stream that never " \
            "arrives will wedge the thread and show 0 packets forever"
        assert 0 < rx.sock.gettimeout() <= 2.0
    finally:
        rx.sock.close()
