"""
PacketQueue's loss counters count packets that never arrived.

A packet that arrives past the reorder window is pushed out of order;
it fills a hole the counters already charged for, so it is not a loss.
"""

import socket
import time

from rfmux import streamer
from test.packet_helpers import pfb_datagram


def _stats_after(order):
    """Push PFB packets with sequence numbers *order*, one per
    receive_batch so the reorder stage (window 1) sees them in that
    order; returns the queue's stats."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver = streamer.PFBPacketReceiver(sock, reorder_window=1,
                                          queue_max_size=100,
                                          flush_threshold=1)
    try:
        for seq in order:
            send.sendto(pfb_datagram(0, seq=seq), ("127.0.0.1", port))
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if receiver.receive_batch(batch_size=256, timeout_ms=50):
                    break
            else:
                raise AssertionError(f"packet {seq} never arrived")
        receiver.flush_all()
        ((_serial, _module, queue),) = receiver.get_all_queues()
        assert queue.size() == len(order)
        return queue.get_stats()
    finally:
        send.close()
        sock.close()


def test_a_packet_past_the_reorder_window_is_not_a_loss():
    # Window 1 releases 6, 8 before 7 arrives: pushed 6, 8, 7, 9, 10.
    stats = _stats_after([6, 8, 9, 7, 10])
    assert stats.packets_missing == 0
    assert stats.sequence_gaps == 1


def test_a_packet_that_never_arrives_is_a_loss():
    stats = _stats_after([6, 8, 9, 10])
    assert stats.packets_missing == 1
    assert stats.sequence_gaps == 1
