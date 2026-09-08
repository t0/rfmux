"""
The batched PFB getter equals the per-packet conversion.

PacketQueue.pop_pfb_batch hands the fast source a drain's worth of
packets demuxed into one row per interleaved channel.  Its output is
held to what np.array(pkt) and the per-packet timestamp arithmetic
produce for the same packets, sent through a real receiver on a
loopback port.
"""

import socket
import time

import numpy as np
import pytest

from rfmux import streamer
from test.packet_helpers import pfb_packet, stamp

MODE = {1: 0, 2: 1, 4: 2}           # groups -> wire mode


def _packet(seq, slots, rng, *, num_samples=100, recent=True, t_s=43200.0):
    pkt = pfb_packet(1, t_s=t_s, seq=seq, slots=slots,
                     num_samples=num_samples)
    pkt.raw_samples = rng.integers(-2**23, 2**23, size=2 * num_samples,
                                   dtype=np.int32)
    if not recent:
        # A day no disciplined stamp carries, so a getter that took it
        # would show it.
        pkt.ts = stamp(t_s, y=1, d=1, recent=False)
    return pkt


def _receive(packets):
    """Send *packets* through a receiver on a loopback port; returns
    the socket, the receiver and its queue once every packet is in.
    Sent in small bursts, each received before the next, so no kernel
    buffer size is assumed.  The receiver owns the packet type the
    queued packets point at, so it must outlive the drain."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver = streamer.PFBPacketReceiver(sock, reorder_window=1,
                                          queue_max_size=10000,
                                          flush_threshold=1)
    queue = None
    deadline = time.monotonic() + 10.0
    for k in range(0, len(packets), 8):
        for pkt in packets[k:k + 8]:
            send.sendto(bytes(pkt), ("127.0.0.1", port))
        want = min(k + 8, len(packets)) - 1     # the newest is held back
        while time.monotonic() < deadline:
            receiver.receive_batch(batch_size=256, timeout_ms=50)
            for _serial, _module, q in receiver.get_all_queues():
                queue = q
            if queue is not None and queue.size() >= want:
                break
    send.close()
    receiver.flush_all()
    assert queue is not None and queue.size() == len(packets), \
        f"{None if queue is None else queue.size()} of {len(packets)} arrived"
    return sock, receiver, queue


def _drain(queue, max_packets=1000):
    out = []
    while (batch := queue.pop_pfb_batch(max_packets)) is not None:
        out.append(batch)
    return out


def _per_packet(pkt):
    ts = pkt.ts
    seconds = (ts.h * 3600 + ts.m * 60 + ts.s + ts.ss / streamer.SS_PER_SECOND
               if ts.recent else np.nan)
    return np.array(pkt), seconds, bool(ts.recent), pkt.seq


@pytest.mark.parametrize("slots", [(1,), (3, 7), (1, 2, 5, 9)])
def test_batch_equals_per_packet_conversion(slots):
    rng = np.random.default_rng(3)
    groups = len(slots)
    packets = [_packet(k, slots, rng, recent=(k % 7 != 3),
                       t_s=43200.0 + k * 4.1e-5) for k in range(60)]
    sock, receiver, queue = _receive(packets)
    try:
        batches = _drain(queue)
        assert len(batches) == 1
        (samples, seconds, recent, seq, mode, wire_slots, num_samples,
         _year, _yday) = batches[0]
        assert num_samples == 100
        time_samples = num_samples // groups
        assert samples.shape == (groups, 60 * time_samples)
        assert samples.dtype == np.complex128
        assert mode == MODE[groups]
        assert tuple(wire_slots[:groups]) == tuple(ch - 1 for ch in slots)
        for i, pkt in enumerate(packets):
            s, t, r, q = _per_packet(pkt)
            for g in range(groups):
                np.testing.assert_array_equal(
                    samples[g, i * time_samples:(i + 1) * time_samples],
                    s[g::groups])
            assert recent[i] == r and seq[i] == q
            if r:
                assert seconds[i] == t
            else:
                assert np.isnan(seconds[i])
    finally:
        sock.close()


def test_trailing_samples_short_of_a_group_are_dropped():
    """Ten samples over four channels: two per channel, the last two
    of the packet belong to no complete time sample."""
    rng = np.random.default_rng(6)
    packets = [_packet(k, (1, 2, 3, 4), rng, num_samples=10) for k in range(3)]
    sock, receiver, queue = _receive(packets)
    try:
        samples = _drain(queue)[0][0]
        assert samples.shape == (4, 3 * 2)
        for i, pkt in enumerate(packets):
            np.testing.assert_array_equal(samples[:, 2 * i:2 * i + 2],
                                          np.array(pkt)[:8].reshape(2, 4).T)
    finally:
        sock.close()


def test_day_comes_from_the_first_disciplined_stamp():
    rng = np.random.default_rng(7)
    packets = [_packet(k, (1,), rng, recent=(k >= 2)) for k in range(5)]
    sock, receiver, queue = _receive(packets)
    try:
        assert _drain(queue)[0][7:] == (26, 245)
    finally:
        sock.close()


def test_day_is_zero_without_a_disciplined_stamp():
    rng = np.random.default_rng(8)
    sock, receiver, queue = _receive([_packet(k, (1,), rng, recent=False)
                            for k in range(3)])
    try:
        assert _drain(queue)[0][7:] == (0, 0)
    finally:
        sock.close()


def test_a_batch_holds_one_layout():
    """A batch stops short where the mode, a slot or the sample count
    changes; the next layout waits for the next call."""
    rng = np.random.default_rng(4)
    packets = ([_packet(k, (1,), rng) for k in range(5)]
               + [_packet(5 + k, (1, 2), rng) for k in range(3)]
               + [_packet(8 + k, (2,), rng) for k in range(2)]
               + [_packet(10 + k, (1,), rng, num_samples=50) for k in range(2)])
    sock, receiver, queue = _receive(packets)
    try:
        layouts = [(mode, slots[0], num_samples, len(seq))
                   for _s, _t, _r, seq, mode, slots, num_samples, _y, _d
                   in _drain(queue)]
        assert layouts == [(0, 0, 100, 5), (1, 0, 100, 3),
                           (0, 1, 100, 2), (0, 0, 50, 2)]
    finally:
        sock.close()


def test_max_packets_bounds_a_batch():
    rng = np.random.default_rng(5)
    sock, receiver, queue = _receive([_packet(k, (1,), rng) for k in range(10)])
    try:
        assert [len(b[3]) for b in _drain(queue, max_packets=4)] == [4, 4, 2]
    finally:
        sock.close()


def test_an_empty_queue_pops_none():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        receiver = streamer.PFBPacketReceiver(sock, reorder_window=1)
        assert receiver.get_queue(serial=1, module=0).pop_pfb_batch() is None
    finally:
        sock.close()
