"""
The batched readout getter equals the per-packet conversion.

PacketQueue.pop_readout_batch hands a consumer a drain's worth of
packets demuxed into arrays, so Periscope touches one object per frame
instead of one per packet.  Its output is held to what np.array(pkt)
and the per-packet timestamp arithmetic produce for the same packets,
sent through a real receiver on a loopback port.
"""

import socket
import time

import numpy as np
import pytest

from rfmux import streamer
from test.packet_helpers import readout_packet

SHORT = streamer.SHORT_PACKET_VERSION
LONG = streamer.LONG_PACKET_VERSION


def _packet(seq, version, rng, *, recent=True, fir_stage=0x8, t_s=43200.0):
    pkt = readout_packet(seq, version=version, module=2, serial=24,
                         fir_stage=fir_stage, t_s=t_s, recent=recent)
    pkt.raw_samples = rng.integers(-2**23, 2**23, size=2 * len(pkt),
                                   dtype=np.int32)
    return pkt


SENTINEL_SEQ = 1 << 20


def _receive(packets):
    """Send *packets* through a receiver on a loopback port; returns
    its queue once every packet is in.  The receiver's reorder stage
    holds the newest packet until a later one arrives, so a sentinel
    follows them; the tests drop it by its sequence number."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(0.2)
    port = sock.getsockname()[1]
    send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for pkt in packets:
        send.sendto(bytes(pkt), ("127.0.0.1", port))
    send.sendto(bytes(_packet(SENTINEL_SEQ, packets[-1].version,
                              np.random.default_rng(0))),
                ("127.0.0.1", port))
    send.close()
    receiver = streamer.ReadoutPacketReceiver(sock, reorder_window=1,
                                              queue_max_size=10000,
                                              flush_threshold=1)
    deadline = time.monotonic() + 3.0
    queue = None
    while time.monotonic() < deadline:
        receiver.receive_batch(batch_size=256, timeout_ms=50)
        for serial, module, q in receiver.get_all_queues():
            queue = q
        if queue is not None and queue.size() >= len(packets):
            break
    # The sentinel itself may still be held by the reorder stage.
    assert queue is not None and queue.size() >= len(packets), \
        f"{None if queue is None else queue.size()} of {len(packets)} arrived"
    return sock, receiver, queue


def _drain(queue, max_packets=1000):
    """Every batch, the sentinel dropped."""
    out = []
    while True:
        batch = queue.pop_readout_batch(max_packets)
        if batch is None:
            return out
        keep = batch[4] != SENTINEL_SEQ
        if keep.all():
            out.append(batch)
        elif keep.any():
            out.append(tuple(a[keep] for a in batch[:5]) + batch[5:])


def _per_packet(pkt):
    ts = pkt.ts
    seconds = (ts.h * 3600 + ts.m * 60 + ts.s + ts.ss / streamer.SS_PER_SECOND
               if ts.recent else np.nan)
    return np.array(pkt), seconds, bool(ts.recent), pkt.fir_stage, pkt.seq


@pytest.mark.parametrize("version", [SHORT, LONG])
def test_batch_equals_per_packet_conversion(version):
    rng = np.random.default_rng(3)
    packets = [_packet(k, version, rng, recent=(k % 7 != 3),
                       t_s=43200.0 + k * 2.6e-5) for k in range(300)]
    sock, receiver, queue = _receive(packets)
    try:
        batches = _drain(queue)
        assert len(batches) == 1
        samples, seconds, recent, fir, seq, year, yday = batches[0]
        assert samples.shape == (300, len(packets[0]))
        assert samples.dtype == np.complex128
        assert (year, yday) == (26, 245)
        for i, pkt in enumerate(packets):
            s, t, r, f, q = _per_packet(pkt)
            np.testing.assert_array_equal(samples[i], s)
            assert recent[i] == r and fir[i] == f and seq[i] == q
            if r:
                assert seconds[i] == t
            else:
                assert np.isnan(seconds[i])
    finally:
        sock.close()


def test_a_batch_holds_one_packet_width():
    """A batch stops short at a size change; the other width waits."""
    rng = np.random.default_rng(4)
    packets = ([_packet(k, SHORT, rng) for k in range(5)]
               + [_packet(5 + k, LONG, rng) for k in range(3)]
               + [_packet(8 + k, SHORT, rng) for k in range(2)])
    sock, receiver, queue = _receive(packets)
    try:
        widths = [(b[0].shape[0], b[0].shape[1]) for b in _drain(queue)]
        assert widths == [(5, 128), (3, 1024), (2, 128)]
    finally:
        sock.close()


def test_max_packets_bounds_a_batch():
    rng = np.random.default_rng(5)
    sock, receiver, queue = _receive([_packet(k, SHORT, rng) for k in range(10)])
    try:
        sizes = [b[0].shape[0] for b in _drain(queue, max_packets=4)]
        assert sizes == [4, 4, 2]           # the sentinel made a fourth, dropped
    finally:
        sock.close()
