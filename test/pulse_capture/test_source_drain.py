"""The stream sources drain what the socket holds per event-loop wake,
so a stream's throughput follows its data rate rather than how often
the loop schedules it."""

import asyncio
import contextlib
import socket
import sys
import threading
import time

import numpy as np
import pytest

from rfmux import streamer
from rfmux.pulse_capture import sources as src
from rfmux.streamer import Timestamp, TimestampSource


def _slow_packet(seq, module=1, t_s=43200.0):
    pkt = streamer.ReadoutPacket(magic=streamer.STREAMER_MAGIC, version=5,
                                 serial=156, num_modules=1, flags=0,
                                 fir_stage=6, module=module - 1, seq=seq)
    pkt[:] = np.zeros(len(pkt), dtype=complex)
    h = int(t_s // 3600); m = int(t_s % 3600 // 60); s_ = int(t_s % 60)
    ss = int((t_s % 1) * streamer.SS_PER_SECOND)
    pkt.ts = Timestamp(y=26, d=244, h=h, m=m, s=s_, ss=ss, c=0, sbs=0,
                       source=TimestampSource.TEST, recent=True)
    return bytes(pkt)


@contextlib.contextmanager
def _loopback_pair():
    """(receiver, sender, port) on an OS-chosen loopback port."""
    recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
    recv.bind(("127.0.0.1", 0))
    port = recv.getsockname()[1]
    send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        yield recv, send, port
    finally:
        recv.close()
        send.close()


class _SlowSink:
    """What run_slow_source needs of a capture session."""

    def __init__(self):
        self.channels = [1]
        self.count = 0
        self.stamps = []

    def feed_block(self, ch, i_vals, q_vals, timestamps):
        self.count += len(i_vals)
        self.stamps.extend(np.asarray(timestamps).tolist())


def _patched_socket(monkeypatch, sock_by_port):
    @contextlib.contextmanager
    def fake(host, port=None, **kw):
        yield sock_by_port[port]
    monkeypatch.setattr(src.streamer, "get_multicast_socket", fake)
    # A quiet socket otherwise holds the source for the 60 s production
    # timeout after the senders finish.
    monkeypatch.setattr(src.streamer, "STREAMER_TIMEOUT", 0.3)
    # The receiver holds its newest reorder_window packets until more
    # arrive; a finite burst must not sit in it until the stop.
    monkeypatch.setattr(src, "_PFB_REORDER_WINDOW", 1)
    monkeypatch.setattr(src, "_PFB_FLUSH_EVERY", 1)


def test_slow_drain_is_lossless_and_ordered(monkeypatch):
    """Batching per wake feeds the packets sent, in order."""
    N = 400
    with _loopback_pair() as (recv, send, port):
        _patched_socket(monkeypatch, {streamer.STREAMER_PORT: recv})
        sink = _SlowSink()

        def pump():
            for k in range(N):
                send.sendto(_slow_packet(k, t_s=43200.0 + k / 596.0),
                            ("127.0.0.1", port))
                if k % 10 == 0:
                    # Small bursts: rmem_max commonly clamps the buffer
                    # to ~200 kB (~25 packets), and a kernel drop is not
                    # the drain's fault.
                    time.sleep(0.002)

        th = threading.Thread(target=pump)
        th.start()
        deadline = time.monotonic() + 5.0
        covered = asyncio.run(src.run_slow_source(
            sink, "127.0.0.1", module=1,
            should_stop=lambda: (sink.count >= N
                                 or time.monotonic() > deadline)))
        th.join()

    assert sink.count >= 0.98 * N, f"lost {N - sink.count} of {N} packets"
    stamps = np.array(sink.stamps)
    assert np.all(np.diff(stamps) > 0), "reordered within the drain"
    assert covered == pytest.approx((N - 1) / 596.0, rel=0.05)


def test_many_datagrams_per_wake(monkeypatch):
    """A backlog is drained per wake, not per awaited receive: with a
    preloaded backlog the awaited receives are far fewer than the
    packets processed."""
    # Small enough to sit in a socket buffer even when rmem_max clamps
    # the 16 MB request to ~200 kB.
    N = 20
    with _loopback_pair() as (recv, send, port):
        _patched_socket(monkeypatch, {streamer.STREAMER_PORT: recv})
        # run_slow_source flushes datagrams that predate the capture;
        # this test IS a preloaded backlog.
        monkeypatch.setattr(src, "_flush", lambda sock: None)
        for k in range(N):
            send.sendto(_slow_packet(k, t_s=43200.0 + k / 596.0),
                        ("127.0.0.1", port))
        time.sleep(0.1)              # let the kernel land them all

        awaited = {"n": 0}
        loop_cls = asyncio.new_event_loop().__class__
        orig = loop_cls.sock_recv

        async def counting_sock_recv(self, sock, n):
            awaited["n"] += 1
            return await orig(self, sock, n)

        monkeypatch.setattr(loop_cls, "sock_recv", counting_sock_recv)

        sink = _SlowSink()
        deadline = time.monotonic() + 3.0
        asyncio.run(src.run_slow_source(
            sink, "127.0.0.1", module=1,
            should_stop=lambda: (sink.count >= N
                                 or time.monotonic() > deadline)))

    assert sink.count == N, f"only {sink.count}/{N} packets arrived"
    # One awaited receive gets the first datagram; the drain takes the
    # rest synchronously.  One per wake would need ~N.
    assert awaited["n"] <= 3, \
        f"{awaited['n']} awaited receives for {N} packets"


def _pfb_packet(module0, t_s=43200.0, seq=0, slots=(1,), values=None):
    """A PFB packet whose slots carry *slots*' channels, interleaved;
    *values* gives one constant per slot."""
    pkt = streamer.PFBPacket()
    pkt.magic = streamer.PFB_PACKET_MAGIC
    pkt.module = module0                     # 0-indexed on the wire
    pkt.seq = seq
    pkt.mode = {1: 0, 2: 1, 4: 2}[len(slots)]
    for i, ch in enumerate(slots):
        setattr(pkt, f"slot{i + 1}", ch - 1)      # 0-indexed on the wire
    pkt.num_samples = 100
    data = np.zeros(100, dtype=complex)
    for i, v in enumerate(values or ()):
        data[i::len(slots)] = v
    pkt[:] = data
    h = int(t_s // 3600); m = int(t_s % 3600 // 60); s_ = int(t_s % 60)
    ss = int((t_s % 1) * streamer.SS_PER_SECOND)
    pkt.ts = Timestamp(y=26, d=244, h=h, m=m, s=s_, ss=ss, c=0, sbs=0,
                       source=TimestampSource.TEST, recent=True)
    return bytes(pkt)


@pytest.mark.filterwarnings("ignore:PFB socket buffer")
def test_pfb_source_keeps_only_its_module(monkeypatch):
    """Every module's PFB streamer shares the port; packets from another
    module are not this capture's samples."""
    with _loopback_pair() as (recv, send, port):
        _patched_socket(monkeypatch, {streamer.PFB_STREAMER_PORT: recv})
        monkeypatch.setattr(src, "_flush", lambda sock: None)
        for k in range(12):
            send.sendto(_pfb_packet(k % 2, t_s=43200.0 + k * 1e-4, seq=k),
                        ("127.0.0.1", port))
        time.sleep(0.1)

        class _Sink:
            channels = [1]
            fed = 0                           # samples

            def feed_block(self, ch, i, q, t):
                _Sink.fed += len(i)

        deadline = time.monotonic() + 3.0
        asyncio.run(src.run_pfb_source(
            _Sink(), "127.0.0.1", [1], module=2,
            should_stop=lambda: time.monotonic() > deadline or _Sink.fed >= 600))
    assert _Sink.fed == 600                   # six packets of 100


@pytest.mark.filterwarnings("ignore:PFB socket buffer")
def test_pfb_source_restores_sequence_order(monkeypatch):
    """Datagrams arrive slightly out of order; the samples are fed in
    sequence order, with per-sample times that never step back."""
    with _loopback_pair() as (recv, send, port):
        _patched_socket(monkeypatch, {streamer.PFB_STREAMER_PORT: recv})
        monkeypatch.setattr(src, "_flush", lambda sock: None)
        order = list(range(40))
        for i in range(1, 40, 4):            # swap neighbours, as seen on the wire
            order[i], order[i + 1] = order[i + 1], order[i]
        for k in order:
            send.sendto(_pfb_packet(1, t_s=43200.0 + k * 4.096e-4, seq=1000 + k),
                        ("127.0.0.1", port))
        time.sleep(0.1)

        stamps = []

        class _Sink:
            channels = [1]

            def feed_block(self, ch, i, q, t):
                stamps.extend(np.asarray(t).tolist())

        deadline = time.monotonic() + 3.0
        asyncio.run(src.run_pfb_source(
            _Sink(), "127.0.0.1", [1], module=2,
            should_stop=lambda: time.monotonic() > deadline or len(stamps) >= 4000))
    assert len(stamps) == 4000
    assert np.all(np.diff(stamps) > 0), "samples fed out of order"


def test_slow_ingest_feeds_blocks_in_time_order():
    fed = []
    ingest = src.SlowIngest(lambda ch, i, q, t: fed.append(np.asarray(t)),
                            max_packets=8, max_age_s=1e9)
    order = list(range(20))
    for i in range(1, 20, 4):
        order[i], order[i + 1] = order[i + 1], order[i]
    for k in order:
        ingest.add((1,), np.array([complex(k, 0)]), 43200.0 + k / 596.0)
    ingest.flush()
    stamps = np.concatenate(fed)
    assert np.array_equal(stamps, 43200.0 + np.arange(20) / 596.0)
    assert ingest.elapsed == pytest.approx(19 / 596.0)


class _Sink:
    def __init__(self):
        self.channels = [1]
        self.calls = []                       # (channel, I values)

    def feed_block(self, ch, i, q, t):
        self.calls.append((ch, np.asarray(i).copy(), np.asarray(t).copy()))


def _run_pfb(monkeypatch, blobs, channels, stop_after):
    with _loopback_pair() as (recv, send, port):
        _patched_socket(monkeypatch, {streamer.PFB_STREAMER_PORT: recv})
        monkeypatch.setattr(src, "_flush", lambda sock: None)
        for b in blobs:
            send.sendto(b, ("127.0.0.1", port))
        time.sleep(0.1)
        sink = _Sink()
        sink.channels = channels
        sink.source = {}
        deadline = time.monotonic() + 3.0
        fed = lambda: sum(len(c[1]) for c in sink.calls)
        asyncio.run(src.run_pfb_source(
            sink, "127.0.0.1", channels, module=2,
            should_stop=lambda: time.monotonic() > deadline or fed() >= stop_after))
    return sink


@pytest.mark.filterwarnings("ignore:PFB socket buffer")
def test_pfb_source_picks_its_channels_from_a_wider_stream(monkeypatch):
    """The streamer carries channels 3 and 7; the capture wants 7."""
    blobs = [_pfb_packet(1, t_s=43200.0 + k * 4.096e-4, seq=k,
                         slots=(3, 7), values=(3 + 0j, 7 + 0j)) for k in range(20)]
    sink = _run_pfb(monkeypatch, blobs, [7], stop_after=1000)
    assert {c[0] for c in sink.calls} == {7}
    assert all(np.all(c[1] == 7.0 / 256.0) for c in sink.calls)
    assert sum(len(c[1]) for c in sink.calls) == 20 * 50


@pytest.mark.filterwarnings("ignore:PFB socket buffer")
def test_pfb_source_feeds_whole_batches_in_order(monkeypatch):
    """Every packet's samples reach the engine, in stream order, in
    blocks that are whole packets."""
    blobs = [_pfb_packet(1, t_s=43200.0 + k * 4.096e-4, seq=k) for k in range(40)]
    sink = _run_pfb(monkeypatch, blobs, [1], stop_after=4000)
    assert sum(len(c[1]) for c in sink.calls) == 4000
    assert all(len(c[1]) % 100 == 0 for c in sink.calls)
    t = np.concatenate([c[2] for c in sink.calls])
    assert np.all(np.diff(t) > 0)


def test_pfb_source_counts_lost_packets(monkeypatch):
    """A hole in the sequence numbers is counted from the receiver's
    own statistics, missing packets individually."""
    seqs = [k for k in range(40) if not 10 <= k < 17]      # 7 never sent
    blobs = [_pfb_packet(1, t_s=43200.0 + k * 4.096e-4, seq=k) for k in seqs]
    sink = _run_pfb(monkeypatch, blobs, [1], stop_after=3300)
    assert sum(len(c[1]) for c in sink.calls) == 3300
    assert sink.source["lost_packets"] == 7


def test_pfb_buffer_seconds_counts_a_four_channel_stream():
    """The warning threshold is in seconds of stream: four channels at
    2.44 MHz, 1000 samples per 8056-byte packet."""
    import socket as _socket
    import sys as _sys
    from rfmux.pulse_capture import sources
    bytes_per_s = 4 * 2441406.25 / 1000 * streamer.PFB_PACKET_SIZE

    class Sock:
        def getsockopt(self, level, opt):
            held = int(bytes_per_s)         # exactly one second's worth
            return held * 2 if _sys.platform == "linux" else held
    assert abs(sources._pfb_buffer_seconds(Sock()) - 1.0) < 1e-6
    if _sys.platform == "linux":
        with open("/proc/sys/net/core/rmem_max") as f:
            assert sources._rcvbuf_request() == int(f.read())


@pytest.mark.filterwarnings("ignore:PFB socket buffer")
def test_pfb_source_discards_to_bound_its_lag(monkeypatch):
    """When the session says the fast stream is further behind the
    slow one than the bound, the source drops packets by their stamps
    until it is within half the bound, and counts them."""
    with _loopback_pair() as (recv, send, port):
        _patched_socket(monkeypatch, {streamer.PFB_STREAMER_PORT: recv})
        monkeypatch.setattr(src, "_flush", lambda sock: None)
        step = 4.096e-5
        for k in range(200):
            send.sendto(_pfb_packet(1, t_s=43200.0 + k * step, seq=k),
                        ("127.0.0.1", port))
        time.sleep(0.1)
        calls = []

        def lag():
            # Behind by 1 ms once the first batch is in; fine after.
            calls.append(1)
            return 1e-3 if len(calls) == 1 else 0.0

        class _Sink:
            channels = [1]
            source = {}
            stream_lag = staticmethod(lag)
            fed = 0

            def feed_block(self, ch, i, q, t):
                _Sink.fed += len(i)

        deadline = time.monotonic() + 2.0
        monkeypatch.setattr(src, "_PFB_POP_MAX", 20)   # pops small enough to see
        asyncio.run(src.run_pfb_source(
            _Sink(), "127.0.0.1", [1], module=2, max_lag_s=0.5e-3,
            should_stop=lambda: time.monotonic() > deadline))
    flushed = _Sink.source["flushed_packets"]
    # 1 ms behind with a 0.5 ms bound: drop until within 0.25 ms, which
    # is 0.75 ms of packets at one per 41 us.
    assert 15 <= flushed <= 22, flushed
    assert _Sink.fed == (200 - flushed) * 100
