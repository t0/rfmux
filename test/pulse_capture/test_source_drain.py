"""The stream sources drain what the socket holds per event-loop wake,
so a stream's throughput follows its data rate rather than how often
the loop schedules it."""

import asyncio
import contextlib
import math
import socket
import sys
import threading
import time

import numpy as np
import pytest

from rfmux import streamer
from rfmux.pulse_capture import sources as src
from test.packet_helpers import pfb_datagram, readout_packet


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


class _Sink:
    """What a source needs of a capture session, and what it fed."""

    def __init__(self, channels=(1,)):
        self.channels = list(channels)
        self.source = {}
        self.calls = []                       # (channel, I values, stamps)
        self.count = 0
        self.stamps = []

    def feed_block(self, ch, i, q, t):
        self.calls.append((ch, np.asarray(i).copy(), np.asarray(t).copy()))
        self.count += len(i)
        self.stamps.extend(np.asarray(t).tolist())


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
        monkeypatch.setattr(src, "_flush", lambda sock: None)
        sink = _Sink()

        def pump():
            for k in range(N):
                send.sendto(bytes(readout_packet(k, t_s=43200.0 + k / 596.0)),
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

    assert sink.count == N, f"lost {N - sink.count} of {N} packets"
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
            send.sendto(bytes(readout_packet(k, t_s=43200.0 + k / 596.0)),
                        ("127.0.0.1", port))
        time.sleep(0.1)              # let the kernel land them all

        awaited = {"n": 0}
        loop_cls = asyncio.SelectorEventLoop
        orig = loop_cls.sock_recv

        async def counting_sock_recv(self, sock, n):
            awaited["n"] += 1
            return await orig(self, sock, n)

        monkeypatch.setattr(loop_cls, "sock_recv", counting_sock_recv)

        sink = _Sink()
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


def test_pfb_source_keeps_only_its_module(monkeypatch):
    """Every module's PFB streamer shares the port; packets from another
    module are not this capture's samples."""
    with _loopback_pair() as (recv, send, port):
        _patched_socket(monkeypatch, {streamer.PFB_STREAMER_PORT: recv})
        monkeypatch.setattr(src, "_flush", lambda sock: None)
        for k in range(12):
            send.sendto(pfb_datagram(k % 2, t_s=43200.0 + k * 1e-4, seq=k),
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


def _run_pfb(monkeypatch, blobs, channels, stop_after):
    with _loopback_pair() as (recv, send, port):
        _patched_socket(monkeypatch, {streamer.PFB_STREAMER_PORT: recv})
        monkeypatch.setattr(src, "_flush", lambda sock: None)
        for b in blobs:
            send.sendto(b, ("127.0.0.1", port))
        time.sleep(0.1)
        sink = _Sink(channels)
        deadline = time.monotonic() + 3.0
        fed = lambda: sum(len(c[1]) for c in sink.calls)
        asyncio.run(src.run_pfb_source(
            sink, "127.0.0.1", channels, module=2,
            should_stop=lambda: time.monotonic() > deadline or fed() >= stop_after))
    return sink


def test_pfb_source_picks_its_channels_from_a_wider_stream(monkeypatch):
    """The streamer carries channels 3 and 7; the capture wants 7."""
    blobs = [pfb_datagram(1, t_s=43200.0 + k * 4.096e-4, seq=k,
                         slots=(3, 7), values=(3 + 0j, 7 + 0j)) for k in range(20)]
    sink = _run_pfb(monkeypatch, blobs, [7], stop_after=1000)
    assert {c[0] for c in sink.calls} == {7}
    assert all(np.all(c[1] == 7.0 / 256.0) for c in sink.calls)
    assert sum(len(c[1]) for c in sink.calls) == 20 * 50


def test_pfb_source_feeds_every_packet_in_stream_order(monkeypatch):
    """Datagrams arrive a few out of order; every packet's samples reach
    the engine, in stream order, with per-sample times that never step
    back."""
    order = list(range(40))
    for i in range(1, 40, 4):            # swap neighbours, as seen on the wire
        order[i], order[i + 1] = order[i + 1], order[i]
    blobs = [pfb_datagram(1, t_s=43200.0 + k * 4.096e-4, seq=1000 + k)
             for k in order]
    sink = _run_pfb(monkeypatch, blobs, [1], stop_after=4000)
    assert sum(len(c[1]) for c in sink.calls) == 4000
    t = np.concatenate([c[2] for c in sink.calls])
    assert np.all(np.diff(t) > 0)


def test_pfb_source_counts_lost_packets(monkeypatch):
    """A hole in the sequence numbers is counted from the receiver's
    own statistics, missing packets individually."""
    seqs = [k for k in range(40) if not 10 <= k < 17]      # 7 never sent
    blobs = [pfb_datagram(1, t_s=43200.0 + k * 4.096e-4, seq=k) for k in seqs]
    sink = _run_pfb(monkeypatch, blobs, [1], stop_after=3300)
    assert sum(len(c[1]) for c in sink.calls) == 3300
    assert sink.source["lost_packets"] == 7


def test_pfb_buffer_seconds_counts_a_four_channel_stream():
    """The buffer figure is in seconds of stream: four channels at
    2.44 MHz, 1000 samples per 8056-byte packet."""
    bytes_per_s = 4 * 2441406.25 / 1000 * streamer.PFB_PACKET_SIZE

    class Sock:
        def getsockopt(self, level, opt):
            held = int(bytes_per_s)         # exactly one second's worth
            return held * 2 if sys.platform == "linux" else held
    assert abs(src._pfb_buffer_seconds(Sock()) - 1.0) < 1e-6


def test_pfb_source_discards_to_bound_its_lag(monkeypatch):
    """When the session says the fast stream is further behind the
    slow one than the bound, the source drops packets by their stamps
    until it is within half the bound, and counts them."""
    with _loopback_pair() as (recv, send, port):
        _patched_socket(monkeypatch, {streamer.PFB_STREAMER_PORT: recv})
        monkeypatch.setattr(src, "_flush", lambda sock: None)
        step = 4.096e-5
        for k in range(200):
            send.sendto(pfb_datagram(1, t_s=43200.0 + k * step, seq=k),
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


def test_pfb_source_turns_the_loop_over_per_batch(monkeypatch):
    """Each popped batch is followed by a turn of the loop, so a dual
    capture's slow side runs between fast batches."""
    N = 40
    with _loopback_pair() as (recv, send, port):
        _patched_socket(monkeypatch, {streamer.PFB_STREAMER_PORT: recv})
        monkeypatch.setattr(src, "_PFB_POP_MAX", 8)
        monkeypatch.setattr(src, "_flush", lambda s: None)
        for k in range(N):
            send.sendto(pfb_datagram(1, seq=k), ("127.0.0.1", port))
        time.sleep(0.05)

        yields = {"n": 0}
        real_sleep = asyncio.sleep

        async def counting_sleep(delay):
            if delay == 0:
                yields["n"] += 1
            return await real_sleep(delay)
        monkeypatch.setattr(asyncio, "sleep", counting_sleep)

        sink = _Sink()
        asyncio.run(src.run_pfb_source(
            sink, "127.0.0.1", [1], module=2,
            should_stop=lambda: sink.count >= N * 100))
    assert sink.count == N * 100
    # 40 packets in pops of at most 8: one turn after each pop.
    assert yields["n"] >= math.ceil(N / 8), f"{yields['n']} yields"
