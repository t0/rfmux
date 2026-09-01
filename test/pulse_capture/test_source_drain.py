"""The stream sources drain in batches, so scheduling cannot starve one.

One datagram per event-loop wake made a stream's throughput proportional
to how often the loop scheduled it.  In a dual capture the two streams
share one loop, per-packet processing runs milliseconds under load, and
the slow stream -- needing 596 wakes a second -- got a fraction of them:
measured live, 0.8 s of sample time over 300 s of wall time, so noise
training never finished and no trigger ever armed.

Draining whatever the socket already holds per wake (bounded) makes a
stream's throughput depend on its own data rate, not on who else shares
the loop.  These tests pin the drain and the fairness it buys.
"""

import asyncio
import contextlib
import socket
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


def _pfb_packet(t_s=43200.0):
    pkt = streamer.PFBPacket()
    pkt.magic = streamer.PFB_PACKET_MAGIC
    pkt.num_samples = 1000
    pkt[:] = np.zeros(1000, dtype=complex)
    h = int(t_s // 3600); m = int(t_s % 3600 // 60); s_ = int(t_s % 60)
    ss = int((t_s % 1) * streamer.SS_PER_SECOND)
    pkt.ts = Timestamp(y=26, d=244, h=h, m=m, s=s_, ss=ss, c=0, sbs=0,
                       source=TimestampSource.TEST, recent=True)
    return bytes(pkt)


@contextlib.contextmanager
def _loopback_pair():
    """(receiver socket factory patch target, sender) on an OS-chosen port."""
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
    # Quiet sockets otherwise hold the source for the full 60 s
    # production timeout after the test's senders finish.
    monkeypatch.setattr(src.streamer, "STREAMER_TIMEOUT", 0.3)


def test_slow_drain_is_lossless_and_ordered(monkeypatch, qt_app=None):
    """Batching per wake must feed exactly the packets sent, in order."""
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

    # The kernel may drop a packet under load; the drain itself must
    # neither lose nor reorder what it was given.
    assert sink.count >= 0.98 * N, f"lost {N - sink.count} of {N} packets"
    stamps = np.array(sink.stamps)
    assert np.all(np.diff(stamps) > 0), "reordered within the drain"
    assert covered == pytest.approx((N - 1) / 596.0, rel=0.05)


def test_starved_loop_still_feeds_the_slow_stream(monkeypatch):
    """The measured pathology, reproduced and required to stay fixed.

    The fast side is given per-packet processing costs like the ones a
    hunting threshold produces (milliseconds each), on the same loop.
    One-datagram-per-wake let it eat nearly every task step; the slow
    side must now keep up regardless, because it drains its backlog
    whenever it does run.
    """
    DURATION = 2.5
    SLOW_RATE, FAST_RATE = 596.0, 2400.0

    with _loopback_pair() as (slow_recv, slow_send, slow_port), \
            _loopback_pair() as (fast_recv, fast_send, fast_port):
        _patched_socket(monkeypatch, {
            streamer.STREAMER_PORT: slow_recv,
            streamer.PFB_STREAMER_PORT: fast_recv,
        })

        slow_sink = _SlowSink()

        class _FastSink:
            channels = [1]

            def feed_block(self, ch, i_vals, q_vals, timestamps):
                # The churn: 4 ms of held loop per packet, busy rather
                # than sleeping, because the real cost is numpy work
                # that never yields.
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < 0.004:
                    pass

        session = type("S", (), {})()
        session.slow_feed = slow_sink
        session.fast_feed = _FastSink()

        stop_at = time.monotonic() + DURATION
        sent = {"slow": 0}

        def pump(sock, port, rate, mk, name=None):
            t0 = time.monotonic()
            n = 0
            while time.monotonic() < stop_at:
                want = int((time.monotonic() - t0) * rate)
                while n < want:
                    sock.sendto(mk(n), ("127.0.0.1", port))
                    n += 1
                time.sleep(0.002)
            if name:
                sent[name] = n

        ths = [threading.Thread(
                   target=pump,
                   args=(slow_send, slow_port, SLOW_RATE,
                         lambda k: _slow_packet(k, t_s=43200.0 + k / 596.0),
                         "slow")),
               threading.Thread(
                   target=pump,
                   args=(fast_send, fast_port, FAST_RATE,
                         lambda k: _pfb_packet(43200.0 + k / 2441.4)))]
        for t in ths:
            t.start()
        asyncio.run(src.run_dual_source(
            session, "127.0.0.1", [1],
            should_stop=lambda: time.monotonic() > stop_at + 0.3))
        for t in ths:
            t.join()

    got, wanted = slow_sink.count, sent["slow"]
    assert wanted > 500, "sender too slow to make the test meaningful"
    # A liveness bound, deliberately loose: timing on a shared machine
    # is noisy, and the sharp mechanism pin is
    # test_many_datagrams_per_wake below.  Measured here: batched drain
    # ~99%, one-per-wake ~76%, live pathology 0.3%.
    assert got >= 0.6 * wanted, \
        f"slow stream starved: {got}/{wanted} packets " \
        f"({100 * got / wanted:.0f}%)"


def test_many_datagrams_per_wake(monkeypatch):
    """The mechanism itself: a backlog is drained per wake, not per await.

    This is the deterministic pin for the starvation fix.  One datagram
    per awaited receive made throughput proportional to how often the
    event loop scheduled the stream; with a preloaded backlog, the
    number of awaited receives must therefore be far smaller than the
    number of packets processed.
    """
    # Small enough to sit in a socket buffer even when rmem_max clamps
    # the 16 MB request to ~200 kB.
    N = 20
    with _loopback_pair() as (recv, send, port):
        _patched_socket(monkeypatch, {streamer.STREAMER_PORT: recv})
        # run_slow_source flushes datagrams that predate the capture --
        # correct in production, but this test IS a preloaded backlog.
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
    # rest of the backlog synchronously.  One-per-wake needs ~N.
    assert awaited["n"] <= 3, \
        f"{awaited['n']} awaited receives for {N} packets -- " \
        "the drain is back to one datagram per wake"
