"""The stream sources drain what the socket holds per event-loop wake,
so a stream's throughput follows its data rate rather than how often
the loop schedules it."""

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
