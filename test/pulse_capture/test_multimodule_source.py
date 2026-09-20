"""The slow source feeds a session that spans modules: one ingest per
module, keyed by (module, channel), and a silent module is an error
rather than a stall."""

import asyncio
import threading
import time

import numpy as np
import pytest

from rfmux import streamer
from rfmux.pulse_capture import sources as src
from test.packet_helpers import (loopback_pair, patched_socket,
                                 readout_packet)

T0 = 43200.0
FS = 596.0


class _Sink:
    def __init__(self, channels):
        self.channels = list(channels)
        self.fed = {}                         # key -> I values

    def feed_block(self, key, i, q, t):
        self.fed.setdefault(key, []).extend(np.asarray(i).real.tolist())


def _packet(seq, module, value):
    pkt = readout_packet(seq, t_s=T0 + seq / FS, module=module)
    pkt[:] = np.full(len(pkt), value, dtype=complex)
    return bytes(pkt)


def _send_interleaved(send, port, n, modules):
    for k in range(n):
        for m in modules:
            send.sendto(_packet(k, m, 10 * m), ("127.0.0.1", port))
        if k % 10 == 0:
            time.sleep(0.002)


def test_pair_keys_take_each_module_from_its_own_packets(monkeypatch):
    N = 40
    with loopback_pair() as (recv, send, port):
        patched_socket(monkeypatch, {streamer.STREAMER_PORT: recv})
        sink = _Sink([(1, 1), (2, 3)])
        th = threading.Thread(target=_send_interleaved,
                              args=(send, port, N, (1, 2, 3)))
        th.start()
        deadline = time.monotonic() + 5.0
        covered = asyncio.run(src.run_slow_source(
            sink, "127.0.0.1",
            should_stop=lambda: (sum(map(len, sink.fed.values())) >= 2 * N
                                 or time.monotonic() > deadline)))
        th.join()
    assert set(sink.fed) == {(1, 1), (2, 3)}
    assert set(sink.fed[(1, 1)]) == {10.0} and len(sink.fed[(1, 1)]) == N
    assert set(sink.fed[(2, 3)]) == {20.0} and len(sink.fed[(2, 3)]) == N
    assert covered == pytest.approx((N - 1) / FS, rel=0.05)


def test_a_plain_channel_is_read_from_the_module_given(monkeypatch):
    N = 20
    with loopback_pair() as (recv, send, port):
        patched_socket(monkeypatch, {streamer.STREAMER_PORT: recv})
        sink = _Sink([1])
        th = threading.Thread(target=_send_interleaved,
                              args=(send, port, N, (1, 2)))
        th.start()
        deadline = time.monotonic() + 5.0
        asyncio.run(src.run_slow_source(
            sink, "127.0.0.1", module=2,
            should_stop=lambda: (len(sink.fed.get(1, [])) >= N
                                 or time.monotonic() > deadline)))
        th.join()
    assert set(sink.fed) == {1} and set(sink.fed[1]) == {20.0}


def test_the_duration_is_covered_by_the_first_module_to_cover_it(monkeypatch):
    N = 60
    with loopback_pair() as (recv, send, port):
        patched_socket(monkeypatch, {streamer.STREAMER_PORT: recv})
        sink = _Sink([(1, 1), (2, 1)])
        th = threading.Thread(target=_send_interleaved,
                              args=(send, port, N, (1, 2)))
        th.start()
        covered = asyncio.run(src.run_slow_source(
            sink, "127.0.0.1", duration_s=20 / FS))
        th.join()
    assert covered == pytest.approx(20 / FS, abs=1.5 / FS)
    assert 15 <= len(sink.fed[(1, 1)]) <= 25
    assert 15 <= len(sink.fed[(2, 1)]) <= 25


def test_a_module_that_never_sends_anything_is_an_error_too(monkeypatch):
    with loopback_pair() as (recv, send, port):
        patched_socket(monkeypatch, {streamer.STREAMER_PORT: recv})
        sink = _Sink([2])
        t = time.monotonic()
        with pytest.raises(ValueError, match=r"module\(s\) \[2\] sent no"):
            asyncio.run(src.run_slow_source(sink, "127.0.0.1", module=2,
                                            duration_s=5.0))
    assert time.monotonic() - t < 3.0


def test_a_module_still_silent_when_the_duration_is_covered_is_an_error(
        monkeypatch):
    """Module 1 covers a short duration before the silence span runs
    out: the run ends in the error, not in a capture missing a module."""
    with loopback_pair() as (recv, send, port):
        patched_socket(monkeypatch, {streamer.STREAMER_PORT: recv})
        sink = _Sink([(1, 1), (3, 1)])
        stop = threading.Event()

        def pump():
            k = 0
            while not stop.is_set():
                send.sendto(_packet(k, 1, 10.0), ("127.0.0.1", port))
                k += 1
                time.sleep(0.005)

        th = threading.Thread(target=pump)
        th.start()
        try:
            with pytest.raises(ValueError, match=r"module\(s\) \[3\] sent no"):
                asyncio.run(src.run_slow_source(sink, "127.0.0.1",
                                                duration_s=0.1))
        finally:
            stop.set()
            th.join()
    assert len(sink.fed[(1, 1)]) >= 0.1 * FS
