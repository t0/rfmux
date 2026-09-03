"""
Rings and the pulse tap hold the same contents whether packets arrive
one at a time or as one batch per frame.
"""

from types import SimpleNamespace

import numpy as np
import pytest


pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from rfmux.tools.periscope.app import Periscope  # noqa: E402
from rfmux.tools.periscope.utils import Circular  # noqa: E402
from test.packet_helpers import stamp  # noqa: E402

N = 256
CHANNELS = [1, 2, 5]



class _Packet:
    """Just enough packet for np.array(pkt) and len(pkt)."""

    def __init__(self, values):
        self._v = values
        self.ts = SimpleNamespace(recent=False)   # no disciplined stamp

    def __array__(self, dtype=None, copy=None):
        return self._v if dtype is None else self._v.astype(dtype)

    def __len__(self):
        return self._v.shape[0]


def _runtime(qt_app):
    p = Periscope.__new__(Periscope)
    QtWidgets.QMainWindow.__init__(p)
    p.all_chs = list(CHANNELS)
    p.N = N
    p.buf = {c: {k: Circular(N) for k in ("I", "Q", "M")} for c in CHANNELS}
    p.tbuf = {c: Circular(N) for c in CHANNELS}
    p._display_values = []
    p._display_times = []
    p._display_width = -1
    p._display_rows = 0
    p._pulse_tap = None
    p._pulse_tap_channels = None
    p._pulse_tap_cache = None
    return p


def _packets(n_packets, width, rng):
    return [(_Packet((rng.normal(0, 1e5, width)
                      + 1j * rng.normal(0, 1e5, width))),
             0.001 * i)
            for i in range(n_packets)]


@pytest.mark.parametrize("n_packets", [1, 10, 700])
def test_batched_writes_match_per_packet_writes(qt_app, n_packets):
    rng = np.random.default_rng(17)
    width = 128
    packets = _packets(n_packets, width, rng)

    p = _runtime(qt_app)
    for pkt, t_rel in packets:
        p._update_buffers(pkt, t_rel)
    p._flush_display_batch()

    # What writing one sample at a time would have produced.
    want = {c: {k: Circular(N) for k in ("I", "Q", "M")} for c in CHANNELS}
    want_t = {c: Circular(N) for c in CHANNELS}
    for pkt, t_rel in packets:
        samples = np.array(pkt) / 256
        for c in CHANNELS:
            s = samples[c - 1]
            want[c]["I"].add(s.real)
            want[c]["Q"].add(s.imag)
            want[c]["M"].add(np.abs(s))
            want_t[c].add(t_rel)

    for c in CHANNELS:
        for k in ("I", "Q", "M"):
            np.testing.assert_allclose(p.buf[c][k].data(),
                                       want[c][k].data(), rtol=0, atol=0)
        np.testing.assert_allclose(p.tbuf[c].data(), want_t[c].data())


def test_packet_width_change_flushes_rather_than_stacking(qt_app):
    # Short and long packets cannot go in one np.stack; a decimation
    # change mid-frame must not raise or mix columns.
    rng = np.random.default_rng(3)
    p = _runtime(qt_app)
    for pkt, t in _packets(5, 128, rng):
        p._update_buffers(pkt, t)
    for pkt, t in _packets(5, 1024, rng):
        p._update_buffers(pkt, t)
    p._flush_display_batch()
    assert p.buf[1]["I"].count == 10


def test_channels_beyond_the_packet_width_are_skipped(qt_app):
    rng = np.random.default_rng(4)
    p = _runtime(qt_app)
    p.all_chs = [1, 200]              # 200 is not in a short packet
    p.buf[200] = {k: Circular(N) for k in ("I", "Q", "M")}
    p.tbuf[200] = Circular(N)
    for pkt, t in _packets(4, 128, rng):
        p._update_buffers(pkt, t)
    p._flush_display_batch()
    assert p.buf[1]["I"].count == 4
    assert p.buf[200]["I"].count == 0


def test_none_timestamp_becomes_nan(qt_app):
    rng = np.random.default_rng(5)
    p = _runtime(qt_app)
    pkt, _ = _packets(1, 128, rng)[0]
    p._update_buffers(pkt, None)
    p._flush_display_batch()
    assert np.isnan(p.tbuf[1].data()[0])


class _TapSink:
    def __init__(self):
        self.calls = []

    def __call__(self, channels, values, stamps, day):
        self.calls.append((channels, np.atleast_2d(np.asarray(values)).copy(),
                           np.atleast_1d(np.asarray(stamps, dtype=float)).copy(),
                           day))


class _StampedPacket(_Packet):
    def __init__(self, values, seconds):
        super().__init__(values)
        self.ts = stamp(seconds)


def test_a_batch_equals_the_packets_one_at_a_time(qt_app):
    """The batched writer, fed one array for many packets, leaves the
    rings and the pulse tap exactly as the per-packet writer does."""
    from rfmux import streamer
    rng = np.random.default_rng(23)
    width = 128
    n = 300
    secs = 43200.0 + np.arange(n) * 2.6e-5
    packets = [_StampedPacket(rng.normal(0, 1e5, width)
                              + 1j * rng.normal(0, 1e5, width), s)
               for s in secs]
    t_rel = np.arange(n) * 2.6e-5

    def run(feed):
        p = _runtime(qt_app)
        tap = _TapSink()
        p._pulse_tap = tap
        p._pulse_tap_channels = (1, 5)
        p._pulse_tap_cache = None
        p._pulse_tap_day_key = None
        p._pulse_tap_day = None
        feed(p)
        p._flush_display_batch()
        return p, tap

    def per_packet(p):
        for pkt, t in zip(packets, t_rel):
            p._update_buffers(pkt, float(t))

    def batched(p):
        samples = np.stack([np.array(pkt) for pkt in packets])
        seconds = np.array([_seconds(pkt.ts) for pkt in packets])
        p._update_buffers_batch(samples, t_rel, seconds,
                                np.ones(n, dtype=bool), (26, 245))

    def _seconds(ts):
        return ts.h * 3600 + ts.m * 60 + ts.s + ts.ss / streamer.SS_PER_SECOND

    a, tap_a = run(per_packet)
    b, tap_b = run(batched)
    for c in CHANNELS:
        for k in ("I", "Q", "M"):
            np.testing.assert_array_equal(a.buf[c][k].data(), b.buf[c][k].data())
        np.testing.assert_array_equal(a.tbuf[c].data(), b.tbuf[c].data())
    rows_a = np.concatenate([v for _, v, _, _ in tap_a.calls])
    rows_b = np.concatenate([v for _, v, _, _ in tap_b.calls])
    np.testing.assert_array_equal(rows_a, rows_b)
    stamps_a = np.concatenate([s for _, _, s, _ in tap_a.calls])
    stamps_b = np.concatenate([s for _, _, s, _ in tap_b.calls])
    np.testing.assert_array_equal(stamps_a, stamps_b)
    assert {d for _, _, _, d in tap_a.calls} == {d for _, _, _, d in tap_b.calls}
    assert tap_b.calls[0][3] == streamer.ts_day_epoch(packets[0].ts)
