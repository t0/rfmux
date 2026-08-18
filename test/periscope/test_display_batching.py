"""
Batching the display writes must not change what gets plotted.

_update_buffers used to write four values per displayed channel per
packet.  It now buffers packets and writes them a frame at a time,
which is only safe if the ring contents come out identical.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from rfmux.tools.periscope.app import Periscope  # noqa: E402
from rfmux.tools.periscope.utils import Circular  # noqa: E402

N = 256
CHANNELS = [1, 2, 5]


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


class _Packet:
    """Just enough packet for np.array(pkt) and len(pkt)."""

    def __init__(self, values):
        self._v = values

    def __array__(self, dtype=None, copy=None):
        return self._v if dtype is None else self._v.astype(dtype)

    def __len__(self):
        return self._v.shape[0]


def _runtime(qt_app, width=128):
    p = Periscope.__new__(Periscope)
    QtWidgets.QMainWindow.__init__(p)
    p.all_chs = list(CHANNELS)
    p.N = N
    p.buf = {c: {k: Circular(N) for k in ("I", "Q", "M")} for c in CHANNELS}
    p.tbuf = {c: Circular(N) for c in CHANNELS}
    p._display_values = []
    p._display_times = []
    p._display_width = -1
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

    p = _runtime(qt_app, width)
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
