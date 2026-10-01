"""A Units change converts the traces the pulse view draws, not only its
axis labels: every trace is the stored samples times one complex factor,
a rotation by the df calibration's angle and a scale.  Checked for a
slow capture in review and live, and for a both-mode capture, with a
calibration far from either axis so a missing rotation cannot hide."""

import numpy as np
import pytest
from test.qt_helpers import axis_label  # noqa: E402

pytest.importorskip("PyQt6")

from rfmux.core.transferfunctions import VOLTS_PER_ROC  # noqa: E402
from rfmux.pulse_capture import (  # noqa: E402
    DualPulseCaptureSession, PulseCaptureConfig, PulseCaptureSession)
from rfmux.tools.periscope.pulse_capture_panel import (  # noqa: E402
    PulseCapturePanel, UNITS_COUNTS, UNITS_DF, UNITS_VOLTS)
from rfmux.tools.periscope.pulse_capture_task import (  # noqa: E402
    PulseCaptureSignals, PulseCaptureTask)

SLOW_FS, FAST_FS = 1000.0, 20000.0
#: Hz per volt at 80 degrees: a pulse along I in counts lands mostly on
#: the dissipation axis, so the two panels differ in every view.
CAL = 2.0e7 * np.exp(-1j * np.radians(80.0))
#: stored = counts * VPC * CAL (df, Hz); each view's factor on stored.
FACTORS = {UNITS_DF: 1.0, UNITS_VOLTS: 1.0 / CAL,
           UNITS_COUNTS: 1.0 / (VOLTS_PER_ROC * CAL)}
LABELS = {UNITS_DF: ("df (Hz)", "dissipation (Hz)"),
          UNITS_VOLTS: ("I (V)", "Q (V)"),
          UNITS_COUNTS: ("I (counts)", "Q (counts)")}


def _pulse(rng, fs, seconds, start):
    n = int(seconds * fs)
    t = np.arange(n) / fs
    i, q = rng.normal(0, 1e3, n), rng.normal(0, 1e3, n)
    on = t >= start
    i[on] += 6e4 * np.exp(-(t[on] - start) / 0.004)
    q[on] += 2e4 * np.exp(-(t[on] - start) / 0.004)
    return t, i, q


def _feed_slow(session, rng):
    t, i, q = _pulse(rng, SLOW_FS, 2.0, 1.0)
    for lo in range(0, len(t), 50):
        session.feed_block(1, i[lo:lo + 50], q[lo:lo + 50], t[lo:lo + 50])


def _slow_file(tmp_path):
    path = tmp_path / "slow.h5"
    cfg = PulseCaptureConfig(max_pulse_ms=50.0, noise_train_ms=300.0)
    s = PulseCaptureSession(channels=[1], sample_rate=SLOW_FS,
                            hdf5_path=path, tuning={1: {"df_calibration": CAL}},
                            **cfg.session_kwargs(SLOW_FS))
    s.start()
    _feed_slow(s, np.random.default_rng(1))
    s.stop()
    return path


def _dual_file(tmp_path):
    path = tmp_path / "dual.h5"
    cfg = PulseCaptureConfig(max_pulse_ms=50.0, noise_train_ms=100.0)
    dual = DualPulseCaptureSession(
        channels=[1], slow_rate=SLOW_FS, fast_rate=FAST_FS, config=cfg,
        hdf5_path=path, slow_time_offset_s=0.0,
        tuning={1: {"df_calibration": CAL}})
    dual.start()
    rng = np.random.default_rng(7)
    streams = {fs: _pulse(rng, fs, 2.0, 1.0) for fs in (SLOW_FS, FAST_FS)}
    for k in range(40):
        for feed, fs in ((dual.feed_slow_block, SLOW_FS),
                         (dual.feed_fast_block, FAST_FS)):
            t, i, q = streams[fs]
            a, b = int(k * 0.05 * fs), int((k + 1) * 0.05 * fs)
            feed(1, i[a:b], q[a:b], t[a:b])
    dual.stop()
    return path


def _traces(panel):
    """The pulse view's data curves as complex traces, in drawing
    order: the first panel real, the second imaginary."""
    def curves(plot):
        return [np.asarray(c.getData()[1], dtype=float)
                for c in plot.getPlotItem().listDataItems()
                if len(c.getData()[0]) > 8]
    return [i + 1j * q for i, q in zip(curves(panel.pulse_plot_i),
                                       curves(panel.pulse_plot_q))]


def _labels(panel):
    return tuple(axis_label(p, "left")
                 for p in (panel.pulse_plot_i, panel.pulse_plot_q))


def _check_reprojection(panel, stored):
    """Every view's traces are *stored* times that view's factor, on
    both panels, under the labels that name the view."""
    assert panel._stored_state(1) == ("df", "Hz")
    for view, factor in FACTORS.items():
        panel.units_combo.setCurrentText(view)
        assert _labels(panel) == LABELS[view]
        drawn = _traces(panel)
        assert len(drawn) == len(stored)
        for got, want in zip(drawn, stored):
            np.testing.assert_allclose(got, want * factor, rtol=1e-9)
    # And the rotation is a real one: the two views are not multiples
    # of each other on a panel.
    ratio = (stored[0] / CAL).real / stored[0].real
    assert np.ptp(ratio) > 0.1 * abs(np.mean(ratio))


def test_slow_review_reprojects_both_panels(qt_app, tmp_path):
    panel = PulseCapturePanel(dark_mode=False)
    panel.load_from_hdf5(_slow_file(tmp_path))
    wf = panel.reader.get_pulse(1, 1)
    panel._show_pulse(1, 1)
    _check_reprojection(panel, [wf["Amp_I"] + 1j * wf["Amp_Q"]])


def test_slow_live_reprojects_both_panels(qt_app):
    """The live path: the real worker's cache behind the panel, no file,
    the stored basis worked out from the capture's tuning."""
    panel = PulseCapturePanel(dark_mode=False,
                              tuning={1: {1: {"df_calibration": CAL}}})
    cfg = PulseCaptureConfig(max_pulse_ms=50.0, noise_train_ms=300.0)
    session = PulseCaptureSession(
        channels=[1], sample_rate=SLOW_FS, hdf5_path=None,
        tuning={1: {"df_calibration": CAL}}, **cfg.session_kwargs(SLOW_FS))
    signals = PulseCaptureSignals()
    task = PulseCaptureTask(session, signals, mode="slow")
    panel._reset_results([1])
    panel.task = task
    signals.noise_estimated.connect(panel._on_noise_estimated)
    signals.pulse_detected.connect(panel._on_pulse_detected)
    session.start()
    _feed_slow(session, np.random.default_rng(1))
    assert panel.reader is None and panel._pulse_order == [(1, 1)]
    wf = task.get_pulse(1, 1)
    panel._show_pulse(1, 1)
    _check_reprojection(panel, [wf["Amp_I"] + 1j * wf["Amp_Q"]])
    session.stop()
    panel.task = None


def test_both_mode_reprojects_both_streams(qt_app, tmp_path):
    panel = PulseCapturePanel(dark_mode=False)
    panel.load_from_hdf5(_dual_file(tmp_path))
    key = panel._pulse_order[-1]
    pair = panel.reader.get_match(*key)
    panel._show_pair(*key)
    stored = [pair[f"{side}_tod"]["Amp_I"] + 1j * pair[f"{side}_tod"]["Amp_Q"]
              for side in ("fast", "slow")]
    _check_reprojection(panel, stored)


def test_noise_bands_follow_the_view(qt_app, tmp_path):
    """The bands are the statistics projected by the factor the traces
    are converted with."""
    from rfmux.pulse_capture.analysis import project_noise_stats
    from rfmux.pulse_capture.detection import ChannelNoiseStats
    panel = PulseCapturePanel(dark_mode=False)
    panel.load_from_hdf5(_slow_file(tmp_path))
    ns = ChannelNoiseStats(mean_I=100.0, std_I=50.0, mean_Q=-20.0, std_Q=5.0)
    panel.units_combo.setCurrentText(UNITS_VOLTS)
    seen, want = panel._view_noise(1, ns), project_noise_stats(ns, 1 / CAL)
    for name in ("mean_I", "std_I", "mean_Q", "std_Q"):
        assert getattr(seen, name) == pytest.approx(getattr(want, name))


def test_a_failed_redraw_is_reported(qt_app, tmp_path, capsys):
    """The labels change before the redraw; if the redraw fails, new
    labels over old traces must not pass silently."""
    panel = PulseCapturePanel(dark_mode=False)
    panel.load_from_hdf5(_slow_file(tmp_path))
    panel._show_pulse(1, 1)

    def boom(*_):
        raise RuntimeError("no such pulse")
    panel._show_pulse = boom
    panel.units_combo.setCurrentText(UNITS_COUNTS)
    assert "redraw failed" in panel.status_label.text()
    assert "redraw failed" in capsys.readouterr().out


def test_the_noise_record_keeps_its_stored_labels_through_a_units_change(
        qt_app, tmp_path):
    panel = PulseCapturePanel(dark_mode=False)
    panel.load_from_hdf5(_slow_file(tmp_path))
    panel._show_noise_segment(None, 1)
    before = _labels(panel)
    assert before == ("df (Hz)", "dissipation (Hz)")
    panel.units_combo.setCurrentText(UNITS_COUNTS)
    assert _labels(panel) == before


def test_slow_iq_plane_reprojects(qt_app, tmp_path):
    panel = PulseCapturePanel(dark_mode=False)
    panel.load_from_hdf5(_slow_file(tmp_path))
    wf = panel.reader.get_pulse(1, 1)
    stored = wf["Amp_I"] + 1j * wf["Amp_Q"]
    panel._show_pulse(1, 1)
    panel.viewer_tabs.setCurrentWidget(panel.iq_view)
    for view, factor in FACTORS.items():
        panel.units_combo.setCurrentText(view)
        curve = max(panel.iq_plot.getPlotItem().listDataItems(),
                    key=lambda c: len(c.getData()[0]))
        x, y = curve.getData()
        np.testing.assert_allclose(np.asarray(x) + 1j * np.asarray(y),
                                   stored * factor, rtol=1e-9)
