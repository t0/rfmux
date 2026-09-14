"""Pulse Capture panel: what a reviewed file shows, and what the
panel says without a dialog."""
from types import SimpleNamespace

import numpy as np
import pytest

from test.qt_helpers import spin  # noqa: E402

pytest.importorskip("PyQt6")
pytest.importorskip("h5py")

from PyQt6 import QtCore  # noqa: E402
from rfmux.core.transferfunctions import VOLTS_PER_ROC  # noqa: E402
from rfmux.pulse_capture.capture_session import (  # noqa: E402
    PulseCaptureConfig, PulseCaptureSession,
)
from rfmux.pulse_capture.detection import ChannelNoiseStats  # noqa: E402
from rfmux.tools.periscope import pulse_capture_panel as m  # noqa: E402
from rfmux.tools.periscope.pulse_capture_panel import (  # noqa: E402
    PulseCapturePanel,
)
from test.pulse_capture.test_pulse_capture_panel import (  # noqa: E402
    _FakeRuntime, _build_dual_file,
)


def _no_dialogs(monkeypatch):
    """Any dialog fails the test."""
    def boom(*a, **k):
        raise AssertionError(f"dialog raised: {a[2] if len(a) > 2 else a}")
    monkeypatch.setattr(m.QtWidgets.QMessageBox, "warning", boom)
    monkeypatch.setattr(m.QtWidgets.QMessageBox, "information", boom)


def _panel(qt_app, **kw):
    panel = PulseCapturePanel(dark_mode=False, **kw)
    yield panel
    panel.close()
    spin(qt_app)


@pytest.fixture
def panel(qt_app):
    yield from _panel(qt_app)


@pytest.fixture
def calibrated_panel(qt_app):
    """A calibrated channel whose live capture stores quadratures in
    volts, so every view is a conversion away from storage."""
    for panel in _panel(qt_app, tuning={1: {1: {"df_calibration": 2.0e6 + 0j}}}):
        panel.capture_config = PulseCaptureConfig(trigger_basis="iq")
        yield panel


def _build_timed_capture_file(tmp_path, fs=20000.0):
    """A single-stream file made with a sample rate, so its templates
    carry the time axis a live capture's do."""
    path = tmp_path / "timed_review.h5"
    s = PulseCaptureSession(
        channels=[1], sample_rate=fs, hdf5_path=path,
        **PulseCaptureConfig(threshold_sigma=5.0, end_sigma=1.5,
                             max_pulse_ms=50.0,
                             noise_train_ms=20.0).session_kwargs(fs))
    rng = np.random.default_rng(3)
    s.start()
    n = int(0.5 * fs)
    t = np.arange(n) / fs
    sig = rng.normal(0, 1.0, n)
    for t0 in (0.1, 0.25, 0.4):
        mask = t >= t0
        sig[mask] += 50.0 * np.exp(-(t[mask] - t0) / 1e-3)
    s.feed_block(1, sig, rng.normal(0, 1.0, n), 43000.0 + t)
    s.stop()
    assert s.total_pulses == 3
    return path


# ── Review mode ───────────────────────────────────────────────────


def test_a_reviewed_pair_shows_its_decay_constant(qt_app, tmp_path, panel):
    panel.load_from_hdf5(_build_dual_file(tmp_path))
    panel._show_pair(*panel._pulse_order[-1])
    assert "τ =" in panel.pulse_info.text()


def test_a_reviewed_file_fills_the_template_tab(qt_app, tmp_path, panel):
    panel.load_from_hdf5(_build_timed_capture_file(tmp_path))
    assert "Trigger-aligned stack" in panel.template_info.text()
    assert len(panel.template_plot_i.getPlotItem().listDataItems()) >= 1


def test_a_reviewed_dual_file_fills_the_template_tab(qt_app, tmp_path,
                                                     panel):
    panel.load_from_hdf5(_build_dual_file(tmp_path))
    assert set(panel._template_data_by_stream) == {"slow", "fast"}
    assert len(panel.template_plot_i.getPlotItem().listDataItems()) >= 1


def test_a_reviewed_pair_reports_a_short_fast_window(qt_app, panel):
    """The union window travels with the full pair; a fast record that
    stops short of it is packets lost, and the pair view says so."""
    panel._both_mode = True
    panel._reset_results([1])
    t_fast = 1.0 + np.arange(200) / 1e5
    pair = {"channel": 1, "pair_idx": 1, "slow_idx": 1, "fast_idx": 1,
            "time_offset": 0.0, "window": (0.99, 1.005),
            "fast_tod": {"Amp_I": np.zeros(200), "Amp_Q": np.zeros(200),
                         "Time": t_fast}}
    panel._get_pair = lambda ch, i: pair
    panel._get_waveform = lambda ch, idx, stream=None: None
    panel._on_pair_matched({k: pair[k] for k in (
        "channel", "pair_idx", "slow_idx", "fast_idx", "time_offset")})
    panel._show_pair(1, 1)
    text = panel.pulse_info.text()
    assert "fast window incomplete" in text
    assert "first" in text and "last" in text


def test_a_missing_pair_stops_loading(qt_app, panel):
    """When the file has neither the pair nor its records, the view
    reports that after one round of requests instead of asking again on
    every reply."""
    calls = []
    panel.task = SimpleNamespace(
        get_pair=lambda ch, i: None,
        get_pulse=lambda ch, i, s=None: None,
        request_pair=lambda ch, i: calls.append(("pair", ch, i)),
        request_waveform=lambda ch, i, s=None: calls.append((s, ch, i)),
        request_stop=lambda: None, wait=lambda *_: None,
        session=SimpleNamespace(hdf5_path=None))
    panel._both_mode = True
    panel._reset_results([1])
    panel._on_pair_matched({"channel": 1, "pair_idx": 1, "slow_idx": 1,
                            "fast_idx": 1, "time_offset": 0.0})
    panel._show_pair(1, 1)
    assert "loading" in panel.pulse_plot_i.getPlotItem().titleLabel.text
    assert calls and len(calls) == len(set(calls)), calls
    asked = len(calls)
    # Each fetch reports back, the view redraws, and nothing is re-asked.
    for _ in range(3):
        panel._on_waveform_ready(1, 1)
    assert len(calls) == asked
    assert "not available" in panel.pulse_plot_i.getPlotItem().titleLabel.text
    panel.task = None


# ── Units ──────────────────────────────────────────────────────────


def test_the_peak_in_the_info_line_is_in_the_view_units(qt_app,
                                                          calibrated_panel):
    panel = calibrated_panel
    panel._counts = {1: 1}
    panel._pulse_summaries[(1, 1)] = {
        "n_samples": 10, "duration_ms": 1.0, "peak_amp": 3.0e-5,
        "snr": 8.0, "tau_ms": float("nan")}
    panel._get_waveform = lambda ch, idx, stream=None: None
    panel.units_combo.setCurrentText(m.UNITS_COUNTS)
    panel._show_pulse(1, 1)
    expected = 3.0e-5 / VOLTS_PER_ROC
    assert f"peak {expected:.4g} counts" in panel.pulse_info.text()
    panel.units_combo.setCurrentText(m.UNITS_DF)
    panel._show_pulse(1, 1)
    assert f"peak {3.0e-5 * 2.0e6:.4g} Hz" in panel.pulse_info.text()


def test_amplitude_bins_follow_the_view_in_counts_too(qt_app,
                                                       calibrated_panel):
    panel = calibrated_panel
    panel._counts = {1: 3}
    panel._hist_data = {"amplitude_i_edges": np.array([0.0, 1e-5, 2e-5]),
                        "amplitude_i_counts_ch1": np.array([1.0, 2.0])}
    panel.units_combo.setCurrentText(m.UNITS_COUNTS)
    curve = panel.hist_plots["amplitude"].getPlotItem().listDataItems()[0]
    assert np.max(curve.xData) == pytest.approx(2e-5 / VOLTS_PER_ROC)
    label = panel.hist_plots["amplitude"].getPlotItem().getAxis("bottom")
    assert label.labelText == "amplitude (counts)"


def test_amplitude_histogram_overlays_the_two_stored_axes(qt_app,
                                                         calibrated_panel):
    panel = calibrated_panel
    panel._counts = {1: 3}
    panel._hist_data = {"amplitude_i_edges": np.array([0.0, 1e-5, 2e-5]),
                        "amplitude_i_counts_ch1": np.array([1.0, 2.0]),
                        "amplitude_q_edges": np.array([0.0, 1e-5, 2e-5]),
                        "amplitude_q_counts_ch1": np.array([3.0, 0.0])}
    panel._render_histograms()
    curves = panel.hist_plots["amplitude"].getPlotItem().listDataItems()
    names = [c.name() for c in curves]
    assert len(curves) == 2
    # Quadratures stored, so the axes are I and Q; the second is hatched
    assert any(" I (" in n for n in names) and any(" Q (" in n for n in names)
    q = curves[[i for i, n in enumerate(names) if " Q (" in n][0]]
    assert m.pg.mkBrush(q.opts["fillBrush"]).style() == QtCore.Qt.BrushStyle.BDiagPattern
    i = curves[[i for i, n in enumerate(names) if " I (" in n][0]]
    assert m.pg.mkBrush(i.opts["fillBrush"]).style() == QtCore.Qt.BrushStyle.SolidPattern
    # The key naming the two axes leads the legend, and stays when the
    # per-channel names drop out past the listed-channel limit.
    legend = panel.hist_plots["amplitude"].getPlotItem().legend
    labels = [lbl.text for _s, lbl in legend.items]
    assert labels[:2] == ["I: filled", "Q: hatched"]
    many = {}
    for ch in range(1, m.MAX_LISTED_CHANNELS + 3):
        many[f"amplitude_i_counts_ch{ch}"] = np.array([1.0, 2.0])
        many[f"amplitude_q_counts_ch{ch}"] = np.array([3.0, 0.0])
    panel._counts = {ch: 3 for ch in range(1, m.MAX_LISTED_CHANNELS + 3)}
    panel._hist_data = {"amplitude_i_edges": np.array([0.0, 1e-5, 2e-5]),
                        "amplitude_q_edges": np.array([0.0, 1e-5, 2e-5]), **many}
    panel._render_histograms()
    labels = [lbl.text for _s, lbl in legend.items]
    assert labels == ["I: filled", "Q: hatched"]


def test_the_quadrature_view_of_a_hertz_channel_draws_the_raw_pair(qt_app):
    """A channel stored in the frequency basis keeps its raw-quadrature
    peaks beside the stored ones; the volts and counts views draw those,
    named I and Q, and the df view the stored pair."""
    cal = 2.0e6 + 0j
    panel = PulseCapturePanel(dark_mode=False, tuning={1: {1: {"df_calibration": cal}}})
    panel.capture_config = PulseCaptureConfig(trigger_basis="df")
    panel._counts = {1: 3}
    hz = np.array([0.0, 1000.0, 2000.0])
    volts = np.array([0.0, 1e-4, 2e-4])
    panel._hist_data = {
        "amplitude_i_edges": hz, "amplitude_i_counts_ch1": np.array([1.0, 2.0]),
        "amplitude_q_edges": hz, "amplitude_q_counts_ch1": np.array([3.0, 0.0]),
        "amplitude_raw_i_edges": volts, "amplitude_raw_i_counts_ch1": np.array([2.0, 1.0]),
        "amplitude_raw_q_edges": volts, "amplitude_raw_q_counts_ch1": np.array([0.0, 3.0])}
    item = panel.hist_plots["amplitude"].getPlotItem()
    panel.units_combo.setCurrentText(m.UNITS_VOLTS)
    names = [c.name() for c in item.listDataItems()]
    assert all(" I (" in n or " Q (" in n for n in names) and len(names) == 2
    assert np.max(item.listDataItems()[0].xData) == pytest.approx(2e-4)
    panel.units_combo.setCurrentText(m.UNITS_COUNTS)
    assert np.max(item.listDataItems()[0].xData) == pytest.approx(2e-4 / VOLTS_PER_ROC)
    panel.units_combo.setCurrentText(m.UNITS_DF)
    names = [c.name() for c in item.listDataItems()]
    assert all(" df (" in n or " diss (" in n for n in names)
    assert np.max(item.listDataItems()[0].xData) == pytest.approx(2000.0)
    panel.close()
    spin(qt_app)


def test_idle_axes_name_the_default_view(qt_app, panel):
    """Before any data, every tab names the units the selector shows."""
    assert panel.units_combo.currentText() == m.UNITS_VOLTS
    for plot in (panel.pulse_plot_i, panel.template_plot_i):
        assert plot.getPlotItem().getAxis("left").labelText == "I (V)"
    amp = panel.hist_plots["amplitude"].getPlotItem().getAxis("bottom")
    assert amp.labelText == "amplitude (V)"


def test_the_noise_segment_prints_the_stored_unit(qt_app, panel):
    rng = np.random.default_rng(0)
    arr = 1e-5 * (rng.normal(0, 1, 300) + 1j * rng.normal(0, 1, 300))
    panel.task = SimpleNamespace(session=SimpleNamespace(
        noise_data={1: arr}))
    panel.noise_stats = {1: ChannelNoiseStats(mean_I=2e-6, std_I=1.1e-5,
                                              mean_Q=0.0, std_Q=1e-5)}
    panel._show_noise_segment()
    text = panel.pulse_info.text()
    assert "I = 2e-06 ± 1.1e-05 V" in text
    assert panel.pulse_plot_i.getPlotItem().getAxis("left").labelText \
        == "I (V)"
    panel.task = None


def test_units_change_keeps_the_two_stream_noise_strip(qt_app, panel):
    panel._both_mode = True
    panel._reset_results([1])
    panel._noise_by_stream = {
        "slow": {1: ChannelNoiseStats(std_I=1e-5, std_Q=1e-5)},
        "fast": {1: ChannelNoiseStats(std_I=3e-5, std_Q=3e-5)}}
    panel.noise_stats = panel._noise_by_stream["slow"]
    panel.units_combo.setCurrentText(m.UNITS_COUNTS)
    text = panel.noise_label.text()
    assert "slow:" in text and "fast:" in text


# ── Start checks and routine outcomes ──────────────────────────────


def test_slow_mode_refuses_a_module_periscope_is_not_receiving(
        qt_app, tmp_path, monkeypatch):
    warned = []
    monkeypatch.setattr(m.QtWidgets.QMessageBox, "warning",
                        lambda *a, **k: warned.append(a[2]))
    runtime = _FakeRuntime()
    runtime.module = 2
    panel = PulseCapturePanel(periscope=runtime, dark_mode=False, module=1)
    panel._browse_dir = str(tmp_path)
    panel.channels_edit.setText("1")
    panel._on_start()
    assert panel.task is None
    assert warned and "module 2" in warned[0]
    panel.close()
    spin(qt_app)


def test_a_both_mode_capture_is_registered_with_the_session(qt_app, tmp_path,
                                                            panel):
    registered = []
    panel.session_manager = SimpleNamespace(
        is_active=True, session_path=str(tmp_path),
        register_external_file=lambda p, t, l: registered.append((p, t, l)))
    panel.task = SimpleNamespace(session=SimpleNamespace(
        hdf5_path=tmp_path / "pulse_module1_000000.h5", module=1,
        slow=SimpleNamespace(noise_data={}),
        fast=SimpleNamespace(noise_data={})))
    panel._both_mode = True
    panel._on_noise_estimated({"stream": "slow",
                               "stats": {1: ChannelNoiseStats()}})
    panel._on_noise_estimated({"stream": "fast",
                               "stats": {1: ChannelNoiseStats()}})
    assert registered == [(str(tmp_path / "pulse_module1_000000.h5"),
                           "pulse", "module1")]
    panel.task = None


def test_an_empty_export_goes_to_the_status_line(qt_app, panel, monkeypatch):
    _no_dialogs(monkeypatch)
    panel.viewer_tabs.setCurrentIndex(1)
    panel._on_export()
    assert "Nothing to export" in panel.status_label.text()


def test_relabelling_with_all_never_reads_the_board(qt_app, panel,
                                                    monkeypatch):
    _no_dialogs(monkeypatch)
    panel.channels_edit.setText("all")
    assert panel._label_channel() == 1
    panel.units_combo.setCurrentText(m.UNITS_COUNTS)
