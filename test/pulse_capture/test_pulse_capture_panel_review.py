"""Pulse Capture panel: what a reviewed file shows, and what the
panel says without a dialog."""
from types import SimpleNamespace

import numpy as np
import pytest
from test.qt_helpers import axis_label, pulse_rows  # noqa: E402

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


def test_autoscaling_the_template_fits_the_stacked_span(qt_app, tmp_path,
                                                        panel):
    """The time grid runs to half the ring buffer; an autoscale fits the
    bins that hold data, as the first draw does."""
    panel.load_from_hdf5(_build_timed_capture_file(tmp_path))
    vb = panel.template_plot_i.getPlotItem().vb
    drawn = vb.viewRange()[0]
    vb.autoRange()
    fitted = vb.viewRange()[0]
    assert fitted[1] - fitted[0] < 1.2 * (drawn[1] - drawn[0])


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
    assert axis_label(panel.hist_plots["amplitude"], "bottom") == \
        "amplitude (counts)"


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


@pytest.fixture
def tod_only(qt_app, tmp_path, panel):
    """A file of time-ordered data alone, in df with a calibration,
    open in review; (panel, channel, calibration)."""
    from rfmux.algorithms.measurement.tod import write_tod
    from test.pulse_capture.test_overlay import (
        CHANNEL, _dirfile, _recording_file)
    cal = complex(3.0e6, -4.0e6)
    path = write_tod(tmp_path / "tod.h5", [CHANNEL], 1,
                     fastrx=_recording_file(tmp_path),
                     dirfile=_dirfile(tmp_path),
                     tuning={CHANNEL: {"bias_channel": CHANNEL,
                                       "df_calibration": cal}})
    panel.load_from_hdf5(path)
    return panel, CHANNEL, cal


def test_a_tod_file_names_its_streams_and_lists_no_pulses(tod_only):
    panel, channel, _ = tod_only
    status = panel.status_label.text()
    assert "slow:" in status and "fast:" in status
    assert all(row.childCount() == 0 for row in pulse_rows(panel))


def test_the_metadata_item_lists_the_metadata_and_each_calibration(
        tod_only):
    """Every attribute of the metadata group, and under it the channel's
    calibration, its df calibration first."""
    panel, channel, _ = tod_only
    tree = panel.pulse_tree
    meta = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
                if "Metadata" in tree.topLevelItem(i).text(0))
    lines = [meta.child(i).text(0) for i in range(meta.childCount())]
    assert "trigger_basis = df" in lines and "stored_units = Hz" in lines
    cal_item = next(meta.child(i) for i in range(meta.childCount())
                    if meta.child(i).text(0) == f"calibration, channel {channel}")
    assert cal_item.child(0).text(0).startswith("df_calibration = ")


def _tod_file(tmp_path):
    from rfmux.algorithms.measurement.tod import write_tod
    from test.pulse_capture.test_overlay import (
        CHANNEL, _dirfile, _recording_file)
    return write_tod(tmp_path / "tod.h5", [CHANNEL], 1,
                     fastrx=_recording_file(tmp_path),
                     dirfile=_dirfile(tmp_path), trigger_basis="iq"), CHANNEL


def _tod_item(panel):
    tree = panel.pulse_tree
    return next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
                if "Time-ordered data" in tree.topLevelItem(i).text(0))


def test_the_tree_holds_the_pulses_beside_the_time_ordered_data(
        qt_app, tmp_path, panel):
    """Top level: Pulses, holding the channels (or events) as the
    grouping says, and beside it the time-ordered data and metadata."""
    from rfmux.algorithms.measurement.tod import merge_tod, write_tod
    from test.pulse_capture.test_overlay import (
        CHANNEL, _capture, _recording_file)
    pulse = _capture(tmp_path, channels=(CHANNEL,), trigger_basis="iq")
    merge_tod(pulse, write_tod(tmp_path / "tod.h5", [CHANNEL], 1,
                               fastrx=_recording_file(tmp_path),
                               trigger_basis="iq"))
    panel.load_from_hdf5(pulse)
    tree = panel.pulse_tree
    tops = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
    assert tops[0] == "◆ Pulses"
    assert any(t.startswith("≋ Time-ordered data") for t in tops[1:])
    assert tops[-1] == "▦ Metadata"
    assert [r.text(0) for r in pulse_rows(panel)] == \
        [f"▤ Channel {CHANNEL} (1)"]
    panel.group_combo.setCurrentText(m.GROUP_EVENTS)
    assert [r.data(0, QtCore.Qt.ItemDataRole.UserRole)[0]
            for r in pulse_rows(panel)] == ["event"]
    tops = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
    assert tops[0] == "◆ Pulses"


@pytest.fixture
def tod_tab(qt_app, tmp_path, panel):
    """A TOD file in review, its channel open in the Channel TOD View."""
    path, channel = _tod_file(tmp_path)
    panel.load_from_hdf5(path)
    panel._open_tod_viewer(channel)
    return panel.tod_view


def test_double_clicking_a_tod_channel_brings_its_tab_forward(
        qt_app, tmp_path, panel):
    path, channel = _tod_file(tmp_path)
    panel.load_from_hdf5(path)
    tabs, view = panel.viewer_tabs, panel.tod_view
    assert tabs.isTabVisible(tabs.indexOf(view))
    assert tabs.currentWidget() is not view
    panel._on_tree_double_click(_tod_item(panel).child(0), 0)
    assert tabs.currentWidget() is view
    assert view.channel_combo.currentData() == channel


def test_a_wide_view_draws_at_most_two_points_per_bin(tod_tab):
    from rfmux.algorithms.measurement.tod import VIEW_BINS
    for curve in tod_tab.curves["fast"] + tod_tab.curves["slow"]:
        assert 0 < len(curve.getData()[0]) <= 2 * VIEW_BINS


def test_the_fast_stream_is_drawn_under_the_slow(tod_tab):
    fast, slow = tod_tab.curves["fast"], tod_tab.curves["slow"]
    assert max(c.zValue() for c in fast) < min(c.zValue() for c in slow)


def test_a_narrow_view_draws_the_samples_themselves(tod_tab):
    from rfmux.algorithms.measurement.tod import tod_window
    tod_tab.plots[0].setXRange(0.010, 0.011, padding=0)
    tod_tab._refresh()
    x0, x1 = tod_tab.plots[0].getPlotItem().viewRange()[0]
    want = tod_window(tod_tab.f, "fast", tod_tab.key,
                      tod_tab.origin + x0, tod_tab.origin + x1)
    assert want["kind"] == "raw"
    np.testing.assert_allclose(tod_tab.curves["fast"][0].getData()[1],
                               want["I"], rtol=1e-6)


def test_unchecking_a_stream_clears_its_curves(tod_tab):
    tod_tab.stream_checks["slow"].setChecked(False)
    for curve in tod_tab.curves["slow"]:
        x = curve.getData()[0]
        assert x is None or len(x) == 0
    assert all(len(c.getData()[0]) for c in tod_tab.curves["fast"])


def test_the_tod_tab_zooms_to_a_dragged_box_and_back_to_the_whole_run(
        qt_app, tmp_path, panel):
    """Dragging draws a zoom box (the pulse view's viewbox), which sets
    both axes; the wheel is time only; Whole run returns with the
    vertical axis following the data again."""
    import pyqtgraph as pg
    from PyQt6 import QtCore as QC
    path, channel = _tod_file(tmp_path)
    panel.load_from_hdf5(path)
    panel._open_tod_viewer(channel)
    view = panel.tod_view
    vb = view.plots[0].getPlotItem().getViewBox()
    assert vb.state["mouseMode"] == pg.ViewBox.RectMode
    assert vb.state["mouseEnabled"] == [True, False]
    vb.showAxRect(QC.QRectF(0.010, -50.0, 0.001, 100.0), padding=0)
    view._refresh()
    (x0, x1), (y0, y1) = vb.viewRange()
    assert (x0, x1) == pytest.approx((0.010, 0.011))
    assert (y0, y1) == pytest.approx((-50.0, 50.0))
    assert "each drawn" in view.info.text()
    view.reset_btn.click()
    (x0, x1), _ = vb.viewRange()
    assert x1 - x0 == pytest.approx(view.span * 1.02, rel=1e-3)
    assert vb.state["autoRange"][1]


def test_entering_the_tod_tab_draws_the_selected_pulses_channel(
        qt_app, tmp_path, panel):
    """No trip to the Channel box: the tab opens on the channel of the
    pulse selected in Pulse View, and on the first channel with the
    time-ordered data when the selection is on a channel it lacks."""
    from rfmux.algorithms.measurement.tod import merge_tod, write_tod
    from test.pulse_capture.test_overlay import (
        CHANNEL, _capture, _recording_file)
    pulse = _capture(tmp_path, channels=(CHANNEL, 5))
    tod = write_tod(tmp_path / "tod.h5", [CHANNEL, 7], 1,
                    fastrx=_recording_file(tmp_path), trigger_basis="iq")
    import h5py
    with h5py.File(pulse, "a") as f:            # the capture stored volts
        for c in (CHANNEL, 5):
            f[f"channel_{c}"].attrs["stored_units"] = "V"
        f["metadata"].attrs["trigger_basis"] = "iq"
    merge_tod(pulse, tod)
    panel.load_from_hdf5(pulse)
    tabs, view = panel.viewer_tabs, panel.tod_view
    panel._show_pulse(CHANNEL, 1)
    tabs.setCurrentWidget(view)
    assert view.key == CHANNEL and view.curves
    assert view.info.text().startswith("Channel 200:")
    # A selection on a channel the TOD lacks leaves what is drawn.
    tabs.setCurrentIndex(0)
    panel._current_view = (5, 1)
    tabs.setCurrentWidget(view)
    assert view.key == CHANNEL


def test_prev_and_next_step_through_the_channels_over_the_same_window(
        qt_app, tmp_path, panel):
    """Like the pulse tab's: one channel along at a time, stopping at
    either end; the time window zoomed to stays, to compare channels at
    one moment."""
    from rfmux.algorithms.measurement.tod import write_tod
    from test.pulse_capture.test_overlay import CHANNEL, _recording_file
    path = write_tod(tmp_path / "tod.h5", [7, CHANNEL], 1,
                     fastrx=_recording_file(tmp_path), trigger_basis="iq")
    panel.load_from_hdf5(path)
    view = panel.tod_view
    panel._open_tod_viewer(7)
    assert not view.btn_prev.isEnabled() and view.btn_next.isEnabled()
    from PyQt6 import QtCore as QC
    vb = view.plots[0].getPlotItem().getViewBox()
    vb.showAxRect(QC.QRectF(0.010, -50.0, 0.001, 100.0), padding=0)
    view.btn_next.click()
    assert view.key == CHANNEL
    assert view.channel_combo.currentData() == CHANNEL
    assert vb.viewRange()[0] == pytest.approx([0.010, 0.011])
    # The box's levels were channel 7's; channel 200's are its own.
    assert vb.state["autoRange"][1]
    assert view.btn_prev.isEnabled() and not view.btn_next.isEnabled()
    view.btn_next.click()                    # at the end: stays
    assert view.key == CHANNEL
    view.btn_prev.click()
    assert view.key == 7


def test_the_units_choice_redraws_the_tod_tab_as_it_does_the_pulse_view(
        qt_app, tmp_path, panel):
    """The TOD tab takes the same conversion as the pulse and IQ views:
    a channel stored in volts and switched to df is drawn in hertz under
    df and dissipation labels, over the same window."""
    from rfmux.algorithms.measurement.tod import write_tod
    from test.pulse_capture.test_overlay import CHANNEL, _recording_file
    cal = complex(3e6, -4e6)
    path = write_tod(tmp_path / "tod.h5", [CHANNEL], 1,
                     fastrx=_recording_file(tmp_path), trigger_basis="iq",
                     tuning={CHANNEL: {"df_calibration": cal}})
    panel.load_from_hdf5(path)
    view = panel.tod_view
    panel.units_combo.setCurrentText(m.UNITS_VOLTS)
    panel._open_tod_viewer(CHANNEL)
    assert axis_label(view.plots[0], "left") == "I (V)"
    view.plots[0].setXRange(0.010, 0.011, padding=0)
    view._refresh()
    i_volts = view.curves["fast"][0].getData()[1]
    q_volts = view.curves["fast"][1].getData()[1]
    panel.units_combo.setCurrentText(m.UNITS_DF)
    assert axis_label(view.plots[0], "left") == "df (Hz)"
    assert axis_label(view.plots[1], "left") == \
        "dissipation (Hz)"
    assert view.plots[0].getPlotItem().viewRange()[0] == \
        pytest.approx([0.010, 0.011])
    factor = panel._view_coeffs(CHANNEL)[0]
    expected = (i_volts + 1j * q_volts) * factor
    np.testing.assert_allclose(view.curves["fast"][0].getData()[1],
                               expected.real, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(view.curves["fast"][1].getData()[1],
                               expected.imag, rtol=1e-4, atol=1e-6)


def test_the_tod_tab_opens_on_its_first_channel_with_nothing_selected(
        qt_app, tmp_path, panel):
    path, channel = _tod_file(tmp_path)
    panel.load_from_hdf5(path)
    panel.viewer_tabs.setCurrentWidget(panel.tod_view)
    assert panel.tod_view.key == channel and panel.tod_view.curves


def test_the_tod_tab_is_hidden_for_a_file_without_time_ordered_data(
        qt_app, tmp_path, panel):
    from test.pulse_capture.test_overlay import _capture
    path, channel = _tod_file(tmp_path)
    panel.load_from_hdf5(path)
    panel._open_tod_viewer(channel)
    (tmp_path / "plain").mkdir()
    panel.load_from_hdf5(_capture(tmp_path / "plain"))
    tabs, view = panel.viewer_tabs, panel.tod_view
    assert not tabs.isTabVisible(tabs.indexOf(view))
    assert view.f is None and view.channel_combo.count() == 0


def test_idle_axes_name_the_default_view(qt_app, panel):
    """Before any data, every tab names the units the selector shows."""
    assert panel.units_combo.currentText() == m.UNITS_VOLTS
    for plot in (panel.pulse_plot_i, panel.template_plot_i):
        assert axis_label(plot, "left") == "I (V)"
    assert axis_label(panel.hist_plots["amplitude"], "bottom") == \
        "amplitude (V)"


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
    assert axis_label(panel.pulse_plot_i, "left") \
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
