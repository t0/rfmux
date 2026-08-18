"""
Offscreen GUI integration test for the Pulse Capture panel (Phase B).

Drives the real PulseCapturePanel through the real tap → queue →
PulseCaptureTask → PulseCaptureSession path with synthetic samples, and
verifies the tree, status, histograms, waveform cache, and the
finalized HDF5 file.
"""

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
pytest.importorskip("h5py")

from PyQt6 import QtWidgets  # noqa: E402

from rfmux.algorithms.measurement.pulse_hdf5 import PulseHDF5Reader  # noqa: E402
from rfmux.tools.periscope.pulse_capture_panel import PulseCapturePanel  # noqa: E402

SAMPLE_RATE = 38147.0
DT = 1.0 / SAMPLE_RATE


class _FakeRuntime:
    """Stands in for the Periscope main window's tap API."""

    def __init__(self):
        self._pulse_tap = None
        self.tap_channels = None

    def register_pulse_tap(self, callback, channels=None):
        self._pulse_tap = callback
        self.tap_channels = channels

    def unregister_pulse_tap(self):
        self._pulse_tap = None
        self.tap_channels = None


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _spin(qt_app, seconds=0.05):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qt_app.processEvents()
        time.sleep(0.005)


def _spin_until(qt_app, predicate, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qt_app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _make_panel(qt_app, tmp_path, runtime):
    from dataclasses import replace

    panel = PulseCapturePanel(periscope=runtime, dark_mode=True)
    panel._browse_dir = str(tmp_path)
    panel.channels_edit.setText("1")
    panel.threshold_spin.setValue(5.0)
    panel.end_spin.setValue(1.5)
    # Training is derived from the pulse length (20x), which at the
    # default 250 ms would ask for more samples than these tests feed.
    # Override just the training length: max_pulse_ms is left alone
    # because it also sets the baseline-tracking floor, and 250 ms is
    # the right scale for the sample-indexed pulses fed below.
    panel.capture_config = replace(panel.capture_config,
                                   noise_train_ms=1.0)
    return panel


def _tap1(tap, ch, i, q, t):
    """Feed one sample through the packet-shaped tap.

    The tap takes a whole packet (channels, values, timestamp); a slow
    packet carries one sample per channel, so a single sample is just a
    one-entry packet.
    """
    tap((ch,), np.array([complex(i, q)]), t)


def _feed_capture(tap, rng, n=3000, pulse_starts=(100, 900, 1700),
                  tau_samples=40, amp=60.0):
    signal = rng.normal(0, 1.0, n)
    k = np.arange(n)
    for k0 in pulse_starts:
        m = k >= k0
        signal[m] += amp * np.exp(-(k[m] - k0) / tau_samples)
    for i in range(n):
        _tap1(tap, 1, float(signal[i]), float(rng.normal(0, 1.0)),
              i * DT)


def test_live_capture_end_to_end(qt_app, tmp_path):
    runtime = _FakeRuntime()
    panel = _make_panel(qt_app, tmp_path, runtime)
    rng = np.random.default_rng(42)

    panel._on_start()
    assert panel.task is not None
    assert runtime._pulse_tap is not None
    hdf5_path = panel.task.session.hdf5_path
    assert str(hdf5_path).startswith(str(tmp_path))

    # Noise estimation: 1000 samples (session default), timestamps unused
    for _ in range(1000):
        _tap1(runtime._pulse_tap, 1, float(rng.normal(0, 1.0)),
              float(rng.normal(0, 1.0)), None)
    assert _spin_until(qt_app, lambda: panel.noise_stats), \
        "noise_estimated signal never arrived"
    assert "Noise:" in panel.noise_label.text()
    # The Pulse View shows the noise-training segment until pulses arrive
    assert "Noise training" in panel.pulse_info.text()

    # Capture stream with 3 injected pulses
    _feed_capture(runtime._pulse_tap, rng)
    assert _spin_until(
        qt_app, lambda: len(panel._pulse_order) == 3), \
        f"expected 3 pulses, saw {len(panel._pulse_order)}"

    # Tree: channel group shows count, newest first
    ch_item = panel._channel_items[1]
    assert "(3)" in ch_item.text(0)
    assert ch_item.childCount() == 3
    assert "#000003" in ch_item.child(0).text(0)

    # Status line went green/capturing
    assert "Capturing" in panel.status_label.text()

    # Live waveform cache serves the pulse view
    wf = panel.task.get_pulse(1, 1)
    assert wf is not None and len(wf["Amp_I"]) > 0
    panel._show_pulse(1, 1)
    assert "Pulse #000001" in panel.pulse_info.text()
    assert "derived τ" in panel.pulse_info.text()

    # Histograms rendered for all four metrics (flush_every default 50 —
    # force one render from current accumulator state via the session)
    panel._on_histograms(
        panel.task.session.histograms.get_histogram_data())
    for metric in ("snr", "amplitude", "duration_ms", "tau_ms"):
        assert len(panel.hist_plots[metric].getPlotItem().
                   listDataItems()) >= 1, f"no curve in {metric} histogram"

    # Stop → task finishes, tap unregistered, file finalized
    panel._on_stop()
    assert _spin_until(qt_app, lambda: panel.task is None), \
        "task never finished"
    assert runtime._pulse_tap is None
    assert "Stopped" in panel.status_label.text()

    with PulseHDF5Reader(hdf5_path) as reader:
        assert reader.pulse_count(1) == 3
        assert "capture_end" in reader.metadata
        pulse = reader.get_pulse(1, 1)
        assert np.isfinite(pulse["tau_s"])
        hists = reader.get_histograms()
        assert np.sum(hists["tau_ms_counts_ch1"]) == 3

    panel.close()
    _spin(qt_app)


def test_tap_exclusivity(qt_app, tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning",
        staticmethod(lambda *a, **k: warnings.append(a)))

    runtime = _FakeRuntime()
    panel1 = _make_panel(qt_app, tmp_path, runtime)
    panel2 = _make_panel(qt_app, tmp_path, runtime)

    panel1._on_start()
    assert panel1.task is not None
    panel2._on_start()
    assert panel2.task is None, "second panel must not start"
    assert warnings, "second panel should have warned about the busy tap"

    panel1._on_stop()
    assert _spin_until(qt_app, lambda: panel1.task is None)
    panel1.close()
    panel2.close()
    _spin(qt_app)


# ───────────────────────── Phase C: review mode + session ───────────

from rfmux.algorithms.measurement.pulse_capture_session import (  # noqa: E402
    PulseCaptureSession,
)


def _build_capture_file(tmp_path, n_pulses=3):
    """Produce a real capture HDF5 headlessly via PulseCaptureSession."""
    path = tmp_path / "review_source.h5"
    session = PulseCaptureSession(
        channels=[1], threshold_sigma=5.0, end_sigma=1.5,
        margin_fraction=0.2, noise_samples=200, hdf5_path=path,
        histogram_flush_every=2)
    rng = np.random.default_rng(42)
    session.start()
    for _ in range(200):
        session.feed_sample(1, float(rng.normal(0, 1.0)),
                            float(rng.normal(0, 1.0)), None)
    n = 3000
    starts = [100 + 800 * i for i in range(n_pulses)]
    signal = rng.normal(0, 1.0, n)
    k = np.arange(n)
    for k0 in starts:
        m = k >= k0
        signal[m] += 60.0 * np.exp(-(k[m] - k0) / 40.0)
    for i in range(n):
        session.feed_sample(1, float(signal[i]),
                            float(rng.normal(0, 1.0)), i * DT)
    assert session.total_pulses == n_pulses
    session.stop()
    return path


def test_review_mode(qt_app, tmp_path):
    path = _build_capture_file(tmp_path)
    panel = PulseCapturePanel(dark_mode=False)
    panel.load_from_hdf5(path)

    # Controls restored from file metadata, then locked
    assert panel.threshold_spin.value() == pytest.approx(5.0)
    assert panel.end_spin.value() == pytest.approx(1.5)
    assert not panel.btn_start.isEnabled()
    assert not panel.channels_edit.isEnabled()
    assert "Review Mode" in panel.status_label.text()

    # Tree populated newest-first
    ch_item = panel._channel_items[1]
    assert ch_item.childCount() == 3
    assert "(3)" in ch_item.text(0)
    assert "#000003" in ch_item.child(0).text(0)

    # Waveforms come from the reader (no task)
    assert panel.task is None
    wf = panel._get_waveform(1, 2)
    assert wf is not None and len(wf["Amp_I"]) > 0
    panel._show_pulse(1, 2)
    assert "Pulse #000002" in panel.pulse_info.text()

    # Histograms restored from the file
    for metric in ("snr", "amplitude", "duration_ms", "tau_ms"):
        assert len(panel.hist_plots[metric].getPlotItem().
                   listDataItems()) >= 1, f"no curve in {metric}"

    panel.close()
    _spin(qt_app)


def test_identify_and_register(qt_app, tmp_path):
    from rfmux.tools.periscope.session_manager import SessionManager

    h5 = _build_capture_file(tmp_path)
    fake = tmp_path / "not_really.h5"
    fake.write_text("plain text pretending to be hdf5")

    sm = SessionManager()
    session_dir = sm.start_session(str(tmp_path), "session_test")
    assert sm.is_active

    assert sm.identify_file_type(str(h5)) == "pulse"
    assert sm.identify_file_type(str(fake)) is None

    exported = []
    sm.file_exported.connect(lambda p, t: exported.append((p, t)))
    target = session_dir / "pulse_module1_120000.h5"
    target.write_bytes(h5.read_bytes())
    sm.register_external_file(str(target), "pulse", "module1")

    assert exported == [(str(target), "pulse")]
    entries = sm.session_metadata.get("exports", [])
    assert any(e["data_type"] == "pulse"
               and e["filename"] == target.name for e in entries)
    sm.end_session()


# ───────────── Noise-training feedback / channel validation ─────────


def test_noise_progress_stall_visibility(qt_app, tmp_path):
    """Requesting a channel the stream never delivers must be VISIBLE:
    the status line shows per-channel progress with the starved channel
    stuck at 0/N."""
    runtime = _FakeRuntime()
    panel = _make_panel(qt_app, tmp_path, runtime)
    panel.channels_edit.setText("1,2")
    rng = np.random.default_rng(11)

    panel._on_start()
    assert panel.task is not None
    target = panel.task.session.noise_samples
    n_fed = target // 2  # partial fill on ch1 only — channel 2 starves
    for _ in range(n_fed):
        _tap1(runtime._pulse_tap, 1, float(rng.normal(0, 1.0)),
              float(rng.normal(0, 1.0)), None)

    assert _spin_until(
        qt_app,
        lambda: f"Ch1 {n_fed}/{target}" in panel.status_label.text()), \
        f"no progress shown: {panel.status_label.text()!r}"
    assert f"Ch2 0/{target}" in panel.status_label.text()

    panel._on_stop()
    assert _spin_until(qt_app, lambda: panel.task is None)
    panel.close()
    _spin(qt_app)


def test_non_displayed_channels_start_ok(qt_app, tmp_path):
    """Capture channels are decoupled from displayed channels: the tap
    is registered with the requested list, display is irrelevant."""
    runtime = _FakeRuntime()
    runtime.all_chs = [1]  # periscope displays only channel 1
    panel = _make_panel(qt_app, tmp_path, runtime)
    panel.channels_edit.setText("1,2")

    panel._on_start()
    assert panel.task is not None, \
        "non-displayed channels must be capturable"
    assert runtime.tap_channels == [1, 2]

    panel._on_stop()
    assert _spin_until(qt_app, lambda: panel.task is None)
    panel.close()
    _spin(qt_app)


def test_packet_width_validation(qt_app, tmp_path, monkeypatch):
    """Channels beyond the packet width (128 short / 1024 long) abort."""
    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning",
        staticmethod(lambda parent, title, text: warnings.append(text)))

    runtime = _FakeRuntime()
    runtime.is_short_packet = True  # 128-channel packets
    panel = _make_panel(qt_app, tmp_path, runtime)
    panel.channels_edit.setText("1,200")

    panel._on_start()
    assert panel.task is None
    assert warnings and "200" in warnings[0] and "128" in warnings[0]
    panel.close()
    _spin(qt_app)


def test_waveform_fetch_after_eviction_and_stop(qt_app, tmp_path):
    """Evicted waveforms load back from the live HDF5 via the worker;
    after Stop, the finalized file keeps every pulse browsable."""
    runtime = _FakeRuntime()
    panel = _make_panel(qt_app, tmp_path, runtime)
    rng = np.random.default_rng(42)

    panel._on_start()
    panel.task._cache_size = 1  # force eviction of all but the latest
    # Disable follow-latest so no _show_pulse (and hence no automatic
    # cache-refetch) runs while pulses arrive — keeps eviction
    # deterministic for this test.
    panel.follow_check.setChecked(False)
    for _ in range(1000):
        _tap1(runtime._pulse_tap, 1, float(rng.normal(0, 1.0)),
              float(rng.normal(0, 1.0)), None)
    assert _spin_until(qt_app, lambda: panel.noise_stats)
    _feed_capture(runtime._pulse_tap, rng)
    assert _spin_until(qt_app, lambda: len(panel._pulse_order) == 3)

    # Pulse 1 was evicted (cache holds only the newest)
    assert panel.task.get_pulse(1, 1) is None
    panel._show_pulse(1, 1)  # triggers async fetch from the live file
    assert _spin_until(
        qt_app,
        lambda: len(panel.pulse_plot_i.getPlotItem().listDataItems()) >= 1
        and len(panel.pulse_plot_q.getPlotItem().listDataItems()) >= 1), \
        "waveform never loaded from the live HDF5 file"
    assert "Pulse #000001" in panel.pulse_info.text()

    # After stop: reader auto-opens, everything stays browsable
    panel._on_stop()
    assert _spin_until(qt_app, lambda: panel.task is None)
    assert panel.reader is not None
    for idx in (1, 2, 3):
        wf = panel._get_waveform(1, idx)
        assert wf is not None and len(wf["Amp_I"]) > 0
    panel._show_pulse(1, 2)
    assert "Pulse #000002" in panel.pulse_info.text()

    panel.close()
    _spin(qt_app)


def test_channel_default_follows_stream(qt_app, tmp_path):
    runtime = _FakeRuntime()
    runtime.all_chs = [3, 1]
    panel = PulseCapturePanel(periscope=runtime, dark_mode=True)
    assert panel.channels_edit.text() == "1,3"
    panel.close()
    _spin(qt_app)


def _build_dual_file(tmp_path):
    from rfmux.algorithms.measurement.pulse_capture_dual import (
        DualPulseCaptureSession,
    )
    from rfmux.algorithms.measurement.pulse_capture_session import (
        PulseCaptureConfig,
    )
    path = tmp_path / "dual_review.h5"
    cfg = PulseCaptureConfig(threshold_sigma=5.0, end_sigma=1.5,
                             max_pulse_ms=200.0, noise_train_ms=10.0)
    dual = DualPulseCaptureSession(channels=[1], slow_rate=20000.0,
                                   fast_rate=100000.0, config=cfg,
                                   hdf5_path=path)
    rng = np.random.default_rng(9)
    dual.start()
    for feed, n in ((dual.feed_slow, dual.slow.noise_samples + 5),
                    (dual.feed_fast, dual.fast.noise_samples + 5)):
        for _ in range(n):
            feed(1, float(rng.normal()), float(rng.normal()), None)
    for feed, fs in ((dual.feed_slow, 20000.0),
                     (dual.feed_fast, 100000.0)):
        n = int(0.5 * fs)
        t = 0.9 + np.arange(n) / fs
        sig = rng.normal(0, 1.0, n)
        mask = t >= 1.0
        sig[mask] += 50.0 * np.exp(-(t[mask] - 1.0) / 1e-3)
        for i in range(n):
            feed(1, float(sig[i]), float(rng.normal()), float(t[i]))
    dual.stop()
    return path


def test_dual_review_mode(qt_app, tmp_path):
    path = _build_dual_file(tmp_path)
    panel = PulseCapturePanel(dark_mode=False)
    panel.load_from_hdf5(path)

    assert panel._both_mode
    assert "Review Mode" in panel.status_label.text()
    assert not panel.btn_start.isEnabled()

    ch_item = panel._channel_items[1]
    assert ch_item.childCount() >= 1
    assert "pairs" in ch_item.text(0)

    # The matched pair renders: fast line + slow line+markers per plot
    key = panel._pulse_order[-1]
    panel._show_pair(*key)
    assert "Pair #" in panel.pulse_info.text()
    assert len(panel.pulse_plot_i.getPlotItem().listDataItems()) >= 2
    assert len(panel.pulse_plot_q.getPlotItem().listDataItems()) >= 2
    # Legend distinguishes the fast line from the slow markers
    legend = panel.pulse_plot_i.getPlotItem().legend
    labels = [label.text for _sample, label in legend.items]
    assert any("fast" in t for t in labels), labels
    assert any("slow" in t for t in labels), labels

    panel.close()
    _spin(qt_app)


def test_dual_session_hdf5_path_parity(tmp_path):
    """Panel/task read session.hdf5_path on finish — the dual session
    must expose it like the single session (stop-crash regression)."""
    from rfmux.algorithms.measurement.pulse_capture_dual import (
        DualPulseCaptureSession,
    )
    path = tmp_path / "parity.h5"
    dual = DualPulseCaptureSession(channels=[1], slow_rate=20000.0,
                                   hdf5_path=path)
    assert dual.hdf5_path == path
    dual.stop()
    no_file = DualPulseCaptureSession(channels=[1], slow_rate=20000.0)
    assert no_file.hdf5_path is None
    no_file.stop()


def test_follow_latest_coalesces_bursts(qt_app, tmp_path):
    """A burst of pulses with a tiny cache must leave the viewer on the
    NEWEST pulse (cached), not churning on evicted intermediates."""
    runtime = _FakeRuntime()
    panel = _make_panel(qt_app, tmp_path, runtime)
    rng = np.random.default_rng(21)

    panel._on_start()
    panel.task._cache_size = 2  # aggressive eviction
    assert panel.follow_check.isChecked()
    for _ in range(1000):
        _tap1(runtime._pulse_tap, 1, float(rng.normal(0, 1.0)),
              float(rng.normal(0, 1.0)), None)
    assert _spin_until(qt_app, lambda: panel.noise_stats)

    _feed_capture(runtime._pulse_tap, rng, n=8000,
                  pulse_starts=tuple(range(200, 7000, 700)))
    n_expected = len(range(200, 7000, 700))
    # This test is about the coalescing timer, not detector tuning, so
    # wait for the burst to land rather than pinning an exact count.
    assert _spin_until(
        qt_app, lambda: len(panel._pulse_order) >= n_expected), \
        f"saw {len(panel._pulse_order)}/{n_expected} pulses"
    _spin(qt_app, 0.3)
    newest = max(idx for _ch, idx in panel._pulse_order)
    assert _spin_until(
        qt_app,
        lambda: f"#{newest:06d}" in panel.pulse_info.text()
        and len(panel.pulse_plot_i.getPlotItem().listDataItems()) >= 1), \
        f"viewer not on latest (#{newest}): {panel.pulse_info.text()!r}"

    panel._on_stop()
    assert _spin_until(qt_app, lambda: panel.task is None)
    panel.close()
    _spin(qt_app)


def test_stale_dock_entries_are_pruned(qt_app, tmp_path):
    """Closing a Pulse Capture dock must not leave a dangling entry:
    double-clicking its .h5 afterwards previously raised
    'wrapped C/C++ object ... has been deleted'."""
    from types import SimpleNamespace

    from PyQt6 import sip

    from rfmux.tools.periscope.app import Periscope

    panel_live = PulseCapturePanel(dark_mode=False)
    dock_live = QtWidgets.QDockWidget()
    panel_dead = PulseCapturePanel(dark_mode=False)
    dock_dead = QtWidgets.QDockWidget()

    fake = SimpleNamespace(pulse_capture_windows={
        "live": {"window": panel_live, "dock": dock_live},
        "dead": {"window": panel_dead, "dock": dock_dead},
        "empty": {"window": None, "dock": None},
    })

    sip.delete(dock_dead)          # emulate a closed/destroyed dock
    live = Periscope._live_pulse_capture_windows(fake)

    assert len(live) == 1
    assert live[0]["window"] is panel_live
    assert set(fake.pulse_capture_windows) == {"live"}

    panel_live.close()
    panel_dead.close()
    _spin(qt_app)


def test_template_tab_renders(qt_app, tmp_path):
    """Trigger-aligned stack reaches the Template tab and the HDF5."""
    runtime = _FakeRuntime()
    panel = _make_panel(qt_app, tmp_path, runtime)
    rng = np.random.default_rng(4)

    panel._on_start()
    hdf5_path = panel.task.session.hdf5_path
    for _ in range(1000):
        _tap1(runtime._pulse_tap, 1, float(rng.normal(0, 1.0)),
              float(rng.normal(0, 1.0)), None)
    assert _spin_until(qt_app, lambda: panel.noise_stats)

    _feed_capture(runtime._pulse_tap, rng, n=8000,
                  pulse_starts=tuple(range(200, 7000, 700)))
    n_expected = len(range(200, 7000, 700))
    assert _spin_until(
        qt_app, lambda: len(panel._pulse_order) == n_expected)

    # Force a flush so the template payload reaches the GUI
    panel._on_templates(panel.task.session.templates.get_template_data())
    assert len(panel.template_plot_i.getPlotItem().listDataItems()) >= 1
    assert len(panel.template_plot_q.getPlotItem().listDataItems()) >= 1
    assert "Trigger-aligned stack" in panel.template_info.text()

    # Stacked template peaks AT the trigger (index pre_samples)
    acc = panel.task.session.templates.get(1)
    assert acc.n_pulses == n_expected
    mean = acc.mean("I")
    assert int(np.nanargmax(np.abs(mean))) == acc.pre_samples

    panel._on_stop()
    assert _spin_until(qt_app, lambda: panel.task is None)

    with PulseHDF5Reader(hdf5_path) as reader:
        tmpl = reader.get_templates()
        assert "template_I_ch1" in tmpl
        assert np.nanmax(np.abs(tmpl["template_I_ch1"])) > 0

    panel.close()
    _spin(qt_app)


def test_units_toggle_scales_amplitude(qt_app, tmp_path):
    """counts → Hz uses counts × VOLTS_PER_ROC × df_calibration."""
    from rfmux.core.transferfunctions import VOLTS_PER_ROC

    panel = PulseCapturePanel(dark_mode=False,
                              df_calibrations={1: {1: 2.0e6}})
    panel.module_spin.setValue(1)
    assert panel._df_scale(1) == pytest.approx(2.0e6 * VOLTS_PER_ROC)
    assert panel._df_scale(7) is None          # uncalibrated channel
    assert not panel._units_are_hz()
    panel.units_combo.setCurrentText("Hz")
    assert panel._units_are_hz()

    # No calibration at all → no scaling offered
    plain = PulseCapturePanel(dark_mode=False)
    assert plain._df_scale(1) is None
    panel.close()
    plain.close()
    _spin(qt_app)


def test_csv_exports(qt_app, tmp_path):
    """Each viewer tab exports its own CSV."""
    import csv as _csv

    path = _build_capture_file(tmp_path)
    panel = PulseCapturePanel(dark_mode=False)
    panel.load_from_hdf5(path)
    panel._browse_dir = str(tmp_path)

    # Pulse View tab
    panel.viewer_tabs.setCurrentIndex(0)
    panel._show_pulse(*panel._pulse_order[-1])
    panel._on_export()
    # Histograms tab
    panel.viewer_tabs.setCurrentIndex(1)
    panel._on_export()
    # Template tab (needs template data from the file)
    panel._template_data = panel.reader.get_templates()
    panel.viewer_tabs.setCurrentIndex(2)
    panel._on_export()

    written = sorted(p.name for p in tmp_path.glob("*.csv"))
    assert any(n.startswith("pulse_ch") for n in written), written
    assert any(n.startswith("pulse_histograms") for n in written), written
    assert any(n.startswith("pulse_template") for n in written), written

    hist_csv = next(tmp_path.glob("pulse_histograms_*.csv"))
    with open(hist_csv) as fh:
        rows = list(_csv.reader(fh))
    assert rows[0] == ["metric", "channel", "bin_left", "bin_right",
                       "count"]
    assert len(rows) > 10

    panel.close()
    _spin(qt_app)


def test_keyboard_navigation(qt_app, tmp_path):
    path = _build_capture_file(tmp_path)
    panel = PulseCapturePanel(dark_mode=False)
    panel.load_from_hdf5(path)

    panel._navigate_end(first=True)
    first = panel._current_view
    panel._navigate_end(first=False)
    last = panel._current_view
    assert first != last

    panel._navigate(-1)
    assert panel._current_view != last

    start_tab = panel.viewer_tabs.currentIndex()
    panel._cycle_tab()
    assert panel.viewer_tabs.currentIndex() != start_tab
    for _ in range(panel.viewer_tabs.count() - 1):
        panel._cycle_tab()
    assert panel.viewer_tabs.currentIndex() == start_tab

    panel.close()
    _spin(qt_app)


def test_template_view_fits_data_and_uses_zoombox(qt_app, tmp_path):
    """Template axes track the STACKED region (not the whole pre/post
    grid), and plots default to zoombox (RectMode) like the rest of
    Periscope."""
    import pyqtgraph as pg

    from rfmux.algorithms.measurement.pulse_templates import (
        PulseTemplateSet,
    )
    from rfmux.algorithms.measurement.pulse_detection import (
        ChannelNoiseStats,
    )

    panel = PulseCapturePanel(dark_mode=False)
    panel._counts = {1: 5}

    # Stack a few pulses whose data occupies only part of the grid
    ts = PulseTemplateSet(pre_samples=50, post_samples=400,
                          threshold_sigma=5.0, sample_rate=1000.0)
    ns = ChannelNoiseStats(std_I=1.0, std_Q=1.0)
    rng = np.random.default_rng(2)
    for i in range(5):
        n = 160
        k = np.arange(n)
        sig = rng.normal(0, 1.0, n)
        sig[k >= 40] += 80.0 * np.exp(-(k[k >= 40] - 40) / 15.0)
        ts.add_pulse(1, {"Amp_I": sig, "Amp_Q": rng.normal(0, 1.0, n),
                         "Time": k / 1000.0, "pileup": False}, ns)

    panel._on_templates(ts.get_template_data())

    acc = ts.get(1)
    counts = acc.counts
    t_axis = acc.time_axis(1000.0)
    populated = np.nonzero(counts > 0)[0]
    data_lo, data_hi = t_axis[populated[0]], t_axis[populated[-1]]
    grid_span = t_axis[-1] - t_axis[0]

    (x0, x1), (y0, y1) = panel.template_plot_i.getPlotItem().vb.viewRange()
    # X fits the populated region, not the full grid
    assert x1 - x0 < 0.6 * grid_span
    assert x0 <= data_lo + 1e-9 and x1 >= data_hi - 1e-9
    # Y brackets the template peak with sane padding
    peak = float(np.nanmax(np.abs(acc.mean("I"))))
    assert y1 >= peak * 0.9
    assert (y1 - y0) < peak * 4

    for plot in (panel.template_plot_i, panel.template_plot_q,
                 panel.pulse_plot_i, panel.pulse_plot_q,
                 panel.hist_plots["snr"]):
        assert plot.getPlotItem().vb.state["mouseMode"] == \
            pg.ViewBox.RectMode

    panel.close()
    _spin(qt_app)


def _axis_label(plot):
    ax = plot.getPlotItem().getAxis("bottom")
    return ax.labelText, ax.labelUnits


def test_stacked_plots_share_one_x_axis_label(qt_app, tmp_path):
    """Labelling only one of an x-linked pair leaves them showing the
    same data against differently scaled ticks: setting `units` turns on
    pyqtgraph's SI-prefix autoscaling for that axis only."""
    runtime = _FakeRuntime()
    panel = _make_panel(qt_app, tmp_path, runtime)
    assert _axis_label(panel.pulse_plot_i) == _axis_label(panel.pulse_plot_q)
    assert _axis_label(panel.pulse_plot_i) == ("time", "s")
    assert (_axis_label(panel.template_plot_i)
            == _axis_label(panel.template_plot_q))

    # …and it stays consistent when a view switches the axis meaning.
    panel._set_pulse_x_axis("sample")
    assert _axis_label(panel.pulse_plot_i) == _axis_label(panel.pulse_plot_q)
    assert _axis_label(panel.pulse_plot_i) == ("sample", "")
    panel.close()
    _spin(qt_app)


def _legend_labels(plot):
    legend = plot.getPlotItem().legend
    return [label.text for _sample, label in legend.items]


def test_single_pulse_bands_are_drawn_and_named(qt_app, tmp_path):
    """The bands used to be InfiniteLines, which are not PlotDataItems
    and so never reached the legend."""
    runtime = _FakeRuntime()
    panel = _make_panel(qt_app, tmp_path, runtime)
    rng = np.random.default_rng(7)
    panel._on_start()
    for _ in range(1000):
        _tap1(runtime._pulse_tap, 1, float(rng.normal(0, 1.0)),
              float(rng.normal(0, 1.0)), None)
    assert _spin_until(qt_app, lambda: panel.noise_stats)
    _feed_capture(runtime._pulse_tap, rng)
    assert _spin_until(qt_app, lambda: len(panel._pulse_order) >= 1)

    panel._show_pulse(*panel._pulse_order[-1])
    for plot in (panel.pulse_plot_i, panel.pulse_plot_q):
        labels = _legend_labels(plot)
        assert any("baseline" in t for t in labels), labels
        assert any("trigger" in t and "σ" in t for t in labels), labels
        assert any("end" in t and "σ" in t for t in labels), labels
        # One entry per band pair, not two.
        assert sum("trigger" in t for t in labels) == 1, labels

    panel._on_stop()
    _spin(qt_app)
    panel.close()
    _spin(qt_app)


def test_both_mode_annotates_bands_per_stream(qt_app, tmp_path):
    """In 'both' mode the two streams have different noise, so each
    needs its own bands — a single shared set would be ambiguous about
    which threshold an excursion had to clear."""
    path = _build_dual_file(tmp_path)
    panel = PulseCapturePanel(dark_mode=False)
    panel.load_from_hdf5(path)
    panel._show_pair(*panel._pulse_order[-1])

    for plot in (panel.pulse_plot_i, panel.pulse_plot_q):
        labels = _legend_labels(plot)
        assert any(t.startswith("slow ") and "baseline" in t
                   for t in labels), labels
        assert any(t.startswith("fast ") and "baseline" in t
                   for t in labels), labels
        assert any(t.startswith("slow ") and "trigger" in t
                   for t in labels), labels
        assert any(t.startswith("fast ") and "trigger" in t
                   for t in labels), labels

    panel.close()
    _spin(qt_app)


def test_decision_marks_are_drawn_and_described(qt_app, tmp_path):
    """Trigger and leaky-bucket points, so a wrong-looking capture can
    be read against the decisions that produced it."""
    import pyqtgraph as pg

    runtime = _FakeRuntime()
    panel = _make_panel(qt_app, tmp_path, runtime)
    rng = np.random.default_rng(42)
    panel._on_start()
    for _ in range(1000):
        _tap1(runtime._pulse_tap, 1, float(rng.normal(0, 1.0)),
              float(rng.normal(0, 1.0)), None)
    assert _spin_until(qt_app, lambda: panel.noise_stats)
    _feed_capture(runtime._pulse_tap, rng)
    assert _spin_until(qt_app, lambda: len(panel._pulse_order) >= 1)
    panel._show_pulse(*panel._pulse_order[-1])

    info = panel.pulse_info.text()
    assert "trigger @ sample" in info, info
    assert "end confirmed @" in info, info
    assert "bucket" in info, info

    for name, plot in (("I", panel.pulse_plot_i),
                       ("Q", panel.pulse_plot_q)):
        lines = [it for it in plot.getPlotItem().items
                 if isinstance(it, pg.InfiniteLine)]
        # Vertical only — the horizontal band lines are curves now.
        verticals = [it for it in lines if it.angle == 90]
        assert len(verticals) == 3, f"{name}: {len(verticals)} markers"
        labels = [getattr(it, "label", None) for it in verticals]
        if name == "I":
            texts = [lb.textItem.toPlainText() for lb in labels if lb]
            assert set(texts) == {"trigger", "below threshold",
                                  "end confirmed"}, texts
        else:
            # x-linked, so repeating the labels underneath is noise
            assert all(lb is None for lb in labels)

    panel._on_stop()
    _spin(qt_app)
    panel.close()
    _spin(qt_app)


def test_pair_view_marks_come_from_the_triggered_record(qt_app, tmp_path):
    """In 'both' mode the plots show the UNION ring window, which
    carries no decisions — the marks live on the stream's own triggered
    record and are absolute times, so they still land correctly."""
    import pyqtgraph as pg

    path = _build_dual_file(tmp_path)
    panel = PulseCapturePanel(dark_mode=False)
    panel.load_from_hdf5(path)
    panel._show_pair(*panel._pulse_order[-1])

    for name, plot in (("I", panel.pulse_plot_i),
                       ("Q", panel.pulse_plot_q)):
        verticals = [it for it in plot.getPlotItem().items
                     if isinstance(it, pg.InfiniteLine) and it.angle == 90]
        assert verticals, f"{name}: no decision marks in pair view"
        if name == "I":
            texts = [it.label.textItem.toPlainText()
                     for it in verticals if getattr(it, "label", None)]
            assert any("trigger" in t for t in texts), texts
            assert any(t.startswith(("slow ", "fast ")) for t in texts), \
                texts
    panel.close()
    _spin(qt_app)


def test_both_mode_noise_segment_is_plotted(qt_app):
    """In "both" mode the per-stream noise estimate must draw the
    training-segment plot, pulling the record from the dual session's
    inner stream session (regression: the both-mode branch returned
    before plotting, and the dual session has no top-level noise_data)."""
    from types import SimpleNamespace
    from rfmux.algorithms.measurement.pulse_detection import (
        ChannelNoiseStats,
    )

    panel = PulseCapturePanel(dark_mode=False)
    rng = np.random.default_rng(0)
    arr = (rng.normal(0, 1, 500)
           + 1j * rng.normal(0, 1, 500)).astype(np.complex128)
    panel.task = SimpleNamespace(session=SimpleNamespace(
        slow=SimpleNamespace(noise_data={1: arr}),
        fast=SimpleNamespace(noise_data={})))
    panel.follow_check.setChecked(True)

    panel._on_noise_estimated({
        "stream": "slow",
        "stats": {1: ChannelNoiseStats(std_I=1.0, std_Q=1.0)}})
    assert "Noise training segment (slow)" in panel.pulse_info.text()
    assert "(slow)" in panel.pulse_plot_i.getPlotItem().titleLabel.text
    assert len(panel.pulse_plot_i.getPlotItem().listDataItems()) >= 1

    # The fast stream's estimate lands later and takes over the view.
    panel.task.session.fast.noise_data = {1: arr}
    panel._on_noise_estimated({
        "stream": "fast",
        "stats": {1: ChannelNoiseStats(std_I=2.0, std_Q=2.0)}})
    assert "Noise training segment (fast)" in panel.pulse_info.text()

    panel.task = None
    panel.close()
    _spin(qt_app)


# ── "all" channel selection ───────────────────────────────────────


class _BiasedCRS:
    """CRS stub exposing just the get_biased_channels macro surface."""

    def __init__(self, biased):
        self._biased = list(biased)
        self.calls = []

    async def get_biased_channels(self, module, *, max_channels=None,
                                  threshold=0.0):
        self.calls.append((module, max_channels, threshold))
        return [c for c in self._biased
                if max_channels is None or c <= max_channels]


class _RuntimeWithCRS(_FakeRuntime):
    def __init__(self, crs, is_short_packet=False):
        super().__init__()
        self.crs = crs
        self.is_short_packet = is_short_packet


def _panel_with(qt_app, crs, *, short=False, text="all"):
    runtime = _RuntimeWithCRS(crs, is_short_packet=short)
    panel = PulseCapturePanel(periscope=runtime, dark_mode=False)
    panel.channels_edit.setText(text)
    return panel, runtime


def test_all_resolves_to_biased_channels(qt_app):
    crs = _BiasedCRS([1, 4, 17])
    panel, runtime = _panel_with(qt_app, crs)
    assert panel._parse_channels(runtime=runtime) == [1, 4, 17]
    # Long packets by default, so the whole width is in play.
    module, max_channels, _ = crs.calls[-1]
    assert module == int(panel.module_spin.value())
    assert max_channels == 1024


def test_all_is_bounded_by_short_packet_width(qt_app):
    # 200 is biased but unreachable in short-packet mode.
    crs = _BiasedCRS([1, 200])
    panel, runtime = _panel_with(qt_app, crs, short=True)
    assert panel._parse_channels(runtime=runtime) == [1]
    assert crs.calls[-1][1] == 128


@pytest.mark.parametrize("text", ["all", "ALL", "  All  ", "*"])
def test_all_spellings_accepted(qt_app, text):
    crs = _BiasedCRS([2, 3])
    panel, runtime = _panel_with(qt_app, crs, text=text)
    assert panel._parse_channels(runtime=runtime) == [2, 3]


def test_all_without_a_crs_declines(qt_app):
    panel = PulseCapturePanel(periscope=_FakeRuntime(), dark_mode=False)
    panel.channels_edit.setText("all")
    # quiet: the settings dialog must never pop a modal just to size a
    # buffer estimate.
    assert panel._parse_channels(quiet=True) is None


def test_all_with_nothing_biased_declines(qt_app):
    crs = _BiasedCRS([])
    panel, runtime = _panel_with(qt_app, crs)
    assert panel._parse_channels(runtime=runtime, quiet=True) is None


def test_explicit_channel_list_is_unaffected(qt_app):
    crs = _BiasedCRS([9])
    panel, runtime = _panel_with(qt_app, crs, text="1,2")
    assert panel._parse_channels(runtime=runtime) == [1, 2]
    assert crs.calls == []  # no board round trip for an explicit list


@pytest.mark.parametrize("text,expected", [
    ("2-19", list(range(2, 20))),
    ("1,5-8,20", [1, 5, 6, 7, 8, 20]),
    ("1, 3 - 5", [1, 3, 4, 5]),
])
def test_range_syntax_in_the_channels_field(qt_app, text, expected):
    crs = _BiasedCRS([9])
    panel, runtime = _panel_with(qt_app, crs, text=text)
    assert panel._parse_channels(runtime=runtime) == expected
    assert crs.calls == []  # an explicit spec needs no board round trip


def test_bad_spec_is_reported_not_silently_empty(qt_app):
    panel, runtime = _panel_with(qt_app, _BiasedCRS([1]), text="19-2")
    assert panel._parse_channels(runtime=runtime, quiet=True) is None


# ── status strip stays narrow at high channel counts ──────────────

def _noise(std_i, std_q=None):
    from types import SimpleNamespace
    return SimpleNamespace(mean_I=1.0, std_I=std_i,
                           mean_Q=-1.0, std_Q=std_q if std_q else std_i)


def test_noise_progress_summarises_many_channels(qt_app):
    panel = PulseCapturePanel(dark_mode=False)
    wide = {c: 500 for c in range(1, 201)}
    wide[7] = 120                                    # one straggler
    panel._on_noise_progress({"collected": wide, "target": 1000})

    text = panel.status_label.text()
    assert len(text) < 120, f"{len(text)} chars: {text!r}"
    assert "200 ch" in text
    assert "120/1000" in text, "the slowest channel is the useful number"
    # Nothing is lost — the full listing moves to the tooltip.
    assert panel.status_label.toolTip().count("\n") == 199


def test_noise_progress_still_names_a_few_channels(qt_app):
    panel = PulseCapturePanel(dark_mode=False)
    panel._on_noise_progress({"collected": {1: 10, 2: 20}, "target": 100})
    text = panel.status_label.text()
    assert "Ch1" in text and "Ch2" in text


def test_noise_label_summarises_many_channels(qt_app):
    panel = PulseCapturePanel(dark_mode=False)
    stats = {c: _noise(1.0 + c / 100.0) for c in range(1, 201)}
    panel._on_noise_estimated(stats)

    text = panel.noise_label.text()
    assert len(text) < 160, f"{len(text)} chars: {text!r}"
    assert "200 ch" in text
    assert panel.noise_label.toolTip().count("\n") == 199


def test_noise_label_still_lists_a_few_channels(qt_app):
    panel = PulseCapturePanel(dark_mode=False)
    panel._on_noise_estimated({1: _noise(1.0), 2: _noise(2.0)})
    text = panel.noise_label.text()
    assert "Ch1" in text and "Ch2" in text


def test_status_labels_do_not_drive_panel_width(qt_app):
    # Even if some future message is long, the label's size hint must
    # not become the dock's minimum width.
    panel = PulseCapturePanel(dark_mode=False)
    for label in (panel.status_label, panel.noise_label):
        assert (label.sizePolicy().horizontalPolicy()
                is QtWidgets.QSizePolicy.Policy.Ignored)


def test_capturing_status_summarises_many_channels(qt_app):
    panel = PulseCapturePanel(dark_mode=False)
    panel._both_mode = False
    per_ch = {c: 0 for c in range(1, 201)}
    per_ch[42] = 9
    per_ch[7] = 3
    panel._last_stats = {"total_pulses": 12, "rate_per_min": 4.0,
                         "per_channel": per_ch, "elapsed_s": 65}
    panel._refresh_status_line()

    text = panel.status_label.text()
    assert len(text) < 160, f"{len(text)} chars: {text!r}"
    assert "2/200 ch firing" in text
    assert "Ch42: 9" in text, "the busiest channel is the useful one"
    assert panel.status_label.toolTip().count("\n") == 199


def test_capturing_status_before_any_pulse(qt_app):
    panel = PulseCapturePanel(dark_mode=False)
    panel._both_mode = False
    panel._last_stats = {"total_pulses": 0, "rate_per_min": 0.0,
                         "per_channel": {c: 0 for c in range(1, 201)},
                         "elapsed_s": 3}
    panel._refresh_status_line()
    assert "none firing yet" in panel.status_label.text()


def test_capturing_status_still_lists_a_few_channels(qt_app):
    panel = PulseCapturePanel(dark_mode=False)
    panel._both_mode = False
    panel._last_stats = {"total_pulses": 5, "rate_per_min": 1.0,
                         "per_channel": {1: 2, 2: 3}, "elapsed_s": 10}
    panel._refresh_status_line()
    text = panel.status_label.text()
    assert "Ch1: 2" in text and "Ch2: 3" in text
