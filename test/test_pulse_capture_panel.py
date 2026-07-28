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

    def register_pulse_tap(self, callback):
        self._pulse_tap = callback

    def unregister_pulse_tap(self):
        self._pulse_tap = None


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
    panel = PulseCapturePanel(periscope=runtime, dark_mode=True)
    panel._browse_dir = str(tmp_path)
    panel.channels_edit.setText("1")
    panel.threshold_spin.setValue(5.0)
    panel.end_spin.setValue(1.5)
    return panel


def _feed_capture(tap, rng, n=3000, pulse_starts=(100, 900, 1700),
                  tau_samples=40, amp=60.0):
    signal = rng.normal(0, 1.0, n)
    k = np.arange(n)
    for k0 in pulse_starts:
        m = k >= k0
        signal[m] += amp * np.exp(-(k[m] - k0) / tau_samples)
    for i in range(n):
        tap(1, float(signal[i]), float(rng.normal(0, 1.0)), i * DT)


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
        runtime._pulse_tap(1, float(rng.normal(0, 1.0)),
                           float(rng.normal(0, 1.0)), None)
    assert _spin_until(qt_app, lambda: panel.noise_stats), \
        "noise_estimated signal never arrived"
    assert "Noise:" in panel.noise_label.text()

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
