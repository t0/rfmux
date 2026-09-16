"""The IQ Plane tab: the pulse over the sweep it was tuned with, in
one frame, in every view."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("h5py")

from test.qt_helpers import spin  # noqa: E402

from rfmux.core.transferfunctions import VOLTS_PER_ROC  # noqa: E402
from rfmux.pulse_capture.detection import ChannelNoiseStats  # noqa: E402
from rfmux.pulse_capture.hdf5 import PulseHDF5Writer  # noqa: E402
from rfmux.tools.periscope.pulse_capture_panel import (  # noqa: E402
    PulseCapturePanel, UNITS_DF, UNITS_VOLTS)

F0 = 1.0e9
PHASE = 30.0
CAL = 2.0e6 * np.exp(1j * np.radians(-40.0))


def _resonance():
    """A dip in counts, swept at ADC phase zero."""
    f = np.linspace(F0 - 5e4, F0 + 5e4, 41)
    iq = 1000.0 * (1 - 0.8 / (1 + 2j * 20000 * (f / F0 - 1)))
    return f, iq


def _write_capture(path):
    f, iq = _resonance()
    bias = F0 + 1250.0
    turned = iq * np.exp(-1j * np.radians(PHASE))
    point = complex(np.interp(bias, f, turned.real),
                    np.interp(bias, f, turned.imag))
    base = point * VOLTS_PER_ROC          # stored in volts, quadratures
    n = 40
    k = np.arange(n)
    step = np.where(k >= 10, np.exp(-(k - 10) / 8.0), 0.0)
    amp = base + 2e-5 * step * np.conj(CAL) / abs(CAL)
    writer = PulseHDF5Writer(
        path, [1], {1: ChannelNoiseStats(mean_I=base.real, mean_Q=base.imag,
                                         std_I=1e-6, std_Q=1e-6)},
        {"streamer_mode": "slow", "sample_rate_slow": 596.0, "module": 1,
         "volts_per_count": VOLTS_PER_ROC},
        tuning={1: {"frequencies": f, "iq_complex": iq, "bias_frequency": bias,
                    "optimal_phase_degrees": PHASE, "df_calibration": CAL}},
        stored_units={1: "V"})
    writer.append_pulse(1, 1, {
        "Amp_I": amp.real, "Amp_Q": amp.imag, "Time": k / 596.0,
        "trigger_index": 10, "trigger_baseline_I": base.real,
        "trigger_baseline_Q": base.imag})
    writer.finalize()
    return turned, base


def _named(panel):
    return {item.name(): item.getData()
            for item in panel.iq_plot.getPlotItem().listDataItems()
            if item.name()}


def test_the_pulse_and_the_sweep_share_one_frame(qt_app, tmp_path):
    """The baseline sits on the bias point whichever view is up, the
    sweep is turned by minus the phase the bias set, and the frequency
    direction is the df axis of the df view."""
    path = tmp_path / "plane.h5"
    turned, base = _write_capture(path)
    panel = PulseCapturePanel(dark_mode=False)
    panel.load_from_hdf5(path)
    spin(qt_app)

    panel.units_combo.setCurrentText(UNITS_VOLTS)
    spin(qt_app)
    items = _named(panel)
    x, y = items["tuning sweep"]
    np.testing.assert_allclose(x + 1j * y, turned * VOLTS_PER_ROC)
    assert items["bias point"][0][0] == pytest.approx(base.real)
    assert items["trigger baseline"][0][0] == pytest.approx(base.real)
    assert items["trigger baseline"][1][0] == pytest.approx(base.imag)
    assert panel.iq_plot.getPlotItem().getAxis("bottom").labelText == "I (V)"

    panel.units_combo.setCurrentText(UNITS_DF)
    spin(qt_app)
    items = _named(panel)
    bx, by = items["bias point"]
    tx, ty = items["trigger baseline"]
    assert (tx[0], ty[0]) == pytest.approx((bx[0], by[0]))
    dx, dy = items["frequency direction"]
    assert dx[1] > dx[0] and dy[1] == pytest.approx(dy[0], abs=1e-6)
    assert panel.iq_plot.getPlotItem().getAxis("left").labelText == \
        "dissipation (Hz)"
    assert "bias 1000.001250 MHz" in panel.iq_info.toolTip() + panel.iq_info.text()
    panel.close()
    spin(qt_app)


def test_a_long_pulse_keeps_its_path_and_thins_its_markers(qt_app, tmp_path):
    """A fast-mode pulse is hundreds of thousands of samples: the line
    keeps them all, the time-coloured markers are a fixed budget."""
    import pyqtgraph as pg
    from rfmux.tools.periscope.pulse_capture_panel import IQ_PLANE_POINTS

    path = tmp_path / "plane.h5"
    _write_capture(path)
    panel = PulseCapturePanel(dark_mode=False)
    panel.load_from_hdf5(path)
    n = 50_000
    k = np.arange(n, dtype=float)
    panel._get_waveform = lambda *a, **kw: {
        "Amp_I": 1e-3 + 1e-5 * np.exp(-k / 8e3), "Amp_Q": np.full(n, -2e-3),
        "Time": k / 2.44e6, "trigger_index": 100}
    panel._current_view = (1, 1)
    panel._render_iq_plane()
    spin(qt_app)
    assert len(_named(panel)["pulse"][0]) == n
    markers = [i for i in panel.iq_plot.getPlotItem().items
               if isinstance(i, pg.ScatterPlotItem)]
    assert len(markers) == 1 and len(markers[0].data) == IQ_PLANE_POINTS
    panel.close()
    spin(qt_app)


def test_without_a_sweep_the_plane_says_so(qt_app, tmp_path):
    path = tmp_path / "bare.h5"
    PulseHDF5Writer(path, [1], {1: ChannelNoiseStats()},
                    {"streamer_mode": "slow"}).finalize()
    panel = PulseCapturePanel(dark_mode=False)
    panel.load_from_hdf5(path)
    spin(qt_app)
    assert "no sweep in the tuning" in panel.iq_info.toolTip() + panel.iq_info.text()
    assert not _named(panel)
    panel.close()
    spin(qt_app)
