"""In df units every axis is in hertz, and a spectrum is a power density.

The calibration is Hz/V, so both quadratures come out of
``apply_iq_conversion`` in hertz -- neither axis is dimensionless -- and
the spectrum the PSD task returns in this mode is linear power, never an
amplitude density.
"""

import pytest

pytest.importorskip("PyQt6")

import pyqtgraph as pg  # noqa: E402

from rfmux.tools.periscope.app import Periscope  # noqa: E402


def _calibrated():
    p = Periscope.__new__(Periscope)
    p.module = 1
    p.unit_mode = "df"
    p.real_units = False
    p.df_calibrations = {1: {1: 1.0 + 0j}}
    return p


@pytest.mark.parametrize("mode_key, axis", [
    ("T", "left"), ("IQ", "bottom"), ("IQ", "left"), ("F", "left"),
])
def test_every_df_axis_is_in_hertz(qt_app, mode_key, axis):
    pw = pg.PlotWidget()
    _calibrated()._configure_plot_axes(pw, mode_key, [1])
    assert pw.getAxis(axis).labelUnits == "Hz"


@pytest.mark.parametrize("mode_key", ["S", "D"])
def test_the_df_spectrum_is_labelled_a_power_density(qt_app, mode_key):
    pw = pg.PlotWidget()
    _calibrated()._configure_plot_axes(pw, mode_key, [1])
    assert "PSD (Hz²/Hz)" in pw.getAxis("left").labelText
