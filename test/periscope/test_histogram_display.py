"""The amplitude histogram labels the units it bins, and its smoothed
ranges start over when the layout is rebuilt."""

import pytest


pytest.importorskip("PyQt6")

import pyqtgraph as pg  # noqa: E402
from PyQt6 import QtWidgets  # noqa: E402

from rfmux.tools.periscope.app import Periscope  # noqa: E402


def _periscope(unit_mode, real_units, df_calibrations):
    p = Periscope.__new__(Periscope)
    p.module = 1
    p.unit_mode = unit_mode
    p.real_units = real_units
    p.df_calibrations = df_calibrations
    return p


def test_histogram_axis_says_df_units_when_binning_df(qt_app):
    p = _periscope("df", False, {1: {1: 1.0 + 0j}})
    pw = pg.PlotWidget()
    p._configure_plot_axes(pw, "H", [1])
    assert pw.getAxis("bottom").labelUnits == "Hz or unitless"


def test_histogram_axis_falls_back_to_counts_without_calibration(qt_app):
    p = _periscope("df", False, {1: {}})
    pw = pg.PlotWidget()
    p._configure_plot_axes(pw, "H", [1])
    assert pw.getAxis("bottom").labelUnits == "Counts"


def test_layout_rebuild_forgets_smoothed_histogram_ranges(qt_app):
    p = Periscope.__new__(Periscope)
    QtWidgets.QMainWindow.__init__(p)
    p.channel_list = []
    p._clear_current_layout = lambda: None
    p._get_active_modes = lambda: []
    p._restore_auto_range_settings = lambda: None
    p._toggle_iqmag = lambda: None
    # A range smoothed while binning counts.
    p._smooth_range((1, "I"), -1e5, 1e5)
    assert p._hist_ranges

    p._build_layout()

    # The first frame in the new units sets the range outright.
    lo, hi = p._smooth_range((1, "I"), -1e-3, 1e-3)
    assert (lo, hi) == (-1.1e-3, 1.1e-3)
