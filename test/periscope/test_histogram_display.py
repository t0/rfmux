"""The amplitude histogram labels the units it bins, and its smoothed
ranges start over when the layout is rebuilt."""

import pytest


pytest.importorskip("PyQt6")

import pyqtgraph as pg  # noqa: E402

from test.qt_helpers import bare_periscope  # noqa: E402


def _periscope(unit_mode, real_units, df_calibrations):
    return bare_periscope(module=1, unit_mode=unit_mode,
                          real_units=real_units,
                          df_calibrations=df_calibrations)


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
    p = bare_periscope(channel_list=[],
                       _clear_current_layout=lambda: None,
                       _get_active_modes=lambda: [],
                       _restore_auto_range_settings=lambda: None,
                       _toggle_iqmag=lambda: None,
                       _add_tone_columns=lambda n: None)
    # A range smoothed while binning counts.
    p._smooth_range((1, "I"), -1e5, 1e5)
    assert p._hist_ranges

    p._build_layout()

    # The first frame in the new units sets the range outright.
    lo, hi = p._smooth_range((1, "I"), -1e-3, 1e-3)
    assert (lo, hi) == (-1.1e-3, 1.1e-3)
