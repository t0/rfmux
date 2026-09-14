"""df units are offered only when every channel on screen can be drawn in them.

A calibration is measured at the tone a detector is biased at, so it
arrives with the applied bias and never from picking a units option.
Until it has, the radio is dead: the main window neither measures nor
draws half a row in hertz.
"""

import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope.app import Periscope  # noqa: E402
from test.qt_helpers import bare_periscope  # noqa: E402


def test_df_units_need_every_channel_on_screen(qt_app, monkeypatch):
    p = bare_periscope(monkeypatch)
    p.module, p.all_chs = 1, [1, 2]
    p.unit_mode, p.real_units = "counts", False
    p._handle_tuning_ready(1, {1: {"df_calibration": 1 + 0j}})
    assert not p._df_units_available(), "channel 2 has none"
    p._handle_tuning_ready(1, {1: {"df_calibration": 1 + 0j},
                               2: {"df_calibration": 2 + 0j}})
    assert p._df_units_available()


def test_losing_a_calibration_steps_out_of_df_units(qt_app, monkeypatch):
    from rfmux.tools.periscope.utils import QtWidgets

    p = bare_periscope(monkeypatch)
    p.module, p.all_chs = 1, [1]
    p.rb_df_units = QtWidgets.QRadioButton("df Units")
    p.rb_counts = QtWidgets.QRadioButton("Counts")
    p._build_layout = lambda: None
    p._handle_tuning_ready(1, {1: {"df_calibration": 1 + 0j}})
    p.unit_mode, p.real_units = "df", False
    assert p.rb_df_units.isEnabled()

    # The array is regenerated under it.
    p.df_calibrations.pop(1)
    p._update_df_units_enabled()
    assert not p.rb_df_units.isEnabled()
    assert p.unit_mode == "counts" and p.rb_counts.isChecked()


def test_changing_units_measures_nothing(qt_app):
    """The switch is a switch: no sweep hides behind it."""
    assert not hasattr(Periscope, "_measure_df_calibrations")
