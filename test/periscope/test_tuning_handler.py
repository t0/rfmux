"""The main window holds a module's tuning rows and converts its plots
with the df calibration of every row that has one."""

import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope.app import Periscope  # noqa: E402


def test_the_rows_are_held_and_the_calibrations_derived(qt_app, capsys):
    p = Periscope.__new__(Periscope)
    p.tuning, p.df_calibrations = {}, {}
    rows = {3: {"bias_channel": 3, "df_calibration": 1 + 1j,
                "bias_frequency": 1e9},
            7: {"bias_channel": 7, "df_calibration": None}}
    p._handle_tuning_ready(2, rows)
    assert p.tuning == {2: rows}
    assert p.df_calibrations == {2: {3: 1 + 1j}}
    assert "2 detectors on module 2, 1 with a df calibration" in \
        capsys.readouterr().out
