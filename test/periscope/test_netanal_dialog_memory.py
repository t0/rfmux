"""The network analysis dialog remembers its fields between opens."""

import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore  # noqa: E402

from test.qt_helpers import spin  # noqa: E402

from rfmux.tools.periscope.network_analysis_dialog import (  # noqa: E402
    NetworkAnalysisDialog,
)


def _dialog(settings):
    return NetworkAnalysisDialog(
        modules=list(range(1, 9)), dac_scales={m: -0.5 for m in range(1, 9)},
        settings=settings)


def test_the_dialog_remembers_its_values(qt_app, tmp_path):
    settings = QtCore.QSettings(str(tmp_path / "periscope.ini"),
                                QtCore.QSettings.Format.IniFormat)
    dlg = _dialog(settings)
    dlg.fmin_edit.setText("200")
    dlg.fmax_edit.setText("800")
    dlg.cable_length_edit.setText("12.5")
    dlg.amp_edit.setText("1/1000, 0.01")
    dlg.points_edit.setText("1234")
    dlg.samples_edit.setText("30")
    dlg.max_chans_edit.setText("64")
    dlg.max_span_edit.setText("50")
    dlg.clear_channels_cb.setChecked(False)
    dlg.accept()

    again = _dialog(settings)
    assert again.fmin_edit.text() == "200"
    assert again.amp_edit.text() == "1/1000, 0.01"
    params = again.get_parameters()
    assert (params["fmin"], params["fmax"], params["cable_length"]) == \
        (200e6, 800e6, 12.5)
    assert params["amps"] == [0.001, 0.01]
    assert (params["npoints"], params["nsamps"], params["max_chans"],
            params["max_span"]) == (1234, 30, 64, 50e6)
    assert params["clear_channels"] is False

    # Cancel keeps what was last started with.
    again.fmin_edit.setText("300")
    again.reject()
    assert _dialog(settings).fmin_edit.text() == "200"


def test_the_multisweep_dialog_remembers_its_values(qt_app, tmp_path):
    """Every sweep setting comes back, the amplitude generator's fields
    included; the defaults of those are powers in dBm."""
    from rfmux.tools.periscope.multisweep_dialog import MultisweepDialog

    settings = QtCore.QSettings(str(tmp_path / "periscope.ini"),
                                QtCore.QSettings.Format.IniFormat)
    make = lambda: MultisweepDialog(
        section_center_frequencies=[100e6], dac_scales={1: -0.5},
        current_module=1, initial_params={"amps": [0.01]}, settings=settings)
    dlg = make()
    assert (dlg.start_amp_edit.text(), dlg.stop_amp_edit.text()) == ("-65", "-40")
    dlg.span_khz_edit.setText("50")
    dlg.npoints_edit.setText("77")
    dlg.start_amp_edit.setText("-70")
    dlg.iterations_amp_edit.setText("5")
    dlg.sweep_direction_combo.setCurrentIndex(1)
    dlg.apply_nonlinear_fit_checkbox.setChecked(True)
    dlg.accept()

    again = make()
    assert again.span_khz_edit.text() == "50"
    assert again.npoints_edit.text() == "77"
    assert (again.start_amp_edit.text(), again.iterations_amp_edit.text()) == \
        ("-70", "5")
    assert again.sweep_direction_combo.currentIndex() == 1
    assert again.apply_nonlinear_fit_checkbox.isChecked() is True
    again.close()
    spin(qt_app)
