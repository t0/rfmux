"""The Bias KIDs dialog's fit choice reaches bias_kids, preselected to
the fit the sweeps carry."""
import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope.bias_kids_dialog import BiasKidsDialog  # noqa: E402


def test_fit_choice_defaults_to_nonlinear(qt_app):
    assert BiasKidsDialog(None, 1).get_parameters()["fit_method"] == "nonlinear"
    both = BiasKidsDialog(None, 1, fits_present={"nonlinear", "skewed"})
    assert both.get_parameters()["fit_method"] == "nonlinear"


def test_fit_choice_follows_the_only_fit_present(qt_app):
    dlg = BiasKidsDialog(None, 1, fits_present={"skewed"})
    assert dlg.get_parameters()["fit_method"] == "skewed"
    # The threshold reads the nonlinear fit's parameter, so it is greyed
    # out while the skewed fit is chosen.
    assert not dlg.nonlinear_threshold_spin.isEnabled()
    dlg.fit_method_combo.setCurrentIndex(0)
    assert dlg.get_parameters()["fit_method"] == "nonlinear"
    assert dlg.nonlinear_threshold_spin.isEnabled()


def test_the_calibration_options_match_headless_bias_kids(qt_app):
    """The dialog offers the measured-or-fitted choice and the step, at
    the headless defaults; the step greys out under the fit's."""
    dlg = BiasKidsDialog(None, 1)
    p = dlg.get_parameters()
    assert p["measure_calibration"] is True
    assert p["calibration_step"] == 0.05
    dlg.cal_step_spin.setValue(0.1)
    dlg.measure_cal_checkbox.setChecked(False)
    p = dlg.get_parameters()
    assert p["measure_calibration"] is False
    assert p["calibration_step"] == 0.1
    assert not dlg.cal_step_spin.isEnabled()
