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
    dlg.fit_method_combo.setCurrentIndex(0)
    assert dlg.get_parameters()["fit_method"] == "nonlinear"
