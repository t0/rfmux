"""The fit settings window, as the multisweep panel reads it."""
import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope import settings as periscope_settings  # noqa: E402
from rfmux.tools.periscope.fit_settings_panel import (  # noqa: E402
    ALL_AMPLITUDES,
    BIAS_AMPLITUDE,
    FitSettingsPanel,
)


@pytest.fixture
def panel(qt_app, monkeypatch, tmp_path):
    """A settings window whose saves go nowhere the developer's own do."""
    saved = {}
    monkeypatch.setattr(periscope_settings, "get_fit_parameters", lambda: dict(saved))
    monkeypatch.setattr(periscope_settings, "set_fit_parameters", saved.update)
    return FitSettingsPanel(), saved


def test_both_models_are_fitted_unless_one_is_unchecked(panel):
    """The default is every model this window offers."""
    window, _ = panel
    assert window.get_parameters()["models"] == ("skewed", "nonlinear")


def test_a_model_can_be_dropped_and_the_other_kept(panel):
    """The checkboxes are independent: this is not a radio button."""
    window, _ = panel
    window.set_parameters({"models": ("nonlinear",)})
    assert window.get_parameters()["models"] == ("nonlinear",)


def test_the_settings_are_remembered(panel):
    """A change is written through to where the next session reads it."""
    window, saved = panel
    window._model_checks["skewed"].setChecked(False)
    assert saved["models"] == ("nonlinear",)


def test_the_amplitude_choices_are_the_measurement_s(panel):
    """A step means nothing until something has been swept at it, so the
    multisweep panel says which steps there are."""
    window, _ = panel
    assert window.get_parameters()["amplitude_choice"] is ALL_AMPLITUDES

    window.set_amplitude_choices([("All amplitudes", ALL_AMPLITUDES),
                                  ("At bias amplitude", BIAS_AMPLITUDE),
                                  ("Step 0: -60.0 dBm", 0)])
    window.set_amplitude_choice(0)
    assert window.get_parameters()["amplitude_choice"] == 0


def test_a_step_the_next_measurement_lacks_falls_back(panel):
    """Two steps chosen, then a measurement with one: the choice cannot stand,
    and 'all of them' is the answer that is always true."""
    window, _ = panel
    window.set_amplitude_choices([("All amplitudes", ALL_AMPLITUDES),
                                  ("Step 0", 0), ("Step 1", 1)])
    window.set_amplitude_choice(1)

    window.set_amplitude_choices([("All amplitudes", ALL_AMPLITUDES), ("Step 0", 0)])

    assert window.get_parameters()["amplitude_choice"] is ALL_AMPLITUDES


def test_the_model_to_draw_is_one_the_sweeps_have_fits_for(panel):
    """Nothing fitted is nothing to draw, and the group says so by being dead."""
    window, _ = panel
    assert window.get_display_model() is None
    assert not window.display_group.isEnabled()

    window.set_models_fitted(["skewed", "nonlinear"])

    assert window.get_display_model() == "skewed"
    assert window.display_group.isEnabled()


def test_the_model_to_draw_survives_a_refit(panel):
    """Re-running the fits does not move the tab off what it was showing."""
    window, _ = panel
    window.set_models_fitted(["skewed", "nonlinear"])
    window.display_combo.setCurrentIndex(window.display_combo.findData("nonlinear"))

    window.set_models_fitted(["skewed", "nonlinear"])

    assert window.get_display_model() == "nonlinear"


def test_choosing_a_model_to_draw_asks_for_a_redraw(panel):
    """The only setting here that changes what is on screen says so."""
    window, _ = panel
    window.set_models_fitted(["skewed", "nonlinear"])
    redraws = []
    window.display_model_changed.connect(lambda: redraws.append(True))

    window.display_combo.setCurrentIndex(window.display_combo.findData("nonlinear"))
    assert redraws == [True]

    window._model_checks["skewed"].setChecked(False)
    assert redraws == [True], "the fit settings do not change what is drawn"


def test_the_model_drawn_last_session_is_picked_up(panel, monkeypatch):
    """The combo is empty until a measurement arrives, so the choice has to
    outlive that."""
    window, saved = panel
    window.set_models_fitted(["skewed", "nonlinear"])
    window.display_combo.setCurrentIndex(window.display_combo.findData("nonlinear"))
    assert saved["display_model"] == "nonlinear"

    monkeypatch.setattr(periscope_settings, "get_fit_parameters", lambda: dict(saved))
    next_session = FitSettingsPanel()
    next_session.set_models_fitted(["skewed", "nonlinear"])

    assert next_session.get_display_model() == "nonlinear"
