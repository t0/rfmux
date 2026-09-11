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
