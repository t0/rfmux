"""The Fit Results tab's toolbar: which fit is drawn, and over which sweeps."""
import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope.fit_display_toolbar import FitDisplayToolbar  # noqa: E402
from rfmux.tools.periscope.fit_settings_panel import (  # noqa: E402
    ALL_AMPLITUDES,
    BIAS_AMPLITUDE,
)


@pytest.fixture
def toolbar(qt_app):
    """A toolbar saving into this test's own settings file."""
    return FitDisplayToolbar()


def test_nothing_fitted_is_nothing_to_draw(toolbar):
    """The combo is dead until the sweeps carry fits of something."""
    assert toolbar.get_model() is None
    assert not toolbar.model_combo.isEnabled()

    toolbar.set_models_fitted(["skewed", "nonlinear"])

    assert toolbar.get_model() == "skewed"
    assert toolbar.model_combo.isEnabled()


def test_the_model_drawn_survives_a_refit(toolbar):
    """Re-running the fits does not move the tab off what it was showing."""
    toolbar.set_models_fitted(["skewed", "nonlinear"])
    toolbar.model_combo.setCurrentIndex(toolbar.model_combo.findData("nonlinear"))

    toolbar.set_models_fitted(["skewed", "nonlinear"])

    assert toolbar.get_model() == "nonlinear"


def test_choosing_a_model_asks_for_a_redraw(toolbar):
    """Both controls change what is on screen, so both say so."""
    toolbar.set_models_fitted(["skewed", "nonlinear"])
    toolbar.set_amplitude_choices([("All amplitudes", ALL_AMPLITUDES), ("Step 0", 0)])
    redraws = []
    toolbar.display_changed.connect(lambda: redraws.append(True))

    toolbar.model_combo.setCurrentIndex(toolbar.model_combo.findData("nonlinear"))
    toolbar.amplitude_combo.setCurrentIndex(toolbar.amplitude_combo.findData(0))

    assert redraws == [True, True]


def test_every_amplitude_is_the_default(toolbar):
    """A tab that has been told nothing draws everything that was fitted."""
    assert toolbar.get_amplitude() is ALL_AMPLITUDES


def test_a_step_the_next_measurement_lacks_falls_back(toolbar):
    """A step is a step of one measurement's schedule; 'all of them' is the
    answer that is always true."""
    toolbar.set_amplitude_choices([("All amplitudes", ALL_AMPLITUDES),
                                   ("Step 0", 0), ("Step 1", 1)])
    toolbar.amplitude_combo.setCurrentIndex(toolbar.amplitude_combo.findData(1))

    toolbar.set_amplitude_choices([("All amplitudes", ALL_AMPLITUDES), ("Step 0", 0)])

    assert toolbar.get_amplitude() is ALL_AMPLITUDES


def test_what_was_drawn_last_session_is_picked_up(qt_app):
    """The combos are empty until a measurement arrives, so the choices have to
    outlive that."""
    toolbar = FitDisplayToolbar()
    toolbar.set_models_fitted(["skewed", "nonlinear"])
    toolbar.set_amplitude_choices([("All amplitudes", ALL_AMPLITUDES), ("Step 1", 1)])
    toolbar.model_combo.setCurrentIndex(toolbar.model_combo.findData("nonlinear"))
    toolbar.amplitude_combo.setCurrentIndex(toolbar.amplitude_combo.findData(1))

    next_session = FitDisplayToolbar()
    next_session.set_models_fitted(["skewed", "nonlinear"])
    next_session.set_amplitude_choices(
        [("All amplitudes", ALL_AMPLITUDES), ("Step 1", 1)])

    assert next_session.get_model() == "nonlinear"
    assert next_session.get_amplitude() == 1


def test_the_bias_amplitude_is_offered_only_when_there_is_one(toolbar):
    """'At bias' means the step a resonator was biased at, which is nothing
    until a bias has been found; the multisweep panel says when it has."""
    toolbar.set_amplitude_choices([("All amplitudes", ALL_AMPLITUDES), ("Step 0", 0)])
    assert toolbar.amplitude_combo.findData(BIAS_AMPLITUDE) == -1

    toolbar.set_amplitude_choices([("All amplitudes", ALL_AMPLITUDES),
                                   ("At bias amplitude", BIAS_AMPLITUDE),
                                   ("Step 0", 0)])
    toolbar.amplitude_combo.setCurrentIndex(
        toolbar.amplitude_combo.findData(BIAS_AMPLITUDE))

    assert toolbar.get_amplitude() == BIAS_AMPLITUDE
