"""The bias settings window, as the multisweep panel reads it."""
import inspect

import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope import settings as periscope_settings  # noqa: E402
from rfmux.tools.periscope.bias_settings_panel import (  # noqa: E402
    DEFAULTS,
    BiasSettingsPanel,
)
from rfmux.tuning.bias import find_bias_points  # noqa: E402


@pytest.fixture
def panel(qt_app, monkeypatch):
    """A settings window whose saves go nowhere the developer's own do."""
    saved = {}
    monkeypatch.setattr(periscope_settings, "get_bias_parameters", lambda: dict(saved))
    monkeypatch.setattr(periscope_settings, "set_bias_parameters", saved.update)
    return BiasSettingsPanel(), saved


def test_the_settings_are_the_finder_s_arguments(panel):
    """Everything this window produces is something ``find_bias_points``
    takes, so the press is a splat and not a translation."""
    window, _ = panel
    arguments = set(inspect.signature(find_bias_points).parameters)
    assert set(window.get_parameters()) <= arguments


def test_a_fresh_window_asks_for_what_the_library_would_do(panel):
    """No saved settings means the library's own defaults, which is what the
    reset button puts back."""
    window, _ = panel
    assert window.get_parameters() == DEFAULTS


def test_reset_puts_the_library_s_defaults_back(panel):
    """Whatever was chosen, Reset to Defaults is the finder's signature."""
    window, _ = panel
    window.prominence_spin.setValue(0.9)
    window.noise_gate_spin.setValue(20.0)
    window._select(window.method_combo, "hysteresis")
    window._reset()
    assert window.get_parameters() == DEFAULTS


def test_a_changed_setting_is_remembered(panel):
    """A change is written through to where the next session reads it."""
    window, saved = panel
    window.prominence_spin.setValue(0.8)
    assert saved["spike_prominence_factor"] == 0.8


def test_last_session_s_settings_are_what_the_window_opens_with(panel, monkeypatch):
    """Not the library's defaults: what was chosen last time."""
    _, _ = panel
    monkeypatch.setattr(periscope_settings, "get_bias_parameters",
                        lambda: {"noise_gate_factor": 20.0,
                                 "amplitude_method": "derivative"})
    monkeypatch.setattr(periscope_settings, "set_bias_parameters", lambda d: None)
    window = BiasSettingsPanel()
    assert window.get_parameters()["noise_gate_factor"] == 20.0
    assert window.get_parameters()["amplitude_method"] == "derivative"


def test_a_distance_guard_in_hertz_is_the_number_typed(panel):
    """The absolute field is kilohertz on screen and hertz to the library."""
    window, _ = panel
    window._distance_radios["absolute"].setChecked(True)
    window.absolute_spin.setValue(12.5)
    assert window.get_parameters()["max_distance_hz"] == 12.5e3


def test_a_fractional_guard_is_a_fraction_of_the_span_swept(panel):
    """The same setting means the same thing on a measurement swept at
    another span, which is the point of expressing it this way."""
    window, _ = panel
    window._distance_radios["fraction"].setChecked(True)
    window.fraction_spin.setValue(0.25)
    assert window.get_parameters(span_hz=70e3)["max_distance_hz"] == 17.5e3
    assert window.get_parameters(span_hz=200e3)["max_distance_hz"] == 50e3


def test_a_fraction_with_no_span_to_measure_is_no_guard(panel):
    """A fraction of an unknown span is not a distance, so it is not passed
    off as one."""
    window, _ = panel
    window._distance_radios["fraction"].setChecked(True)
    assert window.get_parameters(span_hz=None)["max_distance_hz"] is None


def test_no_limit_is_how_the_guard_is_switched_off(panel):
    """Which is the library's default, and a mode of its own rather than a
    magic number in one of the fields."""
    window, _ = panel
    window._distance_radios["absolute"].setChecked(True)
    window._distance_radios["none"].setChecked(True)
    assert window.get_parameters(span_hz=70e3)["max_distance_hz"] is None


def test_which_distance_field_was_meant_survives_a_session(panel):
    """Hertz alone cannot say whether a fraction or an absolute was typed, so
    the mode is saved beside it."""
    window, saved = panel
    window._distance_radios["fraction"].setChecked(True)
    window.fraction_spin.setValue(0.3)
    assert saved["max_distance_mode"] == "fraction"
    assert saved["max_distance_fraction"] == 0.3


def test_a_one_direction_sweep_cannot_run_the_tests_that_compare_two(panel):
    """'both' and 'hysteresis' compare an upward sweep against a downward one.
    Rather than let the press fail, the window drops to the test that reads a
    single trace."""
    window, _ = panel
    assert window.get_parameters()["amplitude_method"] == "both"
    window.set_directions_swept(["upward"])
    assert window.get_parameters()["amplitude_method"] == "derivative"


def test_both_directions_leave_the_choice_alone(panel):
    """Nothing is taken away from a measurement that can answer for it."""
    window, _ = panel
    window.set_directions_swept(["upward", "downward"])
    assert window.get_parameters()["amplitude_method"] == "both"


def test_a_direction_that_was_not_swept_is_not_offered(panel):
    """Measuring the calibration on a sweep that does not exist is not a
    setting; automatic is."""
    window, _ = panel
    window._select(window.direction_combo, "downward")
    window.set_directions_swept(["upward"])
    assert window.get_parameters()["direction"] is None


def test_only_the_chosen_test_s_settings_are_live(panel):
    """The groups say what controls what: a setting the chosen test does not
    read is greyed out rather than quietly ignored."""
    window, _ = panel
    window._select(window.method_combo, "derivative")
    window._update_enabled()
    assert window.derivative_group.isEnabled()
    assert not window.hysteresis_group.isEnabled()

    window._select(window.method_combo, "hysteresis")
    window._update_enabled()
    assert not window.derivative_group.isEnabled()
    assert window.hysteresis_group.isEnabled()
