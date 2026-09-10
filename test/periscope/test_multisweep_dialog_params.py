"""What the multisweep dialog hands the task.

The dialog is a view over ``crs.multisweep``'s own arguments, so what it emits
is what the library takes: a scalar ``amp`` for one amplitude and an
``AmplitudeSchedule`` for several, and a sequence of both directions rather
than a "both" flag of its own.

It also emits nothing the library does not accept. The Bias Frequency Method
combo, the Rotate Saved Data checkbox and the two fit checkboxes have gone:
bias frequency and fits are analyses now, run on their own buttons, and
nothing on this branch rotates a saved sweep.
"""

import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope.multisweep_dialog import MultisweepDialog  # noqa: E402
from rfmux.tuning import AmplitudeSchedule  # noqa: E402


def _dialog(qt_app, amplitudes="0.001", direction="Upward"):
    dialog = MultisweepDialog(section_center_frequencies=[1.0e9, 1.1e9],
                              dac_scales={1: -0.5}, current_module=1)
    dialog.amp_edit.setText(amplitudes)
    dialog.sweep_direction_combo.setCurrentText(direction)
    return dialog


def test_one_amplitude_is_a_number(qt_app):
    params = _dialog(qt_app, amplitudes="0.001").get_parameters()
    assert params["amp"] == pytest.approx(0.001)


def test_several_amplitudes_are_a_schedule(qt_app):
    """A list is a schedule of steps, which multisweep sweeps in one call."""
    params = _dialog(qt_app, amplitudes="0.001, 0.002, 0.004").get_parameters()

    assert isinstance(params["amp"], AmplitudeSchedule)
    assert params["amp"].steps == pytest.approx((0.001, 0.002, 0.004))
    assert not params["amp"].relative        # absolute levels, not factors


def test_both_directions_is_the_sequence_multisweep_takes(qt_app):
    params = _dialog(qt_app, direction="Both").get_parameters()
    assert params["sweep_direction"] == ("upward", "downward")


def test_one_direction_is_the_string(qt_app):
    params = _dialog(qt_app, direction="Downward").get_parameters()
    assert params["sweep_direction"] == "downward"


def test_a_rerun_seeded_with_both_directions_still_says_both(qt_app):
    """The re-run seeds the dialog from the previous call's own arguments."""
    assert MultisweepDialog._direction_text(("upward", "downward")) == "Both"
    assert MultisweepDialog._direction_text("downward") == "Downward"


def test_nothing_is_emitted_that_multisweep_would_refuse(qt_app):
    params = _dialog(qt_app).get_parameters()
    for gone in ("bias_frequency_method", "rotate_saved_data",
                 "apply_skewed_fit", "apply_nonlinear_fit"):
        assert gone not in params
