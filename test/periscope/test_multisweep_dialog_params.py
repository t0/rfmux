"""What the multisweep dialog hands the task.

The dialog is a view over ``crs.multisweep``'s own arguments: the amplitude
group builds an ``AmplitudeSchedule`` through the constructor each radio names,
and what the dialog emits is what the driver takes. It emits nothing else -- the
Bias Frequency Method combo, the Rotate Saved Data checkbox, the two fit
checkboxes and the fit-frequency option have all gone, because bias frequency
and fits are analyses run on their own buttons.
"""

import inspect

import pytest

pytest.importorskip("PyQt6")

from rfmux.algorithms.measurement.multisweep import multisweep  # noqa: E402
from rfmux.core.resonators import ResonatorCatalog  # noqa: E402
from rfmux.core.transferfunctions import BASE_FREQUENCY  # noqa: E402
from rfmux.tools.periscope.multisweep_dialog import MultisweepDialog  # noqa: E402
from rfmux.tuning import AmplitudeSchedule  # noqa: E402


def _catalog():
    """Two resonators at different amplitudes, so a relative schedule shows."""
    catalog = ResonatorCatalog.from_frequencies(
        [1.0e9, 1.1e9], module=1, amplitude=0.004)
    catalog[catalog.names()[1]].update_bias_point(amplitude=0.008)
    return catalog


def _dialog(qt_app, **params):
    return MultisweepDialog(catalog=_catalog(), dac_scales={1: -0.5},
                            initial_params=params or None)


def test_the_catalog_amplitudes_are_the_default(qt_app):
    """The default sweeps each resonator where the catalog says it is biased."""
    dialog = _dialog(qt_app)
    schedule = dialog.get_parameters()["amp"]

    assert schedule == AmplitudeSchedule()
    assert [step.amplitudes for step in schedule.resolve_steps(dialog.catalog)] == [
        {dialog.catalog.names()[0]: 0.004, dialog.catalog.names()[1]: 0.008}]


def test_one_amplitude_is_one_step_at_that_amplitude(qt_app):
    dialog = _dialog(qt_app)
    dialog.single_radio.setChecked(True)
    dialog.single_amp_edit.setText("0.002")

    schedule = dialog.get_parameters()["amp"]
    assert schedule.nsteps == 1
    assert set(schedule.resolve_steps(dialog.catalog)[0].amplitudes.values()) == {0.002}


def test_a_list_is_an_explicit_schedule(qt_app):
    dialog = _dialog(qt_app)
    dialog.list_radio.setChecked(True)
    dialog.list_amp_edit.setText("0.001, 0.002, 0.004")

    schedule = dialog.get_parameters()["amp"]
    assert schedule == AmplitudeSchedule.explicit([0.001, 0.002, 0.004])


def test_a_ramp_is_a_ramp(qt_app):
    dialog = _dialog(qt_app)
    dialog.ramp_radio.setChecked(True)
    dialog.ramp_start_edit.setText("0.001")
    dialog.ramp_stop_edit.setText("0.008")
    dialog.ramp_steps_edit.setText("4")
    dialog.ramp_spacing.setCurrentText("log")

    assert dialog.get_parameters()["amp"] == AmplitudeSchedule.ramp(
        0.001, 0.008, 4, spacing="log")


def test_multiplicative_keeps_each_resonator_on_its_own_scale(qt_app):
    """The point of the relative schedule: the same factors, different drives."""
    dialog = _dialog(qt_app)
    dialog.multiplicative_radio.setChecked(True)
    dialog.factor_start_edit.setText("0.5")
    dialog.factor_stop_edit.setText("2")
    dialog.factor_steps_edit.setText("3")

    schedule = dialog.get_parameters()["amp"]
    assert schedule == AmplitudeSchedule.multiplicative(0.5, 2.0, 3)
    first, second = dialog.catalog.names()
    steps = schedule.resolve_steps(dialog.catalog)
    assert steps[0].amplitudes[first] == pytest.approx(0.002)
    assert steps[0].amplitudes[second] == pytest.approx(0.004)


def test_both_directions_is_the_sequence_multisweep_takes(qt_app):
    dialog = _dialog(qt_app)
    dialog.upward_cb.setChecked(True)
    dialog.downward_cb.setChecked(True)
    assert dialog.get_parameters()["sweep_direction"] == ("upward", "downward")


def test_one_direction_is_the_string(qt_app):
    dialog = _dialog(qt_app)
    dialog.upward_cb.setChecked(False)
    dialog.downward_cb.setChecked(True)
    assert dialog.get_parameters()["sweep_direction"] == "downward"


def test_no_direction_at_all_is_refused(qt_app):
    """Start is disabled and the status line says why, rather than a modal."""
    dialog = _dialog(qt_app)
    dialog.upward_cb.setChecked(False)
    dialog.downward_cb.setChecked(False)

    assert not dialog.start_btn.isEnabled()
    assert "direction" in dialog.status_label.text()


def test_a_schedule_that_overshoots_full_scale_is_refused(qt_app):
    """Legal factors whose product is not: only resolving them against the
    catalog finds it, which is why validation happens against the array."""
    dialog = _dialog(qt_app)
    dialog.multiplicative_radio.setChecked(True)
    dialog.factor_start_edit.setText("1")
    dialog.factor_stop_edit.setText("500")
    dialog.factor_steps_edit.setText("3")

    assert not dialog.start_btn.isEnabled()
    assert "above full scale" in dialog.status_label.text()
    assert "Step 2" in dialog.status_label.text()


def test_the_summary_is_the_schedules_own_numbers(qt_app):
    """Every number on the line comes from ``describe``, not from the dialog."""
    dialog = _dialog(qt_app)
    dialog.list_radio.setChecked(True)
    dialog.list_amp_edit.setText("0.001, 0.002")
    dialog.upward_cb.setChecked(True)
    dialog.downward_cb.setChecked(True)

    described = dialog.schedule().describe(dialog.catalog, 2, -0.5)
    summary = dialog.summary_label.text()
    assert f"{described['n_sweeps']} sweeps" in summary
    assert f"{described['n_sections']} sections" in summary
    assert f"{described['power_dbm_max']:+.1f} dBm" in summary


def test_the_measurement_name_is_the_files_label(qt_app):
    dialog = _dialog(qt_app)
    dialog.label_edit.setText("a name with spaces")

    assert dialog.get_parameters()["label"] == "a name with spaces"
    assert dialog.filename_label.text().endswith("_a_name_with_spaces.pkl")


def test_custom_frequencies_mint_an_array_of_their_own(qt_app):
    """Typed frequencies become a catalog, which is what multisweep measures;
    it needs an amplitude, and the names are new."""
    dialog = _dialog(qt_app)
    dialog.custom_frequencies_cb.setChecked(True)
    dialog.sections_edit.setText("1000.5, 1100.5")
    dialog.custom_amp_edit.setText("0.003")

    catalog = dialog.get_parameters()["catalog"]
    assert catalog.module == 1
    # On the tone grid, which is where a frequency typed by hand has to land.
    assert [catalog[n].bias.frequency_hz for n in catalog.names()] == [
        pytest.approx(1.0005e9, abs=BASE_FREQUENCY),
        pytest.approx(1.1005e9, abs=BASE_FREQUENCY)]
    assert {catalog[n].bias.amplitude for n in catalog.names()} == {0.003}


def test_a_rerun_is_seeded_with_the_schedule_it_ran(qt_app):
    """Re-run opens on the previous call's own arguments."""
    dialog = _dialog(qt_app, amp=AmplitudeSchedule.multiplicative(0.5, 2.0, 3),
                     span_hz=250e3, npoints_per_sweep=51, nsamps=20,
                     sweep_direction=("upward", "downward"))

    params = dialog.get_parameters()
    assert params["amp"] == AmplitudeSchedule.multiplicative(0.5, 2.0, 3)
    assert params["span_hz"] == pytest.approx(250e3)
    assert params["npoints_per_sweep"] == 51
    assert params["nsamps"] == 20
    assert params["sweep_direction"] == ("upward", "downward")


def test_it_emits_only_arguments_multisweep_accepts(qt_app):
    """The contract that a TypeError before the first sweep is what broke."""
    accepted = set(inspect.signature(multisweep).parameters)
    assert set(_dialog(qt_app).get_parameters()) <= accepted


def test_the_defaults_are_the_drivers_own(qt_app):
    """A default the dialog offers is one the library has."""
    params = _dialog(qt_app).get_parameters()
    defaults = inspect.signature(multisweep).parameters

    assert params["span_hz"] == defaults["span_hz"].default
    assert params["npoints_per_sweep"] == defaults["npoints_per_sweep"].default
    assert params["nsamps"] == defaults["nsamps"].default
