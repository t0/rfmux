"""Test that the multisweep dialog emits valid driver arguments and schedules."""

import inspect
import math

import pytest

pytest.importorskip("PyQt6")

from rfmux.algorithms.measurement.multisweep import multisweep  # noqa: E402
from rfmux.core.resonators import ResonatorCatalog  # noqa: E402
from rfmux.core.dac_scale import DAC_SCALE_OUTPUT_RANGE_DBM  # noqa: E402
from rfmux.core.transferfunctions import BALUN_LOSS_DB, BASE_FREQUENCY  # noqa: E402
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
    dialog = MultisweepDialog(module=1, dac_scales={1: -0.5})
    dialog.center_source_combo.setCurrentIndex(2)
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


@pytest.mark.parametrize("schedule", [
    AmplitudeSchedule.ramp(-60, -40, 3, units="dbm"),
    AmplitudeSchedule.ramp(0.001, 0.01, 4, spacing="linear"),
    AmplitudeSchedule(-50, units="dbm"),
    AmplitudeSchedule.multiplicative(0.5, 2.0, 3, spacing="linear"),
], ids=["dbm-ramp", "linear-ramp", "one-amplitude", "linear-multiples"])
def test_a_rerun_opens_on_the_same_kind_of_schedule(qt_app, schedule):
    """Re-run gives the same schedule, in the same units, and selects the
    radio button that made it."""
    dialog = _dialog(qt_app, amp=schedule)

    rerun = dialog.get_parameters()["amp"]
    assert rerun == schedule
    assert (rerun.units, rerun.spacing) == (schedule.units, schedule.spacing)


def test_it_emits_only_arguments_multisweep_accepts(qt_app):
    """The contract that a TypeError before the first sweep is what broke.
    The task uses ``dac_scale_setting_dbm`` before the sweep."""
    accepted = set(inspect.signature(multisweep).parameters)
    assert set(_dialog(qt_app).get_parameters()) <= accepted | {
        "dac_scale_setting_dbm"}


def test_the_defaults_are_the_drivers_own(qt_app):
    """A default the dialog offers is one the library has."""
    params = _dialog(qt_app).get_parameters()
    defaults = inspect.signature(multisweep).parameters

    assert params["span_hz"] == defaults["span_hz"].default
    assert params["npoints_per_sweep"] == defaults["npoints_per_sweep"].default
    assert params["nsamps"] == defaults["nsamps"].default


def test_previous_centers_preserve_identity_and_current_amplitudes(qt_app):
    catalog = _catalog()
    for resonator in catalog:
        resonator.update_bias_point(bifurcated_at=0.01, dI_df=1e-9, dQ_df=2e-9)
    previous = {"results": {
        0: {"upward": {
            r.name: {"original_center_frequency": r.bias.frequency_hz}
            for r in catalog}},
        1: {"downward": {
            r.name: {"original_center_frequency": r.bias.frequency_hz + 123.45}
            for r in catalog}},
    }}
    original = [(r.name, r.channel, r.bias) for r in catalog]
    dialog = MultisweepDialog(catalog=catalog, previous_sweeps=previous,
                              dac_scales={1: -0.5})
    assert dialog.get_parameters()["catalog"] is catalog
    dialog.center_source_combo.setCurrentIndex(1)
    params = dialog.get_parameters()
    result = params["catalog"]
    assert [(r.name, r.channel, r.bias.amplitude) for r in result] == [
        (name, channel, bias.amplitude) for name, channel, bias in original]
    assert [(r.name, r.channel, r.bias) for r in result] == original
    assert params["center_frequencies"] == {
        name: bias.frequency_hz + 123.45 for name, _, bias in original}
    assert [(r.name, r.channel, r.bias) for r in catalog] == original


def test_previous_centers_require_a_previous_measurement(qt_app):
    dialog = _dialog(qt_app)
    dialog.center_source_combo.setCurrentIndex(1)
    assert not dialog.start_btn.isEnabled()
    assert "No previous multisweep centers" in dialog.status_label.text()


def test_previous_centers_require_every_selected_resonator(qt_app):
    dialog = _dialog(qt_app)
    dialog.previous_sweeps = {"results": {0: {"upward": {}}}}
    dialog.center_source_combo.setCurrentIndex(1)
    assert not dialog.start_btn.isEnabled()
    assert "No previous sweep center for" in dialog.status_label.text()


@pytest.mark.parametrize("kind, expected", [
    ("single", [0.002]),
    ("list", [0.001, 0.002, 0.004]),
    ("ramp", [0.001, 0.002, 0.004]),
])
def test_custom_frequencies_use_absolute_schedule_without_base(qt_app, kind, expected):
    dialog = MultisweepDialog(module=1, dac_scales={1: -0.5})
    dialog.center_source_combo.setCurrentIndex(2)
    dialog.sections_edit.setText("1100, 1000")
    dialog.custom_amp_edit.setText("")
    dialog.single_amp_edit.setText("0.002")
    dialog.ramp_start_edit.setText("0.001")
    dialog.ramp_stop_edit.setText("0.004")
    dialog.ramp_steps_edit.setText("3")
    getattr(dialog, f"{kind}_radio").setChecked(True)
    assert dialog.start_btn.isEnabled()
    assert dialog.custom_amp_edit.isHidden()
    params = dialog.get_parameters()
    steps = params["amp"].resolve_steps(params["catalog"])
    assert [next(iter(step.amplitudes.values())) for step in steps] == pytest.approx(expected)
    assert all(len(set(step.amplitudes.values())) == 1 for step in steps)


def test_custom_frequencies_use_base_for_multiplicative_schedule(qt_app):
    dialog = MultisweepDialog(module=1, dac_scales={1: -0.5})
    dialog.center_source_combo.setCurrentIndex(2)
    dialog.sections_edit.setText("1000, 1100")
    dialog.custom_amp_edit.setText("0.003")
    dialog.multiplicative_radio.setChecked(True)
    dialog.factor_steps_edit.setText("3")
    params = dialog.get_parameters()
    assert not dialog.custom_amp_edit.isHidden()
    assert [next(iter(step.amplitudes.values())) for step in
            params["amp"].resolve_steps(params["catalog"])] == pytest.approx(
                [0.0015, 0.003, 0.006])


def test_sections_describe_frequency_span(qt_app):
    dialog = _dialog(qt_app)
    frequencies = [r.bias.frequency_hz / 1e6 for r in dialog.catalog]
    assert dialog.sections_info_label.text() == (
        f"2 sweep sections, spanning {min(frequencies):.6f} MHz to "
        f"{max(frequencies):.6f} MHz")


def test_custom_centers_preserve_the_active_catalog(qt_app):
    dialog = _dialog(qt_app)
    catalog = dialog.catalog
    for resonator in catalog:
        resonator.update_bias_point(bifurcated_at=0.01, dI_df=1e-9, dQ_df=2e-9)
    before = catalog.to_dict()
    dialog.center_source_combo.setCurrentIndex(2)
    dialog.sections_edit.setText("1100.12345, 1000.54321")
    params = dialog.get_parameters()
    assert params["catalog"].to_dict() == before
    assert params["center_frequencies"] == dict(zip(
        catalog.names(), [1100.12345e6, 1000.54321e6]))


def test_custom_centers_require_one_frequency_per_resonator(qt_app):
    dialog = _dialog(qt_app)
    dialog.center_source_combo.setCurrentIndex(2)
    dialog.sections_edit.setText("1000")
    assert not dialog.start_btn.isEnabled()
    assert "one center per resonator" in dialog.status_label.text()


def test_worker_passes_selected_centers_without_changing_catalog(qt_app):
    from rfmux.tools.periscope.tasks import MultisweepTask, MultisweepSignals

    dialog = _dialog(qt_app)
    dialog.center_source_combo.setCurrentIndex(2)
    dialog.sections_edit.setText("1000.5, 1100.5")
    params = dialog.get_parameters()
    task = MultisweepTask(None, params, MultisweepSignals())
    forwarded = task._multisweep_params()
    assert forwarded["center_frequencies"] == params["center_frequencies"]
    assert forwarded["catalog"].to_dict() == dialog.catalog.to_dict()


def test_loading_sweep_settings_uses_the_tuned_catalog(qt_app, monkeypatch):
    from test.tuning.test_bias import a_schedule
    from rfmux.tuning import find_bias_points
    from rfmux.tools.periscope import multisweep_dialog

    block = a_schedule()
    report = find_bias_points(block, save=False)
    monkeypatch.setattr(multisweep_dialog, "load_multisweep_container",
                        lambda *args: {"module": block})
    dialog = _dialog(qt_app)
    dialog._on_file_selected("sweep.pkl")
    assert dialog.catalog["R0001"].bias.bifurcated_at == pytest.approx(0.004)
    assert dialog.catalog["R0001"].bias.amplitude == report.catalog["R0001"].bias.amplitude



def test_a_ramp_in_dbm_is_a_dbm_schedule(qt_app):
    dialog = _dialog(qt_app)
    dialog.amp_units.set_units("dbm")
    dialog.ramp_radio.setChecked(True)
    dialog.ramp_start_edit.setText("-60")
    dialog.ramp_stop_edit.setText("-40")
    dialog.ramp_steps_edit.setText("5")

    assert dialog.get_parameters()["amp"] == AmplitudeSchedule.ramp(
        -60, -40, 5, units="dbm")


def test_switching_units_keeps_the_power_and_shows_the_other(qt_app):
    """When the user changes the units, the fields change to the new units.
    The label next to each field shows the value in the other units."""
    dialog = _dialog(qt_app)
    dialog.single_amp_edit.setText("0.001")
    dialog.amp_units.combo.setCurrentIndex(dialog.amp_units.combo.findData("dbm"))
    dialog.amp_units.combo.activated.emit(dialog.amp_units.combo.currentIndex())

    dbm = float(dialog.single_amp_edit.text())
    assert dbm == pytest.approx(-0.5 + 20 * math.log10(0.001), abs=0.01)
    assert "normalized" in dialog.amp_units._fields[1][1].text()


def test_units_set_in_code_do_not_rewrite_the_fields(qt_app):
    """When code sets the units, the text of the fields does not change."""
    dialog = _dialog(qt_app)
    dialog.single_amp_edit.setText("-50")
    dialog.amp_units.set_units("dbm")

    assert dialog.single_amp_edit.text() == "-50"



def test_the_dac_scale_from_the_board_is_not_a_change(qt_app):
    """Only a value that the user types goes to the task."""
    assert _dialog(qt_app).get_parameters()["dac_scale_setting_dbm"] is None


def test_a_typed_dac_scale_is_the_setting_with_the_balun_loss_added(qt_app):
    dialog = _dialog(qt_app)
    dialog.dac_scale_field.setValue(1.0)

    assert dialog.get_parameters()["dac_scale_setting_dbm"] == pytest.approx(
        1.0 + BALUN_LOSS_DB)


def test_the_dac_scale_stays_in_the_range_the_board_accepts(qt_app):
    dialog = _dialog(qt_app)
    """The board refuses a value above its limit, so the field stops at or
    below it."""
    dialog.dac_scale_field.setValue(20.0)

    value = dialog.dac_scale_field.value()
    assert DAC_SCALE_OUTPUT_RANGE_DBM[1] - 0.01 <= value <= DAC_SCALE_OUTPUT_RANGE_DBM[1]


def test_a_schedule_that_can_clip_is_a_warning_on_the_status_line(qt_app):
    dialog = _dialog(qt_app)
    dialog.single_radio.setChecked(True)
    dialog.single_amp_edit.setText("0.8")

    assert "can clip" in dialog.status_label.toolTip()
