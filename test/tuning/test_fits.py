"""Behaviour of the resonator fitting layer.

Pure — no board, no driver. Synthetic resonators in, fits written beside them.
The emphasis is on the contract rather than the arithmetic: fits land under
``fits`` keyed by model, nothing derivable is stored, a fit that fails says
why, running one model leaves the others alone, and the selection arguments
pick out the sweeps a caller means.

The arithmetic gets one test each — a resonator planted at a known fr and Qr
has to come back — because a fitter that recovers nothing is worth catching
here rather than on a cryostat.
"""

import numpy as np
import pytest

from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog
from rfmux.tuning.fits import (
    MODELS,
    FitFailed,
    FitReport,
    centered_iq,
    fit_nonlinear_iq,
    fit_section,
    fit_skewed,
    fit_sweeps,
    fit_sweeps_at_bias_amplitude,
    gain_corrected_iq,
    nonlinear_iq,
    nonlinear_model_iq,
    remove_gain,
    skewed_model_magnitude,
)
from rfmux.tuning.multisweep_amplitudes import AmplitudeSchedule
from rfmux.tuning.sweep_results import pack_results

pytestmark = pytest.mark.portable

FR = 1.0e9
QR = 1e4

# The fitters take one module's envelope, so a ladder built here is indexed out
# of the container the packer returns.
MODULE_ID = "crs0030_rmod2"


def a_resonator(fr=FR, Qr=QR, amp=0.5, a=0.0, npoints=201, noise=0.0, seed=0):
    """A synthetic sweep across one resonator: frequencies and complex IQ.

    Generated from the nonlinear model itself, with ``a=0`` giving the linear
    resonator the skewed Lorentzian is a model of. Gain is a plausible readout
    scale so the traces are in something like counts.
    """
    span = 6 * fr / Qr  # the span this fit likes, per fit_nonlinear_iq
    frequencies = np.linspace(fr - span / 2, fr + span / 2, npoints)
    iq = nonlinear_iq(frequencies, fr, Qr, amp, 0.1, a, 1.2e5, 0.4e5)
    if noise:
        rng = np.random.default_rng(seed)
        iq = iq + (rng.normal(0, noise, npoints) + 1j * rng.normal(0, noise, npoints))
    return frequencies, iq


def a_sweep(amplitude=1e-3, **kwargs):
    """One sweep entry, shaped the way multisweep returns them."""
    frequencies, iq = a_resonator(**kwargs)
    return {
        "channel": 1,
        "frequencies": frequencies,
        "iq_counts": iq,
        "iq_volts": iq * 1e-6,
        "original_center_frequency": kwargs.get("fr", FR),
        "sweep_direction": "upward",
        "sweep_amplitude": amplitude,
    }


def a_catalog():
    """Two resonators a megahertz apart, at different bias amplitudes."""
    return ResonatorCatalog(
        [
            Resonator(name="R0001", channel=1,
                      bias=BiasPoint(frequency_hz=FR, amplitude=2e-3)),
            Resonator(name="R0002", channel=2,
                      bias=BiasPoint(frequency_hz=FR + 1e6, amplitude=4e-3)),
        ],
        module=2,
    )


def a_ladder(directions=("upward", "downward")):
    """A packed multiamp_multisweep result over three amplitude steps."""
    catalog = a_catalog()
    schedule = AmplitudeSchedule.ramp(1e-3, 4e-3, 3)
    steps = schedule.steps(catalog)
    sweeps = {
        step.step: {
            direction: {
                name: a_sweep(
                    fr=FR if name == "R0001" else FR + 1e6,
                    amplitude=step.amplitudes[name],
                )
                for name in ("R0001", "R0002")
            }
            for direction in directions
        }
        for step in steps
    }
    return pack_results(
        sweeps,
        module_id=MODULE_ID,
        module=2,
        amp_schedule=schedule,
        directions=directions,
        span_hz=600e3,
        npoints_per_sweep=201,
        nsamps=10,
        catalog=catalog,
    )[MODULE_ID]


# ─── the fits go beside the sweep they describe ───────────────────────────────


def test_a_fit_lands_under_the_entrys_fits_subdict_keyed_by_model():
    entry = a_sweep()
    fit_section(entry)

    assert set(entry["fits"]) == set(MODELS)
    assert entry["fits"]["skewed"]["failed_because"] is None


def test_the_entry_is_written_in_place_and_keeps_what_it_measured():
    entry = a_sweep()
    before = dict(entry)

    returned = fit_section(entry)

    assert returned is entry["fits"]
    for key in ("channel", "frequencies", "iq_counts", "sweep_amplitude"):
        assert entry[key] is before[key]


def test_fitting_one_model_leaves_another_models_results_alone():
    entry = a_sweep()
    fit_section(entry, models=("skewed",))
    skewed = entry["fits"]["skewed"]

    fit_section(entry, models=("nonlinear",))

    assert entry["fits"]["skewed"] is skewed
    assert set(entry["fits"]) == {"skewed", "nonlinear"}


def test_nothing_derivable_from_the_stored_numbers_is_stored():
    """The model curves, the gain-corrected trace and the centred loop are all
    functions of what is stored plus arrays the entry already has. Storing them
    is how a file comes to disagree with itself."""
    entry = a_sweep()
    fit_section(entry)

    assert set(entry["fits"]["skewed"]) == {"params", "errors", "failed_because"}
    assert set(entry["fits"]["nonlinear"]) == {
        "params", "errors", "residual", "gain", "failed_because"
    }
    assert set(entry["fits"]["circle"]) == {"center", "radius", "failed_because"}


def test_the_settings_come_back_on_the_report_rather_than_on_every_entry():
    sweeps = {"R0001": a_sweep()}
    report = fit_sweeps(sweeps, models=("skewed",), approx_Qr=3e4)

    assert report.settings["approx_Qr"] == 3e4
    assert "settings" not in sweeps["R0001"]["fits"]["skewed"]


# ─── the fitters find what was planted ────────────────────────────────────────


def test_the_skewed_fit_recovers_a_planted_resonator():
    frequencies, iq = a_resonator(noise=1e-3)

    params, errors = fit_skewed(frequencies, iq)

    assert params["fr"] == pytest.approx(FR, rel=1e-6)
    assert params["Qr"] == pytest.approx(QR, rel=0.05)
    assert params["Qc"] >= params["Qr"]  # or the resonator would be unphysical
    assert set(errors) == {"fr", "Qr", "Qcre", "Qcim", "A"}


def test_the_nonlinear_fit_recovers_a_planted_resonator_and_its_nonlinearity():
    frequencies, iq = a_resonator(a=0.3, noise=1e-3)
    corrected, _ = remove_gain(frequencies, iq)

    params, _, residual = fit_nonlinear_iq(frequencies, corrected)

    assert params["fr"] == pytest.approx(FR, rel=1e-6)
    assert params["Qr"] == pytest.approx(QR, rel=0.05)
    assert params["a"] == pytest.approx(0.3, abs=0.05)
    assert residual < 0.1


def test_pinning_the_nonlinearity_fits_a_linear_resonator():
    frequencies, iq = a_resonator(a=0.3)
    corrected, _ = remove_gain(frequencies, iq)

    params, _, _ = fit_nonlinear_iq(frequencies, corrected, fit_nonlinearity=False)

    assert params["a"] == pytest.approx(0.0, abs=1e-9)


def test_the_circle_fit_finds_the_loop_centre():
    entry = a_sweep()
    fit_section(entry, models=("circle",))

    circle = entry["fits"]["circle"]
    distances = np.abs(np.asarray(entry["iq_counts"]) - circle["center"])
    assert distances.max() == pytest.approx(circle["radius"], rel=0.05)


def test_the_gain_the_nonlinear_fit_removed_is_the_off_resonance_level():
    entry = a_sweep()
    fit_section(entry, models=("nonlinear",))

    gain = entry["fits"]["nonlinear"]["gain"]
    off_resonance = np.mean(np.abs(entry["iq_counts"][[0, -1]]))
    assert abs(gain) == pytest.approx(off_resonance, rel=0.05)


# ─── a fit that fails says why ────────────────────────────────────────────────


def test_too_few_points_to_fit_is_a_reason_not_a_traceback():
    entry = {
        "frequencies": np.linspace(FR, FR + 1e3, 3),
        "iq_counts": np.ones(3, dtype=complex),
    }
    fits = fit_section(entry, models=("skewed",))

    assert fits["skewed"]["params"] is None
    assert "too few" in fits["skewed"]["failed_because"]


def test_the_single_trace_fitters_raise_rather_than_returning_a_sentinel():
    with pytest.raises(FitFailed, match="too few"):
        fit_skewed(np.linspace(FR, FR + 1e3, 3), np.ones(3, dtype=complex))


def test_a_dead_channel_has_no_gain_to_divide_out():
    frequencies = np.linspace(FR, FR + 1e3, 101)
    with pytest.raises(FitFailed, match="zero at both ends"):
        remove_gain(frequencies, np.zeros(101, dtype=complex))


def test_a_converged_fit_above_max_residual_keeps_its_parameters_and_says_so():
    entry = a_sweep()
    fits = fit_section(entry, models=("nonlinear",), max_residual=1e-12)

    assert fits["nonlinear"]["params"] is not None  # it converged
    assert "max_residual" in fits["nonlinear"]["failed_because"]


def test_one_malformed_entry_does_not_throw_away_the_rest_of_the_batch():
    sweeps = {"good": a_sweep(), "bad": {"frequencies": None, "iq_counts": None}}

    report = fit_sweeps(sweeps, models=("circle",))

    assert [f.name for f in report.fitted] == ["good"]
    assert "nothing to fit" in report.failed[0].failed_because


def test_mismatched_arrays_are_the_callers_mistake_and_raise():
    with pytest.raises(ValueError, match="not the same trace"):
        fit_skewed(np.linspace(FR, FR + 1e3, 10), np.ones(9, dtype=complex))


# ─── reading the fits back ────────────────────────────────────────────────────


def test_the_skewed_model_tracks_the_normalized_measurement():
    entry = a_sweep(noise=1e-3)
    fit_section(entry, models=("skewed",))

    model = skewed_model_magnitude(entry)
    measured = np.abs(entry["iq_counts"] / entry["iq_counts"][-1])

    assert model.shape == measured.shape
    assert np.allclose(model, measured, atol=0.02)


def test_the_nonlinear_model_is_returned_in_counts_beside_the_measurement():
    entry = a_sweep(noise=1e-3)
    fit_section(entry, models=("nonlinear",))

    model = nonlinear_model_iq(entry)

    assert np.allclose(model, entry["iq_counts"], rtol=0, atol=0.02 * np.abs(entry["iq_counts"]).mean())


def test_the_gain_corrected_trace_is_rebuilt_from_the_stored_gain():
    entry = a_sweep()
    fit_section(entry, models=("nonlinear",))

    assert np.allclose(
        gain_corrected_iq(entry),
        entry["iq_counts"] / entry["fits"]["nonlinear"]["gain"],
    )


def test_the_centred_loop_is_rebuilt_from_the_stored_circle_centre():
    entry = a_sweep()
    fit_section(entry, models=("circle",))

    assert np.allclose(
        centered_iq(entry),
        entry["iq_counts"] - entry["fits"]["circle"]["center"],
    )


def test_reading_a_model_that_was_never_run_says_how_to_run_it():
    entry = a_sweep()
    fit_section(entry, models=("circle",))

    with pytest.raises(ValueError, match="no skewed fit"):
        skewed_model_magnitude(entry)


def test_reading_a_model_that_did_not_converge_gives_the_reason():
    entry = {
        "frequencies": np.linspace(FR, FR + 1e3, 3),
        "iq_counts": np.ones(3, dtype=complex),
    }
    fit_section(entry, models=("skewed",))

    with pytest.raises(ValueError, match="did not converge.*too few"):
        skewed_model_magnitude(entry)


# ─── what gets fitted ─────────────────────────────────────────────────────────


def test_a_bare_multisweep_return_is_fitted_as_one_iteration():
    sweeps = {"R0001": a_sweep(), "R0002": a_sweep()}

    report = fit_sweeps(sweeps, models=("circle",))

    assert len(report) == 2
    assert {f.iteration for f in report.fits} == {None}
    assert {f.direction for f in report.fits} == {None}


def test_a_packed_ladder_is_fitted_across_every_iteration_and_direction():
    sweeps = a_ladder()

    report = fit_sweeps(sweeps, models=("circle",))

    # 2 resonators x 3 amplitude steps x 2 directions
    assert len(report) == 12
    assert {f.iteration for f in report.fits} == {0, 1, 2}
    assert {f.direction for f in report.fits} == {"upward", "downward"}


def test_the_selection_arguments_narrow_what_is_fitted():
    sweeps = a_ladder()

    report = fit_sweeps(
        sweeps, models=("circle",), names="R0001", iterations=1, directions="upward"
    )

    assert len(report) == 1
    assert (report.fits[0].name, report.fits[0].iteration) == ("R0001", 1)
    assert "fits" not in sweeps["results"][0]["upward"]["R0001"]


def test_a_single_name_is_one_name_and_not_a_sequence_of_characters():
    sweeps = a_ladder(directions=("upward",))

    report = fit_sweeps(sweeps, models=("circle",), names="R0001")

    assert {f.name for f in report.fits} == {"R0001"}


def test_a_name_that_was_not_swept_says_which_ones_were():
    with pytest.raises(ValueError, match=r"R9999.*R0001"):
        fit_sweeps(a_ladder(), names="R9999")


def test_a_filter_that_selects_nothing_says_what_there_was():
    with pytest.raises(ValueError, match=r"iterations=99.*\[0, 1, 2\]"):
        fit_sweeps(a_ladder(), iterations=99)


def test_iterations_are_refused_on_a_sweep_that_has_only_one():
    with pytest.raises(ValueError, match="single multisweep return"):
        fit_sweeps({"R0001": a_sweep()}, iterations=0)


def test_a_multi_module_return_is_refused_one_module_at_a_time():
    with pytest.raises(TypeError, match="one module at a time"):
        fit_sweeps([{"R0001": a_sweep()}])


def test_a_bare_string_of_models_is_refused_rather_than_read_as_characters():
    with pytest.raises(TypeError, match="not a single string"):
        fit_sweeps({"R0001": a_sweep()}, models="skewed")


def test_an_unknown_model_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="Unknown model"):
        fit_sweeps({"R0001": a_sweep()}, models=("lorentzian",))


# ─── fitting where each resonator is biased ───────────────────────────────────


def test_fitting_at_the_bias_amplitude_picks_one_iteration_per_resonator():
    sweeps = a_ladder(directions=("upward",))

    report = fit_sweeps_at_bias_amplitude(sweeps, models=("circle",))

    # ramp(1e-3, 4e-3, 3) is [1e-3, 2.5e-3, 4e-3]; R0001 is biased at 2e-3 and
    # R0002 at 4e-3, so they are matched to different rungs of the same ladder.
    at = {f.name: f.iteration for f in report.fits}
    assert at == {"R0001": 1, "R0002": 2}


def test_an_explicit_amplitude_overrides_the_catalogs_bias_amplitudes():
    sweeps = a_ladder(directions=("upward",))

    report = fit_sweeps_at_bias_amplitude(
        sweeps, amplitude=1e-3, models=("circle",)
    )

    assert {f.iteration for f in report.fits} == {0}


def test_a_single_multisweep_has_no_amplitudes_to_match_between():
    with pytest.raises(TypeError, match="nothing to match"):
        fit_sweeps_at_bias_amplitude({"R0001": a_sweep()})


# ─── the report ───────────────────────────────────────────────────────────────


def test_the_report_counts_each_model_separately():
    sweeps = a_ladder(directions=("upward",))

    report = fit_sweeps(sweeps, models=("skewed", "circle"))

    assert len(report.for_model("skewed")) == 6
    assert len(report.for_model("circle")) == 6
    assert len(report) == 12


def test_the_report_says_where_a_failure_was_as_well_as_why():
    sweeps = {"R0001": {"frequencies": None, "iq_counts": None}}

    report = fit_sweeps(sweeps, models=("circle",))

    assert isinstance(report, FitReport)
    assert report.fitted == []
    assert report.failed[0].where == "R0001"
    assert "failed" in repr(report)


def test_the_report_labels_a_ladder_fit_with_both_of_its_coordinates():
    sweeps = a_ladder()

    report = fit_sweeps(sweeps, models=("circle",), names="R0001", iterations=2)

    assert {f.where for f in report.fits} == {
        "R0001@2 upward",
        "R0001@2 downward",
    }


def test_progress_is_reported_per_sweep_and_not_per_fit():
    sweeps = a_ladder(directions=("upward",))
    ticks = []

    fit_sweeps(
        sweeps,
        models=("skewed", "circle"),
        progress_callback=lambda done, total: ticks.append((done, total)),
    )

    assert ticks == [(i, 6) for i in range(1, 7)]
