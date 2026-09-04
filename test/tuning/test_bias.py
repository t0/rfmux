"""Behaviour of the bias-finding layer.

Pure — no board, no driver. Synthetic amplitude steps in, a new catalog out.

The emphasis is on the contract: the search stops one amplitude below
bifurcation, nothing that was handed in comes back modified, a resonator that
could not be placed keeps the operating point it arrived with and says why, and
the calibration on a bias point belongs to the frequency printed beside it.

The two bifurcation tests get their arithmetic checked as well, because a
detector that never fires — or always does — is worth catching here rather than
on a cryostat. The nonlinear model supplies the jumped traces: it leans a
resonance over exactly the way drive does, so a large ``a`` gives a sweep with a
real discontinuity in it rather than a spike pasted in by hand.
"""

import numpy as np
import pytest

from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog
from rfmux.core.transferfunctions import BASE_FREQUENCY
from rfmux.tuning.bias import (
    BiasReport,
    bifurcated_by_derivative,
    bifurcated_by_hysteresis,
    find_bias_amplitude,
    find_bias_frequency,
    find_bias_points,
    iq_arc_speed,
    iq_derivatives_at,
    normalized_arc_speed,
)
from rfmux.tuning.fits import nonlinear_iq
from rfmux.tuning.multisweep_amplitudes import AmplitudeSchedule
from rfmux.tuning.sweep_results import pack_results, pack_sweep

pytestmark = pytest.mark.portable

FR = 1.0e9
QR = 1e4
SPAN = 6 * FR / QR  # 600 kHz — the span the resonator models like

#: ``a`` well past bifurcation (4·sqrt(3)/9 ≈ 0.77). High enough that the
#: model's branch solution actually jumps, which is what the detectors look for.
JUMPED = 2.0

MODULE_ID = "crs0030_rmod2"
VOLTS_PER_COUNT = 1e-6


def a_trace(a=0.0, npoints=201, fr=FR, direction="upward"):
    """A synthetic sweep across one resonator: frequencies and complex IQ.

    ``a=0`` is a linear resonator; :data:`JUMPED` leans it far enough over to
    bifurcate. A downward sweep visits the same frequencies in reverse, which
    is what the macro does.
    """
    frequencies = np.linspace(fr - SPAN / 2, fr + SPAN / 2, npoints)
    if direction == "downward":
        frequencies = frequencies[::-1]
    return frequencies, nonlinear_iq(frequencies, fr, QR, 0.5, 0.1, a, 1.2e5, 0.4e5)


def a_sweep(amplitude=1e-3, a=0.0, fr=FR, direction="upward", **kwargs):
    """One sweep entry, shaped the way multisweep returns them."""
    frequencies, iq = a_trace(a=a, fr=fr, direction=direction, **kwargs)
    return {
        "channel": 1,
        "frequencies": frequencies,
        "iq_counts": iq,
        "iq_volts": iq * VOLTS_PER_COUNT,
        "original_center_frequency": fr,
        "sweep_direction": direction,
        "sweep_amplitude": amplitude,
    }


def a_catalog(amplitude=1e-3):
    """Two resonators a megahertz apart."""
    return ResonatorCatalog(
        [
            Resonator(name="R0001", channel=1,
                      bias=BiasPoint(frequency_hz=FR, amplitude=amplitude)),
            Resonator(name="R0002", channel=2,
                      bias=BiasPoint(frequency_hz=FR + 1e6, amplitude=amplitude)),
        ],
        module=2,
    )


def a_step(a=0.0, amplitude=1e-3, directions=("upward",), **kwargs):
    """One resonator at one amplitude step, as ``{direction: entry}``."""
    return {
        direction: a_sweep(amplitude=amplitude, a=a, direction=direction, **kwargs)
        for direction in directions
    }


def amplitude_iterations(nonlinearities=(0.0, 0.0, JUMPED), amplitudes=None, **kwargs):
    """One resonator's ``{iteration: {direction: entry}}``, a step per ``a``."""
    if amplitudes is None:
        amplitudes = [1e-3 * 2**i for i in range(len(nonlinearities))]
    return {
        i: a_step(a=a, amplitude=amp, **kwargs)
        for i, (a, amp) in enumerate(zip(nonlinearities, amplitudes))
    }


def a_multiamp_multisweep(
    nonlinearities=(0.0, 0.0, JUMPED),
    directions=("upward", "downward"),
    catalog=None,
    names=("R0001", "R0002"),
):
    """One module's worth of a packed multiamp_multisweep return.

    Through the real packer, so these tests cannot drift from the shape the
    macros actually produce. Both resonators get the same series of ``a``.
    """
    catalog = a_catalog() if catalog is None else catalog
    schedule = AmplitudeSchedule.ramp(1e-3, 4e-3, len(nonlinearities))
    steps = schedule.steps(catalog)
    sweeps = {
        step.step: {
            direction: {
                name: a_sweep(
                    amplitude=step.amplitudes[name],
                    a=nonlinearities[step.step],
                    fr=FR if name == "R0001" else FR + 1e6,
                    direction=direction,
                )
                for name in names
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
        span_hz=SPAN,
        npoints_per_sweep=201,
        nsamps=10,
        catalog=catalog,
    )[MODULE_ID]


# ─── choosing the amplitude ───────────────────────────────────────────────────


def test_the_amplitude_below_the_first_bifurcated_one_is_chosen():
    iterations = amplitude_iterations((0.0, 0.0, JUMPED, JUMPED))

    choice = find_bias_amplitude(iterations)

    assert choice.iteration == 1
    assert choice.amplitude == pytest.approx(2e-3)
    assert choice.bifurcated_at == pytest.approx(4e-3)
    assert not choice.is_bifurcated


def test_a_sweep_that_never_bifurcates_is_biased_at_its_loudest_step():
    """The schedule did not reach the limit, so the most drive measured is the
    most drive known to be safe."""
    iterations = amplitude_iterations((0.0, 0.0, 0.0))

    choice = find_bias_amplitude(iterations)

    assert choice.iteration == 2
    assert choice.amplitude == pytest.approx(4e-3)
    assert choice.bifurcated_at is None
    assert not choice.is_bifurcated


def test_bifurcation_at_the_quietest_step_says_so_rather_than_going_below():
    iterations = amplitude_iterations((JUMPED, JUMPED))

    choice = find_bias_amplitude(iterations)

    assert choice.iteration == 0
    assert choice.bifurcated_at == pytest.approx(choice.amplitude)
    assert choice.is_bifurcated


def test_a_single_multisweep_is_one_amplitude_step():
    choice = find_bias_amplitude(amplitude_iterations((0.0,)))

    assert choice.iteration == 0
    assert choice.bifurcated_at is None


def test_steps_are_examined_in_amplitude_order_not_the_order_measured():
    """An explicit schedule is free to run high, low, middle. 'One step below'
    is a statement about drive, not about when the sweep happened."""
    iterations = {
        0: a_step(a=JUMPED, amplitude=4e-3),
        1: a_step(a=0.0, amplitude=1e-3),
        2: a_step(a=0.0, amplitude=2e-3),
    }

    choice = find_bias_amplitude(iterations)

    assert choice.iteration == 2
    assert choice.amplitude == pytest.approx(2e-3)


def test_only_the_steps_actually_examined_get_a_verdict():
    """The search stops at the first bifurcated step, so the ones above it were
    never looked at and have nothing to report."""
    iterations = amplitude_iterations((0.0, JUMPED, JUMPED, JUMPED))

    choice = find_bias_amplitude(iterations)

    assert set(choice.checks) == {0, 1}
    assert not choice.checks[0].bifurcated
    assert choice.checks[1].bifurcated


def test_no_sweeps_at_all_says_so():
    with pytest.raises(ValueError, match="no amplitude to choose"):
        find_bias_amplitude({})


def test_the_choice_unpacks_like_the_tuple_it_is():
    iteration, amplitude, bifurcated_at, checks = find_bias_amplitude(
        amplitude_iterations((0.0, 0.0, JUMPED))
    )

    assert (iteration, amplitude) == (1, pytest.approx(2e-3))
    assert bifurcated_at == pytest.approx(4e-3)
    assert set(checks) == {0, 1, 2}


def test_an_unknown_amplitude_method_is_refused_by_name():
    with pytest.raises(ValueError, match="hysteresis"):
        find_bias_amplitude(amplitude_iterations(), method="jumpiness")


# ─── bifurcation by derivative ────────────────────────────────────────────────


def test_a_linear_resonator_is_not_bifurcated():
    check = bifurcated_by_derivative(a_step(a=0.0))

    assert not check.bifurcated
    assert check.method == "derivative"


def test_a_jumped_trace_is_bifurcated():
    check = bifurcated_by_derivative(a_step(a=JUMPED))

    assert check.bifurcated
    assert check.metric > check.threshold


def test_the_factors_are_what_makes_the_test_less_sensitive():
    step = a_step(a=JUMPED)

    assert bifurcated_by_derivative(step).bifurcated
    assert not bifurcated_by_derivative(step, spike_height_factor=1e6).bifurcated


def test_the_derivative_test_reads_the_shape_and_not_the_scale():
    """I and Q are normalized by their own range, so a resonator ten times
    deeper is not ten times more suspicious."""
    step = a_step(a=JUMPED)
    louder = {"upward": dict(step["upward"])}
    louder["upward"]["iq_counts"] = step["upward"]["iq_counts"] * 10

    assert bifurcated_by_derivative(louder).metric == pytest.approx(
        bifurcated_by_derivative(step).metric
    )


def test_one_bifurcated_direction_is_enough_to_call_the_step_bifurcated():
    """A bifurcated resonator jumps whichever way the sweep runs; needing both
    to agree would only lose the one that happened to catch it."""
    mixed = {
        "upward": a_sweep(a=JUMPED, direction="upward"),
        "downward": a_sweep(a=0.0, direction="downward"),
    }

    assert bifurcated_by_derivative(mixed).bifurcated


def test_a_downward_sweep_is_read_the_same_way_as_an_upward_one():
    """Entries arrive high-to-low; every difference taken here wants them the
    other way round."""
    up = bifurcated_by_derivative({"upward": a_sweep(a=JUMPED)})
    down = bifurcated_by_derivative({"downward": a_sweep(a=JUMPED, direction="downward")})

    assert down.bifurcated == up.bifurcated
    assert down.metric == pytest.approx(up.metric)


# ─── bifurcation by hysteresis ────────────────────────────────────────────────


def test_two_directions_that_agree_are_not_bifurcated():
    check = bifurcated_by_hysteresis(a_step(a=0.0, directions=("upward", "downward")))

    assert not check.bifurcated
    assert check.metric == pytest.approx(0.0, abs=1e-9)
    assert check.method == "hysteresis"


def test_the_discrepancy_is_measured_in_loop_radii():
    """The separation between the traces is imposed here rather than simulated:
    what is under test is that a known separation comes back as a number in
    units of the loop, so the threshold means the same thing on every
    resonator."""
    step = a_step(a=0.0, directions=("upward", "downward"))
    up = step["upward"]["iq_counts"]
    radius = np.max(np.abs(up - up.mean()))
    apart = np.zeros_like(up)
    apart[100:110] = 0.4 * radius
    step["downward"]["iq_counts"] = step["downward"]["iq_counts"] + apart[::-1]

    check = bifurcated_by_hysteresis(step)

    assert check.metric == pytest.approx(0.4, rel=1e-6)
    assert check.bifurcated
    assert not bifurcated_by_hysteresis(step, max_discrepancy=0.5).bifurcated


def test_the_discrepancy_does_not_care_how_large_the_loop_is():
    step = a_step(a=0.0, directions=("upward", "downward"))
    step["downward"]["iq_counts"] = a_trace(a=0.5, direction="downward")[1]
    louder = {
        d: {**e, "iq_counts": e["iq_counts"] * 10} for d, e in step.items()
    }

    assert bifurcated_by_hysteresis(louder).metric == pytest.approx(
        bifurcated_by_hysteresis(step).metric
    )


def test_hysteresis_needs_both_directions_and_says_which_is_missing():
    with pytest.raises(ValueError, match="no downward sweep"):
        bifurcated_by_hysteresis(a_step(directions=("upward",)))


def test_a_hysteresis_search_stops_where_the_directions_part():
    """The steps are identical up and down until the third, where the downward
    trace is displaced — so the second step is the operating point."""
    iterations = amplitude_iterations((0.0, 0.0, 0.0), directions=("upward", "downward"))
    top = iterations[2]
    up = top["upward"]["iq_counts"]
    top["downward"]["iq_counts"] = top["downward"]["iq_counts"] + 0.5 * np.max(
        np.abs(up - up.mean())
    )

    choice = find_bias_amplitude(iterations, method="hysteresis")

    assert choice.iteration == 1
    assert choice.bifurcated_at == pytest.approx(4e-3)


# ─── where in the sweep the tone goes ─────────────────────────────────────────


def test_the_iq_derivative_method_lands_on_the_resonance():
    entry = a_sweep(a=0.0)

    frequency_hz = find_bias_frequency(entry)

    assert frequency_hz == pytest.approx(FR, abs=SPAN / 200)


def test_the_minimum_method_lands_at_the_bottom_of_the_dip():
    entry = a_sweep(a=0.0)

    frequency_hz = find_bias_frequency(entry, method="minimum")

    assert frequency_hz == pytest.approx(
        entry["frequencies"][np.argmin(np.abs(entry["iq_counts"]))]
    )


def test_both_methods_answer_with_a_frequency_that_was_measured():
    entry = a_sweep(a=JUMPED)

    for method in ("iq_derivative", "minimum"):
        assert find_bias_frequency(entry, method=method) in set(entry["frequencies"])


def test_the_frequency_method_does_not_judge_its_own_answer():
    """Plausibility needs the sweep centre and a tolerance, so it belongs to
    find_bias_points — which flags rather than refuses, because the answer is
    still the best point on the trace."""
    assert find_bias_frequency(a_sweep(a=JUMPED)) > FR


def test_an_unknown_frequency_method_is_refused_by_name():
    with pytest.raises(ValueError, match="iq_derivative"):
        find_bias_frequency(a_sweep(), method="fitted")


# ─── reading back what a method looked at ─────────────────────────────────────


def test_the_arc_speed_reader_peaks_where_the_frequency_method_puts_the_tone():
    entry = a_sweep(a=0.0)

    frequencies, speed = iq_arc_speed(entry)

    assert frequencies[np.argmax(speed)] == pytest.approx(find_bias_frequency(entry))


def test_the_normalized_speed_reader_is_what_the_detector_differentiates():
    """One step shorter than the sweep, on the midpoints — which is where a
    difference between two points belongs."""
    entry = a_sweep(a=JUMPED)

    frequencies, speed = normalized_arc_speed(entry)

    assert len(speed) == len(entry["frequencies"]) - 1
    assert len(frequencies) == len(speed)
    assert np.diff(speed).max() == pytest.approx(
        bifurcated_by_derivative({"upward": entry}).metric
    )


def test_both_readers_come_back_in_ascending_frequency_order():
    downward = a_sweep(direction="downward")

    for reader in (iq_arc_speed, normalized_arc_speed):
        frequencies, _ = reader(downward)
        assert np.all(np.diff(frequencies) > 0)


def test_a_trace_too_short_to_differentiate_says_so_rather_than_returning_empty():
    stub = {"frequencies": np.array([1.0, 2.0]), "iq_counts": np.array([1 + 1j, 2 + 2j])}

    with pytest.raises(ValueError):
        iq_arc_speed(stub)


# ─── the calibration at that frequency ────────────────────────────────────────


def test_the_derivatives_are_read_off_the_volts_and_are_per_hertz():
    entry = a_sweep(a=0.0)

    dI_df, dQ_df = iq_derivatives_at(entry, FR)
    entry["iq_volts"] = entry["iq_volts"] * 2

    assert iq_derivatives_at(entry, FR)[0] == pytest.approx(2 * dI_df)
    assert iq_derivatives_at(entry, FR)[1] == pytest.approx(2 * dQ_df)


def test_a_sweep_with_no_volts_cannot_be_calibrated():
    entry = a_sweep()
    entry["iq_volts"] = None

    with pytest.raises(ValueError, match="iq_volts"):
        iq_derivatives_at(entry, FR)


# ─── the whole thing ──────────────────────────────────────────────────────────


def test_a_new_catalog_comes_back_and_the_one_swept_is_untouched():
    catalog = a_catalog()
    before = catalog.to_dict()

    report = find_bias_points(a_multiamp_multisweep(), catalog)

    assert isinstance(report, BiasReport)
    assert report.catalog is not catalog
    assert catalog.to_dict() == before
    assert report.catalog.to_dict() != before


def test_the_sweeps_come_back_as_they_went_in():
    """The diagnostics of an analysis do not belong written onto the data the
    analysis was handed."""
    sweeps = a_multiamp_multisweep()
    entry = sweeps["results"][0]["upward"]["R0001"]
    keys = set(entry)

    find_bias_points(sweeps)

    assert set(entry) == keys


def test_the_bias_point_carries_the_calibration_measured_at_it():
    report = find_bias_points(a_multiamp_multisweep())
    bias = report.catalog["R0001"].bias
    finding = report["R0001"]

    assert bias.dI_df == finding.dI_df
    assert bias.dQ_df == finding.dQ_df
    assert bias.df_calibration == pytest.approx(
        1 / complex(finding.dI_df, finding.dQ_df)
    )
    assert bias.bifurcated_at == pytest.approx(finding.bifurcated_at)


def test_the_calibration_belongs_to_the_tone_that_will_be_played():
    """Quantized first, then differentiated: the derivatives are the ones at
    the frequency the hardware will actually put the tone on."""
    report = find_bias_points(a_multiamp_multisweep())
    bias = report.catalog["R0001"].bias
    entry = report_entry(a_multiamp_multisweep(), report["R0001"])

    assert bias.frequency_hz == pytest.approx(
        round(bias.frequency_hz / BASE_FREQUENCY) * BASE_FREQUENCY
    )
    assert (bias.dI_df, bias.dQ_df) == pytest.approx(
        iq_derivatives_at(entry, bias.frequency_hz)
    )


def report_entry(sweeps, finding, direction="upward"):
    """The sweep a finding was measured on."""
    return sweeps["results"][finding.iteration][direction][finding.name]


def test_iq_rotation_is_left_alone_because_it_is_not_measured_from_a_sweep():
    report = find_bias_points(a_multiamp_multisweep())

    assert report.catalog["R0001"].bias.iq_rotation_deg is None


def test_identity_and_channels_survive_the_new_catalog():
    catalog = a_catalog()

    report = find_bias_points(a_multiamp_multisweep(catalog=catalog), catalog)

    assert [r.name for r in report.catalog] == [r.name for r in catalog]
    assert [r.channel for r in report.catalog] == [r.channel for r in catalog]
    assert report.catalog.module == catalog.module


def test_the_amplitude_that_was_chosen_is_the_amplitude_on_the_bias_point():
    report = find_bias_points(a_multiamp_multisweep((0.0, 0.0, JUMPED)))

    for finding in report.findings:
        assert report.catalog[finding.name].bias.amplitude == pytest.approx(
            finding.amplitude
        )


def test_the_catalog_defaults_to_the_one_the_sweep_recorded():
    report = find_bias_points(a_multiamp_multisweep())

    assert len(report.catalog) == 2
    assert all(f.good for f in report.findings)


def test_a_sweep_of_bare_frequencies_has_no_catalog_to_bias():
    sweeps = pack_sweep(
        {"section_0": a_sweep()},
        module_id=MODULE_ID,
        module=2,
        sweep_direction="upward",
        span_hz=SPAN,
        npoints_per_sweep=201,
        nsamps=10,
        center_frequencies=[FR],
    )[MODULE_ID]

    with pytest.raises(ValueError, match="Pass catalog"):
        find_bias_points(sweeps)


def test_a_catalog_resonator_these_sweeps_do_not_cover_is_the_callers_mistake():
    """The catalog and the sweeps come from one measurement, so a resonator with
    no data means a mismatched pair of arguments — not a detector that could not
    be biased."""
    catalog = ResonatorCatalog(
        [*a_catalog(),
         Resonator(name="R0003", channel=3,
                   bias=BiasPoint(frequency_hz=FR + 2e6, amplitude=1e-3))],
        module=2,
    )

    with pytest.raises(KeyError, match="R0003"):
        find_bias_points(a_multiamp_multisweep(), catalog)


def test_a_sweep_with_no_volts_to_calibrate_off_is_the_callers_mistake_too():
    sweeps = a_multiamp_multisweep()
    for by_direction in sweeps["results"].values():
        for sections in by_direction.values():
            sections["R0001"]["iq_volts"] = None

    with pytest.raises(ValueError, match="iq_volts"):
        find_bias_points(sweeps)


def test_there_is_one_finding_per_catalog_resonator_in_channel_order():
    """Sections in the sweep that are not in the catalog are not detectors we
    were asked to bias, so they are not reported on."""
    report = find_bias_points(a_multiamp_multisweep(), a_catalog_of_one())

    assert [f.name for f in report.findings] == ["R0001"]
    with pytest.raises(KeyError):
        report["R0002"]


def a_catalog_of_one():
    return ResonatorCatalog(
        [Resonator(name="R0001", channel=1,
                   bias=BiasPoint(frequency_hz=FR, amplitude=1e-3))],
        module=2,
    )


# ─── a bias point that is a default rather than a measurement ─────────────────


def test_every_resonator_comes_back_with_a_freshly_measured_bias_point():
    catalog = a_catalog()

    report = find_bias_points(a_multiamp_multisweep(catalog=catalog), catalog)

    assert len(report.findings) == len(catalog)
    for resonator in report.catalog:
        assert resonator.bias.dI_df is not None
        assert resonator.bias.df_calibration is not None


def test_bifurcation_at_the_quietest_amplitude_is_biased_anyway_and_flagged():
    report = find_bias_points(a_multiamp_multisweep((JUMPED, JUMPED, JUMPED)))
    finding = report["R0001"]

    assert report.catalog["R0001"].bias.amplitude == pytest.approx(finding.amplitude)
    assert not finding.good
    assert "quietest amplitude" in finding.flagged_because
    assert [f.name for f in report.flagged] == ["R0001", "R0002"]


def test_never_reaching_bifurcation_is_biased_anyway_and_flagged():
    report = find_bias_points(a_multiamp_multisweep((0.0, 0.0, 0.0)))
    finding = report["R0001"]

    assert finding.bifurcated_at is None
    assert not finding.good
    assert "loudest amplitude measured" in finding.flagged_because


def test_an_amplitude_bracketed_by_the_sweep_is_not_flagged():
    report = find_bias_points(a_multiamp_multisweep((0.0, 0.0, JUMPED)))

    assert report.flagged == []
    assert [f.name for f in report.good] == ["R0001", "R0002"]
    assert report["R0001"].flagged_because is None


def with_the_sweep_centre_moved(sweeps, name, by_hz):
    """Rewrite one resonator's recorded sweep centre.

    The synthetic resonances sit exactly at the middle of their own sweeps, so
    the measured peak is always the sweep centre and no distance test can ever
    fire. Moving the recorded centre is the same arithmetic as a resonance that
    was pulled away from where the sweep was aimed.
    """
    for by_direction in sweeps["results"].values():
        for sections in by_direction.values():
            sections[name]["original_center_frequency"] += by_hz
    return sweeps


def test_a_resonance_further_out_than_asked_for_leaves_the_tone_where_it_was():
    """A peak that far off centre is usually a neighbour in the span, or noise
    in a trace the resonance has left. Moving the tone onto it would be worse
    than not moving it at all — so the sweep centre is kept, and flagged."""
    sweeps = with_the_sweep_centre_moved(
        a_multiamp_multisweep((0.0, 0.0, JUMPED)), "R0001", -20e3
    )

    report = find_bias_points(sweeps, max_distance_hz=5e3)
    finding = report["R0001"]

    assert finding.frequency_hz == pytest.approx(FR - 20e3, abs=BASE_FREQUENCY / 2)
    assert finding.frequency_hz == report.catalog["R0001"].bias.frequency_hz
    assert not finding.good
    assert "left where the sweep was centred" in finding.flagged_because

    # Its neighbour was not moved, so it is measured and not flagged.
    assert [f.name for f in report.flagged] == ["R0001"]
    assert report["R0002"].frequency_hz == pytest.approx(
        FR + 1e6, abs=BASE_FREQUENCY / 2
    )


def test_the_calibration_is_measured_where_the_tone_ended_up():
    """Falling back moves the frequency, so the derivatives have to be read
    there rather than at the peak that was rejected."""
    sweeps = with_the_sweep_centre_moved(
        a_multiamp_multisweep((0.0, 0.0, JUMPED)), "R0001", -20e3
    )

    finding = find_bias_points(sweeps, max_distance_hz=5e3)["R0001"]
    entry = sweeps["results"][finding.iteration]["upward"]["R0001"]

    assert (finding.dI_df, finding.dQ_df) == pytest.approx(
        iq_derivatives_at(entry, finding.frequency_hz)
    )
    # Not the peak it rejected, which is 20 kHz away and much steeper.
    assert (finding.dI_df, finding.dQ_df) != pytest.approx(
        iq_derivatives_at(entry, find_bias_frequency(entry))
    )


def test_a_believable_distance_leaves_the_measured_peak_alone():
    sweeps = with_the_sweep_centre_moved(
        a_multiamp_multisweep((0.0, 0.0, JUMPED)), "R0001", -20e3
    )

    report = find_bias_points(sweeps, max_distance_hz=50e3)

    assert report["R0001"].frequency_hz == pytest.approx(FR, abs=BASE_FREQUENCY / 2)
    assert report.flagged == []


def test_only_the_first_concern_is_reported():
    """A resonator whose sweeps never bifurcated has a bigger problem than one
    whose tone landed off centre, and hearing about both at once helps nobody."""
    sweeps = with_the_sweep_centre_moved(
        a_multiamp_multisweep((0.0, 0.0, 0.0)), "R0001", -20e3
    )

    report = find_bias_points(sweeps, max_distance_hz=5e3)

    assert "loudest amplitude measured" in report["R0001"].flagged_because


def test_hysteresis_on_a_one_direction_sweep_is_refused_once_not_per_resonator():
    with pytest.raises(ValueError, match="Sweep both directions"):
        find_bias_points(a_multiamp_multisweep(directions=("upward",)), amplitude_method="hysteresis")


def test_a_direction_that_was_not_swept_is_refused():
    with pytest.raises(ValueError, match="was not swept"):
        find_bias_points(a_multiamp_multisweep(directions=("upward",)), direction="downward")


def test_the_whole_container_is_refused_because_a_report_is_about_one_module():
    sweeps = a_multiamp_multisweep()

    with pytest.raises(TypeError, match="keyed by module"):
        find_bias_points({MODULE_ID: sweeps})


def test_the_settings_come_back_on_the_report_rather_than_on_every_bias_point():
    report = find_bias_points(a_multiamp_multisweep(), max_discrepancy=0.4)

    assert report.settings["amplitude_method"] == "derivative"
    assert report.settings["frequency_method"] == "iq_derivative"
    assert report.settings["max_discrepancy"] == 0.4
    assert report.settings["max_distance_hz"] is None
    assert report.settings["module"] == 2


def test_the_report_reads_like_what_happened():
    report = find_bias_points(a_multiamp_multisweep((0.0, 0.0, 0.0)))

    assert len(report) == 2
    assert "2 biased, 2 flagged" in repr(report)
    assert "loudest amplitude measured" in repr(report)
    with pytest.raises(KeyError):
        report["R9999"]
