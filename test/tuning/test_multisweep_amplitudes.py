"""Behaviour of the pure half of the multi-amplitude multisweep.

Two ends of one contract: AmplitudeSchedule deciding what each iteration probes
at, and the packing and readers deciding what comes back. Pure — no board, no
driver. The emphasis is on what the driver and a Periscope dialog rely on: a
step is one amplitude, amplitudes come back keyed by name, everything that would
fail on the hardware fails here instead, and nothing has to be stored twice to
be readable.
"""

import math
import pickle

import numpy as np
import pytest

from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog
from rfmux.tuning.multisweep_amplitudes import (
    RESULTS_SCHEMA_VERSION,
    AmplitudeSchedule,
    AmplitudeStep,
    collect_amplitude_iterations_for,
    find_iteration_number_matching_amplitude,
    get_amplitudes_at_iteration,
    pack_results,
)

pytestmark = pytest.mark.portable


def a_catalog(amplitudes=(0.001, 0.002, 0.004)):
    """Three resonators, deliberately at different bias amplitudes."""
    return ResonatorCatalog(
        [
            Resonator(
                name=f"R{i + 1:04d}",
                channel=i + 1,
                bias=BiasPoint(frequency_hz=1.0e9 + i * 1e6, amplitude=a),
            )
            for i, a in enumerate(amplitudes)
        ],
        module=2,
    )


def amplitudes_of(schedule, target):
    """The per-step amplitude dicts, which is what most assertions are about."""
    return [step.amplitudes for step in schedule.steps(target)]


# ─── a step is one amplitude ──────────────────────────────────────────────────


def test_the_default_schedule_sweeps_the_catalog_once_as_it_stands():
    catalog = a_catalog()
    steps = AmplitudeSchedule().steps(catalog)

    assert len(steps) == 1
    assert steps[0].amplitudes == {"R0001": 0.001, "R0002": 0.002, "R0003": 0.004}


def test_steps_are_numbered_from_zero_in_measurement_order():
    schedule = AmplitudeSchedule.explicit([0.004, 0.001, 0.002])
    steps = schedule.steps(a_catalog())

    assert [s.step for s in steps] == [0, 1, 2]
    # Explicit means explicit: the order given is the order measured, not sorted.
    assert [s.amplitudes["R0001"] for s in steps] == [0.004, 0.001, 0.002]


def test_nsteps_counts_amplitudes_and_needs_no_catalog():
    schedule = AmplitudeSchedule.ramp(1e-3, 1e-2, 6)

    assert schedule.nsteps == 6
    assert len(schedule) == 6


def test_a_step_carries_its_rung_and_an_absolute_step_carries_none():
    catalog = a_catalog()

    scaled = AmplitudeSchedule.scaled(0.5, 2.0, 3).steps(catalog)
    assert [s.factor for s in scaled] == pytest.approx([0.5, 1.0, 2.0])

    absolute = AmplitudeSchedule.ramp(1e-3, 4e-3, 3).steps(catalog)
    assert [s.factor for s in absolute] == [None, None, None]


# ─── the base ─────────────────────────────────────────────────────────────────


def test_no_base_means_each_resonators_own_bias_amplitude():
    catalog = a_catalog()
    assert amplitudes_of(AmplitudeSchedule.fixed(), catalog) == [
        {"R0001": 0.001, "R0002": 0.002, "R0003": 0.004}
    ]


def test_a_number_is_the_base_for_every_resonator():
    assert amplitudes_of(AmplitudeSchedule.fixed(0.005), a_catalog()) == [
        {"R0001": 0.005, "R0002": 0.005, "R0003": 0.005}
    ]


def test_a_mapping_is_the_base_per_resonator():
    base = {"R0001": 0.004, "R0002": 0.006, "R0003": 0.008}
    assert amplitudes_of(AmplitudeSchedule.fixed(base), a_catalog()) == [base]


def test_a_partial_base_mapping_is_an_error_not_a_fallback():
    schedule = AmplitudeSchedule.fixed({"R0001": 0.004})
    with pytest.raises(ValueError, match="missing an amplitude"):
        schedule.steps(a_catalog())


def test_an_unknown_name_in_the_base_is_an_error():
    schedule = AmplitudeSchedule.fixed(
        {"R0001": 0.004, "R0002": 0.006, "R0003": 0.008, "R0009": 0.01}
    )
    with pytest.raises(ValueError, match="not being swept"):
        schedule.steps(a_catalog())


@pytest.mark.parametrize("sequence", [[0.004, 0.006, 0.008], (0.004, 0.006, 0.008)])
def test_a_positional_base_is_refused_alongside_a_catalog(sequence):
    schedule = AmplitudeSchedule.fixed(sequence)
    with pytest.raises(TypeError, match="positional sequence"):
        schedule.steps(a_catalog())


# ─── the ladder: relative multiplies the base ─────────────────────────────────


def test_a_relative_ladder_scales_every_resonator_by_its_own_base():
    catalog = a_catalog(amplitudes=(0.001, 0.002))
    steps = amplitudes_of(AmplitudeSchedule.scaled(0.5, 2.0, 3), catalog)

    assert steps[0] == pytest.approx({"R0001": 0.0005, "R0002": 0.001})
    assert steps[1] == pytest.approx({"R0001": 0.001, "R0002": 0.002})
    assert steps[2] == pytest.approx({"R0001": 0.002, "R0002": 0.004})


def test_a_relative_ladder_can_scale_a_base_of_your_own_choosing():
    """The gap in the dialog this replaces: per-resonator base *and* a ladder."""
    catalog = a_catalog(amplitudes=(0.001, 0.002))
    schedule = AmplitudeSchedule.scaled(
        1.0, 2.0, 2, base={"R0001": 0.01, "R0002": 0.02}
    )

    assert amplitudes_of(schedule, catalog) == pytest.approx(
        [{"R0001": 0.01, "R0002": 0.02}, {"R0001": 0.02, "R0002": 0.04}]
    )


def test_an_absolute_ladder_ignores_the_catalogs_amplitudes_entirely():
    catalog = a_catalog(amplitudes=(0.001, 0.002, 0.004))
    steps = amplitudes_of(AmplitudeSchedule.ramp(1e-3, 1e-2, 2), catalog)

    assert steps[0] == pytest.approx({"R0001": 1e-3, "R0002": 1e-3, "R0003": 1e-3})
    assert steps[1] == pytest.approx({"R0001": 1e-2, "R0002": 1e-2, "R0003": 1e-2})


def test_an_absolute_ladder_takes_no_base():
    """Its rungs *are* the amplitudes, so a base would have nothing to do."""
    with pytest.raises(ValueError, match="absolute ladder takes no base"):
        AmplitudeSchedule(ladder=(1e-3, 1e-2), relative=False, base=0.004)


# ─── spacing ──────────────────────────────────────────────────────────────────


def test_log_is_the_default_spacing_so_steps_are_equal_in_db():
    ladder = AmplitudeSchedule.ramp(1e-3, 1e-1, 3).ladder

    assert ladder == pytest.approx([1e-3, 1e-2, 1e-1])
    ratios = [b / a for a, b in zip(ladder, ladder[1:])]
    assert ratios == pytest.approx([ratios[0]] * len(ratios))


@pytest.mark.parametrize(
    "constructor, kwargs",
    [(AmplitudeSchedule.ramp, {}), (AmplitudeSchedule.scaled, {})],
)
def test_linear_spacing_is_still_available(constructor, kwargs):
    ladder = constructor(0.2, 0.6, 3, spacing="linear", **kwargs).ladder
    assert ladder == pytest.approx([0.2, 0.4, 0.6])


def test_an_unknown_spacing_is_an_error():
    with pytest.raises(ValueError, match="spacing="):
        AmplitudeSchedule.ramp(1e-3, 1e-2, 3, spacing="geometric")


def test_log_spacing_needs_positive_endpoints():
    with pytest.raises(ValueError, match="positive endpoints"):
        AmplitudeSchedule.scaled(0.0, 2.0, 3)


# ─── nsteps ───────────────────────────────────────────────────────────────────


def test_one_step_between_two_different_endpoints_is_an_error():
    """The dialog this replaces silently kept the start and dropped the stop."""
    with pytest.raises(ValueError, match="which of the two did you mean"):
        AmplitudeSchedule.ramp(1e-3, 1e-2, 1)


def test_one_step_is_fine_when_the_endpoints_agree():
    assert AmplitudeSchedule.ramp(1e-3, 1e-3, 1).ladder == (1e-3,)


@pytest.mark.parametrize("nsteps", [0, -1])
def test_a_schedule_needs_at_least_one_step(nsteps):
    with pytest.raises(ValueError, match="at least one step"):
        AmplitudeSchedule.ramp(1e-3, 1e-2, nsteps)


def test_an_empty_ladder_is_an_error():
    with pytest.raises(ValueError, match="measures nothing"):
        AmplitudeSchedule(ladder=())


# ─── the (0, 1] domain, caught before the first sweep ─────────────────────────


def test_a_ladder_that_overshoots_full_scale_is_caught_with_the_step_and_name():
    catalog = a_catalog(amplitudes=(0.001, 0.5))
    schedule = AmplitudeSchedule.scaled(1.0, 4.0, 3, spacing="linear")

    with pytest.raises(ValueError) as excinfo:
        schedule.steps(catalog)

    message = str(excinfo.value)
    assert "Step 2" in message  # 0.5 × 4 = 2.0
    assert "R0002" in message
    assert "R0001" not in message  # 0.001 × 4 is fine, and is not blamed


def test_an_absolute_rung_above_full_scale_is_caught_at_construction():
    with pytest.raises(ValueError, match=r"outside \(0, 1\]"):
        AmplitudeSchedule.explicit([0.5, 1.5])


def test_a_negative_amplitude_says_it_might_be_dbm():
    with pytest.raises(ValueError, match="dBm"):
        AmplitudeSchedule.explicit([-60.0])


def test_a_relative_rung_of_zero_is_an_error():
    with pytest.raises(ValueError, match="not positive"):
        AmplitudeSchedule.scaled(0.0, 1.0, 3, spacing="linear")


def test_a_zero_base_is_caught_when_it_meets_the_ladder():
    """A base of zero survives construction — nothing knows it yet — and is
    refused once a step is resolved from it."""
    schedule = AmplitudeSchedule.fixed(
        {"R0001": 0.0, "R0002": 0.002, "R0003": 0.004}
    )
    with pytest.raises(ValueError, match="at or below zero"):
        schedule.steps(a_catalog())


def test_full_scale_itself_is_allowed():
    assert AmplitudeSchedule.explicit([1.0]).steps(a_catalog())[0].amplitudes == {
        "R0001": 1.0,
        "R0002": 1.0,
        "R0003": 1.0,
    }


# ─── scheduling by name, for a bare frequency list ────────────────────────────


def test_names_stand_in_for_a_catalog():
    schedule = AmplitudeSchedule.ramp(1e-3, 1e-2, 2)
    steps = schedule.steps(["S0001", "S0002"])

    assert steps[0].amplitudes == pytest.approx({"S0001": 1e-3, "S0002": 1e-3})
    assert steps[1].amplitudes == pytest.approx({"S0001": 1e-2, "S0002": 1e-2})


def test_names_have_no_amplitude_to_fall_back_to():
    with pytest.raises(ValueError, match="required when scheduling by name"):
        AmplitudeSchedule.fixed().steps(["S0001", "S0002"])


def test_a_positional_base_is_accepted_alongside_names():
    """Where the ordering is the caller's own, as multisweep's amp= allows."""
    schedule = AmplitudeSchedule.scaled(1.0, 2.0, 2, base=[0.001, 0.002])
    assert amplitudes_of(schedule, ["low", "high"]) == pytest.approx(
        [{"low": 0.001, "high": 0.002}, {"low": 0.002, "high": 0.004}]
    )


def test_a_mismatched_positional_base_is_an_error():
    schedule = AmplitudeSchedule.fixed([0.001, 0.002, 0.003])
    with pytest.raises(ValueError, match="3 amplitudes for 2 sweeps"):
        schedule.steps(["low", "high"])


def test_a_bare_string_target_is_refused_rather_than_split_into_letters():
    with pytest.raises(TypeError, match="single characters"):
        AmplitudeSchedule.fixed(0.001).steps("S0001")


def test_a_mapping_target_points_at_base():
    with pytest.raises(TypeError, match="pass it as base="):
        AmplitudeSchedule.fixed().steps({"S0001": 0.001})


def test_duplicate_names_are_an_error():
    with pytest.raises(ValueError, match="Duplicate sweep names"):
        AmplitudeSchedule.fixed(0.001).steps(["S0001", "S0001"])


def test_an_empty_target_is_an_error():
    with pytest.raises(ValueError, match="nothing to sweep"):
        AmplitudeSchedule.fixed(0.001).steps([])


def test_names_must_be_strings():
    with pytest.raises(TypeError, match="must be strings"):
        AmplitudeSchedule.fixed(0.001).steps([1, 2])


# ─── describe ─────────────────────────────────────────────────────────────────


def test_describe_reports_the_sweep_count_across_both_axes():
    schedule = AmplitudeSchedule.ramp(1e-3, 1e-2, 4)
    described = schedule.describe(a_catalog(), n_directions=2)

    assert described["nsteps"] == 4
    assert described["n_directions"] == 2
    assert described["n_sweeps"] == 8
    assert described["n_sweep_targets"] == 3


def test_describe_gives_each_resonators_own_amplitude_range():
    catalog = a_catalog(amplitudes=(0.001, 0.002))
    described = AmplitudeSchedule.scaled(1.0, 2.0, 2).describe(catalog)

    assert described["amplitude_range_by_name"] == pytest.approx(
        {"R0001": (0.001, 0.002), "R0002": (0.002, 0.004)}
    )
    assert described["amplitude_min"] == pytest.approx(0.001)
    assert described["amplitude_max"] == pytest.approx(0.004)


def test_describe_reports_spacing_it_did_not_compute_with():
    assert AmplitudeSchedule.ramp(1e-3, 1e-2, 3).describe(a_catalog())["spacing"] == "log"
    assert AmplitudeSchedule.explicit([1e-3]).describe(a_catalog())["spacing"] == "explicit"
    assert AmplitudeSchedule.fixed().describe(a_catalog())["spacing"] == "none"


def test_describe_converts_to_dbm_when_told_the_dac_scale():
    described = AmplitudeSchedule.explicit([0.1]).describe(
        a_catalog(), dac_scale_dbm=-10.0
    )
    assert described["power_dbm_max"] == pytest.approx(-30.0)


# ─── validate ─────────────────────────────────────────────────────────────────


def severities(issues):
    return [severity for severity, _ in issues]


def test_a_sound_schedule_validates_with_only_an_info_line():
    issues = AmplitudeSchedule.ramp(1e-3, 1e-2, 4).validate(a_catalog(), n_directions=2)

    assert severities(issues) == ["info"]
    assert "8 sweeps" in issues[0][1]


def test_validate_reports_an_overshoot_it_would_otherwise_raise_on():
    catalog = a_catalog(amplitudes=(0.001, 0.5))
    schedule = AmplitudeSchedule.scaled(1.0, 4.0, 3, spacing="linear")

    issues = schedule.validate(catalog)

    assert "error" in severities(issues)
    assert any("R0002" in message for _, message in issues)


def test_validate_never_raises_on_a_structurally_bad_input():
    """A dialog rendering a half-entered form wants text, not a traceback."""
    issues = AmplitudeSchedule.fixed({"R0001": 0.004}).validate(a_catalog())

    assert severities(issues) == ["error"]
    assert "missing an amplitude" in issues[0][1]


def test_validate_warns_about_a_repeated_rung():
    issues = AmplitudeSchedule.explicit([1e-3, 1e-3, 1e-2]).validate(a_catalog())

    assert "warning" in severities(issues)
    assert any("twice" in message for _, message in issues)


def test_validate_does_not_complain_about_a_descending_ladder():
    issues = AmplitudeSchedule.explicit([1e-2, 3e-3, 1e-3]).validate(a_catalog())
    assert severities(issues) == ["info"]


# ─── persistence and identity ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "schedule",
    [
        AmplitudeSchedule(),
        AmplitudeSchedule.fixed(0.005),
        AmplitudeSchedule.fixed({"R0001": 0.004}),
        AmplitudeSchedule.scaled(0.5, 2.0, 5),
        AmplitudeSchedule.scaled(0.5, 2.0, 5, spacing="linear", base=[0.001, 0.002]),
        AmplitudeSchedule.ramp(1e-3, 1e-2, 6),
        AmplitudeSchedule.explicit([1e-3, 3e-3, 1e-2]),
    ],
)
def test_a_schedule_round_trips_through_its_dict(schedule):
    restored = AmplitudeSchedule.from_dict(schedule.to_dict())

    assert restored == schedule
    assert restored.spacing == schedule.spacing
    assert restored.ladder == pytest.approx(schedule.ladder)


def test_to_dict_is_plain_builtins_so_it_pickles_with_the_rest_of_a_result():
    d = AmplitudeSchedule.scaled(0.5, 2.0, 3, base={"R0001": 0.004}).to_dict()

    assert pickle.loads(pickle.dumps(d)) == d
    assert all(isinstance(v, float) for v in d["ladder"])
    assert isinstance(d["base"], dict)


def test_a_dict_from_another_version_is_refused():
    d = AmplitudeSchedule.ramp(1e-3, 1e-2, 3).to_dict()
    d["schema_version"] = 2

    with pytest.raises(ValueError, match="schema_version"):
        AmplitudeSchedule.from_dict(d)


def test_two_schedules_that_measure_the_same_thing_are_equal():
    """Spacing is provenance, not behaviour: it does not split identity."""
    generated = AmplitudeSchedule.ramp(1e-3, 1e-2, 2)
    spelled_out = AmplitudeSchedule.explicit([1e-3, 1e-2])

    assert generated == spelled_out
    assert generated.spacing != spelled_out.spacing


def test_the_repr_says_what_the_schedule_will_do():
    assert "6 steps" in repr(AmplitudeSchedule.ramp(1e-3, 1e-2, 6))
    assert "absolute" in repr(AmplitudeSchedule.ramp(1e-3, 1e-2, 6))
    assert "catalog's own" in repr(AmplitudeSchedule.scaled(0.5, 2.0, 3))
    assert "1 step," in repr(AmplitudeSchedule.fixed())


def test_a_step_repr_says_how_many_sweeps_at_what():
    step = AmplitudeSchedule.scaled(2.0, 2.0, 1).steps(a_catalog())[0]
    assert "3 sweeps" in repr(step)
    assert "×2" in repr(step)


def test_a_step_converts_to_a_plain_dict():
    step = AmplitudeSchedule.ramp(1e-3, 1e-3, 1).steps(["S0001"])[0]
    assert step.to_dict() == {
        "step": 0,
        "amplitudes": {"S0001": 1e-3},
        "factor": None,
    }


# ─── the amplitudes go straight into multisweep ───────────────────────────────


def test_a_step_is_accepted_by_multisweeps_own_amplitude_resolution():
    """The contract that lets the driver be a loop and nothing more."""
    from rfmux.algorithms.measurement.multisweep import _resolve_amplitudes

    catalog = a_catalog()
    step = AmplitudeSchedule.scaled(2.0, 2.0, 1).steps(catalog)[0]

    resolved = _resolve_amplitudes(
        [r.name for r in catalog],
        step.amplitudes,
        defaults={r.name: r.bias.amplitude for r in catalog},
        allow_sequence=False,
    )
    assert resolved == pytest.approx({"R0001": 0.002, "R0002": 0.004, "R0003": 0.008})


# ─── the driver's output: packing and reading ─────────────────────────────────


def a_sweep_entry(name, amplitude, direction):
    """One resonator's worth of a multisweep return, cut down to what the
    readers actually touch."""
    return {
        "name": name,
        "sweep_amplitude": amplitude,
        "sweep_direction": direction,
        "original_center_frequency": 1.0e9,
    }


# So `catalog=None` can mean "a frequency-list sweep, which has none" rather
# than "you didn't say".
_UNSET = object()


def packed(
    schedule=None, catalog=_UNSET, names=None, directions=("upward",), **overrides
):
    """A packed result, built the way the driver builds one."""
    if catalog is _UNSET:
        catalog = a_catalog()
    schedule = schedule if schedule is not None else AmplitudeSchedule()
    target = catalog if catalog is not None else list(names)

    sweeps = {
        step.step: {
            direction: {
                name: a_sweep_entry(name, amplitude, direction)
                for name, amplitude in step.amplitudes.items()
            }
            for direction in directions
        }
        for step in schedule.steps(target)
    }

    kwargs = dict(
        module=2,
        amp_schedule=schedule,
        directions=directions,
        span_hz=200e3,
        npoints_per_sweep=101,
        nsamps=10,
        bias_frequency_method="max-diq",
        rotate_saved_data=False,
        apply_df_calibration=True,
        catalog=catalog,
        names=names,
        requested_module=None,
    )
    kwargs.update(overrides)
    return pack_results(sweeps, **kwargs)


def test_pack_results_puts_the_iterations_under_results_keyed_by_number():
    result = packed(
        schedule=AmplitudeSchedule.ramp(1e-3, 1e-2, 3),
        directions=("upward", "downward"),
    )

    assert list(result["results"]) == [0, 1, 2]
    for iteration in result["results"].values():
        assert set(iteration) == {"upward", "downward"}


def test_pack_results_records_the_call_as_made():
    schedule = AmplitudeSchedule.ramp(1e-3, 1e-2, 2)
    catalog = a_catalog()
    result = packed(schedule=schedule, catalog=catalog, nsamps=7)

    params = result["call_params"]
    assert params["amp_schedule"] == schedule.to_dict()
    assert params["catalog"] == catalog.to_dict()
    assert params["nsamps"] == 7
    assert params["module"] is None  # as passed, not as resolved
    assert result["module"] == 2  # resolved, never None
    assert result["schema_version"] == RESULTS_SCHEMA_VERSION


def test_pack_results_is_plain_builtins():
    result = packed(schedule=AmplitudeSchedule.ramp(1e-3, 1e-2, 2))
    assert pickle.loads(pickle.dumps(result)) == result


# ─── collect_amplitude_iterations_for ─────────────────────────────────────────


def test_one_resonators_sweeps_across_every_iteration():
    result = packed(
        schedule=AmplitudeSchedule.scaled(1.0, 4.0, 3),
        catalog=a_catalog(amplitudes=(0.001, 0.002)),
        directions=("upward", "downward"),
    )

    collected = collect_amplitude_iterations_for(result, "R0001")

    assert list(collected) == [0, 1, 2]
    assert set(collected[0]) == {"upward", "downward"}
    assert [
        c["upward"]["sweep_amplitude"] for c in collected.values()
    ] == pytest.approx([0.001, 0.002, 0.004])
    # and only that resonator
    assert all(
        entry["name"] == "R0001"
        for by_direction in collected.values()
        for entry in by_direction.values()
    )


def test_collecting_keeps_the_order_measured_rather_than_sorting():
    """An explicit ladder may run in any order, and re-sorting would lose the
    order things actually happened in."""
    result = packed(schedule=AmplitudeSchedule.explicit([0.01, 0.001, 0.004]))

    collected = collect_amplitude_iterations_for(result, "R0001")

    assert [
        c["upward"]["sweep_amplitude"] for c in collected.values()
    ] == pytest.approx([0.01, 0.001, 0.004])


def test_collecting_a_resonator_that_was_not_swept_is_an_error():
    result = packed()
    with pytest.raises(KeyError, match="was not swept"):
        collect_amplitude_iterations_for(result, "R9999")


def test_a_reader_handed_the_wrong_dict_says_so():
    result = packed()
    with pytest.raises(TypeError, match="not one of its parts"):
        collect_amplitude_iterations_for(result["results"], "R0001")


# ─── get_amplitudes_at_iteration ──────────────────────────────────────────────


def test_the_amplitudes_of_an_iteration_come_from_the_sweeps_themselves():
    result = packed(
        schedule=AmplitudeSchedule.scaled(1.0, 2.0, 2),
        catalog=a_catalog(amplitudes=(0.001, 0.002, 0.004)),
    )

    assert get_amplitudes_at_iteration(result, 0) == pytest.approx(
        {"R0001": 0.001, "R0002": 0.002, "R0003": 0.004}
    )
    assert get_amplitudes_at_iteration(result, 1) == pytest.approx(
        {"R0001": 0.002, "R0002": 0.004, "R0003": 0.008}
    )


def test_an_absolute_ladder_probes_everything_at_the_same_amplitude():
    result = packed(schedule=AmplitudeSchedule.ramp(1e-3, 1e-2, 2))

    amplitudes = get_amplitudes_at_iteration(result, 1)
    assert list(amplitudes.values()) == pytest.approx([1e-2, 1e-2, 1e-2])


def test_asking_for_an_iteration_that_does_not_exist_is_an_error():
    result = packed(schedule=AmplitudeSchedule.ramp(1e-3, 1e-2, 2))
    with pytest.raises(KeyError, match=r"No iteration 5"):
        get_amplitudes_at_iteration(result, 5)


# ─── find_iteration_number_matching_amplitude ─────────────────────────────────


def test_the_iteration_nearest_a_given_amplitude():
    result = packed(schedule=AmplitudeSchedule.explicit([1e-3, 1e-2, 1e-1]))

    assert find_iteration_number_matching_amplitude(result, "R0001", 1e-2) == 1
    # nearest, not exact — a ladder's floats rarely compare equal
    assert find_iteration_number_matching_amplitude(result, "R0001", 9.6e-3) == 1


def test_without_an_amplitude_it_finds_where_the_resonator_is_biased():
    """The usual question: which iteration was taken at the bias point?"""
    catalog = a_catalog(amplitudes=(0.001, 0.002, 0.004))
    result = packed(
        schedule=AmplitudeSchedule.scaled(0.25, 4.0, 5), catalog=catalog
    )

    # scaled() puts ×1 in the middle, so every resonator's bias amplitude is
    # iteration 2 — whatever its own amplitude happens to be.
    for name in ("R0001", "R0002", "R0003"):
        assert find_iteration_number_matching_amplitude(result, name) == 2


def test_a_relative_ladder_gives_each_resonator_its_own_answer():
    """Which is why the reader takes a name: R0001 and R0002 share an iteration
    number and nothing else."""
    catalog = a_catalog(amplitudes=(0.001, 0.01))
    result = packed(schedule=AmplitudeSchedule.scaled(1.0, 4.0, 3), catalog=catalog)

    assert find_iteration_number_matching_amplitude(result, "R0001", 0.004) == 2
    assert find_iteration_number_matching_amplitude(result, "R0002", 0.004) == 0


def test_a_frequency_list_result_has_no_bias_amplitude_to_fall_back_on():
    schedule = AmplitudeSchedule.ramp(1e-3, 1e-2, 2)
    result = packed(
        schedule=schedule,
        catalog=None,
        center_frequencies=[1.0e9, 1.1e9],
        names=["low", "high"],
    )

    with pytest.raises(ValueError, match="no catalog to take one from"):
        find_iteration_number_matching_amplitude(result, "low")

    # but an explicit amplitude still works
    assert find_iteration_number_matching_amplitude(result, "low", 1e-2) == 1
