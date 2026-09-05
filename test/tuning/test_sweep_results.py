"""Behaviour of the shape a sweep comes back in.

Packing and reading, with no board and no driver in sight — which is most of
what a consumer of a sweep ever touches. The emphasis is on the properties the
rest of the codebase leans on: one shape whatever measured it, a module
identifier at the top even for one module, nothing stored twice to be readable,
and readers that refuse the container rather than walking it as if it were an
envelope.
"""

import pickle

import pytest

from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog
from rfmux.tuning.multisweep_amplitudes import AmplitudeSchedule
from rfmux.tuning.sweep_results import (
    RESULTS_SCHEMA_VERSION,
    collect_amplitude_iterations_for,
    find_iteration_matching_amplitude,
    get_amplitudes_at_iteration,
    merge_modules,
    pack_results,
    pack_sweep,
)

pytestmark = pytest.mark.portable

MODULE_ID = "crs0030_rmod2"


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


def a_sweep_entry(amplitude, direction):
    """One resonator's worth of a multisweep return, cut down to what the
    readers actually touch."""
    return {
        "sweep_amplitude": amplitude,
        "sweep_direction": direction,
        "original_center_frequency": 1.0e9,
    }


# So `catalog=None` can mean "a frequency-list sweep, which has none" rather
# than "you didn't say".
_UNSET = object()


def container(
    schedule=None, catalog=_UNSET, names=None, directions=("upward",), **overrides
):
    """A ladder's whole return, keyed by module, as the driver builds it."""
    if catalog is _UNSET:
        catalog = a_catalog()
    schedule = schedule if schedule is not None else AmplitudeSchedule()
    target = catalog if catalog is not None else list(names)

    sweeps = {
        step.step: {
            direction: {
                name: a_sweep_entry(amplitude, direction)
                for name, amplitude in step.amplitudes.items()
            }
            for direction in directions
        }
        for step in schedule.steps(target)
    }

    kwargs = dict(
        module_id=MODULE_ID,
        module=2,
        amp_schedule=schedule,
        directions=directions,
        span_hz=200e3,
        npoints_per_sweep=101,
        nsamps=10,
        catalog=catalog,
        names=names,
        requested_module=None,
    )
    kwargs.update(overrides)
    return pack_results(sweeps, **kwargs)


def packed(**kwargs):
    """One module's envelope — what every reader and fitter is handed."""
    return container(**kwargs)[MODULE_ID]


def swept(sections=None, direction="upward", **overrides):
    """A single sweep's whole return, keyed by module."""
    if sections is None:
        sections = {"R0001": a_sweep_entry(0.001, direction)}

    kwargs = dict(
        module_id=MODULE_ID,
        module=2,
        sweep_direction=direction,
        span_hz=200e3,
        npoints_per_sweep=101,
        nsamps=10,
        amp=None,
        catalog=a_catalog(),
        requested_module=None,
    )
    kwargs.update(overrides)
    return pack_sweep(sections, **kwargs)


# ─── one shape, whatever measured it ──────────────────────────────────────────


def test_a_single_sweep_is_one_iteration_in_one_direction():
    result = swept(direction="downward")[MODULE_ID]

    assert list(result["results"]) == [0]
    assert list(result["results"][0]) == ["downward"]
    assert list(result["results"][0]["downward"]) == ["R0001"]


def test_a_sweep_and_a_ladder_nest_identically():
    """The property the fitters rely on: nothing downstream has to ask which
    macro produced a result."""
    sweep = swept()[MODULE_ID]
    ladder = packed(schedule=AmplitudeSchedule.ramp(1e-3, 1e-2, 3))

    assert set(sweep) == set(ladder)
    for result in (sweep, ladder):
        for by_direction in result["results"].values():
            for sections in by_direction.values():
                assert all(isinstance(entry, dict) for entry in sections.values())


def test_a_sweep_records_the_call_as_made():
    result = swept(amp={"R0001": 0.004}, nsamps=7, direction="downward")[MODULE_ID]
    params = result["call_params"]

    assert params["amp"] == {"R0001": 0.004}
    assert params["sweep_direction"] == "downward"
    assert params["nsamps"] == 7
    assert params["span_hz"] == 200e3
    assert params["catalog"]["module"] == 2
    # Verbatim: the module was worked out from the catalog, not asked for.
    assert params["module"] is None


def test_a_sweep_records_a_bare_amp_verbatim_rather_than_resolving_it():
    """What each resonator was probed at is already `sweep_amplitude` in its own
    entry, so call_params can stay a record of the request."""
    result = swept(amp=0.005)[MODULE_ID]

    assert result["call_params"]["amp"] == 0.005


def test_the_two_packers_agree_on_the_shared_call_params():
    shared = set(swept()[MODULE_ID]["call_params"]) & set(packed()["call_params"])

    assert shared == {
        "catalog",
        "center_frequencies",
        "names",
        "span_hz",
        "npoints_per_sweep",
        "nsamps",
        "module",
    }


def test_the_amplitude_spec_is_the_only_thing_that_differs():
    sweep_only = set(swept()[MODULE_ID]["call_params"]) - set(packed()["call_params"])
    ladder_only = set(packed()["call_params"]) - set(swept()[MODULE_ID]["call_params"])

    assert sweep_only == {"amp", "sweep_direction"}
    assert ladder_only == {"amp_schedule", "directions"}


def test_both_packers_stamp_the_schema_version():
    assert swept()[MODULE_ID]["schema_version"] == RESULTS_SCHEMA_VERSION
    assert packed()["schema_version"] == RESULTS_SCHEMA_VERSION


# ─── the module is the outermost key, even when there is one ──────────────────


def test_one_module_is_still_keyed_by_module():
    """So a script written as a loop over .items() is the same script for one
    module and for four."""
    for result in (swept(), container()):
        assert list(result) == [MODULE_ID]


def test_the_envelope_keeps_the_module_number_as_an_int():
    """The key names the board and module; the int is what goes back into a
    hardware call, and re-parsing it out of the string would be worse."""
    assert swept()[MODULE_ID]["module"] == 2
    assert container()[MODULE_ID]["module"] == 2


def test_merging_modules_keys_them_rather_than_ordering_them():
    merged = merge_modules([
        swept(module_id="crs0030_rmod1", module=1),
        swept(module_id="crs0030_rmod2", module=2),
    ])

    assert list(merged) == ["crs0030_rmod1", "crs0030_rmod2"]
    assert merged["crs0030_rmod1"]["module"] == 1
    assert merged["crs0030_rmod2"]["module"] == 2


def test_merging_a_repeated_module_is_an_error_not_an_overwrite():
    with pytest.raises(ValueError, match="appears twice"):
        merge_modules([swept(), swept()])


def test_merging_nothing_is_an_empty_result_rather_than_an_error():
    assert merge_modules([]) == {}


# ─── the container is refused, not walked ─────────────────────────────────────


@pytest.mark.parametrize(
    "reader",
    [
        lambda r: collect_amplitude_iterations_for(r, "R0001"),
        lambda r: get_amplitudes_at_iteration(r, 0),
        lambda r: find_iteration_matching_amplitude(r, "R0001"),
    ],
)
def test_a_reader_handed_the_whole_container_says_how_to_index_it(reader):
    with pytest.raises(TypeError, match="keyed by module"):
        reader(container())


def test_the_refusal_names_the_modules_it_found():
    whole = merge_modules([
        swept(module_id="crs0030_rmod1", module=1),
        swept(module_id="crs0030_rmod2", module=2),
    ])

    with pytest.raises(TypeError, match=r"crs0030_rmod1.*crs0030_rmod2"):
        get_amplitudes_at_iteration(whole, 0)


def test_one_modules_envelope_is_what_the_readers_take():
    """The other half of the refusal above: indexing in is all it takes."""
    whole = container(schedule=AmplitudeSchedule.ramp(1e-3, 1e-2, 2))

    assert get_amplitudes_at_iteration(whole[MODULE_ID], 0) == pytest.approx(
        {"R0001": 1e-3, "R0002": 1e-3, "R0003": 1e-3}
    )


# ─── packing and reading a ladder ─────────────────────────────────────────────


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
        schedule=AmplitudeSchedule.multiplicative(1.0, 4.0, 3),
        catalog=a_catalog(amplitudes=(0.001, 0.002)),
        directions=("upward", "downward"),
    )

    collected = collect_amplitude_iterations_for(result, "R0001")

    assert list(collected) == [0, 1, 2]
    assert set(collected[0]) == {"upward", "downward"}
    assert [
        c["upward"]["sweep_amplitude"] for c in collected.values()
    ] == pytest.approx([0.001, 0.002, 0.004])
    # and only that resonator. A section entry does not name itself — it is
    # already keyed by one — so this checks the collector handed back the very
    # entries filed under "R0001", not merely ones that look like them.
    assert all(
        entry is result["results"][iteration][direction]["R0001"]
        for iteration, by_direction in collected.items()
        for direction, entry in by_direction.items()
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
        schedule=AmplitudeSchedule.multiplicative(1.0, 2.0, 2),
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


# ─── find_iteration_matching_amplitude ─────────────────────────────────


def test_the_iteration_nearest_a_given_amplitude():
    result = packed(schedule=AmplitudeSchedule.explicit([1e-3, 1e-2, 1e-1]))

    assert find_iteration_matching_amplitude(result, "R0001", 1e-2) == 1
    # nearest, not exact — a ladder's floats rarely compare equal
    assert find_iteration_matching_amplitude(result, "R0001", 9.6e-3) == 1


def test_without_an_amplitude_it_finds_where_the_resonator_is_biased():
    """The usual question: which iteration was taken at the bias point?"""
    catalog = a_catalog(amplitudes=(0.001, 0.002, 0.004))
    result = packed(
        schedule=AmplitudeSchedule.multiplicative(0.25, 4.0, 5), catalog=catalog
    )

    # multiplicative() puts ×1 in the middle, so every resonator's bias amplitude is
    # iteration 2 — whatever its own amplitude happens to be.
    for name in ("R0001", "R0002", "R0003"):
        assert find_iteration_matching_amplitude(result, name) == 2


def test_a_sweep_holding_an_older_catalog_snapshot_still_answers():
    """Catalog schema_version 1 listed its resonators instead of keying them."""
    catalog = a_catalog(amplitudes=(0.001, 0.002, 0.004))
    result = packed(
        schedule=AmplitudeSchedule.multiplicative(0.25, 4.0, 5), catalog=catalog
    )
    snapshot = result["call_params"]["catalog"]
    snapshot["schema_version"] = 1
    snapshot["resonators"] = [
        {"name": name, **entry} for name, entry in snapshot["resonators"].items()
    ]

    assert find_iteration_matching_amplitude(result, "R0002") == 2


def test_a_relative_ladder_gives_each_resonator_its_own_answer():
    """Which is why the reader takes a name: R0001 and R0002 share an iteration
    number and nothing else."""
    catalog = a_catalog(amplitudes=(0.001, 0.01))
    result = packed(schedule=AmplitudeSchedule.multiplicative(1.0, 4.0, 3), catalog=catalog)

    assert find_iteration_matching_amplitude(result, "R0001", 0.004) == 2
    assert find_iteration_matching_amplitude(result, "R0002", 0.004) == 0


def test_a_frequency_list_result_has_no_bias_amplitude_to_fall_back_on():
    schedule = AmplitudeSchedule.ramp(1e-3, 1e-2, 2)
    result = packed(
        schedule=schedule,
        catalog=None,
        center_frequencies=[1.0e9, 1.1e9],
        names=["low", "high"],
    )

    with pytest.raises(ValueError, match="no catalog to take one from"):
        find_iteration_matching_amplitude(result, "low")

    # but an explicit amplitude still works
    assert find_iteration_matching_amplitude(result, "low", 1e-2) == 1
