"""Behaviour of multiamp_multisweep: the loop over amplitude steps.

The driver's only contact with a board is ``crs.multisweep``, so a fake CRS with
one async method exercises all of it — the order sweeps are taken in, the
amplitudes each one is asked for, what is forwarded unchanged, and the shape of
the packed result. The measurement itself belongs to multisweep and is not
re-tested here.
"""

import pytest

from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog
from rfmux.tuning import AmplitudeSchedule
from rfmux.tuning.sweep_results import pack_sweep
from rfmux.algorithms.measurement.multiamp_multisweep import multiamp_multisweep

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


# @macro typechecks its first argument against CRS, so the tests drive the
# undecorated function underneath it — the loop is what is under test, not the
# decorator's class check.
drive_macro = multiamp_multisweep.__wrapped__


class FakeReadoutModule:
    """Enough of a ReadoutModule to name itself, which is how the driver gets
    the module identifier its result is keyed by."""

    def __init__(self, module):
        self._module = module

    def index(self):
        return f"crs0030_rmod{self._module}"


class FakeCRS:
    """A CRS that only knows how to be swept, and records how it was asked.

    The driver's only contact with a board is ``crs.multisweep``, so one async
    method is nearly the whole surface it needs — plus ``crs.module[m].index()``
    for the key its result comes back under.
    """

    def __init__(self, sweep_result=None):
        self.calls = []
        self._sweep_result = sweep_result
        self.module = {m: FakeReadoutModule(m) for m in range(1, 9)}

    async def multisweep(self, catalog=None, **kwargs):
        self.calls.append({"catalog": catalog, **kwargs})
        return self._packed(self._sections(catalog, kwargs), kwargs)

    def _sections(self, catalog, kwargs):
        """The {name: entry} a sweep measured, or whatever a test substituted."""
        if callable(self._sweep_result):
            return self._sweep_result(catalog, kwargs)
        if self._sweep_result is not None:
            return self._sweep_result
        # Stand in for one sweep's measurement: keyed by name, and carrying the
        # amplitude it was asked for, as the real one does.
        names = (
            [r.name for r in catalog]
            if catalog is not None
            else kwargs["names"]
        )
        return {
            name: {
                "name": name,
                "sweep_amplitude": kwargs["amp"][name],
                "sweep_direction": kwargs["sweep_direction"],
            }
            for name in names
        }

    def _packed(self, sections, kwargs):
        """Through the real packer, so the fake cannot drift from the shape the
        driver actually has to unwrap."""
        return pack_sweep(
            sections,
            module_id=self.module[kwargs["module"]].index(),
            module=kwargs["module"],
            sweep_direction=kwargs["sweep_direction"],
            span_hz=kwargs["span_hz"],
            npoints_per_sweep=kwargs["npoints_per_sweep"],
            nsamps=kwargs["nsamps"],
            amp=kwargs["amp"],
        )


async def drive(crs, catalog=None, **kwargs):
    """The macro's own defaults, minus the two every call needs.

    Returns the one module's envelope rather than the container the macro
    returns, because these tests are about the driver's loop. That the result is
    keyed by module at all is checked below, and the container's own behaviour —
    merging, and being refused where an envelope was wanted — lives in
    ``test/tuning/test_sweep_results.py``.
    """
    kwargs.setdefault("span_hz", 200e3)
    kwargs.setdefault("npoints_per_sweep", 101)
    container = await drive_macro(crs, catalog, **kwargs)
    return container[module_id_of(container)]


def module_id_of(container):
    """The single module a driver call comes back under."""
    (module_id,) = container
    return module_id


# ─── the loop: step outer, direction inner ────────────────────────────────────


@pytest.mark.asyncio
async def test_every_step_is_swept_in_every_direction():
    crs = FakeCRS()
    catalog = a_catalog()

    await drive(
        crs,
        catalog,
        amp_schedule=AmplitudeSchedule.ramp(1e-3, 1e-2, 3),
        directions=("upward", "downward"),
    )

    assert len(crs.calls) == 6


@pytest.mark.asyncio
async def test_amplitude_is_the_outer_axis_and_direction_the_inner():
    """Each step's pair is measured together, so amplitude marches
    monotonically — which is what a bifurcation walk wants."""
    crs = FakeCRS()

    await drive(
        crs,
        a_catalog(),
        amp_schedule=AmplitudeSchedule.ramp(1e-3, 1e-1, 3),
        directions=("upward", "downward"),
    )

    assert [
        (c["amp"]["R0001"], c["sweep_direction"]) for c in crs.calls
    ] == pytest.approx([
        (1e-3, "upward"), (1e-3, "downward"),
        (1e-2, "upward"), (1e-2, "downward"),
        (1e-1, "upward"), (1e-1, "downward"),
    ])


@pytest.mark.asyncio
async def test_the_order_of_directions_is_the_order_measured():
    crs = FakeCRS()
    await drive(crs, a_catalog(), directions=("downward", "upward"))

    assert [c["sweep_direction"] for c in crs.calls] == ["downward", "upward"]


@pytest.mark.asyncio
async def test_one_direction_is_the_default():
    crs = FakeCRS()
    await drive(crs, a_catalog(), amp_schedule=AmplitudeSchedule.ramp(1e-3, 1e-2, 4))

    assert len(crs.calls) == 4
    assert {c["sweep_direction"] for c in crs.calls} == {"upward"}


@pytest.mark.asyncio
async def test_each_sweep_gets_the_amplitudes_of_its_own_step():
    crs = FakeCRS()
    catalog = a_catalog(amplitudes=(0.001, 0.002))

    await drive(crs, catalog, amp_schedule=AmplitudeSchedule.multiplicative(1.0, 2.0, 2))

    assert crs.calls[0]["amp"] == pytest.approx({"R0001": 0.001, "R0002": 0.002})
    assert crs.calls[1]["amp"] == pytest.approx({"R0001": 0.002, "R0002": 0.004})


@pytest.mark.asyncio
async def test_no_schedule_means_one_sweep_at_the_catalogs_own_amplitudes():
    crs = FakeCRS()
    catalog = a_catalog()

    result = await drive(crs, catalog, directions=("upward", "downward"))

    assert len(crs.calls) == 2
    assert crs.calls[0]["amp"] == pytest.approx(
        {"R0001": 0.001, "R0002": 0.002, "R0003": 0.004}
    )
    assert list(result["results"]) == [0]


# ─── what is forwarded, and what is not ───────────────────────────────────────


@pytest.mark.asyncio
async def test_the_sweep_parameters_reach_multisweep_unchanged():
    crs = FakeCRS()

    await drive(
        crs,
        a_catalog(),
        span_hz=123e3,
        npoints_per_sweep=57,
        nsamps=42,
    )

    call = crs.calls[0]
    assert call["span_hz"] == 123e3
    assert call["npoints_per_sweep"] == 57
    assert call["nsamps"] == 42


@pytest.mark.asyncio
async def test_the_catalog_is_passed_through_untouched():
    crs = FakeCRS()
    catalog = a_catalog()
    before = catalog.to_dict()

    await drive(crs, catalog, amp_schedule=AmplitudeSchedule.multiplicative(0.5, 2.0, 3))

    assert all(c["catalog"] is catalog for c in crs.calls)
    assert catalog.to_dict() == before  # the driver mutates nothing


@pytest.mark.asyncio
async def test_the_module_is_taken_from_the_catalog():
    crs = FakeCRS()
    result = await drive(crs, a_catalog())

    assert result["module"] == 2
    assert crs.calls[0]["module"] == 2


# ─── a bare frequency list ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_frequency_list_is_swept_at_a_ladder_of_absolute_amplitudes():
    crs = FakeCRS()

    result = await drive(
        crs,
        center_frequencies=[1.0e9, 1.1e9],
        module=2,
        amp_schedule=AmplitudeSchedule.ramp(1e-4, 1e-2, 3),
    )

    assert len(crs.calls) == 3
    assert crs.calls[0]["amp"] == pytest.approx({"S0001": 1e-4, "S0002": 1e-4})
    assert set(result["results"][0]["upward"]) == {"S0001", "S0002"}


@pytest.mark.asyncio
async def test_section_names_are_resolved_once_and_passed_to_every_sweep():
    """So the schedule's keys and the results' keys are the same strings by
    construction, not by both happening to generate S0001…"""
    crs = FakeCRS()

    await drive(
        crs,
        center_frequencies=[1.0e9, 1.1e9],
        module=2,
        amp_schedule=AmplitudeSchedule.ramp(1e-4, 1e-2, 2),
    )

    assert all(c["names"] == ["S0001", "S0002"] for c in crs.calls)


@pytest.mark.asyncio
async def test_supplied_names_are_used_and_recorded():
    crs = FakeCRS()

    result = await drive(
        crs,
        center_frequencies=[1.0e9, 1.1e9],
        names=["low", "high"],
        module=2,
        amp_schedule=AmplitudeSchedule.explicit([1e-3]),
    )

    assert crs.calls[0]["names"] == ["low", "high"]
    assert crs.calls[0]["amp"] == pytest.approx({"low": 1e-3, "high": 1e-3})
    assert result["call_params"]["names"] == ["low", "high"]


@pytest.mark.asyncio
async def test_a_frequency_list_needs_a_schedule_that_carries_amplitudes():
    """There is no bias amplitude to scale, so the schedule has to supply one."""
    crs = FakeCRS()

    with pytest.raises(ValueError, match="required when scheduling by name"):
        await drive(crs, center_frequencies=[1.0e9], module=2)

    assert crs.calls == []


# ─── refusals, all before the first sweep ─────────────────────────────────────


@pytest.mark.asyncio
async def test_an_overshooting_ladder_is_refused_before_anything_is_measured():
    crs = FakeCRS()
    catalog = a_catalog(amplitudes=(0.001, 0.5))

    with pytest.raises(ValueError, match="above full scale"):
        await drive(
            crs,
            catalog,
            amp_schedule=AmplitudeSchedule.multiplicative(1.0, 4.0, 3, spacing="linear"),
        )

    assert crs.calls == []


@pytest.mark.asyncio
async def test_neither_input_is_an_error():
    with pytest.raises(ValueError, match="exactly one of the two"):
        await drive(FakeCRS())


@pytest.mark.asyncio
async def test_both_inputs_is_an_error():
    with pytest.raises(ValueError, match="exactly one of the two"):
        await drive(FakeCRS(), a_catalog(), center_frequencies=[1.0e9])


@pytest.mark.asyncio
async def test_names_alongside_a_catalog_is_an_error():
    with pytest.raises(ValueError, match="applies to center_frequencies only"):
        await drive(FakeCRS(), a_catalog(), names=["a", "b", "c"])


@pytest.mark.asyncio
async def test_a_module_list_is_refused():
    with pytest.raises(ValueError, match="one module per call"):
        await drive(
            FakeCRS(), center_frequencies=[1.0e9], module=[1, 2],
            amp_schedule=AmplitudeSchedule.explicit([1e-3]),
        )


@pytest.mark.asyncio
async def test_a_module_that_disagrees_with_the_catalog_is_an_error():
    with pytest.raises(ValueError, match="does not match the catalog's module"):
        await drive(FakeCRS(), a_catalog(), module=3)


@pytest.mark.asyncio
async def test_a_frequency_list_needs_a_module():
    with pytest.raises(ValueError, match="module is required"):
        await drive(
            FakeCRS(), center_frequencies=[1.0e9],
            amp_schedule=AmplitudeSchedule.explicit([1e-3]),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "directions, match",
    [
        ((), "nothing would be measured"),
        (("sideways",), "Unknown sweep direction"),
        (("upward", "upward"), "repeats"),
    ],
)
async def test_a_bad_direction_axis_is_an_error(directions, match):
    with pytest.raises(ValueError, match=match):
        await drive(FakeCRS(), a_catalog(), directions=directions)


@pytest.mark.asyncio
async def test_a_bare_string_direction_is_refused_rather_than_split():
    with pytest.raises(TypeError, match="not a single string"):
        await drive(FakeCRS(), a_catalog(), directions="upward")


@pytest.mark.asyncio
async def test_a_bare_number_is_not_a_schedule():
    with pytest.raises(TypeError, match="must be an AmplitudeSchedule"):
        await drive(FakeCRS(), a_catalog(), amp_schedule=0.005)


# ─── the packed result ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_results_are_keyed_by_step_then_direction_and_nothing_else():
    crs = FakeCRS()

    result = await drive(
        crs,
        a_catalog(),
        amp_schedule=AmplitudeSchedule.ramp(1e-3, 1e-2, 3),
        directions=("upward", "downward"),
    )

    assert list(result["results"]) == [0, 1, 2]  # integer keys, in order
    for step in result["results"].values():
        assert set(step) == {"upward", "downward"}


@pytest.mark.asyncio
async def test_a_step_swept_once_holds_only_that_direction():
    crs = FakeCRS()
    result = await drive(crs, a_catalog(), directions=("downward",))

    assert set(result["results"][0]) == {"downward"}


@pytest.mark.asyncio
async def test_under_a_direction_is_exactly_what_multisweep_returned():
    sentinel = {"R0001": {"iq_counts": "opaque"}}
    crs = FakeCRS(sweep_result=sentinel)

    result = await drive(crs, a_catalog())

    # Equal, and entry-for-entry the same objects: the driver files what the
    # sweep measured, and does not touch it on the way past.
    assert result["results"][0]["upward"] == sentinel
    assert result["results"][0]["upward"]["R0001"] is sentinel["R0001"]


@pytest.mark.asyncio
async def test_the_amplitude_of_a_step_is_recoverable_without_being_stored_twice():
    """Per-resonator from the sweep, the rung from the schedule."""
    crs = FakeCRS()
    catalog = a_catalog(amplitudes=(0.001, 0.002, 0.004))

    result = await drive(
        crs, catalog, amp_schedule=AmplitudeSchedule.multiplicative(1.0, 4.0, 3)
    )

    step = result["results"][2]["upward"]
    assert step["R0001"]["sweep_amplitude"] == pytest.approx(0.004)
    assert result["call_params"]["amp_schedule"]["ladder"][2] == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_call_params_records_the_arguments_as_given():
    crs = FakeCRS()
    schedule = AmplitudeSchedule.ramp(1e-3, 1e-2, 3)
    catalog = a_catalog()

    result = await drive(
        crs, catalog, amp_schedule=schedule, nsamps=7,
        directions=("upward", "downward"),
    )

    params = result["call_params"]
    assert params["amp_schedule"] == schedule.to_dict()
    assert params["catalog"] == catalog.to_dict()
    assert params["span_hz"] == 200e3
    assert params["npoints_per_sweep"] == 101
    assert params["nsamps"] == 7
    assert params["directions"] == ["upward", "downward"]
    # As given, not as resolved: nothing was passed, so nothing is claimed.
    assert params["module"] is None
    assert params["center_frequencies"] is None
    assert params["names"] is None


@pytest.mark.asyncio
async def test_center_frequencies_are_recorded_only_as_passed():
    """A future step may re-centre between amplitudes; each sweep's own
    original_center_frequency is then the truth, and a top-level copy a lie."""
    crs = FakeCRS()

    result = await drive(
        crs, center_frequencies=[1.0e9, 1.1e9], module=2,
        amp_schedule=AmplitudeSchedule.explicit([1e-3]),
    )

    assert result["call_params"]["center_frequencies"] == [1.0e9, 1.1e9]
    assert "center_frequencies" not in result
    assert "span_hz" not in result


@pytest.mark.asyncio
async def test_the_result_carries_a_schema_version():
    result = await drive(FakeCRS(), a_catalog())
    # A literal, not the constant: bumping the version should mean editing a
    # test, because it is a claim that readers of older files need to know.
    assert result["schema_version"] == 3


@pytest.mark.asyncio
async def test_the_result_is_keyed_by_the_module_it_was_swept_on():
    container = await drive_macro(
        FakeCRS(), a_catalog(), span_hz=200e3, npoints_per_sweep=101
    )

    assert list(container) == ["crs0030_rmod2"]
    assert container["crs0030_rmod2"]["module"] == 2


# ─── callbacks ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sweep_callback_fires_once_per_sweep_in_measurement_order():
    crs = FakeCRS()
    seen = []

    await drive(
        crs,
        a_catalog(),
        amp_schedule=AmplitudeSchedule.ramp(1e-3, 1e-2, 2),
        directions=("upward", "downward"),
        sweep_callback=seen.append,
    )

    assert [(r["step"], r["direction"]) for r in seen] == [
        (0, "upward"), (0, "downward"), (1, "upward"), (1, "downward"),
    ]
    assert [r["completed"] for r in seen] == [1, 2, 3, 4]
    assert {r["total"] for r in seen} == {4}


@pytest.mark.asyncio
async def test_sweep_callback_carries_the_step_amplitudes_and_its_rung():
    crs = FakeCRS()
    seen = []

    await drive(
        crs,
        a_catalog(amplitudes=(0.001,)),
        amp_schedule=AmplitudeSchedule.multiplicative(2.0, 2.0, 1),
        sweep_callback=seen.append,
    )

    assert seen[0]["amplitudes"] == pytest.approx({"R0001": 0.002})
    assert seen[0]["factor"] == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_an_absolute_step_reports_no_rung():
    crs = FakeCRS()
    seen = []

    await drive(
        crs, a_catalog(), amp_schedule=AmplitudeSchedule.explicit([1e-3]),
        sweep_callback=seen.append,
    )

    assert seen[0]["factor"] is None


@pytest.mark.asyncio
async def test_sweep_callback_hands_over_every_sweep_that_finished_before_a_failure():
    """Which is why the driver does not need to return partial results."""
    catalog = a_catalog()
    seen = []

    def fail_on_the_third(catalog_, kwargs):
        if len(seen) == 2:
            raise RuntimeError("the board fell over")
        return {r.name: {"name": r.name} for r in catalog_}

    crs = FakeCRS(sweep_result=fail_on_the_third)

    with pytest.raises(RuntimeError, match="fell over"):
        await drive(
            crs,
            catalog,
            amp_schedule=AmplitudeSchedule.ramp(1e-3, 1e-2, 4),
            sweep_callback=seen.append,
        )

    assert [r["step"] for r in seen] == [0, 1]


@pytest.mark.asyncio
async def test_progress_callback_is_forwarded_untouched():
    """It means progress *within* a sweep, and keeps meaning that."""
    crs = FakeCRS()

    def progress(module, pct):
        pass

    await drive(
        crs, a_catalog(), amp_schedule=AmplitudeSchedule.ramp(1e-3, 1e-2, 2),
        progress_callback=progress,
    )

    assert all(c["progress_callback"] is progress for c in crs.calls)


@pytest.mark.asyncio
async def test_data_callback_is_widened_with_the_coordinates_of_its_sweep():
    """multisweep emits (module, partial); a consumer inside a ladder needs to
    know which sweep that was."""
    seen = []

    def sweep_and_emit_partial_data(catalog_, kwargs):
        kwargs["data_callback"](2, {"partial": True})
        return {r.name: {"name": r.name} for r in catalog_}

    crs = FakeCRS(sweep_result=sweep_and_emit_partial_data)

    await drive(
        crs,
        a_catalog(),
        amp_schedule=AmplitudeSchedule.ramp(1e-3, 1e-2, 2),
        directions=("upward", "downward"),
        data_callback=lambda *args: seen.append(args),
    )

    assert seen == [
        (2, {"partial": True}, 0, "upward"),
        (2, {"partial": True}, 0, "downward"),
        (2, {"partial": True}, 1, "upward"),
        (2, {"partial": True}, 1, "downward"),
    ]


@pytest.mark.asyncio
async def test_no_data_callback_means_none_is_forwarded():
    crs = FakeCRS()
    await drive(crs, a_catalog())

    assert crs.calls[0]["data_callback"] is None


# ─── the driver's output is what the readers expect ───────────────────────────


@pytest.mark.asyncio
async def test_the_readers_work_on_what_the_driver_actually_returns():
    """The tuning-side tests build a packed dict by hand; this one checks the
    hand-built shape has not drifted from the real thing."""
    from rfmux.tuning import (
        collect_amplitude_iterations_for,
        find_iteration_matching_amplitude,
        get_amplitudes_at_iteration,
    )

    crs = FakeCRS()
    catalog = a_catalog(amplitudes=(0.001, 0.002, 0.004))

    result = await drive(
        crs,
        catalog,
        amp_schedule=AmplitudeSchedule.multiplicative(0.25, 4.0, 5),
        directions=("upward", "downward"),
    )

    collected = collect_amplitude_iterations_for(result, "R0002")
    assert list(collected) == [0, 1, 2, 3, 4]
    assert set(collected[0]) == {"upward", "downward"}
    assert [
        c["upward"]["sweep_amplitude"] for c in collected.values()
    ] == pytest.approx([0.0005, 0.001, 0.002, 0.004, 0.008])

    assert get_amplitudes_at_iteration(result, 2) == pytest.approx(
        {"R0001": 0.001, "R0002": 0.002, "R0003": 0.004}
    )

    # ×1 sits in the middle of the ladder, so that is where each resonator is
    # at its own bias amplitude — and the sweep taken there comes back with it.
    at_bias = find_iteration_matching_amplitude(result, "R0002")
    assert list(at_bias) == [2]
    assert at_bias[2]["upward"]["sweep_amplitude"] == pytest.approx(0.002)

    assert list(find_iteration_matching_amplitude(result, "R0002", 0.008)) == [4]


@pytest.mark.asyncio
async def test_the_readers_work_on_a_frequency_list_result_too():
    from rfmux.tuning import (
        collect_amplitude_iterations_for,
        find_iteration_matching_amplitude,
    )

    crs = FakeCRS()
    result = await drive(
        crs,
        center_frequencies=[1.0e9, 1.1e9],
        module=2,
        amp_schedule=AmplitudeSchedule.ramp(1e-4, 1e-2, 3),
    )

    assert list(collect_amplitude_iterations_for(result, "S0002")) == [0, 1, 2]
    assert list(find_iteration_matching_amplitude(result, "S0002", 1e-2)) == [2]

    # No catalog, so no bias amplitude to fall back on.
    with pytest.raises(ValueError, match="no catalog to take one from"):
        find_iteration_matching_amplitude(result, "S0002")
