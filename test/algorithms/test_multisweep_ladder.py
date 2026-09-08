"""Behaviour of multisweep's loop over amplitude steps and directions.

The loop's one collaborator is ``_measure_sweep``, the coroutine that owns
everything needing a board, so substituting that exercises all of it — the
order sweeps are taken in, the amplitudes each one is asked for, what is passed
down unchanged, and the shape of the packed result. The measurement itself is
``_measure_sweep``'s and is not re-tested here; input resolution is in
``test_multisweep.py``.

The narrow call — one amplitude, one direction — goes through this same loop, so
it is tested here too rather than being a separate path to trust.
"""

import numpy as np
import pytest

from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog
from rfmux.tuning import AmplitudeSchedule
from rfmux.algorithms.measurement import multisweep as multisweep_module
from rfmux.algorithms.measurement.multisweep import multisweep

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
drive_macro = multisweep.__wrapped__


class FakeReadoutModule:
    """Enough of a ReadoutModule to name itself, which is how the macro gets
    the module identifier its result is keyed by."""

    def __init__(self, module):
        self._module = module

    def index(self):
        return f"crs0030_rmod{self._module}"


class FakeCRS:
    """A CRS with the two things the loop asks of one, and no measurement.

    Everything that needs a board lives in ``_measure_sweep``, which the
    ``sweeps`` fixture replaces; what is left up here is the decimation query
    that bounds the channel count, and ``crs.module[m].index()`` for the key
    the result comes back under.
    """

    def __init__(self, decimation=6):
        self._decimation = decimation
        self.module = {m: FakeReadoutModule(m) for m in range(1, 9)}

    async def get_decimation(self):
        return self._decimation

    async def multisweep(self, **kwargs):
        """The module fan-out re-enters the macro, so a fake has to as well."""
        return await drive_macro(self, **kwargs)


class RecordedSweeps:
    """Every ``_measure_sweep`` the loop made, and what it was asked for."""

    def __init__(self, measure=None):
        self.calls = []
        self._measure = measure

    async def __call__(self, crs, targets, amplitudes, **kwargs):
        self.calls.append({
            "names": [t.name for t in targets],
            "amplitudes": dict(amplitudes),
            **kwargs,
        })
        if self._measure is not None:
            return self._measure(targets, amplitudes, kwargs)
        # Stand in for one sweep's measurement: keyed by name, and carrying the
        # amplitude and direction it was asked for, as the real one does.
        return {
            t.name: {
                "sweep_amplitude": amplitudes[t.name],
                "sweep_direction": kwargs["sweep_direction"],
            }
            for t in targets
        }

    def __len__(self):
        return len(self.calls)


@pytest.fixture
def sweeps(monkeypatch):
    """The loop's collaborator, replaced and recorded."""
    recorded = RecordedSweeps()
    monkeypatch.setattr(multisweep_module, "_measure_sweep", recorded)
    return recorded


def measuring(monkeypatch, measure):
    """As *sweeps*, but with a test's own stand-in for the measurement."""
    recorded = RecordedSweeps(measure)
    monkeypatch.setattr(multisweep_module, "_measure_sweep", recorded)
    return recorded


async def drive(crs, catalog=None, **kwargs):
    """The macro's own defaults, minus the two every call needs.

    Returns the one module's output rather than the container the macro
    returns, because these tests are about the loop. That the result is keyed
    by module at all is checked below, and the container's own behaviour —
    merging, and being refused where one module's output was wanted — lives in
    ``test/tuning/test_sweep_results.py``.
    """
    kwargs.setdefault("span_hz", 200e3)
    kwargs.setdefault("npoints_per_sweep", 101)
    container = await drive_macro(crs, catalog, **kwargs)
    (module_id,) = container
    return container[module_id]


# ─── the loop: step outer, direction inner ────────────────────────────────────


@pytest.mark.asyncio
async def test_every_step_is_swept_in_every_direction(sweeps):
    await drive(
        FakeCRS(),
        a_catalog(),
        amp=AmplitudeSchedule.ramp(1e-3, 1e-2, 3),
        sweep_direction=("upward", "downward"),
    )

    assert len(sweeps) == 6


@pytest.mark.asyncio
async def test_amplitude_is_the_outer_axis_and_direction_the_inner(sweeps):
    """Each step's pair is measured together, so amplitude marches
    monotonically — which is what a bifurcation walk wants."""
    await drive(
        FakeCRS(),
        a_catalog(),
        amp=AmplitudeSchedule.ramp(1e-3, 1e-1, 3),
        sweep_direction=("upward", "downward"),
    )

    assert [
        (c["amplitudes"]["R0001"], c["sweep_direction"]) for c in sweeps.calls
    ] == pytest.approx([
        (1e-3, "upward"), (1e-3, "downward"),
        (1e-2, "upward"), (1e-2, "downward"),
        (1e-1, "upward"), (1e-1, "downward"),
    ])


@pytest.mark.asyncio
async def test_the_order_of_directions_is_the_order_measured(sweeps):
    await drive(FakeCRS(), a_catalog(), sweep_direction=("downward", "upward"))

    assert [c["sweep_direction"] for c in sweeps.calls] == ["downward", "upward"]


@pytest.mark.asyncio
async def test_one_direction_is_the_default(sweeps):
    await drive(FakeCRS(), a_catalog(), amp=AmplitudeSchedule.ramp(1e-3, 1e-2, 4))

    assert len(sweeps) == 4
    assert {c["sweep_direction"] for c in sweeps.calls} == {"upward"}


@pytest.mark.asyncio
async def test_each_sweep_gets_the_amplitudes_of_its_own_step(sweeps):
    await drive(
        FakeCRS(),
        a_catalog(amplitudes=(0.001, 0.002)),
        amp=AmplitudeSchedule.multiplicative(1.0, 2.0, 2),
    )

    assert sweeps.calls[0]["amplitudes"] == pytest.approx({"R0001": 0.001, "R0002": 0.002})
    assert sweeps.calls[1]["amplitudes"] == pytest.approx({"R0001": 0.002, "R0002": 0.004})


# ─── saying nothing about either axis is one sweep ────────────────────────────


@pytest.mark.asyncio
async def test_no_schedule_means_one_sweep_at_the_catalogs_own_amplitudes(sweeps):
    result = await drive(FakeCRS(), a_catalog())

    assert len(sweeps) == 1
    assert sweeps.calls[0]["amplitudes"] == pytest.approx(
        {"R0001": 0.001, "R0002": 0.002, "R0003": 0.004}
    )
    assert list(result["results"]) == [0]
    assert list(result["results"][0]) == ["upward"]


@pytest.mark.asyncio
async def test_a_bare_amplitude_is_a_ladder_of_one_rung(sweeps):
    """Which is what one amplitude is, so it goes through the same loop rather
    than round a second one."""
    result = await drive(FakeCRS(), a_catalog(), amp=0.005)

    assert len(sweeps) == 1
    assert sweeps.calls[0]["amplitudes"] == pytest.approx(
        {"R0001": 0.005, "R0002": 0.005, "R0003": 0.005}
    )
    assert result["call_params"]["amp_schedule"] == AmplitudeSchedule(0.005).to_dict()


@pytest.mark.asyncio
async def test_a_bare_direction_is_one_direction(sweeps):
    result = await drive(FakeCRS(), a_catalog(), sweep_direction="downward")

    assert [c["sweep_direction"] for c in sweeps.calls] == ["downward"]
    assert result["call_params"]["directions"] == ["downward"]


@pytest.mark.asyncio
async def test_a_bare_amplitude_can_still_be_per_resonator(sweeps):
    await drive(FakeCRS(), a_catalog(), amp={"R0001": 0.1, "R0002": 0.2, "R0003": 0.3})

    assert sweeps.calls[0]["amplitudes"] == pytest.approx(
        {"R0001": 0.1, "R0002": 0.2, "R0003": 0.3}
    )


# ─── what is passed down, and what is not ─────────────────────────────────────


@pytest.mark.asyncio
async def test_the_sweep_parameters_reach_the_measurement_unchanged(sweeps):
    await drive(
        FakeCRS(),
        a_catalog(),
        span_hz=123e3,
        npoints_per_sweep=57,
        nsamps=42,
    )

    call = sweeps.calls[0]
    assert call["span_hz"] == 123e3
    assert call["npoints_per_sweep"] == 57
    assert call["nsamps"] == 42


@pytest.mark.asyncio
async def test_the_catalog_is_read_and_never_written(sweeps):
    catalog = a_catalog()
    before = catalog.to_dict()

    await drive(FakeCRS(), catalog, amp=AmplitudeSchedule.multiplicative(0.5, 2.0, 3))

    assert catalog.to_dict() == before


@pytest.mark.asyncio
async def test_the_module_is_taken_from_the_catalog(sweeps):
    result = await drive(FakeCRS(), a_catalog())

    assert result["module"] == 2
    assert sweeps.calls[0]["module"] == 2


@pytest.mark.asyncio
async def test_the_step_number_travels_with_the_sweep(sweeps):
    """So partial data can say which step it belongs to without the
    measurement having to count its own calls."""
    await drive(
        FakeCRS(),
        a_catalog(),
        amp=AmplitudeSchedule.ramp(1e-3, 1e-2, 3),
        sweep_direction=("upward", "downward"),
    )

    assert [c["step"] for c in sweeps.calls] == [0, 0, 1, 1, 2, 2]


# ─── a bare frequency list ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_frequency_list_is_swept_at_a_ladder_of_absolute_amplitudes(sweeps):
    result = await drive(
        FakeCRS(),
        center_frequencies=[1.0e9, 1.1e9],
        module=2,
        amp=AmplitudeSchedule.ramp(1e-4, 1e-2, 3),
    )

    assert len(sweeps) == 3
    assert sweeps.calls[0]["amplitudes"] == pytest.approx({"S0001": 1e-4, "S0002": 1e-4})
    assert set(result["results"][0]["upward"]) == {"S0001", "S0002"}


@pytest.mark.asyncio
async def test_a_frequency_list_is_recorded_as_the_catalog_it_became(sweeps):
    """The resolved form goes into call_params beside the request, the way a
    bare amp is recorded as the one-rung schedule it became. That is what lets
    the analysis downstream take a result and nothing else."""
    result = await drive(
        FakeCRS(),
        center_frequencies=[1.0e9, 1.1e9],
        module=2,
        amp=AmplitudeSchedule.ramp(1e-4, 1e-2, 3),
    )

    generated = ResonatorCatalog.from_dict(result["call_params"]["catalog"])

    assert generated.module == 2
    assert generated.names(order="frequency") == ["S0001", "S0002"]
    assert [r.channel for r in generated] == [1, 2]
    # The centres exactly as passed — unquantized, so they agree with each
    # sweep's own original_center_frequency — and step 0's amplitude, which is
    # the one the first pass used.
    assert [r.bias.frequency_hz for r in generated] == [1.0e9, 1.1e9]
    assert [r.bias.amplitude for r in generated] == pytest.approx([1e-4, 1e-4])
    # The request is still recorded beside it.
    assert result["call_params"]["center_frequencies"] == [1.0e9, 1.1e9]


@pytest.mark.asyncio
async def test_the_generated_catalog_takes_the_names_that_were_supplied(sweeps):
    result = await drive(
        FakeCRS(),
        center_frequencies=[1.0e9, 1.1e9],
        names=["low", "high"],
        module=2,
        amp=AmplitudeSchedule.explicit([1e-3]),
    )

    generated = ResonatorCatalog.from_dict(result["call_params"]["catalog"])
    assert generated.names(order="frequency") == ["low", "high"]


def a_measured_resonance(targets, amplitudes, kwargs):
    """A stand-in for ``_measure_sweep`` that returns traces with data in them.

    The rest of this file substitutes a stub, because the loop does not care
    what a sweep contains. The one test that hands its result to an analysis
    does, and this is the least resonance that analysis can work on: one circle
    per target, centred where the sweep was.
    """
    entries = {}
    for t in targets:
        frequencies = np.linspace(
            t.center_frequency_hz - 1e5, t.center_frequency_hz + 1e5, 101
        )
        if kwargs["sweep_direction"] == "downward":
            frequencies = frequencies[::-1]
        s21 = 1 - 1 / (1 + 2j * (frequencies - t.center_frequency_hz) / 2e4)
        entries[t.name] = {
            "channel": t.channel,
            "frequencies": frequencies,
            "iq_counts": s21 * 1e3,
            "iq_volts": s21 * 1e-6,
            "original_center_frequency": t.center_frequency_hz,
            "sweep_direction": kwargs["sweep_direction"],
            "sweep_amplitude": amplitudes[t.name],
        }
    return entries


@pytest.mark.asyncio
async def test_a_frequency_list_sweep_can_be_biased(monkeypatch):
    """The point of generating one: find_bias_points reads the array out of the
    sweep, so a bare frequency list is biased like anything else without the
    caller having to build a catalog by hand to hand back to it."""
    from rfmux.tuning import find_bias_points

    measuring(monkeypatch, a_measured_resonance)
    result = await drive(
        FakeCRS(),
        center_frequencies=[1.0e9],
        module=2,
        amp=AmplitudeSchedule.ramp(1e-4, 1e-2, 3),
    )

    report = find_bias_points(result, amplitude_method="derivative", save=False)

    assert [f.name for f in report.findings] == ["S0001"]
    assert report.catalog["S0001"].bias.dI_df is not None


@pytest.mark.asyncio
async def test_nothing_to_sweep_still_records_a_catalog_of_nothing(sweeps):
    with pytest.warns(UserWarning, match="Nothing to sweep"):
        result = await drive(
            FakeCRS(), center_frequencies=[], module=2, amp=1e-3
        )

    generated = ResonatorCatalog.from_dict(result["call_params"]["catalog"])
    assert len(generated) == 0
    assert generated.module == 2


@pytest.mark.asyncio
async def test_section_names_are_resolved_once_for_the_whole_call(sweeps):
    """So the schedule's keys and the results' keys are the same strings by
    construction, not by both happening to generate S0001…"""
    await drive(
        FakeCRS(),
        center_frequencies=[1.0e9, 1.1e9],
        module=2,
        amp=AmplitudeSchedule.ramp(1e-4, 1e-2, 2),
    )

    assert all(c["names"] == ["S0001", "S0002"] for c in sweeps.calls)


@pytest.mark.asyncio
async def test_supplied_names_are_used_and_recorded(sweeps):
    result = await drive(
        FakeCRS(),
        center_frequencies=[1.0e9, 1.1e9],
        names=["low", "high"],
        module=2,
        amp=AmplitudeSchedule.explicit([1e-3]),
    )

    assert sweeps.calls[0]["names"] == ["low", "high"]
    assert sweeps.calls[0]["amplitudes"] == pytest.approx({"low": 1e-3, "high": 1e-3})
    assert result["call_params"]["names"] == ["low", "high"]


@pytest.mark.asyncio
async def test_a_frequency_list_needs_an_amplitude(sweeps):
    """There is no bias amplitude to fall back to, and the complaint names the
    argument the caller actually typed."""
    with pytest.raises(ValueError, match="amp is required"):
        await drive(FakeCRS(), center_frequencies=[1.0e9], module=2)

    assert len(sweeps) == 0


@pytest.mark.asyncio
async def test_a_relative_schedule_over_a_frequency_list_needs_its_own_base(sweeps):
    """The schedule spelling gets the schedule's complaint, because base= is
    what that caller has to fix."""
    with pytest.raises(ValueError, match="required when scheduling by name"):
        await drive(
            FakeCRS(), center_frequencies=[1.0e9], module=2,
            amp=AmplitudeSchedule.multiplicative(0.5, 2.0, 3),
        )

    assert len(sweeps) == 0


# ─── several modules run the whole measurement each ───────────────────────────


@pytest.mark.asyncio
async def test_a_module_list_runs_the_whole_ladder_on_each_module(sweeps):
    container = await drive_macro(
        FakeCRS(),
        center_frequencies=[1.0e9],
        module=[1, 2],
        span_hz=200e3,
        npoints_per_sweep=101,
        amp=AmplitudeSchedule.ramp(1e-3, 1e-2, 3),
        sweep_direction=("upward", "downward"),
    )

    assert list(container) == ["crs0030_rmod1", "crs0030_rmod2"]
    # Six sweeps per module, not six split between them.
    assert sorted(c["module"] for c in sweeps.calls) == [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    for module_id in container:
        assert list(container[module_id]["results"]) == [0, 1, 2]


@pytest.mark.asyncio
async def test_a_module_list_records_each_modules_own_call(sweeps):
    container = await drive_macro(
        FakeCRS(), center_frequencies=[1.0e9], module=[1, 2],
        span_hz=200e3, npoints_per_sweep=101, amp=1e-3,
    )

    # As passed for that module's call — the fan-out is not the measurement.
    assert container["crs0030_rmod1"]["call_params"]["module"] == 1
    assert container["crs0030_rmod2"]["module"] == 2


# ─── refusals, all before the first sweep ─────────────────────────────────────


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
async def test_a_module_list_alongside_a_catalog_is_an_error():
    with pytest.raises(ValueError, match="one call per module"):
        await drive(FakeCRS(), a_catalog(), module=[1, 2])


@pytest.mark.asyncio
async def test_a_module_that_disagrees_with_the_catalog_is_an_error():
    with pytest.raises(ValueError, match="does not match the catalog's module"):
        await drive(FakeCRS(), a_catalog(), module=3)


@pytest.mark.asyncio
async def test_a_frequency_list_needs_a_module():
    with pytest.raises(ValueError, match="module is required"):
        await drive(FakeCRS(), center_frequencies=[1.0e9], amp=1e-3)


@pytest.mark.asyncio
async def test_an_overshooting_ladder_is_refused_before_anything_is_measured(sweeps):
    with pytest.raises(ValueError, match="above full scale"):
        await drive(
            FakeCRS(),
            a_catalog(amplitudes=(0.001, 0.5)),
            amp=AmplitudeSchedule.multiplicative(1.0, 4.0, 3, spacing="linear"),
        )

    assert len(sweeps) == 0


@pytest.mark.asyncio
async def test_a_silent_amplitude_is_refused_rather_than_measured(sweeps):
    """Zero amplitude is not a quiet measurement, it is no measurement."""
    with pytest.raises(ValueError, match="at or below zero amplitude"):
        await drive(FakeCRS(), a_catalog(), amp=0.0)

    assert len(sweeps) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "directions, match",
    [
        ((), "nothing would be measured"),
        (("sideways",), "Unknown sweep direction"),
        (("upward", "upward"), "repeats"),
        ("sideways", "Invalid sweep_direction"),
    ],
)
async def test_a_bad_direction_axis_is_an_error(directions, match):
    with pytest.raises(ValueError, match=match):
        await drive(FakeCRS(), a_catalog(), sweep_direction=directions)


@pytest.mark.asyncio
async def test_a_direction_that_is_neither_a_string_nor_a_sequence_is_an_error():
    with pytest.raises(TypeError, match="must be one of"):
        await drive(FakeCRS(), a_catalog(), sweep_direction=2)


# ─── the packed result ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_results_are_keyed_by_step_then_direction_and_nothing_else(sweeps):
    result = await drive(
        FakeCRS(),
        a_catalog(),
        amp=AmplitudeSchedule.ramp(1e-3, 1e-2, 3),
        sweep_direction=("upward", "downward"),
    )

    assert list(result["results"]) == [0, 1, 2]  # integer keys, in order
    for step in result["results"].values():
        assert set(step) == {"upward", "downward"}


@pytest.mark.asyncio
async def test_a_step_swept_once_holds_only_that_direction(sweeps):
    result = await drive(FakeCRS(), a_catalog(), sweep_direction=("downward",))

    assert set(result["results"][0]) == {"downward"}


@pytest.mark.asyncio
async def test_under_a_direction_is_exactly_what_the_measurement_returned(monkeypatch):
    sentinel = {"R0001": {"iq_counts": "opaque"}}
    measuring(monkeypatch, lambda targets, amplitudes, kwargs: sentinel)

    result = await drive(FakeCRS(), a_catalog())

    # Equal, and entry-for-entry the same objects: the loop files what the
    # sweep measured, and does not touch it on the way past.
    assert result["results"][0]["upward"] == sentinel
    assert result["results"][0]["upward"]["R0001"] is sentinel["R0001"]


@pytest.mark.asyncio
async def test_the_amplitude_of_a_step_is_recoverable_without_being_stored_twice(sweeps):
    """Per-resonator from the sweep, the rung from the schedule."""
    result = await drive(
        FakeCRS(),
        a_catalog(amplitudes=(0.001, 0.002, 0.004)),
        amp=AmplitudeSchedule.multiplicative(1.0, 4.0, 3),
    )

    step = result["results"][2]["upward"]
    assert step["R0001"]["sweep_amplitude"] == pytest.approx(0.004)
    assert result["call_params"]["amp_schedule"]["ladder"][2] == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_call_params_records_the_arguments_as_given(sweeps):
    schedule = AmplitudeSchedule.ramp(1e-3, 1e-2, 3)
    catalog = a_catalog()

    result = await drive(
        FakeCRS(), catalog, amp=schedule, nsamps=7,
        sweep_direction=("upward", "downward"),
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
async def test_a_bare_amp_is_recorded_as_given_rather_than_resolved(sweeps):
    """The one-rung schedule keeps the number the caller typed as its base, so
    call_params is still a record of the request. What each resonator was
    actually probed at is sweep_amplitude on its own entry."""
    result = await drive(FakeCRS(), a_catalog(), amp=0.005)

    assert result["call_params"]["amp_schedule"]["base"] == 0.005
    assert result["call_params"]["amp_schedule"]["ladder"] == [1.0]


@pytest.mark.asyncio
async def test_center_frequencies_are_recorded_only_as_passed(sweeps):
    """A future step may re-centre between amplitudes; each sweep's own
    original_center_frequency is then the truth, and a top-level copy a lie."""
    result = await drive(
        FakeCRS(), center_frequencies=[1.0e9, 1.1e9], module=2,
        amp=AmplitudeSchedule.explicit([1e-3]),
    )

    assert result["call_params"]["center_frequencies"] == [1.0e9, 1.1e9]
    assert "center_frequencies" not in result
    assert "span_hz" not in result


@pytest.mark.asyncio
async def test_the_result_carries_a_schema_version(sweeps):
    result = await drive(FakeCRS(), a_catalog())
    # A literal, not the constant: bumping the version should mean editing a
    # test, because it is a claim that readers of older files need to know.
    assert result["schema_version"] == 6


@pytest.mark.asyncio
async def test_the_result_says_which_driver_made_it(sweeps):
    """The shape is shared with netanal; this is what tells a reader which of
    the two is under 'results'. One amplitude or twenty, it is a multisweep."""
    one = await drive(FakeCRS(), a_catalog())
    many = await drive(
        FakeCRS(), a_catalog(), amp=AmplitudeSchedule.ramp(1e-3, 1e-2, 3)
    )

    assert one["measurement"] == "multisweep"
    assert many["measurement"] == "multisweep"


@pytest.mark.asyncio
async def test_the_result_is_keyed_by_the_module_it_was_swept_on(sweeps):
    container = await drive_macro(
        FakeCRS(), a_catalog(), span_hz=200e3, npoints_per_sweep=101
    )

    assert list(container) == ["crs0030_rmod2"]
    assert container["crs0030_rmod2"]["module"] == 2


@pytest.mark.asyncio
async def test_nothing_to_sweep_is_still_a_well_formed_result(sweeps):
    """A bare {} would be indistinguishable from a caller's own empty dict."""
    empty = ResonatorCatalog([], module=2)

    with pytest.warns(UserWarning, match="Nothing to sweep"):
        result = await drive(
            FakeCRS(), empty, amp=AmplitudeSchedule.ramp(1e-3, 1e-2, 2),
            sweep_direction=("upward", "downward"),
        )

    assert len(sweeps) == 0
    # The steps and directions asked for are still what was asked for.
    assert list(result["results"]) == [0, 1]
    assert result["results"][0] == {"upward": {}, "downward": {}}


# ─── callbacks ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sweep_callback_fires_once_per_sweep_in_measurement_order(sweeps):
    seen = []

    await drive(
        FakeCRS(),
        a_catalog(),
        amp=AmplitudeSchedule.ramp(1e-3, 1e-2, 2),
        sweep_direction=("upward", "downward"),
        sweep_callback=seen.append,
    )

    assert [(r["step"], r["direction"]) for r in seen] == [
        (0, "upward"), (0, "downward"), (1, "upward"), (1, "downward"),
    ]
    assert [r["completed"] for r in seen] == [1, 2, 3, 4]
    assert {r["total"] for r in seen} == {4}


@pytest.mark.asyncio
async def test_sweep_callback_fires_for_a_single_sweep_too(sweeps):
    """One sweep of one is still a sweep that finished."""
    seen = []
    await drive(FakeCRS(), a_catalog(), sweep_callback=seen.append)

    assert [(r["step"], r["direction"], r["total"]) for r in seen] == [
        (0, "upward", 1),
    ]


@pytest.mark.asyncio
async def test_sweep_callback_carries_the_step_amplitudes_and_its_rung(sweeps):
    seen = []

    await drive(
        FakeCRS(),
        a_catalog(amplitudes=(0.001,)),
        amp=AmplitudeSchedule.multiplicative(2.0, 2.0, 1),
        sweep_callback=seen.append,
    )

    assert seen[0]["amplitudes"] == pytest.approx({"R0001": 0.002})
    assert seen[0]["factor"] == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_an_absolute_step_reports_no_rung(sweeps):
    seen = []

    await drive(
        FakeCRS(), a_catalog(), amp=AmplitudeSchedule.explicit([1e-3]),
        sweep_callback=seen.append,
    )

    assert seen[0]["factor"] is None


@pytest.mark.asyncio
async def test_sweep_callback_hands_over_every_sweep_that_finished_before_a_failure(
    monkeypatch,
):
    """Which is why the macro does not need to return partial results."""
    seen = []

    def fail_on_the_third(targets, amplitudes, kwargs):
        if len(seen) == 2:
            raise RuntimeError("the board fell over")
        return {t.name: {"sweep_amplitude": amplitudes[t.name]} for t in targets}

    measuring(monkeypatch, fail_on_the_third)

    with pytest.raises(RuntimeError, match="fell over"):
        await drive(
            FakeCRS(),
            a_catalog(),
            amp=AmplitudeSchedule.ramp(1e-3, 1e-2, 4),
            sweep_callback=seen.append,
        )

    assert [r["step"] for r in seen] == [0, 1]


@pytest.mark.asyncio
async def test_progress_runs_across_the_whole_call(monkeypatch):
    """One call, one progress bar: a ladder reaches 100 once, at the end, not
    once per sweep."""
    seen = []

    def halfway_then_done(targets, amplitudes, kwargs):
        kwargs["report_progress"](0.5)
        return {t.name: {} for t in targets}

    measuring(monkeypatch, halfway_then_done)

    await drive(
        FakeCRS(),
        a_catalog(),
        amp=AmplitudeSchedule.ramp(1e-3, 1e-2, 2),
        sweep_direction=("upward", "downward"),
        progress_callback=lambda module, pct: seen.append(pct),
    )

    # Four sweeps, each reporting its own midpoint and then its completion.
    assert seen == pytest.approx([12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5, 100.0])


@pytest.mark.asyncio
async def test_progress_for_a_single_sweep_is_the_whole_of_it(monkeypatch):
    """Which is what it was before there was an amplitude axis to share."""
    seen = []
    measuring(
        monkeypatch,
        lambda targets, amplitudes, kwargs: (
            kwargs["report_progress"](0.25), {t.name: {} for t in targets}
        )[1],
    )

    await drive(
        FakeCRS(), a_catalog(),
        progress_callback=lambda module, pct: seen.append(pct),
    )

    assert seen == pytest.approx([25.0, 100.0])


@pytest.mark.asyncio
async def test_no_progress_callback_means_the_measurement_is_given_none(sweeps):
    await drive(FakeCRS(), a_catalog())

    assert sweeps.calls[0]["report_progress"] is None


@pytest.mark.asyncio
async def test_data_callback_reaches_the_measurement_with_its_step(sweeps):
    """The measurement emits (module, partial, step, direction) itself; what
    the loop owes it is the step number, which only the loop knows."""
    await drive(
        FakeCRS(),
        a_catalog(),
        amp=AmplitudeSchedule.ramp(1e-3, 1e-2, 2),
        sweep_direction=("upward", "downward"),
        data_callback=print,
    )

    assert [(c["step"], c["sweep_direction"]) for c in sweeps.calls] == [
        (0, "upward"), (0, "downward"), (1, "upward"), (1, "downward"),
    ]
    assert all(c["data_callback"] is print for c in sweeps.calls)


@pytest.mark.asyncio
async def test_no_data_callback_means_none_is_passed_down(sweeps):
    await drive(FakeCRS(), a_catalog())

    assert sweeps.calls[0]["data_callback"] is None


# ─── the loop's output is what the readers expect ─────────────────────────────


@pytest.mark.asyncio
async def test_the_readers_work_on_what_the_macro_actually_returns(sweeps):
    """The tuning-side tests build a packed dict by hand; this one checks the
    hand-built shape has not drifted from the real thing."""
    from rfmux.tuning import (
        collect_amplitude_iterations_for,
        find_iteration_matching_amplitude,
        get_amplitudes_at_iteration,
    )

    result = await drive(
        FakeCRS(),
        a_catalog(amplitudes=(0.001, 0.002, 0.004)),
        amp=AmplitudeSchedule.multiplicative(0.25, 4.0, 5),
        sweep_direction=("upward", "downward"),
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
    at_bias, iteration = find_iteration_matching_amplitude(result, "R0002")
    assert iteration == 2
    assert at_bias["upward"]["sweep_amplitude"] == pytest.approx(0.002)

    assert find_iteration_matching_amplitude(result, "R0002", 0.008)[1] == 4


@pytest.mark.asyncio
async def test_the_readers_work_on_a_frequency_list_result_too(sweeps):
    from rfmux.tuning import (
        collect_amplitude_iterations_for,
        find_iteration_matching_amplitude,
    )

    result = await drive(
        FakeCRS(),
        center_frequencies=[1.0e9, 1.1e9],
        module=2,
        amp=AmplitudeSchedule.ramp(1e-4, 1e-2, 3),
    )

    assert list(collect_amplitude_iterations_for(result, "S0002")) == [0, 1, 2]
    assert find_iteration_matching_amplitude(result, "S0002", 1e-2)[1] == 2

    # The catalog multisweep generated from the list is what the fallback
    # reads, so a frequency-list result has a bias amplitude like any other:
    # step 0's, which for an absolute ramp is its first rung.
    assert find_iteration_matching_amplitude(result, "S0002")[1] == 0


@pytest.mark.asyncio
async def test_a_single_sweep_reads_the_same_way_a_ladder_does(sweeps):
    """The property the merge exists to make true: one entry point, one shape,
    and the readers cannot tell how wide the call was."""
    from rfmux.tuning import collect_amplitude_iterations_for

    result = await drive(FakeCRS(), a_catalog())

    collected = collect_amplitude_iterations_for(result, "R0002")
    assert list(collected) == [0]
    assert collected[0]["upward"]["sweep_amplitude"] == pytest.approx(0.002)
