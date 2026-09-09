"""How multisweep decides what to sweep, what to call it, and at what amplitude.

These cover the resolution step only — the part that turns a catalog, or a bare
list of frequencies, into one normalized list of sweep targets and the amplitude
every pass probes them at. The loop over amplitude steps is in
``test_multisweep_schedule.py``; the measurement loop below both needs a board and
is not exercised here.

The amplitude tests go through ``_resolve_schedule``, which is the path the
macro actually takes: a bare ``amp`` becomes a one-step schedule, and the
complaints on the way have to name ``amp`` rather than the ``base`` that same
value would be called had the caller built the schedule themselves.
"""

import numpy as np
import pytest

from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog
from rfmux.tuning import AmplitudeSchedule
from rfmux.algorithms.measurement.multisweep import (
    _resolve_schedule,
    _resolve_section_names,
    _resolve_sweep_targets,
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


def for_catalog(catalog, amp):
    """The amplitudes one pass over *catalog* would be swept at."""
    resonators = catalog.resonators(order="frequency")
    schedule = _resolve_schedule(
        amp,
        [r.name for r in resonators],
        {r.name: r.bias.amplitude for r in resonators},
    )
    return schedule.resolve_steps(catalog)[0].amplitudes


def for_sections(names, amp):
    """The same, for a bare frequency list, which has nothing to fall back to."""
    schedule = _resolve_schedule(amp, names, None)
    return schedule.resolve_steps(names)[0].amplitudes


# ─── amplitude resolution, with a catalog ─────────────────────────────────────


def test_none_uses_each_resonators_own_bias_amplitude():
    catalog = a_catalog()
    assert for_catalog(catalog, None) == {
        "R0001": 0.001,
        "R0002": 0.002,
        "R0003": 0.004,
    }


def test_a_number_overrides_every_resonator():
    catalog = a_catalog()
    assert for_catalog(catalog, 0.01) == {
        "R0001": 0.01,
        "R0002": 0.01,
        "R0003": 0.01,
    }
    # …and the catalog is untouched by the override.
    assert [r.bias.amplitude for r in catalog] == [0.001, 0.002, 0.004]


def test_a_mapping_overrides_per_resonator():
    catalog = a_catalog()
    amps = {"R0001": 0.5, "R0002": 0.6, "R0003": 0.7}
    assert for_catalog(catalog, amps) == amps


def test_a_partial_mapping_is_an_error_not_a_fallback():
    catalog = a_catalog()
    with pytest.raises(ValueError, match="R0003"):
        for_catalog(catalog, {"R0001": 0.5, "R0002": 0.6})


def test_an_unknown_name_is_an_error():
    catalog = a_catalog()
    with pytest.raises(ValueError, match="R9999"):
        for_catalog(catalog, {"R0001": 0.5, "R0002": 0.6, "R0003": 0.7, "R9999": 0.8})


@pytest.mark.parametrize("sequence", [[0.1, 0.2, 0.3], (0.1, 0.2, 0.3), np.zeros(3)])
def test_a_positional_sequence_is_refused_alongside_a_catalog(sequence):
    """It would depend on catalog ordering, so it is only allowed with a
    caller-supplied frequency list."""
    with pytest.raises(TypeError, match="positional sequence"):
        for_catalog(a_catalog(), sequence)


# ─── amplitude resolution, with a frequency list ──────────────────────────────


def test_a_single_amplitude_applies_to_every_section():
    assert for_sections(["S0001", "S0002"], 0.004) == {"S0001": 0.004, "S0002": 0.004}


@pytest.mark.parametrize(
    "amps", [[0.001, 0.002, 0.003], (0.001, 0.002, 0.003), np.array([1e-3, 2e-3, 3e-3])]
)
def test_one_amplitude_per_section_pairs_off_positionally(amps):
    """The ordering is the caller's own here, unlike a catalog's."""
    assert for_sections(["S0001", "S0002", "S0003"], amps) == pytest.approx(
        {"S0001": 0.001, "S0002": 0.002, "S0003": 0.003}
    )


def test_a_mismatched_amplitude_list_is_an_error():
    with pytest.raises(ValueError, match="2 amplitudes for 3"):
        for_sections(["S0001", "S0002", "S0003"], [0.001, 0.002])


def test_sections_have_no_amplitude_to_fall_back_to():
    with pytest.raises(ValueError, match="amp is required"):
        for_sections(["S0001"], None)


def test_sections_accept_a_mapping_too():
    """Sections are named, so they can be addressed by name like resonators."""
    amps = {"low": 0.001, "high": 0.002}
    assert for_sections(["low", "high"], amps) == amps


# ─── a bare amp and a schedule are the same argument ──────────────────────────


def test_a_bare_amp_becomes_a_one_step_schedule():
    """Which is what one amplitude is — so the loop above has one thing to walk
    whichever way the caller spelled it."""
    schedule = _resolve_schedule(0.005, ["S0001"], None)

    assert isinstance(schedule, AmplitudeSchedule)
    assert schedule.nsteps == 1
    # Kept verbatim, so call_params still records the request.
    assert schedule.base == 0.005


def test_a_schedule_is_passed_through_untouched():
    schedule = AmplitudeSchedule.ramp(1e-3, 1e-2, 4)
    assert _resolve_schedule(schedule, ["S0001"], None) is schedule


def test_a_bare_amp_is_complained_about_in_amps_own_words():
    """A caller who wrote amp= should not be told to fix base=, even though the
    two resolve through one function."""
    with pytest.raises(ValueError, match=r"^amp is missing an amplitude"):
        for_catalog(a_catalog(), {"R0001": 0.5})


def test_a_schedules_base_is_complained_about_in_the_schedules_words():
    """The other half of the same rule: this caller really does have to fix
    base=."""
    with pytest.raises(ValueError, match=r"^base is missing an amplitude"):
        _resolve_schedule(
            AmplitudeSchedule.multiplicative(1.0, 2.0, 2, base={"R0001": 0.5}),
            ["R0001", "R0002"],
            None,
        ).resolve_steps(["R0001", "R0002"])


# ─── section naming ───────────────────────────────────────────────────────────


def test_sections_are_named_s0001_upwards_by_default():
    assert _resolve_section_names([1.0e9, 1.1e9, 1.2e9], None) == [
        "S0001",
        "S0002",
        "S0003",
    ]


def test_section_names_do_not_collide_with_catalog_names():
    """S0001, not R0001 — a result dict says which form produced it."""
    auto = _resolve_section_names([1.0e9] * 3, None)
    assert not set(auto) & {r.name for r in a_catalog()}


def test_supplied_names_are_used_in_the_order_given():
    names = ["upper", "lower", "middle"]
    assert _resolve_section_names([1.0e9, 1.1e9, 1.2e9], names) == names


def test_the_wrong_number_of_names_is_an_error():
    with pytest.raises(ValueError, match="2 names for 3"):
        _resolve_section_names([1.0e9, 1.1e9, 1.2e9], ["a", "b"])


def test_duplicate_names_are_an_error():
    with pytest.raises(ValueError, match="Duplicate section names"):
        _resolve_section_names([1.0e9, 1.1e9], ["same", "same"])


def test_names_must_be_strings():
    with pytest.raises(TypeError, match="must be strings"):
        _resolve_section_names([1.0e9, 1.1e9], ["fine", 2])


# ─── target resolution, with a catalog ────────────────────────────────────────


def test_targets_come_from_the_catalog_in_channel_order():
    catalog = a_catalog()
    targets = _resolve_sweep_targets(catalog, None, None)

    assert [t.name for t in targets] == ["R0001", "R0002", "R0003"]
    assert [t.channel for t in targets] == [1, 2, 3]


def test_a_target_says_what_is_swept_and_not_how_loud():
    """A target is resolved once and swept at every step of the schedule, so an
    amplitude on it could only be one step's."""
    (target,) = _resolve_sweep_targets(a_catalog(amplitudes=(0.001,)), None, None)

    assert not hasattr(target, "amplitude")


def test_sweep_centres_are_the_catalogs_bias_frequencies():
    catalog = a_catalog()
    targets = _resolve_sweep_targets(catalog, None, None)
    assert [t.center_frequency_hz for t in targets] == [
        r.bias.frequency_hz for r in catalog
    ]


def test_the_catalogs_channels_are_used_not_a_fresh_1_to_n():
    """A resonator's channel is a permanent binding; multisweep honours it."""
    catalog = ResonatorCatalog(
        [
            Resonator(
                name="R0001",
                channel=7,
                bias=BiasPoint(frequency_hz=1.0e9, amplitude=0.001),
            ),
            Resonator(
                name="R0002",
                channel=3,
                bias=BiasPoint(frequency_hz=1.1e9, amplitude=0.001),
            ),
        ],
        module=2,
    )
    targets = _resolve_sweep_targets(catalog, None, None)
    # Bias-frequency order, so R0001 leads despite sitting on the higher
    # channel, and each target keeps the channel its resonator was bound to.
    assert [(t.name, t.channel) for t in targets] == [("R0001", 7), ("R0002", 3)]


def test_names_alongside_a_catalog_is_an_error():
    """A catalog's resonators are already named; renaming belongs to it."""
    with pytest.raises(ValueError, match="names applies to center_frequencies"):
        _resolve_sweep_targets(a_catalog(), None, ["a", "b", "c"])


def test_neither_input_is_an_error():
    with pytest.raises(ValueError, match="exactly one"):
        _resolve_sweep_targets(None, None, None)


def test_both_inputs_is_an_error():
    with pytest.raises(ValueError, match="exactly one"):
        _resolve_sweep_targets(a_catalog(), [1.0e9], None)


# ─── target resolution, with a frequency list ─────────────────────────────────


def test_a_frequency_list_is_named_by_section_and_channelled_by_position():
    targets = _resolve_sweep_targets(None, [1.0e9, 1.1e9], None)

    assert [t.name for t in targets] == ["S0001", "S0002"]
    assert [t.channel for t in targets] == [1, 2]
    assert [t.center_frequency_hz for t in targets] == [1.0e9, 1.1e9]


def test_a_frequency_list_keeps_the_order_it_was_given():
    """Unlike a catalog, this form is not sorted — S0001 is what you passed
    first, whatever its frequency."""
    targets = _resolve_sweep_targets(None, [1.5e9, 1.0e9, 1.2e9], None)
    assert [(t.name, t.center_frequency_hz) for t in targets] == [
        ("S0001", 1.5e9),
        ("S0002", 1.0e9),
        ("S0003", 1.2e9),
    ]


def test_supplied_names_become_the_target_names():
    targets = _resolve_sweep_targets(None, [1.0e9, 1.1e9], ["low", "high"])
    assert [(t.name, t.center_frequency_hz) for t in targets] == [
        ("low", 1.0e9),
        ("high", 1.1e9),
    ]
    # Channels are still positional, independent of what the sections are called.
    assert [t.channel for t in targets] == [1, 2]
