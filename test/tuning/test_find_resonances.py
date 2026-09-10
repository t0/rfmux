"""Behaviour of the resonance finder.

Synthetic sweeps with known resonators, so every assertion is against a truth
the test set up itself. The emphasis is on the contract callers rely on: the
dips come back, close pairs are only merged when asked for, and anything the
finder discards is reported rather than silently dropped.
"""

from dataclasses import replace

import numpy as np
import pytest

from rfmux.core.resonators import ResonatorCatalog
from rfmux.tuning import (
    ResonanceSearch,
    find_resonances,
    find_resonances_in_netanal,
    find_sweeps_with_nearby_resonances,
    netanal_trace,
    record_search,
)
from rfmux.tuning.sweep_results import pack_netanal

pytestmark = pytest.mark.portable


# ─── synthetic sweeps ─────────────────────────────────────────────────────────


TRUTH = (1.02e9, 1.04e9, 1.06e9, 1.08e9)


def lorentzian_dip(frequencies, f0, q, depth):
    """A negative-going Lorentzian, full width at half depth f0 / q."""
    return -depth / (1.0 + (2.0 * (frequencies - f0) / (f0 / q)) ** 2)


def a_sweep(
    resonances=TRUTH,
    qs=(2e4, 3e4, 2.5e4, 4e4),
    depths=None,
    fmin=1.0e9,
    fmax=1.1e9,
    npoints=100_000,  # 1 kHz spacing: enough to resolve a 50 kHz-wide dip
    noise=0.0,
    seed=0,
):
    """A netanal-like trace: flat baseline with Lorentzian dips in |S21|."""
    frequencies = np.linspace(fmin, fmax, npoints)
    if depths is None:
        depths = [0.7] * len(resonances)
    magnitude = np.ones(npoints)
    for fr, q, depth in zip(resonances, qs, depths):
        magnitude += lorentzian_dip(frequencies, fr, q, depth)
    if noise:
        magnitude += np.random.default_rng(seed).normal(0, noise, npoints)
    return frequencies, magnitude


def matched(found_hz, expected_hz, tolerance_hz=50e3):
    """How many expected resonances have a hit within tolerance."""
    return sum(
        any(abs(f - e) < tolerance_hz for f in found_hz) for e in expected_hz
    )


# ─── the search finds what is there ───────────────────────────────────────────


def test_finds_every_planted_resonance():
    truth = TRUTH
    frequencies, magnitude = a_sweep(resonances=truth)

    found = find_resonances(frequencies, magnitude, min_Q=1e4, max_Q=1e6)

    assert len(found) == len(truth)
    assert matched(found.resonance_frequencies_hz, truth) == len(truth)
    assert not found.rejected


def test_survives_noise():
    truth = TRUTH
    frequencies, magnitude = a_sweep(resonances=truth, noise=0.01)

    found = find_resonances(
        frequencies, magnitude, min_dip_depth_db=1.0, min_Q=1e4, max_Q=1e6
    )

    assert matched(found.resonance_frequencies_hz, truth) == len(truth)


def test_complex_and_magnitude_input_agree():
    """|S21| is taken either way, so I/Q and a bare magnitude are one input."""
    frequencies, magnitude = a_sweep()
    iq = magnitude * np.exp(1j * 2 * np.pi * frequencies * 1e-8)  # cable delay

    from_magnitude = find_resonances(frequencies, magnitude)
    from_iq = find_resonances(frequencies, iq)

    assert np.array_equal(
        from_magnitude.resonance_frequencies_hz, from_iq.resonance_frequencies_hz
    )


def test_shallow_dips_need_a_lower_threshold():
    """The depth floor is the knob that decides what counts as a resonance."""
    frequencies, magnitude = a_sweep(resonances=(1.05e9,), qs=(2e4,), depths=(0.05,))

    assert len(find_resonances(frequencies, magnitude, min_dip_depth_db=1.0)) == 0
    assert len(find_resonances(frequencies, magnitude, min_dip_depth_db=0.2)) == 1


def test_depth_is_the_dip_depth_in_db():
    """A dip to half the baseline is 6 dB deep, and that is what is reported."""
    frequencies, magnitude = a_sweep(resonances=(1.05e9,), qs=(2e4,), depths=(0.5,))

    found = find_resonances(frequencies, magnitude)

    assert found.candidates[0].depth_db == pytest.approx(6.02, abs=0.1)


# ─── the collision cut ────────────────────────────────────────────────────────


def a_close_pair(separation_hz=200e3, depths=(0.7, 0.4)):
    return a_sweep(
        resonances=(1.05e9, 1.05e9 + separation_hz), qs=(2e4, 2e4), depths=depths
    )


def test_the_default_cut_leaves_real_resonances_alone():
    """0 Hz catches only the same point found twice, so a close pair survives."""
    frequencies, magnitude = a_close_pair()

    found = find_resonances(frequencies, magnitude, min_Q=1e4, max_Q=1e6)

    assert len(found) == 2
    assert not found.rejected


def test_collided_pair_is_cut_entirely():
    """Both members go, not just the shallower: a tone on either one reads the
    other, so neither is operable."""
    frequencies, magnitude = a_close_pair()

    found = find_resonances(
        frequencies, magnitude, min_Q=1e4, max_Q=1e6, min_separation_hz=500e3
    )

    assert len(found) == 0
    assert len(found.rejected) == 2
    assert all("collided" in c.rejected_because for c in found.rejected)


def test_the_cut_is_local_to_the_collision():
    """A well-separated resonator elsewhere in the band is untouched."""
    frequencies, magnitude = a_sweep(
        resonances=(1.02e9, 1.06e9, 1.06e9 + 200e3),
        qs=(2e4, 2e4, 2e4),
        depths=(0.7, 0.7, 0.5),
    )

    found = find_resonances(
        frequencies, magnitude, min_Q=1e4, max_Q=1e6, min_separation_hz=500e3
    )

    assert len(found) == 1
    assert found.candidates[0].frequency_hz == pytest.approx(1.02e9, abs=20e3)
    assert len(found.rejected) == 2


def test_the_cut_chains_through_a_dense_group():
    """Each of three consecutive 200 kHz gaps violates a 300 kHz cut, so all
    three go — even though the outer two are 400 kHz apart."""
    frequencies, magnitude = a_sweep(
        resonances=(1.05e9, 1.05e9 + 200e3, 1.05e9 + 400e3),
        qs=(2e4, 2e4, 2e4),
        depths=(0.7, 0.7, 0.7),
    )

    found = find_resonances(
        frequencies, magnitude, min_Q=1e4, max_Q=1e6, min_separation_hz=300e3
    )

    assert len(found) == 0
    assert len(found.rejected) == 3


def test_the_cut_can_be_skipped_entirely():
    frequencies, magnitude = a_close_pair()

    found = find_resonances(
        frequencies, magnitude, min_Q=1e4, max_Q=1e6, min_separation_hz=None
    )

    assert len(found) == 2


def test_the_cut_is_independent_of_sweep_sampling():
    """The point of taking Hz rather than samples: the answer is the same on a
    coarse and a fine sweep of the same band."""
    results = []
    for npoints in (100_000, 200_000):
        frequencies, magnitude = a_sweep(
            resonances=(1.05e9, 1.05e9 + 200e3),
            qs=(2e4, 2e4),
            depths=(0.7, 0.4),
            npoints=npoints,
        )
        results.append(
            len(
                find_resonances(
                    frequencies,
                    magnitude,
                    min_Q=1e4,
                    max_Q=1e6,
                    min_separation_hz=500e3,
                )
            )
        )
    assert results == [0, 0]


def test_the_default_cut_removes_an_exact_duplicate():
    """What the 0 Hz default is for. Reached directly, because find_peaks
    returns distinct samples and so cannot produce a duplicate on its own — the
    guard is there for candidate lists that have been merged or concatenated."""
    from rfmux.tuning.find_resonances import ResonanceCandidate, _separation_pass

    twice = [
        ResonanceCandidate(1.05e9, index=i, depth_db=3.0, width_hz=5e4, q_estimate=2e4)
        for i in (100, 200)
    ]

    kept, dropped = _separation_pass(twice, min_separation_hz=0.0)

    assert kept == []
    assert len(dropped) == 2
    assert "0 Hz" in dropped[0].rejected_because


def test_a_negative_separation_raises():
    frequencies, magnitude = a_sweep()
    with pytest.raises(ValueError, match="None skips the pass"):
        find_resonances(frequencies, magnitude, min_separation_hz=-1.0)


# ─── expected count ───────────────────────────────────────────────────────────


def test_expected_count_keeps_the_deepest_and_records_the_rest():
    frequencies, magnitude = a_sweep(
        resonances=(1.02e9, 1.05e9, 1.08e9),
        qs=(2e4, 2e4, 2e4),
        depths=(0.7, 0.6, 0.1),
    )

    with pytest.warns(UserWarning, match="kept the 2 deepest"):
        found = find_resonances(
            frequencies, magnitude, min_dip_depth_db=0.2, expected_resonances=2
        )

    assert len(found) == 2
    assert matched(found.resonance_frequencies_hz, (1.02e9, 1.05e9)) == 2
    assert len(found.rejected) == 1
    assert "deepest" in found.rejected[0].rejected_because


def test_too_few_resonances_warns():
    frequencies, magnitude = a_sweep(resonances=(1.05e9,), qs=(2e4,))

    with pytest.warns(UserWarning, match="expected 5"):
        found = find_resonances(frequencies, magnitude, expected_resonances=5)

    assert len(found) == 1


# ─── bad input is refused, not absorbed ───────────────────────────────────────


def test_unsorted_frequencies_raise():
    frequencies, magnitude = a_sweep()
    with pytest.raises(ValueError, match="strictly increasing"):
        find_resonances(frequencies[::-1], magnitude)


def test_mismatched_lengths_raise():
    frequencies, magnitude = a_sweep()
    with pytest.raises(ValueError, match="same points"):
        find_resonances(frequencies, magnitude[:-1])


def test_inverted_q_limits_raise():
    frequencies, magnitude = a_sweep()
    with pytest.raises(ValueError, match="must be below"):
        find_resonances(frequencies, magnitude, min_Q=1e7, max_Q=1e4)


def test_all_zero_response_raises():
    frequencies, _ = a_sweep()
    with pytest.raises(ValueError, match="zero everywhere"):
        find_resonances(frequencies, np.zeros(len(frequencies)))


# ─── the collision cut, run again on multisweep data ──────────────────────────


def a_section(dips=(1.0e9,), span_hz=60e3, npoints=201, q=2e5, depth=0.7):
    """One multisweep section: a fine sweep about 1 GHz holding *dips*."""
    frequencies = np.linspace(1e9 - span_hz / 2, 1e9 + span_hz / 2, npoints)
    magnitude = np.ones(npoints)
    for f0 in dips:
        magnitude += lorentzian_dip(frequencies, f0, q, depth)
    return {
        "frequencies": frequencies,
        "iq_counts": magnitude.astype(complex) * 1e6,
        "sweep_amplitude": 1e-3,
    }


def a_multisweep(sections, iteration=0, direction="upward"):
    """One module's sweep result, holding the given ``{name: section}``."""
    return {"call_params": {}, "results": {iteration: {direction: sections}}}


PAIR_8KHZ = (1e9 - 4e3, 1e9 + 4e3)


def test_a_section_with_one_resonance_is_kept():
    sweeps = a_multisweep({"R0001": a_section()})

    assert find_sweeps_with_nearby_resonances(sweeps, 20e3) == []


def test_a_section_holding_two_resonances_is_culled():
    sweeps = a_multisweep({"R0001": a_section(), "R0002": a_section(PAIR_8KHZ)})

    assert find_sweeps_with_nearby_resonances(sweeps, 20e3) == ["R0002"]


def test_a_pair_wider_than_the_cut_is_left_alone():
    sweeps = a_multisweep({"R0002": a_section(PAIR_8KHZ)})

    assert find_sweeps_with_nearby_resonances(sweeps, 5e3) == []


def test_the_cut_is_inclusive():
    """A pair exactly min_separation_hz apart goes, as in _separation_pass.

    The dips sit on sweep samples and are narrow enough not to drag each other's
    minima inward, so the separation the finder measures is the 7800 Hz planted.
    """
    sweeps = a_multisweep({"R0002": a_section((1e9 - 3900, 1e9 + 3900), q=1e6)})

    assert find_sweeps_with_nearby_resonances(sweeps, 7800) == ["R0002"]
    assert find_sweeps_with_nearby_resonances(sweeps, 7799) == []


def test_an_infinite_cut_asks_only_whether_a_second_dip_exists():
    sweeps = a_multisweep({"R0002": a_section((1e9 - 25e3, 1e9 + 25e3))})

    assert find_sweeps_with_nearby_resonances(sweeps, 20e3) == []
    assert find_sweeps_with_nearby_resonances(sweeps, float("inf")) == ["R0002"]


def test_a_downward_sweep_reads_the_same_as_an_upward_one():
    """Descending frequencies are the same trace, so they give the same answer."""
    section = a_section(PAIR_8KHZ)
    reversed_section = {
        "frequencies": section["frequencies"][::-1],
        "iq_counts": section["iq_counts"][::-1],
    }
    sweeps = a_multisweep({"R0002": reversed_section}, direction="downward")

    assert find_sweeps_with_nearby_resonances(sweeps, 20e3) == ["R0002"]


def test_one_collided_amplitude_is_enough_to_cull():
    """The dips only separate at some amplitudes; any one of them condemns."""
    sweeps = {
        "call_params": {},
        "results": {
            0: {"upward": {"R0002": a_section()}},
            1: {"upward": {"R0002": a_section(PAIR_8KHZ)}},
        },
    }

    assert find_sweeps_with_nearby_resonances(sweeps, 20e3) == ["R0002"]
    assert find_sweeps_with_nearby_resonances(sweeps, 20e3, iteration=0) == []
    assert find_sweeps_with_nearby_resonances(sweeps, 20e3, iteration=1) == ["R0002"]


def test_a_name_is_culled_once_however_many_sweeps_show_it():
    sweeps = {
        "call_params": {},
        "results": {
            i: {d: {"R0002": a_section(PAIR_8KHZ)} for d in ("upward", "downward")}
            for i in (0, 1)
        },
    }

    assert find_sweeps_with_nearby_resonances(sweeps, 20e3) == ["R0002"]


def test_names_come_back_in_the_order_they_were_swept():
    sweeps = a_multisweep(
        {name: a_section(PAIR_8KHZ) for name in ("R0003", "R0001", "R0002")}
    )

    assert find_sweeps_with_nearby_resonances(sweeps, 20e3) == [
        "R0003",
        "R0001",
        "R0002",
    ]


def test_a_dead_channel_is_not_a_collision():
    """All zeros is no evidence, and must not take the whole scan down with it."""
    section = a_section()
    section["iq_counts"] = np.zeros(len(section["frequencies"]), dtype=complex)
    sweeps = a_multisweep({"R0001": section, "R0002": a_section(PAIR_8KHZ)})

    assert find_sweeps_with_nearby_resonances(sweeps, 20e3) == ["R0002"]


def test_dips_closer_than_the_finder_resolves_are_not_seen():
    """min_dip_spacing_hz floors what min_separation_hz can act on."""
    sweeps = a_multisweep({"R0002": a_section(PAIR_8KHZ)})

    assert find_sweeps_with_nearby_resonances(sweeps, 20e3, min_dip_spacing_hz=1e3) == [
        "R0002"
    ]
    assert (
        find_sweeps_with_nearby_resonances(sweeps, 20e3, min_dip_spacing_hz=30e3) == []
    )


def test_a_selection_that_matches_no_sweep_raises():
    """A typo'd direction must not read as an array with nothing to cull."""
    sweeps = a_multisweep({"R0002": a_section(PAIR_8KHZ)})

    with pytest.raises(ValueError, match="selected no sweeps"):
        find_sweeps_with_nearby_resonances(sweeps, 20e3, direction="up")
    with pytest.raises(ValueError, match="selected no sweeps"):
        find_sweeps_with_nearby_resonances(sweeps, 20e3, iteration=7)


def test_a_negative_separation_raises_here_too():
    with pytest.raises(ValueError, match="min_separation_hz"):
        find_sweeps_with_nearby_resonances(a_multisweep({"R0001": a_section()}), -1)


def test_the_whole_container_is_refused():
    sweeps = a_multisweep({"R0001": a_section()})

    with pytest.raises(TypeError, match="crs0000_rmod1"):
        find_sweeps_with_nearby_resonances({"crs0000_rmod1": sweeps}, 20e3)


def test_a_missing_iq_key_says_what_the_sweep_holds():
    sweeps = a_multisweep({"R0001": a_section()})

    with pytest.raises(KeyError, match="iq_volts"):
        find_sweeps_with_nearby_resonances(sweeps, 20e3, iq_key="iq_volts")


# ─── netanal wrapper ──────────────────────────────────────────────────────────


def a_module_netanal(module=1, sweep_direction="upward", **kwargs):
    """One module's netanal output, packed the way take_netanal packs it."""
    frequencies, magnitude = a_sweep(**kwargs)
    iq = magnitude.astype(complex)
    fmin, fmax = frequencies[0], frequencies[-1]
    if sweep_direction == "downward":
        frequencies, iq = frequencies[::-1], iq[::-1]
    return pack_netanal(
        {
            "frequencies": frequencies,
            "iq_counts": iq,
            "iq_volts": iq * 1e-7,
            "sweep_amplitude": 0.001,
            "sweep_direction": sweep_direction,
        },
        module_id=f"crs0000_rmod{module}",
        module=module,
        amp=0.001,
        fmin=fmin,
        fmax=fmax,
        npoints=len(frequencies),
        nsamps=10,
        max_chans=1023,
        max_span=500e6,
        rotate_phase_to_0=True,
        sweep_direction=sweep_direction,
        requested_module=module,
    )[f"crs0000_rmod{module}"]


def a_netanal(module=1, **kwargs):
    """The container take_netanal returns, for one module."""
    return {f"crs0000_rmod{module}": a_module_netanal(module, **kwargs)}


def test_wrapper_unpacks_one_modules_netanal():
    truth = TRUTH
    found = find_resonances_in_netanal(
        a_module_netanal(resonances=truth), min_Q=1e4, max_Q=1e6
    )

    assert matched(found.resonance_frequencies_hz, truth) == len(truth)


def test_wrapper_matches_calling_the_search_directly():
    module_netanal = a_module_netanal()
    trace = netanal_trace(module_netanal)
    direct = find_resonances(trace["frequencies"], trace["iq_counts"])

    assert np.array_equal(
        find_resonances_in_netanal(module_netanal).resonance_frequencies_hz,
        direct.resonance_frequencies_hz,
    )


def test_wrapper_searches_a_downward_netanal():
    """Its trace is descending — the order it was measured in — and the finder
    reads a trace in ascending order, so the wrapper flips it."""
    truth = TRUTH
    found = find_resonances_in_netanal(
        a_module_netanal(resonances=truth, sweep_direction="downward"),
        min_Q=1e4,
        max_Q=1e6,
    )

    assert matched(found.resonance_frequencies_hz, truth) == len(truth)


def test_a_downward_search_carries_the_grid_it_searched():
    """candidate.index indexes search.frequencies_hz, which is ascending — not
    the descending array sitting beside it in the netanal."""
    module_netanal = a_module_netanal(sweep_direction="downward")
    measured = netanal_trace(module_netanal)["frequencies"]

    found = find_resonances_in_netanal(module_netanal)

    assert np.array_equal(found.frequencies_hz, measured[::-1])
    for candidate in found.candidates:
        assert found.frequencies_hz[candidate.index] == pytest.approx(
            candidate.frequency_hz
        )


def test_a_downward_netanal_finds_what_the_upward_one_does():
    """The flip is the only difference the direction makes to the search."""
    truth = TRUTH
    up = find_resonances_in_netanal(a_module_netanal(resonances=truth))
    down = find_resonances_in_netanal(
        a_module_netanal(resonances=truth, sweep_direction="downward")
    )

    assert down.resonance_frequencies_hz == pytest.approx(
        up.resonance_frequencies_hz
    )


def test_wrapper_refuses_the_whole_netanal_and_names_the_modules():
    """One module at a time, as the fitters and the bias finder take one."""
    netanal = a_netanal(1) | a_netanal(2)

    with pytest.raises(TypeError, match="crs0000_rmod1, crs0000_rmod2"):
        find_resonances_in_netanal(netanal)
    # And says which index to pass instead.
    with pytest.raises(TypeError, match=r"netanal\['crs0000_rmod1'\]"):
        find_resonances_in_netanal(netanal)


def test_the_search_is_labelled_by_its_module_unless_you_name_it():
    """A warning from a loop over eight modules has to say which one."""
    assert find_resonances_in_netanal(a_module_netanal(2)).label == "module 2"
    assert find_resonances_in_netanal(
        a_module_netanal(2), label="cold"
    ).label == "cold"


def test_wrapper_says_what_it_looked_for():
    with pytest.raises(TypeError, match="Expected one module's netanal"):
        find_resonances_in_netanal([a_module_netanal()])


def test_wrapper_refuses_a_sweep():
    """A sweep's output walks to the same depth and means something else."""
    module_netanal = (
        a_multisweep({"R0001": a_section()}) | {"measurement": "multisweep"}
    )

    with pytest.raises(TypeError, match="not a netanal"):
        netanal_trace(module_netanal)


def test_wrapper_refuses_an_output_that_does_not_say_what_it_is():
    """Silence is not permission: {name: section} sits where the trace would."""
    with pytest.raises(TypeError, match="not a netanal"):
        netanal_trace(a_multisweep({"R0001": a_section()}))


# ─── the search left in the netanal ───────────────────────────────────────────


def stored_search(module_netanal) -> ResonanceSearch:
    """The search out of a netanal: an index and a from_dict, as a caller does it."""
    return ResonanceSearch.from_dict(
        netanal_trace(module_netanal)["resonance_search"]
    )


def test_the_search_goes_into_the_netanal_beside_the_trace():
    module_netanal = a_module_netanal()
    found = find_resonances_in_netanal(module_netanal, save=False)

    trace = netanal_trace(module_netanal)
    # As builtins and ndarrays, the way everything else in a file is: this goes
    # into a pickle, and a pickled class records its own import path.
    assert isinstance(trace["resonance_search"], dict)
    # And the measurement is still there beside it.
    assert "iq_counts" in trace

    restored = stored_search(module_netanal)
    assert restored.candidates == found.candidates
    assert restored.rejected == found.rejected
    assert np.array_equal(restored.magnitude_db, found.magnitude_db)


def test_a_search_lands_only_in_the_module_that_was_searched():
    """The loop is the caller's, so the other modules are untouched until it runs."""
    netanal = a_netanal(1) | a_netanal(2)
    found = find_resonances_in_netanal(netanal["crs0000_rmod1"], save=False)

    assert stored_search(netanal["crs0000_rmod1"]).candidates == found.candidates
    assert "resonance_search" not in netanal_trace(netanal["crs0000_rmod2"])


def test_searching_again_replaces_the_search_that_was_there():
    """One trace, one search — not a pile of them keyed by nothing."""
    module_netanal = a_module_netanal(resonances=TRUTH)
    find_resonances_in_netanal(module_netanal, min_dip_depth_db=1.0, save=False)
    shallow = find_resonances_in_netanal(
        module_netanal, min_dip_depth_db=0.01, save=False
    )

    stored = stored_search(module_netanal)
    assert stored.settings["min_dip_depth_db"] == 0.01
    assert stored.candidates == shallow.candidates


def test_an_unsearched_netanal_simply_has_no_search_in_it():
    assert "resonance_search" not in netanal_trace(a_module_netanal())


# ─── editing a search by hand ─────────────────────────────────────────────────
#
# The operator is the last rejection pass: what they drop is rejected with a
# reason rather than deleted, so the search stays the whole record of what was
# found and either edit undoes the other.


def test_rejecting_by_hand_keeps_the_candidate_and_its_reason():
    found = find_resonances(*a_sweep(), min_Q=1e4, max_Q=1e6)
    doomed = found.candidates[1]

    rejected = found.reject(doomed.frequency_hz)

    assert len(found.candidates) == len(TRUTH) - 1
    assert found.rejected == [rejected]
    assert rejected.rejected_because == ResonanceSearch.BY_HAND
    # Everything the finder measured about it, unchanged.
    assert replace(rejected, rejected_because=None) == doomed


def test_rejecting_takes_the_nearest_accepted_candidate():
    """A click lands near a resonance, not on its exact sample."""
    found = find_resonances(*a_sweep(), min_Q=1e4, max_Q=1e6)

    rejected = found.reject(TRUTH[2] + 3e3)

    assert rejected.frequency_hz == pytest.approx(TRUTH[2], abs=1e3)


def test_rejecting_with_nothing_accepted_says_so():
    found = find_resonances(*a_sweep(resonances=()), min_Q=1e4, max_Q=1e6,
                            label="module 2")

    with pytest.raises(ValueError, match="module 2 has no accepted candidates"):
        found.reject(1.05e9)


def test_accepting_restores_a_rejected_candidate_as_found():
    found = find_resonances(*a_sweep(), min_Q=1e4, max_Q=1e6)
    doomed = found.candidates[1]
    found.reject(doomed.frequency_hz)

    restored = found.accept(doomed.frequency_hz)

    assert restored == doomed
    assert found.rejected == []
    assert len(found.candidates) == len(TRUTH)


def test_accepting_a_threshold_rejection_is_the_same_move():
    """Nothing marks a by-hand rejection as the only reversible kind: a
    candidate the collision cut threw out comes back the same way."""
    frequencies, magnitude = a_close_pair()
    found = find_resonances(frequencies, magnitude, min_separation_hz=400e3)
    assert len(found.candidates) == 0 and len(found.rejected) == 2
    cut = found.rejected[0]

    restored = found.accept(cut.frequency_hz)

    assert restored == replace(cut, rejected_because=None)
    assert len(found.rejected) == 1


def test_accepting_somewhere_new_measures_the_trace_there():
    """A resonance the finder had no candidate for carries the numbers a found
    one does, read off the same trace with the same scipy calls."""
    shallow_at = 1.05e9
    frequencies, magnitude = a_sweep(
        resonances=TRUTH + (shallow_at,),
        qs=(2e4, 3e4, 2.5e4, 4e4, 3e4),
        depths=[0.7] * 4 + [0.02],
    )
    found = find_resonances(frequencies, magnitude, min_Q=1e4, max_Q=1e6)
    assert len(found.candidates) == len(TRUTH)   # the shallow one is under the floor

    added = found.accept(shallow_at)

    assert added.accepted
    assert added.frequency_hz == pytest.approx(shallow_at, abs=2e3)
    assert added.depth_db == pytest.approx(0.18, abs=0.05)
    assert added.q_estimate == pytest.approx(3e4, rel=0.2)


def test_an_accepted_candidate_lands_on_the_searched_grid():
    """Not wherever the click was: a candidate indexes the trace, so its
    frequency has to be a point of it."""
    found = find_resonances(*a_sweep(), min_Q=1e4, max_Q=1e6)

    added = found.accept(1.05e9 + 137.0)

    assert added.frequency_hz == found.frequencies_hz[added.index]


def test_accepted_candidates_stay_in_frequency_order():
    """``candidates`` is documented as ordered, and ``to_catalog`` numbers
    channels off it."""
    found = find_resonances(*a_sweep(), min_Q=1e4, max_Q=1e6)

    found.accept(1.01e9)
    found.accept(1.07e9)

    frequencies = [c.frequency_hz for c in found.candidates]
    assert frequencies == sorted(frequencies)


def test_accepting_where_one_is_already_accepted_says_so():
    found = find_resonances(*a_sweep(), min_Q=1e4, max_Q=1e6)
    already = found.candidates[0].frequency_hz

    with pytest.raises(ValueError, match="already an accepted resonance"):
        found.accept(already)


def test_a_hand_edited_search_round_trips_through_builtins():
    """The edits are candidates like any other, so the file needs no new keys
    and reads back with the reason the operator's rejection carries."""
    found = find_resonances(*a_sweep(), min_Q=1e4, max_Q=1e6)
    found.reject(TRUTH[0])
    found.accept(1.05e9)

    restored = ResonanceSearch.from_dict(found.to_dict())

    assert restored.candidates == found.candidates
    assert restored.rejected == found.rejected
    assert restored.rejected[0].rejected_because == ResonanceSearch.BY_HAND


def test_a_hand_edit_is_recorded_in_the_netanal_it_searched():
    """``record_search`` is how an edit reaches the measurement, the same way
    the search itself got there."""
    module_netanal = a_module_netanal()
    found = find_resonances_in_netanal(module_netanal, save=False)
    found.reject(found.candidates[0].frequency_hz)

    record_search(module_netanal, found, save=False)

    stored = ResonanceSearch.from_dict(
        netanal_trace(module_netanal)["resonance_search"])
    assert stored.candidates == found.candidates
    assert stored.rejected == found.rejected


# ─── handing the result onward ────────────────────────────────────────────────


def test_result_seeds_a_catalog():
    truth = TRUTH
    found = find_resonances(*a_sweep(resonances=truth), min_Q=1e4, max_Q=1e6)

    catalog = found.to_catalog(module=2, amplitude=0.01)

    assert isinstance(catalog, ResonatorCatalog)
    assert len(catalog) == len(truth)
    assert [r.channel for r in catalog] == [1, 2, 3, 4]
    assert np.allclose(
        [r.bias.frequency_hz for r in catalog],
        found.resonance_frequencies_hz,
    )
    assert all(r.bias.amplitude == 0.01 for r in catalog)


def test_repr_summarises_without_dumping_the_array():
    found = find_resonances(*a_sweep(), min_Q=1e4, max_Q=1e6)
    text = repr(found)

    assert "4 found" in text
    assert "0 rejected" in text


# ─── Persistence ──────────────────────────────────────────────────────────────


def test_a_search_survives_a_round_trip_through_builtins():
    found = find_resonances(*a_sweep(), min_Q=1e4, max_Q=1e6, label="module 2")
    restored = ResonanceSearch.from_dict(found.to_dict())

    assert restored.label == found.label
    assert restored.candidates == found.candidates
    assert restored.rejected == found.rejected
    assert np.array_equal(restored.frequencies_hz, found.frequencies_hz)
    assert np.array_equal(restored.magnitude_db, found.magnitude_db)


def test_a_searchs_dict_holds_no_rfmux_classes():
    """Files have to open on a machine that has never heard of rfmux."""
    d = find_resonances(*a_sweep(), min_Q=1e4, max_Q=1e6).to_dict()

    assert d["schema_version"] == ResonanceSearch.SCHEMA_VERSION
    assert all(type(c).__name__ == "dict" for c in d["candidates"])
    assert all(
        type(v).__module__ in ("builtins", "numpy") for v in d["settings"].values()
    )


def test_a_search_from_another_version_is_refused():
    d = find_resonances(*a_sweep(), min_Q=1e4, max_Q=1e6).to_dict()
    d["schema_version"] = ResonanceSearch.SCHEMA_VERSION + 1

    with pytest.raises(ValueError, match="schema_version"):
        ResonanceSearch.from_dict(d)
