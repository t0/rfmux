"""What ``min_resonance_separation_hz`` promises, in each of its two modes.

The separation has always been ``distance`` in ``scipy.signal.find_peaks``,
which thins: of a group closer than it, the most prominent survives and the
rest are discarded.  That makes the returned list obey the separation while
saying nothing about the array -- the survivor can have a real resonance
beside it, namely the one that was thrown away.  Biasing it reads two
detectors at once.

``require_isolation`` drops the whole group instead, so every resonance that
comes back is one nothing else is near.
"""

import warnings

import numpy as np

from rfmux.algorithms.measurement.fitting import find_resonances

# Wide enough that neighbours never overlap by accident, and finely enough
# sampled that a 30 kHz gap is many points across.
FREQS = np.linspace(1.000e9, 1.010e9, 20001)
SEPARATION_HZ = 100e3

# Two clean resonances, and a pair 30 kHz apart -- well inside the 100 kHz
# separation, so the pair is a collision and neither member is usable.
ISOLATED = [1.0010e9, 1.0040e9]
COLLIDED = [1.00700e9, 1.00703e9]


def _s21(centres, Q=5e4, depth=0.8):
    """Ideal resonators multiplied together."""
    out = np.ones_like(FREQS, dtype=complex)
    for fr in centres:
        out = out * (1 - depth / (1 + 2j * Q * (FREQS - fr) / fr))
    return out


def _find(centres, **kw):
    kw.setdefault("min_resonance_separation_hz", SEPARATION_HZ)
    kw.setdefault("min_dip_depth_db", 1.0)
    kw.setdefault("min_Q", 1e3)
    kw.setdefault("max_Q", 1e7)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = find_resonances(FREQS, _s21(centres), **kw)
    return result, [str(w.message) for w in caught]


def _mhz(result):
    return [round(f / 1e6, 3) for f in result["resonance_frequencies"]]


def test_thinning_keeps_one_of_a_collided_pair():
    """The old behaviour, stated as a test so the change is visible.

    The survivor looks like an ordinary resonance: nothing in the result
    says another one sits 30 kHz away.
    """
    result, _ = _find(ISOLATED + COLLIDED)
    assert _mhz(result) == [1001.0, 1004.0, 1007.0]

    # It is reported with a frequency, a width and a Q like any other.
    assert len(result["resonances_details"]) == 3


def test_isolation_drops_both_members():
    result, _ = _find(ISOLATED + COLLIDED, require_isolation=True)
    assert _mhz(result) == [1001.0, 1004.0]
    assert len(result["resonances_details"]) == 2


def test_isolation_leaves_a_clean_array_alone():
    """Nothing is within the separation, so both modes agree."""
    thinned, _ = _find(ISOLATED)
    isolated, _ = _find(ISOLATED, require_isolation=True)
    assert _mhz(thinned) == _mhz(isolated) == [1001.0, 1004.0]


def test_the_drop_is_reported():
    """Silently returning fewer resonances would look like a bad sweep."""
    _, messages = _find(ISOLATED + COLLIDED, require_isolation=True)
    assert any("Dropped 2 of 4" in m for m in messages), messages
    assert any("1e+05 Hz" in m for m in messages), messages


def test_a_run_of_three_goes_entirely():
    """Isolation is a property of a resonance, not of a pair.

    The middle one is close to both neighbours; the outer two are each
    close to the middle.  None of the three is isolated.
    """
    run = [1.00700e9, 1.00703e9, 1.00706e9]
    result, _ = _find(ISOLATED + run, require_isolation=True)
    assert _mhz(result) == [1001.0, 1004.0]


def test_separation_is_the_boundary():
    """A gap well over the separation is far enough; half of it is not."""
    wide = [1.00700e9, 1.00700e9 + 1.5 * SEPARATION_HZ]
    kept, _ = _find(wide, require_isolation=True)
    assert len(kept["resonance_frequencies"]) == 2

    narrow = [1.00700e9, 1.00700e9 + 0.5 * SEPARATION_HZ]
    dropped, _ = _find(narrow, require_isolation=True)
    assert dropped["resonance_frequencies"] == []


def test_expected_resonances_counts_only_isolated_ones():
    """The top-N is taken after the collisions are removed.

    Filtering afterwards would let a collided peak use up one of the N
    slots and then be dropped, returning fewer than were available.
    """
    result, _ = _find(ISOLATED + COLLIDED, require_isolation=True,
                      expected_resonances=2)
    assert _mhz(result) == [1001.0, 1004.0]


def test_nothing_to_collide_with():
    """One resonance, and none at all: both are trivially isolated."""
    one, _ = _find([1.0040e9], require_isolation=True)
    assert _mhz(one) == [1004.0]

    flat = find_resonances(FREQS, np.ones_like(FREQS, dtype=complex),
                           min_resonance_separation_hz=SEPARATION_HZ,
                           require_isolation=True)
    assert flat["resonance_frequencies"] == []


def test_default_is_the_old_behaviour():
    """Callers that never heard of this keep the results they had."""
    explicit, _ = _find(ISOLATED + COLLIDED, require_isolation=False)
    default, _ = _find(ISOLATED + COLLIDED)
    assert _mhz(default) == _mhz(explicit) == [1001.0, 1004.0, 1007.0]
