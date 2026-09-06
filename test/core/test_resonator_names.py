"""What the namers promise.

The two vendored modules (``syllables``, ``boring``) are upstream's and are
tested there; this covers the rfmux layer on top — that a draw is distinct, that
a frequency-derived name is stable against the things that move, and that a
namer plugs into a catalog.
"""

import random
from functools import partial

import pytest

from rfmux.resonator_names import (
    DEFAULT_LENGTH,
    boring_names,
    numbered_names,
    syllabic_name,
    syllabic_names,
    syllabic_names_from_frequency,
)

pytestmark = pytest.mark.portable


def some_frequencies(n=8, start=4.5e9, spacing=1e6):
    return [start + i * spacing for i in range(n)]


# ─── syllabic_names ───────────────────────────────────────────────────────────


def test_one_name_per_frequency_all_distinct():
    names = syllabic_names(some_frequencies(200))
    assert len(names) == 200
    assert len(set(names)) == 200


def test_names_are_the_default_length():
    assert all(len(n) == DEFAULT_LENGTH for n in syllabic_names(some_frequencies(50)))
    assert all(len(n) == 6 for n in syllabic_names(some_frequencies(50), length=6))


def test_names_are_upper_case_and_ascii():
    """All caps reads as a label rather than as a word."""
    for name in syllabic_names(some_frequencies(50)):
        assert name.isascii() and name.isalpha()
        assert name.isupper()


def test_the_primitive_is_upper_case_too():
    """Case is decided in one place, so both entry points agree."""
    assert syllabic_name(6, rng=0).isupper()


def test_an_unseeded_draw_differs_between_runs():
    """The whole point: a name carries no ordering, so it need not be stable."""
    assert syllabic_names(some_frequencies(20)) != syllabic_names(some_frequencies(20))


def test_a_seed_fixes_the_draw():
    assert syllabic_names(some_frequencies(20), rng=7) == syllabic_names(
        some_frequencies(20), rng=7
    )
    assert syllabic_names(some_frequencies(20), rng=random.Random(7)) == syllabic_names(
        some_frequencies(20), rng=7
    )


def test_avoid_is_honoured():
    taken = syllabic_names(some_frequencies(30), rng=1)
    fresh = syllabic_names(some_frequencies(30), rng=1, avoid=taken)
    assert not set(fresh) & set(taken)


def test_a_length_below_a_syllable_is_refused():
    with pytest.raises(ValueError, match="at least"):
        syllabic_names(some_frequencies(3), length=1)


def test_running_out_says_so_rather_than_returning_a_short_list():
    """Length 3 holds a few thousand strings, so this is reachable."""
    with pytest.raises(ValueError, match="ran out of distinct"):
        syllabic_names(some_frequencies(20_000), length=3)


# ─── syllabic_names_from_frequency ────────────────────────────────────────────


def test_a_derived_name_is_the_same_every_run():
    freqs = some_frequencies(30)
    assert syllabic_names_from_frequency(freqs) == syllabic_names_from_frequency(freqs)


def test_a_derived_name_survives_re_measurement_jitter():
    """Finding the same resonator again moves it a little; the name must hold."""
    freqs = some_frequencies(30)
    nudged = [f + 400 for f in freqs]  # under a tone-grid step
    assert syllabic_names_from_frequency(nudged) == syllabic_names_from_frequency(freqs)


def test_a_derived_name_belongs_to_its_resonator_not_its_position():
    """One extra resonance must not rename everything after it."""
    freqs = some_frequencies(10)
    before = syllabic_names_from_frequency(freqs)
    extra = sorted(freqs + [freqs[4] + 500e3])
    after = syllabic_names_from_frequency(extra)
    for frequency, name in zip(freqs, before):
        assert after[extra.index(frequency)] == name


def test_derived_names_are_distinct_within_a_catalog():
    names = syllabic_names_from_frequency(some_frequencies(200))
    assert len(set(names)) == 200


def test_two_resonators_in_one_bucket_still_get_two_names():
    names = syllabic_names_from_frequency([4.5e9, 4.5e9 + 1.0])
    assert names[0] != names[1]


def test_a_non_positive_quantum_is_refused():
    with pytest.raises(ValueError, match="quantum_hz must be positive"):
        syllabic_names_from_frequency(some_frequencies(3), quantum_hz=0)


# ─── numbered_names ───────────────────────────────────────────────────────────


def test_numbered_names_count_from_one():
    assert numbered_names(some_frequencies(3)) == ["R0001", "R0002", "R0003"]


def test_numbered_names_take_a_prefix_and_a_start():
    assert numbered_names(some_frequencies(2), prefix="kid", start=7) == [
        "kid0007",
        "kid0008",
    ]


def test_numbered_names_skip_what_is_already_taken():
    assert numbered_names(some_frequencies(2), avoid=["R0001"]) == ["R0002", "R0003"]


def test_a_partial_is_a_usable_namer():
    """partial() is what the catalog docstring recommends for a custom prefix."""
    namer = partial(numbered_names, prefix="kid")
    assert namer(some_frequencies(2)) == ["kid0001", "kid0002"]


# ─── the primitives still work as vendored ────────────────────────────────────


def test_the_syllabic_primitive_honours_an_exact_length():
    assert len(syllabic_name(9, rng=0)) == 9


def test_the_boring_primitive_widens_a_batch_as_a_whole():
    assert boring_names(3, start=9999) == ["R09999", "R10000", "R10001"]
