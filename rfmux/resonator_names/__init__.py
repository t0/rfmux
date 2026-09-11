"""Names for resonators, generated on the fly.

This package is derived from and based on the resonator_name_generator project
by Maclean Rouble:
https://github.com/macleaner/resonator_name_generator

Contents (trimmed for rfmux):
- syllables: pronounceable syllabic strings of an exact length, made up rather
  than looked up
- boring: a plain counter, ``R0001…``, for when a number is the right answer

Modifications in this repository include:
- Trimmed to the two modules that generate names without data. Upstream ships
  five curated word lists (~680 kB of human given names, pet names, and nouns)
  and a category-weighting layer to mix them; neither comes along, so nothing
  here reads a file.
- Removal of the build pipeline (``tools/``), which regenerates those lists from
  Wikidata and other sources over the network.
- Added the catalog-facing namers below, which draw one distinct name per
  resonator. Upstream's equivalent lives in its weighting layer.
- Names are upper case here and capitalised upstream (``BOTA``, not ``Bota``).
  The change is made in :func:`syllabic_name` rather than in ``syllables.py``,
  so the vendored modules stay byte-identical and re-vendoring stays a copy.

Original project license: see rfmux/resonator_names/LICENSE (CC0 1.0, upstream
LICENSE retained)

A namer is a function of the resonators' frequencies::

    syllabic_names(frequencies_hz)                # ['BOTA', 'KOZR', 'SPET', …]
    numbered_names(frequencies_hz)                # ['R0001', 'R0002', …]
    syllabic_names_from_frequency(frequencies_hz) # stable per resonator

:meth:`~rfmux.core.resonators.ResonatorCatalog.from_frequencies` takes one of
these, so a catalog can be named any of those ways without the constructor
growing a flag per scheme. Taking the frequencies rather than just a count is
what lets a namer derive a name from the resonator it is naming — see
:func:`syllabic_names_from_frequency` — and lets you write your own.
"""

from __future__ import annotations

import random
from typing import Iterable, Sequence

from .boring import (
    DEFAULT_PREFIX,
    DEFAULT_WIDTH,
    boring_name,
    boring_names,
)
from .syllables import (
    MIN_LENGTH,
    random_syllabic_string as _random_syllabic_string,
)

__all__ = [
    "DEFAULT_LENGTH",
    "DEFAULT_PREFIX",
    "DEFAULT_QUANTUM_HZ",
    "DEFAULT_WIDTH",
    "MIN_LENGTH",
    "boring_name",
    "boring_names",
    "numbered_names",
    "syllabic_name",
    "syllabic_names",
    "syllabic_names_from_frequency",
]

#: How long a generated name is. Four characters is short enough to sit in a
#: legend or a table column without wrapping, and there are ~89,000 reachable
#: strings at this length, so a full module of 1024 draws without trouble.
#:
#: The cost is that four characters is not much room to be distinctive: among
#: 1000 names of this length, expect on the order of a thousand pairs that
#: differ in a single character (``BITA`` and ``BOTA``). Raise this if an array
#: is large and the names are being read aloud or typed; 6 removes essentially
#: all of that.
DEFAULT_LENGTH = 4


def syllabic_name(
    length: int = DEFAULT_LENGTH,
    *,
    rng: random.Random | int | None = None,
) -> str:
    """One made-up, pronounceable name of exactly ``length`` characters.

    Upper case, which is where rfmux differs from the vendored generator: it
    hands back ``Bota`` and this hands back ``BOTA``. Case is decided here, in
    one place, so ``syllables.py`` stays byte-identical to upstream.

    All caps reads as a label rather than as a word, which is what these are —
    nobody is meant to wonder whether ``BOTA`` is an English word they should
    recognise. It also keeps a name visually distinct from the surrounding prose
    in a plot legend or a log line.

    This is the single-name primitive. For a catalog's worth, all distinct, use
    :func:`syllabic_names`.

    :param length: the exact character count, at least :data:`MIN_LENGTH`.
    :param rng: a :class:`random.Random`, or an int seed, for a reproducible draw.
    """
    return _random_syllabic_string(length, rng=rng).upper()


#: Consecutive draws that all turn out to be duplicates before the space is
#: treated as used up. There is no pool to watch empty, so this stands in for
#: one: a run this long means the strings of that length really are exhausted
#: rather than merely unlucky.
_MISSES = 5000


def _distinct(n: int, length: int, rng: random.Random, seen: set[str]) -> list[str]:
    """Draw ``n`` strings none of which is in ``seen``, adding each as it lands."""
    drawn: list[str] = []
    while len(drawn) < n:
        for _ in range(_MISSES):
            candidate = syllabic_name(length, rng=rng)
            if candidate not in seen:
                break
        else:
            raise ValueError(
                f"ran out of distinct syllabic strings of length {length} after "
                f"{len(drawn)} of {n}; a longer length has far more of them "
                f"({length} -> {length + 2} is roughly a hundredfold)"
            )
        seen.add(candidate)
        drawn.append(candidate)
    return drawn


def syllabic_names(
    frequencies_hz: Sequence[float],
    *,
    length: int = DEFAULT_LENGTH,
    rng: random.Random | int | None = None,
    avoid: Iterable[str] = (),
) -> list[str]:
    """One distinct made-up name per frequency, in the order given.

    The default namer. Names are drawn rather than derived, so the frequencies
    are used only for their count — two runs over the same array give different
    names. That is the point: a name that carries no ordering cannot imply one
    that later goes stale, the way ``R0007`` does the moment a resonator is
    removed or retuned.

    When you need the same array to come back with the same names — a demo
    notebook whose prose names a resonator out loud — use
    :func:`syllabic_names_from_frequency`, or pass ``rng`` a seed to fix the
    sequence.

    :param frequencies_hz: the resonators being named; only the count is read.
    :param length: characters per name. See :data:`DEFAULT_LENGTH`.
    :param rng: a :class:`random.Random`, or an int seed, for a reproducible
        draw. Note this fixes the *sequence* of names, not which resonator gets
        which: names are handed out positionally, so one extra resonance shifts
        every name after it onto a different resonator.
    :param avoid: names already in use, e.g. those of resonators already named.
    :raises ValueError: if there are not enough distinct strings of that length.
    """
    if length < MIN_LENGTH:
        raise ValueError(f"length must be at least {MIN_LENGTH}, got {length}")
    resolved = rng if isinstance(rng, random.Random) else random.Random(rng)
    return _distinct(len(frequencies_hz), length, resolved, set(avoid))


#: How coarsely a frequency is bucketed before it seeds a name. Ten kilohertz
#: sits in the gap between the two things this has to survive: re-measuring one
#: resonator moves it by a sweep step or so, which must *not* change its name,
#: while two distinct resonators are a hundred kilohertz apart or more (the
#: usual ``min_resonance_separation_hz``), which must.
DEFAULT_QUANTUM_HZ = 10e3


def syllabic_names_from_frequency(
    frequencies_hz: Sequence[float],
    *,
    length: int = DEFAULT_LENGTH,
    quantum_hz: float = DEFAULT_QUANTUM_HZ,
    avoid: Iterable[str] = (),
) -> list[str]:
    """One made-up name per frequency, derived from that frequency.

    Each name is a function of the resonator it names, so the resonator at
    4.512300 GHz gets the same name in every run, on every machine, against mock
    or real hardware — no seed to pass and nothing to keep in step. Unlike a
    seeded draw, this binds a name to a *resonator* rather than to a position,
    so finding one extra resonance does not rename everything after it.

    That is what makes it worth having for documentation: a notebook can say
    "``BOTA`` is the one that goes nonlinear first" and still be right next
    week. It is off by default because outside that setting the stability buys
    nothing, and a name that is secretly a hash of a frequency invites being
    read as one.

    Two caveats, both from the bucketing:

    - A frequency landing near a bucket edge can fall either side of it between
      runs and take a different name. ``quantum_hz`` trades that risk against
      the next one.
    - Two resonators inside one bucket derive the same name. The later one in
      frequency order redraws, which makes *its* name depend on the array again.
      At the default separations this is rare enough not to matter.

    :param frequencies_hz: the resonators being named, in the order to name them.
    :param length: characters per name. See :data:`DEFAULT_LENGTH`.
    :param quantum_hz: bucket width. See :data:`DEFAULT_QUANTUM_HZ`.
    :param avoid: names already in use.
    :raises ValueError: if there are not enough distinct strings of that length.
    """
    if length < MIN_LENGTH:
        raise ValueError(f"length must be at least {MIN_LENGTH}, got {length}")
    if not quantum_hz > 0:
        raise ValueError(f"quantum_hz must be positive, got {quantum_hz}")

    seen = set(avoid)
    names: list[str] = []
    for frequency in frequencies_hz:
        bucket = round(float(frequency) / quantum_hz)
        # Seeded per resonator, so this draw depends on nothing but the
        # frequency -- until it collides, when _distinct carries on from the
        # same stream rather than starting a shared one.
        names.extend(_distinct(1, length, random.Random(bucket), seen))
    return names


def numbered_names(
    frequencies_hz: Sequence[float],
    *,
    prefix: str = DEFAULT_PREFIX,
    start: int = 1,
    width: int = DEFAULT_WIDTH,
    avoid: Iterable[str] = (),
) -> list[str]:
    """``R0001…``, one per frequency, in the order given.

    The namer for when a number is the right answer — a wafer being screened, a
    figure that has to sort correctly, a reader who wants to know which
    resonator is which without learning a vocabulary. Note what it asserts: the
    catalog's order at the moment it was built, which stops being true as soon
    as a resonator is removed or retuned.

    :param frequencies_hz: the resonators being named; only the count is read.
    :param prefix: what to put in front of the number, verbatim. Must not
        contain whitespace or end in a digit.
    :param start: the first index to use.
    :param width: the minimum number of digits; a batch running past it widens
        as a whole, so the names stay the same width as each other.
    :param avoid: names already in use. Matched on the number, so ``R1`` and
        ``R0001`` both mean index 1.
    """
    return boring_names(
        len(frequencies_hz), prefix, start=start, width=width, avoid=avoid
    )
