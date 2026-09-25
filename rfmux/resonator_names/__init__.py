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
these functions through its ``names`` argument. You can also supply a custom
function that receives sorted frequencies and returns one name per frequency.
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

# Default character count. Increase it to make similar names easier to distinguish.
DEFAULT_LENGTH = 4


def syllabic_name(
    length: int = DEFAULT_LENGTH,
    *,
    rng: random.Random | int | None = None,
) -> str:
    """Generate one uppercase, pronounceable name of exactly ``length`` characters.

    Use :func:`syllabic_names` to generate several distinct names.

    :param length: character count, at least :data:`MIN_LENGTH`.
    :param rng: a :class:`random.Random` instance or integer seed for reproducibility.
    """
    return _random_syllabic_string(length, rng=rng).upper()


# Maximum consecutive duplicate draws before raising ValueError.
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
    """Generate one distinct name per input frequency, in the supplied order.

    Only the number of frequencies is used. Names are random unless ``rng``
    is seeded. A fixed seed repeats the sequence, but inserting a frequency
    changes which names are assigned to later resonators.

    :param frequencies_hz: frequencies to name; only their count is used.
    :param length: characters per name.
    :param rng: a :class:`random.Random` instance or integer seed.
    :param avoid: names already in use, which must not be generated.
    :raises ValueError: if length is too short or distinct-name retries are exhausted.
    """
    if length < MIN_LENGTH:
        raise ValueError(f"length must be at least {MIN_LENGTH}, got {length}")
    resolved = rng if isinstance(rng, random.Random) else random.Random(rng)
    return _distinct(len(frequencies_hz), length, resolved, set(avoid))


# Frequency bucket width in Hz for repeatable name generation.
DEFAULT_QUANTUM_HZ = 10e3


def syllabic_names_from_frequency(
    frequencies_hz: Sequence[float],
    *,
    length: int = DEFAULT_LENGTH,
    quantum_hz: float = DEFAULT_QUANTUM_HZ,
    avoid: Iterable[str] = (),
) -> list[str]:
    """Generate distinct names using rounded frequency buckets as random seeds.

    For each frequency, seed a separate generator with
    ``round(frequency / quantum_hz)``. The same bucket gives the same first
    name. If that name is already used or in ``avoid``, draw again from that
    generator until a distinct name is found.

    Names usually survive small frequency changes and additions to an array.
    They can change when a frequency crosses a bucket boundary or when a
    name collision changes which draw is available. Collisions can occur
    within a bucket or between different buckets.

    :param frequencies_hz: frequencies in Hz, processed in the supplied order.
    :param length: characters per name.
    :param quantum_hz: positive bucket width in Hz; defaults to 10 kHz.
    :param avoid: names already in use.
    :raises ValueError: if length or bucket width is invalid, or retries are exhausted.
    """
    if length < MIN_LENGTH:
        raise ValueError(f"length must be at least {MIN_LENGTH}, got {length}")
    if not quantum_hz > 0:
        raise ValueError(f"quantum_hz must be positive, got {quantum_hz}")

    seen = set(avoid)
    names: list[str] = []
    for frequency in frequencies_hz:
        bucket = round(float(frequency) / quantum_hz)
        # Retry collisions using this bucket's generator.
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

    Numbers follow input order. Catalogs retain these names after retuning or
    removal, so the numbers may no longer match current frequency order.

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
