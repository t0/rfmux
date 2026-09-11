"""Pronounceable syllabic strings of an exact length.

Where upstream draws from curated word lists, this module makes words up::

    >>> random_syllabic_string(6)           # doctest: +SKIP
    'Tavren'
    >>> random_syllabic_string(9)           # doctest: +SKIP
    'Marilena'

The usual way in is :func:`~rfmux.resonator_names.syllabic_names`, which draws a
distinct name per resonator; this module is the primitive underneath, for when a
single word of a known length is all that is wanted.

A syllabic string earns its place when the arrays are large enough that the
lists run dry, when every name has to be the same width for a plot legend or a
filename, or simply when a name that is definitely not anybody's is preferable.

How the length comes out exact
------------------------------

A word is a run of syllables, each of them an *onset* (1-2 consonant letters,
or nothing at the start of a word), a *nucleus* (1-2 vowel letters), and a
*coda* (0-2 consonant letters). The shortest syllable is therefore 2 characters
and the next shortest is 3, and since every integer above 1 is a sum of 2s and
3s, a single rule keeps the generator from ever painting itself into a corner:

    never leave exactly one character to fill at a syllable boundary.

That is the whole of the length logic -- no backtracking, no rejection, no
retrying until something happens to fit. Each unit is drawn from the lengths
still compatible with what remains, and the last syllable lands on the boundary
exactly.

Why it reads as pronounceable
-----------------------------

Three things, beyond the consonant-vowel alternation the syllable shape gives
for free:

- The inventories are English-ish clusters, not arbitrary letter pairs: ``br``
  and ``st`` are onsets, ``bt`` and ``zk`` are not.
- Syllable junctions are constrained. After a syllable that ends in a consonant
  the next one must start with a single consonant, which caps any consonant run
  at three (``Astrid``, not ``Astkrid``); after one that ends in a *stop* that
  consonant must further be a liquid, nasal, or glide (``Sidra`` and ``Abner``,
  never ``Sonksest``); and a doubled letter is only allowed where English
  doubles it.
- Codas are weighted so most syllables are open (end in a vowel), which is what
  makes a word flow rather than clatter.

Nothing here is a phonological model; it is an inventory and three rules, tuned
by reading the output.
"""

from __future__ import annotations

import random
import re
from typing import Container, Iterable, Mapping

__all__ = ["MIN_LENGTH", "random_syllabic_string"]

#: Shorter than this there is no room for a syllable.
MIN_LENGTH = 2


# Weights are relative and were set by eye. Single consonants dominate the
# onsets, clusters are deliberately rare, and the empty onset (drawn only at the
# start of a word) is what yields Aurelia and Elowen rather than every name
# opening on a consonant.
_ONSETS: Mapping[str, float] = {
    "": 3.0,
    "b": 4.0, "c": 3.0, "d": 5.0, "f": 3.0, "g": 3.0, "h": 3.0, "j": 1.5,
    "k": 3.0, "l": 4.0, "m": 5.0, "n": 4.0, "p": 3.0, "r": 4.0, "s": 5.0,
    "t": 5.0, "v": 2.0, "w": 2.0, "y": 1.5, "z": 1.2,
    "ch": 1.2, "sh": 1.2, "th": 1.2, "ph": 0.4, "wh": 0.4,
    "bl": 0.8, "br": 1.0, "cl": 0.8, "cr": 0.8, "dr": 0.8, "fl": 0.8,
    "fr": 0.8, "gl": 0.6, "gr": 0.8, "kr": 0.4, "pl": 0.7, "pr": 0.7,
    "sc": 0.4, "sk": 0.5, "sl": 0.7, "sm": 0.5, "sn": 0.5, "sp": 0.6,
    "st": 0.9, "sw": 0.5, "tr": 1.0, "tw": 0.3,
}

# ``y`` counts as a vowel here, which is what makes Lyra and Myrin possible.
# Digraphs are kept to about one nucleus in seven: they are what stops every
# word sounding the same, but two or three in one word (``Boumbauba``) stop it
# sounding like a word at all.
_NUCLEI: Mapping[str, float] = {
    "a": 9.0, "e": 9.0, "i": 7.0, "o": 7.0, "u": 4.0, "y": 1.5,
    "ae": 0.15, "ai": 0.6, "au": 0.4, "ea": 0.5, "ee": 0.4, "ei": 0.3,
    "eo": 0.2, "ia": 0.6, "ie": 0.45, "io": 0.35, "oa": 0.25, "oe": 0.2,
    "oi": 0.25, "oo": 0.35, "ou": 0.4, "ua": 0.2, "ue": 0.2, "ui": 0.15,
}

# The empty coda carries most of the weight on purpose: roughly three syllables
# in five end on their vowel, which is the difference between Tavina and
# Tarvent. Lower it and long words start to clatter -- the effect is invisible
# at length 5 and unmissable at length 12, so tune it by reading long ones.
_CODAS: Mapping[str, float] = {
    "": 60.0,
    "b": 0.6, "d": 1.5, "f": 0.5, "g": 0.6, "k": 1.5, "l": 3.0, "m": 2.5,
    "n": 5.0, "p": 0.8, "r": 4.0, "s": 3.5, "t": 2.5, "v": 0.3, "x": 0.5,
    "z": 0.4,
    "ck": 0.8, "ft": 0.4, "ll": 0.8, "lm": 0.3, "ls": 0.4, "lt": 0.5,
    "mb": 0.3, "mp": 0.4, "nd": 1.0, "ng": 0.9, "nk": 0.5, "ns": 0.5,
    "nt": 0.9, "rd": 0.6, "rk": 0.5, "rn": 0.7, "rs": 0.5, "rt": 0.6,
    "sh": 0.5, "sk": 0.4, "ss": 0.6, "st": 1.0, "th": 0.6,
}

#: Consonants English is happy to double across a syllable boundary, so that
#: Annika and Bellamy stay reachable while Ngkara does not.
_GEMINABLE = frozenset({"b", "d", "f", "g", "l", "m", "n", "p", "r", "s", "t"})

#: Single-consonant codas sonorous enough that any consonant can follow: Sandro,
#: Marta, Alva, Esta all read fine. A stop or a fricative is an awkward place to
#: restart on another obstruent -- Sidka, Abgo -- so only a liquid, a nasal, or
#: a glide may follow one of those.
_OPEN_AFTER = frozenset("lmnrs")
_AFTER_OBSTRUENT = frozenset({"l", "r", "m", "n", "w", "y"})

#: Two-consonant codas that can sit mid-word, and then only before a liquid.
#: These are the three-consonant runs English actually says -- ``ndr`` in
#: Sandra, ``str`` in Astrid, ``ckl`` in Buckley. The two-consonant codas left
#: out (``rn``, ``ls``, ``ss``, ...) end in a sonorant themselves, and stacking
#: a third consonant on those gives ``rnl`` and ``ssn``; they close a word
#: perfectly well, so they are simply kept to the end of one.
_MEDIAL_CLUSTERS = frozenset(
    {"ck", "ft", "lt", "mb", "mp", "nd", "ng", "nk", "nt", "rd", "rk", "rt",
     "sk", "st", "th"}
)
_AFTER_CLUSTER = frozenset({"l", "r"})

_SINGLE_ONSETS = frozenset(unit for unit in _ONSETS if len(unit) == 1)

#: Shapes the inventories can still produce but that do not read well: any
#: letter three times over, four consonants in a row, a doubled ``y``.
_UGLY = re.compile(r"(.)\1\1|[^aeiouy]{4}|yy")

# A syllabic string passes under nobody's eye before it lands on a plot, so this
# list is the only thing standing between the generator and an awkward caption.
# Over-blocking costs nothing here -- there is no real word to protect, only one
# nonsense string swapped for another -- so it errs wide, and short fragments
# that would be unusable against a real corpus are fine.
_UNSAFE = re.compile(
    "|".join(
        (
            "anal", "anus", "arse", "ass", "bastard", "bitch", "boob", "bugger",
            "chink", "clit", "cock", "coon", "crap", "cum", "cunt", "dick",
            "dyke", "fag", "fart", "fuk", "fuc", "gook", "hitler", "jihad",
            "jizz", "kike", "kunt", "nazi", "negr", "nigg", "paki", "penis",
            "phuk", "piss", "poop", "porn", "pube", "puss", "queer", "rape",
            "rapi", "retard", "scat", "semen", "sex", "shit", "slut", "spic",
            "sperm", "tard", "tit", "turd", "twat", "vagin", "wank", "whore",
            "wog", "wop",
        )
    )
)


def _table(units: Mapping[str, float]) -> dict[int, tuple[tuple[str, ...], tuple[float, ...]]]:
    """Group an inventory by unit length, ready for a weighted draw."""
    grouped: dict[int, list[tuple[str, float]]] = {}
    for unit, weight in units.items():
        grouped.setdefault(len(unit), []).append((unit, weight))
    return {
        length: (tuple(u for u, _ in items), tuple(w for _, w in items))
        for length, items in grouped.items()
    }


_ONSET_TABLE = _table(_ONSETS)
_NUCLEUS_TABLE = _table(_NUCLEI)
_CODA_TABLE = _table(_CODAS)


def _draw(
    table: Mapping[int, tuple[tuple[str, ...], tuple[float, ...]]],
    lengths: Iterable[int],
    rng: random.Random,
    allowed: Container[str] | None = None,
) -> str:
    """Pick one unit of an allowed length, by weight.

    ``allowed``, when given, further restricts which units may be drawn -- it is
    how the junction rules narrow the onset after a closed syllable.
    """
    units: list[str] = []
    weights: list[float] = []
    for length in lengths:
        entry = table.get(length)
        if entry is None:
            continue
        for unit, weight in zip(*entry):
            if allowed is not None and unit not in allowed:
                continue
            units.append(unit)
            weights.append(weight)
    if not units:  # pragma: no cover - the length rules guarantee a candidate
        raise RuntimeError(f"no unit available for lengths {sorted(lengths)}")
    return rng.choices(units, weights=weights)[0]


def _draw_coda(remaining: int, rng: random.Random) -> str:
    """Close a syllable with ``remaining`` characters still to fill.

    Lengths are limited to those leaving 0 or 2-and-up -- one leftover character
    is the state no syllable can fill -- and a two-consonant coda mid-word is
    limited to the clusters something can legally follow.
    """
    units: list[str] = []
    weights: list[float] = []
    for length in (0, 1, 2):
        entry = _CODA_TABLE.get(length)
        if entry is None or length > remaining or remaining - length == 1:
            continue
        word_final = remaining - length == 0
        for unit, weight in zip(*entry):
            if length == 2 and not word_final and unit not in _MEDIAL_CLUSTERS:
                continue
            units.append(unit)
            weights.append(weight)
    return rng.choices(units, weights=weights)[0]


def _onsets_after(coda: str) -> frozenset[str]:
    """Which onsets may open the syllable following ``coda``."""
    if len(coda) == 2:
        return _AFTER_CLUSTER - {coda[-1]}
    allowed = _SINGLE_ONSETS if coda in _OPEN_AFTER else _AFTER_OBSTRUENT
    if coda in _GEMINABLE:
        return allowed  # Annika, Bellamy
    return allowed - {coda}


def _build(length: int, rng: random.Random) -> str:
    """Assemble one word of exactly ``length`` characters, lowercase."""
    parts: list[str] = []
    remaining = length
    coda = ""  # of the previous syllable; empty at the start of the word
    at_start = True

    while remaining:
        # Leaving one character at a syllable boundary is the one unfillable
        # state, so every choice below keeps the remainder at 0 or 2-and-up.
        if at_start:
            onset_lengths = [n for n in (0, 1, 2) if remaining - n >= 1]
            allowed = None
        elif coda:
            # After a closed syllable, one consonant only -- this is what caps
            # a consonant run at three -- and not just any one.
            onset_lengths = [1]
            allowed = _onsets_after(coda)
        else:
            onset_lengths = [n for n in (1, 2) if remaining - n >= 1]
            allowed = None

        onset = _draw(_ONSET_TABLE, onset_lengths, rng, allowed)
        remaining -= len(onset)

        nucleus = _draw(_NUCLEUS_TABLE, [n for n in (1, 2) if remaining - n >= 0], rng)
        remaining -= len(nucleus)

        coda = _draw_coda(remaining, rng)
        remaining -= len(coda)

        parts.append(onset + nucleus + coda)
        at_start = False

    return "".join(parts)


def _resolve_rng(rng: random.Random | int | None) -> random.Random:
    if rng is None:
        return random.Random()
    if isinstance(rng, random.Random):
        return rng
    return random.Random(rng)


def _acceptable(word: str) -> bool:
    return not _UGLY.search(word) and not _UNSAFE.search(word)


def _build_acceptable(length: int, rng: random.Random) -> str:
    for _ in range(1000):
        word = _build(length, rng)
        if _acceptable(word):
            return word
    # Only reachable if the filters were edited into rejecting nearly everything.
    raise RuntimeError(f"could not build an acceptable word of length {length}")


def random_syllabic_string(
    length: int = 6,
    *,
    rng: random.Random | int | None = None,
) -> str:
    """Return one pronounceable syllabic string of exactly ``length`` characters.

    Capitalised, ASCII, no spaces -- the same shape as the words in the curated
    lists, so the two mix without looking sorted.

    For more than one, and for uniqueness across a catalog, use
    :func:`~rfmux.resonator_names.syllabic_names` instead.

    :param length: the exact character count, at least :data:`MIN_LENGTH`.
    :param rng: a :class:`random.Random`, or an int seed, for reproducible draws.
    """
    if length < MIN_LENGTH:
        raise ValueError(f"length must be at least {MIN_LENGTH}, got {length}")
    return _build_acceptable(length, _resolve_rng(rng)).capitalize()
