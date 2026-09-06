"""Names that are just a counter.

Sometimes a memorable name is the wrong answer -- a wafer being screened, a
figure that has to sort correctly, a colleague who wants to know which
resonator is which without learning a vocabulary. So::

    >>> boring_names(3)
    ['R0001', 'R0002', 'R0003']
    >>> boring_names(3, "S")
    ['S0001', 'S0002', 'S0003']
    >>> boring_names(3, "kid")
    ['kid0001', 'kid0002', 'kid0003']

The prefix is taken verbatim -- it is not capitalised or otherwise tidied, so
``kid`` stays lowercase -- and the counter is zero-padded to a fixed width so
that the names sort lexically in the order they were drawn, which is the whole
reason to number them in the first place.

This is a *mode* rather than a category: unlike a syllabic draw, the order
matters and nothing is drawn at random, so the two do not mix. ``R0003`` sitting
between ``BATHE`` and ``KOZRI`` would be neither one thing nor the other.
"""

from __future__ import annotations

from typing import Iterable

__all__ = [
    "DEFAULT_PREFIX",
    "DEFAULT_WIDTH",
    "boring_name",
    "boring_names",
]

#: What ``R0001`` starts with. ``R`` for resonator.
DEFAULT_PREFIX = "R"

#: How many digits the counter is padded to. Four covers any array anyone is
#: plausibly naming, and reads as a deliberate field width rather than as an
#: accident of how many there happened to be.
DEFAULT_WIDTH = 4


def _check_prefix(prefix: str) -> str:
    """Reject prefixes that would make the result ambiguous or unusable."""
    if any(character.isspace() for character in prefix):
        raise ValueError(f"prefix must not contain whitespace, got {prefix!r}")
    if prefix[-1:].isdigit():
        # ``R1`` + ``0001`` is ``R10001``, which is also ``R1`` + ``0001`` read
        # a different way -- and ``avoid`` below has to read it back.
        raise ValueError(f"prefix must not end in a digit, got {prefix!r}")
    return prefix


def boring_name(
    index: int,
    prefix: str = DEFAULT_PREFIX,
    *,
    width: int = DEFAULT_WIDTH,
) -> str:
    """Return the ``index``-th boring name.

        >>> boring_name(1)
        'R0001'
        >>> boring_name(42, "kid")
        'kid0042'

    :param index: the number to print, zero-padded. Counting normally starts at
        one, but any non-negative index works.
    :param prefix: what to put in front of the number, verbatim. Must not
        contain whitespace or end in a digit.
    :param width: the *minimum* number of digits. A larger index simply takes
        the digits it needs.
    """
    _check_prefix(prefix)
    if index < 0:
        raise ValueError(f"index must not be negative, got {index}")
    if width < 1:
        raise ValueError(f"width must be positive, got {width}")
    return f"{prefix}{index:0{width}d}"


def boring_names(
    n: int,
    prefix: str = DEFAULT_PREFIX,
    *,
    start: int = 1,
    width: int = DEFAULT_WIDTH,
    avoid: Iterable[str] = (),
) -> list[str]:
    """Return ``n`` numbered names, all of the same width.

        >>> boring_names(3)
        ['R0001', 'R0002', 'R0003']
        >>> boring_names(3, "S", start=7)
        ['S0007', 'S0008', 'S0009']
        >>> boring_names(3, avoid=["R0001", "R0002"])
        ['R0003', 'R0004', 'R0005']

    Uniform width is the invariant worth keeping -- it is what makes the names
    line up in a legend and sort correctly in a directory listing -- so if the
    count runs past ``width`` digits the whole batch widens together rather than
    part of it:

        >>> boring_names(3, start=9999)
        ['R09999', 'R10000', 'R10001']

    :param n: how many names to return.
    :param prefix: as for :func:`boring_name`.
    :param start: the first index to use.
    :param width: the *minimum* number of digits; see above.
    :param avoid: names already in use, e.g. those of resonators already named.
        A name is skipped when it matches an entry *exactly*, so a second batch
        carries on from where the first stopped without having to be told where
        that was. Nothing is parsed out of the entries: ``R1`` does not stand in
        for ``R0001``, and anything that is not a name this call would have
        produced simply never matches.
    """
    if n < 0:
        raise ValueError(f"n must not be negative, got {n}")
    if start < 0:
        raise ValueError(f"start must not be negative, got {start}")
    if width < 1:
        raise ValueError(f"width must be positive, got {width}")
    _check_prefix(prefix)

    avoided = frozenset(avoid)
    # Widening renames every candidate, so which of them ``avoid`` matches can
    # only be settled at the width they are finally printed at -- hence building
    # the batch and starting over if it turns out to need another digit. That
    # settles after a round or two: the width only ever grows, by at least one
    # digit each time, and the indices needed are bounded by ``start + n +
    # len(avoid)``.
    while True:
        names: list[str] = []
        index = start
        while len(names) < n:
            name = f"{prefix}{index:0{width}d}"
            if name not in avoided:
                names.append(name)
            index += 1
        if not names:
            return names
        needed = max(width, len(str(index - 1)))
        if needed == width:
            return names
        width = needed
