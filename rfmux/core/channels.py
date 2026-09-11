"""Channel and module spellings shared by the command lines and the
dialogs: ranges like ``1,5-8,20`` and per-module ranges like
``2:1-114,3:1-96``."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

#: Spellings of the "every biased channel" wildcard.
ALL_CHANNELS_TOKENS = ("all", "*")
#: Readout modules a board has, numbered 1-4 everywhere in Python
#: (0-3 on the wire: NUM_MODULES in the streamer's packet.h).
MAX_MODULE = 4


def parse_channel_spec(text: str, *, name: str = "channel",
                       max_value: Optional[int] = None,
                       wildcard: bool = True) -> Optional[List[int]]:
    """Parse a channel spec into a sorted, de-duplicated channel list.

    Accepts single channels and inclusive ranges, in any mix::

        "1,2"        -> [1, 2]
        "2-19"       -> [2, 3, ..., 19]
        "1,5-8,20"   -> [1, 5, 6, 7, 8, 20]

    Returns ``None`` for the wildcard (``all`` / ``*``) where *wildcard*
    is allowed, which the caller resolves against the board (see
    ``crs.get_biased_channels``).  Whitespace is ignored anywhere.
    *max_value* bounds the numbers and *name* is what messages call
    them.

    Raises ValueError with a message naming the offending token, since
    the immediate caller is a GUI field showing it back to a human.
    """
    plural = f"{name}s"
    cleaned = "".join(text.split())
    if not cleaned:
        raise ValueError(f"No {plural} given.")
    if cleaned.lower() in ALL_CHANNELS_TOKENS:
        if wildcard:
            return None
        raise ValueError(f"Name the {plural}: {cleaned!r} is not a list.")

    values = set()
    for token in cleaned.split(","):
        if not token:
            continue  # tolerate "1,,2" and a trailing comma
        lo, sep, hi = token.partition("-")
        try:
            start = int(lo)
            stop = int(hi) if sep else start
        except ValueError:
            raise ValueError(
                f"Could not read {token!r}. Use {name} numbers like "
                f"\"1,2\", ranges like \"2-19\""
                + (", or \"all\"." if wildcard else ".")) from None
        if start < 1 or stop < 1:
            raise ValueError(
                f"{plural.capitalize()} are 1-indexed, so {token!r} is out "
                "of range.")
        if stop < start:
            raise ValueError(
                f"Range {token!r} runs backwards -- write "
                f"\"{stop}-{start}\".")
        if max_value is not None and stop > max_value:
            raise ValueError(
                f"{plural.capitalize()} run 1-{max_value}, so {token!r} is "
                "out of range.")
        values.update(range(start, stop + 1))
    if not values:
        raise ValueError(f"No {plural} given.")
    return sorted(values)


def parse_module_channels(spec: Union[str, List[str]], *,
                          max_module: Optional[int] = None,
                          max_channel: Optional[int] = None,
                          ) -> Dict[int, List[int]]:
    """``"2:1-114,3:1-96"`` -> ``{2: [1, ..., 114], 3: [1, ..., 96]}``.

    A module prefix holds until the next one, so ``2:1-10,20-30,3:1``
    gives module 2 both ranges.  *spec* may also be a list of such
    strings (a repeated command-line option).  Every channel needs a
    module: a spec without one is refused, so a caller tells the two
    grammars apart by the colon.
    """
    if not isinstance(spec, str):
        spec = ",".join(spec)
    cleaned = "".join(spec.split())
    result: Dict[int, set] = {}
    module = None
    for token in cleaned.split(","):
        if not token:
            continue
        prefix, colon, channels = token.rpartition(":")
        if colon:
            try:
                module = int(prefix)
            except ValueError:
                raise ValueError(
                    f"Could not read the module in {token!r}. Write "
                    "MODULE:CHANNELS, like \"2:1-114,3:1-96\".") from None
            if module < 1 or (max_module is not None
                              and module > max_module):
                raise ValueError(
                    f"Modules run 1-{max_module}, so {token!r} is out of "
                    "range." if max_module is not None else
                    f"Modules are 1-indexed, so {token!r} is out of range.")
        if module is None:
            raise ValueError(
                f"{token!r} has no module. Write MODULE:CHANNELS, like "
                "\"2:1-114,3:1-96\".")
        result.setdefault(module, set()).update(parse_channel_spec(
            channels, max_value=max_channel, wildcard=False))
    if not result:
        raise ValueError("No channels given.")
    return {m: sorted(chs) for m, chs in sorted(result.items())}
