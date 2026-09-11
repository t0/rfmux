"""Channel keys of a capture: an int for a channel of the file's one
module, a ``(module, channel)`` pair when a capture spans modules.

The names built from a key are the file's contract.  A one-module file
keeps ``channel_<n>`` groups and ``..._ch<n>`` datasets; a pair nests
as ``module_<M>/channel_<n>`` and ``..._m<M>ch<n>``.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple, Union

import numpy as np

ChannelKey = Union[int, Tuple[int, int]]


def channel_suffix(key: ChannelKey) -> str:
    """``ch5`` or ``m2ch5``: the tail of a per-channel dataset name."""
    if isinstance(key, tuple):
        return f"m{key[0]}ch{key[1]}"
    return f"ch{key}"


def channel_group(key: ChannelKey) -> str:
    """``channel_5`` or ``module_2/channel_5``: a channel's group path."""
    if isinstance(key, tuple):
        return f"module_{key[0]}/channel_{key[1]}"
    return f"channel_{key}"


def check_keys(keys: Iterable) -> List[ChannelKey]:
    """*keys* as ints or (module, channel) tuples, never a mix."""
    out: List[ChannelKey] = []
    for k in keys:
        if isinstance(k, (tuple, list, np.ndarray)):
            out.append((int(k[0]), int(k[1])))
        else:
            out.append(int(k))
    if len({isinstance(k, tuple) for k in out}) > 1:
        raise ValueError("channels must all be numbers or all "
                         "(module, channel) pairs")
    return out


def keys_from_attr(attr) -> List[ChannelKey]:
    """Keys from a file's ``channels`` attribute: a 1-D array of channel
    numbers or an (N, 2) array of (module, channel) rows."""
    arr = np.asarray(attr)
    if arr.ndim == 2:
        return [(int(m), int(c)) for m, c in arr]
    return [int(c) for c in arr.reshape(-1)]


def modules_of(keys: Iterable[ChannelKey]) -> List[int]:
    """The modules pair keys name, sorted; empty for int keys."""
    return sorted({k[0] for k in keys if isinstance(k, tuple)})
