"""
Resolving "which channels should I operate on" against the board.

Callers routinely want every channel that is actually carrying a tone
rather than a list they typed by hand.  Reading that back is a
per-channel RPC, so the naive loop is one round trip per channel --
1024 of them for a long-packet module.  :func:`get_biased_channels`
batches the reads into a single tuber context instead.

Registered as a macro, so it is available as::

    channels = await crs.get_biased_channels(module=1)
"""

from __future__ import annotations

from typing import List, Optional

from ...core.hardware_map import macro
from ...core.schema import CRS
from ... import streamer


@macro(CRS, register=True)
async def get_biased_channels(
    crs,
    module: int,
    *,
    max_channels: Optional[int] = None,
    threshold: float = 0.0,
) -> List[int]:
    """1-indexed channels on ``module`` whose bias amplitude is nonzero.

    ``max_channels`` bounds the search; it defaults to the long-packet
    width.  Pass the active slow-stream packet width when the result
    feeds something that reads the stream -- a channel above that width
    is carried by no packet, so it cannot be captured however it is
    biased.

    ``threshold`` is compared against ``abs(amplitude)``, so the default
    of 0.0 keeps any channel that has been given a tone at all.
    """
    if max_channels is None:
        max_channels = streamer.LONG_PACKET_CHANNELS
    max_channels = int(max_channels)
    if max_channels < 1:
        return []

    module = int(module)
    async with crs.tuber_context() as ctx:
        for channel in range(1, max_channels + 1):
            ctx.get_amplitude(channel=channel, module=module)
        amplitudes = await ctx()

    biased: List[int] = []
    for channel, amplitude in enumerate(amplitudes, start=1):
        if amplitude is None:
            continue
        try:
            if abs(float(amplitude)) > threshold:
                biased.append(channel)
        except (TypeError, ValueError):
            # A board that answers with something non-numeric is not a
            # biased channel; skip it rather than failing the whole scan.
            continue
    return biased
