"""Read and program a module's tones in one round trip each.

Periscope's Control mode refreshes from :func:`read_tones` once a second
and programs an edited field with :func:`write_tone`.  Both are plain
coroutines, so they run headlessly against a board or the mock.
"""

from typing import Iterable, Optional

FIELDS = ("frequency", "amplitude", "phase")


async def read_tones(crs, module: int, channels: Iterable[int]) -> dict:
    """The module's NCO and every listed channel's tone, in one batched
    call.

    Returns ``{"nco": Hz, "channels": {channel: {"frequency": Hz from the
    NCO, "amplitude": normalized, "phase": degrees of the DAC (carrier)
    phase}}}``.
    A value the board has never set reads as None.
    """
    channels = list(channels)
    async with crs.tuber_context() as ctx:
        ctx.get_nco_frequency(module=module)
        for ch in channels:
            ctx.get_frequency(channel=ch, module=module)
            ctx.get_amplitude(channel=ch, module=module)
            ctx.get_phase(units=crs.UNITS.DEGREES, target=crs.TARGET.DAC,
                          channel=ch, module=module)
        values = await ctx()
    tones = {}
    for i, ch in enumerate(channels):
        tones[ch] = dict(zip(FIELDS, values[1 + 3 * i:4 + 3 * i]))
    return {"nco": values[0], "channels": tones}


async def write_tone(crs, module: int, channel: int, *,
                     frequency: Optional[float] = None,
                     amplitude: Optional[float] = None,
                     phase: Optional[float] = None) -> dict:
    """Program the given fields of one channel (frequency in Hz from the
    NCO, amplitude normalized, phase in degrees) and return
    :func:`read_tones` for that channel, so the caller shows what the
    board kept rather than what was sent."""
    if any(v is not None for v in (frequency, amplitude, phase)):
        async with crs.tuber_context() as ctx:
            if frequency is not None:
                ctx.set_frequency(float(frequency), channel=channel,
                                  module=module)
            if amplitude is not None:
                ctx.set_amplitude(float(amplitude), channel=channel,
                                  module=module)
            if phase is not None:
                ctx.set_phase(float(phase), units=crs.UNITS.DEGREES,
                              target=crs.TARGET.DAC, channel=channel,
                              module=module)
            await ctx()
    return await read_tones(crs, module, [channel])
