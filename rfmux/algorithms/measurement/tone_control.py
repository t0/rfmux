"""Read and program a module's tones in one round trip each.

Periscope's Control mode refreshes from :func:`read_tones` once a second
and programs an edited field with :func:`write_tone`.  Both are plain
coroutines, so they run headlessly against a board or the mock.
"""

from typing import Iterable, Optional

from .bias_kids import DAC_SCALE_LABEL_OFFSET_DB

FIELDS = ("frequency", "amplitude", "dac_phase", "adc_phase")
PHASE_TARGET = {"dac_phase": "DAC", "adc_phase": "ADC"}


async def read_tones(crs, module: int, channels: Iterable[int]) -> dict:
    """The module's NCO, its labelled DAC scale (dBm, as Periscope
    labels amplitudes) and every listed channel's tone, in one batched
    call: ``{"nco": Hz, "dac_scale": dBm, "channels": {channel:
    {"frequency": Hz from the NCO, "amplitude": normalized, "dac_phase",
    "adc_phase": degrees}}}``.  A value the board has never set reads
    as None."""
    channels = list(channels)
    async with crs.tuber_context() as ctx:
        ctx.get_nco_frequency(module=module)
        ctx.get_dac_scale('DBM', module=module)
        for ch in channels:
            ctx.get_frequency(channel=ch, module=module)
            ctx.get_amplitude(channel=ch, module=module)
            for target in PHASE_TARGET.values():
                ctx.get_phase(units=crs.UNITS.DEGREES,
                              target=getattr(crs.TARGET, target),
                              channel=ch, module=module)
        values = await ctx()
    n = len(FIELDS)
    tones = {ch: dict(zip(FIELDS, values[2 + n * i:2 + n * (i + 1)]))
             for i, ch in enumerate(channels)}
    return {"nco": values[0], "dac_scale": values[1] - DAC_SCALE_LABEL_OFFSET_DB,
            "channels": tones}


async def write_tone(crs, module: int, channel: int, *,
                     frequency: Optional[float] = None,
                     amplitude: Optional[float] = None,
                     dac_phase: Optional[float] = None,
                     adc_phase: Optional[float] = None) -> dict:
    """Program the given fields of one channel (frequency in Hz from the
    NCO, amplitude normalized, phases in degrees) and return
    :func:`read_tones` for that channel, so the caller shows what the
    board kept rather than what was sent."""
    async with crs.tuber_context() as ctx:
        if frequency is not None:
            ctx.set_frequency(float(frequency), channel=channel, module=module)
        if amplitude is not None:
            ctx.set_amplitude(float(amplitude), channel=channel, module=module)
        for field, phase in (("dac_phase", dac_phase), ("adc_phase", adc_phase)):
            if phase is not None:
                ctx.set_phase(float(phase), units=crs.UNITS.DEGREES,
                              target=getattr(crs.TARGET, PHASE_TARGET[field]),
                              channel=channel, module=module)
        await ctx()
    return await read_tones(crs, module, [channel])
