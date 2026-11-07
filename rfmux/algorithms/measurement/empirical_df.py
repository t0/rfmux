"""Calibrate the tones a module is already playing, from one sweep each.

For a board whose tones were set by something other than bias finding --
the simulator biasing its own array, or tones set by hand -- so that their
detectors can be read in hertz.
"""

from __future__ import annotations

from ...core.hardware_map import macro
from ...core.resonators import BiasPoint, Resonator, ResonatorCatalog
from ...core.schema import CRS
from ...resonator_names import syllabic_names_from_frequency
from ...tuning.bias import calibrated_at
from .channel_selection import get_biased_channels


@macro(CRS, register=True)
async def empirically_calibrate_df(
    crs,
    module: int,
    *,
    span_hz: float = 20e3,
    npoints_per_sweep: int = 41,
    nsamps: int = 10,
    save: bool | None = None,
    progress_callback=None,
) -> ResonatorCatalog:
    """A catalog of the tones *module* is playing, each calibrated.

    Every biased channel becomes a resonator at its tone's frequency and
    amplitude, named from its frequency. One upward sweep of ``span_hz``
    around each tone, at its own amplitude, gives the bias point its IQ slope
    at the tone (and so ``df_calibration``), the sweep and its fits, and the
    power it was measured at. ``crs.apply_bias`` then puts the tones back,
    saving the catalog as a ``bias_catalog`` file when *save* says so (None
    follows the autosave setting).

    Returns an empty catalog when no channel is biased.
    """
    channels = await get_biased_channels(crs, module)
    if not channels:
        return ResonatorCatalog([], module=module)
    nco = await crs.get_nco_frequency(module=module)
    async with crs.tuber_context() as ctx:
        for channel in channels:
            ctx.get_frequency(channel=channel, module=module)
            ctx.get_amplitude(channel=channel, module=module)
        values = await ctx()
    tones = [(channel, nco + values[2 * i], values[2 * i + 1])
             for i, channel in enumerate(channels)]
    names = syllabic_names_from_frequency([f for _, f, _ in tones])
    # The frequencies as they are playing, not moved onto the tone grid.
    catalog = ResonatorCatalog(
        [Resonator(name=name, channel=channel,
                   bias=BiasPoint(frequency_hz=f, amplitude=a,
                                  bias_frequency_quantized=False))
         for name, (channel, f, a) in zip(names, tones)],
        module=module)

    sweeps = await crs.multisweep(
        catalog, span_hz=span_hz, npoints_per_sweep=npoints_per_sweep,
        nsamps=nsamps, save=False, progress_callback=progress_callback)
    swept = sweeps[crs.module[module].index()]["results"][0]["upward"]
    for r in catalog:
        r.bias = calibrated_at(swept[r.name], r.bias)

    await crs.apply_bias(catalog, save=save, label="empirical_df")
    return catalog
