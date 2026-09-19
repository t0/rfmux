"""Noise from configured tones, without applying a bias or owning a sender."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from numbers import Integral
from typing import Any

import numpy as np

from ... import streamer
from ...core.dac_scale import dac_scale_dbm
from ...core.hardware_map import macro
from ...core.resonators import ResonatorCatalog
from ...core.schema import CRS
from ...core.transferfunctions import (
    PFB_SAMPLING_FREQ, VOLTS_PER_ROC, decimation_to_sampling,
)
from ...tuning import store
from ...tuning.sweep_results import _packed


def _integer(value: int, name: str, low: int, high: int | None = None) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or value < low or (high is not None and value > high)):
        bounds = f"{low}..{high}" if high is not None else f">= {low}"
        raise ValueError(f"{name} must be an integer {bounds}.")
    return int(value)


def _segments(samples: int, segments: int, name: str) -> tuple[int, int]:
    # Symmetric Hann windows need >2 PFB samples and a non-DC I/Q bin.
    minimum = 4 if name == "PFB" else 2
    samples = _integer(samples, f"{name} samples", minimum)
    segments = _integer(segments, f"{name} nsegments", 1)
    if samples // segments < minimum:
        raise ValueError(f"{name} needs at least {minimum} samples per segment.")
    return samples, segments


def _record(data: Any, reference: str, index: int | None = None) -> dict:
    def values(value: Any) -> np.ndarray:
        return np.array(value if index is None else value[index], copy=True)

    counts = values(data.i) + 1j * values(data.q)
    if reference == "absolute":
        counts /= VOLTS_PER_ROC
    return {
        "iq_counts": counts,
        **{key: values(getattr(data.spectrum, key)) for key in
           ("psd_i", "psd_q", "psd_dual_sideband")},
    }


def _axes(data: Any) -> dict:
    return {key: np.array(getattr(data.spectrum, key), copy=True)
            for key in ("freq_iq", "freq_dsb")}


@macro(CRS, register=True)
async def take_noise_spectrum(
    crs: CRS,
    catalog: ResonatorCatalog | None = None,
    *,
    channels: Sequence[int] | None = None,
    module: int | None = None,
    decimation: int | None = None,
    num_samples: int = 10_000,
    nsegments: int = 10,
    reference: str = "absolute",
    spectrum_cutoff: float = 0.9,
    pfb_samples: int | None = None,
    pfb_nsegments: int | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    save: bool | None = None,
    label: str | None = None,
) -> dict:
    """Measure one module's configured tones; optionally save a noise container.

    Requires a running slow UDP stream and valid board timestamps. Default
    decimation preserves stream settings. A *different* explicit decimation
    selects this module alone, with short packets below stage 3 and long
    packets otherwise; that configuration remains in effect. No sender or
    tone programming is started, stopped or restored.

    PFB RPC captures follow the slow capture sequentially, without resetting
    the NCO. Progress receives {stream, channel, completed, total} after each
    capture (one slow capture plus one per PFB channel). Cancellation or an
    acquisition failure propagates without saving a partial measurement.

    Results contain shared slow timestamps/axes and named resonator records
    with complex readout counts and unchanged helper spectra. Absolute helper
    TOD is converted back from volts. Relative spectra retain the helpers'
    carrier-power bin exception, recorded in acquisition metadata.
    """
    requested_module = module
    if catalog is not None:
        if channels is not None:
            raise ValueError("Supply catalog or channels, not both.")
        if module is not None and module != catalog.module:
            raise ValueError("module must match the catalog's module.")
        module = catalog.module if module is None else module
        snapshot = deepcopy(catalog.to_dict())
        bindings = [(r.name, r.channel) for r in catalog]
    else:
        if channels is None:
            raise ValueError("Without a catalog, supply module and channels.")
        snapshot = None
        bindings = [(f"CH{c:04d}", c) for c in
                    [_integer(c, "channel", 1, streamer.LONG_PACKET_CHANNELS)
                     for c in channels]]
    module = _integer(module, "module", 1, 8)
    bindings = [(name, _integer(c, "channel", 1,
                              streamer.LONG_PACKET_CHANNELS))
                for name, c in bindings]
    if not bindings or len({c for _, c in bindings}) != len(bindings):
        raise ValueError("Channels must be nonempty and unique.")
    num_samples, nsegments = _segments(num_samples, nsegments, "slow")
    if decimation is not None:
        decimation = _integer(decimation, "decimation", 0, 6)
    if reference not in ("absolute", "relative"):
        raise ValueError("reference must be 'absolute' or 'relative'.")
    if not np.isfinite(spectrum_cutoff) or not 0 < spectrum_cutoff <= 1:
        raise ValueError("spectrum_cutoff must be finite and in (0, 1].")
    if pfb_samples is None and pfb_nsegments is not None:
        raise ValueError("pfb_nsegments requires pfb_samples.")
    requested_pfb_nsegments = pfb_nsegments
    if pfb_samples is not None:
        pfb_samples = _integer(pfb_samples, "pfb_samples", 4, 10_000_000)
        pfb_samples, pfb_nsegments = _segments(
            pfb_samples, nsegments if pfb_nsegments is None else pfb_nsegments,
            "PFB")
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable.")

    if module not in crs.modules.module:
        raise ValueError(f"Module {module} is unavailable on this board.")
    high_bank = await crs.get_analog_bank()
    if (module > 4) != bool(high_bank):
        raise ValueError("Module is outside the active analog bank.")
    current_decimation = await crs.get_decimation()
    if current_decimation is None:
        raise ValueError("Enable the slow stream before measuring noise.")
    stage = current_decimation if decimation is None else decimation
    stage = _integer(stage, "board decimation", 0, 6)
    if stage < 3 and max(c for _, c in bindings) > streamer.SHORT_PACKET_CHANNELS:
        raise ValueError("This decimation requires channels within short packets.")
    if pfb_samples is not None and await crs.get_pfb_streamer(module=module):
        raise ValueError("Disable the PFB UDP streamer before PFB RPC capture.")

    scale = await dac_scale_dbm(crs, module)
    nco = await crs.get_nco_frequency(module=module)
    async with crs.tuber_context() as ctx:
        for _, channel in bindings:
            ctx.get_frequency(channel=channel, module=module)
            ctx.get_amplitude(channel=channel, module=module)
        tones = await ctx()
    records = {}
    for (name, channel), frequency, amplitude in zip(
            bindings, tones[::2], tones[1::2]):
        records[name] = {
            "channel": channel,
            "tone_frequency_hz": (None if frequency is None or nco is None
                                  else float(nco + frequency)),
            "amplitude": None if amplitude is None else float(amplitude),
        }
    changed = stage != current_decimation
    if changed:
        await crs.set_decimation(stage, short=stage < 3, module=module)

    total = 1 + (len(bindings) if pfb_samples is not None else 0)

    def progress(stream: str, channel: int | None, completed: int) -> None:
        if progress_callback is not None:
            progress_callback(dict(stream=stream, channel=channel,
                                   completed=completed, total=total))

    slow = await crs.py_get_samples(
        num_samples=num_samples, module=module, channel=None,
        return_spectrum=True, scaling="psd", nsegments=nsegments,
        reference=reference, spectrum_cutoff=float(spectrum_cutoff))
    for name, channel in bindings:
        if channel > len(slow.i):
            raise ValueError(f"Channel {channel} is absent from slow packets.")
        records[name]["slow"] = _record(slow, reference, channel - 1)
    slow_axes = {"timestamps": [store.plain(vars(ts)) for ts in slow.ts],
                 **_axes(slow)}
    del slow
    progress("slow", None, 1)
    if pfb_samples is not None:
        for completed, (name, channel) in enumerate(bindings, start=2):
            pfb = await crs.py_get_pfb_samples(
                nsamps=pfb_samples, channel=channel, module=module,
                nsegments=pfb_nsegments, reference=reference,
                reset_NCO=False, binlim=1e6, trim=False)
            records[name]["pfb"] = {
                **_record(pfb, reference), **_axes(pfb),
                "time_s": np.arange(len(pfb.i)) / PFB_SAMPLING_FREQ,
            }
            progress("pfb", channel, completed)

    result = _packed(
        crs.module[module].index(), module,
        dict(catalog=snapshot, channel_names=dict(bindings),
             naming_rule=None if catalog is not None else "CH{channel:04d}",
             module=requested_module, decimation=decimation,
             num_samples=num_samples, nsegments=nsegments, reference=reference,
             spectrum_cutoff=float(spectrum_cutoff), pfb_samples=pfb_samples,
             pfb_nsegments=requested_pfb_nsegments),
        dict(acquisition=dict(
            decimation=stage, decimation_changed=changed,
            slow_sample_rate_hz=decimation_to_sampling(stage),
            pfb_sample_rate_hz=PFB_SAMPLING_FREQ if pfb_samples else None,
            pfb_nsegments=pfb_nsegments, pfb_binlim_hz=1e6 if pfb_samples else None,
            pfb_trim=False if pfb_samples else None,
            pfb_reset_nco=False if pfb_samples else None,
            reference=reference, iq_units="adc_counts", frequency_units="Hz",
            spectrum_units="dBm/Hz" if reference == "absolute" else "dBc/Hz",
            carrier_bin_units="dBm/Hz" if reference == "absolute" else "dBc",
            carrier_bin_rule="nearest zero frequency in each spectrum",
        ), slow=slow_axes, resonators=records),
        measurement="noise", dac_scale_dbm=scale)
    store.maybe_save(result, "noise", save=save, label=label)
    return result
