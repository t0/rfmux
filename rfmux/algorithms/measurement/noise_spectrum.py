"""
take_noise_spectrum: noise spectra of a module's channels from the slow
stream at a chosen decimation and, optionally, from the PFB stream.
"""

import numpy as np

from ...core.hardware_map import macro
from ...core.schema import CRS
from ...core.transferfunctions import PFB_SAMPLING_FREQ


@macro(CRS, register=True)
async def take_noise_spectrum(
    crs: CRS,
    channels: list,
    decimation: int = 6,
    num_samples: int = 10000,
    num_segments: int = 10,
    reference: str = "relative",
    spectrum_limit: float = 0.9,
    pfb_samples: int = None,
    *,
    module,
):
    """
    Noise spectra of *channels* on *module*.

    Sets the slow stream to *decimation* if it is not there already (short
    packets below stage 4, where long packets exceed the link), takes
    *num_samples* of every channel with the Welch spectrum, and reads each
    channel's amplitude and frequency. With *pfb_samples* set, also takes
    that many PFB samples of each channel in turn.

    Parameters
    ----------
    channels : list of int
        Readout channels (1-based) to report.
    decimation : int
        Slow-stream decimation stage, 0-6.
    num_samples, num_segments : int
        Samples to take and Welch segments to average over.
    reference : {'relative', 'absolute'}
        dBc/Hz relative to the carrier, or dBm/Hz.
    spectrum_limit : float
        Fraction of Nyquist the spectrum is kept up to.
    pfb_samples : int or None
        PFB samples per channel; None leaves the PFB stream alone.

    Returns
    -------
    dict
        Per-channel lists in the order of *channels* (``I``, ``Q``,
        ``single_psd_i``, ``single_psd_q``, ``dual_psd``, ``amplitudes``,
        ``channel_frequencies`` in Hz), the shared axes (``ts``,
        ``freq_iq``, ``freq_dsb``), ``slow_freq_hz`` and ``fast_freq_hz``,
        and ``pfb_enabled`` with the matching ``pfb_*`` entries when set.
    """
    channels = [int(c) for c in channels]
    if not channels:
        raise ValueError("channels must name at least one channel.")
    if not 0 <= decimation <= 6:
        raise ValueError(f"decimation must be 0-6 (got {decimation}).")
    if num_segments < 1 or num_samples < 2 * num_segments:
        raise ValueError(f"num_samples ({num_samples}) must cover at least two points "
                         f"per Welch segment ({num_segments} segments).")
    if reference not in ("relative", "absolute"):
        raise ValueError(f"reference must be 'relative' or 'absolute' (got {reference!r}).")
    if not 0 < spectrum_limit <= 1:
        raise ValueError(f"spectrum_limit must be in (0, 1] (got {spectrum_limit:g}).")
    if pfb_samples is not None and pfb_samples < 2 * num_segments:
        raise ValueError(f"pfb_samples ({pfb_samples}) must cover at least two points per segment.")

    if await crs.get_decimation() != decimation:
        if decimation > 4:
            await crs.set_decimation(decimation, short=False)
        else:
            await crs.set_decimation(decimation, module=module, short=decimation < 4)

    slow = await crs.py_get_samples(num_samples, return_spectrum=True, scaling="psd",
                                    reference=reference, nsegments=num_segments,
                                    spectrum_cutoff=spectrum_limit, channel=None, module=module)

    nco = await crs.get_nco_frequency(module=module)
    amplitudes, frequencies = [], []
    for c in channels:
        amp = await crs.get_amplitude(channel=c, module=module)
        offset = await crs.get_frequency(channel=c, module=module)
        amplitudes.append(0 if amp is None else amp)
        frequencies.append(float(nco + (offset or 0)))

    def pick(per_channel):
        return [per_channel[c - 1] for c in channels]

    data = {
        "reference": reference,
        "ts": slow.ts,
        "I": pick(slow.i),
        "Q": pick(slow.q),
        "freq_iq": slow.spectrum.freq_iq,
        "single_psd_i": pick(slow.spectrum.psd_i),
        "single_psd_q": pick(slow.spectrum.psd_q),
        "freq_dsb": slow.spectrum.freq_dsb,
        "dual_psd": pick(slow.spectrum.psd_dual_sideband),
        "amplitudes": amplitudes,
        "channel_frequencies": frequencies,
        "slow_freq_hz": max(slow.spectrum.freq_iq) / spectrum_limit,
        "fast_freq_hz": PFB_SAMPLING_FREQ / 2,
        "pfb_enabled": pfb_samples is not None,
    }

    if pfb_samples is not None:
        pfb = []
        for c in channels:
            pfb.append(await crs.py_get_pfb_samples(
                pfb_samples, channel=c, module=module, binlim=1e6, trim=False,
                nsegments=num_segments, reference=reference, reset_NCO=False))
        data.update({
            "pfb_ts": list(np.linspace(0, pfb_samples / PFB_SAMPLING_FREQ, pfb_samples)),
            "pfb_I": [p.i for p in pfb],
            "pfb_Q": [p.q for p in pfb],
            "pfb_freq_iq": [p.spectrum.freq_iq for p in pfb],
            "pfb_psd_i": [p.spectrum.psd_i for p in pfb],
            "pfb_psd_q": [p.spectrum.psd_q for p in pfb],
            "pfb_freq_dsb": [p.spectrum.freq_dsb for p in pfb],
            "pfb_dual_psd": [p.spectrum.psd_dual_sideband for p in pfb],
        })
    return data
