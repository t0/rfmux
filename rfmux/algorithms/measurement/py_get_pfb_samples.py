"""
py_get_pfb_samples: an experimental client-side wrapper around get_pfb_samples that implements
the spectral processing and embeds it within the return value. This should probably end up
getting ported to the server-side if possible.

CONSIDERATION: If so, we probably need a way to persistently update the VOLTS_PER_ROC.
Likely this will change or need to get more nuanced as we get better at our transfer functions.

This code retrieves time-domain samples from the PFB, applies droop correction, and returns
both single-sideband and dual-sideband PSD data in either dBc/Hz or dBm/Hz, depending on the
'reference' argument.
"""

import numpy as np
from ...core.hardware_map import macro
from ...core.schema import CRS
from ...core.transferfunctions import VOLTS_PER_ROC
from ...tuning.noise import apply_pfb_correction
from tuber.codecs import TuberResult


@macro(CRS, register=True)
async def py_get_pfb_samples(
    crs: CRS,
    nsamps: int,
    channel: int,
    module: int,
    *,
    binlim: float = 1e6,
    trim: bool = True,
    nsegments: int = 100,
    reference: str = "relative",
    reset_NCO: bool = False,
):
    """
    Acquire time-domain samples from the PFB, apply droop correction,
    and embed the result in the return value.

    This function:
      1) Retrieves pfb_samples from the hardware.
      2) Optionally resets the NCO to center the channel within a bin.
      3) Calls apply_pfb_correction to produce single-sideband and dual-sideband PSDs.
      4) Returns a TuberResult with time-domain arrays "i","q" and a "spectrum" dict
         containing "freq_iq","psd_i","psd_q","freq_dsb","psd_dual_sideband".

    Parameters
    ----------
    crs : CRS
        The CRS device instance.
    nsamps : int
        Number of time-domain samples to collect from the PFB. Max: 1e7
    channel : int
        Which readout channel to acquire from (1..1024).
    module : int
        Module index (1..8) from which to retrieve data.
    binlim : float, optional
        Frequency range (±) for droop correction. Default=1e6.
    trim : bool, optional
        If True, we trim the dual-sideband data as well to be symmetric around zero freq. Default True.
    nsegments : int, optional
        Number of segments to average in linear space. Default=100 => good compromise.
    reference : {'relative','absolute'}, optional
        If 'relative', final PSD data => dBc/Hz (DC bin is the carrier total power).
        If 'absolute', final PSD data => dBm/Hz.
    reset_NCO : bool, optional
        If True, shift the NCO so the channel is exactly at the bin center, measure,
        then restore the original frequencies. Helps get the entire bin bandwidth.

    Returns
    -------
    TuberResult
        A dictionary-like object containing:
          - "i","q": time-domain arrays (counts or volts),
          - "spectrum": sub-dict with "freq_iq","psd_i","psd_q","freq_dsb","psd_dual_sideband",
            containing single-sideband and dual-sideband PSD data in either dBc/Hz or dBm/Hz.

    Notes
    -----
    - This code accumulates in linear (Watts/Hz) across segments, matching
      how Welch accumulates the power for each segment, then does a final
      average. The final step converts to dBm/Hz or dBc/Hz.
    - By using sum(window^2) for U (rather than sum(window^2)/len(window)),
      and not dividing the FFT amplitude by len(segment), we align with the
      typical Hann-window logic in Welch's 'density' approach.
    - The required factor of 2 for single-sideband real signals is applied inside
      `separate_iq_fft_to_i_and_q_linear`, with an additional 1/2 on the DC
      bin to avoid double-counting.
    - If reference='relative' the data are referenced to the total carrier power.
      The DC bin is overwritten to be in power rather than density, such that
      for the dual-sideband data this will be exactly 0dB.
    """

    assert module in crs.module, (
        f"Module {module} invalid. Available: {crs.module}"
    )
    assert 1 <= channel <= 1024, f"Invalid channel: {channel}"

    if reset_NCO:
        # Shift the NCO so channel freq is at the bin center
        nco_freq_orig = await crs.get_nco_frequency(module=module)
        ch_freq_orig = await crs.get_frequency(channel=channel, module=module)

        bin_centers = (625e6 / 512.0) * np.arange(-256, 256)
        b_idx = np.abs(bin_centers - ch_freq_orig).argmin()
        bin_center_freq = bin_centers[b_idx]
        offset_in_bin = ch_freq_orig - bin_center_freq

        await crs.set_nco_frequency(nco_freq_orig + offset_in_bin, module=module)
        await crs.set_frequency(ch_freq_orig - offset_in_bin, channel=channel, module=module)

    # Retrieve final NCO and channel freq
    nco_freq = await crs.get_nco_frequency(module=module)
    ch_freq = await crs.get_frequency(channel=channel, module=module)

    # Grab PFB samples from hardware
    fastsamps = await crs.get_pfb_samples(nsamps, channel=channel, module=module)

    if reset_NCO:
        # restore original freq
        await crs.set_nco_frequency(nco_freq_orig, module=module)
        await crs.set_frequency(ch_freq_orig, channel=channel, module=module)

    # Build complex array from the time-domain I/Q
    pfb_samps = np.array(fastsamps.i, dtype=float) + 1j * np.array(fastsamps.q, dtype=float)

    # Time-domain I/Q arrays
    time_i = pfb_samps.real
    time_q = pfb_samps.imag

    # If reference='absolute', interpret time-domain in volts. Else keep ADC counts.
    if reference.lower() == "absolute":
        time_i = time_i * VOLTS_PER_ROC
        time_q = time_q * VOLTS_PER_ROC

    # Now apply the PFB droop correction => single/dual sideband PSD
    (
        freq_ssb,
        psd_i,
        psd_q,
        freq_dsb,
        psd_dual_sideband,
    ) = apply_pfb_correction(
        pfb_samps,
        nco_freq,
        ch_freq + nco_freq,
        binlim=binlim,
        trim=trim,
        nsegments=nsegments,
        reference=reference,
    )

    # Return results
    return TuberResult(
        i=time_i.tolist(),
        q=time_q.tolist(),
        spectrum=TuberResult(
            freq_iq=freq_ssb.tolist(),
            psd_i=psd_i.tolist(),
            psd_q=psd_q.tolist(),
            freq_dsb=freq_dsb.tolist(),
            psd_dual_sideband=psd_dual_sideband.tolist()))


# NOTE: Statistically, log binning of spectra is not advisable, since 
#       it buries the fact that each point now has a different statistical uncertainty
#       However, it can be visually helpful. Not sure where to put this little tool
#       so for now it lives here.
def logbin_spectrum(freq, psd_db, nbins=50):
    """
    Re-bin (freq, psd_db) data into log-spaced frequency bins, then compute 
    the average power in each bin. Note that the resulting points all now
    now different statistical uncertainties!

    Parameters
    ----------
    freq : ndarray
        Frequency array in Hz (must be > 0).
    psd_db : ndarray
        PSD in dB units (e.g., dBm/Hz).
    nbins : int
        Number of logarithmic bins to use.

    Returns
    -------
    freq_out : ndarray
        Log-bin center frequencies (geometric mean in each bin).
    psd_out : ndarray
        Average PSD in dB for each bin, using linear averaging.
    """
    # 1. Convert the dB values to linear (mW/Hz if psd_db is dBm/Hz)
    psd_lin = 10**(np.array(psd_db) / 10.0)  # from dB to "mW/Hz"
    # If psd_db was dBm/Hz, psd_lin is now in mW/Hz units.

    freq = np.array(freq)
    
    # 2. Determine log-spaced bins in frequency
    #    Make sure we ignore zero or negative freq if present.
    valid = freq > 0
    freq_valid = freq[valid]
    psd_lin_valid = psd_lin[valid]

    fmin = freq_valid.min()
    fmax = freq_valid.max()
    bin_edges = np.logspace(np.log10(fmin), np.log10(fmax), nbins + 1)

    freq_out = []
    psd_out = []

    # 3. For each bin, gather points & average
    for i in range(nbins):
        in_bin = (freq_valid >= bin_edges[i]) & (freq_valid < bin_edges[i+1])
        if not np.any(in_bin):
            continue

        freq_bin = freq_valid[in_bin]
        psd_bin_lin = psd_lin_valid[in_bin]

        # Geometric mean of the freq in that bin
        freq_gmean = np.exp(np.mean(np.log(freq_bin)))

        # Average PSD in linear, then convert to dB
        psd_mean_lin = np.mean(psd_bin_lin)
        psd_mean_db = 10.0 * np.log10(psd_mean_lin + 1e-30)

        freq_out.append(freq_gmean)
        psd_out.append(psd_mean_db)

    return np.array(freq_out), np.array(psd_out)
