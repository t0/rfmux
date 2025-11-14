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
from rfmux.core.transferfunctions import VOLTS_PER_ROC
from rfmux.core.hardware_map import macro
from rfmux.core.schema import CRS
from rfmux.tuber.codecs import TuberResult
from scipy.signal.windows import chebwin
from scipy.interpolate import interp1d


def separate_iq_fft_to_i_and_q_linear(freqs, iqfft, fs, U):
    """
    Reconstruct I- and Q-only FFTs from a complex (IQ) FFT, returning *linear*
    PSD values (Watts/Hz) for each bin. This function is part of a larger
    pipeline that matches the Hann-window normalization used in SciPy's Welch.

    How the normalization works:
      - We define U = sum(window^2), NOT sum(window^2)/N.
      - The factor of 2 below accounts for real-signal reconstruction:
         * I and Q each get a factor of 2 in power for single-sideband.
      - We then divide by (50 * U * fs) at the last stage of the pipeline
        (though part of that division happens externally in apply_pfb_correction
        for the dual-sideband portion).

    Parameters
    ----------
    freqs : ndarray
        Frequency array (unshifted 0..fs layout) used only to determine
        the maximum frequency for psdfreq.
    iqfft : ndarray
        Complex FFT array (length N) of I+jQ data, presumably after windowing
        and optional gain factors. Typically shape (N,).
    fs : float
        Sampling frequency in Hz for the final PSD (e.g., ~2.44 MHz in CRS).
    U : float
        Window power normalization factor = sum(window^2). *Not* divided
        by len(window), so it aligns with Welch's approach.

    Returns
    -------
    psdfreq : ndarray
        Single-sideband frequency axis from 0..(max(freqs)) for N/2 bins.
    re_ps_lin : ndarray
        I-channel PSD (Watts/Hz).
    im_ps_lin : ndarray
        Q-channel PSD (Watts/Hz).

    Notes
    -----
    - We use a factor of 2 for each real channel (I or Q). For real-signal
      single-sideband PSD, this is typical.
    - The DC bin is halved again (re_ps[0] /= 2, im_ps[0] /= 2), to avoid
      double-counting at 0 Hz.
    - The final scaling to dBm/Hz or dBc/Hz (and any referencing) happens
      outside, in apply_pfb_correction.
    """

    N = len(iqfft)

    # Handle DC separately (it's purely real for real signals)
    dc_i = iqfft[0].real
    dc_q = iqfft[0].imag
    
    # For k from 1 to N//2-1, we need indices N-k
    # This gives us [N-1, N-2, ..., N//2+1]
    k_indices = np.arange(1, N // 2)
    neg_k_indices = N - k_indices  # This gives [N-1, N-2, ..., N//2+1]
    
    # Extract positive and negative frequency components
    z_pos = iqfft[k_indices]
    z_neg = iqfft[neg_k_indices]
    
    # Reconstruct I and Q using vectorized operations
    # For I: real parts add, imaginary parts subtract
    ibatch = 0.5 * (z_pos + np.conj(z_neg))
    
    # For Q: we need to be careful with the formula
    # Q[k] = 0.5 * ((z_pos - conj(z_neg)) / j)
    # which is: 0.5 * j * (conj(z_neg) - z_pos)
    qbatch = 0.5j * (np.conj(z_neg) - z_pos)
    
    # Prepend DC values
    ibatch = np.concatenate([[dc_i], ibatch])
    qbatch = np.concatenate([[dc_q], qbatch])
    

    # Factor of 2 for real signals, dividing out 50 ohms * U
    re_ps = 2 * (np.abs(ibatch) ** 2) / (50.0 * U)
    im_ps = 2 * (np.abs(qbatch) ** 2) / (50.0 * U)

    # DC bin fix for real-signal reconstruction
    re_ps[0] /= 2.0
    im_ps[0] /= 2.0

    # Here we define rbw = fs, so re_ps / rbw => power spectral density in W/Hz
    rbw = fs
    re_ps_lin = re_ps / rbw
    im_ps_lin = im_ps / rbw

    psdfreq = np.linspace(0, max(freqs), N // 2, endpoint=False)
    return psdfreq, re_ps_lin, im_ps_lin


def apply_pfb_correction(
    pfb_samples,
    nco_freq,
    channel_freq,
    binlim=1.1e6,
    trim=True,
    nsegments=1,
    reference="relative",
):
    """
    Apply droop correction for a polyphase filter bank (PFB), returning both
    single-sideband (I,Q) and dual-sideband PSDs in either dBm/Hz or dBc/Hz.

    We accumulate in *linear* units (Watts/Hz) across 'nsegments', then
    convert to the final dB scale. This approach aligns with Hann-window
    normalization in Welch:

      U = sum(window^2), no dividing by len(window).
      We do not divide the FFT amplitude by len(segment).
      The final PSD is effectively power / (U * fs).

    If reference='relative', we treat the DC bin in the dual-sideband data
    as the "carrier" and subtract it in dB, producing dBc/Hz. If 'absolute',
    we keep dBm/Hz.

    Parameters
    ----------
    pfb_samples : ndarray
        Complex time-domain samples from the PFB (ADC counts).
    nco_freq : float
        NCO frequency in Hz, used to locate the bin center offset.
    channel_freq : float
        Channel frequency in Hz (NCO freq + channel offset).
    binlim : float, optional
        ± frequency range for droop correction / trimming. Default 1.1e6.
    trim : bool, optional
        If True, narrows the dual-sideband data around zero. Default True.
    nsegments : int, optional
        Number of segments to average in linear space. Default 1 => use all
        samples in one segment.
    reference : str, optional
        'relative' => produce dBc/Hz, referencing the DC bin in dual sideband.
        'absolute' => produce dBm/Hz.

    Returns
    -------
    psdfreq_ssb : ndarray
        Single-sideband frequency axis (0..binlim), possibly trimmed.
    re_psd : ndarray
        I-channel PSD in dBm/Hz or dBc/Hz.
    im_psd : ndarray
        Q-channel PSD in dBm/Hz or dBc/Hz.
    ds_freq : ndarray
        Dual-sideband frequency axis, possibly trimmed around zero if trim=True.
    ds_psd : ndarray
        Dual-sideband PSD in dBm/Hz or dBc/Hz.

    Notes
    -----
    - We build a Hann window per segment, compute an FFT, and apply the
      built-in PFB gain at bin center + droop correction from the PFB filter.
    - The factor-of-2 for single-sideband real signals is handled inside
      `separate_iq_fft_to_i_and_q_linear`, with an extra 1/2 for the DC bin.
    - For 'relative', we sum the DC bin in the dual-sideband data across
      segments to find the "carrier" in linear, then do final referencing.
    """

    comb_sampling_freq = 625e6
    fs = comb_sampling_freq / 256.0

    # Convert ADC counts to volts for droop correction.
    pfb_samples_volts = pfb_samples * VOLTS_PER_ROC

    # Build PFB droop function via chebwin taps, large FFT, and shift
    NTAPS = 4096
    w_taps = chebwin(NTAPS, 103)
    W_full = np.fft.fft(w_taps, 262144)
    W_full /= np.abs(W_full[0]) + 1e-30
    W_full = np.fft.fftshift(W_full)
    pfb_freqs = np.linspace(-comb_sampling_freq, comb_sampling_freq, len(W_full))
    pfb_func = interp1d(pfb_freqs, W_full, bounds_error=True)

    seg_len = len(pfb_samples_volts) // nsegments

    # Linear accumulators for single-sideband (I/Q) and dual-sideband
    re_ps_lin_accum = None
    im_ps_lin_accum = None
    ssb_freq_final = None

    ds_ps_lin_accum = None
    ds_freq_final = None

    # For dBc mode: accumulate the DC bin amplitude in linear units
    carrier_lin_accum = 0.0
    carrier_count = 0

    def _trim_around_zero(freq, data):
        """
        Helper to trim data around zero frequency. We find zero_idx and keep
        symmetric +/- range. Used if trim=True for the dual-sideband.
        """
        zero_idx = np.argmin(np.abs(freq))
        half_len = min(zero_idx, len(freq) - zero_idx)
        return (
            freq[zero_idx - half_len : zero_idx + half_len],
            data[zero_idx - half_len : zero_idx + half_len],
        )

    def _accumulate_linear(accum, freq_accum, new_lin, new_freq):
        """
        Accumulates new_lin into accum in-place, ensuring consistent length min.
        freq_accum is not strictly updated here except to keep shape alignment
        if needed. We do a simple accumulation for multi-segment averaging.
        """
        if accum is None:
            return new_lin, new_freq
        m_len = min(len(accum), len(new_lin))
        accum[:m_len] += new_lin[:m_len]
        return accum, freq_accum

    # Figure out bin centers. The channel offset inside that bin is channel_freq_in_nco_bw
    bin_centers = (comb_sampling_freq / 512.0) * np.arange(-256, 256)
    channel_freq_in_nco_bw = channel_freq - nco_freq

    for seg in range(nsegments):
        # 1) Slice out time segment and apply Hann window
        segment = pfb_samples_volts[seg * seg_len : (seg + 1) * seg_len]
        window = np.hanning(len(segment))
        U = np.sum(window**2)  # sum(window^2) => consistent with Welch

        # 2) SHIFTed FFT: we do np.fft.fft(window*segment), then fftshift
        freqs_shifted = np.fft.fftshift(np.fft.fftfreq(len(segment), d=1.0 / fs))
        fft_shifted = np.fft.fftshift(np.fft.fft(window * segment))

        # 3) Locate bin center offset
        b_idx = np.abs(bin_centers - channel_freq_in_nco_bw).argmin()
        bin_center = bin_centers[b_idx]
        channel_offset_in_bin = channel_freq_in_nco_bw - bin_center

        freqs_in_bin = freqs_shifted + channel_offset_in_bin

        # 4) Apply built-in PFB gain at bin center, droop-correct only up to ±binlim
        built_in_gain = pfb_func(channel_offset_in_bin)
        fft_shifted *= built_in_gain

        valid_idx = np.abs(freqs_in_bin) <= binlim
        freq_corr = freqs_in_bin[valid_idx]
        data_corr = fft_shifted[valid_idx]

        droop = pfb_func(freq_corr) + 1e-30
        specdata_corrected = data_corr / droop

        # 5) Build dual-sideband data in linear. 
        ds_freq_untrimmed = freq_corr - channel_offset_in_bin
        ds_freq_local = ds_freq_untrimmed
        ds_spec_local = specdata_corrected

        # Optionally trim around 0 if trim=True
        if trim:
            ds_freq_local, ds_spec_local = _trim_around_zero(ds_freq_local, ds_spec_local)

        # PSD => ~ (|FFT|^2) / (50*U*fs)
        ds_ps_lin_local = (np.abs(ds_spec_local) ** 2) / (50.0 * U * fs)

        if reference.lower()=='relative' and len(ds_freq_local) > 0:
            # Record the DC bin amplitude for later normalization in linear
            zero_idx = np.argmin(np.abs(ds_freq_local))
            # The DC bin is infinitely narrow => multiply by bin_width to get total power
            bin_width = fs / len(segment)
            carrier_lin_accum += ds_ps_lin_local[zero_idx] * bin_width
            carrier_count += 1

            # Overwrite DC bin with total power => ds_ps_lin_local[zero_idx] * bin_width
            ds_ps_lin_local[zero_idx] *= bin_width

        ds_ps_lin_accum, ds_freq_final = _accumulate_linear(
            ds_ps_lin_accum, ds_freq_final, ds_ps_lin_local, ds_freq_local
        )

        # 6) Single-sideband logic => reconstruct I/Q in separate_iq_fft_to_i_and_q_linear
        fft_corr_shifted = np.zeros_like(fft_shifted, dtype=complex)
        fft_corr_shifted[valid_idx] = specdata_corrected

        fft_corr_unshifted = np.fft.ifftshift(fft_corr_shifted)
        freqs_unshifted = np.fft.fftfreq(len(segment), d=1.0 / fs)

        # separate_iq => returns freq_ssb, re_ps_lin, im_ps_lin in W/Hz
        ssb_freq_local, re_ps_lin_local, im_ps_lin_local = separate_iq_fft_to_i_and_q_linear(
            freqs_unshifted, fft_corr_unshifted, fs, U
        )

        if reference.lower()=='relative':
            # Overwrite the DC bin in single-sideband as well
            re_ps_lin_local[0] *= (fs / len(segment))
            im_ps_lin_local[0] *= (fs / len(segment))

        # clip freq to binlim
        offset_hz = abs(channel_offset_in_bin)
        ssb_clip = max(0, binlim - offset_hz)
        bin_width = fs / len(ssb_freq_local)
        epsilon = 0.8 * bin_width
        freq_mask = ssb_freq_local <= (ssb_clip - epsilon)

        ssb_freq_local = ssb_freq_local[freq_mask]
        re_ps_lin_local = re_ps_lin_local[freq_mask]
        im_ps_lin_local = im_ps_lin_local[freq_mask]

        # Accumulate single-sideband PSDs
        if re_ps_lin_accum is None:
            re_ps_lin_accum = re_ps_lin_local
            im_ps_lin_accum = im_ps_lin_local
            ssb_freq_final = ssb_freq_local
        else:
            m_len = min(len(re_ps_lin_accum), len(re_ps_lin_local))
            re_ps_lin_accum[:m_len] += re_ps_lin_local[:m_len]
            im_ps_lin_accum[:m_len] += im_ps_lin_local[:m_len]

    # 7) Average in linear
    re_ps_lin_accum /= nsegments
    im_ps_lin_accum /= nsegments
    ds_ps_lin_accum /= nsegments

    # Summed carrier across segments => average
    carrier_lin = carrier_lin_accum / max(carrier_count, 1)

    # 8) Convert final linear accumulators to dBm/Hz or dBc/Hz
    re_ps_dbm = 10.0 * np.log10(re_ps_lin_accum / 1e-3 + 1e-30)
    im_ps_dbm = 10.0 * np.log10(im_ps_lin_accum / 1e-3 + 1e-30)
    ds_ps_dbm = 10.0 * np.log10(ds_ps_lin_accum / 1e-3 + 1e-30)

    if reference.lower() == "absolute":
        # => keep dBm/Hz directly
        final_re_psd = re_ps_dbm
        final_im_psd = im_ps_dbm
        final_ds_psd = ds_ps_dbm
    else:
        # => relative => dBc/Hz. Use DS DC bin as unified carrier reference
        carrier_dbm = 10.0 * np.log10(carrier_lin / 1e-3 + 1e-30)

        final_re_psd = re_ps_dbm - carrier_dbm
        final_im_psd = im_ps_dbm - carrier_dbm
        final_ds_psd = ds_ps_dbm - carrier_dbm

    return (
        ssb_freq_final,   # single-sideband freq axis
        final_re_psd,     # I-PSD in dBm/Hz or dBc/Hz
        final_im_psd,     # Q-PSD in dBm/Hz or dBc/Hz
        ds_freq_final,    # dual-sideband freq axis
        final_ds_psd      # dual-sideband PSD in dBm/Hz or dBc/Hz
    )


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

    assert module in crs.modules.module, (
        f"Module {module} invalid. Available: {crs.modules.module}"
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
    results = {
        "i": time_i.tolist(),
        "q": time_q.tolist(),
        "spectrum" : TuberResult({
            "freq_iq": freq_ssb.tolist(),
            "psd_i": psd_i.tolist(),
            "psd_q": psd_q.tolist(),
            "freq_dsb": freq_dsb.tolist(),
            "psd_dual_sideband": psd_dual_sideband.tolist()})
    }

    return TuberResult(results)


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