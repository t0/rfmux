'''
Example run for 4 random channels 

channel = [13, 23, 37, 47]
result = await crs.py_run_pfb_streamer(channel = channel, module = 1, _bandwidth_derating = 0.8, time_run = 0.5) 

#### Plotting spectrum for each channel #####

for i in range(len(channel)):
    plt.plot(result.spectrum.freq_iq, result.spectrum.psd_i[i])
    plt.plot(result.spectrum.freq_iq, result.spectrum.psd_q[i])
    plt.show()
'''

import asyncio, inspect
import enum
import numpy as np
import socket
import time

from ...core.hardware_map import macro
from ...core.schema import CRS
from ...tuber.codecs import TuberResult
from ...core.transferfunctions import VOLTS_PER_ROC
from ... import streamer

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
async def py_run_pfb_streamer(crs : CRS,
                              channel : Union[None, int, List[int]] = None,
                              module : int = 1,
                              *, 
                              _bandwidth_derating: float = 0.5,
                              time_run : float = 10.0,
                              binlim: float = 1e6,
                              trim: bool = True,
                              nsegments: int = 100,
                              reference: str = "relative",
                              reset_NCO: bool = False):
    
    """
    Run the PFB streamer for one or more channels, capture time-domain I/Q data,
    and compute PFB-corrected spectra.

    Overview
    --------
    This macro:
      1. Enables PFB streaming on the specified CRS module and channel(s).
      2. Listens on the PFB multicast socket and captures packets for a fixed
         duration (`time_run`).
      3. Demultiplexes interleaved samples into per-channel complex time streams.
      4. Optionally recenters the channel using the module NCO (`reset_NCO`).
      5. Computes single- and dual-sideband power spectral densities using
         PFB droop correction.
      6. Disables PFB streaming before returning results.

    Parameters
    ----------
    crs : CRS
        Connected CRS control object.

    channel : None | int | list[int], default None
        Channel or channels to stream. If None, PFB streaming is disabled and
        the function returns without capturing data.

    module : int, default 1
        CRS module index (must be in the range 1–8).

    _bandwidth_derating : float, keyword-only
        Bandwidth derating factor passed to the PFB streamer configuration.
        Controls streaming bandwidth share between fast and slow. 

    time_run : float, keyword-only
        Duration (in seconds) to capture PFB packets.

    binlim : float, keyword-only
        Frequency span limit used during spectrum calculation.

    trim : bool, keyword-only
        Whether to trim edge bins during spectral processing.

    nsegments : int, keyword-only
        Number of segments used for PSD estimation.

    reference : str, keyword-only
        Scaling/reference mode for time-domain data and spectra
        (e.g. "relative" or "absolute").

    reset_NCO : bool, keyword-only
        If True, shifts the NCO and channel frequency so the signal is centered
        in a PFB bin before spectral analysis.

    Returns
    -------
    TuberResult
        A result object containing:
          - "i": time-domain I samples (per channel)
          - "q": time-domain Q samples (per channel)
          - "spectrum": a nested TuberResult with frequency axes and
            single- and dual-sideband PSDs for each channel.
    """
    
    assert module in range(1, 9), f"Invalid module {module}! Must be between 1 and 8"

    
    if channel is None:
        print(f"[Pfb streaming]: No channels specified. No Pfb streaming.")
        crs.set_pfb_streamer(channel = None, module = module)
        return
    
    #### Activate streaming ####
    x = await crs.set_pfb_streamer(channel = channel, module = module, _bandwidth_derating = _bandwidth_derating)

    #### rechecking if its active ####
    pfb_state = await crs.get_pfb_streamer(module = module)
    if pfb_state is not None:
        print(f"[Pfb streaming]: Active on channel {channel}")
    else:
        print(f"[Pfb streaming]: No active PFB streaming found")

    #### Opening the socket #####
    host = crs.tuber_hostname
    port = streamer.PFB_STREAMER_PORT
    
    channels = channel if isinstance(channel, list) else [channel]
    n_groups = len(channels)

    slot_lists = [[] for _ in range(n_groups)]
    slices = [slice(k, None, n_groups) for k in range(n_groups)]

    with streamer.get_multicast_socket(host, port=port) as sock:

        # To use asyncio, we need a non-blocking socket
        loop = asyncio.get_running_loop()
        sock.setblocking(False)
        packets = []

        async def receive_attempt():
            start = time.monotonic()
            # Start receiving packets
            while time.monotonic() - start < time_run:
                data = await asyncio.wait_for(
                    loop.sock_recv(sock, streamer.PFB_PACKET_SIZE),
                    streamer.STREAMER_TIMEOUT,
                )

                ##### Tried PFB packet receiver -> was getting empty queue - leaving it for debugging later #####
                ##### Plus this exactly mimics the py_get_samples flow, we can modfiy both codes to use receivers ###
    
                # Parse the received packet
                p = streamer.PFBPacket(data)
                packets.append(p)
                                
                samples = p.samples
                for lst, sl in zip(slot_lists, slices):
                    lst.extend(samples[sl])

                pfb_samps = [np.asarray(lst, dtype=np.complex128) for lst in slot_lists]                
            return sorted(packets, key=lambda p: p.seq), pfb_samps
            
        
        # Allow up to 10 packet-loss retries
        for attempt in range(NUM_ATTEMPTS := 10):
            packets, samples = await receive_attempt()

            sequence_steps = np.diff([p.seq for p in packets])
            if not all(np.diff([p.seq for p in packets]) == 1):
                warnings.warn(
                    f"Discontinuous packet capture! Attempt {attempt+1}/{NUM_ATTEMPTS}..."
                )
                continue
            # passed our tests - break out of the loop
            break
        else:
            raise RuntimeError(
                f"Failed to retrieve contiguous, consistent packet capture in {NUM_ATTEMPTS} attempts!"
            )

    print(f"[Pfb streaming] Shutting off the streaming for channel {channel}")
    await crs.set_pfb_streamer(channel = None, module = module)


    ###### Spectrum and TOD ######
    
    time_list_i = []
    time_list_q = []
    psd_list_i = []
    psd_list_q = []
    psd_list_dual = []
    
    for i in range(len(samples)):
        # Time-domain I/Q arrays
        time_i = samples[i].real
        time_q = samples[i].imag

        channel = channels[i]

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
            samples[i],
            nco_freq,
            ch_freq + nco_freq,
            binlim=binlim,
            trim=trim,
            nsegments=nsegments,
            reference=reference,
        )

        time_list_i.append(time_i.tolist())
        time_list_q.append(time_q.tolist())
        psd_list_i.append(psd_i.tolist())
        psd_list_q.append(psd_q.tolist())
        psd_list_dual.append(psd_dual_sideband.tolist())
        

    # Return results
    results = {
        "i": time_list_i,
        "q": time_list_q,
        "spectrum" : TuberResult({
            "freq_iq": freq_ssb.tolist(),
            "psd_i": psd_list_i,
            "psd_q": psd_list_q,
            "freq_dsb": freq_dsb.tolist(),
            "psd_dual_sideband": psd_list_q})
    }

    return TuberResult(results)