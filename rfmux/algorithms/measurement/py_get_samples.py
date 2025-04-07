"""
py_get_samples: an experimental pure-Python, client-side implementation of the
get_samples() call with dynamic determination of the multicast interface IP address.

This code retrieves time-domain samples from the CRS device. It can optionally
compute a spectrum (via Welch) in either 'psd' (V^2/Hz) or 'ps' (V^2) form.
If reference='relative', we report them in dBc or dBc/Hz (carrier power = DC bin).
If reference='absolute', we report in dBm or dBm/Hz.

Important notes:
  - "average=True" => returns time-domain mean/std dev only (no spectrum).
  - "channel=None" => returns data for all channels.
  - "scaling" => 'psd' or 'ps', determines whether we interpret Welch's data as
    power spectral density (V^2/Hz) or power spectrum (V^2).
  - "nsegments" => how many segments to use for the Welch method (like nperseg),
    or for chunk-based logic. By default 1 => no segmenting beyond the entire data array.
  - "spectrum_cutoff" => fraction of Nyquist (0..1) to keep. Default 0.9 => up to 0.9*(fs/2).

The final output:
  - time-domain arrays in 'results["i"]', 'results["q"]'.
  - if return_spectrum=True, then 'results["spectrum"]' includes frequency axes
    and spectral data, e.g. 'freq_iq', 'freq_dsb' plus entries named:
      '{scaling}_i', '{scaling}_q', '{scaling}_dual_sideband'
    in either dBc or dBm units, depending on 'reference'.

"""

import array
import asyncio
import contextlib
import dataclasses
import enum
import numpy as np
import socket
import sys
import warnings

from ...core.hardware_map import macro
from ...core.schema import CRS
from ...tuber.codecs import TuberResult
from ...core.utils.transferfunctions import VOLTS_PER_ROC
from ... import streamer

# Added import for PSD computation
from scipy.signal import welch

def _cic_correction(frequencies, f_in, R=64, N=6):
    """
    Compute the single-stage CIC (Cascaded Integrator-Comb) filter correction factor.

    CIC filters exhibit passband droop, especially at higher frequencies. This function
    calculates a correction factor to approximately compensate for that droop. The
    correction factor is derived analytically based on the idealized mathematical
    expressions for a CIC filter. However, in firmware implementations, the filter
    coefficients are quantized, so the actual correction will not be perfectly exact,
    but will still be quite close.

    Parameters
    ----------
    frequencies : ndarray
        Frequency bins (in Hz) for which the correction factor is desired. Typically
        these might be FFT bin centers or some other set of discrete frequencies at
        which droop compensation is needed.
    f_in : float
        Input (pre-decimation) sampling rate in Hz. This is the rate at which data
        enters the CIC filter before it is decimated.
    R : int, optional
        The decimation rate. Default is 64.
    N : int, optional
        The number of CIC stages (integrator-comb pairs). Default is 6.

    Returns
    -------
    correction : ndarray
        Array of correction values (dimensionless, same shape as `frequencies`) that can
        be multiplied by the frequency response (or by the time-domain samples after
        the CIC filter) to approximately correct for the droop introduced by the filter.

    Notes
    -----
    1. The correction factor is computed using the ratio of sinc functions raised to
       the power of N:

       .. math::
          H_{corr}(\\omega) =
              \\left(
                  \\frac{\\sin\\left(\\pi f / f_{in}\\right)}
                       {\\sin\\left(\\frac{\\pi f}{R f_{in}}\\right)}
              \\right)^N
              \\times \\frac{1}{R^N}

       where :math:`f` is the frequency, :math:`f_{in}` is the input sample rate,
       :math:`R` is the decimation ratio, and :math:`N` is the number of stages.

    2. At DC (0 Hz), the expression above has an indeterminate form
       :math:`0/0`. We replace those NaN values with the ideal DC gain of 
       :math:`R^N`, then normalize by :math:`R^N`, effectively making the
       correction factor 1 at DC.

    3. In practical hardware/firmware implementations, the CIC coefficients may be
       quantized or truncated, which means that the filter's actual response can deviate
       slightly from the ideal model used here. Therefore, the computed correction
       factor is an approximation that should perform well but will not be absolutely
       precise.
    """
    freq_ratio = frequencies / f_in
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = np.sin(np.pi * freq_ratio)
        denominator = np.sin(np.pi * freq_ratio / R)
        correction = (numerator / denominator) ** N
        # Replace NaNs at DC with the ideal DC gain = R^N
        correction[np.isnan(correction)] = R ** N
    return correction / (R ** N)


def _compute_spectrum(i_data, q_data, fs, dec_stage,
                     scaling='psd', nperseg=None,
                     reference='relative',
                     spectrum_cutoff=0.9):
    """
    Internal function to compute both the I/Q single-sideband PSD and
    the dual-sideband complex PSD in either dBc or dBm, depending on 'reference'.
    The 'scaling' argument can be 'psd' => power spectral density (V^2/Hz)
    or 'ps' => power spectrum (V^2).

    Parameters
    ----------
    i_data : array-like
        Time-domain I (real) samples.
    q_data : array-like
        Time-domain Q (imag) samples.
    fs : float
        Sampling frequency in Hz.
    dec_stage : int
        Decimation stage to define the second CIC correction factor.
    scaling : {'psd','ps'}
        Whether we interpret Welch output as a PSD (V^2/Hz) or total power spectrum (V^2).
    nperseg : int, optional
        Number of samples per Welch segment. Default is all samples (no segmentation).
    reference : {'relative','absolute'}
        If 'relative', we do dBc => referencing the DC bin as the carrier.
        If 'absolute', we do dBm => referencing an absolute scale with 50 ohms.
    spectrum_cutoff : float
        Fraction of Nyquist frequency to retain in the spectrum (default: 0.9).

    Returns
    -------
    dict
        A dictionary containing:
          "freq_iq" : array of frequency bins for single-sideband I/Q,
          "psd_i"   : array of final I data in dBc or dBm,
          "psd_q"   : array of final Q data in dBc or dBm,
          "freq_dsb": array of frequency bins for the dual-sideband data,
          "psd_dual_sideband": array of final dual-sideband data in dBc or dBm.
    """
    if nperseg is None:
        nperseg = len(i_data)

    # Convert 'psd'/'ps' to scipy's 'density'/'spectrum'
    scipy_scaling = 'density' if scaling.lower() == 'psd' else 'spectrum'

    arr_i = np.asarray(i_data)
    arr_q = np.asarray(q_data)
    arr_complex = arr_i + 1j * arr_q


    # Define CIC decimation parameters
    R1 = 64
    R2 = 2 ** dec_stage
    f_in1 = 625e6 / 256.0  # Original samplerate before 1st CIC
    f_in2 = f_in1 / R1     # Samplerate before 2nd CIC

    # Welch for dual-sideband complex
    freq_dsb, psd_c = welch(
        arr_complex, fs=fs, nperseg=nperseg,
        scaling=scipy_scaling,
        return_onesided=False,
        detrend=False # Important because we need the DC information for normalization
    )
    # The CIC corrections are symmetric
    freq_abs = np.abs(freq_dsb)

    # Apply CIC correction
    cic1_corr_c = _cic_correction(freq_abs * R1 * R2, f_in1, R=R1, N=3)
    cic2_corr_c = _cic_correction(freq_abs * R2, f_in2, R=R2, N=6)
    correction_c = cic1_corr_c * cic2_corr_c
    psd_c_corrected = psd_c / correction_c

    # Enforce cutoff
    nyquist = fs / 2
    cutoff_freq = spectrum_cutoff * nyquist
    cutoff_idx_c = freq_abs <= cutoff_freq
    freq_dsb = freq_dsb[cutoff_idx_c]
    psd_c_corrected = psd_c_corrected[cutoff_idx_c]

    # Carrier normalization based on DC bin
    carrier_normalization = psd_c_corrected[0]*(fs/nperseg)

    if reference == 'relative' and scaling == 'psd': # Normalize by the _PS_ of the DC bin:
        ## OVERWRITE the DC bin on the assumption that this is the carrier we are normalizing to
        ## This gives people the correct "reference" right on the plot, as we are correctly assuming
        ## the carrier in total power, not a power density
        psd_c_corrected[0] = psd_c_corrected[0]*(fs/nperseg)
        psd_dual_sideband_db = 10.0 * np.log10(psd_c_corrected / (carrier_normalization + 1e-30))
                
    elif reference == 'relative' and scaling == 'ps': # DC bin already correctly normalized:
        psd_dual_sideband_db = 10.0 * np.log10(psd_c_corrected / ((psd_c_corrected[0]) + 1e-30))
    else:
        # absolute => convert V^2 -> W => dBm
        p_c = psd_c_corrected / 50.0
        psd_dual_sideband_db = 10.0 * np.log10(p_c / 1e-3 + 1e-30)

    # Single-sideband I/Q
    freq_i, psd_i = welch(arr_i, fs=fs, nperseg=nperseg, scaling=scipy_scaling, return_onesided=True, detrend=None)
    freq_q, psd_q = welch(arr_q, fs=fs, nperseg=nperseg, scaling=scipy_scaling, return_onesided=True, detrend=None)
    freq_iq = freq_i

    # CIC Correction
    cic1_corr = _cic_correction(freq_iq * R1 * R2, f_in1, R=R1, N=6)
    cic2_corr = _cic_correction(freq_iq * R2, f_in2, R=R2, N=6)
    correction_iq = cic1_corr * cic2_corr

    psd_i_corrected = psd_i / correction_iq
    psd_q_corrected = psd_q / correction_iq

    cutoff_idx_iq = freq_iq <= cutoff_freq
    freq_iq = freq_iq[cutoff_idx_iq]
    psd_i_corrected = psd_i_corrected[cutoff_idx_iq]
    psd_q_corrected = psd_q_corrected[cutoff_idx_iq]

    # Convert to dBc or dBm
    if reference == 'relative' and scaling == 'psd': # Normalize by the _PS_ of the DC bin
        ## OVERWRITE the DC bin on the assumption that this is the carrier we are normalizing to
        ## This gives people the correct "reference" right on the plot, as we are correctly assuming
        ## the carrier in total power, not a power density
        psd_i_corrected[0] = psd_i_corrected[0]*(fs/nperseg)
        psd_q_corrected[0] = psd_q_corrected[0]*(fs/nperseg)

        ## Then normalize to the TOTAL power (in I and Q)
        psd_i_db = 10.0 * np.log10(psd_i_corrected / (carrier_normalization + 1e-30))
        psd_q_db = 10.0 * np.log10(psd_q_corrected / (carrier_normalization + 1e-30))
    elif reference == 'relative' and scaling == 'ps': # We are already dealing with PS
        psd_i_db = 10.0 * np.log10(psd_i_corrected / (psd_c_corrected[0] + 1e-30))
        psd_q_db = 10.0 * np.log10(psd_q_corrected / (psd_c_corrected[0] + 1e-30))
    else:
        # absolute => convert V^2 to W => W to dBm
        p_i = psd_i_corrected / 50.0
        p_q = psd_q_corrected / 50.0
        psd_i_db = 10.0 * np.log10(p_i / 1e-3 + 1e-30)
        psd_q_db = 10.0 * np.log10(p_q / 1e-3 + 1e-30)

    return {
        "freq_iq": freq_iq,
        "psd_i": psd_i_db,
        "psd_q": psd_q_db,
        "freq_dsb": freq_dsb,
        "psd_dual_sideband": psd_dual_sideband_db,
    }


@macro(CRS, register=True)
async def py_get_samples(crs: CRS,
                         num_samples: int,
                         average: bool = False,
                         channel: int = None,
                         module: int = None,
                         *,
                         return_spectrum: bool = False,
                         scaling: str = 'psd',
                         nsegments: int = 1,
                         reference: str = 'relative',
                         spectrum_cutoff: float = 0.9):
    """
    Asynchronously retrieves samples from the CRS device.

    Parameters
    ----------
    crs : CRS
        The CRS device instance.
    num_samples : int
        Number of samples to collect.
    average : bool, optional
        If True, returns average and std dev only (time-domain).
        If False, returns the full timeseries (and possibly the spectrum).
    channel : int, optional
        Specific channel number to collect data from (1..1024). If None, collects all channels.
    module : int, optional
        The module number from which to retrieve samples. Must be in crs.modules.module.
    return_spectrum : bool, optional
        If True, also compute and return spectral data using welch + droop corrections.
    scaling : {'psd','ps'}, optional
        'psd' => interpret Welch results as V^2/Hz => final dBc/Hz or dBm/Hz
        'ps' => interpret Welch results as V^2 => final dBc or dBm
    nsegments : int, optional
        Number of Welch segments => nperseg = num_samples//nsegments. 
        Default 1 => entire data is one segment. For plots with lots of samples 10 is a good place to start.
    reference : {'relative','absolute'}, optional
        'relative' => dBc or dBc/Hz with DC bin as carrier in spectra, readout counts in TOD
        'absolute' => dBm or dBm/Hz with absolute scaling in spectra, volts in TOD
    spectrum_cutoff : float, optional
        Fraction of Nyquist to retain. Default=0.9 => up to 0.9*(fs/2).

    Returns
    -------
    TuberResult
        Contains the time-domain data (in 'i','q') plus optionally a 'spectrum' dict, with:
          - freq_iq, freq_dsb: frequency axes
          - {scaling}_i, {scaling}_q, {scaling}_dual_sideband
        in dBc or dBm units, depending on 'reference'.
        If channel=None, data arrays contain all channels.

    Notes
    -----
    - If average=True, we only return time-domain mean and std dev (no spectrum).
    - If reference='absolute', we convert time-domain data to volts (VOLTS_PER_ROC).
    - If reference='relative', the data are referenced to the total carrier power. 
      The DC bin is also overwritten to show the DC power rather than a density. 
      For the dual-sideband data this will be exactly 0dB.
    """
    # Ensure 'module' parameter is valid
    assert (
        module in crs.modules.module
    ), f"Unspecified or invalid module! Available modules: {crs.modules.module}"

    if channel is not None:
        assert 1 <= channel <= streamer.NUM_CHANNELS, f"Invalid channel {channel}!"

    # We need to ensure all data we grab from the network was emitted "now" or
    # later, from the perspective of control flow. Because of delays in the
    # signal path and network, we might actually get stale data.  We want to
    # avoid the following pathology:
    #
    # >>> # cryostat is in "before" condition"
    # >>> d.set_amplitude(...)
    # >>> # cryostat is in "after" condition
    # >>> x = await d.get_samples(...) # "x" had better reflect "after" condition!
    #
    # There are two ways data can be stale:
    #
    # 1. Buffers in the end-to-end network mean that any packets we grab "now"
    # might actually be pretty old. We can fix this by grabbing a "now"
    # timestamp from the board, and tossing any data packets that are older
    # than it.
    #
    # 2. Adjusting the "now" timestamp to add a little smidgen of signal-path
    # delay. This is because the signal path experiences group delay due to
    # decimation filters (PFB, CIC) and the timestamp doesn't. There are also
    # digital delays in the data converters (upsampling, downsampling) and
    # analog delays in the system.

    async with crs.tuber_context() as tc:
        ts = tc.get_timestamp()
        high_bank = tc.get_analog_bank()
        (ts, high_bank) = await tc()


    if high_bank:
        assert module > 4, \
                f"Can't retrieve samples from module {module} with set_analog_bank(True)"
        module -= 4
    else:
        assert module <= 4, \
                f"Can't retrieve samples from module {module} with set_analog_bank(False)"

    # Because IRIG-B takes a full second to decode a timestamp, it's polite to
    # stall here if we don't see a recent timestamp for up to a couple of
    # seconds. This allows a set_timestamp_port() call followed by a
    # py_get_samples() call to succeed, which may occur when a board is first
    # initialized after power-up. Note that old packets (with stale timestamps)
    # still need to traverse the network even if the board sees fresh ones,
    # this PC may receive old ones until they flush out.
    attempt = 5
    while attempt>0 and not ts.recent:
        warnings.warn(f"Got a stale timestamp. Trying again ({attempt} attempts remaining...)")
        await asyncio.sleep(1)
        ts = await crs.get_timestamp()
        attempt -= 1

    # Math on timestamps only works if they are valid
    assert ts.recent, "Timestamp wasn't recent - do you have a valid timestamp source?"

    # Ingest timestamp into Pythonic representation and nudge to compensate for
    # FIR delay.
    ts = streamer.Timestamp.from_TuberResult(ts)
    ts.ss += np.uint32(.02 * streamer.SS_PER_SECOND) # 20ms, per experiments at FIR6
    ts.renormalize()

    with streamer.get_multicast_socket(crs.tuber_hostname) as sock:

        # To use asyncio, we need a non-blocking socket
        loop = asyncio.get_running_loop()
        sock.setblocking(False)

        # Variable to track the previous sequence number for continuity checks
        prev_seq = None

        # Allow up to 10 packet-loss retries
        retries = 10

        packets = []
        # Start receiving packets
        while len(packets) < num_samples:
            data = await asyncio.wait_for(
                loop.sock_recv(sock, streamer.STREAMER_LEN), streamer.STREAMER_TIMEOUT
            )

            # Parse the received packet
            p = streamer.DfmuxPacket.from_bytes(data)

            if p.serial != int(crs.serial):
                warnings.warn(
                    f"Packet serial number {p.serial} didn't match CRS serial number {crs.serial}! Two boards on the network? IGMPv3 capable router will fix this warning."
                )

            # Filter packets by module
            if p.module != module - 1:
                continue  # Skip packets from other modules

            # Sanity checks on the packet
            if p.magic != streamer.STREAMER_MAGIC:
                raise RuntimeError(f"Invalid packet magic! {p.magic} != {streamer.STREAMER_MAGIC}")

            if p.version != streamer.STREAMER_VERSION:
                raise RuntimeError(f"Invalid packet version! {p.version} != {streamer.STREAMER_VERSION}")

            # Check if this packet is older than our "now" timestamp
            assert ts.source == p.ts.source, f"Timestamp source changed! {ts.source} vs {p.ts.source}"
            if ts > p.ts:
                continue

            # Update the sequence number continuity check
            if prev_seq is not None and prev_seq + 1 != p.seq:
                if retries > 0:
                    warnings.warn(
                        f"Discontinuous packet capture! Previous sequence {prev_seq} -> current sequence {p.seq}. "
                        f"Retrying capture ({retries} attempts remain.)"
                    )
                    retries -= 1
                    packets = []
                    prev_seq = None
                    continue
                else:
                    raise RuntimeError(
                        f"Discontinuous packet capture! Previous sequence {prev_seq} -> current sequence {p.seq}"
                    )

            # Update the previous sequence number
            prev_seq = p.seq

            # Append the valid packet to the list
            packets.append(p)

    # If average => just return time-domain averages
    if average:
        mean_i = np.zeros(streamer.NUM_CHANNELS)
        mean_q = np.zeros(streamer.NUM_CHANNELS)
        std_i = np.zeros(streamer.NUM_CHANNELS)
        std_q = np.zeros(streamer.NUM_CHANNELS)

        for c in range(streamer.NUM_CHANNELS):
            mean_i[c] = np.mean([p.s[2*c]/256 for p in packets])
            mean_q[c] = np.mean([p.s[2*c+1]/256 for p in packets])
            std_i[c] = np.std([p.s[2*c]/256 for p in packets])
            std_q[c] = np.std([p.s[2*c+1]/256 for p in packets])

        # If reference='absolute', convert to volts
        if reference == 'absolute':
            mean_i *= VOLTS_PER_ROC
            mean_q *= VOLTS_PER_ROC
            std_i *= VOLTS_PER_ROC
            std_q *= VOLTS_PER_ROC

        if channel is None:
            results = {
                "mean": TuberResult(dict(i=mean_i, q=mean_q)),
                "std": TuberResult(dict(i=std_i, q=std_q)),
            }
        else:
            # single channel => pick out channel-1
            results = {
                "mean": TuberResult(dict(i=mean_i[channel-1], q=mean_q[channel-1])),
                "std": TuberResult(dict(i=std_i[channel-1], q=std_q[channel-1])),
            }
        return TuberResult(results)

    # Otherwise build the normal time-domain results
    results = dict(ts=[TuberResult(dataclasses.asdict(p.ts)) for p in packets])

    if channel is None:
        # Return data for all channels
        results["i"] = []
        results["q"] = []
        for c in range(streamer.NUM_CHANNELS):
            i_channel = np.array([p.s[2*c]/256 for p in packets])
            q_channel = np.array([p.s[2*c+1]/256 for p in packets])
            if reference == 'absolute':
                i_channel=i_channel*VOLTS_PER_ROC
                q_channel=q_channel*VOLTS_PER_ROC
            results["i"].append(i_channel.tolist())
            results["q"].append(q_channel.tolist())
    else:
        i_channel = np.array([p.s[2*(channel-1)]/256 for p in packets])
        q_channel = np.array([p.s[2*(channel-1)+1]/256 for p in packets])
        if reference=='absolute':
            i_channel=i_channel*VOLTS_PER_ROC
            q_channel=q_channel*VOLTS_PER_ROC
        results["i"] = i_channel.tolist()
        results["q"] = q_channel.tolist()

    # Optionally compute the spectrum
    if return_spectrum:
        # Convert nsegments => nperseg for Welch
        nperseg = num_samples // nsegments

        # Retrieve decimation stage => helps define final sampling freq
        dec_stage = await crs.get_fir_stage()
        fs = 625e6/(256*64*(2**dec_stage))

        spec_data = {}
        if channel is None:
            # multi-channel => store a list of spectra for each channel
            spec_data["freq_iq"] = None
            spec_data["freq_dsb"] = None
            i_ch_spectra=[]
            q_ch_spectra=[]
            c_ch_spectra=[]
            for c in range(streamer.NUM_CHANNELS):
                i_data = results["i"][c]
                q_data = results["q"][c]
                d = _compute_spectrum(
                    i_data, q_data, fs, dec_stage,
                    scaling=scaling,
                    nperseg=nperseg if nperseg else num_samples,
                    reference=reference,
                    spectrum_cutoff=spectrum_cutoff
                )
                # Store freq_iq, freq_dsb once
                if spec_data["freq_iq"] is None:
                    spec_data["freq_iq"] = d["freq_iq"].tolist()
                if spec_data["freq_dsb"] is None:
                    # shift freq_dsb for plotting
                    spec_data["freq_dsb"] = np.fft.fftshift(d["freq_dsb"]).tolist()

                i_ch_spectra.append(d["psd_i"].tolist())
                q_ch_spectra.append(d["psd_q"].tolist())
                c_ch_spectra.append(np.fft.fftshift(d["psd_dual_sideband"]).tolist())

            # store under {scaling}_i, {scaling}_q, {scaling}_dual_sideband
            spec_data[f"{scaling}_i"] = i_ch_spectra
            spec_data[f"{scaling}_q"] = q_ch_spectra
            spec_data[f"{scaling}_dual_sideband"] = c_ch_spectra
        else:
            i_data = results["i"]
            q_data = results["q"]
            d = _compute_spectrum(
                i_data, q_data, fs, dec_stage,
                scaling=scaling,
                nperseg=nperseg if nperseg else num_samples,
                reference=reference,
                spectrum_cutoff=spectrum_cutoff
            )
            spec_data["freq_iq"] = d["freq_iq"].tolist()
            spec_data[f"{scaling}_i"] = d["psd_i"].tolist()
            spec_data[f"{scaling}_q"] = d["psd_q"].tolist()

            spec_data["freq_dsb"] = np.fft.fftshift(d["freq_dsb"]).tolist()
            spec_data[f"{scaling}_dual_sideband"] = np.fft.fftshift(d["psd_dual_sideband"]).tolist()

        # attach spectrum data to results
        results["spectrum"] = TuberResult(spec_data)

    return TuberResult(results)
