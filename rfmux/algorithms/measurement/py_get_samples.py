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
import asyncio, inspect
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
from ...core.transferfunctions import VOLTS_PER_ROC, spectrum_from_slow_tod
from ... import streamer

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
                         spectrum_cutoff: float = 0.9,
                         _extra_metadata: bool = False):
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
    _extra_metadata : bool, optional
        If True, include extra packet data beyond what get_samples returns.
        This is primarily intended for regression testing.

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

    # Ensure 'module' parameter is valid. We can't validate 'channel' until
    # later on, when we have a packet to inspect. (This is kinder than
    # explicitly querying the readout mode, since older firmware may not
    # support it.)
    assert module in range(1, 9), f"Invalid module {module}! Must be between 1 and 8"

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
        tc.get_timestamp()
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
    ts = streamer.Timestamp(**vars(ts))
    ts.ss += np.uint32(.02 * streamer.SS_PER_SECOND) # 20ms, per experiments at FIR6
    ts.renormalize()

    if crs.tuber_hostname == "rfmux0000.local":
        host = '127.0.0.1'
    else:
        host = crs.tuber_hostname

    with streamer.get_multicast_socket(host) as sock:

        # To use asyncio, we need a non-blocking socket
        loop = asyncio.get_running_loop()
        sock.setblocking(False)

        async def receive_attempt():
            packets = []
            # Start receiving packets
            while len(packets) < num_samples:
                data = await asyncio.wait_for(
                    loop.sock_recv(sock, streamer.LONG_PACKET_SIZE),
                    streamer.STREAMER_TIMEOUT,
                )

                # Parse the received packet
                p = streamer.ReadoutPacket(data)

                if crs.serial == "MOCK0001":
                    packets.append(p)  
                else:
                    if p.serial != int(crs.serial):
                        warnings.warn(
                            f"Packet serial number {p.serial} didn't match CRS serial number {crs.serial}! Two boards on the network? IGMPv3 capable router will fix this warning."
                        )
    
                    # Filter packets by module
                    if p.module != module - 1:
                        continue  # Skip packets from other modules
    
                    # Check if this packet is older than our "now" timestamp
                    assert ts.source == p.ts.source, f"Timestamp source changed! {ts.source} vs {p.ts.source}"
                    if ts > p.ts:
                        continue

                    packets.append(p)

            # Sort packets by sequence number
            return sorted(packets, key=lambda p: p.seq)

        # Allow up to 10 packet-loss retries
        for attempt in range(NUM_ATTEMPTS := 10):
            packets = await receive_attempt()

            sequence_steps = np.diff([p.seq for p in packets])
            if not all(np.diff([p.seq for p in packets]) == 1):
                warnings.warn(
                    f"Discontinuous packet capture! Attempt {attempt+1}/{NUM_ATTEMPTS}..."
                )
                continue

            versions = {p.version for p in packets}
            if len(versions) != 1:
                warnings.warn(
                    f"Packet version {versions} changed! Attempt {attempt+1}/{NUM_ATTEMPTS}..."
                )
                continue

            # passed our tests - break out of the loop
            break
        else:
            raise RuntimeError(
                f"Failed to retrieve contiguous, consistent packet capture in {NUM_ATTEMPTS} attempts!"
            )

    num_channels = packets[0].get_num_channels()
    if channel is not None and channel > num_channels:
        raise ValueError(f"Invalid channel {channel}! Packets only contain {num_channels} channels in this readout mode.")

    # If average => just return time-domain averages
    if average:
        # Stack all samples into a (num_samples, num_channels) array
        samples = np.stack([p.samples for p in packets])

        # If reference='absolute', convert to volts
        if reference == 'absolute':
            samples *= VOLTS_PER_ROC

        # Compute statistics across time axis (axis=0)
        mean_i = np.mean(samples.real, axis=0)
        mean_q = np.mean(samples.imag, axis=0)
        std_i = np.std(samples.real, axis=0)
        std_q = np.std(samples.imag, axis=0)

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
    results = dict(ts=[TuberResult(dict(p.ts)) for p in packets])

    if _extra_metadata:
        results["seq"] = [p.seq for p in packets]

    if channel is None:
        # Return data for all channels
        # Stack all samples into a 2D array: (num_samples, num_channels)
        samples = np.stack([p.samples for p in packets])
        if reference == 'absolute':
            samples *= VOLTS_PER_ROC

        # Transposition produces to (num_channels, num_samples)
        results["i"] = samples.real.T.tolist()
        results["q"] = samples.imag.T.tolist()
    else:
        samples = np.array([p.samples[channel-1] for p in packets])
        if reference=='absolute':
            samples *= VOLTS_PER_ROC

        results["i"] = samples.real.tolist()
        results["q"] = samples.imag.tolist()

    # Optionally compute the spectrum
    if return_spectrum:
        # Convert nsegments => nperseg for Welch
        nperseg = num_samples // nsegments

        dec_stage = await crs.get_decimation()
        fs = 625e6/(256*64*(2**dec_stage))

        

        spec_data = {}
        if channel is None:
            # multi-channel => store a list of spectra for each channel
            spec_data["freq_iq"] = None
            spec_data["freq_dsb"] = None
            i_ch_spectra=[]
            q_ch_spectra=[]
            c_ch_spectra=[]
            for c in range(num_channels):
                i_data = results["i"][c]
                q_data = results["q"][c]
                
                # Calculate spectrum for this channel
                if reference == "absolute": #### since its in volts
                    d = spectrum_from_slow_tod(
                        i_data, q_data, dec_stage,
                        scaling=scaling,
                        nperseg=nperseg if nperseg else num_samples,
                        reference=reference,
                        spectrum_cutoff=spectrum_cutoff,
                        input_units="volts"
                    )
                else:
                    d = spectrum_from_slow_tod(
                        i_data, q_data, dec_stage,
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
            
            if reference == "absolute": #### since its in volts
                d = spectrum_from_slow_tod(
                    i_data, q_data, dec_stage,
                    scaling=scaling,
                    nperseg=nperseg if nperseg else num_samples,
                    reference=reference,
                    spectrum_cutoff=spectrum_cutoff,
                    input_units="volts"
                )
            else:
                d = spectrum_from_slow_tod(
                    i_data, q_data, dec_stage,
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
