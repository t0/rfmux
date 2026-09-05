'''
Example run for 4 random channels 

channel = [13, 23, 37, 47]
result = await crs.py_run_pfb_streamer(channel = channel, module = 1, time_run = 0.5) 

#### Plotting spectrum for each channel #####

for i in range(len(channel)):
    plt.plot(result.spectrum.freq_iq, result.spectrum.psd_i[i])
    plt.plot(result.spectrum.freq_iq, result.spectrum.psd_q[i])
    plt.show()
'''

from typing import Union, List
import asyncio
import numpy as np
import time
import traceback
import warnings

from ...core.hardware_map import macro
from ...core.schema import CRS
from tuber.codecs import TuberResult
from ...core.transferfunctions import PFB_SAMPLING_FREQ, VOLTS_PER_ROC

#: How long stale packet timestamps are tolerated before the capture
#: raises: IRIG-B takes about a second to decode after set_timestamp_port,
#: and packets stamped before that can still be in flight.
STALE_TIMESTAMP_GRACE_S = 5.0
from ... import streamer

from .py_get_pfb_samples import apply_pfb_correction


@macro(CRS, register=True)
async def py_run_pfb_streamer(crs : CRS,
                              channel : Union[None, int, List[int]] = None,
                              module : int = 1,
                              *, 
                              time_run : float = 4.0,
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
      2. Listens on the PFB multicast socket and captures packets until
         ``time_run`` seconds of **sample time** have elapsed.
      3. Demultiplexes interleaved samples into per-channel complex time streams.
      4. Optionally recenters the channel using the module NCO (`reset_NCO`).
      5. Computes single- and dual-sideband power spectral densities using
         PFB droop correction.
      6. Disables PFB streaming before returning results.

    **Sample-time semantics:** ``time_run`` is the capture duration in
    *sample time*, elapsed time in the data stream as the packet
    timestamps measure it.  On hardware that is wall time; on the mock,
    whose physics can run slower than real time, it is simulation time.
    The timestamps have to be disciplined: a board without a timestamp
    source stamps every packet stale, and the capture then raises rather
    than run forever, as ``py_get_samples`` does.

    Parameters
    ----------
    crs : CRS
        Connected CRS control object.

    channel : None | int | list[int], default None
        Channel or channels to stream. If None, PFB streaming is disabled and
        the function returns without capturing data.

    module : int, default 1
        CRS module index (must be in the range 1–8).

    time_run : float, keyword-only
        Duration in seconds of **sample time** (elapsed time in the data
        stream, from the packet timestamps); wall time on hardware,
        simulation time on the mock.

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
    
    if channel is None:
        print(f"[Pfb streaming]: No channels specified. No Pfb streaming.")
        crs.set_pfb_streamer(channel = None, module = module)
        return
    
    try:
        #### Activate streaming ####
        await crs.set_pfb_streamer(channel = channel, module = module)
    
        #### rechecking if its active ####
        pfb_state = await crs.get_pfb_streamer(module = module)
        if pfb_state is not None:
            print(f"[Pfb streaming]: Active on channel {channel}")
        else:
            print(f"[Pfb streaming]: No active PFB streaming found")
    
        #### Opening the socket #####
        # resolve_host, not the raw hostname: a mock CRS built from a
        # serial alone reports rfmux0000.local, which resolves nowhere.
        host = streamer.resolve_host(crs.tuber_hostname)
        port = streamer.PFB_STREAMER_PORT
        
        channels = channel if isinstance(channel, list) else [channel]
        n_groups = len(channels)
    
        with streamer.get_multicast_socket(host, port=port) as sock:
    
            # To use asyncio, we need a non-blocking socket
            loop = asyncio.get_running_loop()
            sock.setblocking(False)
            packets = []

            slices = [slice(k, None, n_groups) for k in range(n_groups)]
    
            async def receive_attempt():
                """Capture packets until *sample time* reaches ``time_run``.

                Sample time comes from the packet timestamps, so the
                capture covers the same stretch of the data stream whether
                it arrives at real time or (the mock at many tones) slower.
                Stale timestamps are tolerated for STALE_TIMESTAMP_GRACE_S,
                the time IRIG-B needs after set_timestamp_port; past that
                the capture raises.  When the stream is arriving slower
                than real time, progress and an ETA are printed.
                """

                slot_lists = [[] for _ in range(n_groups)]

                elapsed_sample_time: float = 0.0
                wall_start = time.monotonic()
                sample_time_origin = None
                stale_since = None

                # Progress estimation state
                _slow_stream = False
                _last_progress_frac = 0.0      # Last reported progress fraction
                _progress_step = 0.10           # Report every 10%
                _warmup_wall = 2.0              # Seconds before first throughput check
                _throughput_ratio = 1.0         # sample_time / wall_time (1.0 = real-time)

                print(f"[Pfb streaming] Capturing for {time_run}s of sample time...")

                while elapsed_sample_time < time_run:
                    data = await asyncio.wait_for(
                        loop.sock_recv(sock, streamer.PFB_PACKET_SIZE),
                        streamer.STREAMER_TIMEOUT,
                    )
    
                    # Parse the received packet
                    p = streamer.PFBPacket(data)
                    packets.append(p)
                                    
                    for lst, sl in zip(slot_lists, slices):
                        lst.extend(p[sl])

                    ts_sec = streamer.ts_to_seconds(p.ts)
                    if ts_sec is None:
                        stale_since = stale_since or time.monotonic()
                        if time.monotonic() - stale_since > STALE_TIMESTAMP_GRACE_S:
                            raise RuntimeError(
                                "Packet timestamps are not recent, so sample "
                                "time cannot be measured. Do you have a valid "
                                "timestamp source? Set one with "
                                "crs.set_timestamp_port(...) and try again.")
                        continue
                    stale_since = None
                    if sample_time_origin is None:
                        sample_time_origin = ts_sec
                    time_samples_in_pkt = p.num_samples // max(n_groups, 1)
                    elapsed_sample_time = (ts_sec - sample_time_origin
                                           + (time_samples_in_pkt - 1) / PFB_SAMPLING_FREQ)

                    # ── Progress estimation ─────
                    wall_elapsed = time.monotonic() - wall_start
                    if (elapsed_sample_time > 0 and wall_elapsed > _warmup_wall
                            and not _slow_stream):
                        _throughput_ratio = elapsed_sample_time / wall_elapsed
                        if _throughput_ratio < 0.5:
                            _slow_stream = True
                            est_total_wall = time_run / _throughput_ratio
                            print(
                                f"[Pfb streaming] Stream arriving at "
                                f"{_throughput_ratio*100:.1f}% of real time "
                                f"(the mock at many tones). "
                                f"Estimated wall time for {time_run}s of sample data: "
                                f"~{est_total_wall:.1f}s"
                            )

                    if _slow_stream and time_run > 0:
                        frac = elapsed_sample_time / time_run
                        if frac - _last_progress_frac >= _progress_step:
                            _last_progress_frac = frac
                            # Recompute throughput from latest data
                            _throughput_ratio = elapsed_sample_time / max(wall_elapsed, 1e-9)
                            remaining_sample = time_run - elapsed_sample_time
                            eta = remaining_sample / max(_throughput_ratio, 1e-9)
                            print(
                                f"[Pfb streaming] Progress: "
                                f"{elapsed_sample_time:.4f}/{time_run}s "
                                f"({frac*100:.0f}%) — "
                                f"wall: {wall_elapsed:.1f}s, "
                                f"ETA: ~{eta:.1f}s remaining"
                            )

                wall_elapsed = time.monotonic() - wall_start
                print(f"[Pfb streaming] Sample time covered: {elapsed_sample_time:.4f}s "
                      f"(wall time: {wall_elapsed:.1f}s)")

                pfb_samps = [np.asarray(lst, dtype=np.complex128) for lst in slot_lists]
                return sorted(packets, key=lambda p: p.seq), pfb_samps
                
            # Allow up to 10 packet-loss retries
            for attempt in range(NUM_ATTEMPTS := 3):
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
                    f"Failed to retrieve contiguous, consistent packet capture in {NUM_ATTEMPTS} attempts! Do either or all. 1.Check network config\n 2.Reduce the number of channels\n 3. Reduce the acquisition time.")
        
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
        return TuberResult(
            i=time_list_i,
            q=time_list_q,
            spectrum=TuberResult(
                freq_iq=freq_ssb.tolist(),
                psd_i=psd_list_i,
                psd_q=psd_list_q,
                freq_dsb=freq_dsb.tolist(),
                psd_dual_sideband=psd_list_dual))

    except Exception as e:
        if e.__class__.__name__ == "TuberRemoteError":
            raise RuntimeError(
                "[Pfb streaming] Remote call failed: bandwidth limit exceeded.\n"
                "Mitigations:\n"
                "  1) Reduce the number of channels requested\n"
                "  2) Increase decimation (e.g., 6)\n"
            )
    
        else:
            print("[Pfb streaming]: Exception occurred:")
            traceback.print_exc()
            raise
    
    finally:
        print(f"[Pfb streaming] Shutting off the streaming")
        await crs.set_pfb_streamer(channel = None, module = module)