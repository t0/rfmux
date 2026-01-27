import array
import asyncio, inspect
import contextlib
import dataclasses
import enum
import numpy as np
import socket
import sys
import warnings
import time
from typing import Union, List

from ...core.hardware_map import macro
from ...core.schema import CRS
from ...tuber.codecs import TuberResult
from ...core.transferfunctions import VOLTS_PER_ROC, spectrum_from_slow_tod
from ... import streamer
from rfmux.tools.periscope.utils import Circular

class PulseCapture:
    """
    Streaming pulse detector and capture buffer for CRS readout packets.

    `PulseCapture` maintains per-channel circular buffers of I, Q, and relative time samples and
    implements a simple threshold-triggered pulse capture state machine. As packets arrive, samples
    are appended into FIFO-ordered circular buffers. When a sample in the selected component
    (`thresh_type` = "I" or "Q") exceeds `thresh` (magnitude compare), capture begins. Capture ends
    once the signal has returned below `noise_level` for >10 consecutive samples. At end-of-pulse,
    the class slices a window from the buffers consisting of a configurable number of pre-trigger
    samples (`pre`) plus the observed post-trigger duration, clamps the window to currently
    available buffer contents, and stores the captured pulse.

    Captured pulses are stored in `self.pulses` as a dictionary indexed by pulse number:
        pulses[pulse_index] = {
            "Amp_I": np.ndarray,   # windowed I samples
            "Amp_Q": np.ndarray,   # windowed Q samples
            "Time":  np.ndarray,   # windowed relative timestamps aligned to samples
        }

    Timing:
      - Relative time is derived from packet timestamps (`pkt.ts`) when `ts.recent` is True.
      - A fixed +20 ms offset is applied to the timestamp (matching the capture-side convention).
      - `start_time` is set on the first valid timestamp and all subsequent "Time" values are
        relative to that start.

    Key attributes:
        buf (dict):
            Per-channel circular buffers: buf[c]["I"], buf[c]["Q"], buf[c]["ts"].
        thresh (float):
            Trigger threshold applied to the selected component (I or Q). Trigger uses absolute
            value to catch inverted pulses/dips.
        noise_level (float):
            Level used to decide when the waveform has returned to baseline. End-of-pulse is
            declared after >10 consecutive samples below this level.
        pre (int):
            Requested number of pre-trigger samples to include in the captured window (clamped
            if insufficient history is available).
        start_time (float | None):
            Reference time (seconds-of-day converted to seconds) of first valid packet used for
            relative timestamps.
        pulses (dict):
            Captured pulse windows as described above.
        pulse_count (int):
            Monotonically increasing pulse index.
        capturing (bool):
            Whether the state machine is currently within a pulse capture.
        abs_n (int):
            Absolute packet/sample counter used to compute post-trigger length and map trigger
            position into the current FIFO buffer view.
        trig_abs (int | None):
            Absolute counter value at the trigger sample.
        trigger_value (float | None):
            Trigger sample value in the selected component (I or Q).

    Args:
        buf_size (int):
            Capacity of each per-channel circular buffer in samples.
        channels (Iterable[int]):
            1-indexed channel numbers to buffer and monitor.
        thresh (float):
            Trigger threshold.
        noise_level (float):
            Baseline/noise level used for end-of-pulse detection.
        thresh_type (str):
            Which component to use for triggering/end detection: "I" (real) or "Q" (imag).
        pre (int, optional):
            Number of pre-trigger samples to attempt to include in the saved pulse window.

    Usage:
        Instantiate once, then feed packets via `_update_buffer_and_capture(pkt)` as they arrive.
        After streaming, read `start_time` and `pulses` for results.
    """

    def __init__(self, buf_size, channels, thresh, noise_level, thresh_type, pre=20):
        self.buf = {}
        self.channels = channels
        self.buf_size = buf_size

        for c in self.channels:
            self.buf[c] = {k: Circular(buf_size) for k in ("I", "Q", "ts")}

        self.thresh = thresh
        self.noise_level = noise_level
        self.pre = pre

        self.start_time = None

        self.pulses = {}
        self.pulse_count = 0

        self.capturing = False
        self.end_ptr_count = 0

        # absolute time index (increment once per packet)
        self.abs_n = 0
        self.trig_abs = None
        self.trigger_value = None

        self.data_type = thresh_type #### Can be 'I' or 'Q' 

    def _calculate_relative_timestamp(self, pkt) -> float | None:
        ts = pkt.ts
        if ts.recent:
            ts.ss += int(0.02 * streamer.SS_PER_SECOND)
            ts.renormalize()
            t_now = ts.h * 3600 + ts.m * 60 + ts.s + ts.ss / streamer.SS_PER_SECOND
            if self.start_time is None:
                self.start_time = t_now
            return t_now - self.start_time
        return None

    def _pulse_window(self, circ: Circular, start: int, end: int) -> np.ndarray:
        """
        Extract pulse in FIFO-ordered window [start:end) from circ.data(), copying out.
        start/end are validated/clamped by caller.
        """
        d = circ.data()  # length = circ.count (<= N), FIFO oldest->newest
        return d[start:end].copy()

    def _update_buffer_and_capture(self, pkt):
        samples = pkt.samples / 256.0
        arr_time = self._calculate_relative_timestamp(pkt)

        self.abs_n += 1 ## Absolute packet counter ##

        for c in self.channels:
            sample = samples[c - 1]

            self.buf[c]["I"].add(float(sample.real))
            self.buf[c]["Q"].add(float(sample.imag))
            self.buf[c]["ts"].add(arr_time)

            if self.data_type == "Q":
                data_val = sample.imag
            else:
                data_val = sample.real

            # --- trigger detection ---
            if (abs(data_val) > abs(self.thresh)) and (not self.capturing): ### absolute to make sure we catch 180 rotated pulses (or dips)
                self.capturing = True
                self.end_ptr_count = 0
                self.trig_abs = self.abs_n
                self.trigger_value = float(data_val)

                print("\nFound trigger abs_n", self.trig_abs,
                      "time", arr_time,
                      "Sample value", data_val)

            # --- end condition and save ---
            if self.capturing:
                if abs(data_val) < abs(self.noise_level):
                    self.end_ptr_count += 1
                else:
                    self.end_ptr_count = 0

                if self.end_ptr_count > 10:

                    ### Finding the start and end indices of the pulse in the buffer ###
                    
                    post = self.abs_n - self.trig_abs #### How big is the burst ###
                    if post <= 0:
                        # reset
                        self.capturing = False
                        self.end_ptr_count = 0
                        self.trig_abs = None
                        self.trigger_value = None
                        continue

                    # Use CURRENT available FIFO/buffer length
                    L = self.buf[c]["I"].count  

                    trig_fifo = (L - 1) - (self.abs_n - self.trig_abs) ### indices in buffer data that has the trigger

                    # If trig_fifo is out of view, the trigger sample has already been overwritten
                    # or we advanced too far. Skip save.
                    if trig_fifo < 0 or trig_fifo >= L:
                        print("Skipping save: trigger not in current buffer view",
                              {"L": L, "trig_fifo": trig_fifo, "abs_n": self.abs_n, "trig_abs": self.trig_abs})
                        # reset
                        self.capturing = False
                        self.end_ptr_count = 0
                        self.trig_abs = None
                        self.trigger_value = None
                        continue

                    # Compute desired window bounds in FIFO/buffer indices
                    pre_desired = self.pre ### How early you want to collect it
                    start = trig_fifo - pre_desired
                    end = trig_fifo + post  # exclusive if post counts samples after trigger

                    #### Making sure we catch the entire pulse ####

                    # Clamp to [0, L]
                    start_clamped = max(0, start)
                    end_clamped = min(L, end)

                    # If we had to clamp start upward, we effectively have fewer pre samples
                    pre_actual = trig_fifo - start_clamped

                    # Require at least (trigger + 1) sample in window
                    if end_clamped <= start_clamped:
                        print("Skipping save: empty window after clamping",
                              {"start": start_clamped, "end": end_clamped, "L": L})
                        # reset
                        self.capturing = False
                        self.end_ptr_count = 0
                        self.trig_abs = None
                        self.trigger_value = None
                        continue

                    #### Actually storing the data ######

                    I_win = self._pulse_window(self.buf[c]["I"], start_clamped, end_clamped)
                    Q_win = self._pulse_window(self.buf[c]["Q"], start_clamped, end_clamped)
                    ts_win = self._pulse_window(self.buf[c]["ts"], start_clamped, end_clamped)

                    self.pulse_count += 1
                    self.pulses[self.pulse_count] = {
                        "Amp_I": np.array(I_win),
                        "Amp_Q": np.array(Q_win),
                        "Time": np.array(ts_win),
                    }

                    #### Logging capture ####
                    if self.data_type == "Q":
                        data_win = Q_win
                    else:
                        data_win = I_win
                        
                    print("Pulse:", self.pulse_count,
                          "Saving Now: trigger sample (window)", data_win[pre_actual],
                          "end sample", data_win[-1],
                          "len", len(data_win),
                          "pre_actual", pre_actual,
                          "trigger sample (at trigger time)", self.trigger_value)

                    # reset
                    self.capturing = False
                    self.end_ptr_count = 0
                    self.trig_abs = None
                    self.trigger_value = None
    

def get_noise_level(packets, thresh, channels, thresh_type):
    """
    Estimate per-channel noise levels from a set of CRS readout packets.

    This function extracts complex samples from the provided readout packets, selects either
    the I or Q component based on `thresh_type`, and computes noise statistics using only
    samples that fall below the specified threshold. Samples above threshold are assumed to
    be signal-contaminated and are excluded from the noise estimate.

    For each channel, the median of the below-threshold samples is used as the noise level.
    This heuristic is intended to be robust for bursts on the order of ~250–300 samples wide.

    Args:
        packets (List[ReadoutPacket]):
            List of CRS readout packets containing complex sample data.
        thresh (float):
            Threshold used to exclude signal-dominated samples from the noise estimate.
        channels (Iterable[int]):
            1-indexed channel numbers for which noise levels should be computed.
        thresh_type (str):
            Component to analyze for thresholding: "I" (real) or "Q" (imaginary).

    Returns:
        Tuple[List[float], dict]:
            - noise_levels: List of estimated noise levels, one per channel, in the same order
              as `channels`.
            - noise_data: Dictionary mapping channel number to the array of samples used in the
              noise calculation (i.e., samples below threshold).
    """
    ### Find ideal solution ####
    samples = np.stack([p.samples for p in packets]) / 256.0
    real_all = samples.real
    imag_all = samples.imag

    noise_levels = []
    noise_data = {}

    for c in channels:
        real_ch = real_all[:, c-1]
        imag_ch = imag_all[:, c-1]

        
        if thresh_type == "Q":
            thresh_ch = imag_ch
        else:
            thresh_ch = real_ch

        below_thresh = [] 
    
        for r in thresh_ch:
            if r < thresh:
                below_thresh.append(r) #### collecting all the samples below the threshold #####
                
        below_thresh = np.array(below_thresh)
        noise_data[c] = below_thresh
        
        max_val = np.max(below_thresh)
        min_val = np.min(below_thresh)
        std = np.std(below_thresh)
        mean = np.mean(below_thresh)
        median = np.mean(below_thresh)

        print("Channel:", c,
              "Min:", min_val,
              "Max:", max_val,
              "Std:", std,
              "Mean:", mean,
              "Median:", median,
              "Sample_below:", len(below_thresh))
        
        noise_levels.append(median) ##### This is a pretty good approximation for ~ 250-300 samples wide burst ####

    return noise_levels, noise_data

@macro(CRS, register=True)
async def slow_trigger_capture(crs: CRS,
                               channel : Union[None, int, List[int]] = None,
                               module: int = None,
                               *,
                               time_run : float = 10.0,
                               thresh : float = None, 
                               noise : float = None, ### if None run a calculation script
                               buf_size : int = 5000,
                               thresh_type : str = "I", #### or can be Q #####
                               in_volts : bool = False, ### add volts conversion  ####
                               _extra_metadata: bool = False):

    """
    Capture threshold-triggered pulses from CRS multicast readout packets over a fixed time window.

    This coroutine listens to the CRS multicast stream, filters packets to a selected module (and
    optional channel subset), and performs simple pulse detection using `PulseCapture`. The capture
    begins after establishing a “now” reference timestamp from the CRS tuber context, then discards
    any packets older than that timestamp to ensure consistent time ordering.

    If `noise` is not provided, an initial noise estimate is computed from a short pre-capture
    packet grab (currently ~1000 packets) using `get_noise_level(...)`. The estimated noise is then
    passed into `PulseCapture` along with the configured thresholding parameters.

    Notes / assumptions:
      - Module indexing depends on the active analog bank. If the analog bank is enabled, the
        requested `module` must be > 4 and is mapped by subtracting 4. If disabled, `module` must
        be <= 4.
      - A valid and recent timestamp source is required (`ts.recent` must be True).
      - A small timestamp offset (~20 ms) is applied before capture to avoid edge effects seen in
        FIR6 experiments.
      - Packets are filtered by CRS serial number (warns if mismatched) and by module.
      - Packets older than the reference timestamp are ignored.
      - This function uses asyncio with a non-blocking multicast socket; packet receive operations
        are bounded by `streamer.STREAMER_TIMEOUT`.

    Args:
        crs (CRS):
            The CRS device handle providing access to tuber context, decimation, serial number,
            and hostname information.
        channel (None | int | List[int], optional):
            Channel(s) to monitor. If None, uses the full channel range based on current
            decimation: channels 1..128 when decimation < 4, otherwise channels 1..1024.
            (***** CURRENT VERSION ONLY RUNS FOR A SINGLE CHANNEL - Work in Progress *****)
        module (int):
            1-indexed module selection from the user’s perspective. 
        time_run (float, keyword-only):
            Capture duration in seconds. Packets are received until this wall-clock interval
            elapses.
        thresh (float, keyword-only):
            Detection threshold used by `PulseCapture` / `get_noise_level`. Interpretation depends
            on `thresh_type`.
        noise (float | None, keyword-only):
            Noise level for pulse detection. If None, a noise estimation step is run using an
            initial grab of packets.
        buf_size (int, keyword-only):
            Internal buffer size passed to `PulseCapture` for maintaining sample history and
            pulse extraction.
        thresh_type (str, keyword-only):
            Thresholding domain/type indicator. Either "I" or "Q".
        in_volts (bool, keyword-only):
            Placeholder for optional volts conversion. (**** Currently not applied in this routine. ****)
        _extra_metadata (bool, keyword-only):
            Placeholder for including additional metadata in output. (**** Currently not applied.****)

    Returns:
        Tuple[float, Any]:
            (pcap.start_time, pcap.pulses)
            - start_time: The monotonic (or capture-defined) start time recorded by PulseCapture.
            - pulses: The returned `pulses` object is a dictionary indexed by pulse number. 
            Each entry contains windowed waveform data for a detected pulse with the following fields:
                  - "Amp_I": numpy array of in-phase (I) samples
                  - "Amp_Q": numpy array of quadrature (Q) samples
                  - "Time": numpy array of timestamps aligned with the I/Q samples
    """
    
    ##### I don't think it has to be killed because of dropped packets unlike Noise Spectrum, but we can maybe keep a drop counter ######
    ##### Not inserting any receiver_attempt loops here, as it is in py_get_samples #####
    
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

    # Math on timestamps only works if they are valid
    assert ts.recent, "Timestamp wasn't recent - do you have a valid timestamp source?"

    ts = streamer.Timestamp(**vars(ts))
    ts.ss += np.uint32(.02 * streamer.SS_PER_SECOND) # 20ms, per experiments at FIR6
    ts.renormalize()

    dec = await crs.get_decimation()
    if channel is None:
        if dec < 4:
            channels = np.arange(1,129)
        else:
            channels = np.arange(1,1024)
    else:
        channels = channel if isinstance(channel, list) else [channel]
            
    print("The host is", crs.tuber_hostname)
    if crs.tuber_hostname == "rfmux0000.local":
        host = '127.0.0.1'
    else:
        host = crs.tuber_hostname


    ##### Add noise level estimate here #####
    ### Taking only 1000 noise samples, will fail if the burst is 1000 samples wide. Find ideal strategy ###

    if noise is None:
        with streamer.get_multicast_socket(host) as sock:
    
            # To use asyncio, we need a non-blocking socket
            loop = asyncio.get_running_loop()
            sock.setblocking(False)
            noise_packets = []
    
            while len(noise_packets) <= 1000: 
    
                data = await asyncio.wait_for(
                    loop.sock_recv(sock, streamer.LONG_PACKET_SIZE),
                    streamer.STREAMER_TIMEOUT,
                )
                p = streamer.ReadoutPacket(data)
                noise_packets.append(p)
    
    
        noise_level, noise_data = get_noise_level(noise_packets, thresh, channels, thresh_type)
    
        noise = noise_level[0] ### running it only for one channel now ###
    
        print("Updated the noise to", noise)
            
    pcap = PulseCapture(buf_size, channels, thresh, noise, thresh_type)
    with streamer.get_multicast_socket(host) as sock:

        # To use asyncio, we need a non-blocking socket
        loop = asyncio.get_running_loop()
        sock.setblocking(False)

        start = time.monotonic()
        print("Starting capture at socket", sock)
        # Start receiving packets
        while time.monotonic() - start < time_run:
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

                pcap._update_buffer_and_capture(p)


    return pcap.start_time, pcap.pulses, noise_data