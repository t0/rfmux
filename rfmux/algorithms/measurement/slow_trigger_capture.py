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

from ...core.hardware_map import macro
from ...core.schema import CRS
from ...tuber.codecs import TuberResult
from ...core.transferfunctions import VOLTS_PER_ROC, spectrum_from_slow_tod
from ... import streamer
from rfmux.tools.periscope.utils import Circular

''' 
Remember a buffer is FIFO. So if you got a pulse at 9900 for a buffer of size 10000 and it has a width of 200 sample.
The burst will be written as follows - [9900:10000] and then [0:100].
You have ovewritten the first 100 values in the buffer, so make sure you have saved that info.

Also the buffer used by periscope returns already FIFO ordered data, we do not have to touch it again, we just have to figure out the correct indices to call. 
'''

# class PulseCapture():

#     def __init__(self, buf_size, channels, thresh, noise_level):
        
#         self.buf = {}
#         self.channels = channels
#         self.buf_size = buf_size
        
#         for c in self.channels:
#             self.buf[c] = {k: Circular(buf_size) for k in ("I", "Q", "ts")}
            
#         self.thresh = thresh
#         self.noise_level = noise_level
        
#         self.start_time = None
        
#         self.pulses = {}
#         self.pulse_count = 0
#         self.capturing = False
#         self.trigger_ptr = None
#         self.end_ptr_count = 0

#         self.abs_n = 0          # absolute sample counter (monotonic)
#         self.trig_abs = None

#     def _calculate_relative_timestamp(self, pkt) -> float | None:
#         """
#         Calculate a relative timestamp for a packet.

#         If the packet's timestamp is recent, it's adjusted slightly and
#         converted to seconds relative to the first packet's timestamp.

#         Args:
#             pkt: The incoming packet object.

#         Returns:
#             float | None: Relative timestamp in seconds, or None if not recent.
#         """
#         # streamer from .utils
#         ts = pkt.ts
#         if ts.recent:
#             # Apply a small offset to ensure timestamps are strictly increasing for plotting
#             ts.ss += int(0.02 * streamer.SS_PER_SECOND); ts.renormalize()
#             t_now = ts.h * 3600 + ts.m * 60 + ts.s + ts.ss / streamer.SS_PER_SECOND
#             if self.start_time is None: self.start_time = t_now
#             return t_now - self.start_time


#     def _ordered(self, circ):
#         '''
#         We need this, since the lesser value of pointer points to an older value of time in the buffer.
#         Say we started dump while data is being parsed. So time pointer value at 120 is 15 seconds. 
#         '''
#         d = circ.data()
#         p = circ.ptr
#         return np.concatenate([d[p:], d[:p]]) ## Making sure the time data gets ordered ##


#     # def _calculate_noise_level(self, samples):
        


#     def _update_buffer_and_capture(self, pkt):
#         samples = pkt.samples / 256
#         seq = pkt.seq
#         ts = pkt.ts
#         arr_time = self._calculate_relative_timestamp(pkt)
    
#         for c in self.channels: 
#             sample = samples[c-1]
    
#             self.buf[c]["I"].add(sample.real)
#             self.buf[c]["Q"].add(sample.imag)
#             self.buf[c]["ts"].add(arr_time)

#             ptr = self.buf[c]["I"].ptr

#             self.abs_n += 1

#             # if self.thresh is not None:
#             if sample.real > self.thresh and not self.capturing:
#                 print("\nFound trigger at ptr", ptr, "at time", arr_time, "Sample value", sample.real)
#                 self.trigger_ptr = ptr - 1
#                 self.capturing = True
#                 self.end_ptr_count = 0
#                 self.trig_abs = self.abs_n

#             if self.capturing:
#                 if sample.real < self.noise_level:
#                     self.end_ptr_count += 1
#                 else:
#                     self.end_ptr_count = 0

#                 if self.end_ptr_count > 10: ### Stop after 10 noise samples

#                     start = self.trigger_ptr - 20
#                     end = ptr

#                     if (start > 1) and (end < self.buf_size): #### This might be missing pulses #####
#                         print("Saving capture at ptr", end, "at time", arr_time, "Sample value", sample.real)
#                         self.pulse_count += 1
                        
#                         I_ord  = self._ordered(self.buf[c]["I"])
#                         ts_ord = self._ordered(self.buf[c]["ts"])

#                         # I_ord = self.buf[c]["I"].data()
#                         # ts_ord = self.buf[c]["ts"].data()

#                         print("Saving Now: Sample value trigger was", I_ord[start+20], "the end value is", I_ord[end-1])
                        
#                         self.pulses[self.pulse_count] = {
#                             "Amp":  I_ord[start:end],
#                             "Time": ts_ord[start:end]
#                         }
    
#                     self.trigger_ptr = None
#                     self.capturing = False
#                     self.end_ptr_count = 0


class PulseCapture:

    def __init__(self, buf_size, channels, thresh, noise_level, pre=20):
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

            # --- trigger detection ---
            if (sample.real > self.thresh) and (not self.capturing):
                self.capturing = True
                self.end_ptr_count = 0
                self.trig_abs = self.abs_n
                self.trigger_value = float(sample.real)

                print("\nFound trigger abs_n", self.trig_abs,
                      "time", arr_time,
                      "Sample value", sample.real)

            # --- end condition and save ---
            if self.capturing:
                if sample.real < self.noise_level:
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
                    L = self.buf[c]["I"].count  # <= buf_size

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
                    ts_win = self._pulse_window(self.buf[c]["ts"], start_clamped, end_clamped)

                    self.pulse_count += 1
                    self.pulses[self.pulse_count] = {
                        "Amp": I_win,
                        "Time": ts_win,
                    }

                    #### Logging capture ####
                    print("Pulse:", self.pulse_count,
                          "Saving Now: trigger sample (window)", I_win[pre_actual],
                          "end sample", I_win[-1],
                          "len", len(I_win),
                          "pre_actual", pre_actual,
                          "trigger sample (at trigger time)", self.trigger_value)

                    # reset
                    self.capturing = False
                    self.end_ptr_count = 0
                    self.trig_abs = None
                    self.trigger_value = None
    

@macro(CRS, register=True)
async def slow_trigger_capture(crs: CRS,
                               channel : Union[None, int, List[int]] = None,
                               module: int = None,
                               *,
                               time_run : float = 10.0,
                               thresh : float = None, 
                               noise : float, ### Will have to initialize and maybe add a script to calculate
                               buf_size : int = 5000,
                               in_volts : bool = False ### add volts conversion  ####
                               _extra_metadata: bool = False):


    ##### I don't think it has to be killed because of dropped packets unlike Noise, but we can maybe keep a drop counter ######
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

    pcap = PulseCapture(buf_size, channels, thresh, noise)
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


    return pcap.start_time, pcap.pulses