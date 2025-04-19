#!/usr/bin/env python3
"""
emulator_runner.py

Run Periscope in “emulation mode” by injecting synthetic DfmuxPacket data
into its queue, rather than reading from the network.

Modifications:
--------------
1) Up to 15 channels.
2) Channel 1 -> Distinct pink noise arrays for I and Q.
3) Other channels -> DC offset in IQ space, plus
   - Odd channels > single sine wave
   - Even channels > sum of two sine waves
   - Random noise (Gaussian)
"""

import threading
import time
import datetime
import math
import array
import argparse
import socket
import random

import numpy as np
from PyQt6 import QtWidgets

import rfmux.streamer as streamer
import rfmux.tools.periscope as ps
from rfmux.streamer import (
    DfmuxPacket,
    Timestamp,
    TimestampPort,
    STREAMER_MAGIC,
    STREAMER_VERSION,
    NUM_CHANNELS,
    SS_PER_SECOND,
)


# ──────────────────────────────────────────────────────────────────────────
# 1/f (Pink) noise generator (Voss/McCartney)
# ──────────────────────────────────────────────────────────────────────────
def generate_pink_noise(N=100_000) -> np.ndarray:
    """
    Generate an array of pink (1/f) noise using a simple multi‑bin method.
    Range is approximately [-1, +1].
    """
    n_levels = int(math.ceil(math.log2(N)))
    levels = [0.0] * n_levels
    pink = np.zeros(N, dtype=float)

    running_sum = 0.0
    for i in range(N):
        # Identify which bit is changing
        changed_bit = (i & -i).bit_length() - 1
        if changed_bit >= 0:
            levels[changed_bit] = random.uniform(-1, 1)
        running_sum = sum(levels)
        pink[i] = running_sum

    # Normalize amplitude ~[-1, +1]
    pink /= np.abs(pink).max()
    return pink


# ──────────────────────────────────────────────────────────────────────────
# Dummy socket to prevent real network usage.
# ──────────────────────────────────────────────────────────────────────────
class DummySock:
    def settimeout(self, t):
        pass
    def recv(self, _):
        raise socket.timeout
    def close(self):
        pass

streamer.get_multicast_socket = lambda host: DummySock()


# ──────────────────────────────────────────────────────────────────────────
# Emulation injector
# ──────────────────────────────────────────────────────────────────────────
class EmuInjector(threading.Thread):
    """Generate and inject synthetic DfmuxPacket objects at a fixed rate."""

    def __init__(self, queue, module: int, rate_hz: float = 50.0, channels=(1,2,3,4,5,6,7,8)):
        super().__init__(daemon=True)
        self.queue      = queue
        self.module     = module
        self.rate_hz    = rate_hz
        self.channels   = channels
        self.keep_running = True
        self.seq        = 0

        # Pink noise arrays for CH=1: distinct I, Q
        self._pink_I = generate_pink_noise(N=200_000)
        self._pink_Q = generate_pink_noise(N=200_000)
        self._pink_i_idx = 0
        self._pink_q_idx = 0

        # Initialize random seeds for channel param generation
        self._chan_params = {}
        for ch in self.channels:
            if ch == 1:
                continue  # pink noise -> no param
            else:
                self._chan_params[ch] = self.init_signals(ch)

    def init_signals(self, ch: int):
        """
        Create random parameters for each channel.
        - DC offset for I and Q
        - 1 or 2 sine waves (odd=1, even=2)
        - random noise amplitude
        """
        random.seed(ch * 513)  # deterministic but unique per channel

        n_waves = 2 if (ch % 2 == 0) else 1
        waves = []
        for _ in range(n_waves):
            amp   = random.uniform(500, 3000)
            freq  = random.uniform(0.2, 3.0)   # in Hz
            phase = random.uniform(0, 2*math.pi)
            waves.append((amp, freq, phase))

        noise_amp = random.uniform(0, 300)

        # random DC offset in IQ space
        dc_i = random.uniform(-800, 800)
        dc_q = random.uniform(-800, 800)

        return {
            "waves"     : waves,
            "noise_amp" : noise_amp,
            "dc_i"      : dc_i,
            "dc_q"      : dc_q,
        }

    def get_ch1_sample(self) -> (int, int):
        """
        Distinct pink noise for I and Q, each scaled to ~[-1000, +1000].
        """
        val_i = self._pink_I[self._pink_i_idx]
        val_q = self._pink_Q[self._pink_q_idx]
        self._pink_i_idx = (self._pink_i_idx + 1) % len(self._pink_I)
        self._pink_q_idx = (self._pink_q_idx + 1) % len(self._pink_Q)

        i_val = int(val_i * 1000)
        q_val = int(val_q * 1000)
        return i_val, q_val

    def get_other_ch_sample(self, ch: int, t_now: float) -> (int, int):
        """
        For channels != 1: DC offset + sine wave(s) + random noise
        """
        cp = self._chan_params[ch]
        total_i = cp["dc_i"]
        total_q = cp["dc_q"]

        # Add each wave
        for (amp, freq, phase) in cp["waves"]:
            angle = 2 * math.pi * freq * t_now + phase
            total_i += amp * math.sin(angle)
            total_q += amp * math.cos(angle)

        # Add random noise
        noise_i = random.gauss(0, cp["noise_amp"])
        noise_q = random.gauss(0, cp["noise_amp"])
        total_i += noise_i
        total_q += noise_q

        return int(total_i), int(total_q)

    def make_packet(self) -> DfmuxPacket:
        """
        Create a synthetic DfmuxPacket with the current time stamp.
        """
        # Basic packet metadata
        magic       = STREAMER_MAGIC
        version     = STREAMER_VERSION
        serial      = 0
        num_modules = 1
        block       = 0
        fir_stage   = 0
        module_idx  = self.module - 1
        seq         = self.seq
        self.seq    = (self.seq + 1) & 0xFFFFFFFF

        # Prepare data array: NUM_CHANNELS*2
        body = array.array("i", [0] * (NUM_CHANNELS * 2))
        t_now = time.time()

        for ch in self.channels:
            idx = 2 * (ch - 1)
            if ch == 1:
                ival, qval = self.get_ch1_sample()
            else:
                ival, qval = self.get_other_ch_sample(ch, t_now)

            body[idx]     = ival
            body[idx + 1] = qval

        # Build timestamp from system clock
        dt = datetime.datetime.now(datetime.timezone.utc)
        y = dt.year % 100
        d = dt.timetuple().tm_yday
        h, m, s = dt.hour, dt.minute, dt.second
        ss = int(dt.microsecond * SS_PER_SECOND / 1e6)
        c = 0
        sbs = 0
        ts = Timestamp(
            y=y, d=d, h=h, m=m, s=s, ss=ss,
            c=c, sbs=sbs,
            source=TimestampPort.GND,
            recent=True,
        )

        return DfmuxPacket(
            magic=magic,
            version=version,
            serial=serial,
            num_modules=num_modules,
            block=block,
            fir_stage=fir_stage,
            module=module_idx,
            seq=seq,
            s=body,
            ts=ts,
        )

    def run(self):
        interval = 1.0 / self.rate_hz
        while self.keep_running:
            pkt = self.make_packet()
            self.queue.put(pkt)
            time.sleep(interval)

    def stop(self):
        self.keep_running = False


# ──────────────────────────────────────────────────────────────────────────
# Main: launch Periscope + emulator
# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Run Periscope with a local data emulator."
    )
    ap.add_argument("hostname", help="Ignored in emulation.")
    ap.add_argument("-m", "--module", type=int, default=1,
                    help="Module number (1‑based).")
    ap.add_argument("-c", "--channels", default="1",
                    help="Comma list of channels up to 15, e.g. '1,2,3,4,5'.")
    ap.add_argument("-r", "--rate", type=float, default=50.0,
                    help="Emulation packet rate (Hz).")
    args, extras = ap.parse_known_args()

    # 1) Launch Periscope non‑blocking
    viewer, app = ps.launch(
        hostname=args.hostname,
        module=args.module,
        channels=args.channels,
        fps=30.0,
        blocking=False
    )

    # 2) Stop its real UDPReceiver immediately
    viewer.receiver.stop()
    viewer.receiver.wait()

    # 3) Start our synthetic-data injector thread
    #    which writes into the same queue that Periscope reads from.
    channel_list = []
    for tok in args.channels.split(","):
        tok = tok.strip()
        if tok.isdigit():
            ch_val = int(tok)
            if 1 <= ch_val <= 15:
                channel_list.append(ch_val)
    channel_list = channel_list or [1]  # default: just channel 1 if none parse

    injector = EmuInjector(
        queue=viewer.receiver.queue,
        module=args.module,
        rate_hz=args.rate,
        channels=channel_list
    )
    injector.start()

    # 4) Run the Qt loop
    try:
        app.exec()
    finally:
        injector.stop()
        injector.join()


if __name__ == "__main__":
    main()
