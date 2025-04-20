#!/usr/bin/env python3
"""
emulator_runner.py

Run Periscope in “emulation mode” by injecting synthetic DfmuxPacket data
into its queue, rather than reading from the network.

Modifications:
--------------
1) Up to 15 channels.
2) Channel 1 -> Distinct pink noise arrays for I and Q, plus random DC offset (>100).
3) Other channels -> DC offset (>100) + 1 or 2 sine waves + random Gaussian noise.
4) All signals can be negative or positive; DC offset has random phase.
5) All signals have some random noise component.
6) Sampling rate derived from decimation stage (1..6), default=6, ~595 Hz.
7) Packet timestamps reflect this exact sampling rate (no drift).
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
# Sampling rate helper
# ──────────────────────────────────────────────────────────────────────────
def decimation_to_sampling(dec: int) -> float:
    """
    Return sampling rate (Hz) for a given decimation stage in [1..6].
    Formula: 625e6 / 256 / 64 / 2^dec
    """
    return 625e6 / 256.0 / 64.0 / (2.0 ** dec)


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
    """Generate and inject synthetic DfmuxPacket objects at a fixed sampling rate."""

    def __init__(self, queue, module: int, dec: int = 6, channels=(1,2,3,4,5,6,7,8)):
        super().__init__(daemon=True)
        self.queue         = queue
        self.module        = module
        self.dec           = max(1, min(6, dec))  # clamp dec between 1..6
        self.samp_rate_hz  = decimation_to_sampling(self.dec)
        self.keep_running  = True
        self.seq           = 0
        self.channels      = channels

        # Pink noise arrays for CH=1: distinct I, Q
        self._pink_I = generate_pink_noise(N=200_000)
        self._pink_Q = generate_pink_noise(N=200_000)
        self._pink_i_idx = 0
        self._pink_q_idx = 0

        # Random DC offset (>100) for channel 1
        mag_1 = random.uniform(100, 800)  # ensure DC "power" > 100
        phase_1 = random.uniform(0, 2*math.pi)
        self._ch1_dc_i = mag_1 * math.cos(phase_1)
        self._ch1_dc_q = mag_1 * math.sin(phase_1)

        # Initialize random seeds & params for channels != 1
        self._chan_params = {}
        for ch in self.channels:
            if ch != 1:
                self._chan_params[ch] = self.init_signals(ch)

        # For perfect time increments at the computed sampling rate:
        self._sample_counter = 0
        self._t_start = time.time()

    def init_signals(self, ch: int):
        """
        Create random parameters for each channel:
        - DC offset in IQ space with magnitude > 100
        - 1 sine wave if odd channel, 2 if even
        - random Gaussian noise amplitude
        """
        random.seed(ch * 513)  # deterministic but unique per channel

        # DC offset: random magnitude > 100, random phase
        mag  = random.uniform(100, 800)
        phase = random.uniform(0, 2*math.pi)
        dc_i = mag * math.cos(phase)
        dc_q = mag * math.sin(phase)

        # Sine waves
        n_waves = 3 if (ch % 2 == 0) else 1
        waves = []
        for _ in range(n_waves):
            amp   = random.uniform(500, 3000)    # amplitude
            freq  = random.uniform(-300, 300)     # frequency in Hz
            wph   = random.uniform(0, 2*math.pi) # phase
            waves.append((amp, freq, wph))

        noise_amp = random.uniform(0, 300)

        return {
            "dc_i"      : dc_i,
            "dc_q"      : dc_q,
            "waves"     : waves,
            "noise_amp" : noise_amp
        }

    def get_ch1_sample(self) -> (int, int):
        """
        Distinct pink noise for I and Q, scaled to ~[-1000, +1000],
        plus a DC offset ensuring power > 100.
        """
        val_i = self._pink_I[self._pink_i_idx]
        val_q = self._pink_Q[self._pink_q_idx]
        self._pink_i_idx = (self._pink_i_idx + 1) % len(self._pink_I)
        self._pink_q_idx = (self._pink_q_idx + 1) % len(self._pink_Q)

        # Scale pink noise to ~[-1000, +1000]
        i_val = val_i * 1000
        q_val = val_q * 1000

        # Add DC offset
        i_val += self._ch1_dc_i
        q_val += self._ch1_dc_q

        return int(i_val), int(q_val)

    def get_other_ch_sample(self, ch: int, t_now: float) -> (int, int):
        """
        For channels != 1: DC offset + (1 or 2) sine waves + random Gaussian noise
        """
        cp = self._chan_params[ch]
        total_i = cp["dc_i"]
        total_q = cp["dc_q"]

        # Add sine wave(s)
        for (amp, freq, ph) in cp["waves"]:
            angle = 2 * math.pi * freq * t_now + ph
            # interpret I as sine, Q as cosine
            total_i += amp * math.cos(angle)
            total_q += amp * math.sin(angle)

        # Add Gaussian noise
        noise_i = random.gauss(0, cp["noise_amp"])
        noise_q = random.gauss(0, cp["noise_amp"])
        total_i += noise_i
        total_q += noise_q

        return int(total_i), int(total_q)

    def make_packet(self) -> DfmuxPacket:
        """
        Create a synthetic DfmuxPacket with a timestamp reflecting
        the exact sampling interval (1 / samp_rate_hz).
        """
        magic       = STREAMER_MAGIC
        version     = STREAMER_VERSION
        serial      = 0
        num_modules = 1
        block       = 0
        fir_stage   = 0
        module_idx  = self.module - 1
        seq         = self.seq
        self.seq    = (self.seq + 1) & 0xFFFFFFFF

        # Compute ideal sampling time for this sample index
        t_samp = self._t_start + self._sample_counter / self.samp_rate_hz
        self._sample_counter += 1

        # Prepare data array: NUM_CHANNELS*2
        body = array.array("i", [0] * (NUM_CHANNELS * 2))

        for ch in self.channels:
            idx = 2 * (ch - 1)
            if ch == 1:
                i_val, q_val = self.get_ch1_sample()
            else:
                i_val, q_val = self.get_other_ch_sample(ch, t_samp)
            body[idx]     = i_val
            body[idx + 1] = q_val

        # Convert t_samp to a UTC datetime
        dt = datetime.datetime.fromtimestamp(t_samp, tz=datetime.timezone.utc)
        y = dt.year % 100
        d = dt.timetuple().tm_yday
        h, m, s = dt.hour, dt.minute, dt.second
        # Convert microseconds to sub‑second ticks
        ss = int(dt.microsecond * SS_PER_SECOND / 1e6)

        # Additional fields for the custom Timestamp
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
        # We'll pace the emission to match the actual sampling rate in real time
        interval = 1.0 / self.samp_rate_hz
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
    ap.add_argument("-d", "--dec", type=int, default=6,
                    help="Decimation stage (1..6). Determines sampling rate.")
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
        dec=args.dec,
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
