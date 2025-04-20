#!/usr/bin/env python3
"""
emulator_runner.py
==================

Standalone test fixture that feeds *Periscope* with synthetic **DfmuxPacket**
objects instead of real UDP multicast traffic.  This is invaluable for GUI
debugging, CI, and offline development where hardware is unavailable.

Signal Model
------------
* **Channel 1** – independent pink‑noise (1/f) for *I* and *Q*, plus a random
  DC vector (‖DC‖ > 100).
* **Channels 2‥15** – one or two sinewaves, random DC vector (‖DC‖ > 100),
  and added Gaussian noise.

Timing
------
A perfect sampling interval *Δt = 1 / f_s* is derived from the decimation
stage::

    f_s(dec) = (625 MHz / 256 / 64) / 2**dec   # dec ∈ [1, 6]

Timestamps are generated from a monotonic counter so that Periscope’s
decimation‑stage inference remains accurate.

Usage Example
-------------
    $ python emulator_runner.py ignored_host -m 1 -c 1,2,3,4 -d 6

The *hostname* argument is ignored but preserved so that the CLI mirrors the
real `rfmux.tools.periscope` entry‑point.
"""

import argparse
import array
import datetime as _dt
import math
import queue
import random
import socket
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import rfmux.streamer as streamer
import rfmux.tools.periscope as ps
from rfmux.streamer import (
    DfmuxPacket,
    NUM_CHANNELS,
    SS_PER_SECOND,
    STREAMER_LEN,
    STREAMER_MAGIC,
    STREAMER_VERSION,
    Timestamp,
    TimestampPort,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLING_BASE_HZ: float = 625e6 / 256.0 / 64.0  # ≈ 38 147.46 Hz for dec=0
MAX_DEC_STAGE: int = 6
DEFAULT_DEC_STAGE: int = 6
MAX_USER_CHANNEL: int = 15            # Physical hardware limit for this fixture
MIN_DC_MAG: float = 100.0
DC_MAG_RANGE: Tuple[float, float] = (MIN_DC_MAG, 800.0)
SINE_AMP_RANGE: Tuple[float, float] = (500.0, 3000.0)
SINE_FREQ_RANGE: Tuple[float, float] = (-300.0, 300.0)  # Hz
GAUSS_NOISE_RANGE: Tuple[float, float] = (0.0, 300.0)

# ---------------------------------------------------------------------------
# Dummy socket patch (avoids real network traffic)
# ---------------------------------------------------------------------------


class _DummySock:
    """Bare‑minimum socket object satisfying the interface used by UDPReceiver."""

    def settimeout(self, _t: float) -> None:  # noqa: D401
        pass

    def recv(self, _size: int) -> bytes:  # noqa: D401
        raise socket.timeout

    def close(self) -> None:  # noqa: D401
        pass


# Monkey‑patch once at import‑time
streamer.get_multicast_socket = lambda _host: _DummySock()  # type: ignore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def decimation_to_sampling(dec: int) -> float:
    """
    Convert decimation stage to sampling frequency ``f_s``.

    Parameters
    ----------
    dec
        Decimation stage ∈ [1, 6].

    Returns
    -------
    float
        Sampling rate in Hertz.
    """
    dec = int(dec)
    if dec < 1 or dec > MAX_DEC_STAGE:
        raise ValueError(f"decimation must be 1 … {MAX_DEC_STAGE}")
    return SAMPLING_BASE_HZ / (2.0 ** dec)


class PinkNoise:
    """
    Fast reusable 1/f noise generator (Voss‑McCartney).

    Notes
    -----
    * Uses a fixed internal buffer for performance; indexes wrap around.
    * Thread‑safe as long as each instance is confined to a single thread.
    """

    def __init__(self, size: int = 200_000, *, seed: int | None = None) -> None:
        self._size = size
        self._rng = random.Random(seed)
        self._levels: List[float] = [0.0] * int(math.ceil(math.log2(size)))
        self._buf = np.zeros(size, dtype=np.float32)
        self._fill()
        # Normalise to ±1
        self._buf /= np.abs(self._buf).max()
        self._idx: int = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def next(self) -> float:
        """Return the next pink‑noise sample (range ≈ ±1)."""
        val = float(self._buf[self._idx])
        self._idx = (self._idx + 1) % self._size
        return val

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _fill(self) -> None:
        running_sum = 0.0
        for i in range(self._size):
            bit = (i & -i).bit_length() - 1
            if bit >= 0:
                self._levels[bit] = self._rng.uniform(-1.0, 1.0)
            running_sum = sum(self._levels)
            self._buf[i] = running_sum


# ---------------------------------------------------------------------------
# Emulator injector thread
# ---------------------------------------------------------------------------


class EmuInjector(threading.Thread):
    """
    Background thread that pushes synthetic **DfmuxPacket** objects into the
    *UDPReceiver* queue at the exact nominal sampling cadence.

    Parameters
    ----------
    pkt_queue
        Thread‑safe queue belonging to the real Periscope UDPReceiver.
    module
        Module number (1‑based) to encode in packets.
    dec
        Decimation stage (1…6).  Determines *f_s*.
    channels
        Iterable of channel indices (1…15).  Values > 15 are silently ignored.
    """

    def __init__(
        self,
        pkt_queue: "queue.Queue[DfmuxPacket]",
        *,
        module: int = 1,
        dec: int = DEFAULT_DEC_STAGE,
        channels: Sequence[int] = (1,),
    ) -> None:
        super().__init__(daemon=True)
        self._queue = pkt_queue
        self._module = int(module)
        self._dec = max(1, min(int(dec), MAX_DEC_STAGE))
        self._fs_hz = decimation_to_sampling(self._dec)
        self._channels: Tuple[int, ...] = tuple(
            ch for ch in channels if 1 <= ch <= MAX_USER_CHANNEL
        ) or (1,)

        # Per‑channel signal parameters ------------------------------------
        self._pink_I = PinkNoise(seed=1234)
        self._pink_Q = PinkNoise(seed=5678)
        self._pink_lock = threading.Lock()  # protects index wrap‑around

        self._ch1_dc_i, self._ch1_dc_q = self._random_dc_vector()

        # deterministic per‑channel RNGs so that repeated runs are stable
        self._chan_rng: Dict[int, random.Random] = {
            ch: random.Random(ch * 513) for ch in self._channels if ch != 1
        }
        self._chan_params: Dict[int, Dict[str, object]] = {
            ch: self._init_other_channel(ch) for ch in self._channels if ch != 1
        }

        # Timebase ----------------------------------------------------------
        self._t0_monotonic = time.monotonic()
        self._sample_idx: int = 0
        self._running = threading.Event()
        self._running.set()

    # ------------------------------------------------------------------ #
    # Thread life‑cycle
    # ------------------------------------------------------------------ #

    def run(self) -> None:  # noqa: D401
        """Main producer loop – exits when :py:meth:`stop` is called."""
        dt = 1.0 / self._fs_hz
        try:
            while self._running.is_set():
                pkt = self._make_packet()
                self._queue.put(pkt)
                time.sleep(dt)
        except KeyboardInterrupt:
            # Allow Ctrl‑C during development to terminate quickly
            self._running.clear()

    def stop(self) -> None:
        """Request orderly shutdown."""
        self._running.clear()

    # ------------------------------------------------------------------ #
    # Channel initialisation helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _random_dc_vector() -> Tuple[float, float]:
        mag = random.uniform(*DC_MAG_RANGE)
        phase = random.uniform(0.0, 2.0 * math.pi)
        return mag * math.cos(phase), mag * math.sin(phase)

    def _init_other_channel(self, ch: int) -> Dict[str, object]:
        rng = self._chan_rng[ch]
        dc_i, dc_q = self._random_dc_vector()

        # -------------------------------------------------------------- #
        # Number of sine components per channel 
        # * **Channels 2–4** – deterministic number of sinewaves   
        #   (Ch 2 → 1, Ch 3 → 2, Ch 4 → 3), random DC vector, Gaussian noise. 
        # * **Channels 5–15** – *random* 1–3 sinewaves, random DC vector, 
        #   Gaussian noise.
        # -------------------------------------------------------------- #
        if 2 <= ch <= 4:
            n_waves = ch - 1
        else:
            n_waves = rng.randint(1, 25)
        waves = [
            (
                rng.uniform(*SINE_AMP_RANGE),
                rng.uniform(*SINE_FREQ_RANGE),
                rng.uniform(0.0, 2.0 * math.pi),
            )
            for _ in range(n_waves)
        ]
        noise_amp = rng.uniform(*GAUSS_NOISE_RANGE)
        return dict(dc_i=dc_i, dc_q=dc_q, waves=waves, noise_amp=noise_amp)

    # ------------------------------------------------------------------ #
    # Sample synthesis
    # ------------------------------------------------------------------ #

    def _ch1_sample(self) -> Tuple[int, int]:
        """Return one *I,Q* pair for channel 1."""
        with self._pink_lock:
            i_val = self._pink_I.next()
            q_val = self._pink_Q.next()
        i_val = i_val * 1_000.0 + self._ch1_dc_i
        q_val = q_val * 1_000.0 + self._ch1_dc_q
        return int(i_val), int(q_val)

    def _other_sample(self, ch: int, t_now: float) -> Tuple[int, int]:
        prm = self._chan_params[ch]
        total_i: float = prm["dc_i"]  # type: ignore[assignment]
        total_q: float = prm["dc_q"]  # type: ignore[assignment]
        for amp, freq, ph in prm["waves"]:  # type: ignore[misc]
            angle = 2.0 * math.pi * freq * t_now + ph
            total_i += amp * math.cos(angle)
            total_q += amp * math.sin(angle)
        rng = self._chan_rng[ch]
        total_i += rng.gauss(0.0, prm["noise_amp"])  # type: ignore[arg-type]
        total_q += rng.gauss(0.0, prm["noise_amp"])  # type: ignore[arg-type]
        return int(total_i), int(total_q)

    # ------------------------------------------------------------------ #
    # Timestamp helper
    # ------------------------------------------------------------------ #

    def _next_sample_time(self) -> float:
        """
        Ideal absolute UTC timestamp for the next sample, *including* the
        initial offset from monotonic time.  Ensures long‑term drift < 1 sample.
        """
        t_rel = self._sample_idx / self._fs_hz
        self._sample_idx += 1
        return time.time() + (t_rel - (time.monotonic() - self._t0_monotonic))

    # ------------------------------------------------------------------ #
    # Packet builder
    # ------------------------------------------------------------------ #

    def _make_packet(self) -> DfmuxPacket:
        t_abs = self._next_sample_time()
        dt_utc = _dt.datetime.fromtimestamp(t_abs, tz=_dt.timezone.utc)

        # ------------------------------------------------------------------
        # Header fields
        # ------------------------------------------------------------------
        seq = self._sample_idx & 0xFFFFFFFF
        body = array.array("i", [0] * (NUM_CHANNELS * 2))

        # ------------------------------------------------------------------
        # Per‑channel sample generation
        # ------------------------------------------------------------------
        for ch in self._channels:
            idx = 2 * (ch - 1)
            if ch == 1:
                i_val, q_val = self._ch1_sample()
            else:
                i_val, q_val = self._other_sample(ch, t_abs)
            body[idx] = i_val
            body[idx + 1] = q_val

        # ------------------------------------------------------------------
        # Timestamp → CRS format
        # ------------------------------------------------------------------
        y, d = dt_utc.year % 100, dt_utc.timetuple().tm_yday
        h, m, s = dt_utc.hour, dt_utc.minute, dt_utc.second
        ss = int(dt_utc.microsecond * SS_PER_SECOND / 1e6)
        ts = Timestamp(
            y=y,
            d=d,
            h=h,
            m=m,
            s=s,
            ss=ss,
            c=0,
            sbs=0,
            source=TimestampPort.GND,
            recent=True,
        )

        return DfmuxPacket(
            magic=STREAMER_MAGIC,
            version=STREAMER_VERSION,
            serial=0,
            num_modules=1,
            block=0,
            fir_stage=0,
            module=self._module - 1,
            seq=seq,
            s=body,
            ts=ts,
        )


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _parse_channels(txt: str) -> Tuple[int, ...]:
    """
    Convert comma‑separated channel string into a sorted tuple of unique ints.

    Raises
    ------
    argparse.ArgumentTypeError
        If *txt* is malformed.
    """
    chans: List[int] = []
    for tok in txt.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if not tok.isdigit():
            raise argparse.ArgumentTypeError(f"invalid channel: {tok!r}")
        val = int(tok)
        if not (1 <= val <= MAX_USER_CHANNEL):
            raise argparse.ArgumentTypeError(
                f"channel {val} out of range 1…{MAX_USER_CHANNEL}"
            )
        chans.append(val)
    return tuple(sorted(set(chans))) or (1,)


# ---------------------------------------------------------------------------
# Main launcher
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: D401
    """Command‑line entry point."""
    ap = argparse.ArgumentParser(description="Run Periscope with synthetic data.")
    ap.add_argument("hostname", help="Ignored – kept for CLI symmetry.")
    ap.add_argument("-m", "--module", type=int, default=1)
    ap.add_argument("-c", "--channels", default="1", help="Comma list up to 15.")
    ap.add_argument(
        "-d", "--dec", type=int, default=DEFAULT_DEC_STAGE, choices=range(1, 7)
    )
    args, _extras = ap.parse_known_args()

    chan_tuple = _parse_channels(args.channels)

    # ------------------------------------------------------------------ #
    # 1) Launch Periscope (non‑blocking)
    # ------------------------------------------------------------------ #
    viewer, qt_app = ps.launch(
        hostname=args.hostname,
        module=args.module,
        channels=",".join(map(str, chan_tuple)),
        fps=30.0,
        blocking=False,
    )

    # ------------------------------------------------------------------ #
    # 2) Replace its real UDPReceiver with our injector
    # ------------------------------------------------------------------ #
    viewer.receiver.stop()
    viewer.receiver.wait()

    injector = EmuInjector(
        pkt_queue=viewer.receiver.queue,
        module=args.module,
        dec=args.dec,
        channels=chan_tuple,
    )
    injector.start()

    # ------------------------------------------------------------------ #
    # 3) Enter Qt event‑loop
    # ------------------------------------------------------------------ #
    try:
        qt_app.exec()
    finally:
        injector.stop()
        injector.join()


if __name__ == "__main__":
    main()
