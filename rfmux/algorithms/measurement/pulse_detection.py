"""
Streamer-agnostic pulse detection and capture.

This module provides the core pulse detection state machine (``PulseCapture``)
and noise-estimation utility, independent of the packet format or sample rate.
It can be driven by either the slow readout stream or the fast PFB stream.

Triggering is **sigma-based**: for each channel the noise statistics (mean, std)
are computed independently for both I and Q.  A pulse is detected when either
component deviates by more than ``threshold_sigma`` standard deviations from
its baseline mean.  This catches both positive excursions and negative dips on
whichever component carries the larger signal.

Usage::

    # 1. Collect noise samples and compute stats
    noise_stats, noise_data = estimate_noise_stats(samples_by_channel, channels)

    # 2. Create PulseCapture with sigma threshold
    pcap = PulseCapture(
        buf_size=5000,
        channels=[1, 2, 3],
        noise_stats=noise_stats,       # per-channel {mean_I, std_I, mean_Q, std_Q}
        threshold_sigma=5.0,           # trigger at 5σ on EITHER I or Q
        sample_rate=38147.0,
    )

    # 3. Feed samples
    for channel, i_val, q_val, timestamp in sample_stream:
        pcap.process_sample(channel, i_val, q_val, timestamp)

    # 4. Results
    print(pcap.pulses)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# ───────────────────────── Circular Buffer ──────────────────────────

class Circular:
    """Lock-free ring buffer backed by a doubled numpy array."""

    def __init__(self, size: int, dtype=float) -> None:
        self.N = size
        self.buf = np.zeros(size * 2, dtype=dtype)
        self.ptr = 0
        self.count = 0

    def add(self, value):
        self.buf[self.ptr] = value
        self.buf[self.ptr + self.N] = value
        self.ptr = (self.ptr + 1) % self.N
        self.count = min(self.count + 1, self.N)

    def data(self) -> np.ndarray:
        """Return FIFO-ordered view (oldest → newest), length = count."""
        if self.count < self.N:
            return self.buf[: self.count]
        return self.buf[self.ptr : self.ptr + self.N]


# ───────────────────────── Per-Channel Noise Stats ──────────────────

@dataclass
class ChannelNoiseStats:
    """Noise statistics for one channel, computed independently for I and Q."""
    mean_I: float = 0.0
    std_I: float = 1.0
    mean_Q: float = 0.0
    std_Q: float = 1.0


# ───────────────────────── Per-Channel Capture State ────────────────

@dataclass
class _ChState:
    capturing: bool = False
    end_ptr_count: int = 0
    trig_abs: Optional[int] = None
    trigger_value_I: Optional[float] = None
    trigger_value_Q: Optional[float] = None
    ch_sample_n: int = 0  # Per-channel sample counter (for buffer arithmetic)
    warmup_done: bool = False  # True once baseline confirmed after capture start
    warmup_count: int = 0      # Leaky-bucket counter for baseline confirmation


# ───────────────────────── PulseCapture ─────────────────────────────

class PulseCapture:
    """Streaming multi-channel pulse detector with dual I/Q sigma-based triggering.

    For each channel, both I and Q are monitored independently.  A pulse is
    detected when **either** component deviates from its baseline mean by more
    than ``threshold_sigma`` standard deviations.  This catches both positive
    pulses and negative dips on whichever component carries the signal.

    End-of-pulse is declared when **both** I and Q return to within
    ``end_sigma`` (default 1.0) standard deviations for a configurable
    time duration.

    Parameters
    ----------
    buf_size : int
        Circular buffer capacity per channel.
    channels : list[int]
        1-indexed channel numbers to monitor.
    noise_stats : dict[int, ChannelNoiseStats]
        Per-channel noise statistics (mean/std for I and Q).
    threshold_sigma : float
        Number of standard deviations above noise mean to trigger.
    end_sigma : float
        Number of standard deviations — signal must return within this
        to declare pulse end (default 1.0σ).
    sample_rate : float
        Expected sample rate (Hz).
    pre : int
        Pre-trigger samples to include in the pulse window.
    end_samples : int
        Number of samples signal must stay below end_sigma to declare
        pulse over (using leaky-bucket counting).
    """

    def __init__(
        self,
        buf_size: int,
        channels: List[int],
        noise_stats: Dict[int, ChannelNoiseStats],
        threshold_sigma: float = 5.0,
        end_sigma: float = 1.0,
        sample_rate: float = 38147.0,
        pre: int = 20,
        end_samples: int = 100,
    ):
        self.channels = list(channels)
        self.buf_size = buf_size
        self.sample_rate = sample_rate
        self.threshold_sigma = threshold_sigma
        self.end_sigma = end_sigma
        self.pre = pre

        self.end_samples = max(1, end_samples)

        # Per-channel noise stats
        self.noise_stats = noise_stats

        # Per-channel circular buffers
        self.buf: Dict[int, Dict[str, Circular]] = {}
        for c in self.channels:
            self.buf[c] = {k: Circular(buf_size) for k in ("I", "Q", "ts")}

        # Build channel → index lookup
        self._ch_set = set(self.channels)

        # Per-channel capture state
        self.state: Dict[int, _ChState] = {c: _ChState() for c in self.channels}

        # Results
        self.start_time: Optional[float] = None
        self.pulses: Dict[str, dict] = {f"Channel {c}": {} for c in self.channels}
        self.pulse_count: Dict[str, int] = {f"Channel {c}": 0 for c in self.channels}

        # When True, no new triggers start but in-progress captures complete.
        # Set by the receive loop when time_run is reached.
        self.freeze_triggers: bool = False

        # Absolute sample counter
        self.abs_n = 0

    # ── Public API ────────────────────────────────────────────────

    def process_sample(
        self,
        channel: int,
        i_val: float,
        q_val: float,
        timestamp: Optional[float],
    ) -> None:
        """Ingest a single I/Q sample for *channel*.

        Triggers on EITHER I or Q exceeding threshold_sigma from baseline.
        """
        if channel not in self._ch_set:
            return

        self.abs_n += 1

        # Update circular buffers
        self.buf[channel]["I"].add(i_val)
        self.buf[channel]["Q"].add(q_val)
        self.buf[channel]["ts"].add(timestamp)

        # Get noise stats for this channel
        ns = self.noise_stats.get(channel, ChannelNoiseStats())
        st = self.state[channel]
        st.ch_sample_n += 1  # Per-channel counter for buffer arithmetic

        # Compute deviations in sigma units
        dev_I = abs(i_val - ns.mean_I) / max(ns.std_I, 1e-30)
        dev_Q = abs(q_val - ns.mean_Q) / max(ns.std_Q, 1e-30)

        # ── Warmup: confirm baseline before enabling triggers ─────
        # Don't trigger until the signal has been confirmed at baseline.
        # This prevents capturing a pulse already in progress when the
        # capture starts.  Requirements for warmup completion:
        #   1. At least `pre` samples collected (buffer has room)
        #   2. Leaky-bucket baseline counter has reached `end_samples`
        #      (signal has been consistently at baseline)
        if not st.warmup_done:
            # Use threshold_sigma (not end_sigma) for warmup — stronger
            # condition ensures the signal is fully at baseline, not just
            # in the tail of a decaying pulse where it might briefly dip
            # below end_sigma.
            if dev_I < self.threshold_sigma and dev_Q < self.threshold_sigma:
                st.warmup_count += 1
            else:
                st.warmup_count = max(0, st.warmup_count - 1)
            if st.warmup_count > self.end_samples and st.ch_sample_n > self.pre:
                st.warmup_done = True

        # ── Trigger: EITHER I or Q exceeds threshold_sigma ────────
        if (not st.capturing
            and not self.freeze_triggers
            and st.warmup_done
            and (dev_I > self.threshold_sigma or dev_Q > self.threshold_sigma)):
            st.capturing = True
            st.end_ptr_count = 0
            st.trig_abs = st.ch_sample_n
            st.trigger_value_I = i_val
            st.trigger_value_Q = q_val

        # ── End condition: BOTH I and Q back within end_sigma ─────
        #
        # Uses a leaky-bucket counter instead of requiring strictly
        # consecutive samples.  A below-threshold sample increments the
        # counter by 1; an above-threshold sample decrements by 1 (capped
        # at 0).  This makes the end condition robust to individual noisy
        # samples that would otherwise reset the counter — critical for
        # high-rate PFB data where Gaussian noise fluctuations frequently
        # exceed 1.5σ on individual samples even during quiet inter-pulse
        # periods.
        if st.capturing:
            if dev_I < self.end_sigma and dev_Q < self.end_sigma:
                st.end_ptr_count += 1
            else:
                st.end_ptr_count = max(0, st.end_ptr_count - 1)

            if st.end_ptr_count > self.end_samples:
                self._save_pulse(channel)

    # ── Internal helpers ──────────────────────────────────────────

    def _save_pulse(self, channel: int) -> None:
        st = self.state[channel]
        # Use per-channel sample counter for correct buffer arithmetic
        # (abs_n is shared across all channels, causing 2x offset with 2 ch)
        # Subtract the leaky-bucket confirmation count so the saved window
        # ends approximately where the signal returned to baseline, not
        # where we confirmed it returned.  Keep a small margin (5 samples)
        # to capture the actual return transition.
        confirmation_trim = max(0, st.end_ptr_count - 5)
        post = st.ch_sample_n - (st.trig_abs or st.ch_sample_n) - confirmation_trim

        if post <= 0 or st.trig_abs is None:
            self._reset(channel)
            return

        L = self.buf[channel]["I"].count
        trig_fifo = (L - 1) - (st.ch_sample_n - st.trig_abs)

        if trig_fifo < 0 or trig_fifo >= L:
            self._reset(channel)
            return

        start = max(0, trig_fifo - self.pre)
        end = min(L, trig_fifo + post)
        if end <= start:
            self._reset(channel)
            return

        I_win = self._window(self.buf[channel]["I"], start, end)
        Q_win = self._window(self.buf[channel]["Q"], start, end)
        ts_win = self._window(self.buf[channel]["ts"], start, end)

        ch_key = f"Channel {channel}"
        self.pulse_count[ch_key] += 1
        k = self.pulse_count[ch_key]
        self.pulses[ch_key][k] = {
            "Amp_I": np.array(I_win),
            "Amp_Q": np.array(Q_win),
            "Time": np.array(ts_win),
        }

        self._reset(channel)

    def get_window_by_time(
        self,
        channel: int,
        t_start: float,
        t_end: float,
    ) -> Optional[dict]:
        """Extract a time window from the circular buffer for *channel*.

        This is used for continuous TOD extraction: when a pulse is
        detected in one stream, the corresponding time window can be
        retrieved from the *other* stream's ``PulseCapture`` even if no
        pulse was triggered there.

        Parameters
        ----------
        channel : int
            1-indexed channel number.
        t_start, t_end : float
            Time bounds (in the same units as the timestamps fed to
            ``process_sample``).

        Returns
        -------
        dict or None
            ``{"Amp_I": ndarray, "Amp_Q": ndarray, "Time": ndarray}``
            covering the requested window, or ``None`` if *channel* is
            unknown or no data overlaps the window.
        """
        if channel not in self.buf:
            return None

        ts_data = self.buf[channel]["ts"].data()
        if len(ts_data) == 0:
            return None

        # Build a boolean mask for the requested time window.
        # Timestamps may be None for samples that arrived before the
        # reference was established; treat those as outside the window.
        mask = np.zeros(len(ts_data), dtype=bool)
        for idx in range(len(ts_data)):
            t = ts_data[idx]
            if t is not None and t_start <= t <= t_end:
                mask[idx] = True

        if not np.any(mask):
            return None

        I_data = self.buf[channel]["I"].data()
        Q_data = self.buf[channel]["Q"].data()

        return {
            "Amp_I": np.array(I_data[mask]),
            "Amp_Q": np.array(Q_data[mask]),
            "Time": np.array(ts_data[mask]),
        }

    @staticmethod
    def _window(circ: Circular, start: int, end: int) -> np.ndarray:
        return circ.data()[start:end].copy()

    def _reset(self, channel: int) -> None:
        st = self.state[channel]
        st.capturing = False
        st.end_ptr_count = 0
        st.trig_abs = None
        st.trigger_value_I = None
        st.trigger_value_Q = None


# ───────────────────────── Noise Estimation ─────────────────────────

def _robust_std(x: np.ndarray) -> float:
    """MAD-based robust standard deviation estimator.

    Uses the Median Absolute Deviation (MAD) scaled by 1.4826 to provide
    a consistent estimator of σ for Gaussian-distributed data.  The MAD
    has a breakdown point of 50%, meaning it remains accurate even when
    up to half the samples are outliers (e.g. pulse events contaminating
    the noise estimation window).
    """
    if len(x) == 0:
        return 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    robust = 1.4826 * mad
    # Fall back to naive std if MAD gives zero (e.g. constant signal)
    return robust if robust > 0 else float(np.std(x))


def estimate_noise_stats(
    samples_by_channel: Dict[int, np.ndarray],
    channels: List[int],
) -> tuple[Dict[int, ChannelNoiseStats], Dict[int, np.ndarray]]:
    """Estimate per-channel noise statistics independently for I and Q.

    Uses **robust estimators** (median for location, MAD-based σ for scale)
    so that pulse events in the noise estimation window do not inflate the
    noise estimate and hide subsequent pulses from detection.

    Parameters
    ----------
    samples_by_channel : dict[int, ndarray]
        Complex sample arrays keyed by channel number (dtype complex128).
    channels : list[int]
        Channel numbers to estimate noise for.

    Returns
    -------
    noise_stats : dict[int, ChannelNoiseStats]
        Per-channel noise statistics (mean and std for both I and Q).
    raw_data : dict[int, ndarray]
        The raw complex samples used per channel.
    """
    noise_stats: Dict[int, ChannelNoiseStats] = {}
    raw_data: Dict[int, np.ndarray] = {}

    for c in channels:
        if c not in samples_by_channel or len(samples_by_channel[c]) == 0:
            noise_stats[c] = ChannelNoiseStats()
            raw_data[c] = np.array([], dtype=np.complex128)
            continue

        arr = samples_by_channel[c]
        raw_data[c] = arr

        # Robust estimators: median (location) + MAD-based σ (scale)
        # These resist contamination from pulse outliers in the noise window.
        robust_mean_I = float(np.median(arr.real))
        robust_mean_Q = float(np.median(arr.imag))
        robust_std_I = _robust_std(arr.real)
        robust_std_Q = _robust_std(arr.imag)

        # Diagnostic: detect pulse contamination by comparing naive vs robust
        naive_std_I = float(np.std(arr.real))
        naive_std_Q = float(np.std(arr.imag))
        ratio_I = naive_std_I / max(robust_std_I, 1e-30)
        ratio_Q = naive_std_Q / max(robust_std_Q, 1e-30)
        if ratio_I > 1.3 or ratio_Q > 1.3:
            print(f"[noise_stats] Ch {c}: pulse contamination detected — "
                  f"I ratio={ratio_I:.2f} (naive={naive_std_I:.2f}, robust={robust_std_I:.2f}), "
                  f"Q ratio={ratio_Q:.2f} (naive={naive_std_Q:.2f}, robust={robust_std_Q:.2f})")

        noise_stats[c] = ChannelNoiseStats(
            mean_I=robust_mean_I,
            std_I=robust_std_I,
            mean_Q=robust_mean_Q,
            std_Q=robust_std_Q,
        )

    return noise_stats, raw_data


# ── Legacy API compat (for slow_trigger_capture backward compat) ──

def estimate_noise_levels(
    samples_by_channel: Dict[int, np.ndarray],
    thresholds: Sequence[float],
    channels: List[int],
    thresh_type: str = "I",
) -> tuple[List[float], Dict[int, np.ndarray]]:
    """Legacy noise estimation — returns median of below-threshold samples.

    Kept for backward compatibility with ``slow_trigger_capture``.
    New code should use ``estimate_noise_stats`` instead.
    """
    noise_levels: List[float] = []
    noise_data: Dict[int, np.ndarray] = {}

    for idx, c in enumerate(channels):
        if c not in samples_by_channel:
            noise_levels.append(0.0)
            noise_data[c] = np.array([])
            continue

        arr = samples_by_channel[c]
        component = arr.imag if thresh_type == "Q" else arr.real
        thresh = thresholds[idx]

        below = component[component < thresh]
        noise_data[c] = below

        if len(below) > 0:
            noise_levels.append(float(np.median(below)))
        else:
            noise_levels.append(0.0)

    return noise_levels, noise_data
