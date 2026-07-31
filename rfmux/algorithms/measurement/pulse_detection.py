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
from typing import Callable, Dict, List, Optional, Sequence


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
    re_trigger_ready: bool = False  # True once signal drops below threshold during capture (pileup detection)
    prev_max_dev: float = 0.0      # Previous sample's max deviation in σ units (for derivative-based pileup)
    active_duration: Optional[int] = None  # Frozen pulse duration (trigger → below threshold) for adaptive end


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
    margin_fraction : float
        Fraction of pulse duration to use as pre-trigger margin and
        adaptive end-of-pulse confirmation count.  Default 0.1 (10%).
    min_pulse_samples : int
        Minimum pulse core duration (trigger → end) in samples.
        Pulses shorter than this are discarded as glitches.  Default 0.
    enable_pileup : bool
        Enable derivative-based pileup detection.  When True, a new
        pulse arriving during the tail of a previous one is split into
        a separate event.  Default True.
    """

    # Minimum leaky-bucket end confirmation count — prevents premature
    # termination on very short pulses or when margin_fraction is tiny.
    _MIN_END_SAMPLES: int = 10

    def __init__(
        self,
        buf_size: int,
        channels: List[int],
        noise_stats: Dict[int, ChannelNoiseStats],
        threshold_sigma: float = 5.0,
        end_sigma: float = 1.0,
        sample_rate: float = 38147.0,
        margin_fraction: float = 0.1,
        min_pulse_samples: int = 0,
        enable_pileup: bool = True,
        on_pulse: Optional[Callable[[int, int, dict], None]] = None,
        accumulate: bool = True,
        baseline_track_samples: int = 0,
    ):
        self.channels = list(channels)
        self.buf_size = buf_size
        self.sample_rate = sample_rate
        self.threshold_sigma = threshold_sigma
        self.end_sigma = end_sigma
        self.margin_fraction = margin_fraction
        self.min_pulse_samples = min_pulse_samples
        self.enable_pileup = enable_pileup

        # Callback for streaming consumers (HDF5, GUI, etc.)
        # Signature: on_pulse(channel: int, pulse_idx: int, pulse_data: dict)
        self.on_pulse = on_pulse

        # When False, pulses are NOT accumulated in self.pulses dict.
        # Use accumulate=False with on_pulse for streaming consumers
        # that write to HDF5 and don't need all pulses in memory.
        self._accumulate = accumulate

        # Per-channel noise stats
        self.noise_stats = noise_stats

        # Baseline tracking (0 = frozen, the historical behaviour).
        # Under 1/f noise the true baseline random-walks away from the
        # value captured at training while sigma stays put, so triggers
        # fire on drift and the end condition can become unsatisfiable.
        # An EMA of the quiet samples follows the drift instead.
        # Validity window: tau_pulse << tau_track << 1/f_drift.
        self.set_baseline_track_samples(baseline_track_samples)

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

    def set_baseline_track_samples(self, n: int) -> None:
        """Set the baseline EMA time constant, in samples (0 = frozen).

        Settable after construction so a re-estimation can install a
        freshly measured window without rebuilding the engine and
        losing the ring buffers.
        """
        self.baseline_track_samples = max(0, int(n))
        self._baseline_alpha = (1.0 / self.baseline_track_samples
                                if self.baseline_track_samples > 0 else 0.0)

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

        # ── Baseline tracking (clamped EMA) ───────────────────────
        # Every sample contributes, but its influence is CLAMPED to
        # +/- end_sigma * sigma.  Gating on "not capturing" or "quiet"
        # instead would deadlock: drift parks the signal outside the
        # band, the engine starts capturing and never finishes, and a
        # gated tracker can neither update nor bootstrap out of it.
        #
        # Clamping gives both properties we need:
        #  - a rare pulse shifts the baseline by at most
        #    (duration / tau_track) * end_sigma * sigma, which is
        #    negligible when tau_track >> pulse length;
        #  - persistent drift, offset in the same direction on every
        #    sample, is followed at up to end_sigma * sigma per
        #    tau_track — enough to escape a stuck capture.
        if self._baseline_alpha > 0.0:
            a = self._baseline_alpha
            clamp_I = self.end_sigma * max(ns.std_I, 1e-30)
            clamp_Q = self.end_sigma * max(ns.std_Q, 1e-30)
            innov_I = i_val - ns.mean_I
            innov_Q = q_val - ns.mean_Q
            if innov_I > clamp_I:
                innov_I = clamp_I
            elif innov_I < -clamp_I:
                innov_I = -clamp_I
            if innov_Q > clamp_Q:
                innov_Q = clamp_Q
            elif innov_Q < -clamp_Q:
                innov_Q = -clamp_Q
            ns.mean_I += a * innov_I
            ns.mean_Q += a * innov_Q

        # ── Trigger: EITHER I or Q exceeds threshold_sigma ────────
        # NOTE: Baseline confirmation is handled by _wait_for_baseline()
        # in trigger_capture.py BEFORE the capture loop starts.  This
        # ensures the first sample fed to PulseCapture is already at a
        # known-good baseline — no warmup phase needed here.
        if (not st.capturing
            and not self.freeze_triggers
            and (dev_I > self.threshold_sigma or dev_Q > self.threshold_sigma)):
            st.capturing = True
            st.end_ptr_count = 0
            st.trig_abs = st.ch_sample_n
            st.trigger_value_I = i_val
            st.trigger_value_Q = q_val

        # ── End condition & pileup detection ──────────────────────
        #
        # Two ways a pulse capture ends:
        #
        # (A) **Return to baseline**: BOTH I and Q stay within end_sigma
        #     for end_samples leaky-bucket counts.  Normal single-pulse.
        #
        # (B) **Pileup re-trigger** (derivative-based): The signal
        #     dropped below threshold_sigma at some point during capture
        #     (partial decay), AND we observe a large *jump* in deviation
        #     from baseline — the sample-to-sample increase in max
        #     deviation exceeds threshold_sigma.  This catches the sharp
        #     rising edge of a new pulse while ignoring slow noise
        #     fluctuations that wander above threshold.
        #
        #     For uncorrelated noise with σ=1 (in deviation units), the
        #     sample-to-sample difference has σ_diff ≈ √2 ≈ 1.4.
        #     A derivative jump of threshold_sigma (e.g. 3–5σ) is thus
        #     >>  noise, ensuring only real pulse arrivals trigger splits.
        #
        # The leaky-bucket counter is robust to individual noisy samples
        # that would otherwise reset the counter — critical for
        # high-rate PFB data where Gaussian noise fluctuations frequently
        # exceed 1.5σ on individual samples even during quiet inter-pulse
        # periods.
        if st.capturing:
            max_dev = max(dev_I, dev_Q)

            # ── Freeze active_duration ────────────────────────────
            # Freeze the pulse's active duration once the signal starts
            # returning to baseline.  Two triggers (first one wins):
            #   1. Signal drops below threshold_sigma (also enables
            #      pileup re-trigger detection).
            #   2. Signal drops below end_sigma (both I and Q) — catches
            #      large-amplitude pulses whose tails never cross below
            #      threshold_sigma before reaching end_sigma.
            # Without this freeze, the adaptive end target grows with
            # pulse_so_far, creating a race condition that extends windows.
            if max_dev < self.threshold_sigma:
                if not st.re_trigger_ready:
                    st.re_trigger_ready = True
                if st.active_duration is None:
                    st.active_duration = st.ch_sample_n - (st.trig_abs or st.ch_sample_n)
            elif (self.enable_pileup
                  and not st.re_trigger_ready
                  and (st.ch_sample_n - (st.trig_abs or st.ch_sample_n)) > self._MIN_END_SAMPLES
                  and max_dev < st.prev_max_dev):
                # Signal is still above threshold but decaying — enable
                # pileup re-trigger for large-amplitude pulses.
                # NOTE: do NOT freeze active_duration here — on noisy
                # PFB data, max_dev < prev_max_dev fires on the first
                # random noise dip after trigger, freezing active_duration
                # at just ~11 samples and causing premature termination.
                st.re_trigger_ready = True

            # ── Pileup detection (derivative-based, optional) ─────
            if self.enable_pileup:
                dev_jump = max_dev - st.prev_max_dev
                st.prev_max_dev = max_dev

                if (st.re_trigger_ready
                        and dev_jump > self.threshold_sigma
                        and (dev_I > self.threshold_sigma
                             or dev_Q > self.threshold_sigma)):
                    self._save_pulse(channel, pileup=True)
                    if not self.freeze_triggers:
                        st.capturing = True
                        st.end_ptr_count = 0
                        st.trig_abs = st.ch_sample_n
                        st.trigger_value_I = i_val
                        st.trigger_value_Q = q_val
                        st.re_trigger_ready = False
                        st.prev_max_dev = max_dev
                    return
            else:
                st.prev_max_dev = max_dev

            # ── Normal end: leaky-bucket baseline confirmation ────
            if dev_I < self.end_sigma and dev_Q < self.end_sigma:
                st.end_ptr_count += 1
                # Also freeze active_duration if not yet frozen —
                # catches pulses that skip past threshold_sigma.
                if st.active_duration is None:
                    st.active_duration = st.ch_sample_n - (st.trig_abs or st.ch_sample_n)
            else:
                st.end_ptr_count = max(0, st.end_ptr_count - 1)

            # Use frozen active_duration for stable end target
            ref_duration = st.active_duration or (
                st.ch_sample_n - (st.trig_abs or st.ch_sample_n))
            adaptive_end = max(
                self._MIN_END_SAMPLES,
                int(self.margin_fraction * ref_duration))

            if st.end_ptr_count > adaptive_end:
                self._save_pulse(channel)

    # ── Internal helpers ──────────────────────────────────────────

    def _save_pulse(self, channel: int, pileup: bool = False) -> None:
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

        # Glitch rejection: discard pulses shorter than min_pulse_samples
        if post < self.min_pulse_samples:
            self._reset(channel)
            return

        L = self.buf[channel]["I"].count
        trig_fifo = (L - 1) - (st.ch_sample_n - st.trig_abs)

        if trig_fifo < 0 or trig_fifo >= L:
            self._reset(channel)
            return

        # Pre-trigger margin: margin_fraction of pulse core duration,
        # minimum 2 samples to always show trigger context.
        # No post-trigger margin — the leaky-bucket end condition
        # already includes the return-to-baseline transition.
        pre_margin = max(2, int(self.margin_fraction * post))

        start = max(0, trig_fifo - pre_margin)
        end = min(L, trig_fifo + post)
        if end <= start:
            self._reset(channel)
            return

        I_win = self._window(self.buf[channel]["I"], start, end)
        Q_win = self._window(self.buf[channel]["Q"], start, end)
        ts_win = self._window(self.buf[channel]["ts"], start, end)

        pulse_data = {
            "Amp_I": np.array(I_win),
            "Amp_Q": np.array(Q_win),
            "Time": np.array(ts_win),
            "pileup": pileup,
        }

        ch_key = f"Channel {channel}"
        self.pulse_count[ch_key] += 1
        k = self.pulse_count[ch_key]

        # Notify streaming consumers (HDF5 writer, GUI, histograms, etc.)
        if self.on_pulse is not None:
            self.on_pulse(channel, k, pulse_data)

        # Accumulate in memory (for backward compat with trigger_capture macro).
        # Streaming consumers should set accumulate=False to avoid unbounded
        # memory growth during long captures.
        if self._accumulate:
            self.pulses[ch_key][k] = pulse_data

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
        st.re_trigger_ready = False
        st.prev_max_dev = 0.0
        st.active_duration = None


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

        # ── Noise estimation: median + high-pass MAD ──────────
        # Baseline mean: median is robust to asymmetric pulse
        # contamination (up to 50% outliers).
        # Noise σ: use the running difference (np.diff) as a
        # high-pass filter that removes the baseline level and
        # exponential decay tails.  For stationary Gaussian noise,
        # std(diff) = √2 × σ_noise, so σ = MAD(diff) / √2.
        # The MAD on diff is extremely robust because pulse onsets
        # are only 1-2 samples out of thousands — well under the
        # 50% breakdown point.
        robust_mean_I = float(np.median(arr.real))
        robust_mean_Q = float(np.median(arr.imag))
        robust_std_I = _robust_std(np.diff(arr.real)) / np.sqrt(2)
        robust_std_Q = _robust_std(np.diff(arr.imag)) / np.sqrt(2)

        # Refine baseline mean using the now-correct σ to clip
        # pulse outliers.  The median can be biased when pulses
        # cross zero (asymmetric contamination), but 3σ clipping
        # with the correct σ from diff/MAD accurately rejects them.
        clip = ((np.abs(arr.real - robust_mean_I) < 3 * robust_std_I) &
                (np.abs(arr.imag - robust_mean_Q) < 3 * robust_std_Q))
        clean = arr[clip]
        if len(clean) > 10:
            robust_mean_I = float(np.mean(clean.real))
            robust_mean_Q = float(np.mean(clean.imag))

        noise_stats[c] = ChannelNoiseStats(
            mean_I=robust_mean_I,
            std_I=robust_std_I,
            mean_Q=robust_mean_Q,
            std_Q=robust_std_Q,
        )

    return noise_stats, raw_data


# ─────────────── Baseline-tracking window from noise data ───────────

#: Fewest independent block pairs before a lag is worth fitting.  This
#: bounds the longest lag examined; confidence is handled by the error
#: model rather than by this alone.
_MIN_ALLAN_PAIRS = 8


def _allan_curve(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Overlapping, MAD-based Allan deviation at octave-spaced lags.

    For a block length ``m``, the variance of the block mean falls as
    1/m under white noise but saturates under 1/f, so the curve
    separates the two.  Differences of *adjacent* block means are used
    (rather than deviations from a global mean) so any slow trend
    cancels to first order.

    Every block position is used, not just the disjoint ones — the
    overlapping estimator wrings far more confidence out of a short
    record at long lags, which is exactly where a training window is
    poorest.  Reported degrees of freedom stay at the *independent*
    pair count, so the error bars remain conservative.

    The scale of the differences comes from the same MAD estimator the
    noise fit uses: a pulse landing in the training window spoils a few
    blocks, and the median ignores them.

    Returns ``(lags, sigma_allan, dof)``.
    """
    n = len(x)
    csum = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64))])
    lags, sig, dof = [], [], []
    m = 1
    while n // m >= _MIN_ALLAN_PAIRS + 1:
        k = n - 2 * m + 1
        # Difference of adjacent length-m block means, at every offset.
        d = (csum[2 * m:2 * m + k]
             - 2.0 * csum[m:m + k] + csum[:k]) / float(m)
        # std(diff) = sqrt(2) * sigma_allan for the Allan definition.
        sig.append(_robust_std(d) / np.sqrt(2.0))
        lags.append(m)
        dof.append(n // m - 1)
        m *= 2
    return (np.asarray(lags, dtype=float),
            np.asarray(sig, dtype=float),
            np.asarray(dof, dtype=float))


def _knee_samples(x: np.ndarray, n_sigma: float
                  ) -> tuple[Optional[float], dict]:
    """Locate the white/flicker crossover in *x*, in samples.

    Fits sigma_A^2(m) = A/m + B — A the white-noise power, B the flicker
    floor — and returns the lag where an EMA's white-noise averaging
    error falls to the flicker floor, N = A / 2B.  Below N the tracker
    is only smoothing white noise; above it the baseline has already
    moved on.

    ``None`` means B did not stand clear of its own uncertainty, i.e.
    the record shows no drift.  That test matters: the variance of an
    Allan point is 2*sigma_A^4/dof, so at a handful of pairs a lone
    upward fluctuation looks exactly like a flicker floor.
    """
    lags, sig, dof = _allan_curve(x)
    diag = {"lags": lags.tolist(), "allan": sig.tolist()}
    if len(lags) < 3 or not np.all(sig > 0):
        diag["reason"] = "record too short for an Allan fit"
        return None, diag

    # Inverse-variance weighted fit.  The uncertainty on an Allan
    # variance is proportional to the variance itself, so equal
    # weighting would let the short lags (where sigma_A^2 is largest)
    # swamp the very points that constrain B.
    var = 2.0 * sig ** 4 / dof
    design = np.column_stack([1.0 / lags, np.ones_like(lags)])
    dtw = design.T / var
    normal = dtw @ design
    try:
        cov = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        diag["reason"] = "degenerate Allan fit"
        return None, diag
    A, B = (cov @ (dtw @ sig ** 2)).tolist()
    sigma_B = float(np.sqrt(max(cov[1, 1], 0.0)))
    diag.update(white_var=A, flicker_var=B, flicker_var_err=sigma_B)

    if A <= 0:
        diag["reason"] = "degenerate fit (no white-noise term)"
        return None, diag
    if B <= n_sigma * sigma_B:
        diag["reason"] = (f"no drift: flicker floor under {n_sigma:g}x its "
                          "own uncertainty")
        return None, diag

    diag["reason"] = "knee from white/flicker crossover"
    return A / (2.0 * B), diag


def recommend_baseline_track_samples(
    samples_by_channel: Dict[int, np.ndarray],
    channels: List[int],
    *,
    min_samples: int = 0,
    max_samples: Optional[int] = None,
    n_sigma: float = 2.0,
) -> tuple[int, Dict[str, object]]:
    """Choose the baseline EMA time constant from the noise training data.

    The tracker has to sit in the window ``pulse << tau << drift``.  Its
    upper end is a property of the detector, not a user preference, and
    the training record already contains it: the 1/f knee, where
    averaging longer stops improving the baseline estimate.  Below the
    knee the EMA is only smoothing white noise; above it the baseline
    has already moved.

    Parameters
    ----------
    samples_by_channel, channels
        As for :func:`estimate_noise_stats` — the same arrays.
    min_samples
        Hard floor, normally a multiple of the longest expected pulse so
        the tracker cannot absorb pulse tails.
    max_samples
        Hard ceiling; defaults to the training length, since drift
        slower than the record is not something it measured.
    n_sigma
        Detection threshold: how far the fitted flicker floor must
        stand above its own uncertainty to count as real drift.  Set
        at 2 rather than a stricter 3 because the two errors are not
        symmetric — a marginal false positive lands the window near
        record/9, still far above the pulse floor and harmless, while
        a miss lets the drift the tracker exists for go unfollowed.
        (Measured: 2 sigma gives ~5% false positives on white noise
        and finds a knee at 1/30 of the record 9 times in 10; 3 sigma
        gives no false positives but finds that knee half the time.)

    Returns
    -------
    (samples, info)
        ``samples`` is ready to hand to :class:`PulseCapture`; 0 means
        "leave the baseline frozen" and is only returned when the caller
        allows it via ``max_samples=0``.  ``info`` carries the per-channel
        fits and a human-readable ``summary`` for display.
    """
    per_channel: Dict[int, dict] = {}
    knees: List[float] = []
    n_train = 0

    for c in channels:
        arr = samples_by_channel.get(c)
        if arr is None or len(arr) < 32:
            continue
        n_train = max(n_train, len(arr))
        k_I, d_I = _knee_samples(np.real(arr), n_sigma)
        k_Q, d_Q = _knee_samples(np.imag(arr), n_sigma)
        found = [k for k in (k_I, k_Q) if k is not None]
        # Both quadratures share one EMA, so the faster-drifting one sets
        # the pace for the channel.
        knee = min(found) if found else None
        per_channel[c] = {"I": d_I, "Q": d_Q, "knee_samples": knee}
        if knee is not None:
            knees.append(knee)

    if max_samples is None:
        max_samples = n_train if n_train else 0

    info: Dict[str, object] = {
        "per_channel": per_channel,
        "min_samples": int(min_samples),
        "max_samples": int(max_samples),
        "n_train": int(n_train),
        "drift_detected": bool(knees),
    }

    if not per_channel:
        info["summary"] = "no usable training data — baseline left frozen"
        return 0, info

    if knees:
        # Median across channels: one bad fit should not drag the whole
        # module, and the floor below bounds the damage if it does.
        raw = float(np.median(knees))
        info["knee_samples"] = raw
        basis = "1/f knee"
    else:
        # Nothing drifting within the record, so track no faster than the
        # span over which we confirmed that.
        raw = float(max_samples)
        info["knee_samples"] = None
        basis = "no drift measured"

    chosen = int(round(max(min_samples, min(raw, float(max_samples)))))
    info["raw_samples"] = raw
    info["samples"] = chosen

    # Lead with what was actually used — the basis is secondary when a
    # clamp overrode it.
    if chosen == min_samples and raw < min_samples:
        if min_samples > max_samples:
            info["summary"] = (
                f"floored at {chosen:,} samples: the {max_samples:,}-sample "
                "training window is too short to measure drift on that "
                "timescale — lengthen noise training")
        else:
            info["summary"] = (
                f"floored at {chosen:,} samples — the {basis} "
                f"({raw:,.0f}) is too fast to separate from a pulse")
    elif chosen == max_samples and raw > max_samples:
        info["summary"] = (f"capped at {chosen:,} samples "
                           f"({basis}, {raw:,.0f})")
    elif knees:
        info["summary"] = f"{basis} at {chosen:,} samples"
    else:
        info["summary"] = (f"{basis}; tracking at the training span, "
                           f"{chosen:,} samples")
    return chosen, info


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
