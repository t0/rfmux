"""
Streamer-agnostic pulse detection and capture.

This module provides the core pulse detection state machine (``PulseCapture``)
and noise-estimation utility, independent of the packet format or sample rate.
It can be driven by either the slow readout stream or the fast PFB stream.

Triggering demands two things of a pulse, on either quadrature:

- **Amplitude**: the sample deviates from the baseline mean by more than
  ``threshold_sigma`` standard deviations, for ``trigger_samples``
  consecutive samples.  Catches positive excursions and negative dips on
  whichever component carries the signal.
- **Edge**: the deviation *grew* by more than ``threshold_sigma`` jump-σ
  within the last ``edge_lookback`` samples.  The jump is a difference of
  two raw samples, so the baseline cancels out of it — 1/f wander that
  drifts across the threshold band cannot fake it, however stale the
  baseline estimate is.  Real pulses rise fast; drift does not.

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

    # 3. Feed samples.  Completed pulses arrive through on_pulse; the
    #    detector holds only the ring buffer, so a capture can run
    #    indefinitely without growing.
    for channel, i_val, q_val, timestamp in sample_stream:
        pcap.process_sample(channel, i_val, q_val, timestamp)
"""

from __future__ import annotations

import math

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

_SQRT2 = math.sqrt(2.0)


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

    def recent(self, k: int):
        """Value written k adds ago (k=0 → newest).  Requires k < count."""
        return self.buf[(self.ptr - 1 - k) % self.N]


# ───────────────────────── Per-Channel Noise Stats ──────────────────

@dataclass
class ChannelNoiseStats:
    """Noise statistics for one channel, computed independently for I and Q.

    ``jump_std_*`` is the robust σ of the lag-K difference
    ``x[n] - x[n-K]`` at the edge-detector lookback K, measured from the
    training record.  It absorbs both filter correlation and whatever
    1/f power exists at that lag, so the edge threshold holds a constant
    false rate on any detector.  0 means "not measured" — consumers fall
    back to the white-noise value √2·std.
    """
    mean_I: float = 0.0
    std_I: float = 1.0
    mean_Q: float = 0.0
    std_Q: float = 1.0
    jump_std_I: float = 0.0
    jump_std_Q: float = 0.0


# ───────────────────────── Per-Channel Capture State ────────────────

@dataclass
class _ChState:
    capturing: bool = False
    end_ptr_count: int = 0
    trig_abs: Optional[int] = None
    # When the trigger FIRED, as opposed to where the window is dated
    # (trig_abs may sit up to a lookback earlier when drift had parked
    # the run above threshold).  Capture-relative edge references clip
    # to this, so they can never reach past the physical pulse onset.
    fire_abs: int = 0
    # Pre-pulse level snapshot (median of the edge taps at fire time):
    # a baseline-free end reference.  The rolling median can lag 1/f by
    # several σ, leaving the amplitude test parked above threshold for
    # a whole capture — but returning to the level the pulse ROSE FROM
    # is decisive evidence the pulse is over, however stale the mean.
    anchor_I: float = 0.0
    anchor_Q: float = 0.0
    # This capture began at a pileup split, so it sits on the previous
    # pulse's decaying tail: its peak and tau are pedestal-biased, and
    # it carries the pileup flag when saved however it ends.
    pileup_child: bool = False
    ch_sample_n: int = 0  # Per-channel sample counter (for buffer arithmetic)
    re_trigger_ready: bool = False  # True once the current pulse is seen decaying (pileup re-arm)
    active_duration: Optional[int] = None  # Frozen pulse duration (trigger → below threshold) for adaptive end
    # Run of consecutive above-threshold samples — the trigger is dated
    # to the start of the run, not to the sample that confirmed it.
    above_run: int = 0
    run_start_abs: int = 0
    # First sample of the current statistics epoch: lag-K references
    # never reach past it (see reset_edge_history).
    epoch_start: int = 0
    # Rolling-baseline bookkeeping: decimation phase, and insertions
    # since the last re-estimate.
    decim_n: int = 0
    since_refresh: int = 0


# ───────────────────────── PulseCapture ─────────────────────────────

class PulseCapture:
    """Streaming multi-channel pulse detector with dual I/Q sigma-based triggering.

    For each channel, both I and Q are monitored independently.  A pulse
    must satisfy two conditions on **either** component: deviate from the
    baseline mean by more than ``threshold_sigma`` standard deviations
    (for ``trigger_samples`` consecutive samples), and have *arrived*
    fast — the deviation must have grown by more than ``threshold_sigma``
    jump-σ over the last ``edge_lookback`` samples.  The second test is a
    difference of two raw samples, so the baseline cancels out of it:
    slow 1/f wander that crosses the threshold band cannot fake it.

    End-of-pulse is declared when **both** I and Q return to within
    ``end_sigma`` (default 1.0) standard deviations — of the tracked
    mean, or of the pre-pulse *anchor* (the level this pulse rose from,
    snapshotted at the trigger; baseline-free like the edge test) — for
    a configurable time duration, or unconditionally at
    ``max_capture_samples``.  A mean estimate lagging 1/f can therefore
    delay nothing: the pulse ends when the signal is back where it
    started.  The hard stop bounds the STATE MACHINE only: the saved
    window ends at the below-threshold instant plus a
    ``margin_fraction`` tail, which is where the eye puts the end of
    the pulse, not where the confirmation finished.

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
        Fraction of pulse duration kept as pre-trigger margin and as
        post-pulse tail after the below-threshold instant, and the
        adaptive end-of-pulse confirmation count.  Default 0.1 (10%).
    min_pulse_samples : int
        Minimum pulse core duration (trigger → end) in samples.
        Pulses shorter than this are discarded as glitches.  Default 0.
    trigger_samples : int
        Consecutive samples that must exceed ``threshold_sigma`` before
        a capture starts.  1 restores single-sample triggering; 2 (the
        default) removes essentially all accidental triggers, which
        otherwise arrive at ~1.4 Hz per channel on the PFB stream at
        5 sigma.  The capture is still dated to the first sample of the
        run, so nothing is lost from the rising edge.
    enable_pileup : bool
        Enable pileup splitting.  When True, a new pulse arriving during
        the tail of a previous one — a fresh edge after the current pulse
        was seen decaying — is split into a separate event.  EVERY
        fragment of a split chain carries the ``pileup`` flag (the first
        has its tail cut, the rest sit on a pedestal), so downstream
        consumers can exclude them — templates already do.  Default
        True.  Requires the edge detector; inert when ``edge_lookback``
        is 0.
    baseline_window : int
        Samples spanned by the rolling-baseline median (0 = frozen
        baseline).  Must be long compared with a pulse: the median
        tolerates pulses up to ~50% duty, so the ring buffer itself
        (1.5x the max pulse) is too short and the training window is
        the natural choice.
    edge_lookback : int, optional
        Lag K of the edge detector, in samples.  None (the default)
        derives it from the ring: ~10% of the longest recordable pulse,
        far longer than any physical rise and far shorter than 1/f
        wander.  0 disables the edge gate — amplitude-only triggering,
        for A/B debugging only.
    max_capture_samples : int, optional
        Hard stop: any capture reaching this length is saved as-is.  It
        is flagged ``truncated`` only when the pulse had not yet dropped
        below threshold — a stop reached because drift stalled the end
        confirmation still saved a complete pulse.  None (the default)
        derives it from the ring (~80%, i.e. 1.2x the max pulse the ring
        was sized for), so a capture can never outlive the buffer and
        silently lose its rising edge.  0 disables the stop.
    """

    # Minimum leaky-bucket end confirmation count — prevents premature
    # termination on very short pulses or when margin_fraction is tiny.
    _MIN_END_SAMPLES: int = 10

    #: Entries kept for the rolling-baseline median.  Fixed, so cost and
    #: memory do not scale with the window: a decimated reservoir tracks
    #: the full-stream median to ~0.01 sigma.
    _BASELINE_RESERVOIR: int = 4096

    #: Fraction of the ring a capture may fill before the hard stop.
    #: With the ring sized at 1.5x the max expected pulse, 0.8 puts the
    #: stop at 1.2x the max pulse, leaving room for the pre-trigger
    #: margin in the same ring.
    HARD_STOP_RING_FRACTION: float = 0.8

    @staticmethod
    def default_edge_lookback(buf_size: int,
                              margin_fraction: float = 0.1) -> int:
        """Edge-detector lag derived from the ring: ~margin_fraction of
        the longest recordable pulse.  Shared with noise estimation so
        the measured jump-σ is taken at the same lag the detector uses."""
        return max(1, int(round(
            margin_fraction * PulseCapture.HARD_STOP_RING_FRACTION
            * buf_size)))

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
        trigger_samples: int = 2,
        enable_pileup: bool = True,
        on_pulse: Optional[Callable[[int, int, dict], None]] = None,
        baseline_window: int = 0,
        edge_lookback: Optional[int] = None,
        max_capture_samples: Optional[int] = None,
    ):
        self.channels = list(channels)
        self.buf_size = buf_size
        self.sample_rate = sample_rate
        self.threshold_sigma = threshold_sigma
        self.end_sigma = end_sigma
        self.margin_fraction = margin_fraction
        self.min_pulse_samples = min_pulse_samples
        self.trigger_samples = max(1, int(trigger_samples))
        self.enable_pileup = enable_pileup

        if edge_lookback is None:
            edge_lookback = self.default_edge_lookback(buf_size,
                                                       margin_fraction)
        self.edge_lookback = max(0, int(edge_lookback))
        if max_capture_samples is None:
            max_capture_samples = int(round(
                self.HARD_STOP_RING_FRACTION * buf_size))
        self.max_capture_samples = max(0, int(max_capture_samples))

        # Callback for streaming consumers (HDF5, GUI, etc.)
        # Signature: on_pulse(channel: int, pulse_idx: int, pulse_data: dict)
        self.on_pulse = on_pulse

        # Per-channel noise stats
        self.noise_stats = noise_stats

        # ── Rolling baseline ──────────────────────────────────────
        # Under 1/f the true baseline wanders away from the value fixed
        # at training while sigma stays put, so deviations grow with no
        # signal: triggers fire on drift, and the end condition can
        # become unsatisfiable because the signal never comes back
        # inside a band centred on a stale mean.
        #
        # sigma and the mean want different things.  sigma is
        # stationary and is measured from the long training record by a
        # high-pass (diff/MAD) estimator that drift cannot corrupt.
        # The mean is the part that moves, so it is re-estimated
        # continuously as the MEDIAN of a window long compared with a
        # pulse.
        #
        # The median, not a mean or a clamped average: it IGNORES
        # pulses rather than merely bounding their pull, holding to
        # ~0.15 sigma at 10% pulse duty and only breaking down near
        # 50%.  Excluding flagged-pulse samples instead would be worse,
        # not better — drift can park the signal outside the band, and
        # a capture that can then never end would gate the update off
        # permanently.  A plain median over everything has no such
        # state and walks out of that on its own.
        self.baseline_window = max(0, int(baseline_window))
        # A fixed-size decimated reservoir keeps memory and refresh cost
        # constant however long the window is; every M-th sample
        # preserves the amplitude distribution (block averaging would
        # smear pulses across neighbours).
        self._bl_capacity = min(self.baseline_window,
                                self._BASELINE_RESERVOIR)
        self._bl_decim = max(1, self.baseline_window
                             // max(1, self._bl_capacity))
        self._bl_refresh = max(8, self._bl_capacity // 8)
        self._bl_min = min(64, self._bl_capacity)
        self._bl: Dict[int, Dict[str, Circular]] = {}
        if self.baseline_window > 0:
            for c in self.channels:
                self._bl[c] = {k: Circular(self._bl_capacity)
                               for k in ("I", "Q")}

        # Per-channel circular buffers
        self.buf: Dict[int, Dict[str, Circular]] = {}
        for c in self.channels:
            self.buf[c] = {k: Circular(buf_size) for k in ("I", "Q", "ts")}

        # Build channel → index lookup
        self._ch_set = set(self.channels)

        # Per-channel capture state
        self.state: Dict[int, _ChState] = {c: _ChState() for c in self.channels}

        # Results.  Completed pulses leave through on_pulse and are not
        # retained here — the detector's memory is the ring buffer and
        # nothing else, so a capture can run for as long as you like.
        self.start_time: Optional[float] = None
        self.pulse_count: Dict[str, int] = {f"Channel {c}": 0 for c in self.channels}

        # When True, no new triggers start but in-progress captures complete.
        # Set by the receive loop when time_run is reached.
        self.freeze_triggers: bool = False

        # Absolute sample counter
        self.abs_n = 0

    # ── Public API ────────────────────────────────────────────────

    def reset_edge_history(self) -> None:
        """Start a new statistics epoch: lag-K references stop reaching
        into samples taken under the previous baseline mean.

        Call after swapping ``noise_stats`` mid-stream (re-estimation).
        Samples on the far side of a mean shift difference against the
        new ones as enormous jumps — enough to both arm and fire a
        pileup split inside a pulse's own rise.  The edge detector runs
        with a shortened lag until a full lookback of new-epoch samples
        exists, exactly as it does at stream start."""
        for st in self.state.values():
            st.epoch_start = st.ch_sample_n

    def _refresh_baseline(self, channel: int) -> None:
        """Re-centre this channel's band on the median of the window."""
        bl = self._bl.get(channel)
        if bl is None or bl["I"].count < self._bl_min:
            return
        ns = self.noise_stats.get(channel)
        if ns is None:
            return
        ns.mean_I = float(np.median(bl["I"].data()))
        ns.mean_Q = float(np.median(bl["Q"].data()))

    def process_sample(
        self,
        channel: int,
        i_val: float,
        q_val: float,
        timestamp: Optional[float],
    ) -> None:
        """Ingest a single I/Q sample for *channel*.

        Triggers when EITHER I or Q is threshold_sigma past the baseline
        (amplitude, confirmed over trigger_samples) AND rose that
        significantly within the edge lookback (edge, baseline-free).
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

        # ── Rolling baseline ──────────────────────────────────────
        # Fed FIRST so everything this sample derives from the same
        # baseline estimate.  A median refresh can jump the mean; a
        # deviation computed before the refresh compared against an
        # edge reference computed after it reads as a spurious 5σ+
        # rise on that one sample.
        if self.baseline_window > 0:
            st.decim_n += 1
            if st.decim_n >= self._bl_decim:
                st.decim_n = 0
                self._bl[channel]["I"].add(i_val)
                self._bl[channel]["Q"].add(q_val)
                st.since_refresh += 1
                if st.since_refresh >= self._bl_refresh:
                    st.since_refresh = 0
                    self._refresh_baseline(channel)

        # Compute deviations in sigma units
        raw_I = abs(i_val - ns.mean_I)
        raw_Q = abs(q_val - ns.mean_Q)
        dev_I = raw_I / max(ns.std_I, 1e-30)
        dev_Q = raw_Q / max(ns.std_Q, 1e-30)
        max_dev = max(dev_I, dev_Q)
        js_I = ns.jump_std_I if ns.jump_std_I > 0 else _SQRT2 * ns.std_I
        js_Q = ns.jump_std_Q if ns.jump_std_Q > 0 else _SQRT2 * ns.std_Q

        # ── Trigger confirmation ──────────────────────────────────
        # A threshold crossing must persist for trigger_samples
        # consecutive samples.  A single sample is not evidence of a
        # pulse: at 5 sigma on two quadratures the per-sample false
        # rate is ~1.1e-6, which is one spurious trigger every ~23 s
        # per channel on the slow stream and about 1.4 PER SECOND on
        # the PFB stream.  Requiring two consecutive samples costs a
        # real pulse nothing — anything above threshold for a single
        # sample carries no measurable rise or decay anyway — while
        # cutting the accidental rate by orders of magnitude.
        if max_dev > self.threshold_sigma:
            if st.above_run == 0:
                st.run_start_abs = st.ch_sample_n
            st.above_run += 1
        else:
            st.above_run = 0
        eligible = st.above_run >= self.trigger_samples

        # ── Edge detector (lag-K outward jump) ────────────────────
        # How much the deviation from baseline GREW over the last
        # edge_lookback samples, per quadrature, in units of the
        # measured lag-K jump-σ.  While both samples sit on the same
        # side of the mean the baseline cancels exactly — this is the
        # 1/f-immune half of the trigger.  Positive = rising away from
        # baseline, negative = decaying back toward it, so decay tails
        # cannot re-fire it.
        edge_ok = False
        edge_taps = None
        if self.edge_lookback > 0:
            # The usable lag shortens near the start of the stream and
            # after a statistics epoch reset — a reference from before
            # either would difference across a mean shift and read as
            # a jump.
            lmax = min(self.edge_lookback,
                       self.buf[channel]["I"].count - 1,
                       st.ch_sample_n - st.epoch_start - 1)
            if lmax >= 1:
                bI = self.buf[channel]["I"]
                bQ = self.buf[channel]["Q"]
                # Three taps across the lookback (K, K/2, K/4), judged
                # against the MEDIAN.  A single reference K back is
                # blind to pulse trains whose period matches K — it
                # lands on the previous pulse and the identical
                # envelopes cancel — so several taps are needed, and
                # the median tolerates ONE polluted tap in either
                # direction: a tap on a previous pulse must not blind
                # the trigger (min-like), and a tap on a mean-crossing
                # transit — a pulse traversing a stale mean leaves
                # near-zero-deviation samples in the ring — must not
                # fake a rise out of parked drift (max-like).  All taps
                # ride the same 1/f drift, so wander immunity is
                # unchanged.
                tap_vals_I: list = []
                tap_vals_Q: list = []
                for tap in (lmax, lmax // 2, lmax // 4):
                    if tap >= 1:
                        tap_vals_I.append(bI.recent(tap))
                        tap_vals_Q.append(bQ.recent(tap))
                devs_I = sorted(abs(v - ns.mean_I) for v in tap_vals_I)
                devs_Q = sorted(abs(v - ns.mean_Q) for v in tap_vals_Q)
                ref_I = devs_I[len(devs_I) // 2]
                ref_Q = devs_Q[len(devs_Q) // 2]
                edge_ok = (
                    (raw_I - ref_I) / max(js_I, 1e-30)
                    > self.threshold_sigma
                    or (raw_Q - ref_Q) / max(js_Q, 1e-30)
                    > self.threshold_sigma)
                edge_taps = (tap_vals_I, tap_vals_Q)

        # ── Trigger: amplitude AND edge ───────────────────────────
        # With the edge gate the trigger can fire at any point in the
        # above-threshold run, not only the sample that completed the
        # confirmation — a rise spread over several samples earns its
        # jump as it grows.  Without the edge detector (edge_lookback
        # 0, debug only) the legacy fire-once-per-run rule stands in
        # for it: a run that outlives one capture must not re-fire.
        if self.edge_lookback > 0:
            trigger_ok = eligible and edge_ok
        else:
            trigger_ok = st.above_run == self.trigger_samples
        if not st.capturing and not self.freeze_triggers and trigger_ok:
            st.capturing = True
            st.end_ptr_count = 0
            st.fire_abs = st.ch_sample_n
            # Pre-pulse anchor: the level the pulse rose from, as the
            # median of the edge taps.  Baseline-free — the end tests
            # compare against where the signal actually WAS, so a mean
            # estimate lagging the 1/f wander cannot hold the capture
            # open after the pulse has visibly returned.
            if edge_taps is not None:
                vi, vq = edge_taps
                st.anchor_I = sorted(vi)[len(vi) // 2]
                st.anchor_Q = sorted(vq)[len(vq) // 2]
            else:
                st.anchor_I = ns.mean_I
                st.anchor_Q = ns.mean_Q
            # Date the trigger to where the excursion began, so the
            # pre-trigger margin and the stacking alignment do not
            # slip by the confirmation length.  Capped one lookback
            # deep: if drift parked the run above threshold long ago,
            # the pulse that fired the edge began within the last K
            # samples, not at the ancient crossing.
            st.trig_abs = st.run_start_abs
            if self.edge_lookback > 0:
                st.trig_abs = max(st.trig_abs,
                                  st.ch_sample_n - self.edge_lookback)

        # ── End condition & pileup detection ──────────────────────
        #
        # Three ways a pulse capture ends:
        #
        # (A) **Return to baseline**: BOTH I and Q stay within end_sigma
        #     — of the tracked mean, or of the pre-pulse ANCHOR (the
        #     level this pulse rose from, baseline-free) — for
        #     end_samples leaky-bucket counts.  Normal single-pulse.
        #
        # (B) **Pileup re-trigger**: the current pulse was seen decaying
        #     (below threshold, or a strong inward lag-K jump), and a
        #     fresh outward edge arrives — the same amplitude+edge test
        #     as a new trigger.  The event is split.
        #
        # (C) **Hard stop**: the capture reaches max_capture_samples.
        #     Saved as-is and flagged truncated.  A baseline that
        #     drifted during the capture can push the end_sigma band off
        #     the settled signal; without this bound that capture would
        #     run forever while the ring silently wraps over the rising
        #     edge.  Worst case is now one max-pulse of dead time.
        #
        # The leaky-bucket counter is robust to individual noisy samples
        # that would otherwise reset the counter — critical for
        # high-rate PFB data where Gaussian noise fluctuations frequently
        # exceed 1.5σ on individual samples even during quiet inter-pulse
        # periods.
        if st.capturing:
            since_trig = st.ch_sample_n - (st.trig_abs or st.ch_sample_n)
            since_fire = st.ch_sample_n - st.fire_abs

            # ── Capture-relative edge tests ───────────────────────
            # Taps CLIPPED to samples since the trigger FIRED: a
            # reference reaching further back lands on the previous
            # pulse — or, when drift parked the run and the window was
            # dated a lookback early, on pre-pulse wander — and either
            # reads as false decay evidence or as a false rise.
            # Unclipped, one split shredded the successor capture
            # sample by sample.
            #
            # decaying_now: the signal sits far below the loudest tap
            # of its own capture — decay evidence for pulses whose
            # tails never cross below threshold.
            # rising_above_self: the signal rose above the NEAREST tap
            # — a new pulse stacks on top of the tail, while a smooth
            # decay sits below its own recent level by construction.
            decaying_now = False
            rising_above_self = False
            near_vals = None
            if self.edge_lookback > 0 and since_fire >= 1:
                span = min(self.edge_lookback, since_fire,
                           self.buf[channel]["I"].count - 1,
                           st.ch_sample_n - st.epoch_start - 1)
                if span >= 1:
                    bI = self.buf[channel]["I"]
                    bQ = self.buf[channel]["Q"]
                    hi_I = hi_Q = 0.0
                    for tap in (span, span // 2, span // 4):
                        if tap >= 1:
                            hi_I = max(hi_I,
                                       abs(bI.recent(tap) - ns.mean_I))
                            hi_Q = max(hi_Q,
                                       abs(bQ.recent(tap) - ns.mean_Q))
                    decaying_now = (
                        (raw_I - hi_I) / max(js_I, 1e-30)
                        < -self.threshold_sigma
                        or (raw_Q - hi_Q) / max(js_Q, 1e-30)
                        < -self.threshold_sigma)
                    near = max(1, min(self.edge_lookback // 4, span))
                    near_vals = (bI.recent(near), bQ.recent(near))
                    rising_above_self = (
                        (raw_I - abs(near_vals[0] - ns.mean_I))
                        / max(js_I, 1e-30) > self.threshold_sigma
                        or (raw_Q - abs(near_vals[1] - ns.mean_Q))
                        / max(js_Q, 1e-30) > self.threshold_sigma)

            # ── Baseline-free return test ─────────────────────────
            # Back at the pre-pulse anchor on BOTH quadratures.  The
            # amplitude tests below compare against the tracked mean,
            # which can lag 1/f by several σ and park dev above
            # end_sigma (or even threshold_sigma) for an entire
            # capture; the anchor is where the signal actually sat
            # when this pulse rose, so returning to it ends the
            # capture no matter what the mean estimate is doing.
            returned = (
                abs(i_val - st.anchor_I)
                < self.end_sigma * max(ns.std_I, 1e-30)
                and abs(q_val - st.anchor_Q)
                < self.end_sigma * max(ns.std_Q, 1e-30))

            # ── Freeze active_duration / arm pileup re-trigger ────
            # The duration freezes once the signal starts returning to
            # baseline; without the freeze the adaptive end target grows
            # with the capture and extends windows.  Decay evidence also
            # arms the pileup re-trigger: dropping below threshold, or —
            # for large pulses whose tails stay above it — sitting far
            # below the capture's own recent level.
            if max_dev < self.threshold_sigma or returned:
                st.re_trigger_ready = True
                if st.active_duration is None:
                    st.active_duration = since_trig
            elif (decaying_now and not st.re_trigger_ready
                    and since_fire > self._MIN_END_SAMPLES):
                st.re_trigger_ready = True

            # ── Pileup split ──────────────────────────────────────
            # A confirmed run that rose above the current pulse's own
            # recent level, after the current pulse was seen decaying.
            if (self.enable_pileup and self.edge_lookback > 0
                    and st.re_trigger_ready and eligible
                    and rising_above_self):
                self._save_pulse(channel, pileup=True)
                if not self.freeze_triggers:
                    st.capturing = True
                    st.end_ptr_count = 0
                    st.fire_abs = st.ch_sample_n
                    # The new rise happened within the nearest tap's
                    # window — the run may date back to the previous
                    # pulse's onset if the tail never crossed below
                    # threshold.
                    st.trig_abs = max(
                        st.run_start_abs,
                        st.ch_sample_n - max(1, self.edge_lookback // 4))
                    # The piled-up pulse decays onto the previous
                    # pulse's tail, not the pre-pulse level: anchor on
                    # the tail just before the new rise.
                    if near_vals is not None:
                        st.anchor_I, st.anchor_Q = near_vals
                    # Every fragment of a chain is pileup-affected —
                    # this one sits on the previous pulse's pedestal.
                    st.pileup_child = True
                return

            # ── Normal end: leaky-bucket baseline confirmation ────
            # Fed by either test: inside the end band of the tracked
            # mean, or back at the pre-pulse anchor.
            if returned or (dev_I < self.end_sigma
                            and dev_Q < self.end_sigma):
                st.end_ptr_count += 1
                # Also freeze active_duration if not yet frozen —
                # catches pulses that skip past threshold_sigma.
                if st.active_duration is None:
                    st.active_duration = since_trig
            else:
                st.end_ptr_count = max(0, st.end_ptr_count - 1)

            # Use frozen active_duration for stable end target
            ref_duration = st.active_duration or since_trig
            adaptive_end = max(
                self._MIN_END_SAMPLES,
                int(self.margin_fraction * ref_duration))

            if st.end_ptr_count > adaptive_end:
                self._save_pulse(channel)
            elif (self.max_capture_samples > 0
                    and since_trig >= self.max_capture_samples):
                self._save_pulse(channel, truncated=True)

    # ── Internal helpers ──────────────────────────────────────────

    def _save_pulse(self, channel: int, pileup: bool = False,
                    truncated: bool = False) -> None:
        st = self.state[channel]
        # Use per-channel sample counter for correct buffer arithmetic
        # (abs_n is shared across all channels, causing 2x offset with 2 ch)
        raw_post = st.ch_sample_n - (st.trig_abs or st.ch_sample_n)
        core = st.active_duration
        if pileup or core is None:
            # No below-threshold instant to anchor on: a split ends at
            # the split sample, and a hard stop that never saw the pulse
            # end keeps everything it has.  Trim off the leaky-bucket
            # confirmation count (less a 5-sample margin) so the window
            # ends near where the signal settled, not where the bucket
            # finished counting.
            post = raw_post - max(0, st.end_ptr_count - 5)
        else:
            # The pulse visibly ended at below-threshold (core samples
            # after the trigger).  Save margin_fraction of it as tail
            # and drop the slow confirmation stretch — the leaky bucket
            # (or the hard stop) only bounds the STATE MACHINE, it no
            # longer stretches the data.  In particular a hard stop
            # reached because drift stalled the bucket still saved a
            # complete pulse, so it is NOT flagged truncated.
            tail = max(self._MIN_END_SAMPLES,
                       int(self.margin_fraction * core))
            post = min(raw_post, core + tail)
            truncated = False

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

        # Pre-trigger margin: margin_fraction of the saved length,
        # minimum 2 samples to always show trigger context.  The tail
        # margin after below-threshold was already folded into post.
        pre_margin = max(2, int(self.margin_fraction * post))

        start = max(0, trig_fifo - pre_margin)
        end = min(L, trig_fifo + post)
        if end <= start:
            self._reset(channel)
            return

        I_win = self._window(self.buf[channel]["I"], start, end)
        Q_win = self._window(self.buf[channel]["Q"], start, end)
        ts_win = self._window(self.buf[channel]["ts"], start, end)

        # Where the state machine actually acted, so a capture can be
        # read back against the decisions that produced it.
        #
        # The end index is normally PAST the last saved sample: the
        # window ends at below-threshold plus the tail margin, while
        # the state machine keeps running until the leaky bucket (or
        # the hard stop) releases it.  Times are carried alongside the
        # indices for exactly that reason.
        ts_all = self.buf[channel]["ts"].data()
        trigger_index = trig_fifo - start
        end_index = (L - 1) - start
        below_index = (trigger_index + st.active_duration
                       if st.active_duration is not None else None)

        pulse_data = {
            "Amp_I": np.array(I_win),
            "Amp_Q": np.array(Q_win),
            "Time": np.array(ts_win),
            # The pileup parameter means "this capture was ENDED by a
            # split" (it sets the window policy above); the flag also
            # covers captures that BEGAN at a split — every fragment of
            # a chain is pedestal-biased and marked.
            "pileup": bool(pileup or st.pileup_child),
            "truncated": truncated,
            "trigger_index": int(trigger_index),
            "end_index": int(end_index),
            "trigger_time": float(ts_all[trig_fifo]),
            "end_time": float(ts_all[L - 1]),
            "end_confirm_samples": int(st.end_ptr_count),
            "end_confirm_target": int(max(
                self._MIN_END_SAMPLES,
                int(self.margin_fraction * (
                    st.active_duration
                    if st.active_duration is not None
                    else st.ch_sample_n - st.trig_abs)))),
        }
        if below_index is not None:
            pulse_data["below_threshold_index"] = int(below_index)
            if 0 <= below_index < len(ts_win):
                pulse_data["below_threshold_time"] = float(
                    ts_win[below_index])

        ch_key = f"Channel {channel}"
        self.pulse_count[ch_key] += 1
        k = self.pulse_count[ch_key]

        # The only way a completed pulse leaves the detector.  A consumer
        # that wants them all in memory (trigger_capture) collects them
        # here; one that streams to disk writes them here.  Either way
        # the detector keeps nothing.
        if self.on_pulse is not None:
            self.on_pulse(channel, k, pulse_data)

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
        st.re_trigger_ready = False
        st.active_duration = None
        st.pileup_child = False


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
    jump_lag: int = 0,
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
    jump_lag : int
        Edge-detector lookback K in samples.  When positive, the σ of
        the lag-K difference ``x[n] - x[n-K]`` is measured from the
        record (``jump_std_*``) — self-calibrating the edge threshold to
        the filter correlation and 1/f power actually present at that
        lag.  Records too short for the lag fall back to the
        white-noise value √2·σ.

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

        # ── Lag-K jump σ for the edge detector ────────────────
        # Measured directly from the record's lag-K differences, so
        # correlated filters and 1/f power at that lag are priced in
        # rather than assumed away.  MAD keeps pulse edges (a tiny
        # minority of lag pairs) from inflating it.
        if jump_lag > 0 and len(arr) > jump_lag + 16:
            jump_std_I = _robust_std(arr.real[jump_lag:]
                                     - arr.real[:-jump_lag])
            jump_std_Q = _robust_std(arr.imag[jump_lag:]
                                     - arr.imag[:-jump_lag])
        else:
            jump_std_I = _SQRT2 * robust_std_I
            jump_std_Q = _SQRT2 * robust_std_Q

        noise_stats[c] = ChannelNoiseStats(
            mean_I=robust_mean_I,
            std_I=robust_std_I,
            mean_Q=robust_mean_Q,
            std_Q=robust_std_Q,
            jump_std_I=jump_std_I,
            jump_std_Q=jump_std_Q,
        )

    return noise_stats, raw_data


