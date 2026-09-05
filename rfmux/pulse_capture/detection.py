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
    )

    # 3. Feed samples.  Completed pulses arrive through on_pulse; the
    #    engine holds only the ring buffer, so a capture can run
    #    indefinitely without growing.
    for channel, i_val, q_val, timestamp in sample_stream:
        pcap.process_sample(channel, i_val, q_val, timestamp)
"""

from __future__ import annotations

import math

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from . import walk as _walk

_SQRT2 = math.sqrt(2.0)

# ── Ring geometry ───────────────────────────────────────────────────
# Every time scale in the engine is a fraction of the longest pulse
# the ring was sized for, so ONE user-facing number (max_pulse_ms) sets
# them all.  The two constants live here, in the layer that owns the
# ring, and PulseCaptureConfig derives from them — otherwise a bare
# PulseCapture(buf_size=...) and a config-driven one resolve different
# lags from the same intent.

#: End-of-pulse threshold, in sigma: the one definition the engine, the
#: session and the config share.
DEFAULT_END_SIGMA = 1.0

#: Ring headroom over the longest expected pulse.  The pre-trigger
#: margin and the end-confirmation tail share the ring with the pulse.
BUFFER_SAFETY: float = 1.5

#: Fraction of the ring a capture may fill before the hard stop.  With
#: the ring at 1.5x the max expected pulse, 0.8 puts the stop at 1.2x
#: that pulse, leaving room for the pre-trigger margin in the same ring.
HARD_STOP_RING_FRACTION: float = 0.8


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

    def extend(self, values) -> None:
        """Append many values at once, as ``add`` would one at a time.

        Leaves ``ptr`` and ``count`` exactly where the equivalent run of
        ``add`` calls would, including when *values* is longer than the
        ring: only the last N survive, but they land at the position the
        full sequence would have left them at.

        Also Periscope's display path: a batch of samples lands in one
        call rather than one ``add`` per sample.

        Values are coerced to the buffer's dtype, so a ``None``
        timestamp becomes NaN exactly as ``add`` leaves it.  Dropping
        that coercion silently produces an object array instead.
        """
        v = np.asarray(values, dtype=self.buf.dtype)
        m = v.shape[0]
        if m == 0:
            return
        keep = v[-self.N:] if m > self.N else v
        k = keep.shape[0]
        # Where the surviving values begin once all m have gone by.
        start = (self.ptr + m - k) % self.N
        head = min(k, self.N - start)
        # Both halves of the doubled buffer, so data() stays a contiguous
        # view no matter where the write wrapped.
        self.buf[start:start + head] = keep[:head]
        self.buf[start + self.N:start + self.N + head] = keep[:head]
        if head < k:
            tail = k - head
            self.buf[:tail] = keep[head:]
            self.buf[self.N:self.N + tail] = keep[head:]
        self.ptr = (self.ptr + m) % self.N
        self.count = min(self.count + m, self.N)

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
    # (trig_abs may sit up to a lookback earlier when drift had held
    # the run above threshold).  Capture-relative edge references clip
    # to this, so they can never reach past the physical pulse onset.
    fire_abs: int = 0
    # Pre-pulse level snapshot (median of the edge taps at fire time):
    # a baseline-free end reference.  The rolling median can lag 1/f by
    # several σ, leaving the amplitude test above threshold for
    # a whole capture — but returning to the level the pulse ROSE FROM
    # is decisive evidence the pulse is over, however stale the mean.
    anchor_I: float = 0.0
    anchor_Q: float = 0.0
    # The band the trigger was tested against, frozen at the trigger:
    # the baseline refresh re-centres the stats object afterwards, so
    # it no longer says what the trigger saw.
    trig_mean_I: float = 0.0
    trig_mean_Q: float = 0.0
    trig_std_I: float = 0.0
    trig_std_Q: float = 0.0
    # The quadrature whose deviation started the current above-threshold
    # run, and the one the capture triggered on.
    run_quad: str = ""
    trig_quad: str = ""
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
    """Streaming multi-channel pulse-detection engine, triggering on I and Q.

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
    started.

    How much of that is SAVED is ``save_to_end_confirmed``.  Off (the
    default) ends the window at the below-threshold instant plus a
    ``margin_fraction`` tail — where the eye puts the end of the pulse,
    rather than where the confirmation finished — which keeps window
    length a property of the pulse instead of the baseline, at the cost
    of the tail.  On keeps every sample the state machine saw,
    confirmation tail included.  Either way ``duration_ms`` is measured from the threshold
    crossings, so it does not move with this setting.

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
        otherwise arrive at ~2.8 Hz per channel on the PFB stream at
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
        wander.  0 disables the edge test: amplitude-only triggering,
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

    #: Walk blocks with the transcribed state machine (walk.walk) rather
    #: than process_sample per sample.  False keeps the per-sample path,
    #: which is the reference the tests hold the walk to.
    use_walk = True

    #: Crossings closer than this, in samples, are walked as one island
    #: by process_block's per-sample path (use_walk off) rather than
    #: bulked apart.
    _MERGE_GAP = 32

    #: Entries kept for the rolling-baseline median.  Fixed, so cost and
    #: memory do not scale with the window: a decimated reservoir tracks
    #: the full-stream median to ~0.01 sigma.
    _BASELINE_RESERVOIR: int = 4096

    @staticmethod
    def default_edge_lookback(buf_size: int,
                              margin_fraction: float = 0.1) -> int:
        """Edge-detector lag derived from the ring: ``margin_fraction``
        of the longest pulse the ring was sized for (the ring is
        ``BUFFER_SAFETY`` times that pulse).  Shared with noise
        estimation so the measured jump-σ is taken at the same lag the
        edge detector uses, and equal by construction to
        ``PulseCaptureConfig.edge_lookback_samples`` for the same
        intent."""
        return max(1, int(round(
            margin_fraction * buf_size / BUFFER_SAFETY)))

    @staticmethod
    def default_max_capture_samples(buf_size: int) -> int:
        """Hard stop derived from the ring, so a capture can never
        outlive the buffer and silently lose its rising edge."""
        return max(0, int(round(HARD_STOP_RING_FRACTION * buf_size)))

    def __init__(
        self,
        buf_size: int,
        channels: List[int],
        noise_stats: Dict[int, ChannelNoiseStats],
        threshold_sigma: float = 5.0,
        end_sigma: float = DEFAULT_END_SIGMA,
        margin_fraction: float = 0.1,
        min_pulse_samples: int = 0,
        trigger_samples: int = 2,
        enable_pileup: bool = True,
        save_to_end_confirmed: bool = False,
        min_end_samples: int = 10,
        on_pulse: Optional[Callable[[int, int, dict], None]] = None,
        baseline_window: int = 0,
        edge_lookback: Optional[int] = None,
        max_capture_samples: Optional[int] = None,
    ):
        self.channels = list(channels)
        self.buf_size = buf_size
        self.threshold_sigma = threshold_sigma
        self.end_sigma = end_sigma
        self.margin_fraction = margin_fraction
        self.min_pulse_samples = min_pulse_samples
        self.trigger_samples = max(1, int(trigger_samples))
        self.enable_pileup = enable_pileup
        self.save_to_end_confirmed = save_to_end_confirmed
        # Floor under the end-confirmation count.  end_ptr_count counts
        # up while both quadratures are settled and down when they are
        # not, so an isolated noisy sample does not restart the
        # confirmation; without a floor a very short pulse would end its
        # capture almost as soon as it began.  A sample count: 17 ms at
        # 596 Hz, 4 us on the PFB stream.
        self.min_end_samples = max(1, int(min_end_samples))

        if edge_lookback is None:
            edge_lookback = self.default_edge_lookback(buf_size,
                                                       margin_fraction)
        self.edge_lookback = max(0, int(edge_lookback))
        if max_capture_samples is None:
            max_capture_samples = self.default_max_capture_samples(buf_size)
        self.max_capture_samples = max(0, int(max_capture_samples))

        # Callback for streaming consumers (HDF5, GUI, etc.)
        # Signature: on_pulse(channel: int, pulse_idx: int, pulse_data: dict)
        self.on_pulse = on_pulse

        # Per-channel noise stats
        self.noise_stats = noise_stats

        # ── Rolling baseline ──────────────────────────────────────
        # 1/f drift moves the true baseline while sigma stays put, so a
        # mean fixed at training turns quiet samples into deviations:
        # triggers fire on the drift, and the end condition can become
        # unsatisfiable because the signal never returns inside a band
        # centred on a stale mean.
        #
        # sigma comes from the training record, as the MAD about a
        # block-median baseline, which drift cannot corrupt.  The mean is re-estimated as a
        # running median over a window long compared with a pulse: a
        # median because it ignores pulses rather than being pulled by
        # them, holding to ~0.15 sigma at 10% pulse duty.
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
        _walk.warm_up()

        # Build channel → index lookup
        self._ch_set = set(self.channels)

        # Per-channel capture state
        self.state: Dict[int, _ChState] = {c: _ChState() for c in self.channels}

        # Results.  Completed pulses leave through on_pulse and are not
        # retained here — the engine's memory is the ring buffer and
        # nothing else, so a capture can run for as long as you like.
        self.start_time: Optional[float] = None
        #: Per-channel pulse counter, keyed by channel number.  It
        #: generates the index that names every HDF5 group, so it is
        #: not merely a statistic.
        self.pulse_count: Dict[int, int] = {c: 0 for c in self.channels}

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

    # ── Block ingestion ───────────────────────────────────────────

    def _samples_until_refresh(self, st: "_ChState") -> int:
        """Samples from now until a rolling-baseline refresh moves the
        mean.  ``0`` when the baseline is frozen (no refresh ever)."""
        if self.baseline_window <= 0:
            return 0
        to_insert = self._bl_decim - st.decim_n
        inserts_left = self._bl_refresh - st.since_refresh - 1
        return inserts_left * self._bl_decim + to_insert

    def _bulk_quiet(self, channel: int, st: "_ChState", I: np.ndarray,
                    Q: np.ndarray, T: np.ndarray) -> None:
        """Absorb a run of samples that provably cannot do anything.

        Valid only when the engine is not capturing, no sample in the
        run reaches ``threshold_sigma``, and the baseline mean does not
        move within it.  Under those three conditions process_sample's
        entire body reduces to: append to the ring, feed the decimated
        baseline reservoir, and advance the counters.  No trigger can
        fire, no capture can end, and the edge detector — which only
        runs on eligible samples — never looks.
        """
        n = I.shape[0]
        bufs = self.buf[channel]
        bufs["I"].extend(I)
        bufs["Q"].extend(Q)
        bufs["ts"].extend(T)

        if self.baseline_window > 0:
            # First insertion lands where decim_n would have wrapped.
            first = self._bl_decim - 1 - st.decim_n
            if first < n:
                idx = np.arange(first, n, self._bl_decim)
                self._bl[channel]["I"].extend(I[idx])
                self._bl[channel]["Q"].extend(Q[idx])
                st.since_refresh += idx.shape[0]
            st.decim_n = (st.decim_n + n) % self._bl_decim

        self.abs_n += n
        st.ch_sample_n += n
        # Nothing in the run was above threshold, by precondition.
        st.above_run = 0

    def process_block(
        self,
        channel: int,
        i_vals,
        q_vals,
        timestamps,
    ) -> None:
        """Ingest many samples at once, identically to calling
        :meth:`process_sample` on each in turn.

        Split by what a sample can do.  A run in which no sample reaches
        ``threshold_sigma``, with no capture open and no baseline
        refresh inside, cannot trigger, cannot end anything, and is
        never looked at by the edge detector: it enters the ring and the
        baseline reservoir, which :meth:`_bulk_quiet` does with numpy.
        Everything else is walked by :func:`walk.walk`, the state
        machine compiled, or by process_sample when ``use_walk`` is
        off.  A baseline refresh falls on exactly one sample, and that
        sample always goes through process_sample.
        """
        if channel not in self._ch_set:
            return

        I = np.ascontiguousarray(i_vals, dtype=np.float64)
        Q = np.ascontiguousarray(q_vals, dtype=np.float64)
        T = np.ascontiguousarray(timestamps, dtype=np.float64)
        n = I.shape[0]
        if not (n == Q.shape[0] == T.shape[0]):
            raise ValueError("process_block needs equal-length arrays")

        st = self.state[channel]
        pos = 0
        while pos < n:
            if st.capturing:
                # Mid-pulse: every sample can end the capture, split it,
                # or hit the hard stop.  Walked, up to the sample a
                # baseline refresh falls on, which process_sample takes.
                if self.use_walk:
                    until = self._samples_until_refresh(st)
                    if until == 1:
                        self.process_sample(channel, I[pos], Q[pos], T[pos])
                        pos += 1
                        continue
                    stop = n if until <= 0 else min(n, pos + until - 1)
                    pos = self._walk_block(channel, st, I, Q, T, pos, stop)
                    continue
                self.process_sample(channel, I[pos], Q[pos], T[pos])
                pos += 1
                continue

            # How far ahead the current baseline mean is still the one
            # the deviations will be measured against.  _bulk_quiet
            # cannot re-centre the band, so the sample that moves the
            # mean has to go through process_sample.
            until = self._samples_until_refresh(st)
            if until == 1:
                self.process_sample(channel, I[pos], Q[pos], T[pos])
                pos += 1
                continue
            end = n if until <= 0 else min(n, pos + until - 1)

            ns = self.noise_stats.get(channel)
            if ns is None:
                ns = ChannelNoiseStats()
            si = max(ns.std_I, 1e-30)
            sq = max(ns.std_Q, 1e-30)
            seg_I = I[pos:end]
            seg_Q = Q[pos:end]
            above = ((np.abs(seg_I - ns.mean_I) / si > self.threshold_sigma)
                     | (np.abs(seg_Q - ns.mean_Q) / sq
                        > self.threshold_sigma))

            hits = np.flatnonzero(above)
            if hits.shape[0] == 0:
                self._bulk_quiet(channel, st, seg_I, seg_Q, T[pos:end])
                pos = end
                continue

            # Absorb the quiet lead-in, walk only the hit ISLANDS, and
            # bulk the provably-quiet gaps between them.  A gap sample,
            # not capturing and below threshold, does exactly what
            # _bulk_quiet does: reset the trigger run, feed the ring and
            # the baseline reservoir, advance the counters.  The edge
            # detector never looks at it (it runs only on eligible,
            # above-threshold samples), and its lag-K lookback reaches
            # back through bulked samples because they are in the ring.
            # So bulking a gap equals walking it, the same equivalence
            # the lead-in relies on.  Crossings closer than _MERGE_GAP
            # stay in one island: bulking a handful of samples costs
            # more than walking them.
            seg = pos
            first = int(hits[0])
            if first > 0:
                self._bulk_quiet(channel, st, I[seg:seg + first],
                                 Q[seg:seg + first], T[seg:seg + first])
                pos = seg + first
            if self.use_walk:
                # One stretch from the first crossing to the last, gaps
                # included: on a gap sample the walk does what
                # _bulk_quiet does, and one call costs more than
                # walking a gap of any length this side of a refresh.
                # The quiet tail after the last crossing is the next
                # segment's lead-in.
                pos = self._walk_block(channel, st, I, Q, T, pos,
                                       seg + int(hits[-1]) + 1)
                continue
            h = 0
            nhits = hits.shape[0]
            while pos < seg + int(hits[-1]) + 1:
                # pos sits at a crossing (an island start).  Extend the
                # island while the next crossing is within MERGE_GAP.
                while (h + 1 < nhits
                       and int(hits[h + 1]) - int(hits[h]) <= self._MERGE_GAP):
                    h += 1
                walk_stop = seg + int(hits[h]) + 1
                while pos < walk_stop:
                    self.process_sample(channel, I[pos], Q[pos], T[pos])
                    pos += 1
                    if st.capturing:
                        break  # let the capturing branch take over
                if st.capturing:
                    break
                h += 1
                if h >= nhits:
                    break  # quiet tail after the last island -> outer bulk
                nxt = seg + int(hits[h])
                if nxt > pos:
                    self._bulk_quiet(channel, st, I[pos:nxt],
                                     Q[pos:nxt], T[pos:nxt])
                    pos = nxt

    _QUAD = {"": 0, "I": 1, "Q": 2}
    _QUAD_NAMES = ("", "I", "Q")

    def _pack_state(self, st: "_ChState") -> Tuple[np.ndarray, np.ndarray]:
        si = np.empty(_walk.N_INT, dtype=np.int64)
        sf = np.empty(_walk.N_FLT, dtype=np.float64)
        si[_walk.CAPTURING] = 1 if st.capturing else 0
        si[_walk.END_PTR] = st.end_ptr_count
        si[_walk.TRIG_ABS] = -1 if st.trig_abs is None else st.trig_abs
        si[_walk.FIRE_ABS] = st.fire_abs
        si[_walk.RUN_QUAD] = self._QUAD[st.run_quad]
        si[_walk.TRIG_QUAD] = self._QUAD[st.trig_quad]
        si[_walk.PILEUP_CHILD] = 1 if st.pileup_child else 0
        si[_walk.CH_N] = st.ch_sample_n
        si[_walk.RETRIG] = 1 if st.re_trigger_ready else 0
        si[_walk.ACTIVE_DUR] = (-1 if st.active_duration is None
                                else st.active_duration)
        si[_walk.ABOVE_RUN] = st.above_run
        si[_walk.RUN_START] = st.run_start_abs
        si[_walk.EPOCH] = st.epoch_start
        si[_walk.DECIM_N] = st.decim_n
        si[_walk.SINCE_REFRESH] = st.since_refresh
        sf[_walk.ANCHOR_I] = st.anchor_I
        sf[_walk.ANCHOR_Q] = st.anchor_Q
        sf[_walk.TMEAN_I] = st.trig_mean_I
        sf[_walk.TMEAN_Q] = st.trig_mean_Q
        sf[_walk.TSTD_I] = st.trig_std_I
        sf[_walk.TSTD_Q] = st.trig_std_Q
        sf[_walk.NEAR_I] = math.nan
        sf[_walk.NEAR_Q] = math.nan
        return si, sf

    def _unpack_state(self, st: "_ChState", si, sf) -> None:
        st.capturing = bool(si[_walk.CAPTURING])
        st.end_ptr_count = int(si[_walk.END_PTR])
        st.trig_abs = None if si[_walk.TRIG_ABS] < 0 else int(si[_walk.TRIG_ABS])
        st.fire_abs = int(si[_walk.FIRE_ABS])
        st.run_quad = self._QUAD_NAMES[int(si[_walk.RUN_QUAD])]
        st.trig_quad = self._QUAD_NAMES[int(si[_walk.TRIG_QUAD])]
        st.pileup_child = bool(si[_walk.PILEUP_CHILD])
        st.ch_sample_n = int(si[_walk.CH_N])
        st.re_trigger_ready = bool(si[_walk.RETRIG])
        st.active_duration = (None if si[_walk.ACTIVE_DUR] < 0
                              else int(si[_walk.ACTIVE_DUR]))
        st.above_run = int(si[_walk.ABOVE_RUN])
        st.run_start_abs = int(si[_walk.RUN_START])
        st.epoch_start = int(si[_walk.EPOCH])
        st.decim_n = int(si[_walk.DECIM_N])
        st.since_refresh = int(si[_walk.SINCE_REFRESH])
        st.anchor_I = float(sf[_walk.ANCHOR_I])
        st.anchor_Q = float(sf[_walk.ANCHOR_Q])
        st.trig_mean_I = float(sf[_walk.TMEAN_I])
        st.trig_mean_Q = float(sf[_walk.TMEAN_Q])
        st.trig_std_I = float(sf[_walk.TSTD_I])
        st.trig_std_Q = float(sf[_walk.TSTD_Q])

    def _walk_block(self, channel: int, st: "_ChState", I: np.ndarray,
                    Q: np.ndarray, T: np.ndarray, start: int,
                    stop: int) -> int:
        """process_sample over samples start..stop-1 through walk.walk,
        handling what the walk returns for.  Returns the index to
        continue from."""
        bufs = self.buf[channel]
        rI, rQ, rT = bufs["I"], bufs["Q"], bufs["ts"]
        bl = self._bl.get(channel) if self.baseline_window > 0 else None
        ns = self.noise_stats.get(channel, ChannelNoiseStats())
        si, sf = self._pack_state(st)
        out = np.zeros(6, dtype=np.int64)
        pos = start
        while pos < stop:
            if bl is not None:
                bI, bQ = bl["I"].buf, bl["Q"].buf
                bptr, bcount, bN = bl["I"].ptr, bl["I"].count, bl["I"].N
            else:
                bI = bQ = rI.buf
                bptr = bcount = 0
                bN = 1
            _walk.walk(I, Q, T, pos, stop,
                       rI.buf, rQ.buf, rT.buf, rI.ptr, rI.count, rI.N,
                       bI, bQ, bptr, bcount, bN, self._bl_decim,
                       bl is not None,
                       float(ns.mean_I), float(ns.mean_Q),
                       float(ns.std_I), float(ns.std_Q),
                       float(ns.jump_std_I), float(ns.jump_std_Q),
                       float(self.threshold_sigma), float(self.end_sigma),
                       int(self.trigger_samples), int(self.edge_lookback),
                       int(self.min_end_samples), float(self.margin_fraction),
                       int(self.max_capture_samples), bool(self.enable_pileup),
                       bool(self.freeze_triggers), si, sf, out)
            k, reason = int(out[0]), int(out[1])
            rI.ptr = rQ.ptr = rT.ptr = int(out[2])
            rI.count = rQ.count = rT.count = int(out[3])
            if bl is not None:
                bl["I"].ptr = bl["Q"].ptr = int(out[4])
                bl["I"].count = bl["Q"].count = int(out[5])
            self.abs_n += k - pos if reason == _walk.DONE else k - pos + 1
            self._unpack_state(st, si, sf)
            if reason == _walk.DONE:
                return stop
            if reason == _walk.END:
                self._save_pulse(channel)
            elif reason == _walk.HARD_STOP:
                self._save_pulse(channel, truncated=True)
            elif reason == _walk.SPLIT:
                self._save_pulse(channel, pileup=True)
                if not self.freeze_triggers:
                    anchor = ((float(sf[_walk.NEAR_I]),
                               float(sf[_walk.NEAR_Q]))
                              if math.isfinite(sf[_walk.NEAR_I]) else None)
                    self._rearm_after_split(st, ns, anchor)
            si, sf = self._pack_state(st)
            pos = k + 1
        return stop

    def _rearm_after_split(self, st: "_ChState", ns: ChannelNoiseStats,
                           anchor) -> None:
        """Open a capture for the pulse that rose on the tail of the
        one just saved: dated where the nearest tap saw that tail,
        anchored on it (it decays onto the previous pulse's tail, not
        the pre-pulse level), and marked a fragment of a chain."""
        self._begin_capture(st, ns)
        st.trig_abs = max(st.run_start_abs,
                          st.ch_sample_n - max(1, self.min_end_samples))
        if anchor is not None:
            st.anchor_I, st.anchor_Q = anchor
        st.pileup_child = True

    @staticmethod
    def _begin_capture(st: "_ChState", ns: ChannelNoiseStats) -> None:
        """Open a capture on *st*, keeping the band it was decided against."""
        st.capturing = True
        st.end_ptr_count = 0
        st.fire_abs = st.ch_sample_n
        st.trig_mean_I, st.trig_mean_Q = ns.mean_I, ns.mean_Q
        st.trig_std_I, st.trig_std_Q = ns.std_I, ns.std_Q
        st.trig_quad = st.run_quad

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

        # Update circular buffers.  Bound to locals because this method
        # runs once per sample per channel — 2.44 MHz on the PFB stream
        # — and repeated self.buf[channel][...] lookups are a
        # measurable fraction of that budget.
        bufs = self.buf[channel]
        bI = bufs["I"]
        bQ = bufs["Q"]
        bI.add(i_val)
        bQ.add(q_val)
        bufs["ts"].add(timestamp)

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
        # per channel on the slow stream at decimation stage 0 and
        # about 2.8 PER SECOND on the PFB stream.  Requiring two consecutive samples costs a
        # real pulse nothing — anything above threshold for a single
        # sample carries no measurable rise or decay anyway — while
        # cutting the accidental rate by orders of magnitude.
        if max_dev > self.threshold_sigma:
            if st.above_run == 0:
                st.run_start_abs = st.ch_sample_n
                st.run_quad = "I" if dev_I >= dev_Q else "Q"
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
        # Only evaluated when it can matter.  Both results are read in
        # exactly one place — the "not capturing and trigger_ok" branch
        # below — and trigger_ok needs `eligible`, so on a sample that
        # is neither above threshold nor mid-capture the block would be
        # computed and thrown away.  That is nearly every sample of a
        # quiet stream: the engine's hot loop.
        if self.edge_lookback > 0 and eligible and not st.capturing:
            # The usable lag shortens near the start of the stream and
            # after a statistics epoch reset — a reference from before
            # either would difference across a mean shift and read as
            # a jump.
            lmax = min(self.edge_lookback,
                       bI.count - 1,
                       st.ch_sample_n - st.epoch_start - 1)
            if lmax >= 1:
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
                # fake a rise out of held drift (max-like).  All taps
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
        # With the edge test the trigger can fire at any point in the
        # above-threshold run, not only the sample that completed the
        # confirmation — a rise spread over several samples earns its
        # jump as it grows.  Without the edge detector (edge_lookback
        # 0, debug only) a fire-once-per-run rule stands in for it: a
        # run that outlives one capture must not re-fire.
        if self.edge_lookback > 0:
            trigger_ok = eligible and edge_ok
        else:
            trigger_ok = st.above_run == self.trigger_samples
        if not st.capturing and not self.freeze_triggers and trigger_ok:
            self._begin_capture(st, ns)
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
            # deep: if drift held the run above threshold long ago,
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
        #     end_samples confirmation counts.  Normal single-pulse.
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
        #     edge.  Worst case is one max-pulse of dead time.
        #
        # Counting down rather than resetting is what makes an isolated
        # noisy sample harmless.  That matters most on the PFB stream,
        # where Gaussian noise exceeds 1.5σ on individual samples even
        # between pulses.
        if st.capturing:
            since_trig = st.ch_sample_n - (st.trig_abs or st.ch_sample_n)
            since_fire = st.ch_sample_n - st.fire_abs

            # ── Capture-relative edge tests ───────────────────────
            # Taps CLIPPED to samples since the trigger FIRED: a
            # reference reaching further back lands on the previous
            # pulse — or, when drift held the run and the window was
            # dated a lookback early, on pre-pulse wander — and either
            # reads as false decay evidence or as a false rise.
            # Unclipped, a split re-splits the successor capture
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
            # Both results, and near_vals with them, feed nothing but
            # the pileup split below — so with splitting off this is
            # six ring reads per sample of every capture, discarded.
            # Both are judged on the length of the deviation vector,
            # (dev_I, dev_Q) in sigma units, not per quadrature.  A
            # pulse rotates in the IQ plane as it settles -- on the PFB
            # stream Q overshoots and settles while I swings the other
            # way -- and per quadrature that reads as decay on one axis
            # and a rise on the other, which is the split signature.
            # The length is the same whichever way the vector points.
            # Jumps are scaled by the larger of the two lag-K jump-sigmas,
            # in sigma units.
            if (self.enable_pileup and self.edge_lookback > 0
                    and since_fire >= 1):
                span = min(self.edge_lookback, since_fire,
                           bI.count - 1,
                           st.ch_sample_n - st.epoch_start - 1)
                if span >= 1:
                    sI = max(ns.std_I, 1e-30)
                    sQ = max(ns.std_Q, 1e-30)
                    jn = max(js_I / sI, js_Q / sQ, 1e-30)
                    mag = math.hypot(dev_I, dev_Q)
                    hi = 0.0
                    for tap in (span, span // 2, span // 4):
                        if tap >= 1:
                            hi = max(hi, math.hypot(
                                (bI.recent(tap) - ns.mean_I) / sI,
                                (bQ.recent(tap) - ns.mean_Q) / sQ))
                    decaying_now = (mag - hi) / jn < -self.threshold_sigma
                    # The pulse's own recent level: min_end_samples
                    # back, as far as the decay evidence had to wait.
                    near = max(1, min(self.min_end_samples, span))
                    near_vals = (bI.recent(near), bQ.recent(near))
                    near_mag = math.hypot(
                        (near_vals[0] - ns.mean_I) / sI,
                        (near_vals[1] - ns.mean_Q) / sQ)
                    rising_above_self = (
                        (mag - near_mag) / jn > self.threshold_sigma)

            # ── Baseline-free return test ─────────────────────────
            # Back at the pre-pulse anchor on BOTH quadratures.  The
            # amplitude tests below compare against the tracked mean,
            # which can lag 1/f by several σ and hold dev above
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
                    and since_fire > self.min_end_samples):
                st.re_trigger_ready = True

            # ── Pileup split ──────────────────────────────────────
            # A confirmed run that rose above the current pulse's own
            # recent level, after the current pulse was seen decaying.
            if (self.enable_pileup and self.edge_lookback > 0
                    and st.re_trigger_ready and eligible
                    and rising_above_self):
                self._save_pulse(channel, pileup=True)
                if not self.freeze_triggers:
                    self._rearm_after_split(st, ns, near_vals)
                return

            # ── Normal end: baseline confirmation ─────────────────
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
                self.min_end_samples,
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
        # Buffer arithmetic is per channel: ch_sample_n counts only this
        # channel's samples, where abs_n counts every channel's.
        raw_post = st.ch_sample_n - (st.trig_abs or st.ch_sample_n)
        core = st.active_duration
        if pileup or core is None:
            # No below-threshold instant to anchor on: a split ends at
            # the split sample, and a hard stop that never saw the pulse
            # end keeps everything it has.  Trim off the confirmation
            # count (less a 5-sample margin) so the window ends near
            # where the signal settled, not where the counter finished.
            post = raw_post - max(0, st.end_ptr_count - 5)
        elif self.save_to_end_confirmed:
            # Keep everything the state machine saw, confirmation tail
            # included.  Those samples are already in the ring, so this
            # trades disk for a decay tail that is otherwise discarded
            # at the below-threshold instant plus a small margin.
            #
            # It does make the window length depend on how long the
            # confirmation took, which is a baseline property rather
            # than a pulse property.  That is why duration is measured
            # from the threshold crossings (below_threshold_time -
            # trigger_time) and not from the length of this window.
            #
            # +1 because the window end is exclusive and raw_post counts
            # samples SINCE the trigger: without it the sample the end
            # was confirmed on — the whole point of the policy — is the
            # one sample left out.
            post = raw_post + 1
            truncated = False
        else:
            # The pulse visibly ended at below-threshold (core samples
            # after the trigger).  Save margin_fraction of it as tail
            # and drop the slow confirmation stretch: the confirmation
            # count (or max_capture_samples) bounds the STATE MACHINE
            # only, and does not stretch the data.  A capture stopped
            # because drift stalled the confirmation still holds a
            # complete pulse, so it is NOT flagged truncated.
            tail = max(self.min_end_samples,
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
        # Under save_to_end_confirmed the end index is the last saved
        # sample.  Without it the index is normally PAST the window:
        # the data stops at below-threshold plus the tail margin while
        # the state machine keeps running until the confirmation count
        # (or max_capture_samples) releases it.  Times are carried alongside the
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
            # The bands each decision was made against: the trigger's
            # rolling band of that instant, and the end's pre-pulse
            # anchor.  The stats object keeps neither.
            "trigger_baseline_I": float(st.trig_mean_I),
            "trigger_baseline_Q": float(st.trig_mean_Q),
            "trigger_sigma_I": float(st.trig_std_I),
            "trigger_sigma_Q": float(st.trig_std_Q),
            "trigger_quad": st.trig_quad,
            "end_baseline_I": float(st.anchor_I),
            "end_baseline_Q": float(st.anchor_Q),
            "threshold_sigma": float(self.threshold_sigma),
            "end_sigma": float(self.end_sigma),
            "end_confirm_samples": int(st.end_ptr_count),
            "end_confirm_target": int(max(
                self.min_end_samples,
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

        self.pulse_count[channel] += 1
        k = self.pulse_count[channel]

        # The only way a completed pulse leaves the engine.  A consumer
        # that wants them all in memory (trigger_capture) collects them
        # here; one that streams to disk writes them here.  Either way
        # the engine keeps nothing.
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

        # A sample fed without a timestamp holds NaN, which fails both
        # comparisons and so stays outside every window.
        with np.errstate(invalid="ignore"):
            mask = (ts_data >= t_start) & (ts_data <= t_end)

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


def _block_median_baseline(x: np.ndarray, block: int) -> np.ndarray:
    """The record's slow baseline: medians of consecutive *block*-long
    stretches, interpolated between block centres and held at the ends.

    A median per block ignores pulses while they are a minority of it,
    as the engine's rolling median does, and the whole thing is one pass
    over the record whatever the block.  ``block`` 0, or one longer than
    half the record, means one median for the whole record.
    """
    n = len(x)
    block = n if block <= 0 else int(block)
    block = max(64, min(block, n))
    nb = n // block
    if nb < 2:
        return np.full(n, np.median(x))
    cut = nb * block
    meds = np.median(x[:cut].reshape(nb, block), axis=1)
    centres = (np.arange(nb) + 0.5) * block
    return np.interp(np.arange(n), centres, meds)


def estimate_noise_stats(
    samples_by_channel: Dict[int, np.ndarray],
    channels: List[int],
    jump_lag: int = 0,
    baseline_block: int = 0,
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
    baseline_block : int
        Samples per block of the baseline σ is measured against: long
        compared with a pulse, so a pulse stays a minority of its
        block's median, and short compared with the record, so wander
        slower than a few pulses is baseline rather than noise.  The
        session passes three captures.  0 means one median for the
        record.

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

        # ── Noise estimation: median + MAD about the baseline ─
        # Baseline mean: median is robust to asymmetric pulse
        # contamination (up to 50% outliers).
        # Noise σ: the MAD of the samples about a block-median
        # baseline a few pulses long.  The baseline removes drift and
        # pulse tails, and the MAD ignores the pulses themselves.  Not
        # the σ of adjacent differences over √2: that is exact only
        # for white noise, and the CIC decimators correlate
        # neighbouring slow-stream samples enough to read it 1.3x
        # (stage 0) to 1.6x (stages above) low.
        robust_mean_I = float(np.median(arr.real))
        robust_mean_Q = float(np.median(arr.imag))
        robust_std_I = _robust_std(
            arr.real - _block_median_baseline(arr.real, baseline_block))
        robust_std_Q = _robust_std(
            arr.imag - _block_median_baseline(arr.imag, baseline_block))

        # Refine baseline mean using the now-correct σ to clip
        # pulse outliers.  The median can be biased when pulses
        # cross zero (asymmetric contamination), but 3σ clipping
        # with the correct σ from the baseline-subtracted MAD rejects
        # them.
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


