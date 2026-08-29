"""
Live pulse-capture orchestration, independent of any GUI.

:class:`PulseCaptureSession` ties together the pieces that already exist
in this package — noise estimation, the :class:`PulseCapture` detection
engine, the streaming :class:`PulseHDF5Writer`, and the running
:class:`PulseHistogramSet` — into a single object with a start/stop
lifecycle that a GUI (or script) can drive by feeding it samples.

The session is deliberately free of Qt, sockets, and threads: samples
arrive through :meth:`feed_sample`, or a whole block at a time through
:meth:`feed_block`, from whatever source the caller has (the Periscope
slow-stream tap, a PFB receive loop, a synthetic stream in tests), and
results leave through plain callbacks.  The Periscope
``PulseCaptureTask`` is that adapter on the GUI side: it drives
:class:`~.sources.SlowIngest` into :meth:`feed_block` from its own
thread and re-emits the callbacks as Qt signals.  ``SlowIngest`` is the
same class :func:`~.sources.run_slow_source` uses, so the GUI and a
headless script cannot interpret a stream differently.

Lifecycle::

    capture_session = PulseCaptureSession(
        channels=[1, 2],
        threshold_sigma=5.0,
        hdf5_path="capture.h5",
        on_pulse=lambda ch, idx, summary, data: ...,
    )
    capture_session.start()             # begins noise estimation
    for ch, i, q, t in sample_stream:
        capture_session.feed_sample(ch, i, q, t)
    capture_session.stop()              # finalizes the HDF5 file

State machine::

    IDLE ──start()──▶ ESTIMATING ──(noise_samples reached)──▶ CAPTURING
                          ▲                                       │
                          └──────────re_estimate_noise()──────────┘
    any ──stop()──▶ STOPPED (terminal; HDF5 finalized)

Thread safety: none.  All methods must be called from a single thread —
in Periscope, the capture task thread (h5py writes must stay on one
thread).

:class:`DualPulseCaptureSession`, below, composes two of these — the slow
readout stream and the fast PFB stream — with an
:class:`IncrementalPulseMatcher` so pulses seen in both become pairs.  It
lives here rather than in its own module because it is the same object
one layer up: it shares :class:`_CallbackHost`, the state machine and the
config, and a module that reaches into another's private base class was
never two modules.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .. import streamer
from ..core.transferfunctions import PFB_SAMPLING_FREQ

from .detection import (
    DEFAULT_END_SIGMA,
    BUFFER_SAFETY,
    HARD_STOP_RING_FRACTION,
    ChannelNoiseStats,
    PulseCapture,
    estimate_noise_stats,
)
from .analysis import pulse_summary
from .accumulators import PulseHistogramSet, PulseTemplateSet

from .hdf5 import DualPulseHDF5Writer, PulseHDF5Writer


#: The detection knobs, named once.
#:
#: They travel together: PulseCaptureConfig.session_kwargs() produces
#: them, the session holds them, PulseCapture consumes them, and the
#: HDF5 file records them.  Every hand-maintained copy of the list is a
#: chance for those to disagree.
DETECTION_PARAMS = (
    "threshold_sigma",
    "end_sigma",
    "margin_fraction",
    "min_pulse_samples",
    "trigger_samples",
    "enable_pileup",
    "save_to_end_confirmed",
    "baseline_window",
    "edge_lookback",
    "max_capture_samples",
)


class CaptureState(Enum):
    IDLE = "idle"
    ESTIMATING = "estimating"
    CAPTURING = "capturing"
    STOPPED = "stopped"


class _CallbackHost:
    """Callback dispatch shared by the single- and dual-stream sessions.

    Every result leaves a session through a plain callback, and a
    callback that raises must not take the capture down with it: a GUI
    repaint failing is no reason to stop writing pulses to disk.  So
    exceptions are caught and reported through ``on_error`` — which is
    itself allowed to fail silently, because at that point there is
    nowhere left to report to.
    """

    #: Class-level default so _error() is safe before __init__ has run
    #: far enough to assign the instance attribute.
    on_error: Optional[Callable] = None

    def _callback(self, cb: Optional[Callable], *args) -> None:
        if cb is None:
            return
        try:
            cb(*args)
        except Exception as e:
            self._error(f"Callback {getattr(cb, '__name__', cb)!r} "
                        f"raised: {e}")

    #: Class-level default so _to_writer() is safe on a host that has not
    #: opened a writer (hdf5_path=None, or h5py missing).
    writer = None

    def _to_writer(self, method: str, *args, what: str = "") -> None:
        """Call *method* on the HDF5 writer, if there is one.

        Ten copies of ``if self.writer is not None: try: ... except:
        self._error(...)`` said the same thing ten times, and one of the
        writer calls had been left unguarded.  A failing write must not
        take the capture down: the samples keep flowing and the error is
        reported, which is the same rule :meth:`_callback` follows.
        """
        if self.writer is None:
            return
        try:
            getattr(self.writer, method)(*args)
        except Exception as e:
            self._error(f"HDF5 {what or method} failed: {e}")

    def _error(self, message: str) -> None:
        if self.on_error is not None:
            try:
                self.on_error(message)
            except Exception:
                pass


@dataclass
class PulseCaptureConfig:
    """User-facing pulse-capture parameters, in physical units.

    Time-like quantities are milliseconds; :meth:`session_kwargs`
    converts them to samples for a given stream rate, and auto-sizes
    the ring buffer from the longest expected pulse (the buffer is the
    hard ceiling on recordable pulse length — a capture that outlasts
    it silently loses its rising edge).
    """

    threshold_sigma: float = 5.0
    end_sigma: float = DEFAULT_END_SIGMA
    #: Consecutive samples that must clear the threshold to trigger.
    #: 0 = choose it from the stream rate, which is the right default:
    #: the accidental rate scales with sample rate, so one sample is
    #: plenty of evidence at 596 Hz (2.5 accidentals per HOUR) and
    #: nowhere near enough at 2.44 MHz (2.8 per SECOND).  Demanding two
    #: everywhere would reject real pulses on a heavily decimated slow
    #: stream, where a fast pulse spans less than one sample.
    trigger_samples: int = 0
    #: Accidental-trigger budget used to pick trigger_samples, per
    #: channel per minute.
    max_accidental_per_min: float = 1.0
    margin_fraction: float = 0.1
    min_pulse_ms: float = 0.0      # 0 = no glitch rejection
    #: Longest pulse the ring must hold, and the basis for the floor
    #: under the baseline tracking window.  Estimate it generously — a
    #: capture that outlasts the ring loses its rising edge.
    max_pulse_ms: float = 250.0
    #: Training length override, in SAMPLE time.  0 (the default)
    #: derives it from the pulse length — see noise_train_span_ms().
    noise_train_ms: float = 0.0
    enable_pileup: bool = True
    #: Keep samples all the way to the end-of-pulse CONFIRMATION rather
    #: than stopping a ``margin_fraction`` tail past the below-threshold
    #: instant.  The extra samples are already in the ring, so this
    #: costs disk, not acquisition.
    #:
    #: Off gives windows whose length tracks the pulse; on gives windows
    #: whose length also tracks how long the end confirmation took to be
    #: satisfied, which depends on where the baseline was wandering.  On
    #: the mock that widens the spread across identical injected pulses
    #: from roughly 1.3x to 5.6x.
    #:
    #: Default on, because that variability is a property of the saved
    #: TAIL and does not reach ``duration_ms``, which is measured from
    #: the threshold crossings.  Turn it off when
    #: the tail costs more than it is worth: PFB captures, where windows
    #: already carry 64x the samples, or high count rates, where longer
    #: windows overlap and raise the pileup fraction.
    save_to_end_confirmed: bool = True

    #: Ring geometry, owned by pulse_detection so the engine's bare
    #: defaults and these derivations cannot disagree.
    BUFFER_SAFETY = BUFFER_SAFETY
    _MIN_BUF = 1000
    _MIN_NOISE = 200
    #: Ceiling on the held training record, per channel.  complex128, so
    #: this is 32 MB/channel.
    _MAX_NOISE = 2_000_000
    #: Fewest ring-lengths the rolling-baseline median must span, so a
    #: max-length pulse stays a clear minority of its window.
    BASELINE_MIN_RINGS = 8
    #: Training window as a multiple of the max pulse length.  Tying it
    #: to the pulse scale is what makes one setting work across every
    #: stream rate: the training window has to be long compared with a
    #: pulse (so the fit sees baseline, not signal), and that ratio —
    #: not any absolute duration — is the thing that matters.
    NOISE_TRAIN_PULSES = 20
    #: Hard stop on a capture, as a multiple of the max pulse length —
    #: the ring fraction expressed against the pulse rather than the
    #: ring, so it is the same stop the engine's own default computes.
    #: The ring (1.5x) still has room for the pre-trigger margin, and a
    #: baseline that drifted mid-capture can delay the end condition but
    #: never wedge the detector.
    HARD_STOP_FACTOR = HARD_STOP_RING_FRACTION * BUFFER_SAFETY

    # ── ms → samples (per stream rate) ────────────────────────────

    def min_pulse_samples(self, sample_rate: float) -> int:
        return int(round(self.min_pulse_ms * 1e-3 * sample_rate))

    def buf_size(self, sample_rate: float) -> int:
        need = self.max_pulse_ms * 1e-3 * sample_rate * self.BUFFER_SAFETY
        return max(self._MIN_BUF, int(math.ceil(need)))

    def noise_samples(self, sample_rate: float) -> int:
        """Training length in samples, memory-bounded.

        The record is held whole (it is fitted, not streamed), so a
        duration that is comfortable on the slow stream is not on the
        PFB one: 30 s at 2.44 MHz is 73.2M samples per channel.  The cap
        binds only at fast rates, where a long span is neither needed
        nor achievable — describe() reports the duration actually used.
        """
        want = int(round(self.noise_train_span_ms() * 1e-3 * sample_rate))
        return max(self._MIN_NOISE, min(want, self._MAX_NOISE))

    def noise_train_span_ms(self) -> float:
        """Effective training length.

        Derived from the pulse length by default: the window must be
        long compared with a pulse for the fit to see baseline rather
        than signal, and expressing that as a ratio means it follows
        whatever pulse scale the user sets instead of needing its own
        answer.  A positive noise_train_ms overrides it.
        """
        if self.noise_train_ms > 0:
            return self.noise_train_ms
        return self.NOISE_TRAIN_PULSES * self.max_pulse_ms

    def baseline_window_samples(self, sample_rate: float) -> int:
        """Span of the rolling-baseline median, in samples.

        The same window the noise fit used: long compared with a pulse,
        which is exactly the requirement.  Floored against the ring in
        case the training length was overridden short — the median only
        ignores pulses while they are a minority of its window, and the
        ring holds one max-length pulse.
        """
        return max(self.noise_samples(sample_rate),
                   self.BASELINE_MIN_RINGS * self.buf_size(sample_rate))

    def _cross_prob(self) -> float:
        """Per-sample probability that noise alone clears the threshold
        on either quadrature."""
        p1 = math.erfc(self.threshold_sigma / math.sqrt(2.0))
        return 1.0 - (1.0 - p1) ** 2

    def trigger_samples_for(self, sample_rate: float) -> int:
        """Confirmation length: the fewest samples that hold accidental
        triggers under the budget at this rate.

        Deriving it rather than fixing it is what lets one setting work
        across a 2000x span of stream rates: at 596 Hz the answer is 1,
        at 38 kHz and above it is 2.
        """
        if self.trigger_samples > 0:
            return self.trigger_samples
        p = self._cross_prob()
        n = 1
        while (60.0 * sample_rate * p ** n > self.max_accidental_per_min
               and n < 16):
            n += 1
        return n

    def accidental_rate_hz(self, sample_rate: float) -> float:
        """Expected triggers per second per channel on noise alone.

        Gaussian tail on either quadrature, raised to the confirmation
        length.  Real samples are correlated by the CIC/PFB response so
        this understates the confirmed rate, but the single-sample
        figure is exact and is the one that bites: at 5 sigma it is
        ~1.4 Hz per channel on the PFB stream.
        """
        return (sample_rate
                * self._cross_prob() ** self.trigger_samples_for(sample_rate))

    def max_pulse_samples(self, sample_rate: float) -> int:
        return max(1, int(round(self.max_pulse_ms * 1e-3 * sample_rate)))

    def edge_lookback_samples(self, sample_rate: float) -> int:
        """Edge-detector lag K: margin_fraction of the max pulse.

        Long enough to contain any physical rise (KID rise times are a
        small fraction of the decay the pulse length is set from), short
        enough that 1/f wander moves negligibly across it.  The jump-σ
        the edge threshold uses is measured from the training record at
        exactly this lag."""
        return max(1, int(round(self.margin_fraction
                                * self.max_pulse_samples(sample_rate))))

    def max_capture_samples(self, sample_rate: float) -> int:
        """Hard stop on a capture — HARD_STOP_FACTOR × the max pulse,
        floored so the stop can never sit inside the end-confirmation
        count of a legitimate short pulse."""
        return max(32, int(round(self.HARD_STOP_FACTOR
                                 * self.max_pulse_samples(sample_rate))))

    def session_kwargs(self, sample_rate: float) -> Dict[str, Any]:
        """Keyword arguments for :class:`PulseCaptureSession`."""
        return {
            "threshold_sigma": self.threshold_sigma,
            "end_sigma": self.end_sigma,
            "margin_fraction": self.margin_fraction,
            "min_pulse_samples": self.min_pulse_samples(sample_rate),
            "trigger_samples": self.trigger_samples_for(sample_rate),
            "enable_pileup": self.enable_pileup,
            "save_to_end_confirmed": self.save_to_end_confirmed,
            "buf_size": self.buf_size(sample_rate),
            "noise_samples": self.noise_samples(sample_rate),
            "baseline_window": self.baseline_window_samples(sample_rate),
            "edge_lookback": self.edge_lookback_samples(sample_rate),
            "max_capture_samples": self.max_capture_samples(sample_rate),
        }

    def describe(self, sample_rate: float,
                 n_channels: int = 1) -> Dict[str, Any]:
        """Derived quantities at a given stream rate (for display)."""
        buf = self.buf_size(sample_rate)
        return {
            "sample_rate_hz": sample_rate,
            "min_pulse_samples": self.min_pulse_samples(sample_rate),
            "noise_samples": self.noise_samples(sample_rate),
            "noise_train_span_ms": self.noise_train_span_ms(),
            "noise_train_actual_ms":
                self.noise_samples(sample_rate) / sample_rate * 1e3,
            "buf_samples": buf,
            "buf_mb_per_channel": buf * 3 * 8 / 1e6,
            "buf_mb_total": buf * 3 * 8 * n_channels / 1e6,
            "max_recordable_ms": buf / sample_rate * 1e3,
            "baseline_window": self.baseline_window_samples(sample_rate),
            "baseline_window_ms":
                self.baseline_window_samples(sample_rate)
                / sample_rate * 1e3,
            "trigger_samples": self.trigger_samples_for(sample_rate),
            "accidental_per_min":
                60.0 * self.accidental_rate_hz(sample_rate),
            "accidental_per_min_unconfirmed":
                60.0 * sample_rate * self._cross_prob(),
            "edge_lookback": self.edge_lookback_samples(sample_rate),
            "edge_lookback_ms":
                self.edge_lookback_samples(sample_rate) / sample_rate * 1e3,
            "max_capture_samples": self.max_capture_samples(sample_rate),
            "max_capture_ms":
                self.max_capture_samples(sample_rate) / sample_rate * 1e3,
            # For a marginal pulse in white noise the edge test is the
            # stricter of the two: it compares against √2·σ.  Shown so
            # the user knows the effective amplitude floor.
            "edge_floor_sigma": self.threshold_sigma * math.sqrt(2.0),
        }

    def validate(self, sample_rate: Optional[float] = None
                 ) -> List[tuple]:
        """[(severity, message), ...] — severities error/warning/info."""
        issues: List[tuple] = []
        if self.threshold_sigma <= 0:
            issues.append(("error", "Threshold σ must be positive."))
        if self.end_sigma <= 0:
            issues.append(("error", "End σ must be positive."))
        if self.end_sigma >= self.threshold_sigma:
            issues.append(("error",
                           "End σ must sit below the trigger threshold "
                           f"({self.end_sigma:g} ≥ "
                           f"{self.threshold_sigma:g})."))
        elif self.end_sigma < 1.2:
            issues.append((
                "warning",
                f"End σ {self.end_sigma:g} < 1.2: the end condition "
                "needs BOTH I and Q inside the band (~47% per sample at "
                "1.0σ on Gaussian noise) — pulse termination becomes a "
                "random walk. 1.5σ recommended."))
        if self.threshold_sigma < 3:
            issues.append(("warning",
                           f"Threshold {self.threshold_sigma:g}σ will "
                           "trigger frequently on plain noise."))
        if self.trigger_samples < 0:
            issues.append(("error",
                           "Trigger confirmation cannot be negative."))
        if not 0 <= self.margin_fraction <= 1:
            issues.append(("error",
                           "Margin fraction must be within 0–1."))
        if self.max_pulse_ms <= 0:
            issues.append(("error", "Max pulse length must be positive."))
        if self.min_pulse_ms < 0:
            issues.append(("error", "Min pulse length cannot be negative."))
        if self.min_pulse_ms and self.min_pulse_ms >= self.max_pulse_ms:
            issues.append(("error",
                           "Min pulse length must be below max pulse "
                           "length."))
        if self.noise_train_ms < 0:
            issues.append(("error",
                           "Noise training override cannot be negative."))
        if sample_rate:
            acc = 60.0 * self.accidental_rate_hz(sample_rate)
            if acc > 1.0:
                issues.append((
                    "warning",
                    f"Noise alone will trigger about {acc:,.0f} times per "
                    f"minute per channel at {self.threshold_sigma:g}σ and "
                    f"{sample_rate:,.0f} Hz. Raise the threshold or the "
                    "confirmation length."))
            elif acc > 0.01:
                issues.append((
                    "info",
                    f"Accidental trigger rate ≈ {acc:.2g}/min per channel."))
            if self.min_pulse_ms > 0 \
                    and self.min_pulse_samples(sample_rate) < 2:
                issues.append((
                    "warning",
                    f"Min pulse {self.min_pulse_ms:g} ms is under 2 "
                    f"samples at {sample_rate:,.0f} Hz — the glitch "
                    "filter will be ineffective."))
            mb = self.describe(sample_rate)["buf_mb_per_channel"]
            if mb > 100:
                issues.append((
                    "warning",
                    f"Ring buffer is {mb:.0f} MB per channel at this "
                    "rate — consider a shorter max pulse length."))
        return issues


class PulseCaptureSession(_CallbackHost):
    """Callback-driven live pulse capture (noise → detect → HDF5 → histograms).

    Parameters
    ----------
    channels : list[int]
        1-indexed channel numbers to monitor.
    module : int
        Module number (metadata only).
    streamer_mode : str
        "slow", "fast", or "both" (metadata only — the session is
        agnostic to where samples come from).
    threshold_sigma, end_sigma, margin_fraction, min_pulse_samples,
    enable_pileup, buf_size :
        Passed through to :class:`PulseCapture`.
    sample_rate : float, optional
        Nominal sample rate in Hz (metadata only; all timing derives
        from the timestamps fed to :meth:`feed_sample`).
    noise_samples : int
        Samples per channel to accumulate before estimating noise.
    hdf5_path : str or Path, optional
        When given, a :class:`PulseHDF5Writer` streams every pulse to
        this file; when None, no file is written.
    df_calibrations : dict[int, float], optional
        Per-channel Hz-per-count calibration, stored in the HDF5 file.
    histogram_flush_every : int
        Flush histograms to HDF5 and fire ``on_histograms`` every N
        pulses (and once at stop).  Default 50.
    progress_interval_s : float
        Floor on the wall time between noise-training progress
        callbacks.  They fire per channel per block, so a wide capture
        emits thousands a second without this.  Default 0.1.
    histogram_flush_interval_s : float
        Also flush when this long has passed since the last one, so the
        live view keeps up at low count rates instead of waiting for
        the 50th pulse.  The pulse count still bounds the work at high
        rates; this only bounds the WAIT at low ones.  Default 0.5, so
        a quiet capture costs at most two flushes a second.

    Callbacks (all optional; exceptions are caught and routed to
    ``on_error``):

    - ``on_noise(noise_stats: dict[int, ChannelNoiseStats])`` — after
      each (re-)estimation completes.
    - ``on_pulse(channel, pulse_idx, summary, pulse_data)`` — per
      detected pulse; *summary* is :func:`pulse_summary` scalars,
      *pulse_data* the waveform dict.
    - ``on_stats(stats: dict)`` — after each pulse (cheap counters).
    - ``on_histograms(data: dict)`` — every ``histogram_flush_every``
      pulses and at stop.
    - ``on_error(message: str)``.
    """

    def __init__(
        self,
        channels: List[int],
        *,
        module: int = 1,
        streamer_mode: str = "slow",
        threshold_sigma: float = 5.0,
        end_sigma: float = DEFAULT_END_SIGMA,
        margin_fraction: float = 0.1,
        min_pulse_samples: int = 0,
        trigger_samples: int = 2,
        enable_pileup: bool = True,
        save_to_end_confirmed: bool = True,
        buf_size: int = 5000,
        sample_rate: Optional[float] = None,
        noise_samples: int = 1000,
        baseline_window: int = 0,
        edge_lookback: Optional[int] = None,
        max_capture_samples: Optional[int] = None,
        hdf5_path: Optional[str | Path] = None,
        df_calibrations: Optional[Dict[int, float]] = None,
        histogram_flush_every: int = 50,
        histogram_flush_interval_s: float = 0.5,
        progress_interval_s: float = 0.1,
        on_noise: Optional[Callable] = None,
        on_pulse: Optional[Callable] = None,
        on_stats: Optional[Callable] = None,
        on_histograms: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_progress: Optional[Callable] = None,
        on_templates: Optional[Callable] = None,
    ):
        self.channels = list(channels)
        self.module = module
        self.streamer_mode = streamer_mode
        self.threshold_sigma = threshold_sigma
        self.end_sigma = end_sigma
        self.margin_fraction = margin_fraction
        self.min_pulse_samples = min_pulse_samples
        self.trigger_samples = max(1, int(trigger_samples))
        self.enable_pileup = enable_pileup
        self.save_to_end_confirmed = save_to_end_confirmed
        self.buf_size = buf_size
        self.sample_rate = sample_rate
        self.noise_samples = int(noise_samples)
        self.baseline_window = int(baseline_window)
        # Resolved here (not in the engine) so noise estimation measures
        # the jump-σ at exactly the lag the edge detector will use — but
        # through the engine's own resolvers, so there is one definition
        # of each default rather than a copy that can drift.
        if edge_lookback is None:
            edge_lookback = PulseCapture.default_edge_lookback(
                buf_size, margin_fraction)
        self.edge_lookback = max(0, int(edge_lookback))
        if max_capture_samples is None:
            max_capture_samples = PulseCapture.default_max_capture_samples(
                buf_size)
        self.max_capture_samples = max(0, int(max_capture_samples))
        self.hdf5_path = Path(hdf5_path) if hdf5_path is not None else None
        self.df_calibrations = df_calibrations
        self.histogram_flush_every = int(histogram_flush_every)
        self.histogram_flush_interval_s = float(histogram_flush_interval_s)
        self._last_flush_t = 0.0
        self.progress_interval_s = float(progress_interval_s)
        self._last_progress_t = 0.0
        self._progress_dirty = False

        self.on_noise = on_noise
        self.on_pulse = on_pulse
        self.on_stats = on_stats
        self.on_histograms = on_histograms
        self.on_error = on_error
        self.on_templates = on_templates
        self.on_progress = on_progress

        self.histograms = PulseHistogramSet(threshold_sigma=threshold_sigma)

        self.templates: PulseTemplateSet
        self._build_templates()

        self.state = CaptureState.IDLE
        self.noise_stats: Dict[int, ChannelNoiseStats] = {}
        # Raw complex noise-training arrays from the last estimation
        # (kept for visualization: what the estimator actually saw).
        self.noise_data: Dict[int, np.ndarray] = {}
        self.pcap: Optional[PulseCapture] = None
        self.writer = None

        # Noise-estimation accumulation.  Preallocated numpy rather
        # than a list of Python complex objects: the record is held
        # whole, and at 4x the memory per sample a long training run
        # on a fast stream would not fit.
        self._noise_buf: Dict[int, np.ndarray] = {}
        self._noise_n: Dict[int, int] = {}
        #: Samples that arrived after a channel filled its training
        #: quota but before the SESSION left ESTIMATING (other channels
        #: still training).  Held rather than dropped -- see feed_block.
        self._pending_post_noise: Dict[int, tuple] = {}

        # Counters / timing (sample time, from fed timestamps)
        self.pulse_counts: Dict[int, int] = {c: 0 for c in self.channels}
        self.total_pulses = 0
        self.dropped_invalid_ts = 0
        self._first_ts: Optional[float] = None
        self._last_ts: Optional[float] = None
        self._pulses_since_flush = 0

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Begin the session: enter noise estimation."""
        if self.state is not CaptureState.IDLE:
            raise RuntimeError(f"start() called in state {self.state.name}")
        self._alloc_noise_buffers()
        self.state = CaptureState.ESTIMATING

    def re_estimate_noise(self) -> None:
        """Freeze triggering and collect a fresh noise estimate.

        Valid only while CAPTURING.  Pulse capture resumes automatically
        (with the updated statistics) once ``noise_samples`` new samples
        per channel have been collected.
        """
        if self.state is not CaptureState.CAPTURING:
            raise RuntimeError(
                f"re_estimate_noise() called in state {self.state.name}")
        if self.pcap is not None:
            self.pcap.freeze_triggers = True
        self._alloc_noise_buffers()
        self.state = CaptureState.ESTIMATING

    def stop(self) -> None:
        """End the session and finalize the HDF5 file.  Idempotent.

        Any capture still in progress is abandoned (its samples remain
        in the ring buffer only).
        """
        if self.state is CaptureState.STOPPED:
            return
        if self.pcap is not None:
            self.pcap.freeze_triggers = True
        self._flush_histograms()
        self._to_writer("finalize", what="finalize")
        self.state = CaptureState.STOPPED

    # ── Sample ingestion ──────────────────────────────────────────

    def feed_sample(
        self,
        channel: int,
        i_val: float,
        q_val: float,
        timestamp: Optional[float],
    ) -> None:
        """Ingest one I/Q sample.  Dispatches on the session state.

        During ESTIMATING, samples accumulate for the noise fit
        (timestamps are not needed).  During CAPTURING, samples with a
        None/NaN timestamp are dropped and counted — all pulse timing
        derives from these timestamps, so NaN would silently poison
        durations and tau.
        """
        if self.state is CaptureState.ESTIMATING:
            buf = self._noise_buf.get(channel)
            n = self._noise_n.get(channel, 0)
            if buf is None or n >= self.noise_samples:
                self._maybe_finish_estimation()
                return
            buf[n] = complex(i_val, q_val)
            n += 1
            self._noise_n[channel] = n
            # Same rule as _absorb_noise: a channel that has just filled
            # always reports, otherwise the wall clock decides.  Two
            # different rate limits for one callback was one too many.
            self._progress_dirty = True
            if (n >= self.noise_samples
                    or time.monotonic() - self._last_progress_t
                    >= self.progress_interval_s):
                self._emit_progress()
            if n >= self.noise_samples:
                self._maybe_finish_estimation()
            return

        if self.state is not CaptureState.CAPTURING or self.pcap is None:
            return

        if timestamp is None or not math.isfinite(timestamp):
            self.dropped_invalid_ts += 1
            return

        if self._first_ts is None:
            self._first_ts = float(timestamp)
        self._last_ts = float(timestamp)

        self.pcap.process_sample(channel, float(i_val), float(q_val),
                                 float(timestamp))

    def feed_block(
        self,
        channel: int,
        i_vals,
        q_vals,
        timestamps,
    ) -> None:
        """Ingest many samples of one channel at once.

        Equivalent to :meth:`feed_sample` per element, but hands whole
        arrays to :meth:`PulseCapture.process_block`, which absorbs
        quiet stretches with numpy instead of a Python loop.  That is
        what makes the 2.44 MHz PFB stream tractable — see
        process_block for why the per-sample path cannot get there.

        A block may straddle the end of noise training, so it is split
        at the transition and each part dispatched on the state it
        belongs to.
        """
        I = np.asarray(i_vals, dtype=np.float64)
        Q = np.asarray(q_vals, dtype=np.float64)
        T = np.asarray(timestamps, dtype=np.float64)
        n = I.shape[0]
        if not (n == Q.shape[0] == T.shape[0]):
            raise ValueError("feed_block needs equal-length arrays")

        pos = 0
        while pos < n:
            if self.state is CaptureState.ESTIMATING:
                taken = self._absorb_noise(channel, I[pos:], Q[pos:])
                if taken == 0:
                    if pos < n and channel in self._noise_buf:
                        # This channel has its quota but the session is
                        # still ESTIMATING because another channel has
                        # not.  Dropping here would cost up to a whole
                        # block -- 1000 samples on the PFB stream --
                        # where per-sample feeding loses at most one.
                        # Hold it for the transition instead.
                        self._hold_post_noise(channel, I[pos:], Q[pos:],
                                              T[pos:])
                    return  # unknown channel, or state did not advance
                pos += taken
                continue

            if self.state is not CaptureState.CAPTURING or self.pcap is None:
                return

            seg_I, seg_Q, seg_T = I[pos:], Q[pos:], T[pos:]
            # Same rule as feed_sample: a sample with no usable
            # timestamp is dropped and counted, because every pulse
            # duration and tau is measured from these.
            good = np.isfinite(seg_T)
            if not good.all():
                self.dropped_invalid_ts += int((~good).sum())
                seg_I, seg_Q, seg_T = seg_I[good], seg_Q[good], seg_T[good]
            if seg_T.shape[0]:
                if self._first_ts is None:
                    self._first_ts = float(seg_T[0])
                self._last_ts = float(seg_T[-1])
                self.pcap.process_block(channel, seg_I, seg_Q, seg_T)
            return

    def _hold_post_noise(self, channel: int, I: np.ndarray, Q: np.ndarray,
                         T: np.ndarray) -> None:
        """Stash a block tail until the session starts capturing."""
        held = self._pending_post_noise.get(channel)
        if held is None:
            self._pending_post_noise[channel] = (I.copy(), Q.copy(),
                                                 T.copy())
        else:
            self._pending_post_noise[channel] = (
                np.concatenate((held[0], I)),
                np.concatenate((held[1], Q)),
                np.concatenate((held[2], T)))

    def _drain_post_noise(self) -> None:
        """Feed everything held during training, now that we capture."""
        pending, self._pending_post_noise = self._pending_post_noise, {}
        for channel, (I, Q, T) in pending.items():
            self.feed_block(channel, I, Q, T)

    def _emit_progress(self) -> None:
        self._progress_dirty = False
        self._last_progress_t = time.monotonic()
        self._callback(self.on_progress, {
            "state": self.state.value,
            "collected": dict(self._noise_n),
            "target": self.noise_samples,
        })

    def flush_progress(self) -> None:
        """Report training counts that the rate limit is still holding.

        Call when the sample flow pauses.  The limit exists to stop a
        fast stream from flooding the listener, not to hide the latest
        state -- and a channel the stream never delivers is only
        visible as one that stopped advancing, which needs the last
        update to actually arrive.
        """
        if self.state is CaptureState.ESTIMATING and self._progress_dirty:
            self._emit_progress()

    def _absorb_noise(self, channel: int, I: np.ndarray,
                      Q: np.ndarray) -> int:
        """Take as much of a block as the training record still wants.

        Returns how many samples were consumed — the caller re-dispatches
        the remainder, which by then belongs to the capturing state.
        """
        buf = self._noise_buf.get(channel)
        n = self._noise_n.get(channel, 0)
        if buf is None or n >= self.noise_samples:
            self._maybe_finish_estimation()
            return 0

        take = min(self.noise_samples - n, I.shape[0])
        buf[n:n + take] = I[:take] + 1j * Q[:take]
        n += take
        self._noise_n[channel] = n

        # Rate-limited, because this fires per CHANNEL per block: 128
        # channels training at stage 0 is ~15,500 callbacks a second,
        # each copying a 128-entry dict and, in Periscope, crossing a
        # Qt queued connection to a handler that rebuilds a
        # 128-line label.  That storm is itself enough to make the GUI
        # miss packets, which then slows the very training it reports.
        # A channel that has just finished always reports, so the
        # display still settles on the final counts.
        self._progress_dirty = True
        now = time.monotonic()
        if (n >= self.noise_samples
                or now - self._last_progress_t >= self.progress_interval_s):
            self._emit_progress()
        if n >= self.noise_samples:
            self._maybe_finish_estimation()
        return take

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Current counters, rate, and elapsed sample time."""
        elapsed = 0.0
        if self._first_ts is not None and self._last_ts is not None:
            elapsed = self._last_ts - self._first_ts
        rate_per_min = (
            60.0 * self.total_pulses / elapsed if elapsed > 0 else 0.0)
        return {
            "state": self.state.value,
            "total_pulses": self.total_pulses,
            "per_channel": dict(self.pulse_counts),
            "elapsed_s": elapsed,
            "rate_per_min": rate_per_min,
            "dropped_invalid_ts": self.dropped_invalid_ts,
            "hdf5_path": str(self.hdf5_path) if self.hdf5_path else None,
            "baseline_window": self.baseline_window,
            "baseline_window_ms": (
                self.baseline_window / self.sample_rate * 1e3
                if self.sample_rate else None),
        }

    # ── Internals ─────────────────────────────────────────────────

    def _build_templates(self) -> None:
        """Size the trigger-aligned stacking window from the ring, so it
        covers the longest expected pulse without unbounded memory."""
        post = max(64, min(self.buf_size // 2, 20000))
        self.templates = PulseTemplateSet(
            pre_samples=max(8, post // 10), post_samples=post,
            threshold_sigma=self.threshold_sigma,
            sample_rate=self.sample_rate)

    def _alloc_noise_buffers(self) -> None:
        self._noise_buf = {c: np.empty(self.noise_samples,
                                       dtype=np.complex128)
                           for c in self.channels}
        self._noise_n = {c: 0 for c in self.channels}
        self._pending_post_noise = {}

    def _maybe_finish_estimation(self) -> None:
        if any(self._noise_n.get(c, 0) < self.noise_samples
               for c in self.channels):
            return

        samples = {c: self._noise_buf[c][:self._noise_n[c]]
                   for c in self.channels}
        self.noise_stats, self.noise_data = estimate_noise_stats(
            samples, self.channels, jump_lag=self.edge_lookback)

        if self.pcap is None:
            self._build_engine_and_writer()
        else:
            # Re-estimation: swap stats in place and resume triggering.
            # The edge detector must not difference across the swap —
            # samples taken under the old mean read as huge jumps
            # against the new one.
            self.pcap.noise_stats = self.noise_stats
            self.pcap.reset_edge_history()
            if self.writer is not None:
                self.writer.update_noise_stats(self.noise_stats)
            self.pcap.freeze_triggers = False

        self.state = CaptureState.CAPTURING
        self._callback(self.on_noise, self.noise_stats)
        # Only now: a listener must hear "noise estimated" before it
        # hears about a pulse found in the samples we held back.
        self._drain_post_noise()

    def _build_engine_and_writer(self) -> None:
        # One dict, two consumers: what the engine runs on is what the
        # file records, with no second list to keep in step.
        detection = {name: getattr(self, name) for name in DETECTION_PARAMS}

        self.pcap = PulseCapture(
            buf_size=self.buf_size,
            channels=self.channels,
            noise_stats=self.noise_stats,
            on_pulse=self._on_engine_pulse,
            **detection,
        )

        if self.hdf5_path is not None:
            capture_params = {
                **detection,
                "streamer_mode": self.streamer_mode,
                "module": self.module,
            }
            if self.sample_rate:
                key = ("sample_rate_fast" if self.streamer_mode == "fast"
                       else "sample_rate_slow")
                capture_params[key] = self.sample_rate
            try:
                self.writer = PulseHDF5Writer(
                    self.hdf5_path,
                    self.channels,
                    self.noise_stats,
                    capture_params,
                    df_calibrations=self.df_calibrations,
                )
            except Exception as e:
                self.writer = None
                self._error(f"Could not open HDF5 file {self.hdf5_path}: {e}")

    def _on_engine_pulse(self, channel: int, pulse_idx: int,
                         pulse_data: dict) -> None:
        ns = self.noise_stats.get(channel)
        summary = pulse_summary(pulse_data, ns, self.threshold_sigma)

        self.pulse_counts[channel] = self.pulse_counts.get(channel, 0) + 1
        self.total_pulses += 1
        self._pulses_since_flush += 1

        self._to_writer("append_pulse", channel, pulse_idx, pulse_data,
                        what=f"write for pulse ch{channel}#{pulse_idx}")

        self.histograms.add_pulse(channel, pulse_data, ns)
        self.templates.add_pulse(channel, pulse_data, ns)

        self._callback(self.on_pulse, channel, pulse_idx, summary, pulse_data)
        self._callback(self.on_stats, self.stats())

        # Count OR clock: the count keeps a busy capture from flushing
        # on every pulse, the clock keeps a quiet one from looking dead
        # until the 50th arrives.
        if self._pulses_since_flush and (
                self._pulses_since_flush >= self.histogram_flush_every
                or (time.monotonic() - self._last_flush_t
                    >= self.histogram_flush_interval_s)):
            self._flush_histograms()

    def _flush_histograms(self) -> None:
        self._pulses_since_flush = 0
        self._last_flush_t = time.monotonic()
        if self.histograms.total_pulses() == 0:
            return
        data = self.histograms.get_histogram_data()
        self._to_writer("update_histograms", data, what="histogram update")
        self._callback(self.on_histograms, data)

        tmpl = self.templates.get_template_data()
        if tmpl:
            self._to_writer("update_templates", tmpl,
                            what="template update")
            self._callback(self.on_templates, tmpl)


# ───────────────────────── Incremental matcher ──────────────────────

@dataclass
class _Pending:
    pulse_idx: int
    mid: float
    summary: dict


class IncrementalPulseMatcher:
    """Streaming version of trigger_capture's ``_match_pulses``.

    Pulses from the two streams are paired per channel when their
    midpoint times fall within ``window_s`` (best match wins).  A pulse
    that finds no partner within ``grace_s`` of stream time is emitted
    as a one-sided pair.  ``on_pair`` receives dicts::

        {"channel", "pair_idx", "slow_idx" | None, "fast_idx" | None,
         "time_offset" | None, "slow_summary" | None,
         "fast_summary" | None}
    """

    def __init__(self, window_s: float = 0.05, grace_s: float = 0.25,
                 on_pair: Optional[Callable[[dict], None]] = None):
        self.window_s = window_s
        self.grace_s = grace_s
        self.on_pair = on_pair
        self._pending: Dict[str, Dict[int, List[_Pending]]] = {
            "slow": {}, "fast": {}}
        self._pair_counts: Dict[int, int] = {}
        # Per-stream clocks: a pulse in stream S may only be declared
        # one-sided once the OTHER stream's time has provably passed it
        # — robust to stream skew (sequential feeding, stalled socket).
        self._latest: Dict[str, Optional[float]] = {
            "slow": None, "fast": None}
        self.matched = 0
        self.unmatched = 0

    @staticmethod
    def _other(stream: str) -> str:
        return "fast" if stream == "slow" else "slow"

    def advance_time(self, stream: str, now: float) -> None:
        """Advance *stream*'s clock from stream time (not just pulse
        arrivals) so the OTHER stream's one-sided pulses expire
        ~grace_s after the event — while ring buffers still cover it —
        instead of waiting for the next pulse or capture stop."""
        if not math.isfinite(now):
            return
        latest = self._latest[stream]
        if latest is None or now > latest:
            self._latest[stream] = now
            self._expire()

    def add(self, stream: str, channel: int, pulse_idx: int,
            summary: dict) -> None:
        mid = summary.get("timestamp", 0.0) \
            + summary.get("duration_s", 0.0) / 2.0
        if not math.isfinite(mid):
            return

        latest = self._latest[stream]
        if latest is None or mid > latest:
            self._latest[stream] = mid

        # Best partner in the other stream's pending list
        others = self._pending[self._other(stream)].setdefault(channel, [])
        best_i, best_diff = -1, self.window_s
        for i, cand in enumerate(others):
            diff = abs(cand.mid - mid)
            if diff <= best_diff:
                best_i, best_diff = i, diff
        if best_i >= 0:
            partner = others.pop(best_i)
            mine = _Pending(pulse_idx, mid, summary)
            slow, fast = ((mine, partner) if stream == "slow"
                          else (partner, mine))
            self.matched += 1
            self._emit(channel, slow=slow, fast=fast)
        else:
            self._pending[stream].setdefault(channel, []).append(
                _Pending(pulse_idx, mid, summary))

        self._expire()

    def flush(self) -> None:
        """Emit every remaining pending pulse as one-sided (on stop)."""
        self._expire(force=True)

    # ── internals ─────────────────────────────────────────────────

    def _expire(self, force: bool = False) -> None:
        for stream in ("slow", "fast"):
            if force:
                cutoff = float("inf")
            else:
                other_latest = self._latest[self._other(stream)]
                if other_latest is None:
                    continue  # other stream has no time yet — can't judge
                cutoff = other_latest - self.grace_s
            for channel, pend in self._pending[stream].items():
                keep: List[_Pending] = []
                for p in pend:
                    if p.mid < cutoff:
                        self.unmatched += 1
                        self._emit(channel,
                                   slow=p if stream == "slow" else None,
                                   fast=p if stream == "fast" else None)
                    else:
                        keep.append(p)
                self._pending[stream][channel] = keep

    def _emit(self, channel: int, slow: Optional[_Pending],
              fast: Optional[_Pending]) -> None:
        self._pair_counts[channel] = self._pair_counts.get(channel, 0) + 1
        pair = {
            "channel": channel,
            "pair_idx": self._pair_counts[channel],
            "slow_idx": slow.pulse_idx if slow else None,
            "fast_idx": fast.pulse_idx if fast else None,
            "slow_summary": slow.summary if slow else None,
            "fast_summary": fast.summary if fast else None,
            "time_offset": (slow.mid - fast.mid)
            if slow and fast else None,
        }
        if self.on_pair is not None:
            self.on_pair(pair)


# ───────────────────────── Dual session ─────────────────────────────

class DualPulseCaptureSession(_CallbackHost):
    """Slow + fast capture with live matching and one dual HDF5 file.

    Callbacks (all optional):

    - ``on_noise(stream, {ch: ChannelNoiseStats})``
    - ``on_pulse(stream, channel, pulse_idx, summary, pulse_data)``
    - ``on_pair(pair_dict)`` — pair_dict as from the matcher, plus
      ``slow_tod``/``fast_tod`` cross-stream windows on one-sided pairs
      when the other ring still covers the interval
    - ``on_stats(stats_dict)`` — merged per-stream stats + pair counts
    - ``on_histograms(stream, data)``
    - ``on_error(message)``
    """

    def __init__(
        self,
        channels: List[int],
        *,
        module: int = 1,
        slow_rate: float,
        fast_rate: float = PFB_SAMPLING_FREQ,
        config: Optional[PulseCaptureConfig] = None,
        hdf5_path=None,
        df_calibrations: Optional[Dict[int, float]] = None,
        match_window_s: float = 0.05,
        match_grace_s: float = 0.25,
        on_noise: Optional[Callable] = None,
        on_pulse: Optional[Callable] = None,
        on_pair: Optional[Callable] = None,
        on_stats: Optional[Callable] = None,
        on_histograms: Optional[Callable] = None,
        on_templates: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ):
        self.channels = list(channels)
        self.module = module
        self.config = config or PulseCaptureConfig()
        # Parity with PulseCaptureSession (panel/task read this)
        self.hdf5_path = Path(hdf5_path) if hdf5_path is not None else None
        self.on_noise = on_noise
        self.on_pulse = on_pulse
        self.on_pair = on_pair
        self.on_stats = on_stats
        self.on_histograms = on_histograms
        self.on_templates = on_templates
        self.on_error = on_error

        self.writer = None
        if hdf5_path is not None:
            try:
                self.writer = DualPulseHDF5Writer(
                    hdf5_path, self.channels,
                    capture_params={
                        "streamer_mode": "both",
                        "threshold_sigma": self.config.threshold_sigma,
                        "end_sigma": self.config.end_sigma,
                        "margin_fraction": self.config.margin_fraction,
                        "enable_pileup": self.config.enable_pileup,
                        "save_to_end_confirmed":
                            self.config.save_to_end_confirmed,
                        "module": module,
                        "sample_rate_slow": slow_rate,
                        "sample_rate_fast": fast_rate,
                    },
                    df_calibrations=df_calibrations)
            except Exception as e:
                self._error(f"Could not open HDF5 file "
                            f"{hdf5_path}: {e}")

        self.matcher = IncrementalPulseMatcher(
            window_s=match_window_s, grace_s=match_grace_s,
            on_pair=self._on_matcher_pair)
        self._last_advance: Dict[str, float] = {}

        self.slow = self._make_stream("slow", slow_rate)
        self.fast = self._make_stream("fast", fast_rate)
        #: Source-compatible facades: run_slow_source/run_pfb_source
        #: read ``channels`` and call ``feed_block``, so routing those
        #: names through the per-stream feeds is all it takes for
        #: stream time to drive matcher expiry.  ``feed_sample`` stays
        #: for callers that genuinely have one sample at a time.
        self.slow_feed = SimpleNamespace(channels=self.channels,
                                         feed_sample=self.feed_slow,
                                         feed_block=self.feed_slow_block)
        self.fast_feed = SimpleNamespace(channels=self.channels,
                                         feed_sample=self.feed_fast,
                                         feed_block=self.feed_fast_block)

    def _make_stream(self, stream: str,
                     sample_rate: float) -> PulseCaptureSession:
        kwargs = self.config.session_kwargs(sample_rate)
        # Union-window extraction happens up to grace_s after a pulse
        # (single-trigger expiry): the ring must cover the full window
        # PLUS the grace, or the extraction races the ring.
        grace = self.matcher.grace_s
        min_buf = int((self.config.max_pulse_ms / 1e3 * 1.5
                       + grace + 0.1) * sample_rate)
        kwargs["buf_size"] = max(kwargs["buf_size"], min_buf)
        return PulseCaptureSession(
            channels=self.channels,
            module=self.module,
            streamer_mode=stream,
            sample_rate=sample_rate,
            hdf5_path=None,  # the dual writer owns the file
            on_noise=lambda ns, s=stream: self._on_stream_noise(s, ns),
            on_pulse=lambda ch, idx, summary, data, s=stream:
                self._on_stream_pulse(s, ch, idx, summary, data),
            on_stats=lambda _s, s=stream: self._emit_stats(),
            on_histograms=lambda data, s=stream:
                self._on_stream_histograms(s, data),
            on_templates=lambda data, s=stream:
                self._on_stream_templates(s, data),
            on_error=self._error,
            **kwargs,
        )

    # ── Lifecycle / feeding ───────────────────────────────────────

    def start(self) -> None:
        self.slow.start()
        self.fast.start()

    def feed_slow(self, ch: int, i: float, q: float, t) -> None:
        self.slow.feed_sample(ch, i, q, t)
        self._advance_matcher("slow", t)

    def feed_fast(self, ch: int, i: float, q: float, t) -> None:
        self.fast.feed_sample(ch, i, q, t)
        self._advance_matcher("fast", t)

    def feed_slow_block(self, ch: int, i_vals, q_vals, timestamps) -> None:
        """Block form of :meth:`feed_slow`."""
        self._feed_block("slow", self.slow, ch, i_vals, q_vals, timestamps)

    def feed_fast_block(self, ch: int, i_vals, q_vals, timestamps) -> None:
        """Block form of :meth:`feed_fast`."""
        self._feed_block("fast", self.fast, ch, i_vals, q_vals, timestamps)

    def _feed_block(self, stream: str, session, ch: int, i_vals, q_vals,
                    timestamps) -> None:
        """Feed one stream a block and advance its clock once.

        Matcher time advances off the last usable timestamp in the
        block rather than per sample -- _advance_matcher is throttled to
        20 ms of stream time anyway, so the per-sample calls bought
        nothing.
        """
        session.feed_block(ch, i_vals, q_vals, timestamps)
        stamps = np.asarray(timestamps, dtype=np.float64)
        usable = stamps[np.isfinite(stamps)]
        if usable.size:
            self._advance_matcher(stream, float(usable[-1]))

    def _advance_matcher(self, stream: str, t) -> None:
        # Throttled: expiry sweep at most every 20 ms of stream time
        if t is not None and t - self._last_advance.get(stream,
                                                        float("-inf")) > 0.02:
            self._last_advance[stream] = t
            self.matcher.advance_time(stream, t)

    def flush_progress(self) -> None:
        """Release held training progress on both streams."""
        for session in (self.slow, self.fast):
            session.flush_progress()

    def re_estimate_noise(self) -> None:
        """Freeze both streams and retrain their noise statistics."""
        for session in (self.slow, self.fast):
            if session.state is CaptureState.CAPTURING:
                session.re_estimate_noise()

    def stop(self) -> None:
        self.slow.stop()
        self.fast.stop()
        self.matcher.flush()
        self._to_writer("finalize", what="finalize")

    @property
    def state(self) -> Dict[str, str]:
        return {"slow": self.slow.state.value,
                "fast": self.fast.state.value}

    def stats(self) -> Dict[str, Any]:
        return {
            "slow": self.slow.stats(),
            "fast": self.fast.stats(),
            "pairs_matched": self.matcher.matched,
            "pairs_unmatched": self.matcher.unmatched,
            "total_pulses": (self.slow.total_pulses
                             + self.fast.total_pulses),
        }

    # ── Internal wiring ───────────────────────────────────────────

    def _on_stream_noise(self, stream: str, noise_stats: dict) -> None:
        self._to_writer("set_noise_stats", stream, noise_stats,
                        what="noise write")
        self._sync_capture_start()
        self._callback(self.on_noise, stream, noise_stats)

    def _sync_capture_start(self) -> None:
        """Hold triggering until BOTH streams have finished training.

        The two streams need the same training span in sample time but
        reach it at different moments — the fast one collects its
        samples far quicker and is additionally capped, so it can
        finish seconds ahead.  Left alone it triggers into a partner
        that has no ring yet, and every pair comes out one-sided with
        "window unavailable".  Its buffers keep filling while frozen,
        so nothing is lost by waiting.
        """
        both_ready = all(s.state is CaptureState.CAPTURING
                         for s in (self.slow, self.fast))
        for s in (self.slow, self.fast):
            if s.pcap is not None:
                s.pcap.freeze_triggers = not both_ready

    def _on_stream_pulse(self, stream: str, channel: int, pulse_idx: int,
                         summary: dict, pulse_data: dict) -> None:
        self._to_writer("append_pulse", stream, channel, pulse_idx,
                        pulse_data,
                        what=f"write for {stream} ch{channel}#{pulse_idx}")
        self._callback(self.on_pulse, stream, channel, pulse_idx,
                       summary, pulse_data)
        self.matcher.add(stream, channel, pulse_idx, summary)

    def _on_stream_histograms(self, stream: str, data: dict) -> None:
        self._to_writer("update_histograms", stream, data,
                        what="histogram update")
        self._callback(self.on_histograms, stream, data)

    def _on_stream_templates(self, stream: str, data: dict) -> None:
        self._to_writer("update_templates", stream, data,
                        what="template update")
        self._callback(self.on_templates, stream, data)

    def _on_matcher_pair(self, pair: dict) -> None:
        # EVERY pair carries both streams over the UNION time window
        # (widest interval spanned by the trigger(s), plus margin),
        # extracted from the ring buffers.  The per-stream triggered
        # records stay untouched — metrics (SNR, tau, histograms) are
        # computed on the physically-triggered cores only; the union
        # windows are the pair's display/analysis data.
        try:
            union = self._union_window(pair)
            if union is not None:
                t0, t1 = union
                for stream, session in (("slow", self.slow),
                                        ("fast", self.fast)):
                    if session.pcap is not None:
                        pair[f"{stream}_tod"] = \
                            session.pcap.get_window_by_time(
                                pair["channel"], t0, t1)
        except Exception as e:
            self._error(f"Union-window extraction failed: {e}")

        self._to_writer("append_match", pair["channel"], pair,
                        what="match write")
        self._callback(self.on_pair, pair)
        self._emit_stats()


    @staticmethod
    def _union_window(pair: dict) -> Optional[tuple]:
        """[t0, t1] spanning every available trigger window + 10% margin."""
        t0 = t1 = None
        for key in ("slow_summary", "fast_summary"):
            summ = pair.get(key)
            if not summ:
                continue
            s0 = summ.get("timestamp", 0.0)
            s1 = s0 + summ.get("duration_s", 0.0)
            if not (math.isfinite(s0) and math.isfinite(s1)):
                continue
            t0 = s0 if t0 is None else min(t0, s0)
            t1 = s1 if t1 is None else max(t1, s1)
        if t0 is None:
            return None
        margin = max((t1 - t0) * 0.1, 1e-4)
        return (t0 - margin, t1 + margin)

    def _emit_stats(self) -> None:
        self._callback(self.on_stats, self.stats())
