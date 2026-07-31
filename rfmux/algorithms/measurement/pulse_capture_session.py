"""
Live pulse-capture orchestration, independent of any GUI.

:class:`PulseCaptureSession` ties together the pieces that already exist
in this package — noise estimation, the :class:`PulseCapture` detection
engine, the streaming :class:`PulseHDF5Writer`, and the running
:class:`PulseHistogramSet` — into a single object with a start/stop
lifecycle that a GUI (or script) can drive by feeding it samples.

The session is deliberately free of Qt, sockets, and threads: samples
arrive through :meth:`feed_sample` from whatever source the caller has
(the Periscope slow-stream tap, a PFB receive loop, a synthetic stream
in tests), and results leave through plain callbacks.  The Periscope
``PulseCaptureTask`` is expected to be a thin adapter that pumps a
thread-safe queue into :meth:`feed_sample` and re-emits the callbacks
as Qt signals.

Lifecycle::

    session = PulseCaptureSession(
        channels=[1, 2],
        threshold_sigma=5.0,
        hdf5_path="capture.h5",
        on_pulse=lambda ch, idx, summary, data: ...,
    )
    session.start()                     # begins noise estimation
    for ch, i, q, t in sample_stream:
        session.feed_sample(ch, i, q, t)
    session.stop()                      # finalizes the HDF5 file

State machine::

    IDLE ──start()──▶ ESTIMATING ──(noise_samples reached)──▶ CAPTURING
                          ▲                                       │
                          └──────────re_estimate_noise()──────────┘
    any ──stop()──▶ STOPPED (terminal; HDF5 finalized)

Thread safety: none.  All methods must be called from a single thread —
in Periscope, the capture task thread (h5py writes must stay on one
thread).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .pulse_detection import (
    ChannelNoiseStats,
    PulseCapture,
    estimate_noise_stats,
    measure_pulse_scale,
    recommend_baseline_track_samples,
)
from .pulse_analysis import pulse_summary
from .pulse_histograms import PulseHistogramSet
from .pulse_templates import PulseTemplateSet

try:
    from .pulse_hdf5 import PulseHDF5Writer
except ImportError:  # pragma: no cover - h5py missing
    PulseHDF5Writer = None  # type: ignore[assignment]


class CaptureState(Enum):
    IDLE = "idle"
    ESTIMATING = "estimating"
    CAPTURING = "capturing"
    STOPPED = "stopped"


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
    end_sigma: float = 1.5
    #: Consecutive samples that must clear the threshold to trigger.
    #: 0 = choose it from the stream rate, which is the right default:
    #: the accidental rate scales with sample rate, so one sample is
    #: plenty of evidence at 596 Hz (2.5 accidentals per HOUR) and
    #: nowhere near enough at 1.22 MHz (1.4 per SECOND).  Demanding two
    #: everywhere would reject real pulses on a heavily decimated slow
    #: stream, where a fast pulse spans less than one sample.
    trigger_samples: int = 0
    #: Accidental-trigger budget used to pick trigger_samples, per
    #: channel per minute.
    max_accidental_per_min: float = 1.0
    margin_fraction: float = 0.1
    min_pulse_ms: float = 0.0      # 0 = no glitch rejection
    #: Longest pulse the ring must hold.  Used as the starting guess and
    #: as the fallback when ``max_pulse_auto`` finds nothing to measure.
    max_pulse_ms: float = 250.0
    #: Measure the pulse length from the training record rather than
    #: taking ``max_pulse_ms``.  This is the parameter users can least
    #: supply from first principles, and it sets both the ring buffer
    #: and the floor under the baseline window.
    #:
    #: Off by default because it costs a second detection pass over the
    #: training record, run synchronously when training ends — on a fast
    #: stream that stalls the receive loop long enough to drop packets.
    #: Turn it on together with a training window long enough to contain
    #: a useful number of pulses.
    max_pulse_auto: bool = False
    #: Training length, in SAMPLE time — so the wall-clock cost scales
    #: with how fast the stream actually runs.  Deliberately short by
    #: default: on the mock, and on the PFB stream generally, a long
    #: training window is expensive, and the measurements below degrade
    #: gracefully to their fallbacks rather than failing.  Raise it
    #: (seconds are reasonable on hardware) when you want the pulse
    #: scale and the 1/f knee actually measured rather than guessed.
    noise_train_ms: float = 50.0
    enable_pileup: bool = True
    #: Measure the baseline tracking window from the noise training data
    #: instead of taking ``baseline_track_ms``.  The upper end of the
    #: usable window is the 1/f knee, which is a property of the
    #: detector rather than a preference, so this is the default.
    baseline_track_auto: bool = True
    #: EMA time constant for baseline tracking when ``baseline_track_auto``
    #: is off (0 = frozen baseline).  Needed under 1/f noise, where the
    #: true baseline drifts away from the training-time mean.  Must
    #: satisfy pulse length << baseline_track_ms << drift timescale.
    baseline_track_ms: float = 0.0

    #: buffer headroom over max_pulse_ms (pre-trigger margin +
    #: end-confirmation tail both live in the same ring)
    BUFFER_SAFETY = 1.5
    _MIN_BUF = 1000
    _MIN_NOISE = 200
    #: Ceiling on the held training record, per channel.  complex128, so
    #: this is 32 MB/channel.
    _MAX_NOISE = 2_000_000
    #: How many max-length pulses the tracker must span.  A clamped EMA
    #: moves by at most (pulse / tau) * end_sigma * sigma during a
    #: pulse, so 20x bounds that bite at a few hundredths of a sigma.
    BASELINE_PULSE_FACTOR = 20
    #: Independent block pairs wanted before the knee is measurable —
    #: mirrors _MIN_ALLAN_PAIRS in the estimator.
    _KNEE_PAIRS = 9

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
        PFB one: 30 s at 1.22 MHz is 36.6M samples per channel.  The cap
        binds only at fast rates, where a long span is neither needed
        nor achievable — describe() reports the duration actually used.
        """
        want = int(round(self.noise_train_ms * 1e-3 * sample_rate))
        return max(self._MIN_NOISE, min(want, self._MAX_NOISE))

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

    def baseline_track_samples(self, sample_rate: float) -> int:
        if self.baseline_track_ms <= 0:
            return 0
        return max(1, int(round(
            self.baseline_track_ms * 1e-3 * sample_rate)))

    def baseline_track_min_samples(self, sample_rate: float) -> int:
        """Floor for the auto-measured window: the tracker must not be
        fast enough to absorb the tail of the longest expected pulse."""
        return self.BASELINE_PULSE_FACTOR * self.max_pulse_samples(sample_rate)

    def knee_measurable_ms(self, sample_rate: float) -> float:
        """Longest drift timescale the training window can resolve.

        The Allan fit needs several independent block pairs at its
        longest lag, so a record of N samples says nothing about drift
        slower than about N / 9.
        """
        return (self.noise_samples(sample_rate) / self._KNEE_PAIRS
                / sample_rate * 1e3)

    def session_kwargs(self, sample_rate: float) -> Dict[str, Any]:
        """Keyword arguments for :class:`PulseCaptureSession`."""
        return {
            "threshold_sigma": self.threshold_sigma,
            "end_sigma": self.end_sigma,
            "margin_fraction": self.margin_fraction,
            "min_pulse_samples": self.min_pulse_samples(sample_rate),
            "trigger_samples": self.trigger_samples_for(sample_rate),
            "enable_pileup": self.enable_pileup,
            "buf_size": self.buf_size(sample_rate),
            "noise_samples": self.noise_samples(sample_rate),
            "max_pulse_auto": self.max_pulse_auto,
            "buffer_safety": self.BUFFER_SAFETY,
            "baseline_pulse_factor": self.BASELINE_PULSE_FACTOR,
            "baseline_track_auto": self.baseline_track_auto,
            "baseline_track_samples":
                self.baseline_track_samples(sample_rate),
            "baseline_track_min_samples":
                self.baseline_track_min_samples(sample_rate),
        }

    def describe(self, sample_rate: float,
                 n_channels: int = 1) -> Dict[str, Any]:
        """Derived quantities at a given stream rate (for display)."""
        buf = self.buf_size(sample_rate)
        return {
            "sample_rate_hz": sample_rate,
            "min_pulse_samples": self.min_pulse_samples(sample_rate),
            "noise_samples": self.noise_samples(sample_rate),
            "noise_train_actual_ms":
                self.noise_samples(sample_rate) / sample_rate * 1e3,
            "buf_samples": buf,
            "buf_mb_per_channel": buf * 3 * 8 / 1e6,
            "buf_mb_total": buf * 3 * 8 * n_channels / 1e6,
            "max_recordable_ms": buf / sample_rate * 1e3,
            "baseline_track_samples":
                self.baseline_track_samples(sample_rate),
            "baseline_track_auto": self.baseline_track_auto,
            "baseline_track_min_samples":
                self.baseline_track_min_samples(sample_rate),
            "baseline_track_min_ms":
                self.baseline_track_min_samples(sample_rate)
                / sample_rate * 1e3,
            "knee_measurable_ms": self.knee_measurable_ms(sample_rate),
            "max_pulse_auto": self.max_pulse_auto,
            "trigger_samples": self.trigger_samples_for(sample_rate),
            "accidental_per_min":
                60.0 * self.accidental_rate_hz(sample_rate),
            "accidental_per_min_unconfirmed": 60.0 * sample_rate * (
                1.0 - (1.0 - math.erfc(
                    self.threshold_sigma / math.sqrt(2.0))) ** 2),
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
        if self.noise_train_ms <= 0:
            issues.append(("error",
                           "Noise training length must be positive."))
        if self.baseline_track_auto:
            # The knee is only visible if the training record reaches it;
            # otherwise the fit honestly reports "no drift measured" and
            # the fallback is used.  Say what it would take to do better.
            floor_ms = self.BASELINE_PULSE_FACTOR * self.max_pulse_ms
            issues.append((
                "info",
                "Baseline window measured from the noise training data "
                f"(the 1/f knee), floored at {floor_ms:g} ms = "
                f"{self.BASELINE_PULSE_FACTOR}x the max pulse length."))
            if sample_rate:
                meas = self.knee_measurable_ms(sample_rate)
                if meas < floor_ms:
                    want = self._KNEE_PAIRS * floor_ms
                    want_s = (f"{want / 1000:,.1f} s" if want >= 1000
                              else f"{want:,.0f} ms")
                    issues.append((
                        "warning",
                        f"Noise training ({self.noise_train_ms:g} ms) can "
                        f"only resolve drift faster than {meas:.3g} ms, "
                        f"below the {floor_ms:g} ms floor — the knee "
                        "cannot be measured and the floor will be used. "
                        f"Train for ≥ {want_s} to measure it."))
        elif self.baseline_track_ms < 0:
            issues.append(("error",
                           "Baseline tracking time cannot be negative."))
        elif self.baseline_track_ms > 0:
            # The tracker must be far slower than a pulse or it eats the
            # signal, and far faster than the drift or it never catches
            # it: pulse << tau_track << drift timescale.
            if self.baseline_track_ms < 10 * self.max_pulse_ms:
                issues.append((
                    "warning",
                    f"Baseline tracking ({self.baseline_track_ms:g} ms) "
                    f"is under 10x the max pulse length "
                    f"({self.max_pulse_ms:g} ms) — the tracker will "
                    "absorb pulse tails and suppress triggers. Use "
                    f"≥ {100 * self.max_pulse_ms:g} ms where possible."))
            else:
                issues.append((
                    "info",
                    f"Baseline tracked with a {self.baseline_track_ms:g} "
                    "ms time constant — drift slower than that is "
                    "followed; faster drift still reaches the trigger."))

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


class PulseCaptureSession:
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
    histogram_config : dict, optional
        Extra kwargs for :class:`PulseHistogramSet` (ranges and bin
        counts).  ``threshold_sigma`` is injected automatically.
    histogram_flush_every : int
        Flush histograms to HDF5 and fire ``on_histograms`` every N
        pulses (and once at stop).  Default 50.

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
        end_sigma: float = 1.0,
        margin_fraction: float = 0.1,
        min_pulse_samples: int = 0,
        trigger_samples: int = 2,
        enable_pileup: bool = True,
        buf_size: int = 5000,
        sample_rate: Optional[float] = None,
        noise_samples: int = 1000,
        max_pulse_auto: bool = False,
        buffer_safety: float = 1.5,
        baseline_pulse_factor: int = 20,
        baseline_track_auto: bool = False,
        baseline_track_samples: int = 0,
        baseline_track_min_samples: int = 0,
        hdf5_path: Optional[str | Path] = None,
        df_calibrations: Optional[Dict[int, float]] = None,
        histogram_config: Optional[Dict[str, Any]] = None,
        histogram_flush_every: int = 50,
        progress_every: int = 100,
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
        self.buf_size = buf_size
        self.sample_rate = sample_rate
        self.noise_samples = int(noise_samples)
        self.max_pulse_auto = bool(max_pulse_auto)
        self.buffer_safety = float(buffer_safety)
        self.baseline_pulse_factor = int(baseline_pulse_factor)
        #: Diagnostics from the last pulse-scale measurement.
        self.pulse_scale_info: Optional[Dict[str, Any]] = None
        self.baseline_track_auto = bool(baseline_track_auto)
        self.baseline_track_samples = int(baseline_track_samples)
        self.baseline_track_min_samples = int(baseline_track_min_samples)
        #: Diagnostics from the last auto-measurement (None when off).
        self.baseline_track_info: Optional[Dict[str, Any]] = None
        self.hdf5_path = Path(hdf5_path) if hdf5_path is not None else None
        self.df_calibrations = df_calibrations
        self.histogram_flush_every = int(histogram_flush_every)

        self.on_noise = on_noise
        self.on_pulse = on_pulse
        self.on_stats = on_stats
        self.on_histograms = on_histograms
        self.on_error = on_error
        self.on_templates = on_templates
        self.on_progress = on_progress
        self.progress_every = max(1, int(progress_every))

        hist_kwargs = dict(histogram_config or {})
        hist_kwargs["threshold_sigma"] = threshold_sigma
        self.histograms = PulseHistogramSet(**hist_kwargs)

        self.templates: PulseTemplateSet
        self._resize_templates()

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
        if self.writer is not None:
            try:
                self.writer.finalize()
            except Exception as e:  # pragma: no cover - defensive
                self._error(f"HDF5 finalize failed: {e}")
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
            if n % self.progress_every == 0 or n == self.noise_samples:
                self._callback(self.on_progress, {
                    "state": self.state.value,
                    "collected": dict(self._noise_n),
                    "target": self.noise_samples,
                })
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
            "baseline_track_samples": self.baseline_track_samples,
            "baseline_track_ms": (
                self.baseline_track_samples / self.sample_rate * 1e3
                if self.sample_rate else None),
            "baseline_track_summary": (
                self.baseline_track_info.get("summary")
                if self.baseline_track_info else None),
        }

    # ── Internals ─────────────────────────────────────────────────

    def _resize_templates(self) -> None:
        """(Re)build the stacking window from the current ring size.

        Called again if the measured pulse scale enlarges the ring —
        only ever before the first pulse arrives, so nothing is lost.
        """
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

    def _maybe_finish_estimation(self) -> None:
        if any(self._noise_n.get(c, 0) < self.noise_samples
               for c in self.channels):
            return

        samples = {c: self._noise_buf[c][:self._noise_n[c]]
                   for c in self.channels}
        self.noise_stats, self.noise_data = estimate_noise_stats(
            samples, self.channels)

        # The training record carries more than the trigger threshold:
        # the pulse timescale and the drift timescale are both in it.
        # Measure the pulse scale FIRST — it sets the ring buffer and
        # the floor under the tracking window.
        if self.max_pulse_auto and self.pcap is None:
            rec, self.pulse_scale_info = measure_pulse_scale(
                samples, self.channels, self.noise_stats,
                threshold_sigma=self.threshold_sigma,
                end_sigma=self.end_sigma,
                trigger_samples=self.trigger_samples,
                margin_fraction=self.margin_fraction,
                enable_pileup=self.enable_pileup,
                safety=self.buffer_safety)
            if rec:
                self.buf_size = max(self.buf_size, rec)
                self.baseline_track_min_samples = int(
                    self.baseline_pulse_factor
                    * self.pulse_scale_info["p99_samples"])
                self._resize_templates()

        # The same record that gives the trigger threshold also carries
        # the drift timescale, so measure the tracking window here
        # rather than making the user guess it.
        if self.baseline_track_auto:
            self.baseline_track_samples, self.baseline_track_info = \
                recommend_baseline_track_samples(
                    samples, self.channels,
                    min_samples=self.baseline_track_min_samples)

        if self.pcap is None:
            self._build_engine_and_writer()
        else:
            # Re-estimation: swap stats in place and resume triggering.
            self.pcap.noise_stats = self.noise_stats
            self.pcap.set_baseline_track_samples(self.baseline_track_samples)
            if self.writer is not None:
                self.writer.update_noise_stats(self.noise_stats)
            self.pcap.freeze_triggers = False

        self.state = CaptureState.CAPTURING
        self._callback(self.on_noise, self.noise_stats)

    def _build_engine_and_writer(self) -> None:
        self.pcap = PulseCapture(
            buf_size=self.buf_size,
            channels=self.channels,
            noise_stats=self.noise_stats,
            threshold_sigma=self.threshold_sigma,
            end_sigma=self.end_sigma,
            sample_rate=self.sample_rate or 0.0,
            margin_fraction=self.margin_fraction,
            min_pulse_samples=self.min_pulse_samples,
            trigger_samples=self.trigger_samples,
            enable_pileup=self.enable_pileup,
            baseline_track_samples=self.baseline_track_samples,
            on_pulse=self._on_engine_pulse,
            accumulate=False,
        )

        if self.hdf5_path is not None:
            if PulseHDF5Writer is None:
                self._error("h5py not available — capturing without HDF5")
                return
            capture_params = {
                "streamer_mode": self.streamer_mode,
                "threshold_sigma": self.threshold_sigma,
                "end_sigma": self.end_sigma,
                "margin_fraction": self.margin_fraction,
                "min_pulse_samples": self.min_pulse_samples,
                "trigger_samples": self.trigger_samples,
                "enable_pileup": self.enable_pileup,
                "module": self.module,
                "baseline_track_samples": self.baseline_track_samples,
                "baseline_track_auto": self.baseline_track_auto,
            }
            if self.baseline_track_info:
                capture_params["baseline_track_basis"] = str(
                    self.baseline_track_info.get("summary", ""))
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

        if self.writer is not None:
            try:
                self.writer.append_pulse(channel, pulse_idx, pulse_data)
            except Exception as e:
                self._error(f"HDF5 write failed for pulse "
                            f"ch{channel}#{pulse_idx}: {e}")

        self.histograms.add_pulse(channel, pulse_data, ns)
        self.templates.add_pulse(channel, pulse_data, ns)

        self._callback(self.on_pulse, channel, pulse_idx, summary, pulse_data)
        self._callback(self.on_stats, self.stats())

        if self._pulses_since_flush >= self.histogram_flush_every:
            self._flush_histograms()

    def _flush_histograms(self) -> None:
        self._pulses_since_flush = 0
        if self.histograms.total_pulses() == 0:
            return
        data = self.histograms.get_histogram_data()
        if self.writer is not None:
            try:
                self.writer.update_histograms(data)
            except Exception as e:
                self._error(f"HDF5 histogram update failed: {e}")
        self._callback(self.on_histograms, data)

        tmpl = self.templates.get_template_data()
        if tmpl:
            if self.writer is not None:
                try:
                    self.writer.update_templates(tmpl)
                except Exception as e:
                    self._error(f"HDF5 template update failed: {e}")
            self._callback(self.on_templates, tmpl)

    def _callback(self, cb: Optional[Callable], *args) -> None:
        if cb is None:
            return
        try:
            cb(*args)
        except Exception as e:
            self._error(f"Callback {getattr(cb, '__name__', cb)!r} "
                        f"raised: {e}")

    def _error(self, message: str) -> None:
        if self.on_error is not None:
            try:
                self.on_error(message)
            except Exception:
                pass
