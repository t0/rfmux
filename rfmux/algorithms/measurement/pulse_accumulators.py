"""
Running accumulations over detected pulses: histograms and templates.

Both structures here follow the same pattern -- a per-channel
``*Accumulator`` holding running sums, and a ``*Set`` that owns one per
channel -- and both are driven the same way: a
:class:`~.pulse_capture_session.PulseCaptureSession` calls ``add_pulse``
for every pulse it detects and flushes them together to HDF5 and to the
live view.  They are two products of one stream, so they live in one
module.

Cost is O(1) per pulse and memory is independent of pulse count, which
is what makes them usable on a live capture that runs for hours.

**Histograms** bin the scalar summary of each pulse (amplitude, SNR,
duration, tau) into fixed bins::

    histograms = PulseHistogramSet()
    for channel, pulse_data in pulse_stream:
        histograms.add_pulse(channel, pulse_data, noise_stats[channel])
    data = histograms.get_histogram_data()

**Templates** stack the pulse waveforms themselves on a common time
origin, beating the noise down as 1/sqrt(N) to expose the underlying
shape -- the rise, the decay, and any structure buried under
single-pulse noise.  The residual RMS around the template separates
genuine pulse-to-pulse variation from measurement noise::

    templates = PulseTemplateSet(pre_samples=20, post_samples=200,
                                 threshold_sigma=5.0)
    for channel, pulse_data in stream:
        templates.add_pulse(channel, pulse_data, noise_stats[channel])
    t = templates.get(channel)
    t.mean("I"), t.residual_rms("I"), t.counts, t.time_axis(sample_rate)

Template alignment is on the **trigger crossing** -- the first sample
whose deviation exceeds ``threshold_sigma`` in either quadrature, which
is exactly the condition :class:`~.pulse_detection.PulseCapture`
triggers on.  Aligning on the window start instead would smear the
stack, because the pre-trigger margin is a fraction of each pulse's own
length; aligning on the peak would bias the rise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .pulse_detection import ChannelNoiseStats
from .pulse_analysis import pulse_summary


# ═══════════════════════════ Histograms ═════════════════════════

# ───────────────────────── Single Histogram ─────────────────────────

@dataclass
class HistogramAccumulator:
    """Fixed-bin running histogram that updates incrementally.

    Bins are defined at construction and never change.  Each call to
    :meth:`add` increments the appropriate bin in O(1).  Memory usage
    is O(n_bins), independent of the number of values added.

    Parameters
    ----------
    bin_edges : ndarray
        Monotonically increasing array of N+1 bin edges defining N bins.
    """

    bin_edges: np.ndarray
    counts: np.ndarray = field(init=False)

    def __post_init__(self):
        self.counts = np.zeros(len(self.bin_edges) - 1, dtype=np.int64)

    def add(self, value: float) -> None:
        """Increment the bin containing *value*."""
        idx = np.searchsorted(self.bin_edges, value, side="right") - 1
        if 0 <= idx < len(self.counts):
            self.counts[idx] += 1

    def add_many(self, values: np.ndarray) -> None:
        """Increment bins for an array of values (vectorized)."""
        indices = np.searchsorted(self.bin_edges, values, side="right") - 1
        valid = (indices >= 0) & (indices < len(self.counts))
        np.add.at(self.counts, indices[valid], 1)

    @property
    def bin_centers(self) -> np.ndarray:
        """Return the center of each bin."""
        return (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    @property
    def total(self) -> int:
        """Total number of values added across all bins."""
        return int(np.sum(self.counts))

    def expand_double(self) -> bool:
        """Double the range by pairwise-merging bins (exact, O(n_bins)).

        Requires a zero-based range and an even bin count; bin count is
        unchanged (the upper half starts empty).  Returns False when the
        accumulator is not expandable.
        """
        n = len(self.counts)
        if n < 2 or n % 2 or self.bin_edges[0] != 0:
            return False
        merged = self.counts[0::2] + self.counts[1::2]
        self.counts = np.zeros(n, dtype=np.int64)
        self.counts[: n // 2] = merged
        self.bin_edges = self.bin_edges * 2.0
        return True

    def reset(self) -> None:
        """Zero all bin counts."""
        self.counts[:] = 0


# ───────────────────────── Pulse Histogram Set ──────────────────────

class PulseHistogramSet:
    """Collection of running histograms for pulse capture statistics.

    Maintains per-channel histograms for:

    - **amplitude**: Peak excursion from baseline (max of I and Q)
    - **duration_ms**: Time above threshold (trigger → below-threshold)
      in milliseconds, not the length of the saved window
    - **snr**: Peak signal-to-noise ratio in σ units
    - **tau_ms**: Fit-free decay constant in ms (only binned when
      derivable — requires ``threshold_sigma``)

    Histograms are auto-created for each channel on the first pulse.

    Parameters
    ----------
    amp_range : tuple[float, float]
        Min/max for amplitude histogram bins (in ADC counts or Hz).
    amp_bins : int
        Number of amplitude histogram bins.
    duration_range_ms : tuple[float, float]
        Min/max for duration histogram bins (in ms).
    duration_bins : int
        Number of duration bins.
    snr_range : tuple[float, float]
        Min/max for SNR histogram bins (in σ).
    snr_bins : int
        Number of SNR bins.
    tau_range_ms : tuple[float, float]
        Min/max for derived-tau histogram bins (in ms).
    tau_bins : int
        Number of tau bins.
    threshold_sigma : float, optional
        Trigger threshold used during capture.  Required for the
        derived-tau metric; when None, tau is NaN and never binned.
    """

    def __init__(
        self,
        amp_range: Tuple[float, float] = (0, 5000),
        amp_bins: int = 100,
        duration_range_ms: Tuple[float, float] = (0, 50),
        duration_bins: int = 100,
        snr_range: Tuple[float, float] = (0, 50),
        snr_bins: int = 100,
        tau_range_ms: Tuple[float, float] = (0.0, 10.0),
        tau_bins: int = 100,
        threshold_sigma: Optional[float] = None,
    ):
        self.amp_edges = np.linspace(amp_range[0], amp_range[1], amp_bins + 1)
        self.dur_edges = np.linspace(
            duration_range_ms[0], duration_range_ms[1], duration_bins + 1)
        self.snr_edges = np.linspace(snr_range[0], snr_range[1], snr_bins + 1)
        self.tau_edges = np.linspace(
            tau_range_ms[0], tau_range_ms[1], tau_bins + 1)
        self.threshold_sigma = threshold_sigma

        # Per-channel accumulators: {channel: {metric: HistogramAccumulator}}
        self.histograms: Dict[int, Dict[str, HistogramAccumulator]] = {}

    def _ensure_channel(self, channel: int) -> None:
        """Create histogram accumulators for a channel if not yet present."""
        if channel not in self.histograms:
            self.histograms[channel] = {
                "amplitude": HistogramAccumulator(self.amp_edges.copy()),
                "duration_ms": HistogramAccumulator(self.dur_edges.copy()),
                "snr": HistogramAccumulator(self.snr_edges.copy()),
                "tau_ms": HistogramAccumulator(self.tau_edges.copy()),
            }

    def add_pulse(
        self,
        channel: int,
        pulse_data: dict,
        noise_stats: Optional[ChannelNoiseStats] = None,
    ) -> Dict[str, float]:
        """Update all histograms with a new pulse.

        Parameters
        ----------
        channel : int
            Channel number.
        pulse_data : dict
            Pulse data dict with ``Amp_I``, ``Amp_Q``, ``Time`` arrays.
        noise_stats : ChannelNoiseStats, optional
            Noise statistics for baseline subtraction and SNR.

        Returns
        -------
        dict
            The full :func:`pulse_summary` dict (``peak_amp``, ``snr``,
            ``duration_ms``, ``tau_ms``, ...).
        """
        self._ensure_channel(channel)
        h = self.histograms[channel]

        summary = pulse_summary(pulse_data, noise_stats, self.threshold_sigma)

        for metric, value in (("amplitude", summary["peak_amp"]),
                              ("snr", summary["snr"]),
                              ("duration_ms", summary["duration_ms"]),
                              ("tau_ms", summary["tau_ms"])):
            if metric == "tau_ms" and not np.isfinite(value):
                continue
            self._ensure_range(metric, value)
            h[metric].add(value)

        return summary

    # Template-edge attribute per metric (used when new channels appear)
    _EDGE_ATTRS = {
        "amplitude": "amp_edges",
        "duration_ms": "dur_edges",
        "snr": "snr_edges",
        "tau_ms": "tau_edges",
    }

    def _ensure_range(self, metric: str, value: float) -> None:
        """Auto-expand *metric*'s bins (across ALL channels, in lockstep)
        until *value* fits.  Values beyond a fixed range would otherwise
        be silently dropped — the classic 'histogram never updates'
        failure when a signal is brighter than the configured range.
        """
        if not np.isfinite(value) or value < 0:
            return
        attr = self._EDGE_ATTRS[metric]
        for _ in range(64):  # 2**64 dynamic range — effectively unbounded
            edges = getattr(self, attr)
            if value < edges[-1]:
                return
            expanded = [acc.expand_double()
                        for acc in (ch[metric]
                                    for ch in self.histograms.values())]
            if expanded and not all(expanded):
                return  # not expandable (odd bins / nonzero base)
            setattr(self, attr, edges * 2.0)

    def get_channel_histograms(
        self, channel: int,
    ) -> Optional[Dict[str, HistogramAccumulator]]:
        """Return the histogram accumulators for a channel, or None."""
        return self.histograms.get(channel)

    def get_histogram_data(self) -> Dict[str, np.ndarray]:
        """Return all histogram data as a flat dict.

        Suitable for HDF5 serialization via
        :meth:`PulseHDF5Writer.update_histograms`.

        Returns
        -------
        dict[str, ndarray]
            Keys like ``"amplitude_bins"``, ``"amplitude_counts_ch1"``,
            ``"duration_ms_edges"``, etc.
        """
        result: Dict[str, np.ndarray] = {}
        for ch, metrics in self.histograms.items():
            for name, acc in metrics.items():
                result[f"{name}_bins"] = acc.bin_centers
                result[f"{name}_edges"] = acc.bin_edges
                result[f"{name}_counts_ch{ch}"] = acc.counts.copy()
        return result

    def total_pulses(self, channel: Optional[int] = None) -> int:
        """Return total pulse count, optionally for a specific channel."""
        if channel is not None:
            h = self.histograms.get(channel)
            if h is None:
                return 0
            return h["amplitude"].total
        return sum(
            h["amplitude"].total for h in self.histograms.values())

    def reset_all(self) -> None:
        """Zero all histogram counts across all channels."""
        for ch_histograms in self.histograms.values():
            for acc in ch_histograms.values():
                acc.reset()


# ═══════════════════════════ Templates ══════════════════════════

def find_trigger_index(
    pulse_data: dict,
    noise_stats: Optional[ChannelNoiseStats],
    threshold_sigma: float,
) -> Optional[int]:
    """Index of the first sample crossing ``threshold_sigma``.

    Returns None when no sample crosses (or stats are missing) — the
    caller should skip such a pulse rather than stack it misaligned.
    """
    if noise_stats is None or threshold_sigma <= 0:
        return None
    amp_I = np.asarray(pulse_data["Amp_I"], dtype=np.float64)
    amp_Q = np.asarray(pulse_data["Amp_Q"], dtype=np.float64)
    if len(amp_I) == 0:
        return None
    dev_I = np.abs(amp_I - noise_stats.mean_I) / max(noise_stats.std_I, 1e-30)
    dev_Q = np.abs(amp_Q - noise_stats.mean_Q) / max(noise_stats.std_Q, 1e-30)
    crossed = np.nonzero((dev_I > threshold_sigma)
                         | (dev_Q > threshold_sigma))[0]
    return int(crossed[0]) if len(crossed) else None


class PulseTemplateAccumulator:
    """Running trigger-aligned stack for one channel (one stream).

    The grid spans ``pre_samples`` before the trigger to
    ``post_samples`` after it.  Pulses contribute only where they
    overlap the grid, so ``counts`` varies across the window — always
    check it before trusting the tail of a template.
    """

    def __init__(self, pre_samples: int = 20, post_samples: int = 200,
                 threshold_sigma: float = 5.0):
        self.pre_samples = int(pre_samples)
        self.post_samples = int(post_samples)
        self.threshold_sigma = float(threshold_sigma)
        n = self.pre_samples + self.post_samples
        self._sum = {"I": np.zeros(n), "Q": np.zeros(n)}
        self._sumsq = {"I": np.zeros(n), "Q": np.zeros(n)}
        self.counts = np.zeros(n, dtype=np.int64)
        self.n_pulses = 0
        self.n_skipped = 0

    # ── Accumulation ──────────────────────────────────────────────

    def add(self, pulse_data: dict,
            noise_stats: Optional[ChannelNoiseStats]) -> bool:
        """Stack one pulse.  Returns False when it can't be aligned or
        is pileup-affected.

        Pileup fragments are excluded outright: the first fragment's
        tail is cut at the split, and the successor sits on the
        previous pulse's decaying pedestal — both would bias the mean
        template and inflate the residual RMS."""
        if pulse_data.get("pileup"):
            self.n_skipped += 1
            return False
        trig = find_trigger_index(pulse_data, noise_stats,
                                  self.threshold_sigma)
        if trig is None:
            self.n_skipped += 1
            return False

        amp = {
            "I": np.asarray(pulse_data["Amp_I"], dtype=np.float64)
            - noise_stats.mean_I,
            "Q": np.asarray(pulse_data["Amp_Q"], dtype=np.float64)
            - noise_stats.mean_Q,
        }
        n_samples = len(amp["I"])

        # Overlap of [trig - pre, trig + post) with [0, n_samples)
        src_lo = max(0, trig - self.pre_samples)
        src_hi = min(n_samples, trig + self.post_samples)
        if src_hi <= src_lo:
            self.n_skipped += 1
            return False
        dst_lo = src_lo - trig + self.pre_samples
        dst_hi = dst_lo + (src_hi - src_lo)

        for quad in ("I", "Q"):
            seg = amp[quad][src_lo:src_hi]
            self._sum[quad][dst_lo:dst_hi] += seg
            self._sumsq[quad][dst_lo:dst_hi] += seg * seg
        self.counts[dst_lo:dst_hi] += 1
        self.n_pulses += 1
        return True

    # ── Results ───────────────────────────────────────────────────

    def mean(self, quad: str = "I") -> np.ndarray:
        """Baseline-subtracted mean template (NaN where no data)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(self.counts > 0,
                           self._sum[quad] / np.maximum(self.counts, 1),
                           np.nan)
        return out

    def residual_rms(self, quad: str = "I") -> np.ndarray:
        """Per-bin RMS spread about the template (NaN where n < 2)."""
        n = self.counts
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = self._sum[quad] / np.maximum(n, 1)
            var = self._sumsq[quad] / np.maximum(n, 1) - mean * mean
            out = np.where(n > 1, np.sqrt(np.maximum(var, 0.0)), np.nan)
        return out

    def time_axis(self, sample_rate: float) -> np.ndarray:
        """Seconds relative to the trigger (negative = pre-trigger)."""
        n = self.pre_samples + self.post_samples
        return (np.arange(n) - self.pre_samples) / float(sample_rate)

    def to_dict(self, sample_rate: Optional[float] = None) -> dict:
        """Flat arrays for HDF5 / transport."""
        data = {
            "template_I": self.mean("I"),
            "template_Q": self.mean("Q"),
            "residual_I": self.residual_rms("I"),
            "residual_Q": self.residual_rms("Q"),
            "counts": self.counts.copy(),
            "n_pulses": self.n_pulses,
            "pre_samples": self.pre_samples,
            "post_samples": self.post_samples,
        }
        if sample_rate:
            data["time_s"] = self.time_axis(sample_rate)
        return data

    def reset(self) -> None:
        for quad in ("I", "Q"):
            self._sum[quad][:] = 0.0
            self._sumsq[quad][:] = 0.0
        self.counts[:] = 0
        self.n_pulses = 0
        self.n_skipped = 0


class PulseTemplateSet:
    """Per-channel trigger-aligned templates."""

    def __init__(self, pre_samples: int = 20, post_samples: int = 200,
                 threshold_sigma: float = 5.0,
                 sample_rate: Optional[float] = None):
        self.pre_samples = int(pre_samples)
        self.post_samples = int(post_samples)
        self.threshold_sigma = float(threshold_sigma)
        self.sample_rate = sample_rate
        self.templates: Dict[int, PulseTemplateAccumulator] = {}

    def _ensure(self, channel: int) -> PulseTemplateAccumulator:
        if channel not in self.templates:
            self.templates[channel] = PulseTemplateAccumulator(
                self.pre_samples, self.post_samples, self.threshold_sigma)
        return self.templates[channel]

    def add_pulse(self, channel: int, pulse_data: dict,
                  noise_stats: Optional[ChannelNoiseStats]) -> bool:
        return self._ensure(channel).add(pulse_data, noise_stats)

    def get(self, channel: int) -> Optional[PulseTemplateAccumulator]:
        return self.templates.get(channel)

    def total_pulses(self, channel: Optional[int] = None) -> int:
        if channel is not None:
            acc = self.templates.get(channel)
            return acc.n_pulses if acc else 0
        return sum(a.n_pulses for a in self.templates.values())

    def get_template_data(self) -> Dict[str, np.ndarray]:
        """Flat dict keyed like the histogram data (for HDF5)."""
        result: Dict[str, np.ndarray] = {}
        for ch, acc in self.templates.items():
            for key, value in acc.to_dict(self.sample_rate).items():
                result[f"{key}_ch{ch}"] = (
                    value if isinstance(value, np.ndarray)
                    else np.asarray(value))
        return result

    def reset_all(self) -> None:
        for acc in self.templates.values():
            acc.reset()
