"""
Trigger-aligned pulse stacking (template averaging).

Averaging many pulses on a common time origin beats down the noise as
1/sqrt(N) and exposes the underlying pulse shape — the rise, the decay,
and any structure buried under single-pulse noise.  The residual RMS
around the template is equally informative: it separates genuine
pulse-to-pulse variation from measurement noise.

Alignment is on the **trigger crossing** — the first sample whose
deviation exceeds ``threshold_sigma`` in either quadrature, which is
exactly the condition :class:`~.pulse_detection.PulseCapture` triggers
on.  Aligning on the window start instead would smear the stack,
because the pre-trigger margin is a fraction of each pulse's own
length; aligning on the peak would bias the rise.

Accumulation is running (sum, sum of squares, per-bin count), so cost
is O(1) per pulse and memory is O(window), independent of pulse count.

Usage::

    templates = PulseTemplateSet(pre_samples=20, post_samples=200,
                                 threshold_sigma=5.0)
    for channel, pulse_data in stream:
        templates.add_pulse(channel, pulse_data, noise_stats[channel])

    t = templates.get(channel)
    t.mean("I"), t.residual_rms("I"), t.counts, t.time_axis(sample_rate)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .pulse_detection import ChannelNoiseStats


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
