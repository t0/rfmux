"""
Running histogram accumulators for pulse capture statistics.

These accumulators maintain fixed-bin histograms that update
incrementally with O(1) per pulse.  Memory usage is O(n_bins)
regardless of the number of pulses processed, making them suitable
for long-running live captures.

Usage::

    histograms = PulseHistogramSet()
    for channel, pulse_data in pulse_stream:
        histograms.add_pulse(channel, pulse_data, noise_stats[channel])

    # Get data for plotting or HDF5 serialization
    data = histograms.get_histogram_data()
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .pulse_detection import ChannelNoiseStats
from .pulse_analysis import pulse_summary


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
