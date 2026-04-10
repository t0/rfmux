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

    def reset(self) -> None:
        """Zero all bin counts."""
        self.counts[:] = 0


# ───────────────────────── Pulse Histogram Set ──────────────────────

class PulseHistogramSet:
    """Collection of running histograms for pulse capture statistics.

    Maintains per-channel histograms for:

    - **amplitude**: Peak excursion from baseline (max of I and Q)
    - **duration_ms**: Pulse window duration in milliseconds
    - **snr**: Peak signal-to-noise ratio in σ units

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
    """

    def __init__(
        self,
        amp_range: Tuple[float, float] = (0, 5000),
        amp_bins: int = 100,
        duration_range_ms: Tuple[float, float] = (0, 50),
        duration_bins: int = 100,
        snr_range: Tuple[float, float] = (0, 50),
        snr_bins: int = 100,
    ):
        self.amp_edges = np.linspace(amp_range[0], amp_range[1], amp_bins + 1)
        self.dur_edges = np.linspace(
            duration_range_ms[0], duration_range_ms[1], duration_bins + 1)
        self.snr_edges = np.linspace(snr_range[0], snr_range[1], snr_bins + 1)

        # Per-channel accumulators: {channel: {metric: HistogramAccumulator}}
        self.histograms: Dict[int, Dict[str, HistogramAccumulator]] = {}

    def _ensure_channel(self, channel: int) -> None:
        """Create histogram accumulators for a channel if not yet present."""
        if channel not in self.histograms:
            self.histograms[channel] = {
                "amplitude": HistogramAccumulator(self.amp_edges.copy()),
                "duration_ms": HistogramAccumulator(self.dur_edges.copy()),
                "snr": HistogramAccumulator(self.snr_edges.copy()),
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
            Computed metrics: ``peak_amp``, ``snr``, ``duration_ms``.
        """
        self._ensure_channel(channel)
        h = self.histograms[channel]

        amp_I = np.asarray(pulse_data["Amp_I"])
        amp_Q = np.asarray(pulse_data["Amp_Q"])
        time_arr = np.asarray(pulse_data["Time"], dtype=np.float64)

        # Peak amplitude (max of I and Q excursion from baseline)
        if noise_stats is not None:
            peak_I = float(np.max(np.abs(amp_I - noise_stats.mean_I)))
            peak_Q = float(np.max(np.abs(amp_Q - noise_stats.mean_Q)))
            peak_amp = max(peak_I, peak_Q)
            max_std = max(noise_stats.std_I, noise_stats.std_Q, 1e-30)
            snr = peak_amp / max_std
        else:
            peak_amp = float(max(np.max(np.abs(amp_I)), np.max(np.abs(amp_Q))))
            snr = 0.0

        h["amplitude"].add(peak_amp)
        h["snr"].add(snr)

        # Duration from timestamps
        valid_mask = np.isfinite(time_arr)
        valid_times = time_arr[valid_mask]
        if len(valid_times) > 1:
            duration_ms = float(
                np.max(valid_times) - np.min(valid_times)) * 1e3
        else:
            duration_ms = 0.0
        h["duration_ms"].add(duration_ms)

        return {"peak_amp": peak_amp, "snr": snr, "duration_ms": duration_ms}

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
