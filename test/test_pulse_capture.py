"""
Unit tests for Phase 1 pulse capture infrastructure:
- Callback-driven PulseCapture
- PulseHDF5Writer / PulseHDF5Reader round-trip
- PulseHistogramSet incremental updates
- Integration: PulseCapture → on_pulse → HDF5Writer → Reader
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from rfmux.algorithms.measurement.pulse_detection import (
    PulseCapture,
    ChannelNoiseStats,
    Circular,
    estimate_noise_stats,
)
from rfmux.algorithms.measurement.pulse_histograms import (
    HistogramAccumulator,
    PulseHistogramSet,
)

# HDF5 imports are deferred — tests that need them use the
# requires_h5py marker or skip individually.
try:
    import h5py
    from rfmux.algorithms.measurement.pulse_hdf5 import (
        PulseHDF5Writer,
        PulseHDF5Reader,
    )
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

requires_h5py = pytest.mark.skipif(
    not HAS_H5PY, reason="h5py not installed")


# ───────────────────────── Helpers ──────────────────────────────────

def _make_noise_stats(mean_I=0.0, std_I=10.0, mean_Q=0.0, std_Q=10.0):
    return ChannelNoiseStats(
        mean_I=mean_I, std_I=std_I, mean_Q=mean_Q, std_Q=std_Q)


def _make_pulse_data(n_samples=50, baseline_I=0.0, baseline_Q=0.0,
                     peak_I=100.0, peak_Q=20.0, t_start=1.0,
                     sample_rate=38147.0, pileup=False):
    """Synthesize a pulse: baseline → sharp rise → exponential decay."""
    t = np.arange(n_samples) / sample_rate + t_start
    # Simple exponential pulse shape
    rise = 5  # samples to rise
    tau = n_samples / 4  # decay constant in samples
    pulse_shape = np.zeros(n_samples)
    for i in range(n_samples):
        if i < rise:
            pulse_shape[i] = (i / rise)
        else:
            pulse_shape[i] = np.exp(-(i - rise) / tau)

    amp_I = baseline_I + peak_I * pulse_shape
    amp_Q = baseline_Q + peak_Q * pulse_shape
    return {
        "Amp_I": amp_I,
        "Amp_Q": amp_Q,
        "Time": t,
        "pileup": pileup,
    }


def _generate_synthetic_stream(
    channels, noise_stats, n_baseline=200, n_pulse=80,
    n_tail=100, peak_amplitude=200.0, sample_rate=38147.0,
):
    """Generate a synthetic I/Q sample stream with embedded pulses.

    Yields (channel, i_val, q_val, timestamp) tuples.
    The stream has: baseline → pulse → tail for each channel.
    """
    rng = np.random.default_rng(42)
    t = 0.0
    dt = 1.0 / sample_rate

    # Baseline
    for _ in range(n_baseline):
        for ch in channels:
            ns = noise_stats[ch]
            i_val = ns.mean_I + rng.normal(0, ns.std_I)
            q_val = ns.mean_Q + rng.normal(0, ns.std_Q)
            yield ch, i_val, q_val, t
        t += dt

    # Pulse (exponential rise + decay on I, small perturbation on Q)
    for j in range(n_pulse):
        frac = j / max(1, n_pulse - 1)
        # Sharp rise then exponential decay
        if j < 5:
            env = j / 5.0
        else:
            env = np.exp(-(j - 5) / 20.0)
        for ch in channels:
            ns = noise_stats[ch]
            i_val = ns.mean_I + peak_amplitude * env + rng.normal(0, ns.std_I)
            q_val = ns.mean_Q + peak_amplitude * 0.1 * env + rng.normal(0, ns.std_Q)
            yield ch, i_val, q_val, t
        t += dt

    # Tail (back to baseline)
    for _ in range(n_tail):
        for ch in channels:
            ns = noise_stats[ch]
            i_val = ns.mean_I + rng.normal(0, ns.std_I)
            q_val = ns.mean_Q + rng.normal(0, ns.std_Q)
            yield ch, i_val, q_val, t
        t += dt


# ═══════════════════════════════════════════════════════════════════
#  Circular Buffer Tests
# ═══════════════════════════════════════════════════════════════════

class TestCircular:
    def test_basic_add_and_data(self):
        c = Circular(5)
        for v in [1, 2, 3]:
            c.add(v)
        assert list(c.data()) == [1, 2, 3]
        assert c.count == 3

    def test_wraparound(self):
        c = Circular(3)
        for v in [1, 2, 3, 4, 5]:
            c.add(v)
        assert list(c.data()) == [3, 4, 5]
        assert c.count == 3

    def test_empty(self):
        c = Circular(10)
        assert len(c.data()) == 0
        assert c.count == 0


# ═══════════════════════════════════════════════════════════════════
#  PulseCapture Callback Tests
# ═══════════════════════════════════════════════════════════════════

class TestPulseCaptureCallback:
    """Test the on_pulse callback and accumulate flag."""

    def _run_detection(self, on_pulse=None, accumulate=True):
        channels = [1]
        noise_stats = {1: _make_noise_stats(std_I=5.0, std_Q=5.0)}

        pcap = PulseCapture(
            buf_size=5000,
            channels=channels,
            noise_stats=noise_stats,
            threshold_sigma=5.0,
            end_sigma=1.5,
            sample_rate=38147.0,
            on_pulse=on_pulse,
            accumulate=accumulate,
        )
        pcap.start_time = 0.0

        for ch, i_val, q_val, ts in _generate_synthetic_stream(
            channels, noise_stats, peak_amplitude=100.0,
        ):
            pcap.process_sample(ch, i_val, q_val, ts)

        return pcap

    def test_callback_is_called(self):
        """on_pulse callback should be called when a pulse is detected."""
        captured = []

        def on_pulse(channel, pulse_idx, pulse_data):
            captured.append((channel, pulse_idx, pulse_data.copy()))

        pcap = self._run_detection(on_pulse=on_pulse)

        # Should have detected at least one pulse
        assert len(captured) > 0
        ch, idx, data = captured[0]
        assert ch == 1
        assert idx == 1
        assert "Amp_I" in data
        assert "Amp_Q" in data
        assert "Time" in data
        assert len(data["Amp_I"]) > 0

    def test_callback_with_accumulate_true(self):
        """With accumulate=True (default), pulses are also in self.pulses."""
        captured = []

        def on_pulse(channel, pulse_idx, pulse_data):
            captured.append(pulse_idx)

        pcap = self._run_detection(on_pulse=on_pulse, accumulate=True)

        assert len(captured) > 0
        # Same pulses should be in self.pulses
        ch_key = "Channel 1"
        assert len(pcap.pulses[ch_key]) == len(captured)
        for idx in captured:
            assert idx in pcap.pulses[ch_key]

    def test_callback_with_accumulate_false(self):
        """With accumulate=False, self.pulses stays empty."""
        captured = []

        def on_pulse(channel, pulse_idx, pulse_data):
            captured.append(pulse_idx)

        pcap = self._run_detection(on_pulse=on_pulse, accumulate=False)

        assert len(captured) > 0
        # self.pulses should be empty
        assert len(pcap.pulses["Channel 1"]) == 0
        # But pulse_count should still track
        assert pcap.pulse_count["Channel 1"] == len(captured)

    def test_backward_compat_no_callback(self):
        """Without on_pulse, behavior is identical to original code."""
        pcap = self._run_detection(on_pulse=None, accumulate=True)

        total = sum(len(v) for v in pcap.pulses.values())
        assert total > 0
        # Verify pulse data structure
        ch_key = "Channel 1"
        first_pulse = pcap.pulses[ch_key][1]
        assert "Amp_I" in first_pulse
        assert "Amp_Q" in first_pulse
        assert "Time" in first_pulse
        assert "pileup" in first_pulse


# ═══════════════════════════════════════════════════════════════════
#  Histogram Accumulator Tests
# ═══════════════════════════════════════════════════════════════════

class TestHistogramAccumulator:
    def test_single_add(self):
        edges = np.array([0, 10, 20, 30])
        h = HistogramAccumulator(edges)
        h.add(5.0)
        assert h.total == 1
        assert h.counts[0] == 1  # bin [0, 10)
        assert h.counts[1] == 0
        assert h.counts[2] == 0

    def test_add_many(self):
        edges = np.array([0, 10, 20, 30])
        h = HistogramAccumulator(edges)
        h.add_many(np.array([5, 15, 25, 5, 15]))
        assert h.total == 5
        assert h.counts[0] == 2
        assert h.counts[1] == 2
        assert h.counts[2] == 1

    def test_out_of_range(self):
        edges = np.array([0, 10, 20])
        h = HistogramAccumulator(edges)
        h.add(-5.0)  # below range
        h.add(25.0)  # above range
        assert h.total == 0  # neither should be counted

    def test_bin_centers(self):
        edges = np.array([0, 10, 20, 30])
        h = HistogramAccumulator(edges)
        np.testing.assert_array_equal(h.bin_centers, [5, 15, 25])

    def test_reset(self):
        edges = np.array([0, 10, 20])
        h = HistogramAccumulator(edges)
        h.add(5.0)
        h.add(15.0)
        assert h.total == 2
        h.reset()
        assert h.total == 0
        assert np.all(h.counts == 0)


class TestPulseHistogramSet:
    def test_add_pulse(self):
        hs = PulseHistogramSet(
            amp_range=(0, 200), amp_bins=10,
            duration_range_ms=(0, 10), duration_bins=10,
            snr_range=(0, 30), snr_bins=10,
        )
        ns = _make_noise_stats(std_I=10.0, std_Q=10.0)
        pulse = _make_pulse_data(n_samples=50, peak_I=100.0, peak_Q=20.0)
        metrics = hs.add_pulse(1, pulse, ns)

        assert metrics["peak_amp"] > 0
        assert metrics["snr"] > 0
        assert metrics["duration_ms"] > 0
        assert hs.total_pulses() == 1
        assert hs.total_pulses(channel=1) == 1

    def test_multi_channel(self):
        hs = PulseHistogramSet(amp_range=(0, 200), amp_bins=10)
        ns = _make_noise_stats(std_I=10.0, std_Q=10.0)
        pulse = _make_pulse_data(n_samples=50, peak_I=100.0)

        hs.add_pulse(1, pulse, ns)
        hs.add_pulse(1, pulse, ns)
        hs.add_pulse(2, pulse, ns)

        assert hs.total_pulses() == 3
        assert hs.total_pulses(channel=1) == 2
        assert hs.total_pulses(channel=2) == 1

    def test_get_histogram_data(self):
        hs = PulseHistogramSet(amp_range=(0, 200), amp_bins=10)
        ns = _make_noise_stats(std_I=10.0, std_Q=10.0)
        pulse = _make_pulse_data(n_samples=50, peak_I=100.0)
        hs.add_pulse(1, pulse, ns)

        data = hs.get_histogram_data()
        assert "amplitude_bins" in data
        assert "amplitude_counts_ch1" in data
        assert "snr_bins" in data
        assert "duration_ms_bins" in data

    def test_reset_all(self):
        hs = PulseHistogramSet(amp_range=(0, 200), amp_bins=10)
        ns = _make_noise_stats(std_I=10.0, std_Q=10.0)
        pulse = _make_pulse_data(n_samples=50, peak_I=100.0)
        hs.add_pulse(1, pulse, ns)
        assert hs.total_pulses() == 1
        hs.reset_all()
        assert hs.total_pulses() == 0


# ═══════════════════════════════════════════════════════════════════
#  HDF5 Writer/Reader Tests
# ═══════════════════════════════════════════════════════════════════

@requires_h5py
class TestPulseHDF5:
    def _make_writer(self, path, channels=None, capture_params=None):
        if channels is None:
            channels = [1, 2]
        noise_stats = {ch: _make_noise_stats(std_I=10.0, std_Q=10.0)
                       for ch in channels}
        if capture_params is None:
            capture_params = {
                "streamer_mode": "slow",
                "threshold_sigma": 3.0,
                "end_sigma": 1.5,
                "module": 1,
            }
        return PulseHDF5Writer(path, channels, noise_stats, capture_params)

    def test_writer_creates_file(self, tmp_path):
        path = tmp_path / "test.h5"
        writer = self._make_writer(path)
        assert path.exists()
        assert writer.is_open
        writer.finalize()
        assert not writer.is_open

    def test_writer_metadata(self, tmp_path):
        path = tmp_path / "test.h5"
        writer = self._make_writer(path)
        writer.finalize()

        with PulseHDF5Reader(path) as reader:
            assert reader.metadata["streamer_mode"] == "slow"
            assert reader.metadata["threshold_sigma"] == 3.0
            assert reader.channels == [1, 2]
            assert "capture_start" in reader.metadata
            assert "capture_end" in reader.metadata

    def test_append_and_read_pulse(self, tmp_path):
        path = tmp_path / "test.h5"
        writer = self._make_writer(path, channels=[1])
        pulse = _make_pulse_data(n_samples=30, peak_I=150.0, peak_Q=25.0)
        writer.append_pulse(channel=1, pulse_idx=1, pulse_data=pulse)
        writer.finalize()

        with PulseHDF5Reader(path) as reader:
            assert reader.pulse_count(1) == 1
            loaded = reader.get_pulse(1, 1)
            assert loaded is not None
            np.testing.assert_array_almost_equal(
                loaded["Amp_I"], pulse["Amp_I"])
            np.testing.assert_array_almost_equal(
                loaded["Amp_Q"], pulse["Amp_Q"])
            np.testing.assert_array_almost_equal(
                loaded["Time"], pulse["Time"])
            assert loaded["pileup"] == False
            assert loaded["n_samples"] == 30
            assert loaded["peak_I"] > 0
            assert loaded["peak_snr_I"] > 0
            assert loaded["duration_s"] > 0

    def test_multiple_pulses_multiple_channels(self, tmp_path):
        path = tmp_path / "test.h5"
        writer = self._make_writer(path, channels=[1, 2])

        for ch in [1, 2]:
            for idx in range(1, 6):
                pulse = _make_pulse_data(
                    n_samples=20 + idx, peak_I=50.0 * idx)
                writer.append_pulse(channel=ch, pulse_idx=idx,
                                    pulse_data=pulse)
        writer.finalize()

        with PulseHDF5Reader(path) as reader:
            assert reader.pulse_count(1) == 5
            assert reader.pulse_count(2) == 5
            # Verify last pulse on each channel
            p1 = reader.get_pulse(1, 5)
            p2 = reader.get_pulse(2, 5)
            assert p1 is not None
            assert p2 is not None
            assert p1["n_samples"] == 25
            assert p2["n_samples"] == 25

    def test_pulse_metadata_only(self, tmp_path):
        path = tmp_path / "test.h5"
        writer = self._make_writer(path, channels=[1])
        pulse = _make_pulse_data(n_samples=40, peak_I=200.0, pileup=True)
        writer.append_pulse(channel=1, pulse_idx=1, pulse_data=pulse)
        writer.finalize()

        with PulseHDF5Reader(path) as reader:
            meta = reader.get_pulse_metadata(1, 1)
            assert meta is not None
            assert meta["pileup"] == True
            assert meta["n_samples"] == 40
            assert meta["peak_I"] > 0

    def test_iter_pulse_metadata(self, tmp_path):
        path = tmp_path / "test.h5"
        writer = self._make_writer(path, channels=[1])
        for idx in range(1, 4):
            pulse = _make_pulse_data(n_samples=30 + idx)
            writer.append_pulse(channel=1, pulse_idx=idx, pulse_data=pulse)
        writer.finalize()

        with PulseHDF5Reader(path) as reader:
            metas = list(reader.iter_pulse_metadata(1))
            assert len(metas) == 3
            assert metas[0]["pulse_idx"] == 1
            assert metas[2]["pulse_idx"] == 3

    def test_noise_stats_roundtrip(self, tmp_path):
        path = tmp_path / "test.h5"
        ns_in = _make_noise_stats(mean_I=100.0, std_I=12.5,
                                   mean_Q=200.0, std_Q=8.3)
        noise_stats = {1: ns_in}
        writer = PulseHDF5Writer(
            path, [1], noise_stats,
            {"streamer_mode": "slow", "threshold_sigma": 3.0})
        writer.finalize()

        with PulseHDF5Reader(path) as reader:
            ns_out = reader.noise_stats(1)
            assert ns_out.mean_I == pytest.approx(100.0)
            assert ns_out.std_I == pytest.approx(12.5)
            assert ns_out.mean_Q == pytest.approx(200.0)
            assert ns_out.std_Q == pytest.approx(8.3)

    def test_df_calibration(self, tmp_path):
        path = tmp_path / "test.h5"
        noise_stats = {1: _make_noise_stats()}
        writer = PulseHDF5Writer(
            path, [1], noise_stats,
            {"streamer_mode": "slow"},
            df_calibrations={1: 42.5},
        )
        writer.finalize()

        with PulseHDF5Reader(path) as reader:
            assert reader.df_calibration(1) == pytest.approx(42.5)
            assert reader.df_calibration(99) is None

    def test_nonexistent_pulse_returns_none(self, tmp_path):
        path = tmp_path / "test.h5"
        writer = self._make_writer(path, channels=[1])
        writer.finalize()

        with PulseHDF5Reader(path) as reader:
            assert reader.get_pulse(1, 999) is None
            assert reader.get_pulse(99, 1) is None
            assert reader.get_pulse_metadata(1, 999) is None

    def test_histogram_roundtrip(self, tmp_path):
        path = tmp_path / "test.h5"
        writer = self._make_writer(path, channels=[1])

        hist_data = {
            "amplitude_bins": np.array([5.0, 15.0, 25.0]),
            "amplitude_counts_ch1": np.array([10, 20, 5], dtype=np.int64),
        }
        writer.update_histograms(hist_data)
        writer.finalize()

        with PulseHDF5Reader(path) as reader:
            loaded = reader.get_histograms()
            np.testing.assert_array_equal(
                loaded["amplitude_bins"], hist_data["amplitude_bins"])
            np.testing.assert_array_equal(
                loaded["amplitude_counts_ch1"],
                hist_data["amplitude_counts_ch1"])


# ═══════════════════════════════════════════════════════════════════
#  Integration: PulseCapture → on_pulse → HDF5 → Reader
# ═══════════════════════════════════════════════════════════════════

@requires_h5py
class TestIntegration:
    """End-to-end: synthetic pulses → PulseCapture with callback →
    HDF5 writer → reader verification."""

    def test_capture_to_hdf5_roundtrip(self, tmp_path):
        path = tmp_path / "integration.h5"
        channels = [1]
        noise_stats = {1: _make_noise_stats(std_I=5.0, std_Q=5.0)}

        # Set up HDF5 writer
        writer = PulseHDF5Writer(
            path, channels, noise_stats,
            {"streamer_mode": "slow", "threshold_sigma": 5.0},
        )

        # Set up histogram set
        histograms = PulseHistogramSet(
            amp_range=(0, 300), amp_bins=30,
            snr_range=(0, 50), snr_bins=25,
        )

        # Callback: write to HDF5 + update histograms
        def on_pulse(channel, pulse_idx, pulse_data):
            writer.append_pulse(channel, pulse_idx, pulse_data)
            histograms.add_pulse(channel, pulse_data, noise_stats[channel])

        # Create PulseCapture with callback, no accumulation
        pcap = PulseCapture(
            buf_size=5000,
            channels=channels,
            noise_stats=noise_stats,
            threshold_sigma=5.0,
            end_sigma=1.5,
            sample_rate=38147.0,
            on_pulse=on_pulse,
            accumulate=False,
        )
        pcap.start_time = 0.0

        # Feed synthetic stream
        for ch, i_val, q_val, ts in _generate_synthetic_stream(
            channels, noise_stats, peak_amplitude=100.0,
        ):
            pcap.process_sample(ch, i_val, q_val, ts)

        # Finalize
        writer.update_histograms(histograms.get_histogram_data())
        writer.finalize()

        # Verify: no accumulation in memory
        assert len(pcap.pulses["Channel 1"]) == 0

        # Verify: HDF5 has the pulses
        n_detected = pcap.pulse_count["Channel 1"]
        assert n_detected > 0

        with PulseHDF5Reader(path) as reader:
            assert reader.pulse_count(1) == n_detected

            # Read first pulse
            pulse = reader.get_pulse(1, 1)
            assert pulse is not None
            assert len(pulse["Amp_I"]) > 0
            assert pulse["peak_snr_I"] > 0

            # Histograms present
            hists = reader.get_histograms()
            assert "amplitude_counts_ch1" in hists
            assert np.sum(hists["amplitude_counts_ch1"]) == n_detected

        # Verify histogram set agrees
        assert histograms.total_pulses() == n_detected
