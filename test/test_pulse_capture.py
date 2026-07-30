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


# ───────────────────────── Phase A: pulse_analysis ──────────────────

from rfmux.algorithms.measurement.pulse_analysis import (
    derive_tau,
    pulse_peaks,
    pulse_summary,
)
from rfmux.algorithms.measurement.pulse_capture_session import (
    CaptureState,
    PulseCaptureSession,
)


def _make_decay_pulse(tau_s=1.5e-3, amp_sigma=40.0, sigma=1.0,
                      sample_rate=38147.0, n_samples=400, k0=10,
                      quadrature="I", mean=0.0):
    """Instant step rise at sample k0, clean exponential decay, no noise."""
    dt = 1.0 / sample_rate
    t = np.arange(n_samples) * dt
    shape = np.zeros(n_samples)
    k = np.arange(n_samples)
    shape[k >= k0] = np.exp(-(k[k >= k0] - k0) * dt / tau_s)
    signal = mean + amp_sigma * sigma * shape
    flat = np.full(n_samples, mean)
    if quadrature == "I":
        amp_I, amp_Q = signal, flat
    else:
        amp_I, amp_Q = flat, signal
    return {"Amp_I": amp_I, "Amp_Q": amp_Q, "Time": t, "pileup": False}


class TestPulseAnalysis:
    def test_snr_canonical_definition(self):
        ns = _make_noise_stats(std_I=10.0, std_Q=2.0)
        pulse = _make_pulse_data(peak_I=100.0, peak_Q=20.0)
        peaks = pulse_peaks(pulse, ns)
        # max(peak_I, peak_Q) / max(std_I, std_Q) — NOT per-component
        assert peaks["peak_amp"] == pytest.approx(100.0, rel=1e-6)
        assert peaks["snr"] == pytest.approx(10.0, rel=1e-6)

    def test_summary_keys_complete(self):
        ns = _make_noise_stats()
        summary = pulse_summary(_make_pulse_data(), ns, threshold_sigma=5.0)
        for key in ("n_samples", "pileup", "peak_I", "peak_Q", "peak_amp",
                    "snr", "duration_s", "duration_ms", "timestamp",
                    "tau_s", "tau_ms"):
            assert key in summary

    def test_derive_tau_recovers_tau(self):
        tau_true = 1.5e-3
        ns = ChannelNoiseStats(mean_I=0.0, std_I=1.0, mean_Q=0.0, std_Q=1.0)
        pulse = _make_decay_pulse(tau_s=tau_true, amp_sigma=40.0)
        tau = derive_tau(pulse, ns, threshold_sigma=5.0)
        assert np.isfinite(tau)
        assert tau == pytest.approx(tau_true, rel=0.05)

    def test_derive_tau_dominant_quadrature(self):
        tau_true = 1.5e-3
        ns = ChannelNoiseStats(mean_I=0.0, std_I=1.0, mean_Q=0.0, std_Q=1.0)
        pulse = _make_decay_pulse(tau_s=tau_true, amp_sigma=40.0,
                                  quadrature="Q")
        tau = derive_tau(pulse, ns, threshold_sigma=5.0)
        assert np.isfinite(tau)
        assert tau == pytest.approx(tau_true, rel=0.05)

    def test_derive_tau_nan_low_snr(self):
        ns = ChannelNoiseStats(std_I=1.0, std_Q=1.0)
        pulse = _make_decay_pulse(amp_sigma=6.0)  # below 1.3 * 5σ margin
        assert np.isnan(derive_tau(pulse, ns, threshold_sigma=5.0))

    def test_derive_tau_nan_no_crossing(self):
        ns = ChannelNoiseStats(std_I=1.0, std_Q=1.0)
        # Window ends long before the envelope decays through threshold
        pulse = _make_decay_pulse(tau_s=50e-3, amp_sigma=40.0, n_samples=60)
        assert np.isnan(derive_tau(pulse, ns, threshold_sigma=5.0))

    def test_derive_tau_nan_bad_times(self):
        ns = ChannelNoiseStats(std_I=1.0, std_Q=1.0)
        pulse = _make_decay_pulse(amp_sigma=40.0)
        pulse["Time"] = np.full(len(pulse["Time"]), np.nan)
        assert np.isnan(derive_tau(pulse, ns, threshold_sigma=5.0))

    def test_derive_tau_nan_without_noise_stats(self):
        pulse = _make_decay_pulse(amp_sigma=40.0)
        assert np.isnan(derive_tau(pulse, None, threshold_sigma=5.0))


# ───────────────────────── Phase A: tau histogram ───────────────────

class TestTauHistogram:
    def test_tau_binned_when_derivable(self):
        hist = PulseHistogramSet(threshold_sigma=5.0)
        ns = ChannelNoiseStats(std_I=1.0, std_Q=1.0)
        hist.add_pulse(1, _make_decay_pulse(amp_sigma=40.0), ns)
        assert hist.get_channel_histograms(1)["tau_ms"].total == 1

    def test_tau_not_binned_without_threshold(self):
        hist = PulseHistogramSet()  # threshold_sigma=None
        ns = ChannelNoiseStats(std_I=1.0, std_Q=1.0)
        summary = hist.add_pulse(1, _make_decay_pulse(amp_sigma=40.0), ns)
        assert np.isnan(summary["tau_ms"])
        assert hist.get_channel_histograms(1)["tau_ms"].total == 0

    def test_backward_compat_positional_args(self):
        hist = PulseHistogramSet((0, 1000), 50, (0, 10), 50, (0, 20), 50)
        ns = _make_noise_stats()
        hist.add_pulse(1, _make_pulse_data(), ns)
        metrics = hist.get_channel_histograms(1)
        assert set(metrics.keys()) == {
            "amplitude", "duration_ms", "snr", "tau_ms"}

    def test_add_pulse_returns_full_summary(self):
        hist = PulseHistogramSet(threshold_sigma=5.0)
        ns = _make_noise_stats()
        summary = hist.add_pulse(1, _make_pulse_data(), ns)
        for key in ("peak_amp", "snr", "duration_ms", "tau_ms", "peak_I"):
            assert key in summary


# ───────────────────────── Phase A: HDF5 derived attrs ──────────────

@requires_h5py
class TestHDF5DerivedAttrs:
    def test_roundtrip_unified_attrs(self, tmp_path):
        ns = {1: ChannelNoiseStats(std_I=1.0, std_Q=1.0)}
        pulse = _make_decay_pulse(tau_s=1.5e-3, amp_sigma=40.0)
        writer = PulseHDF5Writer(
            tmp_path / "cap.h5", [1], ns,
            {"streamer_mode": "slow", "threshold_sigma": 5.0})
        writer.append_pulse(1, 1, pulse)
        writer.finalize()

        with PulseHDF5Reader(tmp_path / "cap.h5") as reader:
            loaded = reader.get_pulse(1, 1)
            assert loaded["peak_amp"] == pytest.approx(40.0, rel=1e-6)
            assert loaded["snr"] == pytest.approx(40.0, rel=1e-6)
            expected_tau = derive_tau(pulse, ns[1], 5.0)
            assert loaded["tau_s"] == pytest.approx(expected_tau, rel=1e-9)
            meta = reader.get_pulse_metadata(1, 1)
            assert "tau_s" in meta and "snr" in meta and "peak_amp" in meta


# ───────────────────────── Phase A: PulseCaptureSession ─────────────

def _feed_noise(session, n, rng, mean=0.0, sigma=1.0, channel=1):
    for _ in range(n):
        session.feed_sample(channel, mean + rng.normal(0, sigma),
                            mean + rng.normal(0, sigma), None)


def _feed_capture_stream(session, n, pulse_starts, rng, *, tau_samples=40,
                         amp=60.0, mean=0.0, sigma=1.0, channel=1,
                         sample_rate=38147.0, t_offset=0.0):
    dt = 1.0 / sample_rate
    signal = mean + rng.normal(0, sigma, n)
    k = np.arange(n)
    for k0 in pulse_starts:
        m = k >= k0
        signal[m] += amp * sigma * np.exp(-(k[m] - k0) * dt / (tau_samples * dt))
    for i in range(n):
        session.feed_sample(channel, float(signal[i]),
                            mean + float(rng.normal(0, sigma)),
                            t_offset + i * dt)
    return t_offset + n * dt


@requires_h5py
class TestPulseCaptureSession:
    def _make_session(self, tmp_path, **overrides):
        events = {"noise": [], "pulses": [], "stats": [], "hists": [],
                  "errors": []}
        kwargs = dict(
            channels=[1],
            threshold_sigma=5.0,
            # 1.5σ: the leaky-bucket end condition increments when BOTH
            # I and Q are within end_sigma.  At 1.0σ that probability is
            # ~0.47 on Gaussian noise (negative drift — termination only
            # by lucky random walk); at 1.5σ it is ~0.75 (prompt).
            end_sigma=1.5,
            margin_fraction=0.2,
            buf_size=4000,
            sample_rate=38147.0,
            noise_samples=200,
            hdf5_path=tmp_path / "session.h5",
            histogram_flush_every=2,
            on_noise=lambda ns: events["noise"].append(ns),
            on_pulse=lambda ch, idx, summary, data:
                events["pulses"].append((ch, idx, summary)),
            on_stats=lambda s: events["stats"].append(s),
            on_histograms=lambda d: events["hists"].append(d),
            on_error=lambda msg: events["errors"].append(msg),
        )
        kwargs.update(overrides)
        return PulseCaptureSession(**kwargs), events

    def test_lifecycle_and_detection(self, tmp_path):
        session, events = self._make_session(tmp_path)
        rng = np.random.default_rng(42)

        session.start()
        assert session.state is CaptureState.ESTIMATING
        _feed_noise(session, 200, rng)
        assert session.state is CaptureState.CAPTURING
        assert len(events["noise"]) == 1

        _feed_capture_stream(session, 3000, [100, 900, 1700], rng)
        assert session.total_pulses == 3
        assert session.pulse_counts[1] == 3
        assert len(events["pulses"]) == 3
        for _, _, summary in events["pulses"]:
            assert summary["snr"] > 5.0
            assert np.isfinite(summary["tau_ms"])
        assert len(events["stats"]) == 3
        assert len(events["hists"]) >= 1  # flush_every=2
        assert not events["errors"]

        stats = session.stats()
        assert stats["elapsed_s"] > 0
        assert stats["rate_per_min"] > 0

        session.stop()
        assert session.state is CaptureState.STOPPED

        with PulseHDF5Reader(tmp_path / "session.h5") as reader:
            assert reader.pulse_count(1) == 3
            assert "capture_end" in reader.metadata
            hists = reader.get_histograms()
            assert np.sum(hists["tau_ms_counts_ch1"]) == 3
            pulse = reader.get_pulse(1, 1)
            assert np.isfinite(pulse["tau_s"])

    def test_invalid_timestamps_dropped(self, tmp_path):
        session, events = self._make_session(tmp_path)
        rng = np.random.default_rng(1)
        session.start()
        _feed_noise(session, 200, rng)
        session.feed_sample(1, 0.0, 0.0, None)
        session.feed_sample(1, 0.0, 0.0, float("nan"))
        assert session.dropped_invalid_ts == 2
        assert session.total_pulses == 0
        session.stop()

    def test_re_estimation_updates_noise(self, tmp_path):
        session, events = self._make_session(tmp_path)
        rng = np.random.default_rng(7)
        session.start()
        _feed_noise(session, 200, rng, mean=0.0)
        t_end = _feed_capture_stream(session, 1000, [100], rng)
        assert session.total_pulses == 1

        session.re_estimate_noise()
        assert session.state is CaptureState.ESTIMATING
        _feed_noise(session, 200, rng, mean=5.0)  # shifted baseline
        assert session.state is CaptureState.CAPTURING
        assert len(events["noise"]) == 2
        assert session.noise_stats[1].mean_I == pytest.approx(5.0, abs=1.0)

        _feed_capture_stream(session, 1000, [100], rng, mean=5.0,
                             t_offset=t_end)
        assert session.total_pulses == 2
        session.stop()

        with PulseHDF5Reader(tmp_path / "session.h5") as reader:
            assert reader.noise_stats(1).mean_I == pytest.approx(5.0, abs=1.0)
            assert reader.pulse_count(1) == 2

    def test_stop_idempotent(self, tmp_path):
        session, _ = self._make_session(tmp_path)
        rng = np.random.default_rng(3)
        session.start()
        _feed_noise(session, 200, rng)
        session.stop()
        session.stop()
        assert session.state is CaptureState.STOPPED

    def test_session_without_hdf5(self, tmp_path):
        session, events = self._make_session(tmp_path, hdf5_path=None)
        rng = np.random.default_rng(5)
        session.start()
        _feed_noise(session, 200, rng)
        _feed_capture_stream(session, 1000, [100], rng)
        assert session.total_pulses == 1
        assert session.writer is None
        assert not events["errors"]
        session.stop()

    def test_start_twice_raises(self, tmp_path):
        session, _ = self._make_session(tmp_path)
        session.start()
        with pytest.raises(RuntimeError):
            session.start()


# ───────────────────────── Histogram auto-expansion ─────────────────

class TestHistogramAutoExpand:
    def test_expand_double_merges_pairs(self):
        acc = HistogramAccumulator(np.linspace(0, 10, 11))  # 10 bins
        acc.add(0.5)   # bin 0
        acc.add(1.5)   # bin 1
        acc.add(9.5)   # bin 9
        assert acc.expand_double()
        assert acc.bin_edges[-1] == 20
        assert len(acc.counts) == 10
        assert acc.counts[0] == 2      # bins 0+1 merged
        assert acc.counts[4] == 1      # bins 8+9 merged
        assert acc.total == 3

    def test_expand_requires_zero_base_and_even_bins(self):
        assert not HistogramAccumulator(
            np.linspace(1, 10, 11)).expand_double()   # nonzero base
        assert not HistogramAccumulator(
            np.linspace(0, 10, 10)).expand_double()   # 9 bins (odd)

    def test_out_of_range_snr_expands_instead_of_dropping(self):
        """The 'SNR histogram never updates' bug: values beyond the
        configured range now expand the bins instead of vanishing."""
        hist = PulseHistogramSet(snr_range=(0, 50), snr_bins=100,
                                 threshold_sigma=5.0)
        ns = ChannelNoiseStats(std_I=1.0, std_Q=1.0)
        # SNR ≈ 400 — 8x beyond the default range
        hist.add_pulse(1, _make_decay_pulse(amp_sigma=400.0), ns)
        acc = hist.get_channel_histograms(1)["snr"]
        assert acc.total == 1, "pulse must be binned, not dropped"
        assert acc.bin_edges[-1] >= 400

    def test_expansion_keeps_channels_in_lockstep(self):
        hist = PulseHistogramSet(snr_range=(0, 50), threshold_sigma=5.0)
        ns = ChannelNoiseStats(std_I=1.0, std_Q=1.0)
        hist.add_pulse(1, _make_decay_pulse(amp_sigma=20.0), ns)   # in range
        hist.add_pulse(2, _make_decay_pulse(amp_sigma=400.0), ns)  # expands
        e1 = hist.get_channel_histograms(1)["snr"].bin_edges
        e2 = hist.get_channel_histograms(2)["snr"].bin_edges
        assert np.array_equal(e1, e2), \
            "per-channel histograms must share bin edges"
        assert hist.get_channel_histograms(1)["snr"].total == 1
        assert hist.get_channel_histograms(2)["snr"].total == 1


# ───────────────────────── PulseCaptureConfig ───────────────────────

from rfmux.algorithms.measurement.pulse_capture_session import (
    PulseCaptureConfig,
)


class TestPulseCaptureConfig:
    def test_ms_to_samples_scales_with_rate(self):
        cfg = PulseCaptureConfig(min_pulse_ms=1.0)
        assert cfg.min_pulse_samples(19073.486328125) == 19
        assert cfg.min_pulse_samples(1220703.125) == 1221

    def test_buffer_autosizing_and_floor(self):
        cfg = PulseCaptureConfig(max_pulse_ms=250.0)
        # 0.25 s * 19073 Hz * 1.5 safety
        assert cfg.buf_size(19073.486328125) == 7153
        # tiny rate hits the floor
        assert cfg.buf_size(10.0) == 1000

    def test_session_kwargs_match_session_signature(self):
        cfg = PulseCaptureConfig(min_pulse_ms=0.5, max_pulse_ms=100.0)
        kwargs = cfg.session_kwargs(19073.486328125)
        from rfmux.algorithms.measurement.pulse_capture_session import (
            PulseCaptureSession,
        )
        session = PulseCaptureSession(channels=[1], **kwargs)
        assert session.threshold_sigma == cfg.threshold_sigma
        assert session.buf_size == cfg.buf_size(19073.486328125)

    def test_validate_end_at_or_above_threshold_is_error(self):
        cfg = PulseCaptureConfig(threshold_sigma=3.0, end_sigma=3.0)
        assert any(s == "error" for s, _ in cfg.validate())

    def test_validate_low_end_sigma_warns(self):
        cfg = PulseCaptureConfig(end_sigma=1.0)
        issues = cfg.validate()
        assert any(s == "warning" and "random walk" in m
                   for s, m in issues)
        assert not any(s == "error" for s, _ in issues)

    def test_validate_min_above_max_is_error(self):
        cfg = PulseCaptureConfig(min_pulse_ms=300.0, max_pulse_ms=100.0)
        assert any(s == "error" for s, _ in cfg.validate())

    def test_validate_subsample_min_pulse_warns(self):
        cfg = PulseCaptureConfig(min_pulse_ms=0.001)  # 1 µs
        issues = cfg.validate(sample_rate=596.0464477539062)
        assert any("ineffective" in m for _, m in issues)

    def test_describe_fields(self):
        d = PulseCaptureConfig().describe(19073.486328125, n_channels=2)
        for key in ("min_pulse_samples", "buf_samples",
                    "buf_mb_per_channel", "buf_mb_total",
                    "max_recordable_ms", "noise_samples"):
            assert key in d
        assert d["buf_mb_total"] == pytest.approx(
            2 * d["buf_mb_per_channel"])


# ───────────────────────── Template stacking ────────────────────────

from rfmux.algorithms.measurement.pulse_templates import (
    PulseTemplateAccumulator,
    PulseTemplateSet,
    find_trigger_index,
)


def _shifted_pulse(shift, amp=50.0, tau=20.0, n=200, sigma=1.0, seed=0):
    """Pulse whose trigger sits at sample `shift` (+ noise)."""
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    sig = rng.normal(0, sigma, n)
    m = k >= shift
    sig[m] += amp * np.exp(-(k[m] - shift) / tau)
    return {"Amp_I": sig,
            "Amp_Q": rng.normal(0, sigma, n),
            "Time": k / 1000.0,
            "pileup": False}


class TestPulseTemplates:
    def test_trigger_index_finds_crossing(self):
        ns = ChannelNoiseStats(std_I=1.0, std_Q=1.0)
        p = _shifted_pulse(shift=40)
        assert find_trigger_index(p, ns, 5.0) == 40

    def test_trigger_index_none_when_no_crossing(self):
        ns = ChannelNoiseStats(std_I=1.0, std_Q=1.0)
        p = _shifted_pulse(shift=40, amp=1.0)  # never reaches 5σ
        assert find_trigger_index(p, ns, 5.0) is None

    def test_stack_aligns_on_trigger_not_window_start(self):
        """Pulses with DIFFERENT trigger offsets must stack coherently:
        the mean peak stays full amplitude instead of smearing."""
        ns = ChannelNoiseStats(std_I=1.0, std_Q=1.0)
        acc = PulseTemplateAccumulator(pre_samples=10, post_samples=100,
                                       threshold_sigma=5.0)
        for i, shift in enumerate([20, 45, 63, 80]):
            assert acc.add(_shifted_pulse(shift, seed=i), ns)
        assert acc.n_pulses == 4

        mean = acc.mean("I")
        # Trigger sits at index pre_samples; peak must be there
        assert int(np.nanargmax(mean)) == acc.pre_samples
        assert mean[acc.pre_samples] == pytest.approx(50.0, rel=0.15)
        # Pre-trigger region is baseline
        assert abs(np.nanmean(mean[:acc.pre_samples - 2])) < 2.0

    def test_noise_averages_down(self):
        ns = ChannelNoiseStats(std_I=1.0, std_Q=1.0)
        acc = PulseTemplateAccumulator(pre_samples=20, post_samples=60,
                                       threshold_sigma=5.0)
        for i in range(50):
            acc.add(_shifted_pulse(shift=40, seed=100 + i), ns)
        pre = acc.mean("I")[:15]           # pre-trigger = pure noise
        # 50 stacked pulses → sigma/sqrt(50) ≈ 0.14; allow generous margin
        assert np.nanstd(pre) < 0.5

    def test_counts_and_residual(self):
        ns = ChannelNoiseStats(std_I=1.0, std_Q=1.0)
        acc = PulseTemplateAccumulator(pre_samples=10, post_samples=50,
                                       threshold_sigma=5.0)
        for i in range(5):
            acc.add(_shifted_pulse(shift=30, seed=i), ns)
        assert acc.counts[acc.pre_samples] == 5
        resid = acc.residual_rms("I")
        assert np.isfinite(resid[acc.pre_samples])
        assert resid[acc.pre_samples] >= 0

    def test_time_axis_zero_at_trigger(self):
        acc = PulseTemplateAccumulator(pre_samples=10, post_samples=20)
        t = acc.time_axis(1000.0)
        assert t[acc.pre_samples] == pytest.approx(0.0)
        assert t[0] == pytest.approx(-0.01)

    def test_set_per_channel_and_export(self):
        ns = ChannelNoiseStats(std_I=1.0, std_Q=1.0)
        ts = PulseTemplateSet(pre_samples=5, post_samples=30,
                              threshold_sigma=5.0, sample_rate=1000.0)
        ts.add_pulse(1, _shifted_pulse(shift=20), ns)
        ts.add_pulse(2, _shifted_pulse(shift=25, seed=9), ns)
        assert ts.total_pulses() == 2
        assert ts.total_pulses(1) == 1
        data = ts.get_template_data()
        for key in ("template_I_ch1", "residual_I_ch1", "counts_ch1",
                    "time_s_ch1", "template_I_ch2"):
            assert key in data

    def test_unalignable_pulse_is_skipped(self):
        ns = ChannelNoiseStats(std_I=1.0, std_Q=1.0)
        acc = PulseTemplateAccumulator(threshold_sigma=5.0)
        assert not acc.add(_shifted_pulse(shift=40, amp=1.0), ns)
        assert acc.n_pulses == 0 and acc.n_skipped == 1


# ───────────────────── Tracked baseline (1/f robustness) ────────────

def _wandering_stream(n, amp, sigma=1.0, seed=0, cycles=3.0,
                      pulse_at=None, pulse_amp=60.0, tau=25.0,
                      fs=1000.0):
    """Slow BACK-AND-FORTH baseline wander + white noise.

    A monotonic ramp would leave the engine stuck mid-capture forever
    (see test_frozen_baseline_gets_stuck); a wander crosses the
    threshold repeatedly, which is what 1/f actually looks like and
    what produces countable false triggers.
    """
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    baseline = amp * np.sin(2 * np.pi * cycles * k / n)
    sig = baseline + rng.normal(0, sigma, n)
    if pulse_at is not None:
        m = k >= pulse_at
        sig[m] += pulse_amp * np.exp(-(k[m] - pulse_at) / tau)
    return sig, k / fs


def _run(track_samples, amp, seed=0, n=20000, pulse_at=None,
         monotonic=False):
    """Feed a drifting stream; return (pcap, noise_stats)."""
    ns = {1: ChannelNoiseStats(mean_I=0.0, std_I=1.0,
                               mean_Q=0.0, std_Q=1.0)}
    pcap = PulseCapture(
        buf_size=5000, channels=[1], noise_stats=ns,
        threshold_sigma=5.0, end_sigma=1.5, margin_fraction=0.1,
        accumulate=True, baseline_track_samples=track_samples)
    if monotonic:
        rng = np.random.default_rng(seed)
        k = np.arange(n)
        sig = amp * k / n + rng.normal(0, 1.0, n)
        t = k / 1000.0
    else:
        sig, t = _wandering_stream(n, amp, seed=seed, pulse_at=pulse_at)
    rng2 = np.random.default_rng(seed + 500)
    for i in range(n):
        pcap.process_sample(1, float(sig[i]),
                            float(rng2.normal(0, 1.0)), float(t[i]))
    return pcap, ns[1]


class TestTrackedBaseline:
    def test_frozen_baseline_gets_stuck_on_monotonic_drift(self):
        """The nastiest failure mode: once drift parks the signal past
        threshold, the end condition (BOTH quadratures inside end_sigma
        of the FROZEN mean) can never be satisfied, so the capture runs
        forever and no pulse is ever emitted."""
        pcap, ns = _run(track_samples=0, amp=15.0, monotonic=True)
        assert pcap.state[1].capturing, \
            "expected the engine to be stuck mid-capture"
        assert pcap.pulse_count["Channel 1"] == 0
        assert ns.mean_I == 0.0, "frozen baseline must not move"

    def test_tracking_survives_monotonic_drift(self):
        pcap, ns = _run(track_samples=500, amp=15.0, monotonic=True)
        assert not pcap.state[1].capturing, \
            "tracking should have kept the engine free"
        assert ns.mean_I > 5.0, f"baseline barely moved ({ns.mean_I:.2f})"

    def test_tracking_suppresses_false_triggers_on_wander(self):
        frozen, _ = _run(track_samples=0, amp=8.0)
        tracked, _ = _run(track_samples=500, amp=8.0)
        n_frozen = frozen.pulse_count["Channel 1"]
        n_tracked = tracked.pulse_count["Channel 1"]
        assert n_frozen > 0, "wander should trigger a frozen baseline"
        assert n_tracked < n_frozen, \
            f"tracking did not help ({n_tracked} vs {n_frozen})"

    def test_tracking_does_not_absorb_a_pulse(self):
        """A real pulse must still be detected, and must not drag the
        baseline toward the signal."""
        pcap, ns = _run(track_samples=500, amp=0.0, pulse_at=8000)
        assert pcap.pulse_count["Channel 1"] >= 1, \
            "tracking swallowed the pulse"
        assert abs(ns.mean_I) < 1.0, \
            f"pulse pulled the baseline to {ns.mean_I:.2f}"

    def test_disabled_by_default(self):
        ns = {1: ChannelNoiseStats(std_I=1.0, std_Q=1.0)}
        pcap = PulseCapture(buf_size=100, channels=[1], noise_stats=ns)
        assert pcap.baseline_track_samples == 0
        assert pcap._baseline_alpha == 0.0


class TestBaselineConfig:
    def test_ms_to_samples(self):
        cfg = PulseCaptureConfig(baseline_track_ms=500.0)
        assert cfg.baseline_track_samples(1000.0) == 500
        assert PulseCaptureConfig().baseline_track_samples(1000.0) == 0

    def test_session_kwargs_include_it(self):
        cfg = PulseCaptureConfig(baseline_track_ms=250.0)
        kwargs = cfg.session_kwargs(1000.0)
        assert kwargs["baseline_track_samples"] == 250

    def test_validate_warns_when_faster_than_pulses(self):
        cfg = PulseCaptureConfig(max_pulse_ms=50.0,
                                 baseline_track_ms=100.0)
        issues = cfg.validate()
        assert any(s == "warning" and "absorb pulse tails" in m
                   for s, m in issues)

    def test_validate_accepts_a_sane_window(self):
        cfg = PulseCaptureConfig(max_pulse_ms=50.0,
                                 baseline_track_ms=5000.0)
        issues = cfg.validate()
        assert not any(s == "error" for s, _ in issues)
        assert any("Baseline tracked" in m for _, m in issues)

    def test_negative_is_an_error(self):
        issues = PulseCaptureConfig(baseline_track_ms=-1.0).validate()
        assert any(s == "error" for s, _ in issues)
