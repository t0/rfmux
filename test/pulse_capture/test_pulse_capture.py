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
def _collecting_capture(*args, **kwargs):
    """A PulseCapture that also keeps every pulse in ``.pulses``.

    The detector deliberately retains nothing — completed pulses leave
    through ``on_pulse`` and the only memory it holds is the ring buffer,
    so a capture can run indefinitely.  Tests that assert on whole
    waveforms need them kept somewhere, so this re-creates the old
    ``{"Channel N": {idx: pulse_data}}`` dict here, in the tests, rather
    than in production code.  Any caller-supplied on_pulse still runs.
    """
    pcap = PulseCapture(*args, **kwargs)
    pcap.pulses = {f"Channel {c}": {} for c in pcap.channels}
    user_cb = pcap.on_pulse

    def _collect(channel, pulse_idx, pulse_data):
        pcap.pulses[f"Channel {channel}"][pulse_idx] = pulse_data
        if user_cb is not None:
            user_cb(channel, pulse_idx, pulse_data)

    pcap.on_pulse = _collect
    return pcap


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
    """on_pulse is the only way a completed pulse leaves the detector."""

    def _run_detection(self, on_pulse=None, factory=_collecting_capture):
        channels = [1]
        noise_stats = {1: _make_noise_stats(std_I=5.0, std_Q=5.0)}

        pcap = factory(
            buf_size=5000,
            channels=channels,
            noise_stats=noise_stats,
            threshold_sigma=5.0,
            end_sigma=1.5,
            sample_rate=38147.0,
            on_pulse=on_pulse,
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

    def test_callback_sees_every_counted_pulse(self):
        """The callback fires exactly once per counted pulse."""
        captured = []

        def on_pulse(channel, pulse_idx, pulse_data):
            captured.append(pulse_idx)

        pcap = self._run_detection(on_pulse=on_pulse)

        assert len(captured) > 0
        assert pcap.pulse_count["Channel 1"] == len(captured)
        assert captured == sorted(captured), "indices should be monotone"

    def test_detector_retains_no_pulses(self):
        """The detector holds the ring buffer and nothing else.

        This is what lets a capture run indefinitely: a detector that
        kept every pulse would grow without bound, which is precisely
        what the removed ``accumulate`` flag used to do by default.
        Constructed raw, NOT through _collecting_capture, since that
        helper deliberately re-adds the retention for other tests.
        """
        captured = []

        pcap = self._run_detection(
            on_pulse=lambda ch, idx, data: captured.append(idx),
            factory=PulseCapture)

        assert len(captured) > 0, "no pulses detected — test is vacuous"
        assert pcap.pulse_count["Channel 1"] == len(captured)
        assert not hasattr(pcap, "pulses"), (
            "PulseCapture must not accumulate pulses in memory")

    def test_backward_compat_no_callback(self):
        """Without on_pulse, behavior is identical to original code."""
        pcap = self._run_detection(on_pulse=None)

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

        # The raw detector, not _collecting_capture: this is the streaming
        # path, where pulses go to disk through the callback and nothing
        # is held in memory.
        pcap = PulseCapture(
            buf_size=5000,
            channels=channels,
            noise_stats=noise_stats,
            threshold_sigma=5.0,
            end_sigma=1.5,
            sample_rate=38147.0,
            on_pulse=on_pulse,
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
                    "max_recordable_ms", "noise_samples",
                    "edge_lookback", "edge_lookback_ms",
                    "max_capture_samples", "max_capture_ms",
                    "edge_floor_sigma"):
            assert key in d
        assert d["buf_mb_total"] == pytest.approx(
            2 * d["buf_mb_per_channel"])

    def test_edge_and_hard_stop_follow_max_pulse(self):
        """Both new time scales derive from max_pulse_ms — no knobs."""
        cfg = PulseCaptureConfig(max_pulse_ms=100.0)
        assert cfg.edge_lookback_samples(1000.0) == 10   # 10% of pulse
        assert cfg.max_capture_samples(1000.0) == 120    # 1.2x pulse
        kw = cfg.session_kwargs(1000.0)
        assert kw["edge_lookback"] == 10
        assert kw["max_capture_samples"] == 120
        d = cfg.describe(1000.0)
        assert d["edge_lookback_ms"] == pytest.approx(10.0)
        assert d["max_capture_ms"] == pytest.approx(120.0)
        assert d["edge_floor_sigma"] == pytest.approx(
            cfg.threshold_sigma * np.sqrt(2.0))


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

    def test_pileup_fragments_are_not_stacked(self):
        """Both fragments of a split carry the pileup flag; neither may
        enter the template — the first has its tail cut and the second
        sits on a pedestal, both biasing the mean and residual RMS."""
        ns = ChannelNoiseStats(std_I=1.0, std_Q=1.0)
        acc = PulseTemplateAccumulator(threshold_sigma=5.0)
        piled = _shifted_pulse(shift=40)
        piled["pileup"] = True
        assert not acc.add(piled, ns)
        assert acc.n_pulses == 0 and acc.n_skipped == 1
        assert acc.add(_shifted_pulse(shift=40), ns)
        assert acc.n_pulses == 1


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


def _run(baseline_window, amp, seed=0, n=20000, pulse_at=None,
         monotonic=False, **pcap_kw):
    """Feed a drifting stream; return (pcap, noise_stats)."""
    ns = {1: ChannelNoiseStats(mean_I=0.0, std_I=1.0,
                               mean_Q=0.0, std_Q=1.0)}
    pcap = _collecting_capture(
        buf_size=5000, channels=[1], noise_stats=ns,
        threshold_sigma=5.0, end_sigma=1.5, margin_fraction=0.1, baseline_window=baseline_window, **pcap_kw)
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


class TestRollingBaseline:
    """sigma is stationary and comes from the long training record; the
    mean drifts, so it is re-estimated continuously as the median of a
    window long compared with a pulse."""

    def test_slow_drift_cannot_trigger_even_a_frozen_baseline(self):
        """Monotonic drift used to park the signal past threshold and
        deadlock the engine mid-capture.  The edge gate now vetoes the
        trigger outright — over one lookback the drift moves a fraction
        of a σ — so nothing fires at all, stale baseline or not."""
        pcap, ns = _run(baseline_window=0, amp=15.0, monotonic=True)
        assert not pcap.state[1].capturing, \
            "the engine must never wedge mid-capture"
        assert pcap.pulse_count["Channel 1"] == 0, \
            "slow drift is not a pulse"
        assert ns.mean_I == 0.0, "frozen baseline must not move"

    def test_rolling_baseline_survives_monotonic_drift(self):
        pcap, ns = _run(baseline_window=1000, amp=15.0, monotonic=True)
        assert not pcap.state[1].capturing, \
            "the rolling median should have kept the engine free"
        assert ns.mean_I > 5.0, f"baseline barely moved ({ns.mean_I:.2f})"

    def test_a_step_capture_always_terminates(self):
        """A 15σ step IS a fast event, so it triggers; the signal then
        never returns to the old baseline.  Frozen baseline: the hard
        stop truncates the capture at the ring's capacity and the edge
        gate keeps the parked signal from re-firing — exactly one
        flagged event.  Rolling baseline: the median walks to the new
        level and the leaky bucket ends the capture normally, sooner."""
        def step(window):
            ns = {1: ChannelNoiseStats(mean_I=0.0, std_I=1.0,
                                       mean_Q=0.0, std_Q=1.0)}
            pcap = _collecting_capture(
                buf_size=5000, channels=[1], noise_stats=ns,
                threshold_sigma=5.0, end_sigma=1.5,
                baseline_window=window)
            rng = np.random.default_rng(1)
            for k in range(20000):
                v = rng.normal(0, 1.0) + (15.0 if k >= 5000 else 0.0)
                pcap.process_sample(1, float(v),
                                    float(rng.normal(0, 1.0)), k * 1e-3)
            return pcap, ns[1]

        stuck, frozen_ns = step(0)
        assert not stuck.state[1].capturing, "hard stop did not fire"
        assert stuck.pulse_count["Channel 1"] == 1, \
            "one step, one event — the parked tail must not re-fire"
        d = stuck.pulses["Channel 1"][1]
        assert d["truncated"], "a capture ended by the hard stop is flagged"
        assert frozen_ns.mean_I == 0.0

        freed, ns = step(2000)
        assert not freed.state[1].capturing, "did not recover"
        assert freed.pulse_count["Channel 1"] == 1, \
            "the stuck capture should close out as one event"
        assert not freed.pulses["Channel 1"][1]["truncated"], \
            "the rolling median should end the capture before the stop"
        assert len(freed.pulses["Channel 1"][1]["Amp_I"]) < \
            len(d["Amp_I"]), "baseline recovery should beat the hard stop"
        assert ns.mean_I == pytest.approx(15.0, abs=0.2)

    def test_wander_never_triggers_with_the_edge_gate(self):
        """4σ wander crosses the threshold band repeatedly, but over one
        edge lookback it moves ~0.2σ — the edge gate vetoes it at ANY
        baseline window, including the frozen and too-slow ones that
        used to storm.  Suppression no longer depends on the median
        keeping up with the drift."""
        for window in (0, 1000, 4000):
            pcap, _ = _run(baseline_window=window, amp=4.0)
            assert pcap.pulse_count["Channel 1"] == 0, \
                f"wander triggered at baseline_window={window}"

    def test_edge_gate_off_restores_the_wander_storm(self):
        """Ties the suppression to the gate: amplitude-only triggering
        (edge_lookback=0, debug) still storms on a frozen baseline."""
        legacy, _ = _run(baseline_window=0, amp=4.0, edge_lookback=0)
        assert legacy.pulse_count["Channel 1"] >= 4

    def test_a_pulse_riding_on_wander_still_triggers(self):
        pcap, _ = _run(baseline_window=1000, amp=4.0, pulse_at=8000)
        assert pcap.pulse_count["Channel 1"] >= 1, \
            "the edge gate must not eat real pulses"

    def test_a_pulse_does_not_move_the_baseline(self):
        """The median ignores pulses outright rather than bounding their
        pull, so no clamp is needed."""
        pcap, ns = _run(baseline_window=1000, amp=0.0, pulse_at=8000)
        assert pcap.pulse_count["Channel 1"] >= 1, \
            "the baseline swallowed the pulse"
        assert abs(ns.mean_I) < 0.2, \
            f"pulse pulled the baseline to {ns.mean_I:.2f}"

    def test_reservoir_is_bounded_however_long_the_window(self):
        """Cost and memory must not scale with the window: a decimated
        reservoir tracks the full-stream median."""
        ns = {1: ChannelNoiseStats(std_I=1.0, std_Q=1.0)}
        big = _collecting_capture(buf_size=100, channels=[1], noise_stats=ns,
                           baseline_window=10_000_000)
        assert big._bl_capacity == PulseCapture._BASELINE_RESERVOIR
        assert big._bl_decim > 1
        small = _collecting_capture(buf_size=100, channels=[1], noise_stats=ns,
                             baseline_window=500)
        assert small._bl_capacity == 500 and small._bl_decim == 1

    def test_tracks_an_offset_to_the_right_value(self):
        ns = {1: ChannelNoiseStats(mean_I=0.0, std_I=1.0,
                                   mean_Q=0.0, std_Q=1.0)}
        pcap = _collecting_capture(buf_size=500, channels=[1], noise_stats=ns,
                            threshold_sigma=50.0, baseline_window=4000)
        rng = np.random.default_rng(4)
        for k in range(20000):
            pcap.process_sample(1, float(7.0 + rng.normal(0, 1.0)),
                                float(-3.0 + rng.normal(0, 1.0)),
                                k * 1e-3)
        assert ns[1].mean_I == pytest.approx(7.0, abs=0.2)
        assert ns[1].mean_Q == pytest.approx(-3.0, abs=0.2)

    def test_off_by_default(self):
        ns = {1: ChannelNoiseStats(std_I=1.0, std_Q=1.0)}
        pcap = _collecting_capture(buf_size=100, channels=[1], noise_stats=ns)
        assert pcap.baseline_window == 0


class TestBaselineConfig:
    def test_window_is_the_training_span_floored_against_the_ring(self):
        """One window, one requirement: long compared with a pulse.
        The ring holds one max-length pulse, so the window is floored
        against it in case training was overridden short."""
        # Once the ring is above its own minimum size, training (20x the
        # pulse) beats the ring floor (8 x 1.5x the pulse) and wins.
        cfg = PulseCaptureConfig(max_pulse_ms=1000.0)
        kw = cfg.session_kwargs(1000.0)
        assert kw["baseline_window"] == cfg.noise_samples(1000.0) == 20000
        assert kw["baseline_window"] > cfg.buf_size(1000.0)

        # Training overridden far too short: the ring floor takes over,
        # so the median never runs in a window a pulse could dominate.
        short = PulseCaptureConfig(max_pulse_ms=1000.0, noise_train_ms=1.0)
        kw = short.session_kwargs(1000.0)
        assert kw["baseline_window"] == (short.BASELINE_MIN_RINGS
                                         * short.buf_size(1000.0))
        assert kw["baseline_window"] > short.noise_samples(1000.0)


# ─────────────────── Edge gate calibration & epochs ─────────────────

class TestEdgeCalibration:
    """The edge threshold is threshold_sigma × the measured lag-K
    jump-σ, so its false rate is constant whatever correlation or 1/f
    power the detector actually has at that lag."""

    def test_jump_std_is_sqrt2_on_white_noise(self):
        rng = np.random.default_rng(11)
        arr = (rng.normal(0, 1, 50000)
               + 1j * rng.normal(0, 1, 50000)).astype(np.complex128)
        stats, _ = estimate_noise_stats({1: arr}, [1], jump_lag=500)
        assert stats[1].jump_std_I == pytest.approx(np.sqrt(2), rel=0.05)
        assert stats[1].jump_std_Q == pytest.approx(np.sqrt(2), rel=0.05)

    def test_jump_std_grows_with_wander_at_the_lag(self):
        """1/f power at the lookback lag inflates the measured jump-σ —
        the edge threshold widens to match instead of firing on it."""
        rng = np.random.default_rng(12)
        k = np.arange(50000)
        wander = 5.0 * np.sin(2 * np.pi * k / 5000)
        arr = (wander + rng.normal(0, 1, 50000)
               + 1j * rng.normal(0, 1, 50000)).astype(np.complex128)
        stats, _ = estimate_noise_stats({1: arr}, [1], jump_lag=500)
        assert stats[1].jump_std_I > 2.5, \
            "wander at the lag must be priced into the jump-σ"
        assert stats[1].jump_std_Q == pytest.approx(np.sqrt(2), rel=0.05)

    def test_short_record_falls_back_to_sqrt2_sigma(self):
        rng = np.random.default_rng(13)
        arr = (rng.normal(0, 2.0, 100)
               + 1j * rng.normal(0, 2.0, 100)).astype(np.complex128)
        stats, _ = estimate_noise_stats({1: arr}, [1], jump_lag=500)
        assert stats[1].jump_std_I == pytest.approx(
            np.sqrt(2) * stats[1].std_I)

    def test_true_pileup_still_splits(self):
        """A second pulse arriving on the first one's tail is a fresh
        rise above the NEAREST tap, so the split fires; the first
        pulse's own smooth decay never does (it sits below its own
        recent level by construction)."""
        ns = {1: ChannelNoiseStats(mean_I=0.0, std_I=1.0,
                                   mean_Q=0.0, std_Q=1.0)}
        pcap = _collecting_capture(buf_size=4000, channels=[1], noise_stats=ns,
                            threshold_sigma=5.0, end_sigma=1.5)
        rng = np.random.default_rng(9)
        for k in range(3000):
            v = rng.normal(0, 1.0)
            if k >= 800:
                v += 60.0 * np.exp(-(k - 800) / 40.0)
            if k >= 930:
                v += 60.0 * np.exp(-(k - 930) / 40.0)
            pcap.process_sample(1, float(v), float(rng.normal(0, 1.0)),
                                k * 1e-3)
        assert pcap.pulse_count["Channel 1"] == 2, \
            "the overlapped pair should split into two events"
        assert pcap.pulses["Channel 1"][1]["pileup"], \
            "the fragment cut by the split is flagged"
        assert pcap.pulses["Channel 1"][2]["pileup"], \
            "the successor sits on the first pulse's pedestal — " \
            "it is pileup-affected too"

    def test_a_single_pulse_is_never_split(self):
        """Sweep seeds: no split may fire anywhere on a lone pulse's
        rise, top, or tail."""
        for seed in range(8):
            ns = {1: ChannelNoiseStats(mean_I=0.0, std_I=1.0,
                                       mean_Q=0.0, std_Q=1.0)}
            pcap = _collecting_capture(buf_size=4000, channels=[1],
                                noise_stats=ns, threshold_sigma=5.0,
                                end_sigma=1.5)
            rng = np.random.default_rng(seed)
            for k in range(3000):
                v = rng.normal(0, 1.0)
                if k >= 800:
                    v += 60.0 * np.exp(-(k - 800) / 40.0)
                pcap.process_sample(1, float(v),
                                    float(rng.normal(0, 1.0)), k * 1e-3)
            assert pcap.pulse_count["Channel 1"] == 1, f"seed {seed}"
            assert not pcap.pulses["Channel 1"][1]["pileup"], f"seed {seed}"

    def test_epoch_reset_blocks_cross_epoch_references(self):
        """After a re-estimation shifts the mean, a lag-K reference into
        old-epoch samples differences across the shift and reads as a
        huge jump — enough to arm AND fire a pileup split inside a
        pulse's own rise.  reset_edge_history() shortens the lag until
        a full lookback of new-epoch samples exists."""
        ns = {1: ChannelNoiseStats(mean_I=0.0, std_I=1.0,
                                   mean_Q=0.0, std_Q=1.0)}
        pcap = _collecting_capture(buf_size=4000, channels=[1], noise_stats=ns,
                            threshold_sigma=5.0, end_sigma=1.5)
        rng = np.random.default_rng(7)
        for k in range(1000):
            pcap.process_sample(1, float(rng.normal(0, 1)),
                                float(rng.normal(0, 1)), k * 1e-3)
        # The baseline moved; a re-estimation swaps the stats in place.
        ns[1].mean_I = 5.0
        ns[1].mean_Q = 5.0
        pcap.reset_edge_history()
        for k in range(1000, 2000):
            j = k - 1000
            v = 5.0 + rng.normal(0, 1) \
                + (60.0 * np.exp(-(j - 500) / 40.0) if j >= 500 else 0.0)
            pcap.process_sample(1, float(v),
                                5.0 + float(rng.normal(0, 1)), k * 1e-3)
        assert pcap.pulse_count["Channel 1"] == 1, \
            "the pulse must arrive whole — no cross-epoch splits"
        assert not pcap.pulses["Channel 1"][1]["pileup"]


# ─────────────────────────── Hard stop ──────────────────────────────

class TestHardStop:
    """A capture may end late, but it must always end: the hard stop
    bounds every capture at the ring's capacity (1.2x the max pulse the
    ring was sized for) — see also
    TestRollingBaseline.test_a_step_capture_always_terminates."""

    def test_defaults_derive_from_the_ring(self):
        ns = {1: ChannelNoiseStats(std_I=1.0, std_Q=1.0)}
        pcap = _collecting_capture(buf_size=5000, channels=[1], noise_stats=ns)
        assert pcap.max_capture_samples == 4000  # 0.8 x ring
        assert pcap.edge_lookback == 400         # 0.1 x 0.8 x ring

    def test_a_normal_pulse_is_not_truncated(self):
        ns = {1: ChannelNoiseStats(mean_I=0.0, std_I=1.0,
                                   mean_Q=0.0, std_Q=1.0)}
        pcap = _collecting_capture(buf_size=4000, channels=[1], noise_stats=ns,
                            threshold_sigma=5.0, end_sigma=1.5)
        rng = np.random.default_rng(3)
        for k in range(3000):
            v = rng.normal(0, 1.0)
            if k >= 800:
                v += 60.0 * np.exp(-(k - 800) / 40.0)
            pcap.process_sample(1, float(v), float(rng.normal(0, 1.0)),
                                k * 1e-3)
        d = pcap.pulses["Channel 1"][1]
        assert d["truncated"] is False

    def test_stale_baseline_capture_ends_at_the_pre_pulse_anchor(self):
        """The reported failure: drift parks the signal 6σ from the
        (stale) mean, a pulse fires on top of it, and after the pulse
        the amplitude test STAYS above threshold — the below-threshold
        freeze and the end bucket both starve, and the capture used to
        run to the hard stop with 'end confirmed' far, far past the
        pulse.  The pre-pulse anchor ends it where the pulse actually
        returned."""
        ns = {1: ChannelNoiseStats(mean_I=0.0, std_I=1.0,
                                   mean_Q=0.0, std_Q=1.0)}
        pcap = _collecting_capture(buf_size=4000, channels=[1], noise_stats=ns,
                            threshold_sigma=5.0, end_sigma=1.5)
        rng = np.random.default_rng(11)
        for k in range(3000):
            v = 6.0 + rng.normal(0, 1.0)  # parked: dev never drops below 5
            if k >= 800:
                v += 60.0 * np.exp(-(k - 800) / 40.0)
            pcap.process_sample(1, float(v), float(rng.normal(0, 1.0)),
                                k * 1e-3)
        assert pcap.pulse_count["Channel 1"] == 1
        d = pcap.pulses["Channel 1"][1]
        assert d["truncated"] is False, \
            "the anchor must end the capture, not the hard stop"
        n = len(d["Amp_I"])
        assert n < 400, f"window still bloated ({n} samples)"
        # The end mark (bucket confirmation) sits past the window but
        # within the same neighborhood — not at the hard stop.
        assert d["end_index"] < 1000

    def test_stalled_confirmation_is_not_flagged_truncated(self):
        """A hard stop reached because the end confirmation stalled —
        the pulse itself long over — saved a COMPLETE pulse: with
        save_to_end_confirmed off the window ends at below-threshold +
        tail and the truncated flag stays off.  Truncated is reserved
        for pulses still above threshold when the stop fired."""
        ns = {1: ChannelNoiseStats(mean_I=0.0, std_I=1.0,
                                   mean_Q=0.0, std_Q=1.0)}
        pcap = _collecting_capture(buf_size=2000, channels=[1], noise_stats=ns,
                            threshold_sigma=5.0, end_sigma=1.5,
                            save_to_end_confirmed=False)
        rng = np.random.default_rng(5)
        # Pulse decays to a 3σ plateau: below threshold (so the pulse
        # "ends"), but never inside the 1.5σ end band — the bucket can
        # only stall until the hard stop.
        for k in range(3000):
            v = rng.normal(0, 1.0)
            if k >= 200:
                v += max(3.0, 60.0 * np.exp(-(k - 200) / 40.0))
            pcap.process_sample(1, float(v), float(rng.normal(0, 1.0)),
                                k * 1e-3)
        assert pcap.pulse_count["Channel 1"] == 1
        d = pcap.pulses["Channel 1"][1]
        assert d["truncated"] is False, \
            "a complete pulse with a stalled bucket is not truncated"
        below = d["below_threshold_index"]
        assert (len(d["Amp_I"]) - 1) - below <= \
            max(10, int(0.1 * (below - d["trigger_index"]))) + 2, \
            "the stalled confirmation stretch must not be saved"

    def test_stalled_confirmation_is_kept_when_saving_to_confirmed(self):
        """Same stall, default save policy: the stretch IS saved, and
        the pulse is still not truncated.  This is the case the option
        exists for — the samples are in the ring either way, and which
        of them reach disk is a policy choice, not a detection one."""
        ns = {1: ChannelNoiseStats(mean_I=0.0, std_I=1.0,
                                   mean_Q=0.0, std_Q=1.0)}
        pcap = _collecting_capture(buf_size=2000, channels=[1], noise_stats=ns,
                                   threshold_sigma=5.0, end_sigma=1.5)
        rng = np.random.default_rng(5)
        for k in range(3000):
            v = rng.normal(0, 1.0)
            if k >= 200:
                v += max(3.0, 60.0 * np.exp(-(k - 200) / 40.0))
            pcap.process_sample(1, float(v), float(rng.normal(0, 1.0)),
                                k * 1e-3)
        assert pcap.pulse_count["Channel 1"] == 1
        d = pcap.pulses["Channel 1"][1]
        assert d["truncated"] is False, \
            "the save policy must not change what counts as truncated"
        below = d["below_threshold_index"]
        tail = (len(d["Amp_I"]) - 1) - below
        assert tail > max(10, int(0.1 * (below - d["trigger_index"]))) + 2, \
            "the confirmation stretch should be saved under this policy"
        # The window now runs to where the state machine stopped.
        assert d["end_index"] == len(d["Amp_I"]) - 1

    @requires_h5py
    def test_truncated_flag_survives_hdf5(self, tmp_path):
        d = _make_pulse_data()
        d["truncated"] = True
        ns = ChannelNoiseStats(std_I=1.0, std_Q=1.0)
        w = PulseHDF5Writer(tmp_path / "trunc.h5", [1], {1: ns}, {})
        w.append_pulse(1, 1, d, ns)
        w.finalize()
        with PulseHDF5Reader(tmp_path / "trunc.h5") as r:
            assert r.get_pulse(1, 1)["truncated"] is True
            assert r.get_pulse_metadata(1, 1)["truncated"]


# ────────────────────── Trigger confirmation ────────────────────────

def _spike_stream(pcap, *, spike_len, amp=8.0, n=400, at=200):
    """Quiet noise-free baseline with one excursion of *spike_len*."""
    for k in range(n):
        val = amp if at <= k < at + spike_len else 0.0
        pcap.process_sample(1, val, 0.0, k / 1000.0)


def _mk(trigger_samples, **kw):
    ns = {1: ChannelNoiseStats(mean_I=0.0, std_I=1.0,
                               mean_Q=0.0, std_Q=1.0)}
    return _collecting_capture(
        buf_size=1000, channels=[1], noise_stats=ns,
        threshold_sigma=5.0, end_sigma=1.5, margin_fraction=0.1, trigger_samples=trigger_samples, **kw)


class TestTriggerConfirmation:
    """One sample over threshold is not evidence of a pulse.  At 5σ on
    two quadratures the per-sample accidental probability is ~1.1e-6,
    which is ~1.4 triggers per second per channel on the PFB stream."""

    def test_single_sample_spike_is_rejected(self):
        pcap = _mk(2)
        _spike_stream(pcap, spike_len=1)
        assert pcap.pulse_count["Channel 1"] == 0

    def test_single_sample_spike_triggers_when_disabled(self):
        pcap = _mk(1)
        _spike_stream(pcap, spike_len=1)
        assert pcap.pulse_count["Channel 1"] == 1, \
            "trigger_samples=1 must reproduce the old behaviour"

    def test_two_sample_excursion_still_triggers(self):
        pcap = _mk(2)
        _spike_stream(pcap, spike_len=2)
        assert pcap.pulse_count["Channel 1"] == 1

    def test_longer_confirmation_rejects_shorter_runs(self):
        assert _run_len(3, 2) == 0
        assert _run_len(3, 3) == 1

    def test_trigger_is_dated_to_the_start_of_the_run(self):
        """Confirmation must not eat the rising edge: the capture is
        timestamped at the first sample over threshold, not the one
        that confirmed it."""
        one = _mk(1)
        _spike_stream(one, spike_len=40, at=200)
        many = _mk(4)
        _spike_stream(many, spike_len=40, at=200)
        w1 = one.pulses["Channel 1"][1]
        w4 = many.pulses["Channel 1"][1]
        assert w1["Time"][0] == w4["Time"][0], (
            f"capture window slipped by the confirmation "
            f"({w1['Time'][0]} vs {w4['Time'][0]})")
        assert len(w1["Amp_I"]) == len(w4["Amp_I"])

    def test_sustained_excursion_is_captured_once(self):
        """A capture that ends while the signal is still above threshold
        must not immediately re-fire — repeatedly re-capturing one long
        excursion was how a drifting baseline produced pulse storms."""
        pcap = _mk(2)
        for k in range(4000):
            pcap.process_sample(1, 8.0 if k >= 100 else 0.0, 0.0,
                                k / 1000.0)
        assert pcap.pulse_count["Channel 1"] <= 1

    def test_accidental_rate_matches_the_gaussian_tail(self):
        cfg = PulseCaptureConfig(threshold_sigma=5.0, trigger_samples=1)
        # 2 * Phi(-5) per quadrature, either of two -> ~1.147e-6/sample
        assert cfg.accidental_rate_hz(1.0) == pytest.approx(1.147e-6,
                                                            rel=1e-2)
        # PFB rate: the number that makes single-sample triggering
        # unusable at 1.22 MHz.
        assert cfg.accidental_rate_hz(1220703.125) == pytest.approx(
            1.4, rel=0.05)
        confirmed = PulseCaptureConfig(threshold_sigma=5.0,
                                       trigger_samples=2)
        assert confirmed.accidental_rate_hz(1220703.125) < 1e-5

    def test_confirmation_length_follows_the_stream_rate(self):
        """One sample is ample evidence at 596 Hz and nowhere near
        enough at 1.22 MHz, so a fixed length cannot serve both: at the
        slow end it would reject real pulses that span less than one
        sample."""
        cfg = PulseCaptureConfig(threshold_sigma=5.0)
        assert cfg.trigger_samples_for(596.0464477539062) == 1
        assert cfg.trigger_samples_for(38147.0) == 2
        assert cfg.trigger_samples_for(1220703.125) == 2
        # Every choice keeps accidentals inside the stated budget.
        for fs in (596.0464477539062, 38147.0, 1220703.125):
            assert (60 * cfg.accidental_rate_hz(fs)
                    <= cfg.max_accidental_per_min)

    def test_explicit_confirmation_overrides_the_rate(self):
        cfg = PulseCaptureConfig(trigger_samples=4)
        assert cfg.trigger_samples_for(596.0) == 4
        assert cfg.session_kwargs(596.0)["trigger_samples"] == 4

    def test_a_lower_threshold_demands_more_confirmation(self):
        low = PulseCaptureConfig(threshold_sigma=3.0)
        high = PulseCaptureConfig(threshold_sigma=6.0)
        assert (low.trigger_samples_for(1220703.125)
                > high.trigger_samples_for(1220703.125))

    def test_validate_reports_the_accidental_rate(self):
        loud = PulseCaptureConfig(threshold_sigma=5.0, trigger_samples=1)
        issues = loud.validate(sample_rate=1220703.125)
        assert any(s == "warning" and "times per minute" in m
                   for s, m in issues)


def _run_len(trigger_samples, spike_len):
    pcap = _mk(trigger_samples)
    _spike_stream(pcap, spike_len=spike_len)
    return pcap.pulse_count["Channel 1"]


# ────────────────── Training window is memory-bounded ───────────────

class TestTrainingWindow:
    def test_training_length_is_memory_bounded(self):
        """Raising the window by hand must stay safe: the record is held
        whole, and 30 s at 1.22 MHz is 36.6M samples per channel."""
        cfg = PulseCaptureConfig(noise_train_ms=30_000.0)
        assert cfg.noise_samples(19073.486328125) == 572_205
        assert cfg.noise_samples(1220703.125) == cfg._MAX_NOISE


# ─────────────── Where the state machine actually acted ─────────────

class TestDecisionMarks:
    """A saved capture should be reviewable against the decisions that
    produced it, not just its samples."""

    def _run(self, trigger_samples=2, start=800, tau=40.0, amp=60.0,
             n=3000, seed=3, save_to_end_confirmed=True):
        ns = {1: ChannelNoiseStats(mean_I=0.0, std_I=1.0,
                                   mean_Q=0.0, std_Q=1.0)}
        pcap = _collecting_capture(
            buf_size=4000, channels=[1], noise_stats=ns,
            threshold_sigma=5.0, end_sigma=1.5, margin_fraction=0.1,
            trigger_samples=trigger_samples,
            save_to_end_confirmed=save_to_end_confirmed)
        rng = np.random.default_rng(seed)
        for k in range(n):
            v = rng.normal(0, 1.0)
            if k >= start:
                v += amp * np.exp(-(k - start) / tau)
            pcap.process_sample(1, float(v), float(rng.normal(0, 1.0)),
                                k * 1e-3)
        return pcap.pulses["Channel 1"][1]

    def test_trigger_index_lands_on_the_onset(self):
        d = self._run(start=800)
        t = np.asarray(d["Time"])
        assert d["trigger_time"] == pytest.approx(0.800, abs=2e-3)
        assert t[d["trigger_index"]] == pytest.approx(d["trigger_time"])
        # It is the pre-trigger margin in from the window start.
        assert d["trigger_index"] > 0
        # …and it is where the signal is largest, not somewhere in noise.
        assert abs(d["Amp_I"][d["trigger_index"]]) > 20

    def test_window_ends_at_below_threshold_plus_tail(self):
        """With save_to_end_confirmed off, the saved window ends where
        the eye puts the end of the pulse — the below-threshold instant
        plus a margin_fraction tail.  The leaky bucket only bounds the
        state machine, so the end-confirmed mark sits well past the
        last saved sample."""
        d = self._run(save_to_end_confirmed=False)
        n = len(d["Amp_I"])
        below = d["below_threshold_index"]
        core = below - d["trigger_index"]
        tail = (n - 1) - below
        expected = max(10, int(0.1 * core))
        assert abs(tail - expected) <= 2, \
            f"tail {tail} vs margin-derived {expected}"
        assert d["end_index"] > n - 1
        assert d["end_time"] > d["Time"][-1]

    def test_window_runs_to_confirmation_by_default(self):
        """Default policy: the window ends exactly where the state
        machine did, so the end mark is the last sample rather than a
        pointer past the data."""
        d = self._run()
        n = len(d["Amp_I"])
        assert d["end_index"] == n - 1
        assert d["end_time"] == pytest.approx(d["Time"][-1])
        assert d["below_threshold_index"] < n - 1, \
            "below-threshold must sit inside the window, not at its edge"

    def test_saved_tail_is_longer_when_saving_to_confirmation(self):
        """The two policies differ only in how much tail reaches disk —
        same trigger, same below-threshold instant."""
        on = self._run()
        off = self._run(save_to_end_confirmed=False)
        assert on["trigger_time"] == pytest.approx(off["trigger_time"])
        assert on["below_threshold_time"] == pytest.approx(
            off["below_threshold_time"])
        assert len(on["Amp_I"]) > len(off["Amp_I"])

    def test_duration_does_not_move_with_the_save_policy(self):
        """The reason the policy can default to on: duration measures
        the threshold crossings, so it describes the pulse and not how
        long the leaky bucket took to be satisfied."""
        ns = ChannelNoiseStats(mean_I=0.0, std_I=1.0,
                               mean_Q=0.0, std_Q=1.0)
        on = pulse_summary(self._run(), ns, threshold_sigma=5.0)
        off = pulse_summary(self._run(save_to_end_confirmed=False), ns,
                            threshold_sigma=5.0)
        assert on["n_samples"] > off["n_samples"], \
            "the windows must actually differ, or this proves nothing"
        assert on["duration_ms"] == pytest.approx(off["duration_ms"])

    def test_bucket_count_reaches_its_target(self):
        d = self._run()
        assert d["end_confirm_samples"] > d["end_confirm_target"]

    def test_below_threshold_sits_between_trigger_and_end(self):
        d = self._run(tau=40.0, amp=60.0)
        below = d["below_threshold_index"]
        assert d["trigger_index"] < below < d["end_index"]
        # 60σ decaying with τ=40 crosses 5σ after τ·ln(12) ≈ 99 samples.
        assert below - d["trigger_index"] == pytest.approx(99, abs=20)

    def test_marks_survive_a_round_trip_through_hdf5(self):
        import tempfile
        from rfmux.algorithms.measurement.pulse_hdf5 import (
            PulseHDF5Reader, PulseHDF5Writer)

        d = self._run()
        ns = ChannelNoiseStats(mean_I=0.0, std_I=1.0,
                               mean_Q=0.0, std_Q=1.0)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "marks.h5"
            w = PulseHDF5Writer(path, [1], {1: ns}, {})
            w.append_pulse(1, 1, d, ns)
            w.finalize()
            with PulseHDF5Reader(path) as r:
                got = r.get_pulse(1, 1)
        for key in ("trigger_index", "end_index", "below_threshold_index",
                    "end_confirm_samples", "end_confirm_target"):
            assert got[key] == d[key], key
        assert got["end_time"] == pytest.approx(d["end_time"])
