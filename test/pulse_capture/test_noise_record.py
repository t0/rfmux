"""A capture file keeps the noise training record of each channel: the
samples the noise statistics were fitted to, in the stored units."""

import numpy as np
import pytest

pytest.importorskip("h5py")

from rfmux.pulse_capture.capture_session import PulseCaptureSession  # noqa: E402
from rfmux.pulse_capture.detection import ChannelNoiseStats  # noqa: E402
from rfmux.pulse_capture.hdf5 import (  # noqa: E402
    DualPulseHDF5Writer, PulseHDF5Reader, PulseHDF5Writer)

RECORD = np.arange(6, dtype=float) + 1j * np.arange(6, 0, -1)


def test_the_record_round_trips_per_channel(tmp_path):
    path = tmp_path / "n.h5"
    PulseHDF5Writer(path, [1, 2], {1: ChannelNoiseStats()},
                    {"streamer_mode": "slow"},
                    noise_data={1: RECORD}).finalize()
    with PulseHDF5Reader(path) as r:
        np.testing.assert_array_equal(r.noise_training(1), RECORD)
        assert r.noise_training(2) is None


def test_a_re_estimation_replaces_the_record(tmp_path):
    path = tmp_path / "n.h5"
    w = PulseHDF5Writer(path, [1], {1: ChannelNoiseStats()},
                        {"streamer_mode": "slow"}, noise_data={1: RECORD})
    w.update_noise_stats({1: ChannelNoiseStats()}, {1: RECORD[:3] * 2})
    w.finalize()
    with PulseHDF5Reader(path) as r:
        np.testing.assert_array_equal(r.noise_training(1), RECORD[:3] * 2)


def test_a_dual_file_keeps_one_record_per_stream(tmp_path):
    path = tmp_path / "d.h5"
    w = DualPulseHDF5Writer(path, [1], {"streamer_mode": "both"})
    w.set_noise_stats("slow", {1: ChannelNoiseStats()}, {1: RECORD})
    w.set_noise_stats("fast", {1: ChannelNoiseStats()}, {1: RECORD * 3})
    w.finalize()
    with PulseHDF5Reader(path) as r:
        np.testing.assert_array_equal(r.noise_training(1, "slow"), RECORD)
        np.testing.assert_array_equal(r.noise_training(1, "fast"), RECORD * 3)


def test_a_session_writes_the_samples_it_trained_on(tmp_path):
    session = PulseCaptureSession(
        channels=[1], threshold_sigma=5.0, end_sigma=1.5, buf_size=4000,
        sample_rate=38147.0, noise_samples=200,
        hdf5_path=tmp_path / "session.h5")
    rng = np.random.default_rng(1)
    session.start()
    for k in range(200):
        session.feed_sample(1, rng.normal(), rng.normal(), k / 38147.0)
    trained = np.array(session.noise_data[1])
    session.stop()
    with PulseHDF5Reader(tmp_path / "session.h5") as r:
        record = r.noise_training(1)
    assert len(record) == 200
    np.testing.assert_array_equal(record, trained)
