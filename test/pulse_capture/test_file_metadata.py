"""A capture file records its times as they were asked for, in
milliseconds, and as the sample counts they became: a single-stream
file once, a both-mode file once per stream."""

import numpy as np

from rfmux.pulse_capture.capture_session import (
    DualPulseCaptureSession, PulseCaptureConfig, PulseCaptureSession)
from rfmux.pulse_capture.detection import RATE_PARAMS
from rfmux.pulse_capture.hdf5 import PulseHDF5Reader

CONFIG = PulseCaptureConfig(max_pulse_ms=40.0, noise_train_ms=100.0,
                            pre_pulse_ms=4.0, post_pulse_ms=6.0,
                            trigger_basis="iq",
                            noise_capture_interval_s=0.5)
MS = ("pre_pulse_ms", "post_pulse_ms", "min_pulse_ms", "max_pulse_ms",
      "noise_train_ms")


def _feed(feed, channels, fs, seconds, rng):
    n = int(0.05 * fs)
    for k in range(int(seconds / 0.05)):
        t = k * 0.05 + np.arange(n) / fs
        for ch in channels:
            feed(ch, rng.normal(0, 1, n), rng.normal(0, 1, n), t)


def _single(tmp_path) -> dict:
    path = tmp_path / "single.h5"
    s = PulseCaptureSession(channels=[1, 2], sample_rate=1000.0,
                            hdf5_path=path, **CONFIG.session_kwargs(1000.0))
    s.start()
    _feed(s.feed_block, [1, 2], 1000.0, 1.0, np.random.default_rng(1))
    s.stop()
    with PulseHDF5Reader(path) as r:
        return dict(r.metadata)


def _dual(tmp_path) -> dict:
    path = tmp_path / "dual.h5"
    d = DualPulseCaptureSession(channels=[1, 2], slow_rate=1000.0,
                                fast_rate=20000.0, config=CONFIG,
                                hdf5_path=path, slow_time_offset_s=0.0)
    d.start()
    rng = np.random.default_rng(1)
    for k in range(20):
        for feed, fs in ((d.feed_slow_block, 1000.0),
                         (d.feed_fast_block, 20000.0)):
            n = int(0.05 * fs)
            t = k * 0.05 + np.arange(n) / fs
            for ch in (1, 2):
                feed(ch, rng.normal(0, 1, n), rng.normal(0, 1, n), t)
    d.stop()
    with PulseHDF5Reader(path) as r:
        return dict(r.metadata)


def test_a_single_stream_file_records_both_forms(tmp_path):
    meta = _single(tmp_path)
    for name in MS:
        assert meta[name] == CONFIG.times_ms()[name], name
    kwargs = CONFIG.session_kwargs(1000.0)
    for name in ("pre_samples", "post_samples", "max_capture_samples"):
        assert meta[name] == kwargs[name], name
    assert set(RATE_PARAMS) <= set(meta)


def test_a_both_mode_file_records_both_forms_per_stream(tmp_path):
    meta = _dual(tmp_path)
    for name in MS:
        assert meta[name] == CONFIG.times_ms()[name], name
    for stream, fs in (("slow", 1000.0), ("fast", 20000.0)):
        kwargs = CONFIG.session_kwargs(fs)
        for name in ("pre_samples", "post_samples", "max_capture_samples"):
            assert meta[f"{name}_{stream}"] == kwargs[name], (name, stream)
        assert {f"{n}_{stream}" for n in RATE_PARAMS} <= set(meta)


def test_the_noise_sample_window_is_recorded(tmp_path):
    want = CONFIG.noise_capture_window_ms * 1e-3
    assert _single(tmp_path)["noise_capture_window_s"] == want
    assert _dual(tmp_path)["noise_capture_window_s"] == want
