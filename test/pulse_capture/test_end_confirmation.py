"""The end-confirmation floor is a real setting, not a hidden constant.

A capture ends once the confirmation bucket exceeds
``max(min_end_samples, margin_fraction * core)``.  For a short pulse the
floor wins, so it alone decides how far past below-threshold the end
mark lands -- seen live as an end mark 16.8 ms after a 5 ms pulse at
596 Hz, which is ten samples, which was a class constant.
"""

import numpy as np
import pytest

from rfmux.pulse_capture.capture_session import (
    PulseCaptureConfig, PulseCaptureSession)

FS = 1000.0


def _end_gap(min_end_samples):
    """Samples from below-threshold to end-confirmed for a clean, short
    pulse whose margin_fraction * core is far below the floor."""
    got = []
    s = PulseCaptureSession(
        channels=[1], sample_rate=FS, noise_samples=300, hdf5_path=None,
        threshold_sigma=5.0, end_sigma=1.5, margin_fraction=0.1,
        min_end_samples=min_end_samples, save_to_end_confirmed=True,
        trigger_samples=1,
        on_pulse=lambda ch, idx, summ, data: got.append(data))
    s.start()
    rng = np.random.default_rng(2)
    n = 300
    s.feed_block(1, rng.normal(0, 1, n), rng.normal(0, 1, n), np.arange(n) / FS)
    # A clean 4-sample pulse, then exact baseline: every post-pulse sample
    # is inside the band, so the bucket fills by one per sample and the
    # end falls exactly where the floor says.
    m = 400
    i = np.zeros(m); i[10:14] = 60.0
    s.feed_block(1, i, np.zeros(m), (n + np.arange(m)) / FS)
    s.stop()
    assert got, "the fixture did not trigger"
    d = got[0]
    return d["end_index"] - d["below_threshold_index"]


def test_floor_sets_the_end_of_a_short_pulse():
    """Raising the floor by 30 samples moves the end mark by 30."""
    assert _end_gap(40) - _end_gap(10) == 30


def test_floor_reaches_the_engine_through_the_config():
    cfg = PulseCaptureConfig(min_end_samples=25)
    kw = cfg.session_kwargs(FS)
    assert kw["min_end_samples"] == 25
    kw["noise_samples"] = 200          # the engine is built after training
    s = PulseCaptureSession(channels=[1], sample_rate=FS, hdf5_path=None, **kw)
    s.start()
    rng = np.random.default_rng(1)
    s.feed_block(1, rng.normal(0, 1, 200), rng.normal(0, 1, 200),
                 np.arange(200) / FS)
    assert s.pcap is not None and s.pcap.min_end_samples == 25
    s.stop()


def test_floor_is_described_in_time_at_the_rate():
    d = PulseCaptureConfig(min_end_samples=10).describe(596.0)
    assert d["min_end_samples"] == 10
    assert d["min_end_ms"] == pytest.approx(16.78, abs=0.05)


def test_floor_below_one_is_refused():
    issues = PulseCaptureConfig(min_end_samples=0).validate(FS)
    assert any(sev == "error" and "floor" in msg for sev, msg in issues)


def _short_pulse_summary(save_full_tail):
    got = []
    s = PulseCaptureSession(
        channels=[1], sample_rate=FS, noise_samples=300, hdf5_path=None,
        threshold_sigma=5.0, end_sigma=1.5, margin_fraction=0.1,
        min_end_samples=10, save_to_end_confirmed=save_full_tail,
        trigger_samples=1,
        on_pulse=lambda ch, idx, summ, data: got.append((summ, data)))
    s.start()
    rng = np.random.default_rng(2)
    n = 300
    s.feed_block(1, rng.normal(0, 1, n), rng.normal(0, 1, n), np.arange(n) / FS)
    m = 400
    i = np.zeros(m); i[10:14] = 60.0
    s.feed_block(1, i, np.zeros(m), (n + np.arange(m)) / FS)
    s.stop()
    assert got
    return got[0]


def test_saved_extent_follows_the_full_tail_setting():
    """With full-tail saving OFF, the summary's saved extent ends where
    the saved data ends -- before the confirmation instant.  A display
    extent built on the confirmation instant drew the discarded tail
    straight back out of the ring, so the checkbox changed nothing on
    screen."""
    summ, data = _short_pulse_summary(save_full_tail=False)
    last_saved = float(np.max(data["Time"]))
    assert summ["saved_end_time"] == pytest.approx(last_saved)
    assert summ["saved_end_time"] < data["end_time"], \
        "saved extent must stop before end-confirmed when the tail is off"

    summ, data = _short_pulse_summary(save_full_tail=True)
    assert summ["saved_end_time"] == pytest.approx(float(np.max(data["Time"])))
    assert summ["saved_end_time"] == pytest.approx(data["end_time"])
