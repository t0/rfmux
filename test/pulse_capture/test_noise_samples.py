"""Noise samples: every channel over one window, taken at random
moments whatever the samples hold, filed as events of kind "noise"."""

import numpy as np
import pytest

from rfmux.core.transferfunctions import VOLTS_PER_ROC
from rfmux.pulse_capture import (
    DualPulseCaptureSession, NoiseSampler, PulseCaptureConfig,
    PulseCaptureSession, PulseHDF5Reader)

FS = 1000.0
CHANNELS = [1, 2, 3]


# ── the schedule ──────────────────────────────────────────────────

def test_waits_are_normally_distributed_about_the_interval():
    s = NoiseSampler(interval_s=2.0, window_s=0.06,
                     rng=np.random.default_rng(0))
    waits = np.array([s._wait() for _ in range(20000)])
    assert waits.mean() == pytest.approx(2.0, rel=0.01)
    assert waits.std() == pytest.approx(NoiseSampler.JITTER * 2.0, rel=0.03)
    # Normal, not exponential: symmetric, the median at the mean, and
    # two thirds of the waits within one sigma.
    assert np.median(waits) == pytest.approx(2.0, rel=0.01)
    assert np.mean(np.abs(waits - 2.0) < 0.5) == pytest.approx(0.683, abs=0.01)


def test_samples_never_overlap_however_short_the_interval():
    s = NoiseSampler(interval_s=0.01, window_s=0.06, pre_s=0.005,
                     rng=np.random.default_rng(1))
    s.take(0.0)
    windows, now = [], 0.0
    while len(windows) < 50:
        now += 0.001
        got = s.take(now)
        if got is not None:
            windows.append(got[1])
    assert all(b[0] >= a[1] - 1e-12 for a, b in zip(windows, windows[1:]))


def test_the_window_starts_the_pre_pulse_time_before_the_moment():
    s = NoiseSampler(interval_s=1.0, window_s=0.06, pre_s=0.005,
                     rng=np.random.default_rng(2))
    assert s.take(10.0) is None                   # starts the schedule
    moment, (t0, t1) = s.take(100.0)
    assert t0 == pytest.approx(moment - 0.005)
    assert t1 - t0 == pytest.approx(0.06)


# ── a capture ─────────────────────────────────────────────────────

def _capture(tmp_path=None, pulses=(), seconds=6.0, **config_kw):
    """Noise on three channels for *seconds*, with a clean pulse at each
    of *pulses* (channel, start sample)."""
    cfg = PulseCaptureConfig(max_pulse_ms=50.0, noise_train_ms=300.0,
                             trigger_samples=1, trigger_basis="iq",
                             **config_kw)
    events = []
    path = None if tmp_path is None else tmp_path / "noise.h5"
    s = PulseCaptureSession(
        channels=CHANNELS, sample_rate=FS, hdf5_path=path,
        on_event=events.append, noise_rng=np.random.default_rng(5),
        time_offset_s=0.0,               # the fed times are the stored ones
        **cfg.session_kwargs(FS))
    s.start()
    rng = np.random.default_rng(3)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    data = {ch: rng.normal(0, 1, n) for ch in CHANNELS}
    for ch, start in pulses:
        data[ch][start:start + 4] += 60.0
    for lo in range(0, n, 50):
        for ch in CHANNELS:
            s.feed_block(ch, data[ch][lo:lo + 50], rng.normal(0, 1, 50),
                         t[lo:lo + 50])
    s.stop()
    return events, path, cfg, data


def test_noise_samples_hold_every_channel_over_one_fixed_window():
    events, _, cfg, data = _capture(noise_capture_interval_s=1.0)
    assert 3 <= len(events) <= 8
    length = round(cfg.noise_capture_window_ms * 1e-3 * FS)
    for e in events:
        assert e["kind"] == "noise" and e["members"] == []
        assert sorted(e["dump"]) == CHANNELS
        for ch in CHANNELS:
            times = e["dump"][ch]["Time"]
            assert abs(len(times) - length) <= 1
            # The samples are the stream's own, whatever they hold,
            # in the units the capture stores (volts, uncalibrated).
            lo = int(round(times[0] * FS))
            np.testing.assert_allclose(
                e["dump"][ch]["Amp_I"],
                data[ch][lo:lo + len(times)] * VOLTS_PER_ROC, rtol=1e-12)
    waits = np.diff([e["trigger_time"] for e in events])
    assert np.all(np.abs(waits - 1.0) < 4 * NoiseSampler.JITTER)


def test_a_pulse_inside_a_noise_sample_is_named_but_not_needed(tmp_path):
    """The sample is taken regardless; a pulse that fell inside it is
    listed as a member, and is still a pulse of its channel."""
    dry, _, _, _ = _capture(noise_capture_interval_s=1.0)
    t0, _ = dry[1]["window"]
    start = int(round(t0 * FS)) + 20             # same seed, same schedule
    events, path, _, _ = _capture(tmp_path, pulses=[(2, start)],
                                  noise_capture_interval_s=1.0)
    assert [e["window"] for e in events] == [e["window"] for e in dry]
    hit = events[1]
    assert [(m["channel"], m["pulse_idx"]) for m in hit["members"]] == [(2, 1)]
    assert sorted(hit["dump"]) == CHANNELS
    assert all(e["members"] == [] for e in events if e is not hit)
    with PulseHDF5Reader(path) as r:
        assert r.pulse_count(2) == 1
        stored = r.get_event(hit["event_idx"])
        assert stored["kind"] == "noise" and stored["dumped"] == CHANNELS
        assert [(m["channel"], m["pulse_idx"]) for m in stored["members"]] \
            == [(2, 1)]
        assert r.get_event(1)["members"] == []
        assert r.metadata["noise_capture_interval_s"] == 1.0


def test_a_sample_is_as_long_as_a_typical_saved_record():
    """The longest possible record until pulses show what a typical one
    is, then the median of those saved."""
    s = NoiseSampler(interval_s=1.0, window_s=0.06)
    for length in (0.010, 0.012, 0.011, 0.500):
        s.observe(length)
    assert s.window_s == 0.06
    s.observe(0.013)
    assert s.window_s == pytest.approx(0.012)

    pulses = [(1 + k % 3, 2000 + 150 * k) for k in range(12)]
    events, _, cfg, _ = _capture(pulses=pulses, noise_capture_interval_s=1.0)
    lengths = [e["window"][1] - e["window"][0]
               for e in events if e["kind"] == "noise"]
    assert lengths[0] == pytest.approx(cfg.noise_capture_window_ms * 1e-3)
    assert lengths[-1] < 0.5 * lengths[0]


def test_pulses_are_not_made_events_by_noise_samples_alone():
    events, _, _, _ = _capture(pulses=[(1, 2500)],
                               noise_capture_interval_s=1.0)
    assert {e["kind"] for e in events} == {"noise"}


def test_noise_samples_and_coincidence_events_share_one_numbering():
    events, _, _, _ = _capture(pulses=[(1, 2500), (2, 2502)],
                               noise_capture_interval_s=1.0,
                               coincidence_window_ms=5.0)
    assert [e["event_idx"] for e in events] == list(range(1, len(events) + 1))
    kinds = [e["kind"] for e in events]
    assert kinds.count("pulses") == 1 and kinds.count("noise") >= 3


def test_none_are_taken_unless_asked_for():
    events, _, _, _ = _capture()
    assert events == []


def test_both_mode_takes_both_streams_of_every_channel():
    cfg = PulseCaptureConfig(max_pulse_ms=50.0, noise_train_ms=100.0,
                             trigger_basis="iq", noise_capture_interval_s=0.5)
    events = []
    dual = DualPulseCaptureSession(
        channels=[1, 2], slow_rate=1000.0, fast_rate=20000.0, config=cfg,
        slow_time_offset_s=0.0, on_event=events.append,
        noise_rng=np.random.default_rng(5))
    dual.start()
    rng = np.random.default_rng(7)
    for k in range(60):
        for feed, fs in ((dual.feed_slow_block, 1000.0),
                         (dual.feed_fast_block, 20000.0)):
            n = int(0.05 * fs)
            t = k * 0.05 + np.arange(n) / fs
            for ch in (1, 2):
                feed(ch, rng.normal(0, 1, n), rng.normal(0, 1, n), t)
    dual.stop()
    assert len(events) >= 3 and {e["kind"] for e in events} == {"noise"}
    for e in events:
        t0, t1 = e["window"]
        for ch in (1, 2):
            for side, fs in (("slow_tod", 1000.0), ("fast_tod", 20000.0)):
                times = e["dump"][ch][side]["Time"]
                assert times[0] - t0 < 1.5 / fs and t1 - times[-1] < 1.5 / fs


def test_config_validation():
    assert any(sev == "error" and "noise capture" in msg.lower() for sev, msg
               in PulseCaptureConfig(noise_capture_interval_s=-1).validate())
    assert any(sev == "warning" and "back to back" in msg for sev, msg in
               PulseCaptureConfig(noise_capture_interval_s=0.05).validate())
