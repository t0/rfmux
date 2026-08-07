"""
Tests for dual (slow+fast) pulse capture: the incremental matcher, the
DualPulseCaptureSession composition, and the dual HDF5 layout.

All synthetic — no sockets: samples are fed directly into the dual
session's two streams with a shared clock.
"""

import numpy as np
import pytest

from rfmux.algorithms.measurement.pulse_capture_dual import (
    DualPulseCaptureSession,
    IncrementalPulseMatcher,
)
from rfmux.algorithms.measurement.pulse_capture_session import (
    CaptureState,
    PulseCaptureConfig,
)

try:
    import h5py  # noqa: F401
    from rfmux.algorithms.measurement.pulse_hdf5 import PulseHDF5Reader
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

requires_h5py = pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")


def _summary(t, dur=0.001):
    return {"timestamp": t, "duration_s": dur}


class TestIncrementalMatcher:
    def test_pairs_within_window(self):
        pairs = []
        m = IncrementalPulseMatcher(window_s=0.05, on_pair=pairs.append)
        m.add("slow", 1, 1, _summary(1.000))
        assert pairs == []
        m.add("fast", 1, 1, _summary(1.010))
        assert len(pairs) == 1
        p = pairs[0]
        assert p["slow_idx"] == 1 and p["fast_idx"] == 1
        assert p["time_offset"] == pytest.approx(-0.010, abs=1e-6)
        assert m.matched == 1

    def test_best_candidate_wins(self):
        pairs = []
        m = IncrementalPulseMatcher(window_s=0.05, on_pair=pairs.append)
        m.add("fast", 1, 1, _summary(1.040))
        m.add("fast", 1, 2, _summary(1.005))
        m.add("slow", 1, 1, _summary(1.000))
        assert len(pairs) == 1
        assert pairs[0]["fast_idx"] == 2  # closer midpoint

    def test_outside_window_not_matched(self):
        pairs = []
        m = IncrementalPulseMatcher(window_s=0.05, on_pair=pairs.append)
        m.add("slow", 1, 1, _summary(1.0))
        m.add("fast", 1, 1, _summary(1.2))
        assert pairs == []  # both pending, neither expired yet

    def test_grace_expiry_emits_one_sided(self):
        pairs = []
        m = IncrementalPulseMatcher(window_s=0.05, grace_s=0.25,
                                    on_pair=pairs.append)
        m.add("slow", 1, 1, _summary(1.0))
        # Slow-stream time alone must NOT expire it — only the OTHER
        # stream passing the moment proves it was one-sided.
        m.add("slow", 1, 2, _summary(2.0))
        assert pairs == []
        m.advance_time("fast", 1.3)  # fast clock passes mid + grace
        assert len(pairs) == 1
        assert pairs[0]["slow_idx"] == 1
        assert pairs[0]["fast_idx"] is None
        assert m.unmatched == 1

    def test_flush_drains_everything(self):
        pairs = []
        m = IncrementalPulseMatcher(on_pair=pairs.append)
        m.add("slow", 1, 1, _summary(1.0))
        m.add("fast", 2, 1, _summary(1.0))  # different channel
        m.flush()
        assert len(pairs) == 2
        assert all((p["slow_idx"] is None) != (p["fast_idx"] is None)
                   for p in pairs)

    def test_channels_are_independent(self):
        pairs = []
        m = IncrementalPulseMatcher(window_s=0.05, on_pair=pairs.append)
        m.add("slow", 1, 1, _summary(1.0))
        m.add("fast", 2, 1, _summary(1.0))
        assert pairs == []  # ch1-slow must not pair with ch2-fast


# ───────────────────────── Dual session (synthetic) ─────────────────

SLOW_FS = 20_000.0
FAST_FS = 100_000.0


def _feed_noise(feed, n, rng):
    for _ in range(n):
        feed(1, float(rng.normal(0, 1.0)), float(rng.normal(0, 1.0)),
             None)


def _feed_span(feed, fs, t0, t1, rng, pulse_starts=(), amp=50.0,
               tau_s=1e-3):
    n = int((t1 - t0) * fs)
    t = t0 + np.arange(n) / fs
    sig = rng.normal(0, 1.0, n)
    for ps in pulse_starts:
        mask = t >= ps
        sig[mask] += amp * np.exp(-(t[mask] - ps) / tau_s)
    for i in range(n):
        feed(1, float(sig[i]), float(rng.normal(0, 1.0)), float(t[i]))


@requires_h5py
def test_dual_session_end_to_end(tmp_path):
    events = {"pairs": [], "pulses": [], "errors": []}
    # max_pulse_ms also sizes the rings — the fast ring must still
    # cover a one-sided pulse's window when it expires (grace period),
    # or the cross-stream TOD cannot be extracted.
    cfg = PulseCaptureConfig(threshold_sigma=5.0, end_sigma=1.5,
                             max_pulse_ms=200.0, noise_train_ms=10.0)
    path = tmp_path / "dual.h5"
    dual = DualPulseCaptureSession(
        channels=[1], slow_rate=SLOW_FS, fast_rate=FAST_FS,
        config=cfg, hdf5_path=path, match_grace_s=0.25,
        on_pair=lambda p: events["pairs"].append(p),
        on_pulse=lambda s, ch, idx, summ, _d:
            events["pulses"].append((s, ch, idx)),
        on_error=lambda m: events["errors"].append(m))

    rng = np.random.default_rng(3)
    dual.start()
    _feed_noise(dual.feed_slow, dual.slow.noise_samples + 10, rng)
    _feed_noise(dual.feed_fast, dual.fast.noise_samples + 10, rng)
    assert dual.slow.state is CaptureState.CAPTURING
    assert dual.fast.state is CaptureState.CAPTURING

    # One pulse at t=1.0 visible in BOTH streams (shared clock) …
    _feed_span(dual.feed_slow, SLOW_FS, 0.9, 1.4, rng,
               pulse_starts=[1.0])
    _feed_span(dual.feed_fast, FAST_FS, 0.9, 1.4, rng,
               pulse_starts=[1.0])
    # … then a slow-only pulse at t=2.0 (fast stream stays at baseline
    # but keeps flowing so its ring covers the window) …
    _feed_span(dual.feed_slow, SLOW_FS, 1.9, 2.4, rng,
               pulse_starts=[2.0])
    _feed_span(dual.feed_fast, FAST_FS, 1.9, 2.1, rng)
    # … and a second slow pulse far later to expire the one above.
    _feed_span(dual.feed_slow, SLOW_FS, 2.9, 3.4, rng,
               pulse_starts=[3.0])
    dual.stop()

    assert not events["errors"], events["errors"]
    assert ("slow", 1, 1) in events["pulses"]
    assert ("fast", 1, 1) in events["pulses"]

    matched = [p for p in events["pairs"]
               if p["slow_idx"] and p["fast_idx"]]
    one_sided = [p for p in events["pairs"]
                 if (p["slow_idx"] is None) != (p["fast_idx"] is None)]
    assert len(matched) == 1
    assert len(one_sided) == 2  # t=2.0 (expired) + t=3.0 (flushed)
    assert matched[0]["time_offset"] == pytest.approx(0.0, abs=0.05)

    # Union windows: BOTH streams span the widest trigger window —
    # the fast trace must not be a clipped slice of the slow one.
    m = matched[0]
    assert m.get("slow_tod") is not None
    assert m.get("fast_tod") is not None
    slow_dur = m["slow_summary"]["duration_s"]
    fast_t = np.asarray(m["fast_tod"]["Time"], float)
    fast_span = float(np.max(fast_t) - np.min(fast_t))
    assert fast_span >= 0.9 * slow_dur, \
        f"fast union window {fast_span*1e3:.2f} ms < slow trigger " \
        f"window {slow_dur*1e3:.2f} ms"

    # The expired t=2.0 pulse should carry a fast-stream TOD window
    expired = [p for p in one_sided
               if p["slow_summary"]
               and abs(p["slow_summary"]["timestamp"] - 2.0) < 0.05]
    assert expired and expired[0].get("fast_tod") is not None
    assert len(expired[0]["fast_tod"]["Amp_I"]) > 0

    stats = dual.stats()
    assert stats["pairs_matched"] == 1
    assert stats["pairs_unmatched"] == 2
    assert stats["slow"]["total_pulses"] == 3
    assert stats["fast"]["total_pulses"] == 1

    # ── Dual file round-trip ──────────────────────────────────────
    with PulseHDF5Reader(path) as reader:
        assert reader.dual
        assert reader.streams == ["slow", "fast"]
        assert reader.metadata["streamer_mode"] == "both"
        assert reader.pulse_count(1, "slow") == 3
        assert reader.pulse_count(1, "fast") == 1
        wf = reader.get_pulse(1, 1, stream="fast")
        assert wf is not None and np.isfinite(wf["snr"])
        assert reader.pair_count(1) == 3
        pairs = list(reader.iter_matches(1))
        assert sum(1 for p in pairs
                   if p["slow_idx"] and p["fast_idx"]) == 1
        assert any("fast_tod" in p for p in pairs)
        assert np.sum(reader.get_histograms("slow")
                      ["amplitude_counts_ch1"]) == 3
        ns = reader.noise_stats(1, "slow")
        assert ns.std_I > 0


class TestAdvanceTime:
    def test_other_stream_time_drives_expiry(self):
        pairs = []
        m = IncrementalPulseMatcher(window_s=0.05, grace_s=0.25,
                                    on_pair=pairs.append)
        m.add("slow", 1, 1, _summary(1.0))
        m.advance_time("fast", 1.2)
        assert pairs == []          # still within grace
        m.advance_time("fast", 1.3)
        assert len(pairs) == 1      # expired by stream time alone
        assert pairs[0]["fast_idx"] is None

    def test_own_stream_time_never_expires_own_pulses(self):
        """Stream skew safety: if the fast socket stalls, slow pulses
        must stay pending rather than being declared one-sided."""
        pairs = []
        m = IncrementalPulseMatcher(grace_s=0.25, on_pair=pairs.append)
        m.add("slow", 1, 1, _summary(1.0))
        m.advance_time("slow", 100.0)
        assert pairs == []
        m.add("fast", 1, 1, _summary(1.01))  # late partner still matches
        assert len(pairs) == 1
        assert pairs[0]["fast_idx"] == 1

    def test_advance_ignores_backwards_and_nan(self):
        pairs = []
        m = IncrementalPulseMatcher(grace_s=0.25, on_pair=pairs.append)
        m.add("slow", 1, 1, _summary(1.0))
        m.advance_time("fast", float("nan"))
        m.advance_time("fast", 0.5)
        assert pairs == []


@requires_h5py
def test_one_sided_emits_live_with_cross_tod(tmp_path):
    """A one-sided pulse must surface ~grace after the event from
    baseline stream time alone (no later pulse, no stop) — with the
    other stream's TOD window while its ring still covers it."""
    events = {"pairs": []}
    cfg = PulseCaptureConfig(threshold_sigma=5.0, end_sigma=1.5,
                             max_pulse_ms=200.0, noise_train_ms=10.0)
    dual = DualPulseCaptureSession(
        channels=[1], slow_rate=SLOW_FS, fast_rate=FAST_FS,
        config=cfg, hdf5_path=None, match_grace_s=0.25,
        on_pair=lambda p: events["pairs"].append(p))
    rng = np.random.default_rng(5)
    dual.start()
    _feed_noise(dual.feed_slow, dual.slow.noise_samples + 10, rng)
    _feed_noise(dual.feed_fast, dual.fast.noise_samples + 10, rng)

    # Slow-only pulse at t=1.0; then BASELINE ONLY on both streams
    _feed_span(dual.feed_slow, SLOW_FS, 0.9, 1.2, rng,
               pulse_starts=[1.0])
    _feed_span(dual.feed_fast, FAST_FS, 0.9, 1.2, rng)
    assert events["pairs"] == []  # within grace so far

    _feed_span(dual.feed_slow, SLOW_FS, 1.2, 1.5, rng)
    _feed_span(dual.feed_fast, FAST_FS, 1.2, 1.5, rng)

    assert len(events["pairs"]) == 1, \
        "one-sided pair must emit from stream time, not at stop"
    pair = events["pairs"][0]
    assert pair["slow_idx"] == 1 and pair["fast_idx"] is None
    assert pair.get("fast_tod") is not None, \
        "cross-stream TOD must be captured while the ring covers it"
    assert len(pair["fast_tod"]["Amp_I"]) > 0
    dual.stop()


def test_streams_start_capturing_together():
    """The fast stream reaches its training target far sooner (more
    samples per second, and a cap on top), so left alone it triggers
    into a partner with no ring yet and every pair comes out one-sided
    with 'window unavailable'."""
    import numpy as np
    from rfmux.algorithms.measurement.pulse_capture_session import (
        CaptureState)

    dual = DualPulseCaptureSession(
        channels=[1], slow_rate=1000.0, fast_rate=10000.0,
        config=PulseCaptureConfig(threshold_sigma=5.0, end_sigma=1.5,
                                  max_pulse_ms=20.0, noise_train_ms=100.0))
    dual.start()
    rng = np.random.default_rng(0)

    # Feed the fast stream alone until it finishes training.
    k = 0
    while dual.fast.state is not CaptureState.CAPTURING and k < 200_000:
        dual.feed_fast(1, float(rng.normal(0, 1)), float(rng.normal(0, 1)),
                       k * 1e-4)
        k += 1
    assert dual.fast.state is CaptureState.CAPTURING
    assert dual.slow.state is CaptureState.ESTIMATING
    assert dual.fast.pcap.freeze_triggers, \
        "fast must hold until its partner has a ring"

    # Now bring the slow stream up; both should be live afterwards.
    j = 0
    while dual.slow.state is not CaptureState.CAPTURING and j < 200_000:
        dual.feed_slow(1, float(rng.normal(0, 1)), float(rng.normal(0, 1)),
                       j * 1e-3)
        j += 1
    assert dual.slow.state is CaptureState.CAPTURING
    assert not dual.fast.pcap.freeze_triggers
    assert not dual.slow.pcap.freeze_triggers
    dual.stop()


def test_stream_feeds_present_the_source_facade():
    """run_slow_source/run_pfb_source read ``channels`` and call
    ``feed_sample``; the dual session's facades must satisfy exactly
    that, and must route through feed_slow/feed_fast so stream time
    advances the matcher.  Only the socket sources exercise this path,
    and those need the acquisition tier — so pin the contract here."""
    dual = DualPulseCaptureSession(
        channels=[1, 2], slow_rate=1000.0, fast_rate=10000.0,
        config=PulseCaptureConfig(max_pulse_ms=20.0, noise_train_ms=100.0))
    dual.start()

    for feed, session in ((dual.slow_feed, dual.slow),
                          (dual.fast_feed, dual.fast)):
        assert feed.channels == [1, 2]
        before = session._noise_n[1]
        feed.feed_sample(1, 0.5, -0.5, 1.0)
        assert session._noise_n[1] == before + 1, \
            "the facade must reach the underlying session"

    # ...and through feed_slow/feed_fast, so the matcher clock moved.
    assert dual._last_advance["slow"] == 1.0
    assert dual._last_advance["fast"] == 1.0
    dual.stop()
