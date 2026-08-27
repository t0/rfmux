"""
Histograms and templates must keep up with a quiet capture.

The pulse-count trigger alone means a low count rate shows nothing
until the 50th pulse arrives, which on a real detector can be minutes.
A clock trigger bounds the wait without letting a busy capture flush on
every pulse.
"""

import numpy as np

from rfmux.pulse_capture.session import (
    CaptureState, PulseCaptureSession)

FS = 1000.0
NOISE = 300


def _session(hist_events, tmpl_events, **kwargs):
    return PulseCaptureSession(
        channels=[1], sample_rate=FS, noise_samples=NOISE, hdf5_path=None,
        on_histograms=lambda d: hist_events.append(d),
        on_templates=lambda d: tmpl_events.append(d),
        **kwargs)


def _train_and_pulse(session, n_pulses, rng):
    """Train, then feed `n_pulses` well-separated pulses."""
    noise = rng.normal(0, 1.0, NOISE)
    session.feed_block(1, noise, noise, np.arange(NOISE) / FS)
    assert session.state is CaptureState.CAPTURING

    t = NOISE / FS
    span = 120
    for _ in range(n_pulses):
        k = np.arange(span)
        shape = 80.0 * np.exp(-k / 10.0)
        i_vals = shape + rng.normal(0, 1.0, span)
        q_vals = rng.normal(0, 1.0, span)
        stamps = t + k / FS
        session.feed_block(1, i_vals, q_vals, stamps)
        t += span / FS


def test_clock_trigger_flushes_before_the_50th_pulse():
    rng = np.random.default_rng(1)
    hist, tmpl = [], []
    # interval 0 => every pulse is overdue, which is the low-rate limit.
    s = _session(hist, tmpl, histogram_flush_every=50,
                 histogram_flush_interval_s=0.0)
    s.start()
    _train_and_pulse(s, 3, rng)

    assert hist, "no histogram update before the pulse-count threshold"
    assert tmpl, "no template update before the pulse-count threshold"
    s.stop()


def test_pulse_count_still_bounds_a_busy_capture():
    rng = np.random.default_rng(1)
    hist, tmpl = [], []
    # A clock that never fires: only the count may trigger a flush.
    s = _session(hist, tmpl, histogram_flush_every=50,
                 histogram_flush_interval_s=1e9)
    s.start()
    _train_and_pulse(s, 5, rng)

    assert hist == [], "flushed on the clock when it should not have"
    s.stop()
    # stop() always flushes, so nothing is lost either way.
    assert hist, "stop() must still flush what was accumulated"


def test_default_interval_is_live_enough_to_watch():
    # Not a behavioural assertion so much as a guard on the default:
    # half a second is the slowest a live view should feel.
    s = PulseCaptureSession(channels=[1], sample_rate=FS, hdf5_path=None)
    assert 0 < s.histogram_flush_interval_s <= 0.5


def test_flush_does_not_fire_with_nothing_to_report():
    rng = np.random.default_rng(2)
    hist, tmpl = [], []
    s = _session(hist, tmpl, histogram_flush_every=50,
                 histogram_flush_interval_s=0.0)
    s.start()
    noise = rng.normal(0, 1.0, NOISE)
    s.feed_block(1, noise, noise, np.arange(NOISE) / FS)
    # Training finished but no pulse has been seen: an overdue clock
    # must not manufacture empty updates.
    assert hist == []
    s.stop()
