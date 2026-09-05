"""What the Periscope capture worker owes its callers: requests reach
the session, the tail handed over at stop is captured, a capture that
cannot run never opens a file, and losses are reported."""

import asyncio
import queue
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from rfmux.pulse_capture.capture_session import (
    CaptureState,
    DualPulseCaptureSession,
    PulseCaptureConfig,
    PulseCaptureSession,
)
from rfmux.pulse_capture.detection import ChannelNoiseStats
from rfmux.tools.periscope.pulse_capture_task import PulseCaptureTask


class _Recorder:
    """Stands in for PulseCaptureSignals: every emit is recorded by
    signal name."""

    def __init__(self):
        self.emitted = defaultdict(list)

    def __getattr__(self, name):
        sink = self.emitted[name]
        return SimpleNamespace(emit=lambda *args: sink.append(args))


def _bare_task(session, mode="slow"):
    """A PulseCaptureTask stripped to what the worker-side methods
    touch, no Qt thread behind it."""
    t = PulseCaptureTask.__new__(PulseCaptureTask)
    t.session = session
    t.signals = _Recorder()
    t.mode = mode
    t.sample_queue = queue.Queue()
    return t


def _capturing_dual():
    cfg = PulseCaptureConfig(noise_train_ms=50.0)
    d = DualPulseCaptureSession(
        channels=[1], module=1, slow_rate=1000.0, fast_rate=10000.0,
        config=cfg, hdf5_path=None)
    d.start()
    rng = np.random.default_rng(3)
    for feed, inner, fs in ((d.feed_slow_block, d.slow, 1000.0),
                            (d.feed_fast_block, d.fast, 10000.0)):
        n = inner.noise_samples + 10
        t = 1000.0 + np.arange(n) / fs
        feed(1, rng.normal(size=n), rng.normal(size=n), t)
    assert d.state == {"slow": "capturing", "fast": "capturing"}, d.state
    return d


def test_reestimate_reaches_both_streams_of_a_dual_session():
    d = _capturing_dual()
    t = _bare_task(d, mode="both")
    try:
        assert t._handle_control(("__reestimate__",))
        assert d.slow.state is CaptureState.ESTIMATING
        assert d.fast.state is CaptureState.ESTIMATING
        assert t.signals.emitted["error"] == []
    finally:
        d.stop()


def test_reestimate_outside_capturing_is_reported_not_dropped():
    s = PulseCaptureSession(channels=[1], noise_samples=100, hdf5_path=None)
    s.start()                      # ESTIMATING: nothing to re-estimate yet
    t = _bare_task(s)
    try:
        t._handle_control(("__reestimate__",))
        (message,), = t.signals.emitted["error"]
        assert "ESTIMATING" in message
    finally:
        s.stop()


def test_stop_drains_what_the_tap_handed_over():
    """request_stop puts the gathered tail on the queue and then raises
    the flag; a worker that was mid-item must still feed it."""
    fed = []
    t = _bare_task(SimpleNamespace(flush_progress=lambda: None))
    for k in range(2):
        rows = np.full((3, 1), complex(k + 1, 0))
        stamps = 1000.0 + k * 0.003 + np.arange(3) / 1000.0
        t.sample_queue.put_nowait(((1,), [rows], [stamps]))

    def feed(ch, i_vals, q_vals, stamps):
        fed.extend(i_vals.tolist())

    asyncio.run(t._slow_tap_pump(lambda: True, feed))
    assert fed == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]


def test_a_failed_streamer_check_does_not_start_the_session():
    """A dual session opens its HDF5 file on start(); a capture the
    board cannot feed must fail before that, or it replaces the
    previous file with an empty one."""
    started = []

    class _CRS:
        async def get_pfb_streamer(self, module=1):
            return None

    session = SimpleNamespace(channels=[1], start=lambda: started.append(1),
                              stop=lambda: None)
    t = _bare_task(session, mode="fast")
    t.crs, t.module, t.host = _CRS(), 1, "127.0.0.1"
    t._stop_requested = False
    t.isInterruptionRequested = lambda: False

    t.run()
    assert started == []
    assert t.signals.emitted["failed"], "the refusal must be reported"
    assert t.signals.emitted["finished"] == [()]


def test_tap_queue_overflow_is_reported_once(qt_app):
    signals = _Recorder()
    t = PulseCaptureTask(SimpleNamespace(), signals, queue_size=1)
    t._send_control(("__day__", 1.0))          # the one slot
    for _ in range(3):
        t.enqueue_packet((1,), np.array([1 + 0j]), 0.0)
        t.flush_tap()
    (message,), = signals.emitted["error"]
    assert "dropped" in message


@pytest.mark.parametrize("mode", ["slow", "both"])
def test_noise_estimate_payload_is_a_snapshot(qt_app, mode):
    """The engine re-centres the live stats after every baseline
    refresh; the GUI must keep the estimate it was told."""
    signals = _Recorder()
    inner = SimpleNamespace(streamer_mode="slow")
    session = SimpleNamespace(slow=inner, fast=inner)
    PulseCaptureTask(session, signals, mode=mode)

    live = {1: ChannelNoiseStats(mean_I=1.0)}
    if mode == "both":
        session.on_noise("slow", live)
        (payload,), = signals.emitted["noise_estimated"]
        payload = payload["stats"]
    else:
        session.on_noise(live)
        (payload,), = signals.emitted["noise_estimated"]

    live[1].mean_I = 5.0
    assert payload[1].mean_I == 1.0
