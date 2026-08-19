"""
The GUI packet drain must stop at a deadline.

Draining "until the queue is empty" is only bounded while processing
outruns arrival.  A wide pulse capture inverts that, and the old
unbounded loop then never returned to the Qt event loop -- a frozen
window rather than dropped packets.
"""

import os
import threading
import time
from types import SimpleNamespace

import pytest


pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from rfmux.tools.periscope.app import Periscope  # noqa: E402

REFRESH_MS = 33
#: Must exceed PeriscopeRuntime._DRAIN_DEADLINE_S; the point of the
#: backstop is that it is generous, not tight.
DEADLINE_S = 0.25



class _NeverEmptyQueue:
    """A stream arriving faster than the GUI can process it."""

    def __init__(self):
        self.pops = 0

    def empty(self):
        return False

    def try_pop(self):
        self.pops += 1
        return SimpleNamespace(to_python=lambda: SimpleNamespace())


class _FiniteQueue:
    def __init__(self, n):
        self.remaining = n

    def empty(self):
        return self.remaining <= 0

    def try_pop(self):
        if self.remaining <= 0:
            return None
        self.remaining -= 1
        return SimpleNamespace(to_python=lambda: SimpleNamespace())


def _runtime(qt_app, queue, per_packet_s=0.0):
    p = Periscope.__new__(Periscope)
    QtWidgets.QMainWindow.__init__(p)
    p.refresh_ms = REFRESH_MS
    p.drain_overruns = 0
    p.pkt_cnt = 0
    p.is_mock_mode = False
    p.receiver = SimpleNamespace(queue=queue)
    p._calculate_relative_timestamp = lambda pkt: 0.0
    p.processed = 0
    # The drain flushes the display batch on its way out.
    p._display_values = []
    p._display_times = []
    p._display_width = -1
    p._pulse_tap_frame_end = None
    p.all_chs = []
    p.buf = {}
    p.tbuf = {}

    def _update_buffers(pkt, t_rel):
        p.processed += 1
        if per_packet_s:
            time.sleep(per_packet_s)

    p._update_buffers = _update_buffers
    return p


def _drain(p, timeout=5.0):
    """Run one drain pass, failing rather than hanging on a regression."""
    done = threading.Event()

    def run():
        p._process_incoming_packets()
        done.set()

    t0 = time.monotonic()
    threading.Thread(target=run, daemon=True).start()
    returned = done.wait(timeout=timeout)
    assert returned, "drain did not return — the loop is unbounded again"
    return time.monotonic() - t0


def test_drain_returns_under_sustained_overload(qt_app):
    # 1 ms/packet against a queue that never empties: the unbounded loop
    # ran forever here.  Driven from a worker thread with a join
    # timeout so a regression fails the test instead of hanging the
    # whole suite.
    p = _runtime(qt_app, _NeverEmptyQueue(), per_packet_s=0.001)

    elapsed = _drain(p)

    assert p.drain_overruns == 1, "overrun should be counted"
    # Bounded by the backstop, with room for a packet of overshoot and
    # scheduling noise.
    assert elapsed < DEADLINE_S * 2, f"took {elapsed*1e3:.1f} ms"
    assert p.processed > 0, "should still make progress each frame"


def test_backstop_is_not_a_throughput_budget(qt_app):
    # A frame's worth of packets must never be cut short.  At stage 0
    # that is ~1270 packets per 33 ms frame; the drain has to swallow
    # them without tripping the backstop.
    p = _runtime(qt_app, _FiniteQueue(1270), per_packet_s=0.0)
    _drain(p)
    assert p.processed == 1270
    assert p.drain_overruns == 0, \
        "the backstop fired on an ordinary frame — it is too tight"


def test_drain_makes_progress_across_frames(qt_app):
    q = _NeverEmptyQueue()
    p = _runtime(qt_app, q, per_packet_s=0.001)
    for _ in range(3):
        _drain(p)
    assert p.drain_overruns == 3
    assert p.processed >= 3


def test_normal_load_drains_fully_without_overrun(qt_app):
    p = _runtime(qt_app, _FiniteQueue(25))
    p._process_incoming_packets()
    assert p.processed == 25
    assert p.drain_overruns == 0
