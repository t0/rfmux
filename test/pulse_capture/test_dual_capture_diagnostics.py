"""Dual (both-mode) capture: what a capture that cannot work reports,
how pair windows are taken as each ring catches up, and what a pair
carries."""

import asyncio
import contextlib
import socket

import numpy as np
import pytest

from rfmux import streamer
from rfmux.pulse_capture.sources import run_pfb_source
from test.qt_helpers import spin


def _private_pfb_socket(monkeypatch):
    """Point run_pfb_source at a private loopback socket nothing sends
    to, so the real PFB multicast group cannot leak packets in."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))

    @contextlib.contextmanager
    def fake(host, port=None, **kw):
        yield sock
    monkeypatch.setattr(streamer, "get_multicast_socket", fake)
    return sock


class _NeverFed:
    channels = [1]

    def feed_block(self, ch, i_vals, q_vals, timestamps):
        raise AssertionError("a silent socket must feed nothing")


def test_silent_pfb_socket_is_an_error(monkeypatch):
    """Zero packets ever is a configuration problem, not an empty
    capture: returning 0.0 would end a dual capture as though it had
    been stopped."""
    monkeypatch.setattr(streamer, "STREAMER_TIMEOUT", 0.2)
    _private_pfb_socket(monkeypatch)
    with pytest.raises(TimeoutError, match="fast streamer is not sending"):
        asyncio.run(run_pfb_source(_NeverFed(), "127.0.0.1", [1]))


class _Signals:
    def __init__(self, sink):
        class _E:
            def emit(_self, msg):
                sink.append(msg)
        self.error = _E()


def _watchdog_task(states):
    """A PulseCaptureTask stripped to what the watchdog touches."""
    from types import SimpleNamespace

    from rfmux.tools.periscope.pulse_capture_task import PulseCaptureTask

    t = PulseCaptureTask.__new__(PulseCaptureTask)
    errors = []
    t.signals = _Signals(errors)
    t.session = SimpleNamespace(state=states)
    return t, errors


def test_watchdog_names_a_stream_still_training_and_nothing_else(monkeypatch):
    async def fast_sleep(_s):
        pass
    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    t, errors = _watchdog_task({"slow": "estimating", "fast": "capturing"})
    ticks = {"n": 0}

    def stop():
        ticks["n"] += 1
        return ticks["n"] > 2

    asyncio.run(t._dual_watchdog(stop))
    assert errors, "a stuck stream produced no warning"
    assert "slow stream" in errors[0]
    assert "no stream may trigger" in errors[0]

    t, errors = _watchdog_task({"slow": "capturing", "fast": "capturing"})
    asyncio.run(t._dual_watchdog(lambda: False))  # returns on its own
    assert errors == []


def test_dual_streams_receive_the_calibrations():
    """trigger_basis="df" in a dual capture rotates both streams:
    stored_units is "Hz" when the rotation happened, "V" when not."""
    from rfmux.pulse_capture.capture_session import (
        DualPulseCaptureSession, PulseCaptureConfig)

    cfg = PulseCaptureConfig(trigger_basis="df", noise_train_ms=50.0)
    d = DualPulseCaptureSession(
        channels=[1], module=1, slow_rate=1000.0, fast_rate=10000.0,
        config=cfg, hdf5_path=None,
        df_calibrations={1: 2.0e6 + 0j})
    d.start()
    t = np.arange(50) / 1000.0
    z = np.zeros(50)
    d.feed_slow_block(1, z, z, 1000.0 + t)
    d.feed_fast_block(1, z, z, 1000.0 + t / 10.0)
    d.stop()

    assert d.slow.trigger_basis == "df"
    assert d.slow.stored_units.get(1) == "Hz", d.slow.stored_units
    assert d.fast.stored_units.get(1) == "Hz", d.fast.stored_units


def _task_with_pfb(active):
    """A PulseCaptureTask stripped to the streamer check."""
    from rfmux.tools.periscope.pulse_capture_task import PulseCaptureTask

    calls = []

    class _CRS:
        async def get_pfb_streamer(self, module=1):
            return active

        async def set_pfb_streamer(self, channel=None, module=1):
            calls.append(channel)

    t = PulseCaptureTask.__new__(PulseCaptureTask)
    t.crs = _CRS()
    t.module = 2
    t.host = "127.0.0.1"
    t._stop_requested = False
    t.isInterruptionRequested = lambda: False
    errors = []
    t.signals = _Signals(errors)
    t.signals.failed = t.signals.error
    return t, errors, calls


@pytest.mark.parametrize("active, wanted, ok", [
    ([1, 2], [1, 2], True), ([2, 1], [1, 2], True), (None, [1, 2], False),
    ([], [1, 2], False), ([1], [1, 2], False), ([1, 2, 3], [1, 2], False),
    (1, [1], True), (2, [1], False)])
def test_capture_uses_the_streamer_as_configured(active, wanted, ok):
    """The capture reads what the board streams and never sets it: the
    streamed channel set must be the captured set, in any order.  The
    board reports a single channel as a bare integer."""
    t, errors, calls = _task_with_pfb(active)
    problem = asyncio.run(t._pfb_mismatch(wanted))
    assert (problem is None) is ok
    if not ok:
        assert "Streamer Configuration" in problem
    assert calls == []


def test_an_unreadable_streamer_report_is_named():
    t, errors, calls = _task_with_pfb(object())
    problem = asyncio.run(t._pfb_mismatch([1]))
    assert problem and "get_pfb_streamer" in problem


def test_a_mode_the_streamer_cannot_feed_fails_before_running(monkeypatch):
    from types import SimpleNamespace
    from rfmux.pulse_capture import sources
    t, errors, calls = _task_with_pfb(None)
    t.session = SimpleNamespace(channels=[1])

    async def never(*a, **k):
        raise AssertionError("must not start a source")
    monkeypatch.setattr(sources, "run_dual_source", never)
    asyncio.run(t._run_both())
    assert errors and "off" in errors[0]
    assert calls == []


def test_stream_lag_is_reported_in_stats():
    from rfmux.pulse_capture.capture_session import (
        DualPulseCaptureSession, PulseCaptureConfig)

    cfg = PulseCaptureConfig(threshold_sigma=5.0, noise_train_ms=50.0,
                             max_pulse_ms=50.0)
    d = DualPulseCaptureSession(channels=[1], module=1, slow_rate=1000.0,
                                fast_rate=100000.0, config=cfg, hdf5_path=None)
    d.start()
    t = 43000.0
    # Slow fed to t+2.0, fast only to t+0.5: fast trails by 1.5 s.
    d.feed_slow_block(1, np.zeros(2000), np.zeros(2000), t + np.arange(2000) / 1000.0)
    d.feed_fast_block(1, np.zeros(50000), np.zeros(50000), t + np.arange(50000) / 100000.0)
    d.stop()

    st = d.stats()
    assert st["stream_lag_s"] == pytest.approx(1.5, abs=0.05)
    assert st["ring_overlap_s"] is not None and st["ring_overlap_s"] > 0


def test_status_line_warns_on_lag(qt_app):
    """Green inside half the ring overlap, a warning colour past it, a
    stronger one past the overlap, and the note reaches the status line."""
    from rfmux.tools.periscope.pulse_capture_panel import PulseCapturePanel

    panel = PulseCapturePanel(dark_mode=False)
    panel._both_mode = True
    ok, tip = panel._stream_lag_signal({"stream_lag_s": 0.02, "ring_overlap_s": 0.4})
    assert tip is None
    amber, tip = panel._stream_lag_signal({"stream_lag_s": 0.25, "ring_overlap_s": 0.4})
    assert amber != ok and "fast stream" in tip[0]
    red, tip = panel._stream_lag_signal({"stream_lag_s": 3.5, "ring_overlap_s": 0.4})
    assert red not in (ok, amber)

    panel._last_stats = {"pairs_matched": 0, "pairs_unmatched": 8,
                         "slow": {"total_pulses": 2}, "fast": {"total_pulses": 6},
                         "stream_lag_s": 3.5, "ring_overlap_s": 0.4}
    panel._refresh_status_line()
    assert "behind" in panel.status_label.text()
    assert panel.status_label.toolTip()
    panel.close()
    spin(qt_app)


def _dual_for_deferral():
    from rfmux.pulse_capture.capture_session import (
        DualPulseCaptureSession, PulseCaptureConfig)
    cfg = PulseCaptureConfig(threshold_sigma=5.0, end_sigma=1.5,
                             max_pulse_ms=50.0, noise_train_ms=50.0)
    pairs = []
    d = DualPulseCaptureSession(
        channels=[1], module=1, slow_rate=1000.0, fast_rate=100000.0,
        config=cfg, hdf5_path=None, pair_window_wait_s=3.0,
        on_pair=lambda p: pairs.append(p), on_error=lambda m: None)
    d.start()
    return d, pairs


def _feed_to(d, stream, t_from, t_to, fs):
    n = int(round((t_to - t_from) * fs))
    t = t_from + np.arange(n) / fs
    z = np.zeros(n)
    (d.feed_slow_block if stream == "slow" else d.feed_fast_block)(1, z, z, t)


def _pair(idx, t_trig, T0=43000.0, fast_core=0.05):
    return {"channel": 1, "pair_idx": idx, "slow_idx": idx, "fast_idx": idx,
            "slow_summary": {"timestamp": T0 + t_trig, "duration_s": 0.05},
            "fast_summary": {"timestamp": T0 + t_trig, "duration_s": fast_core},
            "time_offset": 0.0}


def test_each_window_is_taken_when_its_own_ring_covers_it():
    """The pair forms while the slow ring is behind the window's end:
    the fast window is taken at once, the slow one when its ring reaches
    the window's end, and the pair goes out then."""
    d, pairs = _dual_for_deferral()
    T0 = 43000.0
    _feed_to(d, "fast", T0, T0 + 1.00, 100000.0)
    _feed_to(d, "slow", T0, T0 + 0.60, 1000.0)
    # Union window ends at T0+0.80: past the slow ring, inside the fast.
    pair = _pair(1, 0.70, fast_core=0.10)
    d._on_matcher_pair(pair)

    assert pairs == []
    assert pair.get("fast_tod") is not None, "fast window should be taken now"
    assert "slow_tod" not in pair, "slow window must wait for its ring"

    _feed_to(d, "slow", T0 + 0.60, T0 + 0.90, 1000.0)
    assert len(pairs) == 1
    st = pairs[0]["slow_tod"]
    assert st is not None
    assert float(np.nanmax(st["Time"])) >= T0 + 0.80 - 0.002, \
        "slow window stops short of the window end"
    d.stop()


def test_pairs_are_emitted_in_order_behind_a_waiting_one():
    d, pairs = _dual_for_deferral()
    T0 = 43000.0
    _feed_to(d, "fast", T0, T0 + 1.00, 100000.0)
    _feed_to(d, "slow", T0, T0 + 0.60, 1000.0)
    d._on_matcher_pair(_pair(1, 0.70))      # waits on the slow ring
    d._on_matcher_pair(_pair(2, 0.30))      # both rings cover it
    assert pairs == [], "the ready pair must not overtake the waiting one"
    _feed_to(d, "slow", T0 + 0.60, T0 + 0.90, 1000.0)
    assert [p["pair_idx"] for p in pairs] == [1, 2]
    d.stop()


def test_a_stream_that_never_catches_up_does_not_strand_the_pair():
    """Past pair_window_wait_s of the other stream's time, the pair goes
    out with what there is."""
    d, pairs = _dual_for_deferral()
    d._pair_window_wait_s = 0.2
    T0 = 43000.0
    _feed_to(d, "fast", T0, T0 + 0.85, 100000.0)    # 0.85 < 0.755 + 0.2
    _feed_to(d, "slow", T0, T0 + 0.60, 1000.0)
    d._on_matcher_pair(_pair(1, 0.70))
    assert pairs == []
    _feed_to(d, "fast", T0 + 1.00, T0 + 1.20, 100000.0)   # slow never comes
    assert len(pairs) == 1, "pair stranded behind a stream that never came"
    assert pairs[0].get("slow_tod") is None \
        or float(np.nanmax(pairs[0]["slow_tod"]["Time"])) < T0 + 0.75
    d.stop()


def test_pfb_batch_yields_to_the_loop(monkeypatch):
    """A long PFB batch turns the loop over every _PFB_YIELD_EVERY
    packets, so a capture walk cannot hold it for a whole batch."""
    import time

    from rfmux.pulse_capture import sources as src

    sock = _private_pfb_socket(monkeypatch)
    monkeypatch.setattr(streamer, "STREAMER_TIMEOUT", 0.3)
    send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = sock.getsockname()[1]
    pkt = streamer.PFBPacket()
    pkt.magic = streamer.PFB_PACKET_MAGIC
    pkt.num_samples = 100
    blob = bytes(pkt)
    N = 40
    for _ in range(N):
        send.sendto(blob, ("127.0.0.1", port))
    monkeypatch.setattr(src, "_flush", lambda s: None)
    time.sleep(0.05)

    yields = {"n": 0}
    real_sleep = asyncio.sleep

    async def counting_sleep(d):
        if d == 0:
            yields["n"] += 1
        return await real_sleep(d)
    monkeypatch.setattr(asyncio, "sleep", counting_sleep)

    class _Sink:
        channels = [1]
        fed = 0

        def feed_block(self, ch, i, q, t):
            _Sink.fed += 1

    asyncio.run(src.run_pfb_source(
        _Sink(), "127.0.0.1", [1],
        should_stop=lambda: _Sink.fed >= N))
    send.close()
    # 40 packets in one drain: yields at k=8,16,24,32 plus one after.
    assert yields["n"] >= 4, f"only {yields['n']} yields for {N} packets"


def test_union_window_spans_the_saved_record():
    """The pair's window covers the saved record, not the core, and a
    summary without saved_end_time still gets the core."""
    from rfmux.pulse_capture.capture_session import DualPulseCaptureSession as D
    T = 43000.0
    pair = {"slow_summary": {"timestamp": T, "start_time": T,
                             "trigger_time": T + 0.002, "duration_s": 0.004,
                             "saved_end_time": T + 0.016},
            "fast_summary": None}
    t0, t1 = D._union_window(pair)
    assert t1 >= T + 0.016 and t0 <= T

    old = {"slow_summary": {"timestamp": T, "duration_s": 0.004},
           "fast_summary": None}
    t0, t1 = D._union_window(old)
    assert t1 == pytest.approx(T + 0.004 + 0.0004, abs=1e-6)


def test_matcher_pairs_on_the_trigger_instant():
    """Two records of one event start at different pre-margins and have
    different core lengths; the matcher pairs on the trigger, and the
    offset it reports is trigger to trigger."""
    from rfmux.pulse_capture.capture_session import IncrementalPulseMatcher
    pairs = []
    m = IncrementalPulseMatcher(window_s=0.05, grace_s=0.25,
                                on_pair=lambda p: pairs.append(p))
    T = 43000.0
    # Record starts 115 ms apart, outside the 50 ms window; one trigger.
    m.add("slow", 1, 1, {"timestamp": T - 0.005, "trigger_time": T,
                         "duration_s": 0.010})
    m.add("fast", 1, 1, {"timestamp": T - 0.120, "trigger_time": T - 0.0016,
                         "duration_s": 0.002})
    assert m.matched == 1 and pairs and pairs[0]["slow_idx"] == 1 \
        and pairs[0]["fast_idx"] == 1, "one event, two records, no match"
    assert pairs[0]["time_offset"] == pytest.approx(0.0016, abs=1e-9)


def test_band_pair_is_one_legend_entry_that_hides_both(qt_app):
    """Each +/- band is a single item, so its legend entry hides both
    lines and no unnamed twin is left behind."""
    from rfmux.pulse_capture.detection import ChannelNoiseStats
    from rfmux.tools.periscope.pulse_capture_panel import PulseCapturePanel

    panel = PulseCapturePanel(dark_mode=False)
    plot = panel.pulse_plot_i
    ns = ChannelNoiseStats(mean_I=10.0, std_I=2.0, mean_Q=0.0, std_Q=1.0)
    panel.threshold_spin.setValue(5.0)
    panel.end_spin.setValue(1.5)
    panel._annotate_noise_bands(plot, "I", ns, 0.0, 1.0, "#888888")

    named = {}
    for item in plot.getPlotItem().listDataItems():
        if item.name():
            named.setdefault(item.name(), []).append(item)
    trig = named["±5σ trigger"]
    assert len(trig) == 1
    y = np.asarray(trig[0].yData, float)
    levels = set(np.round(y[np.isfinite(y)], 6))
    assert levels == {20.0, 0.0}, levels
    for item in plot.getPlotItem().listDataItems():
        if not item.name():
            yy = np.asarray(item.yData, float)
            assert not (set(np.round(yy[np.isfinite(yy)], 6)) & levels)
    panel.close()
    spin(qt_app)


def _decision_labels(plot):
    import pyqtgraph as pg
    return [item.label.format for item in plot.getPlotItem().items
            if isinstance(item, pg.InfiniteLine) and item.label is not None]


def test_end_confirmed_mark_follows_the_full_tail_setting(qt_app):
    """No confirmation mark when the tail was not saved to it."""
    from rfmux.tools.periscope.pulse_capture_panel import PulseCapturePanel

    panel = PulseCapturePanel(dark_mode=False)
    T = 43000.0
    wf = {"Time": T + np.arange(20) / 1000.0,
          "trigger_index": 2, "trigger_time": T + 0.002,
          "below_threshold_index": 6, "below_threshold_time": T + 0.006,
          "end_index": 40, "end_time": T + 0.040}   # past the data

    panel.capture_config.save_to_end_confirmed = False
    panel.pulse_plot_i.clear()
    panel._annotate_decisions(panel.pulse_plot_i, wf, T, "I")
    labels = _decision_labels(panel.pulse_plot_i)
    assert any("trigger" in l for l in labels)
    assert any("below threshold" in l for l in labels)
    assert not any("end confirmed" in l for l in labels), labels

    panel.capture_config.save_to_end_confirmed = True
    panel.pulse_plot_i.clear()
    panel._annotate_decisions(panel.pulse_plot_i, wf, T, "I")
    assert any("end confirmed" in l for l in _decision_labels(panel.pulse_plot_i))
    panel.close()
    spin(qt_app)


def test_review_mode_restores_the_full_tail_setting(qt_app, tmp_path):
    """An opened file sets the policy its records were made under."""
    from rfmux.pulse_capture.detection import ChannelNoiseStats
    from rfmux.pulse_capture.hdf5 import PulseHDF5Writer
    from rfmux.tools.periscope.pulse_capture_panel import PulseCapturePanel

    path = tmp_path / "tail_off.h5"
    PulseHDF5Writer(path, [1], {1: ChannelNoiseStats()},
                    {"streamer_mode": "slow", "sample_rate": 596.0,
                     "save_to_end_confirmed": False}).finalize()
    panel = PulseCapturePanel(dark_mode=False)
    panel.capture_config.save_to_end_confirmed = True
    panel.load_from_hdf5(path)
    assert panel._saves_full_tail() is False
    panel.close()
    spin(qt_app)


def test_dual_stats_carry_the_median_skew():
    d, pairs = _dual_for_deferral()
    T0 = 43000.0
    _feed_to(d, "fast", T0, T0 + 1.0, 100000.0)
    _feed_to(d, "slow", T0, T0 + 1.0, 1000.0)
    for k, off in enumerate((0.0010, 0.0016, 0.0030)):
        p = _pair(k + 1, 0.3)
        for key in ("slow_summary", "fast_summary"):
            p[key].update(duration_s=0.01, start_time=T0 + 0.3,
                          saved_end_time=T0 + 0.32)
        p["time_offset"] = off
        d._on_matcher_pair(p)
    st = d.stats()
    assert st["stream_skew_n"] == 3
    assert st["stream_skew_s"] == pytest.approx(0.0016)
    d.stop()


def test_both_mode_status_counts_samples_dropped_per_stream(qt_app):
    """A stream whose packets carry no usable timestamp loses every
    sample; the status line says which stream and how many."""
    from rfmux.tools.periscope.pulse_capture_panel import PulseCapturePanel
    panel = PulseCapturePanel(dark_mode=False)
    panel._both_mode = True
    panel._last_stats = {"pairs_matched": 0, "pairs_unmatched": 0,
                         "slow": {"total_pulses": 1, "dropped_invalid_ts": 0},
                         "fast": {"total_pulses": 0, "dropped_invalid_ts": 24000}}
    panel._refresh_status_line()
    assert "fast 24000 dropped (no timestamp)" in panel.status_label.text()
    assert "slow 0" not in panel.status_label.text()
    panel.close()
    spin(qt_app)
