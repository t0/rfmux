"""A dual capture that cannot work says so, instead of counting quietly.

Both diagnostics come from the same live debugging session against real
hardware: a both-mode capture with the PFB streamer off dies silently at
the socket timeout, and one with it on can starve the slow stream so
badly (0.8 s of sample time over 300 s of wall time, measured) that
noise training never finishes and the trigger freeze never lifts.
Either way the user watches counters that can never produce a pair.
"""

import asyncio

import numpy as np
import pytest

from rfmux import streamer
from rfmux.pulse_capture.sources import run_pfb_source
from test.qt_helpers import spin


import contextlib
import socket


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
    """Zero packets ever is a configuration problem, not an empty capture.

    Returning 0.0 made run_dual_source treat it as one side finishing,
    which stopped the other side too: the capture ended as though the
    user had pressed stop, with nothing naming the silent socket.
    """
    monkeypatch.setattr(streamer, "STREAMER_TIMEOUT", 0.2)
    _private_pfb_socket(monkeypatch)
    with pytest.raises(TimeoutError, match="fast streamer is not sending"):
        asyncio.run(run_pfb_source(_NeverFed(), "127.0.0.1", [1]))


def test_a_stream_that_stops_mid_capture_still_returns(monkeypatch):
    """The error is for silence from the START; a stream that dies later
    keeps today's behaviour -- return what was covered."""
    monkeypatch.setattr(streamer, "STREAMER_TIMEOUT", 0.2)

    sent = {"n": 0}
    real_wait_for = asyncio.wait_for

    async def one_packet_then_silence(coro, timeout):
        if sent["n"] == 0:
            sent["n"] += 1
            coro.close()
            pkt = streamer.PFBPacket()
            pkt.magic = streamer.PFB_PACKET_MAGIC
            pkt.num_samples = 100
            return bytes(pkt)
        return await real_wait_for(coro, timeout)

    class _Sink:
        channels = [1]
        fed = 0

        def feed_block(self, ch, i_vals, q_vals, timestamps):
            _Sink.fed += 1

    _private_pfb_socket(monkeypatch)
    monkeypatch.setattr(asyncio, "wait_for", one_packet_then_silence)
    covered = asyncio.run(run_pfb_source(_Sink(), "127.0.0.1", [1]))
    assert covered > 0.0
    assert _Sink.fed == 1


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


def test_watchdog_names_the_stream_still_training(monkeypatch):
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


def test_watchdog_is_silent_once_both_capture(monkeypatch):
    async def fast_sleep(_s):
        pass
    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    t, errors = _watchdog_task({"slow": "capturing", "fast": "capturing"})
    asyncio.run(t._dual_watchdog(lambda: False))  # returns on its own
    assert errors == []


def test_dual_streams_receive_the_calibrations():
    """trigger_basis="df" in a dual capture rotates, not just labels.

    The calibrations reached only the dual writer, so the per-stream
    sessions fell back to volts channel by channel: both streams
    triggered on unrotated quadratures while the single-stream path,
    given the same arguments, rotated.  stored_units is the observable
    -- "Hz" when the rotation happened, "V" when it silently did not.
    """
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


def test_stale_teardown_does_not_disable_a_newer_capture():
    """The PFB enable and its teardown belong to one capture.

    An old task's finally can run after a new capture has enabled the
    streamer; disabling it then turns the stream off underneath the
    running capture -- a board everyone believes is streaming fast, and
    a dual capture that counts pulses and never pairs them, which is
    the state a live board was found in.
    """
    from rfmux.tools.periscope.pulse_capture_task import PulseCaptureTask

    calls = []

    class _CRS:
        async def set_pfb_streamer(self, channel=None, module=1):
            calls.append(channel)

    def make():
        t = PulseCaptureTask.__new__(PulseCaptureTask)
        t.crs = _CRS()
        t.module = 2
        t.signals = _Signals([])
        return t

    async def scenario():
        old, new = make(), make()
        old_claim = await old._claim_pfb([1])
        new_claim = await new._claim_pfb([1])
        # The stale teardown arrives after the new enable: skipped.
        await old._release_pfb(old_claim)
        assert calls == [[1], [1]], calls
        # The owner's own teardown still disables.
        await new._release_pfb(new_claim)
        assert calls == [[1], [1], None], calls

    asyncio.run(scenario())


def test_stream_lag_is_reported_in_stats():
    """The dual session reports how far the fast stream trails the slow.

    The matcher already keeps each stream's processed time, so a lag
    that would make cross-stream windows unavailable is observable
    without new bookkeeping.
    """
    import numpy as np
    from rfmux.pulse_capture.capture_session import (
        DualPulseCaptureSession, PulseCaptureConfig)

    cfg = PulseCaptureConfig(threshold_sigma=5.0, noise_train_ms=50.0,
                             max_pulse_ms=50.0)
    d = DualPulseCaptureSession(channels=[1], module=1, slow_rate=1000.0,
                                fast_rate=100000.0, config=cfg, hdf5_path=None)
    d.start()
    t = 43000.0
    # Feed slow up to t+2.0 but fast only to t+0.5: fast trails by 1.5 s.
    d.feed_slow_block(1, np.zeros(2000), np.zeros(2000), t + np.arange(2000) / 1000.0)
    d.feed_fast_block(1, np.zeros(50000), np.zeros(50000), t + np.arange(50000) / 100000.0)
    d.stop()

    st = d.stats()
    assert st["stream_lag_s"] == pytest.approx(1.5, abs=0.05)
    assert st["ring_overlap_s"] is not None and st["ring_overlap_s"] > 0


def test_status_line_warns_and_tooltips_on_lag(qt_app):
    """A drifting fast stream colours the status line and explains itself,
    the way the dropped-packet indicator does."""
    from rfmux.tools.periscope.pulse_capture_panel import PulseCapturePanel

    panel = PulseCapturePanel(dark_mode=False)
    panel._both_mode = True

    # Healthy: small lag well inside the ring overlap -> green, no note.
    ok, tip = panel._stream_lag_signal(
        {"stream_lag_s": 0.02, "ring_overlap_s": 0.4})
    assert ok == "#4CC38A" and tip is None

    # Amber: past half the overlap.
    amber, tip = panel._stream_lag_signal(
        {"stream_lag_s": 0.25, "ring_overlap_s": 0.4})
    assert amber == "#E5A23B" and "fast stream" in tip[0]

    # Red: lag exceeds the overlap -> windows are failing now.
    red, tip = panel._stream_lag_signal(
        {"stream_lag_s": 3.5, "ring_overlap_s": 0.4})
    assert red == "#E5484D"
    assert "unavailable" in tip[1] and "raise the threshold" in tip[1]

    # And it reaches the actual label + tooltip through the status line.
    panel._last_stats = {"pairs_matched": 0, "pairs_unmatched": 8,
                         "slow": {"total_pulses": 2}, "fast": {"total_pulses": 6},
                         "stream_lag_s": 3.5, "ring_overlap_s": 0.4}
    panel._refresh_status_line()
    assert "behind" in panel.status_label.text()
    assert "unavailable" in panel.status_label.toolTip()

    panel.close()
    spin(qt_app)


def _coincident_pulse_streams(slow_fs=1000.0, fast_fs=100000.0, seconds=3.0,
                              t_pulse=1.5, tau=0.02, amp=60.0, seed=3):
    """The same pulse on both streams, on one clock."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, fs in (("slow", slow_fs), ("fast", fast_fs)):
        n = int(seconds * fs)
        t = np.arange(n) / fs
        sig = rng.normal(0, 1.0, n)
        m = t >= t_pulse
        sig[m] += amp * np.exp(-(t[m] - t_pulse) / tau)
        out[name] = (sig, rng.normal(0, 1.0, n), t)
    return out


def _run_dual_with_slow_lag(lag_s, pair_window_wait_s=3.0):
    """Feed fast at real time and slow LAGGING by lag_s of stream time --
    the slow ring is behind when the fast trigger lands and the pair
    matches.  Returns the pairs emitted."""
    from rfmux.pulse_capture.capture_session import (
        DualPulseCaptureSession, PulseCaptureConfig)

    pairs = []
    cfg = PulseCaptureConfig(threshold_sigma=5.0, end_sigma=1.5,
                             max_pulse_ms=50.0, noise_train_ms=300.0)
    d = DualPulseCaptureSession(
        channels=[1], module=1, slow_rate=1000.0, fast_rate=100000.0,
        config=cfg, hdf5_path=None, pair_window_wait_s=pair_window_wait_s,
        on_pair=lambda p: pairs.append(p), on_error=lambda m: None)
    d.start()
    st = _coincident_pulse_streams()
    t0 = 43000.0
    sl = 0.05
    n_slices = int(3.0 / sl)
    lag_slices = int(round(lag_s / sl))
    for k in range(n_slices + lag_slices):
        # fast slice k; slow slice (k - lag) -- slow trails.
        if k < n_slices:
            a, b = int(k * sl * 100000), int((k + 1) * sl * 100000)
            i, q, t = st["fast"]
            d.feed_fast_block(1, i[a:b], q[a:b], t0 + t[a:b])
        ks = k - lag_slices
        if 0 <= ks < n_slices:
            a, b = int(ks * sl * 1000), int((ks + 1) * sl * 1000)
            i, q, t = st["slow"]
            d.feed_slow_block(1, i[a:b], q[a:b], t0 + t[a:b])
    d.stop()
    return pairs


def test_matched_pair_has_both_windows_when_slow_trails():
    """Integration sanity: slow fed one slice behind fast (the lag is in
    wall time, so the fast ring still holds the pulse), the pair matches,
    and both union windows are present and end together."""
    pairs = _run_dual_with_slow_lag(lag_s=0.05)
    matched = [p for p in pairs if p["slow_idx"] and p["fast_idx"]]
    assert matched, "no matched pair"
    p = matched[0]
    assert p.get("slow_tod") is not None and p.get("fast_tod") is not None
    s_end = float(np.nanmax(p["slow_tod"]["Time"]))
    f_end = float(np.nanmax(p["fast_tod"]["Time"]))
    assert s_end >= f_end - 0.002


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


def test_each_window_is_taken_when_its_own_ring_covers_it():
    """The pair forms while the slow ring is behind the window's end.

    Seen live: after a fast capture walked its samples, the slow ring
    sat behind; the pair matched the instant the fast trigger landed and
    its slow window was cut at the ring's newest sample, mid-pulse,
    though the samples were on their way.  The fast window must be taken
    at once (that ring is current and small), the slow one only once the
    slow ring reaches the window's end, and the pair emitted then.
    """
    d, pairs = _dual_for_deferral()
    T0 = 43000.0
    # Fast ring: current, well past the window.  Slow ring: behind it.
    _feed_to(d, "fast", T0, T0 + 1.00, 100000.0)
    _feed_to(d, "slow", T0, T0 + 0.60, 1000.0)
    # A matched pair whose union window ends at T0+0.80 -- beyond what
    # the slow ring holds, inside what the fast ring holds.
    pair = {"channel": 1, "pair_idx": 1, "slow_idx": 1, "fast_idx": 1,
            "slow_summary": {"timestamp": T0 + 0.70, "duration_s": 0.05},
            "fast_summary": {"timestamp": T0 + 0.70, "duration_s": 0.10},
            "time_offset": 0.0}
    d._on_matcher_pair(pair)

    # Not emitted yet: the fast window was taken, the slow one is waiting.
    assert pairs == []
    assert pair.get("fast_tod") is not None, "fast window should be taken now"
    assert "slow_tod" not in pair, "slow window must wait for its ring"

    # The slow ring reaches the window's end -> slow window taken, pair out.
    _feed_to(d, "slow", T0 + 0.60, T0 + 0.90, 1000.0)
    assert len(pairs) == 1
    st = pairs[0]["slow_tod"]
    assert st is not None
    assert float(np.nanmax(st["Time"])) >= T0 + 0.80 - 0.002, \
        "slow window stops short of the window end"
    d.stop()


def test_pairs_are_emitted_in_order_behind_a_waiting_one():
    """A later, complete pair does not overtake an earlier waiting one."""
    d, pairs = _dual_for_deferral()
    T0 = 43000.0
    _feed_to(d, "fast", T0, T0 + 1.00, 100000.0)
    _feed_to(d, "slow", T0, T0 + 0.60, 1000.0)
    waiting = {"channel": 1, "pair_idx": 1, "slow_idx": 1, "fast_idx": 1,
               "slow_summary": {"timestamp": T0 + 0.70, "duration_s": 0.05},
               "fast_summary": {"timestamp": T0 + 0.70, "duration_s": 0.05},
               "time_offset": 0.0}
    ready = {"channel": 1, "pair_idx": 2, "slow_idx": 2, "fast_idx": 2,
             "slow_summary": {"timestamp": T0 + 0.30, "duration_s": 0.05},
             "fast_summary": {"timestamp": T0 + 0.30, "duration_s": 0.05},
             "time_offset": 0.0}
    d._on_matcher_pair(waiting)
    d._on_matcher_pair(ready)
    assert pairs == [], "the ready pair must not overtake the waiting one"
    _feed_to(d, "slow", T0 + 0.60, T0 + 0.90, 1000.0)
    assert [p["pair_idx"] for p in pairs] == [1, 2]
    d.stop()


def test_a_stream_that_never_catches_up_does_not_strand_the_pair():
    """The wait is bounded: past pair_window_wait_s of the other
    stream's time, the pair goes out with what there is."""
    d, pairs = _dual_for_deferral()
    d._pair_window_wait_s = 0.2
    T0 = 43000.0
    # Fast is just past the window (0.85 < 0.755 + 0.2): still waiting.
    _feed_to(d, "fast", T0, T0 + 0.85, 100000.0)
    _feed_to(d, "slow", T0, T0 + 0.60, 1000.0)
    pair = {"channel": 1, "pair_idx": 1, "slow_idx": 1, "fast_idx": 1,
            "slow_summary": {"timestamp": T0 + 0.70, "duration_s": 0.05},
            "fast_summary": {"timestamp": T0 + 0.70, "duration_s": 0.05},
            "time_offset": 0.0}
    d._on_matcher_pair(pair)
    assert pairs == []
    # Only the FAST stream keeps going; slow never reaches the window.
    _feed_to(d, "fast", T0 + 1.00, T0 + 1.20, 100000.0)
    assert len(pairs) == 1, "pair stranded behind a stream that never came"
    assert "slow_tod" not in pairs[0] or pairs[0]["slow_tod"] is None \
        or float(np.nanmax(pairs[0]["slow_tod"]["Time"])) < T0 + 0.75
    d.stop()


def test_pfb_batch_yields_to_the_loop(monkeypatch):
    """A long PFB batch turns the loop over every _PFB_YIELD_EVERY
    packets, so a capture walk cannot hold it for a whole batch."""
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
    import time as _t
    _t.sleep(0.05)

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
    # 40 packets in one drain -> yields at k=8,16,24,32 inside the batch,
    # plus the one after the batch.  One-per-batch would give ~1.
    assert yields["n"] >= 4, f"only {yields['n']} yields for {N} packets"


def test_union_window_spans_the_saved_record():
    """The pair's window covers the whole saved record, not the core.

    duration_s is trigger-to-below-threshold by design (a pulse
    property, for the histograms).  Used as the display extent it cut
    the drawn data mid-pulse while the marks, read from the record, ran
    on to end-confirmed: 4.2 ms of trace under a 16 ms end mark.
    """
    from rfmux.pulse_capture.capture_session import DualPulseCaptureSession as D
    T = 43000.0
    pair = {"slow_summary": {"timestamp": T + 0.000, "start_time": T + 0.000,
                             "trigger_time": T + 0.002, "duration_s": 0.004,
                             "saved_end_time": T + 0.016},
            "fast_summary": None}
    t0, t1 = D._union_window(pair)
    assert t1 >= T + 0.016, f"window ends at +{(t1-T)*1e3:.1f} ms, record at +16 ms"
    assert t0 <= T + 0.000

    # A summary from before end_time was carried still gets the core.
    old = {"slow_summary": {"timestamp": T, "duration_s": 0.004},
           "fast_summary": None}
    t0, t1 = D._union_window(old)
    assert t1 == pytest.approx(T + 0.004 + 0.0004, abs=1e-6)


def test_matcher_anchors_on_the_trigger_not_the_record_start():
    """Two streams' records of ONE event start at different pre-margins
    (each is a fraction of that stream's own saved length).  Anchoring
    the midpoint on the record start pushed them apart by that
    difference; anchoring on the trigger keeps them together."""
    from rfmux.pulse_capture.capture_session import IncrementalPulseMatcher
    pairs = []
    m = IncrementalPulseMatcher(window_s=0.05, grace_s=0.25,
                                on_pair=lambda p: pairs.append(p))
    T = 43000.0
    # Same trigger, same core; the fast record kept a 120 ms pre-margin
    # (a long confirmation tail), the slow one 5 ms.  Record starts
    # differ by 115 ms -- well outside the 50 ms window.
    m.add("slow", 1, 1, {"timestamp": T - 0.005, "trigger_time": T,
                         "duration_s": 0.004})
    m.add("fast", 1, 1, {"timestamp": T - 0.120, "trigger_time": T,
                         "duration_s": 0.004})
    assert m.matched == 1 and pairs and pairs[0]["slow_idx"] == 1 \
        and pairs[0]["fast_idx"] == 1, "one event, two records, no match"


def test_band_pair_is_one_legend_entry_that_hides_both(qt_app):
    """Each +/- band is a single item: switching its legend entry off
    removes both lines, not just the one that carried the name."""
    import pyqtgraph as pg
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
    assert len(trig) == 1, "the trigger band is two items again"
    y = np.asarray(trig[0].yData, float)
    levels = set(np.round(y[np.isfinite(y)], 6))
    assert levels == {20.0, 0.0}, f"both +/- lines must be in the one item: {levels}"
    # And no unnamed twin lurking at either level.
    for item in plot.getPlotItem().listDataItems():
        if not item.name():
            yy = np.asarray(item.yData, float)
            assert not (set(np.round(yy[np.isfinite(yy)], 6)) & levels), \
                "an unnamed line still sits at a band level"
    panel.close()
    spin(qt_app)


def test_fast_and_slow_traces_are_told_apart_by_hue():
    from rfmux.tools.periscope.pulse_capture_panel import FAST_IQ_COLORS
    from rfmux.tools.periscope.utils import IQ_COLORS
    for q in ("I", "Q"):
        assert FAST_IQ_COLORS[q].lower() != IQ_COLORS[q].lower()


def _decision_labels(plot):
    import pyqtgraph as pg
    out = []
    for item in plot.getPlotItem().items:
        if isinstance(item, pg.InfiniteLine) and item.label is not None:
            out.append(item.label.format)
    return out


def test_end_confirmed_mark_follows_the_full_tail_setting(qt_app):
    """With the tail not saved there is no confirmation instant in the
    data, so no mark for it -- it was drawn past the end of the trace
    regardless of the checkbox."""
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
    """An opened file carries the policy that made its records, so the
    marks drawn over them match it."""
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
