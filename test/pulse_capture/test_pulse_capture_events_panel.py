"""The Pulse Capture panel grouped by events: the same pulses as under
their channels, an event's channels drawn together, and the channels
that did not trigger shown with it except while following."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6 import QtCore  # noqa: E402

from rfmux.pulse_capture import (  # noqa: E402
    PulseCaptureConfig, PulseCaptureSession)
from rfmux.tools.periscope.pulse_capture_panel import (  # noqa: E402
    GROUP_CHANNELS, GROUP_EVENTS, PulseCapturePanel)

FS = 1000.0
ROLE = QtCore.Qt.ItemDataRole.UserRole


def _capture_file(tmp_path, **config_kw):
    """Channels 1 and 2 pulse 3 ms apart, then channel 3 alone 300 ms
    later; channel 3 is quiet during the first event."""
    cfg = PulseCaptureConfig(max_pulse_ms=50.0, noise_train_ms=300.0,
                             trigger_samples=1, trigger_basis="iq",
                             **config_kw)
    path = tmp_path / "events.h5"
    s = PulseCaptureSession(channels=[1, 2, 3], sample_rate=FS,
                            hdf5_path=path, **cfg.session_kwargs(FS))
    s.start()
    rng = np.random.default_rng(3)
    n = 300
    for ch in (1, 2, 3):
        s.feed_block(ch, rng.normal(0, 1, n), rng.normal(0, 1, n),
                     np.arange(n) / FS)
    starts = {1: [100], 2: [103], 3: [400]}
    t = (n + np.arange(800)) / FS
    for lo in range(0, 800, 50):
        for ch in (1, 2, 3):
            i = np.zeros(50)
            for start in starts[ch]:
                if 0 <= start - lo < 50:
                    i[start - lo:start - lo + 4] = 60.0
            s.feed_block(ch, i, np.zeros(50), t[lo:lo + 50])
    s.stop()
    return path


def _review(path, grouping=GROUP_CHANNELS):
    panel = PulseCapturePanel(dark_mode=False)
    panel.group_combo.setCurrentText(grouping)
    panel.load_from_hdf5(path)
    return panel


def _tree(panel):
    """{top-level role: [child roles]} for the pulse and event rows."""
    out = {}
    for k in range(panel.pulse_tree.topLevelItemCount()):
        top = panel.pulse_tree.topLevelItem(k)
        role = top.data(0, ROLE)
        if role and role[0] in ("channel", "event"):
            out[tuple(role)] = [tuple(top.child(j).data(0, ROLE))
                                for j in range(top.childCount())]
    return out


def _pulses(tree):
    return sorted(r for rows in tree.values() for r in rows
                  if r[0] == "pulse")


def _curve_names(panel):
    return sorted(c.name() for c in panel.pulse_plot_i.getPlotItem()
                  .listDataItems())


def test_both_groupings_hold_the_same_pulses(qt_app, tmp_path):
    panel = _review(_capture_file(tmp_path, coincidence_window_ms=5.0,
                                  dump_all_channels=True))
    by_channel = _tree(panel)
    assert set(by_channel) == {("channel", 1), ("channel", 2), ("channel", 3)}
    panel.group_combo.setCurrentText(GROUP_EVENTS)
    by_event = _tree(panel)
    assert _pulses(by_event) == _pulses(by_channel) == \
        [("pulse", 1, 1), ("pulse", 2, 1), ("pulse", 3, 1)]
    # Newest first: the lone pulse, then the coincidence, each with the
    # channels that were saved without a trigger.
    assert list(by_event) == [("event", 2), ("event", 1)]
    assert by_event[("event", 1)] == [("pulse", 1, 1), ("pulse", 2, 1),
                                      ("dump", 1, 3)]
    panel.group_combo.setCurrentText(GROUP_CHANNELS)
    assert _tree(panel) == by_channel


def test_an_event_draws_its_channels_together(qt_app, tmp_path):
    panel = _review(_capture_file(tmp_path, coincidence_window_ms=5.0,
                                  dump_all_channels=True), GROUP_EVENTS)
    panel._show_event(1)
    assert _curve_names(panel) == ["Ch1", "Ch2", "Ch3 (no trigger)"]
    assert "2 channels triggered: Ch1, Ch2" in panel.pulse_info.text()
    # One time axis from the event's first trigger: channel 2's pulse
    # rises 3 ms after channel 1's.
    rise = {}
    for c in panel.pulse_plot_i.getPlotItem().listDataItems():
        x, y = c.getData()
        rise[c.name()] = x[np.argmax(y > 0.5 * y.max())]
    assert rise["Ch1"] == pytest.approx(0.0, abs=1e-9)
    assert rise["Ch2"] - rise["Ch1"] == pytest.approx(0.003)


def test_event_traces_sit_about_the_level_each_pulse_triggered_from(
        qt_app, tmp_path):
    """The baseline wanders after training; the level recorded at the
    trigger follows it, so a drifted channel still starts at zero."""
    panel = _review(_capture_file(tmp_path, coincidence_window_ms=5.0),
                    GROUP_EVENTS)
    wf = panel.reader.get_pulse(1, 1)
    drifted = dict(wf, Amp_I=wf["Amp_I"] + 1.0,
                   trigger_baseline_I=wf["trigger_baseline_I"] + 1.0)
    panel._get_waveform = lambda ch, idx, stream=None: (
        drifted if (ch, idx) == (1, 1) else panel.reader.get_pulse(ch, idx))
    panel._show_event(1)
    curve = next(c for c in panel.pulse_plot_i.getPlotItem().listDataItems()
                 if c.name() == "Ch1")
    assert abs(curve.getData()[1][0]) < 1e-6


def test_following_leaves_out_the_channels_that_did_not_trigger(
        qt_app, tmp_path):
    panel = _review(_capture_file(tmp_path, coincidence_window_ms=5.0,
                                  dump_all_channels=True), GROUP_EVENTS)
    panel.follow_check.setChecked(True)
    panel._show_latest()
    assert panel._current_event == 2
    assert _curve_names(panel) == ["Ch3"]
    assert "not drawn while following" in panel.pulse_info.text()


def test_a_capture_without_events_is_grouped_from_its_trigger_times(
        qt_app, tmp_path):
    """The grouping needs nothing the file does not have; the window is
    the capture settings', and there is no dump to show."""
    panel = _review(_capture_file(tmp_path))
    panel.capture_config.coincidence_window_ms = 5.0
    panel.group_combo.setCurrentText(GROUP_EVENTS)
    assert list(_tree(panel).values()) == [
        [("pulse", 3, 1)], [("pulse", 1, 1), ("pulse", 2, 1)]]
    panel.capture_config.coincidence_window_ms = 1.0
    panel._rebuild_tree()
    assert len(_tree(panel)) == 3


def test_prev_and_next_step_through_events(qt_app, tmp_path):
    panel = _review(_capture_file(tmp_path, coincidence_window_ms=5.0),
                    GROUP_EVENTS)
    panel._show_event(2)
    panel._navigate(-1)
    assert panel._current_event == 1
    panel._navigate(+1)
    assert panel._current_event == 2


def test_a_live_event_joins_the_tree(qt_app, tmp_path):
    """What the worker announces when an event closes: a row at the top
    with the pulses already registered beneath it."""
    panel = PulseCapturePanel(dark_mode=False)
    panel._reset_results([1, 2])
    panel.group_combo.setCurrentText(GROUP_EVENTS)
    for ch in (1, 2):
        panel._on_pulse_detected(ch, 1, {"trigger_time": 10.0 + ch * 1e-3,
                                         "n_samples": 40, "snr": 9.0})
    panel._on_event_closed({
        "event_idx": 1, "trigger_time": 10.001, "window": (9.99, 10.05),
        "members": [{"channel": 1, "pulse_idx": 1, "trigger_time": 10.001},
                    {"channel": 2, "pulse_idx": 1, "trigger_time": 10.002}],
        "dumped": []})
    assert _tree(panel) == {("event", 1): [("pulse", 1, 1), ("pulse", 2, 1)]}


def test_the_grouping_sits_above_the_pulse_list(qt_app):
    """It groups the list, so it is with the list and not inside one of
    the view tabs."""
    panel = PulseCapturePanel(dark_mode=False)
    assert panel.group_combo.parent().parent() is \
        panel.pulse_tree.parent()
    assert not panel.viewer_tabs.isAncestorOf(panel.group_combo)


def test_a_both_mode_event_is_made_of_pairs_and_draws_either_stream(
        qt_app, tmp_path):
    from test.pulse_capture.test_events_dual import _capture
    _, path = _capture(tmp_path, coincidence_window_ms=5.0,
                       dump_all_channels=True)
    panel = _review(path, GROUP_EVENTS)
    tree = _tree(panel)
    # Newest first.  Channel 1 triggered on both streams, channel 2 on
    # the slow one alone, and channel 3 was saved without a trigger.
    assert [[r[0] for r in rows] for rows in tree.values()] == [
        ["pair", "dump", "dump"], ["pair", "pair", "dump"]]
    assert panel.event_stream_box.isVisibleTo(panel)

    panel._show_event(1)
    assert "Ch1, Ch2 (slow)" in panel.pulse_info.text()
    assert _curve_names(panel) == sorted([
        "Ch1 slow", "Ch1 fast", "Ch2 slow", "Ch2 fast",
        "Ch3 slow (no trigger)", "Ch3 fast (no trigger)"])
    panel.event_stream_combo.setCurrentText("slow")
    assert _curve_names(panel) == ["Ch1 slow", "Ch2 slow",
                                   "Ch3 slow (no trigger)"]
