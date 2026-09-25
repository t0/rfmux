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


def test_a_both_mode_event_draws_the_fast_lines_under_the_slow_points(
        qt_app, tmp_path):
    """As a pair draws them: the slow samples, sparse, stay visible over
    the dense fast trace, for every channel of the event."""
    from test.pulse_capture.test_events_dual import _capture
    _, path = _capture(tmp_path, coincidence_window_ms=5.0,
                       dump_all_channels=True)
    panel = _review(path, GROUP_EVENTS)
    panel._show_event(1)
    for plot in (panel.pulse_plot_i, panel.pulse_plot_q):
        curves = plot.getPlotItem().listDataItems()
        slow = [c.zValue() for c in curves if " slow" in c.name()]
        fast = [c.zValue() for c in curves if " fast" in c.name()]
        assert slow and fast and min(slow) > max(fast)


def _double_click(panel, role):
    for k in range(panel.pulse_tree.topLevelItemCount()):
        top = panel.pulse_tree.topLevelItem(k)
        for j in range(top.childCount()):
            if tuple(top.child(j).data(0, ROLE) or ()) == role:
                panel._on_tree_double_click(top.child(j), 0)
                return
    raise AssertionError(f"no row {role}")


def test_a_no_trigger_row_fills_the_pulse_view_and_the_plane(
        qt_app, tmp_path):
    """It is not a pulse, but it is that channel's data for the event,
    and it is shown the way a pulse is."""
    panel = _review(_capture_file(tmp_path, coincidence_window_ms=5.0,
                                  dump_all_channels=True), GROUP_EVENTS)
    saved = panel.reader.get_event(1)["dump"][3]
    _double_click(panel, ("dump", 1, 3))
    assert "Channel 3" in panel.pulse_info.text()
    assert "[no trigger]" in panel.pulse_info.text()
    for plot, name in ((panel.pulse_plot_i, "Amp_I"),
                       (panel.pulse_plot_q, "Amp_Q")):
        curve = max(plot.getPlotItem().listDataItems(),
                    key=lambda c: len(c.getData()[0]))
        assert len(curve.getData()[0]) == len(saved[name])
    assert panel._iq_source()[0] == 3
    np.testing.assert_array_equal(panel._iq_source()[1]["Amp_I"],
                                  saved["Amp_I"])
    # And a pulse row afterwards is a pulse again.
    _double_click(panel, ("pulse", 1, 1))
    assert panel._current_dump is None and panel._iq_source()[0] == 1


def test_a_noise_training_row_replaces_a_no_trigger_view(qt_app, tmp_path):
    """The training segment stays drawn when a late read arrives for the
    no-trigger row that was viewed before it."""
    panel = _review(_capture_file(tmp_path, coincidence_window_ms=5.0,
                                  dump_all_channels=True), GROUP_EVENTS)
    _double_click(panel, ("dump", 1, 3))
    panel.group_combo.setCurrentText(GROUP_CHANNELS)
    _double_click(panel, ("noise", None, 2))
    panel._on_waveform_ready(3, 1)
    assert "Noise training" in panel.pulse_info.text()
    assert panel._iq_source() is None or panel._iq_source()[0] != 3


def test_a_both_mode_no_trigger_row_draws_both_streams(qt_app, tmp_path):
    from test.pulse_capture.test_events_dual import _capture
    _, path = _capture(tmp_path, coincidence_window_ms=5.0,
                       dump_all_channels=True)
    panel = _review(path, GROUP_EVENTS)
    _double_click(panel, ("dump", 1, 3))
    names = _curve_names(panel)
    assert any(n.startswith("fast") for n in names)
    assert any(n.startswith("slow") for n in names)
    panel.iq_stream_combo.setCurrentText("fast")
    fast = panel.reader.get_event(1)["dump"][3]["fast_tod"]
    assert len(panel._iq_source()[1]["Time"]) == len(fast["Time"])


def test_views_name_the_frequency_the_channel_is_biased_at(qt_app, tmp_path):
    panel = _review(_capture_file(tmp_path, coincidence_window_ms=5.0,
                                  dump_all_channels=True), GROUP_EVENTS)
    panel._tuning_rows = {1: {"bias_frequency": 1.2345e9}, 2: {}, 3: {}}
    panel._show_pulse(1, 1)
    assert "Channel 1 (1234.500000 MHz)" in panel.pulse_info.text()
    panel._show_pulse(2, 1)                      # no tuning, no claim
    assert "MHz" not in panel.pulse_info.text()
    panel._show_event(1)
    assert "Ch1 (1234.500000 MHz), Ch2" in panel.pulse_info.text()


def test_the_strip_reports_activity_once_there_are_pulses(qt_app, tmp_path):
    """The busiest channel, and coincident against lone pulses; the
    noise each channel trained to only until a pulse arrives."""
    panel = PulseCapturePanel(dark_mode=False)
    panel._reset_results([1, 2, 3])
    panel._refresh_noise_label()
    assert panel.noise_label.text().startswith("Noise:")

    panel = _review(_capture_file(tmp_path, coincidence_window_ms=5.0))
    text = panel.noise_label.text()
    assert "Most active:  Ch" in text and "1 of 3 pulses" in text
    assert "coincident:  2 in 1 event" in text and "alone:  1" in text
    assert "Noise" not in text
    assert "Pulses per channel" in panel.noise_label.toolTip()
    panel._refresh_noise_label()                 # a view change keeps it
    assert panel.noise_label.text() == text


def test_the_strip_says_when_coincidence_is_off(qt_app, tmp_path):
    panel = _review(_capture_file(tmp_path))
    assert "coincidence window off" in panel.noise_label.text()


def test_a_units_change_redraws_the_event_and_the_no_trigger_views(
        qt_app, tmp_path):
    """The traces move with the axis labels, not only the labels."""
    from rfmux.core.transferfunctions import VOLTS_PER_ROC
    from rfmux.tools.periscope.pulse_capture_panel import (
        UNITS_COUNTS, UNITS_VOLTS)
    panel = _review(_capture_file(tmp_path, coincidence_window_ms=5.0,
                                  dump_all_channels=True), GROUP_EVENTS)
    panel.units_combo.setCurrentText(UNITS_VOLTS)

    def peak():
        return max(np.max(np.abs(c.getData()[1]))
                   for c in panel.pulse_plot_i.getPlotItem().listDataItems())

    panel._show_event(1)
    volts = peak()
    panel.units_combo.setCurrentText(UNITS_COUNTS)
    assert panel._current_event == 1
    assert peak() == pytest.approx(volts / VOLTS_PER_ROC, rel=1e-6)
    label = panel.pulse_plot_i.getPlotItem().getAxis("left").labelText
    assert label == "I (counts) − baseline"

    _double_click(panel, ("dump", 1, 3))
    panel.units_combo.setCurrentText(UNITS_VOLTS)
    assert panel._current_dump == (1, 3)
    assert panel.pulse_plot_i.getPlotItem().getAxis("left").labelText \
        == "I (V)"


@pytest.fixture(scope="module")
def noise_file(tmp_path_factory):
    """A capture with noise samples, one of them holding a pulse; the
    tests that read it do not change it."""
    from test.pulse_capture.test_noise_samples import _capture
    dry, _, _, _ = _capture(noise_capture_interval_s=1.0)
    start = int(round(dry[1]["window"][0] * FS)) + 20
    events, path, _, _ = _capture(tmp_path_factory.mktemp("noise"),
                                  pulses=[(2, start)],
                                  noise_capture_interval_s=1.0)
    return events, path


def test_noise_samples_are_tagged_in_the_event_list(qt_app, noise_file):
    events, path = noise_file
    panel = _review(path, GROUP_EVENTS)
    labels = [panel.pulse_tree.topLevelItem(k).text(0)
              for k in range(len(events))]
    assert all(text.startswith("◇ Noise sample #") for text in labels)
    # Every channel under each, and the pulse that fell inside one.
    tree = _tree(panel)
    assert tree[("event", 2)] == [("pulse", 2, 1), ("dump", 2, 1),
                                  ("dump", 2, 2), ("dump", 2, 3)]
    assert tree[("event", 1)] == [("dump", 1, 1), ("dump", 1, 2),
                                  ("dump", 1, 3)]
    assert f"noise samples:  {len(events)}" in panel.noise_label.text()


def test_a_noise_sample_shows_the_time_it_was_taken_at(qt_app, noise_file):
    _, path = noise_file
    panel = _review(path, GROUP_EVENTS)
    panel._events[0]["trigger_utc"] = "2026-09-02T16:00:01.250000Z"
    panel._rebuild_tree()
    rows = [panel.pulse_tree.topLevelItem(k)
            for k in range(panel.pulse_tree.topLevelItemCount())]
    row = next(r for r in rows if r.data(0, ROLE) == ("event", 1))
    assert row.text(1) == "16:00:01.250000"
    panel._show_event(1)
    assert "taken at 2026-09-02T16:00:01.250000Z" in panel.pulse_info.text()


def test_a_noise_sample_draws_every_channel(qt_app, noise_file):
    _, path = noise_file
    panel = _review(path, GROUP_EVENTS)
    panel._show_event(2)
    assert _curve_names(panel) == ["Ch1", "Ch2", "Ch3"]
    text = panel.pulse_info.text()
    assert text.startswith("Noise sample #000002") and "Ch2" in text
    panel._show_event(1)
    assert "no pulse triggered inside it" in panel.pulse_info.text()
    _double_click(panel, ("dump", 1, 3))
    assert "[noise sample]" in panel.pulse_info.text()


def test_following_passes_over_noise_samples(qt_app, noise_file):
    """A noise sample is for the statistics, not for keeping up with
    the capture: with nothing else to follow, nothing is drawn."""
    _, path = noise_file
    panel = _review(path, GROUP_EVENTS)
    panel.follow_check.setChecked(True)
    panel._show_latest()
    assert panel._current_event is None
