"""Coincidence events: pulses on different channels that triggered
within the window of an event's first trigger, indexed beside the
per-channel pulses, with the channels that did not trigger dumped over
the same span when asked for."""

import numpy as np
import pytest

from rfmux.pulse_capture import (
    EventGrouper, PulseCaptureConfig, PulseCaptureSession, PulseHDF5Reader,
    events_of, group_by_trigger)

FS = 1000.0
CHANNELS = [1, 2, 3]


# ── the rule ──────────────────────────────────────────────────────

def test_triggers_inside_the_window_are_one_event():
    groups = group_by_trigger([(1.000, 1, 1), (1.004, 2, 1), (1.020, 3, 1)],
                              window_s=0.005)
    assert [[(ch, idx) for _, ch, idx in g] for g in groups] == \
        [[(1, 1), (2, 1)], [(3, 1)]]


def test_the_window_runs_from_the_first_trigger_so_events_cannot_chain():
    """Pulses 4 ms apart for ever would be one endless event if each
    only had to be near the one before it."""
    triggers = [(k * 0.004, 1 + k % 3, k) for k in range(10)]
    groups = group_by_trigger(triggers, window_s=0.005)
    assert all(g[-1][0] - g[0][0] <= 0.005 for g in groups)
    assert len(groups) == 5


def _summary(t, start=None, end=None):
    return {"trigger_time": t, "start_time": t - 0.002 if start is None
            else start, "saved_end_time": t + 0.01 if end is None else end}


def test_a_long_pulse_that_ends_late_still_joins_its_event():
    """Pulses arrive when they end.  Channel 2 triggered second and
    ended first; channel 1's arrives after it, and after stream time
    has passed the window, but before the hold has run out."""
    got = []
    g = EventGrouper(window_s=0.005, hold_s=0.06, on_event=got.append)
    g.add(2, 1, _summary(1.003))
    g.advance(1.030)                     # past the window, inside the hold
    assert got == []
    g.add(1, 1, _summary(1.000, end=1.050))
    g.advance(1.064)
    assert got == []
    g.advance(1.066)                     # first trigger + window + hold
    assert len(got) == 1
    assert [(m["channel"], m["pulse_idx"]) for m in got[0]["members"]] == \
        [(1, 1), (2, 1)]
    assert got[0]["window"] == pytest.approx((0.998, 1.050))


def test_flush_closes_what_is_pending():
    got = []
    g = EventGrouper(window_s=0.005, hold_s=10.0, on_event=got.append)
    g.add(1, 1, _summary(1.0))
    g.add(2, 1, _summary(5.0))
    g.flush()
    assert [e["event_idx"] for e in got] == [1, 2]


# ── a capture ─────────────────────────────────────────────────────

def _capture(tmp_path=None, channels=CHANNELS, pulses=((1, 100), (2, 103)),
             block=50, **config_kw):
    """A session fed noise, then a clean pulse on each of *pulses*
    (channel, start sample), then quiet.  Returns (events, path)."""
    cfg = PulseCaptureConfig(max_pulse_ms=50.0, noise_train_ms=300.0,
                             pre_pulse_ms=5.0, post_pulse_ms=5.0,
                             trigger_samples=1, **config_kw)
    events = []
    path = None if tmp_path is None else tmp_path / "events.h5"
    s = PulseCaptureSession(
        channels=channels, sample_rate=FS, hdf5_path=path,
        on_event=events.append, **cfg.session_kwargs(FS))
    s.start()
    rng = np.random.default_rng(3)
    n = 300
    t = np.arange(n) / FS
    for ch in channels:
        s.feed_block(ch, rng.normal(0, 1, n), rng.normal(0, 1, n), t)
    m = 600
    t = (n + np.arange(m)) / FS
    starts = dict(pulses)
    for lo in range(0, m, block):        # a block of each channel in turn
        for ch in channels:
            i = np.zeros(block)
            if ch in starts and 0 <= starts[ch] - lo < block:
                i[starts[ch] - lo:starts[ch] - lo + 4] = 60.0
            s.feed_block(ch, i, np.zeros(block), t[lo:lo + block])
    s.stop()
    return events, path


def test_coincident_pulses_on_two_channels_are_one_event():
    events, _ = _capture(coincidence_window_ms=5.0)
    assert len(events) == 1
    assert [(m["channel"], m["pulse_idx"]) for m in events[0]["members"]] \
        == [(1, 1), (2, 1)]
    assert "dump" not in events[0]


def test_pulses_further_apart_than_the_window_are_separate_events():
    events, _ = _capture(coincidence_window_ms=2.0)
    assert [[m["channel"] for m in e["members"]] for e in events] == [[1], [2]]


def test_no_events_are_recorded_unless_asked_for():
    events, path = _capture(None)
    assert events == []


def test_the_dump_covers_the_event_window_on_every_other_channel():
    events, _ = _capture(coincidence_window_ms=5.0, dump_all_channels=True)
    (event,) = events
    assert sorted(event["dump"]) == [3]          # 1 and 2 triggered
    t0, t1 = event["window"]
    times = event["dump"][3]["Time"]
    assert times[0] == pytest.approx(t0) and times[-1] == pytest.approx(t1)
    assert len(times) == round((t1 - t0) * FS) + 1
    assert len(event["dump"][3]["Amp_I"]) == len(times)


def test_an_event_waits_for_the_slowest_channel_before_it_is_dumped():
    """Channels arrive a block at a time.  With 300-sample blocks the
    first channel is past the event's closing time before the others
    have been fed the pulse's span at all; the event closes on the
    slowest of them, so their rings hold the window by then."""
    events, _ = _capture(block=300, coincidence_window_ms=5.0,
                         dump_all_channels=True)
    (event,) = events
    assert [m["channel"] for m in event["members"]] == [1, 2]
    t0, t1 = event["window"]
    times = event["dump"][3]["Time"]
    assert times[0] == pytest.approx(t0) and times[-1] == pytest.approx(t1)


def test_the_dump_alone_makes_each_pulse_an_event():
    events, _ = _capture(dump_all_channels=True)
    assert [[m["channel"] for m in e["members"]] for e in events] == [[1], [2]]
    assert [sorted(e["dump"]) for e in events] == [[2, 3], [1, 3]]


# ── the file ──────────────────────────────────────────────────────

def test_events_round_trip_through_the_file(tmp_path):
    events, path = _capture(tmp_path, coincidence_window_ms=5.0,
                            dump_all_channels=True)
    with PulseHDF5Reader(path) as r:
        assert r.event_count == 1
        assert r.metadata["coincidence_window_s"] == pytest.approx(0.005)
        assert r.metadata["dump_all_channels"]
        event = r.get_event(1)
        assert [(m["channel"], m["pulse_idx"]) for m in event["members"]] \
            == [(1, 1), (2, 1)]
        assert event["window"] == pytest.approx(events[0]["window"])
        assert event["dumped"] == [3]
        np.testing.assert_array_equal(event["dump"][3]["Amp_I"],
                                      events[0]["dump"][3]["Amp_I"])
        # The pulses are where they always were.
        assert r.pulse_count(1) == 1 and r.pulse_count(2) == 1
        # Browsing does not read the dumped samples.
        (lean,) = r.iter_events()
        assert "dump" not in lean and lean["dumped"] == [3]


def test_an_event_links_to_its_pulses(tmp_path):
    """A generic HDF5 tool opens an event and finds its pulses: each
    link is the pulse the members table names."""
    import h5py
    _, path = _capture(tmp_path, coincidence_window_ms=5.0)
    with h5py.File(path, "r") as f:
        links = f["events/event_000001/pulses"]
        assert sorted(links) == ["channel_1_pulse_000001",
                                 "channel_2_pulse_000001"]
        for name, target in (("channel_1_pulse_000001", "channel_1/pulse_000001"),
                             ("channel_2_pulse_000001", "channel_2/pulse_000001")):
            assert isinstance(links.get(name, getlink=True), h5py.SoftLink)
            assert links[name] == f[target]
        # A walk of the file still meets each pulse once, under its channel.
        seen = []
        f.visit(seen.append)
        assert sum(s.endswith("pulse_000001") for s in seen) == 2


def test_a_file_without_events_is_laid_out_as_before(tmp_path):
    _, path = _capture(tmp_path)
    with PulseHDF5Reader(path) as r:
        assert r.event_count == 0 and "events" not in r.f


def test_a_file_without_events_can_be_grouped_afterwards(tmp_path):
    """The grouping needs only trigger times, which every pulse has."""
    _, path = _capture(tmp_path)
    with PulseHDF5Reader(path) as r:
        together = events_of(r, window_s=0.005)
        apart = events_of(r, window_s=0.002)
    assert [[m["channel"] for m in e["members"]] for e in together] == [[1, 2]]
    assert [[m["channel"] for m in e["members"]] for e in apart] == [[1], [2]]


def test_recorded_events_can_be_regrouped_with_another_window(tmp_path):
    _, path = _capture(tmp_path, coincidence_window_ms=5.0)
    with PulseHDF5Reader(path) as r:
        assert len(events_of(r)) == 1
        assert len(events_of(r, window_s=0.002)) == 2


def test_events_across_modules_keep_their_channel_keys(tmp_path):
    keys = [(1, 5), (2, 5), (2, 6)]
    events, path = _capture(tmp_path, channels=keys,
                            pulses=(((1, 5), 100), ((2, 5), 102)),
                            coincidence_window_ms=5.0, dump_all_channels=True)
    with PulseHDF5Reader(path) as r:
        event = r.get_event(1)
    assert [m["channel"] for m in event["members"]] == [(1, 5), (2, 5)]
    assert event["dumped"] == [(2, 6)]
    import h5py
    with h5py.File(path, "r") as f:
        links = f["events/event_000001/pulses"]
        assert sorted(links) == ["module_1_channel_5_pulse_000001",
                                 "module_2_channel_5_pulse_000001"]
        assert links["module_2_channel_5_pulse_000001"] \
            == f["module_2/channel_5/pulse_000001"]


def test_events_need_a_sample_rate():
    with pytest.raises(ValueError, match="sample_rate"):
        PulseCaptureSession(channels=[1], coincidence_window_s=0.005)


def test_config_validation():
    assert any(sev == "error" and "coincidence" in msg.lower() for sev, msg in
               PulseCaptureConfig(coincidence_window_ms=-1).validate())
    assert any(sev == "warning" and "unrelated" in msg for sev, msg in
               PulseCaptureConfig(coincidence_window_ms=80.0,
                                  max_pulse_ms=50.0).validate())
