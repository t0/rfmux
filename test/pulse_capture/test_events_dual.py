"""Coincidence events in a both-mode capture: a channel's share of an
event is its pair, whichever of its streams triggered, and the channels
without one are read from both rings over the event's window."""

import numpy as np
import pytest

from rfmux.pulse_capture import (
    DualPulseCaptureSession, PulseCaptureConfig, PulseHDF5Reader, events_of)

SLOW_FS, FAST_FS = 1000.0, 20000.0
CHANNELS = [1, 2, 3]
TAU = 0.004

#: (stream, channel, start): channel 1 pulses on both streams at 1.0 s,
#: channel 2 on the slow stream alone 2 ms later, and channel 3 on the
#: fast stream alone a second after that.
PULSES = [("slow", 1, 1.000), ("fast", 1, 1.000), ("slow", 2, 1.002),
          ("fast", 3, 2.000)]


def _capture(tmp_path=None, **config_kw):
    cfg = PulseCaptureConfig(max_pulse_ms=50.0, noise_train_ms=100.0,
                             trigger_basis="iq", **config_kw)
    got = {"events": [], "pairs": [], "errors": []}
    path = None if tmp_path is None else tmp_path / "dual_events.h5"
    dual = DualPulseCaptureSession(
        channels=CHANNELS, slow_rate=SLOW_FS, fast_rate=FAST_FS, config=cfg,
        hdf5_path=path, slow_time_offset_s=0.0,
        on_event=got["events"].append, on_pair=got["pairs"].append,
        on_error=got["errors"].append)
    dual.start()
    rng = np.random.default_rng(7)
    feeds = {"slow": (dual.feed_slow_block, SLOW_FS),
             "fast": (dual.feed_fast_block, FAST_FS)}
    step = 0.05
    for k in range(int(3.2 / step)):             # 50 ms of each stream a turn
        for stream, (feed, fs) in feeds.items():
            n = int(round(step * fs))
            t = k * step + np.arange(n) / fs
            for ch in CHANNELS:
                i = rng.normal(0, 1, n)
                for s, c, start in PULSES:
                    if (s, c) == (stream, ch):
                        on = t >= start
                        i[on] += 60.0 * np.exp(-(t[on] - start) / TAU)
                feed(ch, i, rng.normal(0, 1, n), t)
    dual.stop()
    assert not got["errors"], got["errors"]
    return got, path


def _sides(pairs, event):
    """{channel: which streams triggered} for an event's members."""
    by_key = {(p["channel"], p["pair_idx"]): p for p in pairs}
    out = {}
    for m in event["members"]:
        p = by_key[(m["channel"], m["pulse_idx"])]
        out[m["channel"]] = tuple(s for s in ("slow", "fast")
                                  if p[f"{s}_idx"] is not None)
    return out


def test_pairs_on_different_channels_are_one_event():
    got, _ = _capture(coincidence_window_ms=5.0)
    first, second = got["events"]
    assert _sides(got["pairs"], first) == {1: ("slow", "fast"), 2: ("slow",)}
    assert _sides(got["pairs"], second) == {3: ("fast",)}
    assert "dump" not in first


def test_a_window_too_short_separates_them():
    got, _ = _capture(coincidence_window_ms=0.5)
    assert [sorted(_sides(got["pairs"], e)) for e in got["events"]] == \
        [[1], [2], [3]]


def test_no_events_unless_asked_for():
    got, _ = _capture()
    assert got["events"] == [] and len(got["pairs"]) == 3


def test_the_dump_takes_both_streams_of_every_other_channel():
    got, _ = _capture(coincidence_window_ms=5.0, dump_all_channels=True)
    first, second = got["events"]
    assert sorted(first["dump"]) == [3]
    assert sorted(second["dump"]) == [1, 2]
    t0, t1 = first["window"]
    for side, fs in (("slow_tod", SLOW_FS), ("fast_tod", FAST_FS)):
        times = first["dump"][3][side]["Time"]
        assert times[0] - t0 < 1.5 / fs and t1 - times[-1] < 1.5 / fs, side


def test_events_round_trip_through_a_dual_file(tmp_path):
    got, path = _capture(tmp_path, coincidence_window_ms=5.0,
                         dump_all_channels=True)
    with PulseHDF5Reader(path) as r:
        assert r.dual and r.event_count == 2
        assert r.metadata["coincidence_window_s"] == pytest.approx(0.005)
        assert r.metadata["dump_all_channels"]
        event = r.get_event(1)
        assert [(m["channel"], m["pulse_idx"]) for m in event["members"]] \
            == [(m["channel"], m["pulse_idx"])
                for m in got["events"][0]["members"]]
        # A member is a pair of the file.
        assert r.get_match(1, event["members"][0]["pulse_idx"]) is not None
        assert event["dumped"] == [3]
        for side in ("slow_tod", "fast_tod"):
            np.testing.assert_array_equal(
                event["dump"][3][side]["Amp_I"],
                got["events"][0]["dump"][3][side]["Amp_I"])
        # Grouped afresh, a dual file's pairs come out the same way.
        regrouped = events_of(r, window_s=0.005)
    assert [[m["channel"] for m in e["members"]] for e in regrouped] == \
        [[1, 2], [3]]
