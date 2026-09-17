"""What a viewer of a capture computes before it draws: noise statistics
taken into the viewed axes, the level a trace is drawn about, and how a
capture's events divide."""

import numpy as np
import pytest

from rfmux.pulse_capture.analysis import baseline_level, project_noise_stats
from rfmux.pulse_capture.detection import ChannelNoiseStats
from rfmux.pulse_capture.events import (
    event_counts, events_from_triggers, lean_event, pair_trigger_time)

pytestmark = pytest.mark.portable


def test_noise_spreads_are_projected_onto_the_viewed_axes():
    """A channel noisier along df than along dissipation, viewed 80
    degrees around: the wide band belongs to the second axis."""
    ns = ChannelNoiseStats(mean_I=100.0, std_I=50.0, mean_Q=-20.0, std_Q=5.0)
    factor = 0.5 * np.exp(1j * np.radians(80.0))
    seen = project_noise_stats(ns, factor)
    theta = np.radians(80.0)
    assert seen.std_I == pytest.approx(
        0.5 * np.hypot(50.0 * np.cos(theta), 5.0 * np.sin(theta)))
    assert seen.std_Q == pytest.approx(
        0.5 * np.hypot(50.0 * np.sin(theta), 5.0 * np.cos(theta)))
    assert seen.std_Q > 4 * seen.std_I
    assert complex(seen.mean_I, seen.mean_Q) == pytest.approx(
        complex(100.0, -20.0) * factor)


def test_a_jump_spread_that_was_not_measured_stays_unmeasured():
    ns = ChannelNoiseStats(mean_I=0.0, std_I=1.0, mean_Q=0.0, std_Q=1.0,
                           jump_std_I=0.0, jump_std_Q=2.0)
    seen = project_noise_stats(ns, np.exp(0.5j))
    assert (seen.jump_std_I, seen.jump_std_Q) == (0.0, 0.0)


def test_a_pulse_is_drawn_about_the_level_it_triggered_from():
    wf = {"Time": np.arange(6.0), "Amp_I": np.full(6, 9.0),
          "trigger_baseline_I": 4.0}
    assert baseline_level(wf, "I", t_ref=3.0) == 4.0


def test_a_window_without_a_trigger_is_drawn_about_its_earlier_median():
    wf = {"Time": np.arange(8.0),
          "Amp_Q": np.array([1.0, 2.0, 3.0, 4.0, 50.0, 50.0, 50.0, 50.0])}
    assert baseline_level(wf, "Q", t_ref=4.0) == 2.5
    assert baseline_level(wf, "Q", t_ref=2.0) == np.median(wf["Amp_Q"])


def test_a_pair_triggers_at_the_earlier_of_its_streams():
    assert pair_trigger_time({"slow_summary": {"trigger_time": 2.0},
                              "fast_summary": {"trigger_time": 1.5}}) == 1.5
    assert pair_trigger_time({"slow_summary": None,
                              "fast_summary": {"trigger_time": 1.5}}) == 1.5
    assert np.isnan(pair_trigger_time({}))


def test_event_counts_divide_a_capture():
    events = events_from_triggers(
        [(0.100, 1, 1), (0.103, 2, 1), (0.400, 3, 1), (0.402, 3, 2)], 0.005)
    events.append({"kind": "noise", "members": []})
    assert event_counts(events) == {
        "coincident_events": 1, "coincident_pulses": 2, "noise_samples": 1}


def test_a_lean_event_names_its_samples_without_carrying_them():
    event = {"event_idx": 4, "kind": "pulses", "trigger_time": 1.0,
             "window": (0.9, 1.2),
             "members": [{"channel": 1, "pulse_idx": 7, "trigger_time": 1.0,
                          "summary": {"snr": 9.0}}],
             "dump": {3: {"Amp_I": np.zeros(4)}, 2: {"Amp_I": np.zeros(4)}}}
    lean = lean_event(event)
    assert lean["dumped"] == [2, 3] and "dump" not in lean
    assert lean["members"] == [
        {"channel": 1, "pulse_idx": 7, "trigger_time": 1.0}]
