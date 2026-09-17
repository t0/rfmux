"""Coincidence: pulses on different channels grouped into events.

An event opens at its first trigger and takes every pulse, on any
channel, that triggered within the coincidence window of that first
trigger.  Measuring from the first trigger rather than from the nearest
member bounds an event's length at the window: a steady rate of
unrelated pulses cannot chain into one event that never ends.

Pulses stay where they are, under their channels; an event is an index
over them, so a capture can be browsed by channel or by event, and a
file that was captured without events can still be grouped afterwards
(:func:`group_by_trigger`) from the trigger times its pulses carry.
What cannot be recovered afterwards is the data of the channels that
did not trigger, which the session takes from their rings when an event
closes (``dump_all_channels``).

A noise sample is an event nothing triggered: every channel over one
window, taken at a random moment whatever the samples hold, for the
statistics of the noise.  :class:`NoiseSampler` decides the moments.
"""

from __future__ import annotations

import math

import numpy as np
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .channel_keys import ChannelKey

#: One pulse as the grouping sees it.
Trigger = Tuple[float, ChannelKey, int]      # trigger time, channel, index


def group_by_trigger(triggers: Iterable[Trigger],
                     window_s: float) -> List[List[Trigger]]:
    """*triggers* grouped into events, each in trigger order."""
    events: List[List[Trigger]] = []
    for trig in sorted((t for t in triggers if math.isfinite(t[0])),
                       key=lambda t: t[0]):
        if events and trig[0] - events[-1][0][0] <= window_s:
            events[-1].append(trig)
        else:
            events.append([trig])
    return events


def event_window(summaries: Iterable[dict]) -> Optional[Tuple[float, float]]:
    """First saved sample to last saved sample over an event's pulses:
    the span a channel that did not trigger is dumped over."""
    spans = [(s.get("start_time"), s.get("saved_end_time"))
             for s in summaries]
    spans = [(a, b) for a, b in spans
             if a is not None and b is not None
             and math.isfinite(a) and math.isfinite(b)]
    if not spans:
        return None
    return min(a for a, _ in spans), max(b for _, b in spans)


class NoiseSampler:
    """When to take noise samples, and how long each is.

    The waits between samples are drawn from a normal distribution
    centred on ``interval_s``, ``JITTER`` of it wide, and never shorter
    than the window, so two samples cannot overlap.  The window starts
    ``pre_s`` before the chosen moment, as a pulse's record starts
    before its trigger.

    A sample is as long as a typical pulse record: the median length of
    the records the capture has saved (:meth:`observe`), or
    ``window_s`` until ``MIN_RECORDS`` of them have been.
    """

    #: Standard deviation of the wait, as a fraction of the interval.
    JITTER = 0.25
    #: Saved records before their median sets the sample's length, and
    #: how many of the latest it is the median of.
    MIN_RECORDS = 5
    RECORDS_KEPT = 200

    def __init__(self, interval_s: float, window_s: float, pre_s: float = 0.0,
                 rng: Optional[np.random.Generator] = None):
        self.interval_s = float(interval_s)
        self.default_window_s = float(window_s)
        self.pre_s = float(pre_s)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.next_t: Optional[float] = None
        self._lengths: List[float] = []

    def observe(self, length_s: float) -> None:
        """A pulse record of *length_s* was saved."""
        if math.isfinite(length_s) and length_s > 0:
            self._lengths = (self._lengths + [float(length_s)]
                             )[-self.RECORDS_KEPT:]

    @property
    def window_s(self) -> float:
        if len(self._lengths) < self.MIN_RECORDS:
            return self.default_window_s
        return float(np.median(self._lengths))

    def _wait(self) -> float:
        return max(self.window_s, float(self.rng.normal(
            self.interval_s, self.JITTER * self.interval_s)))

    @property
    def window(self) -> Optional[Tuple[float, float]]:
        """The span of the next sample; None until :meth:`take`
        has been given the stream's time."""
        if self.next_t is None:
            return None
        t0 = self.next_t - self.pre_s
        return t0, t0 + self.window_s

    def take(self, now: float) -> Optional[Tuple[float, Tuple[float, float]]]:
        """(moment, window) of a sample whose window the stream has
        passed, scheduling the next; None when none is due.  The first
        call only starts the schedule."""
        if self.next_t is None:
            self.next_t = now + self._wait()
            return None
        window = self.window
        if now < window[1]:
            return None
        moment = self.next_t
        self.next_t = max(moment, now - self.window_s) + self._wait()
        return moment, window


class EventGrouper:
    """Groups pulses into events as a capture runs.

    A pulse reaches :meth:`add` when it ends, which is not the order
    pulses triggered in: a long pulse that triggered first arrives after
    a short one that triggered later.  So an event is closed only once
    no pulse that belongs to it can still be open, ``hold_s`` (the hard
    stop, the longest a capture can run) past the end of its window, on
    the clock of the channel that is furthest behind.

    In a both-mode capture the members are the matched pairs rather
    than the pulses (``pulse_idx`` is then the pair's index), and a pair
    can be held up after its pulses end, waiting for its partner or for
    a ring; :meth:`advance` takes a ``settled`` test for that.

    ``on_event`` receives::

        {"event_idx", "kind": "pulses" | "noise", "trigger_time",
         "window": (t0, t1) | None,
         "members": [{"channel", "pulse_idx", "trigger_time",
                      "summary"}, ...]}

    with the members in trigger order.  A noise sample
    (:meth:`add_noise`) is an event of kind ``"noise"``: its members are
    whatever pulses triggered inside its window, often none, and it
    carries the ``dump`` it was given.  With ``window_s`` None pulses
    are not grouped into events of their own; they are only kept long
    enough to be named by a noise sample.
    """

    def __init__(self, window_s: Optional[float], hold_s: float,
                 on_event: Optional[Callable[[dict], None]] = None):
        self.group_pulses = window_s is not None
        self.window_s = max(0.0, float(window_s or 0.0))
        self.hold_s = max(0.0, float(hold_s))
        self.on_event = on_event
        self.event_count = 0
        self._pending: List[dict] = []
        self._noise: List[dict] = []
        # Pulses a noise sample still to close might contain.
        self._recent: List[dict] = []

    @property
    def deadline(self) -> float:
        """Stream time at which the oldest pending event can close;
        infinite with nothing pending."""
        return min(self._pulses_deadline(), self._noise_deadline())

    def _pulses_deadline(self) -> float:
        if not self._pending:
            return math.inf
        return (min(m["trigger_time"] for m in self._pending)
                + self.window_s + self.hold_s)

    def _noise_deadline(self) -> float:
        if not self._noise:
            return math.inf
        return self._noise[0]["window"][1] + self.hold_s

    def add(self, channel: ChannelKey, pulse_idx: int, summary: dict) -> None:
        t = summary.get("trigger_time")
        if t is None or not math.isfinite(t):
            return
        member = {"channel": channel, "pulse_idx": int(pulse_idx),
                  "trigger_time": float(t), "summary": summary}
        if self.group_pulses:
            self._pending.append(member)
        self._recent.append(member)

    def add_noise(self, moment: float, window: Tuple[float, float],
                  dump: dict) -> None:
        """A noise sample taken over *window*; it closes, like any
        event, once no pulse that triggered inside it can still be
        open."""
        self._noise.append({"trigger_time": float(moment),
                            "window": (float(window[0]), float(window[1])),
                            "dump": dump})

    def advance(self, now: float,
                settled: Optional[Callable[[float], bool]] = None) -> None:
        """Close every event no open pulse can still join; *now* is the
        clock of the slowest channel.  *settled*, given the end of an
        event's window, says whether everything that triggered by then
        has been added: a caller that holds pulses back after they end
        answers False while it holds one."""
        while now >= self.deadline:
            if settled is not None and not settled(
                    self.deadline - self.hold_s):
                break
            self._close_next()
        # A pulse older than any sample still open can name nothing.
        oldest = (self._noise[0]["window"][0] if self._noise
                  else now - self.hold_s)
        self._recent = [m for m in self._recent
                        if m["trigger_time"] >= min(oldest, now - self.hold_s)]

    def flush(self) -> None:
        """Close what is pending: the capture has stopped."""
        while self._pending or self._noise:
            self._close_next()

    def _close_next(self) -> None:
        if self._noise_deadline() < self._pulses_deadline():
            self._close_noise()
        else:
            self._close_oldest()

    def _close_noise(self) -> None:
        sample = self._noise.pop(0)
        t0, t1 = sample["window"]
        self.event_count += 1
        event = {
            "event_idx": self.event_count, "kind": "noise",
            "trigger_time": sample["trigger_time"],
            "window": sample["window"],
            "members": sorted((m for m in self._recent
                               if t0 <= m["trigger_time"] <= t1),
                              key=lambda m: m["trigger_time"]),
            "dump": sample["dump"],
        }
        if self.on_event is not None:
            self.on_event(event)

    def _close_oldest(self) -> None:
        self._pending.sort(key=lambda m: m["trigger_time"])
        first = self._pending[0]["trigger_time"]
        members = [m for m in self._pending
                   if m["trigger_time"] - first <= self.window_s]
        self._pending = self._pending[len(members):]
        self.event_count += 1
        event = {
            "event_idx": self.event_count, "kind": "pulses",
            "trigger_time": first,
            "window": event_window(m["summary"] for m in members),
            "members": members,
        }
        if self.on_event is not None:
            self.on_event(event)


def event_channels(event: dict) -> List[ChannelKey]:
    """The channels that triggered in *event*, in trigger order, each
    once."""
    seen: Dict[ChannelKey, None] = {}
    for m in event["members"]:
        seen.setdefault(m["channel"], None)
    return list(seen)


def events_of(reader, window_s: Optional[float] = None,
              stream: Optional[str] = None) -> List[dict]:
    """The events of a capture file, without dumped samples.

    With *window_s* None these are the events the capture recorded.
    Given a window, or for a file that recorded none, the file's pulses
    are grouped afresh from their trigger times, which needs nothing the
    capture did not already keep; such events have no dump.  A dual
    file's pairs are grouped, each at the earlier of its triggers, as
    its capture groups them; *stream* groups one stream's pulses
    instead.
    """
    if window_s is None and reader.event_count:
        return list(reader.iter_events())
    if reader.dual and stream is None:
        triggers = []
        for channel in reader.channels:
            for pair in reader.iter_matches(channel):
                times = [reader.get_pulse_metadata(channel, idx, side)
                         .get("trigger_time")
                         for side in ("slow", "fast")
                         for idx in [pair.get(f"{side}_idx")] if idx]
                times = [t for t in times if t is not None]
                if times:
                    triggers.append((float(min(times)), channel,
                                     int(pair["pair_idx"])))
    else:
        triggers = [(float(meta.get("trigger_time", meta.get("timestamp"))),
                     channel, int(meta["pulse_idx"]))
                    for channel in reader.channels
                    for meta in reader.iter_pulse_metadata(channel, stream)]
    return [{"event_idx": k, "kind": "pulses",
             "trigger_time": group[0][0], "window": None,
             "dumped": [],
             "members": [{"channel": ch, "pulse_idx": idx, "trigger_time": t}
                         for t, ch, idx in group]}
            for k, group in enumerate(
                group_by_trigger(triggers, window_s or 0.0), start=1)]
