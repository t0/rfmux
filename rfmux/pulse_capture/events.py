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
"""

from __future__ import annotations

import math
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

        {"event_idx", "trigger_time", "window": (t0, t1) | None,
         "members": [{"channel", "pulse_idx", "trigger_time",
                      "summary"}, ...]}

    with the members in trigger order.
    """

    def __init__(self, window_s: float, hold_s: float,
                 on_event: Optional[Callable[[dict], None]] = None):
        self.window_s = max(0.0, float(window_s))
        self.hold_s = max(0.0, float(hold_s))
        self.on_event = on_event
        self.event_count = 0
        self._pending: List[dict] = []

    @property
    def deadline(self) -> float:
        """Stream time at which the oldest pending event can close;
        infinite with nothing pending."""
        if not self._pending:
            return math.inf
        return (min(m["trigger_time"] for m in self._pending)
                + self.window_s + self.hold_s)

    def add(self, channel: ChannelKey, pulse_idx: int, summary: dict) -> None:
        t = summary.get("trigger_time")
        if t is None or not math.isfinite(t):
            return
        self._pending.append({"channel": channel, "pulse_idx": int(pulse_idx),
                              "trigger_time": float(t), "summary": summary})

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
            self._close_oldest()

    def flush(self) -> None:
        """Close what is pending: the capture has stopped."""
        while self._pending:
            self._close_oldest()

    def _close_oldest(self) -> None:
        self._pending.sort(key=lambda m: m["trigger_time"])
        first = self._pending[0]["trigger_time"]
        members = [m for m in self._pending
                   if m["trigger_time"] - first <= self.window_s]
        self._pending = self._pending[len(members):]
        self.event_count += 1
        event = {
            "event_idx": self.event_count,
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
    return [{"event_idx": k, "trigger_time": group[0][0], "window": None,
             "dumped": [],
             "members": [{"channel": ch, "pulse_idx": idx, "trigger_time": t}
                         for t, ch, idx in group]}
            for k, group in enumerate(
                group_by_trigger(triggers, window_s or 0.0), start=1)]
