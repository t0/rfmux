"""Sample time must survive the discontinuities a real stream has.

``SlowIngest`` accumulates capture duration from packet timestamps
monotonically and clamped per packet, rather than taking last minus
first. Two things break the naive version, and neither can be reached by
running the mock: a decimation change restarts the stream clock, and the
timestamp wraps to zero at the day boundary. Simulated captures are
short and never cross midnight.

That is the reason this logic is a plain object with its own tests
instead of something only a live socket loop can exercise.
"""

import numpy as np
import pytest

from rfmux.pulse_capture import SlowIngest


def _ingest(**kw):
    """An ingest whose feed records blocks instead of a session."""
    fed = []

    def feed(channel, i, q, stamps):
        fed.append((channel, len(stamps)))

    ingest = SlowIngest(feed, **kw)
    return ingest, fed


def _sample(n=1):
    return np.ones(n, dtype=np.complex128)


@pytest.mark.portable
def test_steady_stream_accumulates_its_duration():
    ingest, _ = _ingest()
    for i in range(11):
        ingest.advance(100.0 + i * 0.1)
    assert ingest.elapsed == pytest.approx(1.0)


@pytest.mark.portable
def test_the_day_boundary_does_not_rewind_the_clock():
    """Timestamps are seconds of day: 23:59:59.9 -> 00:00:00.0."""
    ingest, _ = _ingest()
    ingest.advance(86399.8)
    ingest.advance(86399.9)
    ingest.advance(0.0)          # midnight
    ingest.advance(0.1)
    assert ingest.elapsed == pytest.approx(0.2), \
        "a backwards step was counted as elapsed time"


@pytest.mark.portable
def test_a_long_gap_is_a_discontinuity_not_elapsed_time():
    """A decimation change restarts the clock; do not bill it."""
    ingest, _ = _ingest()
    ingest.advance(10.0)
    ingest.advance(10.1)
    ingest.advance(600.0)        # stream restarted somewhere else
    ingest.advance(600.1)
    assert ingest.elapsed == pytest.approx(0.2)


@pytest.mark.portable
def test_a_repeated_timestamp_adds_nothing():
    ingest, _ = _ingest()
    for _ in range(5):
        ingest.advance(42.0)
    assert ingest.elapsed == 0.0


@pytest.mark.portable
def test_missing_timestamps_are_survivable():
    """Packets without a recent timestamp arrive as None."""
    ingest, _ = _ingest()
    ingest.advance(1.0)
    ingest.advance(None)
    ingest.advance(1.1)
    assert ingest.elapsed == pytest.approx(0.1)


@pytest.mark.portable
def test_duration_completes_on_sample_time():
    ingest, _ = _ingest(duration_s=0.5)
    assert not ingest.complete
    for i in range(5):
        ingest.advance(i * 0.1)
    assert not ingest.complete, "0.4 s of steps should not finish 0.5 s"
    ingest.advance(0.5)
    ingest.advance(0.6)
    assert ingest.complete


@pytest.mark.portable
def test_without_a_duration_it_never_completes():
    ingest, _ = _ingest()
    for i in range(100):
        ingest.advance(i * 0.1)
    assert not ingest.complete


@pytest.mark.portable
def test_add_keeps_time_as_well_as_blocks():
    """The per-packet duty is one call, so the two cannot drift."""
    ingest, fed = _ingest(max_packets=2, max_age_s=1e9)
    ingest.add((1, 2), _sample(2), 5.0)
    ingest.add((1, 2), _sample(2), 5.1)
    assert ingest.elapsed == pytest.approx(0.1)
    assert fed, "two packets at max_packets=2 should have flushed a block"


@pytest.mark.portable
def test_a_packet_with_no_wanted_channel_is_still_time_passing():
    """Otherwise a capture whose channels are all past the packet
    width would never reach its duration and would run forever."""
    ingest, fed = _ingest(duration_s=0.2)
    ingest.add((), _sample(0), 1.0)
    ingest.add((), _sample(0), 1.3)
    assert ingest.elapsed == pytest.approx(0.3)
    assert ingest.complete
    assert not fed
