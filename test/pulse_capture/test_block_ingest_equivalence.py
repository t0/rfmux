"""
Headless and GUI must ingest the slow stream identically.

Both routes go through SlowIngest now, but the accumulator
only earns that trust if a block of samples produces exactly what the
same samples produce one at a time.  These tests feed identical
synthetic packets down each path and compare what the sessions saw.
"""

import numpy as np
import pytest

from rfmux.pulse_capture.capture_session import (
    CaptureState, PulseCaptureSession)
from rfmux.pulse_capture.sources import (
    SlowIngest, columns_for_width)

from test.pulse_capture.ingest_helpers import (  # noqa: E402
    FS, NOISE, packets as _packets, run_blocks as _run_blocks,
    run_per_sample as _run_per_sample, session as _session)


@pytest.mark.parametrize("channels", [(1,), (1, 2, 3)])
def test_block_and_sample_ingest_agree(channels):
    rng = np.random.default_rng(11)
    packets = _packets(channels, 1400, rng)

    by_sample, by_block = [], []
    s1 = _run_per_sample(channels, packets, by_sample)
    s2 = _run_blocks(channels, packets, by_block)

    assert s1.state is s2.state is CaptureState.STOPPED
    # Blocks dispatch a whole channel at a time, so the CALLBACK order
    # differs from interleaved per-sample feeding.  What must not
    # differ is which pulses were found, and when.
    assert sorted(by_block) == sorted(by_sample), \
        "block ingest changed what was detected"
    assert by_block, "the fixture should produce pulses at all"


def test_unusable_timestamps_are_dropped_not_poisoned():
    channels = (1,)
    rng = np.random.default_rng(9)
    packets = _packets(channels, 900, rng)
    pulses = []
    s = _session(channels, pulses)
    s.start()
    acc = SlowIngest(s.feed_block, max_packets=64, max_age_s=1e9)
    for n, (values, ts) in enumerate(packets):
        acc.add(channels, values, None if n % 100 == 0 else ts)
        if acc.ready:
            acc.flush()
    acc.flush()
    s.stop()
    # Those samples are accounted for, not silently turned into NaN
    # timestamps inside the engine.
    assert s.dropped_invalid_ts > 0


def test_columns_for_width_drops_unreachable_channels():
    kept, idx = columns_for_width((1, 5, 200), 128)
    assert kept == (1, 5)
    assert idx.tolist() == [0, 4]

    kept, idx = columns_for_width((1, 5, 200), 1024)
    assert kept == (1, 5, 200)
    assert idx.tolist() == [0, 4, 199]


def test_accumulator_flushes_on_channel_change():
    seen = []
    acc = SlowIngest(
        lambda ch, i, q, t: seen.append((ch, len(i))),
        max_packets=1000, max_age_s=1e9)
    acc.add((1, 2), np.array([1 + 1j, 2 + 2j]), 0.0)
    acc.add((1, 2), np.array([3 + 3j, 4 + 4j]), 0.1)
    assert seen == [], "nothing is due yet"
    # A different channel set must not be stacked onto the old columns.
    acc.add((5,), np.array([9 + 9j]), 0.2)
    assert seen == [(1, 2), (2, 2)]


def test_held_tail_is_not_transformed_twice():
    """The block held across the end of noise training keeps its basis.

    When one channel fills its noise quota before another, feed_block
    stashes the rest of that channel's block and replays it once the
    session starts capturing.  The replay used to go back through
    feed_block, which applies the basis transform -- so those samples,
    and only those, were transformed a second time.

    Checked against the samples the engine actually receives, rather
    than against what was detected: the doubled stretch is short and
    sits at the transition, so whether it changes a trigger depends on
    where the pulses happen to fall.  It needs several channels, because
    with one nothing is ever held.
    """
    import numpy as np
    from rfmux.core.transferfunctions import VOLTS_PER_ROC

    channels = (1, 2, 3)
    rng = np.random.default_rng(11)
    packets = _packets(channels, 1400, rng)

    # Everything the engine was handed, per channel, in order.
    seen = {c: [] for c in channels}
    session = PulseCaptureSession(
        channels=list(channels), sample_rate=FS, noise_samples=NOISE,
        on_pulse=lambda *a: None)

    # Patched on the class, because the drain happens inside the same
    # call that builds the engine -- wrapping session.pcap afterwards
    # misses exactly the samples in question.
    from rfmux.pulse_capture import detection as _d
    real_block = _d.PulseCapture.process_block
    real_sample = _d.PulseCapture.process_sample

    def spy_block(self, ch, I, Q, T):
        seen.setdefault(ch, []).extend(np.asarray(I).tolist())
        return real_block(self, ch, I, Q, T)

    def spy_sample(self, ch, i, q, t):
        seen.setdefault(ch, []).append(float(i))
        return real_sample(self, ch, i, q, t)

    _d.PulseCapture.process_block = spy_block
    _d.PulseCapture.process_sample = spy_sample
    try:
        session.start()
        acc = SlowIngest(session.feed_block, max_packets=256, max_age_s=1e9)
        for values, ts in packets:
            acc.add(channels, values, ts)
        acc.flush()
        session.stop()
    finally:
        _d.PulseCapture.process_block = real_block
        _d.PulseCapture.process_sample = real_sample
    assert any(seen.values())

    # Raw counts the engine should have received, once converted.
    raw = {c: [float(v[i].real) for v, _ in packets]
           for i, c in enumerate(channels)}
    for ch in channels:
        got = np.array(seen[ch])
        # Every value must be some raw sample times exactly one factor.
        ratios = got / VOLTS_PER_ROC
        near = np.array([np.min(np.abs(np.array(raw[ch]) - r)) for r in ratios])
        assert np.all(near < 1e-6), (
            f"ch{ch}: {int((near >= 1e-6).sum())} of {len(got)} samples reached "
            "the engine with the conversion applied more than once")


def _run_block_adds(channels, packets, pulses, rows=64, max_packets=256, **kw):
    """Blocks of *rows* packets through SlowIngest.add_block."""
    s = _session(channels, pulses, **kw)
    s.start()
    acc = SlowIngest(s.feed_block, max_packets=max_packets, max_age_s=1e9)
    for k in range(0, len(packets), rows):
        chunk = packets[k:k + rows]
        acc.add_block(channels, np.stack([v for v, _ in chunk]),
                      np.array([t for _, t in chunk]))
    acc.flush()
    s.stop()
    return s


@pytest.mark.parametrize("rows", [1, 7, 64, 5000])
def test_block_adds_agree_with_packet_adds(rows):
    """add_block, the batched tap's entry, is add over the block."""
    rng = np.random.default_rng(9)
    channels = (1, 2, 3)
    packets = _packets(channels, 2400, rng, pulse_starts=(600, 1500))
    by_packet, by_block = [], []
    _run_blocks(channels, packets, by_packet)
    _run_block_adds(channels, packets, by_block, rows=rows)
    assert sorted(by_block) == sorted(by_packet)
    assert by_packet


