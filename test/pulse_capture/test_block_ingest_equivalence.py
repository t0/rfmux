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

FS = 1000.0
DT = 1.0 / FS
NOISE = 400


def _session(channels, pulses, **kw):
    return PulseCaptureSession(
        channels=list(channels), sample_rate=FS, noise_samples=NOISE,
        hdf5_path=None,
        on_pulse=lambda ch, idx, summary, data: pulses.append(
            (ch, idx, round(float(summary["timestamp"]), 9))),
        **kw,
    )


def _packets(channels, n, rng, pulse_starts=(600, 900), tau=15, amp=80.0):
    """(values, timestamp) per packet — one complex sample per channel."""
    k = np.arange(n)
    shape = np.zeros(n)
    for k0 in pulse_starts:
        m = k >= k0
        shape[m] += amp * np.exp(-(k[m] - k0) / tau)
    out = []
    for i in range(n):
        vals = (rng.normal(0, 1.0, len(channels))
                + 1j * rng.normal(0, 1.0, len(channels)))
        vals = vals + shape[i]          # same pulse on every channel
        out.append((vals, i * DT))
    return out


def _run_per_sample(channels, packets, pulses, **kw):
    s = _session(channels, pulses, **kw)
    s.start()
    for values, ts in packets:
        for column, ch in enumerate(channels):
            v = values[column]
            s.feed_sample(ch, float(v.real), float(v.imag), ts)
    s.stop()
    return s


def _run_blocks(channels, packets, pulses, max_packets=256, **kw):
    s = _session(channels, pulses, **kw)
    s.start()
    acc = SlowIngest(s.feed_block, max_packets=max_packets,
                               max_age_s=1e9)   # size-driven only
    for values, ts in packets:
        acc.add(channels, values, ts)
    acc.flush()
    s.stop()
    return s


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


@pytest.mark.parametrize("max_packets", [1, 7, 64, 4096])
def test_block_size_does_not_change_the_answer(max_packets):
    # Where the block boundaries land must not matter — including a
    # block larger than the whole capture, and blocks of one.
    channels = (1, 2)
    rng = np.random.default_rng(5)
    packets = _packets(channels, 1200, rng)

    reference, chunked = [], []
    _run_per_sample(channels, packets, reference)
    _run_blocks(channels, packets, chunked, max_packets=max_packets)
    assert sorted(chunked) == sorted(reference)


def test_blocks_straddling_the_end_of_noise_training():
    # The transition out of ESTIMATING lands mid-block here; feed_block
    # has to split it rather than drop the remainder.
    channels = (1,)
    rng = np.random.default_rng(3)
    packets = _packets(channels, 1400, rng)

    reference, chunked = [], []
    _run_per_sample(channels, packets, reference)
    # NOISE=400 is not a multiple of 256, so a boundary falls inside.
    _run_blocks(channels, packets, chunked, max_packets=256)
    assert sorted(chunked) == sorted(reference)


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


def test_dense_scattered_crossings_agree():
    """Block and per-sample ingest agree when crossings scatter.

    process_block walks hit ISLANDS and bulks the quiet gaps between
    them.  A clean single pulse is one island and never exercises a
    second; this drives the threshold down into the noise so a block
    holds many crossings with wide gaps -- the multi-island, gap-bulk
    path -- and requires the two ingest routes to still detect exactly
    the same pulses at exactly the same times.
    """
    rng = np.random.default_rng(4)
    channels = (1,)
    # A couple of real pulses on top of noise, at a low threshold so the
    # noise itself crosses constantly: many islands per block.
    packets = _packets(channels, 3000, rng, pulse_starts=(1200, 2100),
                        tau=15, amp=30.0)
    kw = dict(threshold_sigma=2.5, end_sigma=1.5)

    by_sample, by_block = [], []
    _run_per_sample(channels, packets, by_sample, **kw)
    # Vary the block boundaries: the gaps must not depend on where a
    # block happens to split.
    for mp in (1, 7, 64, 256, 4096):
        by_block.clear()
        _run_blocks(channels, packets, by_block, max_packets=mp, **kw)
        assert sorted(by_block) == sorted(by_sample), \
            f"block ingest disagreed at max_packets={mp}: " \
            f"{len(by_block)} vs {len(by_sample)} pulses"
    assert by_sample, "the fixture should trigger at 2.5 sigma"


def _step_packets(channels, n, rng, start, length, amp):
    """Noise with a level shift of *amp* on the first channel."""
    out = []
    for i in range(n):
        vals = (rng.normal(0, 1.0, len(channels))
                + 1j * rng.normal(0, 1.0, len(channels)))
        if start <= i < start + length:
            vals[0] += amp
        out.append((vals, i / FS))
    return out


def test_a_level_shift_agrees_between_the_routes():
    """The latched stretch after a hard stop goes through the bulk path
    in block ingest and through process_sample per sample; both must
    produce the one truncated record and nothing else."""
    rng = np.random.default_rng(8)
    channels = (1,)
    packets = _step_packets(channels, 9000, rng, start=1000, length=6000,
                            amp=60.0)
    by_sample, by_block = [], []
    s1 = _run_per_sample(channels, packets, by_sample)
    assert len(by_sample) == 1, by_sample
    for mp in (1, 64, 4096):
        by_block.clear()
        s2 = _run_blocks(channels, packets, by_block, max_packets=mp)
        assert by_block == by_sample, f"max_packets={mp}"
        assert s2.pcap.state[1].latched is s1.pcap.state[1].latched
