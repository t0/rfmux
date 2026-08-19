"""
Headless and GUI must ingest the slow stream identically.

Both routes go through SlowBlockAccumulator now, but the accumulator
only earns that trust if a block of samples produces exactly what the
same samples produce one at a time.  These tests feed identical
synthetic packets down each path and compare what the sessions saw.
"""

import numpy as np
import pytest

from rfmux.pulse_capture.session import (
    CaptureState, PulseCaptureSession)
from rfmux.pulse_capture.sources import (
    SlowBlockAccumulator, columns_for_width)

FS = 1000.0
DT = 1.0 / FS
NOISE = 400


def _session(channels, pulses):
    return PulseCaptureSession(
        channels=list(channels), sample_rate=FS, noise_samples=NOISE,
        hdf5_path=None,
        on_pulse=lambda ch, idx, summary, data: pulses.append(
            (ch, idx, round(float(summary["timestamp"]), 9))),
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


def _run_per_sample(channels, packets, pulses):
    s = _session(channels, pulses)
    s.start()
    for values, ts in packets:
        for column, ch in enumerate(channels):
            v = values[column]
            s.feed_sample(ch, float(v.real), float(v.imag), ts)
    s.stop()
    return s


def _run_blocks(channels, packets, pulses, max_packets=256):
    s = _session(channels, pulses)
    s.start()
    acc = SlowBlockAccumulator(s.feed_block, max_packets=max_packets,
                               max_age_s=1e9)   # size-driven only
    for values, ts in packets:
        acc.add_and_flush_if_ready(channels, values, ts)
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
    acc = SlowBlockAccumulator(s.feed_block, max_packets=64, max_age_s=1e9)
    for n, (values, ts) in enumerate(packets):
        acc.add(channels, values, None if n % 100 == 0 else ts)
        if acc.ready:
            acc.flush()
    acc.flush()
    s.stop()
    # Those samples are accounted for, not silently turned into NaN
    # timestamps inside the detector.
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
    acc = SlowBlockAccumulator(
        lambda ch, i, q, t: seen.append((ch, len(i))),
        max_packets=1000, max_age_s=1e9)
    acc.add((1, 2), np.array([1 + 1j, 2 + 2j]), 0.0)
    acc.add((1, 2), np.array([3 + 3j, 4 + 4j]), 0.1)
    assert seen == [], "nothing is due yet"
    # A different channel set must not be stacked onto the old columns.
    acc.add((5,), np.array([9 + 9j]), 0.2)
    assert seen == [(1, 2), (2, 2)]
