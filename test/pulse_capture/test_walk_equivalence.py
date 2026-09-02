"""
walk.walk is process_sample transcribed; this holds it to the original.

One stream carries every branch of the capturing state machine: a
clean pulse, a pileup chain, a step that never returns (hard stop), a
run parked above threshold by drift with a pulse on top, and a window
with triggers frozen.  It is ingested three ways -- per sample, in
blocks without the walk, in blocks with it -- and every field of every
record must agree bitwise.
"""

import math

import numpy as np
import pytest

from rfmux.pulse_capture.detection import PulseCapture
from rfmux.pulse_capture.sources import SlowIngest

from test.pulse_capture.test_block_ingest_equivalence import (
    DT, FS, _session)

N = 4600
FREEZE = (3600, 3900)           # triggers frozen on these packets
HARD_STOP = 300                 # max_capture_samples


def _stream(channels, rng):
    k = np.arange(N, dtype=float)
    shape = np.zeros(N)

    def pulse(k0, amp, tau=15):
        m = k >= k0
        shape[m] += amp * np.exp(-(k[m] - k0) / tau)

    pulse(600, 80.0)
    for k0, amp in ((900, 60.0), (912, 50.0), (924, 70.0)):     # pileup
        pulse(k0, amp)
    shape[1300:2100] += 40.0                                    # step
    ramp = np.clip((k - 2400) / 400.0, 0, 1) * 7.0              # drift
    ramp -= np.clip((k - 3100) / 400.0, 0, 1) * 7.0
    shape += ramp
    pulse(3000, 60.0)                                           # on drift
    pulse(3700, 80.0)                                           # frozen
    pulse(4100, 80.0)
    out = []
    for i in range(N):
        vals = (rng.normal(0, 1.0, len(channels))
                + 1j * rng.normal(0, 1.0, len(channels)))
        out.append((vals + shape[i], i * DT))
    return out


def _norm(x):
    if isinstance(x, dict):
        return {k: _norm(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_norm(v) for v in x.tolist()]
    if isinstance(x, (float, np.floating)):
        return repr(float(x))
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x


def _sorted(records):
    return sorted(records, key=lambda r: (r[0], float(r[2]["trigger_time"])))


def _records(channels, packets, *, blocks, use_walk, max_packets=64):
    records = []
    s = _session(channels, [], threshold_sigma=5.0, end_sigma=2.0,
                 max_capture_samples=HARD_STOP)
    s.on_pulse = lambda ch, idx, summary, data: records.append(
        (ch, idx, _norm(summary), _norm(data)))
    PulseCapture.use_walk = use_walk
    try:
        s.start()
        acc = SlowIngest(s.feed_block, max_packets=max_packets,
                         max_age_s=1e9) if blocks else None
        for i, (values, ts) in enumerate(packets):
            if i in FREEZE:
                if acc is not None:
                    acc.flush()
                s.pcap.freeze_triggers = (i == FREEZE[0])
            if acc is not None:
                acc.add(channels, values, ts)
            else:
                for column, ch in enumerate(channels):
                    v = values[column]
                    s.feed_sample(ch, float(v.real), float(v.imag), ts)
        if acc is not None:
            acc.flush()
        s.stop()
    finally:
        PulseCapture.use_walk = True
    return records


@pytest.mark.parametrize("channels", [(1,), (1, 2)])
@pytest.mark.parametrize("max_packets", [1, 37, 4096])
def test_walk_matches_process_sample(channels, max_packets):
    packets = _stream(channels, np.random.default_rng(11))
    ref = _records(channels, packets, blocks=False, use_walk=False)
    loop = _records(channels, packets, blocks=True, use_walk=False,
                    max_packets=max_packets)
    walked = _records(channels, packets, blocks=True, use_walk=True,
                      max_packets=max_packets)
    # Per-sample ingest interleaves channels within a packet; block
    # ingest finishes one channel's block before the next.  The records
    # are the same, their order across channels is not.
    assert _sorted(loop) == _sorted(ref)
    assert _sorted(walked) == _sorted(ref)

    # The fixture must actually reach the branches it claims to.
    flags = [(r[2]["pileup"], r[2]["truncated"], float(r[2]["trigger_time"]))
             for r in ref if r[0] == channels[0]]
    assert any(p for p, _, _ in flags), "no pileup split"
    assert any(t for _, t, _ in flags), "no hard stop"
    assert not any(FREEZE[0] * DT <= t < FREEZE[1] * DT
                   for _, _, t in flags), "a trigger fired while frozen"
    assert any(2.7 <= t <= 3.1 for _, _, t in flags), \
        "no trigger on the drifted run"
    assert any(4.05 <= t <= 4.15 for _, _, t in flags)
