"""
walk.walk is process_sample transcribed; this holds it to the original.

One stream carries every branch of the capturing state machine: a
clean pulse, a pileup chain, a step that never returns (hard stop), a
run parked above threshold by drift with a pulse on top, and a window
with triggers frozen.  It is ingested three ways -- per sample, in
blocks without the walk, in blocks with it -- and every field of every
record must agree bitwise.
"""

import numpy as np
import pytest

from rfmux.pulse_capture import walk
from rfmux.pulse_capture.detection import PulseCapture
from rfmux.pulse_capture.sources import SlowIngest

from test.pulse_capture.ingest_helpers import DT, session as _session

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


def _records(channels, packets, *, blocks, use_walk, max_packets=64,
             monkeypatch=None, **kw):
    """use_walk: False for the loop, "python" for the walk uncompiled,
    True for the walk as compiled."""
    if use_walk == "python":
        monkeypatch.setattr(walk, "walk", walk.walk.py_func)
        use_walk = True
    records = []
    s = _session(channels, [], max_capture_samples=HARD_STOP, **kw)
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


# A rolling baseline of 60 samples refreshes many times inside a
# 300-sample capture: each refresh is a walk re-entry through
# process_sample.  2.5 sigma puts the threshold in the noise: crossings
# everywhere, splits at every boundary.
@pytest.mark.parametrize("channels", [(1,), (1, 2)])
@pytest.mark.parametrize("max_packets", [1, 37, 4096])
@pytest.mark.parametrize("baseline_window", [0, 60])
@pytest.mark.parametrize("threshold", [5.0, 2.5])
def test_walk_matches_process_sample(channels, max_packets, baseline_window,
                                     threshold, monkeypatch):
    kw = dict(threshold_sigma=threshold, end_sigma=threshold * 0.4,
              baseline_window=baseline_window)
    if baseline_window:
        pc = PulseCapture(channels=[1], buf_size=1024, noise_stats={},
                          baseline_window=baseline_window)
        assert pc._bl_decim * pc._bl_refresh < HARD_STOP
    packets = _stream(channels, np.random.default_rng(11))
    ref = _records(channels, packets, blocks=False, use_walk=False, **kw)
    loop = _records(channels, packets, blocks=True, use_walk=False,
                    max_packets=max_packets, **kw)
    compiled = _records(channels, packets, blocks=True, use_walk=True,
                        max_packets=max_packets, **kw)
    with monkeypatch.context() as m:
        python = _records(channels, packets, blocks=True,
                          use_walk="python", max_packets=max_packets,
                          monkeypatch=m, **kw)
    # Per-sample ingest interleaves channels within a packet; block
    # ingest finishes one channel's block before the next.  The records
    # are the same, their order across channels is not.
    assert _sorted(loop) == _sorted(ref)
    assert _sorted(python) == _sorted(ref)
    assert _sorted(compiled) == _sorted(ref)

    # The fixture must actually reach the branches it claims to.
    flags = [(r[2]["pileup"], r[2]["truncated"], float(r[2]["trigger_time"]))
             for r in ref if r[0] == channels[0]]
    assert not any(FREEZE[0] * DT <= t < FREEZE[1] * DT
                   for _, _, t in flags), "a trigger fired while frozen"
    # A rolling baseline absorbs the step and the drift, and 2.5 sigma
    # triggers on noise: the branches below are reached as designed
    # only with the fixed baseline at 5 sigma.
    if threshold < 5.0 or baseline_window:
        return
    assert any(p for p, _, _ in flags), "no pileup split"
    assert any(t for _, t, _ in flags), "no hard stop"
    assert any(2.7 <= t <= 3.1 for _, _, t in flags), \
        "no trigger on the drifted run"
    assert any(4.05 <= t <= 4.15 for _, _, t in flags)


def test_walk_is_compiled_without_the_gil():
    """While one engine walks, the receiver and the other stream's engine
    must keep running: the walk is compiled and releases the lock."""
    walk.warm_up()
    assert walk.walk.targetoptions.get("nogil") is True
    assert walk.walk.signatures
