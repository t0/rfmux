"""
A pulse whose deviation rotates in the IQ plane is one pulse.

One quadrature decays while the other rises; judged per quadrature
that is a pileup, judged on the length of the deviation vector, and
with the rise measured against the pulse's own recent level, it is
one record.  The profile is a recorded fast-stream pulse in sigma
units.
"""

import numpy as np

from rfmux.pulse_capture.detection import ChannelNoiseStats, PulseCapture

DEV_I = [1, 0, -1, -1, 1, 5, 10, 2, -11, -14, -14, -12, -11, -14, -15,
         -13, -13, -13, -15, -15, -13, -13, -15, -15, -13, -11, -10, -12,
         -15, -13, -12, -13, -13, -14, -13, -11, -10, -10, -10, -9, -9,
         -11, -12, -12, -11, -8, -8, -9, -9, -9, -8, -7, -7, -7, -7, -7,
         -7, -7, -7, -6, -7, -7, -6, -5, -4, -6, -7, -7, -7, -5, -2, -2,
         -4]
DEV_Q = [1, 2, 1, -1, -1, 1, 13, 29, 32, 28, 25, 23, 25, 25, 23, 21, 22,
         22, 20, 18, 17, 18, 18, 15, 12, 12, 14, 16, 14, 11, 12, 12, 11,
         9, 8, 8, 9, 9, 8, 8, 9, 9, 7, 6, 4, 3, 5, 5, 4, 4, 3, 3, 3, 2,
         1, 1, 1, 1, 1, 2, 2, 1, 0, 0, 2, 3, 3, 1, 0, -1, 0, 2, 2]

FAST_RATE = 2441406.25


def _engine(records, **kw):
    ns = ChannelNoiseStats(mean_I=0.0, mean_Q=0.0, std_I=1.0, std_Q=1.0,
                           jump_std_I=2.06, jump_std_Q=2.06)
    return PulseCapture(
        channels=[1], buf_size=8192, noise_stats={1: ns},
        threshold_sigma=5.0, end_sigma=1.5, trigger_samples=2,
        edge_lookback=61035, min_end_samples=10, margin_fraction=0.1,
        max_capture_samples=732422, baseline_window=0,
        on_pulse=lambda ch, k, d: records.append(d), **kw)


def _feed(pc, I, Q, rng, lead=300, tail=400):
    n = 0
    for i, q in zip(rng.normal(0, 1, lead), rng.normal(0, 1, lead)):
        pc.process_sample(1, float(i), float(q), n / FAST_RATE); n += 1
    for i, q in zip(I, Q):
        pc.process_sample(1, float(i), float(q), n / FAST_RATE); n += 1
    for i, q in zip(rng.normal(0, 1, tail), rng.normal(0, 1, tail)):
        pc.process_sample(1, float(i), float(q), n / FAST_RATE); n += 1


def test_a_rotating_pulse_is_one_record():
    records = []
    _feed(_engine(records), DEV_I, DEV_Q, np.random.default_rng(0))
    assert len(records) == 1
    assert not records[0]["pileup"]
    assert records[0].get("below_threshold_index") is not None


def test_a_pulse_on_the_tail_still_splits():
    """The same pulse again, 40 samples after the first, on its tail."""
    I = list(DEV_I) + [0] * 40
    Q = list(DEV_Q) + [0] * 40
    k0 = 40
    for j in range(len(DEV_I)):
        I[k0 + j] += DEV_I[j]
        Q[k0 + j] += DEV_Q[j]
    records = []
    _feed(_engine(records), I, Q, np.random.default_rng(0))
    assert len(records) == 2
    assert all(r["pileup"] for r in records)
