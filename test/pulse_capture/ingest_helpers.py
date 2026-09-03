"""A synthetic slow stream and the sessions that ingest it, shared by
the tests that hold one ingest path to another."""
import numpy as np

from rfmux.pulse_capture.capture_session import PulseCaptureSession
from rfmux.pulse_capture.sources import SlowIngest

FS = 1000.0
DT = 1.0 / FS
NOISE = 400


def session(channels, pulses, **kw):
    return PulseCaptureSession(
        channels=list(channels), sample_rate=FS, noise_samples=NOISE,
        hdf5_path=None,
        on_pulse=lambda ch, idx, summary, data: pulses.append(
            (ch, idx, round(float(summary["timestamp"]), 9))),
        **kw,
    )


def packets(channels, n, rng, pulse_starts=(600, 900), tau=15, amp=80.0):
    """(values, timestamp) per packet: one complex sample per channel,
    the same pulses on every channel."""
    k = np.arange(n)
    shape = np.zeros(n)
    for k0 in pulse_starts:
        m = k >= k0
        shape[m] += amp * np.exp(-(k[m] - k0) / tau)
    out = []
    for i in range(n):
        vals = (rng.normal(0, 1.0, len(channels))
                + 1j * rng.normal(0, 1.0, len(channels)))
        out.append((vals + shape[i], i * DT))
    return out


def run_per_sample(channels, packets_, pulses, **kw):
    s = session(channels, pulses, **kw)
    s.start()
    for values, ts in packets_:
        for column, ch in enumerate(channels):
            v = values[column]
            s.feed_sample(ch, float(v.real), float(v.imag), ts)
    s.stop()
    return s


def run_blocks(channels, packets_, pulses, max_packets=256, **kw):
    s = session(channels, pulses, **kw)
    s.start()
    acc = SlowIngest(s.feed_block, max_packets=max_packets,
                     max_age_s=1e9)   # size-driven only
    for values, ts in packets_:
        acc.add(channels, values, ts)
    acc.flush()
    s.stop()
    return s
