"""A slow-stream capture file for the review tests: noise on every
channel, with *events* (a key's list of start times in seconds) as
decaying pulses on it, and the tuning rows given."""

import numpy as np

from rfmux.pulse_capture.capture_session import (PulseCaptureConfig,
                                                 PulseCaptureSession)

FS = 20000.0


def capture_file(path, channels, module, *, tuning=None, events=None,
                 seconds=0.3, seed=1):
    s = PulseCaptureSession(
        channels=list(channels), module=module, sample_rate=FS,
        hdf5_path=path, tuning=tuning,
        **PulseCaptureConfig(threshold_sigma=5.0, end_sigma=1.5,
                             max_pulse_ms=50.0,
                             noise_train_ms=20.0).session_kwargs(FS))
    rng = np.random.default_rng(seed)
    s.start()
    n = int(seconds * FS)
    t = np.arange(n) / FS
    for key in channels:
        sig = rng.normal(0, 1.0, n)
        for t0 in (events or {}).get(key, ()):
            mask = t >= t0
            sig[mask] += 50.0 * np.exp(-(t[mask] - t0) / 1e-3)
        s.feed_block(key, sig, rng.normal(0, 1.0, n), 43000.0 + t)
    s.stop()
    return path
