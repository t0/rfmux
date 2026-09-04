"""measure_df_calibrations against a synthetic board: one read per sweep
point for every channel, progress per point, and a slope that repeats
under noise because it is fitted, not differenced."""
import asyncio
import contextlib

import numpy as np
import pytest

from rfmux.algorithms.measurement.df_calibration import (
    _local_fit, measure_df_calibrations)
from rfmux.core.transferfunctions import VOLTS_PER_ROC

FR = {1: 1.000e9, 2: 1.050e9}
LINEWIDTH = 6e3


class _Board:
    """Two Lorentzian resonators, each biased at its own frequency, with
    readout noise and a slow drift of the resonance during a sweep."""

    def __init__(self, seed, noise_counts=40.0, drift_hz=300.0):
        self.rng = np.random.default_rng(seed)
        self.nco = 0.0
        self.freq = {ch: FR[ch] for ch in FR}
        self.noise = noise_counts
        self.drift = drift_hz
        self.reads = 0
        self.t = 0.0

    async def get_nco_frequency(self, module=1):
        return self.nco

    async def get_frequency(self, channel, module=1):
        return self.freq[channel]

    def tuber_context(self):
        board = self

        class Ctx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def set_frequency(self, f, channel, module=1):
                board.freq[channel] = f

            async def __call__(self):
                return None
        return Ctx()

    async def get_samples(self, n, channel=None, module=1):
        self.reads += 1
        self.t += 0.01
        i, q = [], []
        for ch in range(1, 1025):
            if ch in self.freq:
                fr = FR[ch] + self.drift * np.sin(2 * np.pi * self.t)
                x = 2 * (self.freq[ch] - fr) / LINEWIDTH
                s21 = 1 - 0.7 / (1 + 1j * x)
                base = 2e5 * s21
                i.append((base.real + self.rng.normal(0, self.noise, n)).tolist())
                q.append((base.imag + self.rng.normal(0, self.noise, n)).tolist())
            else:
                i.append([0.0] * n)
                q.append([0.0] * n)
        return {"i": i, "q": q}


def _measure(board, **kw):
    # The macro checks its first argument is a CRS; the function under
    # it is what the synthetic board exercises.
    return asyncio.run(measure_df_calibrations.__wrapped__(
        board, channels=[1, 2], module=1, **kw))


def test_one_read_per_point_and_progress_per_point():
    board = _Board(1)
    seen = []
    cals = _measure(board, span_hz=20e3, resolution_hz=500.0,
                    progress=lambda d, t: seen.append((d, t)))
    assert board.reads == 41
    assert seen == [(k, 41) for k in range(41)]
    assert set(cals) == {1, 2}
    assert board.freq == {1: FR[1], 2: FR[2]}, "channels restored"


def test_calibration_repeats_under_noise_and_drift():
    a = _measure(_Board(2))
    b = _Board(3)
    b = _measure(b)
    for ch in (1, 2):
        assert abs(b[ch]) / abs(a[ch]) == pytest.approx(1.0, abs=0.1)
        assert abs(np.degrees(np.angle(b[ch] / a[ch]))) < 5.0


def test_local_fit_returns_the_fitted_centre_and_leaves_the_edges():
    offsets = np.linspace(-10e3, 10e3, 41)
    clean = (offsets / 1e4) ** 2 + 1j * (offsets / 1e4)   # a cubic fits exactly
    iq = clean + 0.05 * (np.random.default_rng(0).normal(size=41)
                         + 1j * np.random.default_rng(1).normal(size=41))
    smooth = _local_fit(offsets, iq)
    centre = np.abs(offsets) <= 5e3
    assert np.array_equal(smooth[~centre], iq[~centre])
    # 21 points with 0.05 of noise each, fitted by four parameters: the
    # fit sits closer to the clean curve than the points did.
    resid = np.sqrt(np.mean(np.abs(smooth[centre] - clean[centre]) ** 2))
    noise = np.sqrt(np.mean(np.abs(iq[centre] - clean[centre]) ** 2))
    assert resid < 0.6 * noise
