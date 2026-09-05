"""measure_df_calibrations against a synthetic board: one read per sweep
point for every channel, progress per point, and a slope that repeats
under noise because it is fitted, not differenced."""
import asyncio

import numpy as np
import pytest

from rfmux.algorithms.measurement.df_calibration import (
    df_calibration_from_sweep, measure_df_calibrations)
from rfmux.algorithms.measurement.fitting import identify_bifurcation
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
        return self.freq.get(channel)

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
        return _Samples(i, q)


class _Samples:
    def __init__(self, i, q):
        self.i, self.q = i, q


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


def test_a_bifurcated_channel_is_warned_about_and_still_reported():
    """Too much bias power bifurcates the resonance: the sweep steps
    across a jump and no slope there means much.  The caveat is said;
    the number is still reported, since the channel is still biased."""
    board = _Board(4)
    real_get = board.get_samples

    async def jumping(n, channel=None, module=1):
        s = await real_get(n, channel=channel, module=module)
        # Channel 2 lands on the other branch past the bias point.
        if board.freq[2] > FR[2]:
            s.q[1] = [v + 6e5 for v in s.q[1]]
        return s
    board.get_samples = jumping
    with pytest.warns(UserWarning, match="channel 2.*bifurcated"):
        cals = _measure(board)
    assert 1 in cals and 2 in cals


def test_the_flow_detector_tells_a_jump_from_a_steep_resonance():
    f = np.linspace(-10e3, 10e3, 41)
    smooth = 1 - 0.7 / (1 + 2j * f / 6e3)
    assert not identify_bifurcation(smooth)
    jumped = smooth.copy()
    jumped[21:] += 2j                    # the other branch
    assert identify_bifurcation(jumped)


@pytest.mark.parametrize("span, n, skew, off_fr", [
    (20e3, 41, 0.0, 0.0), (20e3, 41, 0.02, 0.0), (20e3, 41, 0.0, 1e3),
    (200e3, 101, 0.0, 0.0)])
def test_fitted_slope_is_exact_on_a_resonance(span, n, skew, off_fr):
    """The nonlinear resonator model's slope at the bias, with a sloped
    baseline, off the dip, and at multisweep's coarse sampling."""
    fr, lw = 1.0e9, 6e3
    fb = fr + off_fr
    f = np.linspace(fb - span / 2, fb + span / 2, n)

    def s21(ff):
        return (1 + skew * (ff - fr) / 1e5) * (1 - 0.7 / (1 + 2j * (ff - fr) / lw))
    # The sweep is handed over in counts; the calibration is per volt.
    got = 1.0 / df_calibration_from_sweep(f, s21(f) / VOLTS_PER_ROC, fb)
    exact = (s21(fb + 0.01) - s21(fb - 0.01)) / 0.02
    assert abs(got) / abs(exact) == pytest.approx(1.0, abs=1e-2)
    assert abs(np.degrees(np.angle(got / exact))) < 0.5


def test_a_channel_with_no_tone_is_skipped_with_a_warning():
    board = _Board(5)
    with pytest.warns(UserWarning, match="channel 3 has no tone"):
        cals = asyncio.run(measure_df_calibrations.__wrapped__(
            board, channels=[1, 3], module=1))
    assert set(cals) == {1}
    assert board.freq == {1: FR[1], 2: FR[2]}, "no channel was moved but 1"
