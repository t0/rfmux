"""
The trained sigma is the samples' scatter about the baseline.

It used to be the sigma of adjacent differences over sqrt(2), exact for
white noise and low by 1.3x to 1.6x on the CIC-decimated slow stream,
whose neighbouring samples are correlated.  These pin the estimator on
the noise the streams actually carry: correlated, drifting, and with
pulses in the training record.
"""

import numpy as np
import pytest

from rfmux.pulse_capture.detection import (
    _block_median_baseline, estimate_noise_stats)

N = 60_000


def _complex(re, im):
    return (np.asarray(re, float) + 1j * np.asarray(im, float)).astype(
        np.complex128)


def _cic_noise(rng, n, stages=6, r=16, sigma=1.0):
    """White noise through an *stages*-fold boxcar of length *r*,
    decimated by r, scaled to unit variance: the slow stream's shape."""
    h = np.array([1.0])
    for _ in range(stages):
        h = np.convolve(h, np.ones(r))
    x = np.convolve(rng.normal(0, 1, n * r + len(h)), h)[len(h):][::r][:n]
    return sigma * x / np.sqrt(np.dot(h, h))


def _sigma(arr, **kw):
    stats, _ = estimate_noise_stats({1: arr}, [1], **kw)
    return stats[1].std_I, stats[1].std_Q


def test_white_noise_is_unchanged():
    rng = np.random.default_rng(1)
    s_i, s_q = _sigma(_complex(rng.normal(0, 1, N), rng.normal(0, 2, N)),
                      baseline_block=2000)
    assert s_i == pytest.approx(1.0, rel=0.03)
    assert s_q == pytest.approx(2.0, rel=0.03)


def test_correlated_noise_reads_its_true_scatter():
    """The regression: on CIC-shaped noise the difference estimator
    reported 0.62 for unit sigma."""
    rng = np.random.default_rng(2)
    x = _cic_noise(rng, N)
    assert np.std(x) == pytest.approx(1.0, rel=0.05), "fixture"
    s_i, _ = _sigma(_complex(x, rng.normal(0, 1, N)), baseline_block=2000)
    assert s_i == pytest.approx(1.0, rel=0.05)


def test_drift_and_pulses_are_not_noise():
    rng = np.random.default_rng(3)
    k = np.arange(N)
    x = rng.normal(0, 1, N) + 8.0 * np.sin(2 * np.pi * k / 20_000)
    # 2% of the record under 12-sigma pulses with exponential tails.
    # Their tails are scatter about the baseline as the engine sees it,
    # so a few percent of inflation is the true answer, not an error.
    for start in rng.integers(0, N - 120, 10):
        x[start:start + 120] += 12.0 * np.exp(-np.arange(120) / 80)
    s_i, _ = _sigma(_complex(x, rng.normal(0, 1, N)), baseline_block=1000)
    assert s_i == pytest.approx(1.0, rel=0.08)


def test_frozen_baseline_is_one_median():
    x = np.arange(1000, dtype=float)
    assert np.all(_block_median_baseline(x, 0) == np.median(x))


def test_baseline_tracks_the_window_and_holds_the_ends():
    x = np.concatenate([np.zeros(500), np.full(500, 10.0)])
    b = _block_median_baseline(x, 100)
    assert b[0] == 0.0 and b[-1] == 10.0
    assert 0.0 < b[500] < 10.0
    # A block longer than the record is one median.
    assert np.all(_block_median_baseline(x, 5000) == np.median(x))
