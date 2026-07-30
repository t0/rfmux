"""
TLS 1/f noise generator: spectrum, common-mode behaviour, determinism.

The common-mode test is the load-bearing one — the slow and PFB streams
must see the SAME wander at the same absolute time, since physically it
is one resonator moving.
"""

import numpy as np
import pytest

from rfmux.mock.tls_noise import TLSNoiseGenerator


def _psd_slope(y, fs, f_lo, f_hi):
    """Least-squares log-log slope of the PSD over [f_lo, f_hi]."""
    from scipy.signal import welch
    f, p = welch(y, fs=fs, nperseg=min(len(y) // 8, 65536))
    band = (f >= f_lo) & (f <= f_hi) & (p > 0)
    return np.polyfit(np.log10(f[band]), np.log10(p[band]), 1)[0]


class TestSpectrum:
    @pytest.mark.parametrize("alpha", [0.5, 1.0])
    def test_psd_slope_matches_alpha(self, alpha):
        gen = TLSNoiseGenerator(n_resonators=1, fractional_rms=1e-6,
                                alpha=alpha, corner_hz=50.0, decades=3,
                                n_poles=8, seed=1)
        fs = 1.0 / gen.dt
        t = np.arange(int(400.0 * fs)) * gen.dt
        y = gen.values_at(t)[:, 0]
        # Fit inside the band, away from both corners
        slope = _psd_slope(y, fs, 0.5, 20.0)
        assert slope == pytest.approx(-alpha, abs=0.25), \
            f"alpha={alpha}: fitted slope {slope:.2f}"

    def test_rms_is_near_requested(self):
        target = 2e-7
        gen = TLSNoiseGenerator(n_resonators=1, fractional_rms=target,
                                alpha=1.0, corner_hz=50.0, seed=3)
        t = np.arange(int(200.0 / gen.dt)) * gen.dt
        y = gen.values_at(t)[:, 0]
        # A finite window under-samples the slowest poles, so the
        # realised RMS is a fraction of the band-integrated target.
        assert 0.2 * target < np.std(y) < 3.0 * target

    def test_white_noise_is_not_produced(self):
        """Sanity: the process must be strongly correlated, unlike the
        white noise already in the model."""
        gen = TLSNoiseGenerator(n_resonators=1, corner_hz=50.0, seed=5)
        t = np.arange(20000) * gen.dt
        y = gen.values_at(t)[:, 0]
        lag1 = np.corrcoef(y[:-1], y[1:])[0, 1]
        assert lag1 > 0.9, f"lag-1 autocorrelation {lag1:.3f} too low"


class TestCommonMode:
    def test_same_absolute_time_same_value_across_rates(self):
        """The slow (~kHz) and PFB (~MHz) streams query the same
        generator at different rates; at coincident times they must
        agree exactly."""
        gen = TLSNoiseGenerator(n_resonators=2, corner_hz=50.0, seed=7)
        slow_fs, fast_fs = 596.0, 1_220_703.125
        # Coincident instants: every slow sample is also a fast sample time
        t_slow = np.arange(200) / slow_fs
        slow_vals = gen.values_at(t_slow)
        fast_vals = np.array([gen.value_at(float(t)) for t in t_slow])
        assert np.allclose(slow_vals, fast_vals, rtol=0, atol=1e-18)

    def test_out_of_order_and_repeat_queries_agree(self):
        """PFB batches run ahead of their slow frame, so queries arrive
        out of order — that must not change any value."""
        gen = TLSNoiseGenerator(n_resonators=2, corner_hz=50.0, seed=11)
        t_probe = [0.5, 0.25, 0.75, 0.25, 0.5]
        first = {t: gen.value_at(t).copy() for t in (0.25, 0.5, 0.75)}
        for t in t_probe:
            assert np.array_equal(gen.value_at(t), first[t])

    def test_channels_are_independent(self):
        gen = TLSNoiseGenerator(n_resonators=2, corner_hz=50.0, seed=13)
        t = np.arange(5000) * gen.dt
        vals = gen.values_at(t)
        corr = abs(np.corrcoef(vals[:, 0], vals[:, 1])[0, 1])
        assert corr < 0.5, f"resonators correlated (r={corr:.2f})"


class TestDeterminism:
    def test_same_seed_same_realisation(self):
        a = TLSNoiseGenerator(n_resonators=1, corner_hz=50.0, seed=17)
        b = TLSNoiseGenerator(n_resonators=1, corner_hz=50.0, seed=17)
        t = np.arange(500) * a.dt
        assert np.allclose(a.values_at(t), b.values_at(t))

    def test_different_seed_differs(self):
        a = TLSNoiseGenerator(n_resonators=1, corner_hz=50.0, seed=17)
        b = TLSNoiseGenerator(n_resonators=1, corner_hz=50.0, seed=18)
        t = np.arange(500) * a.dt
        assert not np.allclose(a.values_at(t), b.values_at(t))


class TestMemoryBound:
    def test_history_is_trimmed(self):
        gen = TLSNoiseGenerator(n_resonators=1, corner_hz=50.0, seed=19,
                                max_history_s=1.0)
        gen.value_at(30.0)
        assert len(gen._values) <= int(1.0 / gen.dt) + 2
        # Still usable after trimming; old queries clamp rather than fail
        assert np.isfinite(gen.value_at(0.0)).all()
        assert np.isfinite(gen.value_at(30.5)).all()

    def test_zero_length_query(self):
        gen = TLSNoiseGenerator(n_resonators=3, corner_hz=50.0, seed=23)
        assert gen.values_at(np.array([])).shape == (0, 3)
