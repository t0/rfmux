"""The linearised Kerr resonator about the solver's steady states:
rates, probe response and idler, against Rouble et al.
arXiv:2607.09178 and against the solver itself."""
import numpy as np
import pytest

from rfmux.mock import kerr


def test_rates_are_the_papers_eq_21_and_22(kerr_model):
    """Far from the fold both rates are kappa/2 with a beat; at the
    critical point (x0,c = sqrt(3)/(2 Qr) toward the shift, n_c =
    kappa / (sqrt(3) |K|)) the slow rate is zero and the fast one
    kappa, eq. 22."""
    env, i = kerr_model.env, kerr_model.i
    kappa, K, omega_r = env['kappa'][i], env['K'][i], env['omega_r'][i]
    # Weak drive, a linewidth above resonance: the linear resonator.
    a, b = kerr.coefficients(env, i, 1e-9 + 0j, omega_r + kappa)
    slow, fast = kerr.rates(a, b)
    assert slow.real == pytest.approx(-kappa / 2, rel=1e-6)
    assert fast.real == pytest.approx(-kappa / 2, rel=1e-6)
    assert abs(slow.imag) == pytest.approx(kappa, rel=1e-6)
    # The critical point.
    omega_c = omega_r + np.sign(K) * np.sqrt(3) / 2 * kappa
    n_c = kappa / (np.sqrt(3) * abs(K))
    a, b = kerr.coefficients(env, i, np.sqrt(n_c) + 0j, omega_c)
    slow, fast = kerr.rates(a, b)
    assert slow == pytest.approx(0.0, abs=1e-9 * kappa)
    assert fast == pytest.approx(-kappa, rel=1e-9)


def test_the_slow_rate_vanishes_at_the_solvers_fold(kerr_model):
    """Swept down at 0.01, the last steady state before the solver
    jumps is at the fold: the slow rate there is a small fraction of
    kappa/2, where a linewidth earlier the two rates were kappa/2 with
    a beat.  Forty linewidths of shift put the fold's splitting within
    the last few hundred hertz, so the approach is in 100 Hz steps."""
    env, i, solve = kerr_model.env, kerr_model.i, kerr_model.solve
    f_r = env['omega_r'][i] / (2 * np.pi)
    kappa = env['kappa'][i]

    def slow_rate(f, I):
        a, b = kerr.coefficients(env, i, I[i], 2 * np.pi * f)
        return kerr.rates(a, b)[0].real

    def sweep(f0, step, seed):
        last = None
        for f in f0 - np.arange(0, 4e5, step):
            I = solve(f, 0.01, seed)
            if last is not None and abs(I[i]) < 0.5 * abs(last[1][i]):
                return last
            last = (f, I)
            seed = I
        raise AssertionError("no jump")
    coarse = sweep(f_r, 1e3, None)
    f_fold, I_fold = sweep(coarse[0], 1e2, coarse[1])
    assert abs(slow_rate(f_fold, I_fold)) < 0.25 * kappa / 2
    f_before = f_fold + kappa / (2 * np.pi)
    seed = None
    for f in f_r - np.arange(0, f_r - f_before + 1, 1e3):
        seed = solve(f, 0.01, seed)
    assert abs(slow_rate(f, seed)) > 0.5 * kappa / 2


def test_the_dc_probe_response_is_the_solvers_slope(kerr_model):
    """At Omega = 0 the probe and idler deviations are both static and
    their sum is the change of the steady state per unit drive, which
    the solver gives by finite difference: to 20%, the derivative of a
    response the Kerr model matches to 3% in value.  On the upper state
    near the fold the probe's deviation exceeds the linear response
    several times over (the paper's responsivity enhancement)."""
    env, i, solve = kerr_model.env, kerr_model.i, kerr_model.solve
    f_r = env['omega_r'][i] / (2 * np.pi)
    for f, amp in ((f_r, 0.003), (f_r - 1.0e5, 0.01)):
        seed = None
        if f != f_r:
            for fs in f_r - np.arange(0, f_r - f + 1, 5e3):
                seed = solve(fs, amp, seed)
        I0 = solve(f, amp, seed)
        d = 1e-3 * amp
        slope = (solve(f, amp + d, I0)[i] - solve(f, amp - d, I0)[i]) / (2 * d)
        a, b = kerr.coefficients(env, i, I0[i], 2 * np.pi * f)
        u_plus, u_minus = kerr.probe_response(a, b, env['D'][i], 0.0)
        assert u_plus + u_minus == pytest.approx(slope, rel=0.2)
    assert kerr.probe_gain(env, i, I0[i], 2 * np.pi * f, 0.0) > 3


def test_the_idler_needs_the_pump(kerr_model):
    """The idler is the pump's doing: it vanishes with the pump, and
    with a pump it is the probe's deviation times |b| / |i Omega - a*|."""
    env, i, m = kerr_model.env, kerr_model.i, kerr_model.m
    omega_g = env['omega_r'][i] - env['kappa'][i]
    a, b = kerr.coefficients(env, i, 0j, omega_g)
    assert b == 0
    assert kerr.probe_response(a, b, env['D'][i], 1e4)[1] == 0
    assert kerr.probe_gain(env, i, 0j, omega_g, 1e4) == pytest.approx(1.0)
    I0 = 4e-3 * m.Istar * np.exp(0.3j)
    a, b = kerr.coefficients(env, i, I0, omega_g)
    u_plus, u_minus = kerr.probe_response(a, b, env['D'][i], 1e4)
    assert abs(u_minus) == pytest.approx(
        abs(b) * abs(u_plus) / abs(1j * 1e4 - np.conj(a)), rel=1e-9)
    assert abs(u_minus) > 0.1 * abs(u_plus)
