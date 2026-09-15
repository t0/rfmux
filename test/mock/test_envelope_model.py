"""The Kerr resonator each mock circuit is near its resonance, in the
solver's current: extracted from the linear response and checked
against the steady-state solver."""
import asyncio
import contextlib
import io

import numpy as np
import pytest

from rfmux.mr_resonator import jit_physics as jp


def _model(n=3):
    from rfmux.mock.crs import ServerMockCRS
    crs = ServerMockCRS("0000")
    with contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(crs.generate_resonators(
            {"num_resonances": n, "resonator_random_seed": 5,
             "auto_bias_kids": False}))
    m = crs._resonator_model
    m.nqp_noise_enabled = False
    m._tls_generator = None
    m._ensure_arrays()
    # The base Lk and R of the QP model at rest, as every evaluation
    # installs them (the generation's differ by a few parts in 1e3).
    m._compute_nqp_state(0.0)
    return m


def _solver(m):
    base_Lk, base_R, base_Lg = m._base_arrays()
    L0 = base_Lk + base_Lg + m.L_junk_array
    k0 = m.mr_lekids[0]

    def solve(f, amp, seed=None):
        return jp.converged_lekid_parameters(
            float(f), amp, L0, base_R, m.C_array, m.Cc_array, base_Lk,
            base_Lg, m.L_junk_array, k0.input_atten_dB, complex(k0.ZLNA),
            m.Istar, 1e-12, 500, initial_currents=seed)[2]
    return solve


def _cubic(env, i, f, amp):
    """|I| of the steady states of the Kerr resonator i at f."""
    Delta = 2 * np.pi * f - env['omega_r'][i]
    K, kappa = env['K'][i], env['kappa'][i]
    roots = np.roots([K ** 2, -2 * K * Delta, (kappa / 2) ** 2 + Delta ** 2,
                      -abs(env['D'][i] * amp) ** 2])
    real = roots[np.abs(roots.imag) < 1e-6 * np.abs(roots).max()].real
    return np.sqrt(np.sort(real[real > 0]))


def test_the_solver_current_at_a_negligible_drive_is_the_linear_response():
    """The iterate stops once the inductance stops changing; the current
    returned is the one that inductance carries, so at a drive too small
    to move it the current is the linear response, not a damped step
    toward it."""
    m = _model()
    solve = _solver(m)
    i = int(np.argsort(m.resonator_frequencies)[1])
    f = float(m.resonator_frequencies[i])
    base_Lk, base_R, base_Lg = m._base_arrays()
    L0 = base_Lk + base_Lg + m.L_junk_array
    k0 = m.mr_lekids[0]
    lin = jp.linear_currents(np.array([f]), L0[i], base_R[i], m.C_array[i],
                             m.Cc_array[i], k0.input_atten_dB,
                             complex(k0.ZLNA))[0]
    assert solve(f, 1e-9)[i] / 1e-9 == pytest.approx(lin, rel=1e-6)


def test_the_kerr_cubic_matches_the_solver_on_resonance_and_at_the_fold():
    """Below bifurcation the cubic's current on resonance is the
    solver's to a few percent; above it, the cubic's bistable region
    ends where the solver, swept down, jumps to the low state."""
    m = _model()
    env = m.envelope_parameters()
    solve = _solver(m)
    i = int(np.argsort(m.resonator_frequencies)[1])
    f_r = env['omega_r'][i] / (2 * np.pi)
    assert env['K'][i] < 0                     # a softening nonlinearity
    assert 1e4 < env['kappa'][i] / (2 * np.pi) < 1e6
    for amp in (0.001, 0.003):
        got = abs(solve(f_r, amp)[i])
        assert _cubic(env, i, f_r, amp).max() == pytest.approx(got, rel=0.03)
    # The solver's jump, sweeping down at 0.01 in 1 kHz steps.
    seed, prev, jump = None, None, None
    for f in f_r - np.arange(0, 4e5, 1e3):
        I = solve(f, 0.01, seed)
        seed = I
        if prev is not None and abs(I[i]) < 0.5 * abs(prev):
            jump = f
            break
        prev = I[i]
    assert jump is not None
    fs = f_r - np.arange(0, 4e5, 1e2)
    bistable = [f for f in fs if len(_cubic(env, i, f, 0.01)) == 3]
    assert min(bistable) == pytest.approx(jump, abs=0.03 * (f_r - jump))


def test_the_output_coupling_reproduces_the_dip():
    """S21 on resonance is the through transmission plus c times the
    resonant current: the affine output relation gives the dip."""
    m = _model()
    env = m.envelope_parameters()
    i = int(np.argsort(m.resonator_frequencies)[1])
    f_r = env['omega_r'][i] / (2 * np.pi)
    base_Lk, base_R, base_Lg = m._base_arrays()
    L0 = base_Lk + base_Lg + m.L_junk_array
    k0 = m.mr_lekids[0]
    on = jp.s21_of_one(np.array([f_r]), L0[i], base_R[i], m.C_array[i],
                       m.Cc_array[i], complex(k0.ZLNA), k0.GLNA,
                       k0.input_atten_dB, k0.system_termination)[0]
    peak = env['D'][i] / (env['kappa'][i] / 2)
    assert abs(env['S21_bg'][i]) == pytest.approx(1.0, abs=1e-3)
    assert abs(on) < 0.5                       # a real dip
    assert env['S21_bg'][i] + env['c'][i] * peak == pytest.approx(on, rel=1e-6)
