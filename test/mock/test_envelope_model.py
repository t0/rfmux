"""The Kerr resonator each mock circuit is near its resonance, in the
solver's current: extracted from the linear response and checked
against the steady-state solver."""
import numpy as np
import pytest

from rfmux.mr_resonator import jit_physics as jp


def test_the_solver_current_at_a_negligible_drive_is_the_linear_response(kerr_model):
    """The iterate stops once the inductance stops changing; the current
    returned is the one that inductance carries, so at a drive too small
    to move it the current is the linear response, not a damped step
    toward it."""
    km = kerr_model
    i, m = km.i, km.m
    f = float(m.resonator_frequencies[i])
    lin = jp.linear_currents(np.array([f]), km.L0[i], km.base_R[i],
                             m.C_array[i], m.Cc_array[i],
                             km.k0.input_atten_dB, complex(km.k0.ZLNA))[0]
    assert km.solve(f, 1e-9)[i] / 1e-9 == pytest.approx(lin, rel=1e-6)


def test_the_kerr_cubic_matches_the_solver_on_resonance_and_at_the_fold(kerr_model):
    """For every resonator: below bifurcation the cubic's current on
    resonance is the solver's to a few percent; above it, the cubic's
    bistable region ends where the solver, swept down, jumps to the
    low state."""
    km = kerr_model
    env = km.env
    for j in range(len(km.m.mr_lekids)):
        f_r = env['omega_r'][j] / (2 * np.pi)
        assert env['K'][j] < 0                 # a softening nonlinearity
        assert 1e4 < env['kappa'][j] / (2 * np.pi) < 1e6
        for amp in (0.001, 0.003):
            got = abs(km.solve(f_r, amp)[j])
            assert km.cubic(f_r, amp, j).max() == pytest.approx(got, rel=0.03)
        # The solver's jump, sweeping down at 0.01 in 1 kHz steps.
        seed, prev, jump = None, None, None
        for f in f_r - np.arange(0, 4e5, 1e3):
            I = km.solve(f, 0.01, seed)
            seed = I
            if prev is not None and abs(I[j]) < 0.5 * abs(prev):
                jump = f
                break
            prev = I[j]
        assert jump is not None
        fs = f_r - np.arange(0, 4e5, 1e2)
        bistable = [f for f in fs if len(km.cubic(f, 0.01, j)) == 3]
        assert min(bistable) == pytest.approx(jump, abs=0.03 * (f_r - jump))


def test_the_output_coupling_reproduces_the_dip(kerr_model):
    """S21 on resonance is the through transmission plus c times the
    resonant current: the affine output relation gives the dip."""
    km = kerr_model
    env, i, m = km.env, km.i, km.m
    f_r = env['omega_r'][i] / (2 * np.pi)
    on = jp.s21_of_one(np.array([f_r]), km.L0[i], km.base_R[i], m.C_array[i],
                       m.Cc_array[i], complex(km.k0.ZLNA), km.k0.GLNA,
                       km.k0.input_atten_dB, km.k0.system_termination)[0]
    peak = env['D'][i] / (env['kappa'][i] / 2)
    assert abs(env['S21_bg'][i]) == pytest.approx(1.0, abs=1e-3)
    assert abs(on) < 0.5                       # a real dip
    assert env['S21_bg'][i] + env['c'][i] * peak == pytest.approx(on, rel=1e-6)
