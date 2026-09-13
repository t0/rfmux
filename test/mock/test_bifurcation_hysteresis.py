"""A bifurcated resonance in the mock is hysteretic, as a real one is:
a sweep down rides the deep branch to the fold, a sweep up jumps at
the other fold, and below the fold the two directions agree."""

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
    fgen = sorted(m.resonator_frequencies)[1]
    wide = np.linspace(fgen - 3e6, fgen + 3e6, 3001)
    f0 = float(wide[np.argmin(m.s21_sweep(wide, 0.001))])
    return m, f0


def _sweep(m, points, amp):
    return np.array([abs(m.s21_lc_response(float(f), amp)) for f in points])


def test_up_and_down_agree_below_the_fold():
    m, f0 = _model()
    grid = np.linspace(f0 - 3e5, f0 + 1e5, 81)
    up = _sweep(m, grid, 0.003)
    down = _sweep(m, grid[::-1], 0.003)[::-1]
    # To the solver's residual, amplified on the dip's steep side.
    np.testing.assert_allclose(up, down, atol=1e-3)
    assert up.min() < 0.5


def test_a_bifurcated_resonance_is_hysteretic():
    """Amplitude 0.01 bifurcates this resonator between about -205 and
    -65 kHz: the way down keeps the deep dip to the lower fold, the way
    up sees the shallow side and jumps at the upper one."""
    m, f0 = _model()
    grid = np.linspace(f0 - 3e5, f0 + 1e5, 81)          # 5 kHz steps
    up = _sweep(m, grid, 0.01)
    down = _sweep(m, grid[::-1], 0.01)[::-1]
    assert down.min() < up.min() - 0.3
    assert grid[np.argmin(down)] < grid[np.argmin(up)] - 50e3
    jump_up = grid[np.argmax(np.abs(np.diff(up)))]
    jump_down = grid[np.argmax(np.abs(np.diff(down)))]
    assert jump_down < jump_up - 50e3
    # The lower fold is at -204.9 kHz: the deep branch is held to it,
    # not left a grid step or two early.
    assert jump_down < f0 - 200e3
    # The cache serves each direction its own branch on a repeat.
    np.testing.assert_allclose(_sweep(m, grid[::-1], 0.01)[::-1], down,
                               atol=1e-6)
    np.testing.assert_allclose(_sweep(m, grid, 0.01), up, atol=1e-6)


def test_the_batched_sweep_takes_the_same_branch():
    m, f0 = _model()
    grid = np.linspace(f0 - 3e5, f0 + 1e5, 81)
    swept = m.s21_sweep(grid[::-1], 0.01)[::-1]
    m._branch_memory.clear()
    np.testing.assert_allclose(swept, _sweep(m, grid[::-1], 0.01)[::-1],
                               rtol=1e-9)


def test_each_module_keeps_its_own_branches():
    """Channel 1 of module 1 sweeps down through the bifurcation while
    channel 1 of module 2 sits on another resonator, the modules taking
    turns as the streamer has them: module 1 keeps its deep branch."""
    m, f0 = _model()
    crs = m.mock_crs
    grid = np.linspace(f0 - 3e5, f0 + 1e5, 81)
    down = _sweep(m, grid[::-1], 0.01)[::-1]
    m._branch_memory.clear()
    m._convergence_cache.clear()
    other = sorted(m.resonator_frequencies)[0]
    fs = 625e6 / 256 / 64
    for mod in (1, 2):
        crs._nco_frequencies[mod] = 0.0
        crs._amplitudes[(mod, 1)] = 0.01 if mod == 1 else 0.001
        crs._phases[(mod, 1)] = 0.0
    crs._frequencies[(2, 1)] = other
    seen = []
    for f in grid[::-1]:
        crs._frequencies[(1, 1)] = f
        seen.append(m.calculate_module_response_coupled(
            1, num_samples=2, sample_rate=fs)[1][0])
        m.calculate_module_response_coupled(2, num_samples=2, sample_rate=fs)
    np.testing.assert_allclose(np.abs(seen[::-1]) / 0.01, down, atol=1e-3)


def test_a_moved_tone_is_followed_in_sub_steps_only_where_it_jumps_branch(monkeypatch):
    """One seeded solve per point where the current moves smoothly (a
    netanal, a dip search); the sub-steps only where one step from the
    previous point lands on the other branch."""
    m, f0 = _model()
    calls = []
    real = jp.converged_lekid_parameters

    def counting(*a, **k):
        calls.append(a[0])
        return real(*a, **k)
    monkeypatch.setattr(jp, "converged_lekid_parameters", counting)
    far = f0 + 3e6
    _sweep(m, [far, far + 1e4], 0.01)            # a netanal's 10 kHz step
    assert len(calls) == 2
    calls.clear()
    _sweep(m, np.arange(f0 + 1e5, f0 - 1.95e5 - 1, -5e3), 0.01)
    n_ride = len(calls)
    calls.clear()
    _sweep(m, [f0 - 2.1e5], 0.01)                # across the fold at -204.9 kHz
    assert len(calls) > 1
    assert n_ride < 2 * 60                        # 60 points, few retaken


def test_a_tone_switched_off_leaves_its_resonator_at_rest():
    """Inside the bistable region a tone that arrived from above is on
    the deep branch; switched off and back on at the same frequency it
    finds the resonator at rest, on the low branch."""
    m, f0 = _model()
    crs = m.mock_crs
    fs = 625e6 / 256 / 64
    crs._nco_frequencies[1] = 0.0
    crs._phases[(1, 1)] = 0.0
    inside = f0 - 1.5e5                       # bistable at 0.01: -205..-65 kHz
    for f in np.arange(f0 + 1e5, inside - 1, -5e3):
        crs._frequencies[(1, 1)] = float(f)
        crs._amplitudes[(1, 1)] = 0.01
        deep = m.calculate_module_response_coupled(
            1, num_samples=2, sample_rate=fs)[1][0]
    crs._amplitudes[(1, 1)] = 0.0
    m.calculate_module_response_coupled(1, num_samples=2, sample_rate=fs)
    crs._amplitudes[(1, 1)] = 0.01
    back = m.calculate_module_response_coupled(
        1, num_samples=2, sample_rate=fs)[1][0]
    m._branch_memory.clear()
    m._convergence_cache.clear()
    rest = _sweep(m, [inside], 0.01)[0]
    assert abs(deep) / 0.01 < rest - 0.3
    assert abs(back) / 0.01 == pytest.approx(rest, abs=1e-3)


def test_the_seeded_solver_holds_the_deep_branch_and_converges():
    """Where the fixed damping ran to its cap: seeded from the previous
    point, the adaptive step converges on the deep branch in a few
    dozen iterations at most."""
    m, f0 = _model()
    m._extract_param_arrays()
    n = len(m.mr_lekids)
    base_Lk = np.array([m.base_lekid_params[i]["Lk"] for i in range(n)])
    base_Lg = np.array([m.base_lekid_params[i]["Lg"] for i in range(n)])
    k0 = m.mr_lekids[0]
    L, R = m.L_array.copy(), m.R_array.copy()
    I = np.zeros(n, dtype=complex)
    worst = 0
    for f in np.linspace(f0 + 1e5, f0 - 4e5, 251):
        L, R, I, its = jp.converged_lekid_parameters(
            float(f), 0.03, L, R, m.C_array, m.Cc_array, base_Lk, base_Lg,
            m.L_junk_array, k0.input_atten_dB, complex(k0.ZLNA), m.Istar,
            1e-9, 500, initial_currents=I)
        worst = max(worst, its)
    assert worst < 100
    # At the last point the seeded solve is still on the deep branch,
    # carrying several times the current a solve from rest lands on;
    # both are a small fraction of Istar, a shift of a linewidth
    # needing a 1e-4 change in Lk.
    _, _, at_rest, _ = jp.converged_lekid_parameters(
        float(f), 0.03, L, R, m.C_array, m.Cc_array, base_Lk, base_Lg,
        m.L_junk_array, k0.input_atten_dB, complex(k0.ZLNA), m.Istar,
        1e-9, 500)
    assert abs(I[1]) > 4 * abs(at_rest[1])
