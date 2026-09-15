"""Fixtures shared by the mock physics tests."""
import asyncio
import contextlib
import io
import types

import numpy as np
import pytest

from rfmux.mr_resonator import jit_physics as jp


@pytest.fixture
def kerr_model():
    """Seed 5, three resonators, noise off, the QP model's rest state
    installed as every evaluation installs it (the generation's base
    Lk differs by a few parts in 1e3): the model, its Kerr envelope
    parameters, the index of the middle resonator, a seeded solver
    returning the currents, and the Kerr cubic's |I| at a point."""
    from rfmux.mock.crs import ServerMockCRS
    crs = ServerMockCRS("0000")
    with contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(crs.generate_resonators(
            {"num_resonances": 3, "resonator_random_seed": 5,
             "auto_bias_kids": False}))
    m = crs._resonator_model
    m.nqp_noise_enabled = False
    m._tls_generator = None
    m._ensure_arrays()
    m._compute_nqp_state(0.0)
    base_Lk, base_R, base_Lg = m._base_arrays()
    L0 = base_Lk + base_Lg + m.L_junk_array
    k0 = m.mr_lekids[0]

    def solve(f, amp, seed=None):
        return jp.converged_lekid_parameters(
            float(f), amp, L0, base_R, m.C_array, m.Cc_array, base_Lk,
            base_Lg, m.L_junk_array, k0.input_atten_dB, complex(k0.ZLNA),
            m.Istar, 1e-12, 500, initial_currents=seed)[2]

    env = m.envelope_parameters()
    i = int(np.argsort(m.resonator_frequencies)[1])

    def cubic(f, amp):
        Delta = 2 * np.pi * f - env['omega_r'][i]
        K, kappa = env['K'][i], env['kappa'][i]
        roots = np.roots([K ** 2, -2 * K * Delta,
                          (kappa / 2) ** 2 + Delta ** 2,
                          -abs(env['D'][i] * amp) ** 2])
        real = roots[np.abs(roots.imag) < 1e-6 * np.abs(roots).max()].real
        return np.sqrt(np.sort(real[real > 0]))

    return types.SimpleNamespace(m=m, env=env, i=i, solve=solve, cubic=cubic,
                                 L0=L0, base_R=base_R, k0=k0)
