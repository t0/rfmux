"""The batched S21 sweep is the single-point path, minus the noise.

The mock's auto-bias finds each resonator's transmission minimum by
sweeping |S21| over thousands of points.  s21_sweep exists so that does
not pay the whole single-point path per point; these tests pin that it
reports what that path would have.
"""

import asyncio
import contextlib
import io

import numpy as np


def _model(n=3):
    from rfmux.mock.crs import ServerMockCRS
    crs = ServerMockCRS("0000")
    with contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(crs.generate_resonators(
            {"num_resonances": n, "resonator_random_seed": 5,
             "auto_bias_kids": False}))
    m = crs._resonator_model
    # The per-point path draws QP noise and TLS on top of the same
    # physics; take both out so the curves can be compared exactly.
    m.nqp_noise_enabled = False
    m._tls_generator = None
    return crs, m


def test_sweep_matches_per_point_path():
    crs, m = _model()
    f0 = sorted(m.resonator_frequencies)[1]
    grid = np.linspace(f0 - 3e6, f0 + 3e6, 400)
    amp = 0.01
    swept = m.s21_sweep(grid, amp)
    single = np.array([abs(m.s21_lc_response(f, amp)) for f in grid])
    np.testing.assert_allclose(swept, single, rtol=1e-9)
    assert swept.min() < 0.95 * swept.max(), "grid should straddle a dip"


def test_sweep_on_a_fresh_model_is_not_a_different_model():
    """A fresh model still holds generation-time Lk; the single-point
    path replaces it with the QP state's before converging.  The sweep
    must start from the same place, or the dip lands over a MHz off."""
    crs, m = _model()
    f0 = sorted(m.resonator_frequencies)[0]
    assert m._nqp_state_t is None
    fresh = crs._find_s21_dip_frequency(f0, 0.01)
    m.s21_lc_response(f0, 0.01)
    assert crs._find_s21_dip_frequency(f0, 0.01) == fresh


def test_sweep_leaves_lekid_state_alone():
    crs, m = _model()
    f0 = sorted(m.resonator_frequencies)[0]
    m.s21_lc_response(f0, 0.01)
    before = [(k.Lk, k.R, k.L) for k in m.mr_lekids]
    m.s21_sweep(np.linspace(f0 - 3e6, f0 + 3e6, 50), 0.01)
    assert [(k.Lk, k.R, k.L) for k in m.mr_lekids] == before
