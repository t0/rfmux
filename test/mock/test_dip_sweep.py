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
import pytest


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


def test_two_stage_search_finds_the_brute_force_dip():
    """The locating pass at 50 kHz plus multisweep's 200 kHz / 101
    points lands on the same minimum a dense sweep over the whole
    coupling-shift window finds, to the fine grid's step."""
    from rfmux.mock.config import BIAS_DBM, bias_amplitude_from_dbm
    crs, m = _model()
    amp = bias_amplitude_from_dbm(BIAS_DBM)
    for f0 in sorted(m.resonator_frequencies):
        dense = np.linspace(f0 - 3e6, f0 + 3e6, 3001)
        ref = dense[np.argmin(m.s21_sweep(dense, amp))]
        found = crs._find_s21_dip_frequency(f0, amp)
        assert abs(found - ref) <= 2e3, (f0, found, ref)
        assert abs(found - f0) > 200e3, \
            "the fixture should exercise the coupling shift"


def test_bias_power_round_trips_through_dbm():
    from rfmux.mock.config import (
        BIAS_DBM, bias_amplitude_from_dbm, bias_dbm_from_amplitude, defaults)
    amp = defaults()["bias_amplitude"]
    assert bias_dbm_from_amplitude(amp) == pytest.approx(BIAS_DBM)
    assert bias_amplitude_from_dbm(bias_dbm_from_amplitude(0.01)) == \
        pytest.approx(0.01)


def test_sweep_leaves_lekid_state_alone():
    crs, m = _model()
    f0 = sorted(m.resonator_frequencies)[0]
    m.s21_lc_response(f0, 0.01)
    before = [(k.Lk, k.R, k.L) for k in m.mr_lekids]
    m.s21_sweep(np.linspace(f0 - 3e6, f0 + 3e6, 50), 0.01)
    assert [(k.Lk, k.R, k.L) for k in m.mr_lekids] == before
