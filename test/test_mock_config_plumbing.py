"""
Mock config plumbing regressions.

Two bugs found while designing the TLS/1-f noise model:

1. Several sites read ``mock_crs.physics_config`` (no underscore) — an
   attribute that does not exist — so the dialog's cache-tuning
   settings and ``get_samples``' scale factor were silently ignored and
   the hardcoded fallbacks always won.
2. ``update_qp_densities_for_time`` is a monotonic ratchet, and with PFB
   enabled its batches advance the clock past the slow frame — so the
   slow emitter must pass ``pulse_time`` explicitly or its samples get
   evaluated at the PFB's (later) time.
"""

import inspect

import pytest


def test_no_dead_physics_config_references():
    """The attribute is _physics_config; a bare 'physics_config' lookup
    silently returns the default forever."""
    from rfmux.mock import crs as crs_mod
    from rfmux.mock import resonator_model as model_mod

    for module in (crs_mod, model_mod):
        src = inspect.getsource(module)
        for pattern in ("'physics_config'", '"physics_config"'):
            for line in src.splitlines():
                if pattern in line and "_physics_config" not in line:
                    pytest.fail(f"{module.__name__}: dead lookup: "
                                f"{line.strip()}")


def test_cache_tuning_reads_the_real_attribute():
    """The convergence-cache steps come from _compute_cache_key_params;
    they must read _physics_config so dialog settings take effect."""
    from rfmux.mock.resonator_model import MockResonatorModel

    src = inspect.getsource(
        MockResonatorModel._compute_cache_key_params)
    assert "_physics_config" in src
    assert "cache_freq_step" in src
    for line in src.splitlines():
        if "physics_config" in line:
            assert "_physics_config" in line, line.strip()


def test_slow_emitter_passes_explicit_pulse_time():
    """Otherwise the slow stream inherits the PFB batches' later clock."""
    from rfmux.mock import udp_streamer

    src = inspect.getsource(udp_streamer.MockCRSStreamer._emit_slow_packet)
    assert "pulse_time=t_frame" in src, \
        "slow emitter must pin pulse_time to its own frame time"


def test_ratchet_still_guards_pulse_generation():
    """The monotonic guard stays — it protects the Poisson dt draw from
    being double-counted by out-of-order calls."""
    from rfmux.mock.resonator_model import MockResonatorModel

    src = inspect.getsource(
        MockResonatorModel.update_qp_densities_for_time)
    assert "<= self.last_update_time" in src


def test_qp_noise_does_not_defeat_the_convergence_cache():
    """White QP noise is a fresh draw per call; folding it into the nqp
    used for the cache key made every call a miss (85% -> 4% hit rate
    at 10% noise, a 12x slowdown). It must be applied as a post-cache
    perturbation instead, so the hit rate is independent of it."""
    import asyncio

    from rfmux.mock.crs import ServerMockCRS

    rates = {}
    for noise in (0.001, 0.1):
        crs = ServerMockCRS("0000")
        asyncio.run(crs.generate_resonators({
            "num_resonances": 2, "resonator_random_seed": 3,
            "auto_bias_kids": True, "bias_amplitude": 0.001,
            "nqp_noise_enabled": True, "nqp_noise_std_factor": noise}))
        m = crs._resonator_model
        m._convergence_cache.clear()
        m._convergence_stats = {"full": 0, "skipped": 0,
                                "last_reason": None}
        for k in range(200):
            m._s21_lc_response_internal(1.0e9, 0.001, pulse_time=k * 1e-6)
        st = m._convergence_stats
        rates[noise] = st["skipped"] / max(st["full"] + st["skipped"], 1)

    assert rates[0.1] > 0.5 * rates[0.001], (
        f"cache hit rate collapses with noise: "
        f"{rates[0.001]:.2f} -> {rates[0.1]:.2f}")


def test_qp_noise_still_reaches_the_signal():
    """The perturbation must not be optimised away.

    Absolute scatter is a bad probe: the response is dominated by R
    (s_R ~ 1) while Lk barely moves (s_Lk ~ 3e-5), and the S21 dip sits
    off the nominal resonator frequency, so the sensitivity at any
    given probe point is arbitrary.  What must hold regardless is
    PROPORTIONALITY — 10x the noise gives 10x the scatter — plus a
    deterministic no-noise case.
    """
    import asyncio

    import numpy as np

    from rfmux.mock.crs import ServerMockCRS

    scatter = {}
    for noise in (0.0, 0.01, 0.1):
        crs = ServerMockCRS("0000")
        asyncio.run(crs.generate_resonators({
            "num_resonances": 2, "resonator_random_seed": 3,
            "auto_bias_kids": True, "bias_amplitude": 0.001,
            "nqp_noise_enabled": noise > 0,
            "nqp_noise_std_factor": max(noise, 1e-12)}))
        m = crs._resonator_model
        f_probe = float(m.resonator_frequencies[0])
        vals = [abs(m._s21_lc_response_internal(f_probe, 0.001,
                                                pulse_time=k * 1e-6))
                for k in range(300)]
        scatter[noise] = float(np.std(vals))

    assert scatter[0.01] > 0.0, "noise did not reach the output at all"
    assert scatter[0.0] < scatter[0.01] / 5.0, \
        "no-noise case should be far quieter than the noisy ones"
    ratio = scatter[0.1] / scatter[0.01]
    assert 5.0 < ratio < 20.0, \
        f"scatter should scale ~10x with 10x noise, got {ratio:.1f}x"


def test_nqp_linearisation_matches_the_exact_kernel():
    """The post-cache perturbation is a linearisation; confirm it
    reproduces the exact physics kernel over the usable noise range."""
    import asyncio

    import numpy as np

    from rfmux.mock.crs import ServerMockCRS
    from rfmux.mr_resonator import jit_physics

    crs = ServerMockCRS("0000")
    asyncio.run(crs.generate_resonators({
        "num_resonances": 2, "resonator_random_seed": 3,
        "auto_bias_kids": True, "bias_amplitude": 0.001}))
    m = crs._resonator_model
    s_Lk, s_R = m._nqp_sensitivity()

    n = len(m.mr_complex_resonators)
    base = np.array(m.base_nqp_values[:n])
    cr0 = m.mr_complex_resonators[0]
    common = (np.array([cr.readout_f for cr in m.mr_complex_resonators]),
              np.full(n, cr0.T), np.full(n, cr0.Delta0),
              np.full(n, cr0.N0), np.full(n, cr0.sigmaN),
              np.full(n, cr0.thickness), np.full(n, cr0.width),
              np.full(n, cr0.length), np.full(n, cr0.R_spoiler))
    R0, Lk0 = jit_physics.vectorized_update_params_from_nqp(base, *common)
    for eps in (0.01, 0.1):
        R1, Lk1 = jit_physics.vectorized_update_params_from_nqp(
            base * (1.0 + eps), *common)
        assert np.allclose((Lk1 - Lk0) / Lk0, s_Lk * eps, rtol=1e-3)
        assert np.allclose((R1 - R0) / R0, s_R * eps, rtol=1e-3)
