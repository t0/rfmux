"""
Mock config plumbing regressions.

Two bugs found while designing the TLS/1-f noise model:

1. Several sites read ``mock_crs.physics_config`` (no underscore) — an
   attribute that does not exist — so the dialog's cache-tuning
   settings and ``get_samples``' scale factor were silently ignored and
   the hardcoded fallbacks always won. Tested by observing that a
   setting reaches the value it drives, rather than by grepping the
   source for the spelling: the failure mode is a silent default, and
   that is visible in behaviour whatever the lookup looks like.
2. ``update_qp_densities_for_time`` is a monotonic ratchet, and with PFB
   enabled its batches advance the clock past the slow frame — so the
   slow emitter must pass ``pulse_time`` explicitly or its samples get
   evaluated at the PFB's (later) time.
3. ``generate_resonators`` rebuilds mr_lekids / mr_complex_resonators /
   base_nqp_values incrementally without holding ``_physics_lock``, so a
   streamer sample landing mid-rebuild saw mismatched lengths and raised
   "operands could not be broadcast together with shapes (4,) (3,)".
"""

import types

import pytest


def test_cache_tuning_takes_effect():
    """A dialog setting must reach the convergence-cache key.

    The original bug was a lookup of ``physics_config`` (no underscore),
    which ``getattr(..., default)`` turned into a silent fallback: the
    tuning controls appeared to work and changed nothing. Asserting on
    the returned step, rather than on the source, catches that however
    it is spelled.
    """
    from rfmux.mock.resonator_model import MockResonatorModel

    model = MockResonatorModel.__new__(MockResonatorModel)
    model.mr_lekids = []
    model.base_nqp_values = []
    model.mock_crs = types.SimpleNamespace(_physics_config={})

    _, _, default_step, _, _ = (0,) + model._compute_cache_key_params(1e9)
    assert default_step == pytest.approx(0.0001)

    model.mock_crs._physics_config = {"cache_freq_step": 5.0}
    _, _, tuned_step, _, _ = (0,) + model._compute_cache_key_params(1e9)
    assert tuned_step == pytest.approx(5.0), \
        "cache_freq_step did not reach the cache key — the config is " \
        "being read from somewhere that does not exist"


def test_slow_emitter_evaluates_at_its_own_frame_time():
    """Otherwise the slow stream inherits the PFB batches' later clock.

    ``update_qp_densities_for_time`` is a monotonic ratchet, so once a
    PFB batch has advanced it past ``t_frame`` the slow frame would be
    evaluated at the PFB's time unless it pins ``pulse_time`` itself.
    """
    from rfmux.mock.udp_streamer import MockCRSStreamer

    seen = {}

    class _Stop(Exception):
        pass

    def _record(*args, **kwargs):
        seen.update(kwargs)
        raise _Stop

    streamer = MockCRSStreamer.__new__(MockCRSStreamer)
    streamer.mock_crs = types.SimpleNamespace(
        _short_packets=True,
        _physics_config={},
        _resonator_model=types.SimpleNamespace(
            calculate_module_response_coupled=_record),
    )
    streamer.seq_counters = {1: 0}

    t_frame = 12.5
    with pytest.raises(_Stop):
        streamer._emit_slow_packet(1, t_frame, dec=6)

    assert seen.get("pulse_time") == t_frame, \
        f"slow emitter must pin pulse_time to its own frame time, got {seen}"
    assert seen.get("start_time") == t_frame


def test_ratchet_still_guards_pulse_generation():
    """The monotonic guard protects the Poisson dt draw from being
    double-counted by out-of-order calls: a time at or before the last
    update must be a no-op, not a second draw."""
    from rfmux.mock.resonator_model import MockResonatorModel

    model = MockResonatorModel.__new__(MockResonatorModel)
    model.last_update_time = 10.0
    model._nqp_state_t = "untouched"

    # The model is deliberately bare: nothing past the guard can run on
    # it, so falling through raises instead of quietly doing the wrong
    # thing. Report that as the guard failing, not as a missing stub.
    for backwards in (10.0, 9.5, 0.0):
        try:
            model.update_qp_densities_for_time(backwards)
        except Exception as exc:  # pragma: no cover - only on regression
            pytest.fail(
                f"update_qp_densities_for_time({backwards}) ran past the "
                f"monotonic guard with last_update_time=10.0 ({exc!r}). "
                f"An out-of-order call must be a no-op.")
        assert model.last_update_time == 10.0, \
            f"clock moved backwards on {backwards}"
        assert model._nqp_state_t == "untouched", \
            f"memo dropped for a non-advancing time ({backwards})"


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


def _mk_model(n=3, **extra):
    import asyncio

    from rfmux.mock.crs import ServerMockCRS

    crs = ServerMockCRS("0000")
    cfg = {"num_resonances": n, "resonator_random_seed": 3,
           "auto_bias_kids": False, "bias_amplitude": 0.001}
    cfg.update(extra)
    asyncio.run(crs.generate_resonators(cfg))
    return crs, crs._resonator_model


def test_nqp_sensitivity_tracks_resonator_count():
    """The sensitivity arrays are filled lazily from
    mr_complex_resonators.  A call landing mid-rebuild used to pin a
    short array for the rest of the run, and every later sample then
    crashed the streamer in the QP perturbation."""
    crs, m = _mk_model(3)
    # Fill the cache against a deliberately short (mid-rebuild) set.
    full_cr = list(m.mr_complex_resonators)
    full_nqp = list(m.base_nqp_values)
    m.mr_complex_resonators = full_cr[:2]
    m.base_nqp_values = full_nqp[:2]
    assert len(m._nqp_sensitivity()[0]) == 2

    # Rebuild completes — the cache must follow, not stay at 2.
    m.mr_complex_resonators = full_cr
    m.base_nqp_values = full_nqp
    s_Lk, s_R = m._nqp_sensitivity()
    assert len(s_Lk) == 3 and len(s_R) == 3


def test_partial_rebuild_degrades_instead_of_raising():
    """Belt and braces: even with a short sensitivity array in hand, the
    S21 path must return a value rather than take the streamer down."""
    crs, m = _mk_model(3, nqp_noise_enabled=True, nqp_noise_std_factor=0.01)
    m._nqp_sensitivity()
    m._nqp_sens_cache = (2, m._nqp_sens_cache[1][:2], m._nqp_sens_cache[2][:2])
    val = m._s21_lc_response_internal(float(m.resonator_frequencies[0]),
                                      0.001, pulse_time=1e-6)
    assert val == val  # not NaN


def test_generate_resonators_holds_the_physics_lock():
    """Serialising the rebuild against the streamer thread is the actual
    fix; the clamps above only stop it being fatal."""
    from rfmux.mock.resonator_model import MockResonatorModel

    crs, m = _mk_model(2)
    held = []
    real = MockResonatorModel._generate_resonators_locked

    def spy(self, *a, **k):
        held.append(self._physics_lock._is_owned())
        return real(self, *a, **k)

    MockResonatorModel._generate_resonators_locked = spy
    try:
        m.generate_resonators(num_resonances=2, config=crs._physics_config)
    finally:
        MockResonatorModel._generate_resonators_locked = real
    assert held == [True]


def test_streaming_across_a_regeneration_does_not_crash():
    """The reported failure, reproduced: sample S21 from one thread while
    another regenerates the resonator set."""
    import threading

    crs, m = _mk_model(3, nqp_noise_enabled=True, nqp_noise_std_factor=0.01)
    errors = []
    stop = threading.Event()

    def sampler():
        k = 0
        while not stop.is_set():
            try:
                # Same entry point and lock discipline as the PFB batch loop.
                with m._physics_lock:
                    m._s21_lc_response_internal(1.0e9, 0.001,
                                                pulse_time=k * 1e-6)
            except Exception as exc:          # noqa: BLE001 — that's the test
                errors.append(exc)
                return
            k += 1

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    try:
        for n in (5, 3, 6, 2):
            m.generate_resonators(num_resonances=n,
                                  config=dict(crs._physics_config,
                                              num_resonances=n))
    finally:
        stop.set()
        t.join(timeout=10)
    assert not errors, f"streamer thread died: {errors[0]!r}"


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
