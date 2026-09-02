"""The hoisted batch path is the reference loop with the constants
hoisted: same arithmetic, same noise draws, same cache decisions, same
end state.  Pinned against the loop on one seed, with pulses in flight."""
import asyncio

import numpy as np
import pytest

FS = 2441406.25
N = 64


def _model(seed, mode, pulses=True):
    from rfmux.mock.crs import ServerMockCRS
    crs = ServerMockCRS("0000")
    cfg = {"num_resonances": 2, "resonator_random_seed": seed,
           "auto_bias_kids": True, "bias_amplitude": 0.001}
    if pulses:
        cfg.update({"pulse_mode": "periodic", "pulse_period": 0.0005,
                    "pulse_tau_rise": 1e-6, "pulse_tau_decay": 1e-4,
                    "pulse_amplitude": 3.0})
    asyncio.run(crs.generate_resonators(cfg))
    crs._physics_config["physics_batch_mode"] = mode
    return crs, crs._resonator_model


def _run(crs, m, n_batches, seed):
    np.random.seed(seed)
    out = []
    for k in range(n_batches):
        t = k * N / FS
        r = m.calculate_module_response_coupled(
            1, num_samples=N, sample_rate=FS, start_time=t, pulse_time=t)
        out.append(np.stack([r[ch] for ch in sorted(r)]))
    return np.stack(out)


@pytest.mark.parametrize("pulses", [False, True])
def test_hoisted_matches_reference(pulses):
    a_crs, a = _model(11, "reference", pulses)
    b_crs, b = _model(11, "hoisted", pulses)
    ra = _run(a_crs, a, 40, 7)
    rb = _run(b_crs, b, 40, 7)
    rel = np.max(np.abs(ra - rb) / np.maximum(np.abs(ra), 1e-300))
    assert rel < 1e-9, f"max relative deviation {rel:.3e}"
    # Same end state, so the slow emitter reads the same resonators.
    for la, lb in zip(a.mr_lekids, b.mr_lekids):
        assert (la.L, la.R, la.Lk) == pytest.approx((lb.L, lb.R, lb.Lk), rel=1e-12)
    assert a._convergence_stats["full"] == b._convergence_stats["full"]
    assert a._convergence_stats["skipped"] == b._convergence_stats["skipped"]
    assert a._nqp_state_t == b._nqp_state_t


def test_hoisted_dispatches_once_per_batch(monkeypatch):
    """The per-sample work is gone: one Lk/R kernel dispatch per batch."""
    from rfmux.mr_resonator import jit_physics
    crs, m = _model(11, "hoisted")
    calls = {"n": 0}
    real = jit_physics.vectorized_update_params_from_nqp

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(jit_physics, "vectorized_update_params_from_nqp", counting)
    _run(crs, m, 10, 7)
    assert calls["n"] <= 2 * 10, f"{calls['n']} kernel dispatches for 10 batches"
