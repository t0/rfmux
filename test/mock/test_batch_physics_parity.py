"""The hoisted batch path is the reference loop with the constants
hoisted: same arithmetic, same noise draws, same cache decisions, same
end state.  Pinned against the loop on one seed, with pulses in flight."""
import numpy as np
import pytest

FS = 2441406.25
N = 64


def test_hoisted_matches_reference(batch):
    a_crs, a = batch.model(11, "reference", pulses=True)
    b_crs, b = batch.model(11, "hoisted", pulses=True)
    ra = batch.run(a_crs, a, 40, 7, fs=FS, n=N)
    rb = batch.run(b_crs, b, 40, 7, fs=FS, n=N)
    rel = np.max(np.abs(ra - rb) / np.maximum(np.abs(ra), 1e-300))
    assert rel < 1e-9, f"max relative deviation {rel:.3e}"
    # Same end state, so the slow emitter reads the same resonators.
    for la, lb in zip(a.mr_lekids, b.mr_lekids):
        assert (la.L, la.R, la.Lk) == pytest.approx((lb.L, lb.R, lb.Lk), rel=1e-12)
    assert a._convergence_stats["full"] == b._convergence_stats["full"]
    assert a._convergence_stats["skipped"] == b._convergence_stats["skipped"]
    assert a._nqp_state_t == b._nqp_state_t


def test_hoisted_dispatches_once_per_batch(batch, monkeypatch):
    """The per-sample work is gone: at most two Lk/R kernel dispatches
    per batch, where the loop made one per sample."""
    from rfmux.mr_resonator import jit_physics
    crs, m = batch.model(11, "hoisted")
    # One-time work (the noise-sensitivity finite difference) lands in
    # the first batch; the claim is about the per-batch cost after it.
    batch.run(crs, m, 1, 7, fs=FS, n=N)
    calls = {"n": 0}
    real = jit_physics.vectorized_update_params_from_nqp

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(jit_physics, "vectorized_update_params_from_nqp", counting)
    batch.run(crs, m, 10, 7, fs=FS, n=N)
    assert calls["n"] <= 2 * 10, f"{calls['n']} kernel dispatches for 10 batches"


def _brute_pairs(obs, tones, bw):
    o, t = np.asarray(obs), np.asarray(tones)
    grid = np.abs(t[None, :] - o[:, None]) <= bw
    return set(zip(*np.nonzero(grid)))


@pytest.mark.parametrize("bw", [298.0, 38147.0])
def test_coupled_pairs_match_the_grid(bw):
    """The pair search finds exactly the grid's pairs: clustered tones,
    tones sitting on the bandwidth boundary, and every tone on one
    frequency."""
    from rfmux.mock.resonator_model import MockResonatorModel as M
    rng = np.random.default_rng(3)
    obs = np.sort(rng.uniform(-2.4e8, 2.4e8, 200))
    tones = np.concatenate([obs, obs[:20] + bw, obs[20:40] - bw,
                            obs[40:60] + 0.3 * bw, obs[60:70] + 1.5 * bw,
                            rng.uniform(-2.4e8, 2.4e8, 50)])
    oi, ti, d = M._coupled_pairs(obs, tones, bw)
    assert set(zip(oi.tolist(), ti.tolist())) == _brute_pairs(obs, tones, bw)
    assert np.array_equal(d, tones[ti] - obs[oi])
    same = np.full(300, 1.0e8)
    oi, ti, _ = M._coupled_pairs(same, same, bw)
    assert len(oi) == 300 * 300


def _configure(crs, n_channels, clustered):
    rng = np.random.default_rng(9)
    crs._nco_frequencies[1] = 1.2e9
    freqs = rng.uniform(-2.0e8, 2.0e8, n_channels)
    if clustered:
        freqs[1::2] = freqs[::2][: n_channels // 2] + rng.uniform(-200, 200, n_channels // 2)
    for ch, f in enumerate(freqs, start=1):
        crs._frequencies[(1, ch)] = float(f)
        crs._amplitudes[(1, ch)] = 0.001
        crs._phases[(1, ch)] = 0.0


@pytest.mark.parametrize("pulse_time", [None, 1.0])
def test_batch_paths_agree_with_many_coupled_channels(batch, pulse_time):
    """Forty channels, half of them within one bandwidth of another so
    the pair sum has off-diagonal terms, with and without per-sample
    pulse times."""
    outs, stats = [], []
    for mode in ("reference", "hoisted"):
        crs, m = batch.model(11, mode, pulses=True)
        _configure(crs, 40, clustered=True)
        crs._fir_stage = 6
        np.random.seed(5)
        r = m.calculate_module_response_coupled(
            1, num_samples=10, sample_rate=596.0, start_time=1.0,
            pulse_time=pulse_time)
        outs.append(np.stack([r[ch] for ch in sorted(r)]))
        stats.append((m._convergence_stats["full"],
                      m._convergence_stats["skipped"],
                      list(m._recent_cache_results)))
    ra, rb = outs
    rel = np.max(np.abs(ra - rb) / np.maximum(np.abs(ra), 1e-300))
    assert rel < 1e-9, f"max relative deviation {rel:.3e}"
    assert stats[0] == stats[1]


def test_a_kept_state_carries_the_r_of_its_instant(batch):
    """A pulse raises the QP density, and with it R as well as Lk: the
    state kept for each QP key holds the R of that instant, so a pulse
    moves the dissipation quadrature and not only the frequency."""
    crs, m = batch.model(11, "hoisted", pulses=True)
    R0 = np.array([lk.R for lk in m.mr_lekids])
    batch.run(crs, m, 40, 7, fs=FS, n=N)
    seen = {tuple(R) for ts in m._tone_states.values()
            for _, R, _, _ in ts.runs.values()}
    assert len(seen) > 1
    assert all(np.all(np.asarray(R) >= R0) for R in seen)
    assert any(np.any(np.asarray(R) > R0) for R in seen)


def test_converge_tones_chunks_agree_with_one_call(batch, monkeypatch):
    """A call whose runs times resonators exceed RUN_ELEMENTS_PER_CALL is
    split into tone chunks; the split must not change a value."""
    from rfmux.mr_resonator import jit_physics
    crs, m = batch.model(11, "hoisted", pulses=True)
    whole = batch.run(crs, m, 40, 7, fs=FS, n=N)
    monkeypatch.setattr(jit_physics, "RUN_ELEMENTS_PER_CALL", 3 * 11)
    crs, m = batch.model(11, "hoisted", pulses=True)
    chunked = batch.run(crs, m, 40, 7, fs=FS, n=N)
    assert np.array_equal(whole, chunked)


def test_a_pulse_makes_the_dip_shallower(batch):
    """More quasiparticles mean more loss: the swept dip of a resonator
    with a pulse in flight is shallower than at rest, as well as moved
    down in frequency."""
    crs, m = batch.model(3, "hoisted", pulses=False)
    i = int(np.argsort(m.resonator_frequencies)[1])
    fgen = float(m.resonator_frequencies[i])
    grid = np.linspace(fgen - 3e6, fgen + 3e6, 3001)
    rest = m.s21_sweep(grid, 1e-4)
    m.add_pulse_event(i, 0.5, amplitude=3.0)
    m.update_qp_densities_for_time(0.5 + 2e-6)
    pulsed = m.s21_sweep(grid, 1e-4)
    assert rest.min() < 0.9, "no dip on the grid"
    assert pulsed.min() > rest.min() + 0.01
    assert grid[np.argmin(pulsed)] < grid[np.argmin(rest)]
