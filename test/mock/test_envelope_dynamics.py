"""The current of a resonator with a pulse in flight relaxes toward
each new steady state at its own rates, and the transient reaches the
stream."""
import numpy as np
import pytest

from rfmux.mock import kerr
from rfmux.mr_resonator import jit_physics as jp


def test_a_step_decays_at_half_the_linewidth_at_low_drive():
    """Without a pump term the two rates are kappa/2: a deviation from
    the steady state shrinks by exp(-kappa t / 2) whatever it does in
    phase, sample after sample."""
    kappa = 2e5
    a = -kappa / 2 - 1j * 3e4
    b = 0j
    I_ss = 1e-5 + 2e-6j
    field = I_ss + (3e-6 - 1e-6j)
    dt = 4e-7
    for k in range(1, 40):
        field = jp.kerr_step(field, I_ss, a, b, dt)
        assert abs(field - I_ss) == pytest.approx(
            abs(3e-6 - 1e-6j) * np.exp(-kappa * k * dt / 2), rel=1e-9)


def test_near_the_fold_the_slow_mode_lingers(kerr_model):
    """About the solver's last state before its fold, a deviation four
    ring-downs later is many times what a linear resonator keeps."""
    env, i, solve = kerr_model.env, kerr_model.i, kerr_model.solve
    f_r = env['omega_r'][i] / (2 * np.pi)
    kappa = env['kappa'][i]
    seed, last = None, None
    for f in f_r - np.arange(0, 4e5, 1e2):
        I = solve(f, 0.01, seed)
        if last is not None and abs(I[i]) < 0.5 * abs(last[1][i]):
            break
        last = (f, I)
        seed = I
    f, I = last
    a, b = kerr.coefficients(env, i, I[i], 2 * np.pi * f)
    # A deviation along the slow eigenvector: the real 2x2 is
    # Re(a) I + M, M^2 = disc I; its slow direction is M v = +sqrt(disc) v.
    disc = abs(b) ** 2 - a.imag ** 2
    assert disc > 0
    M = np.array([[b.real, b.imag - a.imag], [b.imag + a.imag, -b.real]])
    w, v = np.linalg.eig(M)
    slow = v[:, np.argmax(w.real)]
    u0 = 1e-7 * complex(slow[0], slow[1])
    t = 4.0 / kappa
    field = jp.kerr_step(I[i] + u0, I[i], a, b, t)
    linear = abs(u0) * np.exp(-kappa * t / 2)
    assert abs(field - I[i]) > 5 * linear


def test_the_slow_mode_never_grows():
    """Past the fold the linearisation would grow; the step clamps the
    split at kappa/2, so a tone parked there keeps a bounded deviation
    (the two directions are not orthogonal, so it can turn, not grow)."""
    kappa = 2e5
    a = -kappa / 2 - 1j * 1e6
    b = 1.2e6 * np.exp(0.7j)               # |b| > |Im a| + kappa/2
    I_ss = 1e-5 + 0j
    u0 = 2e-7 + 1e-7j
    field = I_ss + u0
    seen = []
    for _ in range(1000):
        field = jp.kerr_step(field, I_ss, a, b, 2.6e-5)
        seen.append(abs(field - I_ss))
    assert max(seen) < 10 * abs(u0)
    assert seen[-1] <= seen[499] * (1 + 1e-9)


def test_a_block_without_pulses_is_unchanged_by_the_dynamics(batch):
    """No pulse, no transient: with the dynamics on the block is the
    quasi-static one bit for bit."""
    outs = []
    for on in (True, False):
        crs, m = batch.model(11, "hoisted", pulses=False)
        crs._physics_config["envelope_dynamics"] = on
        outs.append(batch.run(crs, m, 20, 7))
    assert np.array_equal(outs[0], outs[1])


def test_a_pulse_rings_the_resonator(batch):
    """The first samples after a pulse starts differ from the
    quasi-static response by the transient; ten ring-downs later what
    remains is the ringing of the quantised QP steps along the decay,
    a hundred times smaller.  The reference and hoisted paths agree."""
    outs = {}
    for mode in ("hoisted", "reference"):
        for on in (True, False):
            crs, m = batch.model(11, mode, pulses=False)
            crs._physics_config["envelope_dynamics"] = on
            m.nqp_noise_enabled = False
            m._tls_generator = None
            crs._fir_stage = 0
            rate = 625e6 / 256 / 64
            n = 64
            i = int(np.argsort(m.resonator_frequencies)[0])
            m.add_pulse_event(i, 4.5 / rate, amplitude=3.0)
            t = 0.0
            blocks = []
            for _ in range(3):
                m.advance_pulses_to(t + (n - 1) / rate, n, 1.0 / rate)
                r = m.calculate_module_response_coupled(
                    1, num_samples=n, sample_rate=rate, start_time=t,
                    pulse_time=t)
                blocks.append(np.stack([r[ch] for ch in sorted(r)]))
                t += n / rate
            outs[mode, on] = np.concatenate(blocks, axis=1)
    kappa = m.envelope_parameters()['kappa'][i]
    diff = np.abs(outs["hoisted", True] - outs["hoisted", False])
    scale = np.abs(outs["hoisted", False]).max()
    early = diff[:, 5:8].max() / scale
    late = diff[:, int(5 + 10 * kappa ** -1 * rate) + 5:].max() / scale
    assert early > 1e-3
    assert late < 1e-2 * early
    rel = np.max(np.abs(outs["hoisted", True] - outs["reference", True])
                 / np.maximum(np.abs(outs["reference", True]), 1e-300))
    assert rel < 1e-9
