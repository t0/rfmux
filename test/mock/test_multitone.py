"""Tones sharing a resonator each shift the resonance the others see:
an instantaneous nonlinearity puts twice the other tones' |I|^2 into a
tone's solve."""
import numpy as np
import pytest


def _place(crs, module, channel, frequency, amplitude):
    crs._frequencies[(module, channel)] = frequency
    crs._amplitudes[(module, channel)] = amplitude
    crs._phases[(module, channel)] = 0.0


def _response(m, module):
    r = m.calculate_module_response_coupled(module, num_samples=1)
    return {ch: complex(np.atleast_1d(v)[0]) for ch, v in r.items()}


def test_a_strong_tone_moves_a_weak_ones_resonance(kerr_model):
    """A pump on the upper state a hundred kilohertz below resonance
    shifts the resonance a weak probe sees by 2 K |I_pump|^2, so the
    probe's dip moves down by that much; a probe alone does not move."""
    km = kerr_model
    m, env, i = km.m, km.env, km.i
    crs = m.mock_crs
    crs._nco_frequencies[1] = 0.0
    f_r = env['omega_r'][i] / (2 * np.pi)
    grid = f_r + np.arange(-6e5, 5e4, 2e3)

    def probe_dip():
        seen = []
        for f in grid[::-1]:
            _place(crs, 1, 1, f, 1e-4)
            seen.append(abs(_response(m, 1)[1]))
        return grid[::-1][int(np.argmin(seen))]
    alone = probe_dip()
    assert abs(alone - f_r) < 5e3
    # The pump arrives from above, so it takes the upper state.
    for f in np.arange(f_r + 1e5, f_r - 1e5 - 1, -5e3):
        _place(crs, 1, 2, f, 0.01)
        _response(m, 1)
    for _ in range(3):                         # the pair settles
        _response(m, 1)
    n_pump = abs(m._tone_states[(1, 2)].currents[i]) ** 2
    shift = 2 * env['K'][i] * n_pump / (2 * np.pi)
    assert shift < -5e4
    pumped = probe_dip()
    assert pumped - alone == pytest.approx(shift, rel=0.15)


def test_tones_on_different_resonators_leave_each_other_alone(kerr_model):
    """A strong tone on one resonator carries next to no current in
    another, so a tone there is unchanged."""
    km = kerr_model
    m = km.m
    crs = m.mock_crs
    crs._nco_frequencies[1] = 0.0
    order = np.argsort(m.resonator_frequencies)
    f_a = float(m.resonator_frequencies[order[0]])
    f_b = float(m.resonator_frequencies[order[2]])
    _place(crs, 1, 1, f_a, 1e-3)
    alone = _response(m, 1)[1]
    _place(crs, 1, 2, f_b, 1e-2)
    _response(m, 1)
    with_other = _response(m, 1)[1]
    assert with_other == pytest.approx(alone, rel=1e-9)


def test_the_batch_paths_agree_with_two_tones_in_one_resonance(batch):
    """Reference and hoisted paths take the same background."""
    outs = []
    for mode in ("reference", "hoisted"):
        crs, m = batch.model(11, mode, pulses=True)
        f1 = crs._frequencies[(1, 1)]
        _place(crs, 1, 3, f1 + 2e4, 0.003)
        outs.append(batch.run(crs, m, 12, 7))
    rel = np.max(np.abs(outs[0] - outs[1]) / np.maximum(np.abs(outs[0]), 1e-300))
    assert rel < 1e-9
    m_states = [k for k in m._tone_states if k[0] == 1]
    assert (1, 3) in m_states
    assert m._tone_states[(1, 1)].background is not None
    assert m._tone_states[(1, 1)].background.max() > 0
