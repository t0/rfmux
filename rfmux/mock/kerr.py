"""The Kerr resonator about a driven steady state, in the mock's
current units (Rouble et al., arXiv:2607.09178).

Near its resonance each mock resonator is the envelope equation

    dI/dt = -(kappa/2 + i (omega_g - omega_r - K |I|^2)) I + D A

for the resonant current I under a drive of amplitude A at omega_g
(``MockResonatorModel.envelope_parameters`` gives omega_r, kappa, K,
D and the output coupling c per resonator).  Its steady states are
the solver's; about one of them, I0, a deviation u obeys

    du/dt = a u + b u*,   a = -kappa/2 - i (omega_g - omega_r - 2 K n),
                          b = i K I0^2,   n = |I0|^2,

a real 2x2 system whose eigenvalues -kappa/2 +- sqrt(|b|^2 - Im(a)^2)
are the paper's eq. 21: two decay rates of kappa/2 with a beat far
from the fold, splitting into a slow and a fast rate near it, the slow
one reaching zero at the fold.  A weak probe at omega_g + Omega drives
u = u+ e^{i Omega t} + u- e^{-i Omega t}: u+ is the probe's
transmitted deviation, u- the idler at omega_g - Omega, and |u+| over
its value without the pump the parametric gain.
"""
import numpy as np

from rfmux.mr_resonator import jit_physics


def coefficients(env, i, I0, omega_g):
    """(a, b) of du/dt = a u + b u* about the steady state *I0* of
    resonator *i* driven at *omega_g*: jit_physics.kerr_coefficients
    at the rest QP density, the state's inductance built from |I0|^2."""
    n = abs(I0) ** 2
    L0, R0 = float(env['L0'][i]), float(env['R0'][i])
    Lk0 = float(env['alpha_k'][i]) * L0
    Lk = Lk0 * (1.0 + n / env['Istar'] ** 2)
    return jit_physics.kerr_coefficients(
        float(omega_g), L0 + Lk - Lk0, R0, Lk, complex(I0),
        float(env['Istar']), float(env['omega_r'][i]),
        float(env['kappa'][i]), L0, R0, float(env['K'][i]))


def rates(a, b):
    """The two eigenvalues of the deviation, slow first: their real
    parts are the decay rates, an imaginary part a beat."""
    root = np.sqrt(abs(b) ** 2 - a.imag ** 2 + 0j)
    return a.real + root, a.real - root


def probe_response(a, b, D, Omega):
    """(u+, u-) per unit probe amplitude at omega_g + Omega: the
    probe's transmitted deviation and the idler at omega_g - Omega.
    At Omega = 0 both are static and u+ + u- is the change of the
    steady state per unit drive."""
    # (i Omega - a) u+ - b conj(u-) = D,  (-i Omega - a) u- - b conj(u+) = 0
    denom = 1j * Omega - a - abs(b) ** 2 / (1j * Omega - np.conj(a))
    u_plus = D / denom
    u_minus = np.conj(np.conj(b) * u_plus / (1j * Omega - np.conj(a)))
    return u_plus, u_minus


def probe_gain(env, i, I0, omega_g, Omega):
    """|u+| with the pump over |u+| without it, at omega_g + Omega."""
    a, b = coefficients(env, i, I0, omega_g)
    a0 = -env['kappa'][i] / 2 - 1j * (omega_g - env['omega_r'][i])
    with_pump = abs(probe_response(a, b, env['D'][i], Omega)[0])
    without = abs(env['D'][i] / (1j * Omega - a0))
    return with_pump / without
