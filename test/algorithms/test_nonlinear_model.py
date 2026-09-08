"""The nonlinear resonator model is Swenson et al. 2013 eq. 13: stored
energy pulls the resonance to lower frequency, and the fitter can
recover the nonlinearity of a resonance that leans that way."""
import numpy as np
import pytest

from rfmux.algorithms.measurement.fitting_nonlinear import (
    fit_nonlinear_iq, get_y_nonlinear, nonlinear_iq)

FR, QR = 1.0e9, 5.0e4


def test_detuning_satisfies_swenson_eq_13():
    yg = np.linspace(-4, 4, 201)
    for a in (0.1, 0.5, 0.7, 0.76):
        y = get_y_nonlinear(yg, a)
        assert np.allclose(y, yg + a / (1 + 4 * y**2), atol=1e-8)
        assert get_y_nonlinear(float(yg[57]), a) == pytest.approx(y[57], abs=1e-8)


def test_resonance_pulls_to_lower_frequency():
    f = np.linspace(FR - 60e3, FR + 60e3, 6001)
    z = nonlinear_iq(f, FR, QR, 0.5, 0.0, 0.5, 1.0, 0.0)
    f_dip = f[np.argmin(np.abs(z))]
    # y = 0 at yg = -a: the dip sits a*fr/Qr below fr.
    assert f_dip - FR == pytest.approx(-0.5 * FR / QR, abs=50.0)


def test_fit_recovers_nonlinearity_of_a_pulled_resonance():
    f = np.linspace(FR - 100e3, FR + 100e3, 101)
    truth = (FR, QR, 0.6, 0.1, 0.35, 1.0, 0.0)
    z = nonlinear_iq(f, *truth)
    _, popt, _, residual = fit_nonlinear_iq(f, z)
    assert residual < 1e-3
    assert popt[4] == pytest.approx(0.35, abs=0.02)
    assert popt[0] == pytest.approx(FR, abs=100.0)
