"""The nonlinear model follows the measured direction through bifurcation."""

import numpy as np
import pytest

from rfmux.tuning.fits import (
    fit_nonlinear_iq, get_y_nonlinear, nonlinear_iq, nonlinear_model_iq,
)

pytestmark = pytest.mark.portable


def test_directions_agree_below_bifurcation_and_separate_above():
    detuning = np.linspace(-2, 1, 501)
    for a in (0.3, 0.85):
        up = get_y_nonlinear(detuning, a, sweep_direction="upward")
        down = get_y_nonlinear(detuning, a, sweep_direction="downward")
        for y in (up, down):
            np.testing.assert_allclose(y, detuning + a / (1 + 4 * y*y), atol=1e-10)
        if a == 0.3:
            np.testing.assert_allclose(up, down, atol=1e-10)
        else:
            assert np.max(down - up) > 0.2


@pytest.mark.parametrize("direction", ["upward", "downward"])
def test_fit_and_reconstruction_keep_direction_after_sorting(direction):
    frequencies = np.linspace(1e9 - 60e3, 1e9 + 60e3, 301)
    if direction == "downward":
        frequencies = frequencies[::-1]
    initial = [1e9, 5e4, 0.6, 0.1, 0.85, 1.0, 0.0]
    iq = nonlinear_iq(frequencies, *initial, sweep_direction=direction)
    params, _, residual = fit_nonlinear_iq(
        frequencies, iq, p0=initial, sweep_direction=direction)
    assert residual < 1e-8
    entry = {
        "frequencies": frequencies, "sweep_direction": direction,
        "fits": {"nonlinear": {"params": params, "gain": 2.0}},
    }
    np.testing.assert_allclose(nonlinear_model_iq(entry), 2 * iq, atol=1e-8)
