"""Off commands and array rebuilds discard the previous driven state."""

import asyncio

import numpy as np
import pytest

from test.mock.test_bifurcation_hysteresis import _model, _place, _response, _sweep


@pytest.mark.parametrize("command", ["off", "clear"])
def test_shutdown_resets_state_without_an_intervening_read(command):
    model, f0 = _model()
    crs = model.mock_crs
    inside = f0 - 1.5e5
    for f in np.arange(f0 + 1e5, inside - 1, -5e3):
        _place(crs, 1, 1, f, 0.01)
        deep = _response(model, 1)[1][0]
    if command == "off":
        asyncio.run(crs.set_amplitude(0.0, channel=1, module=1))
    else:
        asyncio.run(crs.clear_channels(module=1))
    _place(crs, 1, 1, inside, 0.01)
    back = abs(_response(model, 1)[1][0]) / 0.01
    rest = _sweep(model, [inside], 0.01, tone=(2, 1))[0]
    assert abs(deep) / 0.01 < rest - 0.3
    assert back == pytest.approx(rest, abs=1e-3)


def test_public_regeneration_without_auto_bias_leaves_tones_off():
    model, f0 = _model()
    crs = model.mock_crs
    original = model.resonator_frequencies.copy()
    _place(crs, 1, 1, f0, 0.01)
    asyncio.run(crs.generate_resonators({"auto_bias_kids": False}))
    assert not (asyncio.run(crs.get_amplitude(channel=1, module=1)) or 0)
    np.testing.assert_allclose(crs._resonator_model.resonator_frequencies, original)
