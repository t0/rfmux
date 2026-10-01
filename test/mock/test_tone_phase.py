"""The two phases a channel carries reach the stream as one rotation:
the ADC phase turns the samples by minus its value, the DAC phase turns
the carrier by plus its value."""
import numpy as np
import pytest


def _response(m, crs, dac_deg, adc_deg):
    crs._dac_phases[(1, 1)] = dac_deg
    crs._phases[(1, 1)] = adc_deg
    r = m.calculate_module_response_coupled(1, num_samples=1)
    return complex(np.atleast_1d(r[1])[0])


def test_an_unknown_phase_target_is_refused(kerr_model):
    import asyncio
    crs = kerr_model.m.mock_crs
    with pytest.raises(ValueError, match="target"):
        asyncio.run(crs.set_phase(1.0, target="DAC", channel=1, module=1))


@pytest.mark.parametrize("dac_deg, adc_deg, turn_deg", [
    (90.0, 0.0, 90.0), (0.0, 90.0, -90.0), (30.0, 30.0, 0.0)])
def test_phases_rotate_the_response(kerr_model, dac_deg, adc_deg, turn_deg):
    m = kerr_model.m
    crs = m.mock_crs
    crs._nco_frequencies[1] = 0.0
    f_r = kerr_model.env['omega_r'][kerr_model.i] / (2 * np.pi)
    crs._frequencies[(1, 1)] = f_r + 5e4
    crs._amplitudes[(1, 1)] = 0.001
    ref = _response(m, crs, 0.0, 0.0)
    turned = _response(m, crs, dac_deg, adc_deg)
    assert turned == pytest.approx(ref * np.exp(1j * np.deg2rad(turn_deg)),
                                   rel=1e-9)
