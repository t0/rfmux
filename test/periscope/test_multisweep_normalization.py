"""A normalised multisweep trace is referenced at its lowest frequency,
so an upward and a downward sweep of the same resonance compare."""
import numpy as np
import pytest

from rfmux.tools.periscope.utils import UnitConverter


@pytest.mark.parametrize("unit_mode", ["counts", "dbm", "volts"])
def test_a_downward_sweep_normalises_at_the_same_point_as_an_upward_one(unit_mode):
    f = np.linspace(1e9 - 1.5e5, 1e9 + 1.5e5, 101)
    # A skewed dip: the two ends of the span differ.
    mag = 1000 * (1 - 0.8 / (1 + ((f - 1e9 - 2e4) / 2e4) ** 2)) * (1 + (f - f[0]) / 3e6)
    iq = mag.astype(complex)
    up = UnitConverter.convert_amplitude(
        mag, iq, unit_mode, normalize=True,
        ref_index=UnitConverter.sweep_reference(f))
    down = UnitConverter.convert_amplitude(
        mag[::-1], iq[::-1], unit_mode, normalize=True,
        ref_index=UnitConverter.sweep_reference(f[::-1]))[::-1]
    np.testing.assert_allclose(up, down, rtol=1e-12, atol=1e-12)
    ref = 0.0 if unit_mode == "dbm" else 1.0
    assert up[0] == pytest.approx(ref)
