"""Drive normalization is independent of sweep acquisition direction."""
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope.utils import UnitConverter

pytestmark = pytest.mark.portable


@pytest.mark.parametrize("unit_mode", ["counts", "dbm", "volts"])
def test_a_downward_sweep_normalises_at_the_same_point_as_an_upward_one(unit_mode):
    f = np.linspace(1e9 - 1.5e5, 1e9 + 1.5e5, 101)
    # A skewed dip: the two ends of the span differ.
    mag = 1000 * (1 - 0.8 / (1 + ((f - 1e9 - 2e4) / 2e4) ** 2)) * (1 + (f - f[0]) / 3e6)
    up = UnitConverter.convert_amplitude(
        mag, unit_mode, normalize=True, drive=0.005, dac_scale=1.0)
    down = UnitConverter.convert_amplitude(
        mag[::-1], unit_mode, normalize=True, drive=0.005, dac_scale=1.0)[::-1]
    np.testing.assert_allclose(up, down, rtol=1e-12, atol=1e-12)
