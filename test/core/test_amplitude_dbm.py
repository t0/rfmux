"""Tone power in dBm against a module's DAC scale, and back."""
import numpy as np
import pytest

from rfmux.core.transferfunctions import (
    convert_amplitude_to_dbm, convert_dbm_to_amplitude)


def test_full_scale_is_the_dac_scale_and_a_tenth_is_20_db_down():
    assert convert_amplitude_to_dbm(1.0, -0.5) == pytest.approx(-0.5)
    assert convert_amplitude_to_dbm(0.1, -0.5) == pytest.approx(-20.5)
    assert convert_amplitude_to_dbm(0.0, -0.5) == -np.inf


def test_the_two_are_inverse_and_take_arrays():
    dbm = np.array([-70.0, -55.0, -40.0])
    amp = convert_dbm_to_amplitude(dbm, 1.0)
    np.testing.assert_allclose(convert_amplitude_to_dbm(amp, 1.0), dbm)
    assert isinstance(convert_dbm_to_amplitude(-55.0, 1.0), float)


def test_periscope_labels_with_the_same_arithmetic():
    pytest.importorskip("PyQt6")
    from rfmux.tools.periscope.utils import UnitConverter
    assert UnitConverter.format_probe_label(0.001, dac_scale=-0.5) == \
        f"{convert_amplitude_to_dbm(0.001, -0.5):.1f} dBm"
    assert UnitConverter.format_probe_label(0.0, dac_scale=-0.5) == "-inf dBm"
