"""The sweep a tuning row carries, put under the channel's samples."""

import numpy as np
import pytest

from rfmux.pulse_capture.analysis import (
    display_transform, frequency_direction, tuning_sweep)
from rfmux.core.transferfunctions import apply_iq_conversion

F = np.linspace(1.0e9, 1.0e9 + 6.0, 7)
IQ = np.exp(1j * np.linspace(0.0, 1.0, 7)) * 1000.0


def test_the_sweep_is_turned_by_minus_the_phase_the_bias_set():
    """The board turns samples by minus the programmed phase; a sweep
    taken at phase zero is turned the same way to sit under them."""
    row = {"frequencies": F, "iq_complex": IQ, "optimal_phase_degrees": 30.0,
           "bias_frequency": 1.0e9 + 2.5}
    f, iq, point = tuning_sweep(row)
    np.testing.assert_allclose(iq, IQ * np.exp(-1j * np.radians(30.0)))
    assert point == pytest.approx(0.5 * (iq[2] + iq[3]))


def test_a_row_without_a_phase_or_bias_is_the_sweep_as_stored():
    f, iq, point = tuning_sweep({"frequencies": F[::-1], "iq_complex": IQ[::-1],
                                 "optimal_phase_degrees": None})
    np.testing.assert_array_equal(f, F)
    np.testing.assert_array_equal(iq, IQ)
    assert point is None
    assert tuning_sweep({"df_calibration": 1.0}) is None


def test_the_frequency_direction_is_the_real_axis_of_the_df_view():
    cal = 2.0e6 * np.exp(1j * np.radians(-40.0))
    d = frequency_direction(cal)
    assert abs(d) == pytest.approx(1.0)
    factor, units = display_transform(cal, "iq", "V", "df", "Hz")
    first, second = apply_iq_conversion(d.real, d.imag, factor)
    assert units == "Hz" and first > 0 and second == pytest.approx(0.0, abs=1e-9)
    assert frequency_direction(None) is None
