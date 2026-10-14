"""The sweep a tuning row carries, put under the channel's samples."""

import numpy as np
import pytest

from rfmux.pulse_capture.analysis import tuning_sweep
from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog
from rfmux.core.transferfunctions import VOLTS_PER_ROC
from rfmux.tuning.tuning_record import tuning_rows

F = np.linspace(1.0e9, 1.0e9 + 6.0, 7)
IQ = np.exp(1j * np.linspace(0.0, 1.0, 7)) * 1000.0


def test_catalog_calibration_sweep_is_shown_in_counts_without_loop_rotation():
    bias = BiasPoint(
        frequency_hz=F[2], amplitude=0.01, bias_frequency_quantized=False,
        iq_rotation_deg=30.0,
        bias_sweep={"frequencies": F[::-1],
                    "iq_volts": IQ[::-1] * VOLTS_PER_ROC})
    catalog = ResonatorCatalog([Resonator(name="BOTA", channel=1, bias=bias)],
                               module=1)
    row = tuning_rows(catalog)[1]
    f, iq, point = tuning_sweep(row)
    np.testing.assert_array_equal(f, F)
    np.testing.assert_allclose(iq, IQ)
    assert point == pytest.approx(IQ[2])
    np.testing.assert_array_equal(row["iq_volts"], IQ[::-1] * VOLTS_PER_ROC)


def test_the_sweep_is_turned_by_minus_the_phase_the_bias_set():
    """The board turns samples by minus the programmed phase; a sweep
    taken at phase zero is turned the same way to sit under them."""
    row = {"frequencies": F, "iq_complex": IQ, "optimal_phase_degrees": 30.0,
           "bias_frequency": 1.0e9 + 2.5}
    f, iq, point = tuning_sweep(row)
    np.testing.assert_allclose(iq, IQ * np.exp(-1j * np.radians(30.0)))
    assert point == pytest.approx(0.5 * (iq[2] + iq[3]))


def test_a_rotation_the_multisweep_applied_is_undone_too():
    """"Rotate saved data" turns the stored sweep in place; the samples
    were never turned, so the sweep is turned back under them."""
    row = {"frequencies": F, "iq_complex": IQ, "optimal_phase_degrees": 10.0,
           "applied_rotation_degrees": 25.0}
    _f, iq, _p = tuning_sweep(row)
    np.testing.assert_allclose(iq, IQ * np.exp(-1j * np.radians(35.0)))


def test_a_row_without_a_phase_or_bias_is_the_sweep_as_stored():
    f, iq, point = tuning_sweep({"frequencies": F[::-1], "iq_complex": IQ[::-1],
                                 "optimal_phase_degrees": None})
    np.testing.assert_array_equal(f, F)
    np.testing.assert_array_equal(iq, IQ)
    assert point is None
    assert tuning_sweep({"df_calibration": 1.0}) is None
