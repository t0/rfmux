"""One volts-to-dBm convention: volts are peak amplitudes everywhere, so
a carrier reads the same power through convert_roc_to_dbm and through
the spectrum path's DC bin."""
import numpy as np
import pytest

from rfmux.core.transferfunctions import (
    TERMINATION,
    VOLTS_PER_ROC,
    convert_roc_to_dbm,
    spectrum_from_slow_tod,
    volts_squared_to_dbm,
)

pytestmark = pytest.mark.portable


def test_volts_squared_to_dbm_is_the_peak_convention():
    v_peak = 0.1
    # Peak 0.1 V into 50 ohm: 0.1 mW, 0 dBm... times 1/2 for peak: -3.01 dBm.
    assert volts_squared_to_dbm(v_peak**2) == pytest.approx(
        10 * np.log10(v_peak**2 / 2 / TERMINATION * 1e3))
    assert volts_squared_to_dbm(np.array([0.0]), floor=1e-30)[0] == pytest.approx(-300.0)


def test_carrier_power_agrees_between_the_two_paths():
    counts = 1880796.46          # the full-scale carrier VOLTS_PER_ROC is fitted to
    n = 4096
    i = np.full(n, counts) * VOLTS_PER_ROC
    q = np.zeros(n)
    out = spectrum_from_slow_tod(i, q, dec_stage=6, scaling="ps", nsegments=1,
                                 reference="absolute", input_units="volts")
    dc_bin = out["psd_dual_sideband"][np.argmin(np.abs(out["freq_dsb"]))]
    assert dc_bin == pytest.approx(convert_roc_to_dbm(counts), abs=0.05)
