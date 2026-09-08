"""The nonlinear fit reads the simulator's pull the right way round.

The mock's kinetic inductance grows with drive, Lk = Lk0 (1 + |I|^2/I*^2),
so its resonance moves down in frequency as the tone gets stronger.  In
the fitter's model that is a positive nonlinearity a, growing with power,
and a fitted fr above the transmission minimum.  bias_kids' amplitude
guard compares a with a threshold, so a fitter that read the pull with
the wrong sign would pass every amplitude without a test noticing.

Headless: the model's single-point S21 on multisweep's own grid (200 kHz,
101 points), the fitter that bias_kids runs.
"""
import asyncio
import contextlib
import io

import numpy as np
import pytest

from rfmux.algorithms.measurement.df_calibration import ensure_fits
from rfmux.mock.config import bias_amplitude_from_dbm

LOW_DBM, HIGH_DBM = -55.0, -50.0


@pytest.fixture(scope="module")
def fits():
    from rfmux.mock.crs import ServerMockCRS
    crs = ServerMockCRS("0000")
    with contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(crs.generate_resonators(
            {"num_resonances": 3, "resonator_random_seed": 5,
             "auto_bias_kids": False,
             "tls_noise_enabled": False, "nqp_noise_enabled": False}))
    model = crs._resonator_model
    out = {}
    for dbm in (LOW_DBM, HIGH_DBM):
        amp = bias_amplitude_from_dbm(dbm)
        entries = []
        for f0 in sorted(model.resonator_frequencies):
            dip = crs._find_s21_dip_frequency(f0, amp)
            grid = np.linspace(dip - 100e3, dip + 100e3, 101)
            z = np.array([model.s21_lc_response(f, amp) for f in grid])
            entries.append({"frequencies": grid, "iq_complex": z,
                            "original_center_frequency": dip})
        ensure_fits(entries)
        out[dbm] = entries
    return out


def test_the_pull_is_downward(fits):
    for entry in fits[LOW_DBM]:
        assert entry["nonlinear_fit_success"]
        p = entry["nonlinear_fit_params"]
        assert p["a"] > 0, p
        f_min = entry["frequencies"][np.argmin(np.abs(entry["iq_complex"]))]
        assert p["fr"] > f_min, (p["fr"], f_min)


def test_the_nonlinearity_rises_with_drive(fits):
    for low, high in zip(fits[LOW_DBM], fits[HIGH_DBM]):
        assert high["nonlinear_fit_success"]
        assert high["nonlinear_fit_params"]["a"] > low["nonlinear_fit_params"]["a"]
