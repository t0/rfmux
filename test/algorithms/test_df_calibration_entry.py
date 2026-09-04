"""df_calibration_for_entry: the best fit a multisweep result carries
decides the calibration; a fit is run, and kept, only when it has
none."""
import warnings

import numpy as np
import pytest

from rfmux.algorithms.measurement import df_calibration as dc
from rfmux.algorithms.measurement.fitting_nonlinear import nonlinear_iq
from rfmux.core.transferfunctions import VOLTS_PER_ROC

FR, QR = 1.0e9, 1.5e5
PARAMS = {"fr": FR, "Qr": QR, "amp": 0.6, "phi": 0.1, "a": 0.0,
          "i0": 1.0, "q0": 0.2}


def _entry(span=20e3, n=41):
    f = np.linspace(FR - span / 2, FR + span / 2, n)
    z = nonlinear_iq(f, *[PARAMS[k] for k in ("fr", "Qr", "amp", "phi", "a", "i0", "q0")])
    return {"frequencies": f, "iq_complex": z * 2e5 / VOLTS_PER_ROC / 2e5,
            "bias_frequency": FR}


def _exact(entry):
    f = entry["frequencies"]
    z = np.asarray(entry["iq_complex"]) * VOLTS_PER_ROC
    return 1.0 / ((np.interp(FR + 1, f, z.real) - np.interp(FR - 1, f, z.real)
                   + 1j * (np.interp(FR + 1, f, z.imag) - np.interp(FR - 1, f, z.imag))) / 2)


def test_nonlinear_params_are_used_without_fitting(monkeypatch):
    entry = _entry()
    entry["nonlinear_fit_params"] = dict(PARAMS)
    entry["gain_complex"] = 1.0
    monkeypatch.setattr(dc, "fit_for_calibration",
                        lambda *a, **k: pytest.fail("should not fit"))
    cal = dc.df_calibration_for_entry(entry)
    assert cal is not None
    # The stored parameters are the truth here, up to the counts-to-volts
    # scale the entry's IQ carries.
    assert abs(cal) / abs(_exact(entry)) == pytest.approx(1.0, rel=0.05)


def test_skewed_params_are_the_fallback(monkeypatch):
    entry = _entry()
    entry["fit_params"] = {"fr": FR, "Qr": QR, "Qcre": QR / 0.6, "Qcim": 0.0}
    monkeypatch.setattr(dc, "fit_for_calibration",
                        lambda *a, **k: pytest.fail("should not fit"))
    cal = dc.df_calibration_for_entry(entry)
    assert cal is not None and np.isfinite(cal)
    assert abs(cal) / abs(_exact(entry)) == pytest.approx(1.0, rel=0.1)


def test_no_fit_means_a_warning_a_fit_and_a_stored_fit():
    entry = _entry()
    with pytest.warns(UserWarning, match="carries no fit"):
        cal = dc.df_calibration_for_entry(entry)
    assert cal is not None and np.isfinite(cal)
    assert entry["nonlinear_fit_params"]["fr"] == pytest.approx(FR, rel=1e-4)
    assert entry["nonlinear_fit_success"] is True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        again = dc.df_calibration_for_entry(entry)
    assert again == cal


def test_fit_if_missing_false_returns_none():
    assert dc.df_calibration_for_entry(_entry(), fit_if_missing=False) is None


def test_bias_frequency_from_fit_uses_stored_params(monkeypatch):
    # A symmetric resonance: the IQ trajectory is fastest, and |S21|
    # least, on resonance.  The sweep grid straddles fr without a point
    # on it, so a grid-bound answer would be off by half a step.
    entry = _entry(n=40)
    entry["nonlinear_fit_params"] = dict(PARAMS)
    entry["gain_complex"] = 1.0
    monkeypatch.setattr(dc, "fit_for_calibration",
                        lambda *a, **k: pytest.fail("should not fit"))
    step = entry["frequencies"][1] - entry["frequencies"][0]
    assert abs(dc.bias_frequency_from_fit(entry) - FR) < step / 20
    # |S21| is least on resonance only without the mismatch angle.
    entry["nonlinear_fit_params"]["phi"] = 0.0
    assert abs(dc.bias_frequency_from_fit(entry, "min-s21") - FR) < step / 20


def test_bias_frequency_from_fit_fits_and_keeps_the_fit():
    entry = _entry(n=40)
    f_bias = dc.bias_frequency_from_fit(entry)
    assert f_bias == pytest.approx(FR, abs=50.0)
    assert entry["nonlinear_fit_params"]["fr"] == pytest.approx(FR, rel=1e-4)
    assert entry["nonlinear_fit_success"] is True


def test_bias_frequency_from_fit_declines_unknown_method_and_failed_fit(monkeypatch):
    assert dc.bias_frequency_from_fit(_entry(), None) is None
    monkeypatch.setattr(dc, "fit_for_calibration", lambda *a, **k: None)
    assert dc.bias_frequency_from_fit(_entry()) is None
