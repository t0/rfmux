"""multisweep's fit stage adds the chosen fits to every detector, and its
frequency helpers give one value per detector in index order."""

import pytest

from rfmux.algorithms.measurement import fitting
from rfmux.algorithms.measurement.multisweep import (
    bias_frequencies, fit_multisweep, fitted_frequencies)

FR = (100.0e6, 250.0e6, 400.0e6)


@pytest.fixture
def results():
    out = {}
    for k, fr in enumerate(FR, start=1):
        freqs, iq, _ = fitting.generate_test_resonator_skewed(fr=fr, noise_level=0.001)
        out[k] = {"frequencies": freqs, "iq_complex": iq,
                  "original_center_frequency": fr, "bias_frequency": fr + 50.0}
    return out


def test_skewed_fit_lands_on_each_resonance(results):
    fitted = fit_multisweep(results, skewed=True, max_workers=2)
    assert sorted(fitted) == [1, 2, 3]
    for k, fr in enumerate(FR, start=1):
        assert fitted[k]["skewed_fit_applied"] and fitted[k]["skewed_fit_success"]
        assert fitted[k]["fit_params"]["fr"] == pytest.approx(fr, abs=fr * 1e-5)
        assert len(fitted[k]["skewed_model_mag"]) == len(results[k]["frequencies"])
    assert "fit_params" not in results[1], "the input is not modified"


def test_no_fit_chosen_only_marks_the_entries(results):
    fitted = fit_multisweep(results)
    assert all(not e["skewed_fit_applied"] and not e["nonlinear_fit_applied"] for e in fitted.values())
    assert "fit_params" not in fitted[1]


def test_frequency_helpers_follow_detector_order(results):
    assert bias_frequencies(results) == [fr + 50.0 for fr in FR]
    fitted = fit_multisweep(results, skewed=True)
    assert fitted_frequencies(fitted) == [fitted[k]["fit_params"]["fr"] for k in (1, 2, 3)]
    assert fitted_frequencies(results) == bias_frequencies(results), "without a fit, the bias point"
