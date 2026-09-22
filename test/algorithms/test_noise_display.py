"""Saved noise products preserve units and I/Q correlation."""

import numpy as np
import pytest

from rfmux.algorithms.measurement.noise_display import noise_display_products
from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog
from rfmux.core.transferfunctions import (
    VOLTS_PER_ROC, decimation_to_sampling, spectrum_from_slow_tod,
)

pytestmark = pytest.mark.portable


def noise_block(calibrated=False, reference="absolute"):
    fs = decimation_to_sampling(6)
    signal = 3 + np.sin(2 * np.pi * np.arange(128) / 16)
    iq = signal * (1 + 1j)
    spectrum = spectrum_from_slow_tod(
        iq.real, iq.imag, 6, nsegments=2, reference=reference, input_units="volts")
    catalog = ResonatorCatalog(module=1, resonators=[
        Resonator(name=name, channel=channel, bias=BiasPoint(
            frequency_hz=4e9, amplitude=.1,
            dI_df=2. if calibrated else None, dQ_df=2. if calibrated else None))
        for name, channel in (("A", 2), ("B", 7))])
    return dict(measurement="noise", module=1, call_params=dict(
        catalog=catalog.to_dict(), nsegments=2, spectrum_cutoff=.9),
        results=dict(info=dict(iq_units="adc_counts", reference=reference,
                     decimation=6, slow_sample_rate_hz=fs),
                     shared_slow=dict(freq_iq=spectrum["freq_iq"]), resonators={
                         r.name: dict(channel=r.channel, slow_data=dict(
                             iq_counts=iq / VOLTS_PER_ROC,
                             psd_i=spectrum["psd_i"], psd_q=spectrum["psd_q"]))
                         for r in catalog}))


def test_absolute_voltage_spectra_are_read_without_recomputation(monkeypatch):
    block = noise_block()
    monkeypatch.setattr("rfmux.algorithms.measurement.noise_display.spectrum_from_slow_tod",
                        lambda *a, **k: pytest.fail("must read saved PSD"))
    products = noise_display_products(block, "B")
    np.testing.assert_allclose(products["iq"].real[:2], [3, 3 + np.sin(np.pi / 8)])
    expected = .1 * 10 ** (
        block["results"]["resonators"]["B"]["slow_data"]["psd_i"] / 10)
    np.testing.assert_allclose(products["psd_i"], expected)


def test_df_rotation_preserves_correlation():
    block = noise_block(calibrated=True)
    products = noise_display_products(block, "A", units="df")
    volts = noise_display_products(block, "A")
    np.testing.assert_allclose(products["iq"].real, volts["iq"].real / 2)
    np.testing.assert_allclose(products["iq"].imag, 0, atol=1e-15)
    np.testing.assert_allclose(products["psd_i"], volts["psd_i"] / 4, atol=1e-25)
    assert products["psd_q"].max() < 1e-25


def test_relative_spectra_reconstruct_absolute_voltage_units():
    absolute = noise_display_products(noise_block(), "A")
    relative = noise_display_products(noise_block(reference="relative"), "A")
    np.testing.assert_allclose(relative["psd_i"], absolute["psd_i"], atol=1e-25)


def test_timestream_does_not_compute_spectra(monkeypatch):
    block = noise_block(calibrated=True)
    monkeypatch.setattr("rfmux.algorithms.measurement.noise_display.spectrum_from_slow_tod",
                        lambda *a, **k: pytest.fail("TOD needs no PSD"))
    assert "frequency_hz" not in noise_display_products(
        block, "A", units="df", include_psd=False)


def test_prepared_spectra_are_read_without_recalculation(monkeypatch):
    from rfmux.algorithms.measurement.noise_display import prepare_noise_display
    block = noise_block(calibrated=True)
    prepare_noise_display(block)
    expected = noise_display_products(block, "A", units="df")
    monkeypatch.setattr("rfmux.algorithms.measurement.noise_display.spectrum_from_slow_tod",
                        lambda *a, **k: pytest.fail("must read prepared PSD"))
    actual = noise_display_products(block, "A", units="df")
    np.testing.assert_array_equal(actual["psd_i"], expected["psd_i"])


@pytest.mark.parametrize("reference", ["absolute", "relative"])
def test_pfb_df_psd_uses_same_calibration_and_correction(reference):
    from rfmux.algorithms.measurement.py_get_pfb_samples import apply_pfb_correction
    from rfmux.core.transferfunctions import PFB_SAMPLING_FREQ
    block = noise_block(calibrated=True, reference=reference)
    info = block["results"]["info"]
    info.update(nco_frequency_hz=4e9, pfb_binlim_hz=1e6,
                pfb_trim=False, pfb_nsegments=2)
    iq = block["results"]["resonators"]["A"]["slow_data"]["iq_counts"]
    freq, pi, pq, _, _ = apply_pfb_correction(
        iq, 4e9, 4e9, binlim=1e6, trim=False, nsegments=2, reference=reference)
    record = block["results"]["resonators"]["A"]
    block["results"]["shared_pfb"] = dict(
        time_s=np.arange(len(iq)) / PFB_SAMPLING_FREQ)
    record.update(bias_frequency_hz=4e9, pfb_data=dict(
        iq_counts=iq,
        freq_iq=freq, psd_i=pi, psd_q=pq))
    products = noise_display_products(block, "A", stream="pfb", units="df")
    f, expected, _, _, _ = apply_pfb_correction(
        iq / (2 + 2j), 4e9, 4e9, binlim=1e6, trim=False,
        nsegments=2, reference="absolute")
    np.testing.assert_array_equal(products["frequency_hz"], f)
    np.testing.assert_allclose(products["psd_i"], .1 * 10 ** (expected / 10))
