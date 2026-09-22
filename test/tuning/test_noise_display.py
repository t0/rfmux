"""Noise df products preserve native data and I/Q correlation."""

import numpy as np
import pytest

from rfmux.tuning.noise import noise_to_df
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


def test_native_products_are_retained_and_input_is_not_mutated():
    block = noise_block(calibrated=True)
    source = block["results"]["resonators"]["A"]["slow_data"]
    source["display_psds"] = {"obsolete": object()}
    converted = noise_to_df(block)
    data = converted["results"]["resonators"]["A"]["slow_data"]
    assert data["iq_counts"] is source["iq_counts"]
    assert data["psd_i"] is source["psd_i"]
    assert data["psd_q"] is source["psd_q"]
    assert "time_s" not in data
    assert "display_psds" not in data
    assert "display_psds" in source


def test_df_rotation_preserves_correlation():
    block = noise_block(calibrated=True)
    converted = noise_to_df(block)
    data = converted["results"]["resonators"]["A"]["slow_data"]
    volts = data["iq_counts"] * VOLTS_PER_ROC
    np.testing.assert_allclose(data["df_hz"].real, volts.real / 2)
    np.testing.assert_allclose(data["df_hz"].imag, 0, atol=1e-15)
    np.testing.assert_allclose(
        data["psd_df"], .1 * 10 ** (data["psd_i"] / 10) / 4,
        atol=1e-25,
    )
    assert data["psd_dissipation"].max() < 1e-25
    assert converted["results"]["info"]["df_timestream_units"] == "Hz"
    assert converted["results"]["info"]["df_spectrum_units"] == "Hz²/Hz"


def test_uncalibrated_resonators_keep_only_native_products():
    converted = noise_to_df(noise_block())
    data = converted["results"]["resonators"]["A"]["slow_data"]
    assert "df_hz" not in data
    assert "psd_df" not in data
    assert "df_timestream_units" not in converted["results"]["info"]


@pytest.mark.parametrize("reference", ["absolute", "relative"])
def test_pfb_df_psd_uses_same_calibration_and_correction(reference):
    from rfmux.tuning.noise import apply_pfb_correction
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
    converted = noise_to_df(block)
    products = converted["results"]["resonators"]["A"]["pfb_data"]
    f, expected, _, _, _ = apply_pfb_correction(
        iq / (2 + 2j), 4e9, 4e9, binlim=1e6, trim=False,
        nsegments=2, reference="absolute")
    np.testing.assert_allclose(products["psd_df"], .1 * 10 ** (expected / 10))
