"""Prepare and read display products for noise module blocks."""

from __future__ import annotations

import numpy as np

from ...core.resonators import ResonatorCatalog
from ...core.transferfunctions import (
    VOLTS_PER_ROC, convert_dbm_to_volts, spectrum_from_slow_tod,
)
from .py_get_pfb_samples import apply_pfb_correction


def noise_catalog(block: dict) -> ResonatorCatalog | None:
    snapshot = block["call_params"].get("catalog")
    return ResonatorCatalog.from_dict(snapshot) if snapshot else None


def noise_display_products(
    block: dict, name: str, *, stream: str = "slow", units: str = "volts",
    include_psd: bool = True, include_tod: bool = True,
    catalog: ResonatorCatalog | None = None,
) -> dict:
    """Return complex TOD and component PSDs in V/V²/Hz or Hz/Hz²/Hz.

    Prepared spectra are read directly. Without them, absolute voltage
    spectra need only a unit conversion; relative and df spectra need the
    saved IQ and the acquisition's spectral settings.
    The df rotation retains I/Q correlation; rotating separate powers would
    lose it. No mean subtraction or carrier omission changes the measurement.
    """
    if block.get("measurement") != "noise":
        raise ValueError("Expected a noise module block.")
    if units not in ("volts", "df") or stream not in ("slow", "pfb"):
        raise ValueError("Use volts/df units and slow/pfb stream.")
    result, params = block["results"], block["call_params"]
    info = result["info"]
    if info["iq_units"] not in ("volts", "counts", "adc_counts"):
        raise ValueError("Expected noise IQ in volts or counts.")
    record = result["resonators"][name]
    data = record[f"{stream}_data"]
    saved = data.get("display_psds", {}).get(units)
    if include_psd and not include_tod and saved is not None:
        return dict(saved)
    iq = (np.asarray(data["iq_volts"]) if info["iq_units"] == "volts" else
          np.asarray(data["iq_counts"]) * VOLTS_PER_ROC)
    if units == "df":
        catalog = catalog if catalog is not None else noise_catalog(block)
        factor = catalog[name].bias.df_calibration if catalog else None
        if factor is None:
            raise ValueError(f"{name} has no df calibration.")
        iq = iq * factor
    time = (np.asarray(result["shared_pfb"]["time_s"]) if stream == "pfb" else
            np.arange(len(iq)) / info["slow_sample_rate_hz"])
    products = dict(time_s=time, iq=iq) if include_tod else {}
    if not include_psd:
        return products
    if saved is not None:
        return dict(products, **saved)
    shared = result["shared_slow"] if stream == "slow" else data
    frequency = np.asarray(shared["freq_iq"])
    if units == "volts" and info["reference"] == "absolute":
        psd_i, psd_q = data["psd_i"], data["psd_q"]
    elif stream == "slow":
        spectrum = spectrum_from_slow_tod(
            iq.real, iq.imag, dec_stage=info["decimation"],
            nsegments=params["nsegments"], reference="absolute",
            spectrum_cutoff=params["spectrum_cutoff"], input_units="volts")
        frequency = spectrum["freq_iq"]
        psd_i, psd_q = spectrum["psd_i"], spectrum["psd_q"]
    else:
        nco = info.get("nco_frequency_hz")
        if nco is None:
            raise ValueError("This file lacks the NCO frequency needed to "
                             "reconstruct calibrated PFB spectra.")
        frequency, psd_i, psd_q, _, _ = apply_pfb_correction(
            iq / VOLTS_PER_ROC, nco, record["bias_frequency_hz"],
            binlim=info["pfb_binlim_hz"],
            trim=info["pfb_trim"],
            nsegments=info["pfb_nsegments"], reference="absolute")
    return dict(products, frequency_hz=frequency,
                psd_i=convert_dbm_to_volts(psd_i) ** 2,
                psd_q=convert_dbm_to_volts(psd_q) ** 2)


def prepare_noise_display(block: dict) -> None:
    """Store linear voltage and available df spectra alongside helper spectra."""
    catalog = noise_catalog(block)
    for name, record in block["results"]["resonators"].items():
        units = ["volts"]
        if catalog and catalog[name].bias.df_calibration is not None:
            units.append("df")
        for stream in ("slow", "pfb"):
            data_key = f"{stream}_data"
            if data_key not in record:
                continue
            spectra = record[data_key].setdefault("display_psds", {})
            for unit in units:
                products = noise_display_products(
                    block, name, stream=stream, units=unit,
                    include_tod=False, catalog=catalog)
                spectra[unit] = {key: products[key] for key in
                                 ("frequency_hz", "psd_i", "psd_q")}
