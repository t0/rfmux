"""Processing helpers for measured noise timestreams."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..core.transferfunctions import spectrum_from_slow_tod
from . import store


_SPECTRUM_KEYS = ("psd_i", "psd_q", "psd_dual_sideband")


def remove_common_mode(
    noise_module_output: dict,
    *,
    save: bool | None = None,
    label: str | None = None,
) -> dict:
    """Remove the leading common SVD mode from one module's slow noise data.

    The decomposition is performed on the complex, mean-subtracted detector
    timestream matrix.  Each cleaned timestream retains its original mean, so
    its carrier and relative-reference PSD remain well defined.  The PFB data
    is not changed.

    The module block gains ``common_mode_removal``, containing the detector
    order, offsets, complete reduced SVD, and removed rank-one matrix.  Thus
    the input matrix is exactly ``cleaned + common_mode`` and the centered
    matrix can be recreated as ``cleaned - offset + common_mode``.
    Each processed resonator gains ``common_mode_removed=True``; its
    ``slow_data`` gains the cleaned IQ and PSD arrays with a
    ``_common_mode_removed`` suffix.

    Args:
        noise_module_output: one module block from ``measure_noise``, such as
            ``noise_output[crs.module[m].index()]``.
        save: update the source file or create a noise file.  None follows the
            configured autosave setting.
        label: filename label for a first save; an existing filename is kept.

    Returns:
        dict: the stored ``common_mode_removal`` entry.

    Raises:
        TypeError: if the input is not a mapping.
        ValueError: if it is not a compatible noise block, or fewer than two
            equal-length, finite slow timestreams are available.
    """
    if not isinstance(noise_module_output, dict):
        raise TypeError("Expected one module's noise output as a dict.")
    if noise_module_output.get("measurement") != "noise":
        raise ValueError("Expected one module's measure_noise output.")

    results = noise_module_output.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("Noise output has no results mapping.")
    resonators = results.get("resonators")
    info = results.get("info")
    if not isinstance(resonators, Mapping) or not isinstance(info, Mapping):
        raise ValueError("Noise output lacks resonator records or stream info.")

    iq_units = info.get("iq_units")
    if iq_units == "volts":
        iq_key, input_units = "iq_volts", "volts"
    elif iq_units in ("counts", "adc_counts"):
        iq_key, input_units = "iq_counts", "adc_counts"
    else:
        raise ValueError("Noise IQ units must be volts or counts.")

    names = list(resonators)
    if len(names) < 2:
        raise ValueError("Common-mode removal needs at least two resonators.")

    timestreams = []
    for name in names:
        record = resonators[name]
        slow = record.get("slow_data") if isinstance(record, Mapping) else None
        if not isinstance(slow, Mapping) or iq_key not in slow:
            raise ValueError(f"Resonator {name!r} has no slow {iq_key} timestream.")
        iq = np.asarray(slow[iq_key])
        if iq.ndim != 1:
            raise ValueError(
                f"Resonator {name!r} slow timestream must be one-dimensional."
            )
        if not np.issubdtype(iq.dtype, np.number) or not np.all(np.isfinite(iq)):
            raise ValueError(
                f"Resonator {name!r} slow timestream must be finite numeric data."
            )
        timestreams.append(iq.astype(np.complex128, copy=False))

    lengths = {len(iq) for iq in timestreams}
    if len(lengths) != 1:
        raise ValueError("All slow timestreams must have the same length.")

    params = noise_module_output.get("call_params")
    if not isinstance(params, Mapping):
        raise ValueError("Noise output has no acquisition parameters.")
    try:
        decimation = int(info["decimation"])
        nsegments = int(params["nsegments"])
        spectrum_cutoff = float(params["spectrum_cutoff"])
        reference = info["reference"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Noise output lacks its slow-spectrum settings.") from error
    sample_count = next(iter(lengths))
    if nsegments < 1 or sample_count < nsegments:
        raise ValueError("Slow timestreams need at least one sample per segment.")

    timestream_matrix = np.stack(timestreams, axis=0)
    offset = timestream_matrix.mean(axis=1, keepdims=True)
    centered = timestream_matrix - offset
    u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    common_mode = (u[:, :1] * singular_values[0]) @ vh[:1, :]
    cleaned = centered - common_mode + offset

    spectra = []
    for iq in cleaned:
        spectra.append(spectrum_from_slow_tod(
            iq.real,
            iq.imag,
            dec_stage=decimation,
            scaling="psd",
            nsegments=nsegments,
            reference=reference,
            spectrum_cutoff=spectrum_cutoff,
            input_units=input_units,
        ))

    removal = {
        "method": "complex_svd",
        "resonator_names": names,
        "rank": 1,
        "full_matrices": False,
        "mean_restored": True,
        "iq_key": iq_key,
        "offset": offset,
        "u": u,
        "singular_values": singular_values,
        "vh": vh,
        "common_mode": common_mode,
    }
    noise_module_output["common_mode_removal"] = removal
    suffix = "_common_mode_removed"
    for name, iq, spectrum in zip(names, cleaned, spectra):
        record = resonators[name]
        record["common_mode_removed"] = True
        slow = record["slow_data"]
        slow[f"{iq_key}{suffix}"] = iq
        for key in _SPECTRUM_KEYS:
            slow[f"{key}{suffix}"] = np.asarray(spectrum[key])

    store.maybe_save(noise_module_output, "noise", save=save, label=label)
    return removal


__all__ = ["remove_common_mode"]
