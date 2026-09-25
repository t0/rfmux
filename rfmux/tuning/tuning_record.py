"""Convert catalog bias points to per-channel tuning rows for pulse captures.

The rows include each bias point's calibration sweep. Read them back as a
catalog with :func:`catalog_from_tuning`, or as a multisweep for plotting with
:func:`multisweep_from_tuning`.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np

from ..core.resonators import BiasPoint, Resonator, ResonatorCatalog
from ..core.transferfunctions import VOLTS_PER_ROC
from .multisweep_amplitudes import AmplitudeSchedule
from .sweep_results import pack_multisweep

__all__ = ["tuning_rows", "catalog_from_tuning", "multisweep_from_tuning"]

#: The sweep a BiasPoint carries, flattened into the row and read back out.
_SWEEP_FIELDS = BiasPoint.BIAS_SWEEP_KEYS


def tuning_rows(
    catalog: ResonatorCatalog,
    *,
    nco_frequency_hz: Optional[float] = None,
    dac_scale_dbm: Optional[float] = None,
    nsamps: Optional[int] = None,
) -> Dict[int, dict]:
    """Return ``{channel: row}`` with each resonator's bias and calibration.

    Optional measurement settings are added to every row. A setting passed
    as None is omitted.
    """
    provenance = {k: v for k, v in
                  (("nco_frequency_hz", nco_frequency_hz),
                   ("dac_scale_dbm", dac_scale_dbm),
                   ("nsamps", nsamps)) if v is not None}
    rows: Dict[int, dict] = {}
    for r in catalog:
        bias = r.bias
        row = {
            "name": r.name,
            "bias_frequency": bias.frequency_hz,
            "amplitude": bias.amplitude,
            # Hz/V at the bias point, from the IQ derivatives measured there.
            "df_calibration": bias.df_calibration,
            # Loop rotation is stored separately from the df calibration.
            "iq_rotation_deg": bias.iq_rotation_deg,
            "bifurcated_at": bias.bifurcated_at,
        }
        if bias.bias_sweep is not None:
            row.update({k: bias.bias_sweep[k] for k in _SWEEP_FIELDS
                        if k in bias.bias_sweep})
        row.update(provenance)
        rows[r.channel] = row
    return rows


def _bias_point(row: Mapping) -> BiasPoint:
    """One row's tone and the calibration measured at it."""
    cal = row.get("df_calibration")
    d = None if cal in (None, 0) else 1.0 / complex(cal)
    sweep = {k: row[k] for k in _SWEEP_FIELDS if row.get(k) is not None}
    # A stored sweep requires both frequency and IQ arrays.
    if any(k not in sweep for k in BiasPoint._SWEEP_TRACES):
        sweep = {}
    return BiasPoint(
        frequency_hz=float(row["bias_frequency"]),
        amplitude=float(row["amplitude"]),
        dI_df=None if d is None else float(d.real),
        dQ_df=None if d is None else float(d.imag),
        iq_rotation_deg=row.get("iq_rotation_deg"),
        bifurcated_at=row.get("bifurcated_at"),
        bias_sweep=sweep or None,
    )


def catalog_from_tuning(rows: Mapping[int, dict], module: int,
                        **kwargs) -> ResonatorCatalog:
    """Build a catalog from a capture's tuning rows.

    Skip rows without a bias frequency or amplitude. Use ``"channel <n>"``
    when a row has no name. Pass remaining keywords to ResonatorCatalog.
    """
    resonators = []
    for channel, row in sorted(rows.items()):
        if not isinstance(row, dict):
            continue
        if row.get("bias_frequency") is None or row.get("amplitude") is None:
            continue
        resonators.append(Resonator(
            name=str(row.get("name") or f"channel {int(channel)}"),
            channel=int(channel),
            bias=_bias_point(row),
        ))
    return ResonatorCatalog(resonators, module=module, **kwargs)


def multisweep_from_tuning(rows: Mapping[int, dict], module: int, *,
                           module_id: str) -> dict:
    """Pack stored calibration sweeps as ``{module_id: block}`` for plotting.

    The result has one amplitude step. Each resonator keeps its own amplitude
    and recorded sweep direction; a missing direction defaults to upward.
    ``nsamps`` and ``dac_scale_dbm`` are read from the rows when available,
    and otherwise remain None.

    Raise ValueError if no row contains a calibration sweep.
    """
    catalog = catalog_from_tuning(rows, module)
    by_direction: Dict[str, Dict[str, dict]] = {}
    npoints, nsamps, dac_scale = 0, None, None
    for r in catalog:
        sweep = r.bias.bias_sweep
        if sweep is None:
            continue
        frequencies = np.asarray(sweep["frequencies"], dtype=float)
        iq_volts = np.asarray(sweep["iq_volts"])
        entry = {
            "channel": r.channel,
            "frequencies": frequencies,
            # Reconstruct counts from the stored voltage trace using VOLTS_PER_ROC.
            "iq_counts": iq_volts / VOLTS_PER_ROC,
            "iq_volts": iq_volts,
            "original_center_frequency": float(
                sweep.get("original_center_frequency", r.bias.frequency_hz)),
            "sweep_direction": sweep.get("sweep_direction") or "upward",
            "sweep_amplitude": float(
                sweep.get("sweep_amplitude", r.bias.amplitude)),
        }
        by_direction.setdefault(entry["sweep_direction"], {})[r.name] = entry
        npoints = max(npoints, frequencies.size)
        row = rows.get(r.channel) or {}
        nsamps = nsamps or row.get("nsamps")
        # Tested against None rather than truthiness: 0.0 dBm is a scale.
        if dac_scale is None:
            dac_scale = row.get("dac_scale_dbm")

    if not by_direction:
        raise ValueError("no tuning row carries a sweep to show")

    return pack_multisweep(
        {0: by_direction},
        module_id=module_id,
        module=module,
        amp_schedule=AmplitudeSchedule(),
        directions=list(by_direction),
        span_hz=_span(by_direction),
        npoints_per_sweep=npoints,
        nsamps=nsamps,
        catalog=catalog,
        dac_scale_dbm=dac_scale,
    )


def _span(by_direction: Mapping[str, Mapping[str, dict]]) -> float:
    """Return the largest frequency span among the stored sweeps, in Hz."""
    return max(
        (float(e["frequencies"].max() - e["frequencies"].min())
         for entries in by_direction.values() for e in entries.values()
         if e["frequencies"].size > 1),
        default=0.0,
    )
