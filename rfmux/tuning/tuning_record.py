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
    """``{channel: row}`` for every resonator in *catalog*.

    The keyword arguments are facts about the measurement rather than about
    the array -- a catalog holds no NCO on purpose -- so they are passed in
    and stamped onto every row. One left out is left out of the rows: a
    reader finds the field absent rather than a number nothing measured.
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
            # A separate quantity: the angle the IQ loop is rotated by, which
            # is not derivable from df_calibration and is not measured yet.
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
    # The scalars without the traces do not describe a sweep anything can be
    # read off, and BiasPoint refuses the pair. A row like that has no sweep.
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
    """The catalog a capture's tuning rows describe.

    The inverse of :func:`tuning_rows`. A row with no tone -- no bias
    frequency or no amplitude -- is not a resonator we can say anything
    about, so it is left out rather than given a placeholder. A row with no
    name is named after its channel, which is what it is known by in the
    file it came from.
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
    """A capture's tuning rows as one multisweep, for reading and plotting.

    One iteration: the sweep each resonator is biased at is the one sweep
    it has, so the schedule is ``AmplitudeSchedule()`` -- one pass, each
    resonator at its own amplitude -- and that is what the rows record
    rather than a step everything shares. Resonators biased on traces taken
    in different directions land in the direction they were measured in.

    Goes through the same packer a measurement does, so what comes back is
    a container and not a shape that resembles one. ``nsamps`` and
    ``dac_scale_dbm`` are provenance the file may not carry; they reach the
    container as whatever the rows say, or None.

    Raises:
        ValueError: if no row carries a sweep. There is nothing to draw,
            and an empty container would read as a measurement of nothing.
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
            # The counts the volts were converted from, by the one constant
            # that conversion uses. Kept out of a stored sweep for exactly
            # that reason; rebuilt here because a sweep entry has it.
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
    """The widest sweep in the set, which is the span they were taken at."""
    return max(
        (float(e["frequencies"].max() - e["frequencies"].min())
         for entries in by_direction.values() for e in entries.values()
         if e["frequencies"].size > 1),
        default=0.0,
    )
