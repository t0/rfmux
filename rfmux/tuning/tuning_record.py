"""The tuning record a capture stores beside each channel's pulses.

One row per readout channel, built from the catalog's bias points. The row
is what ``rfmux.pulse_capture`` writes into a capture file's ``tuning``
group and reads back: the HDF5 layer types a row by value and never looks
at the field names, so this module is the only place that says what a row
holds.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..core.resonators import ResonatorCatalog

__all__ = ["tuning_rows"]

#: The sweep keys a BiasPoint's bias_sweep carries, flattened into the row.
_SWEEP_FIELDS = (
    "frequencies",
    "iq_volts",
    "original_center_frequency",
    "sweep_amplitude",
    "sweep_direction",
)


def tuning_rows(
    catalog: ResonatorCatalog,
    *,
    nco_frequency_hz: Optional[float] = None,
    dac_scale_dbm: Optional[float] = None,
) -> Dict[int, dict]:
    """``{channel: row}`` for every resonator in *catalog* with a bias point.

    ``nco_frequency_hz`` and ``dac_scale_dbm`` are facts about the board at
    capture time rather than about the array -- a catalog holds no NCO on
    purpose -- so they are passed in and stamped onto every row.
    """
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
            row.update({k: bias.bias_sweep.get(k) for k in _SWEEP_FIELDS})
        if nco_frequency_hz is not None:
            row["nco_frequency_hz"] = float(nco_frequency_hz)
        if dac_scale_dbm is not None:
            row["dac_scale_dbm"] = float(dac_scale_dbm)
        rows[r.channel] = row
    return rows
