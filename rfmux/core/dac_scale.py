"""Read a module's DAC full-scale power, and check the drive of a tone comb
against it."""

from typing import Iterable, Optional

import numpy as np

from .transferfunctions import BALUN_LOSS_DB, CREST_FACTOR

__all__ = [
    "DAC_SCALE_CURRENT_RANGE_A",
    "DAC_SCALE_OUTPUT_RANGE_DBM",
    "DAC_SCALE_RECOMMENDED_SETTING_DBM",
    "DAC_SCALE_SETTING_RANGE_DBM",
    "QUANTIZATION_PEAK_FRACTION",
    "comb_drive_issues",
    "comb_peak_fraction",
    "dac_scale_dbm",
    "dac_scale_setting_dbm",
]

#: The drive currents that ``set_dac_scale`` accepts, in amps (crs-mkids,
#: crs::set_dac_scale). The board refuses a value outside this range.
DAC_SCALE_CURRENT_RANGE_A = (6.425e-6, 40.5e-3)

#: The same range as DAC scale settings in dBm. The board converts dBm to a
#: current into 100 ohm: W = 3.125 * I**2.
DAC_SCALE_SETTING_RANGE_DBM = tuple(
    float(10 * np.log10(3.125 * i ** 2 * 1e3)) for i in DAC_SCALE_CURRENT_RANGE_A)

#: The recommended DAC scale setting: 20 mA, +1.0 dBm (DS926, DAC at 3.0 V).
DAC_SCALE_RECOMMENDED_SETTING_DBM = 1.0

#: The same range at the output, after the balun loss.
DAC_SCALE_OUTPUT_RANGE_DBM = tuple(
    s - BALUN_LOSS_DB for s in DAC_SCALE_SETTING_RANGE_DBM)

#: A comb with an estimated peak below this fraction of DAC full scale uses
#: few DAC levels, so quantization noise can increase.
QUANTIZATION_PEAK_FRACTION = 0.01


def comb_peak_fraction(amplitudes: Iterable[float]) -> float:
    """Return the estimated peak of tones that play together, as a fraction of
    DAC full scale.

    With random phases, the RMS is sqrt(sum(a**2) / 2) and the peak is about
    CREST_FACTOR times the RMS. The sum of the amplitudes is the largest
    possible peak, so the estimate is not more than that sum.
    """
    a = np.asarray(list(amplitudes), dtype=float)
    return float(min(a.sum(), CREST_FACTOR * np.sqrt((a ** 2).sum() / 2)))


def comb_drive_issues(combs: Iterable[Iterable[float]],
                      dac_scale_dbm: Optional[float] = None
                      ) -> list[tuple[str, str]]:
    """Return ``(kind, message)`` for each problem of the drive.

    *combs* is the normalized amplitudes of the tones that play together, one
    collection for each step. *kind* is ``"clipping"`` if a step can go above
    full scale, or ``"quantization"`` if a step stays far below it. The message
    tells how to change the DAC scale, if *dac_scale_dbm* (at the output)
    permits a change in the data-sheet range.
    """
    peaks = [comb_peak_fraction(c) for c in combs]
    if not peaks:
        return []
    low, high = DAC_SCALE_OUTPUT_RANGE_DBM
    issues = []
    if max(peaks) > 1.0:
        advice = (" Raise the DAC scale and lower the normalized amplitude, to "
                  "keep the same power."
                  if dac_scale_dbm is None or dac_scale_dbm < high
                  else " Lower the amplitude.")
        issues.append(("clipping",
                       f"The tones reach an estimated {max(peaks):.2f} of DAC "
                       f"full scale (crest factor {CREST_FACTOR:g}). The "
                       f"output can clip.{advice}"))
    if min(peaks) < QUANTIZATION_PEAK_FRACTION:
        advice = (" Lower the DAC scale and raise the normalized amplitude, to "
                  "keep the same power."
                  if dac_scale_dbm is None or dac_scale_dbm > low
                  else " Raise the amplitude.")
        issues.append(("quantization",
                       f"The tones reach only an estimated {min(peaks):.3f} of "
                       f"DAC full scale. Quantization noise can increase."
                       f"{advice}"))
    return issues


async def dac_scale_setting_dbm(crs, module: int) -> Optional[float]:
    """Return the module's DAC scale as programmed with ``set_dac_scale``, in dBm.

    Return None if the board reports no scale or analog banking makes the
    module inaccessible. Other board errors propagate to the caller.
    """
    try:
        scale = await crs.get_dac_scale('DBM', module=module)
    except Exception as e:
        if "Can't access module" in str(e) and "analog banking" in str(e):
            return None
        raise
    return None if scale is None else float(scale)


async def dac_scale_dbm(crs, module: int) -> Optional[float]:
    """Return the module's DAC full-scale power at the output in dBm: the
    programmed scale less ``BALUN_LOSS_DB``."""
    setting = await dac_scale_setting_dbm(crs, module)
    return None if setting is None else setting - BALUN_LOSS_DB
