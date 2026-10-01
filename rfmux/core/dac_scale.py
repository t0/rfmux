"""Read a module's DAC full-scale power for amplitude conversion."""

from typing import Optional

from .transferfunctions import BALUN_LOSS_DB

__all__ = ["dac_scale_dbm"]


async def dac_scale_dbm(crs, module: int) -> Optional[float]:
    """Return the module's DAC full-scale power at the output in dBm: the
    programmed scale less ``BALUN_LOSS_DB``.

    Return None if the board reports no scale or analog banking makes the
    module inaccessible. Other board errors propagate to the caller.
    """
    try:
        scale = await crs.get_dac_scale('DBM', module=module)
    except Exception as e:
        if "Can't access module" in str(e) and "analog banking" in str(e):
            return None
        raise
    return None if scale is None else float(scale) - BALUN_LOSS_DB
