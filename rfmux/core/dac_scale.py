"""Read a module's DAC full-scale power for amplitude conversion."""

from typing import Optional

__all__ = ["DAC_SCALE_LABEL_OFFSET_DB", "dac_scale_dbm"]

#: Subtracted from the board's DAC scale. Keep zero unless a physical loss
#: is established; changing it shifts all reported and saved drive powers.
DAC_SCALE_LABEL_OFFSET_DB = 0.0


async def dac_scale_dbm(crs, module: int) -> Optional[float]:
    """Return the module's DAC full-scale power in dBm, with the label offset.

    Return None if the board reports no scale or analog banking makes the
    module inaccessible. Other board errors propagate to the caller.
    """
    try:
        scale = await crs.get_dac_scale('DBM', module=module)
    except Exception as e:
        if "Can't access module" in str(e) and "analog banking" in str(e):
            return None
        raise
    return None if scale is None else float(scale) - DAC_SCALE_LABEL_OFFSET_DB
