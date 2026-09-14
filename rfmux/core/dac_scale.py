"""The module DAC scale, as amplitudes are labelled against it.

A normalized amplitude is turned into a power by adding the module's DAC
scale to ``20*log10(amplitude)`` -- :meth:`rfmux.core.resonators.BiasPoint.
power_dbm` and :func:`rfmux.tuning.multisweep_amplitudes.describe` both do
exactly that. This module is where the scale is read from the board, so
that arithmetic has one input and every label agrees.

.. warning::

   **UNEXPLAINED CONSTANT:** ``DAC_SCALE_LABEL_OFFSET_DB`` subtracts a
   fixed number of dB from the board's own answer before anything is
   labelled. It arrived as 1.5 dB with no recorded physical motivation and
   is **0.0** here, so a label is the board's own number. Every power this
   package reports moves with it, so if the 1.5 dB turns out to mean
   something -- a fixed loss between the DAC and the connector, say -- put
   it back *with the reason written beside it*, and expect every recorded
   power from before that change to be off by the difference.
"""

from typing import Optional

__all__ = ["DAC_SCALE_LABEL_OFFSET_DB", "dac_scale_dbm"]

#: Subtracted from the board's DAC scale before amplitudes are labelled
#: against it. See the module warning before changing this: it shifts every
#: power this package reports, in files as well as on screen.
DAC_SCALE_LABEL_OFFSET_DB = 0.0


async def dac_scale_dbm(crs, module: int) -> Optional[float]:
    """The module's DAC scale in dBm as amplitudes are labelled against it.

    None when the board reports none: a module the analog banking does not
    expose has no scale, which is an answer rather than a failure.
    """
    try:
        scale = await crs.get_dac_scale('DBM', module=module)
    except Exception as e:
        if "Can't access module" in str(e) and "analog banking" in str(e):
            return None
        raise
    return None if scale is None else float(scale) - DAC_SCALE_LABEL_OFFSET_DB
