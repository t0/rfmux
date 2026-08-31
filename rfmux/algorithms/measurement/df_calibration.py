"""
Measure the df calibration at the current bias point.

``bias_kids`` produces a calibration as a by-product of a full sweep and
fit.  This is the same measurement on its own: sweep a narrow span around
where a channel is already biased, differentiate the I/Q trajectory, and
invert -- :func:`~rfmux.core.transferfunctions.convert_iq_to_df`.

Host-side, and it uses nothing but ``set_frequency`` and ``get_samples``,
so it runs against a board or against the simulator with the same code.
A simulated board has no calibration of its own to hand out: the number
is a measurement, not a property of the hardware.

Sweeping moves the channel's frequency and puts it back, so this changes
board state briefly.  Callers that must not disturb a tuned array should
take the calibration from ``bias_kids`` instead.
"""

import warnings
from typing import Dict, List

import numpy as np

from ...core.hardware_map import macro
from ...core.schema import CRS
from ...core.transferfunctions import convert_iq_to_df, convert_roc_to_volts

__all__ = ["measure_df_calibrations"]


@macro(CRS, register=True)
async def measure_df_calibrations(
    crs: CRS,
    channels: List[int],
    module: int = 1,
    span_hz: float = 100e3,
    resolution_hz: float = 500.0,
    n_samples: int = 10,
) -> Dict[int, complex]:
    """``{channel: calibration}`` measured where each channel sits now.

    Parameters
    ----------
    channels : list[int]
        Channels to measure.  Each is swept around its own bias frequency
        and restored afterwards.
    module : int
        Module index (1-based).
    span_hz : float
        Full width of the sweep, centred on the bias point.
    resolution_hz : float
        Spacing between points.  This is the parameter that matters:
        the spline has to resolve the resonance to differentiate it, so
        the spacing must be well inside the linewidth.  A few kHz of
        linewidth against 100 Hz spacing gives tens of points across the
        feature.  Coarser than the linewidth and the fit runs through
        the dip rather than around it -- the magnitude still looks
        plausible while the phase, which is the whole rotation, is
        wrong.
    n_samples : int
        Samples averaged per point.

    Returns
    -------
    dict
        Complex calibration per channel, as ``bias_kids`` reports it:
        magnitude is hertz per volt, phase is the angle from the (I, Q)
        axes to the frequency direction.  Channels whose sweep gives no
        usable derivative are left out rather than guessed at.
    """
    nco = await crs.get_nco_frequency(module=module)
    out: Dict[int, complex] = {}

    for channel in channels:
        rel = await crs.get_frequency(channel=channel, module=module)
        bias = nco + rel
        cal = None
        try:
            half = 0.5 * span_hz
            n_points = max(5, int(round(span_hz / resolution_hz)) + 1)
            freqs = np.linspace(bias - half, bias + half, n_points)
            iq = np.empty(n_points, dtype=complex)
            for k, f in enumerate(freqs):
                await crs.set_frequency(f - nco, channel=channel,
                                        module=module)
                s = await crs.get_samples(n_samples, channel=channel,
                                          module=module)
                # A board hands back an object with .i/.q; an in-process
                # caller can see the underlying dict.
                si = s["i"] if isinstance(s, dict) else s.i
                sq = s["q"] if isinstance(s, dict) else s.q
                iq[k] = np.mean(np.asarray(si)) + 1j * np.mean(np.asarray(sq))

            cal = convert_iq_to_df(np.array([1.0 + 0j]), bias, freqs,
                                   convert_roc_to_volts(iq))[0]
        except Exception as exc:
            # Skipping one channel is fine -- it simply gets no
            # calibration -- but skipping all of them silently would
            # look identical to a board that has none, so say which.
            warnings.warn(f"df calibration failed on channel {channel}: "
                          f"{exc}", stacklevel=2)
            cal = None
        finally:
            await crs.set_frequency(bias - nco, channel=channel,
                                    module=module)

        if cal is not None and np.isfinite(cal) and cal != 0:
            out[int(channel)] = complex(cal)

    return out
