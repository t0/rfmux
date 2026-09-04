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
from typing import Dict, List, Optional

import numpy as np

from ...core.hardware_map import macro
from ...core.schema import CRS
from ...core.transferfunctions import convert_iq_to_df, convert_roc_to_volts

__all__ = ["measure_df_calibrations"]


def _local_fit(offsets, iq_volts, degree: int = 3):
    """The sweep replaced by a cubic fitted through its central half.

    The conversion differentiates a spline through the points at the
    bias frequency, and a spline's derivative at one point is set by
    its nearest neighbours: on a real sweep, whose points carry drift
    as well as noise, repeated measurements disagreed by a factor of
    two in magnitude.  A cubic over the central half of the span uses
    every point there; repeated sweeps then agree to a few percent.
    The fitted curve is what gets differentiated, so the conversion
    itself is untouched.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    half = 0.25 * (offsets[-1] - offsets[0])
    keep = np.abs(offsets) <= half
    if keep.sum() <= degree + 1:
        return iq_volts
    fit_i = np.polyfit(offsets[keep], iq_volts.real[keep], degree)
    fit_q = np.polyfit(offsets[keep], iq_volts.imag[keep], degree)
    smooth = np.array(iq_volts, dtype=complex)
    smooth[keep] = (np.polyval(fit_i, offsets[keep])
                    + 1j * np.polyval(fit_q, offsets[keep]))
    return smooth


@macro(CRS, register=True)
async def measure_df_calibrations(
    crs: CRS,
    channels: Optional[List[int]] = None,
    module: int = 1,
    span_hz: float = 20e3,
    resolution_hz: float = 500.0,
    n_samples: int = 10,
    progress=None,
) -> Dict[int, complex]:
    """``{channel: calibration}`` measured where each channel sits now.

    Parameters
    ----------
    channels : list[int], optional
        Channels to measure.  Each is swept around its own bias frequency
        and restored afterwards.  None means every channel the module
        reports as biased (get_biased_channels).
    module : int
        Module index (1-based).
    span_hz : float
        Full width of the sweep, centred on the bias point.  The slope
        is fitted over the central half of it, so about three
        linewidths is right: wide enough for the fit to see the
        curvature, narrow enough that a cubic still describes it.
    resolution_hz : float
        Spacing between points.  The fit wants a dozen or more points
        across its window, so the spacing must be well inside the
        linewidth.
    n_samples : int
        Samples averaged per point.
    progress : callable, optional
        Called as ``progress(done, total)`` per sweep point.

    Returns
    -------
    dict
        Complex calibration per channel, as ``bias_kids`` reports it:
        magnitude is hertz per volt, phase is the angle from the (I, Q)
        axes to the frequency direction.  Channels whose sweep gives no
        usable derivative are left out rather than guessed at.
    """
    if channels is None:
        from .channel_selection import get_biased_channels
        channels = await get_biased_channels(crs, module)
    channels = [int(c) for c in channels or []]
    if not channels:
        return {}
    nco = await crs.get_nco_frequency(module=module)
    bias: Dict[int, float] = {}
    for channel in channels:
        rel = await crs.get_frequency(channel=channel, module=module)
        bias[channel] = nco + rel

    half = 0.5 * span_hz
    n_points = max(5, int(round(span_hz / resolution_hz)) + 1)
    offsets = np.linspace(-half, half, n_points)
    iq = {ch: np.full(n_points, np.nan, dtype=complex) for ch in channels}
    try:
        # Every channel steps together: one batched frequency write and
        # one read of the module per point, not a round trip per channel
        # per point.  On the simulator each read runs the whole array's
        # physics, so per channel the old loop was minutes at 100 tones.
        for k, off in enumerate(offsets):
            if progress is not None:
                progress(k, n_points)
            async with crs.tuber_context() as ctx:
                for ch in channels:
                    ctx.set_frequency(bias[ch] + off - nco, channel=ch,
                                      module=module)
                await ctx()
            s = await crs.get_samples(n_samples, module=module)
            # A board hands back an object with .i/.q; an in-process
            # caller can see the underlying dict.  Both index channels
            # from zero.
            si = s["i"] if isinstance(s, dict) else s.i
            sq = s["q"] if isinstance(s, dict) else s.q
            for ch in channels:
                iq[ch][k] = (np.mean(np.asarray(si[ch - 1]))
                             + 1j * np.mean(np.asarray(sq[ch - 1])))
    finally:
        async with crs.tuber_context() as ctx:
            for ch in channels:
                ctx.set_frequency(bias[ch] - nco, channel=ch, module=module)
            await ctx()

    out: Dict[int, complex] = {}
    for ch in channels:
        try:
            cal = convert_iq_to_df(
                np.array([1.0 + 0j]), bias[ch], bias[ch] + offsets,
                _local_fit(offsets, convert_roc_to_volts(iq[ch])))[0]
        except Exception as exc:
            # Skipping one channel is fine -- it simply gets no
            # calibration -- but skipping all of them silently would
            # look identical to a board that has none, so say which.
            warnings.warn(f"df calibration failed on channel {ch}: "
                          f"{exc}", stacklevel=2)
            continue
        if cal is not None and np.isfinite(cal) and cal != 0:
            out[ch] = complex(cal)
    return out
