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
from .fitting import identify_bifurcation

__all__ = ["measure_df_calibrations", "df_calibration_from_sweep"]


def df_calibration_from_sweep(freqs, iq_volts, f_bias) -> complex:
    """The calibration ``bias_kids`` reports, from a sweep: multiply IQ
    in volts by it to get frequency shift + j dissipation.

    It is the inverse of d(I + jQ)/df at *f_bias*, taken from the
    nonlinear resonator model the rest of the flow fits
    (:func:`~.fitting_nonlinear.fit_nonlinear_iq`, after its gain
    removal) and differentiated there.  Differentiating a spline
    through the sweep points instead handed the slope to the two
    nearest, and on a real sweep, whose points carry drift as well as
    noise, repeated measurements disagreed by a factor of two; against
    a noise-free truth on the simulator the model's slope is within a
    few percent.  If the fit does not converge the spline is what
    remains, with a warning.
    """
    from .fitting_nonlinear import (
        estimate_and_remove_gain, fit_nonlinear_iq, nonlinear_iq)
    f = np.asarray(freqs, dtype=np.float64)
    z = np.asarray(iq_volts, dtype=complex)
    order = np.argsort(f)
    f, z = f[order], z[order]
    corrected, gain_mag, gain_phase = estimate_and_remove_gain(f, z)
    gain = gain_mag * np.exp(1j * gain_phase)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, popt, _, residual = fit_nonlinear_iq(f, corrected)
    if np.isfinite(residual) and residual < 0.1:
        # The model is analytic: a step far inside any linewidth.
        h = 1e-4 * popt[0] / max(popt[1], 1.0)
        z_pm = nonlinear_iq(np.array([f_bias - h, f_bias + h]), *popt)
        slope = gain * (z_pm[1] - z_pm[0]) / (2 * h)
    else:
        warnings.warn("resonance fit did not converge; the df calibration "
                      "is the spline derivative through the sweep points",
                      stacklevel=2)
        slope = 1.0 / convert_iq_to_df(np.array([1.0 + 0j]), f_bias, f, z)[0]
    return complex(1.0 / slope)


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
        Full width of the sweep, centred on the bias point.  A
        resonance is fitted to it, so a few linewidths is enough; the
        fit's own guidance is six linewidths, and three measures the
        same slope in a third of the time.
    resolution_hz : float
        Spacing between points, well inside the linewidth so the fit
        has a dozen or more points across the resonance.
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
        if identify_bifurcation(iq[ch]):
            # Too much bias power: the resonance is bifurcated and the
            # sweep steps across the jump, so no estimate of the slope
            # there means much.  The number is still reported, and the
            # channel still biased, with the caveat said out loud.
            warnings.warn(f"channel {ch}: the sweep jumps, so the "
                          f"resonance is bifurcated at this bias power and "
                          f"its df calibration is unreliable", stacklevel=2)
        try:
            cal = df_calibration_from_sweep(
                bias[ch] + offsets, convert_roc_to_volts(iq[ch]), bias[ch])
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
