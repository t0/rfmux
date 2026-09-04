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
from ...core.transferfunctions import convert_roc_to_volts

__all__ = ["measure_df_calibrations", "df_calibration_from_sweep",
           "fit_resonance_slope", "sweep_jumps"]


def fit_resonance_slope(freqs, iq_volts, f_bias):
    """d(I + jQ)/df at *f_bias*, from a resonance fitted to the sweep.

    The model is a Lorentzian on a linear baseline,
    ``c0 + c1 (f - f_bias) + d / (1 + 2jQ (f - fr) / fr)`` with complex
    c0, c1, d and real Q, fr, differentiated analytically at the bias
    frequency.  Differentiating a spline through the points instead
    hands the slope to the nearest neighbours, and on a real sweep,
    whose points carry drift as well as noise, repeated measurements
    disagreed by a factor of two; a polynomial through the points has
    no such variance but reads a Lorentzian's central slope low by the
    curvature it cannot follow (25% over three quarters of a
    linewidth).  A resonance fitted to its own shape is unbiased and
    uses every point.
    """
    from scipy.optimize import least_squares
    f = np.asarray(freqs, dtype=np.float64)
    z = np.asarray(iq_volts, dtype=complex)
    x = f - f_bias
    n_edge = max(2, len(f) // 10)
    c0 = 0.5 * (z[:n_edge].mean() + z[-n_edge:].mean())
    k_dip = int(np.argmax(np.abs(z - c0)))
    d = z[k_dip] - c0
    # Linewidth from where the deviation from the baseline halves.
    dev = np.abs(z - c0)
    half = np.flatnonzero(dev >= 0.5 * dev[k_dip])
    width = max(f[half[-1]] - f[half[0]], f[1] - f[0])
    fr0 = f[k_dip]
    q0 = fr0 / width
    scale = max(abs(d), 1e-30)
    # The slope is wanted at the bias, and a real resonance is only a
    # Lorentzian near its centre (drive skews it), so points are
    # weighted down with distance from the bias on the scale of the
    # linewidth: the fit describes the resonance where it is read.
    weight = np.exp(-0.5 * (x / width) ** 2)

    def model(p):
        c0r, c0i, c1r, c1i, dr, di, q, fr = p
        return ((c0r + 1j * c0i) + (c1r + 1j * c1i) * x
                + (dr + 1j * di) / (1 + 2j * q * (f - fr) / fr))

    def resid(p):
        r = weight * (model(p) - z) / scale
        return np.concatenate([r.real, r.imag])

    p0 = [c0.real, c0.imag, 0.0, 0.0, d.real, d.imag, q0, fr0]
    fit = least_squares(resid, p0, x_scale="jac")
    c0r, c0i, c1r, c1i, dr, di, q, fr = fit.x
    den = 1 + 2j * q * (f_bias - fr) / fr
    return (c1r + 1j * c1i) + (dr + 1j * di) * (-2j * q / fr) / den ** 2


def sweep_jumps(iq, threshold: float = 0.5) -> bool:
    """True when neighbouring sweep points are further apart than half
    the sweep's whole extent in the IQ plane: a bifurcated resonance's
    jump.  A smooth resonance sampled a dozen times across its
    linewidth moves a tenth of its extent per point; on the simulator
    the largest step is 0.07 at -55 dBm, 0.29 at -50 where the shape
    is skewed but continuous, and 0.76 once it bifurcates at -48.
    """
    iq = np.asarray(iq, dtype=complex)
    if len(iq) < 3:
        return False
    extent = 2 * np.max(np.abs(iq - iq.mean()))
    if not np.isfinite(extent) or extent == 0:
        return False
    return bool(np.max(np.abs(np.diff(iq))) > threshold * extent)


def df_calibration_from_sweep(freqs, iq_volts, f_bias) -> complex:
    """The calibration ``bias_kids`` reports: multiply IQ in volts by it
    to get frequency shift + j dissipation.  The inverse of the fitted
    slope, so the same number convert_iq_to_df gives for unit IQ."""
    return complex(1.0 / fit_resonance_slope(freqs, iq_volts, f_bias))


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
        if sweep_jumps(iq[ch]):
            # Too much bias power: the resonance is bifurcated and the
            # sweep steps across the jump.  There is no slope there to
            # calibrate with, whatever estimates it.
            warnings.warn(f"df calibration skipped on channel {ch}: the "
                          f"sweep jumps, so the resonance is bifurcated at "
                          f"this bias power", stacklevel=2)
            continue
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
