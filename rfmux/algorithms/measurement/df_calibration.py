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

__all__ = ["measure_df_calibrations", "df_calibration_from_sweep",
           "df_calibration_for_entry", "slope_from_nonlinear",
           "slope_from_skewed", "fit_for_calibration"]


def _finite(v) -> bool:
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def slope_from_nonlinear(f_bias, params, gain_complex=1.0) -> complex:
    """d(I + jQ)/df at *f_bias* of the nonlinear resonator model with
    *params* (fr, Qr, amp, phi, a, i0, q0, as fit_nonlinear_iq names
    them), scaled back by the gain its fit removed."""
    from .fitting_nonlinear import nonlinear_iq
    p = [params[k] for k in ("fr", "Qr", "amp", "phi", "a", "i0", "q0")]
    # The model is analytic: a step far inside any linewidth.
    h = 1e-4 * p[0] / max(p[1], 1.0)
    z_pm = nonlinear_iq(np.array([f_bias - h, f_bias + h]), *p)
    return complex(gain_complex * (z_pm[1] - z_pm[0]) / (2 * h))


def slope_from_skewed(freqs, iq_volts, f_bias, fit_params) -> complex:
    """d(I + jQ)/df at *f_bias* from a skewed-Lorentzian fit.

    That fit is to |S21| alone, so it has the resonance's shape (fr,
    Qr, complex Qc) but not the IQ trajectory's orientation or scale.
    One complex gain, the least-squares factor aligning the model's
    trajectory to the sweep's, supplies both; the model is then
    differentiated at the bias.
    """
    fr, qr = float(fit_params["fr"]), float(fit_params["Qr"])
    qe = float(fit_params["Qcre"]) + 1j * float(fit_params["Qcim"])

    def model(ff):
        return 1 - (qr / qe) / (1 + 2j * qr * (np.asarray(ff) - fr) / fr)
    f = np.asarray(freqs, dtype=np.float64)
    z = np.asarray(iq_volts, dtype=complex)
    m = model(f)
    gain = np.vdot(m, z) / np.vdot(m, m)
    h = 1e-4 * fr / max(qr, 1.0)
    z_pm = model(np.array([f_bias - h, f_bias + h]))
    return complex(gain * (z_pm[1] - z_pm[0]) / (2 * h))


def fit_for_calibration(freqs, iq_volts):
    """The nonlinear fit the calibration prefers, on a sweep in volts:
    ``(params, gain_complex)`` as fit_nonlinear_iq_multisweep stores
    them, or None when it does not converge.

    Sampling sets the accuracy, not the model: at 500 Hz spacing on a
    6 kHz linewidth the slope's phase is within 2 degrees of truth on
    the simulator, at the multisweep's 2 kHz spacing within 7, because
    the fitted resonance frequency is then uncertain by a few hundred
    hertz and the slope's phase turns quickly with distance from
    resonance."""
    from .fitting_nonlinear import estimate_and_remove_gain, fit_nonlinear_iq
    f = np.asarray(freqs, dtype=np.float64)
    z = np.asarray(iq_volts, dtype=complex)
    order = np.argsort(f)
    f, z = f[order], z[order]
    corrected, gain_mag, gain_phase = estimate_and_remove_gain(f, z)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, popt, _, residual = fit_nonlinear_iq(f, corrected)
    if not (np.isfinite(residual) and residual < 0.1):
        return None
    names = ("fr", "Qr", "amp", "phi", "a", "i0", "q0")
    return dict(zip(names, (float(v) for v in popt))), gain_mag * np.exp(1j * gain_phase)


def df_calibration_for_entry(entry, *, fit_if_missing=True, warn=True):
    """The calibration for one multisweep result, from the best fit it
    carries: the nonlinear IQ fit if it has one, else the skewed fit,
    else (with a warning, since a fit is being run for it) the nonlinear
    fit, which is then stored on the entry so it is not run again.
    The sweep's IQ is in counts, as multisweep stores it.  Returns None
    when nothing can be fitted.
    """
    f = np.asarray(entry["frequencies"], dtype=np.float64)
    order = np.argsort(f)
    f = f[order]
    iq = convert_roc_to_volts(np.asarray(entry["iq_complex"])[order])
    f_bias = float(entry.get("bias_frequency",
                             entry.get("original_center_frequency")))
    nl = entry.get("nonlinear_fit_params")
    if nl and entry.get("nonlinear_fit_success", True) and _finite(nl.get("fr")):
        slope = slope_from_nonlinear(f_bias, nl, entry.get("gain_complex", 1.0))
        return complex(1.0 / slope)
    sk = entry.get("fit_params")
    if sk and all(_finite(sk.get(k)) for k in ("fr", "Qr", "Qcre", "Qcim")):
        return complex(1.0 / slope_from_skewed(f, iq, f_bias, sk))
    if not fit_if_missing:
        return None
    if warn:
        warnings.warn(f"resonance at {f_bias / 1e6:.3f} MHz carries no fit; "
                      f"running the nonlinear fit for its df calibration",
                      stacklevel=2)
    fitted = fit_for_calibration(f, iq)
    if fitted is None:
        warnings.warn(f"resonance at {f_bias / 1e6:.3f} MHz: the fit did not "
                      f"converge; the df calibration is the spline derivative "
                      f"through the sweep points", stacklevel=2)
        return complex(convert_iq_to_df(np.array([1.0 + 0j]), f_bias, f, iq)[0])
    params, gain = fitted
    entry["nonlinear_fit_params"] = params
    entry["gain_complex"] = gain
    entry["nonlinear_fit_success"] = True
    return complex(1.0 / slope_from_nonlinear(f_bias, params, gain))


def df_calibration_from_sweep(freqs, iq_volts, f_bias) -> complex:
    """The calibration ``bias_kids`` reports, from a sweep in volts:
    multiply IQ in volts by it to get frequency shift + j dissipation.
    The inverse of the nonlinear resonator model's slope at *f_bias*,
    the model fitted to the sweep by the flow's own fitter;
    differentiating a spline through the points instead scattered by a
    factor of two on real sweeps.  Falls back to that spline, with a
    warning, when the fit does not converge.
    """
    fitted = fit_for_calibration(freqs, iq_volts)
    if fitted is None:
        warnings.warn("resonance fit did not converge; the df calibration "
                      "is the spline derivative through the sweep points",
                      stacklevel=2)
        f = np.asarray(freqs, dtype=np.float64)
        order = np.argsort(f)
        z = np.asarray(iq_volts, dtype=complex)[order]
        return complex(convert_iq_to_df(np.array([1.0 + 0j]), f_bias,
                                        f[order], z)[0])
    params, gain = fitted
    return complex(1.0 / slope_from_nonlinear(f_bias, params, gain))


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
