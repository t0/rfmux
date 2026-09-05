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
           "df_calibration_for_entry", "bias_frequency_from_fit",
           "ensure_fits", "fits_present", "fitted_linewidth",
           "step_slope_correction"]


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


def _skewed_model(freqs, iq_volts, fit_params):
    """The skewed-Lorentzian fit as an IQ trajectory: ``model(f)`` in
    the sweep's volts.  The fit is to |S21| alone, so it has the
    resonance's shape (fr, Qr, complex Qc) but not the trajectory's
    orientation or scale; one complex gain, the least-squares factor
    aligning the model to the sweep, supplies both."""
    fr, qr = float(fit_params["fr"]), float(fit_params["Qr"])
    qe = float(fit_params["Qcre"]) + 1j * float(fit_params["Qcim"])

    def shape(ff):
        return 1 - (qr / qe) / (1 + 2j * qr * (np.asarray(ff, dtype=np.float64) - fr) / fr)
    m = shape(np.asarray(freqs, dtype=np.float64))
    gain = np.vdot(m, np.asarray(iq_volts, dtype=complex)) / np.vdot(m, m)
    return lambda ff: gain * shape(ff)


def slope_from_skewed(freqs, iq_volts, f_bias, fit_params) -> complex:
    """d(I + jQ)/df at *f_bias* of the skewed-Lorentzian fit, in volts
    per hertz."""
    model = _skewed_model(freqs, iq_volts, fit_params)
    h = 1e-4 * float(fit_params["fr"]) / max(float(fit_params["Qr"]), 1.0)
    z_pm = model(np.array([f_bias - h, f_bias + h]))
    return complex((z_pm[1] - z_pm[0]) / (2 * h))


def _has_nonlinear_fit(entry) -> bool:
    nl = entry.get("nonlinear_fit_params")
    return bool(nl and entry.get("nonlinear_fit_success", True) and _finite(nl.get("fr")))


def _has_skewed_fit(entry) -> bool:
    sk = entry.get("fit_params")
    return bool(sk and all(_finite(sk.get(k)) for k in ("fr", "Qr", "Qcre", "Qcim")))


def _sorted_sweep(entry):
    f = np.asarray(entry["frequencies"], dtype=np.float64)
    order = np.argsort(f)
    return f[order], convert_roc_to_volts(np.asarray(entry["iq_complex"])[order])


def fit_for_calibration(freqs, iq_volts):
    """The nonlinear fit the calibration prefers, on a sweep in volts:
    ``(params, gain_complex)`` as fit_nonlinear_iq_multisweep stores
    them, or None when it does not converge.

    On the simulator the slope is within 3% and 2 degrees of truth
    from -60 to -55 dBm, at 500 Hz spacing and at the multisweep's
    2 kHz spacing alike; at -50 dBm, near bifurcation, within 20% and
    5 degrees."""
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


def bias_frequency_from_fit(entry, method="max-diq", fit="nonlinear"):
    """The bias frequency the multisweep's *method* picks, read off the
    resonance *fit* ("nonlinear" or "skewed") the entry carries.

    "max-diq" is where the IQ trajectory moves fastest with frequency,
    "min-s21" where |S21| is least.  On the raw sweep both are the
    largest of a noisy quantity on the sweep's own grid, so they carry
    its noise and its spacing (2 kHz at the multisweep's defaults, a
    third of a linewidth); on the model they are the extremum of a
    smooth curve, found on a fine grid.  Returns None when the entry
    has no such fit or the method is not one of those two.
    """
    if method not in ("max-diq", "min-s21"):
        return None
    if fit == "nonlinear" and _has_nonlinear_fit(entry):
        from .fitting_nonlinear import nonlinear_iq
        nl = entry["nonlinear_fit_params"]
        p = [nl[k] for k in ("fr", "Qr", "amp", "phi", "a", "i0", "q0")]
        model = lambda ff: nonlinear_iq(np.asarray(ff, dtype=np.float64), *p)
    elif fit == "skewed" and _has_skewed_fit(entry):
        model = _skewed_model(*_sorted_sweep(entry), entry["fit_params"])
    else:
        return None
    f = np.asarray(entry["frequencies"], dtype=np.float64)
    grid = np.linspace(f.min(), f.max(), 4001)
    z = model(grid)
    if method == "min-s21":
        return float(grid[np.argmin(np.abs(z))])
    return float(grid[np.argmax(np.abs(np.gradient(z, grid)))])


def fitted_linewidth(entry, prefer="nonlinear"):
    """fr / Qr from the fit the entry carries (the *prefer*red one
    first), in hertz, or None without a fit."""
    order = ("skewed", "nonlinear") if prefer == "skewed" else ("nonlinear", "skewed")
    for fit in order:
        if fit == "nonlinear" and _has_nonlinear_fit(entry):
            p = entry["nonlinear_fit_params"]
            return float(p["fr"]) / max(float(p["Qr"]), 1.0)
        if fit == "skewed" and _has_skewed_fit(entry):
            p = entry["fit_params"]
            return float(p["fr"]) / max(float(p["Qr"]), 1.0)
    return None


def step_slope_correction(entry, f_bias, step_hz, prefer="nonlinear"):
    """What a central difference over +-*step_hz* at *f_bias* reads,
    relative to the true slope there, on the fitted resonance: the
    complex ratio to divide a stepped measurement by.  The step is a
    fraction of the linewidth, so this is a second-order correction
    and the fit only has to have the curvature about right.  1 without
    a fit."""
    order = ("skewed", "nonlinear") if prefer == "skewed" else ("nonlinear", "skewed")
    for fit in order:
        if fit == "nonlinear" and _has_nonlinear_fit(entry):
            from .fitting_nonlinear import nonlinear_iq
            nl = entry["nonlinear_fit_params"]
            p = [nl[k] for k in ("fr", "Qr", "amp", "phi", "a", "i0", "q0")]
            model = lambda ff: nonlinear_iq(np.asarray(ff, dtype=np.float64), *p)
            break
        if fit == "skewed" and _has_skewed_fit(entry):
            model = _skewed_model(*_sorted_sweep(entry), entry["fit_params"])
            break
    else:
        return 1.0 + 0j
    lw = fitted_linewidth(entry, prefer)
    h = 1e-4 * lw
    z = model(np.array([f_bias - step_hz, f_bias + step_hz, f_bias - h, f_bias + h]))
    stepped = (z[1] - z[0]) / (2 * step_hz)
    true = (z[3] - z[2]) / (2 * h)
    if not (np.isfinite(stepped) and np.isfinite(true)) or true == 0:
        return 1.0 + 0j
    return complex(stepped / true)


def df_calibration_for_entry(entry, *, prefer="nonlinear"):
    """The calibration for one multisweep result at its bias frequency,
    from the fit it carries: the *prefer*red one ("nonlinear" or
    "skewed") if present, else the other.  The sweep's IQ is in counts,
    as multisweep stores it.  None when the entry carries no fit.
    """
    f_bias = float(entry.get("bias_frequency",
                             entry.get("original_center_frequency")))
    order = ("skewed", "nonlinear") if prefer == "skewed" else ("nonlinear", "skewed")
    for fit in order:
        if fit == "nonlinear" and _has_nonlinear_fit(entry):
            # The flow fits the sweep in counts, so the model's slope is
            # counts per hertz; the calibration is hertz per volt.
            slope = slope_from_nonlinear(f_bias, entry["nonlinear_fit_params"],
                                         entry.get("gain_complex", 1.0))
            return complex(1.0 / convert_roc_to_volts(slope))
        if fit == "skewed" and _has_skewed_fit(entry):
            f, iq = _sorted_sweep(entry)
            return complex(1.0 / slope_from_skewed(f, iq, f_bias, entry["fit_params"]))
    return None


def fits_present(entries) -> set:
    """Which resonance fits the *entries* carry: a subset of
    {"nonlinear", "skewed"}."""
    present = set()
    for entry in entries:
        if _has_nonlinear_fit(entry):
            present.add("nonlinear")
        if _has_skewed_fit(entry):
            present.add("skewed")
    return present


def ensure_fits(entries, fit="nonlinear") -> int:
    """Run the resonance *fit* ("nonlinear" or "skewed") on every entry
    in *entries* (an iterable of multisweep result dicts) that lacks
    it, with the flow's own batch fitter, storing what the flow's fit
    step would store.  Entries that already carry the fit are left
    alone.  Returns how many were fitted.

    Cost: 21 ms per sweep for the nonlinear fit, 3.5 ms for the skewed,
    on 101-point sweeps.  The batch fitter's thread pool makes the
    nonlinear fit slower, not faster (the work holds the GIL), so it is
    not used.
    """
    if fit == "nonlinear":
        from .fitting_nonlinear import fit_nonlinear_iq_multisweep
        has, keys = _has_nonlinear_fit, ("nonlinear_fit_params", "nonlinear_fit_errors",
                                         "nonlinear_fit_residual", "nonlinear_fit_success",
                                         "gain_complex", "iq_gain_corrected")
        run = lambda batch: fit_nonlinear_iq_multisweep(batch, parallel=False)
    elif fit == "skewed":
        from .fitting import fit_skewed_multisweep
        has, keys = _has_skewed_fit, ("fit_params", "iq_centered")
        run = fit_skewed_multisweep
    else:
        raise ValueError(f"unknown fit {fit!r}: expected 'nonlinear' or 'skewed'")
    todo = [e for e in entries if not has(e)]
    if not todo:
        return 0
    # The batch fitters key by integer index and hand back copies.
    fitted = run({i: dict(e) for i, e in enumerate(todo)})
    for i, e in enumerate(todo):
        for k in keys:
            if k in fitted.get(i, {}):
                e[k] = fitted[i][k]
    return len(todo)


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
