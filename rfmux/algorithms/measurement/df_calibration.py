"""
The df calibration and bias point of a resonance, from the fit its sweep
carries.

The calibration is the inverse of the fitted resonator model's slope at
the bias frequency: multiply IQ in volts by it to get frequency shift
+ j dissipation, in hertz.  ``bias_kids`` works from these helpers: it
gives every sweep the fit it lacks (``ensure_fits``), reads the bias
point off the fitted curve (``bias_frequency_from_fit``), corrects its
stepped-tone measurement for the curve's curvature
(``step_slope_correction``) and takes the fit's own calibration
(``df_calibration_for_entry``) as the fallback and the cross-check.

``measure_df_calibrations`` is the standalone measurement: it sweeps a
narrow span around where each channel is biased, fits the same model
and inverts its slope, falling back to a spline through the points when
no fit is usable.  It runs against a board or the simulator with the
same code; a simulated board has no calibration of its own to hand out.
Sweeping moves every channel's frequency and puts it back.
"""

import warnings
from typing import Dict, List, Optional

import numpy as np

from ...core.hardware_map import macro
from ...core.schema import CRS
from ...core.transferfunctions import convert_iq_to_df, convert_roc_to_volts
from .fitting import identify_bifurcation, s21_skewed
from .fitting_nonlinear import nonlinear_iq

__all__ = ["measure_df_calibrations", "df_calibration_from_sweep",
           "df_calibration_for_entry", "bias_frequency_from_fit",
           "ensure_fits", "fits_present", "fitted_linewidth",
           "step_slope_correction", "BIFURCATION_A", "NONLINEAR_PARAMS"]

# fit_nonlinear_iq's parameters, in the order nonlinear_iq takes them.
NONLINEAR_PARAMS = ("fr", "Qr", "amp", "phi", "a", "i0", "q0")

# The nonlinearity at which the resonator model becomes multivalued
# (Swenson et al. 2013): a fit at or past it describes no single curve.
BIFURCATION_A = 4 * np.sqrt(3) / 9


def _finite(v) -> bool:
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def _has_nonlinear_fit(entry) -> bool:
    """Whether the entry carries a nonlinear fit worth reading: one that
    converged, and whose nonlinearity is below bifurcation."""
    nl = entry.get("nonlinear_fit_params")
    return bool(nl and entry.get("nonlinear_fit_success", True)
                and _finite(nl.get("fr")) and _finite(nl.get("a"))
                and float(nl["a"]) < BIFURCATION_A)


def _has_skewed_fit(entry) -> bool:
    sk = entry.get("fit_params")
    return bool(sk and all(_finite(sk.get(k)) for k in ("fr", "Qr", "Qcre", "Qcim")))


def _sorted_sweep(entry):
    f = np.asarray(entry["frequencies"], dtype=np.float64)
    order = np.argsort(f)
    return f[order], np.asarray(entry["iq_complex"], dtype=complex)[order]


def _skewed_model(freqs, iq, fit_params):
    """The skewed-Lorentzian fit as an IQ trajectory ``model(f)`` in the
    sweep's units.  The fit is to |S21| alone, so it has the resonance's
    shape (fr, Qr, complex Qc) but not the trajectory's orientation or
    scale; one complex gain, the least-squares factor aligning the model
    to the sweep, supplies both."""
    fr, qr = float(fit_params["fr"]), float(fit_params["Qr"])
    qe = float(fit_params["Qcre"]) + 1j * float(fit_params["Qcim"])

    def shape(ff):
        return 1 - (qr / qe) / (1 + 2j * qr * (np.asarray(ff, dtype=np.float64) - fr) / fr)
    m = shape(np.asarray(freqs, dtype=np.float64))
    gain = np.vdot(m, np.asarray(iq, dtype=complex)) / np.vdot(m, m)
    return lambda ff: gain * shape(ff)


def _order(prefer):
    return ("skewed", "nonlinear") if prefer == "skewed" else ("nonlinear", "skewed")


def _fitted_model(entry, fits):
    """The first of *fits* ("nonlinear", "skewed") the entry carries, as
    ``(model, linewidth)``: ``model(f)`` is the fitted IQ trajectory in
    the sweep's counts, the linewidth fr / Qr in hertz.  None when it
    carries none of them."""
    for fit in fits:
        if fit == "nonlinear" and _has_nonlinear_fit(entry):
            p = entry["nonlinear_fit_params"]
            args = [float(p[k]) for k in NONLINEAR_PARAMS]
            gain = entry.get("gain_complex") or 1.0
            model = lambda ff: gain * nonlinear_iq(np.asarray(ff, dtype=np.float64), *args)
        elif fit == "skewed" and _has_skewed_fit(entry):
            p = entry["fit_params"]
            model = _skewed_model(*_sorted_sweep(entry), p)
        else:
            continue
        return model, float(p["fr"]) / max(float(p["Qr"]), 1.0)
    return None


def _slope(model, f, h) -> complex:
    """Central difference of *model* over +-*h* at *f*."""
    z = model(np.array([f - h, f + h]))
    return complex((z[1] - z[0]) / (2 * h))


def bias_frequency_from_fit(entry, method="max-diq", fit="nonlinear"):
    """The bias frequency the multisweep's *method* picks, read off the
    resonance *fit* ("nonlinear" or "skewed") the entry carries.

    "max-diq" is where the IQ trajectory moves fastest with frequency,
    "min-s21" where |S21| is least.  On the raw sweep both are the
    largest of a noisy quantity on the sweep's own grid, so they carry
    its noise and its spacing (2 kHz at Periscope's multisweep defaults
    of 200 kHz over 101 points, a third of a linewidth); on the model
    they are the extremum of a smooth curve, found on a fine grid.
    Returns None when the entry has no such fit or the method is not
    one of those two.
    """
    if method not in ("max-diq", "min-s21"):
        return None
    found = _fitted_model(entry, (fit,))
    if found is None:
        return None
    model, _ = found
    f = np.asarray(entry["frequencies"], dtype=np.float64)
    grid = np.linspace(f.min(), f.max(), 4001)
    z = model(grid)
    if method == "min-s21":
        return float(grid[np.argmin(np.abs(z))])
    return float(grid[np.argmax(np.abs(np.gradient(z, grid)))])


def fitted_linewidth(entry, prefer="nonlinear"):
    """fr / Qr from the fit the entry carries (the *prefer*red one
    first), in hertz, or None without a fit."""
    found = _fitted_model(entry, _order(prefer))
    return None if found is None else found[1]


def step_slope_correction(entry, f_bias, step_hz, prefer="nonlinear"):
    """What a central difference over +-*step_hz* at *f_bias* reads,
    relative to the true slope there, on the fitted resonance: the
    complex ratio to divide a stepped measurement by.  The step is a
    fraction of the linewidth, so this is a second-order correction
    and the fit only has to have the curvature about right.  1 without
    a fit."""
    found = _fitted_model(entry, _order(prefer))
    if found is None:
        return 1.0 + 0j
    model, lw = found
    stepped = _slope(model, f_bias, step_hz)
    # The model is analytic: a step far inside any linewidth.
    true = _slope(model, f_bias, 1e-4 * lw)
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
    found = _fitted_model(entry, _order(prefer))
    if found is None:
        return None
    model, lw = found
    # The model's slope is counts per hertz; the calibration is hertz
    # per volt.
    return complex(1.0 / convert_roc_to_volts(_slope(model, f_bias, 1e-4 * lw)))


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


def _record_fit(entry, fit) -> None:
    """The status and model-curve keys Periscope's fit step records
    beside the fitter's own, so the digest reads a fit run here the
    same way.  The nonlinear curve is gain-free, as the digest expects
    it."""
    if fit == "nonlinear":
        entry["nonlinear_fit_applied"] = True
        entry.setdefault("nonlinear_fit_success", False)
        p = entry.get("nonlinear_fit_params")
        if entry["nonlinear_fit_success"] and p:
            f = np.asarray(entry["frequencies"], dtype=np.float64)
            entry["nonlinear_model_iq"] = nonlinear_iq(f, *[float(p[k]) for k in NONLINEAR_PARAMS])
    else:
        entry["skewed_fit_applied"] = True
        entry["skewed_fit_success"] = _has_skewed_fit(entry)
        p = entry.get("fit_params")
        if entry["skewed_fit_success"] and _finite(p.get("A")):
            f = np.asarray(entry["frequencies"], dtype=np.float64)
            entry["skewed_model_mag"] = s21_skewed(f, p["fr"], p["Qr"], p["Qcre"], p["Qcim"], p["A"])


def ensure_fits(entries, fit="nonlinear") -> int:
    """Run the resonance *fit* ("nonlinear" or "skewed") on every entry
    in *entries* (an iterable of multisweep result dicts) the fitter
    has not been over, with the flow's own batch fitter, storing what
    the flow's fit step stores.  An entry that carries the fitter's
    result, converged or not, is left alone: the fitter is
    deterministic, so running it again changes nothing.  Returns how
    many were fitted.

    Cost: 21 ms per sweep for the nonlinear fit, 3.5 ms for the skewed,
    on 101-point sweeps.  The batch fitter's thread pool makes the
    nonlinear fit slower, not faster (the work holds the GIL), so it is
    not used.
    """
    if fit == "nonlinear":
        from .fitting_nonlinear import fit_nonlinear_iq_multisweep
        seen, keys = "nonlinear_fit_params", ("nonlinear_fit_params", "nonlinear_fit_errors",
                                              "nonlinear_fit_residual", "nonlinear_fit_success",
                                              "gain_complex", "iq_gain_corrected")
        run = lambda batch: fit_nonlinear_iq_multisweep(batch, parallel=False)
    elif fit == "skewed":
        from .fitting import fit_skewed_multisweep
        seen, keys = "fit_params", ("fit_params", "iq_centered")
        run = fit_skewed_multisweep
    else:
        raise ValueError(f"unknown fit {fit!r}: expected 'nonlinear' or 'skewed'")
    todo = [e for e in entries if seen not in e]
    if not todo:
        return 0
    # The batch fitters key by integer index; the skewed one writes in
    # place, so hand them copies.
    fitted = run({i: dict(e) for i, e in enumerate(todo)})
    for i, e in enumerate(todo):
        for k in keys:
            if k in fitted.get(i, {}):
                e[k] = fitted[i][k]
        _record_fit(e, fit)
    return len(todo)


def df_calibration_from_sweep(freqs, iq_counts, f_bias) -> complex:
    """The calibration from one sweep in counts: the inverse of the
    nonlinear resonator model's slope at *f_bias*, the model fitted by
    the flow's own fitter.  Differentiating a spline through the points
    instead scatters by a factor of two on real sweeps, so that is the
    fallback, with a warning, when no fit is usable.
    """
    entry = {"frequencies": np.asarray(freqs, dtype=np.float64),
             "iq_complex": np.asarray(iq_counts, dtype=complex),
             "original_center_frequency": f_bias, "bias_frequency": f_bias}
    ensure_fits([entry])
    cal = df_calibration_for_entry(entry)
    if cal is None:
        warnings.warn("no usable resonance fit; the df calibration is the "
                      "spline derivative through the sweep points", stacklevel=2)
        f, z = _sorted_sweep(entry)
        cal = complex(convert_iq_to_df(np.array([1.0 + 0j]), f_bias, f,
                                       convert_roc_to_volts(z))[0])
    return cal


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
        reports as biased (get_biased_channels).  A channel with no tone
        is skipped, with a warning.
    module : int
        Module index (1-based).
    span_hz : float
        Full width of the sweep, centred on the bias point.  A
        resonance is fitted to it, so a few linewidths is enough; the
        fit's own guidance is six linewidths, and three measures the
        same slope in half the time.
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
        multiply IQ in volts by it to get frequency shift + j
        dissipation in hertz.  Its magnitude is hertz per volt; its
        phase is minus the angle of the frequency direction in the
        (I, Q) plane, so the product turns that direction onto the real
        axis.  Channels whose sweep gives no usable derivative are left
        out rather than guessed at.
    """
    if channels is None:
        from .channel_selection import get_biased_channels
        channels = await get_biased_channels(crs, module)
    channels = [int(c) for c in channels]
    if not channels:
        return {}
    nco = await crs.get_nco_frequency(module=module)
    bias: Dict[int, float] = {}
    for channel in channels:
        rel = await crs.get_frequency(channel=channel, module=module)
        if rel is None:
            warnings.warn(f"channel {channel} has no tone on module {module}; "
                          f"skipped", stacklevel=2)
            continue
        bias[channel] = nco + rel
    channels = list(bias)
    if not channels:
        return {}

    half = 0.5 * span_hz
    n_points = max(5, int(round(span_hz / resolution_hz)) + 1)
    offsets = np.linspace(-half, half, n_points)
    iq = {ch: np.full(n_points, np.nan, dtype=complex) for ch in channels}
    try:
        # Every channel steps together: one batched frequency write and
        # one read of the module per point, not a round trip per channel
        # per point.
        for k, off in enumerate(offsets):
            if progress is not None:
                progress(k, n_points)
            async with crs.tuber_context() as ctx:
                for ch in channels:
                    ctx.set_frequency(bias[ch] + off - nco, channel=ch,
                                      module=module)
                await ctx()
            s = await crs.get_samples(n_samples, module=module)
            # Samples index channels from zero.
            for ch in channels:
                iq[ch][k] = (np.mean(np.asarray(s.i[ch - 1]))
                             + 1j * np.mean(np.asarray(s.q[ch - 1])))
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
            cal = df_calibration_from_sweep(bias[ch] + offsets, iq[ch], bias[ch])
        except Exception as exc:
            # Skipping one channel is fine -- it simply gets no
            # calibration -- but skipping all of them silently would
            # look identical to a board that has none, so say which.
            warnings.warn(f"df calibration failed on channel {ch}: "
                          f"{exc}", stacklevel=2)
            continue
        if np.isfinite(cal) and cal != 0:
            out[ch] = complex(cal)
    return out
