"""Fit skewed, nonlinear, and circle models to measured resonator sweeps.

:func:`fit_sweeps` takes one module's multisweep result, writes results under
each entry's ``fits[model]``, and returns a report dictionary. Use the model
readers below to reconstruct curves from the stored parameters.

The nonlinear model and fitter are adapted from citkid
(https://github.com/loganfoote/citkid), Apache License 2.0. This version omits
Numba and cable delay, and estimates gain from the sweep's frequency extrema.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from . import store
from .store import plain
from .sweep_results import (
    _iteration_matching_amplitude,
    _iterations,
)

__all__ = [
    "MODELS",
    "BIFURCATION_A",
    "FitFailed",
    "fit_sweeps",
    "fit_sweeps_at_bias_amplitude",
    "fit_section",
    "FIT_PARAMS",
    "collect_fit_params",
    "skewed_model_magnitude",
    "nonlinear_model_iq",
    "gain_corrected_iq",
    "centered_iq",
    "s21_skewed",
    "fit_skewed",
    "nonlinear_iq",
    "get_y_nonlinear",
    "guess_p0_nonlinear",
    "fit_nonlinear_iq",
    "remove_gain",
    "calculate_residuals",
    "circle_fit_pratt",
    "center_resonance_iq_circle",
]

#: The models :func:`fit_sweeps` knows how to run, and the default set.
MODELS = ("skewed", "nonlinear", "circle")

#: Parameters of the nonlinear model, in the order its fitter works in.
NONLINEAR_PARAMS = ("fr", "Qr", "amp", "phi", "a", "i0", "q0")

#: Bifurcation threshold (Swenson et al. 2013). Above this value, the model
#: selects a stable branch using the acquisition direction.
BIFURCATION_A = 4 * np.sqrt(3) / 9

#: Parameters of the skewed Lorentzian. ``Qc`` and ``Qi`` are derived from the
#: other three Qs; their errors propagate the fitted Qs' full covariance.
SKEWED_PARAMS = ("fr", "Qr", "Qc", "Qi", "Qcre", "Qcim", "A")
SKEWED_FITTED_PARAMS = ("fr", "Qr", "Qcre", "Qcim", "A")

#: Parameters exposed in tables and displays. Circle fits store centre and
#: radius separately; derived nonlinear Qc and Qi are not listed here.
FIT_PARAMS = {"skewed": SKEWED_PARAMS, "nonlinear": NONLINEAR_PARAMS}


class FitFailed(Exception):
    """A single-trace fit failed; batch adapters record the reason."""


@contextmanager
def _quiet_optimizer():
    """Suppress covariance warnings while a batch runs.

    Set the process-wide filter before workers start and restore it after
    they join; per-thread contexts would race when restoring the filters.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        yield


# ─── The entry points ─────────────────────────────────────────────────────────


def fit_sweeps(
    ms_module_output,
    *,
    models: Sequence[str] = MODELS,
    names=None,
    iterations=None,
    directions=None,
    approx_Qr: float = 1e4,
    normalize: bool = True,
    fr_limit_hz: float | None = None,
    fit_nonlinearity: bool = True,
    n_extrema_points: int = 5,
    max_residual: float = 0.1,
    max_workers: int | None = None,
    progress_callback=None,
    save=None,
    label=None,
) -> dict:
    """Fit selected sweeps in one module and write results into each entry.

    Results go under ``entry["fits"][model]`` with ``failed_because`` (None
    on success). Skewed and nonlinear fits store parameters and errors; circle
    fits store centre and radius. Nonlinear fits also store gain and residual.
    A converged fit with excessive residual keeps its parameters
    and records the failure reason. Rerunning a model replaces only its fits.

    Args:
        ms_module_output: one module's multisweep output,
            ``multisweep_output[crs.module[m].index()]``.
        models: model names from :data:`MODELS`; all three by default.
        names: one resonator name, a sequence or set, or None for all.
        iterations: one amplitude-step index, a sequence or set, or None for all.
            Use :func:`fit_sweeps_at_bias_amplitude` when each resonator needs
            a different step.
        directions: one direction, a sequence or set, or None for all.
        approx_Qr: initial Qr estimate for the skewed fit.
        normalize: divide by the last trace point before the skewed fit.
            :func:`skewed_model_magnitude` returns this normalized scale.
        fr_limit_hz: skewed-fit frequency bound around the sweep centre.
            None uses 37.5% of the sweep span.
        fit_nonlinearity: fit ``a``; False constrains it near zero.
        n_extrema_points: points at each frequency end used to estimate gain.
        max_residual: maximum acceptable nonlinear-fit residual.
        max_workers: worker threads; None uses ``min(4, cpu_count)``.
        progress_callback: called as ``(completed, total)`` as sweep results
            are collected in input order.
        save: save the fitted sweeps to their existing file, or create one.
            None uses ``store.autosave_enabled()``.
        label: filename label for a first save; existing filenames are kept.

    Returns:
        dict: ``schema_version`` (1), ``fits`` and ``settings``. Each fit row
        contains ``name``, ``model``, ``iteration``, ``direction`` and
        ``failed_because`` (None on success). Parameters stay in the sweeps.
        The report uses builtin values and can be saved directly.

    Raises:
        TypeError: input is not a supported module result.
        ValueError: unknown model or a filter that selects no sweeps.
    """
    sections = _select(
        ms_module_output,
        names=names,
        iterations=iterations,
        directions=directions,
    )
    report = _fit(
        sections,
        module=ms_module_output.get("module"),
        models=models,
        approx_Qr=approx_Qr,
        normalize=normalize,
        fr_limit_hz=fr_limit_hz,
        fit_nonlinearity=fit_nonlinearity,
        n_extrema_points=n_extrema_points,
        max_residual=max_residual,
        max_workers=max_workers,
        progress_callback=progress_callback,
    )
    store.maybe_save(ms_module_output, "multisweep", save=save, label=label)
    return report


def fit_sweeps_at_bias_amplitude(
    ms_module_output,
    *,
    amplitude: float | None = None,
    names=None,
    directions=None,
    save=None,
    label=None,
    **settings,
) -> dict:
    """Fit each resonator at the measured amplitude nearest its bias amplitude.

    Matching is done per resonator using
    :func:`~rfmux.tuning.sweep_results.find_iteration_matching_amplitude`.
    There is no maximum matching distance; inspect the chosen amplitudes if
    closeness matters.

    Args:
        ms_module_output: one module's multisweep output.
        amplitude: target amplitude in DAC units, or None for each member's
            bias amplitude in the recorded catalog.
        names: resonator names to fit, or None for all.
        directions: sweep directions to fit, or None for all.
        save: save the fitted sweeps, as in :func:`fit_sweeps`.
        label: filename label for a first save.
        **settings: model and fitter settings, as in :func:`fit_sweeps`.

    Returns:
        dict: outcomes and settings, as returned by :func:`fit_sweeps`.
    """
    sections = _select(ms_module_output, names=names, directions=directions)
    wanted = {s["name"] for s in sections}
    at_bias = {
        name: _iteration_matching_amplitude(ms_module_output, name, amplitude)
        for name in wanted
    }
    sections = [
        s for s in sections if s["iteration"] == at_bias[s["name"]]
    ]
    if not sections:
        raise ValueError(
            f"Nothing to fit: no sweep of {sorted(wanted)[:4]} at its bias "
            f"amplitude survived directions={directions!r}."
        )
    report = _fit(sections, module=ms_module_output.get("module"), **settings)
    store.maybe_save(ms_module_output, "multisweep", save=save, label=label)
    return report


def fit_section(
    entry: dict,
    *,
    models: Sequence[str] = MODELS,
    approx_Qr: float = 1e4,
    normalize: bool = True,
    fr_limit_hz: float | None = None,
    fit_nonlinearity: bool = True,
    n_extrema_points: int = 5,
    max_residual: float = 0.1,
) -> dict:
    """Fit one sweep entry, in place, and return its ``fits`` subdict.

    Args:
        entry: one sweep, as ``multisweep`` returns it: ``frequencies`` and
            ``iq_counts`` are what get fitted.
        models: which models to run, from :data:`MODELS`.
        approx_Qr, normalize, fr_limit_hz, fit_nonlinearity, n_extrema_points,
        max_residual: as :func:`fit_sweeps`.

    Returns:
        dict: the entry's ``fits`` subdict, not a copy.

    Raises:
        ValueError: for an unknown model name, or an entry with no
            ``frequencies`` / ``iq_counts``.
    """
    models = _resolve_models(models)

    frequencies = entry.get("frequencies")
    iq_counts = entry.get("iq_counts")
    if frequencies is None or iq_counts is None:
        missing = [
            k for k in ("frequencies", "iq_counts") if entry.get(k) is None
        ]
        raise ValueError(
            f"This sweep entry has no {' or '.join(missing)}, so there is "
            f"nothing to fit. Its keys are {sorted(entry)}."
        )

    fits = entry.setdefault("fits", {})
    for model in models:
        fits[model] = _FITTERS[model](
            np.asarray(frequencies, dtype=float),
            np.asarray(iq_counts),
            sweep_direction=entry.get("sweep_direction"),
            approx_Qr=approx_Qr,
            normalize=normalize,
            fr_limit_hz=fr_limit_hz,
            fit_nonlinearity=fit_nonlinearity,
            n_extrema_points=n_extrema_points,
            max_residual=max_residual,
        )
    return fits


# ─── Reading the fits back ────────────────────────────────────────────────────


def skewed_model_magnitude(entry: Mapping) -> np.ndarray:
    """The ``|S21|`` the skewed fit predicts, on the entry's own frequencies.

    With ``normalize=True``, compare against
    ``np.abs(entry["iq_counts"] / entry["iq_counts"][-1])``.

    Raises:
        ValueError: if this entry has no converged skewed fit.
    """
    params = _params_of(entry, "skewed")
    return s21_skewed(
        np.asarray(entry["frequencies"], dtype=float),
        params["fr"],
        params["Qr"],
        params["Qcre"],
        params["Qcim"],
        params["A"],
    )


def nonlinear_model_iq(entry: Mapping) -> np.ndarray:
    """The complex trace the nonlinear fit predicts, in readout counts.

    Multiply the gain-corrected model by the stored gain, if present.

    Raises:
        ValueError: if this entry has no converged nonlinear fit.
    """
    params = _params_of(entry, "nonlinear")
    gain = entry["fits"]["nonlinear"].get("gain")
    model = nonlinear_iq(
        np.asarray(entry["frequencies"], dtype=float),
        *(params[p] for p in NONLINEAR_PARAMS),
        sweep_direction=entry.get("sweep_direction"),
    )
    return model if gain is None else model * gain


def gain_corrected_iq(entry: Mapping) -> np.ndarray:
    """``iq_counts`` with the readout gain the nonlinear fit estimated divided out.

    Raises:
        ValueError: if this entry has no nonlinear fit with a gain estimate.
    """
    fit = (entry.get("fits") or {}).get("nonlinear")
    gain = (fit or {}).get("gain")
    if gain is None:
        raise ValueError(
            "This sweep has no nonlinear-fit gain estimate. Run "
            "fit_sweeps(..., models=('nonlinear',)) on it first."
        )
    return np.asarray(entry["iq_counts"]) / gain


def centered_iq(entry: Mapping) -> np.ndarray:
    """``iq_counts`` with the fitted circle centre subtracted.

    Raises:
        ValueError: if this entry has no converged circle fit.
    """
    fit = (entry.get("fits") or {}).get("circle")
    if not fit or fit.get("center") is None:
        raise ValueError(
            "This sweep has no circle fit. Run "
            "fit_sweeps(..., models=('circle',)) on it first."
        )
    return np.asarray(entry["iq_counts"]) - fit["center"]


def collect_fit_params(
    ms_module_output,
    model: str,
    *,
    names=None,
    iterations=None,
    directions=None,
) -> list[dict]:
    """Collect one model's parameters, one row per sweep.

    For example::

        rows = collect_fit_params(ms_module_output, "skewed")
        plt.hist([r["params"]["Qi"] for r in rows], bins=40)

    Omit fits without parameters. Include fits rejected by ``max_residual``;
    filter on ``failed_because`` to exclude them.

    Args:
        ms_module_output: one module's multisweep output, as
            :func:`fit_sweeps` takes it.
        model: which model's parameters, from :data:`FIT_PARAMS`.
        names: resonators to include; every one of them by default.
        iterations: amplitude steps to include; every one by default.
        directions: sweep directions to include; both by default.

    Returns:
        list[dict]: ``name``, ``iteration``, ``direction``, ``amplitude``,
        ``params``, ``errors`` and ``failed_because``, in the order the sweeps
        were measured. ``params`` and ``errors`` are the entry's own dicts, not
        copies.
    """
    if model not in FIT_PARAMS:
        raise ValueError(
            f"Unknown model {model!r}: named parameters come from "
            f"{tuple(FIT_PARAMS)}."
            + (" The circle fit records a centre and a radius; read those off "
               "the entry, or use centered_iq()." if model == "circle" else "")
        )

    sections = _select(
        ms_module_output, names=names, iterations=iterations,
        directions=directions, allow_empty=True,
    )
    rows = []
    for section in sections:
        fit = (section["entry"].get("fits") or {}).get(model)
        if fit is None or fit.get("params") is None:
            continue
        rows.append(
            {
                "name": section["name"],
                "iteration": section["iteration"],
                "direction": section["direction"],
                "amplitude": section["entry"].get("sweep_amplitude"),
                "params": fit["params"],
                "errors": fit.get("errors") or {},
                "failed_because": fit.get("failed_because"),
            }
        )
    return rows


def _params_of(entry: Mapping, model: str) -> dict:
    """Return stored parameters, raising ValueError if absent."""
    fit = (entry.get("fits") or {}).get(model)
    if fit is None:
        raise ValueError(
            f"This sweep has no {model} fit. Run "
            f"fit_sweeps(..., models=({model!r},)) on it first."
        )
    if fit.get("params") is None:
        raise ValueError(
            f"The {model} fit on this sweep did not converge: "
            f"{fit.get('failed_because')}"
        )
    return fit["params"]


# ─── Selecting what to fit ────────────────────────────────────────────────────


def _walk(ms_module_output):
    """Yield sweep dictionaries with their coordinates and original entry."""
    for iteration, by_direction in _iterations(ms_module_output).items():
        for direction, sections in by_direction.items():
            for name, entry in sections.items():
                yield {
                    "name": name,
                    "iteration": int(iteration),
                    "direction": direction,
                    "entry": entry,
                }


def _as_filter(wanted, what: str) -> set | None:
    """Return a set, treating a scalar as one item and None as all items."""
    if wanted is None:
        return None
    if isinstance(wanted, (str, int, np.integer)):
        return {wanted}
    if not isinstance(wanted, (Sequence, set, frozenset)):
        raise TypeError(
            f"{what} must be a single value or an iterable of them, got "
            f"{type(wanted).__name__}."
        )
    return set(wanted)


def _filter_names(names, available: set[str]) -> set[str]:
    """Select names, rejecting any absent from the sweeps."""
    wanted = _as_filter(names, "names")
    if wanted is None:
        return set(available)
    unknown = sorted(wanted - available)
    if unknown:
        listed = sorted(available)[:4]
        more = " …" if len(available) > 4 else ""
        raise ValueError(
            f"names {unknown} were not swept. The names in play are "
            f"{listed}{more}."
        )
    return wanted


def _select(
    ms_module_output, *, names=None, iterations=None, directions=None,
    allow_empty: bool = False,
) -> list[dict]:
    """Select sweeps in stored order, optionally allowing no matches."""
    sections = list(_walk(ms_module_output))
    if not sections and not allow_empty:
        raise ValueError("There are no sweeps in this result to fit.")

    keep_names = _filter_names(names, {s["name"] for s in sections})
    keep_iterations = _as_filter(iterations, "iterations")
    keep_directions = _as_filter(directions, "directions")

    selected = [
        s
        for s in sections
        if s["name"] in keep_names
        and (keep_iterations is None or s["iteration"] in keep_iterations)
        and (keep_directions is None or s["direction"] in keep_directions)
    ]
    if not selected and not allow_empty:
        raise ValueError(
            f"Nothing to fit: names={names!r}, iterations={iterations!r}, "
            f"directions={directions!r} selected none of the "
            f"{len(sections)} sweeps in this result. It has iterations "
            f"{sorted({s['iteration'] for s in sections})} and directions "
            f"{sorted({s['direction'] for s in sections})}."
        )
    return selected


def _resolve_models(models) -> tuple[str, ...]:
    """Validate model names and return a tuple."""
    if isinstance(models, str):
        raise TypeError(
            f"models={models!r}: pass a sequence, not a single string — "
            f"({models!r},) for one model, {MODELS} for all of them. (A bare "
            f"string would read as a sequence of characters.)"
        )
    resolved = tuple(models)
    if not resolved:
        raise ValueError(
            f"models is empty: nothing would be fitted. Pass at least one of "
            f"{MODELS}."
        )
    unknown = [m for m in resolved if m not in MODELS]
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}. Must be from {MODELS}.")
    return resolved


# ─── Running the fits ─────────────────────────────────────────────────────────


def _fit(
    sections: list[dict],
    *,
    module: int | None = None,
    models: Sequence[str] = MODELS,
    approx_Qr: float = 1e4,
    normalize: bool = True,
    fr_limit_hz: float | None = None,
    fit_nonlinearity: bool = True,
    n_extrema_points: int = 5,
    max_residual: float = 0.1,
    max_workers: int | None = None,
    progress_callback=None,
) -> dict:
    """Fit selected sweeps and collect outcomes in stored order.

    Each worker runs all models for one sweep. Record the module once in the
    report settings.
    """
    models = _resolve_models(models)
    settings = {
        "models": models,
        "approx_Qr": approx_Qr,
        "normalize": normalize,
        "fr_limit_hz": fr_limit_hz,
        "fit_nonlinearity": fit_nonlinearity,
        "n_extrema_points": n_extrema_points,
        "max_residual": max_residual,
    }
    per_section = dict(settings)
    del per_section["models"]

    # Record the module without passing it to the single-sweep fitter.
    settings["module"] = module

    def fit_one(section: dict) -> dict:
        return fit_section(section["entry"], models=models, **per_section)

    if max_workers is None:
        max_workers = min(4, os.cpu_count() or 1)

    fits: list[dict] = []
    total = len(sections)
    with _quiet_optimizer(), ThreadPoolExecutor(
        max_workers=max(1, max_workers)
    ) as executor:
        # Preserve sweep order in the report and progress callbacks.
        submitted = [executor.submit(fit_one, s) for s in sections]

        for completed, (section, future) in enumerate(
            zip(sections, submitted), start=1
        ):
            try:
                result = future.result()
                reasons = {m: result[m]["failed_because"] for m in models}
            except Exception as exc:
                # Record malformed sweeps in the report and continue.
                reasons = {m: f"{type(exc).__name__}: {exc}" for m in models}

            fits.extend(
                dict(
                    name=section["name"],
                    model=model,
                    iteration=section["iteration"],
                    direction=section["direction"],
                    failed_because=reasons[model],
                )
                for model in models
            )
            if progress_callback is not None:
                progress_callback(completed, total)

    return {"schema_version": 1, "fits": fits, "settings": plain(settings)}


def _skewed_fit(frequencies, iq_counts, *, approx_Qr, normalize, fr_limit_hz, **_):
    """Run the skewed Lorentzian and shape the result for storage."""
    try:
        params, errors = fit_skewed(
            frequencies,
            iq_counts,
            approx_Qr=approx_Qr,
            normalize=normalize,
            fr_limit_hz=fr_limit_hz,
        )
    except FitFailed as exc:
        return {"params": None, "errors": None, "failed_because": str(exc)}
    return {"params": params, "errors": errors, "failed_because": None}


def _nonlinear_fit(
    frequencies, iq_counts, *, fit_nonlinearity, n_extrema_points, max_residual,
    sweep_direction=None, **_
):
    """Remove the gain, run the nonlinear model, and shape the result."""
    result = {
        "params": None,
        "errors": None,
        "residual": float("inf"),
        "gain": None,
        "failed_because": None,
    }
    try:
        corrected, gain = remove_gain(
            frequencies, iq_counts, n_extrema_points=n_extrema_points
        )
    except FitFailed as exc:
        result["failed_because"] = str(exc)
        return result

    result["gain"] = gain
    try:
        params, errors, residual = fit_nonlinear_iq(
            frequencies, corrected, fit_nonlinearity=fit_nonlinearity,
            sweep_direction=sweep_direction
        )
    except FitFailed as exc:
        result["failed_because"] = str(exc)
        return result

    result["params"] = params
    result["errors"] = errors
    result["residual"] = residual
    if residual > max_residual:
        # Keep rejected parameters for inspection.
        result["failed_because"] = (
            f"residual {residual:.3g} is above max_residual={max_residual:g}"
        )
    return result


def _circle_fit(frequencies, iq_counts, **_):
    """Fit an algebraic circle and format the stored result."""
    xc, yc, radius = circle_fit_pratt(iq_counts.real, iq_counts.imag)
    if xc is None:
        return {
            "center": None,
            "radius": None,
            "failed_because": "the circle fit did not solve; see circle_fit_pratt",
        }
    return {
        "center": complex(xc, yc),
        "radius": float(radius),
        "failed_because": None,
    }


_FITTERS = {
    "skewed": _skewed_fit,
    "nonlinear": _nonlinear_fit,
    "circle": _circle_fit,
}


# ─── The skewed Lorentzian ────────────────────────────────────────────────────


def s21_skewed(f, fr, Qr, Qcre, Qcim, A):
    """Skewed Lorentzian model for ``|S21|``, after the hidfmux implementation.

    Complex coupling ``Qe = Qcre + 1j*Qcim`` produces the asymmetry.
    A multiplicative penalty discourages effective ``Qc`` near or below ``Qr``;
    invalid parameters return infinity.

    Args:
        f (np.ndarray): Frequencies (Hz).
        fr (float): Resonance frequency (Hz).
        Qr (float): Total (loaded) quality factor.
        Qcre (float): Real part of the complex coupling Q.
        Qcim (float): Imaginary part of the complex coupling Q.
        A (float): Overall scale. Near 1 when the trace was normalized.

    Returns:
        np.ndarray: modelled ``|S21|``, ``inf`` where the parameters are
        outside the model's accepted domain.
    """
    if Qcre <= 1e-9 or Qr <= 1e-9 or abs(fr) < 1e-12:
        return np.full_like(f, np.inf)

    Qe = Qcre + 1j * Qcim
    Qc_eff = abs(Qe) ** 2 / Qcre

    penalty_factor = 1.0
    if Qc_eff < Qr * 1.05:  # within 5% of the physical boundary
        ratio = Qc_eff / Qr
        if ratio < 0.5:  # far into the unphysical regime
            return np.full_like(f, np.inf)
        elif ratio < 1.0:  # unphysical, but not extreme
            penalty_factor = 1 + 100 * (1 - ratio) ** 2
        else:  # physical, but close enough to discourage
            penalty_factor = 1 + 5 * (1.05 - ratio) ** 2

    x = (f - fr) / fr
    with np.errstate(divide="ignore", invalid="ignore"):
        s21 = A * (1 - (Qr / Qe) / (1 + 2j * Qr * x))

    magnitude = np.abs(s21) * penalty_factor
    magnitude[~np.isfinite(magnitude)] = np.inf
    return magnitude


def fit_skewed(
    frequencies,
    s21,
    *,
    approx_Qr: float = 1e4,
    normalize: bool = True,
    fr_limit_hz: float | None = None,
) -> tuple[dict, dict]:
    """Fit :func:`s21_skewed` to one trace's magnitude.

    Args:
        frequencies (np.ndarray): Frequencies (Hz).
        s21 (np.ndarray): The complex trace. Only its magnitude is fitted.
        approx_Qr (float): Initial guess for Qr.
        normalize (bool): Divide the trace by its last point first, so ``A``
            comes out near 1.
        fr_limit_hz (float | None): Bound ``fr`` to within this much of the
            trace's middle frequency. None uses 37.5% of the span, which keeps
            the fit off a neighbour that leaked into the edge.

    Returns:
        tuple[dict, dict]: ``(params, errors)``.

        *params* holds ``fr``, ``Qr``, ``Qc``, ``Qi``, ``Qcre``, ``Qcim`` and
        ``A``. ``Qi`` may be ``inf`` — a lossless resonator is a fit result,
        not a failure. *errors* holds one-sigma uncertainties for all seven
        parameters. ``Qc`` and ``Qi`` use first-order propagation of the full
        fit covariance, including correlations. The approximation can be poor
        near vanishing internal loss; ``Qi = inf`` has infinite uncertainty.

    Raises:
        FitFailed: if the fit did not converge, or converged on something
            unphysical. The message says which.
        ValueError: if the two arrays do not describe the same trace.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    s21 = np.asarray(s21)
    if frequencies.shape != s21.shape:
        raise ValueError(
            f"frequencies has {frequencies.shape} points and s21 has "
            f"{s21.shape}: these are not the same trace."
        )
    if frequencies.size < 5:
        raise FitFailed(
            f"{frequencies.size} points is too few to fit five parameters"
        )

    if normalize:
        if abs(s21[-1]) < 1e-15:
            raise FitFailed(
                "the last sweep point is zero, so the trace cannot be "
                "normalized to it"
            )
        s21 = s21 / s21[-1]
    magnitude = np.abs(s21)

    # Centre the frequency bound on the sweep's middle sample.
    middle = frequencies[frequencies.size // 2]
    if fr_limit_hz is None:
        fr_limit_hz = abs(frequencies[-1] - frequencies[0]) * 0.375
    fr_low = max(frequencies.min(), middle - fr_limit_hz)
    fr_high = min(frequencies.max(), middle + fr_limit_hz)
    if not fr_high > fr_low:
        raise FitFailed(
            f"fr_limit_hz={fr_limit_hz:g} leaves no room to fit fr in around "
            f"{middle * 1e-6:.3f} MHz"
        )

    within = np.flatnonzero((frequencies >= fr_low) & (frequencies <= fr_high))
    fr_guess = frequencies[within[np.argmin(magnitude[within])]]
    a_guess = float(np.mean(magnitude[within]))

    # Qcre starts at 1.5 * Qr so the initial guess is on the physical side of
    # the Qc > Qr boundary s21_skewed penalises.
    initial = [fr_guess, approx_Qr, 1.5 * approx_Qr, 0.0, a_guess]
    bounds = (
        [fr_low, 1e2, 1.5e2, -np.inf, 0],
        [fr_high, 1e9, 1e9, np.inf, np.inf],
    )

    try:
        fitted, covariance = curve_fit(
            s21_skewed, frequencies, magnitude, p0=initial, bounds=bounds, maxfev=5000
        )
    except (RuntimeError, ValueError) as exc:
        raise FitFailed(f"the optimizer gave up: {exc}") from None
    except Exception as exc:  # linalg failures inside curve_fit, mostly
        raise FitFailed(f"the fit failed unexpectedly: {exc}") from None

    if not np.all(np.isfinite(fitted)):
        raise FitFailed("the fit returned non-finite parameters")
    if not np.all(np.isfinite(covariance)):
        raise FitFailed(
            "the covariance matrix is not finite, so the fit did not constrain "
            "the parameters"
        )
    errors = np.sqrt(np.diag(covariance))

    fr, Qr, Qcre, Qcim, A = fitted
    Qe = Qcre + 1j * Qcim
    Qc = abs(Qe) ** 2 / Qcre if Qcre > 1e-9 else np.nan
    if not (Qcre > 1e-9 and Qr > 1e-9 and Qc >= Qr):
        raise FitFailed(
            f"the fit converged on an unphysical resonator: Qc={Qc:.4g} is "
            f"below Qr={Qr:.4g}, which would radiate more than it stores"
        )

    with np.errstate(divide="ignore"):
        inverse_Qi = 1.0 / Qr - 1.0 / Qc
    # Qi = inf when the internal loss vanishes, which is a lossless resonator
    # rather than a failed fit.
    Qi = np.inf if inverse_Qi <= 1e-15 else 1.0 / inverse_Qi

    params = dict(
        zip(SKEWED_PARAMS, (float(fr), float(Qr), float(Qc), float(Qi),
                            float(Qcre), float(Qcim), float(A)))
    )
    errors = {
        name: float(err) for name, err in zip(SKEWED_FITTED_PARAMS, errors)
    }
    # Qc = Qcre + Qcim**2 / Qcre; Qi = 1 / (1/Qr - 1/Qc).
    # Gradients are ordered (Qr, Qcre, Qcim), matching the covariance slice.
    qc_gradient = np.array([0.0, 1 - (Qcim / Qcre)**2, 2 * Qcim / Qcre])
    q_covariance = covariance[1:4, 1:4]
    errors["Qc"] = float(np.sqrt(qc_gradient @ q_covariance @ qc_gradient))
    if np.isinf(Qi):
        errors["Qi"] = float("inf")
    else:
        qi_gradient = -(Qi / Qc)**2 * qc_gradient
        qi_gradient[0] = (Qi / Qr)**2
        errors["Qi"] = float(np.sqrt(qi_gradient @ q_covariance @ qi_gradient))
    return params, errors


# ─── The nonlinear resonator model (adapted from citkid) ──────────────────────


def _resolve_sweep_direction(frequencies, sweep_direction: str | None) -> str:
    if sweep_direction is None:
        f = np.asarray(frequencies).reshape(-1)
        sweep_direction = "downward" if len(f) > 1 and f[-1] < f[0] else "upward"
    if sweep_direction not in ("upward", "downward"):
        raise ValueError("sweep_direction must be upward or downward")
    return sweep_direction


def nonlinear_iq(f, fr, Qr, amp, phi, a, i0, q0, *,
                 sweep_direction: str | None = None):
    r"""Transmission through a nonlinear resonator.

    .. code-block:: text

                        /                           (j phi)   \
            (i0+j*q0) * |1 -        Qr             e^           |
                        |     --------------  X  ------------   |
                         \     Qc * cos(phi)       (1+ 2jy)    /

    where ``y`` is the detuning from the power-shifted resonance and ``yg`` the
    generator's detuning from the low-power resonance ``fr``, related by
    Swenson et al. 2013 (J. Appl. Phys. 113, 104501), eq. 13::

        y = yg + a / (1 + 4 y^2)      where yg = Qr * (f - fr) / fr

    The stored energy pulls the resonance to lower frequency, so the dip sits
    below ``fr`` and a sweep leans that way; for ``a > 4*sqrt(3)/9`` the relation
    is multivalued (bifurcation) and the curve has a step where
    :func:`get_y_nonlinear` reaches the end of the selected stable branch.

    This model has no cable-delay parameter.

    Args:
        f (np.ndarray): Frequencies (Hz).
        fr (float): Resonance frequency (Hz).
        Qr (float): Total (loaded) quality factor.
        amp (float): ``Qr / Qc``, so ``0 < amp < 1``.
        phi (float): Impedance-mismatch rotation between resonator and readout
            (radians).
        a (float): Nonlinearity. Bifurcation is at :data:`BIFURCATION_A`,
            ``4*sqrt(3)/9 ≈ 0.7698``; a linear resonator sits near 0.
        i0 (float): Real part of the overall gain and phase offset.
        q0 (float): Imaginary part of the same.
        sweep_direction: Acquisition direction; inferred from frequency order
            when omitted. Assumes the sweep entered from outside bistability.
            Direction alone does not specify a state prepared inside it.

    Returns:
        np.ndarray: complex S21 at each frequency.
    """
    yg = Qr * (f - fr) / fr
    y = get_y_nonlinear(
        yg, a, sweep_direction=_resolve_sweep_direction(f, sweep_direction))
    resonator = 1.0 - (amp / np.cos(phi)) * np.exp(1.0j * phi) / (1.0 + 2.0j * y)
    return (i0 + 1.0j * q0) * resonator


def get_y_nonlinear(yg, a, *, sweep_direction: str = "upward"):
    """The detuning ``y`` that solves Swenson et al. 2013 eq. 13,
    ``y = yg + a / (1 + 4 y^2)``.

    For positive ``a``, the root lies in ``[yg, yg + a]``. Below bifurcation
    (``a < 4*sqrt(3)/9``), there is one root. Twelve bisection steps narrow the
    bracket, followed by four clipped Newton steps.
    Above bifurcation, upward sweeps take the smallest root and downward
    sweeps the largest. These are the stable branches reached from outside
    the bistable interval. A derivative across their jumps is meaningless.
    ``a == 0`` is the linear case and returns *yg* untouched.

    Args:
        yg (float | np.ndarray): Generator detuning from the low-power
            resonance, ``Qr * (f - fr) / fr``.
        a (float): Nonlinearity parameter.
        sweep_direction: "upward" or "downward".

    Returns:
        float | np.ndarray: detuning from the power-shifted resonance, in units
        of ``fr / Qr``, matching *yg*'s shape.
    """
    sweep_direction = _resolve_sweep_direction(yg, sweep_direction)
    if a == 0:
        return yg
    scalar = np.isscalar(yg)
    yg = np.atleast_1d(np.asarray(yg, dtype=np.float64))
    lo = yg.copy()
    hi = yg + a
    if a > BIFURCATION_A:
        # P(y) = (y - yg)(1 + 4y²) - a. Its stationary points
        # bracket the middle root; restrict bisection to the selected outer
        # root wherever three roots exist.
        discriminant = yg * yg - 0.75
        delta = np.sqrt(np.maximum(discriminant, 0))
        left, right = (yg - delta) / 3, (yg + delta) / 3
        if sweep_direction == "upward":
            valid = ((discriminant > 0) & (left > lo) & (left < hi)
                     & ((left - yg) * (1 + 4 * left * left) >= a))
            hi = np.where(valid, left, hi)
        else:
            valid = ((discriminant > 0) & (right > lo) & (right < hi)
                     & ((right - yg) * (1 + 4 * right * right) <= a))
            lo = np.where(valid, right, lo)
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        above = mid - a / (1 + 4 * mid * mid) > yg
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    y = 0.5 * (lo + hi)
    for _ in range(4):
        denominator = 1 + 4 * y * y
        residual = y - a / denominator - yg
        slope = 1 + 8 * a * y / (denominator * denominator)
        step = residual / np.where(np.abs(slope) > 1e-12, slope, 1e-12)
        y = np.clip(y - step, lo, hi)
    return float(y[0]) if scalar else y


def remove_gain(frequencies, iq, *, n_extrema_points: int = 5):
    """Divide out the readout gain, estimated from the sweep's own ends.

    Assumes both ends are off-resonance. Their complex mean estimates gain
    and phase without a separate gain scan.

    Args:
        frequencies (np.ndarray): Frequencies (Hz). Used only for its length.
        iq (np.ndarray): The complex trace.
        n_extrema_points (int): Points to average at each end, limited to
            ``max(1, min(n_extrema_points, len(frequencies) // 4))``.

    Returns:
        tuple[np.ndarray, complex]: the gain-corrected trace, and the complex
        gain that was divided out.

    Raises:
        FitFailed: if the estimated gain magnitude is at most ``1e-10``.
    """
    n_average = max(1, min(n_extrema_points, len(frequencies) // 4))
    gain = complex(
        (np.mean(iq[:n_average]) + np.mean(iq[-n_average:])) / 2.0
    )

    if abs(gain) <= 1e-10:
        raise FitFailed(
            "the trace is zero at both ends, so there is no readout gain to "
            "divide out"
        )
    return iq / gain, gain


def guess_p0_nonlinear(f, z) -> list[float]:
    """Initial guesses for :func:`fit_nonlinear_iq`, read off the trace.

    Args:
        f (np.ndarray): Frequencies (Hz).
        z (np.ndarray): The complex trace, gain-corrected.

    Returns:
        list[float]: ``[fr, Qr, amp, phi, a, i0, q0]``.
    """
    magnitude = np.abs(z)
    deepest = np.argmin(magnitude)
    fr_guess = f[deepest]

    # TODO: Estimate Qr from the crossings around the dip; the outermost
    # samples above the threshold can measure the sweep span instead.
    # Estimate Qr from the span of samples more than 3 dB above the minimum.
    magnitude_db = 20 * np.log10(magnitude)
    above_3db = magnitude_db > magnitude_db[deepest] + 3
    Qr_guess = 1e4
    if np.any(above_3db):
        first, last = np.flatnonzero(above_3db)[[0, -1]]
        if last > first:
            Qr_guess = fr_guess / (f[last] - f[first])
    Qr_guess = np.clip(Qr_guess, 1e3, 1e7)

    # amp = Qr/Qc, which is how deep the dip goes.
    off_resonance = np.mean([magnitude[0], magnitude[-1]])
    amp_guess = 0.5
    if off_resonance > 0:
        amp_guess = np.clip(1 - magnitude[deepest] / off_resonance, 0.1, 0.99)

    phase = np.unwrap(np.angle(z))
    phi_guess = np.clip((phase[-1] - phase[0]) / 2, -np.pi / 2, np.pi / 2)

    return [
        fr_guess,
        Qr_guess,
        amp_guess,
        phi_guess,
        0.01,  # Start near the linear regime.
        float(np.real(np.mean(z[[0, -1]]))),
        float(np.imag(np.mean(z[[0, -1]]))),
    ]


def fit_nonlinear_iq(
    frequencies,
    z,
    *,
    fit_nonlinearity: bool = True,
    bounds=None,
    p0=None,
    max_iterations: int = 3,
    sweep_direction: str | None = None,
) -> tuple[dict, dict, float]:
    """Fit :func:`nonlinear_iq` to one gain-corrected complex trace.

    Fit stacked real and imaginary parts. Scale ``fr`` by ``1e-6`` and ``Qr``
    by ``1e-4`` during optimization.

    Args:
        frequencies (np.ndarray): Frequencies (Hz).
        z (np.ndarray): The complex trace, with the readout gain already
            divided out — see :func:`remove_gain`.
        fit_nonlinearity (bool): Fit ``a``. False constrains it to
            ``[-1e-10, 1e-10]`` to approximate a linear resonator.
        bounds (tuple[list, list] | None): Lower and upper bounds on
            ``[fr, Qr, amp, phi, a, i0, q0]``. None uses the sweep's own
            frequency range for ``fr``, ``Qr`` in ``[1e3, 1e7]``, ``amp`` in
            ``[0.01, 0.99]``, ``phi`` in ``[-pi/2, pi/2]``, ``a`` in
            ``[0, 0.9]`` and ``i0``, ``q0`` in ``[-100, 100]``. The upper bound
            on ``a`` permits fits above :data:`BIFURCATION_A`, using the
            stable branch selected by ``sweep_direction``.
        p0 (list | None): Initial guesses. None reads them off the trace with
            :func:`guess_p0_nonlinear`.
        sweep_direction: Acquisition direction; inferred before sorting if
            omitted. The model assumes entry from outside bistability.
        max_iterations (int): Maximum optimizer attempts. Refine a fit below
            residual 0.1; otherwise reduce the amplitude guess. Stop below
            residual ``1e-3`` or on an optimizer exception.

    Returns:
        tuple[dict, dict, float]: ``(params, errors, residual)``.

        *params* holds the seven fitted parameters plus ``Qc`` and ``Qi``,
        derived from ``Qr`` and ``amp`` where ``amp < 1``. *residual* is the
        RMS error over the mean magnitude — dimensionless, and the number
        :func:`fit_sweeps` thresholds on.

    Raises:
        FitFailed: if no iteration converged. The message carries the
            optimizer's own complaint.
        ValueError: if the two arrays do not describe the same trace.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    z = np.asarray(z)
    if frequencies.shape != z.shape:
        raise ValueError(
            f"frequencies has {frequencies.shape} points and z has {z.shape}: "
            f"these are not the same trace."
        )

    sweep_direction = _resolve_sweep_direction(frequencies, sweep_direction)
    order = np.argsort(frequencies)
    frequencies = frequencies[order]
    z = z[order]

    p0 = list(guess_p0_nonlinear(frequencies, z) if p0 is None else p0)
    if bounds is None:
        bounds = (
            #  fr,               Qr,  amp,      phi,    a,    i0,    q0
            [frequencies.min(), 1e3, 0.01, -np.pi / 2, 0.0, -1e2, -1e2],
            [frequencies.max(), 1e7, 0.99, np.pi / 2, 0.9, 1e2, 1e2],
        )
    bounds = ([float(b) for b in bounds[0]], [float(b) for b in bounds[1]])

    if not fit_nonlinearity:
        # curve_fit needs a non-degenerate interval, so pin a to a sliver.
        p0[4] = 0.0
        bounds[0][4], bounds[1][4] = -1e-10, 1e-10

    # Scale fr to MHz and Qr to units of 1e4.
    scale = [1e-6, 1e-4, 1, 1, 1, 1, 1]

    def model(f, fr_scaled, Qr_scaled, amp, phi, a, i0, q0):
        modelled = nonlinear_iq(
            f, fr_scaled / scale[0], Qr_scaled / scale[1], amp, phi, a, i0, q0,
            sweep_direction=sweep_direction
        )
        return np.hstack((np.real(modelled), np.imag(modelled)))

    stacked = np.hstack((np.real(z), np.imag(z)))
    current_p0 = [p * s for p, s in zip(p0, scale)]
    scaled_bounds = (
        [b * s for b, s in zip(bounds[0], scale)],
        [b * s for b, s in zip(bounds[1], scale)],
    )

    best, best_errors, best_residual = None, None, np.inf
    complaint = "the optimizer did not converge"

    for _ in range(max_iterations):
        try:
            fitted_scaled, covariance = curve_fit(
                model,
                frequencies,
                stacked,
                p0=current_p0,
                bounds=scaled_bounds,
                maxfev=5000,
            )
        except Exception as exc:
            complaint = str(exc)
            break

        fitted = [p / s for p, s in zip(fitted_scaled, scale)]
        errors = [e / s for e, s in zip(np.sqrt(np.diag(covariance)), scale)]
        residual = calculate_residuals(z, nonlinear_iq(
            frequencies, *fitted, sweep_direction=sweep_direction))

        if residual < best_residual:
            best, best_errors, best_residual = fitted, errors, residual

        if residual < 1e-3:
            break
        if residual < 0.1:
            current_p0 = list(fitted_scaled)  # good; refine from here
        else:
            # Poor fit: the depth guess is the usual culprit, so shrink it.
            current_p0[2] = max(current_p0[2] * 0.7, scaled_bounds[0][2])

    if best is None:
        raise FitFailed(complaint)

    params = {name: float(v) for name, v in zip(NONLINEAR_PARAMS, best)}
    errors = {name: float(e) for name, e in zip(NONLINEAR_PARAMS, best_errors)}

    # Custom bounds may permit amp >= 1; omit derived Qs in that case.
    if params["amp"] < 1:
        Qc = params["Qr"] / params["amp"]
        params["Qc"] = Qc
        params["Qi"] = 1.0 / (1.0 / params["Qr"] - 1.0 / Qc)

    return params, errors, float(best_residual)


def calculate_residuals(measured, modelled) -> float:
    """Return complex RMS error divided by the measured mean magnitude.

    Return the unnormalized RMS error when the mean magnitude is zero.
    """
    rms = np.sqrt(np.mean(np.abs(measured - modelled) ** 2))
    mean_magnitude = np.mean(np.abs(measured))
    return float(rms / mean_magnitude if mean_magnitude > 0 else rms)


# ─── The IQ circle ────────────────────────────────────────────────────────────


def circle_fit_pratt(x, y):
    """Fit an algebraic circle using centred moments and a pseudoinverse.

    Minimizes an algebraic residual, not geometric distance to the circle.
    Degenerate point sets are not explicitly rejected.

    Args:
        x (np.ndarray): Real components (I).
        y (np.ndarray): Imaginary components (Q).

    Returns:
        tuple: ``(xc, yc, radius)``, or ``(None, None, None)`` for fewer than
        three points, a failed solve, or an invalid result.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if n < 3:
        return None, None, None

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    u = x - x_mean
    v = y - y_mean

    Suu = np.sum(u**2)
    Svv = np.sum(v**2)
    Suv = np.sum(u * v)

    # Solve B @ [xc, yc] = C for the centre relative to the centroid.
    B = np.array([[Suu, Suv], [Suv, Svv]])
    C = np.array(
        [
            0.5 * (np.sum(u**3) + np.sum(u * v**2)),
            0.5 * (np.sum(v**3) + np.sum(u**2 * v)),
        ]
    )
    try:
        # Pseudo-inverse: a sweep that barely curves gives a near-singular B.
        xc_relative, yc_relative = np.linalg.pinv(B) @ C
    except np.linalg.LinAlgError:
        return None, None, None

    radius_squared = xc_relative**2 + yc_relative**2 + (Suu + Svv) / n
    if radius_squared < 0:
        return None, None, None

    xc = xc_relative + x_mean
    yc = yc_relative + y_mean
    radius = np.sqrt(radius_squared)
    if not (np.isfinite(xc) and np.isfinite(yc) and np.isfinite(radius)):
        return None, None, None

    return float(xc), float(yc), float(radius)


def center_resonance_iq_circle(iq):
    """Fit and subtract an IQ circle centre; return the input if fitting fails."""
    xc, yc, _ = circle_fit_pratt(np.asarray(iq).real, np.asarray(iq).imag)
    if xc is None:
        return iq
    return iq - complex(xc, yc)
