"""
The legacy dict-walking analysis surface for multisweep data.

The resonance finder and the resonator fitters that used to live here have
moved to :mod:`rfmux.tuning.find_resonances` and :mod:`rfmux.tuning.fits`.
They are analysis over two arrays rather than board operations, so they belong
with the other pure layers, and they are re-exported below so callers reaching
for ``fitting.s21_skewed`` still find it.

What is left in this module is the old *dict-walking* API — the functions that
take a whole multisweep return and write their results back into it. Every one
of them reads the pre-schema-2 contract: ``iq_complex``,
``original_center_frequency``, integer keys. ``multisweep`` returns
``iq_counts`` keyed by resonator name and has done since the sweep stopped
doing anything but measure, so these work on files saved before that change and
on nothing else.

For sweeps taken since, use :func:`rfmux.tuning.fit_sweeps`, which reads
``iq_counts`` and writes each model's results into the sweep entry's ``fits``
subdict rather than flat beside the measurement.
"""

import numpy as np
import warnings

# Re-exported for callers that reach for them by attribute — Periscope does.
# The implementations live in rfmux/tuning/fits.py.
from ...tuning.fits import (
    FitFailed,
    center_resonance_iq_circle,
    circle_fit_pratt,
    fit_skewed,
    s21_skewed,
)

# Listed so a linter reports the imports above as re-exports rather than as
# five unused names, and so this module's surface is stated in one place.
__all__ = [
    # Re-exported from rfmux.tuning.fits.
    "FitFailed",
    "center_resonance_iq_circle",
    "circle_fit_pratt",
    "fit_skewed",
    "s21_skewed",
    # Still implemented here.
    "identify_bifurcation",
    "find_resonances",
    "fit_skewed_multisweep",
]


def identify_bifurcation(iq_complex: np.ndarray, threshold_factor: float = 5.0, min_peak_prominence_factor: float = 0.5, min_points_for_detection: int = 10) -> bool:
    """
    Identifies potential bifurcations in a resonator sweep by looking for
    sharp discontinuities (peaks) in the running difference of IQ point norms.

    A bifurcation often manifests as a sudden jump in the IQ trace.

    Args:
        iq_complex (np.ndarray): Complex S21 array for a single resonance sweep.
        threshold_factor (float): Multiplier for the standard deviation of
                                  the diff_norms to set a peak height threshold.
                                  Peaks must be above median + threshold_factor * std.
        min_peak_prominence_factor (float): Factor to multiply by the median of diff_norms
                                            to set a minimum prominence for a peak.
                                            Helps avoid flagging noise as bifurcation.
        min_points_for_detection (int): Minimum number of IQ points required to attempt
                                        bifurcation detection.

    Returns:
        bool: True if a potential bifurcation is detected, False otherwise.
    """
    if not isinstance(iq_complex, np.ndarray) or iq_complex.ndim != 1:
        warnings.warn("identify_bifurcation: iq_complex must be a 1D numpy array.")
        return False
    
    if len(iq_complex) < min_points_for_detection:
        # Not enough points to reliably detect bifurcation via this method
        return False

    try:
        from scipy.signal import find_peaks # Local import to keep it self-contained if moved
    except ImportError:
        warnings.warn("identify_bifurcation: scipy.signal.find_peaks is required but not found. Cannot detect bifurcation.")
        return False

    norms = np.abs(iq_complex)
    if len(norms) < 2: # Need at least two points to calculate a difference
        return False
        
    diff_norms = np.abs(np.diff(norms))
    if len(diff_norms) == 0: # Should not happen if len(norms) >= 2
        return False

    median_diff = np.median(diff_norms)
    std_diff = np.std(diff_norms)

    # Avoid issues if std_diff is zero (e.g., perfectly smooth line or very few points)
    if std_diff < 1e-9: # Effectively zero
        # If std is zero, any deviation is significant if median_diff is also small.
        # If median_diff is large, then small deviations are not peaks.
        # This case is unlikely for real data with bifurcations.
        # We can simply check if any diff_norm is significantly larger than median_diff.
        # A simple check: if max diff is much larger than median.
        if median_diff < 1e-9: # All diffs are zero or near zero
            return False 
        # If median is non-zero but std is zero, it means all diffs are the same.
        # No peaks possible in this scenario.
        return False


    # Height threshold: significantly above the typical point-to-point variation
    height_threshold = median_diff + threshold_factor * std_diff
    
    # Prominence threshold: the peak must stand out relative to its surroundings
    # by a factor of the median difference. This helps filter out noise on a
    # generally "bumpy" trace if the bumps aren't sharp discontinuities.
    prominence_threshold = min_peak_prominence_factor * median_diff
    # Ensure prominence is at least a small absolute value if median_diff is tiny
    prominence_threshold = max(prominence_threshold, 1e-5 * np.max(norms) if np.max(norms) > 0 else 1e-5)


    peaks, properties = find_peaks(diff_norms, height=height_threshold, prominence=prominence_threshold)

    if len(peaks) > 0:
        # print(f"Bifurcation detected: {len(peaks)} peaks found. Example peak height: {properties['peak_heights'][0] if 'peak_heights' in properties else 'N/A'}, Prominence: {properties['prominences'][0] if 'prominences' in properties else 'N/A'}")
        # print(f"Thresholds used: height > {height_threshold:.4g}, prominence > {prominence_threshold:.4g}")
        # print(f"Diff_norms stats: median={median_diff:.4g}, std={std_diff:.4g}")
        return True
        
    return False


def find_resonances(
    frequencies: np.ndarray,
    iq_complex: np.ndarray,
    expected_resonances: int | None = None,
    min_dip_depth_db: float = 1.0,
    min_Q: float = 1e4,
    max_Q: float = 1e7,
    min_resonance_separation_hz: float = 100e3,
    data_exponent: float = 2.0,
    module_identifier: str | int | None = None,
):
    """
    Deprecated. Use :func:`rfmux.tuning.find_resonances` instead.

    Kept so existing callers keep working; it forwards to the implementation in
    ``rfmux/tuning/find_resonances.py`` and converts the result back to the dict
    this function has always returned. There is only one search algorithm, and
    it lives there — see that module for the parameters, which are the same ones
    under slightly tidier names.

    Two things to know if you are reading old output:

    * ``min_resonance_separation_hz`` means something stricter than it used to.
      It was ``find_peaks(distance=...)``, which counted samples and kept the
      tallest member of a close group. It is now a collision cut in Hz: any
      candidate with a neighbour inside the separation is removed *along with
      that neighbour*, because a tone on either member of a collided pair reads
      the other one too. Well-separated arrays are unaffected; a collided pair
      that used to yield one resonance now yields none. Callers that want the
      old permissiveness should pass a much smaller value — the new default in
      ``rfmux.tuning`` is 0 Hz, which cuts exact duplicates only.
    * ``data_exponent`` is accepted and ignored. Raising ``|S21|`` to a power is
      a multiplier in dB, so it scaled dips and noise together and could not
      change what was found; its only real effect here was that the prominence
      threshold was *not* scaled with it, making the exponent a disguised way of
      dividing ``min_dip_depth_db`` by it. With it gone, a caller that passed the
      default of 2.0 is effectively asking for twice the dip depth it used to
      get. Halve ``min_dip_depth_db`` to match the old behaviour.
    * ``prominence_db`` is the dip depth in dB, where it used to be that depth
      multiplied by ``data_exponent``. With the old default of 2.0 the number is
      half what this function printed before.

    Returns
    -------
    dict
        - 'resonance_frequencies': list[float] — frequencies (Hz).
        - 'resonances_details': list[dict] — one dict per resonance, with
          'frequency', 'prominence_db', 'width_hz' and 'q_estimated'.
    """
    warnings.warn(
        "fitting.find_resonances is deprecated; use rfmux.tuning.find_resonances "
        "(or find_resonances_in_netanal for a netanal result), which returns a "
        "ResonanceSearch instead of a dict.",
        DeprecationWarning,
        stacklevel=2,
    )

    from ...tuning.find_resonances import find_resonances as _find_resonances

    # Preserved from the original: a degenerate trace warns and yields nothing
    # rather than raising, because callers (the Periscope panel among them) rely
    # on getting empty lists back.
    if len(frequencies) < 3 or len(frequencies) != len(iq_complex):
        warnings.warn(
            f"Resonance finding skipped for {module_identifier or 'data'}: "
            f"Insufficient or mismatched data points."
        )
        return {'resonance_frequencies': [], 'resonances_details': []}

    found = _find_resonances(
        frequencies,
        iq_complex,
        min_dip_depth_db=min_dip_depth_db,
        min_Q=min_Q,
        max_Q=max_Q,
        min_separation_hz=min_resonance_separation_hz,
        expected_resonances=expected_resonances,
        label=str(module_identifier) if module_identifier is not None else None,
    )

    return {
        'resonance_frequencies': found.resonance_frequencies_hz.tolist(),
        'resonances_details': [
            {
                'frequency': c.frequency_hz,
                'prominence_db': c.depth_db,
                'width_hz': c.width_hz,
                'q_estimated': c.q_estimate,
            }
            for c in found.candidates
        ],
    }



def _legacy_skewed_params(
    frequencies, iq_complex, approx_Q_for_fit, normalize_fit, fr_lim_fit
) -> dict:
    """The dict ``fit_skewed`` used to return, rebuilt from the one it returns now.

    It raises :class:`~rfmux.tuning.fits.FitFailed` with a reason these days
    rather than filling all fourteen fields with the string ``'nan'``. The
    deprecated walker below is the only thing that still reads the old shape,
    so the translation lives with it rather than in the new module.
    """
    names = ("fr", "Qr", "Qc", "Qi", "Qcre", "Qcim", "A")
    try:
        params, errors = fit_skewed(
            frequencies,
            iq_complex,
            approx_Qr=approx_Q_for_fit,
            normalize=normalize_fit,
            fr_limit_hz=fr_lim_fit,
        )
    except FitFailed as exc:
        warnings.warn(f"Skewed fit failed: {exc}")
        return {key: "nan" for name in names for key in (name, f"{name}_err")}

    legacy = {}
    for name in names:
        legacy[name] = params[name]
        # Qc and Qi are derived rather than fitted, so they have no error.
        legacy[f"{name}_err"] = errors.get(name, "nan")
    return legacy


def fit_skewed_multisweep(
    multisweep_data: dict | list[dict],
    approx_Q_for_fit: float = 1e4,
    fit_resonances: bool = True,
    center_iq_circle: bool = True,
    normalize_fit: bool = True,
    fr_lim_fit: float | None = None
):
    """
    Deprecated. Use :func:`rfmux.tuning.fit_sweeps` instead.

    Reads ``iq_complex`` and ``original_center_frequency`` off each entry and
    expects integer keys — the contract multisweep had before it stopped doing
    anything but measure. It works on data saved under that contract and
    nothing newer.

    It also writes flat: ``fit_params`` and ``iq_centered`` land beside the
    measurement they describe, where the replacement puts them under ``fits``
    keyed by model. And it stores the centred trace, where the replacement
    stores the circle's centre and rebuilds the trace on request — one complex
    number instead of an N-point copy of data the entry already has.

    Args:
        multisweep_data (dict | list[dict]): The data returned by the
                                             `rfmux.algorithms.measurement.multisweep.multisweep` function.
                                             For single module: dict with final center frequencies as keys.
                                             For multiple modules: list of such dictionaries.
                                             Each value dict contains 'frequencies', 'iq_complex', and other fields.
        approx_Q_for_fit (float, optional): Initial Qr guess for the skewed fit.
                                            Defaults to 1e4.
        fit_resonances (bool, optional): If True, perform skewed Lorentzian fitting.
                                         Defaults to True.
        center_iq_circle (bool, optional): If True, perform IQ circle centering.
                                           Defaults to True.
        normalize_fit (bool, optional): Whether to normalize S21 data before fitting in `fit_skewed`.
                                        Defaults to True.
        fr_lim_fit (float | None, optional): Fit fr only within +/- fr_lim Hz of the center frequency
                                             in `fit_skewed`. Defaults to None (use full range).

    Returns:
        dict | list[dict]: The input `multisweep_data` with 'fit_params' and 'iq_centered'
                           added to each resonance data dictionary if the respective operations were performed.
    """
    warnings.warn(
        "fitting.fit_skewed_multisweep is deprecated; use "
        "rfmux.tuning.fit_sweeps, which reads iq_counts and writes its results "
        "into each sweep entry's 'fits' subdict.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _fit_skewed_multisweep(
        multisweep_data,
        approx_Q_for_fit,
        fit_resonances,
        center_iq_circle,
        normalize_fit,
        fr_lim_fit,
    )


def _fit_skewed_multisweep(
    multisweep_data,
    approx_Q_for_fit,
    fit_resonances,
    center_iq_circle,
    normalize_fit,
    fr_lim_fit,
):
    """:func:`fit_skewed_multisweep` without the warning, so recursing is quiet."""
    # Handle multi-module case (list of dictionaries)
    if isinstance(multisweep_data, list):
        return [_fit_skewed_multisweep(
            module_data, approx_Q_for_fit, fit_resonances, center_iq_circle, normalize_fit, fr_lim_fit
        ) for module_data in multisweep_data]

    # Handle single module case (dictionary)
    if not isinstance(multisweep_data, dict):
        warnings.warn("fit_skewed_multisweep: input is not a dictionary or list. Returning as is.")
        return multisweep_data

    # Process each resonance in the multisweep data
    for res_key, resonance_data in multisweep_data.items():
        if not isinstance(resonance_data, dict):
            warnings.warn(f"fit_skewed_multisweep: resonance data for key {res_key} is not a dictionary. Skipping.")
            continue

        # Expect index-based keys only
        if not isinstance(res_key, (int, np.integer)):
            warnings.warn(f"fit_skewed_multisweep: Expected integer index key, got {type(res_key)} for key {res_key}. Skipping.")
            continue

        frequencies = resonance_data.get('frequencies')
        iq_complex = resonance_data.get('iq_complex')
        original_cf = resonance_data.get('original_center_frequency')

        if original_cf is None:
            warnings.warn(f"fit_skewed_multisweep: 'original_center_frequency' missing for index {res_key}. Skipping.")
            continue

        if frequencies is None or iq_complex is None:
            warnings.warn(f"fit_skewed_multisweep: 'frequencies' or 'iq_complex' missing for index {res_key}. Skipping.")
            continue

        # Initialize new result keys (don't overwrite existing data)
        resonance_data['fit_params'] = None
        resonance_data['iq_centered'] = None

        if fit_resonances:
            try:
                # Set fr_lim based on the sweep span if not provided and we have frequency data
                if fr_lim_fit is None and len(frequencies) > 1:
                    freq_span = frequencies[-1] - frequencies[0]
                    # Use 75% of the sweep span as the fitting limit
                    auto_fr_lim = abs(freq_span) * 0.375
                else:
                    auto_fr_lim = fr_lim_fit

                resonance_data['fit_params'] = _legacy_skewed_params(
                    frequencies, iq_complex, approx_Q_for_fit, normalize_fit, auto_fr_lim
                )

            except Exception as e:
                warnings.warn(f"Fitting failed for resonance at {original_cf*1e-6:.3f} MHz during post-processing: {e}")
                # resonance_data['fit_params'] remains None

        if center_iq_circle:
            try:
                # Always center the current iq_complex data
                resonance_data['iq_centered'] = center_resonance_iq_circle(iq_complex)

            except Exception as e:
                warnings.warn(f"IQ centering failed for resonance at {original_cf*1e-6:.3f} MHz during post-processing: {e}")
                # resonance_data['iq_centered'] remains None

    return multisweep_data
