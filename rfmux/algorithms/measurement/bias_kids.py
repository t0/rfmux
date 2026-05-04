"""
bias_kids: Bias finding and hardware programming for KIDs readout.

Two independent operations are provided:

1. **find_bias_points** — pure analysis; examines multi-amplitude multisweep
   data to identify the optimal bias amplitude and frequency for each
   resonator, then updates ``res_info_dict`` in-place.

2. **apply_bias** — async hardware step; reads the bias conditions stored in
   ``res_info_dict`` and programmes the CRS channels.

Keeping the two steps separate allows the user to inspect the results of the
bias-finding analysis (in the Periscope multisweep panel) before deciding to
commit them to hardware.

Bias-frequency refinement
-------------------------
Rather than using the sweep centre frequency (which may be off from the true
resonance by up to half the sweep span), the resonance frequency is located by
finding the point of maximum ``|dI/df + j·dQ/df|`` — the IQ arc-length speed —
within the sweep.  This method is robust and works directly on the measured
sweep data without requiring a successful resonance fit.

Bifurcation detection
---------------------
Bifurcation is detected by examining the derivative of the normalised IQ
arc-length speed with respect to frequency.  A bifurcated resonance produces a
characteristic positive spike immediately followed by a negative spike in this
derivative.  This approach is more reliable than nonlinearity-parameter-based
methods because it does not require a successful nonlinear fit.

df-calibration
--------------
The complex calibration factor ``df_cal = 1 / (dI/df + j·dQ/df)`` (in Hz/V,
evaluated at the bias point using ``iq_volts``) is the sole source for
Periscope's df-unit display mode.

Reference
---------
Algorithm ported from hidfmux/analysis/find_bias.py (Maclean Rouble,
McGill Cosmology).
"""

import numpy as np
import asyncio
import warnings
from typing import Dict, Optional, Any, Callable, Literal

from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline


__all__ = [
    'compute_iq_derivative_spline',
    'find_max_derivative_frequency',
    'detect_bifurcation_derivative',
    'find_bias_points',
    'apply_bias',
]


# ── Section A: Derivative utilities ──────────────────────────────────────────

def compute_iq_derivative_spline(
    frequencies: np.ndarray,
    iq: np.ndarray,
) -> tuple:
    """
    Fit cubic splines to I(f) and Q(f) and return their derivative callables.

    This is the low-level building block used by :func:`find_max_derivative_frequency`.
    It is exposed separately so that other analysis functions can compute IQ
    derivatives without repeating the spline fitting.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency array in Hz.  Need not be sorted — data are sorted internally.
    iq : np.ndarray
        Complex IQ data (e.g. ``iq_volts`` from a multisweep entry).

    Returns
    -------
    dI_df : scipy.interpolate.PPoly
        Derivative spline of the I (real) component.  Callable as
        ``dI_df(f)`` → float or ndarray.
    dQ_df : scipy.interpolate.PPoly
        Derivative spline of the Q (imaginary) component.  Callable as
        ``dQ_df(f)`` → float or ndarray.
    """
    sort_idx = np.argsort(frequencies)
    freq_sorted = frequencies[sort_idx]
    iq_sorted = iq[sort_idx]

    dI_df = CubicSpline(freq_sorted, iq_sorted.real).derivative()
    dQ_df = CubicSpline(freq_sorted, iq_sorted.imag).derivative()
    return dI_df, dQ_df


def find_max_derivative_frequency(
    frequencies: np.ndarray,
    iq: np.ndarray,
    reference_freq: float,
    max_distance_hz: float = 100e3,
) -> tuple[float, float, float]:
    """
    Find the frequency of maximum ``|dI/df + j·dQ/df|`` within the sweep.

    The point of maximum IQ arc-length speed corresponds to the resonance
    frequency for a standard resonator sweep.  A sanity check is applied: if
    the identified point is more than *max_distance_hz* away from
    *reference_freq* it is assumed to be a glitch or outlier, and
    *reference_freq* is returned instead.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency array in Hz.
    iq : np.ndarray
        Complex IQ data.  Should be in volts (``iq_volts``) so that the
        returned derivatives are in V/Hz and can be used to compute the
        ``df_calibration`` factor.
    reference_freq : float
        Reference frequency (Hz) for the sanity check.  Typically the fitted
        resonance frequency ``fr``, or the sweep centre frequency as a
        fallback.
    max_distance_hz : float
        Maximum allowed distance (Hz) between the max-derivative point and
        *reference_freq* before falling back to *reference_freq*.
        Default: 100 kHz.

    Returns
    -------
    bias_freq : float
        Refined bias frequency in Hz.
    dI_df_at_bias : float
        dI/df evaluated at *bias_freq* in the same units as *iq* per Hz.
    dQ_df_at_bias : float
        dQ/df evaluated at *bias_freq* in the same units as *iq* per Hz.
    """
    dI_df, dQ_df = compute_iq_derivative_spline(frequencies, iq)

    sort_idx = np.argsort(frequencies)
    freq_sorted = frequencies[sort_idx]

    # Evaluate |dI/df + j*dQ/df| at each measured frequency point
    deriv_mag = np.abs(dI_df(freq_sorted) + 1j * dQ_df(freq_sorted))
    f_max_deriv = float(freq_sorted[np.argmax(deriv_mag)])

    # Sanity check
    if abs(f_max_deriv - reference_freq) < max_distance_hz:
        bias_freq = f_max_deriv
    else:
        warnings.warn(
            f"Max-derivative point ({f_max_deriv * 1e-6:.4f} MHz) is "
            f"{abs(f_max_deriv - reference_freq) * 1e-3:.1f} kHz from the "
            f"reference ({reference_freq * 1e-6:.4f} MHz) — exceeds "
            f"max_distance_hz = {max_distance_hz * 1e-3:.0f} kHz. "
            f"Falling back to reference frequency."
        )
        bias_freq = reference_freq

    dI_df_at_bias = float(dI_df(bias_freq))
    dQ_df_at_bias = float(dQ_df(bias_freq))

    return bias_freq, dI_df_at_bias, dQ_df_at_bias


# ── Section B: Bifurcation detection ─────────────────────────────────────────

def detect_bifurcation_derivative(
    frequencies: np.ndarray,
    iq: np.ndarray,
    spike_prominence_factor: float = 2.0,
    spike_height_factor: float = 3.0,
    return_diagnostics: bool = False,
):
    """
    Detect bifurcation via spikes in the derivative of the IQ arc-length speed.

    **Method**

    The normalised IQ arc-length speed is::

        dist = sqrt((I_speed / df)^2 + (Q_speed / df)^2)

    where I and Q are each normalised by their own range.  The first derivative
    of this (``arc_speed_gradient = diff(dist)``) exhibits a characteristic
    positive spike immediately followed by a negative spike when the resonance
    is bifurcated — the IQ trace "jumps" and then returns to the baseline.

    Spike detection thresholds:

    * Prominence: ``(dist.max() - dist.min()) / spike_prominence_factor``
    * Height:     ``spike_height_factor * std(arc_speed_gradient)``

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency array in Hz.
    iq : np.ndarray
        Complex IQ data (counts or volts — only the *shape* matters).
    spike_prominence_factor : float
        Larger value → less sensitive to bifurcation.  Default: 2.0.
    spike_height_factor : float
        Larger value → less sensitive to bifurcation.  Default: 3.0.
    return_diagnostics : bool
        If ``True``, return a ``(bool, dict)`` tuple where the dict contains
        all intermediate arrays and thresholds computed during detection.
        Default: ``False`` (returns only the boolean, preserving the original
        API for all existing callers).

    Returns
    -------
    bool or (bool, dict)
        When *return_diagnostics* is ``False`` (default): ``True`` if
        bifurcation is detected, ``False`` otherwise.

        When *return_diagnostics* is ``True``: a ``(is_bifurcated, diag)``
        tuple where *diag* is a dictionary with the following keys:

        * ``I_speed`` — normalised per-step I increments, ``diff(I/I_range)``
        * ``Q_speed`` — normalised per-step Q increments, ``diff(Q/Q_range)``
        * ``freq_steps_hz`` — per-step frequency intervals (Hz)
        * ``arc_length_speed`` — normalised IQ arc-length speed per Hz
        * ``arc_speed_gradient`` — ``diff(arc_length_speed)``
        * ``freq_arc_speed`` — frequency midpoints for *arc_length_speed*
        * ``freq_arc_speed_gradient`` — frequency midpoints for *arc_speed_gradient*
        * ``spike_prominence_threshold`` — computed prominence threshold
        * ``spike_height_threshold`` — computed height threshold
        * ``uppeaks`` — indices into *arc_speed_gradient* of positive spikes
        * ``downpeaks`` — indices into *arc_speed_gradient* of negative spikes
        * ``is_bifurcated`` — same as the leading boolean
    """
    sort_idx = np.argsort(frequencies)
    freq_sorted = np.asarray(frequencies)[sort_idx]
    iq_sorted = np.asarray(iq)[sort_idx]

    iarr = iq_sorted.real
    qarr = iq_sorted.imag

    irange = float(iarr.max() - iarr.min())
    qrange = float(qarr.max() - qarr.min())

    if irange == 0.0 or qrange == 0.0:
        if return_diagnostics:
            empty_diag = {
                'I_speed': np.array([]), 'Q_speed': np.array([]),
                'freq_steps_hz': np.array([]),
                'arc_length_speed': np.array([]), 'arc_speed_gradient': np.array([]),
                'freq_arc_speed': np.array([]), 'freq_arc_speed_gradient': np.array([]),
                'spike_prominence_threshold': 0.0, 'spike_height_threshold': 0.0,
                'uppeaks': np.array([], dtype=int), 'downpeaks': np.array([], dtype=int),
                'is_bifurcated': False,
            }
            return False, empty_diag
        return False  # Degenerate sweep

    idiffnorm = np.diff(iarr / irange)
    qdiffnorm = np.diff(qarr / qrange)
    fdiff = np.diff(freq_sorted)

    if np.any(fdiff == 0):
        if return_diagnostics:
            empty_diag = {
                'I_speed': idiffnorm, 'Q_speed': qdiffnorm,
                'freq_steps_hz': fdiff,
                'arc_length_speed': np.array([]), 'arc_speed_gradient': np.array([]),
                'freq_arc_speed': np.array([]), 'freq_arc_speed_gradient': np.array([]),
                'spike_prominence_threshold': 0.0, 'spike_height_threshold': 0.0,
                'uppeaks': np.array([], dtype=int), 'downpeaks': np.array([], dtype=int),
                'is_bifurcated': False,
            }
            return False, empty_diag
        return False

    dist = np.sqrt(idiffnorm ** 2 + qdiffnorm ** 2) / np.abs(fdiff)
    distdiff = np.diff(dist)

    # Frequency midpoints for the step-level arrays (length N-1) and gradient (length N-2)
    freq_arc_speed = 0.5 * (freq_sorted[:-1] + freq_sorted[1:])
    freq_arc_speed_gradient = 0.5 * (freq_arc_speed[:-1] + freq_arc_speed[1:])

    if len(distdiff) < 2:
        if return_diagnostics:
            diag = {
                'I_speed': idiffnorm, 'Q_speed': qdiffnorm,
                'freq_steps_hz': fdiff,
                'arc_length_speed': dist, 'arc_speed_gradient': distdiff,
                'freq_arc_speed': freq_arc_speed,
                'freq_arc_speed_gradient': freq_arc_speed_gradient,
                'spike_prominence_threshold': 0.0, 'spike_height_threshold': 0.0,
                'uppeaks': np.array([], dtype=int), 'downpeaks': np.array([], dtype=int),
                'is_bifurcated': False,
            }
            return False, diag
        return False

    spike_prominence = (dist.max() - dist.min()) / spike_prominence_factor
    spike_height = spike_height_factor * float(np.std(distdiff))

    uppeaks, _ = find_peaks(distdiff, prominence=spike_prominence, height=spike_height)
    downpeaks, _ = find_peaks(-distdiff, prominence=spike_prominence, height=spike_height)

    is_bifurcated = False
    if len(uppeaks) > 0 and len(downpeaks) > 0:
        if downpeaks[0] == uppeaks[0] + 1:
            is_bifurcated = True

    if return_diagnostics:
        diag = {
            'I_speed':                    idiffnorm,
            'Q_speed':                    qdiffnorm,
            'freq_steps_hz':              fdiff,
            'arc_length_speed':           dist,
            'arc_speed_gradient':         distdiff,
            'freq_arc_speed':             freq_arc_speed,
            'freq_arc_speed_gradient':    freq_arc_speed_gradient,
            'spike_prominence_threshold': spike_prominence,
            'spike_height_threshold':     spike_height,
            'uppeaks':                    uppeaks,
            'downpeaks':                  downpeaks,
            'is_bifurcated':              is_bifurcated,
        }
        return is_bifurcated, diag

    return is_bifurcated


# ── Section C: Multi-amplitude analysis ──────────────────────────────────────

def find_bias_points(
    results_by_detector: Dict[str, Dict],
    res_info_dict: Dict[str, Dict],
    spike_prominence_factor: float = 2.0,
    spike_height_factor: float = 3.0,
    max_deriv_distance_hz: float = 100e3,
    reference_freq_source: Literal["bias_frequency", "fit_fr", "sweep_center"] = "bias_frequency",
    fit_selected_amplitude: bool = True,
) -> Dict[str, Dict]:
    """
    Analyse multi-amplitude multisweep results to find optimal bias points.

    For each resonator code the algorithm:

    1. Sorts the available sweep entries by amplitude (lowest → highest).
    2. Steps through amplitudes calling :func:`detect_bifurcation_derivative`.
       When bifurcation is first detected at amplitude *a*, the previous
       amplitude *a - 1* is selected as the bias amplitude.  If *a* is the
       lowest available amplitude, *a* itself is used.
    3. If no bifurcation is found, the highest available amplitude is selected,
       ensuring every resonator always receives a bias point.
    4. On the selected entry:

       * Calls :func:`find_max_derivative_frequency` on the ``iq_volts`` data
         to obtain the refined ``bias_frequency`` and ``dI_df``, ``dQ_df``
         in V/Hz.
       * Computes ``df_calibration = 1 / (dI_df + j·dQ_df)``  (Hz/V).
       * Optionally runs the nonlinear fitter as a post-selection diagnostic.
         This is skipped when the selected entry is itself bifurcated.

    5. Updates ``res_info_dict[code]`` in-place with all new fields.

    Parameters
    ----------
    results_by_detector : dict
        ``{code: {iteration_index: entry_dict}}`` as stored in
        ``MultisweepPanel.results_by_detector``.  Each *entry_dict* should
        contain ``frequencies``, ``iq_counts``, ``iq_volts``, and
        ``sweep_amplitude``.
    res_info_dict : dict
        Resonator registry ``{code: {bias_frequency, bias_amplitude,
        channel_number}}``.  **Updated in-place** and also returned.
    spike_prominence_factor : float
        Passed to :func:`detect_bifurcation_derivative`.  Default: 2.0.
    spike_height_factor : float
        Passed to :func:`detect_bifurcation_derivative`.  Default: 3.0.
    max_deriv_distance_hz : float
        Passed to :func:`find_max_derivative_frequency`.  Default: 100 kHz.
    reference_freq_source : {"bias_frequency", "fit_fr", "sweep_center"}
        Which frequency to use as the sanity-check reference for
        :func:`find_max_derivative_frequency`:

        * ``"bias_frequency"`` — ``res_info_dict[code]["bias_frequency"]``
          (the sweep-centre frequency set during the multisweep run).
        * ``"fit_fr"`` — the fitted resonance frequency from ``fit_params``
          or ``nonlinear_fit_params`` in the selected entry; falls back to
          ``sweep_center_frequency`` when absent.
        * ``"sweep_center"`` — always uses ``entry["sweep_center_frequency"]``.

    fit_selected_amplitude : bool
        If ``True`` (default), run the nonlinear fitter on the selected
        amplitude's sweep as a diagnostic.  The fit is always skipped when
        the selected entry is itself identified as bifurcated, to avoid
        feeding a pathological sweep to the fitter.

    Returns
    -------
    res_info_dict : dict
        The same dict passed in, updated in-place.  New / updated fields per
        resonator code:

        * ``bias_frequency`` (float) — refined via max-derivative method
        * ``bias_amplitude`` (float) — selected amplitude
        * ``dI_df`` (float) — dI/df at the bias point (V/Hz)
        * ``dQ_df`` (float) — dQ/df at the bias point (V/Hz)
        * ``df_calibration`` (complex) — ``1/(dI_df + j·dQ_df)``  (Hz/V)
        * ``bifurcated_at`` (float or None) — amplitude where bifurcation
          was first detected; ``None`` if not detected
        * ``bias_found`` (bool)
        * ``nonlinear_fit_params`` (dict or None) — diagnostic fit result
        * ``nonlinear_fit_success`` (bool)
    """
    from rfmux.algorithms.measurement.fitting_nonlinear import (
        fit_nonlinear_iq,
        estimate_and_remove_gain,
    )

    for code, iter_dict in results_by_detector.items():

        if code not in res_info_dict:
            warnings.warn(
                f"find_bias_points: code {code!r} not found in res_info_dict — skipping."
            )
            continue

        # --- Collect entries and sort by amplitude (lowest first) ---
        entries = list(iter_dict.values())
        entries_sorted = sorted(
            entries,
            key=lambda e: (e.get('sweep_amplitude') or 0.0),
        )

        bifurcated_at = None
        selected_entry = None
        selected_amplitude = None
        selected_is_bifurcated = False

        # --- Amplitude selection ---
        for i, entry in enumerate(entries_sorted):
            amp = entry.get('sweep_amplitude')
            if amp is None:
                continue

            frequencies = np.asarray(entry.get('frequencies', []))
            iq_counts = entry.get('iq_counts')

            if len(frequencies) < 4 or iq_counts is None:
                continue

            is_bifurcated = detect_bifurcation_derivative(
                frequencies,
                np.asarray(iq_counts),
                spike_prominence_factor=spike_prominence_factor,
                spike_height_factor=spike_height_factor,
            )

            if is_bifurcated:
                bifurcated_at = amp
                if i > 0:
                    # Use the entry one step below the bifurcated one
                    prev = entries_sorted[i - 1]
                    if prev.get('sweep_amplitude') is not None:
                        selected_entry = prev
                        selected_amplitude = prev['sweep_amplitude']
                        selected_is_bifurcated = False
                    else:
                        # Previous entry has no amplitude — use current
                        selected_entry = entry
                        selected_amplitude = amp
                        selected_is_bifurcated = True
                else:
                    # Bifurcation at the very lowest amplitude
                    selected_entry = entry
                    selected_amplitude = amp
                    selected_is_bifurcated = True
                break

        if selected_entry is None:
            # No bifurcation found — always fall back to the highest amplitude
            for entry in reversed(entries_sorted):
                if entry.get('sweep_amplitude') is not None:
                    selected_entry = entry
                    selected_amplitude = entry['sweep_amplitude']
                    selected_is_bifurcated = False
                    break
            if selected_entry is None:
                warnings.warn(
                    f"find_bias_points: {code!r} — no valid sweep entries found."
                )
                res_info_dict[code]['bias_found'] = False
                continue

        # --- Determine reference frequency ---
        if reference_freq_source == "bias_frequency":
            ref_freq = res_info_dict[code].get('bias_frequency')

        elif reference_freq_source == "fit_fr":
            fp  = selected_entry.get('fits', {}).get('skewed', {}).get('fit_params') or {}
            nlp = selected_entry.get('fits', {}).get('nonlinear', {}).get('nonlinear_fit_params') or {}
            ref_freq = fp.get('fr') or nlp.get('fr')
            # Treat the string 'nan' (from old skewed fitter) as missing
            if isinstance(ref_freq, str):
                ref_freq = None
            if ref_freq is None:
                ref_freq = selected_entry.get('sweep_center_frequency')

        else:  # "sweep_center"
            ref_freq = selected_entry.get('sweep_center_frequency')

        # Ultimate fallback
        if ref_freq is None:
            ref_freq = (
                selected_entry.get('bias_frequency')
                or selected_entry.get('sweep_center_frequency')
            )

        if ref_freq is None:
            warnings.warn(
                f"find_bias_points: {code!r} — cannot determine reference frequency."
            )
            res_info_dict[code]['bias_found'] = False
            continue

        # --- Max-derivative bias frequency (using iq_volts for calibration) ---
        frequencies = np.asarray(selected_entry['frequencies'])
        iq_volts = selected_entry.get('iq_volts')

        if iq_volts is not None:
            iq_for_deriv = np.asarray(iq_volts)
        else:
            warnings.warn(
                f"find_bias_points: {code!r} — 'iq_volts' absent; "
                f"falling back to 'iq_counts'. "
                f"df_calibration will not be in correct voltage units."
            )
            iq_for_deriv = np.asarray(selected_entry['iq_counts'])

        # Voltage-calibrated splines (for df_calibration and bias-frequency finding)
        dI_df_sp = dQ_df_sp = None
        try:
            dI_df_sp, dQ_df_sp = compute_iq_derivative_spline(frequencies, iq_for_deriv)
        except Exception:
            pass  # Handled below

        try:
            bias_freq, dI_df, dQ_df = find_max_derivative_frequency(
                frequencies,
                iq_for_deriv,
                reference_freq=float(ref_freq),
                max_distance_hz=max_deriv_distance_hz,
            )
        except Exception as exc:
            warnings.warn(
                f"find_bias_points: {code!r} — max-derivative computation failed "
                f"({exc}); using reference frequency."
            )
            bias_freq = float(ref_freq)
            try:
                if dI_df_sp is None:
                    dI_df_sp, dQ_df_sp = compute_iq_derivative_spline(frequencies, iq_for_deriv)
                dI_df = float(dI_df_sp(bias_freq))
                dQ_df = float(dQ_df_sp(bias_freq))
            except Exception:
                dI_df = float('nan')
                dQ_df = float('nan')

        # --- df_calibration ---
        deriv_complex = dI_df + 1j * dQ_df
        if abs(deriv_complex) > 0:
            df_calibration = 1.0 / deriv_complex
        else:
            warnings.warn(
                f"find_bias_points: {code!r} — zero IQ derivative at bias point; "
                f"df_calibration undefined."
            )
            df_calibration = complex(float('nan'), float('nan'))

        # --- Optional nonlinear diagnostic fit ---
        nlfit_params = None
        nlfit_success = False

        if fit_selected_amplitude and not selected_is_bifurcated:
            try:
                iq_counts_arr = np.asarray(selected_entry['iq_counts'])
                iq_corr, _, _ = estimate_and_remove_gain(frequencies, iq_counts_arr)
                _, popt, _, residual = fit_nonlinear_iq(frequencies, iq_corr)
                param_names = ['fr', 'Qr', 'amp', 'phi', 'a', 'i0', 'q0']
                nlfit_params = dict(zip(param_names, popt))
                # Derived Qc / Qi
                Qr = popt[1]
                amp_param = popt[2]
                if 0 < amp_param < 1:
                    Qc = Qr / amp_param
                    Qi_inv = 1.0 / Qr - 1.0 / Qc
                    if Qi_inv > 0:
                        nlfit_params['Qc'] = Qc
                        nlfit_params['Qi'] = 1.0 / Qi_inv
                nlfit_success = bool(residual < 0.1)
            except Exception as exc:
                warnings.warn(
                    f"find_bias_points: {code!r} — nonlinear diagnostic fit failed: {exc}"
                )

        # --- Build bias_finding sub-dict and attach to selected_entry ---
        # Re-run detect_bifurcation_derivative on the selected entry with
        # return_diagnostics=True to capture all intermediate arrays.
        iq_counts_sel = selected_entry.get('iq_counts')
        bif_diag: dict = {}
        if iq_counts_sel is not None:
            try:
                _, bif_diag = detect_bifurcation_derivative(
                    frequencies,
                    np.asarray(iq_counts_sel),
                    spike_prominence_factor=spike_prominence_factor,
                    spike_height_factor=spike_height_factor,
                    return_diagnostics=True,
                )
            except Exception as exc:
                warnings.warn(
                    f"find_bias_points: {code!r} — bifurcation diagnostics failed: {exc}"
                )

        # Normalized splines: fit CubicSpline to I/I_range and Q/Q_range so that
        # the derivative is in units of 1/Hz — matching I_speed / Q_speed scale.
        I_speed_sp = Q_speed_sp = None
        sort_idx = np.argsort(frequencies)
        freq_sorted = frequencies[sort_idx]
        iq_sorted = iq_for_deriv[sort_idx]
        iarr = iq_sorted.real
        qarr = iq_sorted.imag
        irange = float(iarr.max() - iarr.min()) if len(iarr) > 0 else 0.0
        qrange = float(qarr.max() - qarr.min()) if len(qarr) > 0 else 0.0
        try:
            if irange > 0 and qrange > 0:
                I_speed_sp = CubicSpline(freq_sorted, iarr / irange).derivative()
                Q_speed_sp = CubicSpline(freq_sorted, qarr / qrange).derivative()
        except Exception as exc:
            warnings.warn(
                f"find_bias_points: {code!r} — normalized spline fitting failed: {exc}"
            )

        selected_entry['bias_finding'] = {
            # Sorted frequency axis (ascending, required by splines)
            'frequencies_sorted':       freq_sorted,

            # Voltage-calibrated derivative splines (V/Hz — for df_calibration physics)
            'dI_df_spline':             dI_df_sp,
            'dQ_df_spline':             dQ_df_sp,
            'dI_df_at_bias':            dI_df,
            'dQ_df_at_bias':            dQ_df,

            # Normalized derivative splines (1/Hz — matching I_speed/Q_speed scale, for display)
            'I_speed_spline':           I_speed_sp,
            'Q_speed_spline':           Q_speed_sp,

            # Discrete arrays from the bifurcation algorithm
            'I_speed':                  bif_diag.get('I_speed', np.array([])),
            'Q_speed':                  bif_diag.get('Q_speed', np.array([])),
            'freq_steps_hz':            bif_diag.get('freq_steps_hz', np.array([])),
            'arc_length_speed':         bif_diag.get('arc_length_speed', np.array([])),
            'arc_speed_gradient':       bif_diag.get('arc_speed_gradient', np.array([])),
            'freq_arc_speed':           bif_diag.get('freq_arc_speed', np.array([])),
            'freq_arc_speed_gradient':  bif_diag.get('freq_arc_speed_gradient', np.array([])),

            # Bias point
            'bias_frequency':           bias_freq,
            'bias_amplitude':           selected_amplitude,

            # Nested sub-dicts
            'diagnostics': {
                'spike_prominence_threshold': bif_diag.get('spike_prominence_threshold', float('nan')),
                'spike_height_threshold':     bif_diag.get('spike_height_threshold', float('nan')),
                'uppeaks':                    bif_diag.get('uppeaks', np.array([], dtype=int)),
                'downpeaks':                  bif_diag.get('downpeaks', np.array([], dtype=int)),
                'is_bifurcated':              selected_is_bifurcated,
                'bifurcated_at':              bifurcated_at,
            },
            'settings': {
                'spike_prominence_factor':    spike_prominence_factor,
                'spike_height_factor':        spike_height_factor,
                'max_deriv_distance_hz':      max_deriv_distance_hz,
                'reference_freq_source':      reference_freq_source,
            },
        }

        # --- Update res_info_dict in-place ---
        res_info_dict[code].update({
            'bias_frequency':        bias_freq,
            'bias_amplitude':        selected_amplitude,
            'dI_df':                 dI_df,
            'dQ_df':                 dQ_df,
            'df_calibration':        df_calibration,
            'bifurcated_at':         bifurcated_at,
            'bias_found':            True,
            'nonlinear_fit_params':  nlfit_params,
            'nonlinear_fit_success': nlfit_success,
        })

        print(
            f"[find_bias] {code}: bias_freq={bias_freq * 1e-6:.4f} MHz, "
            f"amp={selected_amplitude:.4f}"
            + (f", bifurcated_at={bifurcated_at:.4f}" if bifurcated_at else "")
            + (f", nl_fit={'OK' if nlfit_success else 'failed'}"
               if fit_selected_amplitude and not selected_is_bifurcated else "")
        )

    return res_info_dict


# ── Section D: Hardware programming ──────────────────────────────────────────

# PFB frequency bin spacing in Hz (= 625 MHz / 2^21)
_PFB_BIN_HZ: float = 298.0232238769531


async def apply_bias(
    crs,
    module: int,
    res_info_dict: Dict[str, Dict],
    progress_callback: Optional[Callable[[int, float], None]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Programme the CRS hardware channels with bias conditions from *res_info_dict*.

    Only resonator codes where ``res_info_dict[code]["bias_found"] == True``
    are programmed.  All tone settings (frequency, amplitude, phase=0) are
    committed in a single ``tuber_context`` batch for efficiency.

    The bias frequency is quantised to the nearest PFB frequency bin before
    programming, and the channel frequency is computed relative to the current
    NCO frequency.

    Parameters
    ----------
    crs :
        The CRS hardware object.
    module : int
        Target module number.
    res_info_dict : dict
        Resonator registry as updated by :func:`find_bias_points`.
    progress_callback : callable, optional
        Called as ``progress_callback(module, percent)`` when complete.

    Returns
    -------
    apply_report : dict
        ``{code: dict}`` where each value contains:

        * ``apply_successful`` (bool)
        * ``channel_number`` (int)
        * ``bias_frequency`` (float) — the (unquantised) refined bias frequency
        * ``quantized_bias_frequency`` (float) — the frequency actually programmed
        * ``channel_frequency`` (float) — frequency relative to NCO
        * ``bias_amplitude`` (float)
    """
    nco_freq = await crs.get_nco_frequency(module=module)

    apply_report: Dict[str, Dict[str, Any]] = {}

    codes_to_apply = [
        code for code, info in res_info_dict.items()
        if info.get('bias_found', False)
    ]

    if not codes_to_apply:
        warnings.warn("apply_bias: no codes with bias_found=True — nothing to programme.")
        return apply_report

    async with crs.tuber_context() as ctx:
        for code in codes_to_apply:
            info = res_info_dict[code]
            channel = int(info['channel_number'])
            bias_freq = float(info['bias_frequency'])
            amplitude = float(info['bias_amplitude'])

            # Quantise to nearest PFB bin
            quantized_freq = round(bias_freq / _PFB_BIN_HZ) * _PFB_BIN_HZ
            channel_freq = quantized_freq - nco_freq

            ctx.set_frequency(channel_freq, channel=channel, module=module)
            ctx.set_amplitude(amplitude, channel=channel, module=module)
            ctx.set_phase(
                0.0,
                units=crs.UNITS.DEGREES,
                target=crs.TARGET.ADC,
                channel=channel,
                module=module,
            )

            apply_report[code] = {
                'apply_successful':         True,
                'channel_number':           channel,
                'bias_frequency':           bias_freq,
                'quantized_bias_frequency': quantized_freq,
                'channel_frequency':        channel_freq,
                'bias_amplitude':           amplitude,
            }

        await ctx()

    if progress_callback is not None:
        progress_callback(module, 100.0)

    print(
        f"[apply_bias] Module {module}: programmed {len(apply_report)} / "
        f"{len(res_info_dict)} resonators."
    )

    return apply_report
