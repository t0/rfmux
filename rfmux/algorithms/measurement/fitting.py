"""
Analysis functions for rfmux, including resonance fitting and IQ circle manipulation.
"""

import numpy as np
from scipy.optimize import curve_fit
import warnings
from typing import Dict, Tuple, List, Optional, Union
import pickle # Moved import pickle to top level

def s21_skewed(f, f0, Qr, Qcre, Qcim, A):
    """
    Skewed Lorentzian model for S21 magnitude.
    Based on hidfmux implementation.

    Uses soft penalty for near-unphysical parameters instead of hard cutoff.

    Args:
        f (np.ndarray): Frequency array (Hz).
        f0 (float): Resonance frequency (Hz).
        Qr (float): Resonator quality factor.
        Qcre (float): Real part of complex coupling quality factor Qc.
        Qcim (float): Imaginary part of complex coupling quality factor Qc.
        A (float): Amplitude scaling factor.

    Returns:
        np.ndarray: Modelled S21 magnitude with soft penalty for unphysical parameters.
    """
    # Basic parameter validation
    if Qcre <= 1e-9 or Qr <= 1e-9 or abs(f0) < 1e-12:
        return np.full_like(f, np.inf)
    
    Qe = Qcre + 1j * Qcim
    Qc_eff = abs(Qe)**2 / Qcre
    
    # Calculate soft penalty factor for near-unphysical parameters
    # Allow parameters slightly below the hard boundary
    penalty_factor = 1.0
    if Qc_eff < Qr * 1.05:  # Within 5% of boundary
        # Smooth penalty that increases as we approach unphysical regime
        ratio = Qc_eff / Qr
        if ratio < 0.5:  # Far into unphysical regime
            return np.full_like(f, np.inf)
        elif ratio < 1.0:  # Unphysical but not extreme
            # Quadratic penalty that grows smoothly
            penalty_factor = 1 + 100 * (1 - ratio)**2
        else:  # Near boundary but physical (1.0 <= ratio < 1.05)
            # Mild penalty to discourage but not prohibit
            penalty_factor = 1 + 5 * (1.05 - ratio)**2
    
    # Calculate S21
    x = (f - f0) / f0
    with np.errstate(divide='ignore', invalid='ignore'):
        s21_complex = A * (1 - (Qr / Qe) / (1 + 2j * Qr * x))
    
    # Apply penalty to magnitude
    magnitude = np.abs(s21_complex) * penalty_factor
    
    # Handle any numerical issues
    magnitude[~np.isfinite(magnitude)] = np.inf
    
    return magnitude

def fit_skewed(freq, s21_iq, approxQr=1e4, normalize=True, fr_lim=None):
    """
    Fits the s21_skewed model to complex S21 data using scipy.optimize.curve_fit.
    Based on hidfmux implementation.

    Args:
        freq (np.ndarray): Frequency array (Hz).
        s21_iq (np.ndarray): Complex S21 array.
        approxQr (float): Initial guess for Qr. Defaults to 1e4.
        normalize (bool): Normalize s21_iq to its last point before fitting. Defaults to True.
        fr_lim (float, optional): Fit fr only within +/- fr_lim Hz of the center frequency. Defaults to None (use full range).

    Returns:
        dict: Dictionary of fitted parameters ('fr', 'Qr', 'Qc', 'Qi', 'Qcre', 'Qcim', 'A')
              and their errors ('_err'), or 'nan' string values on failure.
    """
    param_names = ['fr', 'Qr', 'Qc', 'Qi', 'Qcre', 'Qcim', 'A']
    fit_dict = {}
    bad_fit_flag = False

    freq = np.asarray(freq)
    s21_iq = np.asarray(s21_iq)

    if len(freq) < 5: # Need sufficient points for fitting
        warnings.warn("Skewed fit failed: Not enough data points.")
        bad_fit_flag = True

    # Normalize S21 data if requested and possible
    s21_iq_norm = s21_iq
    if normalize:
        if len(s21_iq) > 0 and np.abs(s21_iq[-1]) > 1e-15:
            s21_iq_norm = s21_iq / s21_iq[-1]
        else:
            # Cannot normalize if last point is zero or data is empty
            warnings.warn("Could not normalize S21 data (last point near zero or data empty). Using unnormalized data for fit.")
            # Proceed with unnormalized data, but the 'A' parameter might be less meaningful

    s21_mag = np.abs(s21_iq_norm)

    if not bad_fit_flag:
        # Determine frequency bounds for fr fit
        f_center = freq[len(freq)//2]
        if fr_lim is not None:
            fr_lbound = max(min(freq), f_center - fr_lim)
            fr_ubound = min(max(freq), f_center + fr_lim)
        else:
            fr_lbound = min(freq)
            fr_ubound = max(freq)

        # Initial guess for fr is the minimum magnitude point within bounds
        search_indices = np.where((freq >= fr_lbound) & (freq <= fr_ubound))[0]
        if len(search_indices) > 0:
             fr_guess_idx = search_indices[np.argmin(s21_mag[search_indices])]
             fr_guess = freq[fr_guess_idx]
        else:
             fr_guess = f_center # Fallback if bounds are too narrow or no points within bounds

        # Initial guesses and bounds for curve_fit
        # p0 = [fr, Qr, Qcre, Qcim, A]
        # Use Qcre = 1.5 * approxQr to ensure initial Qc > Qr
        init = [fr_guess, approxQr, 1.5 * approxQr, 0., np.mean(s21_mag[search_indices]) if len(search_indices) > 0 else 1.0]
        # Bounds: ([fr_low, Qr_low, Qcre_low, Qcim_low, A_low], [fr_high, Qr_high, Qcre_high, Qcim_high, A_high])
        # Ensure Qcre lower bound > Qr to help maintain physical validity
        bounds = ([fr_lbound, 1e2, 1.5e2, -np.inf, 0], [fr_ubound, 1e9, 1e9, np.inf, np.inf])

        try:
            # Fit the magnitude data
            s21fitp, s21cov = curve_fit(s21_skewed, freq, s21_mag, p0=init, bounds=bounds, maxfev=5000) # Increased maxfev
            # Check if covariance calculation was successful
            if not np.all(np.isfinite(s21cov)):
                 raise ValueError("Covariance matrix calculation failed (contains inf/nan).")
            errs = np.sqrt(np.diag(s21cov)) # This can fail if cov is not positive definite

            f0, Qr_fit = s21fitp[0:2]
            Qe_re, Qe_im = s21fitp[2:4]
            A_fit = s21fitp[4]
            Qe = Qe_re + 1j * Qe_im

            # Calculate derived Q values, handle potential division by zero or invalid results
            # Ensure denominators are not zero and results are physical
            if Qe_re > 1e-9 and Qr_fit > 1e-9 and abs(Qe)**2 / Qe_re >= Qr_fit: # Check physical validity
                 Qc_fit = abs(Qe)**2 / Qe_re
                 # Use np.errstate to prevent warnings/errors for 1/inf or 1/0
                 with np.errstate(divide='ignore'):
                      inv_Qr = 1.0 / Qr_fit
                      inv_Qc = 1.0 / Qc_fit
                      inv_Qi = inv_Qr - inv_Qc
                      # Check if Qi would be negative or zero
                      if inv_Qi <= 1e-15: # Allow for small numerical errors near Qi=inf
                           Qi_fit = np.inf
                      else:
                           Qi_fit = 1.0 / inv_Qi

                 if not np.isfinite(Qi_fit): Qi_fit = np.inf # Handle infinite Qi explicitly
            else: # Unphysical result from fit
                 Qc_fit = np.nan
                 Qi_fit = np.nan
                 bad_fit_flag = True # Mark as bad if derived Qs are unphysical

            param_vals = [f0, Qr_fit, Qc_fit, Qi_fit, Qe_re, Qe_im, A_fit]

            # Calculate errors, handle potential issues with derived values
            errf0, errQr, errQere, errQeim, errA = errs[[0, 1, 2, 3, 4]]
            try:
                 # Error propagation for Qc, Qi is complex and sensitive.
                 # Providing NaN as a placeholder, as simple propagation can be misleading.
                 # Proper error requires Jacobian/delta method or MC simulation.
                 errQc = np.nan
                 errQi = np.nan
            except Exception:
                 errQc, errQi = np.nan, np.nan

            param_errs = [errf0, errQr, errQc, errQi, errQere, errQeim, errA]

            # Final check for non-finite results in primary fitted parameters
            if not np.all(np.isfinite(s21fitp)): bad_fit_flag = True
            # Check derived Qs as well
            if not np.isfinite(Qc_fit) or not np.isfinite(Qi_fit):
                 # Allow infinite Qi, but NaN Qc or NaN Qi (other than inf) is bad
                 if not (np.isinf(Qi_fit) and np.isfinite(Qc_fit)):
                      bad_fit_flag = True


        except (RuntimeError, ValueError) as e:
            # Catch fit failures (convergence, bounds, etc.) or cov issues
            warnings.warn(f"Skewed fit failed for resonance near {fr_guess*1e-6:.3f} MHz: {e}")
            bad_fit_flag = True
        except Exception as e: # Catch other potential errors like linalg errors in curve_fit
            warnings.warn(f"Skewed fit failed near {fr_guess*1e-6:.3f} MHz with unexpected error: {e}")
            bad_fit_flag = True

    # Populate dictionary with results or 'nan' strings
    if bad_fit_flag:
        for name in param_names:
            fit_dict[name] = 'nan'
            fit_dict[f'{name}_err'] = 'nan'
    else:
        for i, name in enumerate(param_names):
            # Store finite numbers, replace inf with 'inf' string for consistency if needed, nan with 'nan'
            val = param_vals[i]
            err = param_errs[i]
            fit_dict[name] = val if np.isfinite(val) else ('inf' if np.isinf(val) else 'nan')
            fit_dict[f'{name}_err'] = err if np.isfinite(err) else 'nan' # Errors usually shouldn't be inf

    return fit_dict


# Circle fitting using Pratt's method (hyper-LMS)
# Adapted from various online sources, e.g., based on Chernov's implementation notes
def circle_fit_pratt(x, y):
    """
    Fits a circle to a set of points using Pratt's method (hyper-LMS).

    Args:
        x (np.ndarray): Real components (I).
        y (np.ndarray): Imaginary components (Q).

    Returns:
        tuple: (xc, yc, R) - Center coordinates and radius, or (None, None, None) on failure.
    """
    n = len(x)
    if n < 3:
        warnings.warn("Circle fit failed: Need at least 3 points.")
        return None, None, None

    # Calculate moments
    x = np.asarray(x)
    y = np.asarray(y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_c = x - x_mean
    y_c = y - y_mean

    Suu = np.sum(x_c**2)
    Svv = np.sum(y_c**2)
    Suv = np.sum(x_c * y_c)
    Suuu = np.sum(x_c**3)
    Svvv = np.sum(y_c**3)
    Suuv = np.sum(x_c**2 * y_c)
    Suvv = np.sum(x_c * y_c**2)

    # Form the linear system matrix B and vector C
    B = np.array([
        [Suu, Suv],
        [Suv, Svv]
    ])
    C = np.array([
        0.5 * (Suuu + Suvv),
        0.5 * (Svvv + Suuv)
    ])

    # Solve B * [xc, yc]^T = C for the center relative to the mean
    try:
        # Use pseudo-inverse for robustness if B is near singular
        B_inv = np.linalg.pinv(B)
        xc_rel, yc_rel = B_inv @ C
    except np.linalg.LinAlgError:
        warnings.warn("Circle fit failed: Linear system solution failed.")
        return None, None, None

    # Calculate absolute center coordinates
    xc = xc_rel + x_mean
    yc = yc_rel + y_mean

    # Calculate radius
    # R^2 = xc_rel^2 + yc_rel^2 + (Suu + Svv)/n
    R_sq = xc_rel**2 + yc_rel**2 + (Suu + Svv) / n
    if R_sq < 0:
        # This can happen with noisy data or poor fits
        warnings.warn("Circle fit failed: Calculated radius squared is negative.")
        return None, None, None
    R = np.sqrt(R_sq)

    # Check for NaN results which indicate failure
    if not (np.isfinite(xc) and np.isfinite(yc) and np.isfinite(R)):
         warnings.warn("Circle fit failed: Result contains non-finite values.")
         return None, None, None

    return xc, yc, R


def center_resonance_iq_circle(s21_iq):
    """
    Centers the resonance loop in the IQ plane by fitting a circle
    and subtracting its center.

    Args:
        s21_iq (np.ndarray): Complex S21 array.

    Returns:
        np.ndarray: Centered complex S21 array, or the original array if circle fitting fails.
    """
    if len(s21_iq) < 3: # Need at least 3 points to fit a circle
         warnings.warn("Cannot center circle: less than 3 data points.")
         return s21_iq # Return original data

    xc, yc, R = circle_fit_pratt(s21_iq.real, s21_iq.imag)

    if xc is not None and yc is not None:
        # Fit successful, subtract the center
        iq_centered = s21_iq - (xc + 1j * yc)
        return iq_centered
    else:
        # Circle fit failed, return original data
        warnings.warn("Circle centering failed; returning original IQ data.")
        return s21_iq


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
    Finds resonances (dips) in network analysis data.

    Parameters
    ----------
    frequencies : np.ndarray
        Sorted frequency points (Hz).
    iq_complex : np.ndarray
        Complex I/Q data corresponding to 'frequencies'.
    expected_resonances : int | None, optional
        Optional target number of resonances for resilience logic, by default None.
    min_dip_depth_db : float, optional
        Minimum depth (prominence) a dip must have in dB to be considered a peak.
        Corresponds to `prominence` in `scipy.signal.find_peaks`, by default 1.0.
        Note: For shallow resonances (e.g., overcoupled or low Q), you may need
        to reduce this to 0.3-0.5 dB.
    min_Q : float, optional
        Minimum estimated quality factor. Used to calculate the maximum allowed width
        of a resonance feature, by default 1e4.
        Note: For broad resonances, you may need to reduce this to 1000-5000.
    max_Q : float, optional
        Maximum estimated quality factor. Used to calculate the minimum allowed width
        of a resonance feature, by default 1e7.
    min_resonance_separation_hz : float, optional
        Minimum frequency separation between identified peaks. Corresponds to `distance`
        in `scipy.signal.find_peaks`, by default 100e3.
    data_exponent : float, optional
        Exponent applied to the magnitude data before dB conversion to potentially
        enhance peak visibility, by default 2.0.
    module_identifier : str | int | None, optional
        An identifier for the module/data source, used for more informative warnings.

    Returns
    -------
    dict
        A dictionary containing:
        - 'resonance_frequencies': list[float] - Identified resonance peak frequencies (Hz).
        - 'resonances_details': list[dict] - Detailed information for each resonance.
                                             Each dict contains: 'frequency', 'prominence_db',
                                             'width_hz', 'q_estimated'.
        Returns empty lists if no resonances are found or an error occurs.
        
    Notes
    -----
    The default parameters are optimized for typical high-Q superconducting resonators
    with clear dips of several dB. For other resonator types, parameter adjustment
    may be necessary.
    """
    results = {
        'resonance_frequencies': [],
        'resonances_details': []
    }
    if len(frequencies) < 2 or len(iq_complex) < 2 or len(frequencies) != len(iq_complex):
        warnings.warn(f"Resonance finding skipped for {module_identifier or 'data'}: Insufficient or mismatched data points.")
        return results

    try:
        # a. Preprocessing:
        magnitude = np.abs(iq_complex)
        if np.any(magnitude <= 1e-18):
            min_positive = np.min(magnitude[magnitude > 1e-18]) if np.any(magnitude > 1e-18) else 1e-9
            magnitude = np.maximum(magnitude, min_positive * 0.1)

        ref_mag = np.median(magnitude[-min(10, len(magnitude)):])
        if ref_mag <= 1e-18: ref_mag = 1.0

        shifted_mag = magnitude**data_exponent
        ref_shifted = ref_mag**data_exponent
        mag_db = 20 * np.log10(shifted_mag / ref_shifted)

        # b. Calculate find_peaks Parameters:
        pt_spacing = np.mean(np.diff(frequencies))
        if not np.isfinite(pt_spacing) or pt_spacing <= 0:
            pt_spacing = (frequencies[-1] - frequencies[0]) / (len(frequencies) - 1) if len(frequencies) > 1 else 1e6

        median_freq = np.median(frequencies)
        max_width_hz = median_freq / min_Q if min_Q > 0 else np.inf
        min_width_hz = median_freq / max_Q if max_Q > 0 else 0

        max_width_pts = np.ceil(max_width_hz / pt_spacing) if pt_spacing > 0 and np.isfinite(max_width_hz) else len(frequencies)
        min_width_pts = np.ceil(min_width_hz / pt_spacing) if pt_spacing > 0 and np.isfinite(min_width_hz) else 0
        max_width_pts = max(1, int(max_width_pts))
        min_width_pts = max(0, int(min(min_width_pts, max_width_pts)))


        min_separation_pts = np.ceil(min_resonance_separation_hz / pt_spacing) if pt_spacing > 0 else 1
        min_separation_pts = max(1, int(min_separation_pts))

        # c. Call find_peaks:
        # Ensure find_peaks is imported if not already at the top of the file
        from scipy import signal as sp_signal # Use an alias to avoid conflict if 'signal' is used elsewhere
        peaks, properties = sp_signal.find_peaks(
            -mag_db,
            prominence=min_dip_depth_db,
            width=(min_width_pts, max_width_pts),
            distance=min_separation_pts
        )

        # d. Apply Resilience Logic:
        if expected_resonances is not None and len(peaks) != expected_resonances:
            if len(peaks) > expected_resonances:
                prominences = properties['prominences']
                sorted_prominence_indices = np.argsort(prominences)[::-1]
                top_indices_unsorted = sorted_prominence_indices[:expected_resonances]
                top_indices_sorted = np.sort(top_indices_unsorted)
                peaks = peaks[top_indices_sorted]
                properties = {key: val[top_indices_sorted] for key, val in properties.items()}
                warnings.warn(f"Found {len(prominences)} peaks for {module_identifier or 'data'}, selected top {expected_resonances} most prominent.")
            else:
                warnings.warn(f"Found {len(peaks)} peaks for {module_identifier or 'data'}, expected {expected_resonances}. Consider adjusting parameters.")

        # e. Format Results:
        found_frequencies = frequencies[peaks]
        results['resonance_frequencies'] = found_frequencies.tolist()

        resonances_details_list = []
        for i, peak_idx in enumerate(peaks):
            freq_hz = found_frequencies[i]
            width_pts_val = properties['widths'][i]
            width_hz_val = width_pts_val * pt_spacing
            q_est_val = freq_hz / width_hz_val if width_hz_val > 0 else np.inf
            resonances_details_list.append({
                'frequency': freq_hz,
                'prominence_db': properties['prominences'][i],
                'width_hz': width_hz_val,
                'q_estimated': q_est_val
            })
        results['resonances_details'] = resonances_details_list

    except Exception as e:
        warnings.warn(f"Resonance finding failed for {module_identifier or 'data'}: {type(e).__name__} - {e}. Returning empty lists.")
        results['resonance_frequencies'] = []
        results['resonances_details'] = []

    return results


def fit_skewed_multisweep(
    multisweep_data: dict | list[dict],
    approx_Q_for_fit: float = 1e4,
    fit_resonances: bool = True,
    center_iq_circle: bool = True,
    normalize_fit: bool = True,
    fr_lim_fit: float | None = None
):
    """
    Processes the output of the multisweep measurement function to add fitting
    and IQ centering results.

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
    # Handle multi-module case (list of dictionaries)
    if isinstance(multisweep_data, list):
        return [fit_skewed_multisweep(
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
        bias_freq = resonance_data.get('bias_frequency', original_cf)
        
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
                # Use the bias frequency for fitting constraints
                fit_center_freq = bias_freq if bias_freq is not None else original_cf
                
                # Set fr_lim based on the sweep span if not provided and we have frequency data
                if fr_lim_fit is None and len(frequencies) > 1:
                    freq_span = frequencies[-1] - frequencies[0]
                    # Use 75% of the sweep span as the fitting limit
                    auto_fr_lim = abs(freq_span) * 0.375
                else:
                    auto_fr_lim = fr_lim_fit

                fit_params_result = fit_skewed(
                    frequencies, iq_complex,
                    approxQr=approx_Q_for_fit,
                    normalize=normalize_fit,
                    fr_lim=auto_fr_lim
                )
                resonance_data['fit_params'] = fit_params_result
                
                # Log successful fit for debugging
                if fit_params_result.get('fr') != 'nan':
                    fitted_freq = fit_params_result.get('fr', 'nan')
                    fitted_Qr = fit_params_result.get('Qr', 'nan')
                    # if fitted_freq != 'nan' and fitted_Qr != 'nan':
                    #     print(f"Fitted resonance: original_cf={original_cf*1e-6:.3f} MHz, "
                    #           f"fitted_fr={fitted_freq*1e-6:.3f} MHz, Qr={fitted_Qr:.0f}")
                        
            except Exception as e:
                warnings.warn(f"Fitting failed for resonance at {original_cf*1e-6:.3f} MHz during post-processing: {e}")
                # resonance_data['fit_params'] remains None

        if center_iq_circle:
            try:
                # Always center the current iq_complex data
                iq_centered_result = center_resonance_iq_circle(iq_complex)
                resonance_data['iq_centered'] = iq_centered_result
                
                # Provide some feedback on centering
                if iq_centered_result is not None and len(iq_centered_result) > 0:
                    # Calculate how much the centering moved the data
                    original_mean = np.mean(iq_complex)
                    centered_mean = np.mean(iq_centered_result)
                    shift_magnitude = abs(original_mean - centered_mean)
                    # if shift_magnitude > 1e-6:  # Only log if significant shift
                    #     print(f"IQ centering applied to {original_cf*1e-6:.3f} MHz: "
                    #           f"shifted by {shift_magnitude:.6f} units")
                        
            except Exception as e:
                warnings.warn(f"IQ centering failed for resonance at {original_cf*1e-6:.3f} MHz during post-processing: {e}")
                # resonance_data['iq_centered'] remains None
                
    return multisweep_data


# --- Testing Functions ---

def generate_test_resonator_skewed(
    fr: float = 100e6,
    Qr: float = 1e4,
    Qc: float = 2e4,
    phi: float = 0.1,
    n_points: int = 201,
    span_factor: float = 6.0,
    noise_level: float = 0.01,
    gain_mag: float = 0.8,
    gain_phase: float = 0.2
):
    """
    Generate synthetic linear resonator data for testing skewed Lorentzian fitting.
    
    Parameters
    ----------
    fr : float
        Resonance frequency in Hz
    Qr : float 
        Resonator quality factor (loaded Q)
    Qc : float
        Coupling quality factor  
    phi : float
        Impedance mismatch phase
    n_points : int
        Number of frequency points
    span_factor : float
        Frequency span as multiple of fr/Qr
    noise_level : float
        Fractional noise level
    gain_mag : float
        Overall gain magnitude
    gain_phase : float
        Overall gain phase in radians
        
    Returns
    -------
    frequencies : np.ndarray
        Frequency array
    iq_data : np.ndarray
        Complex S21 data with gain and noise
    true_params : dict
        True parameter values
    """
    # Generate frequency array
    span = span_factor * fr / Qr
    frequencies = np.linspace(fr - span/2, fr + span/2, n_points)
    
    # Calculate derived parameters
    Qcre = Qc * np.cos(phi)
    Qcim = Qc * np.sin(phi)
    amp = gain_mag
    
    # For synthetic data, use a simple model that generates physically valid data
    # without the constraints of the s21_skewed function
    x = (frequencies - fr) / fr
    
    # Generate complex S21 directly using standard resonator model
    # S21 = A * (1 - (Qr/Qc) * exp(j*phi) / (1 + 2j*Qr*x))
    Qc_complex = Qcre + 1j * Qcim
    resonator_response = 1 - (Qr / Qc_complex) / (1 + 2j * Qr * x)
    
    # Apply gain
    s21_ideal = amp * resonator_response * np.exp(1j * gain_phase)
    
    # Add noise
    noise_real = np.random.normal(0, noise_level * gain_mag, n_points)
    noise_imag = np.random.normal(0, noise_level * gain_mag, n_points)
    iq_data = s21_ideal + noise_real + 1j * noise_imag
    
    # Calculate Qi from Qr and Qc
    Qi = 1 / (1/Qr - 1/Qc)
    
    true_params = {
        'fr': fr,
        'Qr': Qr,
        'Qc': Qc,
        'Qi': Qi,
        'Qcre': Qcre,
        'Qcim': Qcim,
        'A': amp,
        'phi': phi,
        'gain_mag': gain_mag,
        'gain_phase': gain_phase
    }
    
    return frequencies, iq_data, true_params


def test_fit_skewed():
    """Test the skewed Lorentzian fitting function."""
    print("\nTesting skewed Lorentzian fitting...")
    print("=" * 50)
    
    # Test 1: High Q resonator
    print("\nTest 1: High Q resonator")
    f, z, true_params = generate_test_resonator_skewed(
        fr=150e6, Qr=10000, Qc=20000, phi=0.05,
        noise_level=0.005, gain_mag=0.9
    )
    
    # Fit the data
    fit_result = fit_skewed(f, z, approxQr=1e4, normalize=True)
    
    print(f"\nTrue vs Fitted parameters:")
    fr_fit = fit_result.get('fr', 'nan')
    Qr_fit = fit_result.get('Qr', 'nan')
    Qc_fit = fit_result.get('Qc', 'nan')
    Qi_fit = fit_result.get('Qi', 'nan')
    
    fr_fit_str = f"{fr_fit*1e-6:.3f} MHz" if isinstance(fr_fit, (int, float)) else str(fr_fit)
    Qr_fit_str = f"{Qr_fit:.0f}" if isinstance(Qr_fit, (int, float)) else str(Qr_fit)
    Qc_fit_str = f"{Qc_fit:.0f}" if isinstance(Qc_fit, (int, float)) else str(Qc_fit)
    Qi_fit_str = f"{Qi_fit:.0f}" if isinstance(Qi_fit, (int, float)) else str(Qi_fit)
    
    print(f"  fr: {true_params['fr']*1e-6:.3f} MHz vs {fr_fit_str}")
    print(f"  Qr: {true_params['Qr']:.0f} vs {Qr_fit_str}")
    print(f"  Qc: {true_params['Qc']:.0f} vs {Qc_fit_str}")
    print(f"  Qi: {true_params['Qi']:.0f} vs {Qi_fit_str}")
    
    # Test 2: Low Q resonator
    print("\n" + "="*50)
    print("\nTest 2: Low Q resonator")
    f, z, true_params = generate_test_resonator_skewed(
        fr=200e6, Qr=1000, Qc=2000, phi=-0.1,
        noise_level=0.01, gain_mag=1.1
    )
    
    fit_result = fit_skewed(f, z, approxQr=1e3, normalize=True)
    
    print(f"\nTrue vs Fitted parameters:")
    fr_fit = fit_result.get('fr', 'nan')
    Qr_fit = fit_result.get('Qr', 'nan')
    Qc_fit = fit_result.get('Qc', 'nan')
    
    fr_fit_str = f"{fr_fit*1e-6:.3f} MHz" if isinstance(fr_fit, (int, float)) else str(fr_fit)
    Qr_fit_str = f"{Qr_fit:.0f}" if isinstance(Qr_fit, (int, float)) else str(Qr_fit)
    Qc_fit_str = f"{Qc_fit:.0f}" if isinstance(Qc_fit, (int, float)) else str(Qc_fit)
    
    print(f"  fr: {true_params['fr']*1e-6:.3f} MHz vs {fr_fit_str}")
    print(f"  Qr: {true_params['Qr']:.0f} vs {Qr_fit_str}")
    print(f"  Qc: {true_params['Qc']:.0f} vs {Qc_fit_str}")


def test_circle_fitting():
    """Test circle fitting and IQ centering functions."""
    print("\nTesting circle fitting...")
    print("=" * 50)
    
    # Generate a circle in IQ plane with offset
    n_points = 100
    theta = np.linspace(0, 2*np.pi, n_points)
    radius = 0.3
    center_x = 0.5
    center_y = -0.2
    
    # Ideal circle
    x_ideal = center_x + radius * np.cos(theta)
    y_ideal = center_y + radius * np.sin(theta)
    
    # Add noise
    noise_level = 0.02
    x_noisy = x_ideal + np.random.normal(0, noise_level, n_points)
    y_noisy = y_ideal + np.random.normal(0, noise_level, n_points)
    
    # Fit circle
    xc_fit, yc_fit, r_fit = circle_fit_pratt(x_noisy, y_noisy)
    
    print(f"\nTrue vs Fitted circle parameters:")
    xc_str = f"{xc_fit:.3f}" if xc_fit is not None else "None"
    yc_str = f"{yc_fit:.3f}" if yc_fit is not None else "None"
    r_str = f"{r_fit:.3f}" if r_fit is not None else "None"
    
    print(f"  Center X: {center_x:.3f} vs {xc_str}")
    print(f"  Center Y: {center_y:.3f} vs {yc_str}")
    print(f"  Radius: {radius:.3f} vs {r_str}")
    
    # Test IQ centering
    iq_data = x_noisy + 1j * y_noisy
    iq_centered = center_resonance_iq_circle(iq_data)
    
    if iq_centered is not None:
        center_original = np.mean(iq_data)
        center_after = np.mean(iq_centered)
        print(f"\nIQ Centering:")
        print(f"  Original center: {center_original:.3f}")
        print(f"  Center after: {center_after:.3f}")
        print(f"  Improvement: {abs(center_after) / abs(center_original):.1%} reduction")


def test_find_resonances():
    """Test resonance finding function."""
    print("\nTesting resonance finding...")
    print("=" * 50)
    
    # Generate multiple resonances
    freq_start = 100e6
    freq_stop = 500e6
    n_points = 2000  # More points for better resolution
    frequencies = np.linspace(freq_start, freq_stop, n_points)
    
    # Create multiple resonances with good separation
    resonance_freqs = [150e6, 250e6, 350e6, 450e6]
    resonance_Qrs = [5000, 8000, 6000, 7000]
    
    # Generate S21 data with properly deep resonances
    # Apply resonances individually to avoid cumulative baseline effects
    s21 = np.ones(n_points, dtype=complex)
    
    for i, (fr, Qr) in enumerate(zip(resonance_freqs, resonance_Qrs)):
        # Use critical coupling (Qc = Qr) for deep dips
        # This gives theoretically infinite dB dips at resonance
        # But we'll limit it to avoid numerical issues
        Qc = Qr * 1.01  # Just slightly undercoupled for ~40 dB dips
        x = (frequencies - fr) / fr
        
        # Standard resonator model
        resonator_response = 1 - (Qr / Qc) / (1 + 2j * Qr * x)
        s21 *= resonator_response
    
    # Apply overall cable loss
    s21 *= 0.9  # -0.9 dB cable loss
    
    # Add realistic noise
    noise = 0.0005 * (np.random.normal(0, 1, n_points) + 1j * np.random.normal(0, 1, n_points))
    s21 += noise
    
    # Find resonances with slightly relaxed parameters for robustness
    result = find_resonances(
        frequencies, s21,
        expected_resonances=4,
        min_dip_depth_db=0.5,  # Slightly relaxed for test robustness
        min_Q=3000,  # Slightly relaxed
        max_Q=2e7
    )
    
    print(f"\nExpected {len(resonance_freqs)} resonances:")
    for i, fr in enumerate(resonance_freqs):
        print(f"  {i+1}: {fr*1e-6:.1f} MHz")
    
    print(f"\nFound {len(result['resonance_frequencies'])} resonances:")
    found_freqs = result['resonance_frequencies']
    
    # Check if we found the right resonances (within 1 MHz tolerance)
    matches = 0
    for expected_fr in resonance_freqs:
        for found_fr in found_freqs:
            if abs(expected_fr - found_fr) < 1e6:  # 1 MHz tolerance
                matches += 1
                break
    
    if matches == len(resonance_freqs):
        print(f"\n✓ All {len(resonance_freqs)} expected resonances were found!")
    else:
        print(f"\n✗ Only {matches} out of {len(resonance_freqs)} expected resonances were found.")
    
    # Show details
    for i, res in enumerate(result['resonances_details']):
        print(f"  {i+1}: {res['frequency']*1e-6:.1f} MHz, Q≈{res['q_estimated']:.0f}, depth={res['prominence_db']:.1f} dB")


def test_fit_skewed_multisweep():
    """Test the multisweep fitting function."""
    print("\nTesting fit_skewed_multisweep...")
    print("=" * 50)
    
    # Create synthetic multisweep data
    test_multisweep_data = {}
    
    # Add three resonances
    for i, (fr, Qr, Qc) in enumerate([
        (100e6, 5000, 10000),
        (150e6, 8000, 16000),
        (200e6, 10000, 20000)
    ]):
        f, z, _ = generate_test_resonator_skewed(
            fr=fr, Qr=Qr, Qc=Qc, phi=0.05,
            noise_level=0.01, gain_mag=0.9
        )
        
        # Simulate multisweep output format
        test_multisweep_data[fr] = {
            'frequencies': f,
            'iq_complex': z,
            'original_center_frequency': fr,
            'recalculation_method_applied': 'none',
            'key_frequency_is_recalculated': False,
            'rotation_tod': None,
            'applied_rotation_degrees': 0.0,
            'sweep_direction': 'upward'
        }
    
    # Process with fitting
    fitted_data = fit_skewed_multisweep(
        test_multisweep_data,
        approx_Q_for_fit=1e4,
        fit_resonances=True,
        center_iq_circle=True
    )
    
    print("\nFitting results:")
    for cf, data in fitted_data.items():
        if data['fit_params'] and data['fit_params'].get('fr') != 'nan':
            print(f"\n  Resonance at {cf*1e-6:.0f} MHz:")
            print(f"    Fitted fr: {data['fit_params']['fr']*1e-6:.3f} MHz")
            print(f"    Qr: {data['fit_params']['Qr']:.0f}")
            print(f"    Qc: {data['fit_params']['Qc']:.0f}")
            print(f"    Qi: {data['fit_params']['Qi']:.0f}")
            print(f"    IQ centered: {'Yes' if data['iq_centered'] is not None else 'No'}")


def run_all_tests():
    """Run all test functions."""
    print("Running all fitting.py tests...")
    print("="*60)
    
    test_fit_skewed()
    test_circle_fitting()
    test_find_resonances()
    test_fit_skewed_multisweep()
    
    print("\n" + "="*60)
    print("All tests completed!")


# Run tests if executed directly
if __name__ == "__main__":
    run_all_tests()

def add_bifurcation_flags_to_multisweep_data(
    pickle_filepath_or_data: Union[str, Dict], 
    output_pickle_filepath: Optional[str] = None,
    bifurcation_threshold_factor: float = 10.0,
    bifurcation_min_peak_prominence_factor: float = 1.5
) -> Dict:
    """
    Loads multisweep data from a pickle file (or uses an existing data dictionary),
    identifies bifurcated resonances using identify_bifurcation, adds an 
    'is_bifurcated' flag to each resonance sweep, and optionally saves
    the modified data to a new pickle file if an output path is provided.

    Args:
        pickle_filepath_or_data (Union[str, Dict]): Path to the input multisweep 
                                                     pickle file or the already loaded data dictionary.
        output_pickle_filepath (Optional[str]): Path to save the modified data.
                                                If None, data is modified in memory.
                                                Only used if pickle_filepath_or_data is a string path.
        bifurcation_threshold_factor (float): Threshold factor for identify_bifurcation.
        bifurcation_min_peak_prominence_factor (float): Min peak prominence factor for identify_bifurcation.

    Returns:
        Dict: The loaded and modified multisweep data dictionary.
    """

    if isinstance(pickle_filepath_or_data, str):
        pickle_filepath = pickle_filepath_or_data
        print(f"Loading data from: {pickle_filepath} for bifurcation analysis.")
        try:
            with open(pickle_filepath, 'rb') as f:
                all_data = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: File not found at {pickle_filepath}")
            raise
        except Exception as e:
            print(f"Error loading pickle file {pickle_filepath}: {e}")
            raise
    elif isinstance(pickle_filepath_or_data, dict):
        all_data = pickle_filepath_or_data # Use the provided dictionary
        print("Processing provided data dictionary for bifurcation analysis.")
    else:
        raise TypeError("pickle_filepath_or_data must be a file path (str) or a dictionary.")


    if not isinstance(all_data, dict) or 'results_by_iteration' not in all_data:
        warnings.warn("Loaded data is not in the expected format or 'results_by_iteration' key is missing. Bifurcation flagging skipped.")
        return all_data

    sweep_data_container = all_data.get('results_by_iteration')
    if not isinstance(sweep_data_container, dict):
        warnings.warn("'results_by_iteration' does not contain a dictionary. Bifurcation flagging skipped.")
        return all_data

    print("Identifying bifurcated resonances...")
    bifurcated_resonances_count = 0
    total_resonances_checked = 0

    for iteration_key, iteration_data in sweep_data_container.items():
        if isinstance(iteration_data, dict) and 'data' in iteration_data:
            resonances_in_iteration = iteration_data['data']
            if isinstance(resonances_in_iteration, dict):
                for res_key, res_data_dict in resonances_in_iteration.items(): # res_key should be int
                    total_resonances_checked +=1
                    if isinstance(res_data_dict, dict) and 'iq_complex' in res_data_dict:
                        iq_data = res_data_dict['iq_complex']
                        if isinstance(iq_data, np.ndarray):
                            is_bifurcated = identify_bifurcation(
                                iq_data,
                                threshold_factor=bifurcation_threshold_factor,
                                min_peak_prominence_factor=bifurcation_min_peak_prominence_factor
                            )
                            res_data_dict['is_bifurcated'] = is_bifurcated
                            if is_bifurcated:
                                bifurcated_resonances_count +=1
                        else:
                            res_data_dict['is_bifurcated'] = False 
                            warnings.warn(f"IQ data for iteration {iteration_key}, res key {res_key} is not a numpy array. Flagged as not bifurcated.")
                    else:
                        if isinstance(res_data_dict, dict):
                             res_data_dict['is_bifurcated'] = False
                        warnings.warn(f"Data structure issue or missing 'iq_complex' for iteration {iteration_key}, res key {res_key}. Flagged as not bifurcated.")
            else:
                warnings.warn(f"Iteration {iteration_key} 'data' field is not a dictionary.")
        else:
            warnings.warn(f"Iteration {iteration_key} is not a dictionary or missing 'data' field.")

    
    print(f"Bifurcation analysis complete. Found {bifurcated_resonances_count} bifurcated resonances out of {total_resonances_checked} checked.")

    if isinstance(pickle_filepath_or_data, str) and output_pickle_filepath:
        try:
            with open(output_pickle_filepath, 'wb') as f_out:
                pickle.dump(all_data, f_out)
            print(f"Modified data with bifurcation flags saved to: {output_pickle_filepath}")
        except Exception as e:
            print(f"Error saving modified pickle file to {output_pickle_filepath}: {e}")
            
    return all_data
