"""
Analysis functions for rfmux, including resonance fitting and IQ circle manipulation.
"""

import numpy as np
from scipy.optimize import curve_fit
import warnings

def s21_skewed(f, f0, Qr, Qcre, Qcim, A):
    """
    Skewed Lorentzian model for S21 magnitude.
    Based on hidfmux implementation.

    Prevents negative Qi based on parameters.

    Args:
        f (np.ndarray): Frequency array (Hz).
        f0 (float): Resonance frequency (Hz).
        Qr (float): Resonator quality factor.
        Qcre (float): Real part of complex coupling quality factor Qc.
        Qcim (float): Imaginary part of complex coupling quality factor Qc.
        A (float): Amplitude scaling factor.

    Returns:
        np.ndarray: Modelled S21 magnitude. Returns np.inf for unphysical parameters.
    """
    Qe = Qcre + 1j * Qcim
    # Check for physical validity (prevents negative Qi)
    # Ensure Qcre is positive and the condition for positive Qi holds
    # For Qi > 0, we need Qc > Qr (equivalently, abs(Qe)**2 / Qcre >= Qr)
    if Qcre <= 1e-9 or Qr <= 1e-9 or abs(Qe)**2 / Qcre <= Qr:
         # Return a large value or np.inf to guide fitter away
         return np.full_like(f, np.inf)
    else:
        # Avoid division by zero if f0 is zero
        if abs(f0) < 1e-12:
             return np.full_like(f, np.inf) # Or handle appropriately

        x = (f - f0) / f0
        # Use np.errstate to handle potential division by zero if Qe is zero (though prevented above)
        with np.errstate(divide='ignore', invalid='ignore'):
             # Calculate complex S21
             s21_complex = A * (1 - (Qr / Qe) / (1 + 2j * Qr * x))

        # Return magnitude, handle potential NaNs from division by zero if necessary
        magnitude = np.abs(s21_complex)
        # Replace non-finite values (NaN, Inf) resulting from calculation issues
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
        init = [fr_guess, approxQr, approxQr, 0., np.mean(s21_mag[search_indices]) if len(search_indices) > 0 else 1.0] # Guess Qcim=0 initially, A based on mean in region
        # Bounds: ([fr_low, Qr_low, Qcre_low, Qcim_low, A_low], [fr_high, Qr_high, Qcre_high, Qcim_high, A_high])
        # Loosen Q bounds slightly compared to hidfmux example
        bounds = ([fr_lbound, 1e2, 1e2, -np.inf, 0], [fr_ubound, 1e9, 1e9, np.inf, np.inf])

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

# --- More functions will be added below ---

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
    min_Q : float, optional
        Minimum estimated quality factor. Used to calculate the maximum allowed width
        of a resonance feature, by default 1e4.
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


def process_multisweep_results(
    multisweep_data_dict: dict,
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
        multisweep_data_dict (dict): The dictionary returned by the
                                     `rfmux.algorithms.measurement.multisweep.multisweep` function.
                                     It's expected to have center frequencies as keys, and each
                                     value is a dict with 'frequencies' and 'iq_complex'.
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
        dict: The input `multisweep_data_dict` with 'fit_params' and 'iq_centered'
              added to each sub-dictionary if the respective operations were performed.
    """
    if not isinstance(multisweep_data_dict, dict):
        warnings.warn("process_multisweep_results: input is not a dictionary. Returning as is.")
        return multisweep_data_dict

    for cf, data in multisweep_data_dict.items():
        if not isinstance(data, dict):
            warnings.warn(f"process_multisweep_results: item for cf {cf} is not a dictionary. Skipping.")
            continue

        frequencies = data.get('frequencies')
        iq_complex = data.get('iq_complex')

        if frequencies is None or iq_complex is None:
            warnings.warn(f"process_multisweep_results: 'frequencies' or 'iq_complex' missing for cf {cf}. Skipping.")
            continue

        # Initialize keys in case operations are skipped
        data['fit_params'] = None
        data['iq_centered'] = None

        if fit_resonances:
            try:
                fit_params_result = fit_skewed(
                    frequencies, iq_complex,
                    approxQr=approx_Q_for_fit,
                    normalize=normalize_fit,
                    fr_lim=fr_lim_fit
                )
                data['fit_params'] = fit_params_result
            except Exception as e:
                warnings.warn(f"Fitting failed for resonance at {cf*1e-6:.3f} MHz during post-processing: {e}")
                # data['fit_params'] remains None

        if center_iq_circle:
            try:
                # If fitting was done and successful, and resulted in centered data,
                # it might be preferable to center the *original* iq_complex
                # or the one from fit_params if it's more robust.
                # For now, always center the original iq_complex.
                iq_centered_result = center_resonance_iq_circle(iq_complex)
                data['iq_centered'] = iq_centered_result
            except Exception as e:
                warnings.warn(f"IQ centering failed for resonance at {cf*1e-6:.3f} MHz during post-processing: {e}")
                # data['iq_centered'] remains None
                
    return multisweep_data_dict