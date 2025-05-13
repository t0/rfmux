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
