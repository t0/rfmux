"""
Nonlinear resonator fitting functions for rfmux, inspired by citkid.
Implements advanced fitting procedures for KID resonators including nonlinearity.

This module contains code adapted from the citkid project:
https://github.com/loganfoote/citkid

Original citkid code is licensed under the Apache License, Version 2.0.
See https://github.com/loganfoote/citkid/blob/main/LICENSE for details.

Modifications and adaptations for rfmux:
- Removed JIT compilation (Numba) dependencies
- Removed cable delay tau parameter (handled separately in rfmux)
- Modified gain estimation to use frequency extrema instead of separate gain scan
- Adapted to work with rfmux multisweep data structures
"""

import numpy as np
from scipy.optimize import curve_fit, fmin
import warnings
from typing import Dict, Tuple, List, Optional, Union, Any
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def nonlinear_iq(f: np.ndarray, fr: float, Qr: float, amp: float, phi: float, 
                 a: float, i0: float, q0: float) -> np.ndarray:
    r"""
    Describes the transmission through a nonlinear resonator.
    
    The model is:
                    /                           (j phi)   \
        (i0+j*q0) * |1 -        Qr             e^           |
                    |     --------------  X  ------------   |
                     \     Qc * cos(phi)       (1+ 2jy)    /

    where the nonlinearity of y is described by:
        yg = y + a/(1+y^2)  where yg = Qr*xg and xg = (f-fr)/fr

    Note: Cable delay tau is not included as rfmux handles it separately.

    Parameters
    ----------
    f : np.ndarray
        Array of frequencies in Hz
    fr : float
        Resonance frequency in Hz
    Qr : float
        Total quality factor (loaded Q)
    amp : float
        Qr / Qc, where Qc is the coupling quality factor (0 < amp < 1)
    phi : float
        Rotation parameter for impedance mismatch between KID and readout 
        circuit (radians)
    a : float
        Nonlinearity parameter. Bifurcation occurs at a = 4*sqrt(3)/9 ≈ 0.77.
        For linear resonators, a ≈ 0.
    i0 : float
        I (real) gain factor
    q0 : float
        Q (imaginary) gain factor
        Note: i0 + j*q0 describes the overall constant gain and phase offset

    Returns
    -------
    z : np.ndarray
        Array of complex IQ data (S21) corresponding to input frequencies
    """
    # Calculate normalized frequency shift
    deltaf = f - fr
    yg = Qr * deltaf / fr
    
    # Solve for y including nonlinearity
    y = get_y_nonlinear(yg, a)
    
    # Calculate the overall gain/phase offset
    s21_gain = (i0 + 1.j * q0)
    
    # Calculate resonator response
    s21_res = 1.0 - (amp / np.cos(phi)) * np.exp(1.j * phi) / (1.0 + 2.j * y)
    
    # Total transmission
    z = s21_gain * s21_res
    
    return z


def get_y_nonlinear(yg: Union[float, np.ndarray], a: float) -> Union[float, np.ndarray]:
    """
    Calculates the largest real root of the nonlinear equation:
        yg = y + a / (1 + y^2)
    
    This describes the frequency-pulling effect in a nonlinear resonator.
    Fully vectorized implementation using NumPy operations.
    
    Parameters
    ----------
    yg : float or np.ndarray
        Unmodified (linear) resonance shift: yg = Qr * (f - fr) / fr
    a : float
        Nonlinearity parameter
        
    Returns
    -------
    y : float or np.ndarray
        Largest real root of the nonlinear equation
    """
    if a == 0:
        # Linear case - no frequency pulling
        return yg
    
    # Handle scalar case
    if np.isscalar(yg):
        return _solve_single_y(yg, a)
    
    # Vectorized Newton's method for arrays
    yg = np.asarray(yg)
    y = yg.copy()  # Initial guess
    
    # Perform Newton iterations
    for _ in range(50):  # Maximum iterations
        # Vectorized function evaluation: f(y) = y + a/(1+y^2) - yg
        y_sq = y * y
        f_val = y + a / (1 + y_sq) - yg
        
        # Vectorized derivative: f'(y) = 1 - 2*a*y/(1+y^2)^2
        denom_sq = (1 + y_sq) ** 2
        f_prime = 1 - 2 * a * y / denom_sq
        
        # Update where derivative is not too small
        mask = np.abs(f_prime) > 1e-10
        if not np.any(mask):
            break
            
        # Newton update: y_new = y - f(y)/f'(y)
        y[mask] -= f_val[mask] / f_prime[mask]
        
        # Check convergence
        if np.all(np.abs(f_val) < 1e-10):
            break
    
    return y


def _solve_single_y(yg: float, a: float) -> float:
    """Solve the nonlinear equation for a single yg value."""
    # The equation y + a/(1+y^2) - yg = 0 can be rearranged to:
    # y(1+y^2) + a - yg(1+y^2) = 0
    # y^3 + y - yg*y^2 - yg + a = 0
    # y^3 - yg*y^2 + y + (a - yg) = 0
    
    # For small a and yg not too large, we can use Newton's method
    # starting from y = yg
    y = yg
    for _ in range(50):  # Maximum iterations
        f_val = y + a / (1 + y**2) - yg
        f_prime = 1 - 2 * a * y / (1 + y**2)**2
        
        if abs(f_prime) < 1e-10:
            break
            
        y_new = y - f_val / f_prime
        
        if abs(y_new - y) < 1e-10:
            break
            
        y = y_new
    
    return y


def estimate_and_remove_gain(frequencies: np.ndarray, iq_complex: np.ndarray, 
                           n_extrema_points: int = 5) -> Tuple[np.ndarray, float, float]:
    """
    Estimates and removes the gain from IQ data using points at frequency extrema.
    
    Instead of using a separate gain scan, this function estimates the off-resonance
    baseline by averaging points at the frequency extrema of the sweep.
    
    Parameters
    ----------
    frequencies : np.ndarray
        Frequency array in Hz
    iq_complex : np.ndarray
        Complex S21 data
    n_extrema_points : int, optional
        Number of points to average at each extremum (default: 5)
        
    Returns
    -------
    iq_corrected : np.ndarray
        Gain-corrected complex S21 data
    gain_mag : float
        Estimated gain magnitude
    gain_phase : float
        Estimated gain phase in radians
    """
    n_points = len(frequencies)
    n_avg = min(n_extrema_points, n_points // 4)
    
    # Get points at frequency extrema
    low_freq_avg = np.mean(iq_complex[:n_avg])
    high_freq_avg = np.mean(iq_complex[-n_avg:])
    
    # Average the two extrema for better estimate
    gain_complex = (low_freq_avg + high_freq_avg) / 2.0
    
    # Extract magnitude and phase
    gain_mag = np.abs(gain_complex)
    gain_phase = np.angle(gain_complex)
    
    # Remove the gain
    if gain_mag > 1e-10:  # Avoid division by very small numbers
        iq_corrected = iq_complex / gain_complex
    else:
        warnings.warn("Estimated gain magnitude is very small. Skipping gain correction.")
        iq_corrected = iq_complex.copy()
    
    return iq_corrected, gain_mag, gain_phase


def guess_p0_nonlinear(f: np.ndarray, z: np.ndarray) -> List[float]:
    """
    Make initial parameter guesses for nonlinear resonator fitting.
    
    Parameters
    ----------
    f : np.ndarray
        Frequency array in Hz
    z : np.ndarray
        Complex S21 data (should be gain-corrected)
        
    Returns
    -------
    p0 : list
        Initial parameter guesses [fr, Qr, amp, phi, a, i0, q0]
    """
    # Find resonance frequency as minimum of |S21|
    mag = np.abs(z)
    min_idx = np.argmin(mag)
    fr_guess = f[min_idx]
    
    # Estimate Q from 3dB bandwidth
    mag_db = 20 * np.log10(mag)
    min_db = mag_db[min_idx]
    target_db = min_db + 3  # 3dB point
    
    # Find points closest to 3dB level
    above_3db = mag_db > target_db
    if np.any(above_3db):
        # Find bandwidth
        first_above = np.where(above_3db)[0][0]
        last_above = np.where(above_3db)[0][-1]
        if last_above > first_above:
            bandwidth = f[last_above] - f[first_above]
            Qr_guess = fr_guess / bandwidth
        else:
            Qr_guess = 1e4  # Default
    else:
        Qr_guess = 1e4  # Default
    
    # Limit Q to reasonable range
    Qr_guess = np.clip(Qr_guess, 1e3, 1e7)
    
    # Estimate amplitude (coupling)
    # amp = Qr/Qc, related to the depth of the resonance
    mag_off_res = np.mean([mag[0], mag[-1]])
    mag_on_res = mag[min_idx]
    
    if mag_off_res > 0:
        depth = 1 - mag_on_res / mag_off_res
        amp_guess = np.clip(depth, 0.1, 0.99)
    else:
        amp_guess = 0.5
    
    # Phase guess - look at phase change across resonance
    phase = np.unwrap(np.angle(z))
    phase_change = phase[-1] - phase[0]
    phi_guess = np.clip(phase_change / 2, -np.pi/2, np.pi/2)
    
    # Nonlinearity - start with small value
    a_guess = 0.01
    
    # Gain parameters - should be close to (1, 0) if gain is removed properly
    i0_guess = np.real(np.mean(z[[0, -1]]))
    q0_guess = np.imag(np.mean(z[[0, -1]]))
    
    return [fr_guess, Qr_guess, amp_guess, phi_guess, a_guess, i0_guess, q0_guess]


def fit_nonlinear_iq(f: np.ndarray, z: np.ndarray, 
                     bounds: Optional[Tuple[List[float], List[float]]] = None,
                     p0: Optional[List[float]] = None,
                     fit_nonlinearity: bool = True,
                     max_iterations: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Fit a nonlinear resonator model to S21 sweep data.
    
    It is assumed that the system gain has been removed from the data before fitting.
    The optimal span of the data is 6 * fr / Qr.
    
    Parameters
    ----------
    f : np.ndarray
        Frequencies in Hz
    z : np.ndarray
        Complex S21 data (gain-corrected)
    bounds : tuple of lists, optional
        Lower and upper bounds for parameters
    p0 : list, optional
        Initial parameter guesses [fr, Qr, amp, phi, a, i0, q0]
    fit_nonlinearity : bool, optional
        If False, fixes a=0 (linear resonator). Default: True
    max_iterations : int, optional
        Maximum number of fitting iterations. Default: 3
        
    Returns
    -------
    p0 : np.ndarray
        Initial parameter guesses used
    popt : np.ndarray
        Optimized parameters
    perr : np.ndarray
        Parameter uncertainties (standard errors)
    residual : float
        Fitting residual
    """
    # Sort data by frequency
    sort_idx = np.argsort(f)
    f = f[sort_idx]
    z = z[sort_idx]
    
    # Get initial guess if not provided
    if p0 is None:
        p0 = guess_p0_nonlinear(f, z)
    else:
        p0 = list(p0)  # Make a copy
    
    # Set default bounds if not provided
    if bounds is None:
        #                 fr,     Qr,  amp,         phi,       a,    i0,   q0
        bounds = ([f.min(), 1e3, 0.01, -np.pi/2,    0, -1e2, -1e2],
                  [f.max(), 1e7, 0.99,  np.pi/2,  0.9,  1e2,  1e2])
    
    # If not fitting nonlinearity, fix a=0
    if not fit_nonlinearity:
        p0[4] = 0.0
        bounds[0][4] = -1e-10  # Small negative to satisfy bounds requirement
        bounds[1][4] = 1e-10   # Small positive to satisfy bounds requirement
    
    # Parameter scaling for better numerical stability
    #            fr,      Qr,  amp, phi, a,  i0, q0
    scaler = [1e-6, 1e-4,   1,   1, 1,  1,  1]
    
    # Scale parameters
    p0_scaled = [p * s for p, s in zip(p0, scaler)]
    bounds_scaled = (
        [b * s for b, s in zip(bounds[0], scaler)],
        [b * s for b, s in zip(bounds[1], scaler)]
    )
    
    # Stack real and imaginary parts for fitting
    z_stacked = np.hstack((np.real(z), np.imag(z)))
    
    # Fitting function that returns stacked real/imag
    def fit_func(freq, fr_s, Qr_s, amp, phi, a, i0, q0):
        # Unscale parameters
        fr = fr_s / scaler[0]
        Qr = Qr_s / scaler[1]
        
        z_model = nonlinear_iq(freq, fr, Qr, amp, phi, a, i0, q0)
        return np.hstack((np.real(z_model), np.imag(z_model)))
    
    # Iterative fitting with residual checking
    best_popt = None
    best_perr = None
    best_residual = np.inf
    
    current_p0 = p0_scaled.copy()
    current_bounds = [b.copy() for b in bounds_scaled]
    
    for iteration in range(max_iterations):
        try:
            # Fit the model
            popt_scaled, pcov = curve_fit(
                fit_func, f, z_stacked, 
                p0=current_p0, 
                bounds=current_bounds,
                maxfev=5000
            )
            
            # Unscale parameters
            popt = [p / s for p, s in zip(popt_scaled, scaler)]
            
            # Calculate uncertainties
            perr = np.sqrt(np.diag(pcov))
            perr = [p / s for p, s in zip(perr, scaler)]
            
            # Calculate residual
            z_fit = nonlinear_iq(f, *popt)
            residual = calculate_residuals(z, z_fit)
            
            # Check if this is the best fit so far
            if residual < best_residual:
                best_popt = popt
                best_perr = perr
                best_residual = residual
            
            # Check convergence
            if residual < 1e-3:
                break
            elif residual < 0.1:
                # Good fit, try to refine
                current_p0 = popt_scaled
            else:
                # Poor fit, adjust amplitude guess
                current_p0[2] *= 0.7  # Reduce amplitude
                if current_p0[2] < current_bounds[0][2]:
                    current_p0[2] = current_bounds[0][2]
                    
        except Exception as e:
            warnings.warn(f"Fitting iteration {iteration} failed: {str(e)}")
            if iteration == 0:
                # First iteration failed, return guess
                return np.array(p0), np.array(p0), np.zeros(7), np.inf
            else:
                # Use best result so far
                break
    
    return np.array(p0), np.array(best_popt), np.array(best_perr), best_residual


def calculate_residuals(z_data: np.ndarray, z_fit: np.ndarray) -> float:
    """
    Calculate normalized fitting residuals.
    
    Parameters
    ----------
    z_data : np.ndarray
        Measured complex S21 data
    z_fit : np.ndarray
        Fitted complex S21 data
        
    Returns
    -------
    residual : float
        Normalized residual (RMS error divided by mean magnitude)
    """
    diff = z_data - z_fit
    rms_error = np.sqrt(np.mean(np.abs(diff)**2))
    mean_mag = np.mean(np.abs(z_data))
    
    if mean_mag > 0:
        return rms_error / mean_mag
    else:
        return rms_error


def _fit_single_resonance(args: Tuple[Union[int, np.integer], Dict, bool, int]) -> Tuple[Union[int, np.integer], Dict]:
    """
    Fit a single resonance - extracted for parallel execution.
    
    Parameters
    ----------
    args : tuple
        (res_key, resonance_data, fit_nonlinearity, n_extrema_points)
        
    Returns
    -------
    res_key : int or np.integer
        The resonance index key
    updated_data : dict
        Dictionary with fitting results added
    """
    res_key, resonance_data, fit_nonlinearity, n_extrema_points = args
    
    # Make a copy to avoid modifying the original
    updated_data = resonance_data.copy()
    
    frequencies = resonance_data.get('frequencies')
    iq_complex = resonance_data.get('iq_complex')
    original_cf = resonance_data.get('original_center_frequency')
    
    if frequencies is None or iq_complex is None or original_cf is None:
        # Mark as failed
        updated_data['nonlinear_fit_params'] = None
        updated_data['nonlinear_fit_errors'] = None
        updated_data['nonlinear_fit_residual'] = np.inf
        updated_data['nonlinear_fit_success'] = False
        return res_key, updated_data
    
    try:
        # Step 1: Estimate and remove gain
        iq_corrected, gain_mag, gain_phase = estimate_and_remove_gain(
            frequencies, iq_complex, n_extrema_points
        )
        
        # Store gain info
        updated_data['gain_complex'] = gain_mag * np.exp(1j * gain_phase)
        updated_data['iq_gain_corrected'] = iq_corrected
        
        # Step 2: Fit nonlinear model
        p0, popt, perr, residual = fit_nonlinear_iq(
            frequencies, iq_corrected, 
            fit_nonlinearity=fit_nonlinearity
        )
        
        # Step 3: Store results
        param_names = ['fr', 'Qr', 'amp', 'phi', 'a', 'i0', 'q0']
        
        updated_data['nonlinear_fit_params'] = {
            name: value for name, value in zip(param_names, popt)
        }
        
        updated_data['nonlinear_fit_errors'] = {
            name: error for name, error in zip(param_names, perr)
        }
        
        updated_data['nonlinear_fit_residual'] = residual
        updated_data['nonlinear_fit_success'] = residual < 0.1
        
        # Calculate derived parameters
        Qr = popt[1]
        amp = popt[2]
        if amp < 1:
            Qc = Qr / amp
            Qi = 1 / (1/Qr - 1/Qc)
            updated_data['nonlinear_fit_params']['Qc'] = Qc
            updated_data['nonlinear_fit_params']['Qi'] = Qi
            
    except Exception as e:
        warnings.warn(f"Nonlinear fitting failed for {original_cf*1e-6:.3f} MHz: {e}")
        updated_data['nonlinear_fit_params'] = None
        updated_data['nonlinear_fit_errors'] = None
        updated_data['nonlinear_fit_residual'] = np.inf
        updated_data['nonlinear_fit_success'] = False
    
    return res_key, updated_data


def fit_nonlinear_iq_multisweep(
    multisweep_data: Union[Dict, List[Dict]],
    fit_nonlinearity: bool = True,
    n_extrema_points: int = 5,
    verbose: bool = False,
    parallel: bool = True,
    max_workers: Optional[int] = None
) -> Union[Dict, List[Dict]]:
    """
    Process multisweep results with CITKID-style nonlinear resonator fitting.
    
    This function takes the output from rfmux.multisweep and adds nonlinear
    fitting results to each resonance. It performs gain estimation and removal
    using frequency extrema, then fits the nonlinear resonator model.
    
    Parameters
    ----------
    multisweep_data : dict or list of dict
        Output from rfmux.multisweep. For single module: dict with final 
        center frequencies as keys. For multiple modules: list of such dicts.
    fit_nonlinearity : bool, optional
        If True, fits the nonlinearity parameter 'a'. If False, assumes 
        linear resonator (a=0). Default: True
    n_extrema_points : int, optional
        Number of points at frequency extrema to use for gain estimation.
        Default: 5
    verbose : bool, optional
        If True, prints fitting progress and results. Default: False
    parallel : bool, optional
        If True, use ThreadPoolExecutor for parallel fitting. Default: True
    max_workers : int or None, optional
        Maximum number of worker threads. If None, uses min(4, cpu_count).
        Default: None
        
    Returns
    -------
    results : dict or list of dict
        Input data with added fitting results. Each resonance dict gains:
        - 'gain_complex': Estimated complex gain
        - 'iq_gain_corrected': Gain-corrected IQ data
        - 'nonlinear_fit_params': Dict with fit parameters
        - 'nonlinear_fit_errors': Dict with parameter uncertainties
        - 'nonlinear_fit_residual': Fitting residual
        - 'nonlinear_fit_success': Boolean success flag
    """
    # Handle multi-module case
    if isinstance(multisweep_data, list):
        return [fit_nonlinear_iq_multisweep(
            module_data, fit_nonlinearity, n_extrema_points, verbose, parallel, max_workers
        ) for module_data in multisweep_data]
    
    # Process single module
    if not isinstance(multisweep_data, dict):
        warnings.warn("fit_nonlinear_iq_multisweep: input is not a dict or list")
        return multisweep_data
    
    # Filter valid resonances
    valid_resonances = []
    for res_key, resonance_data in multisweep_data.items():
        if not isinstance(resonance_data, dict):
            continue
        
        # Expect index-based keys only
        if not isinstance(res_key, (int, np.integer)):
            if verbose:
                print(f"Skipping non-integer key {res_key}: expected index-based keys only")
            continue
        
        # Check required fields
        if (resonance_data.get('frequencies') is not None and 
            resonance_data.get('iq_complex') is not None and
            resonance_data.get('original_center_frequency') is not None):
            valid_resonances.append((res_key, resonance_data))
        elif verbose:
            cf = resonance_data.get('original_center_frequency')
            if cf:
                print(f"Skipping resonance at {cf*1e-6:.3f} MHz: missing data")
            else:
                print(f"Skipping resonance at index {res_key}: missing data")
    
    # Decide whether to use parallel processing
    if parallel and len(valid_resonances) > 1:
        # Set up worker pool
        if max_workers is None:
            max_workers = min(4, os.cpu_count() or 1)
        
        if verbose:
            print(f"\nProcessing {len(valid_resonances)} resonances in parallel with {max_workers} workers...")
        
        # Prepare arguments for parallel execution
        args_list = [(res_key, res_data, fit_nonlinearity, n_extrema_points) 
                      for res_key, res_data in valid_resonances]
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(_fit_single_resonance, args): args[0] 
                       for args in args_list}
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                completed += 1
                try:
                    res_key, updated_data = future.result()
                    multisweep_data[res_key].update(updated_data)
                    
                    if verbose:
                        cf = updated_data.get('original_center_frequency', 0)
                        success = updated_data.get('nonlinear_fit_success', False)
                        status = "✓" if success else "✗"
                        print(f"  [{completed}/{len(valid_resonances)}] {status} Resonance at {cf*1e-6:.3f} MHz")
                        
                        if success and updated_data.get('nonlinear_fit_params'):
                            params = updated_data['nonlinear_fit_params']
                            print(f"      Fitted fr: {params['fr']*1e-6:.3f} MHz, "
                                  f"Qr: {params['Qr']:.0f}, a: {params.get('a', 0):.3f}")
                        
                except Exception as e:
                    res_key = futures[future]
                    warnings.warn(f"Parallel fitting failed for resonance {res_key}: {e}")
    else:
        # Sequential processing
        if verbose and len(valid_resonances) > 0:
            print(f"\nProcessing {len(valid_resonances)} resonances sequentially...")
        
        for i, (res_key, resonance_data) in enumerate(valid_resonances):
            _, updated_data = _fit_single_resonance(
                (res_key, resonance_data, fit_nonlinearity, n_extrema_points)
            )
            multisweep_data[res_key].update(updated_data)
            
            if verbose:
                cf = updated_data.get('original_center_frequency', 0)
                success = updated_data.get('nonlinear_fit_success', False)
                status = "✓" if success else "✗"
                print(f"  [{i+1}/{len(valid_resonances)}] {status} Resonance at {cf*1e-6:.3f} MHz")
                
                if success and updated_data.get('nonlinear_fit_params'):
                    params = updated_data['nonlinear_fit_params']
                    print(f"      Fitted fr: {params['fr']*1e-6:.3f} MHz, "
                          f"Qr: {params['Qr']:.0f}, a: {params.get('a', 0):.3f}")
    
    return multisweep_data


# --- Testing Functions ---

def generate_test_resonator_data(
    fr: float = 100e6,
    Qr: float = 1e4, 
    amp: float = 0.5,
    phi: float = 0.1,
    a: float = 0.3,
    n_points: int = 201,
    span_factor: float = 6.0,
    noise_level: float = 0.01,
    gain_mag: float = 0.8,
    gain_phase: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Generate synthetic resonator data for testing.
    
    Parameters
    ----------
    fr : float
        Resonance frequency in Hz
    Qr : float
        Quality factor
    amp : float
        Coupling parameter (Qr/Qc)
    phi : float
        Impedance mismatch phase
    a : float
        Nonlinearity parameter
    n_points : int
        Number of frequency points
    span_factor : float
        Frequency span as multiple of fr/Qr
    noise_level : float
        Fractional noise level
    gain_mag : float
        Overall gain magnitude
    gain_phase : float
        Overall gain phase
        
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
    
    # Generate ideal resonator response
    i0, q0 = 1.0, 0.0  # Unity gain for ideal response
    iq_ideal = nonlinear_iq(frequencies, fr, Qr, amp, phi, a, i0, q0)
    
    # Apply gain
    gain_complex = gain_mag * np.exp(1j * gain_phase)
    iq_with_gain = iq_ideal * gain_complex
    
    # Add noise
    noise_real = np.random.normal(0, noise_level, n_points)
    noise_imag = np.random.normal(0, noise_level, n_points)
    iq_data = iq_with_gain + noise_real + 1j * noise_imag
    
    true_params = {
        'fr': fr,
        'Qr': Qr,
        'amp': amp,
        'phi': phi,
        'a': a,
        'gain_mag': gain_mag,
        'gain_phase': gain_phase
    }
    
    return frequencies, iq_data, true_params


def test_nonlinear_fitting():
    """
    Test the nonlinear fitting procedures with synthetic data.
    """
    print("Testing nonlinear resonator fitting...")
    print("=" * 50)
    
    # Test 1: Linear resonator (a=0)
    print("\nTest 1: Linear resonator (a=0)")
    f, z, true_params = generate_test_resonator_data(
        fr=150e6, Qr=5000, amp=0.7, phi=0.05, a=0.0,
        noise_level=0.005, gain_mag=0.9, gain_phase=0.3
    )
    
    # Create multisweep-like data structure
    test_data = {
        150e6: {
            'frequencies': f,
            'iq_complex': z,
            'original_center_frequency': 150e6
        }
    }
    
    # Fit the data
    fitted_data = fit_nonlinear_iq_multisweep(test_data, fit_nonlinearity=False, verbose=True)
    
    # Check results
    fit_params = fitted_data[150e6]['nonlinear_fit_params']
    print(f"\nTrue vs Fitted parameters:")
    print(f"  fr: {true_params['fr']*1e-6:.3f} vs {fit_params['fr']*1e-6:.3f} MHz")
    print(f"  Qr: {true_params['Qr']:.0f} vs {fit_params['Qr']:.0f}")
    print(f"  amp: {true_params['amp']:.3f} vs {fit_params['amp']:.3f}")
    print(f"  phi: {true_params['phi']:.3f} vs {fit_params['phi']:.3f}")
    
    # Test 2: Nonlinear resonator
    print("\n" + "="*50)
    print("\nTest 2: Nonlinear resonator (a=0.4)")
    f, z, true_params = generate_test_resonator_data(
        fr=200e6, Qr=10000, amp=0.5, phi=-0.1, a=0.4,
        noise_level=0.01, gain_mag=1.1, gain_phase=-0.5
    )
    
    test_data = {
        200e6: {
            'frequencies': f,
            'iq_complex': z,
            'original_center_frequency': 200e6
        }
    }
    
    fitted_data = fit_nonlinear_iq_multisweep(test_data, fit_nonlinearity=True, verbose=True)
    
    fit_params = fitted_data[200e6]['nonlinear_fit_params']
    print(f"\nTrue vs Fitted parameters:")
    print(f"  fr: {true_params['fr']*1e-6:.3f} vs {fit_params['fr']*1e-6:.3f} MHz")
    print(f"  Qr: {true_params['Qr']:.0f} vs {fit_params['Qr']:.0f}")
    print(f"  amp: {true_params['amp']:.3f} vs {fit_params['amp']:.3f}")
    print(f"  phi: {true_params['phi']:.3f} vs {fit_params['phi']:.3f}")
    print(f"  a: {true_params['a']:.3f} vs {fit_params['a']:.3f}")
    
    # Test 3: Multiple resonances
    print("\n" + "="*50)
    print("\nTest 3: Multiple resonances")
    
    test_data = {}
    for i, (fr, a) in enumerate([(100e6, 0.1), (150e6, 0.3), (200e6, 0.5)]):
        f, z, _ = generate_test_resonator_data(
            fr=fr, Qr=8000, amp=0.6, phi=0.0, a=a,
            noise_level=0.008
        )
        test_data[fr] = {
            'frequencies': f,
            'iq_complex': z,
            'original_center_frequency': fr
        }
    
    fitted_data = fit_nonlinear_iq_multisweep(test_data, verbose=True)
    
    print("\nSummary of fits:")
    for cf in fitted_data:
        params = fitted_data[cf]['nonlinear_fit_params']
        if params:
            print(f"  {cf*1e-6:.0f} MHz: Qr={params['Qr']:.0f}, a={params['a']:.3f}, "
                  f"residual={fitted_data[cf]['nonlinear_fit_residual']:.3e}")


# Run tests if executed directly
if __name__ == "__main__":
    test_nonlinear_fitting()
