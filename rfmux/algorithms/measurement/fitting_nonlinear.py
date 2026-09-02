"""
The legacy dict-walking surface for nonlinear resonator fitting.

The model and its fitter have moved to :mod:`rfmux.tuning.fits` — they are
analysis over two arrays rather than board operations, so they belong with the
other pure layers — and are re-exported below so callers reaching for
``fitting_nonlinear.nonlinear_iq`` still find it. The citkid attribution moved
with them.

What is left here is :func:`fit_nonlinear_iq_multisweep`, which reads the
pre-schema-2 multisweep contract — ``iq_complex``,
``original_center_frequency``, integer keys — and writes its results flat onto
each entry. ``multisweep`` returns ``iq_counts`` keyed by resonator name and
has done since the sweep stopped doing anything but measure, so this works on
files saved before that change and on nothing else.

For sweeps taken since, use :func:`rfmux.tuning.fit_sweeps`, which writes the
nonlinear fit under the entry's ``fits`` subdict and stores the estimated gain
rather than a gain-corrected copy of a trace the entry already holds.
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Union
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Re-exported for callers that reach for them by attribute — Periscope does.
# The implementations live in rfmux/tuning/fits.py.
from ...tuning.fits import (
    FitFailed,
    calculate_residuals,
    fit_nonlinear_iq,
    get_y_nonlinear,
    guess_p0_nonlinear,
    nonlinear_iq,
    remove_gain,
)

# Listed so a linter reports the imports above as re-exports rather than as
# seven unused names, and so this module's surface is stated in one place.
__all__ = [
    # Re-exported from rfmux.tuning.fits.
    "FitFailed",
    "calculate_residuals",
    "fit_nonlinear_iq",
    "get_y_nonlinear",
    "guess_p0_nonlinear",
    "nonlinear_iq",
    "remove_gain",
    # Still implemented here.
    "estimate_and_remove_gain",
    "fit_nonlinear_iq_multisweep",
]


def estimate_and_remove_gain(frequencies, iq_complex, n_extrema_points: int = 5):
    """Deprecated. Use :func:`rfmux.tuning.fits.remove_gain`.

    The same estimate, split back into a magnitude and a phase — which is what
    the replacement's single complex gain was always multiplied back together
    from.
    """
    corrected, gain = remove_gain(
        frequencies, iq_complex, n_extrema_points=n_extrema_points
    )
    return corrected, float(np.abs(gain)), float(np.angle(gain))


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

    def mark_failed():
        updated_data['nonlinear_fit_params'] = None
        updated_data['nonlinear_fit_errors'] = None
        updated_data['nonlinear_fit_residual'] = np.inf
        updated_data['nonlinear_fit_success'] = False
        return res_key, updated_data

    if frequencies is None or iq_complex is None or original_cf is None:
        return mark_failed()

    try:
        # Step 1: Estimate and remove gain
        iq_corrected, gain = remove_gain(
            frequencies, iq_complex, n_extrema_points=n_extrema_points
        )
        updated_data['gain_complex'] = gain
        updated_data['iq_gain_corrected'] = iq_corrected

        # Step 2: Fit the nonlinear model. Qc and Qi come back derived.
        params, errors, residual = fit_nonlinear_iq(
            frequencies, iq_corrected, fit_nonlinearity=fit_nonlinearity
        )
    except Exception as e:
        warnings.warn(f"Nonlinear fitting failed for {original_cf*1e-6:.3f} MHz: {e}")
        return mark_failed()

    # Step 3: Store results
    updated_data['nonlinear_fit_params'] = params
    updated_data['nonlinear_fit_errors'] = errors
    updated_data['nonlinear_fit_residual'] = residual
    updated_data['nonlinear_fit_success'] = residual < 0.1

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
    Deprecated. Use :func:`rfmux.tuning.fit_sweeps` instead.

    Reads ``iq_complex`` and ``original_center_frequency`` off each entry and
    expects integer keys — the contract multisweep had before it stopped doing
    anything but measure. It works on data saved under that contract and
    nothing newer.

    It also writes flat: ``nonlinear_fit_params`` and friends land beside the
    measurement they describe, where the replacement puts them under
    ``fits["nonlinear"]``. And it stores ``iq_gain_corrected``, an N-point copy
    of a trace the entry already holds divided by a number the entry also
    holds; the replacement stores the gain and rebuilds the trace on request.

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
    warnings.warn(
        "fitting_nonlinear.fit_nonlinear_iq_multisweep is deprecated; use "
        "rfmux.tuning.fit_sweeps, which reads iq_counts and writes its results "
        "into each sweep entry's 'fits' subdict.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _fit_nonlinear_iq_multisweep(
        multisweep_data, fit_nonlinearity, n_extrema_points, verbose, parallel, max_workers
    )


def _fit_nonlinear_iq_multisweep(
    multisweep_data,
    fit_nonlinearity,
    n_extrema_points,
    verbose,
    parallel,
    max_workers,
):
    """:func:`fit_nonlinear_iq_multisweep` without the warning, so recursing is quiet."""
    # Handle multi-module case
    if isinstance(multisweep_data, list):
        return [_fit_nonlinear_iq_multisweep(
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
