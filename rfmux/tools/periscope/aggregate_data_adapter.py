"""
Data adapter for converting Periscope multisweep format to aggregate plotter format.

This module transforms the iteration-based data structure used by MultisweepPanel
into a detector-based format compatible with aggregate plotting and statistical analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional


def extract_multisweep_data(multisweep_panel) -> Dict[str, Any]:
    """
    Extract and convert multisweep data from a MultisweepPanel instance.
    
    Converts from Periscope's iteration-based format to a detector-based format
    suitable for aggregate plotting and analysis.
    
    Parameters
    ----------
    multisweep_panel : MultisweepPanel
        The multisweep panel instance containing results_by_iteration
        
    Returns
    -------
    dict
        Dictionary with structure:
        {
            'data': {detector_id: {amp_idx: {'freq': [...], 'iq': [...], 'amp': float}}},
            'fit_params': {detector_id: {amp_idx: {...fit parameters...}}},
            'bias_info': {detector_id: {'bias_freq': float, 'bias_amplitude': float}},
            'module': int,
            'dac_scales': dict
        }
    """
    results = multisweep_panel.results_by_iteration
    
    if not results:
        return {
            'data': {},
            'fit_params': {},
            'bias_info': {},
            'module': multisweep_panel.target_module,
            'dac_scales': getattr(multisweep_panel, 'dac_scales', {})
        }
    
    # Group data by detector (also extracts fit parameters)
    detector_data, fit_params = _group_by_detector(results)
    
    # Extract bias information
    bias_info = _extract_bias_info(detector_data)
    
    return {
        'data': detector_data,
        'fit_params': fit_params,
        'bias_info': bias_info,
        'module': multisweep_panel.target_module,
        'dac_scales': getattr(multisweep_panel, 'dac_scales', {})
    }


def _group_by_detector(results_by_iteration: Dict[int, Dict]) -> Tuple[Dict[int, Dict[int, Dict]], Dict[int, Dict[int, Dict]]]:
    """
    Reorganize iteration-based data into detector-based structure.
    
    Converts from:
        {iteration: {'amplitude': X, 'data': {detector: {...}}}}
    To:
        {detector: {amp_idx: {'freq': [...], 'iq': [...], 'amp': X}}}
        
    Parameters
    ----------
    results_by_iteration : dict
        MultisweepPanel.results_by_iteration
        
    Returns
    -------
    tuple
        (detector_data, fit_params) - Both indexed by detector and amplitude
    """
    detector_dict = {}
    fit_params_dict = {}
    
    # Build amplitude index mapping
    amp_to_index = {}
    for iter_idx in sorted(results_by_iteration.keys()):
        iteration_data = results_by_iteration[iter_idx]
        amp = iteration_data.get('amplitude', 0.0)
        if amp not in amp_to_index:
            amp_to_index[amp] = len(amp_to_index)
    
    # Reorganize by detector
    for iter_idx in sorted(results_by_iteration.keys()):
        iteration_data = results_by_iteration[iter_idx]
        amp = iteration_data.get('amplitude', 0.0)
        amp_idx = amp_to_index[amp]
        
        detector_data = iteration_data.get('data', {})
        
        for detector_id, det_data in detector_data.items():
            if detector_id not in detector_dict:
                detector_dict[detector_id] = {}
                fit_params_dict[detector_id] = {}
            
            # Extract sweep data
            freq = det_data.get('frequencies', np.array([]))
            # Use iq_complex_volts for normalized voltage units
            iq = det_data.get('iq_complex_volts', det_data.get('iq_complex', np.array([])))
            bias_freq = det_data.get('bias_frequency', None)
            
            # Store sweep data in detector-indexed structure
            detector_dict[detector_id][amp_idx] = {
                'freq': np.asarray(freq),
                'iq': np.asarray(iq),
                'amp': amp,
                'bias_frequency': bias_freq,
                'iteration': iter_idx,
                'direction': iteration_data.get('direction', 'unknown')
            }
            
            # Extract fit parameters (prefer nonlinear if available)
            fit_params = {}
            
            # Check for nonlinear fit parameters first
            if 'nonlinear_fit_params' in det_data and det_data['nonlinear_fit_params']:
                fit_params = dict(det_data['nonlinear_fit_params'])
            # Fall back to skewed fit parameters
            elif 'fit_params' in det_data and det_data['fit_params']:
                fit_params = dict(det_data['fit_params'])
            
            # Store fit parameters
            fit_params_dict[detector_id][amp_idx] = fit_params
    
    return detector_dict, fit_params_dict


def _extract_bias_info(detector_data: Dict[int, Dict[int, Dict]]) -> Dict[int, Dict]:
    """
    Extract bias frequency and amplitude information for each detector.
    
    Parameters
    ----------
    detector_data : dict
        Detector-indexed data from _group_by_detector
        
    Returns
    -------
    dict
        Bias info indexed by detector
        {detector_id: {'bias_freq': X, 'bias_amplitude': Y}}
    """
    bias_info = {}
    
    for detector_id, amp_dict in detector_data.items():
        # Use first available amplitude data to extract bias info
        if amp_dict:
            first_sweep = amp_dict[min(amp_dict.keys())]
            bias_info[detector_id] = {
                'bias_freq': first_sweep.get('bias_frequency', None),
                'bias_amplitude': first_sweep.get('amp', None)
            }
    
    return bias_info


def get_parameter_arrays(fit_params: Dict[int, Dict[int, Dict]], 
                        param_name: str,
                        amp_idx: Optional[int] = None) -> Tuple[List, List]:
    """
    Extract arrays of a specific parameter across all detectors.
    
    Useful for histogram generation and statistical analysis.
    
    Parameters
    ----------
    fit_params : dict
        Fit parameters from extract_multisweep_data
    param_name : str
        Parameter to extract (e.g., 'fr', 'Qr', 'Qi', 'Qc', 'a')
    amp_idx : int, optional
        Specific amplitude index to extract. If None, uses first available.
        
    Returns
    -------
    tuple
        (values, detector_ids) - Lists of parameter values and corresponding detector IDs.
        NaN values are filtered out.
    """
    values = []
    detector_ids = []
    
    for detector_id, amp_dict in fit_params.items():
        # Select amplitude index
        if amp_idx is not None:
            if amp_idx not in amp_dict:
                continue
            params = amp_dict[amp_idx]
        else:
            # Use first available amplitude
            if not amp_dict:
                continue
            params = amp_dict[min(amp_dict.keys())]
        
        # Extract parameter
        if param_name in params:
            val = params[param_name]
            # Filter out NaN and 'nan' string values
            if val != 'nan' and not (isinstance(val, float) and np.isnan(val)):
                values.append(float(val))
                detector_ids.append(detector_id)
    
    return values, detector_ids


def calculate_grid_size(total_count: int, 
                       default_low: int = 5, 
                       default_high: int = 7) -> Tuple[int, int]:
    """
    Calculate optimal grid dimensions for multi-panel plots.
    
    Adapted from hidfmux's get_nperrow function.
    
    Parameters
    ----------
    total_count : int
        Total number of plots to display
    default_low : int
        Default columns for medium-sized arrays
    default_high : int
        Default columns for large arrays
        
    Returns
    -------
    tuple
        (nrows, ncols) for grid layout
    """
    if total_count > 30:
        nperrow = default_high
    elif total_count < 10:
        nperrow = min(total_count, default_low)
    else:
        nperrow = default_low
    
    nrows = int(np.ceil(total_count / nperrow))
    return nrows, nperrow


def filter_failed_fits(param_dict: Dict[int, Any]) -> Dict[int, Any]:
    """
    Filter out detectors with failed fits (NaN values).
    
    Parameters
    ----------
    param_dict : dict
        Dictionary indexed by detector ID
        
    Returns
    -------
    dict
        Filtered dictionary with failed fits removed
    """
    filtered = {}
    
    for detector_id, value in param_dict.items():
        # Check for NaN in various forms
        is_nan = False
        
        if value == 'nan':
            is_nan = True
        elif isinstance(value, (float, np.floating)):
            is_nan = np.isnan(value)
        elif isinstance(value, dict):
            # Check if any critical parameter is NaN
            critical_params = ['fr', 'Qr']
            for param in critical_params:
                if param in value:
                    val = value[param]
                    if val == 'nan' or (isinstance(val, (float, np.floating)) and np.isnan(val)):
                        is_nan = True
                        break
        
        if not is_nan:
            filtered[detector_id] = value
    
    return filtered
