"""
Data adapter for converting Periscope multisweep format to aggregate plotter format.

This module transforms the iteration-first data structure used by MultisweepPanel
`{iter_idx: {code: entry_dict}}` into a detector-based format compatible with
aggregate plotting and statistical analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional


def extract_multisweep_data(multisweep_panel) -> Dict[str, Any]:
    """
    Extract and convert multisweep data from a MultisweepPanel instance.

    Converts from Periscope's iteration-first format to a detector-based format
    suitable for aggregate plotting and analysis.

    Parameters
    ----------
    multisweep_panel : MultisweepPanel
        The multisweep panel instance containing ``results``
        (``{iter_idx: {code: entry_dict}}``).

    Returns
    -------
    dict
        Dictionary with structure:
        {
            'data': {code: {amp_idx: {'freq': [...], 'iq': [...], 'amp': float}}},
            'fit_params': {code: {amp_idx: {...fit parameters...}}},
            'bias_info': {code: {'bias_freq': float, 'bias_amplitude': float}},
            'module': int,
            'dac_scales': dict
        }
    """
    results = getattr(multisweep_panel, 'results', {})

    if not results:
        return {
            'data': {},
            'fit_params': {},
            'bias_info': {},
            'module': multisweep_panel.target_module,
            'dac_scales': getattr(multisweep_panel, 'dac_scales', {})
        }

    # Group data by detector and extract fit parameters
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


def _group_by_detector(results: Dict[int, Dict]) -> Tuple[Dict, Dict]:
    """
    Reorganise iteration-first data into detector-based structure.

    Converts from:
        ``{iter_idx: {code: entry_dict}}``
    To:
        ``{code: {amp_idx: {'freq': [...], 'iq': [...], 'amp': X, ...}}}``

    An *amp_idx* (0-based integer) is assigned per unique
    ``sweep_amplitude_normalized`` value, sorted ascending.

    Parameters
    ----------
    results : dict
        ``MultisweepPanel.results`` — ``{iter_idx: {code: entry_dict}}``.

    Returns
    -------
    tuple
        ``(detector_data, fit_params)`` — both indexed by detector code then
        amplitude index.
    """
    # Build amplitude index mapping from all unique amplitudes across all entries
    amp_to_index: dict = {}
    for iter_idx, code_dict in sorted(results.items()):
        for code, entry in code_dict.items():
            amp = entry.get('sweep_amplitude_normalized', 0.0)
            if amp not in amp_to_index:
                amp_to_index[amp] = len(amp_to_index)

    detector_dict: dict = {}
    fit_params_dict: dict = {}

    # Reorganise by detector code
    for iter_idx, code_dict in sorted(results.items()):
        for code, entry in code_dict.items():
            amp = entry.get('sweep_amplitude_normalized', 0.0)
            amp_idx = amp_to_index[amp]
            direction = entry.get('sweep_direction', 'unknown')

            if code not in detector_dict:
                detector_dict[code] = {}
                fit_params_dict[code] = {}

            # Prefer iq_volts (already in physical units); fall back to iq_counts
            freq = entry.get('frequencies', np.array([]))
            iq = entry.get('iq_volts', entry.get('iq_counts', np.array([])))
            bias_freq = entry.get('bias_frequency')

            detector_dict[code][amp_idx] = {
                'freq': np.asarray(freq),
                'iq': np.asarray(iq),
                'amp': amp,
                'bias_frequency': bias_freq,
                'iteration': iter_idx,
                'direction': direction,
            }

            # Extract fit parameters — prefer nonlinear, fall back to skewed
            _nl = entry.get('fits', {}).get('nonlinear', {})
            _sk = entry.get('fits', {}).get('skewed', {})
            fit_params: dict = {}
            if _nl.get('nonlinear_fit_params'):
                fit_params = dict(_nl['nonlinear_fit_params'])
            elif _sk.get('fit_params'):
                fit_params = dict(_sk['fit_params'])

            fit_params_dict[code][amp_idx] = fit_params

    return detector_dict, fit_params_dict


def _extract_bias_info(detector_data: Dict[str, Dict[int, Dict]]) -> Dict[str, Dict]:
    """
    Extract bias frequency and amplitude information for each detector.

    Parameters
    ----------
    detector_data : dict
        Detector-indexed data from :func:`_group_by_detector`.

    Returns
    -------
    dict
        Bias info indexed by detector code:
        ``{code: {'bias_freq': X, 'bias_amplitude': Y}}``.
    """
    bias_info = {}

    for code, amp_dict in detector_data.items():
        if amp_dict:
            first_sweep = amp_dict[min(amp_dict.keys())]
            bias_info[code] = {
                'bias_freq': first_sweep.get('bias_frequency'),
                'bias_amplitude': first_sweep.get('amp'),
            }

    return bias_info


def get_parameter_arrays(fit_params: Dict[str, Dict[int, Dict]],
                         param_name: str,
                         amp_idx: Optional[int] = None) -> Tuple[List, List]:
    """
    Extract arrays of a specific parameter across all detectors.

    Useful for histogram generation and statistical analysis.

    Parameters
    ----------
    fit_params : dict
        Fit parameters from :func:`extract_multisweep_data`.
    param_name : str
        Parameter to extract (e.g. ``'fr'``, ``'Qr'``, ``'Qi'``, ``'Qc'``, ``'a'``).
    amp_idx : int, optional
        Specific amplitude index to extract. If *None*, uses the first available.

    Returns
    -------
    tuple
        ``(values, detector_codes)`` — lists of parameter values and the
        corresponding detector codes. NaN values are filtered out.
    """
    values = []
    detector_codes = []

    for code, amp_dict in fit_params.items():
        if amp_idx is not None:
            if amp_idx not in amp_dict:
                continue
            params = amp_dict[amp_idx]
        else:
            if not amp_dict:
                continue
            params = amp_dict[min(amp_dict.keys())]

        if param_name in params:
            val = params[param_name]
            if val == 'nan' or (isinstance(val, float) and np.isnan(val)):
                continue
            values.append(float(val))
            detector_codes.append(code)

    return values, detector_codes


def calculate_grid_size(total_count: int,
                        default_low: int = 5,
                        default_high: int = 7) -> Tuple[int, int]:
    """
    Calculate optimal grid dimensions for multi-panel plots.

    Parameters
    ----------
    total_count : int
        Total number of plots to display.
    default_low : int
        Default columns for medium-sized arrays.
    default_high : int
        Default columns for large arrays.

    Returns
    -------
    tuple
        ``(nrows, ncols)`` for grid layout.
    """
    if total_count > 30:
        nperrow = default_high
    elif total_count < 10:
        nperrow = min(total_count, default_low)
    else:
        nperrow = default_low

    nrows = int(np.ceil(total_count / nperrow))
    return nrows, nperrow


def filter_failed_fits(param_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter out detectors with failed fits (NaN values).

    Parameters
    ----------
    param_dict : dict
        Dictionary indexed by detector code.

    Returns
    -------
    dict
        Filtered dictionary with failed fits removed.
    """
    filtered = {}

    for code, value in param_dict.items():
        is_nan = False

        if value == 'nan':
            is_nan = True
        elif isinstance(value, (float, np.floating)):
            is_nan = np.isnan(value)
        elif isinstance(value, dict):
            for param in ('fr', 'Qr'):
                if param in value:
                    v = value[param]
                    if v == 'nan' or (isinstance(v, (float, np.floating)) and np.isnan(v)):
                        is_nan = True
                        break

        if not is_nan:
            filtered[code] = value

    return filtered
