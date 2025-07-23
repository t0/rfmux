"""
bias_kids: A measurement algorithm for biasing KIDs at their optimal operating points
based on multisweep characterization data.
"""

import numpy as np
import asyncio
import warnings
from typing import Union, Dict, List, Optional, Any, Tuple, Callable



def _extract_data_from_gui_format(gui_results: Dict) -> Tuple[Optional[Dict[Tuple[float, str], Dict[int, Dict]]], Dict]:
    """
    Extract detector data from GUI multisweep results format.
    
    Args:
        gui_results: Dictionary with 'results_by_iteration' key
        
    Returns:
        Tuple of (multiamp_data, metadata)
        where multiamp_data is {(amplitude, direction): {detector_idx: data}}
        Returns (None, {}) if not GUI format
    """
    if 'results_by_iteration' not in gui_results:
        # Not GUI format, return None to indicate not applicable
        return None, {}
        
    multiamp_data = {}
    metadata = {
        'iterations': [],
        'amplitudes': set(),
        'directions': set()
    }
    
    for iteration_data in gui_results['results_by_iteration']:
        iteration = iteration_data.get('iteration')
        amplitude = iteration_data.get('amplitude')
        direction = iteration_data.get('direction', 'upward')
        data = iteration_data.get('data', {})
        
        key = (amplitude, direction)
        multiamp_data[key] = data
        
        metadata['iterations'].append({
            'iteration': iteration,
            'amplitude': amplitude,
            'direction': direction
        })
        metadata['amplitudes'].add(amplitude)
        metadata['directions'].add(direction)
    
    return multiamp_data, metadata


async def bias_kids(
    crs,
    multisweep_results: Union[Dict, List[Dict]],
    nonlinear_threshold: float = 0.77,
    fallback_to_lowest: bool = True,
    *,
    module: Optional[Union[int, List[int]]] = None,
    progress_callback: Optional[Callable] = None
) -> Union[Dict[int, Dict], List[Dict[int, Dict]]]:
    """
    Bias KIDs at their optimal operating points based on multisweep characterization.
    
    This algorithm analyzes multisweep results to find the best amplitude for each
    detector (highest amplitude that is not bifurcated and has nonlinear parameter < threshold),
    then programs the detectors with the appropriate frequency, phase, and amplitude.
    
    Args:
        crs: The CRS object to use for hardware communication.
        multisweep_results: Can be one of:
                           - Dict with 'results_by_iteration' key: Multi-amplitude GUI format
                           - Dict[int, Dict]: Single amplitude results
                           - List[Dict]: Multiple modules
        nonlinear_threshold (float): Maximum acceptable nonlinear parameter 'a'.
                                   Defaults to 0.77.
        fallback_to_lowest (bool): If True and no suitable amplitude found,
                                 use the lowest available amplitude. If False,
                                 skip the detector. Defaults to True.
        module (int | list[int], optional): Target module(s). If None, extracted from results.
        progress_callback (callable, optional): Function called with (module, progress_percentage).
        
    Returns:
        Dictionary or list of dictionaries containing only the biased detectors' data.
        Each entry includes the original multisweep data plus:
        - 'bias_channel': The assigned channel number (1-based)
        - 'bias_amplitude': The amplitude selected for biasing
        - 'bifurcation_suspected': Whether any amplitude showed bifurcation
        - 'bias_successful': Whether the detector was successfully biased
    """
    
    # Check if this is GUI format with multiple amplitudes
    if isinstance(multisweep_results, dict) and 'results_by_iteration' in multisweep_results:
        # Extract multi-amplitude data from GUI format
        multiamp_data, metadata = _extract_data_from_gui_format(multisweep_results)
        
        if multiamp_data is not None:
            # Find optimal configuration for each detector
            optimal_configs = analyze_multiamp_data(multiamp_data, nonlinear_threshold, fallback_to_lowest)
            
            # Convert to single result set using optimal configurations
            single_results = {}
            for det_idx, config in optimal_configs.items():
                single_results[det_idx] = config['selected_data'].copy()
                # Add metadata about amplitude selection
                single_results[det_idx]['selected_amplitude'] = config['selected_amplitude']
                single_results[det_idx]['bifurcation_ever_seen'] = config.get('bifurcation_ever_seen', False)
                
            # Proceed with single-amplitude logic using optimal results
            multisweep_results = single_results
        
    # Handle list of results (multiple modules)
    elif isinstance(multisweep_results, list):
        if module is None:
            # Extract module numbers from the data if possible
            # This would require the multisweep results to contain module info
            # For now, assume modules are sequential starting from 1
            module = list(range(1, len(multisweep_results) + 1))
        elif not isinstance(module, list):
            module = [module]
            
        if len(multisweep_results) != len(module):
            raise ValueError(f"Number of result sets ({len(multisweep_results)}) "
                           f"doesn't match number of modules ({len(module)})")
        
        # Process each module
        tasks = []
        for mod_idx, (mod_results, mod_num) in enumerate(zip(multisweep_results, module)):
            tasks.append(
                bias_kids(
                    crs=crs,
                    multisweep_results=mod_results,
                    nonlinear_threshold=nonlinear_threshold,
                    fallback_to_lowest=fallback_to_lowest,
                    module=mod_num,
                    progress_callback=progress_callback
                )
            )
        
        return await asyncio.gather(*tasks)
    
    # Single module processing
    if module is None:
        raise ValueError("Module must be specified for single result set")
    
    # Get current NCO frequency
    nco_freq = await crs.get_nco_frequency(module=module)
    
    # Analyze each detector to find optimal bias point
    bias_configs = {}
    total_detectors = len(multisweep_results)
    
    for det_idx, det_data in multisweep_results.items():
        # Check if this is multi-amplitude data
        # Multi-amplitude data would need to be organized differently
        # For now, assume single amplitude per detector
        
        # Extract key parameters
        is_bifurcated = det_data.get('is_bifurcated', False)
        nonlinear_params = det_data.get('nonlinear_fit_params', {})
        nonlinear_a = nonlinear_params.get('a', float('inf'))
        
        # Check if detector meets criteria
        suitable = not is_bifurcated and nonlinear_a < nonlinear_threshold
        
        if suitable or fallback_to_lowest:
            # Prepare bias configuration
            bias_freq = det_data.get('bias_frequency', det_data.get('original_center_frequency'))
            sweep_amp = det_data.get('sweep_amplitude')
            rotation_deg = det_data.get('applied_rotation_degrees', 0.0)
            
            if bias_freq is None or sweep_amp is None:
                warnings.warn(f"Detector {det_idx}: Missing bias frequency or amplitude")
                continue
                
            # Channel assignment: det_idx is already 1-based from multisweep
            channel = det_idx
            
            bias_configs[det_idx] = {
                'channel': int(channel),  # Ensure it's a Python int
                'frequency': float(bias_freq - nco_freq),  # Ensure it's a Python float
                'amplitude': float(sweep_amp),  # Ensure it's a Python float
                'phase': -float(rotation_deg),  # Ensure it's a Python float
                'suitable': suitable,
                'bifurcation_suspected': is_bifurcated,
                'nonlinear_a': nonlinear_a
            }
            
            if not suitable:
                warnings.warn(f"Detector {det_idx}: No suitable amplitude found "
                            f"(bifurcated={is_bifurcated}, a={nonlinear_a:.3f}). "
                            f"Using fallback amplitude.")
    
    # # Clear all channels first
    # max_channels = 1024
    # async with crs.tuber_context() as ctx:
    #     for ch in range(1, max_channels + 1):
    #         ctx.set_amplitude(0, channel=ch, module=module)
    #     await ctx()
    
    # Program the selected detectors
    successfully_biased = {}
    
    async with crs.tuber_context() as ctx:
        for det_idx, config in bias_configs.items():
            try:
                ctx.set_frequency(config['frequency'], channel=config['channel'], module=module)
                ctx.set_amplitude(config['amplitude'], channel=config['channel'], module=module)
                ctx.set_phase(config['phase'], units=crs.UNITS.DEGREES, target=crs.TARGET.ADC, channel=config['channel'], module=module)
                # Copy the original multisweep data and add bias info
                biased_data = multisweep_results[det_idx].copy()
                biased_data['bias_channel'] = config['channel']
                biased_data['bifurcation_suspected'] = config['bifurcation_suspected']
                biased_data['bias_successful'] = True
                
                successfully_biased[det_idx] = biased_data
                
            except Exception as e:
                warnings.warn(f"Failed to bias detector {det_idx}: {e}")
                
        await ctx()
    
    # Progress callback
    if progress_callback:
        progress_callback(module, 100.0)
    
    # Log summary
    print(f"Module {module}: Successfully biased {len(successfully_biased)}/{total_detectors} detectors")
    
    return successfully_biased


def analyze_multiamp_data(
    multiamp_data: Dict[Tuple[float, str], Dict[int, Dict]],
    nonlinear_threshold: float = 0.77,
    fallback_to_lowest: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Analyze multi-amplitude multisweep results to find optimal bias points.
    
    Args:
        multiamp_data: Dictionary with (amplitude, direction) tuple as key, detector data as value
        nonlinear_threshold: Maximum acceptable nonlinear parameter
        fallback_to_lowest: Whether to use lowest amplitude as fallback
        
    Returns:
        Dictionary with detector index as key, optimal configuration as value
    """
    optimal_configs = {}
    
    # Get all unique amplitudes and sort them
    amplitudes = sorted(set(amp for amp, _ in multiamp_data.keys()), reverse=True)
    
    # Get all detector indices from the first entry
    first_data = next(iter(multiamp_data.values()))
    detector_indices = list(first_data.keys())
    
    for det_idx in detector_indices:
        # Collect data across all amplitudes for this detector
        amp_analysis = []
        bifurcation_ever_seen = False
        
        for amp in amplitudes:
            # Check both directions if available
            for direction in ['upward', 'downward']:
                key = (amp, direction)
                if key not in multiamp_data:
                    continue
                    
                if det_idx not in multiamp_data[key]:
                    continue
                    
                det_data = multiamp_data[key][det_idx]
                is_bifurcated = det_data.get('is_bifurcated', False)
                nonlinear_params = det_data.get('nonlinear_fit_params', {})
                nonlinear_a = nonlinear_params.get('a', float('inf'))
                nonlinear_success = det_data.get('nonlinear_fit_success', False)
                
                if is_bifurcated:
                    bifurcation_ever_seen = True
                
                # Only consider nonlinear parameter if fit was successful
                if nonlinear_success:
                    suitable = not is_bifurcated and nonlinear_a < nonlinear_threshold
                else:
                    suitable = not is_bifurcated
                    
                amp_analysis.append({
                    'amplitude': amp,
                    'direction': direction,
                    'is_bifurcated': is_bifurcated,
                    'nonlinear_a': nonlinear_a,
                    'nonlinear_fit_success': nonlinear_success,
                    'suitable': suitable,
                    'data': det_data
                })
        
        # Find the highest suitable amplitude
        suitable_entries = [a for a in amp_analysis if a['suitable']]
        
        if suitable_entries:
            # Use the first suitable (which is highest amplitude due to sort)
            optimal = suitable_entries[0]
        elif fallback_to_lowest and amp_analysis:
            # No suitable amplitude found, use lowest amplitude
            # Sort by amplitude (ascending) to get lowest
            sorted_by_amp = sorted(amp_analysis, key=lambda x: x['amplitude'])
            optimal = sorted_by_amp[0]
        else:
            optimal = None
            
        if optimal:
            optimal_configs[det_idx] = {
                'selected_amplitude': optimal['amplitude'],
                'selected_direction': optimal['direction'],
                'selected_data': optimal['data'],
                'bifurcation_ever_seen': bifurcation_ever_seen,
                'all_amplitudes': amp_analysis
            }
    
    return optimal_configs
