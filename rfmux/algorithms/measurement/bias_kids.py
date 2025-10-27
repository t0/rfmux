"""
bias_kids: A measurement algorithm for biasing KIDs at their optimal operating points
based on multisweep characterization data.
"""

import numpy as np
import asyncio
import warnings
from typing import Union, Dict, List, Optional, Any, Tuple, Callable
from scipy.signal import butter, filtfilt



def bandpass_filter(data: np.ndarray, fs: float, lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
    """
    Apply a bandpass filter to the data.
    
    Args:
        data: Input signal
        fs: Sampling frequency (Hz)
        lowcut: Low frequency cutoff (Hz)
        highcut: High frequency cutoff (Hz)
        order: Filter order (default: 4)
        
    Returns:
        Filtered signal
    """
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)


async def find_optimal_phases_parallel(
    crs,
    bias_configs: Dict[int, Dict],
    module: int,
    num_samples: int = 300,
    apply_bandpass: bool = True,
    fs: float = 597,
    lowcut: float = 5,
    highcut: float = 20,
    phase_step: int = 5
) -> Dict[int, Tuple[float, float]]:
    """
    Find optimal ADC phases for multiple channels in parallel.
    
    Args:
        crs: CRS object
        bias_configs: Dictionary of bias configurations {det_idx: config}
        module: Module number
        num_samples: Number of samples to collect at each phase
        apply_bandpass: Whether to apply bandpass filter to Q data
        fs: Sampling frequency (Hz) - only used if apply_bandpass is True
        lowcut: Bandpass filter low cutoff (Hz) - only used if apply_bandpass is True
        highcut: Bandpass filter high cutoff (Hz) - only used if apply_bandpass is True
        phase_step: Phase step size in degrees
        
    Returns:
        Dictionary of {det_idx: (best_phase_degrees, max_std_q)}
    """
    optimal_phases = {}
    
    # Initialize best phases and stds for each detector
    for det_idx in bias_configs:
        optimal_phases[det_idx] = (0.0, -np.inf)
    
    # Scan through phases
    for phase in range(0, 360, phase_step):
        # Set phase for all channels simultaneously
        async with crs.tuber_context() as ctx:
            for det_idx, config in bias_configs.items():
                ctx.set_phase(phase, units=crs.UNITS.DEGREES, target=crs.TARGET.ADC, 
                             channel=config['channel'], module=module)
            await ctx()
        
        # Collect samples for all channels at once
        samples = await crs.get_samples(num_samples, channel=None, module=module, average=False)
        
        # Process each detector's Q data
        for det_idx, config in bias_configs.items():
            channel_idx = config['channel'] - 1  # Convert to 0-based index
            
            # Extract Q data for this channel
            q_data = np.array(samples.q[channel_idx])
            
            # Calculate standard deviation (with or without bandpass filter)
            try:
                if apply_bandpass:
                    q_processed = bandpass_filter(q_data, fs, lowcut, highcut)
                else:
                    q_processed = q_data
                    
                std_q = float(np.std(q_processed))
                
                # Update if this is better than the current best
                current_best_phase, current_best_std = optimal_phases[det_idx]
                if std_q > current_best_std:
                    optimal_phases[det_idx] = (float(phase), std_q)
                    
            except Exception as e:
                if apply_bandpass:
                    warnings.warn(f"Bandpass filter failed for detector {det_idx} at phase {phase}°: {e}")
                else:
                    warnings.warn(f"Failed to process Q data for detector {det_idx} at phase {phase}°: {e}")
                continue
    
    # Report results
    filter_desc = "filtered " if apply_bandpass else ""
    for det_idx, (best_phase, best_std) in optimal_phases.items():
        print(f"Detector {det_idx}: Optimal phase = {best_phase}°, {filter_desc}std(Q) = {best_std:.4f}")
    
    return optimal_phases


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
    optimize_phase: bool = False,
    bandpass_params: Optional[Dict[str, float]] = None,
    num_phase_samples: int = 300,
    phase_step: int = 5,
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
        optimize_phase (bool): If True, scan through ADC phases to find the phase that
                             maximizes variance in bandpass-filtered Q timestream.
                             Defaults to False.
        bandpass_params (dict, optional): Parameters for bandpass filter used in phase optimization.
                                        Keys: 'lowcut' (Hz), 'highcut' (Hz), 'fs' (sampling freq Hz).
                                        Defaults: {'lowcut': 5, 'highcut': 20, 'fs': 597}.
        num_phase_samples (int): Number of samples to collect for phase optimization.
                               Defaults to 300.
        phase_step (int): Phase step size in degrees for optimization scan.
                        Defaults to 5.
        module (int | list[int], optional): Target module(s). If None, extracted from results.
        progress_callback (callable, optional): Function called with (module, progress_percentage).
        
    Returns:
        Dictionary or list of dictionaries containing only the biased detectors' data.
        Each entry includes the original multisweep data plus:
        - 'bias_channel': The assigned channel number (1-based)
        - 'bias_amplitude': The amplitude selected for biasing
        - 'bifurcation_suspected': Whether any amplitude showed bifurcation
        - 'bias_successful': Whether the detector was successfully biased
        - 'optimal_phase_degrees': The optimal ADC phase found (if phase optimization enabled)
        - 'phase_optimization_std': The bandpass-filtered Q std at optimal phase
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
                    optimize_phase=optimize_phase,
                    bandpass_params=bandpass_params,
                    num_phase_samples=num_phase_samples,
                    phase_step=phase_step,
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
    
    # Base frequency (Nyquist frequency) for quantization
    base_freq = 298.0232238769531  # Hz
    
    # Set default bandpass parameters if not provided
    if bandpass_params is None:
        bandpass_params = {'lowcut': 5, 'highcut': 20, 'fs': 597}
    
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
            
            if bias_freq is None or sweep_amp is None:
                warnings.warn(f"Detector {det_idx}: Missing bias frequency or amplitude")
                continue
                
            # Channel assignment: det_idx is already 1-based from multisweep
            channel = det_idx
            
            # Quantize the absolute bias frequency to nearest multiple of base frequency
            quantized_bias_freq = round(bias_freq / base_freq) * base_freq
            
            # Calculate channel frequency relative to NCO
            channel_freq = quantized_bias_freq - nco_freq
            
            bias_configs[det_idx] = {
                'channel': int(channel),  # Ensure it's a Python int
                'frequency': float(channel_freq),  # Channel frequency relative to NCO
                'original_bias_frequency': float(bias_freq),  # Store original for reference
                'quantized_bias_frequency': float(quantized_bias_freq),  # Quantized absolute frequency
                'amplitude': float(sweep_amp),  # Ensure it's a Python float
                'phase': 0.0,  # No phase rotation applied anymore
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
    
    # First, set up all tones without phase optimization
    async with crs.tuber_context() as ctx:
        for det_idx, config in bias_configs.items():
            try:
                ctx.set_frequency(config['frequency'], channel=config['channel'], module=module)
                ctx.set_amplitude(config['amplitude'], channel=config['channel'], module=module)
                ctx.set_phase(config['phase'], units=crs.UNITS.DEGREES, target=crs.TARGET.ADC, channel=config['channel'], module=module)
            except Exception as e:
                print(f"[Bias] Failed to set up detector {det_idx}: {e}")
                continue
        await ctx()
    
    # Now perform phase optimization if requested
    if optimize_phase:
        print(f"Optimizing phases for {len(bias_configs)} detectors in parallel...")
        
        # Determine if bandpass filter should be applied
        # If bandpass_params contains 'apply_bandpass', use that value, otherwise default to True
        apply_bandpass = True
        if bandpass_params is not None:
            apply_bandpass = bool(bandpass_params.get('apply_bandpass', True))
        
        # Find optimal phases for all detectors in parallel
        optimal_phases = await find_optimal_phases_parallel(
            crs=crs,
            bias_configs=bias_configs,
            module=module,  # type: ignore  # module is guaranteed to be int at this point
            num_samples=num_phase_samples,
            apply_bandpass=apply_bandpass,
            fs=bandpass_params.get('fs', 597) if bandpass_params else 597,
            lowcut=bandpass_params.get('lowcut', 5) if bandpass_params else 5,
            highcut=bandpass_params.get('highcut', 20) if bandpass_params else 20,
            phase_step=phase_step
        )
    else:
        # No optimization - all phases are 0
        optimal_phases = {det_idx: (0.0, None) for det_idx in bias_configs}
    
    # Apply the optimal phases and create result data
    for det_idx, config in bias_configs.items():
        try:
            optimal_phase, phase_std = optimal_phases.get(det_idx, (0.0, None))
            
            # Set the optimal phase for this channel
            if optimal_phase != 0.0:
                await crs.set_phase(optimal_phase, crs.UNITS.DEGREES, crs.TARGET.ADC, 
                                   channel=config['channel'], module=module)
            
            # Copy the original multisweep data and add bias info
            biased_data = multisweep_results[det_idx].copy()
            biased_data['bias_channel'] = config['channel']
            biased_data['bifurcation_suspected'] = config['bifurcation_suspected']
            biased_data['bias_successful'] = True
            biased_data['optimal_phase_degrees'] = optimal_phase
            biased_data['phase_optimization_std'] = phase_std
            
            # If we have df calibration and applied a phase, rotate it
            if 'df_calibration' in biased_data and biased_data['df_calibration'] is not None and optimal_phase != 0.0:
                # Rotate the df calibration by the applied phase
                phase_rad = np.radians(optimal_phase)
                rotation_factor = np.exp(1j * phase_rad)
                biased_data['df_calibration'] *= rotation_factor
                biased_data['df_calibration_rotated'] = True
            
            successfully_biased[det_idx] = biased_data
            
        except Exception as e:
            warnings.warn(f"Failed to bias detector {det_idx}: {e}")
    
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
