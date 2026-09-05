"""
bias_kids: A measurement algorithm for biasing KIDs at their optimal operating points
based on multisweep characterization data.
"""

import numpy as np
import asyncio
import warnings
from .df_calibration import (bias_frequency_from_fit, df_calibration_for_entry,
                             ensure_fits)
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
) -> Dict[int, Tuple[float, float]]:
    """
    The ADC phase per detector that puts its timestream's principal axis
    along Q, from one set of samples taken at phase zero.

    Signal and noise move mostly along the resonance's frequency
    direction, so the principal component of (I, Q) is that direction
    and one PCA gives the angle.  The board rotates samples by +phase,
    so a principal axis at theta lands on Q at 90 - theta.

    Args:
        crs: CRS object
        bias_configs: Dictionary of bias configurations {det_idx: config}
        module: Module number
        num_samples: Number of samples in the one set
        apply_bandpass: Whether to bandpass the samples before the PCA
        fs, lowcut, highcut: The bandpass, in Hz

    Returns:
        Dictionary of {det_idx: (phase_degrees, std along the principal axis)}
    """
    async with crs.tuber_context() as ctx:
        for det_idx, config in bias_configs.items():
            ctx.set_phase(0.0, units=crs.UNITS.DEGREES, target=crs.TARGET.ADC,
                          channel=config['channel'], module=module)
        await ctx()
    samples = await crs.get_samples(num_samples, channel=None, module=module, average=False)

    optimal_phases = {}
    for det_idx, config in bias_configs.items():
        k = config['channel'] - 1
        i = np.asarray(samples.i[k], dtype=float)
        q = np.asarray(samples.q[k], dtype=float)
        if apply_bandpass:
            try:
                i = bandpass_filter(i, fs, lowcut, highcut)
                q = bandpass_filter(q, fs, lowcut, highcut)
            except Exception as e:
                warnings.warn(f"Bandpass filter failed for detector {det_idx}: {e}; "
                              f"using the raw samples")
        w, v = np.linalg.eigh(np.cov(np.vstack((i, q))))
        scale = float(np.mean(i * i + q * q))
        if not np.isfinite(w).all() or w.max() <= 1e-18 * max(scale, 1e-300):
            # No variation to find an axis in (a simulated board without
            # noise, say): leave the phase alone rather than rotate by
            # the angle of rounding error.
            warnings.warn(f"Detector {det_idx}: the samples show no variation, "
                          f"so no ADC phase was chosen")
            optimal_phases[det_idx] = (0.0, 0.0)
            continue
        pc = v[:, np.argmax(w)]
        # An axis, not a direction: fold the eigenvector's sign away.
        theta = (np.degrees(np.arctan2(pc[1], pc[0])) + 90.0) % 180.0 - 90.0
        phase = (90.0 - theta) % 360.0
        optimal_phases[det_idx] = (float(phase), float(np.sqrt(w.max())))
        print(f"Detector {det_idx}: principal axis at {theta:.1f} deg, ADC phase {phase:.1f} deg")
    return optimal_phases


def _extract_data_from_gui_format(gui_results: Dict) -> Tuple[Optional[Dict[int, Dict[int, Dict]]], Dict]:
    """
    Extract detector data from the legacy GUI multisweep results format.

    Converts the old iteration-indexed format into the canonical
    detector-indexed format used by :func:`analyze_multiamp_data`.

    Args:
        gui_results: Dictionary with 'results_by_iteration' key.

    Returns:
        Tuple of (results_by_detector, metadata).
        ``results_by_detector`` is ``{detector_id: {iteration_index: entry_dict}}``.
        Returns (None, {}) if not GUI format.
    """
    if 'results_by_iteration' not in gui_results:
        return None, {}

    results_by_detector: Dict[int, Dict[int, Dict]] = {}
    metadata: Dict[str, Any] = {
        'iterations': [],
        'amplitudes': set(),
        'directions': set()
    }

    for iteration_data in gui_results['results_by_iteration']:
        iteration = iteration_data.get('iteration')
        amplitude = iteration_data.get('amplitude')
        direction = iteration_data.get('direction', 'upward')
        data = iteration_data.get('data', {})

        metadata['iterations'].append({
            'iteration': iteration,
            'amplitude': amplitude,
            'direction': direction
        })
        metadata['amplitudes'].add(amplitude)
        metadata['directions'].add(direction)

        for detector_id, det_data in data.items():
            if detector_id not in results_by_detector:
                results_by_detector[detector_id] = {}
            # Store with iteration index as key (new canonical format)
            # Ensure amplitude/direction are inside the entry
            entry = dict(det_data)
            entry['amplitude'] = amplitude
            entry['direction'] = direction
            entry['iteration'] = iteration
            results_by_detector[detector_id][iteration] = entry

    return results_by_detector, metadata


async def measure_calibrations_by_step(crs, bias_configs: Dict[int, Dict], module: int,
                                       step_hz: float = 300.0, num_samples: int = 100
                                       ) -> Dict[int, complex]:
    """The df calibration of every detector in *bias_configs*, measured
    where it sits: each tone steps *step_hz* down, then up, in lockstep,
    one read of the module at each, and the calibration is the inverse
    of the complex slope, in hertz per volt.  The ADC phase in force
    applies to the samples, so the result is in the frame the detector
    is read in.  Detectors whose samples do not move are left out."""
    from ...core.transferfunctions import convert_roc_to_volts
    z = {}
    for sign in (-1.0, 1.0):
        async with crs.tuber_context() as ctx:
            for config in bias_configs.values():
                ctx.set_frequency(config['frequency'] + sign * step_hz,
                                  channel=config['channel'], module=module)
            await ctx()
        samples = await crs.get_samples(num_samples, channel=None, module=module, average=False)
        z[sign] = {det: complex(np.mean(samples.i[config['channel'] - 1]),
                                np.mean(samples.q[config['channel'] - 1]))
                   for det, config in bias_configs.items()}
    async with crs.tuber_context() as ctx:
        for config in bias_configs.values():
            ctx.set_frequency(config['frequency'], channel=config['channel'], module=module)
        await ctx()
    out = {}
    for det in bias_configs:
        slope = convert_roc_to_volts((z[1.0][det] - z[-1.0][det]) / (2.0 * step_hz))
        if np.isfinite(slope) and slope != 0:
            out[det] = complex(1.0 / slope)
    return out


def _suitable(det_data: Dict, nonlinear_threshold: float):
    """Whether an amplitude can be biased at: its sweep does not jump
    and, where it carries a nonlinear fit, the fitted nonlinearity is
    below the threshold.  A failed fit leaves None behind and counts as
    no fit.  Returns (suitable, is_bifurcated, a, has_fit)."""
    is_bifurcated = bool(det_data.get('is_bifurcated', False))
    params = det_data.get('nonlinear_fit_params') or {}
    a = params.get('a', float('inf'))
    has_fit = bool(det_data.get('nonlinear_fit_success', False)) and np.isfinite(a)
    suitable = not is_bifurcated and (a < nonlinear_threshold if has_fit else True)
    return suitable, is_bifurcated, a, has_fit


def _fit_entries(entries, fit_method: str) -> int:
    """Give every entry the fit bias_kids works from; say how many
    needed it, since at thousands of resonators that is seconds."""
    n = ensure_fits(entries, fit_method)
    if n:
        print(f"[Bias] {fit_method} fit run for {n} of {len(entries)} sweeps "
              f"that had none")
    return n


def _bias_point_from_fit(entry: Dict, fit_method: str) -> None:
    """Move the entry's bias frequency to the fitted curve's version of
    the multisweep's choice, when the entry carries that fit."""
    method = entry.get('recalculation_method_applied', 'max-diq')
    f_fit = bias_frequency_from_fit(entry, method, fit_method)
    if f_fit is not None and np.isfinite(f_fit):
        entry['bias_frequency'] = float(f_fit)
        entry['bias_frequency_source'] = fit_method


async def bias_kids(
    crs,
    multisweep_results: Union[Dict, List[Dict]],
    nonlinear_threshold: float = 0.77,
    fallback_to_lowest: bool = True,
    optimize_phase: bool = False,
    bandpass_params: Optional[Dict[str, float]] = None,
    num_phase_samples: int = 300,
    fit_method: str = "nonlinear",
    measure_calibration: bool = True,
    calibration_step_hz: float = 300.0,
    *,
    module: Optional[Union[int, List[int]]] = None,
    progress_callback: Optional[Callable] = None
) -> Union[Dict[int, Dict], List[Dict[int, Dict]]]:
    """
    Bias KIDs at their optimal operating points based on multisweep characterization.

    Every sweep is given the resonance fit named by *fit_method*, run here
    with the flow's own fitter for the sweeps that lack it and left alone
    for those that carry it.  From that fit come the amplitude choice
    (nonlinear fit only: the skewed fit has no nonlinearity parameter,
    so with it only the jump detector rules an amplitude out), the bias
    frequency (the multisweep's max-diq or min-s21 point read off the
    fitted curve rather than the raw sweep grid), and the df calibration
    at that frequency.  The fits and the bias frequency are written back
    onto the entries handed in, so a second call does not fit again.
    
    Args:
        crs: The CRS object to use for hardware communication.
        multisweep_results: Can be one of:
                           - Dict with 'results_by_detector' key: Detector-indexed multi-amplitude format
                           - Dict with 'results_by_iteration' key: Legacy iteration-indexed format
                           - Dict[int, Dict]: Single amplitude results
                           - List[Dict]: Multiple modules
        nonlinear_threshold (float): Maximum acceptable nonlinear parameter 'a'.
                                   Defaults to 0.77.
        fallback_to_lowest (bool): If True and no suitable amplitude found,
                                 use the lowest available amplitude. If False,
                                 skip the detector. Defaults to True.
        optimize_phase (bool): If True, set each detector's ADC phase so the
                             timestream's principal axis lies along Q, from one
                             set of samples.  Defaults to False.
        bandpass_params (dict, optional): Parameters for bandpass filter used in phase optimization.
                                        Keys: 'lowcut' (Hz), 'highcut' (Hz), 'fs' (sampling freq Hz).
                                        Defaults: {'lowcut': 5, 'highcut': 20, 'fs': 597}.
        num_phase_samples (int): Number of samples to collect for phase optimization.
                               Defaults to 300.
        fit_method (str): "nonlinear" (default) or "skewed": the resonance fit the
                        amplitude choice and bias frequency come from.
        measure_calibration (bool): Measure the df calibration where each detector
                        ends up, by stepping every tone calibration_step_hz down and
                        up in lockstep and reading the samples: two reads for the
                        module, no model.  On the simulator the direction is within
                        0.7 degrees of truth where the fit's is within 4.  The fit's
                        calibration is kept as 'df_calibration_fit'.  Defaults to True.
        calibration_step_hz (float): Half the tone step for that measurement, well
                        inside a linewidth.  Defaults to 300 Hz.
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
        - 'bias_frequency': Where the detector was biased; from the fit when
          'bias_frequency_source' names one, else the multisweep's own choice
        - 'df_calibration': Hz per volt at the bias point: measured, or from the fit
        - 'df_calibration_source': "measured" or "fit"
        - 'df_calibration_fit': the fit's calibration, when both exist
    """
    
    # Detect multi-amplitude format and find optimal bias points
    results_by_detector = None

    if isinstance(multisweep_results, dict) and 'results_by_detector' in multisweep_results:
        # Detector-indexed format (current canonical format)
        results_by_detector = multisweep_results['results_by_detector']

    elif isinstance(multisweep_results, dict) and 'results_by_iteration' in multisweep_results:
        # Legacy iteration-indexed format — convert to detector-indexed
        results_by_detector, _ = _extract_data_from_gui_format(multisweep_results)

    if results_by_detector is not None:
        if fit_method == "nonlinear":
            # The amplitude choice reads the fitted nonlinearity of every
            # amplitude.  The skewed fit has none to read, so with it
            # only the chosen amplitude is fitted, below.
            _fit_entries([e for d in results_by_detector.values() for e in d.values()],
                         fit_method)
        optimal_configs = analyze_multiamp_data(results_by_detector, nonlinear_threshold, fallback_to_lowest)

        # The chosen entries themselves, so the fit and the bias
        # frequency written below land on the caller's results.
        single_results = {}
        for det_idx, config in optimal_configs.items():
            entry = config['selected_data']
            entry['selected_amplitude'] = config['selected_amplitude']
            entry['bifurcation_ever_seen'] = config.get('bifurcation_ever_seen', False)
            single_results[det_idx] = entry

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
                    fit_method=fit_method,
                    measure_calibration=measure_calibration,
                    calibration_step_hz=calibration_step_hz,
                    module=mod_num,
                    progress_callback=progress_callback
                )
            )
        
        return await asyncio.gather(*tasks)
    
    # Single module processing
    if module is None:
        raise ValueError("Module must be specified for single result set")
    
    # Get current NCO frequency
    _fit_entries(list(multisweep_results.values()), fit_method)
    for det_data in multisweep_results.values():
        _bias_point_from_fit(det_data, fit_method)

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
        suitable, is_bifurcated, nonlinear_a, _ = _suitable(det_data, nonlinear_threshold)
        
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
        )
    else:
        # No optimization - all phases are 0
        optimal_phases = {det_idx: (0.0, None) for det_idx in bias_configs}
    
    # Every channel's phase, then the measurement at the point each
    # detector now sits at, in the frame it now has.
    async with crs.tuber_context() as ctx:
        for det_idx, config in bias_configs.items():
            optimal_phase = optimal_phases.get(det_idx, (0.0, None))[0]
            if optimal_phase != 0.0:
                ctx.set_phase(optimal_phase, units=crs.UNITS.DEGREES, target=crs.TARGET.ADC,
                              channel=config['channel'], module=module)
        await ctx()
    measured = {}
    if measure_calibration and bias_configs:
        try:
            measured = await measure_calibrations_by_step(crs, bias_configs, module,
                                                          step_hz=calibration_step_hz)
        except Exception as exc:
            warnings.warn(f"Module {module}: measuring the df calibration failed "
                          f"({exc}); using the fit's")

    disagree = []
    for det_idx, config in bias_configs.items():
        try:
            optimal_phase, phase_std = optimal_phases.get(det_idx, (0.0, None))

            # Copy the original multisweep data and add bias info
            biased_data = multisweep_results[det_idx].copy()
            biased_data['bias_channel'] = config['channel']
            biased_data['bifurcation_suspected'] = config['bifurcation_suspected']
            biased_data['bias_successful'] = True
            biased_data['optimal_phase_degrees'] = optimal_phase
            biased_data['phase_optimization_std'] = phase_std
            
            # The calibration from the best fit this result carries: the
            # flow's nonlinear fit if it ran, else the skewed fit, else
            # what multisweep fitted for it.
            try:
                cal = df_calibration_for_entry(biased_data, prefer=fit_method)
                if cal is not None:
                    biased_data['df_calibration'] = cal
            except Exception as exc:
                warnings.warn(f"Detector {det_idx}: df calibration from the fit "
                              f"failed ({exc}); keeping the multisweep's")

            if 'df_calibration' in biased_data and biased_data['df_calibration'] is not None and optimal_phase != 0.0:
                # The board rotates the samples by +phase; the calibration
                # multiplies them, so it turns the other way to keep
                # samples * calibration the same frequency shift.
                biased_data['df_calibration'] *= np.exp(-1j * np.radians(optimal_phase))
                biased_data['df_calibration_rotated'] = True

            fit_cal = biased_data.get('df_calibration')
            meas_cal = measured.get(det_idx)
            if meas_cal is not None:
                biased_data['df_calibration'] = meas_cal
                biased_data['df_calibration_source'] = "measured"
                if fit_cal is not None:
                    biased_data['df_calibration_fit'] = fit_cal
                    ratio = meas_cal / fit_cal
                    turn = abs(np.degrees(np.angle(ratio)))
                    if turn > 5.0 or not 0.7 < abs(ratio) < 1.4:
                        disagree.append((det_idx, turn, abs(ratio)))
            elif fit_cal is not None:
                biased_data['df_calibration_source'] = "fit"

            successfully_biased[det_idx] = biased_data
            
        except Exception as e:
            warnings.warn(f"Failed to bias detector {det_idx}: {e}")
    
    if disagree:
        worst = max(disagree, key=lambda d: d[1])
        warnings.warn(f"Module {module}: the measured df calibration disagrees "
                      f"with the fit's on {len(disagree)} of {len(bias_configs)} "
                      f"detectors (worst: detector {worst[0]}, {worst[1]:.1f} deg, "
                      f"magnitude ratio {worst[2]:.2f}); the measured one is used")

    # Progress callback
    if progress_callback:
        progress_callback(module, 100.0)
    
    # Log summary
    print(f"Module {module}: Successfully biased {len(successfully_biased)}/{total_detectors} detectors")
    
    return successfully_biased


def analyze_multiamp_data(
    results_by_detector: Dict[int, Dict],
    nonlinear_threshold: float = 0.77,
    fallback_to_lowest: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Analyze multi-amplitude multisweep results to find optimal bias points:
    the highest amplitude whose sweep does not jump and, where the entry
    carries a nonlinear fit, whose nonlinearity is below the threshold.

    Args:
        results_by_detector: Detector-indexed data keyed by iteration index:
            ``{detector_id: {iteration_index: entry_dict}}``
            Each entry_dict contains 'amplitude', 'direction', and sweep data.
        nonlinear_threshold: Maximum acceptable nonlinear parameter
        fallback_to_lowest: Whether to use lowest amplitude as fallback

    Returns:
        Dictionary with detector index as key, optimal configuration as value.
    """
    optimal_configs = {}

    for det_idx, iter_dict in results_by_detector.items():
        amp_analysis = []
        bifurcation_ever_seen = False

        # Iterate entries sorted by amplitude (highest first) for deterministic order
        sorted_entries = sorted(
            iter_dict.values(),
            key=lambda e: (-(e.get('amplitude') or 0), e.get('direction', ''))
        )

        for det_data in sorted_entries:
            amp = det_data.get('amplitude')
            direction = det_data.get('direction', 'upward')
            if amp is None:
                continue

            suitable, is_bifurcated, nonlinear_a, nonlinear_success = _suitable(
                det_data, nonlinear_threshold)
            if is_bifurcated:
                bifurcation_ever_seen = True

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
            optimal = suitable_entries[0]
        elif fallback_to_lowest and amp_analysis:
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
