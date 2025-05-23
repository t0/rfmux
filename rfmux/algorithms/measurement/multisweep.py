"""
multisweep: A measurement algorithm for performing simultaneous, targeted,
high-resolution frequency sweeps around multiple specified center frequencies.
Optionally fits resonances and centers IQ data.
"""

import numpy as np
import time
import asyncio
import warnings
from collections import defaultdict

from rfmux.core.hardware_map import macro
from rfmux.core.schema import CRS
from typing import Optional, Tuple, Dict, Any # Added for type hinting

def _get_recalculated_center_freq(
    original_cf_hz: float,
    sweep_freqs_hz: np.ndarray,
    sweep_iq: np.ndarray,
    method: Optional[str]
) -> Tuple[float, str]:
    """
    Helper to recalculate center frequency based on sweep data and method.
    Returns the new frequency and the method string actually applied.
    """
    if sweep_iq.size == 0 or sweep_freqs_hz.size == 0:
        return original_cf_hz, "none"

    actual_method_applied = "none"
    new_frequency_hz = original_cf_hz

    if method == "min-s21":
        s21_mag = np.abs(sweep_iq)
        if np.any(s21_mag):
            min_mag_idx = np.argmin(s21_mag)
            recalculated_freq = sweep_freqs_hz[min_mag_idx]
            if np.isfinite(recalculated_freq):
                new_frequency_hz = recalculated_freq
                actual_method_applied = "min-s21"
            else:
                warnings.warn(f"Recalculated 'min-s21' frequency for original_cf {original_cf_hz*1e-6:.3f} MHz is not finite. Using original.")
        else:
            warnings.warn(f"Cannot recalculate 'min-s21' for original_cf {original_cf_hz*1e-6:.3f} MHz: all S21 magnitudes are zero.")
    elif method == "max-dq":
        if sweep_iq.size < 2 or sweep_freqs_hz.size < 2: # Need at least 2 points for gradient
             warnings.warn(f"Cannot recalculate 'max-dq' for original_cf {original_cf_hz*1e-6:.3f} MHz: not enough sweep points ({sweep_iq.size}).")
             return original_cf_hz, "none"
        phase_rad = np.angle(sweep_iq)
        unwrapped_phase_rad = np.unwrap(phase_rad)
        # Ensure frequencies are sorted for gradient, though linspace should ensure this.
        # If frequencies might not be monotonic, sort them and corresponding unwrapped_phase_rad.
        dq_df = np.gradient(unwrapped_phase_rad, sweep_freqs_hz)
        if np.any(dq_df):
            max_abs_dq_df_idx = np.argmax(np.abs(dq_df))
            recalculated_freq = sweep_freqs_hz[max_abs_dq_df_idx]
            if np.isfinite(recalculated_freq):
                new_frequency_hz = recalculated_freq
                actual_method_applied = "max-dq"
            else:
                warnings.warn(f"Recalculated 'max-dq' frequency for original_cf {original_cf_hz*1e-6:.3f} MHz is not finite. Using original.")
        else:
            warnings.warn(f"Cannot recalculate 'max-dq' for original_cf {original_cf_hz*1e-6:.3f} MHz: dQ/df is zero everywhere.")
    elif method is not None:
        warnings.warn(f"Unknown recalculate_center_frequencies method: '{method}'. No recalculation performed.")

    return new_frequency_hz, actual_method_applied

@macro(CRS, register=True)
async def multisweep(
    crs: CRS,
    center_frequencies: list[float],
    span_hz: float,
    npoints_per_sweep: int,
    amp: float,
    nsamps: int = 10,
    recalculate_center_frequencies: Optional[str] = None, # Options: "min-s21", "max-dq", or None
    sweep_direction: str = "upward", # Options: "upward", "downward"
    *,
    module,
    progress_callback=None,
    data_callback=None,
):
    """
    Perform simultaneous, high-resolution frequency sweeps around multiple center frequencies.

    This algorithm dedicates one channel per resonance and sweeps all resonances in parallel.
    The NCO is re-tuned for different groups of resonances (NCO regions) if their
    combined span exceeds the NCO's instantaneous bandwidth. No phase stitching
    is performed between data collected from different NCO regions.
    This version allows for advanced recalculation of center frequencies and
    TOD-based rotation of the sweep data.

    Args:
        crs (CRS): The CRS object (injected by macro).
        center_frequencies (list[float]): List of center frequencies (Hz) for each resonance.
        span_hz (float): Total frequency width (Hz) of each sweep.
        npoints_per_sweep (int): Number of points to measure within each sweep's span.
        amp (float): Amplitude (normalized DAC units) for all tones.
        nsamps (int, optional): Number of samples to average per frequency point for the main sweep.
                                Defaults to 10. The TOD acquisition for rotation uses 1000 samples.
        recalculate_center_frequencies (Optional[str], optional):
            Determines how/if center frequencies are recalculated and how sweep data is rotated.
            - "min-s21": Recalculates center frequency to the point of minimum |S21| in the sweep.
                         Acquires a TOD at this new frequency. Rotates sweep data to minimize
                         the I component of this TOD's mean.
            - "max-dq": Recalculates center frequency to the point of maximum |d(phase)/df| in the sweep.
                        Acquires a TOD at this new frequency. Rotates sweep data to align the
                        principal component of this TOD (maximizing variance) with the I-axis.
            - None: No recalculation. No TOD-based rotation is performed.
            Defaults to None.
        module (int | list[int]): The target readout module(s).
        progress_callback (callable, optional): Function called with (module, progress_percentage).
        data_callback (callable, optional): Function called with intermediate results during acquisition.

    Returns:
        dict: A dictionary where keys are the final center frequencies (float, Hz).
              The nature of this key (original or recalculated) is indicated within the value.
              The value for each key is another dictionary containing:
              {
                  key_freq: {
                      'frequencies': np.ndarray (Hz), # Original sweep frequencies
                      'iq_complex': np.ndarray (complex), # Final, possibly rotated, sweep IQ data
                      'phase_degrees': np.ndarray (degrees), # Derived from final iq_complex
                      'original_center_frequency': float,
                      'recalculation_method_applied': str, # "min-s21", "max-dq", or "none"
                      'key_frequency_is_recalculated': bool,
                      'rotation_tod': Optional[np.ndarray], # 1000-sample IQ TOD, if acquired
                      'applied_rotation_degrees': Optional[float] # Rotation applied to sweep data
                  },
                  ...
              }
              If running on multiple modules, returns a list of these dictionaries.
    """

    # --- Handle parallel execution if module is a list ---
    if isinstance(module, list):
        if not module:
            raise ValueError("Module list cannot be empty.")

        # Ensure all modules are in [1..4] or all are in [5..8]
        in_first_bank = all(1 <= m <= 4 for m in module)
        in_second_bank = all(5 <= m <= 8 for m in module)
        if not (in_first_bank or in_second_bank):
            raise ValueError(
                f"Module list must be entirely in [1..4] or [5..8], got: {module}"
            )

        tasks = []
        for m in module:
            # Call the same macro again, but for a single module=m
            tasks.append(crs.multisweep(
                center_frequencies=center_frequencies,
                span_hz=span_hz,
                npoints_per_sweep=npoints_per_sweep,
                amp=amp,
                nsamps=nsamps,
                recalculate_center_frequencies=recalculate_center_frequencies,
                module=m, # Pass single module here
                progress_callback=progress_callback,
                data_callback=data_callback,
            ))
        # Results will be a list of dictionaries, one per module
        results_list = await asyncio.gather(*tasks)
        return results_list
    # --- End parallel execution handling ---

    # --- Validate inputs for single module execution ---
    if len(center_frequencies) == 0:
        warnings.warn("Received empty list for center_frequencies. Returning empty dictionary.")
        return {}
    
    # Check if number of resonances exceeds maximum channels
    max_channels = 1024  # Standard max channels per module
    if len(center_frequencies) > max_channels:
        raise ValueError(f"Number of resonances ({len(center_frequencies)}) exceeds maximum channels ({max_channels})")
    
    if npoints_per_sweep < 2:
        raise ValueError("npoints_per_sweep must be at least 2.")
    if span_hz <= 0:
        raise ValueError("span_hz must be positive.")
    if amp <= 0:
        warnings.warn(f"Amplitude amp={amp} is non-positive. Results may be invalid.")
    # --- End input validation ---

    # --- Define Constants ---
    MAX_NCO_SPAN_HZ = 500e6

    # --- Generate sweep frequencies ---
    resonance_data = {}

    for cf in center_frequencies:
        # Generate points for this sweep based on direction
        if sweep_direction == "upward":
            sweep_points = np.linspace(
                cf - span_hz / 2,
                cf + span_hz / 2,
                npoints_per_sweep,
                endpoint=True
            )
        elif sweep_direction == "downward":
            sweep_points = np.linspace(
                cf + span_hz / 2,
                cf - span_hz / 2,
                npoints_per_sweep,
                endpoint=True
            )
        else:
            raise ValueError(f"Invalid sweep_direction: {sweep_direction}. Must be 'upward' or 'downward'.")

        resonance_data[cf] = {
            'frequencies': sweep_points,
            'iq_complex': np.zeros(npoints_per_sweep, dtype=np.complex128), # Pre-allocate array
        }

    # --- Group resonances by NCO regions ---
    nco_regions = []
    nco_frequencies = [] # Store NCO freq for each region
    sorted_cfs = sorted(center_frequencies)

    # Handle special case of a single resonance
    if len(sorted_cfs) == 1:
        nco_regions.append(sorted_cfs)
    else:
        current_region = [sorted_cfs[0]]
        region_min = sorted_cfs[0] - span_hz/2
        region_max = sorted_cfs[0] + span_hz/2
        
        for i in range(1, len(sorted_cfs)):
            cf = sorted_cfs[i]
            cf_min = cf - span_hz/2
            cf_max = cf + span_hz/2

            # If adding this CF would exceed max NCO span, start a new region
            if cf_max - region_min > MAX_NCO_SPAN_HZ:
                nco_regions.append(current_region)
                current_region = [cf]
                region_min = cf_min
                region_max = cf_max
            else:
                # Add to current region and update bounds
                current_region.append(cf)
                region_max = max(region_max, cf_max)
        
        # Add the last region
        if current_region:
            nco_regions.append(current_region)


    # --- Calculate all NCO frequencies upfront ---
    for region_cfs_calc in nco_regions:
        region_min_f = min(cf - span_hz/2 for cf in region_cfs_calc)
        region_max_f = max(cf + span_hz/2 for cf in region_cfs_calc)
        nco_frequencies.append((region_min_f + region_max_f) / 2)

    # --- Measurement Loop ---
    total_nco_regions = len(nco_regions)

    for region_idx, region_cfs in enumerate(nco_regions):
        # --- Set Current NCO Frequency ---
        current_nco_freq = nco_frequencies[region_idx]
        await crs.set_nco_frequency(current_nco_freq, module=module)


        # --- Sweep Points within the Region ---
        # Create channel mapping for this region's resonances
        channel_mapping = {cf: i+1 for i, cf in enumerate(region_cfs)}

        # Loop through sweep points
        for point_idx in range(npoints_per_sweep):
            # Configure resonance channels for this sweep point
            async with crs.tuber_context() as ctx:
                # Set resonance channels
                for cf in region_cfs:
                    channel = channel_mapping[cf]
                    freq = resonance_data[cf]['frequencies'][point_idx]
                    freq_rel = freq - current_nco_freq # Use current_nco_freq
                    ctx.set_frequency(freq_rel, channel=channel, module=module)
                    ctx.set_amplitude(amp, channel=channel, module=module)

                # Zero out unused resonance channels
                active_res_channels = set(channel_mapping.values())
                for ch in range(1, max_channels + 1): # max_channels still relevant for general channel count
                     if ch not in active_res_channels:
                          ctx.set_amplitude(0, channel=ch, module=module) # Zeros freq implicitly if amp=0
                await ctx()

            # Acquire samples for all active resonance channels
            samples = await crs.get_samples(nsamps, average=True, channel=None, module=module)

            # Process samples for each resonance in this region
            for cf in region_cfs:
                channel_idx = channel_mapping[cf] - 1 # 0-based index
                # Get raw IQ
                i_val = samples.mean.i[channel_idx]
                q_val = samples.mean.q[channel_idx]
                raw_iq_val = i_val + 1j * q_val
                
                # Store raw IQ value directly
                resonance_data[cf]['iq_complex'][point_idx] = raw_iq_val

            # --- Progress update ---
            if progress_callback:
                # Calculate progress as percentage
                region_progress = region_idx / total_nco_regions
                point_progress = point_idx / npoints_per_sweep
                overall_progress = (region_progress + point_progress / total_nco_regions) * 100
                progress_callback(module, overall_progress)
            
            # Call data callback with intermediate results if provided
            if data_callback:
                 # Create intermediate results dictionary (using raw IQ for this point)
                 intermediate_results = {}
                 for cb_cf in center_frequencies: # Iterate over all original center_frequencies
                     # Check if this cf is part of the current region_cfs
                     # and if data has been populated up to point_idx
                     if cb_cf in resonance_data and resonance_data[cb_cf]['iq_complex'].size >= point_idx + 1:
                         current_point_count = point_idx + 1 # Include current point
                         frequencies_cb = resonance_data[cb_cf]['frequencies'][:current_point_count]
                         iq_data_cb = resonance_data[cb_cf]['iq_complex'][:current_point_count]
                         intermediate_results[cb_cf] = {
                             'frequencies': frequencies_cb,
                             'iq_complex': iq_data_cb
                         }
                 if intermediate_results: # Only call if there's data to send
                    data_callback(module, intermediate_results)
        
        # --- Post-Sweep Processing for this NCO Region (TOD acquisition and rotation) ---
        if recalculate_center_frequencies is not None: # Only proceed if a method is specified
            # Step 1: Determine all final_center_freq and methods for this region
            resonances_needing_tod = [] # List of (cf_original, final_center_freq, channel_hw)
            for cf_original in region_cfs:
                res_data_entry = resonance_data[cf_original]
                final_center_freq, recalc_method_used = _get_recalculated_center_freq(
                    original_cf_hz=cf_original,
                    sweep_freqs_hz=res_data_entry['frequencies'],
                    sweep_iq=res_data_entry['iq_complex'], # Use raw sweep IQ for recalc
                    method=recalculate_center_frequencies
                )
                res_data_entry['final_center_frequency'] = final_center_freq
                res_data_entry['recalculation_method_applied'] = recalc_method_used
                res_data_entry['rotation_tod'] = None # Initialize
                res_data_entry['applied_rotation_degrees'] = 0.0 # Initialize

                if recalc_method_used in ["min-s21", "max-dq"]:
                    resonances_needing_tod.append({
                        "original_cf": cf_original,
                        "final_cf": final_center_freq,
                        "channel_hw": channel_mapping[cf_original], # 1-based hardware channel
                        "recalc_method": recalc_method_used
                    })
            
            # Step 2: Acquire TODs in batch if any are needed
            if resonances_needing_tod:
                async with crs.tuber_context() as ctx:
                    # Turn off all channels first
                    for ch_iter in range(1, max_channels + 1):
                        ctx.set_amplitude(0, channel=ch_iter, module=module)
                    
                    # Set up channels that need TOD
                    for res_info in resonances_needing_tod:
                        freq_for_tod_rel = res_info["final_cf"] - current_nco_freq
                        ctx.set_frequency(freq_for_tod_rel, channel=res_info["channel_hw"], module=module)
                        ctx.set_amplitude(amp, channel=res_info["channel_hw"], module=module)
                    await ctx()

                try:
                    # Acquire all TODs simultaneously
                    all_tod_samples = await crs.get_samples(50, average=False, channel=None, module=module)
                    
                    # Distribute TODs to respective resonance_data entries
                    for res_info in resonances_needing_tod:
                        channel_idx_0based = res_info["channel_hw"] - 1
                        tod_i_channel_data = np.array(all_tod_samples.i[channel_idx_0based])
                        tod_q_channel_data = np.array(all_tod_samples.q[channel_idx_0based])
                        resonance_data[res_info["original_cf"]]['rotation_tod'] = tod_i_channel_data + 1j * tod_q_channel_data
                except Exception as e:
                    warnings.warn(f"Batch TOD acquisition failed for NCO region {region_idx} (module {module}): {e}")
                    # Mark all relevant TODs as None if batch failed
                    for res_info in resonances_needing_tod:
                        resonance_data[res_info["original_cf"]]['rotation_tod'] = None
            
            # Step 3: Calculate and apply rotations using the (now populated) TODs
            for cf_original in region_cfs: # Iterate again to apply rotations
                res_data_entry = resonance_data[cf_original]
                recalc_method_used = res_data_entry['recalculation_method_applied']
                
                if recalc_method_used not in ["min-s21", "max-dq"] or res_data_entry['rotation_tod'] is None:
                    # Ensure these are set if no TOD-based rotation happens
                    if res_data_entry.get('final_center_frequency') is None : # Should have been set in step 1
                         res_data_entry['final_center_frequency'] = cf_original
                    if res_data_entry.get('recalculation_method_applied') is None:
                         res_data_entry['recalculation_method_applied'] = "none"
                    res_data_entry['rotation_tod'] = None # Ensure it's None
                    res_data_entry['applied_rotation_degrees'] = 0.0
                    continue # Skip to next resonance if no valid method or no TOD

                rotation_angle_rad = 0.0
                tod_iq = res_data_entry['rotation_tod']

                if tod_iq.size > 0: # Check if TOD has data
                    if recalc_method_used == "min-s21":
                        mean_tod = np.mean(tod_iq)
                        rotation_angle_rad = (np.pi / 2) - np.angle(mean_tod)
                    elif recalc_method_used == "max-dq":
                        if tod_iq.size > 1:
                            data_matrix = np.vstack((tod_iq.real, tod_iq.imag))
                            try:
                                covariance_matrix = np.cov(data_matrix)
                                if np.all(np.isfinite(covariance_matrix)):
                                    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
                                    # Eigenvectors are columns, eigenvector corresponding to largest eigenvalue is pc1
                                    pc1_idx = np.argmax(eigenvalues)
                                    pc1_vector_complex = eigenvectors[0, pc1_idx] + 1j * eigenvectors[1, pc1_idx]
                                    rotation_angle_rad = -np.angle(pc1_vector_complex)
                                else:
                                    warnings.warn(f"Covariance matrix for 'max-dq' rotation of cf {cf_original*1e-6:.3f} MHz contains non-finite values. Skipping rotation.")
                            except np.linalg.LinAlgError:
                                 warnings.warn(f"PCA failed for 'max-dq' rotation of cf {cf_original*1e-6:.3f} MHz. Skipping rotation.")
                        else:
                            warnings.warn(f"Not enough TOD points for 'max-dq' PCA rotation for cf {cf_original*1e-6:.3f} MHz ({tod_iq.size} points). Skipping rotation.")
                
                if rotation_angle_rad != 0.0:
                    rotation_factor = np.exp(1j * rotation_angle_rad)
                    res_data_entry['iq_complex'] *= rotation_factor
                    
                    # Also rotate the TOD if it exists and is not empty
                    if res_data_entry['rotation_tod'] is not None and res_data_entry['rotation_tod'].size > 0:
                        res_data_entry['rotation_tod'] *= rotation_factor
                        
                    res_data_entry['applied_rotation_degrees'] = np.degrees(rotation_angle_rad)
                # If no rotation was calculated (e.g. TOD was empty or issues), applied_rotation_degrees remains 0.0

    # --- Format final results for each resonance ---
    results_by_center_freq = {}
    for original_cf, data_entry in resonance_data.items():
        final_iq_complex = data_entry['iq_complex']
        
        # Determine the key for the output dictionary
        current_key_for_results = data_entry.get('final_center_frequency', original_cf)
        recalc_method_applied = data_entry.get('recalculation_method_applied', "none")
        key_is_recalculated = (current_key_for_results != original_cf) and (recalc_method_applied != "none")

        result_dict = {
            'frequencies': data_entry['frequencies'],
            'iq_complex': final_iq_complex,
            'phase_degrees': np.degrees(np.angle(final_iq_complex)),
            'original_center_frequency': original_cf,
            'recalculation_method_applied': recalc_method_applied,
            'key_frequency_is_recalculated': key_is_recalculated,
            'rotation_tod': data_entry.get('rotation_tod'),
            'applied_rotation_degrees': data_entry.get('applied_rotation_degrees'),
            'sweep_direction': sweep_direction
        }
        results_by_center_freq[current_key_for_results] = result_dict
    
    # --- Hardware Cleanup ---
    try:
        async with crs.tuber_context() as ctx:
            # Zero out all potentially used channels
            for ch in range(1, max_channels + 1):
                ctx.set_amplitude(0, channel=ch, module=module)
            await ctx()
    except Exception as e:
        warnings.warn(f"Hardware cleanup failed for module {module}: {e}")
    
    return results_by_center_freq
