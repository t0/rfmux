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
# fitting import removed as it's no longer used directly here

@macro(CRS, register=True)
async def multisweep(
    crs: CRS,
    center_frequencies: list[float],
    span_hz: float,
    npoints_per_sweep: int,
    amp: float,
    nsamps: int = 10,
    global_phase_ref_to_zero: bool = True,
    recalculate_center_frequencies: bool = True,
    *,
    module,
    progress_callback=None,
    data_callback=None,
):
    """
    Perform simultaneous, high-resolution frequency sweeps around multiple center frequencies.

    This algorithm dedicates one channel per resonance and sweeps all resonances in parallel.
    Each iteration adjusts all channel frequencies according to the sweep parameters.
    Phase stitching is applied across NCO region boundaries when necessary.

    Args:
        crs (CRS): The CRS object (injected by macro).
        center_frequencies (list[float]): List of center frequencies (Hz) for each resonance.
        span_hz (float): Total frequency width (Hz) of each sweep.
        npoints_per_sweep (int): Number of points to measure within each sweep's span.
        amp (float): Amplitude (normalized DAC units) for all tones.
        nsamps (int, optional): Number of samples to average per frequency point. Defaults to 10.
        global_phase_ref_to_zero (bool, optional): If True, apply a global phase rotation so the first
                                                  measured point has zero phase. Defaults to True.
        recalculate_center_frequencies (bool, optional): If True, the keys of the returned dictionary
                                                       will be the recalculated center frequencies based
                                                       on the minimum S21 magnitude point. Otherwise,
                                                       keys are the original `center_frequencies`.
                                                       Defaults to False.
        module (int | list[int]): The target readout module(s).
        progress_callback (callable, optional): Function called with (module, progress_percentage).
        data_callback (callable, optional): Function called with intermediate results during acquisition.

    Returns:
        dict: A dictionary where keys are center frequencies (float, Hz). If
              `recalculate_center_frequencies` is True, these keys are the recalculated
              frequencies based on minimum S21 magnitude; otherwise, they are the
              original `center_frequencies` passed as input.
              The value for each key is another dictionary containing the results for that sweep:
              {
                  key_freq: { # key_freq is either original or recalculated center frequency
                      'frequencies': np.ndarray (Hz),
                      'iq_complex': np.ndarray (complex) - Original, stitched IQ data,
                      'phase_degrees': np.ndarray (degrees)
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
                global_phase_ref_to_zero=global_phase_ref_to_zero,
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
    max_channels = 1023  # Standard max channels per module
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
    STITCH_CHANNEL = 1024 # Use the last channel for NCO stitching reference

    # --- Generate sweep frequencies and find global minimum frequency ---
    resonance_data = {}
    abs_min_freq = float('inf')
    ref_cf = None
    ref_point_idx = -1

    for cf in center_frequencies:
        # Generate points for this sweep
        sweep_points = np.linspace(
            cf - span_hz / 2,
            cf + span_hz / 2,
            npoints_per_sweep,
            endpoint=True
        )
        # Find the absolute minimum frequency across all sweeps
        current_min_freq = sweep_points[0]
        if current_min_freq < abs_min_freq:
            abs_min_freq = current_min_freq
            ref_cf = cf
            ref_point_idx = 0 # The minimum is always the first point

        resonance_data[cf] = {
            'frequencies': sweep_points,
            'iq_complex': np.zeros(npoints_per_sweep, dtype=np.complex128), # Pre-allocate array
            # 'last_point_for_stitching': None, # No longer used per-resonance
            'is_rotated': np.zeros(npoints_per_sweep, dtype=bool) # Track global rotation status
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

    # --- Check channel limit if stitching is needed ---
    if len(nco_regions) > 1 and len(center_frequencies) > max_channels - 1:
         raise ValueError(
             f"Cannot perform NCO stitching: {len(center_frequencies)} resonances requested, "
             f"but channel {STITCH_CHANNEL} is required for stitching, leaving {max_channels - 1} available."
         )

    # Initialize global phase rotation factor
    global_rotation = None # Will be calculated based on the point at abs_min_freq
    # Initialize stitching factors (Region 0 needs no stitching)
    region_stitch_rotations = {0: 1.0 + 0j}
    # Store the IQ value measured by the stitch channel at the end of the previous region
    iq_stitch_prev_rotated = None

    # --- Calculate all NCO frequencies upfront ---
    # nco_frequencies was already initialized before the loop in the previous step
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

        # --- Calculate Stitching Rotation (Start of Region > 0) ---
        if region_idx > 0:
            # Stitch frequency is midpoint between previous and current NCO
            stitch_freq = (nco_frequencies[region_idx - 1] + current_nco_freq) / 2
            stitch_freq_rel = stitch_freq - current_nco_freq

            # Measure stitch frequency with current NCO using dedicated channel
            async with crs.tuber_context() as ctx:
                ctx.set_frequency(stitch_freq_rel, channel=STITCH_CHANNEL, module=module)
                # Use a fixed amplitude for stitching? Or the sweep amp? Using 'amp' for now.
                ctx.set_amplitude(amp, channel=STITCH_CHANNEL, module=module)
                # Zero out all other channels
                for ch in range(1, max_channels + 1):
                    if ch != STITCH_CHANNEL:
                        ctx.set_amplitude(0, channel=ch, module=module)
                await ctx()

            # Use specific channel argument for get_samples
            samples_stitch_current = await crs.get_samples(nsamps, average=True, channel=STITCH_CHANNEL, module=module)
            # Samples for single channel might be structured differently, assume index 0
            # Check CRS.get_samples documentation if this fails.
            if hasattr(samples_stitch_current.mean, 'i') and samples_stitch_current.mean.i.size > 0:
                 iq_stitch_current_raw = samples_stitch_current.mean.i[0] + 1j * samples_stitch_current.mean.q[0]
            else:
                 warnings.warn(f"No data received from stitch channel {STITCH_CHANNEL} at start of region {region_idx+1}. Cannot calculate stitch factor.")
                 iq_stitch_current_raw = 0 + 0j # Avoid error below

            # Apply global rotation if defined
            iq_stitch_current_rotated = iq_stitch_current_raw * global_rotation if global_rotation is not None else iq_stitch_current_raw

            # Calculate stitch factor using previous region's rotated stitch measurement
            if iq_stitch_prev_rotated is not None and abs(iq_stitch_current_rotated) > 1e-15:
                stitch_rotation = iq_stitch_prev_rotated / iq_stitch_current_rotated
                region_stitch_rotations[region_idx] = stitch_rotation
            else:
                # Only warn if previous IQ was expected (i.e., not None)
                if iq_stitch_prev_rotated is not None:
                    warnings.warn(f"Cannot calculate NCO stitch factor for region {region_idx+1}. "
                                  f"Prev IQ: {iq_stitch_prev_rotated}, Current IQ: {iq_stitch_current_rotated}. Using unity rotation.")
                # If prev_iq was None (e.g. first stitch attempt failed), silently use unity.
                region_stitch_rotations[region_idx] = 1.0 + 0j
        # Else: region_idx == 0, stitch_rotation remains 1.0 (from initialization)

        # --- Sweep Points within the Region ---
        # Create channel mapping for this region's resonances
        channel_mapping = {cf: i+1 for i, cf in enumerate(region_cfs)}
        # Get the stitch rotation factor for this region
        current_stitch_rotation = region_stitch_rotations[region_idx]

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

                # Zero out unused resonance channels AND the stitch channel
                active_res_channels = set(channel_mapping.values())
                for ch in range(1, max_channels + 1):
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
                iq_val_globally_rotated = raw_iq_val # Start with raw value
                rotated_this_step = False # Tracks global rotation application

                # --- Global Phase Rotation Handling ---
                is_reference_point = (cf == ref_cf and point_idx == ref_point_idx)
                if global_phase_ref_to_zero:
                    if is_reference_point:
                        if abs(raw_iq_val) > 1e-15:
                            global_rotation = abs(raw_iq_val) / raw_iq_val
                            iq_val_globally_rotated = raw_iq_val * global_rotation
                            rotated_this_step = True
                        else:
                            global_rotation = 1.0 # Cannot rotate, use unity
                            iq_val_globally_rotated = raw_iq_val
                            warnings.warn(f"Reference point for global phase rotation (freq={abs_min_freq*1e-6:.3f} MHz) has near-zero amplitude. Skipping rotation.")
                    elif global_rotation is not None:
                        # Apply previously calculated global rotation
                        iq_val_globally_rotated = raw_iq_val * global_rotation
                        rotated_this_step = True
                    # Else: global_rotation not yet calculated, keep raw_iq_val for now (will be rotated in post-processing)

                # --- Apply NCO Stitching Rotation ---
                # Apply the rotation factor calculated at the start of this region
                final_iq = iq_val_globally_rotated * current_stitch_rotation

                # Store final processed IQ value
                resonance_data[cf]['iq_complex'][point_idx] = final_iq
                resonance_data[cf]['is_rotated'][point_idx] = rotated_this_step # Mark if globally rotated *in this step*

            # --- Progress update ---
            if progress_callback:
                # Calculate progress as percentage
                region_progress = region_idx / total_nco_regions
                point_progress = point_idx / npoints_per_sweep
                overall_progress = (region_progress + point_progress / total_nco_regions) * 100
                progress_callback(module, overall_progress)
            
            # Call data callback with intermediate results if provided
            if data_callback:
                 # Create intermediate results dictionary (using final processed IQ for this point)
                 intermediate_results = {}
                 for cb_cf in center_frequencies:
                     current_point_count = point_idx + 1 # Include current point
                     frequencies = resonance_data[cb_cf]['frequencies'][:current_point_count]
                     # Need to handle potential post-processing rotation for points before ref point
                     # For simplicity, callback sends data as processed *so far*.
                     # A more accurate callback would require applying pending global rotation here too.
                     iq_data = resonance_data[cb_cf]['iq_complex'][:current_point_count]
                     intermediate_results[cb_cf] = {
                         'frequencies': frequencies,
                         'iq_complex': iq_data
                     }
                 data_callback(module, intermediate_results)

        # --- Measure Stitch Point for Next Region (End of Current Region) ---
        if region_idx < total_nco_regions - 1:
            # Stitch frequency is midpoint between current and next NCO
            next_nco_freq = nco_frequencies[region_idx + 1]
            stitch_freq = (current_nco_freq + next_nco_freq) / 2
            stitch_freq_rel = stitch_freq - current_nco_freq

            # Measure stitch frequency with current NCO using dedicated channel
            async with crs.tuber_context() as ctx:
                ctx.set_frequency(stitch_freq_rel, channel=STITCH_CHANNEL, module=module)
                ctx.set_amplitude(amp, channel=STITCH_CHANNEL, module=module)
                # Zero out all other channels
                for ch in range(1, max_channels + 1):
                    if ch != STITCH_CHANNEL:
                        ctx.set_amplitude(0, channel=ch, module=module)
                await ctx()

            samples_stitch_prev = await crs.get_samples(nsamps, average=True, channel=STITCH_CHANNEL, module=module)
            if hasattr(samples_stitch_prev.mean, 'i') and samples_stitch_prev.mean.i.size > 0:
                 iq_stitch_prev_raw = samples_stitch_prev.mean.i[0] + 1j * samples_stitch_prev.mean.q[0]
            else:
                 warnings.warn(f"No data received from stitch channel {STITCH_CHANNEL} at end of region {region_idx+1}. Stitching to next region may fail.")
                 iq_stitch_prev_raw = 0 + 0j

            # Apply global rotation if defined (use the rotation potentially defined *during* this region)
            iq_stitch_prev_rotated = iq_stitch_prev_raw * global_rotation if global_rotation is not None else iq_stitch_prev_raw
            # This value (iq_stitch_prev_rotated) will be used at the start of the next region loop iteration

    # --- Post-processing: Apply global rotation to points measured *before* the reference point ---
    if global_phase_ref_to_zero and global_rotation is not None:
        for cf in center_frequencies:
            for point_idx in range(npoints_per_sweep):
                # If this point wasn't rotated during the main loop AND rotation is defined
                if not resonance_data[cf]['is_rotated'][point_idx]:
                    resonance_data[cf]['iq_complex'][point_idx] *= global_rotation
                    # No need to update 'is_rotated' here, it's only for the loop logic

    # --- Format final results for each resonance ---
    results_by_center_freq = {}
    for cf, data in resonance_data.items():
        frequencies = data['frequencies']
        iq_complex = data['iq_complex'] # Already a numpy array

        # Calculate phase in degrees
        phase_degrees = np.degrees(np.angle(iq_complex))
        
        # Initialize result dictionary
        result_dict = {
            'frequencies': frequencies,
            'iq_complex': iq_complex,
            'phase_degrees': phase_degrees
        }
        
        current_key_for_results = cf # Default to original center frequency

        if recalculate_center_frequencies:
            if iq_complex.size > 0:
                s21_mag = np.abs(iq_complex)
                if np.any(s21_mag): # Check if there's any non-zero magnitude
                    min_mag_idx = np.argmin(s21_mag)
                    new_center_freq = frequencies[min_mag_idx]
                    if np.isfinite(new_center_freq):
                        current_key_for_results = new_center_freq
                    else:
                        warnings.warn(f"Recalculated center frequency for original cf {cf*1e-6:.3f} MHz (module {module}) is not finite. Using original cf as key.")
                else:
                    # All magnitudes are zero
                    warnings.warn(f"Cannot recalculate center frequency for original cf {cf*1e-6:.3f} MHz (module {module}): all S21 magnitudes are zero. Using original cf as key.")
            else:
                # iq_complex is empty
                warnings.warn(f"Cannot recalculate center frequency for original cf {cf*1e-6:.3f} MHz (module {module}): IQ data is empty. Using original cf as key.")
        
        # Store results using the determined key
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
