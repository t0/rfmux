"""
take_netanal: A measurement algorithm that handles the book-keeping of assigning NCO, frequency, and channel
pairings in order to measure the complex S21 across a large bandwidth. Often used for finding resonances.
"""

import warnings
import asyncio
import numpy as np
import scipy.signal as signal
from rfmux.core.hardware_map import macro
from rfmux.core.schema import CRS


@macro(CRS, register=True)
async def take_netanal(
    crs : CRS,
    amp: float = 0.001,
    fmin: float = 100e6,
    fmax: float = 2450e6,
    nsamps: int = 10,
    npoints: int = 5000,
    max_chans: int = 1023,
    max_span: float = 500e6,
    rotate_phase_to_0: bool = True,
    *,
    module,
    progress_callback=None,
    data_callback=None,
):
    """
    Perform a network analysis over the frequency range [fmin, fmax].
    Returns the frequencies and complex amplitude and phase at each frequencies.
    The sweep is divided into sub-ranges (chunks) whenever the span exceeds `max_span`.
    Each chunk is associated with a single NCO setting (midpoint of the chunk).

    Exactly one frequency overlaps between consecutive chunks, which is used to
    compute a phase rotation so that the entire measurement is aligned across
    multiple NCO settings.

    Parameters
    ----------
    crs : CRS
        The CRS object used for hardware communication (injected by the macro).
    amp : float, optional
        Amplitude to set on each channel frequency, by default 0.001.
    fmin : float, optional
        Start frequency in Hz, by default 100e6.
    fmax : float, optional
        Stop frequency in Hz, by default 2450e6.
    nsamps : int, optional
        Number of samples to acquire (averaged) per measurement, by default 10.
    npoints : int, optional
        Number of total points across [fmin, fmax], by default 5000.
    max_chans : int, optional
        Maximum number of channels (frequencies) measured per comb iteration,
        by default 1023.
    max_span : float, optional
        Maximum span (Hz) per NCO setting, defaults to the droop-free (non-extended) range of 500MHz.
    rotate_phase_to_0 : bool, optional
        If True, applies an arbitrary global phase rotation to make the first point
        have zero phase. This makes it easier to compare phase responses across
        different measurements, by default True.
    module : int or list of int
        - If an integer, run one measurement on that module.
        - If a list, e.g. [1, 2, 3], run concurrently for each module in the list
          and return a dict keyed by module number.
        - Note -- lists must be within a single analog bank (1-4) or (5-8).
    progress_callback : callable, optional
        Callback function that receives (module, progress_percentage) updates.
    data_callback : callable, optional
        Callback function that receives (module, freqs, amps, phases) updates.

    Returns
    -------
    dict
        A dictionary containing the measurement results.
        Keys:
        - 'frequencies': numpy.ndarray - Sorted frequency points (Hz).
        - 'iq_complex': numpy.ndarray - Complex I/Q data corresponding to 'frequencies'.
        - 'phase_degrees': numpy.ndarray - Phase in degrees corresponding to 'frequencies'.
    """

    # If user passed modules as a list, run in parallel across those modules
    if isinstance(module, list) and len(module) > 0:

        # Ensure all modules are in [1..4] or all are in [5..8]
        if not module:  # empty list
            raise ValueError("Module list is empty.")

        # Check if they all lie within 1..4 OR all lie within 5..8
        in_first_bank = all(1 <= m <= 4 for m in module)
        in_second_bank = all(5 <= m <= 8 for m in module)

        if not (in_first_bank or in_second_bank):
            raise ValueError(
                f"Module list must be entirely in [1..4] or [5..8], got: {module}"
            )
        tasks = []
        for m in module:
            # Call the same macro again, but for a single module=m
            tasks.append(crs.take_netanal(
                amp=amp,
                fmin=fmin,
                fmax=fmax,
                nsamps=nsamps,
                npoints=npoints,
                max_chans=max_chans,
                max_span=max_span,
                rotate_phase_to_0=rotate_phase_to_0,
                module=m,
                progress_callback=progress_callback,
                data_callback=data_callback,
            ))
        results = await asyncio.gather(*tasks)
        return results

    # Generate a global array of frequencies across [fmin, fmax].
    freqs_global = np.linspace(fmin, fmax, npoints, endpoint=True)

    # Warn if total amplitude might exceed rule-of-thumb DAC headroom.
    if max_chans * amp > 3.5:
        warn_msg = (
            f"Total amplitude sum {max_chans * amp:.3f} exceeds crest-factor limit (3.5). "
            "Results may be noisy due to possible DAC clipping."
        )
        warnings.warn(warn_msg)
    
    # Check actual available channels by doing a simple get_samples
    test_samples = await crs.get_samples(1, average=True, channel=None, module=module)
    available_channels = len(test_samples.mean.i) if hasattr(test_samples.mean, 'i') else 0
    
    if max_chans > available_channels:
        error_msg = (
            f"Requested {max_chans} channels, but only {available_channels} channels are available.\n"
            f"This appears to be due to decimation stage settings (short=True limits to 128 channels).\n"
            f"To fix this, either:\n"
            f"  1. Reduce max_chans to {available_channels} or less\n"
            f"  2. Use crs.set_decimation() with the short=False argument to enable up to 1024 channels"
        )
        raise ValueError(error_msg)

    # Identify NCO chunk boundaries by stepping up to max_span each time.
    chunks = []
    i_start = 0
    while i_start < npoints:
        # Candidate end is start freq + max_span.
        f_candidate_stop = freqs_global[i_start] + max_span
        if f_candidate_stop > fmax:
            f_candidate_stop = fmax

        # Find the largest index i_end where freqs_global[i_end] <= f_candidate_stop.
        i_end = np.searchsorted(freqs_global, f_candidate_stop, side='right') - 1
        if i_end <= i_start:
            i_end = i_start

        # Record this chunk (including the boundary freq at i_end).
        chunks.append((i_start, i_end))

        # If we've reached the final point, break.
        if i_end >= npoints - 1:
            break

        # Next chunk reuses the boundary freq at i_end as its first freq.
        i_start = i_end

    # Prepare arrays for final data across all chunks.
    fs_all, iq_all = [], []
    prev_boundary_iq = None  # To store I/Q of the overlap freq from previous chunk.
    first_point_rotation = None  # Store rotation to set first point phase to 0
    first_data_point = None  # Store the very first data point for rotation reference

    # Process each chunk.
    total_chunks = len(chunks)
    
    for i, (start_idx, end_idx) in enumerate(chunks):
        freqs_chunk = freqs_global[start_idx:end_idx + 1]
        if not len(freqs_chunk):
            continue

        # NCO frequency is the midpoint of the chunk.
        nco_freq = 0.5 * (freqs_chunk[0] + freqs_chunk[-1])
        await crs.set_nco_frequency(nco_freq, module=module)

        # Track data at the chunk level
        chunk_fs_full, chunk_iq_full = [], []        

        # Arrays to collect data from this chunk before optional rotation.
        chunk_fs, chunk_iq = [], []
        n_chunk_points = len(freqs_chunk)
        niter = int(np.ceil(n_chunk_points / max_chans))

        # Build comb groups within this chunk.
        for it in range(niter):
            idx_local = it + np.arange(max_chans) * niter
            idx_local = idx_local[idx_local < n_chunk_points]
            if not len(idx_local):
                break

            comb = freqs_chunk[idx_local]

            # Add random offsets to dither IMD tones
            # but ensure they don't exceed NCO bandwidth at the extrema.
            ifreqs = _safe_concatenate_frequencies(comb, nco_freq)

            # Not every internal loop has to use the same number of channels.
            # This block ensures the unused ones are zeroed WHILE programming
            # the others, and avoids zeroing channels again inside the inner loop.
            async with crs.tuber_context() as ctx:
                for j in range(1, max_chans + 1):
                    if j <= len(ifreqs):
                        freq_val = ifreqs[j - 1]
                        # Record which freq is going to channel j
                        chunk_fs.append(freq_val)
                        # Set amplitude/frequency for this used channel
                        ctx.set_frequency(freq_val - nco_freq, channel=j, module=module)
                        if not it:  # only set amplitude once per chunk
                            ctx.set_amplitude(amp, channel=j, module=module)
                    else:
                        if not it: # only zero unused channels once per chunk
                            # Zero out all leftover channels
                            ctx.set_frequency(0, channel=j, module=module)
                            ctx.set_amplitude(0, channel=j, module=module)

                await ctx()

            # Acquire samples and form complex I/Q.
            samples = await crs.get_samples(
                nsamps, average=True, channel=None, module=module
            )
            new_iq_points = []
            for ch in range(len(ifreqs)):
                i_val = samples.mean.i[ch]
                q_val = samples.mean.q[ch]
                iq_val = i_val + 1j * q_val
                
                # Store the very first data point if we haven't seen one yet
                if i == 0 and it == 0 and ch == 0 and first_data_point is None:
                    first_data_point = iq_val
                    # Calculate the phase rotation factor (only once for the entire dataset)
                    if rotate_phase_to_0 and abs(first_data_point) > 1e-15:
                        first_point_rotation = abs(first_data_point) / first_data_point
                
                # Apply the rotation to this point if needed
                if rotate_phase_to_0 and first_point_rotation is not None:
                    iq_val = iq_val * first_point_rotation
                    
                new_iq_points.append(iq_val)
                
            # Add the frequency points
            chunk_fs_full.extend(chunk_fs)
            # Add the (potentially rotated) IQ points
            chunk_iq_full.extend(new_iq_points)
            
            # Clear the temporary arrays for the next iteration
            chunk_fs = []
            chunk_iq = []
            
            if data_callback and chunk_fs_full:
                fs_array = np.array(fs_all + chunk_fs_full)
                iq_array = np.array(iq_all + chunk_iq_full)
                amp_array = np.abs(iq_array)
                phase_array = np.degrees(np.angle(iq_array))
                data_callback(module, fs_array, amp_array, phase_array)

            # Report progress
            if progress_callback:
                progress = ((i * niter + it + 1) / (total_chunks * niter)) * 100
                progress_callback(module, progress)

        # Rotate new NCO chunk so the overlap freq aligns with previous NCO phase.
        if i > 0 and prev_boundary_iq is not None:
            boundary_new = chunk_iq_full[0]
            if abs(boundary_new) > 1e-15:
                rot = prev_boundary_iq / boundary_new
                # Apply rotation to all but the first point (overlap point)
                rotated_iq = [chunk_iq_full[0]] + [iq_val * rot for iq_val in chunk_iq_full[1:]]
                # Now remove the overlap point
                chunk_iq_full = rotated_iq[1:]
                chunk_fs_full = chunk_fs_full[1:]

        # Update the boundary freq's I/Q for use in the next chunk.
        if chunk_iq_full:
            prev_boundary_iq = chunk_iq_full[-1]

        # Accumulate into global arrays.
        fs_all.extend(chunk_fs_full)
        iq_all.extend(chunk_iq_full)
        
        # Report final data update after rotation
        if data_callback:
            fs_array = np.array(fs_all)
            iq_array = np.array(iq_all)
            amp_array = np.abs(iq_array)
            phase_array = np.degrees(np.angle(iq_array))
            data_callback(module, fs_array, amp_array, phase_array)

    # Clean up before exiting
    async with crs.tuber_context() as ctx:
        for j in range(max_chans):
            ctx.set_amplitude(0, channel=j+1, module=module)
        await ctx()

    fs_all_np = np.array(fs_all)
    iq_all_np = np.array(iq_all)

    if len(fs_all_np) == 0: # Handle empty data
        # Return arrays in the expected dictionary structure
        return {
            'frequencies': np.array([]),
            'iq_complex': np.array([]),
            'phase_degrees': np.array([])
        }

    sort_indices = np.argsort(fs_all_np)
    fs_sorted = fs_all_np[sort_indices]
    iq_sorted = iq_all_np[sort_indices]
    phase_sorted = np.degrees(np.angle(iq_sorted))

    result_dict = {
        'frequencies': fs_sorted,
        'iq_complex': iq_sorted,
        'phase_degrees': phase_sorted
    }
            
    return result_dict

def _safe_concatenate_frequencies(comb, nco_freq):
    """
    Safely concatenate frequency arrays with dithering, handling edge cases
    with small numbers of elements.
    
    Parameters
    ----------
    comb : ndarray
        Array of frequencies to dither
    nco_freq : float
        NCO frequency reference
        
    Returns
    -------
    ndarray
        Array of dithered frequencies
    """
    if len(comb) == 0:
        return np.array([])
    
    if len(comb) == 1:
        # Only one frequency - just dither it slightly
        return np.array([comb[0] - 50 * np.sign(comb[0] - nco_freq) * np.random.random()])
    
    if len(comb) == 2:
        # Two frequencies - dither both as edge cases
        return np.array([
            comb[0] - 50 * np.sign(comb[0] - nco_freq) * np.random.random(),
            comb[1] - 50 * np.sign(comb[1] - nco_freq) * np.random.random()
        ])
    
    # Normal case with more than 2 frequencies
    return np.concatenate([
        [comb[0] - 50 * np.sign(comb[0] - nco_freq) * np.random.random()],
        comb[1:-1] + 100 * (np.random.random(len(comb) - 2) - 0.5),
        [comb[-1] - 50 * np.sign(comb[-1] - nco_freq) * np.random.random()]
    ])
