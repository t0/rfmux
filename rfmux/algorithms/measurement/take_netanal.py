"""Measure a wideband complex S21 trace by assigning NCOs and channel tones.

Returns ``{module_id: block}``, with the trace in ``block["results"]`` and
measurement settings in ``block["call_params"]``.
"""

import warnings
import asyncio
import numpy as np
from ...core.dac_scale import dac_scale_dbm
from ...core.hardware_map import macro
from ...core.schema import CRS
from ...core.transferfunctions import (
    CREST_FACTOR,
    convert_dacunits_to_dbm,
    convert_roc_to_volts,
)
from ...tuning import store
from ...tuning.sweep_results import merge_modules, pack_netanal, resolve_direction


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
    sweep_direction: str = "upward",
    module,
    progress_callback=None,
    data_callback=None,
    save=None,
    label=None,
):
    """Measure complex S21 across [fmin, fmax] in one sweep direction.

    Bands wider than ``max_span`` use multiple NCO settings. There is no phase
    stitching between them. Returned frequencies are monotonic in the sweep
    direction, although tones within each band are acquired interleaved.

    Args:
        crs: CRS instance, supplied by the macro.
        amp: per-tone amplitude as a fraction of DAC full scale.
        fmin: lower frequency limit, Hz.
        fmax: upper frequency limit, Hz.
        nsamps: samples averaged per measurement.
        npoints: total frequency points.
        max_chans: maximum tones per comb iteration.
        max_span: maximum bandwidth per NCO setting, Hz.
        rotate_phase_to_0: rotate the trace so the first point in the sweep
            direction has zero phase.
        sweep_direction: ``"upward"`` or ``"downward"``; one per call.
        module: one module or a list within one analog bank (1–4 or 5–8).
            Multiple modules are measured concurrently.
        progress_callback: called with ``(module, percent)``.
        data_callback: called with ``(module, partial)`` during acquisition;
            partial data holds ``frequencies`` and ``iq_counts``.
        save: save one file when the measurement completes. None uses
            ``store.autosave_enabled()``.
        label: label appended to the filename when saving.

    Returns:
        dict: ``{module_id: block}``, keyed by ``crs.module[m].index()``.
        ``block["results"]`` holds ``frequencies`` (Hz), complex ``iq_counts``
        and ``iq_volts``, ``sweep_amplitude``, ``sweep_amplitude_dbm``, and
        ``sweep_direction``.
        Use :func:`rfmux.tuning.netanal_trace` to read it or
        :func:`rfmux.tuning.find_resonances_in_netanal` to search for dips.
    """
    sweep_direction = resolve_direction(sweep_direction)

    # What call_params records: the argument as passed. The fan-out below calls
    # this macro again with a single module, so each module's output ends up
    # recording the module it really is rather than the list that produced it.
    requested_module = module

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
                sweep_direction=sweep_direction,
                module=m,
                progress_callback=progress_callback,
                data_callback=data_callback,
                # The per-module calls do not save. One call is one file, so
                # the fan-out saves once, below, over everything it gathered.
                save=False,
            ))
        # Each of those returns a container of its own, keyed by module, so the
        # several modules merge into one rather than stacking into a list whose
        # order was the only thing saying which element was which.
        merged = merge_modules(await asyncio.gather(*tasks))
        store.maybe_save(merged, "netanal", save=save, label=label)
        return merged

    # Generate a global array of frequencies across [fmin, fmax].
    freqs_global = np.linspace(fmin, fmax, npoints, endpoint=True)

    # The comb's peak against DAC full scale (1.0): with random phases
    # the rms is amp * sqrt(N / 2) and the peak about CREST_FACTOR times
    # that; the coherent sum N * amp bounds it for a few tones.
    peak = min(max_chans * amp, CREST_FACTOR * amp * np.sqrt(max_chans / 2))
    if peak > 1.0:
        warnings.warn(
            f"{max_chans} tones at {amp:g} reach an estimated {peak:.2f} of DAC "
            f"full scale (crest factor {CREST_FACTOR}); results may be noisy "
            "from clipping.")
    
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
    #
    # Chunks partition the frequency list: each starts one point past where the
    # last ended. They used to share a boundary frequency, measured twice so
    # that the phase of one chunk could be rotated onto the previous one's at
    # the point they had in common. That stitch is gone, and with it the reason
    # to measure any frequency twice.
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

        i_start = i_end + 1

    # Downward measures the same chunks in the opposite order. A chunk is a band
    # of the grid and an NCO setting, neither of which the direction changes;
    # what it changes is which end the measurement starts at, and the order the
    # tones inside a chunk are programmed in.
    if sweep_direction == "downward":
        chunks.reverse()

    # Prepare arrays for final data across all chunks.
    fs_all, iq_all = [], []
    first_point_rotation = None  # Store rotation to set first point phase to 0
    first_data_point = None  # Store the very first data point for rotation reference

    # Process each chunk.
    total_chunks = len(chunks)
    
    for i, (start_idx, end_idx) in enumerate(chunks):
        freqs_chunk = freqs_global[start_idx:end_idx + 1]
        if sweep_direction == "downward":
            freqs_chunk = freqs_chunk[::-1]
        if not len(freqs_chunk):
            continue

        # NCO frequency is the midpoint of the chunk.
        nco_freq = 0.5 * (freqs_chunk[0] + freqs_chunk[-1])
        await crs.set_nco_frequency(nco_freq, module=module)

        # Track data at the chunk level
        chunk_fs_full, chunk_iq_full = [], []

        # The frequencies of one comb, collected while they are programmed.
        chunk_fs = []
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

            # Clear the temporary array for the next comb
            chunk_fs = []


            if data_callback and chunk_fs_full:
                data_callback(module, {
                    'frequencies': np.array(fs_all + chunk_fs_full),
                    'iq_counts': np.array(iq_all + chunk_iq_full,
                                          dtype=np.complex128),
                })

            # Report progress
            if progress_callback:
                progress = ((i * niter + it + 1) / (total_chunks * niter)) * 100
                progress_callback(module, progress)

        # Accumulate into global arrays.
        fs_all.extend(chunk_fs_full)
        iq_all.extend(chunk_iq_full)

        # Report the data this chunk added.
        if data_callback:
            data_callback(module, {
                'frequencies': np.array(fs_all),
                'iq_counts': np.array(iq_all, dtype=np.complex128),
            })

    # Clean up before exiting
    async with crs.tuber_context() as ctx:
        for j in range(max_chans):
            ctx.set_amplitude(0, channel=j+1, module=module)
        await ctx()

    fs_all_np = np.array(fs_all)
    iq_all_np = np.array(iq_all, dtype=np.complex128)

    # Sorted in the direction that was swept, not by the order the comb happened
    # to take the tones in: within a chunk they are measured interleaved, so
    # acquisition order is a stride pattern every reader would have to undo. A
    # downward netanal therefore comes back descending, which is what a downward
    # multisweep sweep does and what 'sweep_direction' beside the arrays says.
    sort_indices = np.argsort(fs_all_np)
    if sweep_direction == "downward":
        sort_indices = sort_indices[::-1]
    fs_sorted = fs_all_np[sort_indices]
    iq_sorted = iq_all_np[sort_indices]

    # An empty measurement is still a well-formed result, with empty arrays in
    # it. A bare {} would be indistinguishable from a caller's own empty dict,
    # and the provenance of a netanal that measured nothing is worth as much as
    # any other's.
    scale_dbm = await dac_scale_dbm(crs, module)
    netanal = pack_netanal(
        {
            'frequencies': fs_sorted,
            'iq_counts': iq_sorted,
            'iq_volts': convert_roc_to_volts(iq_sorted),
            'sweep_amplitude': amp,
            'sweep_amplitude_dbm': (
                None if scale_dbm is None else
                float(convert_dacunits_to_dbm(amp, scale_dbm))
            ),
            'sweep_direction': sweep_direction,
        },
        module_id=crs.module[module].index(),
        module=module,
        amp=amp,
        fmin=fmin,
        fmax=fmax,
        npoints=npoints,
        nsamps=nsamps,
        max_chans=max_chans,
        max_span=max_span,
        rotate_phase_to_0=rotate_phase_to_0,
        sweep_direction=sweep_direction,
        requested_module=requested_module,
        # Read here rather than left to whoever opens the file: the amplitudes
        # above are fractions of DAC full scale, and the board that says what
        # full scale is worth is this one, now.
        dac_scale_dbm=scale_dbm,
    )

    store.maybe_save(netanal, "netanal", save=save, label=label)
    return netanal

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
