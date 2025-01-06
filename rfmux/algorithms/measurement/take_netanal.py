"""
take_netanal: A measurement algorithm that handles the book-keeping of assigning NCO, frequency, and channel
pairings in order to measure the complex S21 across a large bandwidth. Often used for finding resonances.
"""

import warnings
import asyncio
import random
import time
import numpy as np
from rfmux.core.hardware_map import macro
from rfmux.core.schema import ReadoutModule


@macro(ReadoutModule, register=True)
async def take_netanal(
    rmod : ReadoutModule,
    amp: float = 0.001,
    fmin: float = 100e6,
    fmax: float = 2450e6,
    nsamps: int = 10,
    npoints: int = 5000,
    max_chans: int = 1023,
    max_span: float = 500e6,
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
    module : int or list of int, optional
        - If an integer, run one measurement on that module (default is 1).
        - If a list, e.g. [1, 2, 3], run concurrently for each module in the list
          and return a dict keyed by module number.
        - Note -- lists must be within a single analog bank (1-4) or (5-8).

    Returns
    -------
    If `module` is a list:
        list
            A list where each value is a tuple corresponding to the modules in the list:
            (fs_sorted, iq_sorted, phase_sorted).
    Otherwise:
        fs_sorted, iq_sorted, phase_sorted : np.ndarray
        Frequency, I/Q, and phase arrays for the single measurement.
    """

    crs = rmod.crs
    module = rmod.module
    # # If user passed modules as a list, run in parallel across those modules
    # if isinstance(module, list) and len(module) > 0:

    #     # Ensure all modules are in [1..4] or all are in [5..8]
    #     if not module:  # empty list
    #         raise ValueError("Module list is empty.")

    #     # Check if they all lie within 1..4 OR all lie within 5..8
    #     in_first_bank = all(1 <= m <= 4 for m in module)
    #     in_second_bank = all(5 <= m <= 8 for m in module)

    #     if not (in_first_bank or in_second_bank):
    #         raise ValueError(
    #             f"Module list must be entirely in [1..4] or [5..8], got: {module}"
    #         )   
    #     tasks = []
    #     for m in module:
    #         # Call the same macro again, but for a single module=m
    #         tasks.append(crs.take_netanal(
    #             amp=amp,
    #             fmin=fmin,
    #             fmax=fmax,
    #             nsamps=nsamps,
    #             npoints=npoints,
    #             max_chans=max_chans,
    #             max_span=max_span,
    #             module=m,
    #         ))
    #     results = await asyncio.gather(*tasks)
    #     return results
    #     # return {m: result for m, result in zip(module, results)}

    # Generate a global array of frequencies across [fmin, fmax].
    freqs_global = np.linspace(fmin, fmax, npoints, endpoint=True)

    # Warn if total amplitude might exceed rule-of-thumb DAC headroom.
    if max_chans * amp > 3.5:
        warn_msg = (
            f"Total amplitude sum {max_chans * amp:.3f} exceeds crest-factor limit (3.5). "
            "Results may be noisy due to possible DAC clipping."
        )
        warnings.warn(warn_msg)

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

    # Process each chunk.
    for i, (start_idx, end_idx) in enumerate(chunks):
        freqs_chunk = freqs_global[start_idx:end_idx + 1]
        if not len(freqs_chunk):
            continue

        # NCO frequency is the midpoint of the chunk.
        nco_freq = 0.5 * (freqs_chunk[0] + freqs_chunk[-1])
        await crs.set_nco_frequency(nco_freq, module=module)

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
            # WARNING: This does assume users aren't taking data with <50Hz
            #          resolution. If so, will need to get more clever and
            #          apply the edge cases to all relevant frequencies and not
            #          just the extrema.
            ifreqs = np.concatenate([
                [comb[0] - 50 * np.sign(comb[0] - nco_freq) * np.random.random()],
                comb[1:-1] + 100 * (np.random.random(len(comb) - 2) - 0.5),
                [comb[-1] - 50 * np.sign(comb[-1] - nco_freq) * np.random.random()]
            ])

            # TODO: Update this to use context manager with the phase rotation is
            #       in the firmware automatically.
            # for j, freq_val in enumerate(ifreqs, start=1):
            #     chunk_fs.append(freq_val)
            #     await crs.set_frequency(freq_val - nco_freq, channel=j, module=module)
            #     await crs.set_amplitude(amp, channel=j, module=module)
            async with crs.tuber_context() as ctx:    
                for j, freq_val in enumerate(ifreqs, start=1):
                    chunk_fs.append(freq_val)
                    ctx.set_frequency(freq_val - nco_freq, channel=j, module=module)
                    ctx.set_amplitude(amp, channel=j, module=module)
                await ctx()

            # Acquire samples and form complex I/Q.
            samples = await crs.py_get_samples(
                nsamps, average=True, channel=None, module=module
            )
            for ch in range(len(ifreqs)):
                i_val = samples.mean.i[ch]
                q_val = samples.mean.q[ch]
                chunk_iq.append(i_val + 1j * q_val)

            async with crs.tuber_context() as ctx:    
                for j in range(max_chans):
                    ctx.set_amplitude(0, channel=j+1, module=module)
                await ctx()

        # Rotate new NCO chunk so the overlap freq aligns with previous NCO phase.
        if i > 0 and prev_boundary_iq is not None:
            boundary_new = chunk_iq[0]
            if abs(boundary_new) > 1e-15:
                rot = prev_boundary_iq / boundary_new
                # Remove the overlap freq from this chunk's arrays, apply rotation to the rest.
                chunk_iq = [iq_val * rot for iq_val in chunk_iq[1:]]
                chunk_fs = chunk_fs[1:]

        # Update the boundary freq's I/Q for use in the next chunk.
        prev_boundary_iq = chunk_iq[-1]

        # Accumulate into global arrays.
        fs_all.extend(chunk_fs)
        iq_all.extend(chunk_iq)

    fs_all = np.array(fs_all)
    iq_all = np.array(iq_all)
    phase_all = np.degrees(np.angle(iq_all))

    # Sort all interleaved and chunked data by ascending frequency.
    combined = sorted(zip(fs_all, iq_all, phase_all), key=lambda x: x[0])
    fs_sorted, iq_sorted, phase_sorted = zip(*combined)

    return np.array(fs_sorted), np.array(iq_sorted), np.array(phase_sorted)
