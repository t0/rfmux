"""Processing helpers for measured noise timestreams."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal.windows import chebwin

from ..core.resonators import ResonatorCatalog
from ..core.transferfunctions import (
    TERMINATION,
    VOLTS_PER_ROC,
    convert_dbm_to_volts,
    spectrum_from_slow_tod,
)
from . import store


_SPECTRUM_KEYS = ("psd_i", "psd_q", "psd_dual_sideband")


def separate_iq_fft_to_i_and_q_linear(freqs, iqfft, fs, U):
    """Reconstruct single-sideband I and Q power spectra from an IQ FFT.

    ``iqfft`` is an unshifted FFT of windowed complex voltage samples.
    ``fs`` is the sample rate in Hz, and ``U`` is ``sum(window**2)``.
    Return ``(psdfreq, psd_i, psd_q)`` with PSDs in W/Hz.

    Reconstruct each real signal from paired positive and negative bins.
    Double non-DC powers for a single-sideband spectrum; use the peak-voltage
    power convention ``V**2 / (2 * TERMINATION)`` and divide by ``U * fs``.

    The output has ``N // 2`` bins. Its frequency axis runs from zero toward
    ``max(freqs)``, excluding that endpoint. Decibel conversion and carrier
    referencing are handled by :func:`apply_pfb_correction`.
    """

    N = len(iqfft)

    # Handle DC separately (it's purely real for real signals)
    dc_i = iqfft[0].real
    dc_q = iqfft[0].imag

    # For k from 1 to N//2-1, we need indices N-k
    # This gives us [N-1, N-2, ..., N//2+1]
    k_indices = np.arange(1, N // 2)
    neg_k_indices = N - k_indices  # This gives [N-1, N-2, ..., N//2+1]

    # Extract positive and negative frequency components
    z_pos = iqfft[k_indices]
    z_neg = iqfft[neg_k_indices]

    # Reconstruct I and Q using vectorized operations
    # For I: real parts add, imaginary parts subtract
    ibatch = 0.5 * (z_pos + np.conj(z_neg))

    # Reconstruct Q using the conjugate of the negative-frequency component.
    qbatch = 0.5j * (np.conj(z_neg) - z_pos)

    # Prepend DC values
    ibatch = np.concatenate([[dc_i], ibatch])
    qbatch = np.concatenate([[dc_q], qbatch])


    # Factor of 2 for real signals; volts are peak amplitudes, so the
    # power is V^2 / 2 / termination (volts_squared_to_dbm's convention).
    re_ps = 2 * (np.abs(ibatch) ** 2) / (2 * TERMINATION * U)
    im_ps = 2 * (np.abs(qbatch) ** 2) / (2 * TERMINATION * U)

    # DC bin fix for real-signal reconstruction
    re_ps[0] /= 2.0
    im_ps[0] /= 2.0

    # Divide by the sample rate to obtain power density in W/Hz.
    rbw = fs
    re_ps_lin = re_ps / rbw
    im_ps_lin = im_ps / rbw

    psdfreq = np.linspace(0, max(freqs), N // 2, endpoint=False)
    return psdfreq, re_ps_lin, im_ps_lin


def apply_pfb_correction(
    pfb_samples,
    nco_freq,
    channel_freq,
    binlim=1.1e6,
    trim=True,
    nsegments=1,
    reference="relative",
):
    """Correct PFB frequency response and calculate I, Q, and complex IQ spectra.

    Split the samples into equal segments, apply a Hann window, and correct
    each FFT for the PFB gain and droop. Average powers before converting to
    decibels. Samples left over after splitting are not used.

    Args:
        pfb_samples: complex PFB samples in ADC counts.
        nco_freq: NCO frequency in Hz.
        channel_freq: absolute tone frequency in Hz.
        binlim: correction limit on either side of the PFB bin centre, in Hz.
            Defaults to 1.1e6.
        trim: trim the corrected dual-sideband spectrum symmetrically around
            zero. Defaults to True.
        nsegments: number of segments to average. Defaults to one.
        reference: ``"absolute"`` for dBm/Hz or ``"relative"`` for dBc/Hz,
            referenced to the dual-sideband DC-bin power. Defaults to relative.

    Returns:
        tuple: ``(ssb_frequency, psd_i, psd_q, dsb_frequency, psd_iq)``.
        Frequencies are in Hz relative to the tone. I and Q spectra are
        single-sideband; the complex IQ spectrum includes both sidebands.
        In relative mode, DC entries represent bin power relative to the
        carrier, in dBc, rather than power density in dBc/Hz.
    """

    comb_sampling_freq = 625e6
    fs = comb_sampling_freq / 256.0

    # Convert ADC counts to volts for droop correction.
    pfb_samples_volts = pfb_samples * VOLTS_PER_ROC

    # Build PFB droop function via chebwin taps, large FFT, and shift
    NTAPS = 4096
    w_taps = chebwin(NTAPS, 103)
    W_full = np.fft.fft(w_taps, 262144)
    W_full /= np.abs(W_full[0]) + 1e-30
    W_full = np.fft.fftshift(W_full)
    pfb_freqs = np.linspace(-comb_sampling_freq, comb_sampling_freq, len(W_full))
    pfb_func = interp1d(pfb_freqs, W_full, bounds_error=True)

    seg_len = len(pfb_samples_volts) // nsegments

    # Linear accumulators for single-sideband (I/Q) and dual-sideband
    re_ps_lin_accum = None
    im_ps_lin_accum = None
    ssb_freq_final = None

    ds_ps_lin_accum = None
    ds_freq_final = None

    # Accumulate DC-bin power for carrier referencing.
    carrier_lin_accum = 0.0
    carrier_count = 0

    def _trim_around_zero(freq, data):
        """Keep equal numbers of bins on either side of the bin nearest zero."""
        zero_idx = np.argmin(np.abs(freq))
        half_len = min(zero_idx, len(freq) - zero_idx)
        return (
            freq[zero_idx - half_len : zero_idx + half_len],
            data[zero_idx - half_len : zero_idx + half_len],
        )

    def _accumulate_linear(accum, freq_accum, new_lin, new_freq):
        """Add overlapping bins to the accumulator in place.

        Keep the first segment's frequency axis and array length.
        """
        if accum is None:
            return new_lin, new_freq
        m_len = min(len(accum), len(new_lin))
        accum[:m_len] += new_lin[:m_len]
        return accum, freq_accum

    # Locate the tone relative to the NCO, then within its nearest PFB bin.
    bin_centers = (comb_sampling_freq / 512.0) * np.arange(-256, 256)
    channel_freq_in_nco_bw = channel_freq - nco_freq

    for seg in range(nsegments):
        # 1) Slice out time segment and apply Hann window
        segment = pfb_samples_volts[seg * seg_len : (seg + 1) * seg_len]
        window = np.hanning(len(segment))
        U = np.sum(window**2)  # sum(window^2) => consistent with Welch

        # 2) SHIFTed FFT: we do np.fft.fft(window*segment), then fftshift
        freqs_shifted = np.fft.fftshift(np.fft.fftfreq(len(segment), d=1.0 / fs))
        fft_shifted = np.fft.fftshift(np.fft.fft(window * segment))

        # 3) Locate bin center offset
        b_idx = np.abs(bin_centers - channel_freq_in_nco_bw).argmin()
        bin_center = bin_centers[b_idx]
        channel_offset_in_bin = channel_freq_in_nco_bw - bin_center

        freqs_in_bin = freqs_shifted + channel_offset_in_bin

        # 4) Apply built-in PFB gain at bin center, droop-correct only up to ±binlim
        built_in_gain = pfb_func(channel_offset_in_bin)
        fft_shifted *= built_in_gain

        valid_idx = np.abs(freqs_in_bin) <= binlim
        freq_corr = freqs_in_bin[valid_idx]
        data_corr = fft_shifted[valid_idx]

        droop = pfb_func(freq_corr) + 1e-30
        specdata_corrected = data_corr / droop

        # 5) Build dual-sideband data in linear.
        ds_freq_untrimmed = freq_corr - channel_offset_in_bin
        ds_freq_local = ds_freq_untrimmed
        ds_spec_local = specdata_corrected

        # Optionally trim around 0 if trim=True
        if trim:
            ds_freq_local, ds_spec_local = _trim_around_zero(ds_freq_local, ds_spec_local)

        # PSD in W/Hz: |FFT|^2 / (2 * termination * U * fs), peak volts
        ds_ps_lin_local = (np.abs(ds_spec_local) ** 2) / (2 * TERMINATION * U * fs)

        if reference.lower()=='relative' and len(ds_freq_local) > 0:
            # Record the DC bin amplitude for later normalization in linear
            zero_idx = np.argmin(np.abs(ds_freq_local))
            # Multiply the DC-bin density by its width to estimate carrier power.
            bin_width = fs / len(segment)
            carrier_lin_accum += ds_ps_lin_local[zero_idx] * bin_width
            carrier_count += 1

            # Overwrite DC bin with total power => ds_ps_lin_local[zero_idx] * bin_width
            ds_ps_lin_local[zero_idx] *= bin_width

        ds_ps_lin_accum, ds_freq_final = _accumulate_linear(
            ds_ps_lin_accum, ds_freq_final, ds_ps_lin_local, ds_freq_local
        )

        # 6) Single-sideband logic => reconstruct I/Q in separate_iq_fft_to_i_and_q_linear
        fft_corr_shifted = np.zeros_like(fft_shifted, dtype=complex)
        fft_corr_shifted[valid_idx] = specdata_corrected

        fft_corr_unshifted = np.fft.ifftshift(fft_corr_shifted)
        freqs_unshifted = np.fft.fftfreq(len(segment), d=1.0 / fs)

        # separate_iq => returns freq_ssb, re_ps_lin, im_ps_lin in W/Hz
        ssb_freq_local, re_ps_lin_local, im_ps_lin_local = separate_iq_fft_to_i_and_q_linear(
            freqs_unshifted, fft_corr_unshifted, fs, U
        )

        if reference.lower()=='relative':
            # Overwrite the DC bin in single-sideband as well
            re_ps_lin_local[0] *= (fs / len(segment))
            im_ps_lin_local[0] *= (fs / len(segment))

        # clip freq to binlim
        offset_hz = abs(channel_offset_in_bin)
        ssb_clip = max(0, binlim - offset_hz)
        bin_width = fs / len(ssb_freq_local)
        epsilon = 0.8 * bin_width
        freq_mask = ssb_freq_local <= (ssb_clip - epsilon)

        ssb_freq_local = ssb_freq_local[freq_mask]
        re_ps_lin_local = re_ps_lin_local[freq_mask]
        im_ps_lin_local = im_ps_lin_local[freq_mask]

        # Accumulate single-sideband PSDs
        if re_ps_lin_accum is None:
            re_ps_lin_accum = re_ps_lin_local
            im_ps_lin_accum = im_ps_lin_local
            ssb_freq_final = ssb_freq_local
        else:
            m_len = min(len(re_ps_lin_accum), len(re_ps_lin_local))
            re_ps_lin_accum[:m_len] += re_ps_lin_local[:m_len]
            im_ps_lin_accum[:m_len] += im_ps_lin_local[:m_len]

    # 7) Average in linear
    re_ps_lin_accum /= nsegments
    im_ps_lin_accum /= nsegments
    ds_ps_lin_accum /= nsegments

    # Summed carrier across segments => average
    carrier_lin = carrier_lin_accum / max(carrier_count, 1)

    # 8) Convert final linear accumulators to dBm/Hz or dBc/Hz
    re_ps_dbm = 10.0 * np.log10(re_ps_lin_accum / 1e-3 + 1e-30)
    im_ps_dbm = 10.0 * np.log10(im_ps_lin_accum / 1e-3 + 1e-30)
    ds_ps_dbm = 10.0 * np.log10(ds_ps_lin_accum / 1e-3 + 1e-30)

    if reference.lower() == "absolute":
        # => keep dBm/Hz directly
        final_re_psd = re_ps_dbm
        final_im_psd = im_ps_dbm
        final_ds_psd = ds_ps_dbm
    else:
        # => relative => dBc/Hz. Use DS DC bin as unified carrier reference
        carrier_dbm = 10.0 * np.log10(carrier_lin / 1e-3 + 1e-30)

        final_re_psd = re_ps_dbm - carrier_dbm
        final_im_psd = im_ps_dbm - carrier_dbm
        final_ds_psd = ds_ps_dbm - carrier_dbm

    return (
        ssb_freq_final,   # single-sideband freq axis
        final_re_psd,     # I-PSD in dBm/Hz or dBc/Hz
        final_im_psd,     # Q-PSD in dBm/Hz or dBc/Hz
        ds_freq_final,    # dual-sideband freq axis
        final_ds_psd      # dual-sideband PSD in dBm/Hz or dBc/Hz
    )


def noise_to_df(
    noise_module_output: dict,
    *,
    save: bool | None = None,
    label: str | None = None,
) -> dict:
    """Return one noise module block with calibrated df products.

    Native IQ timestreams and helper spectra are retained.  A calibrated
    stream gains ``df_hz`` (complex df + j*dissipation), ``psd_df`` and
    ``psd_dissipation`` (both Hz²/Hz).  Component spectra are recomputed from
    the rotated timestream because separate I and Q powers do not retain their
    cross-spectrum.  Resonators without a df calibration remain unchanged.

    The returned block uses copy-on-write dictionaries and shares unchanged
    arrays with the input; the input itself is not modified.  Saving updates
    the noise file the input came from, or creates one for an unsaved block.

    Args:
        noise_module_output: one module block from ``measure_noise``, such as
            ``noise_output[crs.module[m].index()]``.
        save: save the converted block to its existing file, or create a noise
            file.  None follows the configured autosave setting.
        label: filename label for a first save; an existing filename is kept.
    """
    if not isinstance(noise_module_output, dict):
        raise TypeError("Expected one module's noise output as a dict.")
    if noise_module_output.get("measurement") != "noise":
        raise ValueError("Expected one module's measure_noise output.")
    try:
        source_results = noise_module_output["results"]
        source_records = source_results["resonators"]
        info = source_results["info"]
        params = noise_module_output["call_params"]
    except KeyError as error:
        raise ValueError("Noise output lacks results or acquisition settings.") from error

    snapshot = params.get("catalog")
    catalog = ResonatorCatalog.from_dict(snapshot)

    converted = dict(noise_module_output)
    results = dict(source_results)
    results["info"] = converted_info = dict(info)
    converted["results"] = results
    records = {}
    converted_any = False
    for name, source_record in source_records.items():
        record = dict(source_record)
        records[name] = record
        resonator = catalog[name] if catalog is not None and name in catalog else None
        calibration = None if resonator is None else resonator.bias.df_calibration
        for stream in ("slow", "pfb"):
            data_key = f"{stream}_data"
            if data_key not in source_record:
                continue
            data = dict(source_record[data_key])
            data.pop("display_psds", None)
            record[data_key] = data
            if calibration is None:
                continue
            if info["iq_units"] == "volts":
                iq_volts = np.asarray(data["iq_volts"])
            elif info["iq_units"] in ("counts", "adc_counts"):
                iq_volts = np.asarray(data["iq_counts"]) * VOLTS_PER_ROC
            else:
                raise ValueError("Noise IQ units must be volts or counts.")
            df_hz = iq_volts * calibration
            if stream == "slow":
                spectrum = spectrum_from_slow_tod(
                    df_hz.real,
                    df_hz.imag,
                    dec_stage=info["decimation"],
                    nsegments=params["nsegments"],
                    reference="absolute",
                    spectrum_cutoff=params["spectrum_cutoff"],
                    input_units="volts",
                )
                psd_df, psd_dissipation = (
                    spectrum["psd_i"], spectrum["psd_q"])
            else:
                nco = info.get("nco_frequency_hz")
                frequency = source_record.get("bias_frequency_hz")
                if nco is None or frequency is None:
                    raise ValueError(
                        f"{name} lacks the frequencies needed for PFB df spectra."
                    )
                _, psd_df, psd_dissipation, _, _ = apply_pfb_correction(
                    df_hz / VOLTS_PER_ROC,
                    nco,
                    frequency,
                    binlim=info["pfb_binlim_hz"],
                    trim=info["pfb_trim"],
                    nsegments=info["pfb_nsegments"],
                    reference="absolute",
                )
            data["df_hz"] = df_hz
            data["psd_df"] = convert_dbm_to_volts(psd_df) ** 2
            data["psd_dissipation"] = convert_dbm_to_volts(
                psd_dissipation) ** 2
            converted_any = True
    results["resonators"] = records
    if converted_any:
        converted_info["df_timestream_units"] = "Hz"
        converted_info["df_spectrum_units"] = "Hz²/Hz"
    store.maybe_save(converted, "noise", save=save, label=label)
    return converted


def remove_common_mode(
    noise_module_output: dict,
    *,
    save: bool | None = None,
    label: str | None = None,
) -> dict:
    """Remove the leading common SVD mode from one module's slow noise data.

    The decomposition is performed on the complex, mean-subtracted detector
    timestream matrix.  Each cleaned timestream retains its original mean, so
    its carrier and relative-reference PSD remain well defined.  The PFB data
    is not changed.

    The module block gains ``common_mode_removal``, containing the detector
    order, offsets, complete reduced SVD, and removed rank-one matrix.  Thus
    the input matrix is exactly ``cleaned + common_mode`` and the centered
    matrix can be recreated as ``cleaned - offset + common_mode``.
    Each processed resonator gains ``common_mode_removed=True``; its
    ``slow_data`` gains the cleaned IQ and PSD arrays with a
    ``_common_mode_removed`` suffix.

    Args:
        noise_module_output: one module block from ``measure_noise``, such as
            ``noise_output[crs.module[m].index()]``.
        save: update the source file or create a noise file.  None follows the
            configured autosave setting.
        label: filename label for a first save; an existing filename is kept.

    Returns:
        dict: the stored ``common_mode_removal`` entry.

    Raises:
        TypeError: if the input is not a mapping.
        ValueError: if it is not a compatible noise block, or fewer than two
            equal-length, finite slow timestreams are available.
    """
    if not isinstance(noise_module_output, dict):
        raise TypeError("Expected one module's noise output as a dict.")
    if noise_module_output.get("measurement") != "noise":
        raise ValueError("Expected one module's measure_noise output.")

    results = noise_module_output.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("Noise output has no results mapping.")
    resonators = results.get("resonators")
    info = results.get("info")
    if not isinstance(resonators, Mapping) or not isinstance(info, Mapping):
        raise ValueError("Noise output lacks resonator records or stream info.")

    iq_units = info.get("iq_units")
    if iq_units == "volts":
        iq_key, input_units = "iq_volts", "volts"
    elif iq_units in ("counts", "adc_counts"):
        iq_key, input_units = "iq_counts", "adc_counts"
    else:
        raise ValueError("Noise IQ units must be volts or counts.")

    names = list(resonators)
    if len(names) < 2:
        raise ValueError("Common-mode removal needs at least two resonators.")

    timestreams = []
    for name in names:
        record = resonators[name]
        slow = record.get("slow_data") if isinstance(record, Mapping) else None
        if not isinstance(slow, Mapping) or iq_key not in slow:
            raise ValueError(f"Resonator {name!r} has no slow {iq_key} timestream.")
        iq = np.asarray(slow[iq_key])
        if iq.ndim != 1:
            raise ValueError(
                f"Resonator {name!r} slow timestream must be one-dimensional."
            )
        if not np.issubdtype(iq.dtype, np.number) or not np.all(np.isfinite(iq)):
            raise ValueError(
                f"Resonator {name!r} slow timestream must be finite numeric data."
            )
        timestreams.append(iq.astype(np.complex128, copy=False))

    lengths = {len(iq) for iq in timestreams}
    if len(lengths) != 1:
        raise ValueError("All slow timestreams must have the same length.")

    params = noise_module_output.get("call_params")
    if not isinstance(params, Mapping):
        raise ValueError("Noise output has no acquisition parameters.")
    try:
        decimation = int(info["decimation"])
        nsegments = int(params["nsegments"])
        spectrum_cutoff = float(params["spectrum_cutoff"])
        reference = info["reference"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Noise output lacks its slow-spectrum settings.") from error
    sample_count = next(iter(lengths))
    if nsegments < 1 or sample_count < nsegments:
        raise ValueError("Slow timestreams need at least one sample per segment.")

    timestream_matrix = np.stack(timestreams, axis=0)
    offset = timestream_matrix.mean(axis=1, keepdims=True)
    centered = timestream_matrix - offset
    u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    common_mode = (u[:, :1] * singular_values[0]) @ vh[:1, :]
    cleaned = centered - common_mode + offset

    spectra = []
    for iq in cleaned:
        spectra.append(spectrum_from_slow_tod(
            iq.real,
            iq.imag,
            dec_stage=decimation,
            scaling="psd",
            nsegments=nsegments,
            reference=reference,
            spectrum_cutoff=spectrum_cutoff,
            input_units=input_units,
        ))

    removal = {
        "method": "complex_svd",
        "resonator_names": names,
        "rank": 1,
        "full_matrices": False,
        "mean_restored": True,
        "iq_key": iq_key,
        "offset": offset,
        "u": u,
        "singular_values": singular_values,
        "vh": vh,
        "common_mode": common_mode,
    }
    noise_module_output["common_mode_removal"] = removal
    suffix = "_common_mode_removed"
    for name, iq, spectrum in zip(names, cleaned, spectra):
        record = resonators[name]
        record["common_mode_removed"] = True
        slow = record["slow_data"]
        slow[f"{iq_key}{suffix}"] = iq
        for key in _SPECTRUM_KEYS:
            slow[f"{key}{suffix}"] = np.asarray(spectrum[key])

    store.maybe_save(noise_module_output, "noise", save=save, label=label)
    return removal


__all__ = [
    "apply_pfb_correction",
    "noise_to_df",
    "remove_common_mode",
    "separate_iq_fft_to_i_and_q_linear",
]
