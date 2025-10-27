"""
Transfer functions for various hardware items.
"""

import numpy as np

# Added import for PSD computation
from scipy.signal import welch
from scipy.constants import pi, c
from typing import Union, Optional, Dict, Literal
import warnings
from scipy import interpolate


# TODO: Empirical value (should probably be a hybrid)
VOLTS_PER_ROC = (
    (np.sqrt(2)) * np.sqrt(50 * (10 ** (-1.75 / 10)) / 1000) / 1880796.4604246316
)

CREST_FACTOR = 3.5

TERMINATION = 50.0
COMB_SAMPLING_FREQ = 625e6
PFB_SAMPLING_FREQ = COMB_SAMPLING_FREQ / 256

DDS_PHASE_ACC_NBITS = 32  # bits
FREQ_QUANTUM = COMB_SAMPLING_FREQ / 256 / 2**DDS_PHASE_ACC_NBITS

# TODO: verify still appropriate
BASE_FREQUENCY = COMB_SAMPLING_FREQ / 256 / 2**12

# ────────────────── Cable Delay Compensation Functions ───────────────────
SPEED_OF_LIGHT_VACUUM = c  # m/s
VELOCITY_FACTOR_COAX = 0.66
EFFECTIVE_PROPAGATION_SPEED = SPEED_OF_LIGHT_VACUUM * VELOCITY_FACTOR_COAX

# TODO: These should be functions that take the current and attenuation as inputs
# CAN assume -1.75 dBm loss in the balun
# ADC_FS = -1 # dbm
# DAC_FS = -1 # dbm


def convert_roc_to_volts(roc):
    """
    Convenience function for converting a measured value in readout
    counts (ROCs) to voltage units.

    ROCs are a measure of voltage at the board input (TODO: Check).

    Parameters
    ----------
    roc : measured quantity in readout counts (ROCs)


    Returns
    -------

    (float) value in volts
    """

    return roc * VOLTS_PER_ROC


def convert_roc_to_dbm(roc, termination=50.0):
    """
    Convenience function for converting a measured value in readout
    counts (ROCs) to log power units.

    ROCs are a measure of voltage, so an input termination is
    required (default 50 ohm).

    Parameters
    ----------

    roc : measured quantity in readout counts (ROCs)

    termination : the system termination resistance in ohms.
        Default is 50.

    Returns
    -------

    (float) value in dbm
    """

    volts = convert_roc_to_volts(roc)
    dbm = convert_volts_to_dbm(volts, termination)
    return dbm


def convert_volts_to_watts(volts, termination=50.0):
    """
    Convenience function for converting a signal amplitude in
    volts (for a sinusoidal signal) to  power units.

    NOTE that the conversion from volts amplitude to power is
    done using the root-mean-square voltage.

    Parameters
    ----------

    volts : signal amplitude in volts

    termination : the system termination resistance in ohms.
        Default is 50.

    Returns
    -------

    (float) value in watts
    """

    v_rms = volts / np.sqrt(2.0)
    watts = v_rms**2 / termination

    return watts


def convert_volts_to_dbm(volts, termination=50.0):
    """
    Convenience function for converting a signal amplitude in
    volts (for a sinusoidal signal) to log power units.

    NOTE that the conversion from volts amplitude to power is
    done using the root-mean-square voltage.

    Parameters
    ----------

    volts : signal amplitude in volts

    termination : the system termination resistance in ohms.
        Default is 50.

    Returns
    -------

    (float) value in dbm
    """

    watts = convert_volts_to_watts(volts, termination)
    return 10.0 * np.log10(watts * 1e3)


def decimation_to_sampling(dec_stage):
    """
    Convert decimation stage to sampling rate.
    
    Parameters
    ----------
    dec_stage : int
        Decimation stage (0-6)
        
    Returns
    -------
    float
        Sampling rate in Hz
    """
    return 625e6 / 256 / 64 / 2**dec_stage


def _general_single_cic_correction(frequencies, f_in, R=64, N=6):
    """
    Compute the single-stage CIC (Cascaded Integrator-Comb) filter correction factor.

    CIC filters exhibit passband droop, especially at higher frequencies. This function
    calculates a correction factor to approximately compensate for that droop. The
    correction factor is derived analytically based on the idealized mathematical
    expressions for a CIC filter. However, in firmware implementations, the filter
    coefficients are quantized, so the actual correction will not be perfectly exact,
    but will still be quite close.

    The datapath for the CRS includes two different CIC filters, with the second stage
    having a variable decimation rate.

    Parameters
    ----------
    frequencies : ndarray
        Frequency bins (in Hz) for which the correction factor is desired. Typically
        these might be FFT bin centers or some other set of discrete frequencies at
        which droop compensation is needed.
    f_in : float
        Input (pre-decimation) sampling rate in Hz. This is the rate at which data
        enters the CIC filter before it is decimated.
    R : int, optional
        The decimation rate. Default is 64.
    N : int, optional
        The number of CIC stages (integrator-comb pairs). Default is 6.

    Returns
    -------
    correction : ndarray
        Array of correction values (dimensionless, same shape as `frequencies`) that can
        be multiplied by the frequency response (or by the time-domain samples after
        the CIC filter) to approximately correct for the droop introduced by the filter.

    Notes
    -----
    1. The correction factor is computed using the ratio of sinc functions raised to
       the power of N:

       .. math::
          H_{corr}(\\omega) =
              \\left(
                  \\frac{\\sin\\left(\\pi f / f_{in}\\right)}
                       {\\sin\\left(\\frac{\\pi f}{R f_{in}}\\right)}
              \\right)^N
              \\times \\frac{1}{R^N}

       where :math:`f` is the frequency, :math:`f_{in}` is the input sample rate,
       :math:`R` is the decimation ratio, and :math:`N` is the number of stages.

    2. At DC (0 Hz), the expression above has an indeterminate form
       :math:`0/0`. We replace those NaN values with the ideal DC gain of
       :math:`R^N`, then normalize by :math:`R^N`, effectively making the
       correction factor 1 at DC.

    3. In practical hardware/firmware implementations, the CIC coefficients may be
       quantized or truncated, which means that the filter's actual response can deviate
       slightly from the ideal model used here. Therefore, the computed correction
       factor is an approximation that should perform well but will not be absolutely
       precise.
    """
    freq_ratio = frequencies / (f_in / R)
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = np.sin(np.pi * freq_ratio)
        denominator = np.sin(np.pi * freq_ratio / R)
        correction = (numerator / denominator) ** N
        # Replace NaNs at DC with the ideal DC gain = R^N
        correction[np.isnan(correction)] = R**N
    return correction / (R**N)


def compensate_psd_for_cics(frequencies, psd, dec_stage=6, spectrum_cutoff=0.9):
    """
    frequencies : ndarray
        Frequency bins (in Hz) for which the correction factor is desired.
    psd : ndarray
        Input power spectrum or power spectral density
    dec_stage : int
        Decimation stage used for data (0-6). (default: 6)
    spectrum_cutoff : float
        Fraction of Nyquist frequency to retain in the spectrum (default: 0.9).

    Returns
    -------

    Arrays (frequencies, corrected_psd)"""

    # Define CIC decimation parameters
    R1 = 64
    R2 = 2**dec_stage
    f_in1 = 625e6 / 256.0  # Original samplerate before 1st CIC
    f_in2 = f_in1 / R1     # Samplerate before 2nd CIC
    fs = f_in2 / R2        # Final sample rate

    # The CIC corrections are symmetric, take abs in case this is a dual-sideband PSD
    freq_abs = np.abs(frequencies)

    # Apply CIC correction for both CIC stages
    cic1_corr_c = _general_single_cic_correction(freq_abs, f_in1, R=R1, N=3)
    cic2_corr_c = _general_single_cic_correction(freq_abs, f_in2, R=R2, N=6)
    correction = cic1_corr_c * cic2_corr_c
    psd_corrected = psd / correction**2

    # Enforce cutoff
    nyquist = fs / 2
    cutoff_freq = spectrum_cutoff * nyquist
    cutoff_idx_c = freq_abs <= cutoff_freq
    frequencies = frequencies[cutoff_idx_c]
    psd_corrected = psd_corrected[cutoff_idx_c]

    return frequencies, psd_corrected


def spectrum_from_slow_tod(
    i_data: Union[np.ndarray, list],
    q_data: Union[np.ndarray, list],
    dec_stage: int,
    scaling: Literal["psd", "ps"] = "psd",
    nperseg: Optional[int] = None,
    nsegments: Optional[int] = None,
    reference: Literal["relative", "absolute"] = "relative",
    spectrum_cutoff: float = 0.9,
    input_units: Literal["adc_counts", "volts"] = "adc_counts",
) -> Dict[str, np.ndarray]:
    """
    Compute both the I/Q single-sideband PSD and the dual-sideband complex PSD.
    
    This function calculates power spectral density (PSD) or power spectrum (PS) from
    time-domain I/Q data, applying CIC filter corrections and converting to appropriate
    units (dBc or dBm) based on the reference type.

    Parameters
    ----------
    i_data : array-like
        Time-domain I (real) samples. Units must match `input_units` parameter.
    q_data : array-like
        Time-domain Q (imag) samples. Units must match `input_units` parameter.
    dec_stage : int
        Decimation stage (0-6) to define the second CIC correction factor.
    scaling : {'psd','ps'}, default='psd'
        Whether to compute power spectral density (V²/Hz) or power spectrum (V²).
        - 'psd': Power spectral density in V²/Hz (or equivalent)
        - 'ps': Power spectrum in V²
    nperseg : int, optional
        Number of samples per Welch segment. Default uses all samples (no segmentation).
        Cannot be used together with nsegments.
    nsegments : int, optional
        Number of segments to divide the data into for Welch's method. 
        This is a more user-friendly alternative to nperseg. 
        When specified, nperseg = len(data) // nsegments.
        Cannot be used together with nperseg.
    reference : {'relative','absolute'}, default='relative'
        Reference type for the output spectrum:
        - 'relative': Output in dBc, referencing the DC bin as the carrier
        - 'absolute': Output in dBm, referencing an absolute scale with 50Ω termination
    spectrum_cutoff : float, default=0.9
        Fraction of Nyquist frequency to retain in the spectrum.
    input_units : {'adc_counts', 'volts'}, default='adc_counts'
        Units of the input data:
        - 'adc_counts': Input data is in ADC counts (also known as readout counts or ROC).
                        Use this when passing raw data from crs.get_samples() or similar.
                        Corresponds to crs.UNITS.ADC_COUNTS.
        - 'volts': Input data is already converted to volts.
                   Use this when data has been pre-converted using VOLTS_PER_ROC.
                   Corresponds to crs.UNITS.VOLTS.

    Returns
    -------
    dict
        A dictionary containing:
        - "freq_iq" : np.ndarray
            Frequency bins for single-sideband I/Q spectra (Hz)
        - "psd_i" : np.ndarray  
            I-channel spectrum in dBc or dBm (depending on `reference`)
        - "psd_q" : np.ndarray
            Q-channel spectrum in dBc or dBm (depending on `reference`)
        - "freq_dsb" : np.ndarray
            Frequency bins for dual-sideband spectrum (Hz)
        - "psd_dual_sideband" : np.ndarray
            Dual-sideband complex spectrum in dBc or dBm (depending on `reference`)
    
    Raises
    ------
    ValueError
        If `input_units` is not one of the allowed values ('adc_counts' or 'volts').
        If both `nperseg` and `nsegments` are specified.
    
    Notes
    -----
    The function handles unit conversion internally when `reference='absolute'`:
    - If `input_units='adc_counts'`, data is converted to volts using VOLTS_PER_ROC
    - If `input_units='volts'`, data is used as-is
    
    For `reference='relative'`, the input units don't affect the output since
    everything is normalized to the carrier (DC bin) power.
    """
    # Validate that only one of nperseg or nsegments is provided
    if nperseg is not None and nsegments is not None:
        raise ValueError(
            "Cannot specify both 'nperseg' and 'nsegments'. "
            "Please use only one parameter."
        )
    
    # Convert nsegments to nperseg if provided
    if nsegments is not None:
        data_length = len(i_data)
        nperseg = data_length // nsegments
        if nperseg < 1:
            raise ValueError(
                f"nsegments ({nsegments}) is too large for data length ({data_length}). "
                f"Each segment must have at least 1 sample."
            )
    
    # Default behavior if neither is specified
    if nperseg is None:
        nperseg = len(i_data)

    # Convert 'psd'/'ps' to scipy's 'density'/'spectrum'
    scipy_scaling = "density" if scaling.lower() == "psd" else "spectrum"

    # Validate input_units parameter
    valid_units = ["adc_counts", "volts"]
    if input_units not in valid_units:
        raise ValueError(
            f"Invalid input_units '{input_units}'. "
            f"Must be one of {valid_units}. "
            f"Use 'adc_counts' (crs.UNITS.ADC_COUNTS) for raw ADC data or "
            f"'volts' (crs.UNITS.VOLTS) for pre-converted voltage data."
        )
    
    # Handle legacy "roc" value for backwards compatibility
    if input_units == "roc":
        warnings.warn(
            "input_units='roc' is deprecated. Use 'adc_counts' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        input_units = "adc_counts"
    
    arr_i = np.asarray(i_data)
    arr_q = np.asarray(q_data)
    
    # Convert from ADC counts to volts if needed for absolute reference
    if input_units == "adc_counts" and reference == "absolute":
        arr_i = arr_i * VOLTS_PER_ROC
        arr_q = arr_q * VOLTS_PER_ROC
    
    arr_complex = arr_i + 1j * arr_q

    fs = decimation_to_sampling(dec_stage)

    # Welch for dual-sideband complex
    freq_dsb, psd_c = welch(
        arr_complex,
        fs=fs,
        nperseg=nperseg,
        scaling=scipy_scaling,
        return_onesided=False,
        detrend=False,  # Important because we need the DC information for normalization
    )

    # Correct for the CIC1 and CIC2 transfer functions
    freq_dsb, psd_c_corrected = compensate_psd_for_cics(
        frequencies=freq_dsb,
        psd=psd_c,
        dec_stage=dec_stage,
        spectrum_cutoff=spectrum_cutoff,
    )

    # Carrier normalization based on DC bin
    # DC‑bin normalization for relative scale
    carrier_norm_raw = psd_c_corrected[0] * (fs / nperseg)
    # Avoid zero division: floor to machine‑tiny
    eps = np.finfo(float).tiny
    carrier_normalization = max(carrier_norm_raw, eps)

    if (
        reference == "relative" and scaling == "psd"
    ):  # Normalize by the _PS_ of the DC bin:
        ## OVERWRITE the DC bin on the assumption that this is the carrier we are normalizing to
        ## This gives people the correct "reference" right on the plot, as we are correctly assuming
        ## the carrier in total power, not a power density
        psd_c_corrected[0] = psd_c_corrected[0] * (fs / nperseg)
        psd_dual_sideband_db = 10.0 * np.log10(
            psd_c_corrected / (carrier_normalization)
        )

    elif (
        reference == "relative" and scaling == "ps"
    ):  # DC bin already correctly normalized:
        psd_dual_sideband_db = 10.0 * np.log10(
            psd_c_corrected / ((psd_c_corrected[0]))
        )
    elif reference=='absolute':
        # absolute => convert V^2 -> W => dBm
        p_c = psd_c_corrected / 50.0
        # clamp the ratio p_c/1e-3 to avoid log10(0)
        ratio = p_c / 1e-3
        ratio_safe = np.maximum(ratio, eps)
        psd_dual_sideband_db = 10.0 * np.log10(ratio_safe)        

    else: # Keep in counts
        psd_dual_sideband_db = psd_c_corrected

    # Single-sideband I/Q
    freq_i, psd_i = welch(
        arr_i,
        fs=fs,
        nperseg=nperseg,
        scaling=scipy_scaling,
        return_onesided=True,
        detrend=None,
    )
    freq_q, psd_q = welch(
        arr_q,
        fs=fs,
        nperseg=nperseg,
        scaling=scipy_scaling,
        return_onesided=True,
        detrend=None,
    )
    freq_iq = freq_i

    # CIC Correction
    freq_iq, psd_i_corrected = compensate_psd_for_cics(
        frequencies=freq_i,
        psd=psd_i,
        dec_stage=dec_stage,
        spectrum_cutoff=spectrum_cutoff,
    )
    freq_iq, psd_q_corrected = compensate_psd_for_cics(
        frequencies=freq_q,
        psd=psd_q,
        dec_stage=dec_stage,
        spectrum_cutoff=spectrum_cutoff,
    )

    # Claim to avoid divide by zero
    psd_i_corrected = np.maximum(psd_i_corrected, eps)
    psd_q_corrected = np.maximum(psd_q_corrected, eps)  

    # Convert to dBc or dBm
    if (
        reference == "relative" and scaling == "psd"
    ):  # Normalize by the _PS_ of the DC bin
        ## OVERWRITE the DC bin on the assumption that this is the carrier we are normalizing to
        ## This gives people the correct "reference" right on the plot, as we are correctly assuming
        ## the carrier in total power, not a power density
        psd_i_corrected[0] = psd_i_corrected[0] * (fs / nperseg)
        psd_q_corrected[0] = psd_q_corrected[0] * (fs / nperseg)

        ## Then normalize to the TOTAL power (in I and Q)
        psd_i_db = 10.0 * np.log10(psd_i_corrected / (carrier_normalization))
        psd_q_db = 10.0 * np.log10(psd_q_corrected / (carrier_normalization))
    elif reference == "relative" and scaling == "ps":  # We are already dealing with PS
        # floor the DC bin so we never divide by zero
        carrier = max(psd_c_corrected[0], eps)      
        psd_i_db = 10.0 * np.log10(psd_i_corrected / carrier)
        psd_q_db = 10.0 * np.log10(psd_q_corrected / carrier)        
    elif reference == 'absolute':
        # absolute => convert V^2 -> W -> dBm, with zero‑floor
        p_i = psd_i_corrected / 50.0
        p_q = psd_q_corrected / 50.0
    
        # avoid log10(0)
        ratio_i = p_i / 1e-3
        ratio_q = p_q / 1e-3
        ratio_i = np.maximum(ratio_i, eps)
        ratio_q = np.maximum(ratio_q, eps)

        psd_i_db = 10.0 * np.log10(ratio_i)
        psd_q_db = 10.0 * np.log10(ratio_q)
    else: # Keep in counts
        psd_i_db = psd_i_corrected
        psd_q_db = psd_q_corrected

    return {
        "freq_iq": freq_iq,
        "psd_i": psd_i_db,
        "psd_q": psd_q_db,
        "freq_dsb": freq_dsb,
        "psd_dual_sideband": psd_dual_sideband_db,
    }

def fit_cable_delay(freqs: np.ndarray, phases_deg: np.ndarray) -> float:
    """
    Fits the phase vs. frequency data to determine the residual group delay.

    Parameters
    ----------
    freqs : np.ndarray
        Array of frequencies in Hz.
    phases_deg : np.ndarray
        Array of corresponding phases in degrees, as currently displayed
        (i.e., already compensated by any existing cable length setting).

    Returns
    -------
    float
        The calculated additional group delay (tau_additional) in seconds.
        This delay corresponds to the residual phase slope.
    """
    if len(freqs) < 2 or len(phases_deg) < 2 or len(freqs) != len(phases_deg):
        # Not enough data points to perform a fit
        return 0.0

    phases_rad = np.deg2rad(phases_deg)
    unwrapped_phases_rad = np.unwrap(phases_rad)

    # Perform linear fit: phase_rad = slope * freq_hz + intercept
    slope, _ = np.polyfit(freqs, unwrapped_phases_rad, 1)

    # The phase slope d(phase_rad)/d(freq_hz) = 2 * pi * tau
    # So, tau_additional = slope / (2 * pi)
    tau_additional = -slope / (2 * np.pi)
    
    return tau_additional

def calculate_new_cable_length(current_physical_length_m: float, additional_delay_s: float) -> float:
    """
    Calculates the new physical cable length based on the current length and an additional delay.

    Parameters
    ----------
    current_physical_length_m : float
        The current physical cable length setting in meters.
    additional_delay_s : float
        The additional group delay calculated from the phase slope (e.g., by fit_cable_delay).

    Returns
    -------
    float
        The new total physical cable length in meters.
    """
    additional_physical_length_m = additional_delay_s * EFFECTIVE_PROPAGATION_SPEED
    new_physical_length_m = current_physical_length_m + additional_physical_length_m
    return new_physical_length_m

def recalculate_displayed_phase(
    freqs: np.ndarray, 
    phases_deg_currently_displayed: np.ndarray, 
    old_physical_length_m: float, 
    new_physical_length_m: float
) -> np.ndarray:
    """
    Recalculates the phase data with only the delta compensation.
    """
    if len(freqs) == 0:
        return np.array([])

    # Calculate the delta length
    delta_length_m = new_physical_length_m - old_physical_length_m
    
    # Calculate phase shift from delta length
    # A positive delta_length means more cable, which creates positive phase shift
    delta_phase_rad = (2 * np.pi * freqs * delta_length_m) / EFFECTIVE_PROPAGATION_SPEED
    
    # Apply the delta compensation directly
    phases_new_displayed_rad = np.deg2rad(phases_deg_currently_displayed) + delta_phase_rad
    
    return np.rad2deg(phases_new_displayed_rad)


def exp_bin_noise_data(f: np.ndarray, psd: np.ndarray, nbins: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin a noise PSD using exponential bin sizes.
    
    This is a visual aid for cleaner plotting of noise spectra. The exponential
    binning is not statistically rigorous but provides a cleaner visualization
    of noise floors while preserving spectral features.
    
    Parameters
    ----------
    f : array-like
        Frequency data in Hz
    psd : array-like
        Power spectral density data (same shape as f)
    nbins : int, optional
        Number of bins to use (default: 1000)
        
    Returns
    -------
    fbinned : np.ndarray
        Binned frequency data (geometric mean of bin edges)
    psd_binned : np.ndarray
        Binned PSD data (mean of points in each bin)
    """
    f = np.asarray(f)
    psd = np.asarray(psd)
    
    if len(f) < 2:
        return f, psd
    
    # Optimized implementation using numpy operations
    fmin = f[0] + 10e-3  # Small offset to avoid zero
    fmax = f[-1]
    
    # Generate all bin edges at once
    bin_edges = fmin * (fmax / fmin)**(np.linspace(0, 1, nbins))
    
    # Use histogram to bin the data efficiently
    # digitize gives us which bin each frequency belongs to
    bin_indices = np.digitize(f, bin_edges)
    
    # Pre-allocate arrays
    fbinned = []
    psd_binned = []
    
    # Process each bin
    for i in range(1, nbins):
        mask = bin_indices == i
        if np.any(mask):
            # Geometric mean of bin edges for center frequency
            fbinned.append(np.sqrt(bin_edges[i-1] * bin_edges[i]))
            psd_binned.append(np.mean(psd[mask]))
    
    return np.array(fbinned), np.array(psd_binned)


def convert_iq_to_df(iq, fbias, f_calsweep, iq_calsweep):
    '''
    Equation 4.5 of Pete Barry's thesis (https://orca.cardiff.ac.uk/id/eprint/71562/1/2014BarryPPhD.pdf)
    Converts from I,Q voltage units to fequency shift and dissipation.

    NOTE This assumes linearity over the dynamic range being exercised.

    Parameters:
    -----------

    iq : numpy array
        complex I+jQ data in voltage units. Note that this can be an I,Q timestream or an FFT of an I,Q
        timestream 

    fbias : float
        resonant frequency (or, frequency at which to compute the derivatives used in the conversion)

    f_calsweep : arraylike
        array of frequency data from the sweep 

    iq_calsweep : arraylike
        array of I+jQ values from the sweep IN VOLTAGE UNITS, corresponding to f_calsweep

    Returns:
    --------

    df : numpy array
        the converted data, in the format (ffs)+1j*(diss) where ffs is frequency shift (NOT fractional frequency shift)
        and diss is dissipative. To get fractional frequency shift, divide by fbias.    
    '''
    fdi = interpolate.CubicSpline(f_calsweep, iq_calsweep.real).derivative()
    fdq = interpolate.CubicSpline(f_calsweep, iq_calsweep.imag).derivative()
    dIr = fdi(fbias)
    dQr = fdq(fbias)

    df = iq / (dIr +1.j*dQr)

    return df
