"""
Transfer functions for various hardware items.
"""
import numpy as np

# Added import for PSD computation
from scipy.signal import welch


# TODO: Empirical value (should probably be a hybrid)
VOLTS_PER_ROC = (np.sqrt(2)) * np.sqrt(50* (10**(-1.75/10))/1000)/1880796.4604246316 

CREST_FACTOR = 3.5

TERMINATION = 50.
COMB_SAMPLING_FREQ = 625e6
PFB_SAMPLING_FREQ = COMB_SAMPLING_FREQ / 256

DDS_PHASE_ACC_NBITS = 32 # bits
FREQ_QUANTUM = COMB_SAMPLING_FREQ / 256 / 2**DDS_PHASE_ACC_NBITS

# TODO: verify still appropriate
BASE_FREQUENCY = COMB_SAMPLING_FREQ / 256 / 2**12

# TODO: These should be functions that take the current and attenuation as inputs
# ADC_FS = -1 # dbm
# DAC_FS = -1 # dbm


def convert_roc_to_volts(roc):
    '''
    Convenience function for converting a measured value in readout
    counts (ROCs) to voltage units.

    ROCs are a measure of voltage at the board input (TODO: Check).

    Parameters
    ----------
    roc : measured quantity in readout counts (ROCs)


    Returns
    -------
    
    (float) value in volts
    '''

    return roc * VOLTS_PER_ROC


def convert_roc_to_dbm(roc, termination=50.):
    '''
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
    '''

    volts = convert_roc_to_volts(roc)
    dbm = convert_volts_to_dbm(volts, termination)
    return dbm


def convert_volts_to_watts(volts, termination=50.):
    '''
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
    '''

    v_rms = volts / np.sqrt(2.)
    watts = v_rms**2 / termination

    return watts

def convert_volts_to_dbm(volts, termination=50.):
    '''
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
    '''

    watts = convert_volts_to_watts(volts, termination)
    return 10. * np.log10(watts * 1e3)


def decimation_to_sampling(dec):
    return (625e6 / 256 / 64 / 2**dec)


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
    freq_ratio = frequencies / f_in
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = np.sin(np.pi * freq_ratio)
        denominator = np.sin(np.pi * freq_ratio / R)
        correction = (numerator / denominator) ** N
        # Replace NaNs at DC with the ideal DC gain = R^N
        correction[np.isnan(correction)] = R ** N
    return correction / (R ** N)


def compensate_psd_for_cics(frequencies, psd, dec_stage=6, spectrum_cutoff=0.9):
    '''
    frequencies : ndarray
        Frequency bins (in Hz) for which the correction factor is desired.
    psd : ndarray
        Input power spectrum
    dec_stage : int
        Decimation stage used for data. ("FIR stage" in older firmware). (default: 6)
    spectrum_cutoff : float
        Fraction of Nyquist frequency to retain in the spectrum (default: 0.9).

    Returns
    -------

    Arrays (frequencies, corrected_psd)'''

    # Define CIC decimation parameters
    R1 = 64
    R2 = 2**dec_stage
    f_in1 = 625e6 / 256.0  # Original samplerate before 1st CIC
    f_in2 = f_in1 / R1     # Samplerate before 2nd CIC
    fs = f_in2 / R2        # Final sample rate

    # The CIC corrections are symmetric, take abs in case this is a dual-sideband PSD
    freq_abs = np.abs(frequencies)

    # Apply CIC correction for both CIC stages
    cic1_corr_c = _general_single_cic_correction(freq_abs * R1 * R2, f_in1, R=R1, N=3)
    cic2_corr_c = _general_single_cic_correction(freq_abs * R2, f_in2, R=R2, N=6)
    correction = cic1_corr_c * cic2_corr_c
    psd_corrected = psd / correction

    # Enforce cutoff
    nyquist = fs / 2
    cutoff_freq = spectrum_cutoff * nyquist
    cutoff_idx_c = freq_abs <= cutoff_freq
    frequencies = frequencies[cutoff_idx_c]
    psd_corrected = psd_corrected[cutoff_idx_c]

    return frequencies, psd_corrected

def spectrum_from_slow_tod(i_data, q_data, dec_stage,
                     scaling='psd', nperseg=None,
                     reference='relative',
                     spectrum_cutoff=0.9):
    """
    Internal function to compute both the I/Q single-sideband PSD and
    the dual-sideband complex PSD in either dBc or dBm, depending on 'reference'.
    The 'scaling' argument can be 'psd' => power spectral density (V^2/Hz)
    or 'ps' => power spectrum (V^2).

    Parameters
    ----------
    i_data : array-like
        Time-domain I (real) samples.
    q_data : array-like
        Time-domain Q (imag) samples.
    dec_stage : int
        Decimation stage to define the second CIC correction factor.
    scaling : {'psd','ps'}
        Whether we interpret Welch output as a PSD (V^2/Hz) or total power spectrum (V^2).
    nperseg : int, optional
        Number of samples per Welch segment. Default is all samples (no segmentation).
    reference : {'relative','absolute'}
        If 'relative', we do dBc => referencing the DC bin as the carrier.
        If 'absolute', we do dBm => referencing an absolute scale with 50 ohms.
    spectrum_cutoff : float
        Fraction of Nyquist frequency to retain in the spectrum (default: 0.9).

    Returns
    -------
    dict
        A dictionary containing:
          "freq_iq" : array of frequency bins for single-sideband I/Q,
          "psd_i"   : array of final I data in dBc or dBm,
          "psd_q"   : array of final Q data in dBc or dBm,
          "freq_dsb": array of frequency bins for the dual-sideband data,
          "psd_dual_sideband": array of final dual-sideband data in dBc or dBm.
    """
    if nperseg is None:
        nperseg = len(i_data)

    # Convert 'psd'/'ps' to scipy's 'density'/'spectrum'
    scipy_scaling = 'density' if scaling.lower() == 'psd' else 'spectrum'

    arr_i = np.asarray(i_data)
    arr_q = np.asarray(q_data)
    arr_complex = arr_i + 1j * arr_q

    fs = decimation_to_sampling(dec_stage)

    # Welch for dual-sideband complex
    freq_dsb, psd_c = welch(
        arr_complex, fs=fs, nperseg=nperseg,
        scaling=scipy_scaling,
        return_onesided=False,
        detrend=False # Important because we need the DC information for normalization
    )

    # Correct for the CIC1 and CIC2 transfer functions
    freq_dsb, psd_c_corrected = compensate_psd_for_cics(frequencies=freq_dsb, psd=psd_c, dec_stage=dec_stage, spectrum_cutoff=spectrum_cutoff)

    # Carrier normalization based on DC bin
    carrier_normalization = psd_c_corrected[0]*(fs/nperseg)

    if reference == 'relative' and scaling == 'psd': # Normalize by the _PS_ of the DC bin:
        ## OVERWRITE the DC bin on the assumption that this is the carrier we are normalizing to
        ## This gives people the correct "reference" right on the plot, as we are correctly assuming
        ## the carrier in total power, not a power density
        psd_c_corrected[0] = psd_c_corrected[0]*(fs/nperseg)
        psd_dual_sideband_db = 10.0 * np.log10(psd_c_corrected / (carrier_normalization + 1e-30))
                
    elif reference == 'relative' and scaling == 'ps': # DC bin already correctly normalized:
        psd_dual_sideband_db = 10.0 * np.log10(psd_c_corrected / ((psd_c_corrected[0]) + 1e-30))
    else:
        # absolute => convert V^2 -> W => dBm
        p_c = psd_c_corrected / 50.0
        psd_dual_sideband_db = 10.0 * np.log10(p_c / 1e-3 + 1e-30)

    # Single-sideband I/Q
    freq_i, psd_i = welch(arr_i, fs=fs, nperseg=nperseg, scaling=scipy_scaling, return_onesided=True, detrend=None)
    freq_q, psd_q = welch(arr_q, fs=fs, nperseg=nperseg, scaling=scipy_scaling, return_onesided=True, detrend=None)
    freq_iq = freq_i

    # CIC Correction
    freq_iq, psd_i_corrected = compensate_psd_for_cics(frequencies=freq_i, psd=psd_i, dec_stage=dec_stage, spectrum_cutoff=spectrum_cutoff)
    freq_iq, psd_q_corrected = compensate_psd_for_cics(frequencies=freq_q, psd=psd_q, dec_stage=dec_stage, spectrum_cutoff=spectrum_cutoff)

    # Convert to dBc or dBm
    if reference == 'relative' and scaling == 'psd': # Normalize by the _PS_ of the DC bin
        ## OVERWRITE the DC bin on the assumption that this is the carrier we are normalizing to
        ## This gives people the correct "reference" right on the plot, as we are correctly assuming
        ## the carrier in total power, not a power density
        psd_i_corrected[0] = psd_i_corrected[0]*(fs/nperseg)
        psd_q_corrected[0] = psd_q_corrected[0]*(fs/nperseg)

        ## Then normalize to the TOTAL power (in I and Q)
        psd_i_db = 10.0 * np.log10(psd_i_corrected / (carrier_normalization + 1e-30))
        psd_q_db = 10.0 * np.log10(psd_q_corrected / (carrier_normalization + 1e-30))
    elif reference == 'relative' and scaling == 'ps': # We are already dealing with PS
        psd_i_db = 10.0 * np.log10(psd_i_corrected / (psd_c_corrected[0] + 1e-30))
        psd_q_db = 10.0 * np.log10(psd_q_corrected / (psd_c_corrected[0] + 1e-30))
    else:
        # absolute => convert V^2 to W => W to dBm
        p_i = psd_i_corrected / 50.0
        p_q = psd_q_corrected / 50.0
        psd_i_db = 10.0 * np.log10(p_i / 1e-3 + 1e-30)
        psd_q_db = 10.0 * np.log10(p_q / 1e-3 + 1e-30)

    return {
        "freq_iq": freq_iq,
        "psd_i": psd_i_db,
        "psd_q": psd_q_db,
        "freq_dsb": freq_dsb,
        "psd_dual_sideband": psd_dual_sideband_db,
    }