"""
Transfer functions for various hardware items.
"""
import numpy as np


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



