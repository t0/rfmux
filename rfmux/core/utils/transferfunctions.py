"""
Transfer functions for various hardware items.
"""
import numpy as np
from scipy import interpolate
from scipy import signal


# TODO: Empirical value (should probably be a hybrid)
VOLTS_PER_ROC = (np.sqrt(2)) * np.sqrt(50* (10**(-1.75/10))/1000)/1880796.4604246316 

CREST_FACTOR = 3.5

# TODO: Empirical value for digital latency (should probably be derived)
LATENCY = 224e-9

TERMINATION = 50.
COMB_SAMPLING_FREQ = 625e6
PFB_SAMPLING_FREQ = COMB_SAMPLING_FREQ / 256
READOUT_BANDWIDTH = 550e6 # approximately droop-free instantaneous bandwidth

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


def FIR_to_sampling(fir):
    return (625e6 / 256 / 64 / 2**fir)

def cic2_response(f_arr, fs):
    '''
    Compute the response of the second-stage cascaded-integrator 
    comb (CIC) filter, as a function of frequency. This filter 
    defines the shape of each channel's frequency response, so
    the frequencies here are relative to the central frequency
    of the channel.


    Paramters
    ---------

    f_arr : list or array of frequencies within one channel bw.
        Must be less than the demodulated channel sampling frequency.
        
    '''

    CIC_factor = np.log2(625e6 / (256 * 64 * fs))

    f_arr = np.asarray(f_arr)
    
    return np.sinc(f_arr / fs)**CIC_factor

def apply_cic2_comp_asd(f_arr, asd, fs, trim=0.15):
    '''
    Compensate for the response of the second-stage cascaded-
    integrator comb (CIC) filter as a function of frequency in
    an amplitude spectral density measurement.

    Because the channel response drops off quickly towards and 
    above Nyquist, results will be less trustworthy at high
    frequencies.

    Parameters
    ----------

    f_arr : list or array of frequencies within one channel bw.
        Must be less than the demodulated channel sampling frequency.
            
    asd : list or array of amplitude spectral density values as
        a function of frequency

    fs : int or float
        Sampling rate of the demodulated timestream. 

    trim : float
        Percent length of array to trim off at high frequencies,
        to avoid returning data where numerical uncertainties
        are higher in the compensation function.
        
    '''
    compensated = np.asarray(asd) / cic2_response(np.asarray(f_arr), fs)
    trimind = int(np.ceil(trim * len(compensated)))
    
    return f_arr[:-trimind], compensated[:-trimind]

def apply_cic2_comp_psd(f_arr, psd, fs, trim=0.15):
    '''
    NOTE it is not completely correct to apply this correction to
    power data, as any random scatter in the voltage data will
    be forced positive, which may slightly skew the results!
    
    Compensate for the response of the second-stage cascaded-
    integrator comb (CIC) filter as a function of frequency in
    a power spectral density measurement.

    Because the channel response drops off quickly towards and 
    above Nyquist, results will be less trustworthy at high
    frequencies, and so this trims the last trim-percent of the 
    datapoints at high frequency.

    f_arr : list or array of frequencies within one channel bw.
        Must be less than the demodulated channel sampling frequency.
            
    psd : list or array of amplitude spectral density values as
        a function of frequency

    fs : int or float
        Sampling rate of the demodulated timestream. 

    trim : float
        Percent length of array to trim off at high frequencies,
        to avoid returning data where numerical uncertainties
        are higher in the compensation function.
        
    '''
    compensated = np.asarray(psd) / (cic2_response(np.asarray(f_arr), fs))**2

    trimind = int(np.ceil(trim * len(compensated)))
    return f_arr[:-trimind], compensated[:-trimind]


def apply_pfb_correction(pfb_samples, nco_freq, channel_freq, binlim=0.6e6, trim=True):
    '''
    Remove the spectral droop of the polyphase filterbank's window function from timestream
    data acquired by routing the samples directly out of the PFB (the 2 MSPS capture mode).

    Parameters:
    -----------

    pfb_samples : arraylike
        the complex timestream data (array of time-ordered I+jQ samples)

    nco_freq : float
        numerically controlled oscillator frequency setting at which the data was 
        acquired

    channel_freq : float
        readout channel frequency setting at which the data was acquired. Expected to 
        be in real-world frequency units (ie, between 0-3GHz, rather than within -+250MHz 
        relative to the NCO)

    binlim : float, default 0.6e6
        frequency range within a PFB bin to keep data from. The bin's filter window quickly
        goes to very low gain, so we limit the region of data to keep to a range where the
        signal is not too far suppressed. Otherwise, at the extremes of the window, when we 
        remove the droop we are only amplifying noise.

    trim : bool, default True
        whether or not to trim the un-drooped output array to be centred on zero. Optional
        but makes downstream processing much simpler.

    '''

    fs = COMB_SAMPLING_FREQ / 256

    NFFT = 1024  # Synthetic FFT length. Actual cores are 1/4 this size.
    NTAPS = 4096  # Group delay is ~2 us. This is small enough to not tweak.

    w = signal.chebwin(NTAPS, 103) # the window function applied by the PFB
    W = np.fft.fft(w, 262144) # 2^18

    W /= np.abs(W[0]) # normalized filter response in frequency space
    W = np.fft.fftshift(W) # roll W so the zero bin is in the centre

    # now design a frequency array to go with W
    pfb_freqs = np.linspace(-COMB_SAMPLING_FREQ, COMB_SAMPLING_FREQ, len(W))

    # interpolate the response 
    pfb_func = interpolate.interp1d(pfb_freqs, W)

    # FFT the complex timestream so we can apply the corrections
    fftfreqs = np.fft.fftshift(np.fft.fftfreq(len(pfb_samples), d=1./fs))
    fft = np.fft.fftshift(np.fft.fft(np.hanning(len(pfb_samples))*pfb_samples)) 

    # determine which bin this frequency is in and how far it is from the centre
    bin_centres = [x * COMB_SAMPLING_FREQ/512 for x in range(-256, 256)]
    channel_freq_in_nco_bw = channel_freq - nco_freq
    b = abs(np.asarray(bin_centres) - channel_freq_in_nco_bw).argmin()
    this_bin_centre = bin_centres[b]
    channel_freq_in_bin = channel_freq_in_nco_bw - this_bin_centre # tone frequency relative to bin centre
    fftfreqs_in_bin = fftfreqs + channel_freq_in_bin # the FFT frequency array relative to bin centre

    
    # compute the scalar gain factor at this frequency
    builtin_gain_factor = pfb_func(channel_freq_in_bin)
    # remove the scalar gain factor from the FFT'd data
    fft = fft * builtin_gain_factor

    # prepare the droop correction:

    # the correction function will be centered around zero (the bin centre)
    # to avoid over-amplifying noise by un-drooping the extremes of the window's wings,
    # limit the correction region to a range of frequencies (+- binlim) close-ish to zero
    # THIS IS CURRENTLY AN ARBITRARY CHOICE AND MAY NEED TO BE CHOSEN MORE CAREFULLY IN FUTURE
    freqrange_to_correct = fftfreqs_in_bin[np.where((fftfreqs_in_bin > -binlim) & (fftfreqs_in_bin < binlim))[0]]
    specdata_to_correct = fft[np.where((fftfreqs_in_bin > -binlim) & (fftfreqs_in_bin < binlim))[0]]

    # apply the droop correction
    specdata_corrected = specdata_to_correct / pfb_func(freqrange_to_correct)

    # shift the spectrum back to its real-life frequency range:
    output_fftfreqs = freqrange_to_correct - channel_freq_in_bin

    # trim the output arrays to be centred on zero
    # this is optional but reduces headaches later in the processing
    if trim:
        zerof_ind = abs(output_fftfreqs).argmin()
        shortest_range = min([zerof_ind, len(output_fftfreqs)-zerof_ind])
        specdata_corrected = np.asarray(specdata_corrected[zerof_ind-shortest_range:zerof_ind+shortest_range])
        output_fftfreqs = np.asarray(output_fftfreqs[zerof_ind-shortest_range:zerof_ind+shortest_range])

    return output_fftfreqs, specdata_corrected, builtin_gain_factor
