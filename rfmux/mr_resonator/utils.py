'''
kinetic inductance detector modeling

Maclean Rouble

maclean.rouble@mail.mcgill.ca
'''

import numpy as np
from scipy.optimize import curve_fit

def calc_Cc(C, f0, Qc, Z0=50):
    return np.sqrt((8*C) / (2*np.pi*f0*Qc*Z0))

#################
# UTILS
#################

def calc_dphase(Vout):
    x0, y0, R = circle_fit_pratt(Vout.real, Vout.imag)
    phase = np.unwrap(np.arctan2(Vout.imag-y0, Vout.real-x0))
    phase_diff = phase - phase[0]
    return phase_diff

def circle_fit_pratt(x, y):
    """
    Fits a circle to a set of (x, y) points using Pratt's method.
    
    Parameters:
        points (ndarray): Nx2 array of (x, y) points.

    Returns:
        x0 (float): x-coordinate of the circle center
        y0 (float): y-coordinate of the circle center
        R (float): Radius of the circle
    """
    # x = points[:, 0]
    # y = points[:, 1]

    # Construct design matrix
    A = np.column_stack((2*x, 2*y, np.ones_like(x)))
    b = x**2 + y**2

    # Solve the least squares problem: A * p = b
    p = np.linalg.lstsq(A, b, rcond=None)[0]

    x0, y0 = p[0], p[1]  # Circle center
    R = np.sqrt(p[2] + x0**2 + y0**2)  # Circle radius

    return x0, y0, R

def normalize(data, new_min=0, new_max=1):
    """Normalize data to the range [new_min, new_max]."""
    old_min, old_max = np.min(data), np.max(data)
    return (data - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

def square_axes(x):
    '''
    make the axes have the same scale, but allow them to span different ranges, so that the
    resulting plot is square and shows a to-scale representation of the data. (useful for
    IQ circle plots for example)
    
    Parameters:
    -----------
    x : axes object 
    '''
    
    x.set_aspect('equal', adjustable='box')
    x_limits = x.get_xlim()
    y_limits = x.get_ylim()
    max_extent = max(x_limits[1] - x_limits[0], y_limits[1] - y_limits[0])
    x_mid = (x_limits[0] + x_limits[1]) / 2
    y_mid = (y_limits[0] + y_limits[1]) / 2
    x.set_xlim([x_mid - max_extent / 2, x_mid + max_extent / 2])
    x.set_ylim([y_mid - max_extent / 2, y_mid + max_extent / 2])

def exp_bin_noise_data(f, psd, nbins=100):
    '''
    Bin a noise PSD, using exponential bin sizes.
    Not statistically rigorous!!
    
    Parameters:
    -----------
    f : list or np.array
        frequency data
    psd : list or np.array
        power spectral density data
        
    Returns:
    --------
    (np.array) binned frequency data
    (np.array) binned PSD data
    '''
    f = np.asarray(f)
    psd = np.asarray(psd)
    
    fbinned = []
    cavg = []
    std = []

    # nbins = 100
    fmin = f[0] + 10e-3 #######
    fmax = f[-1]

    for k in range(nbins-1):
        (fl, fh) = fmin * (fmax / fmin)**(np.asarray([k,k+1])/(nbins - 1.))
        wh = np.where((f>fl)*(f<=fh))[0]

        if len(wh) != 0:
            fbinned.append(np.mean([fl,fh]))
            cavg.append(np.mean(psd[wh]))
            std.append(np.std(psd[wh]))

    fbinned = np.asarray(fbinned)
    cavg = np.asarray(cavg)
    
    return fbinned, cavg


def rotate_iq_plane(iqdata, n_thetas=50, enforce_positive_i=True, use_mean_value=False, make_plots=False, plot_save_dir=None):

    '''
    rotate the iq plane until the stddev is maximized in the 'Q' direction
    '''

    theta_range = np.linspace(0, np.pi, n_thetas)

    imeans = []
    qmeans = []

    istds = []
    qstds = []

    for theta in theta_range:
        theta = 2*np.pi - theta

        #         thisplane = (idata + 1.j*qdata) * np.exp(1.j*theta)
        thisplane = (iqdata) * np.exp(1.j*theta)

        iplane = thisplane.real
        qplane = thisplane.imag
        imeans.append(np.mean(iplane))
        istds.append(np.std(iplane))
        qmeans.append(np.mean(qplane))
        qstds.append(np.std(qplane))

    if make_plots:
        fig = plt.figure(figsize=(18, 6))
        ax = fig.add_subplot(131)
        bx = fig.add_subplot(132)
        cx = fig.add_subplot(133)

        ax.plot(theta_range, imeans, label='i')
        ax.plot(theta_range, qmeans, label='q')

        bx.plot(theta_range, istds, label='i')
        bx.plot(theta_range, qstds, label='q')

        #ax.axvline(systemic_rotation, color='black', linestyle='--', label='global rotation')
        #bx.axvline(systemic_rotation, color='black', linestyle='--', label='global rotation')

        cx.plot(theta_range, np.asarray(qstds)/np.asarray(istds))
        cx.set_ylabel('Ratio Q/I')

        ax.legend()
        bx.legend()

        ax.set_ylabel('mean')
        bx.set_ylabel('std')
        ax.set_xlabel('reversed IQ rotation')
        bx.set_xlabel('reversed IQ rotation')

        fig.tight_layout()

        if plot_save_dir is not None:
            fig.savefig(os.path.join(plot_save_dir, 'iq_rotation.png'))
            plt.close(fig)

    if use_mean_value:
        ratio = abs(np.asarray(qmeans) / np.asarray(imeans))
    else:
        ratio = np.asarray(qstds) / np.asarray(istds)
    theta_best = 2*np.pi - theta_range[np.argmax(ratio)]
    iqrot = (iqdata) * np.exp(1.j*theta_best)
    
    if enforce_positive_i and np.mean(iqrot)<0:
        iqrot *= np.exp(1.j*np.pi)
        theta_best += np.pi

    return iqrot, theta_best 

def s21_skewed(f, f0, Qr, Qcre, Qcim, A):
    if abs(Qcre + 1j*Qcim)**2/Qcre < Qr: # prevents negative Qi values
        return np.inf
    else:
        x = (f-f0)/f0
        return abs(A*(1 - Qr/(Qcre+1j*Qcim)/(1+2j*Qr*x)))

def fit_skewed(freq, s21_iq, approxQr=1e4, normalize=True, fr_lim=None):
    '''
    Apply Pete's skewed resonance model to a given set of s21 vs frequency data.

    NOTE the "sigma" on the fitter uses an arbitrary value but should instead use an estimate
    of system noise.
    
    Parameters
    ----------
    freq : (list or np array) frequency values from a network analysis

    s21_iq : (list or np array) measured values from a network analysis, in i+jq or magnitude format.

    Returns
    -------

    fit_dict : (dict) dictionary of resonance parameters extracted from fit, with uncertainties. If 
        fit fails, returns 'nan'
    
    '''

    bad_fit_flag = False

    param_names = ['fr', 'Qr', 'Qc', 'Qi', 'Qcre', 'Qcim', 'A']
    fit_dict = {}

    freq = np.asarray(freq)
    s21_iq = np.asarray(s21_iq)

    if normalize:
        s21_iq = s21_iq/s21_iq[-1]

    if fr_lim is None:
        fr_lbound = min(freq)
        fr_ubound = max(freq)
        fr_guess = freq[np.argmin(abs(s21_iq))]
    else:
        fr_lbound = np.mean(freq) - fr_lim
        fr_ubound = np.mean(freq) + fr_lim
        fr_guess = np.mean(freq)
    
    init = [fr_guess, approxQr, approxQr, -approxQr, abs(s21_iq).mean()]
    bounds = ([fr_lbound, 0, 0, -np.inf, 0], [fr_ubound, np.inf, np.inf, np.inf, np.inf])
    try:
        s21fitp, s21cov = curve_fit(s21_skewed, freq, abs(s21_iq), p0=init, bounds=bounds, sigma=np.ones(len(freq))*90)
        errs = np.sqrt(np.diag(s21cov))
        f0, Qr = s21fitp[0:2]
        Qe = s21fitp[2] + 1j*s21fitp[3]
        Qc = abs(Qe)**2/Qe.real
        Qi = 1./(1/Qr - 1/Qc)
        param_vals = [f0, Qr, Qc, Qi, s21fitp[2], s21fitp[3], s21fitp[4]]
        
        errf0, errQr, errQere, errQeim = errs[0:4] # error in f0, Qr, Qe_re, Qe_im
        errQc = np.sqrt(errQere**2 + (2 * errQeim * Qe.imag/Qe.real)**2 + errQere*Qe.imag**2/Qe.real**2)
        errQi = np.sqrt(Qi**4*(errQr**2/Qr**4 + errQc**2/Qc**4))
        errA = errs[-1]
        param_errs = [errf0, errQr, errQc, errQi, errQere, errQeim, errA]

        if np.inf in param_vals:
            print('fit did not converge: found infinity in parameter value.')
            bad_fit_flag = True
        if  np.inf in param_errs:
            print('fit did not converge: found infinity in parameter error.')
            bad_fit_flag = True

        for i in range(len(param_names)):
            fit_dict[param_names[i]] = param_vals[i]
            fit_dict['%s_err'%(param_names[i])] = param_errs[i]


        
    except (RuntimeError, ValueError) as e:
        print(e, '\nfit did not converge')
        bad_fit_flag = True

    if bad_fit_flag:
        for i in range(len(param_names)):
            fit_dict[param_names[i]] = 'nan'
            fit_dict['%s_err'%(param_names[i])] = 'nan'
              
    return fit_dict


##############
# timestreams etc
###################

def calc_rbw(fs, N):
	'''
	resolution bandwidth
	'''

	return fs / N

def make_nqp_timestream_from_Nqp_spectrum(res, frequencies, Nqp_spectrum, rbw, baseline_nqp=None):
    '''
    using the GR noise nqp power spectral density, create a timestream of
    nqp values that corresponds to this spectrum.

    Parameters:
    -----------
    fs : sampling rate

    N : (int) number of samples
    '''

    if baseline_nqp is None:
        baseline_nqp = res.calc_nqp()
    baseline_Nqp = baseline_nqp * res.VL_um3
    
    # rbw = fs / N

    # frequencies = np.fft.fftfreq(N, d=1./fs)
    # frequencies, SN = self.calc_gr_PSD(frange=frequencies)
    SN = Nqp_spectrum

    power_spectrum_N = SN * rbw
    amplitude_spectrum_N = np.asarray(np.sqrt(power_spectrum_N), dtype=complex)

    random_phase_angles = np.random.rand(len(SN)) * 2*np.pi
    random_phases = np.exp(1.j*random_phase_angles)
    amplitude_spectrum_N *= random_phases

    dc_index = abs(frequencies).argmin()
    amplitude_spectrum_N[dc_index] = baseline_Nqp # the average value is the baseline population

    timestream_N = np.fft.ifft(amplitude_spectrum_N).real * len(SN)
    
    return timestream_N / res.VL_um3
