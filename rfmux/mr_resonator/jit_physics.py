"""
This module is derived from and based on the mr_resonator project by Maclean Rouble:
https://github.com/macleaner/mr_resonator

Modifications in this repository include:
- Trimmed to include only functions used by the Mock CRS framework
- Consolidation and integration with local JIT physics approximations

Original project license: see rfmux/mr_resonator/LICENSE (upstream LICENSE retained)

JIT-compiled physics calculations for LEKID resonator simulations.

This module consolidates all numba-accelerated physics calculations,
providing maximum performance for multi-resonator simulations.
All functions are JIT-compiled for 10-25x speedup over pure Python.

Numba is a required dependency for this module.
"""
import platform
import types as _types

import numpy as np
import numba
from numba import jit, prange


# ============================================================================
# Small-n dispatch
# ============================================================================
# numba's parallel=True enters a thread-parallel region at every prange.
# That entry costs a few microseconds, which dwarfs the actual work when
# there are only a handful of resonators — and the convergence solver hits
# three pranges per iteration for ~60 iterations, so it pays that cost
# ~180 times per call.  Measured at n=5 the serial build is 27x faster;
# parallel only pulls ahead above n ~ 1000.  Both builds are compiled and
# selected per call on the array length.
PARALLEL_MIN_N = 1024


def _serial_twin(dispatcher, name, **jit_kwargs):
    """Recompile a parallel dispatcher's source with parallel=False.

    The twin gets its own __qualname__ so numba's on-disk cache keys it
    separately — the two builds share a code object and would otherwise
    collide.
    """
    py = dispatcher.py_func
    twin = _types.FunctionType(py.__code__, py.__globals__, name,
                               py.__defaults__, py.__closure__)
    twin.__qualname__ = name
    return jit(nopython=True, parallel=False, cache=True, **jit_kwargs)(twin)


# Physical constants
H = 6.626e-34  # Planck constant
KB = 1.38e-23  # Boltzmann constant
MU0 = 4e-7 * np.pi  # Permeability of free space [H/m]

# ============================================================================
# Bessel Function Approximations
# ============================================================================

if platform.system() == "Darwin":
    numba.get_num_threads()
    layer = numba.config.THREADING_LAYER

    if layer != "omp":
        print(">>>>>>>>>> You are on Mac with no libomp, the processing will fail <<<<<<<<<<")
        print(">>>>>>>>>> Consult the README.MD <<<<<<<<<<<<<<\n")
    else:
        print("MacOS numba threading layer:", layer)


@jit(nopython=True, cache=True, fastmath=True)
def bessel_k0(x):
    """
    Fast approximation of modified Bessel function K0.
    
    Accurate to ~1e-6 relative error for all x > 0.
    """
    if x < 0.001:
        return 15.0 - np.log(x/2)
    elif x < 2.0:
        log_term = np.log(x / 2.0)
        gamma = 0.5772156649
        k0 = -log_term - gamma
        k0 += 0.25 * x * x * (log_term + gamma - 0.5)
        k0 += 0.015625 * x**4 * (log_term + gamma - 0.75)
        return k0
    else:
        sqrt_term = np.sqrt(np.pi / (2.0 * x))
        exp_term = np.exp(-x)
        series = 1.0 + 1.0/(8.0*x) + 9.0/(128.0*x*x)
        return sqrt_term * exp_term * series


@jit(nopython=True, cache=True, fastmath=True)
def bessel_i0(x):
    """
    Fast approximation of modified Bessel function I0.
    
    Accurate to ~1e-6 relative error for all x.
    """
    if x < 0.0:
        x = -x
    if x < 3.75:
        t = x / 3.75
        t2 = t * t
        return 1.0 + 3.5156229*t2 + 3.0899424*t2*t2 + 1.2067492*t2*t2*t2 + \
               0.2659732*t2*t2*t2*t2 + 0.0360768*t2*t2*t2*t2*t2 + 0.0045813*t2*t2*t2*t2*t2*t2
    else:
        t = 3.75 / x
        exp_term = np.exp(x)
        sqrt_term = 1.0 / np.sqrt(2.0 * np.pi * x)
        series = 0.39894228 + 0.01328592*t + 0.00225319*t*t - 0.00157565*t*t*t + \
                 0.00916281*t*t*t*t - 0.02057706*t*t*t*t*t
        return exp_term * series * sqrt_term


# ============================================================================
# Physics Calculations
# ============================================================================

@jit(nopython=True, cache=True)
def calc_sigma1(f, T, nqp, Delta0, N0, sigmaN):
    """
    Calculate real part of complex conductivity (sigma1).
    
    Parameters
    ----------
    f : float
        Frequency in Hz
    T : float  
        Temperature in K
    nqp : float
        Quasiparticle density
    Delta0 : float
        Zero-temperature gap energy
    N0 : float
        Density of states
    sigmaN : float
        Normal conductivity
        
    Returns
    -------
    float
        Real part of complex conductivity
    """
    zeta = H * f / (2.0 * KB * T)
    K0 = bessel_k0(zeta)
    
    x1 = 2.0 * Delta0 / (H * f)
    x2 = nqp / (N0 * np.sqrt(2.0 * np.pi * KB * T * Delta0))
    
    return x1 * x2 * np.sinh(zeta) * K0 * sigmaN


@jit(nopython=True, cache=True)
def calc_sigma2(f, T, nqp, Delta0, N0, sigmaN):
    """
    Calculate imaginary part of complex conductivity (sigma2).
    
    Parameters
    ----------
    f : float
        Frequency in Hz
    T : float
        Temperature in K
    nqp : float
        Quasiparticle density
    Delta0 : float
        Zero-temperature gap energy
    N0 : float
        Density of states
    sigmaN : float
        Normal conductivity
        
    Returns
    -------
    float
        Imaginary part of complex conductivity
    """
    zeta = H * f / (2.0 * KB * T)
    I0 = bessel_i0(zeta)
    
    x1 = np.pi * Delta0 / (H * f)
    x2 = nqp / (2.0 * N0 * Delta0)
    x3 = np.sqrt(2.0 * Delta0 / (np.pi * KB * T)) * np.exp(-zeta) * I0
    
    return x1 * (1.0 - x2 * (1.0 + x3)) * sigmaN


@jit(nopython=True, cache=True)
def calc_Zs(f, sigma1, sigma2, thickness, width, length):
    """
    Calculate surface impedance.
    
    Parameters
    ----------
    f : float
        Frequency in Hz
    sigma1, sigma2 : float
        Real and imaginary parts of conductivity
    thickness, width, length : float
        Geometric parameters in meters
        
    Returns
    -------
    complex
        Surface impedance
    """
    sigma = sigma1 - 1j * sigma2
    root1 = (1j * 2.0 * np.pi * f * MU0) / sigma
    cotharg = thickness * np.sqrt(1j * 2.0 * np.pi * f * MU0 * sigma)
    Zs = np.sqrt(root1) * (1.0 / np.tanh(cotharg))
    return Zs


@jit(nopython=True, cache=True)
def calc_R_L(f, Zs, length, width, R_spoiler):
    """
    Calculate total resistance and kinetic inductance.
    
    Parameters
    ----------
    f : float
        Frequency in Hz
    Zs : complex
        Surface impedance
    length, width : float
        Geometric parameters in meters
    R_spoiler : float
        Additional resistance
        
    Returns
    -------
    tuple of float
        (R_total, Lk_total)
    """
    R = Zs.real * (length / width) + R_spoiler
    L = (Zs.imag / (2.0 * np.pi * f)) * (length / width)
    return R, L


# ============================================================================
# Vectorized Physics Operations
# ============================================================================

@jit(nopython=True, parallel=True, cache=True)
def _vectorized_update_params_from_nqp_par(
    nqp_array, readout_freqs, T_array, Delta0_array, 
    N0_array, sigmaN_array, thickness_array, 
    width_array, length_array, R_spoiler_array
):
    """
    Update R and Lk for all resonators based on quasiparticle density.
    
    Uses parallel execution for maximum performance with many resonators.
    
    Parameters
    ----------
    nqp_array : ndarray
        Quasiparticle densities for all resonators
    readout_freqs : ndarray
        Readout frequencies for all resonators
    T_array : ndarray
        Temperature for each resonator
    Delta0_array : ndarray
        Gap energy for each resonator
    N0_array : ndarray
        Density of states for each resonator
    sigmaN_array : ndarray
        Normal conductivity for each resonator
    thickness_array : ndarray
        Thickness for each resonator
    width_array : ndarray
        Width for each resonator
    length_array : ndarray
        Length for each resonator
    R_spoiler_array : ndarray
        Spoiler resistance for each resonator
    
    Returns
    -------
    R_array, Lk_array : ndarray
        Resistance and kinetic inductance for all resonators
    """
    n = len(nqp_array)
    R_out = np.zeros(n, dtype=np.float64)
    Lk_out = np.zeros(n, dtype=np.float64)
    
    # Parallel loop over all resonators
    for i in prange(n):
        # Calculate conductivities
        sigma1 = calc_sigma1(
            readout_freqs[i], T_array[i], nqp_array[i],
            Delta0_array[i], N0_array[i], sigmaN_array[i]
        )
        sigma2 = calc_sigma2(
            readout_freqs[i], T_array[i], nqp_array[i],
            Delta0_array[i], N0_array[i], sigmaN_array[i]
        )
        
        # Calculate surface impedance
        Zs = calc_Zs(
            readout_freqs[i], sigma1, sigma2,
            thickness_array[i], width_array[i], length_array[i]
        )
        
        # Calculate R and Lk
        R, Lk = calc_R_L(
            readout_freqs[i], Zs, length_array[i], 
            width_array[i], R_spoiler_array[i]
        )
        
        R_out[i] = R
        Lk_out[i] = Lk
    
    return R_out, Lk_out


_vectorized_update_params_from_nqp_ser = _serial_twin(
    _vectorized_update_params_from_nqp_par,
    "_vectorized_update_params_from_nqp_ser")


def vectorized_update_params_from_nqp(nqp_array, *args):
    """Update R and Lk for all resonators from quasiparticle density."""
    fn = (_vectorized_update_params_from_nqp_par
          if len(nqp_array) >= PARALLEL_MIN_N
          else _vectorized_update_params_from_nqp_ser)
    return fn(nqp_array, *args)


# ============================================================================
# Convergence Loop
# ============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def _p_attenuator(input_atten_dB, z0):
    """(r2, r3) of the P-type input attenuator: r2 in series with the
    generator, r3 shunting the resonator node."""
    att_factor = 10.0 ** (input_atten_dB / 20.0)
    r3 = z0 * ((att_factor + 1) / (att_factor - 1))
    r2 = (z0 / 2.0) * ((10.0 ** (input_atten_dB / 10.0) - 1) / att_factor)
    return r2, r3


@jit(nopython=True, cache=True, fastmath=True)
def _drive_current(w, amplitude, L, R, C, Cc, ZLNA, r2, r3):
    """The current the generator drives through one resonator of total
    inductance L (parallel LC with R in the L branch, Cc in series,
    the LNA across the node): the solver's map F(I) for one element,
    and the linear response when L is the rest value."""
    if C > 0:
        ZC = 1.0 / (1j * w * C)
        ZL = 1j * w * L
        Z_parallel = 1.0 / (1.0 / ZC + 1.0 / (ZL + R))
    else:
        Z_parallel = 1j * w * L + R
    Z_res = Z_parallel + 1.0 / (1j * w * Cc)
    Zsys = 1.0 / (1.0 / Z_res + 1.0 / ZLNA)
    Zp = 1.0 / (1.0 / Zsys + 1.0 / r3)
    I2 = amplitude / (r2 + Zp)
    Iin = I2 * (r3 / (Zsys + r3))
    Zpar = 1.0 / (1.0 / r3 + 1.0 / Z_res + 1.0 / ZLNA)
    return Iin * Zpar / Z_res


@jit(nopython=True, cache=True, fastmath=True)
def linear_currents(freqs, L, R, C, Cc, input_atten_dB, ZLNA):
    """The current per unit drive one resonator at rest carries at each
    of *freqs*: its linear response, as the solver computes it."""
    r2, r3 = _p_attenuator(input_atten_dB, 50.0)
    out = np.empty(len(freqs), dtype=np.complex128)
    for k in range(len(freqs)):
        out[k] = _drive_current(2.0 * np.pi * freqs[k], 1.0, L, R, C, Cc,
                                ZLNA, r2, r3)
    return out


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _converged_lekid_parameters_par(
    frequency, amplitude, 
    L_array, R_array, C_array, Cc_array,
    base_Lk, base_Lg, base_L_junk,
    input_atten_dB, ZLNA,
    Istar, tolerance, max_iterations, initial_currents,
    damp=0.1, damp_min=0.02, damp_max=0.5
):
    """
    Steady state of the readout-current nonlinearity.

    Each resonator's kinetic inductance grows with the current it
    carries, Lk = Lk0 (1 + |I|^2 / Istar^2), which shifts its resonance
    and so changes the current the generator drives through it: the
    steady state is the self-consistent current I = F(I).  This is the
    Kerr-type detuning x = x0 + E/E* of Rouble et al., arXiv:2607.09178,
    with the stored energy written as a current.  Below the critical
    drive (asymmetry parameter a = 4 sqrt(3) / 9) there is one steady
    state; above it the response is bistable, two stable driven states
    (high current on the shifted resonance, low current off it) with an
    unstable one between, and which the resonator is in depends on how
    it got there, so the response is hysteretic.

    The iteration I <- I + d (F(I) - I) starts from *initial_currents*,
    the driven state each resonator was last in, so a swept tone keeps
    its state until that state ceases to exist.  The step d is
    1 / (1 - s), s the real part of the secant slope of F between the
    last two iterates, clamped to [damp_min, damp_max]: near-linear
    points converge in a few iterations, the high-current state (slope
    well below zero) stays stable, and a positive step can never settle
    in the unstable state, whose slope exceeds one.  The first step,
    and any whose slope estimate is degenerate, is *damp*.
    
    Performs the entire convergence calculation in compiled code for
    maximum performance (2-5x speedup over Python loops).
    
    Parameters
    ----------
    frequency : float
        Probe frequency in Hz
    amplitude : float
        Input voltage amplitude
    L_array, R_array, C_array, Cc_array : ndarray
        Circuit parameters for all resonators.  L_array only sets the
        count: the inductance is rebuilt from base_Lk under the
        currents, and the returned L is the converged total.
    base_Lk, base_Lg, base_L_junk : ndarray
        Base inductance values (before current modification).
        Only Lk changes with current; Lg and L_junk are fixed.
    input_atten_dB : float
        Input attenuation in dB
    ZLNA : complex
        LNA impedance
    Istar : float
        Characteristic current
    tolerance : float
        Convergence tolerance
    max_iterations : int
        Maximum convergence iterations
    initial_currents : ndarray (complex)
        The current each resonator starts from
    damp, damp_min, damp_max : float
        The first step, and the bounds of the adaptive step
        
    Returns
    -------
    L_converged : ndarray
        Converged total inductance values (Lk + Lg + L_junk)
    R_converged : ndarray
        Converged resistance values (unchanged)
    currents_converged : ndarray (complex)
        Converged resonator currents
    iterations : int
        Number of iterations to convergence
    """
    n = len(L_array)
    w = 2.0 * np.pi * frequency
    
    # Working arrays: the state the seed currents set
    currents_array = initial_currents.copy()
    current_factors = 1.0 + (np.abs(currents_array)**2 / (Istar * Istar))
    L_work = np.empty(n, dtype=np.float64)
    for i in prange(n):
        L_work[i] = base_Lk[i] * current_factors[i] + base_Lg[i] + base_L_junk[i]
    steps = np.empty(n, dtype=np.float64)
    g_prev = np.zeros(n, dtype=np.complex128)
    I_prev = np.zeros(n, dtype=np.complex128)
    
    r2, r3 = _p_attenuator(input_atten_dB, 50.0)

    actual_iterations = 0

    # Convergence loop
    for iteration in range(max_iterations):
        # Step 1: the current the generator drives through each
        # resonator at its present inductance
        currents_new = np.zeros(n, dtype=np.complex128)
        
        for i in prange(n):
            currents_new[i] = _drive_current(
                w, amplitude, L_work[i], R_array[i], C_array[i], Cc_array[i],
                ZLNA, r2, r3)
        
        # Step 3: the step towards self-consistency.  Each resonator's
        # current I sets its inductance, which sets the current F(I) it
        # would carry; we want I = F(I), so we move by a fraction d of
        # the mismatch g = F(I) - I.  A fixed d converges only if the
        # slope s of F is not too far from 0: the error shrinks by
        # |1 + d (s - 1)| per iteration.  Choosing d = 1 / (1 - s) makes
        # that factor zero, Newton's method on the mismatch with s
        # estimated from the last two iterates (a secant).  s is
        # complex; we keep the real part so d is a plain damping and
        # never rotates the step.  The clamp keeps d positive and
        # bounded: in the unstable driven state s > 1 would make d
        # negative, and a positive d cannot settle there; at the edge of
        # bistability s -> 1 and 1 / (1 - s) blows up.  The first
        # iteration, and any with a degenerate estimate, use the fixed
        # damp.
        g = currents_new - currents_array
        for i in prange(n):
            steps[i] = damp
            if iteration > 0:
                dI = currents_array[i] - I_prev[i]
                if abs(dI) > 1e-300:
                    s = 1.0 + (g[i] - g_prev[i]) / dI      # F = I + g
                    den = 1.0 - s
                    if abs(den) > 1e-300:
                        steps[i] = min(max((1.0 / den).real, damp_min),
                                       damp_max)
            I_prev[i] = currents_array[i]
            g_prev[i] = g[i]
        currents_array = currents_array + steps * g
        
        # Step 4: Calculate new current factors
        new_factors = 1.0 + (np.abs(currents_array)**2 / (Istar * Istar))
        
        # Step 5: Check convergence
        if iteration > 0:
            factor_change = np.max(np.abs(new_factors - current_factors))
            
            # Early stop after minimum iterations
            if iteration >= 3 and factor_change < tolerance:
                actual_iterations = iteration + 1
                break
            
            # Strict convergence
            if factor_change < tolerance * 0.1:
                actual_iterations = iteration + 1
                break
        
        # Step 6: Update factors and inductances
        current_factors = new_factors
        
        # Update L values: Lk changes with current; Lg and L_junk are fixed
        for i in prange(n):
            Lk_work = base_Lk[i] * current_factors[i]
            L_work[i] = Lk_work + base_Lg[i] + base_L_junk[i]
    
    if actual_iterations == 0:
        actual_iterations = max_iterations

    # The current the converged inductance carries.  The iterate stops
    # when the inductance stops changing, which at a drive too small to
    # change it is after the first damped step, a fraction of the way
    # to F(I); the inductance is right either way, the current is F(I).
    for i in prange(n):
        currents_array[i] = _drive_current(
            w, amplitude, L_work[i], R_array[i], C_array[i], Cc_array[i],
            ZLNA, r2, r3)

    return L_work, R_array, currents_array, actual_iterations


_converged_lekid_parameters_ser = _serial_twin(
    _converged_lekid_parameters_par, "_converged_lekid_parameters_ser",
    fastmath=True)


def converged_lekid_parameters(frequency, amplitude, L_array, *args,
                               initial_currents=None, damp=0.1,
                               damp_min=0.02, damp_max=0.5):
    """Self-consistent convergence loop for current-dependent inductance,
    from *initial_currents* (every resonator at rest when None)."""
    fn = (_converged_lekid_parameters_par
          if len(L_array) >= PARALLEL_MIN_N
          else _converged_lekid_parameters_ser)
    if initial_currents is None:
        initial_currents = np.zeros(len(L_array), dtype=np.complex128)
    # Every argument passed: a call that leaves a defaulted one out takes
    # numba's Python dispatch path, ten times the call.
    return fn(frequency, amplitude, L_array, *args,
              np.ascontiguousarray(initial_currents, dtype=np.complex128),
              float(damp), float(damp_min), float(damp_max))


@jit(nopython=True, cache=True)
def states_apart(a, b, Istar, frac):
    """Whether two sets of currents put any resonator in different
    states: the two states of a bifurcated resonance differ by ten
    times in current, adjacent points in one state by a fraction
    (frac).  Under 1e-3 Istar a current changes Lk by 1e-6, the
    resonance by a hundredth of a linewidth: at rest, whatever the
    ratio."""
    for i in range(len(a)):
        big = max(abs(a[i]), abs(b[i]))
        if big > 1e-3 * Istar and abs(a[i] - b[i]) > frac * big:
            return True
    return False


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _converge_tones_par(freqs, amps, has_seed, seed_f, seeds, run_start,
                        base_Lk_runs, base_R_runs, C_array, Cc_array, base_Lg,
                        base_L_junk, input_atten_dB, ZLNA, Istar, tolerance,
                        max_iterations, damp, damp_min, damp_max, follow_hz,
                        max_steps, apart_frac):
    """converge_tones: each tone on its own thread, through its runs in
    order with the per-tone solver."""
    T = len(freqs)
    M = run_start[T]
    n = len(C_array)
    L_out = np.empty((M, n), dtype=np.float64)
    I_out = np.empty((M, n), dtype=np.complex128)
    passes = np.zeros(M, dtype=np.int64)
    for k in prange(T):
        f = freqs[k]
        seed = np.zeros(n, dtype=np.complex128)
        steps = 0
        if has_seed[k]:
            steps = 1
            if follow_hz > 0.0:
                steps = max(1, int(np.ceil(abs(f - seed_f[k]) / follow_hz)))
            if steps <= max_steps:
                seed[:] = seeds[k]
            else:
                steps = 0
        r0 = run_start[k]
        for r in range(r0, run_start[k + 1]):
            R_run = base_R_runs[r]
            L, R, I, its = _converged_lekid_parameters_ser(
                f, amps[k], R_run, R_run, C_array, Cc_array,
                base_Lk_runs[r], base_Lg, base_L_junk, input_atten_dB,
                ZLNA, Istar, tolerance, max_iterations, seed.copy(),
                damp, damp_min, damp_max)
            passes[r] = 1
            if r == r0 and steps > 1 and states_apart(I, seed, Istar,
                                                      apart_frac):
                I = seed.copy()
                for j in range(1, steps + 1):
                    fj = seed_f[k] + (f - seed_f[k]) * j / steps
                    L, R, I, its = _converged_lekid_parameters_ser(
                        fj, amps[k], R_run, R_run, C_array, Cc_array,
                        base_Lk_runs[r], base_Lg, base_L_junk,
                        input_atten_dB, ZLNA, Istar, tolerance,
                        max_iterations, I, damp, damp_min, damp_max)
                passes[r] = 1 + steps
            L_out[r] = L
            I_out[r] = I
            seed = I
    return L_out, I_out, passes


_converge_tones_ser = _serial_twin(_converge_tones_par, "_converge_tones_ser",
                                   fastmath=True)

PARALLEL_MIN_TONES = 8
RUN_ELEMENTS_PER_CALL = 2_000_000    # (runs x resonators) per call, 48 MB out


def converge_tones(freqs, amps, has_seed, seed_f, seeds, run_start,
                   base_Lk_runs, base_R_runs, *args, damp=0.1, damp_min=0.02,
                   damp_max=0.5):
    """Converge every resonator for each tone at each of its QP states,
    the tones in parallel.  Tone k's runs are run_start[k]:run_start[k+1]
    of base_Lk_runs and base_R_runs (the kinetic inductance and the
    resistance the QP density of each instant gives every resonator), in
    time order; the first is seeded from the state
    the tone left its resonators in (seeds[k] at seed_f[k]) when it has
    one (has_seed[k]), else from rest, and each later run from the one
    before.  One step from the seed is the answer where the currents
    move by a fraction; where a resonator jumps state, that state may
    have ended between the two frequencies, and only following the tone
    in steps of follow_hz from where it sat says where, so the first run
    is retaken that way.  A move of more than max_steps of them is a
    new tone, solved from rest.  *args: C, Cc, base_Lg, base_L_junk,
    input_atten_dB, ZLNA, Istar, tolerance, max_iterations, follow_hz,
    max_steps, apart_frac.  Returns (L, currents, solver passes), one
    row per run."""
    freqs = np.ascontiguousarray(freqs, dtype=np.float64)
    amps = np.ascontiguousarray(amps, dtype=np.float64)
    has_seed = np.ascontiguousarray(has_seed, dtype=np.bool_)
    seed_f = np.ascontiguousarray(seed_f, dtype=np.float64)
    seeds = np.ascontiguousarray(seeds, dtype=np.complex128)
    run_start = np.ascontiguousarray(run_start, dtype=np.int64)
    base_Lk_runs = np.ascontiguousarray(base_Lk_runs, dtype=np.float64)
    base_R_runs = np.ascontiguousarray(base_R_runs, dtype=np.float64)
    (C, Cc, base_Lg, base_L_junk, input_atten_dB, ZLNA, Istar, tolerance,
     max_iterations, follow_hz, max_steps, apart_frac) = args
    consts = (C, Cc, base_Lg, base_L_junk, float(input_atten_dB),
              complex(ZLNA), float(Istar), float(tolerance),
              int(max_iterations), float(damp), float(damp_min),
              float(damp_max), float(follow_hz), int(max_steps),
              float(apart_frac))
    n = len(C)
    T = len(freqs)
    per_call = max(1, RUN_ELEMENTS_PER_CALL // max(n, 1))
    out = []
    k0 = 0
    while k0 < T:
        k1 = k0 + 1
        while k1 < T and run_start[k1 + 1] - run_start[k0] <= per_call:
            k1 += 1
        fn = (_converge_tones_par if k1 - k0 >= PARALLEL_MIN_TONES
              else _converge_tones_ser)
        r0, r1 = run_start[k0], run_start[k1]
        out.append(fn(freqs[k0:k1], amps[k0:k1], has_seed[k0:k1],
                      seed_f[k0:k1], seeds[k0:k1],
                      np.ascontiguousarray(run_start[k0:k1 + 1] - r0),
                      base_Lk_runs[r0:r1], base_R_runs[r0:r1], *consts))
        k0 = k1
    if len(out) == 1:
        return out[0]
    return tuple(np.concatenate([o[i] for o in out]) for i in range(3))


# ============================================================================
# S21 Calculation
# ============================================================================

# ============================================================================
# Parallel Resonator S21 Calculation
# ============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def compute_s21_parallel(
    fc, Vin,
    L_array, C_array, R_array, Cc_array,
    ZLNA, GLNA, input_atten_dB, system_termination
):
    """
    Calculate S21 for multiple resonators in parallel on a transmission line.
    
    This properly combines all resonators in parallel before calculating
    the transmission to the load, giving physically correct dips at resonance.
    
    Parameters
    ----------
    fc : float
        Probe frequency in Hz
    Vin : float
        Input voltage amplitude
    L_array : ndarray
        Total inductance for each resonator (Lk + Lg + L_junk)
    C_array : ndarray
        Capacitance for each resonator
    R_array : ndarray
        Resistance for each resonator
    Cc_array : ndarray
        Coupling capacitance for each resonator
    ZLNA : complex
        LNA impedance (load at end of transmission line)
    GLNA : float
        LNA gain
    input_atten_dB : float
        Input attenuation in dB
    system_termination : float
        System termination impedance
        
    Returns
    -------
    complex
        S21 transmission coefficient (V_load / V_input)
    """
    n = len(L_array)
    w = 2.0 * np.pi * fc
    
    # Fixed attenuation factor (not dependent on load)
    att_factor = 10.0**(-input_atten_dB/20.0)  # Negative for attenuation
    
    # Step 1: Calculate impedance of each resonator
    Z_resonators = np.zeros(n, dtype=np.complex128)
    
    for i in range(n):
        # Parallel LC impedance (L_array already includes L_junk)
        if C_array[i] > 0:
            ZC = 1.0 / (1j * w * C_array[i])
            ZL = 1j * w * L_array[i]
            Z_parallel_inv = 1.0/ZC + 1.0/(ZL + R_array[i])
            Z_parallel = 1.0 / Z_parallel_inv
        else:
            Z_parallel = 1j * w * L_array[i] + R_array[i]
        
        # Add coupling capacitor (L_junk already included in L_array)
        ZCc = 1.0 / (1j * w * Cc_array[i])
        Z_resonators[i] = Z_parallel + ZCc
    
    # Step 2: Combine all resonators in parallel
    # 1/Z_total = sum(1/Z_i) for parallel impedances
    Z_total_inv = 0.0 + 0.0j
    for i in range(n):
        Z_total_inv += 1.0 / Z_resonators[i]
    
    # Avoid division by zero - if no resonators, use very high impedance
    if abs(Z_total_inv) > 1e-12:
        Z_total_resonators = 1.0 / Z_total_inv
    else:
        Z_total_resonators = 1e12 + 0j  # Very high impedance (no loading)
    
    # Step 3: Simple transmission line model
    # The resonators shunt current away from the load
    # Transmission coefficient is the voltage divider ratio
    
    # When resonators are off-resonance (high Z), most signal reaches load
    # When resonators are on-resonance (low Z), they shunt signal away
    
    # Transmission coefficient: how much reaches the load
    # S21 = ZLNA / (ZLNA + Z_series) where Z_series represents series loss
    # But our resonators are in parallel, so they act as shunt admittance
    
    # The proper formula for shunt elements on a transmission line:
    # V_load/V_source = 1 / (1 + Z_line/Z_shunt) for matched line
    # For our case with parallel resonators shunting to ground:
    # S21 = Z_shunt / (Z_shunt + Z_line)
    
    # Simplified model: resonators in parallel with load
    # Total load seen = Z_total_resonators || ZLNA
    Z_eff = 1.0 / (1.0/Z_total_resonators + 1.0/ZLNA)
    
    # Transmission coefficient (assuming Z0 source impedance)
    Z0 = system_termination
    S21_raw = Z_eff / (Z_eff + Z0)
    
    # Apply fixed attenuation and gain
    # Factor of 2 converts the voltage divider ratio V_load/V_source = Z/(Z+Z0)
    # into the proper S-parameter S21 = 2*Z/(Z+Z0), which equals 1.0 for a
    # matched thru (Z == Z0).
    S21 = 2.0 * S21_raw * att_factor * GLNA
    
    return S21


@jit(nopython=True, cache=True, fastmath=True)
def s21_of_one(freqs, L, R, C, Cc, ZLNA, GLNA, input_atten_dB,
               system_termination):
    """compute_s21_parallel of one resonator alone, at each of *freqs*
    per unit drive."""
    L1 = np.array([L])
    C1 = np.array([C])
    R1 = np.array([R])
    Cc1 = np.array([Cc])
    out = np.empty(len(freqs), dtype=np.complex128)
    for k in range(len(freqs)):
        out[k] = compute_s21_parallel(freqs[k], 1.0, L1, C1, R1, Cc1, ZLNA,
                                      GLNA, input_atten_dB, system_termination)
    return out


@jit(nopython=True, cache=True, fastmath=True)
def compute_s21_batch(fc, Vin, L2d, C2d, R2d, Cc_array,
                      ZLNA, GLNA, input_atten_dB, system_termination):
    """compute_s21_parallel for each row of (L2d, C2d, R2d): one
    dispatch per batch of samples instead of one per sample.  The
    per-row arithmetic is the same function."""
    n = L2d.shape[0]
    out = np.zeros(n, dtype=np.complex128)
    for k in range(n):
        out[k] = compute_s21_parallel(fc, Vin, L2d[k], C2d[k], R2d[k],
                                      Cc_array, ZLNA, GLNA, input_atten_dB,
                                      system_termination)
    return out
