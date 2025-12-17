"""
This module is derived from and based on the mr_resonator project by Maclean Rouble:
https://github.com/macleaner/mr_resonator

Modifications in this repository include:
- Trimmed to include only functions used by the Mock CRS framework
- Consolidation and integration with local JIT physics approximations

Original project license: see rfmux/mr_resonator/LICENSE (upstream LICENSE retained)
"""

import numpy as np

# Import Bessel approximations from JIT physics module
from .jit_physics import bessel_k0 as bessel_k0_approx, bessel_i0 as bessel_i0_approx

# Import MR_LEKID
from .mr_lekid import MR_LEKID as MR_LEKID


h = 6.626e-34      # Planck's constant [J·s]
kb = 1.38e-23      # Boltzmann constant [J/K]
mu0 = 4e-7 * 3.14159265359  # Permeability of free space [H/m] ≈ 1.257e-6


class MR_complex_resonator(): 
    """
    High-level KID physics model that maps material/geometry/operating-point
    parameters to a circuit-level LEKID (MR_LEKID) used by the mock framework.

    Overview
    --------
    - Computes complex surface impedance Zs from (σ1, σ2) using Gao-inspired
      approximations for superconducting films.
    - Converts Zs to total series resistance R and kinetic inductance Lk using
      geometry scaling (length/width).
    - Creates a MR_LEKID instance with (R, Lk, Lg, C, Cc, ...) and recenters
      the readout frequency via MR_LEKID.compute_fr() for a self-consistent
      operating point.
    - Provides calc_nqp() and related helpers to update base physics from
      quasiparticle number density and temperature.

    Notes
    -----
    - Units are explicitly called out in method docstrings (Hz, Ω, H, F, K).
    - Bessel functions K0/I0 are approximated via jit_physics for speed;
      relative error is typically ~1e-6 for the intended ranges.
    - MockResonatorModel reads attributes like readout_f, T, Delta0, N0,
      sigmaN, thickness, width, length, R_spoiler, and lekid for vectorized
      S21 calculations and convergence.
    """
    
    # Material constants for Aluminum (internal units: µm⁻³ J⁻¹ for N0, seconds for tau0)
    # N0 conversion: 1.72e10 µm⁻³ eV⁻¹ / (1.602e-19 J/eV) = 1.074e29 µm⁻³ J⁻¹
    AL_N0 = 1.72e10 / 1.602e-19     # Density of states [µm⁻³ J⁻¹]
    AL_TAU0 = 438e-9                # Electron-phonon time [s] (de Visser thesis)
    AL_TC = 1.2                     # Critical temperature [K]
    
    def __init__(self, T=0.12, base_readout_f=1e9, VL=540e-18, width=2e-6, thickness=30e-9, 
                 length=None, C=0.5e-12, Cc=0.005e-12, alpha_k=0.5, fix_Lg=None, R_spoiler=0, L_junk=0,
                 # Material parameters - Al defaults (can be overridden for custom materials)
                 Tc=None, N0=None, tau0=None, sigmaN=1./(4*20e-9),
                 # Operating point
                 Popt=1e-18, opt_eff=0.5, pb_eff=0.7, nu_opt=150e9, big_sigma_factor=1e-4, nstar=0,
                 # Readout chain
                 Vin=0.15e-3, input_atten_dB=20, ZLNA=50., GLNA=1,
                 # Deprecated parameter (kept for backward compatibility)
                 material=None,
                 verbose=False):
        """
        High-level KID physics model.
        
        Material Parameters
        -------------------
        Tc : float, optional
            Critical temperature [K]. Default: 1.2 K (Aluminum)
        N0 : float, optional
            Density of states at Fermi level [µm⁻³ J⁻¹]. 
            Default: 1.074e29 (Aluminum: 1.72e10 µm⁻³ eV⁻¹ converted to SI)
        tau0 : float, optional
            Characteristic electron-phonon time [s]. Default: 438 ns (Aluminum)
        sigmaN : float, optional
            Normal state conductivity [S/m]. Default: 1/(4*20nm) for thin Al
        """
        self.T = T
        self.readout_f = base_readout_f
        self.Popt = Popt
        self.opt_eff = opt_eff
        self.pb_eff = pb_eff
        self.nu_opt = nu_opt
        self.big_sigma_factor = big_sigma_factor
        
        # Handle deprecated material parameter
        if material is not None and material != 'Al':
            import warnings
            warnings.warn(
                f"material='{material}' is deprecated. Use explicit Tc, N0, tau0 parameters instead. "
                "Ignoring material name and using provided or default values.",
                DeprecationWarning
            )

        self.R_spoiler = R_spoiler
        self.L_junk = L_junk
               
        self.width = width
        self.thickness = thickness
        if length is not None:
            self.length = length
            VL = self.width * self.thickness * self.length
        else:
            self.length = VL / (self.width * self.thickness)
        self.VL = VL
        self.VL_um3 = VL*1e18 # in um^3; this is conventionally the units for nqp etc
        
        self.sigmaN = sigmaN
    
        # Material properties - use explicit values or Al defaults
        self.Tc = Tc if Tc is not None else self.AL_TC
        self.N0 = N0 if N0 is not None else self.AL_N0
        self.tau0 = tau0 if tau0 is not None else self.AL_TAU0
        
        if self.T >= self.Tc:
            raise ValueError('Error: cannot set operational temperature equal to or above transition temperature.')
        self.nstar = nstar


        self.Delta0 = 1.76 * kb * self.Tc
            
        # compute initial guess at resonator dark conductances and circuit values
        nqp = self.calc_nqp(T=T, Popt=self.Popt, opt_eff=opt_eff, pb_eff=pb_eff)
        self.sigma1_initial = self.calc_sigma1(f=self.readout_f, T=self.T, Popt=self.Popt)
        self.sigma2_initial = self.calc_sigma2(f=self.readout_f, T=self.T, Popt=self.Popt)
        self.sigma_initial = self.sigma1_initial - 1.j*self.sigma2_initial
        self.Zs_initial = self.calc_Zs(f=self.readout_f, sigma=self.sigma_initial)
        self.R_initial, self.Lk_initial = self.calc_R_L(f=self.readout_f, Zs=self.Zs_initial)
        # R_spoiler already included in calc_R_L(), no need to add again
        
        # generate dark resonator
        self.input_atten_dB = input_atten_dB
        self.C = C
        self.Cc = Cc
        self.Vin = Vin
        if fix_Lg is None:
            self.alpha_k = alpha_k
            self.Lg = (self.Lk_initial - self.alpha_k*self.Lk_initial) / self.alpha_k
        else:
            self.Lg = fix_Lg
            self.alpha_k = self.Lk_initial / (self.Lk_initial + self.Lg)
        
        self.lekid_params_initial = dict(R=self.R_initial, Lk=self.Lk_initial, Lg=self.Lg, C=self.C, Cc=self.Cc, Vin=self.Vin, input_atten_dB=self.input_atten_dB, ZLNA=ZLNA, GLNA=GLNA, L_junk=self.L_junk)
        if verbose:
            print('initial parameters:')
            print(self.lekid_params_initial)
        self.lekid_initial = MR_LEKID(**self.lekid_params_initial)
        
        if self.Lk_initial < 0:
            self.readout_f = base_readout_f
            if verbose:
                print('Warning: initial Lk guess is negative.')
        else:
#             self.readout_f = 1./np.sqrt(2*np.pi*(self.Lk_dark + self.Lg) * self.C)
            self.readout_f = self.lekid_initial.compute_fr()
            if verbose:
                print('base readout f: %.4e; readout f now: %.4e'%(base_readout_f, self.readout_f))
        
        # recompute the resonator parameters using the updated readout frequency:
        nqp = self.calc_nqp(T=T, Popt=self.Popt, opt_eff=opt_eff, pb_eff=pb_eff)
        self.sigma1_dark = self.calc_sigma1(f=self.readout_f, T=self.T, Popt=self.Popt)
        self.sigma2_dark = self.calc_sigma2(f=self.readout_f, T=self.T, Popt=self.Popt)
        self.sigma_dark = self.sigma1_dark - 1.j*self.sigma2_dark
        self.Zs_dark = self.calc_Zs(f=self.readout_f, sigma=self.sigma_dark)
        self.R_dark, self.Lk_dark = self.calc_R_L(f=self.readout_f, Zs=self.Zs_dark)
        # R_spoiler already included in calc_R_L(), no need to add again
        
        # generate dark resonator
        self.C = C
        self.Cc = Cc
        self.Vin = Vin
        if fix_Lg is None:
            self.alpha_k = alpha_k
            self.Lg = (self.Lk_dark - self.alpha_k*self.Lk_dark) / self.alpha_k
        else:
            self.Lg = fix_Lg
            self.alpha_k = self.Lk_dark / (self.Lk_dark + self.Lg)
        
        self.lekid_params_dark = dict(R=self.R_dark, Lk=self.Lk_dark, Lg=self.Lg, C=self.C, Cc=self.Cc, Vin=self.Vin, input_atten_dB=self.input_atten_dB, ZLNA=ZLNA, GLNA=GLNA, L_junk=self.L_junk)
        self.lekid = MR_LEKID(**self.lekid_params_dark, verbose=verbose)
        self.readout_f = self.lekid.compute_fr()
        if verbose:
            print(self.lekid_params_dark)

    

    def calc_Zs(self, f, sigma, thickness=None):#, sigma2=None):
        """
        Compute complex surface impedance from complex conductance.

        Parameters
        ----------
        f : float
            Probe frequency [Hz]
        sigma : complex
            Complex conductance σ = σ1 - j σ2 [S/m]
        thickness : float, optional
            Film thickness [m] (defaults to self.thickness)

        Returns
        -------
        complex
            Surface impedance Zs [Ω]
        """
        
        if thickness is None:
            thickness = self.thickness
#         print(thickness)
        root1 = (1.j*2*np.pi*f*mu0)/sigma
        cotharg = thickness * np.sqrt(1.j*2*np.pi*f*mu0*sigma)
        Zs = np.sqrt(root1) * 1./np.tanh(cotharg)
        return Zs


    def calc_R_L(self, f, Zs):
        """
        Convert surface impedance to total series R and kinetic inductance Lk.

        Parameters
        ----------
        f : float
            Probe frequency [Hz]
        Zs : complex
            Surface impedance [Ω]

        Returns
        -------
        tuple[float, float]
            (R_total [Ω], Lk_total [H])

        Notes
        -----
        Geometry scaling: multiply surface quantities by (length/width).
        R_spoiler is added as an extra series resistance term.
        """
        R = (Zs.real ) * (self.length / self.width) + self.R_spoiler
        L = (Zs.imag / (2*np.pi*f)) * (self.length / self.width)
        return R, L
    

    def zeta(self, f, T):
        """
        Dimensionless parameter ζ = h f / (2 k_B T).

        Parameters
        ----------
        f : float
            Frequency [Hz]
        T : float
            Temperature [K]

        Returns
        -------
        float
            ζ (dimensionless)
        """
        return h * f / (2 * kb * T)
    
    def calc_sigma1(self, f=None, nqp=None, T=None, Popt=None, pb_eff=None, opt_eff=None):
        """
        Real part of complex conductivity (σ1) following Gao (Eq. 2.96–2.97).

        Parameters
        ----------
        f : float, optional
            Probe frequency [Hz] (defaults to self.readout_f)
        nqp : float, optional
            Quasiparticle number density [um^-3] (defaults to calc_nqp())
        T : float, optional
            Temperature [K]
        Popt : float, optional
            Optical loading [W]
        pb_eff, opt_eff : float, optional
            Pair-breaking and optical efficiencies

        Returns
        -------
        float
            σ1 [S/m]

        Notes
        -----
        Uses fast bessel_k0 approximation from jit_physics; relative error ~1e-6
        over intended ranges.
        """
        
        if f is None:
            f = self.readout_f
        if T is None:
            T = self.T
        if Popt is None:
            Popt = self.Popt
        if opt_eff is None:
            opt_eff = self.opt_eff
        if pb_eff is None:
            pb_eff = self.pb_eff
        if nqp is None:
            nqp = self.calc_nqp(T=T, Popt=Popt, pb_eff=pb_eff, opt_eff=opt_eff)

        zeta = self.zeta(f=f, T=T)
        
        # Always use fast approximation (JIT-compiled)
        K0 = bessel_k0_approx(zeta)

        x1 = 2 * self.Delta0/(h*f)
        x2 = nqp / (self.N0 * np.sqrt(2*np.pi*kb*T*self.Delta0))

        return x1 * x2 * np.sinh(zeta) * K0 * self.sigmaN

    
    def calc_sigma2(self, f=None, nqp=None, T=None, Popt=None, pb_eff=None, opt_eff=None):
        """
        Imaginary part of complex conductivity (σ2) following Gao (Eq. 2.96–2.97).

        Parameters
        ----------
        f : float, optional
            Probe frequency [Hz] (defaults to self.readout_f)
        nqp : float, optional
            Quasiparticle number density [um^-3] (defaults to calc_nqp())
        T : float, optional
            Temperature [K]
        Popt : float, optional
            Optical loading [W]
        pb_eff, opt_eff : float, optional
            Pair-breaking and optical efficiencies

        Returns
        -------
        float
            σ2 [S/m]

        Notes
        -----
        Uses fast bessel_i0 approximation from jit_physics; relative error ~1e-6
        over intended ranges.
        """
        
        if f is None:
            f = self.readout_f
        if T is None:
            T = self.T
        if Popt is None:
            Popt = self.Popt
        if opt_eff is None:
            opt_eff = self.opt_eff
        if pb_eff is None:
            pb_eff = self.pb_eff
        if nqp is None:
            nqp = self.calc_nqp(T=T, Popt=Popt, pb_eff=pb_eff, opt_eff=opt_eff)
        Delta0 = self.Delta0
        
        zeta = self.zeta(f=f, T=T)
        
        # Always use fast approximation (JIT-compiled)
        I0 = bessel_i0_approx(zeta)

        x1 = np.pi * Delta0 / (h*f)
        x2 = nqp / (2*self.N0*Delta0)
        x3 = np.sqrt(2*Delta0/(np.pi*kb*T)) * np.exp(-zeta) * I0

        return x1 * (1 - x2*(1+x3)) * (self.sigmaN)

        
    def calc_nqp(self, T=None, Popt=None, opt_eff=None, pb_eff=None):
        """
        Quasiparticle number density from thermal + optical generation.

        Parameters
        ----------
        T : float, optional
            Temperature [K]
        Popt : float, optional
            Optical loading [W]
        opt_eff : float, optional
            Optical efficiency (0–1)
        pb_eff : float, optional
            Pair-breaking efficiency (0–1)

        Returns
        -------
        float
            nqp [um^-3]

        Notes
        -----
        nqp = sqrt(n_th^2 + (rate_optical / R)) - nstar
        where R depends on Delta(T), material constants, and tau0.
        """
        
        if T is None:
            T = self.T
        if Popt is None:
            Popt = self.Popt
        if opt_eff is None:
            opt_eff = self.opt_eff
        if pb_eff is None:
            pb_eff = self.pb_eff
            
        Delta = self.calc_Delta_gao(T=T)
        
        R = (2 * Delta)**2 / (2*self.N0 * self.tau0 * (kb * self.Tc)**3)
        
        nth = self.calc_nqp_th(T=T) + self.nstar
        rate_thermal = R * nth**2
        rate_optical = pb_eff * opt_eff * Popt / (Delta * self.VL_um3)
                
        return np.sqrt(nth**2 + rate_optical/R) - self.nstar

    def calc_nqp_th(self, T=None):
        """
        Thermal quasiparticle number density at temperature T.

        Parameters
        ----------
        T : float, optional
            Temperature [K]

        Returns
        -------
        float
            nqp_th [um^-3]

        Notes
        -----
        Uses the standard low-temperature approximation for superconductors.
        """
        
        if T is None:
            T = self.T
        Delta = self.calc_Delta_gao(T)
    
        nqp = 2 * self.N0 * np.sqrt(2 * np.pi * kb * T * Delta) * np.exp(-Delta / (kb * T))
        return nqp

    def calc_Delta_gao(self, T=None):
        """
        Superconducting gap energy Δ(T) via Gao's approximation.

        Parameters
        ----------
        T : float, optional
            Temperature [K]

        Returns
        -------
        float
            Δ(T) [J]

        Validity
        --------
        Representative up to ~0.7 Tc.

        Notes
        -----
        This is used to update R and Lk through conductivities and Zs.
        """
        if T is None:
            T = self.T
        
        innerexp = np.exp(-self.Delta0 / (kb*T))
        outerexp = np.exp( (-2*np.pi*kb * T / self.Delta0) * innerexp)
        return self.Delta0 * outerexp
