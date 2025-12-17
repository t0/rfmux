"""
This module is derived from and based on the mr_resonator project by Maclean Rouble:
https://github.com/macleaner/mr_resonator

Modifications in this repository include:
- Trimmed to include only functions used by the Mock CRS framework
- Consolidation and integration with local JIT physics approximations

Original project license: see rfmux/mr_resonator/LICENSE (upstream LICENSE retained)
"""

import numpy as np
from typing import Optional


class MR_LEKID():
    """
    Circuit-level LEKID model used by the mock framework.

    Model
    -----
    - Parallel RLC (R || jωL || 1/(jωC)) in series with coupling capacitor Cc,
      and optional parasitic inductance L_junk.
    - The network is driven through a p-type, 3-resistor attenuator synthesized for Z0=50 Ω.
    - The LNA is modeled as a complex input impedance ZLNA with linear gain GLNA.

    Usage
    -----
    Instances are created by MR_complex_resonator to carry circuit parameters
    (R, Lk, Lg, C, Cc, etc.). Methods such as compute_Vout() and compute_fr()
    are called by MockResonatorModel during S21 evaluation and convergence.
    """
    
    def __init__(
        self,
        C: float = 1e-12,
        R: float = 1e-6,
        Cc: float = 5e-15,
        Lk: float = 1e-9,
        Lg: Optional[float] = None,
        alpha_k: float = 0.5,
        L_junk: float = 0.0,
        Qi: Optional[float] = None,
        Qc: Optional[float] = None,
        Vin: Optional[float] = None,
        fr_presign2: int = -1,
        system_termination: float = 50.0,
        input_atten_dB: float = 20.0,
        ZLNA: complex = complex(50.0, 0.0),
        GLNA: float = 1.0,
        name: str = 'LR SERIES',
        LNA_noise_temperature: float = 6.0,
        plot_response: bool = False,
        verbose: bool = False
    ):
        
            
        self.Lk = Lk
        if Lg is None:
            # If Lg not provided, derive it from alpha_k parameter
            # alpha_k = Lk / (Lk + Lg) => Lg = Lk * (1 - alpha_k) / alpha_k
            self.Lg = (self.Lk - alpha_k*self.Lk) / alpha_k
        else:
            self.Lg = Lg
        
        self.L_junk = L_junk
        # L is the total inductance affecting resonance frequency and kinetic inductance fraction
        # L_total = Lk + Lg + L_junk (all inductances in the resonator circuit)
        self.L = self.Lk + self.Lg + self.L_junk
        # Effective kinetic inductance fraction (diluted by geometric and junk inductance)
        self.alpha_k = self.Lk / self.L if self.L > 0 else 0.0

        self.C = C
        self.R = R
        self.Cc = Cc
        self.name = name
        
            
         # readout params
        self.system_termination = system_termination
        self.input_atten_dB = input_atten_dB
        self.ZLNA = ZLNA
        self.GLNA = GLNA # LNA gain
        if Vin is None:
            self.Vin = 1e-5 # arbitrary choice
        else:
            self.Vin = Vin
            
        # noise params
        self.LNA_noise_temperature = LNA_noise_temperature
        self.LNA_noise_vrms_per_rtHz = np.sqrt(1.38e-23 * self.LNA_noise_temperature * 4 * self.system_termination) # over a 1 Hz bw

        self.nonres_flag = False # if the imaginary part of the impedance has no real roots
        
        
            
        if verbose:
            print('Created new resonator, %s, with params:'%(self.name))
            #print('Created new resonator, %s, with params:\nLk=%.2e H, Lg=%.2e H, C=%.2e F, Cc=%.2e F, R=%.2e ohm.'%(self.name, self.Lk, self.Lg, self.C, self.Cc, self.R))
            print(self.generate_res_param_string())
        
        
    def parallel_RLC(self, fc, C=None, L=None, R=None):
        """
        Impedance of the parallel RLC subnetwork at a probe frequency.

        Parameters
        ----------
        fc : float
            Probe frequency [Hz]
        C : float, optional
            Capacitance [F]
        L : float, optional
            Total inductance L = Lk + Lg [H]
        R : float, optional
            Series loss resistance [Ω]

        Returns
        -------
        complex
            Parallel RLC impedance [Ω]
        """
        # where fc is the carrier frequency
        # Compute the impedance of the parallel RLC only
    
        if C is None:
            C = self.C
        if L is None:
            L = self.L
        if R is None:
            R = self.R
            
        w = 2*np.pi*fc
        ZC = 1./(1j*w*C)
        ZL = 1j*w*L

        return 1./(1./ZC + 1./(ZL + R))
    
    def total_impedance(self, fc, C=None, L=None, R=None, Cc=None):
        """
        Total device impedance at fc including series elements.

        Composition
        -----------
        Z_total = Z_parallel_RLC + Z_Cc
        
        Note: L_junk is included in self.L (self.L = Lk + Lg + L_junk) for the
        parallel RLC calculation, so it affects the resonance frequency and
        dilutes the kinetic inductance fraction (alpha_k).

        Parameters
        ----------
        fc : float
            Probe frequency [Hz]
        C, L, R : float, optional
            RLC parameters [F, H, Ω] (L already includes L_junk)
        Cc : float, optional
            Coupling capacitor [F]

        Returns
        -------
        complex
            Total device impedance [Ω]
        """
        if C is None:
            C = self.C
        if L is None:
            L = self.L
        if R is None:
            R = self.R
        if Cc is None:
            Cc = self.Cc
        
        Zres = self.parallel_RLC(fc, L=L, C=C, R=R)
        ZCc = 1./(1j*2*np.pi*fc*Cc)

        return Zres + ZCc
    
    
    
    def compute_Vout(self, fc, Vin=None, L=None, C=None, R=None, Cc=None, ZLNA=None, GLNA=None, input_atten_dB=None):
        """
        Compute LNA input voltage from a single LEKID resonator at probe frequency.

        Parameters
        ----------
        fc : float
            Probe frequency [Hz]
        Vin : float, optional
            Source voltage before the input attenuator [V]
        L, C, R, Cc : float, optional
            Circuit parameters [H, F, Ω, F] (defaults to self values)
        ZLNA : complex, optional
            Complex LNA input impedance [Ω]
        GLNA : float, optional
            LNA gain (linear)
        input_atten_dB : float, optional
            Input attenuator value [dB]

        Returns
        -------
        complex
            Vout at the LNA input after GLNA [V]

        Notes
        -----
        - Attenuator is modeled as a p-type 3-resistor network synthesized for Z0=50 Ω.
        - The resonator is in parallel with ZLNA; the attenuator sits in series before
          that parallel combination.
        - This is the building block used by S21 calculations in the mock framework.
        """
        # Model the resonator as a voltage divider between it in parallel
        # with a complex input impedance of the LNA and an input 50 ohm attenuator,
        # return the voltage of the carrier across the resonator at a given frequency
        # as Vout, given an input carrier voltage at that frequency before the attenuator
        
        if C is None:
            C = self.C
        if L is None:
            L = self.L
        if R is None:
            R = self.R
        if Cc is None:
            Cc = self.Cc
        if Vin is None:
            Vin = self.Vin
        if ZLNA is None:
            ZLNA = self.ZLNA
        if GLNA is None:
            GLNA = self.GLNA
        if input_atten_dB is None:
            input_atten_dB = self.input_atten_dB
            
        r1, r2, r3 = self.get_att_vals(input_atten_dB)
        
        parallel = 1. / ( 1./self.total_impedance(fc, L=L, C=C, R=R, Cc=Cc) + 1./ZLNA )
        
        Vres = Vin * self.ptype(parallel, r1, r2, r3)
        Vout = GLNA * Vres
        return Vout
    

    def ptype(self, rl, r1=61.11, r2=247.5, r3=61.11):
        """
        P-type attenuator voltage division given a load.

        Parameters
        ----------
        rl : float or complex
            Load impedance [Ω]
        r1, r2, r3 : float, optional
            Attenuator resistor values synthesized for Z0 [Ω]

        Returns
        -------
        complex
            VL / Vin ratio at the load node (pre-LNA)
        """
        req = 1. / (1./r3 + 1./rl)
        VLoverVin = req / (req + r2)
        return VLoverVin


    
    def get_att_vals(self, att, z0=50.):
        """
        Compute p-type attenuator component values for target Z0 and attenuation.

        Parameters
        ----------
        att : float
            Attenuation [dB]
        z0 : float, optional
            System impedance [Ω]

        Returns
        -------
        tuple[float, float, float]
            (r1, r2, r3) in ohms
        """
        
        r1 = z0 * ((10**(att/20.) +1) / (10**(att/20.) - 1))
        r3 = r1
        r2 = (z0 / 2.) * ((10**(att/10.) - 1) / (10**(att/20.)))

        return r1, r2, r3
    
    def calc_Iin(self, fc, Vin=None, Zres=None):
        """
        Estimate input current into the attenuator-resonator-LNA network.

        Parameters
        ----------
        fc : float
            Probe frequency [Hz]
        Vin : float, optional
            Source voltage before attenuator [V]
        Zres : complex, optional
            Total device impedance at fc [Ω]

        Returns
        -------
        complex
            Input current [A]
        """
        if Vin is None:
            Vin = self.Vin
        if Zres is None:
            Zres = self.total_impedance(fc)
        r1, r2, r3 = self.get_att_vals(self.input_atten_dB)
        Zsys = 1. / ( 1./Zres + 1./self.ZLNA )
        Zp = 1. / ( 1./Zsys + 1./r3 )
        I2 = Vin / (r2 + Zp)
        Iin = I2 * ( r3 / (Zsys + r3) )
        return Iin
    
    def calc_Ires(self, fc, Zres=None, Iin=None, Vin=None, ZLNA=50., Z_other=None):
        """
        Estimate resonator branch current using a current divider approximation.

        Parameters
        ----------
        fc : float
            Probe frequency [Hz]
        Zres : complex, optional
            Resonator impedance at fc [Ω]
        Iin : complex, optional
            Input current [A]
        Vin : float, optional
            Source voltage before attenuator [V]
        ZLNA : complex or float, optional
            LNA input impedance [Ω]
        Z_other : complex, optional
            Optional additional shunt path [Ω]

        Returns
        -------
        complex
            Resonator branch current [A]

        Notes
        -----
        Approximation treats the last-stage attenuator resistor, resonator, and LNA
        input impedance as a three-way current divider. Sufficient for convergence loop.
        """
        
        if Zres is None:
            Zres = self.total_impedance(fc=fc)
        if Vin is None:
            Vin = self.Vin
        if Iin is None:
            Iin = self.calc_Iin(fc=fc, Zres=Zres, Vin=Vin)

        _, _, r3 = self.get_att_vals(self.input_atten_dB)
            
        if Z_other is not None:
            Zpar = 1./ ( 1./r3 + 1./Zres + 1./ZLNA + 1./Z_other )
        else:
            Zpar = 1./ ( 1./r3 + 1./Zres + 1./ZLNA)
        Ires = Iin * Zpar / Zres
        return Ires
    
    
    
    
    ############
    # L and fr #
    ############
    
    def compute_fr(self, L=None, C=None, R=None, Cc=None, presign2 = 1, verbose=False):
        """
        Estimate resonant frequency including series coupling capacitor.

        Method
        ------
        Uses a closed-form expression for Cc + (L+R || C). If no real solution is
        available (critically damped / non-resonant), falls back to searching for the
        magnitude minimum of Vout near the nominal 1/(2π√(LC)).

        Parameters
        ----------
        L, C, R, Cc : float, optional
            Circuit parameters [H, F, Ω, F]
        presign2 : int, optional
            Branch sign selector for the square root (kept for legacy compatibility)
        verbose : bool, optional
            Emit diagnostic prints

        Returns
        -------
        float
            Estimated resonant frequency [Hz]

        Notes
        -----
        Sets self.nonres_flag when the imaginary root test indicates no true resonance.
        """
        # x = -sqrt(-(C^2 R^2)/(2 (C^2 L^2 + C D L^2)) - (C D R^2)/(2 (C^2 L^2 + C D L^2)) - (i sqrt((i C^2 R^2 + i C D R^2 - 2 i C L - i D L)^2 - 4 i (i C^2 L^2 + i C D L^2)))/(2 (C^2 L^2 + C D L^2)) + (C L)/(C^2 L^2 + C D L^2) + (D L)/(2 (C^2 L^2 + C D L^2)))
        if C is None:
            C = self.C
        if L is None:
            L = self.L
        if R is None:
            R = self.R
        if Cc is None:
            Cc = self.Cc
        
        D = Cc
        presign = 1 # take only positive roots
        
        # check if we are going to end up with an imaginary solution:
        num3_part2 = 4 * (C**2 * L**2 + C * D * L**2)
        num3_part1 = (C**2 * R**2 + C * D * R**2 - 2 * C * L - D * L)**2
        if num3_part2 > num3_part1:
            # there is no real root to the imaginary part of the impedance
            # instead, look for the local minimum and call this the resonant frequency
            # though, arguable whether or not this is still a resonance
            
            self.nonres_flag = True
            if verbose:
                print('Found unreal solution! Looking for a local minimum instead.\nnum3 root arg: %.2e (num3 part 1: %.2e, num3 part 1: %.2e)'%(num3_part1-num3_part2, num3_part1, num3_part2))
            guess_fr = 1./(np.pi*2 * np.sqrt(L*C))
            guess_Q = 1./(2*np.pi*R*C)
            guess_bw = guess_fr/guess_Q
            
            span = guess_bw*10
            frange = np.linspace(guess_fr-span*2, guess_fr+span/2, 1000)
            test_mag = abs(self.compute_Vout(frange))
            guess_fr = frange[test_mag.argmin()]
            span2 = guess_bw
            frange = np.linspace(guess_fr-span, guess_fr+span, 1000)
            test_mag = abs(self.compute_Vout(frange))
            better_guess_fr = frange[test_mag.argmin()]
            return better_guess_fr
            
#             numerator3 = -1 * np.sqrt(abs((C**2 * R**2 + C * D * R**2 - 2 * C * L - D * L)**2 - 4 * (C**2 * L**2 + C * D * L**2)))
        else:
            numerator3 = -1 * np.sqrt((C**2 * R**2 + C * D * R**2 - 2 * C * L - D * L)**2 - 4 * (C**2 * L**2 + C * D * L**2))

        quotient1 = -1 * (C**2 * R**2) / (2 * (C**2 * L**2 + C * D * L**2))
        quotient2 = -1 *(C * D * R**2) / (2 * (C**2 * L**2 + C * D * L**2))
#         numerator3 = -1 * np.sqrt((C**2 * R**2 + C * D * R**2 - 2 * C * L - D * L)**2 - 4 * (C**2 * L**2 + C * D * L**2))
        denom3 = (2 * (C**2 * L**2 + C * D * L**2)) 
        quotient4 = (C * L)/(C**2 * L**2 + C * D * L**2)
        quotient5 = (D * L)/(2 * (C**2 * L**2 + C * D * L**2))

        x = presign * np.sqrt( quotient1 + quotient2 + presign2*(numerator3 / denom3) + quotient4 + quotient5 )

        return x / (2 * np.pi)
    



    def compute_Qc(self, Z0=50):
        """
        External coupling quality factor from coupling capacitor.

        Parameters
        ----------
        Z0 : float, optional
            Feedline characteristic impedance [Ω]

        Returns
        -------
        float
            Qc (dimensionless)
        """
        fr = self.compute_fr()
        Qc = (8 * self.C) / (self.Cc**2 * (2 * np.pi * fr * Z0) )
        return Qc

    def compute_Qi(self):
        """
        Internal quality factor from loss R, using L = Lk + Lg.

        Returns
        -------
        float
            Qi (here named Qr in the codebase; returned for compatibility)
        """
        fr = self.compute_fr()
        L = self.Lk + self.Lg
        Qr = np.pi * fr *2 * L / self.R
        return Qr

    # def compute_Qi(self):

    #     Qr = self.compute_Qr()
    #     Qc = self.compute_Qc()
    #     Qi = 1./(1./Qr - 1./Qc)
    #     return Qi
    def compute_Qr(self):
        """
        Loaded quality factor from internal and coupling Q.

        Returns
        -------
        float
            Qr (dimensionless)
        """
        Qi = self.compute_Qi()
        Qc = self.compute_Qc()
        Qr = 1./(1./Qi + 1./Qc)
        return Qr

    def compute_Q_values(self):
        Qr = self.compute_Qr()
        Qi = self.compute_Qi()
        Qc = self.compute_Qc()
        return Qr, Qi, Qc


    #####
    # extras
    #####
    
    def generate_res_param_string(self):
        res_param_string = 'Lk=%.2e H, Lg=%.2e H, C=%.2e F, Cc=%.2e F, R=%.2e ohm'%(self.Lk, self.Lg, self.C, self.Cc, self.R)
        return res_param_string
