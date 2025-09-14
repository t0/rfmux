"""
Mock CRS Device - Resonator Physics Model.
Encapsulates the logic for resonator physics simulation, including S21 response.
"""
import numpy as np
import copy
from . import mock_constants as const

import sys
sys.path.append('/home/maclean/code/')
sys.path.append('/home/maclean/code/mr_resonator/')
import mr_resonator
from mr_resonator.mr_complex_resonator import MR_complex_resonator as MR_complex_resonator
from mr_resonator.mr_lekid import MR_LEKID as MR_LEKID


class MockResonatorModel:
    """
    Handles resonator physics simulation for MockCRS using mr_resonator objects.
    This class now exclusively uses persistent MR_LEKID objects to avoid memory leaks
    and provide resonator physics simulation.
    """
    def __init__(self, mock_crs):
        """
        Initialize the resonator model.

        Parameters
        ----------
        mock_crs : MockCRS
            An instance of the MockCRS to access its state (frequencies, amplitudes, etc.)
        """
        self.mock_crs = mock_crs  # Store a reference to the main MockCRS instance

        # Store persistent mr_resonator objects to avoid memory leaks
        self.mr_lekids = []  # List of persistent MR_LEKID objects
        self.mr_complex_resonators = []  # List of persistent MR_complex_resonator objects
        
        # Store base parameters for each LEKID (as fabricated)
        self.base_lekid_params = []  # Original R, Lk values at T=0, nqp=0
        
        # Layer 1: Quasiparticle effects (affects base Lk and R)
        self.lk_qp_factors = []      # Lk_qp = Lk_base * lk_qp_factor
        self.r_qp_factors = []       # R_qp = R_base * r_qp_factor
        
        # Layer 2: Current effects (affects Lk only, on top of QP effects)
        self.lk_current_factors = []  # Lk_total = Lk_qp * lk_current_factor
        
        # Physical constants
        self.Istar = 1e-6  # Characteristic current [A]
        
        # Resonator metadata (parallel arrays to mr_lekids)
        self.resonator_frequencies = []  # Resonance frequencies for each LEKID
        self.kinetic_inductance_fractions = []  # Per-resonator kinetic inductance fraction
        
        # CIC bandwidth cache for different decimation stages
        self.cic_bandwidths = {
            0: 625e6 / 256 / 64 / 2,      # 19.073 kHz
            1: 625e6 / 256 / 64 / 4,      # 9.537 kHz  
            2: 625e6 / 256 / 64 / 8,      # 4.768 kHz
            3: 625e6 / 256 / 64 / 16,     # 2.384 kHz
            4: 625e6 / 256 / 64 / 32,     # 1.192 kHz
            5: 625e6 / 256 / 64 / 64,     # 596 Hz
            6: 625e6 / 256 / 64 / 128,    # 298 Hz
        }
        
        # Performance caches
        self._s21_cache = {}  # Cache S21 responses
        self._cic_cache = {}  # Cache CIC filter responses
        self._cache_valid = False

    # --- MR_Resonator Methods ---
    def generate_resonators(self, num_resonances=2, config=None):
        '''
        Generate mr_resonator objects with circuit parameters.
        
        Parameters
        ----------
        num_resonances : int
            Number of resonators to generate
        config : dict, optional
            Configuration dictionary with circuit parameters and variations
        '''
        # Get configuration
        if config is None:
            # Use physics_config from MockCRS if available, otherwise use defaults
            if hasattr(self.mock_crs, 'physics_config') and self.mock_crs.physics_config:
                config = self.mock_crs.physics_config
            else:
                from . import mock_crs_helper
                config = mock_crs_helper.DEFAULT_MOCK_CONFIG.copy()
        
        print('Using config:', {k: v for k, v in config.items() if k in ['num_resonances', 'freq_start', 'freq_end', 'T', 'Popt']})
        
        # Extract parameters
        freq_start = config.get('freq_start', 1e9)
        freq_end = config.get('freq_end', 1.5e9)
        
        # Physics parameters (determine Lk and R via quasiparticle density)
        T = config.get('T', 0.12)
        Popt = config.get('Popt', 1e-18)
        
        # Base circuit parameters
        Lg_base = config.get('Lg', 10e-9)
        Cc_base = config.get('Cc', 5e-15)
        L_junk = config.get('L_junk', 0)
        
        # Variations
        C_variation = config.get('C_variation', 0.01)
        Cc_variation = config.get('Cc_variation', 0.1)
        
        # Readout parameters
        Vin = config.get('Vin', 1e-5)
        input_atten_dB = config.get('input_atten_dB', 10)
        system_termination = config.get('system_termination', 50)
        ZLNA = config.get('ZLNA', 50)  # Now a real number
        GLNA = config.get('GLNA', 1.0)

        # temporarily add a value for Istar that is very large so the current dependent effects will be small
        self.Istar = 1e-2
        
        # Clear existing objects to prevent memory leaks
        self.mr_lekids = []
        self.mr_complex_resonators = []
        self.resonator_frequencies = []
        self.kinetic_inductance_fractions = []
        
        # Clear layered parameter tracking
        self.base_lekid_params = []
        self.lk_qp_factors = []
        self.r_qp_factors = []
        self.lk_current_factors = []
        self.resonator_currents = []

        # Step 1: Create a reference MR_complex_resonator to compute Lk and R from T and Popt
        print(f"Computing Lk and R from T={T} K and Popt={Popt} W")
        
        # Create reference resonator with dummy C value to get physics-computed Lk and R
        reference_params = {
            'T': T,
            'Popt': Popt,
            'C': 1e-12,  # Dummy value, just to create the object
            'Cc': Cc_base,
            'fix_Lg': Lg_base,  # Fix Lg to our desired value
            'Vin': Vin,
            'input_atten_dB': input_atten_dB,
            'base_readout_f': (freq_start + freq_end) / 2,  # Middle frequency as initial guess
            'verbose': False,
            'ZLNA': complex(ZLNA, 0),
            'GLNA': GLNA
        }
        
        # Create reference resonator to get physics-computed Lk and R
        ref_resonator = MR_complex_resonator(**reference_params)
        computed_Lk = ref_resonator.lekid.Lk
        computed_R = ref_resonator.lekid.R
        
        print(f"Physics-computed values: Lk={computed_Lk*1e9:.2f} nH, R={computed_R*1e6:.2f} µΩ")
        
        # Step 2: Generate resonators with computed Lk and R, solving for C to get target frequencies
        print(f"Generating {num_resonances} resonators from {freq_start/1e9:.2f} GHz to {freq_end/1e9:.2f} GHz")
        
        for x in range(num_resonances):
            try:
                # Calculate target frequency for this resonator
                if num_resonances == 1:
                    target_freq = (freq_start + freq_end) / 2  # Middle frequency
                else:
                    target_freq = freq_start + (freq_end - freq_start) * x / (num_resonances - 1)
                
                # Calculate C required for target frequency using computed Lk
                # fr = 1/(2π√(LC)) where L = Lk + Lg
                L_total = computed_Lk + Lg_base
                C_required = 1 / (4 * np.pi**2 * target_freq**2 * L_total)
                
                # Apply variations using normal distribution
                C_actual = C_required * (1 + np.random.normal(0, C_variation))
                Cc_actual = Cc_base * (1 + np.random.normal(0, Cc_variation))
                
                # Ensure positive values
                C_actual = max(C_actual, C_required * 0.1)  # At least 10% of required value
                Cc_actual = max(Cc_actual, Cc_base * 0.1)
                
                print(f"Resonator {x}: target_freq={target_freq/1e9:.3f} GHz, C={C_actual*1e12:.2f} pF")
                
                # Create MR_complex_resonator with T, Popt, and calculated C
                complex_res_params = {
                    'T': T,
                    'Popt': Popt,
                    'C': C_actual,
                    'Cc': Cc_actual,
                    'fix_Lg': Lg_base,  # Fix Lg to our desired value
                    'L_junk': L_junk,
                    'Vin': Vin,
                    'input_atten_dB': input_atten_dB,
                    'base_readout_f': target_freq,  # Initial guess for readout frequency
                    'verbose': False,  # Suppress individual resonator output
                    'ZLNA': complex(ZLNA, 0),  # Convert real ZLNA to complex for MR_complex_resonator
                    'GLNA': GLNA
                }
                
                # Create the full MR_complex_resonator (includes physics modeling)
                complex_res = MR_complex_resonator(**complex_res_params)
                
                # Extract the LEKID from the complex resonator
                lekid = complex_res.lekid
                
                # Calculate and store actual resonance frequency
                actual_freq = lekid.compute_fr()
                print(f"  Actual frequency: {actual_freq/1e9:.3f} GHz")
                print(f"  Actual Lk: {lekid.Lk*1e9:.2f} nH, R: {lekid.R*1e6:.2f} µΩ")
                
                # Store the persistent objects (this fixes the memory leak)
                self.mr_lekids.append(lekid)
                self.mr_complex_resonators.append(complex_res)  # Keep for future QP tracking
                
                # Store base parameters (as-fabricated values)
                self.base_lekid_params.append({
                    'R': lekid.R,
                    'Lk': lekid.Lk,
                    'Lg': lekid.Lg,
                    'C': lekid.C,
                    'Cc': lekid.Cc
                })
                
                # Initialize all modification factors to 1 (no modification)
                self.lk_qp_factors.append(1.0)
                self.r_qp_factors.append(1.0)
                self.lk_current_factors.append(1.0)
                self.resonator_currents.append(0.)
                
                # Store metadata (use actual computed frequency)
                self.resonator_frequencies.append(actual_freq)
                self.kinetic_inductance_fractions.append(lekid.alpha_k)
                
            except Exception as e:
                print(f"Warning: Failed to create resonator {x}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with other resonators
                continue
        
        print(f"Successfully created {len(self.mr_lekids)} persistent LEKID objects")
        print(f"Successfully created {len(self.mr_complex_resonators)} persistent MR_complex_resonator objects")
        self.invalidate_caches()


    def s21_lc_response(self, frequency, amplitude=1.0):
        """
        Calculate S21 response with all physics effects.
        
        Includes:
        - Quasiparticle effects (if set)
        - Current-dependent kinetic inductance
        """
        if not self.mr_lekids:
            return 1.0 + 0j
        
        # Update parameters for current amplitude
        self.update_lekids_for_current(frequency, amplitude)
        # print('updating s21"')
        
        # Calculate S21 with updated parameters
        s21_total = 1.0 + 0j
        for lekid in self.mr_lekids:
            try:
                s21_res = lekid.compute_Vout(frequency, Vin=amplitude) / amplitude
                s21_total *= s21_res
            except Exception as e:
                print(f"Warning: LEKID calculation failed: {e}")
                s21_total *= 0.9 + 0.1j
        
        return s21_total

    def update_lekid_parameters(self, lekid_index):
        """
        Apply all layers of modifications to a LEKID's parameters.
        
        Order of operations:
        1. Start with base (as-fabricated) parameters
        2. Apply quasiparticle modifications
        3. Apply current-dependent modifications
        """
        base_params = self.base_lekid_params[lekid_index]
        lekid = self.mr_lekids[lekid_index]
        
        # Layer 1: Apply quasiparticle effects
        Lk_qp = base_params['Lk'] * self.lk_qp_factors[lekid_index]
        R_qp = base_params['R'] * self.r_qp_factors[lekid_index]
        
        # Layer 2: Apply current effects (on top of QP-modified Lk)
        Lk_total = Lk_qp * self.lk_current_factors[lekid_index]
        
        # Update the LEKID object
        lekid.Lk = Lk_total
        lekid.R = R_qp
        lekid.L = lekid.Lk + lekid.Lg
        lekid.alpha_k = lekid.Lk / lekid.L

    def set_quasiparticle_density(self, lekid_index, nqp):
        """
        Set quasiparticle density for a specific resonator.
        
        This affects the BASE kinetic inductance and resistance,
        before any current-dependent effects are applied.
        
        Parameters
        ----------
        lekid_index : int
            Index of the resonator
        nqp : float
            Quasiparticle density (normalized or absolute, TBD)
        """
        # Physics model for QP effects (placeholder - to be refined)
        # These are multiplicative factors on the base values
        
        # Kinetic inductance increases with QP density
        # (fewer Cooper pairs → higher kinetic inductance)
        self.lk_qp_factors[lekid_index] = 1 + nqp * 0.1  # Example scaling
        
        # Resistance increases with QP density
        # (more normal electrons → higher loss)
        self.r_qp_factors[lekid_index] = 1 + nqp * 0.5  # Example scaling
        
        # Apply the updated parameters
        self.update_lekid_parameters(lekid_index)
        self.invalidate_caches()

    def calculate_resonator_currents(self, frequency, Vin, damp=0.1):
        """
        Calculate the current through each resonator using current divider.
        
        This accounts for the fact that resonators are in parallel,
        so we need to consider all their impedances together.
        """
        currents = []
        
        # First, calculate all resonator impedances at this frequency
        impedances = []
        for lekid in self.mr_lekids:
            Z_res = lekid.total_impedance(frequency)
            impedances.append(Z_res)
        
        # Calculate total parallel impedance of all OTHER resonators
        for i, lekid in enumerate(self.mr_lekids):
            # Calculate Z_other: parallel combination of all other resonators
            if len(impedances) > 1:
                Z_other_inv = sum(1/Z for j, Z in enumerate(impedances) if j != i)
                Z_other = 1 / Z_other_inv if Z_other_inv != 0 else None
            else:
                Z_other = None
            
            # Use the LEKID's built-in method to calculate current
            # apply a damping factor to stabilize the iterative solution
            # and because the current cannot physically change instantaneously
            next_I_res = lekid.calc_Ires(
                fc=frequency,
                Zres=impedances[i],
                Vin=Vin,
                Z_other=Z_other
            )
            I_res = self.resonator_currents[i] + damp*(next_I_res - self.resonator_currents[i])
            currents.append(I_res)
            self.resonator_currents[i] = I_res
        
        return currents

    def calculate_current_factors(self, frequency, amplitude):
        """
        Calculate current-dependent Lk factors for all resonators.
        
        Returns the factors (not applied yet) for convergence checking.
        """
        # Calculate currents through each resonator
        currents = self.calculate_resonator_currents(frequency, Vin=amplitude)
        
        # Calculate Lk modification factors
        new_factors = []
        for I_res in currents:
            # Lk_current(I_res) = Lk_qp * (1 + I_res^2 / Istar^2)
            # So the factor is: (1 + I_res^2 / Istar^2)
            factor = 1 + (abs(I_res)**2 / self.Istar**2)
            new_factors.append(factor)
        
        return new_factors

    def update_lekids_for_current(self, frequency, amplitude):
        """
        Update all LEKID parameters based on resonator currents.
        
        This is an iterative process to find self-consistent solution.
        """
        max_iterations = 25
        tolerance = 1e-6
        
        for iteration in range(max_iterations):
            # Calculate new current factors
            new_factors = self.calculate_current_factors(frequency, amplitude)
            
            # Check convergence
            if iteration > 0:
                factor_change = max(abs(new - old) for new, old in 
                                  zip(new_factors, self.lk_current_factors))
                if factor_change < tolerance:
                    # print('current factor change within tolerance!')
                    break
            
            # Apply the new factors
            self.lk_current_factors = new_factors
            
            # Update all LEKID parameters
            for i in range(len(self.mr_lekids)):
                self.update_lekid_parameters(i)

    def set_istar(self, istar):
        """Set the characteristic current for all resonators."""
        self.Istar = istar
        self.invalidate_caches()

    def get_parameter_summary(self, lekid_index):
        """
        Get a summary of all parameter modifications for debugging.
        """
        base = self.base_lekid_params[lekid_index]
        current = self.mr_lekids[lekid_index]
        
        return {
            'base_Lk': base['Lk'],
            'base_R': base['R'],
            'qp_factor_Lk': self.lk_qp_factors[lekid_index],
            'qp_factor_R': self.r_qp_factors[lekid_index],
            'current_factor_Lk': self.lk_current_factors[lekid_index],
            'final_Lk': current.Lk,
            'final_R': current.R,
            'Lk_qp': base['Lk'] * self.lk_qp_factors[lekid_index],
            'R_qp': base['R'] * self.r_qp_factors[lekid_index]
        }
    
    def invalidate_caches(self):
        """Clear caches when resonator parameters change."""
        self._s21_cache.clear()
        self._cic_cache.clear()
        self._cache_valid = False
    
    def _get_cached_s21(self, frequency, amplitude):
        """Get S21 response with caching."""
        # Round frequency to nearest Hz for cache key
        freq_key = round(frequency)
        amp_key = round(amplitude * 1000) / 1000  # 3 decimal places
        
        cache_key = (freq_key, amp_key)
        if cache_key not in self._s21_cache:
            self._s21_cache[cache_key] = self.s21_lc_response(frequency, amplitude)
        
        return self._s21_cache[cache_key]
    
    def _get_cached_cic_response(self, freq_offset, fir_stage):
        """Get CIC response with caching."""
        # Round to nearest 0.1 Hz
        offset_key = round(freq_offset * 10) / 10
        cache_key = (offset_key, fir_stage)
        
        if cache_key not in self._cic_cache:
            self._cic_cache[cache_key] = self._calculate_cic_response(freq_offset, fir_stage)
        
        return self._cic_cache[cache_key]
    
    def _calculate_cic_response(self, freq_offset, fir_stage):
        """
        Calculate CIC filter response at a given frequency offset.
        
        Parameters
        ----------
        freq_offset : float
            Frequency offset from channel center in Hz
        fir_stage : int
            FIR decimation stage (0-6)
            
        Returns
        -------
        float
            Filter response (0-1)
        """
        # Import the CIC model from transferfunctions
        from . import transferfunctions as tf
        
        # CIC parameters
        R1 = 64  # First stage decimation
        R2 = 2**fir_stage  # Second stage decimation
        f_in1 = 625e6 / 256  # Input to first CIC
        f_in2 = f_in1 / R1   # Input to second CIC
        
        # Calculate response for both stages
        # Using absolute value since response is symmetric
        freq_abs = abs(freq_offset)
        
        # Avoid division by zero at DC
        if freq_abs < 0.01:
            return 1.0
            
        # First CIC (3 stages)
        cic1_response = tf._general_single_cic_correction(
            np.array([freq_abs]), f_in1, R=R1, N=3
        )[0]
        
        # Second CIC (6 stages)  
        cic2_response = tf._general_single_cic_correction(
            np.array([freq_abs]), f_in2, R=R2, N=6
        )[0]
        
        # Combined response
        total_response = cic1_response * cic2_response
        
        # Clamp to [0, 1] range
        return np.clip(total_response, 0, 1)
    
    def calculate_module_response_coupled(self, module, num_samples=1, sample_rate=None, start_time=0):
        """
        Calculate coupled response for all channels in a module using vectorized operations.
        
        This implements realistic channel coupling where all channels
        contribute to a composite signal that each channel then observes
        through its own demodulation.
        
        Parameters
        ----------
        module : int
            Module number (1-4)
        num_samples : int
            Number of time samples to generate (for beat frequency simulation)
        sample_rate : float, optional
            Sample rate in Hz. If None, uses decimation-based rate.
        start_time : float
            Starting time in seconds for time-varying signals (e.g., for UDP packets)
            
        Returns
        -------
        dict
            Channel responses keyed by channel number (1-based)
            Each value is either:
            - A complex number (if num_samples=1)
            - A complex array of length num_samples (if num_samples>1)
        """
        # Get NCO frequency for this module
        nco_freq = self.mock_crs.nco_frequencies.get(module, 0)
        
        # Get decimation stage for bandwidth calculation
        fir_stage = self.mock_crs.fir_stage
        bandwidth = self.cic_bandwidths.get(fir_stage, 298)  # Hz
        
        # Determine sample rate if not provided
        if sample_rate is None:
            sample_rate = 625e6 / 256 / 64 / (2**fir_stage)
        
        # Step 1: Collect active tones and observing channels
        active_tone_freqs = []
        active_tone_amps = []
        obs_channels = []
        obs_freqs = []
        
        # Find all configured channels in this module
        configured_channels = set()
        for (mod, ch) in self.mock_crs.frequencies.keys():
            if mod == module:
                configured_channels.add(ch)
        for (mod, ch) in self.mock_crs.amplitudes.keys():
            if mod == module:
                configured_channels.add(ch)
        
        # Collect active tones (transmitting channels)
        for ch in configured_channels:
            freq = self.mock_crs.frequencies.get((module, ch))
            amp = self.mock_crs.amplitudes.get((module, ch))
            
            if freq is not None and amp is not None and amp != 0:
                phase_deg = self.mock_crs.phases.get((module, ch), 0)
                total_freq = freq + nco_freq
                
                # Apply S21 response
                s21_complex = self._get_cached_s21(total_freq, amp)
                
                # Combine amplitude, S21, and phase
                complex_amplitude = amp * s21_complex * np.exp(1j * np.deg2rad(phase_deg))
                
                active_tone_freqs.append(total_freq)
                active_tone_amps.append(complex_amplitude)
        
        # Collect observing channels
        for ch in configured_channels:
            freq = self.mock_crs.frequencies.get((module, ch))
            if freq is not None:
                obs_channels.append(ch)
                obs_freqs.append(freq + nco_freq)
        
        # If no active tones or observers, return empty
        if not active_tone_freqs or not obs_channels:
            return {}
        
        # Convert to numpy arrays for vectorization
        tone_freqs = np.array(active_tone_freqs)  # Shape: (n_tones,)
        tone_amps = np.array(active_tone_amps)     # Shape: (n_tones,)
        obs_freqs_arr = np.array(obs_freqs)        # Shape: (n_obs,)
        
        # Step 2: Vectorized calculation of all frequency differences
        # Broadcasting: (n_obs, 1) - (1, n_tones) = (n_obs, n_tones)
        freq_diffs = tone_freqs[np.newaxis, :] - obs_freqs_arr[:, np.newaxis]
        
        # Step 3: Determine which tones are within bandwidth for each observer
        within_bandwidth = np.abs(freq_diffs) <= bandwidth
        
        # Step 4: Calculate responses for single time point (most common case)
        if num_samples == 1:
            t = start_time
            
            # Vectorized beat frequency calculation
            # Phase for each tone-observer pair at time t
            phases = 2 * np.pi * freq_diffs * t
            
            # Complex exponentials for all pairs
            beat_factors = np.exp(1j * phases)
            
            # Apply CIC response (simplified - just use distance attenuation for now)
            # For better accuracy, could vectorize the CIC calculation
            cic_responses = np.where(
                np.abs(freq_diffs) < 0.1,  # DC components
                1.0,
                np.sinc(freq_diffs / bandwidth) ** 2  # Simplified CIC approximation
            )
            
            # Calculate contributions: (n_obs, n_tones)
            contributions = tone_amps[np.newaxis, :] * beat_factors * cic_responses
            
            # Mask out tones outside bandwidth
            contributions = np.where(within_bandwidth, contributions, 0)
            
            # Sum contributions for each observer
            signals = np.sum(contributions, axis=1)  # Shape: (n_obs,)
            
            # Build response dictionary
            responses = {}
            for i, ch in enumerate(obs_channels):
                responses[ch] = complex(signals[i])
        
        else:
            # Multiple time samples - still vectorized but over time
            t = start_time + np.arange(num_samples) / sample_rate
            responses = {}
            
            for i, ch in enumerate(obs_channels):
                signal = np.zeros(num_samples, dtype=complex)
                
                # Process tones for this observer
                for j, tone_amp in enumerate(tone_amps):
                    if within_bandwidth[i, j]:
                        freq_diff = freq_diffs[i, j]
                        
                        if abs(freq_diff) < 0.1:  # DC
                            signal += tone_amp
                        else:
                            # Simplified CIC response
                            cic_response = np.sinc(freq_diff / bandwidth) ** 2
                            # Time-varying beat
                            signal += tone_amp * cic_response * np.exp(1j * 2 * np.pi * freq_diff * t)
                
                responses[ch] = signal
        
        return responses

    def calculate_channel_response(self, module, channel, frequency, amplitude, phase_degrees):
        """
        Calculate response for a single channel.
        
        This method now uses the coupled module-wide calculation for accuracy
        when multiple channels are active, but falls back to single-channel
        calculation for efficiency when only one channel is active.
        
        Parameters
        ----------
        module, channel : int
            Channel identification
        frequency : float
            Probe frequency in Hz (total, including NCO)
        amplitude : float
            Commanded amplitude (0-1, where 1.0 = full scale)
        phase_degrees : float
            Commanded phase in degrees
            
        Returns
        -------
        complex
            Dimensionless S21 transfer function * commanded_amplitude
            This preserves the amplitude scaling for the UDP streamer
        """
        if amplitude == 0:
            return 0 + 0j # No signal if amplitude is zero
        
        # Check if there are other active channels in this module
        other_active_channels = False
        for (mod, ch) in self.mock_crs.frequencies.keys():
            if mod == module and ch != channel:
                amp = self.mock_crs.amplitudes.get((mod, ch), 0)
                if amp != 0:
                    other_active_channels = True
                    break
        
        if other_active_channels:
            # Use coupled calculation when multiple channels are active
            # Store this channel's settings temporarily
            nco_freq = self.mock_crs.nco_frequencies.get(module, 0)
            self.mock_crs.frequencies[(module, channel)] = frequency - nco_freq
            self.mock_crs.amplitudes[(module, channel)] = amplitude
            self.mock_crs.phases[(module, channel)] = phase_degrees
            
            # Calculate module-wide response
            module_responses = self.calculate_module_response_coupled(module)
            
            # Extract this channel's response
            if channel in module_responses:
                return module_responses[channel]
        
        # Single channel case - use original efficient calculation
        s21_val = self.s21_lc_response(frequency, amplitude)
        phase_rad = np.deg2rad(phase_degrees)
        s21_with_phase = s21_val * np.exp(1j * phase_rad)
        
        # Return S21 * commanded_amplitude (preserves amplitude scaling)
        return s21_with_phase * amplitude
