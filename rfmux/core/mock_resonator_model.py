"""
Mock CRS Device - Resonator Physics Model.
Encapsulates the logic for resonator physics simulation, including S21 response.
"""
import numpy as np
from . import mock_constants as const


class MockResonatorModel:
    """
    Handles resonator physics simulation for MockCRS.
    Includes both the original S21 model and the newer LC resonance model.
    """
    def __init__(self, mock_crs):
        """
        Initialize the resonator model.

        Parameters
        ----------
        mock_crs : MockCRS
            An instance of the MockCRS to access its state (frequencies, amplitudes, etc.)
        """
        self.mock_crs = mock_crs # Store a reference to the main MockCRS instance

        # Parameters for the original S21 model (dynamic resonators)
        self.resonator_frequencies = np.array([]) # Populated by generate_resonators
        self.resonator_Q_factors = np.array([])   # Populated by generate_resonators
        self.num_resonators = 0
        self.ff_factor = 0.01  # Frequency shift scaling factor
        self.p_sky = 0         # External fixed power source
        self.resonator_current_f0s = np.array([])
        self.resonator_pe = np.array([])
        self.resonator_f0_history = []

        # Parameters for the LC resonance model
        self.lc_resonances = [] # List of dicts: {'f0', 'Q', 'coupling'}
        
        # Note: Removed system_gain - S21 should be dimensionless transfer function
        # Final scaling to ADC counts happens in UDP streamer
        
        # Initialize kinetic inductance parameters for each resonator
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

        # Initialize with default LC resonances upon creation
        # self.generate_lc_resonances() # Or let MockCRS call this

    # --- Original S21 Model Methods (Dynamic Resonators) ---
    def initialize_dynamic_resonators(self):
        """Initialize dynamic resonance parameters based on generated resonators."""
        if not self.resonator_frequencies.size or not self.resonator_Q_factors.size:
            self.num_resonators = 0
            self.resonator_current_f0s = np.array([])
            self.resonator_pe = np.array([])
            self.resonator_f0_history = []
            return
        
        self.num_resonators = len(self.resonator_frequencies)
        self.resonator_current_f0s = np.copy(self.resonator_frequencies)
        self.resonator_pe = np.zeros(self.num_resonators)
        self.resonator_f0_history = [[] for _ in range(self.num_resonators)]

    def generate_resonators(self, num_resonators=10, f_start=1e9, f_end=2e9, 
                            nominal_Q=1000, min_spacing=0.3e6):
        """Generate random resonator frequencies and Q factors for the original model."""
        frequencies = []
        Q_factors = []
        config = self.mock_crs.physics_config if hasattr(self.mock_crs, 'physics_config') else {}
        seed = config.get('resonator_random_seed', 42)
        np.random.seed(seed) #### Fixing the seed so that we can recreate the randomness
        
        
        freq_range = f_end - f_start
        if num_resonators * min_spacing > freq_range and num_resonators > 0 : # check num_resonators > 0
             max_resonators = int(freq_range / min_spacing) if min_spacing > 0 else np.inf
             raise ValueError(
                f"Cannot fit {num_resonators} resonators with min_spacing {min_spacing}Hz "
                f"in range {freq_range}Hz. Max possible: {max_resonators}"
            )

        attempts = 0
        max_attempts = num_resonators * 100
        while len(frequencies) < num_resonators and attempts < max_attempts:
            freq = np.random.uniform(f_start, f_end)
            if all(abs(freq - f) >= min_spacing for f in frequencies):
                frequencies.append(freq)
                Q = nominal_Q * np.random.uniform(0.9, 1.1)
                Q_factors.append(Q)
            attempts += 1

        if len(frequencies) < num_resonators:
            raise ValueError("Could not generate the required number of resonators with given constraints.")

        if frequencies: # Ensure not empty before sorting
            sorted_pairs = sorted(zip(frequencies, Q_factors))
            self.resonator_frequencies = np.array([p[0] for p in sorted_pairs])
            self.resonator_Q_factors = np.array([p[1] for p in sorted_pairs])
        else:
            self.resonator_frequencies = np.array([])
            self.resonator_Q_factors = np.array([])
            
        self.initialize_dynamic_resonators()
        self.invalidate_caches()

    def s21_response(self, frequency):
        """Compute the combined S21 response of the original resonator comb."""
        if self.num_resonators == 0:
            return 1.0, 0.0 # No attenuation, zero phase if no resonators

        s21 = 1.0 + 0j
        for idx in range(self.num_resonators):
            f0 = self.resonator_current_f0s[idx]
            Q = self.resonator_Q_factors[idx]
            delta_f = frequency - f0
            s21_resonator = 1 - (1 / (1 + 2j * Q * delta_f / f0))
            s21 *= s21_resonator
        
        return abs(s21), np.angle(s21) # Magnitude and phase in radians

    def update_resonator_frequencies(self, s21_mag_array):
        """Update original model resonance frequencies based on power estimates."""
        if not self.resonator_current_f0s.size:
            return

        pe = (s21_mag_array) ** 2 # Simplified power estimate
        self.resonator_pe = pe
        self.resonator_current_f0s *= (1 - (self.resonator_pe + self.p_sky) * self.ff_factor)

    # --- LC Resonance Model Methods ---
    def generate_lc_resonances(self):
        """Generate LC resonances distributed across spectrum with kinetic inductance parameters."""
        # Get ALL values from stored config or defaults
        config = self.mock_crs.physics_config if hasattr(self.mock_crs, 'physics_config') else {}
        seed = config.get('resonator_random_seed', 42)

        np.random.seed(seed) #### Fixing the seed so that we can recreate the randomness
        
        num_resonances = config.get('num_resonances', const.DEFAULT_NUM_RESONANCES)
        f_start = config.get('freq_start', const.DEFAULT_FREQ_START)
        f_end = config.get('freq_end', const.DEFAULT_FREQ_END)
            
        self.lc_resonances = []
        self.kinetic_inductance_fractions = []
        
        if num_resonances == 0:
            return

        # Use the parameter num_resonances for min_spacing calculation
        min_spacing = (f_end - f_start) / (num_resonances * 2) if num_resonances > 0 else 0
        frequencies = []
        
        attempts = 0
        max_attempts = num_resonances * 100
        while len(frequencies) < num_resonances and attempts < max_attempts:
            f = np.random.uniform(f_start, f_end)
            if all(abs(f - existing) > min_spacing for existing in frequencies):
                frequencies.append(f)
            attempts += 1
        
        if len(frequencies) < num_resonances:
            if const.DEBUG_RESONATOR_GENERATION:
                print(f"Warning: Could only generate {len(frequencies)} LC resonances out of {num_resonances} requested.")
            if not frequencies and num_resonances > 0:
                frequencies = np.linspace(f_start, f_end, num_resonances)

        frequencies.sort()
        
        # Get config values from MockCRS physics_config or fallback to constants
        config = self.mock_crs.physics_config if hasattr(self.mock_crs, 'physics_config') else {}
        q_min = config.get('q_min', const.DEFAULT_Q_MIN)
        q_max = config.get('q_max', const.DEFAULT_Q_MAX)
        q_variation = config.get('q_variation', const.Q_VARIATION)
        coupling_min = config.get('coupling_min', const.DEFAULT_COUPLING_MIN)
        coupling_max = config.get('coupling_max', const.DEFAULT_COUPLING_MAX)
        ki_fraction_default = config.get('kinetic_inductance_fraction', const.DEFAULT_KINETIC_INDUCTANCE_FRACTION)
        ki_variation = config.get('kinetic_inductance_variation', const.KINETIC_INDUCTANCE_VARIATION)
        
        for f0_val in frequencies:
            # Use config values for Q factor range
            Q_val = np.random.uniform(q_min, q_max)
            Q_val *= np.random.uniform(1 - q_variation, 1 + q_variation)
            
            # Use config values for coupling range
            coupling_val = np.random.uniform(coupling_min, coupling_max)
            
            # Generate kinetic inductance fraction for this resonator
            ki_fraction = ki_fraction_default
            ki_fraction *= np.random.uniform(1 - ki_variation, 1 + ki_variation)
            
            self.lc_resonances.append({
                'f0': f0_val, 
                'Q': Q_val, 
                'coupling': coupling_val
            })
            self.kinetic_inductance_fractions.append(ki_fraction)
        
        self.invalidate_caches()

    def s21_lc_response(self, frequency, amplitude=1.0):
        """Calculate S21 response for the LC model with kinetic inductance effects and bifurcation."""
        if not self.lc_resonances:
            return 1.0 + 0j # No attenuation if no LC resonators

        # Get config values from MockCRS physics_config or fallback to constants
        config = self.mock_crs.physics_config if hasattr(self.mock_crs, 'physics_config') else {}
        enable_bifurcation = config.get('enable_bifurcation', const.ENABLE_BIFURCATION)
        base_noise_level = config.get('base_noise_level', const.BASE_NOISE_LEVEL)
        amplitude_noise_coupling = config.get('amplitude_noise_coupling', const.AMPLITUDE_NOISE_COUPLING)
        
        s21_total = 1.0 + 0j
        
        for idx, res in enumerate(self.lc_resonances):
            f0 = res['f0']
            Q = res['Q']
            k = res['coupling']
            
            # Get kinetic inductance fraction for this resonator
            ki_fraction = self.kinetic_inductance_fractions[idx] if idx < len(self.kinetic_inductance_fractions) else const.DEFAULT_KINETIC_INDUCTANCE_FRACTION
            
            if enable_bifurcation:
                # Self-consistent calculation with bifurcation capability
                f0_eff = self._calculate_self_consistent_frequency(frequency, f0, Q, k, amplitude, ki_fraction)
            else:
                # Simple frequency shift without self-consistency
                f0_eff = self._calculate_simple_frequency_shift(f0, amplitude, ki_fraction)
            
            # Q degradation at high power
            Q_eff = Q / (1 + amplitude**2 * 0.1)
            
            # Calculate S21 for this resonator
            delta = (frequency - f0_eff) / f0_eff
            denom = 1 + 2j * Q_eff * delta
            s21_res = 1 - k / denom
            s21_total *= s21_res
        
        # Add measurement noise (dimensionless, relative to S21)
        noise_level = base_noise_level * (1 + amplitude * amplitude_noise_coupling)
        noise = noise_level * (np.random.randn() + 1j * np.random.randn())
        
        return s21_total + noise
    
    def _calculate_simple_frequency_shift(self, f0, amplitude, ki_fraction):
        """Calculate frequency shift without self-consistency (faster)."""
        # Get config values from MockCRS physics_config or fallback to constants
        config = self.mock_crs.physics_config if hasattr(self.mock_crs, 'physics_config') else {}
        power_norm_const = config.get('power_normalization', const.POWER_NORMALIZATION)
        freq_shift_power_law = config.get('frequency_shift_power_law', const.FREQUENCY_SHIFT_POWER_LAW)
        freq_shift_mag = config.get('frequency_shift_magnitude', const.FREQUENCY_SHIFT_MAGNITUDE)
        saturation_power = config.get('saturation_power', const.SATURATION_POWER)
        saturation_sharpness = config.get('saturation_sharpness', const.SATURATION_SHARPNESS)
        
        # Normalized power
        power_norm = (amplitude / power_norm_const) ** freq_shift_power_law
        
        # Apply saturation
        if power_norm > saturation_power:
            saturation_factor = saturation_power + (power_norm - saturation_power) / (1 + (power_norm / saturation_power) ** saturation_sharpness)
            power_norm = saturation_factor
        
        # Frequency shift (downward for KIDs)
        freq_shift_fraction = -freq_shift_mag * ki_fraction * power_norm
        return f0 * (1 + freq_shift_fraction)
    
    def _calculate_self_consistent_frequency(self, frequency, f0, Q, k, amplitude, ki_fraction):
        """Calculate self-consistent frequency with potential bifurcation using damped iteration."""
        # Get config values from MockCRS physics_config or fallback to constants
        config = self.mock_crs.physics_config if hasattr(self.mock_crs, 'physics_config') else {}
        power_norm_const = config.get('power_normalization', const.POWER_NORMALIZATION)
        freq_shift_power_law = config.get('frequency_shift_power_law', const.FREQUENCY_SHIFT_POWER_LAW)
        freq_shift_mag = config.get('frequency_shift_magnitude', const.FREQUENCY_SHIFT_MAGNITUDE)
        saturation_power = config.get('saturation_power', const.SATURATION_POWER)
        saturation_sharpness = config.get('saturation_sharpness', const.SATURATION_SHARPNESS)
        bifurcation_iterations = config.get('bifurcation_iterations', const.BIFURCATION_ITERATIONS)
        bifurcation_tolerance = config.get('bifurcation_convergence_tolerance', const.BIFURCATION_CONVERGENCE_TOLERANCE)
        bifurcation_damping = config.get('bifurcation_damping_factor', const.BIFURCATION_DAMPING_FACTOR)
        
        # Initial guess
        f0_eff = f0
        convergence_history = []
        
        for iteration in range(bifurcation_iterations):
            # Calculate S21 magnitude at current effective frequency
            delta = (frequency - f0_eff) / f0_eff
            denom = 1 + 2j * Q * delta
            s21_res = 1 - k / denom
            circulating_power = abs(s21_res)**2 * amplitude**2
            
            # Calculate new frequency based on circulating power
            power_norm = (circulating_power / power_norm_const) ** freq_shift_power_law
            
            # Apply saturation
            if power_norm > saturation_power:
                saturation_factor = saturation_power + (power_norm - saturation_power) / (1 + (power_norm / saturation_power) ** saturation_sharpness)
                power_norm = saturation_factor
            
            # New effective frequency (downward shift for KIDs)
            freq_shift_fraction = -freq_shift_mag * ki_fraction * power_norm
            f0_new = f0 * (1 + freq_shift_fraction)
            
            # Check convergence
            convergence_error = abs(f0_new - f0_eff) / f0
            convergence_history.append(convergence_error)
            
            if const.DEBUG_BIFURCATION:
                print(f"DEBUG: CONVERGENCE iter {iteration}: error = {convergence_error:.2e}")
            
            if convergence_error < bifurcation_tolerance:
                if const.DEBUG_BIFURCATION:
                    print(f"Bifurcation converged in {iteration + 1} iterations")
                break
            
            # Apply damped iteration to prevent oscillation
            f0_eff = (1 - bifurcation_damping) * f0_eff + bifurcation_damping * f0_new
        
        if const.DEBUG_BIFURCATION and iteration == bifurcation_iterations - 1:
            print(f"Warning: Bifurcation did not converge after {bifurcation_iterations} iterations")
            
        return f0_eff
    
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
    
    def _get_cached_cic_response(self, freq_offset, dec_stage):
        """Get CIC response with caching."""
        # Round to nearest 0.1 Hz
        offset_key = round(freq_offset * 10) / 10
        cache_key = (offset_key, dec_stage)
        
        if cache_key not in self._cic_cache:
            self._cic_cache[cache_key] = self._calculate_cic_response(freq_offset, dec_stage)
        
        return self._cic_cache[cache_key]
    
    def _calculate_cic_response(self, freq_offset, dec_stage):
        """
        Calculate actual CIC filter response (with droop) at a given frequency offset.
        
        This emulates the actual filter behavior that causes amplitude droop
        at higher frequencies, which real data exhibits before correction.
        
        Parameters
        ----------
        freq_offset : float
            Frequency offset from channel center in Hz
        dec_stage : int
            Decimation stage (0-6)
            
        Returns
        -------
        float
            Filter response (0-1) including droop effect
        """
        # Import the CIC correction function from transferfunctions
        from . import transferfunctions as tf
        
        # CIC parameters
        R1 = 64  # First stage decimation
        R2 = 2**dec_stage  # Second stage decimation
        f_in1 = 625e6 / 256  # Input to first CIC
        f_in2 = f_in1 / R1   # Input to second CIC
        
        # Using absolute value since response is symmetric
        freq_abs = abs(freq_offset)
        
        # Avoid division by zero at DC
        if freq_abs < 0.01:
            return 1.0
            
        # Get the correction factors using existing functions
        cic1_correction = tf._general_single_cic_correction(
            np.array([freq_abs]), f_in1, R=R1, N=3
        )[0]
        
        cic2_correction = tf._general_single_cic_correction(
            np.array([freq_abs]), f_in2, R=R2, N=6
        )[0]
        
        # The actual filter response is the inverse of the correction
        # (correction compensates for droop, so 1/correction gives us the droop)
        total_correction = cic1_correction * cic2_correction
        
        # Apply the inverse to get the actual filter response with droop
        if total_correction > 0:
            filter_response = 1.0 / total_correction
        else:
            filter_response = 0.0
        
        # Only clamp to prevent negative values, but don't limit the upper bound
        # CIC droop can make the response much less than 1 at high frequencies
        return max(filter_response, 0.0)
    
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
        dec_stage = self.mock_crs.fir_stage  # Note: still called fir_stage in MockCRS for compatibility
        bandwidth = self.cic_bandwidths.get(dec_stage, 298)  # Hz
        
        # Determine sample rate if not provided
        if sample_rate is None:
            sample_rate = 625e6 / 256 / 64 / (2**dec_stage)
        
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
            
            # Apply proper CIC filter response for each frequency difference
            # This emulates the actual hardware filter behavior with droop
            cic_responses = np.zeros_like(freq_diffs)
            for i in range(len(obs_channels)):
                for j in range(len(tone_freqs)):
                    if within_bandwidth[i, j]:
                        # Use the proper CIC response calculation
                        cic_responses[i, j] = self._get_cached_cic_response(freq_diffs[i, j], dec_stage)
            
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
                        
                        # Use proper CIC filter response
                        cic_response = self._get_cached_cic_response(freq_diff, dec_stage)
                        
                        if abs(freq_diff) < 0.1:  # DC
                            signal += tone_amp * cic_response
                        else:
                            # Time-varying beat with proper CIC filtering
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
