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

        freq_range = f_end - f_start
        if num_resonators * min_spacing > freq_range and num_resonators > 0 : # check num_resonators > 0
             max_resonators = int(freq_range / min_spacing) if min_spacing > 0 else float('inf')
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
    def generate_lc_resonances(self, num_resonances=None, f_start=None, f_end=None):
        """Generate LC resonances distributed across spectrum with kinetic inductance parameters."""
        # Use constants if parameters not provided
        if num_resonances is None:
            num_resonances = const.DEFAULT_NUM_RESONANCES
        if f_start is None:
            f_start = const.DEFAULT_FREQ_START
        if f_end is None:
            f_end = const.DEFAULT_FREQ_END
            
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
        
        for f0_val in frequencies:
            # Use constants for Q factor range
            Q_val = np.random.uniform(const.DEFAULT_Q_MIN, const.DEFAULT_Q_MAX)
            Q_val *= np.random.uniform(1 - const.Q_VARIATION, 1 + const.Q_VARIATION)
            
            # Use constants for coupling range
            coupling_val = np.random.uniform(const.DEFAULT_COUPLING_MIN, const.DEFAULT_COUPLING_MAX)
            
            # Generate kinetic inductance fraction for this resonator
            ki_fraction = const.DEFAULT_KINETIC_INDUCTANCE_FRACTION
            ki_fraction *= np.random.uniform(1 - const.KINETIC_INDUCTANCE_VARIATION, 
                                           1 + const.KINETIC_INDUCTANCE_VARIATION)
            
            self.lc_resonances.append({
                'f0': f0_val, 
                'Q': Q_val, 
                'coupling': coupling_val
            })
            self.kinetic_inductance_fractions.append(ki_fraction)

    def s21_lc_response(self, frequency, amplitude=1.0):
        """Calculate S21 response for the LC model with kinetic inductance effects and bifurcation."""
        if not self.lc_resonances:
            return 1.0 + 0j # No attenuation if no LC resonators

        s21_total = 1.0 + 0j
        
        for idx, res in enumerate(self.lc_resonances):
            f0 = res['f0']
            Q = res['Q']
            k = res['coupling']
            
            # Get kinetic inductance fraction for this resonator
            ki_fraction = self.kinetic_inductance_fractions[idx] if idx < len(self.kinetic_inductance_fractions) else const.DEFAULT_KINETIC_INDUCTANCE_FRACTION
            
            if const.ENABLE_BIFURCATION:
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
        noise_level = const.BASE_NOISE_LEVEL * (1 + amplitude * const.AMPLITUDE_NOISE_COUPLING)
        noise = noise_level * (np.random.randn() + 1j * np.random.randn())
        
        return s21_total + noise
    
    def _calculate_simple_frequency_shift(self, f0, amplitude, ki_fraction):
        """Calculate frequency shift without self-consistency (faster)."""
        # Normalized power
        power_norm = (amplitude / const.POWER_NORMALIZATION) ** const.FREQUENCY_SHIFT_POWER_LAW
        
        # Apply saturation
        if power_norm > const.SATURATION_POWER:
            saturation_factor = const.SATURATION_POWER + (power_norm - const.SATURATION_POWER) / (1 + (power_norm / const.SATURATION_POWER) ** const.SATURATION_SHARPNESS)
            power_norm = saturation_factor
        
        # Frequency shift (downward for KIDs)
        freq_shift_fraction = -const.FREQUENCY_SHIFT_MAGNITUDE * ki_fraction * power_norm
        return f0 * (1 + freq_shift_fraction)
    
    def _calculate_self_consistent_frequency(self, frequency, f0, Q, k, amplitude, ki_fraction):
        """Calculate self-consistent frequency with potential bifurcation."""
        # Initial guess
        f0_eff = f0
        
        for iteration in range(const.BIFURCATION_ITERATIONS):
            # Calculate S21 magnitude at current effective frequency
            delta = (frequency - f0_eff) / f0_eff
            denom = 1 + 2j * Q * delta
            s21_res = 1 - k / denom
            circulating_power = abs(s21_res)**2 * amplitude**2
            
            # Calculate new frequency based on circulating power
            power_norm = (circulating_power / const.POWER_NORMALIZATION) ** const.FREQUENCY_SHIFT_POWER_LAW
            
            # Apply saturation
            if power_norm > const.SATURATION_POWER:
                saturation_factor = const.SATURATION_POWER + (power_norm - const.SATURATION_POWER) / (1 + (power_norm / const.SATURATION_POWER) ** const.SATURATION_SHARPNESS)
                power_norm = saturation_factor
            
            # New effective frequency (downward shift for KIDs)
            freq_shift_fraction = -const.FREQUENCY_SHIFT_MAGNITUDE * ki_fraction * power_norm
            f0_new = f0 * (1 + freq_shift_fraction)
            
            # Check convergence
            print("DEBUG: CONVERGENCE", abs(f0_new - f0_eff) / f0)
            if abs(f0_new - f0_eff) / f0 < const.BIFURCATION_CONVERGENCE_TOLERANCE:
                if const.DEBUG_BIFURCATION:
                    print(f"Bifurcation converged in {iteration + 1} iterations")
                break
                
            f0_eff = f0_new
        print(iteration)
        return f0_eff

    def calculate_channel_response(self, module, channel, frequency, amplitude, phase_degrees):
        """
        Calculate dimensionless S21 transfer function for a given channel's settings.
        
        This returns the complex S21 transfer function that should be applied to
        the commanded amplitude. The final scaling to ADC counts happens in the
        UDP streamer.
        
        Parameters
        ----------
        module, channel : int
            Channel identification (for context, not used in simplified model)
        frequency : float
            Probe frequency in Hz
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
        
        # Calculate S21 response using the LC model (dimensionless)
        s21_val = self.s21_lc_response(frequency, amplitude)
        
        # Apply commanded phase
        phase_rad = np.deg2rad(phase_degrees)
        s21_with_phase = s21_val * np.exp(1j * phase_rad)
        
        # Return S21 * commanded_amplitude (preserves amplitude scaling)
        # The UDP streamer will handle final scaling to ADC counts
        return s21_with_phase * amplitude
