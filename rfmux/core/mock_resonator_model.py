"""
Mock CRS Device - Resonator Physics Model.
Encapsulates the logic for resonator physics simulation, including S21 response.
"""
import numpy as np

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
        
        # System gain factor to scale output amplitudes to realistic levels
        self.system_gain = 100.0  # Adjust this to control overall signal amplitude

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
    def generate_lc_resonances(self, num_resonances=50, f_start=1e9, f_end=2e9):
        """Generate simple LC resonances distributed across spectrum."""
        self.lc_resonances = []
        if num_resonances == 0:
            return

        # Use the parameter num_resonances for min_spacing calculation
        min_spacing = (f_end - f_start) / (num_resonances * 2) if num_resonances > 0 else 0
        frequencies = []
        
        attempts = 0
        # Use the parameter num_resonances for max_attempts
        max_attempts = num_resonances * 100 # Safety break for loop
        while len(frequencies) < num_resonances and attempts < max_attempts:
            f = np.random.uniform(f_start, f_end)
            if all(abs(f - existing) > min_spacing for existing in frequencies):
                frequencies.append(f)
            attempts +=1
        
        if len(frequencies) < num_resonances:
            # Fallback if couldn't generate enough unique frequencies
            # This might happen if f_start, f_end, num_resonances, min_spacing are conflicting
            # For simplicity, just use what was generated, or fill with linear spacing
            print(f"Warning: Could only generate {len(frequencies)} LC resonances out of {num_resonances} requested.")
            if not frequencies and num_resonances > 0 : # if none generated, create some
                 frequencies = np.linspace(f_start, f_end, num_resonances)


        frequencies.sort()
        
        for f0_val in frequencies:
            # High Q for narrow resonances: FWHM = f₀/Q
            # For ~10 kHz FWHM at 1 GHz: Q = 1e9/10e3 = 100,000
            Q_val = np.random.uniform(80000, 120000)
            # Stronger coupling for deeper resonances (10-30 dB)
            # k = 0.68 gives ~10 dB, k = 0.9 gives ~20 dB, k = 0.97 gives ~30 dB
            coupling_val = np.random.uniform(0.68, 0.97)
            self.lc_resonances.append({'f0': f0_val, 'Q': Q_val, 'coupling': coupling_val})

    def s21_lc_response(self, frequency, amplitude=1.0):
        """Calculate S21 response for the LC model, including amplitude dependence."""
        if not self.lc_resonances:
            return 1.0 + 0j # No attenuation if no LC resonators

        s21_total = 1.0 + 0j
        for res in self.lc_resonances:
            f0 = res['f0']
            Q = res['Q']
            k = res['coupling']
            
            # Amplitude-dependent effects (simplified)
            Q_eff = Q / (1 + amplitude**2 * 0.1) # Q degradation
            f0_eff = f0 * (1 + amplitude**2 * 0.001) # Frequency shift
            # k_eff = k * (1 + amplitude * 0.05) # Coupling changes - original had this
            
            delta = (frequency - f0_eff) / f0_eff
            denom = 1 + 2j * Q_eff * delta
            
            s21_res = 1 - k / denom # Using k (original coupling) instead of k_eff
            s21_total *= s21_res
        
        # Add measurement noise (simplified)
        noise_level = 0.001 * (1 + amplitude * 0.1)
        noise = noise_level * (np.random.randn() + 1j * np.random.randn())
        
        return s21_total + noise

    def calculate_channel_response(self, module, channel, frequency, amplitude, phase_degrees):
        """
        Calculate complex S21 for a given channel's settings using the LC model.
        This is the primary method used by MockCRS.get_samples and the UDP streamer.
        """
        # `module` and `channel` are passed for context but not directly used in this simplified model
        # unless more complex per-channel/module effects are added.

        if amplitude == 0:
            return 0 + 0j # No signal if amplitude is zero
        
        # Calculate S21 response using the LC model
        s21_val = self.s21_lc_response(frequency, amplitude)
        
        # Apply commanded phase
        phase_rad = np.deg2rad(phase_degrees)
        s21_with_phase = s21_val * np.exp(1j * phase_rad)
        
        # Scale by amplitude and system gain
        # S21 is a transfer function, output is S21 * input_amplitude * system_gain
        return s21_with_phase * amplitude * self.system_gain
