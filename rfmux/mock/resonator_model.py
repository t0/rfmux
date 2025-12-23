"""
Mock CRS Device - Resonator Physics Model.
Encapsulates the logic for resonator physics simulation, including S21 response.
"""
import numpy as np
import threading

# Import from mr_resonator subpackage using relative imports
from ..mr_resonator.mr_complex_resonator import MR_complex_resonator

# Import JIT-compiled physics functions (numba is required)
from ..mr_resonator import jit_physics


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
        
        # Import default configuration from Single Source of Truth
        from .config import defaults as get_defaults
        default_config = get_defaults()

        # Store persistent mr_resonator objects to avoid memory leaks
        self.mr_lekids = []  # List of persistent MR_LEKID objects
        self.mr_complex_resonators = []  # List of persistent MR_complex_resonator objects
        
        # Store base parameters for each LEKID (as fabricated)
        self.base_lekid_params = []  # Original R, Lk values at T=0, nqp=0
        
        # Store base nqp values computed from physics
        self.base_nqp_values = []    # Base quasiparticle density for each resonator
        
        # Noise configuration from SoT
        self.nqp_noise_enabled = default_config['nqp_noise_enabled']
        self.nqp_noise_std_factor = default_config['nqp_noise_std_factor']
        
        # Current effects (affects Lk only, applied after physics-based base params)
        self.lk_current_factors = []  # Lk_total = Lk_base * lk_current_factor
        
        # Physical constants
        self.Istar = 5e-3  # Characteristic current [A] - This is a physical constant, not a config param
        
        # Resonator metadata (parallel arrays to mr_lekids)
        self.resonator_frequencies = []  # Resonance frequencies for each LEKID
        self.kinetic_inductance_fractions = []  # Per-resonator kinetic inductance fraction
        self.resonator_q_values = []  # Pre-calculated Q values for each resonator
        self.resonator_linewidths = []  # Pre-computed linewidths (f0/Q) for filtering
        
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
        
        # Pulse event tracking for time-dependent QP density - from SoT
        self.pulse_events = []  # List of active pulses
        self.pulse_config = {
            'mode': default_config['pulse_mode'],
            'period': default_config['pulse_period'],
            'probability': default_config['pulse_probability'],
            'tau_rise': default_config['pulse_tau_rise'],
            'tau_decay': default_config['pulse_tau_decay'],
            'amplitude': default_config['pulse_amplitude'],
            'resonators': default_config['pulse_resonators'],
            # Random pulse amplitude distribution
            'random_amp_mode': default_config['pulse_random_amp_mode'],
            'random_amp_min': default_config['pulse_random_amp_min'],
            'random_amp_max': default_config['pulse_random_amp_max'],
            'random_amp_logmean': default_config['pulse_random_amp_logmean'],
            'random_amp_logsigma': default_config['pulse_random_amp_logsigma'],
        }
        self.last_pulse_time = {}  # Track last pulse time for each resonator
        self.last_update_time = 0  # Track last time we updated QP densities
        
        # Vectorized parameter arrays (populated by _extract_param_arrays)
        self._param_arrays_cached = False
        self.L_array = None
        self.R_array = None
        self.C_array = None
        self.Cc_array = None
        self.L_junk_array = None
        
        # Per-operating-point convergence cache from SoT
        self._convergence_cache = {}
        self._convergence_cache_max_size = default_config['convergence_cache_max_size']

        # Statistics tracking
        self._convergence_stats = {
            'full': 0,
            'skipped': 0,
            'last_reason': None
        }
        
        # Rolling cache statistics (last 100 calls)
        self._recent_cache_results = []  # List of True/False for cache hits
        self._stats_counter = 0
        
        # Tolerance settings from SoT - note: these use the old names for backward compat
        # but will be overridden in generate_resonators with new names
        self._tolerance_config = {
            'cache_freq_tolerance': default_config.get('cache_freq_step', 0.0001),
            'cache_amp_tolerance': default_config.get('cache_amp_step', 1e-8),
            'qp_change_threshold': default_config.get('cache_qp_step', 0.0001),
        }
        # Cache/logging controls from SoT
        self._resonator_gen = 0
        self._cache_log_counter = 0
        
        # Physics lock to serialize state updates (prevent race conditions)
        self._physics_lock = threading.RLock()

    def _extract_param_arrays(self):
        """Extract LEKID parameters into numpy arrays for vectorized calculations."""
        if not self.mr_lekids:
            return
        
        n = len(self.mr_lekids)
        self.L_array = np.zeros(n)
        self.R_array = np.zeros(n)
        self.C_array = np.zeros(n)
        self.Cc_array = np.zeros(n)
        self.L_junk_array = np.zeros(n)
        
        for i, lekid in enumerate(self.mr_lekids):
            self.L_array[i] = lekid.L
            self.R_array[i] = lekid.R
            self.C_array[i] = lekid.C
            self.Cc_array[i] = lekid.Cc
            self.L_junk_array[i] = lekid.L_junk
        
        self._param_arrays_cached = True
    
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
            # Use physics_config from MockCRS if available, otherwise use unified defaults
            if hasattr(self.mock_crs, 'physics_config') and self.mock_crs._physics_config:
                config = self.mock_crs._physics_config
            else:
                from .config import defaults
                config = defaults()
        
        print('Using config:', {k: v for k, v in config.items() if k in ['num_resonances', 'freq_start', 'freq_end', 'T', 'Popt']})
        
        # Set random seed for reproducible resonator generation
        # Use separate RandomState for C/Cc variations to avoid pollution from
        # varying iteration counts in the binary search convergence loop
        seed = config.get('resonator_random_seed', 42)
        np.random.seed(seed)  # Global seed for any internal MR_complex_resonator randomness
        variation_rng = np.random.RandomState(seed)  # Dedicated RNG for circuit variations
        
        # Extract parameters
        freq_start = config.get('freq_start', 1e9)
        freq_end = config.get('freq_end', 1.5e9)
        
        # Physics parameters (determine Lk and R via quasiparticle density)
        T = config.get('T', 0.12)
        Popt = config.get('Popt', 1e-18)
        
        # Material & Geometry (use config if provided, otherwise MR_complex_resonator defaults)
        material = config.get('material', 'Al')
        width = config.get('width', 2e-6)  # Default from MR_complex_resonator
        thickness = config.get('thickness', 30e-9)  # Default from MR_complex_resonator
        length = config.get('length', 9000e-6)  # Default derived: VL/(width*thickness) where VL=540e-18
        
        # Custom material properties (from dialog or config)
        # N0 in config is stored in µm⁻³eV⁻¹, must convert to µm⁻³J⁻¹ for MR_complex_resonator
        # Tc, tau0, sigmaN are used directly (no conversion needed)
        custom_Tc = config.get('material_Tc', None)
        custom_N0_eV = config.get('material_N0', None)  # µm⁻³eV⁻¹
        custom_N0 = custom_N0_eV / 1.602e-19 if custom_N0_eV is not None else None  # Convert to µm⁻³J⁻¹
        custom_tau0 = config.get('material_tau0', None)
        custom_sigmaN = config.get('material_sigmaN', None)
        
        if custom_Tc is not None:
            print(f"Using custom material: Tc={custom_Tc}K, N0={custom_N0_eV}µm⁻³eV⁻¹, tau0={custom_tau0}s")
        
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
        
        # Clear existing objects to prevent memory leaks
        self.mr_lekids = []
        self.mr_complex_resonators = []
        self.resonator_frequencies = []
        self.kinetic_inductance_fractions = []
        self.resonator_q_values = []  # Clear Q values
        self.resonator_linewidths = []  # Clear linewidths
        
        # Clear layered parameter tracking
        self.base_lekid_params = []
        self.base_nqp_values = []  # Clear base nqp values
        self.lk_current_factors = []
        self.resonator_currents = []
        
        # Clear cached parameter arrays - CRITICAL for reconfiguration
        # This fixes the array size mismatch when going from more to fewer resonators
        self._param_arrays_cached = False
        self.L_array = None
        self.R_array = None
        self.C_array = None
        self.Cc_array = None
        self.L_junk_array = None
        
        # Also clear the resonator currents array if it exists
        if hasattr(self, 'resonator_currents_array'):
            self.resonator_currents_array = None
        
        # Clear convergence state when resonators change
        self._last_convergence = {'freq': None, 'amp': None, 'nqp_snapshot': None}
        self._convergence_stats = {'full': 0, 'skipped': 0, 'last_reason': None}
        # Also clear convergence cache to avoid size mismatches after reconfiguration
        self._convergence_cache.clear()
        # Bump generation and clear pulses/schedules
        self._resonator_gen += 1
        self.pulse_events = []
        self.last_pulse_time = {}
        self.last_update_time = 0  # Reset time tracking so pulses work after reconfiguration
        
        # Configure noise parameters from config
        # Uses defaults from mock_crs_helper.py if not specified
        self.nqp_noise_enabled = config.get('nqp_noise_enabled', True)
        self.nqp_noise_std_factor = config.get('nqp_noise_std_factor', 0.001)  # Default 0.1% noise if not in config
        
        # Update tolerance settings from config (keep existing if not specified or None)
        self._tolerance_config['cache_freq_tolerance'] = config.get('cache_freq_tolerance', self._tolerance_config['cache_freq_tolerance'])
        self._tolerance_config['cache_amp_tolerance'] = config.get('cache_amp_tolerance', self._tolerance_config['cache_amp_tolerance'])
        self._tolerance_config['qp_change_threshold'] = config.get('qp_change_threshold', self._tolerance_config['qp_change_threshold'])
        
        print(f"Tolerance settings: freq={self._tolerance_config['cache_freq_tolerance']} Hz, "
              f"amp={self._tolerance_config['cache_amp_tolerance']}, "
              f"QP threshold={self._tolerance_config['qp_change_threshold']*100:.1f}%")

        # Update pulse configuration with all pulse parameters from config
        # This ensures tau values persist through reconfiguration
        self.pulse_config.update({
            'mode': config.get('pulse_mode', 'none'),
            'period': config.get('pulse_period', 10.0),
            'probability': config.get('pulse_probability', 0.001),
            'tau_rise': config.get('pulse_tau_rise', 1e-6),
            'tau_decay': config.get('pulse_tau_decay', 1e-1),
            'amplitude': config.get('pulse_amplitude', 2.0),
            'resonators': config.get('pulse_resonators', 'all'),
            'random_amp_mode': config.get('pulse_random_amp_mode', 'fixed'),
            'random_amp_min': config.get('pulse_random_amp_min', 1.5),
            'random_amp_max': config.get('pulse_random_amp_max', 3.0),
            'random_amp_logmean': config.get('pulse_random_amp_logmean', 0.7),
            'random_amp_logsigma': config.get('pulse_random_amp_logsigma', 0.3),
        })
        
        print(f"Pulse config updated: tau_rise={self.pulse_config['tau_rise']}, tau_decay={self.pulse_config['tau_decay']}")

        # Step 1: Create a reference MR_complex_resonator to compute Lk and R from T and Popt
        print(f"Computing Lk and R from T={T} K and Popt={Popt} W")
        
        # Create reference resonator with dummy C value to get physics-computed Lk and R
        # NOTE: Include L_junk so the reference Lk is calculated in the same circuit context
        reference_params = {
            'T': T,
            'Popt': Popt,
            'width': width,
            'thickness': thickness,
            'length': length,
            'C': 1e-12,  # Dummy value, just to create the object
            'Cc': Cc_base,
            'fix_Lg': Lg_base,  # Fix Lg to our desired value
            'L_junk': L_junk,   # Include L_junk for consistent circuit context
            'Vin': Vin,
            'input_atten_dB': input_atten_dB,
            'base_readout_f': (freq_start + freq_end) / 2,  # Middle frequency as initial guess
            'verbose': False,
            'ZLNA': complex(ZLNA, 0),
            'GLNA': GLNA
        }
        # Add custom material properties if provided (N0 already converted to J⁻¹ units)
        if custom_Tc is not None:
            reference_params['Tc'] = custom_Tc
        if custom_N0 is not None:
            reference_params['N0'] = custom_N0
        if custom_tau0 is not None:
            reference_params['tau0'] = custom_tau0
        if custom_sigmaN is not None:
            reference_params['sigmaN'] = custom_sigmaN
        
        # Create reference resonator to get physics-computed Lk and R
        ref_resonator = MR_complex_resonator(**reference_params)
        computed_Lk = ref_resonator.lekid.Lk
        computed_R = ref_resonator.lekid.R
        
        print(f"Physics-computed values: Lk={computed_Lk*1e9:.2f} nH, R={computed_R*1e6:.2f} µΩ")
        
        # Step 2: Generate resonators with iterative C-finding algorithm
        # The key challenge is that Lk depends on frequency, so we need to iterate
        # until the actual resonance frequency matches the target.
        print(f"Generating {num_resonances} resonators from {freq_start/1e9:.2f} GHz to {freq_end/1e9:.2f} GHz")
        
        # Determine bounds for frequency correction
        f_min_bound = min(freq_start, freq_end)
        f_max_bound = max(freq_start, freq_end)
        
        # Tolerance for frequency convergence (0.1% of target)
        freq_tolerance_fraction = 0.001
        max_c_iterations = 20  # Maximum iterations for C-finding
        
        for x in range(num_resonances):
            try:
                # Calculate target frequency for this resonator
                if num_resonances == 1:
                    target_freq = (freq_start + freq_end) / 2  # Middle frequency
                else:
                    target_freq = freq_start + (freq_end - freq_start) * x / (num_resonances - 1)
                
                # Apply Cc variation using dedicated RNG (doesn't change during iteration)
                Cc_actual = Cc_base * (1 + variation_rng.normal(0, Cc_variation))
                Cc_actual = max(Cc_actual, Cc_base * 0.1)
                
                # --- Iterative C-finding algorithm ---
                # Use binary search to find C that gives the target frequency
                # Since Lk depends on frequency (through Mattis-Bardeen), we need to iterate
                
                # Initial guess for C using the reference Lk
                L_total_guess = computed_Lk + Lg_base + L_junk
                C_initial = 1 / (4 * np.pi**2 * target_freq**2 * L_total_guess)
                
                # Set up binary search bounds
                # Start with very wide bounds (factor of 100 in each direction)
                C_low = C_initial / 100
                C_high = C_initial * 100
                C_current = C_initial
                
                print(f"Resonator {x}: target_freq={target_freq/1e9:.3f} GHz")
                
                # Base resonator params (C will be updated during iteration)
                complex_res_params = {
                    'T': T,
                    'Popt': Popt,
                    'C': C_current,
                    'Cc': Cc_actual,
                    'fix_Lg': Lg_base,
                    'L_junk': L_junk,
                    'Vin': Vin,
                    'input_atten_dB': input_atten_dB,
                    'base_readout_f': target_freq,
                    'verbose': False,
                    'ZLNA': complex(ZLNA, 0),
                    'GLNA': GLNA,
                    'width': width,
                    'thickness': thickness,
                    'length': length,
                }
                # Add custom material properties if provided (N0 already converted to J⁻¹ units)
                if custom_Tc is not None:
                    complex_res_params['Tc'] = custom_Tc
                if custom_N0 is not None:
                    complex_res_params['N0'] = custom_N0
                if custom_tau0 is not None:
                    complex_res_params['tau0'] = custom_tau0
                if custom_sigmaN is not None:
                    complex_res_params['sigmaN'] = custom_sigmaN
                
                best_freq = None
                best_C = C_current
                best_error = float('inf')
                
                for iteration in range(max_c_iterations):
                    # Create resonator with current C guess
                    complex_res_params['C'] = C_current
                    complex_res = MR_complex_resonator(**complex_res_params)
                    lekid = complex_res.lekid
                    
                    # Get actual resonance frequency
                    actual_freq = lekid.compute_fr()
                    
                    # Calculate error
                    freq_error = abs(actual_freq - target_freq) / target_freq
                    
                    # Track best result
                    if freq_error < best_error:
                        best_error = freq_error
                        best_freq = actual_freq
                        best_C = C_current
                    
                    # Check convergence
                    if freq_error < freq_tolerance_fraction:
                        if iteration > 0:
                            print(f"  Converged in {iteration+1} iterations: f={actual_freq/1e9:.4f} GHz (error={freq_error*100:.2f}%)")
                        break
                    
                    # Update bounds based on whether we're above or below target
                    if actual_freq > target_freq:
                        # Frequency too high -> need larger C to lower frequency
                        C_low = C_current
                    else:
                        # Frequency too low -> need smaller C to raise frequency
                        C_high = C_current
                    
                    # Binary search: take geometric mean of bounds
                    C_current = np.sqrt(C_low * C_high)
                    
                    # Safety: if bounds have collapsed, break
                    if C_high / C_low < 1.001:
                        print(f"  Bounds collapsed after {iteration+1} iterations: f={actual_freq/1e9:.4f} GHz")
                        break
                else:
                    print(f"  Max iterations reached: f={best_freq/1e9:.4f} GHz (error={best_error*100:.2f}%)")
                
                # Use the best result we found
                complex_res_params['C'] = best_C
                
                # Apply C variation using dedicated RNG now that we have the target C
                C_with_variation = best_C * (1 + variation_rng.normal(0, C_variation))
                C_with_variation = max(C_with_variation, best_C * 0.5)  # At least 50% of target
                complex_res_params['C'] = C_with_variation
                
                # Final resonator creation with variation applied
                complex_res = MR_complex_resonator(**complex_res_params)
                lekid = complex_res.lekid
                actual_freq = lekid.compute_fr()

                print(f"  Actual frequency: {actual_freq/1e9:.4f} GHz")
                print(f"  Circuit: C={lekid.C*1e12:.3f} pF, Cc={lekid.Cc*1e15:.2f} fF, Lg={lekid.Lg*1e9:.2f} nH, Lk={lekid.Lk*1e9:.2f} nH")
                print(f"  Derived: L_total={lekid.L*1e9:.2f} nH, R={lekid.R*1e6:.2f} µΩ, α_k={lekid.alpha_k:.4f}")
                
                # Compute all derived values BEFORE appending to lists to ensure atomicity
                # This prevents partial updates if a calculation fails
                
                # Compute base nqp using calc_nqp() method
                base_nqp = complex_res.calc_nqp()
                print(f"  Base nqp: {base_nqp:.2e}")
                
                # Pre-calculate Q value
                try:
                    Q_value = lekid.compute_Qi()
                    print(f"  Q value: {Q_value:.0f}")
                except:
                    Q_value = 1000  # Fallback Q if calculation fails
                    print(f"  Q value: {Q_value:.0f} (fallback)")
                
                # Pre-compute linewidth
                linewidth = actual_freq / Q_value
                print(f"  Linewidth: {linewidth/1e3:.2f} kHz")
                
                # --- ATOMIC UPDATE START ---
                # Store the persistent objects (this fixes the memory leak)
                self.mr_lekids.append(lekid)
                self.mr_complex_resonators.append(complex_res)  # Keep for future QP tracking
                self.base_nqp_values.append(base_nqp)
                
                # Store base parameters (as-fabricated values)
                self.base_lekid_params.append({
                    'R': lekid.R,
                    'Lk': lekid.Lk,
                    'Lg': lekid.Lg,
                    'C': lekid.C,
                    'Cc': lekid.Cc
                })
                
                # Initialize current factors to 1 (no modification)
                self.lk_current_factors.append(1.0)
                self.resonator_currents.append(0.)
                
                # Store metadata (use actual computed frequency)
                self.resonator_frequencies.append(actual_freq)
                self.kinetic_inductance_fractions.append(lekid.alpha_k)
                self.resonator_q_values.append(Q_value)
                self.resonator_linewidths.append(linewidth)
                # --- ATOMIC UPDATE END ---
                
            except Exception as e:
                print(f"Warning: Failed to create resonator {x}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with other resonators
                continue
        
        print(f"Created {len(self.mr_lekids)} persistent LEKID objects")

        # Configure pulse events if specified in config
        pulse_mode = config.get('pulse_mode', 'none')
        if pulse_mode != 'none':
            self.set_pulse_mode(
                pulse_mode,
                period=config.get('pulse_period', 10.0),
                probability=config.get('pulse_probability', 0.001),
                tau_rise=config.get('pulse_tau_rise', 1e-6),
                tau_decay=config.get('pulse_tau_decay', 1e-3),
                amplitude=config.get('pulse_amplitude', 2.0),
                resonators=config.get('pulse_resonators', 'all'),
                # Random amplitude distribution (random mode)
                random_amp_mode=config.get('pulse_random_amp_mode', 'fixed'),
                random_amp_min=config.get('pulse_random_amp_min', 1.5),
                random_amp_max=config.get('pulse_random_amp_max', 3.0),
                random_amp_logmean=config.get('pulse_random_amp_logmean', 0.7),
                random_amp_logsigma=config.get('pulse_random_amp_logsigma', 0.3),
            )
        
        self.invalidate_caches()

    def s21_lc_response(self, frequency, amplitude=1.0):
        """
        Calculate S21 response with optimized convergence.
        
        Includes:
        - ALL resonators for continuous S21 (no artificial boundaries)
        - Pulse-modified quasiparticle density (if pulses are active)
        - Fresh noisy quasiparticle density each call
        - Physics-based Lk and R from combined nqp
        - OPTIMIZED: Skip convergence for noise-only changes via caching
        
        THREAD SAFETY: This method is protected by _physics_lock because it 
        updates the shared state (self.mr_lekids) during calculation.
        """
        # Acquire lock to prevent race conditions with other threads (e.g. Streamer vs get_samples)
        # This is critical because update_lekids_for_current modifies shared state.
        with self._physics_lock:
            return self._s21_lc_response_internal(frequency, amplitude)

    def _s21_lc_response_internal(self, frequency, amplitude=1.0):
        """Internal implementation of s21_lc_response (assumes lock held)."""
        import time
        t_start = time.perf_counter()
        
        if not self.mr_lekids:
            return 1.0 + 0j
        
        # Calculate effective nqp with fresh noise for ALL resonators (vectorized)
        base_nqp_array = np.array(self.base_nqp_values, dtype=np.float64)
        
        # Start with base values
        effective_nqp_array = base_nqp_array.copy()
        
        # Add pulse contributions (only loop through active pulses)
        for pulse in self.pulse_events:
            i = pulse['resonator_index']
            if i < len(effective_nqp_array):
                pulse_dt = self.last_update_time - pulse['start_time']
                if pulse_dt >= 0:
                    if pulse_dt < pulse['tau_rise']:
                        time_factor = (1 - np.exp(-pulse_dt / pulse['tau_rise']))
                    else:
                        decay_dt = pulse_dt - pulse['tau_rise']
                        time_factor = np.exp(-decay_dt / pulse['tau_decay'])
                    excess_factor = (pulse['amplitude'] - 1.0) * time_factor
                    effective_nqp_array[i] += excess_factor * base_nqp_array[i]
        
        # Add fresh noise (vectorized)
        if self.nqp_noise_enabled:
            # Generate noise for all resonators at once
            noise_array = np.random.normal(0, base_nqp_array * self.nqp_noise_std_factor, len(base_nqp_array))
            effective_nqp_array = np.maximum(0, effective_nqp_array + noise_array)
        else:
            # Ensure non-negative even without noise
            effective_nqp_array = np.maximum(0, effective_nqp_array)
        
        # Convert to list for compatibility with existing code
        effective_nqp = effective_nqp_array.tolist()
        
        # Always update R,Lk from physics (fast, preserves IQ behavior)
        self.update_base_params_from_nqp(effective_nqp)
        
        # === OPTIMIZATION: Check if we need convergence (per-resonator qp-aware key) ===
        # Compute nearest resonator by current f0 estimate
        f0_list = []
        for lek in self.mr_lekids:
            try:
                L_eff = max(lek.L, 1e-30)
                C_eff = max(lek.C, 1e-30)
                f0_list.append(1.0 / (2.0 * np.pi * np.sqrt(L_eff * C_eff)))
            except Exception:
                f0_list.append(0.0)
        if f0_list:
            nearest_idx = int(np.argmin(np.abs(np.array(f0_list) - frequency)))
        else:
            nearest_idx = 0

        # Get cache parameters from config
        phys = getattr(self.mock_crs, 'physics_config', {})
        if not isinstance(phys, dict):
            phys = {}
        
        # Frequency step from config
        freq_step = phys.get('cache_freq_step', 0.0001)  # Default 0.0001 Hz
        if freq_step <= 0:
            freq_step = 0.0001
        freq_key = round(frequency / freq_step) * freq_step

        # Amplitude step from config
        amp_step = phys.get('cache_amp_step', 1e-8)  # Default 1e-8
        if amp_step <= 0:
            amp_step = 1e-8
        amp_key = round(amplitude / amp_step) * amp_step

        # QP step from config (as fraction of base QP)
        qp_step_fraction = phys.get('cache_qp_step', 0.001)  # Default 0.1%
        if qp_step_fraction <= 0:
            qp_step_fraction = 0.001
        
        # Calculate absolute QP step based on median base value
        if self.base_nqp_values:
            base_med = float(np.median(np.array(self.base_nqp_values, dtype=float)))
            qp_step = max(1e-300, abs(base_med) * qp_step_fraction)
        else:
            qp_step = 1e-6  # Fallback
        
        qp_val = effective_nqp[nearest_idx] if 0 <= nearest_idx < len(effective_nqp) else 0.0
        qp_key = round(qp_val / qp_step) * qp_step

        cache_key = (nearest_idx, freq_key, amp_key, qp_key)

        # Check cache
        skip_convergence = False
        cached_data = self._convergence_cache.get(cache_key)
        reason = 'miss'
        if cached_data is not None:
            if cached_data.get('gen') != getattr(self, '_resonator_gen', 0):
                reason = 'gen_changed'
            elif cached_data.get('lekid_count') != len(self.mr_lekids):
                reason = 'count_changed'
            else:
                skip_convergence = True
                reason = 'hit'
        self._convergence_stats['last_reason'] = reason
        
        # Track cache hit/miss for rolling statistics
        self._recent_cache_results.append(skip_convergence)
        if len(self._recent_cache_results) > 100:
            self._recent_cache_results.pop(0)  # Keep only last 100
        
        # Print cache statistics every 100 calls (if logging enabled)
        self._stats_counter += 1
        log_enabled = phys.get('log_cache_decisions', False) if isinstance(phys, dict) else False
        log_every = phys.get('cache_log_interval', 100) if isinstance(phys, dict) else 100
        if log_enabled and (self._stats_counter % max(1, int(log_every)) == 0):
            recent_hits = sum(self._recent_cache_results)
            recent_total = len(self._recent_cache_results)
            hit_rate = (recent_hits / recent_total * 100) if recent_total > 0 else 0
            print(f"[Cache Stats] Last {recent_total} calls: {hit_rate:.1f}% cache hits ({recent_hits} hits, {recent_total - recent_hits} misses)")
        
        # Update parameters based on convergence decision
        if not skip_convergence:
            # Run full convergence
            self.update_lekids_for_current(frequency, amplitude)

            # Update cache capacity from config if provided
            phys = getattr(self.mock_crs, 'physics_config', {})
            if isinstance(phys, dict):
                max_size = phys.get('convergence_cache_max_size', self._convergence_cache_max_size)
            else:
                max_size = self._convergence_cache_max_size
            if isinstance(max_size, int) and max_size > 0:
                self._convergence_cache_max_size = max_size

            # Cache the actual converged values (not factors) for this operating point
            self._convergence_cache[cache_key] = {
                'Lk_values': [lekid.Lk for lekid in self.mr_lekids],
                'R_values': [lekid.R for lekid in self.mr_lekids],
                'L_values': [lekid.L for lekid in self.mr_lekids],
                'frequency': frequency,
                'amplitude': amplitude,
                'qp_key': qp_key,
                'nearest_idx': nearest_idx,
                'gen': getattr(self, '_resonator_gen', 0),
                'lekid_count': len(self.mr_lekids)
            }

            # Limit cache size
            if len(self._convergence_cache) > self._convergence_cache_max_size:
                oldest_key = next(iter(self._convergence_cache))
                del self._convergence_cache[oldest_key]

            # Update statistics
            self._convergence_stats['full'] += 1

        else:
            # Restore cached converged values - apply to ALL resonators
            cached_Lk = cached_data.get('Lk_values')
            cached_R = cached_data.get('R_values')
            cached_L = cached_data.get('L_values')
            
            for i in range(len(self.mr_lekids)):
                lekid = self.mr_lekids[i]
                
                # Restore the exact cached physics state
                if cached_Lk and i < len(cached_Lk):
                    lekid.Lk = cached_Lk[i]
                if cached_R and i < len(cached_R):
                    lekid.R = cached_R[i]
                if cached_L and i < len(cached_L):
                    lekid.L = cached_L[i]
                    lekid.alpha_k = lekid.Lk / lekid.L
            
            # Update statistics
            self._convergence_stats['skipped'] += 1
        
        # Extract parameters for ALL resonators
        # Note: L now includes L_junk (L = Lk + Lg + L_junk)
        n_relevant = len(self.mr_lekids)
        L_subset = np.zeros(n_relevant)
        C_subset = np.zeros(n_relevant)
        R_subset = np.zeros(n_relevant)
        Cc_subset = np.zeros(n_relevant)
        
        for i in range(n_relevant):
            lekid = self.mr_lekids[i]
            L_subset[i] = lekid.L  # Total inductance (includes L_junk)
            C_subset[i] = lekid.C
            R_subset[i] = lekid.R
            Cc_subset[i] = lekid.Cc
        
        # Get common parameters from first LEKID (they should all be the same)
        lekid0 = self.mr_lekids[0]
        
        # Use new parallel S21 calculation that properly combines all resonators
        # This calculates the transmission to the load with all resonators in parallel
        s21_total = jit_physics.compute_s21_parallel(
            fc=frequency,
            Vin=amplitude,
            L_array=L_subset,
            C_array=C_subset,
            R_array=R_subset,
            Cc_array=Cc_subset,
            ZLNA=complex(lekid0.ZLNA),
            GLNA=lekid0.GLNA,
            input_atten_dB=lekid0.input_atten_dB,
            system_termination=lekid0.system_termination
        )
        
        t_vout = time.perf_counter()
        
        return s21_total

    def update_lekids_for_current(self, frequency, amplitude):
        """
        Update LEKID parameters based on resonator currents.
        
        Uses JIT-compiled convergence loop for 2-5x speedup.
        
        Parameters
        ----------
        frequency : float
            Probe frequency in Hz
        amplitude : float
            Probe amplitude  
            
        Convergence tolerance can be configured via physics_config:
        - 1e-9: Ultra high accuracy (default)
        - 1e-7: High accuracy 
        - 1e-5: Balanced
        - 1e-3: Ultra fast (for many channels)
        """
        max_iterations = 500
        tolerance = self.mock_crs._physics_config.get('convergence_tolerance', 1e-9)
        
        # Get LEKID config from first resonator
        lekid0 = self.mr_lekids[0]
        
        # Update all resonators
        self._extract_param_arrays()
        
        n = len(self.mr_lekids)
        
        # Ensure arrays are initialized
        if self.L_array is None or self.R_array is None:
            self._extract_param_arrays()
        
        # Use full arrays
        L_work = self.L_array
        R_work = self.R_array
        C_work = self.C_array
        Cc_work = self.Cc_array
        base_Lk = np.array([self.base_lekid_params[i]['Lk'] for i in range(n)])
        base_Lg = np.array([self.base_lekid_params[i]['Lg'] for i in range(n)])
        base_L_junk = self.L_junk_array  # L_junk is fixed per resonator
        
        # Call JIT-compiled convergence loop
        # Note: L_work = Lk + Lg + L_junk (total resonator inductance)
        # base_L_junk is fixed; only Lk changes with current
        L_converged, R_converged, currents_converged, actual_iterations = \
            jit_physics.converged_lekid_parameters(
                frequency, amplitude,
                L_work, R_work, C_work, Cc_work,
                base_Lk, base_Lg, base_L_junk,
                lekid0.input_atten_dB, complex(lekid0.ZLNA),
                self.Istar, tolerance, max_iterations, damp=0.1
            )
        
        # Extract current factors from converged inductances
        # L_converged = Lk_converged + Lg + L_junk, so Lk_converged = L_converged - Lg - L_junk
        Lk_converged = L_converged - base_Lg - base_L_junk
        current_factors = Lk_converged / base_Lk
        
        # Update LEKID objects with converged values (guard against concurrent reconfigure)
        m = min(n, len(self.mr_lekids))
        for i in range(m):
            lekid = self.mr_lekids[i]
            lekid.Lk = Lk_converged[i]
            lekid.R = R_converged[i]
            lekid.L = L_converged[i]
            lekid.alpha_k = Lk_converged[i] / L_converged[i]

        # Update cached data sized to current resonator count
        mlen = len(self.mr_lekids)
        try:
            # Convert numpy array to list
            factors_array = current_factors[:m]
            base_factors = [float(x) for x in factors_array]
        except Exception:
            base_factors = [1.0] * m
        
        # Extend if needed
        if mlen > m:
            for _ in range(mlen - m):
                base_factors.append(1.0)
        
        self.lk_current_factors = base_factors

        try:
            # Ensure a flat list[complex] regardless of the dtype/shape returned by JIT
            # NOTE: currents_converged contains complex values (phasors).
            # We preserve them as complex to maintain phase information and correct magnitude calculations.
            vec = np.asarray(currents_converged, dtype=complex).reshape(-1)
            curr_list = vec[:m].tolist()
        except Exception:
            curr_list = [0j] * m
        pad_len = mlen - m
        if pad_len > 0:
            curr_list = curr_list + [0j] * pad_len
        self.resonator_currents_array = np.array(curr_list, dtype=complex)
        self.resonator_currents = curr_list

        # Refresh L/R arrays from current objects
        self.L_array = np.array([lek.L for lek in self.mr_lekids])
        self.R_array = np.array([lek.R for lek in self.mr_lekids])
        
        # Log convergence stats occasionally
        if not hasattr(self, '_convergence_counter'):
            self._convergence_counter = 0
        self._convergence_counter += 1

    def update_base_params_from_nqp(self, noisy_nqp_values):
        """
        Update base Lk and R values using physics calculations.
        
        Uses JIT-compiled parallel vectorized calculations for 15-25x speedup.
        
        Parameters
        ----------
        noisy_nqp_values : list
            List of noisy nqp values, one per resonator
        """
        # Prepare arrays for vectorized calculation
        n = len(self.mr_complex_resonators)
        nqp_array = np.array(noisy_nqp_values, dtype=np.float64)
        readout_freqs = np.array([cr.readout_f for cr in self.mr_complex_resonators], dtype=np.float64)
        T_array = np.full(n, self.mr_complex_resonators[0].T, dtype=np.float64)
        Delta0_array = np.full(n, self.mr_complex_resonators[0].Delta0, dtype=np.float64)
        N0_array = np.full(n, self.mr_complex_resonators[0].N0, dtype=np.float64)
        sigmaN_array = np.full(n, self.mr_complex_resonators[0].sigmaN, dtype=np.float64)
        thickness_array = np.full(n, self.mr_complex_resonators[0].thickness, dtype=np.float64)
        width_array = np.full(n, self.mr_complex_resonators[0].width, dtype=np.float64)
        length_array = np.full(n, self.mr_complex_resonators[0].length, dtype=np.float64)
        R_spoiler_array = np.full(n, self.mr_complex_resonators[0].R_spoiler, dtype=np.float64)
        
        # Call JIT-compiled function - computes ALL resonators in parallel
        R_array, Lk_array = jit_physics.vectorized_update_params_from_nqp(
            nqp_array, readout_freqs, T_array, Delta0_array,
            N0_array, sigmaN_array, thickness_array,
            width_array, length_array, R_spoiler_array
        )
        
        # Update all base parameters
        for i in range(n):
            self.base_lekid_params[i]['R'] = R_array[i]
            self.base_lekid_params[i]['Lk'] = Lk_array[i]

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
            'current_factor_Lk': self.lk_current_factors[lekid_index],
            'final_Lk': current.Lk,
            'final_R': current.R
        }
    
    def invalidate_caches(self):
        """Clear caches when resonator parameters change."""
        self._s21_cache.clear()
        self._cic_cache.clear()
        self._cache_valid = False
    
    def update_qp_densities_for_time(self, current_time):
        """Update all resonator QP densities based on active pulses.
        
        Parameters
        ----------
        current_time : float
            Current time in seconds since streaming started
        """
        # Only update if time has advanced
        if current_time <= self.last_update_time:
            return
        
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Check if we should trigger new pulses
        self._check_trigger_pulses(current_time, dt)
        
        # Calculate QP densities for all resonators based on active pulses
        pulse_nqp_values = []
        
        for i in range(len(self.mr_lekids)):
            # Get base nqp for this resonator
            base_nqp = self.base_nqp_values[i] if i < len(self.base_nqp_values) else 0
            
            # Calculate total excess QP contribution from all active pulses
            excess_qp = 0
            
            # Filter active pulses for this resonator
            for pulse in self.pulse_events:
                if pulse['resonator_index'] == i:
                    # Calculate pulse contribution at current time
                    pulse_dt = current_time - pulse['start_time']
                    if pulse_dt >= 0:
                        # Exponential rise and decay model
                        if pulse_dt < pulse['tau_rise']:  # Rising edge
                            time_factor = (1 - np.exp(-pulse_dt / pulse['tau_rise']))
                        else:  # Decay phase
                            rise_time = pulse['tau_rise']
                            decay_dt = pulse_dt - rise_time
                            time_factor = np.exp(-decay_dt / pulse['tau_decay'])
                        
                        # Calculate excess QP relative to base_nqp
                        # amplitude=2.0 means total_nqp = base_nqp + 1.0*base_nqp = 2*base_nqp
                        excess_factor = (pulse['amplitude'] - 1.0) * time_factor
                        excess_qp += excess_factor * base_nqp
            
            # Calculate total absolute QP density
            total_nqp = base_nqp + excess_qp
            
            # Ensure non-negative QP density
            total_nqp = max(0, total_nqp)
            
            pulse_nqp_values.append(total_nqp)
        
        # Update base parameters using physics-based method (same as noise)
        if pulse_nqp_values:
            self.update_base_params_from_nqp(pulse_nqp_values)
            self.invalidate_caches()
        
        # Clean up old pulses (after n decay constants)
        self.pulse_events = [p for p in self.pulse_events 
                           if current_time - p['start_time'] < p['tau_rise'] + p['tau_decay'] * 15]
    
    def _sample_random_pulse_amplitude(self):
        """Sample a pulse amplitude for random mode based on configured distribution."""
        mode = self.pulse_config.get('random_amp_mode', 'fixed')
        if mode == 'uniform':
            amin = float(self.pulse_config.get('random_amp_min', 1.5))
            amax = float(self.pulse_config.get('random_amp_max', 3.0))
            amp = np.random.uniform(amin, amax)
        elif mode == 'lognormal':
            mu = float(self.pulse_config.get('random_amp_logmean', 0.7))
            sigma = float(self.pulse_config.get('random_amp_logsigma', 0.3))
            amp = np.random.lognormal(mean=mu, sigma=sigma)
        else:
            amp = float(self.pulse_config.get('amplitude', 2.0))
        # Enforce non-decreasing QP unless explicitly configured otherwise
        return max(1.0, amp)

    def _check_trigger_pulses(self, current_time, dt):
        """Check if new pulses should be triggered based on mode.
        
        Parameters
        ----------
        current_time : float
            Current time in seconds
        dt : float
            Time step since last update
        """
        mode = self.pulse_config['mode']
        
        if mode == 'none' or mode == 'manual':
            return
        
        # Determine which resonators can receive pulses
        if self.pulse_config['resonators'] == 'all':
            target_resonators = list(range(len(self.mr_lekids)))
        else:
            target_resonators = self.pulse_config['resonators']
        
        if mode == 'periodic':
            # Check each resonator for periodic pulses
            period = self.pulse_config['period']
            for res_idx in target_resonators:
                last_time = self.last_pulse_time.get(res_idx, -period)
                if current_time - last_time >= period:
                    self.add_pulse_event(res_idx, current_time)
                    self.last_pulse_time[res_idx] = current_time
        
        elif mode == 'random':
            # Random pulses based on probability per timestep
            prob = self.pulse_config['probability']
            for res_idx in target_resonators:
                if np.random.random() < prob * dt:
                    amp = self._sample_random_pulse_amplitude()
                    self.add_pulse_event(res_idx, current_time, amplitude=amp)
                    self.last_pulse_time[res_idx] = current_time
    
    def add_pulse_event(self, resonator_index, start_time, amplitude=None):
        """Manually add a pulse event to a specific resonator.
        
        Parameters
        ----------
        resonator_index : int
            Index of the resonator (0-based)
        start_time : float
            Time when the pulse starts (seconds)
        amplitude : float, optional
            Maximum QP density increase. If None, uses config default.
        """
        if resonator_index >= len(self.mr_lekids):
            print(f"Warning: Resonator index {resonator_index} out of range")
            return
        
        pulse = {
            'resonator_index': resonator_index,
            'start_time': start_time,
            'amplitude': amplitude or self.pulse_config['amplitude'],
            'tau_rise': self.pulse_config['tau_rise'],
            'tau_decay': self.pulse_config['tau_decay'],
        }
        self.pulse_events.append(pulse)
        
        # Invalidate caches since parameters will change
        self.invalidate_caches()
    
    def set_pulse_mode(self, mode, **kwargs):
        """Configure pulse generation mode and parameters.
        
        Parameters
        ----------
        mode : str
            Pulse mode: 'periodic', 'random', 'manual', or 'none'
        **kwargs : dict
            Additional configuration parameters:
            - period: Period in seconds (for periodic mode)
            - probability: Probability per timestep (for random mode)
            - tau_rise: Rise time constant in seconds
            - tau_decay: Decay time constant in seconds
            - amplitude: Maximum QP density increase
            - resonators: 'all' or list of resonator indices
        """
        self.pulse_config['mode'] = mode
        
        # Update any provided parameters
        for key, value in kwargs.items():
            if key in self.pulse_config:
                self.pulse_config[key] = value
        
        print(f"Pulse mode set to '{mode}' with config: {self.pulse_config}")
        # Reset pulse schedules so new parameters take effect immediately
        if mode in ('periodic', 'random'):
            self.pulse_events = []
            self.last_pulse_time = {}
        elif mode == 'none':
            self.pulse_events = []
            self.last_pulse_time = {}
    
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
        from ..core import transferfunctions as tf
        
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
        
        Each channel evaluation gets proper pulse handling and fresh noise through
        s21_lc_response, maintaining physics accuracy while keeping the vectorized
        convergence optimization for performance.
        
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
        import time
        t_packet_start = time.perf_counter()
        
        # Update QP densities based on current time (for pulses)
        self.update_qp_densities_for_time(start_time)
        
        t_state_update = time.perf_counter()
        
        # Get NCO frequency for this module using the proper getter
        nco_freq = self.mock_crs._nco_frequencies.get(module)
        
        # Get decimation stage for bandwidth calculation
        dec_stage = self.mock_crs._fir_stage  # Note: still called fir_stage in MockCRS for compatibility
        bandwidth = self.cic_bandwidths.get(dec_stage, 298)  # Hz
        
        # Determine sample rate if not provided
        if sample_rate is None:
            sample_rate = 625e6 / 256 / 64 / (2**dec_stage)
        
        # Step 1: Collect active tones and observing channels
        # Acquire configuration lock to ensure atomic read of frequencies and amplitudes
        active_tone_freqs = []
        active_tone_amps = []
        obs_channels = []
        obs_freqs = []
        
        # Use config lock to ensure we get a consistent snapshot of the channel configuration
        with getattr(self.mock_crs, '_config_lock', threading.RLock()): # Fallback if lock missing
            max_channels = self.mock_crs.channels_per_module()
            
            # Find all configured channels in this module
            configured_channels = set()
            for (mod, ch) in self.mock_crs._frequencies.keys():
                if (mod == module) and (ch <= max_channels):
                    configured_channels.add(ch)
            for (mod, ch) in self.mock_crs._amplitudes.keys():
                if (mod == module) and (ch <= max_channels):
                    configured_channels.add(ch)
            
            # Collect active tones (transmitting channels) using proper getter methods
            # We must do this while holding the lock to prevent "frequency set but amplitude missing" race
            raw_channel_configs = []
            
            for ch in configured_channels:
                freq = self.mock_crs._frequencies.get((module, ch))
                amp = self.mock_crs._amplitudes.get((module, ch))
                phase_deg = self.mock_crs._phases.get((module, ch), 0)
                
                if freq is not None and amp is not None and amp != 0:
                    raw_channel_configs.append((ch, freq, amp, phase_deg))
                
                # Also collect observing channels (if freq exists)
                if freq is not None:
                    obs_channels.append(ch)
                    obs_freqs.append(freq + nco_freq)

        # Process the collected configuration (outside lock where possible, though S21 calculation needs physics lock)
        s21_call_count = 0
        for ch, freq, amp, phase_deg in raw_channel_configs:
            total_freq = freq + nco_freq
            
            # For single sample, pre-compute S21. For multiple samples, compute fresh per sample.
            if num_samples == 1:
                # Apply S21 response using FAST path (state already updated for this packet)
                # Note: s21_lc_response acquires _physics_lock internally
                s21_complex = self.s21_lc_response(total_freq, amp)
                s21_call_count += 1
                
                # Combine amplitude, S21, and phase
                complex_amplitude = amp * s21_complex * np.exp(1j * np.deg2rad(phase_deg))
            else:
                # For multi-sample, just store base amplitude with phase
                # S21 will be evaluated fresh for each sample
                complex_amplitude = amp * np.exp(1j * np.deg2rad(phase_deg))
            
            active_tone_freqs.append(total_freq)
            active_tone_amps.append(complex_amplitude)
        
        t_s21_calc = time.perf_counter()
        
        # If no active tones or observers, return empty
        if not active_tone_freqs or not obs_channels:
            return {}
        
        # Diagnostic logging
        if not hasattr(self, '_packet_timing_counter'):
            self._packet_timing_counter = 0
        self._packet_timing_counter += 1
        
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
            # Multiple time samples - evaluate S21 for each sample to get fresh noise
            t = start_time + np.arange(num_samples) / sample_rate
            responses = {}
            
            # For each observing channel
            for i, ch in enumerate(obs_channels):
                signal = np.zeros(num_samples, dtype=complex)
                obs_freq = obs_freqs_arr[i]
                
                # For each time sample
                for sample_idx in range(num_samples):
                    sample_contribution = 0 + 0j
                    
                    # Process each active tone with fresh S21 evaluation
                    for j in range(len(active_tone_freqs)):
                        if within_bandwidth[i, j]:
                            tone_freq = active_tone_freqs[j]
                            tone_amp_base = active_tone_amps[j]
                            freq_diff = freq_diffs[i, j]
                            
                            # Get fresh S21 for this tone at this time sample
                            # This gives us new QP noise realization for each sample
                            amp_magnitude = abs(tone_amp_base)
                            if amp_magnitude > 0:
                                # Re-evaluate S21 with fresh noise
                                s21_fresh = self.s21_lc_response(tone_freq, amp_magnitude)
                                
                                # Apply phase from original tone
                                phase_original = np.angle(tone_amp_base)
                                tone_amp_fresh = amp_magnitude * s21_fresh * np.exp(1j * phase_original)
                            else:
                                tone_amp_fresh = 0 + 0j
                            
                            # Apply CIC filter response
                            cic_response = self._get_cached_cic_response(freq_diff, dec_stage)
                            
                            # Calculate beat contribution at this time
                            if abs(freq_diff) < 0.1:  # DC
                                sample_contribution += tone_amp_fresh * cic_response
                            else:
                                # Beat frequency at time t[sample_idx]
                                beat_phase = 2 * np.pi * freq_diff * t[sample_idx]
                                sample_contribution += tone_amp_fresh * cic_response * np.exp(1j * beat_phase)
                    
                    signal[sample_idx] = sample_contribution
                
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
        for (mod, ch) in self.mock_crs._frequencies.keys():
            if mod == module and ch != channel:
                amp = self.mock_crs._amplitudes.get((mod, ch), 0)
                if amp != 0:
                    other_active_channels = True
                    break
        
        if other_active_channels:
            # Use coupled calculation when multiple channels are active
            # Store this channel's settings temporarily
            nco_freq = self.mock_crs._nco_frequencies.get(module, 0)
            self.mock_crs._frequencies[(module, channel)] = frequency - nco_freq
            self.mock_crs._amplitudes[(module, channel)] = amplitude
            self.mock_crs._phases[(module, channel)] = phase_degrees
            
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
