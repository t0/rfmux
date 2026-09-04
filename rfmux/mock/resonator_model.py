"""
Mock CRS Device - Resonator Physics Model.
Encapsulates the logic for resonator physics simulation, including S21 response.
"""
import heapq
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
        # TLS (1/f) frequency wander — built in generate_resonators once
        # the resonator count is known; None disables it entirely.
        self._tls_generator = None
        self._nqp_sens_cache = None
        self._nqp_state_t = None
        self._nqp_state_noise = None
        self._nqp_state_values = []
        self._nqp_const_arrays = None
        self._nqp_tiled_cache = None
        self._cache_key_params = {}
        self.tls_noise_enabled = default_config.get('tls_noise_enabled',
                                                    False)
        self.tls_fractional_rms = default_config.get('tls_fractional_rms',
                                                     1e-7)
        self.tls_alpha = default_config.get('tls_alpha', 1.0)
        self.tls_corner_hz = default_config.get('tls_corner_hz', 100.0)
        
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
            # Random pulse tau_decay distribution
            'random_tau_mode': default_config['pulse_random_tau_mode'],
            'random_tau_min': default_config['pulse_random_tau_min'],
            'random_tau_max': default_config['pulse_random_tau_max'],
            'random_tau_logmean': default_config['pulse_random_tau_logmean'],
            'random_tau_logsigma': default_config['pulse_random_tau_logsigma'],
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
    def generate_resonators(self, num_resonances=2, config=None,
                            progress=None):
        """Regenerate the resonator set, atomically w.r.t. the streamer.

        *progress*, if given, is called as ``progress(done, total)``
        before each resonator is built.

        Regeneration empties mr_lekids / mr_complex_resonators /
        base_nqp_values and refills them one resonator at a time.  The
        streamer thread reads those lists under _physics_lock, so without
        holding it here it can sample a half-built set — e.g. 4 lekids
        against 3 complex resonators, which crashed the streamer's noise
        perturbation with a broadcast error.  RLock, so any re-entrant
        s21 call made during generation is fine.
        """
        with self._physics_lock:
            return self._generate_resonators_locked(num_resonances, config,
                                                    progress)

    def _generate_resonators_locked(self, num_resonances=2, config=None,
                                    progress=None):
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
            if getattr(self.mock_crs, '_physics_config', None):
                config = self.mock_crs._physics_config
            else:
                from .config import defaults
                config = defaults()
        
        print('Using config:', {k: v for k, v in config.items() if k in ['num_resonances', 'freq_start', 'freq_end', 'T', 'Popt']})
        
        # Set random seed for reproducible resonator generation.
        # Use separate RandomState for C/Cc variations to avoid pollution from
        # varying iteration counts in the binary search convergence loop.
        # When seed is None (the default from config.py), generate a concrete
        # seed from system entropy and write it back into *config* so that any
        # subsequent save of the config dict (e.g. to session metadata) will
        # reproduce the exact same resonator set.
        seed = config.get('resonator_random_seed', 42)
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31))
            config['resonator_random_seed'] = seed
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
        self.tls_noise_enabled = config.get('tls_noise_enabled', False)
        self.tls_fractional_rms = config.get('tls_fractional_rms', 1e-7)
        self.tls_alpha = config.get('tls_alpha', 1.0)
        self.tls_corner_hz = config.get('tls_corner_hz', 100.0)
        self._tls_generator = None  # rebuilt below once count is known
        self._nqp_sens_cache = None  # depends on resonator params
        self._nqp_state_t = None
        self._nqp_state_noise = None
        self._nqp_state_values = []
        self._nqp_const_arrays = None
        self._nqp_tiled_cache = None
        self._cache_key_params = {}

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
            'random_tau_mode': config.get('pulse_random_tau_mode', 'fixed'),
            'random_tau_min': config.get('pulse_random_tau_min', 5e-4),
            'random_tau_max': config.get('pulse_random_tau_max', 5e-3),
            'random_tau_logmean': config.get('pulse_random_tau_logmean', -6.9),
            'random_tau_logsigma': config.get('pulse_random_tau_logsigma', 0.5),
        })
        
        print(f"Pulse config updated: tau_rise={self.pulse_config['tau_rise']}, tau_decay={self.pulse_config['tau_decay']}, random_tau_mode={self.pulse_config['random_tau_mode']}")

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
            if progress is not None:
                progress(x, num_resonances)
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

        # TLS (1/f) frequency wander — one independent process per
        # resonator, seeded from the resonator seed so mock runs stay
        # reproducible.
        if self.tls_noise_enabled and self.mr_lekids:
            from .tls_noise import TLSNoiseGenerator
            seed = config.get('resonator_random_seed')
            self._tls_generator = TLSNoiseGenerator(
                n_resonators=len(self.mr_lekids),
                fractional_rms=self.tls_fractional_rms,
                alpha=self.tls_alpha,
                corner_hz=self.tls_corner_hz,
                seed=seed,
            )
            print(f"TLS 1/f noise: rms={self.tls_fractional_rms:.2e} "
                  f"df/f, alpha={self.tls_alpha}, "
                  f"corner={self.tls_corner_hz} Hz")
        else:
            self._tls_generator = None

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
                # Random amplitude distribution
                random_amp_mode=config.get('pulse_random_amp_mode', 'fixed'),
                random_amp_min=config.get('pulse_random_amp_min', 1.5),
                random_amp_max=config.get('pulse_random_amp_max', 3.0),
                random_amp_logmean=config.get('pulse_random_amp_logmean', 0.7),
                random_amp_logsigma=config.get('pulse_random_amp_logsigma', 0.3),
                # Random tau_decay distribution
                random_tau_mode=config.get('pulse_random_tau_mode', 'fixed'),
                random_tau_min=config.get('pulse_random_tau_min', 5e-4),
                random_tau_max=config.get('pulse_random_tau_max', 5e-3),
                random_tau_logmean=config.get('pulse_random_tau_logmean', -6.9),
                random_tau_logsigma=config.get('pulse_random_tau_logsigma', 0.5),
            )
        
        self.invalidate_caches()

    def s21_lc_response(self, frequency, amplitude=1.0, pulse_time=None):
        """
        Calculate S21 response with optimized convergence.
        
        Includes:
        - ALL resonators for continuous S21 (no artificial boundaries)
        - Pulse-modified quasiparticle density (if pulses are active)
        - Fresh noisy quasiparticle density each call
        - Physics-based Lk and R from combined nqp
        - OPTIMIZED: Skip convergence for noise-only changes via caching
        
        Parameters
        ----------
        pulse_time : float or None
            If provided, use this time for pulse contribution calculations
            instead of ``self.last_update_time``.  This allows the PFB
            streamer to evaluate S21 at its own simulation time without
            competing with the slow streamer for ``last_update_time``.
        
        THREAD SAFETY: This method is protected by _physics_lock because it 
        updates the shared state (self.mr_lekids) during calculation.
        """
        # Acquire lock to prevent race conditions with other threads (e.g. Streamer vs get_samples)
        # This is critical because update_lekids_for_current modifies shared state.
        with self._physics_lock:
            return self._s21_lc_response_internal(frequency, amplitude, pulse_time)

    def _s21_lc_response_internal(self, frequency, amplitude=1.0, pulse_time=None):
        """Internal implementation of s21_lc_response (assumes lock held).

        Parameters
        ----------
        pulse_time : float or None
            If provided, use this time for pulse contribution calculations
            instead of ``self.last_update_time``.
        """
        import time
        t_start = time.perf_counter()
        
        if not self.mr_lekids:
            return 1.0 + 0j
        
        # Use provided pulse_time or fall back to shared last_update_time
        t_for_pulses = pulse_time if pulse_time is not None else self.last_update_time
        
        # The QP state is computed once per instant and reused by every
        # channel asking at that same timestamp.  The multi-sample (PFB)
        # loop evaluates every (channel, tone) pair at each instant, so
        # without this the pulse sum, the nqp -> Lk/R physics update and
        # the noise draw are all recomputed once per CHANNEL rather than
        # once per sample — 5x wasted work at the default 5 resonators.
        # Sharing across channels at a given instant is also more
        # physically correct: they observe the same resonators at the
        # same moment and should see the same state, not independent
        # noise draws.
        if (self._nqp_state_t is not None
                and t_for_pulses == self._nqp_state_t):
            nqp_noise_frac = self._nqp_state_noise
        else:
            nqp_noise_frac = self._compute_nqp_state(t_for_pulses)
            self._nqp_state_t = t_for_pulses
            self._nqp_state_noise = nqp_noise_frac

        return self._s21_from_current_state(
            frequency, amplitude, t_for_pulses, nqp_noise_frac, t_start)

    def _compute_nqp_state(self, t_for_pulses):
        """Advance QP state (pulses + noise draw) to *t_for_pulses*.

        Writes the resulting Lk/R into ``base_lekid_params`` and returns
        the fractional white-noise draw for this instant (or None).
        """
        # The pulse sum for this one instant, through the same array
        # formula the batch path uses, so the two quantize identically
        # for the convergence-cache key.
        effective_nqp_array = self._batch_nqp(
            np.array([t_for_pulses], dtype=np.float64))[0]

        # White QP noise is applied as a FIRST-ORDER perturbation about
        # the converged operating point, rather than folded into the nqp
        # below and re-converged.  R is proportional to nqp, so this is
        # exact for the dominant term; the Lk contribution is smaller by
        # five orders of magnitude.  See _nqp_sensitivity, and the
        # application site in _s21_lc_response_internal.
        #
        # Folding it in instead would also miss the convergence cache on
        # every call, since it is a fresh draw (measured: 85% hit rate
        # -> 4% at a 10% noise level, a 12x slowdown).
        if self.nqp_noise_enabled and self.nqp_noise_std_factor > 0:
            nqp_noise_frac = np.random.normal(
                0.0, self.nqp_noise_std_factor, len(self.base_nqp_values))
        else:
            nqp_noise_frac = None
        
        # Convert to list for compatibility with existing code
        effective_nqp = effective_nqp_array.tolist()
        
        # Always update R,Lk from physics (fast, preserves IQ behavior)
        self.update_base_params_from_nqp(effective_nqp)
        self._nqp_state_values = effective_nqp
        return nqp_noise_frac

    def _cache_key_gen(self):
        """Generation counter for the cache-key memo.

        Bumped implicitly whenever the resonator set is regenerated or the
        physics config is edited, so stale quantization steps cannot survive
        a reconfigure.
        """
        phys = getattr(self.mock_crs, '_physics_config', {})
        if not isinstance(phys, dict):
            phys = {}
        return (getattr(self, '_resonator_gen', 0),
                len(self.mr_lekids),
                phys.get('cache_freq_step'),
                phys.get('cache_amp_step'),
                phys.get('cache_qp_step'))

    def _cache_keys_for(self, frequency):
        """The cache-key parameters for a tone frequency, memoised per
        frequency and refreshed when the resonators or the config
        change; the memo is bounded."""
        keys = self._cache_key_params.get(frequency)
        if keys is None or keys[0] != self._cache_key_gen():
            keys = ((self._cache_key_gen(),)
                    + self._compute_cache_key_params(frequency))
            if len(self._cache_key_params) > 4096:
                self._cache_key_params.clear()
            self._cache_key_params[frequency] = keys
        return keys

    def _compute_cache_key_params(self, frequency):
        """(nearest_idx, freq_step, amp_step, qp_step) for a tone."""
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

        phys = getattr(self.mock_crs, '_physics_config', {})
        if not isinstance(phys, dict):
            phys = {}

        freq_step = phys.get('cache_freq_step', 0.0001)  # Default 0.0001 Hz
        if not freq_step or freq_step <= 0:
            freq_step = 0.0001

        amp_step = phys.get('cache_amp_step', 1e-8)  # Default 1e-8
        if not amp_step or amp_step <= 0:
            amp_step = 1e-8

        qp_step_fraction = phys.get('cache_qp_step', 0.001)  # Default 0.1%
        if not qp_step_fraction or qp_step_fraction <= 0:
            qp_step_fraction = 0.001

        # Absolute QP step based on median base value
        if self.base_nqp_values:
            base_med = float(np.median(
                np.array(self.base_nqp_values, dtype=float)))
            qp_step = max(1e-300, abs(base_med) * qp_step_fraction)
        else:
            qp_step = 1e-6  # Fallback

        return nearest_idx, freq_step, amp_step, qp_step

    def _s21_from_current_state(self, frequency, amplitude, t_for_pulses,
                                nqp_noise_frac, t_start):
        """Frequency-dependent half: convergence (cached) + S21.

        Assumes the QP state for this instant has already been applied
        by :meth:`_compute_nqp_state`.
        """
        import time

        # === OPTIMIZATION: Check if we need convergence (per-resonator qp-aware key) ===
        # Cache-key inputs are per-call constants: which resonator is
        # nearest to this tone, and the three quantization steps.  Recomputing
        # them (a Python loop over resonators, an np.median, four dict
        # lookups) once per PFB sample cost more than the physics itself, so
        # they are memoised per tone frequency and refreshed only when the
        # resonators or the config actually change.
        keys = self._cache_keys_for(frequency)
        _, nearest_idx, freq_step, amp_step, qp_step = keys

        freq_key = round(frequency / freq_step) * freq_step
        amp_key = round(amplitude / amp_step) * amp_step

        phys = getattr(self.mock_crs, '_physics_config', {})
        if not isinstance(phys, dict):
            phys = {}

        effective_nqp = self._nqp_state_values
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
            phys = getattr(self.mock_crs, '_physics_config', {})
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
        
        # ── White QP noise, as a post-cache perturbation ──────────
        # Linearised about the operating point: a fractional nqp
        # deviation eps maps to fractional changes s_Lk*eps in Lk and
        # s_R*eps in R.  Lk enters the total inductance weighted by
        # alpha_k = Lk/L, so dL/L = alpha_k * s_Lk * eps.
        if nqp_noise_frac is not None:
            s_Lk, s_R = self._nqp_sensitivity()
            # Clamp against every participating array, as the TLS block
            # below does: these are sized from three different lists and
            # a mid-rebuild sample must degrade, not raise.
            m = min(n_relevant, len(nqp_noise_frac), len(s_Lk), len(s_R))
            if m:
                eps = nqp_noise_frac[:m]
                alpha_k = np.array(
                    [self.mr_lekids[i].alpha_k for i in range(m)])
                L_subset[:m] = L_subset[:m] * (
                    1.0 + alpha_k * s_Lk[:m] * eps)
                R_subset[:m] = np.maximum(
                    0.0, R_subset[:m] * (1.0 + s_R[:m] * eps))

        # ── TLS (1/f) frequency wander ────────────────────────────
        # Applied HERE, after any convergence-cache restore, because
        # the cache quantizes QP and restores L/Lk/R verbatim on a hit —
        # a wander injected through nqp would be quantized away.
        # TLS is a surface-dielectric effect, so capacitance is the
        # faithful knob: df/f = -0.5 * dC/C.
        if self._tls_generator is not None:
            y = self._tls_generator.value_at(t_for_pulses)
            n = min(n_relevant, len(y))
            C_subset[:n] = C_subset[:n] * (1.0 - 2.0 * y[:n])

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

    def s21_sweep(self, frequencies, amplitude):
        """Noise-free |S21| over *frequencies*, the state re-converged at
        each point as it is when the tone actually sits there.

        The dip search wants thousands of points.  Through s21_lc_response
        each one pays the whole single-point path (lock, cache lookup,
        list rebuilds), eight times the two kernels it comes down to; and
        the state cannot be converged once for the grid, since a frozen
        state shows a far deeper dip a few linewidths off that moves away
        as soon as the tone follows it.  Warm-started from the previous
        point, convergence takes a few iterations.  The lekids keep the
        Lk/R/L they had; the QP-state memo and the parameter arrays are
        refreshed the way any single-point call refreshes them.
        """
        with self._physics_lock:
            if not self.mr_lekids:
                return np.ones(len(frequencies))
            # The base Lk/R the single-point path converges from are the
            # ones the QP state for this instant installs, not the
            # generation-time values a fresh model still holds.
            t = self.last_update_time
            if self._nqp_state_t is None or t != self._nqp_state_t:
                self._nqp_state_noise = self._compute_nqp_state(t)
                self._nqp_state_t = t
            self._extract_param_arrays()
            n = len(self.mr_lekids)
            base_Lk = np.array([self.base_lekid_params[i]['Lk'] for i in range(n)])
            base_Lg = np.array([self.base_lekid_params[i]['Lg'] for i in range(n)])
            L, R = self.L_array.copy(), self.R_array.copy()
            C, Cc = self.C_array, self.Cc_array
            k0 = self.mr_lekids[0]
            tolerance = self.mock_crs._physics_config.get(
                'convergence_tolerance', 1e-9)
            out = np.empty(len(frequencies))
            for i, f in enumerate(frequencies):
                f = float(f)
                L, R, _, _ = jit_physics.converged_lekid_parameters(
                    f, amplitude, L, R, C, Cc, base_Lk, base_Lg,
                    self.L_junk_array, k0.input_atten_dB, complex(k0.ZLNA),
                    self.Istar, tolerance, 500, damp=0.1)
                out[i] = abs(jit_physics.compute_s21_parallel(
                    fc=f, Vin=amplitude, L_array=L, C_array=C, R_array=R,
                    Cc_array=Cc, ZLNA=complex(k0.ZLNA), GLNA=k0.GLNA,
                    input_atten_dB=k0.input_atten_dB,
                    system_termination=k0.system_termination))
            return out

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

    def _nqp_sensitivity(self):
        """Fractional response of (Lk, R) to a fractional nqp change.

        Computed once per resonator configuration by finite-differencing
        the same physics kernel the model uses, then reused for every
        sample — the operating point moves only slowly, and the white
        noise it serves is a small perturbation about it.

        Returns ``(s_Lk, s_R)`` where ``dLk/Lk = s_Lk * dnqp/nqp``.
        """
        # Keyed on the resonator count, not merely invalidated at the top
        # of generate_resonators: this is computed lazily, so a call that
        # lands mid-rebuild would otherwise cache an array sized for the
        # partial set and keep serving it afterwards.
        n = len(self.mr_complex_resonators)
        if self._nqp_sens_cache is not None and self._nqp_sens_cache[0] == n:
            return self._nqp_sens_cache[1:]

        base = np.array(self.base_nqp_values[:n], dtype=np.float64)
        cr0 = self.mr_complex_resonators[0]
        common = (
            np.array([cr.readout_f for cr in self.mr_complex_resonators],
                     dtype=np.float64),
            np.full(n, cr0.T), np.full(n, cr0.Delta0),
            np.full(n, cr0.N0), np.full(n, cr0.sigmaN),
            np.full(n, cr0.thickness), np.full(n, cr0.width),
            np.full(n, cr0.length), np.full(n, cr0.R_spoiler),
        )
        delta = 1e-3
        R0, Lk0 = jit_physics.vectorized_update_params_from_nqp(
            base, *common)
        R1, Lk1 = jit_physics.vectorized_update_params_from_nqp(
            base * (1.0 + delta), *common)
        with np.errstate(divide="ignore", invalid="ignore"):
            s_Lk = np.where(Lk0 != 0, (Lk1 - Lk0) / (Lk0 * delta), 0.0)
            s_R = np.where(R0 != 0, (R1 - R0) / (R0 * delta), 0.0)
        self._nqp_sens_cache = (n, np.nan_to_num(s_Lk),
                                np.nan_to_num(s_R))
        return self._nqp_sens_cache[1:]

    def update_base_params_from_nqp(self, noisy_nqp_values):
        """
        Update base Lk and R values using physics calculations.
        
        Uses JIT-compiled parallel vectorized calculations for 15-25x speedup.
        
        Parameters
        ----------
        noisy_nqp_values : list
            List of noisy nqp values, one per resonator
        """
        # The material/geometry arrays below are fixed for a given
        # resonator set, but this runs once per PFB sample — rebuilding
        # eight np.full() arrays each time cost more than the physics.
        n = len(self.mr_complex_resonators)
        const = self._nqp_const_arrays
        if const is None or const[0] != n:
            cr0 = self.mr_complex_resonators[0]
            const = (
                n,
                np.array([cr.readout_f for cr in self.mr_complex_resonators],
                         dtype=np.float64),
                np.full(n, cr0.T, dtype=np.float64),
                np.full(n, cr0.Delta0, dtype=np.float64),
                np.full(n, cr0.N0, dtype=np.float64),
                np.full(n, cr0.sigmaN, dtype=np.float64),
                np.full(n, cr0.thickness, dtype=np.float64),
                np.full(n, cr0.width, dtype=np.float64),
                np.full(n, cr0.length, dtype=np.float64),
                np.full(n, cr0.R_spoiler, dtype=np.float64),
            )
            self._nqp_const_arrays = const
        nqp_array = np.asarray(noisy_nqp_values, dtype=np.float64)

        # Call JIT-compiled function - computes ALL resonators in parallel
        R_array, Lk_array = jit_physics.vectorized_update_params_from_nqp(
            nqp_array, *const[1:]
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
        """Advance every resonator's QP density to *current_time*
        under the active pulses; a monotonic ratchet, so an earlier
        time is a no-op."""
        if current_time <= self.last_update_time:
            return
        self.advance_pulses_to(current_time, 1, 0.0)

    def advance_pulses_to(self, t_to, n_steps, dt):
        """The bookkeeping of update_qp_densities_for_time, with the
        trigger checks made at *n_steps* instants *dt* apart ending at
        *t_to* rather than at one.

        Trigger checks at every grid instant, then one evaluation
        across the frame: a pulse contributes nothing before its start,
        so the frame's samples equal a per-instant evaluation.
        """
        t_from = t_to - (n_steps - 1) * dt
        for k in range(n_steps):
            t = t_from + k * dt
            if t <= self.last_update_time:
                continue
            step = t - self.last_update_time
            self.last_update_time = t
            self._check_trigger_pulses(t, step)
        self._nqp_state_t = None
        if self.mr_lekids:
            nqp = self._batch_nqp(np.array([t_to], dtype=np.float64))[0]
            self.update_base_params_from_nqp(nqp.tolist())
            self.invalidate_caches()
        # Keep every event still contributing at the START of the span:
        # the span's samples are evaluated after this, and a block of
        # slow frames is many decay times long.
        self.pulse_events = [p for p in self.pulse_events
                             if t_from - p['start_time']
                             < p['tau_rise'] + p['tau_decay'] * 15]

    def warm_pulse_caches(self, module, sample_rate, block_len,
                          progress=None):
        """Run one pulse on every resonator through the block path the
        streamer uses, so the first real pulse does not stall the stream
        on cold convergence caches and kernels.  Time and pulse state
        are put back; the caches are what remain.  The configured pulse
        parameters decide which keys are warmed, so a pulse amplitude or
        decay changed later warms itself.

        This is the first pulse's whole cost, paid here instead: one
        convergence of the array per tone per distinct quasiparticle
        key along the decay, 36,500 of them (28 s) at 100 tones with
        the default 0.1 s decay.  *progress* is called per block as
        ``progress(done, total)``.
        """
        if not self.mr_lekids:
            return
        with self._physics_lock:
            saved = (self.last_update_time, dict(self.pulse_config),
                     list(self.pulse_events), dict(self.last_pulse_time))
            try:
                self.pulse_config['mode'] = 'periodic'
                self.pulse_config['resonators'] = 'all'
                self.pulse_events = []
                self.last_pulse_time = {}
                dt = 1.0 / sample_rate
                span = (self.pulse_config['tau_rise']
                        + 15 * self.pulse_config['tau_decay'])
                n_blocks = int(np.ceil(span / (block_len * dt))) + 1
                t = self.last_update_time
                for k in range(n_blocks):
                    if progress is not None:
                        progress(k, n_blocks)
                    self.advance_pulses_to(t + (block_len - 1) * dt,
                                           block_len, dt)
                    self.calculate_module_response_coupled(
                        module, num_samples=block_len,
                        sample_rate=sample_rate, start_time=t,
                        pulse_time=t)
                    t += block_len * dt
            finally:
                (self.last_update_time, self.pulse_config,
                 self.pulse_events, self.last_pulse_time) = saved
                self._nqp_state_t = None

    def _sample_random_pulse_amplitude(self):
        """Sample a pulse amplitude based on configured distribution.
        
        Works in all pulse modes (periodic, random, manual).
        Returns the config default when random_amp_mode is 'fixed'.
        """
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

    def _sample_random_pulse_tau(self):
        """Sample a pulse tau_decay based on configured distribution.
        
        In MKID physics, tau_rise is quasi-instantaneous (~µs) and fixed,
        while tau_decay (QP recombination) varies with QP density, temperature,
        material defects, etc.  This method randomises tau_decay only.
        
        Works in all pulse modes (periodic, random, manual).
        Returns the config default when random_tau_mode is 'fixed'.
        """
        mode = self.pulse_config.get('random_tau_mode', 'fixed')
        if mode == 'uniform':
            tmin = float(self.pulse_config.get('random_tau_min', 5e-4))
            tmax = float(self.pulse_config.get('random_tau_max', 5e-3))
            tau = np.random.uniform(tmin, tmax)
        elif mode == 'lognormal':
            mu = float(self.pulse_config.get('random_tau_logmean', -6.9))
            sigma = float(self.pulse_config.get('random_tau_logsigma', 0.5))
            tau = np.random.lognormal(mean=mu, sigma=sigma)
        else:
            tau = float(self.pulse_config.get('tau_decay', 0.1))
        # tau_decay must be strictly positive
        return max(1e-9, tau)

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
                    amp = self._sample_random_pulse_amplitude()
                    tau = self._sample_random_pulse_tau()
                    self.add_pulse_event(res_idx, current_time, amplitude=amp, tau_decay=tau)
                    self.last_pulse_time[res_idx] = current_time
        
        elif mode == 'random':
            # Random pulses based on probability per timestep
            prob = self.pulse_config['probability']
            for res_idx in target_resonators:
                if np.random.random() < prob * dt:
                    amp = self._sample_random_pulse_amplitude()
                    tau = self._sample_random_pulse_tau()
                    self.add_pulse_event(res_idx, current_time, amplitude=amp, tau_decay=tau)
                    self.last_pulse_time[res_idx] = current_time
    
    def add_pulse_event(self, resonator_index, start_time, amplitude=None, tau_decay=None):
        """Manually add a pulse event to a specific resonator.
        
        Parameters
        ----------
        resonator_index : int
            Index of the resonator (0-based)
        start_time : float
            Time when the pulse starts (seconds)
        amplitude : float, optional
            Maximum QP density increase. If None, uses config default.
        tau_decay : float, optional
            Decay time constant in seconds. If None, uses config default.
            Typically sampled from _sample_random_pulse_tau() by the caller.
        """
        if resonator_index >= len(self.mr_lekids):
            print(f"Warning: Resonator index {resonator_index} out of range")
            return
        
        pulse = {
            'resonator_index': resonator_index,
            'start_time': start_time,
            'amplitude': amplitude if amplitude is not None else self.pulse_config['amplitude'],
            'tau_rise': self.pulse_config['tau_rise'],
            'tau_decay': tau_decay if tau_decay is not None else self.pulse_config['tau_decay'],
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
    
    def calculate_module_response_coupled(self, module, num_samples=1, sample_rate=None, start_time=0, pulse_time=None):
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
                s21_complex = self.s21_lc_response(total_freq, amp, pulse_time=pulse_time)
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
        
        tone_freqs = np.array(active_tone_freqs)
        tone_amps = np.array(active_tone_amps)
        n_obs = len(obs_channels)
        n_tones = len(active_tone_freqs)

        # Which tones each observer sees: those within the CIC bandwidth
        # of its own frequency, with the droop at that offset.
        obs_idx, tone_idx, diff = self._coupled_pairs(
            obs_freqs, tone_freqs, bandwidth)
        cic = np.array([self._get_cached_cic_response(d, dec_stage)
                        for d in diff])
        pairs = (obs_idx, tone_idx, diff, cic)

        if num_samples == 1:
            beat = np.exp(2j * np.pi * diff * start_time)
            signals = self._mix_pairs(pairs, tone_amps, beat, n_obs)[:, 0]
            responses = {}
            for i, ch in enumerate(obs_channels):
                responses[ch] = complex(signals[i])

        else:
            t = start_time + np.arange(num_samples) / sample_rate

            # Per-tone invariants, hoisted out of the sample loop.
            tone_mag = np.abs(np.asarray(active_tone_amps))
            tone_phase = np.exp(1j * np.angle(np.asarray(active_tone_amps)))

            observed = np.zeros(n_tones, dtype=bool)
            observed[tone_idx] = True

            phys = getattr(self.mock_crs, '_physics_config', {}) or {}
            mode = phys.get('physics_batch_mode', 'hoisted')
            with self._physics_lock:
                if mode == 'reference':
                    signals = self._batch_response_reference(
                        active_tone_freqs, tone_mag, tone_phase, pairs,
                        observed, t, pulse_time, sample_rate, n_obs, n_tones)
                else:
                    signals = self._batch_response_hoisted(
                        active_tone_freqs, tone_mag, tone_phase, pairs,
                        observed, t, pulse_time, sample_rate, n_obs, n_tones)

            responses = {}
            for i, ch in enumerate(obs_channels):
                responses[ch] = signals[i]

        return responses

    @staticmethod
    def _coupled_pairs(obs_freqs, tone_freqs, bandwidth):
        """(obs_idx, tone_idx, freq_diff) for every observer/tone pair
        within *bandwidth*.

        Found through the sorted tones rather than the full grid: with
        a module fully configured the grid has a million entries and
        only the diagonal survives.  The window is widened by a relative 1e-9
        and the grid's own predicate applied to the candidates, so the
        pair set is the grid's exactly.
        """
        obs = np.asarray(obs_freqs, dtype=np.float64)
        tones = np.asarray(tone_freqs, dtype=np.float64)
        order = np.argsort(tones, kind="stable")
        sorted_tones = tones[order]
        slack = bandwidth * (1.0 + 1e-9) + 1e-9
        lo = np.searchsorted(sorted_tones, obs - slack, side="left")
        hi = np.searchsorted(sorted_tones, obs + slack, side="right")
        counts = hi - lo
        total = int(counts.sum())
        obs_idx = np.repeat(np.arange(len(obs)), counts)
        first = np.repeat(np.cumsum(counts) - counts, counts)
        pos = np.repeat(lo, counts) + (np.arange(total) - first)
        tone_idx = order[pos]
        diff = tones[tone_idx] - obs[obs_idx]
        keep = np.abs(diff) <= bandwidth
        return obs_idx[keep], tone_idx[keep], diff[keep]

    @staticmethod
    def _mix_pairs(pairs, tone_amp, beat, n_obs):
        """Each observer's signal: the sum over its coupled tones of
        amplitude x CIC droop x beat factor, as (n_obs, N).

        *tone_amp* is (n_tones,) for one instant or (n_tones, N);
        *beat* is per pair, (n_pairs,) or (n_pairs, N).
        """
        obs_idx, tone_idx, _diff, cic = pairs
        amp = np.asarray(tone_amp)
        if amp.ndim == 1:
            amp = amp[:, None]
        beat = np.asarray(beat)
        if beat.ndim == 1:
            beat = beat[:, None]
        contrib = amp[tone_idx] * cic[:, None] * beat
        signals = np.zeros((n_obs, contrib.shape[1]), dtype=complex)
        np.add.at(signals, obs_idx, contrib)
        return signals

    def _batch_response_reference(self, active_tone_freqs, tone_mag,
                                  tone_phase, pairs, observed, t,
                                  pulse_time, sample_rate, n_obs, n_tones):
        """The per-sample loop: S21 for every (sample, tone), through the
        same path the slow stream uses one sample at a time.  Kept as the
        reference the hoisted path is checked against, and selectable
        with physics_batch_mode="reference".  Assumes _physics_lock held.

        Per-sample pulse_time makes the QP density (and so the pulse
        shape) evolve across the batch rather than freeze at its start.
        """
        num_samples = len(t)
        signals = np.zeros((n_obs, num_samples), dtype=complex)
        diff = pairs[2]
        # Samples OUTER, channels INNER: every tone at a given instant
        # then shares one QP-state evaluation (the memo in
        # _s21_lc_response_internal); the other way round the time
        # changes on every call and the memo never hits.
        for sample_idx in range(num_samples):
            if pulse_time is not None:
                sample_pulse_time = pulse_time + sample_idx / sample_rate
            else:
                sample_pulse_time = None
            # S21 for a tone at a given instant does not depend on
            # which channel observes it — evaluate once per tone.
            s21_by_tone = np.zeros(n_tones, dtype=complex)
            for j in range(n_tones):
                if tone_mag[j] > 0 and observed[j]:
                    s21_by_tone[j] = self._s21_lc_response_internal(
                        active_tone_freqs[j], tone_mag[j],
                        pulse_time=sample_pulse_time)
            tone_amp_fresh = tone_mag * s21_by_tone * tone_phase
            beat = np.where(np.abs(diff) < 0.1, 1.0,
                            np.exp(2j * np.pi * diff * t[sample_idx]))
            signals[:, sample_idx] = self._mix_pairs(
                pairs, tone_amp_fresh, beat, n_obs)[:, 0]
        return signals

    def _batch_nqp(self, t_arr):
        """nqp per (instant, resonator) for the current pulse set: the
        pulse sum of _compute_nqp_state over an array of instants."""
        base = np.asarray(self.base_nqp_values, dtype=np.float64)
        eff = np.tile(base, (len(t_arr), 1))
        for pulse in self.pulse_events:
            i = pulse['resonator_index']
            if i >= len(base):
                continue
            dt = t_arr - pulse['start_time']
            rise = 1.0 - np.exp(-np.maximum(dt, 0.0) / pulse['tau_rise'])
            decay = np.exp(-np.maximum(dt - pulse['tau_rise'], 0.0)
                           / pulse['tau_decay'])
            tf = np.where(dt < pulse['tau_rise'], rise, decay)
            tf = np.where(dt >= 0, tf, 0.0)
            eff[:, i] += (pulse['amplitude'] - 1.0) * tf * base[i]
        return np.maximum(0, eff)

    def _nqp_const_tiled(self, S):
        """The per-resonator material arrays repeated S times, so one
        kernel dispatch covers S instants."""
        n = len(self.mr_complex_resonators)
        if self._nqp_const_arrays is None:
            self.update_base_params_from_nqp(self.base_nqp_values[:n])
        const = self._nqp_const_arrays
        cached = self._nqp_tiled_cache
        if cached is None or cached[0] != (S, n) or cached[1] is not const:
            tiled = tuple(np.tile(a, S) for a in const[1:])
            cached = ((S, n), const, tiled)
            self._nqp_tiled_cache = cached
        return cached[2]

    def _batch_response_hoisted(self, active_tone_freqs, tone_mag,
                                tone_phase, pairs, observed, t, pulse_time,
                                sample_rate, n_obs, n_tones):
        """The reference loop with everything constant across the batch
        hoisted out of it.  Same arithmetic, same noise draws in the same
        order, same convergence-cache decisions and the same end state.
        Assumes _physics_lock held.

        With pulse_time None every sample shares one instant (the
        reference loop's memo makes them share one QP state and one
        noise draw), so the state is evaluated once and broadcast.
        """
        num_samples = len(t)
        n_res = len(self.mr_lekids)
        if not self.mr_lekids:
            return np.zeros((n_obs, num_samples), dtype=complex)
        if pulse_time is not None:
            t_states = pulse_time + np.arange(num_samples) / sample_rate
        else:
            t_states = np.array([self.last_update_time], dtype=np.float64)
        S = len(t_states)
        row = (np.arange(num_samples) if pulse_time is not None
               else np.zeros(num_samples, dtype=np.int64))

        # ── QP state per instant: pulses, noise, Lk/R, TLS ──────────
        reuse0 = (self._nqp_state_t is not None
                  and t_states[0] == self._nqp_state_t)
        nqp = self._batch_nqp(t_states)                       # (S, n_res)
        eps = None
        if self.nqp_noise_enabled and self.nqp_noise_std_factor > 0:
            eps = np.empty((S, n_res))
            first = 0
            if reuse0 and self._nqp_state_noise is not None:
                eps[0] = self._nqp_state_noise
                first = 1
            if S > first:
                eps[first:] = np.random.normal(
                    0.0, self.nqp_noise_std_factor, (S - first, n_res))
        n_cr = len(self.mr_complex_resonators)
        R_nqp, Lk_nqp = jit_physics.vectorized_update_params_from_nqp(
            nqp[:, :n_cr].ravel(), *self._nqp_const_tiled(S))
        R_nqp = R_nqp.reshape(S, n_cr)
        Lk_nqp = Lk_nqp.reshape(S, n_cr)
        y = (self._tls_generator.values_at(t_states)
             if self._tls_generator is not None else None)

        gen = getattr(self, '_resonator_gen', 0)
        phys = getattr(self.mock_crs, '_physics_config', {})
        if not isinstance(phys, dict):
            phys = {}
        log_enabled = phys.get('log_cache_decisions', False)
        log_every = max(1, int(phys.get('cache_log_interval', 100)))
        lekid0 = self.mr_lekids[0]
        C_const = np.array([lk.C for lk in self.mr_lekids])
        Cc_const = np.array([lk.Cc for lk in self.mr_lekids])
        if eps is not None:
            s_Lk, s_R = self._nqp_sensitivity()

        def set_base(s):
            for i in range(n_cr):
                self.base_lekid_params[i]['R'] = R_nqp[s, i]
                self.base_lekid_params[i]['Lk'] = Lk_nqp[s, i]

        def restore(state):
            L_v, R_v, Lk_v = state
            for i in range(n_res):
                lek = self.mr_lekids[i]
                if i < len(Lk_v):
                    lek.Lk = Lk_v[i]
                if i < len(R_v):
                    lek.R = R_v[i]
                if i < len(L_v):
                    lek.L = L_v[i]
                    lek.alpha_k = lek.Lk / lek.L

        # ── Per tone: cache-key parameters and the runs of one key ────
        # The key is the quantized QP density; within a run of one key
        # the reference converges at the first sample and hits that
        # entry for the rest, so a run is one lookup.
        tones = []
        for j in range(n_tones):
            if not (tone_mag[j] > 0 and observed[j]):
                continue
            frequency = active_tone_freqs[j]
            amplitude = tone_mag[j]
            keys = self._cache_keys_for(frequency)
            _, nearest_idx, freq_step, amp_step, qp_step = keys
            qp_col = (nqp[:, nearest_idx] if 0 <= nearest_idx < n_res
                      else np.zeros(S))
            qp_keys = np.round(qp_col / qp_step) * qp_step
            starts = np.flatnonzero(np.r_[True, qp_keys[1:] != qp_keys[:-1]])
            tones.append({
                'j': j, 'frequency': frequency, 'amplitude': amplitude,
                'nearest_idx': nearest_idx,
                'freq_key': round(frequency / freq_step) * freq_step,
                'amp_key': round(amplitude / amp_step) * amp_step,
                'qp_keys': qp_keys, 'starts': starts, 'states': []})

        def state_at(k, s):
            """The state tone k's run covering sample s holds."""
            tone = tones[k]
            r = int(np.searchsorted(tone['starts'], s, side='right')) - 1
            return tone['states'][r]

        # ── Convergence decisions, one per run, in the reference order ──
        # Samples outer, tones inner: each convergence starts from the
        # state the step before it left in the lekids, which is the
        # previous tone's run at this sample or, for the first tone, the
        # last tone's run at the previous sample.  The cache is a FIFO
        # of bounded size, so an insertion can evict the entry a run in
        # progress is hitting; the reference then converges again at
        # that tone's next step, and so does this, by splitting the run.
        events = [(int(st), k) for k, tone in enumerate(tones)
                  for st in tone['starts']]
        heapq.heapify(events)
        for tone in tones:
            tone['starts'] = []
        live = {}                       # cache key -> tone index of the run using it
        max_size = phys.get('convergence_cache_max_size',
                            self._convergence_cache_max_size)
        if isinstance(max_size, int) and max_size > 0:
            self._convergence_cache_max_size = max_size
        last_reason = self._convergence_stats.get('last_reason')
        misses = set()
        while events:
            s, k = heapq.heappop(events)
            tone = tones[k]
            if tone['starts'] and tone['starts'][-1] == s:
                continue                # a split landing on a run start
            qp_key = float(tone['qp_keys'][s])
            cache_key = (tone['nearest_idx'], tone['freq_key'],
                         tone['amp_key'], qp_key)
            cached = self._convergence_cache.get(cache_key)
            hit = (cached is not None and cached.get('gen') == gen
                   and cached.get('lekid_count') == n_res)
            if hit:
                state = (cached['L_values'], cached['R_values'],
                         cached['Lk_values'])
                last_reason = 'hit'
            else:
                if k > 0:
                    restore(state_at(k - 1, s))
                elif s > 0:
                    restore(state_at(len(tones) - 1, s - 1))
                set_base(s)
                self.update_lekids_for_current(tone['frequency'],
                                               tone['amplitude'])
                state = ([lk.L for lk in self.mr_lekids],
                         [lk.R for lk in self.mr_lekids],
                         [lk.Lk for lk in self.mr_lekids])
                self._convergence_cache[cache_key] = {
                    'Lk_values': state[2], 'R_values': state[1],
                    'L_values': state[0], 'frequency': tone['frequency'],
                    'amplitude': tone['amplitude'], 'qp_key': qp_key,
                    'nearest_idx': tone['nearest_idx'], 'gen': gen,
                    'lekid_count': n_res}
                self._convergence_stats['full'] += 1
                misses.add((s, k))
                last_reason = ('miss' if cached is None
                               else 'gen_changed' if cached.get('gen') != gen
                               else 'count_changed')
                if len(self._convergence_cache) > self._convergence_cache_max_size:
                    evicted = next(iter(self._convergence_cache))
                    del self._convergence_cache[evicted]
                    k_ev = live.pop(evicted, None)
                    if k_ev is not None:
                        # That tone looks the key up again at its next
                        # step: this sample if it comes later in the
                        # tone order, otherwise the next sample.
                        s_ev = s if k_ev > k else s + 1
                        if s_ev < S:
                            heapq.heappush(events, (s_ev, k_ev))
            tone['starts'].append(s)
            tone['states'].append(state)
            live[cache_key] = k
        self._convergence_stats['last_reason'] = last_reason

        # The per-step statistics the reference keeps, from the runs: a
        # step is one (sample, tone), a run's first step is a miss when
        # it converged, every other step is a hit.  Samples sharing one
        # instant are separate steps of which only the first can miss.
        n_steps = num_samples * len(tones)
        self._convergence_stats['skipped'] += n_steps - len(misses)
        first = np.full(S, num_samples)
        np.minimum.at(first, row, np.arange(num_samples))
        # The last hundred steps' hit/miss, for the statistics.
        n_live = len(tones)
        recent = []
        for step in range(max(0, n_steps - 100), n_steps):
            n, k = divmod(step, n_live)
            recent.append(not (n == first[row[n]]
                               and (int(row[n]), k) in misses))
        self._recent_cache_results.extend(recent)
        del self._recent_cache_results[:-100]
        before = self._stats_counter // log_every
        self._stats_counter += n_steps
        if log_enabled and self._stats_counter // log_every > before:
            hits = sum(self._recent_cache_results)
            total = len(self._recent_cache_results)
            print(f"[Cache Stats] Last {total} calls: "
                  f"{hits / total * 100:.1f}% cache hits "
                  f"({hits} hits, {total - hits} misses)")

        # ── Per tone, over all instants: noise and TLS perturbations, S21 ──
        s21 = np.zeros((n_tones, S), dtype=complex)
        for tone in tones:
            L_s = np.empty((S, n_res))
            R_s = np.empty((S, n_res))
            Lk_s = np.empty((S, n_res))
            starts = tone['starts']
            for r, state in enumerate(tone['states']):
                sl = slice(int(starts[r]),
                           int(starts[r + 1]) if r + 1 < len(starts) else S)
                L_s[sl] = state[0][:n_res]
                R_s[sl] = state[1][:n_res]
                Lk_s[sl] = state[2][:n_res]
            if eps is not None:
                m = min(n_res, eps.shape[1], len(s_Lk), len(s_R))
                if m:
                    alpha_k = Lk_s[:, :m] / L_s[:, :m]
                    L_s[:, :m] = L_s[:, :m] * (
                        1.0 + alpha_k * s_Lk[None, :m] * eps[:, :m])
                    R_s[:, :m] = np.maximum(
                        0.0, R_s[:, :m] * (1.0 + s_R[None, :m] * eps[:, :m]))
            C_s = np.tile(C_const, (S, 1))
            if y is not None:
                n = min(n_res, y.shape[1])
                C_s[:, :n] = C_s[:, :n] * (1.0 - 2.0 * y[:, :n])
            s21[tone['j']] = jit_physics.compute_s21_batch(
                float(tone['frequency']), float(tone['amplitude']),
                L_s, C_s, R_s, Cc_const, complex(lekid0.ZLNA), lekid0.GLNA,
                lekid0.input_atten_dB, lekid0.system_termination)

        # ── End state: what the reference leaves after its last sample ──
        if tones:
            restore(state_at(len(tones) - 1, S - 1))
        set_base(S - 1)
        self._nqp_state_t = float(t_states[-1])
        self._nqp_state_noise = (eps[-1] if eps is not None else None)
        self._nqp_state_values = nqp[-1].tolist()

        # ── Mix: tones into observers, per sample ────────────────
        tone_amp = (tone_mag[:, None] * s21 * tone_phase[:, None])[:, row]
        diff = pairs[2]
        beat = np.where(np.abs(diff)[:, None] < 0.1, 1.0,
                        np.exp(2j * np.pi * diff[:, None] * t[None, :]))
        return self._mix_pairs(pairs, tone_amp, beat, n_obs)

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
