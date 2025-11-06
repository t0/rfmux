"""
Mock CRS Device - Resonator Physics Model.
Encapsulates the logic for resonator physics simulation, including S21 response.
"""
import numpy as np
from . import mock_constants as const

# Import from mr_resonator subpackage using relative imports
from ..mr_resonator.mr_complex_resonator import MR_complex_resonator
from ..mr_resonator.mr_lekid import MR_LEKID

# Import JIT-compiled physics functions (numba is required)
from ..mr_resonator import jit_physics
print("[Physics] JIT-compiled physics loaded successfully")


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
        
        # Store base nqp values computed from physics
        self.base_nqp_values = []    # Base quasiparticle density for each resonator
        
        # Noise configuration
        self.nqp_noise_enabled = True
        self.nqp_noise_std_factor = 0.001  # Default 0.1% noise (realistic for detectors)
        
        # Current effects (affects Lk only, applied after physics-based base params)
        self.lk_current_factors = []  # Lk_total = Lk_base * lk_current_factor
        
        # Physical constants
        self.Istar = 5e-3  # Characteristic current [A]
        
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
        
        # Pulse event tracking for time-dependent QP density
        self.pulse_events = []  # List of active pulses
        self.pulse_config = {
            'mode': 'none',      # 'periodic', 'random', 'manual', or 'none'
            'period': 10.0,      # seconds (for periodic mode)
            'probability': 0.001, # per-timestep probability (for random mode)
            'tau_rise': 1e-6,    # rise time constant (seconds)
            'tau_decay': 1e-1,   # decay time constant (seconds)
            'amplitude': 10.0,    # multiplicative factor relative to base_nqp (2.0 = double the base nqp)
            'resonators': 'all', # 'all' or list of resonator indices
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
        
        # Per-operating-point convergence cache
        # Key: (rounded_freq, rounded_amp), Value: dict with factors and state
        self._convergence_cache = {}
        self._convergence_cache_max_size = 100  # Limit cache size
        
        # Statistics tracking
        self._convergence_stats = {
            'full': 0,
            'skipped': 0,
            'last_reason': None
        }
        
        # Tolerance settings for convergence optimization
        self._tolerance_config = {
            'cache_freq_tolerance': 1e3,    # Hz - how closely frequencies must match 
            'cache_amp_tolerance': 1e-6,    # Fractional - how closely amplitudes must match
            'qp_change_threshold': 0.01,    # 1% - fractional QP change to trigger recalc
            'power_change_threshold': 0.001, # 0.1% - fractional power change to trigger recalc
        }

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
    
    def _get_relevant_resonators(self, frequency, linewidth_threshold=10):
        """
        Get indices of resonators that contribute significantly to S21 at given frequency.
        
        A resonator's S21 contribution falls off as (f0/Q)^2 / (f - f0)^2 far from resonance.
        We include resonators within linewidth_threshold linewidths of the probe frequency.
        
        For linewidth_threshold = 10:
        - With typical coupling k=0.5-1.0, contributes <1% to S21 beyond this distance
        
        Parameters
        ----------
        frequency : float
            Probe frequency in Hz
        linewidth_threshold : float
            Number of linewidths beyond which resonators are excluded
            
        Returns
        -------
        list
            Indices of resonators that should be included in S21 calculation
        """
        relevant_indices = []
        
        for i in range(len(self.resonator_frequencies)):
            res_freq = self.resonator_frequencies[i]
            linewidth = self.resonator_linewidths[i]
            
            # Check if resonator is within threshold
            freq_diff = abs(frequency - res_freq)
            if freq_diff <= linewidth_threshold * linewidth:
                relevant_indices.append(i)
        
        return relevant_indices
    
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
        
        # Set random seed for reproducible resonator generation
        seed = config.get('resonator_random_seed', 42)
        np.random.seed(seed)
        
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
        
        # Configure noise parameters from config
        # Uses defaults from mock_crs_helper.py if not specified
        self.nqp_noise_enabled = config.get('nqp_noise_enabled', True)
        self.nqp_noise_std_factor = config.get('nqp_noise_std_factor', 0.001)  # Default 0.1% noise if not in config
        
        # Update tolerance settings from config (keep existing if not specified or None)
        self._tolerance_config['cache_freq_tolerance'] = config.get('cache_freq_tolerance', self._tolerance_config['cache_freq_tolerance'])
        self._tolerance_config['cache_amp_tolerance'] = config.get('cache_amp_tolerance', self._tolerance_config['cache_amp_tolerance'])
        self._tolerance_config['qp_change_threshold'] = config.get('qp_change_threshold', self._tolerance_config['qp_change_threshold'])
        self._tolerance_config['power_change_threshold'] = config.get('power_change_threshold', self._tolerance_config['power_change_threshold'])
        
        print(f"Tolerance settings: freq={self._tolerance_config['cache_freq_tolerance']} Hz, "
              f"amp={self._tolerance_config['cache_amp_tolerance']}, "
              f"QP threshold={self._tolerance_config['qp_change_threshold']*100:.1f}%")

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
                
                # Compute and store base nqp using calc_nqp() method
                base_nqp = complex_res.calc_nqp()
                self.base_nqp_values.append(base_nqp)
                print(f"  Base nqp: {base_nqp:.2e}")
                
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
                
                # Pre-calculate and store Q value
                try:
                    Q_value = lekid.compute_Qi()
                    print(f"  Q value: {Q_value:.0f}")
                except:
                    Q_value = 1000  # Fallback Q if calculation fails
                    print(f"  Q value: {Q_value:.0f} (fallback)")
                self.resonator_q_values.append(Q_value)
                
                # Pre-compute and store linewidth
                linewidth = actual_freq / Q_value
                self.resonator_linewidths.append(linewidth)
                print(f"  Linewidth: {linewidth/1e3:.2f} kHz")
                
            except Exception as e:
                print(f"Warning: Failed to create resonator {x}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with other resonators
                continue
        
        print(f"Successfully created {len(self.mr_lekids)} persistent LEKID objects")
        print(f"Successfully created {len(self.mr_complex_resonators)} persistent MR_complex_resonator objects")
        
        # Analyze and print neighbor relationships based on linewidths
        if self.resonator_frequencies and self.resonator_linewidths:
            print("\n--- Resonator Neighbor Analysis (10 linewidth threshold) ---")
            linewidth_threshold = 10
            
            for i in range(len(self.resonator_frequencies)):
                neighbors = []
                res_freq = self.resonator_frequencies[i]
                linewidth = self.resonator_linewidths[i]
                
                # Check all other resonators
                for j in range(len(self.resonator_frequencies)):
                    if i != j:
                        other_freq = self.resonator_frequencies[j]
                        freq_diff = abs(res_freq - other_freq)
                        if freq_diff <= linewidth_threshold * linewidth:
                            neighbors.append(j)
                
                print(f"  Resonator {i} ({res_freq/1e9:.3f} GHz, LW={linewidth/1e3:.1f} kHz): "
                      f"{len(neighbors)} neighbors within {linewidth_threshold} linewidths")
                
                if neighbors and len(neighbors) <= 5:  # Show details for small neighbor sets
                    neighbor_details = []
                    for n in neighbors:
                        n_freq = self.resonator_frequencies[n]
                        separation_khz = abs(res_freq - n_freq) / 1e3
                        separation_linewidths = abs(res_freq - n_freq) / linewidth
                        neighbor_details.append(f"R{n} ({separation_khz:.0f}kHz, {separation_linewidths:.1f}LW)")
                    print(f"    Neighbors: {', '.join(neighbor_details)}")
            
            # Summary statistics
            total_neighbors = 0
            for i in range(len(self.resonator_frequencies)):
                res_freq = self.resonator_frequencies[i]
                linewidth = self.resonator_linewidths[i]
                neighbors_count = 0
                for j in range(len(self.resonator_frequencies)):
                    if i != j:
                        other_freq = self.resonator_frequencies[j]
                        freq_diff = abs(res_freq - other_freq)
                        if freq_diff <= linewidth_threshold * linewidth:
                            neighbors_count += 1
                total_neighbors += neighbors_count
            avg_neighbors = total_neighbors / len(self.resonator_frequencies)
            print(f"\n  Average neighbors per resonator: {avg_neighbors:.1f}")
            print(f"  This means typical S21 calculations will process ~{int(avg_neighbors)+1} resonators instead of {len(self.resonator_frequencies)}")
            print("--- End Neighbor Analysis ---\n")

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
                resonators=config.get('pulse_resonators', 'all')
            )
        
        self.invalidate_caches()


    def s21_lc_response(self, frequency, amplitude=1.0):
        """
        Calculate S21 response with optimized convergence.
        
        Includes:
        - Frequency-selective resonator filtering (only compute nearby resonators)
        - Pulse-modified quasiparticle density (if pulses are active)
        - Fresh noisy quasiparticle density each call
        - Physics-based Lk and R from combined nqp
        - OPTIMIZED: Skip convergence for noise-only changes
        """
        import time
        t_start = time.perf_counter()
        
        if not self.mr_lekids:
            return 1.0 + 0j
        
        # Get relevant resonators
        relevant_indices = self._get_relevant_resonators(frequency, linewidth_threshold=10)
        
        # If no resonators are relevant at this frequency, return unity (no attenuation)
        if not relevant_indices:
            return 1.0 + 0j
        
        # Calculate effective nqp with fresh noise for ALL resonators
        effective_nqp = []
        for i in range(len(self.base_nqp_values)):
            base_nqp = self.base_nqp_values[i]
            current_nqp = base_nqp
            
            # Add pulses
            for pulse in self.pulse_events:
                if pulse['resonator_index'] == i:
                    pulse_dt = self.last_update_time - pulse['start_time']
                    if pulse_dt >= 0:
                        if pulse_dt < pulse['tau_rise']:
                            time_factor = (1 - np.exp(-pulse_dt / pulse['tau_rise']))
                        else:
                            decay_dt = pulse_dt - pulse['tau_rise']
                            time_factor = np.exp(-decay_dt / pulse['tau_decay'])
                        excess_factor = (pulse['amplitude'] - 1.0) * time_factor
                        current_nqp += excess_factor * base_nqp
            
            # Add fresh noise
            if self.nqp_noise_enabled and current_nqp > 0:
                noise = np.random.normal(0, base_nqp * self.nqp_noise_std_factor)
                current_nqp = max(0, current_nqp + noise)
            
            effective_nqp.append(current_nqp)
        
        # Always update R,Lk from physics (fast, preserves IQ behavior)
        self.update_base_params_from_nqp(effective_nqp)
        
        # === OPTIMIZATION: Check if we need convergence ===
        # Create cache key for this operating point using configurable tolerances
        freq_tol = self._tolerance_config['cache_freq_tolerance']
        amp_tol = self._tolerance_config['cache_amp_tolerance']
        freq_key = round(frequency / freq_tol) * freq_tol
        amp_key = round(amplitude / amp_tol) * amp_tol
        cache_key = (freq_key, amp_key)
        
        # Check if we have cached convergence for this operating point
        skip_convergence = False
        cached_data = self._convergence_cache.get(cache_key)
        
        if cached_data is not None:
            # Improved logic: Check if QP changed significantly from all sources
            # Compare current QP state to cached QP state
            cached_qp = cached_data.get('qp_state', effective_nqp) if isinstance(cached_data, dict) else effective_nqp
            
            # Calculate maximum QP change fraction across all resonators
            max_qp_change = 0.0
            for i in range(len(effective_nqp)):
                if cached_qp[i] > 0:
                    qp_change = abs(effective_nqp[i] - cached_qp[i]) / cached_qp[i]
                    max_qp_change = max(max_qp_change, qp_change)
            
            # Check if power changed significantly (frequency or amplitude)
            freq_change = abs(frequency - freq_key) / freq_key if freq_key > 0 else 0
            amp_change = abs(amplitude - amp_key) / amp_key if amp_key > 0 else 0
            
            # Skip convergence only if both QP and power changes are below thresholds
            qp_threshold = self._tolerance_config['qp_change_threshold']
            power_threshold = self._tolerance_config['power_change_threshold']
            
            skip_convergence = (max_qp_change < qp_threshold and 
                               freq_change < power_threshold and 
                               amp_change < power_threshold)
            
            # Store reason for statistics
            if skip_convergence:
                self._convergence_stats['last_reason'] = f"QP:{max_qp_change:.3f}<{qp_threshold:.3f}"
            else:
                self._convergence_stats['last_reason'] = f"QP:{max_qp_change:.3f}>={qp_threshold:.3f}"
        
        # Update parameters based on convergence decision
        if not skip_convergence:
            # Run full convergence
            self.update_lekids_for_current(frequency, amplitude, relevant_indices)
            
            # Cache the convergence factors and QP state for this operating point
            # Store as a dict with both factors and QP state
            self._convergence_cache[cache_key] = {
                'factors': self.lk_current_factors.copy(),
                'qp_state': effective_nqp.copy()
            }
            
            # Limit cache size
            if len(self._convergence_cache) > self._convergence_cache_max_size:
                # Remove oldest entry (dict preserves insertion order in Python 3.7+)
                oldest_key = next(iter(self._convergence_cache))
                del self._convergence_cache[oldest_key]
            
            # Update statistics
            self._convergence_stats['full'] += 1
            
            # Debug print for convergence trigger
            if self._convergence_stats['full'] % 100 == 0:  # Print every 100th convergence
                total_calls = self._convergence_stats['full'] + self._convergence_stats['skipped']
                skip_rate = (self._convergence_stats['skipped'] / total_calls * 100) if total_calls > 0 else 0
                    
                print(f"[Convergence Triggered] f={frequency/1e9:.6f}GHz, amp={amplitude:.6f} | "
                      f"Cache size: {len(self._convergence_cache)} | "
                      f"Stats: {self._convergence_stats['full']} full, "
                      f"{self._convergence_stats['skipped']} skipped ({skip_rate:.1f}% skip rate) | "
                      f"Reason: {self._convergence_stats.get('last_reason', 'N/A')}")
        else:
            # Reuse cached convergence factors - apply to ALL resonators
            cached_factors = cached_data.get('factors') if isinstance(cached_data, dict) else cached_data
            
            for i in range(len(self.mr_lekids)):
                lekid = self.mr_lekids[i]
                base = self.base_lekid_params[i]
                factor = cached_factors[i] if (cached_factors and i < len(cached_factors)) else 1.0
                
                # Apply cached factor to current base values (updated for noise)
                lekid.Lk = base['Lk'] * factor
                lekid.R = base['R']
                lekid.L = lekid.Lk + lekid.Lg
                lekid.alpha_k = lekid.Lk / lekid.L
            
            # Update statistics
            self._convergence_stats['skipped'] += 1
        
        # Extract subset of parameters for relevant resonators
        n_relevant = len(relevant_indices)
        L_subset = np.zeros(n_relevant)
        C_subset = np.zeros(n_relevant)
        R_subset = np.zeros(n_relevant)
        Cc_subset = np.zeros(n_relevant)
        L_junk_subset = np.zeros(n_relevant)
        
        for idx, i in enumerate(relevant_indices):
            lekid = self.mr_lekids[i]
            L_subset[idx] = lekid.L
            C_subset[idx] = lekid.C
            R_subset[idx] = lekid.R
            Cc_subset[idx] = lekid.Cc
            L_junk_subset[idx] = lekid.L_junk
        
        # Get common parameters from first LEKID (they should all be the same)
        lekid0 = self.mr_lekids[0]
        
        # JIT-compiled S21 calculation for SUBSET of resonators only!
        Vout_array = jit_physics.compute_s21_vectorized(
            fc=frequency,
            Vin=amplitude,
            L_array=L_subset,
            C_array=C_subset,
            R_array=R_subset,
            Cc_array=Cc_subset,
            L_junk_array=L_junk_subset,
            ZLNA=complex(lekid0.ZLNA),
            GLNA=lekid0.GLNA,
            input_atten_dB=lekid0.input_atten_dB,
            system_termination=lekid0.system_termination
        )
        
        # Convert to S21 (divide by input voltage) and multiply all together
        s21_array = Vout_array / amplitude
        s21_total = np.prod(s21_array)
        
        t_vout = time.perf_counter()
        
        # Diagnostic timing (commented out - timing vars removed for optimization)
        # if not hasattr(self, '_timing_counter'):
        #     self._timing_counter = 0
        # self._timing_counter += 1
        # 
        # if self._timing_counter % 10 == 0:  # Log every 10th call
        #     total_ms = (t_vout - t_start) * 1000
        #     # Additional timing would require adding timing points back
        
        return s21_total

    def update_lekid_parameters(self, lekid_index):
        """
        Apply current-dependent modifications to a LEKID's parameters.
        
        Order of operations:
        1. Start with base parameters (already updated by physics from nqp)
        2. Apply current-dependent modifications to Lk only
        """
        base_params = self.base_lekid_params[lekid_index]
        lekid = self.mr_lekids[lekid_index]
        
        # Apply current effects to Lk (R is not affected by current)
        Lk_total = base_params['Lk'] * self.lk_current_factors[lekid_index]
        R_total = base_params['R']  # R comes directly from physics calculation
        
        # Update the LEKID object
        lekid.Lk = Lk_total
        lekid.R = R_total
        lekid.L = lekid.Lk + lekid.Lg
        lekid.alpha_k = lekid.Lk / lekid.L


    def calculate_resonator_currents(self, frequency, Vin, damp=0.1):
        """
        Calculate the current through each resonator.
        
        This is now handled inside the JIT-compiled convergence loop for efficiency.
        This method is kept for compatibility but just returns the cached currents.
        """
        if not self.mr_lekids:
            return []
        
        # Return cached currents from last convergence
        if hasattr(self, 'resonator_currents'):
            return self.resonator_currents
        else:
            return [0] * len(self.mr_lekids)

    def calculate_current_factors(self, frequency, amplitude):
        """
        Calculate current-dependent Lk factors for all resonators.
        
        Uses JIT-compiled helper for maximum performance.
        """
        # This is now handled inside the convergence loop
        # Return cached factors if available
        if hasattr(self, 'lk_current_factors'):
            return self.lk_current_factors
        
        # Otherwise calculate using JIT helper
        if hasattr(self, 'resonator_currents_array'):
            factors = jit_physics.current_factors(self.resonator_currents_array, self.Istar)
            return factors.tolist()
        else:
            return [1.0] * len(self.mr_lekids)

    def update_lekids_for_current(self, frequency, amplitude, relevant_indices=None):
        """
        Update LEKID parameters based on resonator currents.
        
        Uses JIT-compiled convergence loop for 2-5x speedup.
        Can update all resonators or just a subset for frequency-selective optimization.
        
        Parameters
        ----------
        frequency : float
            Probe frequency in Hz
        amplitude : float
            Probe amplitude  
        relevant_indices : list, optional
            Indices of resonators to update. If None, updates all resonators.
            
        Convergence tolerance can be configured via physics_config:
        - 1e-3: Ultra fast (for many channels)
        - 1e-4: High speed (default, good balance)
        - 1e-5: Balanced accuracy
        - 1e-6: High accuracy (slower)
        """
        max_iterations = 50
        tolerance = self.mock_crs.physics_config.get('convergence_tolerance', 1e-9)
        
        # Get LEKID config from first resonator
        lekid0 = self.mr_lekids[0]
        
        # Determine which resonators to update
        if relevant_indices is None:
            # Update all resonators (original behavior)
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
            L_junk_work = self.L_junk_array
            base_Lk = np.array([self.base_lekid_params[i]['Lk'] for i in range(n)])
            base_Lg = np.array([self.base_lekid_params[i]['Lg'] for i in range(n)])
            indices_to_update = list(range(n))
        else:
            # Update only subset (frequency-selective optimization)
            if not relevant_indices:
                return
            
            n = len(relevant_indices)
            
            # Extract subset of parameters
            L_work = np.zeros(n)
            R_work = np.zeros(n)
            C_work = np.zeros(n)
            Cc_work = np.zeros(n)
            L_junk_work = np.zeros(n)
            base_Lk = np.zeros(n)
            base_Lg = np.zeros(n)
            
            for idx, i in enumerate(relevant_indices):
                lekid = self.mr_lekids[i]
                L_work[idx] = lekid.L
                R_work[idx] = lekid.R
                C_work[idx] = lekid.C
                Cc_work[idx] = lekid.Cc
                L_junk_work[idx] = lekid.L_junk
                base_Lk[idx] = self.base_lekid_params[i]['Lk']
                base_Lg[idx] = self.base_lekid_params[i]['Lg']
            
            indices_to_update = relevant_indices
        
        # Call JIT-compiled convergence loop
        L_converged, R_converged, currents_converged, actual_iterations = \
            jit_physics.converged_lekid_parameters(
                frequency, amplitude,
                L_work, R_work, C_work,
                Cc_work, L_junk_work,
                base_Lk, base_Lg,
                lekid0.input_atten_dB, complex(lekid0.ZLNA),
                self.Istar, tolerance, max_iterations, damp=0.1
            )
        
        # Extract current factors from converged inductances
        Lk_converged = L_converged - base_Lg
        current_factors = Lk_converged / base_Lk
        
        # Update LEKID objects with converged values
        for idx, i in enumerate(indices_to_update):
            lekid = self.mr_lekids[i]
            lekid.Lk = Lk_converged[idx]
            lekid.R = R_converged[idx]
            lekid.L = L_converged[idx]
            lekid.alpha_k = Lk_converged[idx] / L_converged[idx]
            
            # Update factors for these specific resonators
            if relevant_indices is not None:
                # Subset update
                self.lk_current_factors[i] = current_factors[idx]
        
        # Update cached data if we updated all resonators
        if relevant_indices is None:
            self.lk_current_factors = current_factors.tolist()
            self.resonator_currents_array = currents_converged
            self.resonator_currents = currents_converged.tolist()
            self.L_array = L_converged
            self.R_array = R_converged
        
        # Log convergence stats occasionally
        if not hasattr(self, '_convergence_counter'):
            self._convergence_counter = 0
        self._convergence_counter += 1
        
        # if self._convergence_counter % 1000 == 0:
        #     n_updated = len(indices_to_update)
        #     print(f"[JIT Convergence] Iterations: {actual_iterations}/{max_iterations} "
        #           f"at f={frequency/1e9:.3f}GHz, updated {n_updated} resonators")

    def generate_noisy_nqp_values(self):
        """
        Generate fresh noisy nqp values for all resonators.
        
        Returns
        -------
        list
            List of noisy nqp values, one per resonator
        """
        noisy_nqp = []
        for base_nqp in self.base_nqp_values:
            if self.nqp_noise_enabled and base_nqp > 0:
                # Generate Gaussian noise with std = base_nqp * noise_factor
                noise = np.random.normal(0, base_nqp * self.nqp_noise_std_factor)
                # Ensure non-negative nqp
                noisy_value = max(0, base_nqp + noise)
                noisy_nqp.append(noisy_value)
            else:
                noisy_nqp.append(base_nqp)
        return noisy_nqp

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
                    self.add_pulse_event(res_idx, current_time)
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
        print('added new pulse with amplitude:', pulse['amplitude'])
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
    
    def _evaluate_s21_at_frequency(self, frequency, amplitude):
        """
        S21 evaluation at a specific frequency - properly handles pulses + noise.
        
        This now properly calls s21_lc_response to ensure each evaluation gets:
        - Fresh QP noise 
        - Current pulse state
        - Vectorized convergence (performance improvement retained)
        
        Parameters
        ----------
        frequency : float
            Probe frequency in Hz
        amplitude : float
            Probe amplitude
            
        Returns
        -------
        complex
            S21 response
        """
        # Use the full physics path that handles pulses + noise + convergence
        # Each channel gets its own fresh QP state as the physics requires
        return self.s21_lc_response(frequency, amplitude)
    
    def _get_cached_s21(self, frequency, amplitude):
        """Get S21 response with caching disabled when noise is enabled."""
        # Disable caching when noise is enabled to ensure fresh noise each call
        if self.nqp_noise_enabled:
            return self.s21_lc_response(frequency, amplitude)
        
        # Use caching when noise is disabled for performance
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
        nco_freq = self.mock_crs.get_nco_frequency(module=module)
        
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
        
        # Collect active tones (transmitting channels) using proper getter methods
        s21_call_count = 0
        for ch in configured_channels:
            freq = self.mock_crs.get_frequency(channel=ch, module=module)
            amp = self.mock_crs.get_amplitude(channel=ch, module=module)
            
            if freq is not None and amp is not None and amp != 0:
                phase_deg = self.mock_crs.phases.get((module, ch), 0)  # No getter for phase yet
                total_freq = freq + nco_freq
                
                # Apply S21 response using FAST path (state already updated for this packet)
                s21_complex = self._evaluate_s21_at_frequency(total_freq, amp)
                s21_call_count += 1
                
                # Combine amplitude, S21, and phase
                complex_amplitude = amp * s21_complex * np.exp(1j * np.deg2rad(phase_deg))
                
                active_tone_freqs.append(total_freq)
                active_tone_amps.append(complex_amplitude)
        
        t_s21_calc = time.perf_counter()
        
        # Collect observing channels
        for ch in configured_channels:
            freq = self.mock_crs.frequencies.get((module, ch))
            if freq is not None:
                obs_channels.append(ch)
                obs_freqs.append(freq + nco_freq)
        
        # If no active tones or observers, return empty
        if not active_tone_freqs or not obs_channels:
            return {}
        
        # Diagnostic logging
        if not hasattr(self, '_packet_timing_counter'):
            self._packet_timing_counter = 0
        self._packet_timing_counter += 1
        
        # if self._packet_timing_counter % 10 == 0:
        #     state_ms = (t_state_update - t_packet_start) * 1000
        #     s21_ms = (t_s21_calc - t_state_update) * 1000
        #     total_ms = (t_s21_calc - t_packet_start) * 1000
        #     print(f"[Packet Timing] Total: {total_ms:.1f}ms | "
        #           f"state_update: {state_ms:.1f}ms | "
        #           f"s21_calculations: {s21_ms:.1f}ms ({s21_call_count} calls)")
        
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
