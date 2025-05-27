"""
Mock CRS Constants - Configuration parameters for the mock resonator system.
Similar to periscope's utils.py, this provides centralized configuration.
"""

# =============================================================================
# Kinetic Inductance and Power Dependence Parameters
# =============================================================================

# Kinetic inductance fraction (fraction of total inductance that's kinetic)
DEFAULT_KINETIC_INDUCTANCE_FRACTION = 0.2  # 20% typical for Al resonators
KINETIC_INDUCTANCE_VARIATION = 0.1  # ±10% variation between resonators

# Power dependence parameters - ENHANCED for visible frequency shifts
FREQUENCY_SHIFT_POWER_LAW = 1.5  # Power law exponent: Δf ∝ P^α (linear for simplicity)
FREQUENCY_SHIFT_MAGNITUDE = 1e-2  # Base magnitude: Δf/f₀ = magnitude * (P/P₀)^α (1000x stronger!)
POWER_NORMALIZATION = 0.001  # Reference power level for normalization (lower = more sensitive)

# Bifurcation parameters
ENABLE_BIFURCATION = True  # Enable self-consistent frequency shift
BIFURCATION_ITERATIONS = 2000  # Number of iterations for self-consistency
BIFURCATION_CONVERGENCE_TOLERANCE = 1e-9  # Convergence criterion for frequency
BIFURCATION_DAMPING_FACTOR = 0.9  # Damping factor to prevent oscillation (0.1-0.5 range)

# Nonlinearity saturation
SATURATION_POWER = 0.1  # Power level where nonlinearity saturates (fixed comment)
SATURATION_SHARPNESS = 2.0  # Sharpness of saturation transition

# =============================================================================
# Resonance Physical Parameters
# =============================================================================

# Q factor range for narrow resonances (FWHM = f₀/Q)
# For ~10 kHz FWHM at 1 GHz: Q = 100,000
DEFAULT_Q_MIN = 80000
DEFAULT_Q_MAX = 120000
Q_VARIATION = 0.2  # ±20% variation between resonators

# Coupling parameters for resonance depth
# k = 0.68 gives ~10 dB, k = 0.9 gives ~20 dB, k = 0.97 gives ~30 dB
DEFAULT_COUPLING_MIN = 0.68
DEFAULT_COUPLING_MAX = 0.97

# Frequency range for resonance generation
DEFAULT_FREQ_START = 1e9  # 1 GHz
DEFAULT_FREQ_END = 2e9    # 2 GHz
DEFAULT_NUM_RESONANCES = 500

# =============================================================================
# Amplitude and Noise Parameters
# =============================================================================

# Noise parameters
BASE_NOISE_LEVEL = 0.001  # Base noise level (relative to S21)
AMPLITUDE_NOISE_COUPLING = 0.01  # How noise scales with amplitude

# ADC simulation parameters  
UDP_NOISE_LEVEL = 10.0  # ADC noise level (in counts)
SCALE_FACTOR = 2**21  # Scaling factor: 2^21 ≈ 2.1 million counts

# =============================================================================
# Physical Constants and Conversions
# =============================================================================

# Superconductor properties (for advanced modeling if needed)
ALUMINUM_TC = 1.2  # Kelvin (critical temperature)
TYPICAL_OPERATING_TEMP = 0.1  # Kelvin

# Frequency shift temperature dependence (if needed)
FREQ_TEMP_COEFFICIENT = -1e-4  # df/f per Kelvin

# =============================================================================
# Performance and Numerical Parameters
# =============================================================================

# Resonance spacing to avoid overlap
MIN_RESONANCE_SPACING_FACTOR = 2.0  # Minimum spacing as multiple of FWHM

# Random seed for reproducible resonator generation (None = random)
RESONATOR_RANDOM_SEED = None

# =============================================================================
# Debug and Development Parameters
# =============================================================================

# Enable debug output
DEBUG_RESONATOR_GENERATION = False
DEBUG_POWER_DEPENDENCE = True  # Enable to see frequency shifts
DEBUG_BIFURCATION = False

# Performance monitoring
ENABLE_PERFORMANCE_TIMING = False
