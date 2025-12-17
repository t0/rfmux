"""
Single Source of Truth (SoT) for MockCRS configuration.

This module centralizes all parameters used by:
- MockCRS (core)
- MockResonatorModel (physics)
- UDP streamer (packet generation)
- Periscope Mock Configuration Dialog (GUI)

External code should import MOCK_DEFAULTS and optionally call apply_overrides()
to merge user-provided overrides with validated defaults.

No other module should define mock defaults or duplicate parameters.
"""

from __future__ import annotations

from typing import Any, Dict
from copy import deepcopy
import re

# =============================================================================
# Unified Default Configuration
# =============================================================================

MOCK_DEFAULTS: Dict[str, Any] = {
    # -------------------------------------------------------------------------
    # Basic resonator distribution
    # -------------------------------------------------------------------------
    "num_resonances": 5,
    "freq_start": 1.0e9,   # Hz
    "freq_end": 1.5e9,     # Hz
    "resonator_random_seed": None,  # int | None

    # -------------------------------------------------------------------------
    # Physics parameters (determine Lk and R via quasiparticle density)
    # -------------------------------------------------------------------------
    "T": 0.12,             # Temperature [K]
    "Popt": 1e-15,         # Optical power [W]

    # -------------------------------------------------------------------------
    # Material & Geometry (optional - uses hardcoded defaults if not specified)
    # -------------------------------------------------------------------------
    "material": "Al",      # Superconductor material
    "width": 2e-6,         # Strip width [m] (2 µm)
    "thickness": 30e-9,    # Film thickness [m] (30 nm)
    "length": 9000e-6,     # Strip length [m] (9000 µm = 9 mm)
    
    # -------------------------------------------------------------------------
    # Circuit parameters (base values for MR_LEKID)
    # -------------------------------------------------------------------------
    "Lg": 10e-9,           # Geometric inductance [H]
    "Cc": 0.01e-12,        # Coupling capacitor [F]
    "L_junk": 0.0,         # Parasitic inductance [H]

    # -------------------------------------------------------------------------
    # Variations (as fractional standard deviations)
    # -------------------------------------------------------------------------
    "C_variation": 0.01,   # 1% variation in capacitance
    "Cc_variation": 0.01,  # 1% variation in coupling capacitor

    # -------------------------------------------------------------------------
    # Readout parameters
    # -------------------------------------------------------------------------
    "Vin": 1e-5,               # Input voltage [V]
    "input_atten_dB": 10.0,    # Input attenuation [dB]
    "system_termination": 50.0,# System impedance [Ω]
    "ZLNA": 50.0,              # LNA input impedance [Ω] (real)
    "GLNA": 10.0**(10.0/20.0), # LNA gain (Vout/Vin)

    # -------------------------------------------------------------------------
    # Noise configuration
    # -------------------------------------------------------------------------
    "nqp_noise_enabled": True,  # Enable noise on quasiparticle density
    "nqp_noise_std_factor": 0.001,  # Std dev as fraction of base nqp (0.1%)

    # -------------------------------------------------------------------------
    # Convergence parameters
    # -------------------------------------------------------------------------
    # 1e-9: Ultra high accuracy (default, allows up to 50 iterations)
    # 1e-7: High accuracy (slower, for few channels)
    # 1e-6: Balanced
    # 1e-5: High speed (faster, for many channels)
    "convergence_tolerance": 1e-9,

    # -------------------------------------------------------------------------
    # Cache quantization steps for convergence reuse
    # -------------------------------------------------------------------------
    "cache_freq_step": 0.0001,         # Hz - frequency quantization step for cache key
    "cache_amp_step": 1e-8,            # Amplitude quantization step
    "cache_qp_step": 0.0001,           # QP quantization as fraction of base QP
    "log_cache_decisions": False,      # enable cache decision logging (rate-limited)
    "cache_log_interval": 100,         # log every N convergence events
    "convergence_cache_max_size": 10000000,  # max cache entries

    # -------------------------------------------------------------------------
    # Automatic KID biasing parameters
    # -------------------------------------------------------------------------
    "auto_bias_kids": False,   # Enable automatic channel configuration
    "bias_amplitude": 0.01,    # Bias amplitude in normalized units (≈ -40 dBm)

    # -------------------------------------------------------------------------
    # UDP streamer (ADC simulation)
    # -------------------------------------------------------------------------
    "udp_noise_level": 10.0,   # Additive ADC noise [counts]
    "scale_factor": 2**21,     # Base 24-bit full-scale (~2.1e6 counts)

    # -------------------------------------------------------------------------
    # Quasiparticle pulse parameters (time-dependent nqp)
    # -------------------------------------------------------------------------
    "pulse_mode": "none",         # 'periodic', 'random', 'manual', or 'none'
    "pulse_period": 2.0,          # seconds (periodic mode)
    "pulse_probability": 0.1,     # per-timestep probability (random mode)
    "pulse_tau_rise": 1e-6,       # seconds
    "pulse_tau_decay": 0.1,       # seconds
    "pulse_amplitude": 2.0,       # multiplicative factor relative to base nqp
    "pulse_resonators": "all",    # 'all' or list of resonator indices

    # Random pulse amplitude distribution (random mode only)
    "pulse_random_amp_mode": "fixed",   # "fixed" | "uniform" | "lognormal"
    "pulse_random_amp_min": 1.5,        # for uniform mode (>= 1.0)
    "pulse_random_amp_max": 3.0,        # for uniform mode (>= min)
    "pulse_random_amp_logmean": 0.7,    # for lognormal mode
    "pulse_random_amp_logsigma": 0.3,   # for lognormal mode (>= 0)
}

# =============================================================================
# Helper Functions
# =============================================================================

def defaults() -> Dict[str, Any]:
    """Return a deep copy of MOCK_DEFAULTS."""
    return deepcopy(MOCK_DEFAULTS)


def apply_overrides(overrides: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Merge user-provided overrides onto defaults, returning a new dict.
    
    Unknown keys are allowed (forward-compatible), but callers should prefer
    the documented keys.

    Parameters
    ----------
    overrides : dict | None
        Optional user-provided configuration overrides

    Returns
    -------
    dict
        Merged configuration dictionary
    """
    cfg = defaults()
    if overrides:
        cfg.update(overrides)
    
    # Normalize/validate a few obvious types
    if cfg.get("resonator_random_seed", None) in ("", "None"):
        cfg["resonator_random_seed"] = None
    
    # Ensure numeric fields that might come in as strings are parsed
    for k in (
        "freq_start", "freq_end", "T", "Popt", "Lg", "Cc", "L_junk",
        "width", "thickness", "length",  # Geometry
        "C_variation", "Cc_variation", "Vin", "input_atten_dB",
        "system_termination", "ZLNA", "GLNA", "nqp_noise_std_factor",
        "convergence_tolerance", "bias_amplitude", "udp_noise_level",
        "scale_factor", "pulse_period", "pulse_probability",
        "pulse_tau_rise", "pulse_tau_decay", "pulse_amplitude",
        "pulse_random_amp_min", "pulse_random_amp_max",
        "pulse_random_amp_logmean", "pulse_random_amp_logsigma",
        "cache_freq_step", "cache_amp_step", "cache_qp_step"
    ):
        if k in cfg and isinstance(cfg[k], str):
            try:
                cfg[k] = float(cfg[k])
            except ValueError:
                pass
    
    for k in ("num_resonances", "cache_log_interval", "convergence_cache_max_size"):
        if k in cfg and isinstance(cfg[k], str):
            try:
                cfg[k] = int(cfg[k])
            except ValueError:
                pass

    # Normalize pulse_resonators: accept "all", CSV string, or list/tuple of ints/strings
    val = cfg.get("pulse_resonators", "all")
    if isinstance(val, str):
        s = val.strip().lower()
        if s == "" or s == "all":
            cfg["pulse_resonators"] = "all"
        else:
            parts = re.split(r"[,\s]+", val.strip())
            indices = []
            for p in parts:
                if not p:
                    continue
                try:
                    indices.append(int(p))
                except ValueError:
                    # ignore non-integer tokens
                    pass
            cfg["pulse_resonators"] = indices if indices else "all"
    elif isinstance(val, (list, tuple)):
        indices = []
        for p in val:
            if isinstance(p, int):
                indices.append(p)
            elif isinstance(p, str):
                try:
                    indices.append(int(p))
                except ValueError:
                    pass
        cfg["pulse_resonators"] = indices if indices else "all"

    # Normalize random amplitude distribution settings
    mode = cfg.get("pulse_random_amp_mode", "fixed")
    if isinstance(mode, str):
        mode_norm = mode.strip().lower()
        if mode_norm not in ("fixed", "uniform", "lognormal"):
            mode_norm = "fixed"
        cfg["pulse_random_amp_mode"] = mode_norm
    else:
        cfg["pulse_random_amp_mode"] = "fixed"

    # Enforce numeric constraints and sane defaults
    try:
        minv = float(cfg.get("pulse_random_amp_min", 1.5))
    except Exception:
        minv = 1.5
    try:
        maxv = float(cfg.get("pulse_random_amp_max", 3.0))
    except Exception:
        maxv = 3.0
    
    # Pulses should not reduce nqp unless explicitly desired; enforce >= 1.0
    if minv < 1.0:
        minv = 1.0
    if maxv < 1.0:
        maxv = 1.0
    # Ensure min <= max
    if maxv < minv:
        maxv = minv
    cfg["pulse_random_amp_min"] = minv
    cfg["pulse_random_amp_max"] = maxv

    try:
        logmean = float(cfg.get("pulse_random_amp_logmean", 0.7))
    except Exception:
        logmean = 0.7
    try:
        logsigma = float(cfg.get("pulse_random_amp_logsigma", 0.3))
    except Exception:
        logsigma = 0.3
    if logsigma < 0.0:
        logsigma = 0.0
    cfg["pulse_random_amp_logmean"] = logmean
    cfg["pulse_random_amp_logsigma"] = logsigma

    return cfg
