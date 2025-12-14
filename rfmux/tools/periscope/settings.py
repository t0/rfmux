"""
Periscope Settings Module
=========================

Centralized QSettings wrapper for remembering user preferences across sessions.

Uses Qt's QSettings which persists settings to:
- Windows: Registry (HKEY_CURRENT_USER\\Software\\rfmux\\periscope)
- macOS: ~/Library/Preferences/com.rfmux.periscope.plist
- Linux: ~/.config/rfmux/periscope.conf

Usage:
    from . import settings
    
    # Get/set last session directory
    last_dir = settings.get_last_session_directory()
    settings.set_last_session_directory("/path/to/sessions")
"""

from PyQt6.QtCore import QSettings
from pathlib import Path
from typing import Optional

# Organization and application name for QSettings
ORGANIZATION = "rfmux"
APPLICATION = "periscope"

# Settings keys
KEY_CONNECTION_MODE = "connection/last_mode"
KEY_CRS_SERIAL = "connection/last_crs_serial"
KEY_MODULE = "connection/last_module"
KEY_SESSION_DIRECTORY = "session/last_base_directory"
KEY_USER_LIBRARY_PATH = "notebook/user_library_path"
KEY_CUSTOM_MATERIALS = "materials/custom_materials"

# Default values
DEFAULT_CONNECTION_MODE = "hardware"
DEFAULT_MODULE = 1


def _get_settings() -> QSettings:
    """Get a QSettings instance with our org/app names."""
    return QSettings(ORGANIZATION, APPLICATION)


# ─────────────────────────────────────────────────────────────────
# Connection Settings
# ─────────────────────────────────────────────────────────────────

def get_last_connection_mode() -> str:
    """
    Get the last used connection mode.
    
    Returns:
        str: "hardware", "mock", or "offline"
    """
    settings = _get_settings()
    return settings.value(KEY_CONNECTION_MODE, DEFAULT_CONNECTION_MODE)


def set_last_connection_mode(mode: str) -> None:
    """
    Save the connection mode.
    
    Args:
        mode: "hardware", "mock", or "offline"
    """
    if mode not in ("hardware", "mock", "offline"):
        raise ValueError(f"Invalid connection mode: {mode}")
    settings = _get_settings()
    settings.setValue(KEY_CONNECTION_MODE, mode)


def get_last_crs_serial() -> str:
    """
    Get the last used CRS serial number.
    
    Returns:
        str: CRS serial number (e.g., "0042") or empty string
    """
    settings = _get_settings()
    return settings.value(KEY_CRS_SERIAL, "")


def set_last_crs_serial(serial: str) -> None:
    """
    Save the CRS serial number.
    
    Args:
        serial: CRS serial number (e.g., "0042")
    """
    settings = _get_settings()
    settings.setValue(KEY_CRS_SERIAL, serial)


def get_last_module() -> int:
    """
    Get the last used module number.
    
    Returns:
        int: Module number (1-8), defaults to 1
    """
    settings = _get_settings()
    value = settings.value(KEY_MODULE, DEFAULT_MODULE)
    # QSettings may return string on some platforms
    return int(value) if value else DEFAULT_MODULE


def set_last_module(module: int) -> None:
    """
    Save the module number.
    
    Args:
        module: Module number (1-8)
    """
    if not 1 <= module <= 8:
        raise ValueError(f"Module must be 1-8, got: {module}")
    settings = _get_settings()
    settings.setValue(KEY_MODULE, module)


# ─────────────────────────────────────────────────────────────────
# Session Settings
# ─────────────────────────────────────────────────────────────────

def get_last_session_directory() -> str:
    """
    Get the last directory used for session creation/loading.
    
    Returns:
        str: Directory path or empty string
    """
    settings = _get_settings()
    return settings.value(KEY_SESSION_DIRECTORY, "")


def set_last_session_directory(path: str) -> None:
    """
    Save the session directory.
    
    Args:
        path: Directory path
    """
    settings = _get_settings()
    settings.setValue(KEY_SESSION_DIRECTORY, str(path))


# ─────────────────────────────────────────────────────────────────
# Notebook Settings
# ─────────────────────────────────────────────────────────────────

def get_user_library_path() -> str:
    """
    Get the user notebook library path (used for symlinks in sessions).
    
    Returns:
        str: Library path or empty string
    """
    settings = _get_settings()
    return settings.value(KEY_USER_LIBRARY_PATH, "")


def set_user_library_path(path: str) -> None:
    """
    Save the user notebook library path.
    
    Args:
        path: Library directory path
    """
    settings = _get_settings()
    settings.setValue(KEY_USER_LIBRARY_PATH, str(path))


# ─────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────

def clear_all() -> None:
    """Clear all saved settings (useful for testing/reset)."""
    settings = _get_settings()
    settings.clear()


def get_settings_path() -> Optional[Path]:
    """
    Get the path to the settings file (for debugging).
    
    Returns:
        Path to settings file, or None if using registry (Windows)
    """
    settings = _get_settings()
    filename = settings.fileName()
    return Path(filename) if filename else None


# ─────────────────────────────────────────────────────────────────
# Custom Materials Settings
# ─────────────────────────────────────────────────────────────────

def get_custom_materials() -> dict:
    """
    Get all user-defined custom materials.
    
    Returns:
        dict: Material name -> properties dict
              e.g., {'Nb': {'Tc': 9.2, 'N0': 3.5e10, 'tau0': 200e-9, 'sigmaN': 5e7}}
    """
    import json
    settings = _get_settings()
    json_str = settings.value(KEY_CUSTOM_MATERIALS, "{}")
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def save_custom_material(name: str, tc: float, n0: float, tau0: float, Rs: Optional[float] = None, thickness_ref_nm: Optional[float] = None, sigmaN: Optional[float] = None) -> None:
    """
    Add or update a custom material.
    
    Args:
        name: Material name/symbol (e.g., "Nb", "TiN")
        tc: Critical temperature [K]
        n0: Density of states at Fermi level [µm⁻³eV⁻¹]
        tau0: Quasiparticle recombination time [s]
        Rs: Sheet resistance [Ω/□] - preferred input method
        thickness_ref_nm: Reference film thickness [nm] where Rs was measured
        sigmaN: Normal state conductivity [S/m] - calculated from Rs if not provided
        
    Note:
        Either (Rs AND thickness_ref_nm) OR sigmaN must be provided.
        If Rs is provided, sigmaN is calculated as: σN = 1/(Rs × thickness)
    """
    import json
    materials = get_custom_materials()
    
    # Calculate sigmaN from Rs if provided
    if Rs is not None and thickness_ref_nm is not None:
        thickness_m = thickness_ref_nm * 1e-9  # nm to m
        sigmaN = 1.0 / (Rs * thickness_m)
        
        mat_data = {
            'Tc': float(tc),
            'N0': float(n0),
            'tau0': float(tau0),
            'Rs': float(Rs),
            'thickness_ref': float(thickness_ref_nm),
            'sigmaN': float(sigmaN)
        }
    elif sigmaN is not None:
        # Legacy: direct sigmaN input (backward compatibility)
        mat_data = {
            'Tc': float(tc),
            'N0': float(n0),
            'tau0': float(tau0),
            'sigmaN': float(sigmaN)
        }
    else:
        # Default Rs and thickness for backward compatibility
        Rs_default = 4.0
        thickness_default = 20.0
        sigmaN_default = 1.0 / (Rs_default * thickness_default * 1e-9)
        mat_data = {
            'Tc': float(tc),
            'N0': float(n0),
            'tau0': float(tau0),
            'Rs': Rs_default,
            'thickness_ref': thickness_default,
            'sigmaN': sigmaN_default
        }
    
    materials[name] = mat_data
    
    settings = _get_settings()
    settings.setValue(KEY_CUSTOM_MATERIALS, json.dumps(materials))


def delete_custom_material(name: str) -> None:
    """
    Remove a custom material.
    
    Args:
        name: Material name to remove
    """
    import json
    materials = get_custom_materials()
    if name in materials:
        del materials[name]
        settings = _get_settings()
        settings.setValue(KEY_CUSTOM_MATERIALS, json.dumps(materials))


def get_material_properties(name: str) -> dict:
    """
    Get properties for a material (built-in or custom).
    
    Args:
        name: Material name (e.g., "Al", "Nb")
        
    Returns:
        dict: {
            'Tc': float,      # Critical temperature [K]
            'N0': float,      # Density of states [µm⁻³eV⁻¹] - NOTE: eV units, needs conversion!
            'tau0': float,    # Quasiparticle recombination time [s]
            'sigmaN': float   # Normal state conductivity [S/m]
        }
        
    Note:
        N0 is stored in µm⁻³eV⁻¹ for human readability. When passing to
        MR_complex_resonator, convert to µm⁻³J⁻¹:
        
            N0_internal = N0_eV / 1.602e-19
        
    Raises:
        ValueError: If material not found
    """
    # Built-in materials
    if name == "Al":
        return {
            'Tc': 1.2,
            'N0': 1.72e10,  # µm⁻³eV⁻¹
            'tau0': 438e-9,  # s
            'Rs': 4.0,  # Ω/□ (typical for thin Al)
            'thickness_ref': 20.0,  # nm (reference thickness)
            'sigmaN': 1./(4*20e-9)  # S/m (calculated from Rs)
        }
    
    # Custom materials
    materials = get_custom_materials()
    if name in materials:
        mat = materials[name].copy()
        # Ensure sigmaN has default if not specified
        if 'sigmaN' not in mat:
            mat['sigmaN'] = 1./(4*20e-9)
        return mat
    
    raise ValueError(f"Material '{name}' not found in built-in or custom materials")
