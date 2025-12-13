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
