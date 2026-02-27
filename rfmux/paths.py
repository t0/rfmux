"""Platform-aware path management for rfmux."""

import os
import shutil
import stat
from pathlib import Path

import rfmux


_REFERENCE_NOTEBOOKS = Path(__file__).with_name("reference-notebooks")


def get_rfmux_data_dir() -> Path:
    """Return the platform data directory for rfmux.

      - Linux/macOS: ~/.local/share/rfmux/
      - Windows:     ~/AppData/Local/rfmux/
    """
    if os.name == "nt":
        return Path.home() / "AppData" / "Local" / "rfmux"
    return Path.home() / ".local" / "share" / "rfmux"


def get_user_notebook_dir() -> Path:
    """Return the default directory for user notebooks."""
    return get_rfmux_data_dir() / "user-notebooks"


def get_reference_notebook_dir() -> Path:
    """Provision shipped notebooks to per-user directory and return path.

    Copies rfmux/reference-notebooks/ to a versioned subdirectory:
      - Linux/macOS: ~/.local/share/rfmux/reference-notebooks/<version>/
      - Windows:     ~/AppData/Local/rfmux/reference-notebooks/<version>/

    Files are made read-only (0o444/0o555) to discourage in-place editing.
    Each version gets its own directory, so upgrades never collide with
    notebooks that are already open.
    """
    dest = get_rfmux_data_dir() / "reference-notebooks" / rfmux.__version__

    if dest.exists():
        return dest

    shutil.copytree(_REFERENCE_NOTEBOOKS, dest)

    # Make files read-only to discourage in-place editing
    for root, dirs, files in os.walk(dest):
        for d in dirs:
            os.chmod(os.path.join(root, d), stat.S_IREAD | stat.S_IEXEC)
        for f in files:
            os.chmod(os.path.join(root, f), stat.S_IREAD)

    return dest
