"""Platform-aware path management for rfmux."""

import os
import shutil
import stat
from pathlib import Path

import rfmux


_REFERENCE_NOTEBOOKS = Path(__file__).with_name("reference-notebooks")
#: The repository's docs/ (guides, release notes, installation): inside
#: the package in a wheel (the streamer's CMake installs it there),
#: beside the package in a source or editable install.
_DOCS = next((d for d in (Path(__file__).with_name("docs"),
                          Path(__file__).resolve().parents[1] / "docs")
              if d.is_dir()),
             Path(__file__).with_name("docs"))
#: What the docs folder is called among the provisioned notebooks.
DOCS_FOLDER = "Guides"


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

    The repository's docs/ folder (shipped in the wheel as rfmux/docs,
    beside the package in a source checkout) is provisioned alongside as
    ``Guides``, the figure scripts left out, so the guides and release
    notes are in the Jupyter session with the notebooks.
    """
    dest = get_rfmux_data_dir() / "reference-notebooks" / rfmux.__version__
    guides = dest / DOCS_FOLDER

    if not dest.exists():
        shutil.copytree(_REFERENCE_NOTEBOOKS, dest)
        _read_only(dest)
    # A version provisioned before the docs came along gets them now;
    # one provisioned as a copy of the whole docs/ tree is redone.
    if (guides / "guides").is_dir():
        for root, _dirs, _files in os.walk(guides):
            os.chmod(root, stat.S_IRWXU)     # folders were made read-only
        shutil.rmtree(guides)
    if _DOCS.is_dir() and not guides.exists():
        _provision_docs(dest)
    return dest


def _provision_docs(dest: Path) -> None:
    """The guides as ``Guides/`` with the installation page beside
    them, and the release notes into ``Release Notes/`` with the
    shipped walkthroughs: the tracked pieces of docs/, nothing else the
    folder may hold."""
    skip = shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints")
    guides = dest / DOCS_FOLDER
    shutil.copytree(_DOCS / "guides", guides, ignore=skip, dirs_exist_ok=True)
    if (_DOCS / "installation.md").is_file():
        shutil.copy2(_DOCS / "installation.md", guides / "installation.md")
    notes = dest / "Release Notes"
    if (_DOCS / "release-notes").is_dir():
        if notes.is_dir():
            os.chmod(notes, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        shutil.copytree(_DOCS / "release-notes", notes, ignore=skip,
                        dirs_exist_ok=True)
    for folder in (guides, notes):
        if folder.is_dir():
            _read_only(folder)
            os.chmod(folder, stat.S_IREAD | stat.S_IEXEC)


def _read_only(dest: Path) -> None:
    """Files 0o444 and folders 0o500 under *dest*, to discourage
    in-place editing."""
    for root, dirs, files in os.walk(dest):
        for d in dirs:
            os.chmod(os.path.join(root, d), stat.S_IREAD | stat.S_IEXEC)
        for f in files:
            os.chmod(os.path.join(root, f), stat.S_IREAD)
