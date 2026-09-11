"""Where rfmux keeps the handful of settings that outlive a Python session.

There is one YAML file, and you own it. ``config_template.yaml`` ships with the
package, documents every setting, and is never read except by :func:`init`,
which copies it to ``rfmux/config.yaml`` for you to edit. That copy is
gitignored, so a working directory full of real cryostat paths stays out of
commits — and because the template is the only tracked copy, an upgrade can add
a setting without ever overwriting your answer to an old one.

Nothing here decides anything. :func:`get` reports what the file says and
:mod:`rfmux.tuning.store` decides what to do about it, which is why the
resolution order — argument, then session override, then environment, then this
file, then the built-in default — lives there rather than here.

Nothing in this module needs a board, a GUI or a hardware map.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml

from .paths import get_rfmux_data_dir

__all__ = [
    "template_path",
    "path",
    "init",
    "get",
    "reload",
]


_TEMPLATE = Path(__file__).with_name("config_template.yaml")

# The user's copy, in preference order. The package-local one is what init()
# writes and what a git checkout wants; the data-directory one is for a
# pip-installed rfmux, where site-packages is not somewhere anyone should be
# editing files.
_CANDIDATES = (
    Path(__file__).with_name("config.yaml"),
    get_rfmux_data_dir() / "config.yaml",
)

# Parsed contents, keyed by the file they came from, so a reload() or a changed
# $RFMUX_CONFIG is not fought by a stale cache. None means "no file found".
_cache: dict[Path | None, dict] = {}


def template_path() -> Path:
    """The shipped template — documentation, and what :func:`init` copies."""
    return _TEMPLATE


def path() -> Path | None:
    """The config file in effect, or ``None`` if you have not made one yet.

    ``$RFMUX_CONFIG`` names a file directly and is not required to exist —
    pointing at a missing file is a mistake worth hearing about, so it raises
    rather than quietly falling through to the search below.
    """
    named = os.environ.get("RFMUX_CONFIG")
    if named:
        candidate = Path(named).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"$RFMUX_CONFIG points at {candidate}, which is not a file."
            )
        return candidate

    for candidate in _CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def init(destination: Path | str | None = None, *, force: bool = False) -> Path:
    """Copy the template somewhere you can edit it, and say where that is.

    Defaults to ``rfmux/config.yaml``, beside the template. Falls back to the
    per-user data directory when the package directory is not writable, which
    is the normal state of affairs for a ``pip install`` into site-packages.

    Refuses to clobber an existing file unless ``force`` — the whole point of
    the copy is that it holds edits worth keeping.
    """
    if destination is not None:
        target = Path(destination).expanduser()
    else:
        target = _CANDIDATES[0]
        if not os.access(target.parent, os.W_OK):
            target = _CANDIDATES[1]

    if target.exists() and not force:
        raise FileExistsError(
            f"{target} already exists. Edit it, or pass force=True to replace "
            f"it with a fresh copy of the template."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(_TEMPLATE, target)
    reload()
    return target


def get(key: str, default=None):
    """Look a dotted key up in the config file, e.g. ``get("store.directory")``.

    Returns ``default`` when there is no config file, when the key is absent,
    or when the key is present but empty — a commented-out setting and a
    missing one mean the same thing to a reader.
    """
    data = _load()
    node = data
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return default if node is None else node


def reload() -> None:
    """Drop the parsed config, so the next :func:`get` re-reads the file.

    For editing the file with a notebook already open.
    """
    _cache.clear()


def _load() -> dict:
    """Parse the active config file, once per file per session."""
    active = path()
    if active in _cache:
        return _cache[active]

    if active is None:
        _cache[active] = {}
        return _cache[active]

    try:
        with active.open() as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        # Naming the file matters more than usual here: the one in the package
        # directory and the one under ~/.local/share are both plausible, and
        # the error otherwise says nothing about which was being read.
        raise ValueError(f"Could not parse the rfmux config at {active}: {e}") from None

    if data is None:
        data = {}  # A file that is all comments, which the template nearly is.
    elif not isinstance(data, dict):
        raise ValueError(
            f"The rfmux config at {active} should be a mapping of settings, "
            f"but its top level is a {type(data).__name__}."
        )

    _cache[active] = data
    return data
