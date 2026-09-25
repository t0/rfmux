"""Save measurements and update the files they came from.

By default, new measurements go into ``~/rfmux_data/ipy_session_YYYYMMDD``.
Use ``set_output_directory(path)`` to save directly in a chosen folder for
this Python session; ``set_output_directory(None)`` restores the dated layout.

Saved blocks carry their path in ``file_metadata``, so fitting or bias finding
can update the same file. Payloads contain builtins and NumPy arrays; convert
rfmux objects with ``to_dict()`` before saving.
"""

from __future__ import annotations

import datetime
import os
import pickle
import warnings
from pathlib import Path

import numpy as np

from .. import config
from .sweep_results import _is_container

__all__ = [
    "FILE_VERSION",
    "METADATA_KEY",
    "save",
    "load",
    "maybe_save",
    "saved_path",
    "plain",
    "output_directory",
    "set_output_directory",
    "session_directory",
    "autosave_enabled",
    "set_autosave",
    "set_created_by",
]


# Version of file_metadata. Measurements and catalogs have their own versions.
FILE_VERSION = 1

METADATA_KEY = "file_metadata"

SESSION_PREFIX = "ipy_session_"

# Fallback when neither config nor environment specifies an output root.
DEFAULT_DIRECTORY = "~/rfmux_data"


# Session-lifetime overrides. None means "nobody has said", which is what sends
# the resolution below on to the environment and then the config file.
_output_directory: Path | None = None
_autosave: bool | None = None
_created_by: str | None = None


# ── where things go ──────────────────────────────────────────────────────────


def output_directory() -> Path:
    """Return the output root without creating it.

    Use the first available setting: :func:`set_output_directory`,
    ``$RFMUX_DATA_DIR``, ``store.directory`` in the config, or ``~/rfmux_data``.
    :func:`session_directory` adds a dated subfolder unless the path was set
    with :func:`set_output_directory`.
    """
    if _output_directory is not None:
        return _output_directory

    named = os.environ.get("RFMUX_DATA_DIR")
    if named:
        return Path(named).expanduser()

    configured = config.get("store.directory")
    if configured:
        return Path(configured).expanduser()

    return Path(DEFAULT_DIRECTORY).expanduser()


def set_output_directory(directory: Path | str | None) -> None:
    """Set the output folder for this Python session, without a dated subfolder.

    Pass None to restore dated folders under the environment or config root.
    """
    global _output_directory
    _output_directory = None if directory is None else Path(directory).expanduser()


def session_directory(*, create: bool = True) -> Path:
    """The active output folder, made if requested and it isn't there.

    An explicit :func:`set_output_directory` destination is used directly.
    Otherwise use today's dated folder under :func:`output_directory`;
    a kernel left open overnight switches to the next day's folder.
    """
    folder = output_directory()
    if _output_directory is None:
        folder = folder / f"{SESSION_PREFIX}{_now():%Y%m%d}"
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    return folder


def autosave_enabled() -> bool:
    """Do measurements save themselves when no ``save=`` says otherwise?

    Resolved like :func:`output_directory`: :func:`set_autosave`, then
    ``$RFMUX_AUTOSAVE``, then ``store.autosave`` in your config file, then on.
    """
    if _autosave is not None:
        return _autosave

    named = os.environ.get("RFMUX_AUTOSAVE")
    if named is not None:
        return named.strip().lower() not in ("0", "false", "no", "off", "")

    return bool(config.get("store.autosave", True))


def set_autosave(enabled: bool | None) -> None:
    """Turn automatic saving on or off for the rest of this Python session.

    ``None`` hands the decision back to the environment and config file. A
    ``save=`` argument on an individual call beats this either way.
    """
    global _autosave
    _autosave = None if enabled is None else bool(enabled)


def set_created_by(who: str | None) -> None:
    """Declare what is driving these measurements, for ``file_metadata``.

    Detected as ``"ipython"`` or ``"script"`` if nobody says. Periscope and the
    CLI name themselves, so a file found later says which tool made it.
    """
    global _created_by
    _created_by = who


# ── saving and loading ───────────────────────────────────────────────────────


def _save(
    data,
    measurement_type: str | None = None,
    *,
    label: str | None = None,
    directory: Path | str | None = None,
    module=None,
    new: bool = False,
) -> Path:
    """Write data to a pickle and return its path.

    If ``file_metadata`` contains a path, update that file. Pass ``new=True``
    to create a separate timestamped file in ``directory`` or the session
    folder. Updating one module preserves the other modules in a readable
    source container.

    ``measurement_type`` starts the filename and may be omitted when saved
    metadata provides it. ``label`` is appended to new filenames; an existing
    filename is kept. ``module`` supplies the metadata module number when
    needed for a standalone dictionary.
    """
    existing = _metadata_of(data)

    if measurement_type is None:
        measurement_type = existing.get("measurement_type") if existing else None
        if measurement_type is None:
            raise ValueError(
                "measurement_type is needed to name the file. It can only be "
                "left out for data that was saved or loaded before, which "
                "carries its own."
            )

    if label is None and existing:
        label = existing.get("label")

    reused = not new and existing.get("path")
    if reused:
        target = Path(existing["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        folder = (
            Path(directory).expanduser() if directory is not None
            else session_directory()
        )
        folder.mkdir(parents=True, exist_ok=True)
        target = _unused(folder / _filename(measurement_type, label, _now()))

    _stamp(
        data,
        measurement_type=measurement_type,
        path=target,
        label=label,
        module=module,
        created=existing.get("created") if reused else None,
    )

    # Before the file is opened for writing: opening it "wb" truncates it, and
    # _spliced has to read what is there.
    payload = _spliced(data, target) if reused else data

    with target.open("wb") as f:
        pickle.dump(payload, f)
    return target


def _spliced(data, target: Path):
    """Replace one module in its saved container, preserving the other modules.

    Match by module number. Return ``data`` unchanged if the source file is
    missing, unreadable, not a container, or has no matching module.
    """
    if _is_container(data) or not isinstance(data, dict):
        return data
    if "results" not in data or not target.exists():
        return data

    try:
        with target.open("rb") as f:
            on_disk = pickle.load(f)
    except Exception:
        # Unreadable or half-written: the data in hand is better than
        # nothing, and refusing to save it would be the worse failure.
        return data

    if not _is_container(on_disk):
        return data

    for module_id, module_output in on_disk.items():
        if module_output.get("module") == data.get("module"):
            on_disk[module_id] = data
            return on_disk
    return data


# The public name. `maybe_save` takes a `save` argument, so it needs a way to
# reach the function that is not the name that argument shadows.
save = _save


def load(path: Path | str):
    """Load a pickle and update its metadata to the path it was loaded from.

    This lets later saves update the opened file even if it was moved or renamed.
    """
    path = Path(path).expanduser()
    with path.open("rb") as f:
        data = pickle.load(f)

    for block, _ in _blocks(data, module=None):
        metadata = block.get(METADATA_KEY)
        if isinstance(metadata, dict):
            metadata["path"] = str(path.resolve())
    return data


def maybe_save(
    data,
    measurement_type: str,
    *,
    save: bool | None = None,
    label: str | None = None,
    module=None,
) -> Path | None:
    """Save when requested, using the autosave setting when ``save`` is None.

    On a write failure, warn and return None. The caller still has the data
    in memory and can retry with :func:`save`.
    """
    if save is False:
        return None
    if save is None and not autosave_enabled():
        return None

    try:
        return _save(data, measurement_type, label=label, module=module)
    except Exception as e:
        warnings.warn(
            f"Could not save this {measurement_type} to "
            f"{output_directory()}: {e}. The data is still in hand — "
            f"rfmux.tuning.store.save(result, {measurement_type!r}, "
            f"directory=...) will write it somewhere else.",
            stacklevel=2,
        )
        return None


def saved_path(data) -> Path | None:
    """Where ``data`` was last written, or ``None`` if it never has been."""
    recorded = _metadata_of(data).get("path")
    return Path(recorded) if recorded else None


def plain(value):
    """Recursively convert NumPy values in settings to Python scalars and lists.

    Dictionary keys become strings, arrays and tuples become lists, and NumPy
    scalars become Python scalars. Other values are returned unchanged.
    Call this on settings, not measured traces that should remain arrays.
    """
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


# ── naming ───────────────────────────────────────────────────────────────────


def _now() -> datetime.datetime:
    """One place to get the time, so tests have one place to freeze it."""
    return datetime.datetime.now()


def _filename(measurement_type: str, label: str | None, when) -> str:
    """``{type}_{YYYYMMDD}_{HHMMSS}_{label}.pkl``, label and all if there is one.

    The date is in the name as well as on the folder, so a file still says when
    it was taken after being copied out of the folder that said so.
    """
    parts = [measurement_type, when.strftime("%Y%m%d_%H%M%S")]
    if label:
        parts.append(str(label).replace(" ", "_").replace("/", "_"))
    return "_".join(parts) + ".pkl"


def _unused(target: Path) -> Path:
    """Return an unused path, adding a numeric suffix if the target exists.

    Try suffixes ``_1`` through ``_999``, then raise FileExistsError.
    """
    if not target.exists():
        return target
    for n in range(1, 1000):
        candidate = target.with_name(f"{target.stem}_{n}{target.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"A thousand files already share the name {target.name}.")


# ── the file_metadata block ──────────────────────────────────────────────────


def _stamp(data, *, measurement_type, path, label, module, created) -> None:
    """Add file metadata to each module block or standalone dictionary.

    Preserve ``created`` when updating a saved file and record the current
    time in ``updated``. A new file gets the current time as ``created``.
    """
    stamped_at = _now().isoformat(timespec="seconds")
    for block, block_module in _blocks(data, module):
        metadata = {
            "file_version": FILE_VERSION,
            "measurement_type": measurement_type,
            "path": str(path.resolve()),
            "created": created or stamped_at,
            "created_by": _who(),
            "rfmux_version": _version(),
        }
        if created:
            metadata["updated"] = stamped_at
        if block_module is not None:
            metadata["module"] = block_module
        if label:
            metadata["label"] = label
        block[METADATA_KEY] = metadata


def _metadata_of(data) -> dict:
    """The first ``file_metadata`` block in ``data``, or an empty dict.

    First rather than all: every block in one file carries the same path and
    label, duplicated so that nobody has to index up a level to find them.
    """
    for block, _ in _blocks(data, module=None):
        metadata = block.get(METADATA_KEY)
        if isinstance(metadata, dict):
            return metadata
    return {}


def _blocks(data, module):
    """Yield dictionaries to stamp, paired with their module numbers.

    For a measurement container, yield each module block and its recorded
    module number. For a standalone dictionary, use an integer ``module``
    argument if supplied, otherwise its own ``module`` field.
    Raise TypeError for other input types.
    """
    if _is_container(data):
        for module_output in data.values():
            yield module_output, module_output.get("module")
        return

    if isinstance(data, dict):
        # One module's output says which module it is; a to_dict() does
        # not, and falls back to whatever the caller passed.
        if isinstance(module, (int, np.integer)):
            yield data, int(module)
        else:
            yield data, data.get("module")
        return

    if isinstance(data, (list, tuple)):
        raise TypeError(
            f"Cannot save a {type(data).__name__} of {len(data)}. Several "
            f"modules come back from a driver as one dict keyed by module "
            f"identifier, not as a list — if this is an old take_netanal "
            f"result, re-measure it, and if you assembled it yourself, key it "
            f"by module."
        )

    raise TypeError(
        f"Cannot save a {type(data).__name__}. Measurement results are dicts, "
        f"and rfmux classes go in through their .to_dict() — pickling the "
        f"class itself records its import path and skips its constructor on "
        f"the way back."
    )


def _who() -> str:
    """Whether this is a notebook or a script, unless something said otherwise."""
    if _created_by is not None:
        return _created_by
    try:
        from IPython import get_ipython
    except ImportError:
        return "script"
    return "ipython" if get_ipython() is not None else "script"


def _version() -> str:
    """The rfmux that wrote the file. Imported late to dodge the import cycle."""
    import rfmux

    return getattr(rfmux, "__version__", "unknown")
