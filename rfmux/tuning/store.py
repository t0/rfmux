"""The output folder: where a measurement goes, and how it finds its way back.

A measurement that only exists in a notebook variable is one kernel restart away
from being a measurement that never happened. So the drivers save by default:
:func:`~rfmux.algorithms.measurement.multisweep.multisweep` and its neighbours
take ``save=`` and hand what they produced to :func:`maybe_save` on the way out.

**The folder is keyed by date, not by session.** Periscope makes one folder per
session because clicking *New Session* is an unambiguous moment. From a notebook
there is no such moment — you open a kernel on Monday and are still in it on
Wednesday — so rfmux makes one ``ipy_session_YYYYMMDD`` folder per day inside
your output directory and puts the date and time in the filename instead::

    ~/rfmux_data/ipy_session_20260904/multisweep_20260904_142231_cooldown3.pkl

That is also why the names look different from Periscope's
``multisweep_module1_142231.pkl``: two tools writing two layouts should be
telling you apart at a glance, not almost-matching.

**Files know where they live.** Every saved payload carries a ``file_metadata``
block recording the path it was written to, so an analysis that modifies data in
place — :func:`~rfmux.tuning.fits.fit_sweeps` writing fits into the sweep entries
— can save it back over the file it came from without anyone passing a path
around. It is stamped inside each module's block rather than at the top of the
file, so you reach it wherever you were already working:
``sweeps["crs0042_rmod2"]["file_metadata"]``.

**Files outlive rfmux.** The payload is builtins and ndarrays: ``pickle.load``
gets you a usable result on a machine with no rfmux installed, and
``file_metadata`` is an ordinary key that every existing reader ignores. Classes
go in through their ``to_dict()`` — never pickled directly, because that records
the class's import path and skips ``__init__`` on the way back, so a renamed
class orphans old files and a malformed one restores into a state the class
would have refused to build.

Nothing here needs a board or a GUI.
"""

from __future__ import annotations

import datetime
import os
import pickle
import warnings
from collections.abc import Sequence
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


# Bumped when the file_metadata block changes shape in a way a reader cannot
# absorb. It versions the envelope, not the measurement: what a sweep looks like
# inside is RESULTS_SCHEMA_VERSION's business, and a catalog's is its own.
FILE_VERSION = 1

METADATA_KEY = "file_metadata"

SESSION_PREFIX = "ipy_session_"

# Used when there is no config file and no environment override. Deliberately
# somewhere you will trip over it — data you cannot find is data you do not
# have, which is the argument against ~/.local/share for this one.
DEFAULT_DIRECTORY = "~/rfmux_data"


# Session-lifetime overrides. None means "nobody has said", which is what sends
# the resolution below on to the environment and then the config file.
_output_directory: Path | None = None
_autosave: bool | None = None
_created_by: str | None = None


# ── where things go ──────────────────────────────────────────────────────────


def output_directory() -> Path:
    """The directory the dated session folders are made in.

    Resolved highest-first: :func:`set_output_directory`, then
    ``$RFMUX_DATA_DIR``, then ``store.directory`` in your config file, then
    ``~/rfmux_data``. Not created here — :func:`session_directory` does that,
    so merely asking where output *would* go never leaves a folder behind.
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
    """Send output somewhere else for the rest of this Python session.

    The notebook knob — one line at the top of a cooldown's notebook, no file to
    edit. ``None`` puts it back to whatever the environment and config say.
    """
    global _output_directory
    _output_directory = None if directory is None else Path(directory).expanduser()


def session_directory(*, create: bool = True) -> Path:
    """Today's folder inside :func:`output_directory`, made if it isn't there.

    Named for the date it was made, not for when you started working: a kernel
    left open overnight starts writing into tomorrow's folder tomorrow, which is
    where you would look for it.
    """
    folder = output_directory() / f"{SESSION_PREFIX}{_now():%Y%m%d}"
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
    """Write ``data`` to a pickle and return where it went.

    ``measurement_type`` leads the filename and is how you recognize the file
    later; it can be omitted only when ``data`` has been saved before and can
    say what it is itself.

    If ``data`` already knows its path — it was saved earlier, or loaded from
    disk — this overwrites that file, which is what makes
    ``fit_sweeps(sweeps, save=True)`` update the sweep it fitted rather than
    leaving a near-copy beside it. Pass ``new=True`` for a fresh timestamped
    file when you want to keep what is already on disk.

    ``label`` is your name for this measurement and goes on the end of the
    filename. ``module`` supplies the module number for payloads that do not
    record it themselves — the netanal ones.
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
    """What actually goes in the file when ``data`` is only part of it.

    :func:`~rfmux.tuning.fits.fit_sweeps` works on **one module's envelope**,
    ``sweeps[module_id]``, and that envelope carries the path of the file it was
    written to — a file holding the whole container, every module of it. Writing
    the envelope over that path would silently throw the other modules away, and
    would leave even a one-module file no longer shaped like a sweep result.

    So a re-save of an envelope reads the container back, puts the envelope in
    the place it came from — matched on module number, which
    :func:`~rfmux.tuning.sweep_results.merge_modules` guarantees is unique
    within a container — and writes the whole thing.
    """
    if _is_container(data) or not isinstance(data, dict):
        return data
    if "results" not in data or not target.exists():
        return data

    try:
        with target.open("rb") as f:
            on_disk = pickle.load(f)
    except Exception:
        # Unreadable or half-written: the envelope in hand is better than
        # nothing, and refusing to save it would be the worse failure.
        return data

    if not _is_container(on_disk):
        return data

    for module_id, envelope in on_disk.items():
        if envelope.get("module") == data.get("module"):
            on_disk[module_id] = data
            return on_disk
    return data


# The public name. `maybe_save` takes a `save` argument, so it needs a way to
# reach the function that is not the name that argument shadows.
save = _save


def load(path: Path | str):
    """Read a pickle back, and tell it where it actually is now.

    Files get copied off the acquisition machine and renamed. The
    ``file_metadata`` path recorded at write time is corrected to where the file
    was really found, so saving it again after a fit writes back to the file you
    opened rather than to a path on a machine you may not even be on.
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
    """The drivers' entry point: honour ``save=``, and never lose data over it.

    ``save=None`` means "whatever :func:`autosave_enabled` says", which is how a
    config file turns autosave off without any call site changing.

    A failed write warns rather than raises. A twenty-minute sweep that made it
    back into memory should not be thrown away because the output directory is
    read-only — you still have the data, and you can save it somewhere else.
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
    """Builtins all the way down: numpy scalars and arrays become floats and lists.

    What the ``to_dict`` methods run their loosely-typed corners through — the
    ``settings`` dicts on the reports hold whatever a caller passed, which is
    routinely a ``np.float64`` picked out of an array. Left as it is that
    pickles as numpy, and a file you need numpy to read is not the
    plain-builtins file this module promises.

    Measured data is exempt and stays as ndarrays: a sweep's IQ has no business
    being a list of a hundred thousand Python floats.
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
    """A path nothing is at yet, suffixing ``_1``, ``_2`` … if need be.

    Two measurements can finish in the same second — the amplitude steps of a
    ladder, saved individually, routinely do — and the loser should not silently
    land on top of the winner.
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
    """Write a ``file_metadata`` block into every part of ``data`` that takes one.

    ``created`` carries the original timestamp through a re-save: the file
    records when the measurement was *taken*, and gains an ``updated`` field to
    say when an analysis last wrote to it. A fit run three days later should not
    make the sweep look three days younger than it is.
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
    """Yield ``(mapping_to_stamp, module_number_or_None)`` for one payload.

    The four shapes a saveable thing arrives in:

    * a packed sweep container, ``{module_id: envelope}`` — one block per
      envelope, each of which already records its own module. Stamping inside
      the envelopes rather than at the top is what keeps
      :func:`~rfmux.tuning.sweep_results._is_container` true of the result, and
      keeps ``find_resonances_in_netanal``'s "all-integer keys means modules"
      test from tripping over a string key.
    * a list of netanal results, what ``take_netanal(module=[1, 2])`` returns —
      one block each, numbered from the ``module`` argument, since a netanal
      result records nothing about which module produced it.
    * a single netanal result, or any class's ``to_dict()`` — one block, at the
      top.
    * anything else — refused, because a payload that cannot be stamped cannot
      be found again, and silently writing one is worse than not writing it.
    """
    if _is_container(data):
        for envelope in data.values():
            yield envelope, envelope.get("module")
        return

    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        modules = module if isinstance(module, (list, tuple)) else None
        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                raise TypeError(
                    f"Cannot save a list whose entry {i} is a "
                    f"{type(entry).__name__} — expected a result dict, as "
                    f"take_netanal(module=[...]) returns."
                )
            yield entry, modules[i] if modules and i < len(modules) else None
        return

    if isinstance(data, dict):
        # One module's envelope says which module it is; a netanal result does
        # not, and falls back to whatever the caller passed.
        if isinstance(module, (int, np.integer)):
            yield data, int(module)
        else:
            yield data, data.get("module")
        return

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
