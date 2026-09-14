"""A session folder as Periscope and the recorder share it: its name,
the metadata file beside its exports, and the export listing the
session browser reads."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import List, Optional

METADATA_FILE = "session_metadata.json"
FOLDER_FORMAT = "session_%Y%m%d_%H%M%S"


def folder_name(now: Optional[datetime.datetime] = None) -> str:
    """``session_YYYYMMDD_HHMMSS``."""
    return (now or datetime.datetime.now()).strftime(FOLDER_FORMAT)


def is_session(path) -> bool:
    return (Path(path) / METADATA_FILE).exists()


def new_metadata(folder) -> dict:
    """The metadata of a session folder just made."""
    folder = Path(folder)
    return {
        "created": datetime.datetime.now().isoformat(),
        "folder_name": folder.name,
        "base_path": str(folder.resolve().parent),
        "exports": [],
        "screenshots": [],
    }


def load_metadata(session) -> dict:
    """The session's metadata, or ``{}`` when it has none it can read."""
    try:
        with open(Path(session) / METADATA_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_metadata(session, metadata: dict) -> None:
    with open(Path(session) / METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def export_filename(data_type: str, identifier: str, ext: str = ".pkl",
                    now: Optional[datetime.datetime] = None) -> str:
    """``<type>_<identifier>_HHMMSS<ext>``: the time alone, the folder
    carrying the date."""
    stamp = (now or datetime.datetime.now()).strftime("%H%M%S")
    clean = identifier.replace(" ", "_").replace("/", "_")
    return f"{data_type}_{clean}_{stamp}{ext}"


def export_entry(filename: str, data_type: str, identifier: str,
                 timestamp: Optional[str] = None) -> dict:
    """An entry of the metadata's ``exports`` list."""
    return {
        "filename": filename,
        "data_type": data_type,
        "identifier": identifier,
        "timestamp": timestamp or datetime.datetime.now().isoformat(),
    }


def open_session(path=None, base=None) -> Path:
    """The session folder at *path*, or a new one under *base* (the
    working directory), with its metadata file."""
    if path is None:
        path = Path(base or ".") / folder_name()
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    if not is_session(path):
        save_metadata(path, new_metadata(path))
    return path


def register_export(session, filename: str, data_type: str,
                    identifier: str, timestamp: Optional[str] = None) -> None:
    """List a file in the session's exports."""
    metadata = load_metadata(session)
    metadata.setdefault("exports", []).append(
        export_entry(filename, data_type, identifier, timestamp))
    save_metadata(session, metadata)


def exports(session, data_type: Optional[str] = None,
            identifier: Optional[str] = None) -> List[dict]:
    """The listed exports, of *data_type* and *identifier* when given,
    newest last by the listing's timestamp (a copied folder keeps no
    file times); an entry whose file is gone is left out."""
    found = [e for e in load_metadata(session).get("exports", [])
             if (data_type is None or e.get("data_type") == data_type)
             and (identifier is None or e.get("identifier") == identifier)
             and (Path(session) / str(e.get("filename", ""))).is_file()]
    return sorted(found, key=lambda e: str(e.get("timestamp", "")))


def latest_export(session, data_type: str, identifier: str) -> Optional[Path]:
    """The newest listed export of *data_type* and *identifier*, or
    None."""
    found = exports(session, data_type, identifier)
    return Path(session) / found[-1]["filename"] if found else None


def newest_session(base) -> Optional[Path]:
    """The newest session folder under *base*, by the stamp in its
    name, or None."""
    found = sorted(p for p in Path(base).expanduser().glob("session_*")
                   if p.is_dir() and is_session(p))
    return found[-1] if found else None
