"""The session dialogs must remember where you put sessions.

Two halves, and both were broken in app.py while the startup dialog got them
right: the folder chooser has to OPEN at the last used directory, and it has to
RECORD what you chose.

Passing "" as the start directory is not neutral.  Qt falls back to its own
process-global last-visited directory, so the dialog quietly follows whatever
other file dialog was opened most recently — which is how it ends up pointing
at a temp directory nobody chose.
"""

import os
import types
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from rfmux.tools.periscope import settings  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _restore_settings():
    """These write the real per-user QSettings, so put them back."""
    directory = settings.get_last_session_directory()
    path = settings.get_last_session_path()
    yield
    settings.set_last_session_directory(directory)
    settings.set_last_session_path(path)


def _capture_dialog(monkeypatch, returns):
    """Patch the folder chooser; record the directory it was opened at."""
    seen = {}

    def fake(parent, caption, directory, options):
        seen["start_dir"] = directory
        return str(returns)

    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory",
                        staticmethod(fake))
    return seen


def test_new_session_opens_at_and_records_the_directory(
        qt_app, tmp_path, monkeypatch):
    from rfmux.tools.periscope.app import Periscope

    remembered = tmp_path / "remembered"
    remembered.mkdir()
    chosen = tmp_path / "chosen"
    chosen.mkdir()
    settings.set_last_session_directory(str(remembered))

    seen = _capture_dialog(monkeypatch, chosen)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("session_x", True)))

    app = types.SimpleNamespace(session_manager=MagicMock())
    Periscope._start_new_session(app)

    assert seen["start_dir"] == str(remembered), \
        "dialog must open at the last session directory, not wherever Qt drifted"
    assert settings.get_last_session_directory() == str(chosen), \
        "the chosen directory must be remembered for next time"
    app.session_manager.start_session.assert_called_once()


def test_load_session_records_the_parent_directory(
        qt_app, tmp_path, monkeypatch):
    """Loading names a session FOLDER, so the base directory is its parent."""
    from rfmux.tools.periscope.app import Periscope

    base = tmp_path / "sessions"
    session = base / "session_20260101_000000"
    session.mkdir(parents=True)
    settings.set_last_session_directory(str(base))

    seen = _capture_dialog(monkeypatch, session)

    app = types.SimpleNamespace(session_manager=MagicMock())
    app.session_manager.load_session.return_value = True
    app._restore_mock_config_from_session = lambda: None
    Periscope._load_session(app)

    assert seen["start_dir"] == str(base)
    assert settings.get_last_session_directory() == str(base)
    assert settings.get_last_session_path() == str(session)


def test_failed_load_does_not_move_the_remembered_directory(
        qt_app, tmp_path, monkeypatch):
    """A load that fails must not repoint the setting at the bad folder."""
    from rfmux.tools.periscope.app import Periscope

    good = tmp_path / "good"
    good.mkdir()
    bad = tmp_path / "bad"
    bad.mkdir()
    settings.set_last_session_directory(str(good))

    _capture_dialog(monkeypatch, bad)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: None))

    app = types.SimpleNamespace(session_manager=MagicMock())
    app.session_manager.load_session.return_value = False
    Periscope._load_session(app)

    assert settings.get_last_session_directory() == str(good)
