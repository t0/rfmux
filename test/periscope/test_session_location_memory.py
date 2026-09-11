"""Where sessions are saved is chosen once, then kept.

A session folder needs a name every time, but the folder it is created in is
the same folder run after run.  Periscope asks for it on the first start that
needs it, remembers it, and only asks again when the user goes looking for it
in the Session menu.

Passing "" as a file dialog's start directory is not neutral.  Qt falls back
to its own process-global last-visited directory, so a dialog opened with ""
quietly follows whatever other file dialog was opened most recently — which is
how it ends up pointing at a temp directory nobody chose.
"""

import types
from unittest.mock import MagicMock

import pytest


pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from rfmux.tools.periscope import app as periscope_app  # noqa: E402
from rfmux.tools.periscope import session_startup_dialog as ssd  # noqa: E402
from rfmux.tools.periscope import settings  # noqa: E402


def _capture_root_chooser(monkeypatch, returns):
    """Patch the session-location chooser; record where it opened.

    Both bindings: app.py calls it directly, and through ``session_root``.
    """
    seen = {}

    def fake(parent, start_dir=""):
        seen["start_dir"] = start_dir
        return None if returns is None else str(returns)

    monkeypatch.setattr(ssd, "choose_session_root", fake)
    monkeypatch.setattr(periscope_app, "choose_session_root", fake)
    return seen


def _capture_folder_dialog(monkeypatch, returns):
    """Patch the plain folder chooser; record where it opened."""
    seen = {}

    def fake(parent, caption, directory, options):
        seen["start_dir"] = directory
        return str(returns)

    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory",
                        staticmethod(fake))
    return seen


def _app():
    app = types.SimpleNamespace(session_manager=MagicMock())
    app.statusBar = MagicMock()
    return app


def test_new_session_uses_the_remembered_location_without_asking(
        qt_app, tmp_path, monkeypatch):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    settings.set_session_root(str(sessions))

    seen = _capture_root_chooser(monkeypatch, tmp_path / "unwanted")
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("session_x", True)))

    app = _app()
    periscope_app.Periscope._start_new_session(app)

    assert seen == {}, "asked for a location that was already chosen"
    app.session_manager.start_session.assert_called_once_with(
        str(sessions), "session_x")


def test_new_session_asks_once_when_no_location_is_known(
        qt_app, tmp_path, monkeypatch):
    chosen = tmp_path / "chosen"
    chosen.mkdir()

    seen = _capture_root_chooser(monkeypatch, chosen)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("session_x", True)))

    app = _app()
    periscope_app.Periscope._start_new_session(app)

    assert seen["start_dir"] == ""
    assert settings.get_session_root() == str(chosen), \
        "the chosen location must be remembered, or the next run asks again"
    app.session_manager.start_session.assert_called_once_with(
        str(chosen), "session_x")


def test_cancelling_the_location_does_not_start_a_session(
        qt_app, monkeypatch):
    _capture_root_chooser(monkeypatch, None)

    app = _app()
    periscope_app.Periscope._start_new_session(app)

    app.session_manager.start_session.assert_not_called()
    assert settings.get_session_root() == ""


def test_the_menu_changes_the_location_for_later_sessions(
        qt_app, tmp_path, monkeypatch):
    old = tmp_path / "old"
    old.mkdir()
    new = tmp_path / "new"
    new.mkdir()
    settings.set_session_root(str(old))

    seen = _capture_root_chooser(monkeypatch, new)

    app = _app()
    periscope_app.Periscope._change_session_root(app)

    assert seen["start_dir"] == str(old), \
        "the chooser must open at the location in force, not wherever Qt drifted"
    assert settings.get_session_root() == str(new)
    app.session_manager.start_session.assert_not_called()


def test_load_session_opens_where_sessions_live(
        qt_app, tmp_path, monkeypatch):
    base = tmp_path / "sessions"
    session = base / "session_20260101_000000"
    session.mkdir(parents=True)
    settings.set_session_root(str(base))

    seen = _capture_folder_dialog(monkeypatch, session)

    app = _app()
    app.session_manager.load_session.return_value = True
    app._restore_mock_config_from_session = lambda: None
    periscope_app.Periscope._load_session(app)

    assert seen["start_dir"] == str(base)
    assert settings.get_last_session_path() == str(session)


def test_load_session_opens_beside_the_session_loaded_last(
        qt_app, tmp_path, monkeypatch):
    """Sessions are usually loaded from where the last one came from."""
    base = tmp_path / "sessions"
    base.mkdir()
    archive = tmp_path / "archive"
    previous = archive / "session_20240101_000000"
    previous.mkdir(parents=True)
    settings.set_session_root(str(base))
    settings.set_last_session_path(str(previous))

    seen = _capture_folder_dialog(monkeypatch, previous)

    app = _app()
    app.session_manager.load_session.return_value = True
    app._restore_mock_config_from_session = lambda: None
    periscope_app.Periscope._load_session(app)

    assert seen["start_dir"] == str(archive)


def test_loading_from_elsewhere_does_not_move_where_new_sessions_go(
        qt_app, tmp_path, monkeypatch):
    """An archive is a place to read from, not the place to write to."""
    base = tmp_path / "sessions"
    base.mkdir()
    archive = tmp_path / "archive" / "session_20240101_000000"
    archive.mkdir(parents=True)
    settings.set_session_root(str(base))

    _capture_folder_dialog(monkeypatch, archive)

    app = _app()
    app.session_manager.load_session.return_value = True
    app._restore_mock_config_from_session = lambda: None
    periscope_app.Periscope._load_session(app)

    assert settings.get_session_root() == str(base)


def test_a_failed_load_is_not_remembered(qt_app, tmp_path, monkeypatch):
    good = tmp_path / "good"
    good.mkdir()
    bad = tmp_path / "bad"
    bad.mkdir()
    settings.set_session_root(str(good))

    _capture_folder_dialog(monkeypatch, bad)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: None))

    app = _app()
    app.session_manager.load_session.return_value = False
    periscope_app.Periscope._load_session(app)

    assert settings.get_session_root() == str(good)
    assert settings.get_last_session_path() == ""
