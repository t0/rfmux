"""The startup dialog names the session location instead of asking for it.

Naming a new session is a per-run decision; where session folders live is not.
The dialog asks for the location on the first start that needs one, then shows
it, and points at the Session menu entry that changes it.
"""

import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope import session_startup_dialog as ssd  # noqa: E402
from rfmux.tools.periscope.session_startup_dialog import (  # noqa: E402
    SESSION_ROOT_MENU_PATH, UnifiedStartupDialog)
from rfmux.tools.periscope import settings  # noqa: E402


@pytest.fixture
def chooser(monkeypatch):
    """Patch the location chooser; record calls, return a canned answer."""
    calls = []

    def fake(parent, start_dir=""):
        calls.append(start_dir)
        return fake.returns

    fake.returns = None
    fake.calls = calls
    monkeypatch.setattr(ssd, "choose_session_root", fake)
    return fake


def _new_session_dialog():
    dlg = UnifiedStartupDialog()
    dlg.rb_mock.setChecked(True)          # no serial to validate
    dlg.rb_new_session.setChecked(True)
    return dlg


def test_a_known_location_is_used_without_asking(qt_app, tmp_path, chooser):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    settings.set_session_root(str(sessions))

    dlg = _new_session_dialog()
    try:
        dlg._validate_and_accept()
        assert chooser.calls == [], "asked for a location already chosen"
        assert dlg.get_configuration()["session_path"] == str(sessions)
    finally:
        dlg.deleteLater()


def test_the_location_is_asked_for_once_and_kept(qt_app, tmp_path, chooser):
    chosen = tmp_path / "chosen"
    chosen.mkdir()
    chooser.returns = str(chosen)

    dlg = _new_session_dialog()
    try:
        dlg._validate_and_accept()
        assert chooser.calls == [""]
        assert dlg.get_configuration()["session_path"] == str(chosen)
        assert settings.get_session_root() == str(chosen)
    finally:
        dlg.deleteLater()


def test_cancelling_the_location_keeps_the_dialog_open(qt_app, chooser):
    chooser.returns = None

    dlg = _new_session_dialog()
    try:
        dlg._validate_and_accept()
        assert dlg.result() == 0, "accepted without a session location"
        assert settings.get_session_root() == ""
    finally:
        dlg.deleteLater()


def test_the_dialog_shows_the_location_and_how_to_change_it(
        qt_app, tmp_path, chooser):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    settings.set_session_root(str(sessions))

    dlg = _new_session_dialog()
    try:
        text = dlg.session_root_label.text()
        assert str(sessions) in text
        assert SESSION_ROOT_MENU_PATH in text
    finally:
        dlg.deleteLater()


def test_the_first_run_still_says_where_to_change_it(qt_app, chooser):
    dlg = _new_session_dialog()
    try:
        assert SESSION_ROOT_MENU_PATH in dlg.session_root_label.text()
    finally:
        dlg.deleteLater()
