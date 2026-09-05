"""Mock mode must not inherit a module the mock has no tones on.

The Module spinbox is hidden whenever the connection is not hardware, so
in mock mode whatever it holds is a leftover from a board session that
the user can neither see nor correct.  The mock streams only the modules
carrying tones, and ``_auto_bias_kids`` in rfmux/mock/crs.py puts every
startup tone on module 1, so restoring module 2 there gives a permanently
blank viewer -- zero packets, zero loss, no error.  It must also not write
that module back, or one mock run would overwrite the module chosen for
the board.
"""

import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope import session_startup_dialog as ssd
from rfmux.tools.periscope.session_startup_dialog import UnifiedStartupDialog


@pytest.fixture
def saved(monkeypatch):
    """Settings in memory, so a test never touches the real config."""
    store = {"module": 2, "mode": None, "serial": None,
             "session_mode": None, "session_dir": None, "session_path": None}
    monkeypatch.setattr(ssd.settings, "get_last_module", lambda: store["module"])
    monkeypatch.setattr(ssd.settings, "set_last_module",
                        lambda m: store.__setitem__("module", m))
    monkeypatch.setattr(ssd.settings, "get_last_connection_mode", lambda: "mock")
    monkeypatch.setattr(ssd.settings, "set_last_connection_mode",
                        lambda m: store.__setitem__("mode", m))
    monkeypatch.setattr(ssd.settings, "get_last_crs_serial", lambda: "0156")
    monkeypatch.setattr(ssd.settings, "set_last_crs_serial",
                        lambda s: store.__setitem__("serial", s))
    monkeypatch.setattr(ssd.settings, "get_last_session_mode", lambda: "none")
    monkeypatch.setattr(ssd.settings, "set_last_session_mode",
                        lambda m: store.__setitem__("session_mode", m))
    monkeypatch.setattr(ssd.settings, "get_last_session_directory", lambda: "")
    monkeypatch.setattr(ssd.settings, "get_last_session_path", lambda: "")
    return store


def _mock_dialog(qt_app):
    dlg = UnifiedStartupDialog()
    dlg.rb_mock.setChecked(True)
    dlg.rb_no_session.setChecked(True)   # keeps file dialogs out of the test
    return dlg


def test_saved_module_is_restored_into_the_spinbox(qt_app, saved):
    """The leftover really is carried in -- that is the trap."""
    dlg = UnifiedStartupDialog()
    try:
        assert dlg.module_input.value() == 2
    finally:
        dlg.deleteLater()


def test_mock_mode_uses_the_module_the_mock_tones(qt_app, saved):
    dlg = _mock_dialog(qt_app)
    try:
        dlg.module_input.setValue(2)     # as restored from board use
        dlg._validate_and_accept()
        assert dlg.connection_mode == UnifiedStartupDialog.CONN_MOCK
        assert dlg.module == UnifiedStartupDialog.MOCK_MODULE == 1, \
            "mock mode used a module the mock puts no startup tones on"
        assert dlg.get_configuration()["module"] == 1
    finally:
        dlg.deleteLater()


def test_mock_mode_does_not_overwrite_the_board_module(qt_app, saved):
    dlg = _mock_dialog(qt_app)
    try:
        dlg.module_input.setValue(2)
        dlg._validate_and_accept()
        assert saved["module"] == 2, \
            "a mock run clobbered the module saved for the board"
    finally:
        dlg.deleteLater()


def test_hardware_mode_still_saves_the_chosen_module(qt_app, saved):
    """The clamp must not leak into the hardware path."""
    dlg = UnifiedStartupDialog()
    try:
        dlg.rb_hardware.setChecked(True)
        dlg.rb_no_session.setChecked(True)
        dlg.serial_input.setText("0156")
        dlg.module_input.setValue(3)
        dlg._validate_and_accept()
        assert dlg.module == 3
        assert saved["module"] == 3
    finally:
        dlg.deleteLater()
