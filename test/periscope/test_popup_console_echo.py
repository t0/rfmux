"""A warning or error pop-up is also printed to the console: a terminal
log or a remote session keeps what the dialog said."""
import pytest

pytest.importorskip("PyQt6")

from PyQt6 import QtCore, QtWidgets  # noqa: E402

from rfmux.tools.periscope.utils import echo_popups_to_console  # noqa: E402


def _close_the_message_box():
    for w in QtWidgets.QApplication.topLevelWidgets():
        if isinstance(w, QtWidgets.QMessageBox) and w.isVisible():
            w.close()


def test_static_error_box_reaches_the_console(qt_app, capsys):
    echo_popups_to_console(qt_app)
    QtCore.QTimer.singleShot(0, _close_the_message_box)
    QtWidgets.QMessageBox.critical(
        None, "Jupyter Server Error", "Failed to start Jupyter Lab: timeout")
    err = capsys.readouterr().err
    assert "ERROR: Jupyter Server Error: Failed to start Jupyter Lab: timeout" in err


def test_warning_box_built_by_hand_reaches_the_console(qt_app, capsys):
    echo_popups_to_console(qt_app)
    box = QtWidgets.QMessageBox(
        QtWidgets.QMessageBox.Icon.Warning, "Fitting", "3 fits failed")
    box.setInformativeText("channels 4, 9, 12")
    box.show()
    box.close()
    err = capsys.readouterr().err
    assert "WARNING: Fitting: 3 fits failed\nchannels 4, 9, 12" in err


def test_installing_twice_prints_once(qt_app, capsys):
    echo_popups_to_console(qt_app)
    echo_popups_to_console(qt_app)
    box = QtWidgets.QMessageBox(
        QtWidgets.QMessageBox.Icon.Warning, "Once", "only")
    box.show()
    box.close()
    assert capsys.readouterr().err.count("Once: only") == 1


def test_a_question_is_not_echoed(qt_app, capsys):
    echo_popups_to_console(qt_app)
    box = QtWidgets.QMessageBox(
        QtWidgets.QMessageBox.Icon.Question, "Overwrite?", "the file exists")
    box.show()
    box.close()
    assert "Overwrite?" not in capsys.readouterr().err
