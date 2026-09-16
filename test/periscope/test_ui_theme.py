"""Dark Mode dresses the whole application, not only the plots."""

import pytest

from test.qt_helpers import bare_periscope, spin

pytest.importorskip("PyQt6")

from PyQt6 import QtGui, QtWidgets  # noqa: E402

from rfmux.tools.periscope import app_runtime  # noqa: E402
from rfmux.tools.periscope.utils import apply_ui_theme  # noqa: E402

Role = QtGui.QPalette.ColorRole


def test_both_modes_are_fusion_and_reach_existing_widgets(qt_app):
    """Dark paints widgets made earlier dark; light paints them light,
    whatever palette the desktop supplied."""
    label = QtWidgets.QLabel("x")
    qt_app.setPalette(QtGui.QPalette(QtGui.QColor("#202830")))  # a dark desktop

    apply_ui_theme(True)
    spin(qt_app, 0.01)
    assert qt_app.style().objectName() == "fusion"
    assert qt_app.palette().color(Role.Window).lightness() < 128
    assert label.palette().color(Role.WindowText).lightness() > 128

    apply_ui_theme(False)
    spin(qt_app, 0.01)
    assert qt_app.style().objectName() == "fusion"
    assert qt_app.palette().color(Role.Window).lightness() > 128
    assert label.palette().color(Role.WindowText).lightness() < 128


def test_toggle_applies_the_ui_theme_before_rebuilding(qt_app, monkeypatch):
    """The View menu action reaches the application theme, not only the
    layout rebuild and the panels' plots."""
    p = bare_periscope(monkeypatch)
    calls = []
    monkeypatch.setattr(app_runtime, "apply_ui_theme", calls.append)
    monkeypatch.setattr(p, "_build_layout", lambda: calls.append("layout"))
    monkeypatch.setattr(p, "_update_dark_mode_in_child_windows", lambda: None)

    p._toggle_dark_mode(True)

    assert p.dark_mode is True
    assert calls == [True, "layout"]
