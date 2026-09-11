"""The main window builds.

MainPlotPanel lays out the action buttons the Periscope window owns, so a
button removed from app.py but left in that layout is an AttributeError at
startup and nothing else catches it: every other Periscope test builds a
panel or a stand-in, never the window.
"""

import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope.app import Periscope  # noqa: E402


def test_offline_window_comes_up_with_its_action_buttons(qt_app):
    p = Periscope("OFFLINE", module=1, chan_str="1", skip_startup_dialog=True)
    try:
        panel = p.main_plot_panel
        assert panel is not None
        buttons = panel.findChildren(type(p.btn_netanal))
        assert p.btn_netanal in buttons
    finally:
        p.close()


def test_window_opens_wide_but_within_its_screen(qt_app):
    p = Periscope("OFFLINE", module=1, chan_str="1", skip_startup_dialog=True)
    try:
        available = p.screen().availableGeometry()
        assert p.width() <= available.width()
        assert p.width() >= min(2700, available.width())
    finally:
        p.close()


def test_the_window_comes_back_the_size_it_was_left(qt_app):
    """Closing saves the geometry; the next window opens on it, not the default."""
    first = Periscope("OFFLINE", module=1, chan_str="1",
                      skip_startup_dialog=True)
    first.resize(640, 480)
    first.close()

    second = Periscope("OFFLINE", module=1, chan_str="1",
                       skip_startup_dialog=True)
    try:
        assert (second.width(), second.height()) == (640, 480)
    finally:
        second.close()


def test_a_geometry_too_big_for_this_screen_is_shrunk_to_fit(qt_app):
    """A window closed on a large monitor, reopened on the laptop alone."""
    first = Periscope("OFFLINE", module=1, chan_str="1",
                      skip_startup_dialog=True)
    available = first.screen().availableGeometry()
    first.resize(available.width() * 3, available.height() * 3)
    first.close()

    second = Periscope("OFFLINE", module=1, chan_str="1",
                       skip_startup_dialog=True)
    try:
        assert second.width() <= available.width()
        assert second.height() <= available.height()
    finally:
        second.close()
