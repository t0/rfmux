"""The main window's status bar wraps: its statistics and the packet
warning must not set a minimum window width wider than a laptop screen,
and what wraps must stay visible."""
import pytest

pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from rfmux.tools.periscope.app import Periscope  # noqa: E402


class _Window(QtWidgets.QMainWindow):
    is_mock_mode = True


def _window_with_longest_texts(qt_app):
    win = _Window()
    win.setCentralWidget(QtWidgets.QWidget())
    Periscope._add_status_bar(win)
    win.fps_label.setText("FPS 30.0")
    win.pps_label.setText("| Packets/s 38147.0")
    win.sim_speed_label.setText("| Sim 1.00x")
    win.packet_loss_label.setText("| Loss: 10.0% missed, 10.0% dropped")
    win.dropped_label.setText("| Lost: 1,234,567 missed / 1,234,567 dropped")
    win.streaming_info_label.setText(
        "| Dec Stage: 6 | Fs: 596.0 Hz | Long Packets (Ch 1-1024)")
    win.info_text.setText(
        "PACKETS MISSED BEFORE PERISCOPE - check the network, the UDP "
        "buffer, or CPU load on this machine")
    win.show()
    return win


def _settle(qt_app):
    for _ in range(5):
        qt_app.processEvents()


def test_status_bar_does_not_set_the_window_width(qt_app):
    win = _window_with_longest_texts(qt_app)
    # Half a 1080p display, the bound the panels are held to.
    assert win.minimumSizeHint().width() < 960
    win.resize(700, 400)
    _settle(qt_app)
    assert win.width() == 700


def test_wrapped_status_text_stays_inside_the_bar(qt_app):
    win = _window_with_longest_texts(qt_app)
    win.resize(1900, 400)
    _settle(qt_app)
    one_row = win.statusBar().height()
    win.resize(700, 400)
    _settle(qt_app)
    bar = win.statusBar()
    assert bar.height() > one_row
    for label in (win.fps_label, win.streaming_info_label, win.info_text):
        bottom = label.mapTo(bar, label.rect().bottomRight())
        assert bottom.y() <= bar.height() and bottom.x() <= bar.width()
