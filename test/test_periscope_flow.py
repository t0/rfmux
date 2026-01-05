import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from rfmux.tools.periscope.app import Periscope
from rfmux.tools.periscope.utils import QtWidgets


@pytest.fixture(scope="session")
def qt_app():
    """
    Provide a QApplication for all UI smoke tests.
    Ensures Qt is initialized only once.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
    app.quit()


@pytest.fixture
def mock_periscope(qt_app):
    """
    Build a minimal Periscope instance for mocked UI tests.
    Equivalent to the old `_build_mock_periscope()`.
    """

    # Allocate object without running Periscope.__init__
    periscope = Periscope.__new__(Periscope)
    QtWidgets.QMainWindow.__init__(periscope)

    # Required minimal CRS mock
    periscope.crs = MagicMock()
    periscope.crs.TIMESTAMP_PORT = SimpleNamespace(
        BACKPLANE="BACKPLANE",
        TEST="TEST",
        SMA="SMA"
    )

    # Basic runtime attributes
    periscope.module = 1
    periscope.pool = MagicMock()
    periscope.pool.start = MagicMock()

    periscope.channel_list = [[1]]
    periscope.dark_mode = False
    periscope.dac_scales = {1: -0.5}
    periscope.raw_data = {1: [1, 2, 3]}
    periscope.resonance_freqs = {1: [90e6, 91e6]}

    periscope.timer = MagicMock()
    periscope.receiver = MagicMock()
    periscope.receiver.stop = MagicMock()
    periscope.receiver.wait = MagicMock()

    # UI bookkeeping
    periscope.netanal_window_count = 0
    periscope.netanal_windows = {}
    periscope.netanal_tasks = {}

    periscope.multisweep_window_count = 0
    periscope.multisweep_windows = {}
    periscope.multisweep_tasks = {}

    # Mock tab widget
    periscope.tabs = MagicMock()
    periscope.tabs.currentIndex.return_value = 0
    periscope.tabs.tabText.return_value = "Module 1"

    # Signals required by smoke test
    periscope.netanal_signals = MagicMock()
    periscope.multisweep_signals = MagicMock()

    dock_manager = MagicMock()

    def _make_dock(widget, title=None, dock_id=None):
        dock = MagicMock()
        dock.widget.return_value = widget
        dock.windowTitle.return_value = title or ""
        return dock

    dock_manager.create_dock.side_effect = _make_dock
    dock_manager.get_dock.return_value = None
    dock_manager.protect_dock = MagicMock()
    dock_manager.remove_dock = MagicMock()
    periscope.dock_manager = dock_manager

    periscope.setDockNestingEnabled = MagicMock()
    periscope.tabifyDockWidget = MagicMock()
    
    for sig_name in (
        "progress",
        "starting_iteration",
        "data_update",
        "completed_iteration",
        "all_completed",
        "error",
        "fitting_progress",
    ):
        setattr(periscope.multisweep_signals, sig_name, MagicMock())

    return periscope


def test_periscope_ui_smoke(mock_periscope):
    """
    Run the mocked UI smoke test using pytest.
    No real dialogs or Qt threads should be launched.
    """

    # The core smoke-test entry point
    mock_periscope.run_ui_mock_smoke_test()


    # Passing = no exceptions
    assert True