"""Helpers shared by the GUI tests: the event loop, a main window built
without its startup, and a stand-in board.

These live in a normal module rather than in ``conftest.py`` because
conftest is not importable by name — ``from conftest import spin`` picks
up whichever conftest is first on sys.path, which in this repo is the
root one. The ``qt_app`` fixture itself does live in
``test/conftest.py``, where pytest finds it by fixture lookup.
"""

import time


def spin(qt_app, seconds=0.05):
    """Pump the Qt event loop for a fixed spell."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qt_app.processEvents()
        time.sleep(0.005)


def spin_until(qt_app, predicate, timeout=8.0):
    """Pump the event loop until *predicate* holds; True if it did.

    Several call sites rely on the default timeout rather than passing one,
    and one passes it positionally.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qt_app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def bare_periscope(monkeypatch=None, *, crs=None, **attrs):
    """A main window built without its startup, holding what the
    multisweep and tuning paths read, with *attrs* set on top.

    Given a *monkeypatch*, a warning or error dialog fails the test.
    """
    from unittest.mock import MagicMock

    import pytest

    from rfmux.tools.periscope.app import Periscope
    from rfmux.tools.periscope.utils import QtWidgets

    if monkeypatch is not None:
        for kind in ("warning", "critical"):
            monkeypatch.setattr(QtWidgets.QMessageBox, kind,
                                lambda *a, **k: pytest.fail(f"dialog: {a[2]}"))
    p = Periscope.__new__(Periscope)
    QtWidgets.QMainWindow.__init__(p)
    p.crs, p.host, p.dark_mode = crs, "OFFLINE", False
    p.multisweep_window_count, p.multisweep_windows = 0, {}
    p.multisweep_tasks = {}
    p.tuning, p.df_calibrations = {}, {}
    p.dock_manager = MagicMock()
    p.dock_manager.get_dock.return_value = None
    p.tabifyDockWidget = MagicMock()
    for name, value in attrs.items():
        setattr(p, name, value)
    return p


class Board:
    """A stand-in CRS recording what it was told."""

    def __init__(self, nco_frequency_hz=1.05e9, dac_scale=-0.5):
        self.calls = []
        self.nco_frequency_hz = nco_frequency_hz
        self.dac_scale = dac_scale

    async def get_dac_scale(self, units="DBM", module=None):
        return self.dac_scale

    async def set_nco_frequency(self, f, module=None):
        self.calls.append(("nco", f, module))

    async def get_nco_frequency(self, module=None):
        return self.nco_frequency_hz
