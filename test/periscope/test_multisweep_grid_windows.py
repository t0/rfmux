"""Multisweep subplots remain inside their tab while appearing."""

import pytest

pytest.importorskip("PyQt6")

import pyqtgraph as pg  # noqa: E402
from PyQt6 import QtCore, QtWidgets  # noqa: E402

from rfmux.tools.periscope.multisweep_grid_helpers import (  # noqa: E402
    update_sweep_grid,
)


class _TopLevelPlotShows(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.count = 0

    def eventFilter(self, obj, event):
        if (isinstance(obj, pg.PlotWidget)
                and event.type() == QtCore.QEvent.Type.Show
                and obj.isWindow()):
            self.count += 1
        return False


def test_multisweep_subplots_never_show_as_windows(qt_app):
    container = QtWidgets.QWidget()
    grid = QtWidgets.QGridLayout(container)
    cache = []
    probe = _TopLevelPlotShows()
    qt_app.installEventFilter(probe)
    try:
        update_sweep_grid(
            grid, {f"KID-{index}": [] for index in range(3)},
            "bias", 0, 3, {}, False, widget_cache=cache,
        )
        assert probe.count == 0
        assert all(plot.parentWidget() is container for plot in cache)
    finally:
        qt_app.removeEventFilter(probe)
        container.close()
        for plot in cache:
            plot.close()
