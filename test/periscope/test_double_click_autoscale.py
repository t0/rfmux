"""A left double-click on a Periscope plot autoscales it to its data,
unless the plot's panel claims the double-click for something else."""
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6 import QtCore  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
import pyqtgraph as pg  # noqa: E402

from rfmux.tools.periscope.utils import ClickableViewBox  # noqa: E402


def _double_click(plot, qt_app):
    """A left double-click in the middle of *plot*, as the mouse sends it."""
    QTest.mouseDClick(plot.viewport(), QtCore.Qt.MouseButton.LeftButton,
                      pos=plot.viewport().rect().center())
    qt_app.processEvents()


def _zoomed_plot(qt_app):
    plot = pg.PlotWidget(viewBox=ClickableViewBox())
    plot.plot(np.arange(100.0), np.linspace(0.0, 1.0, 100))
    plot.resize(400, 300)
    plot.show()
    vb = plot.getViewBox()
    vb.setRange(xRange=(40, 45), yRange=(0.4, 0.45), padding=0)
    qt_app.processEvents()
    return plot, vb


def test_double_click_autoscales_to_the_data(qt_app):
    plot, vb = _zoomed_plot(qt_app)
    _double_click(plot, qt_app)
    (x0, x1), (y0, y1) = vb.viewRange()
    assert x0 <= 0 and x1 >= 99 and y0 <= 0 and y1 >= 1


def test_double_click_leaves_continuous_autoscale_as_it_was(qt_app):
    plot, vb = _zoomed_plot(qt_app)
    vb.enableAutoRange(x=False, y=True)
    _double_click(plot, qt_app)
    assert [bool(a) for a in vb.autoRangeEnabled()] == [False, True]


def test_a_panel_that_takes_the_double_click_keeps_the_view(qt_app):
    plot, vb = _zoomed_plot(qt_app)
    vb.doubleClickedEvent.connect(lambda ev: ev.accept())
    _double_click(plot, qt_app)
    (x0, x1), _ = vb.viewRange()
    assert (x0, x1) == pytest.approx((40, 45))


def test_histogram_plots_share_the_double_click(qt_app):
    from rfmux.tools.periscope.parameter_histograms_panel import (
        ParameterHistogramsPanel)
    vb = ParameterHistogramsPanel._view_box()
    assert isinstance(vb, ClickableViewBox)
    assert vb.state["mouseMode"] == pg.ViewBox.PanMode
