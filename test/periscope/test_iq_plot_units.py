"""The IQ plot converts counts to volts once: _convert_iq_data does it
before the IQ task runs, so the task's bounds and points arrive in
display units and are drawn as they are."""
import numpy as np
import pytest

pytest.importorskip("PyQt6")

import pyqtgraph as pg  # noqa: E402
from PyQt6 import QtCore  # noqa: E402

from rfmux.tools.periscope.app_runtime import PeriscopeRuntime  # noqa: E402


class _RealUnits:
    real_units = True


def test_density_bounds_are_drawn_as_the_task_returned_them(qt_app):
    item = pg.ImageItem()
    hist = np.zeros((4, 4), dtype=np.uint8)
    PeriscopeRuntime._update_density_image(_RealUnits(), item, (hist, (1.0, 3.0, -2.0, 2.0)))
    rect = item.boundingRect()
    # ImageItem.setRect maps the image onto (Imin, Qmin, width, height).
    assert item.mapRectToParent(rect) == QtCore.QRectF(1.0, -2.0, 2.0, 4.0)


def test_scatter_points_are_drawn_as_the_task_returned_them(qt_app):
    item = pg.ScatterPlotItem()
    xs, ys = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    colors = [pg.mkBrush("w")] * 2
    PeriscopeRuntime._update_scatter_plot(_RealUnits(), item, (xs, ys, colors))
    got = item.getData()
    assert np.array_equal(got[0], xs) and np.array_equal(got[1], ys)
