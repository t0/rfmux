"""A drive's colour is one colour, whatever the schedule's length.

Above ``AMPLITUDE_COLORMAP_THRESHOLD`` the colours come off a colormap. An
RGBA array reads to pyqtgraph's scatter plots as one pen per point, so the fit
tab raised on every sweep of four or more amplitudes.
"""

import pytest

pytest.importorskip("PyQt6")

import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402

from rfmux.tools.periscope.multisweep_grid_helpers import (  # noqa: E402
    create_amplitude_color_map)


@pytest.mark.portable
@pytest.mark.parametrize("num_amps", [1, 3, 4, 8])
@pytest.mark.parametrize("dark_mode", [False, True])
def test_an_amplitude_colour_is_a_scatter_symbol_pen(qt_app, num_amps, dark_mode):
    amplitudes = list(np.linspace(0.001, 0.01, num_amps))
    colors = create_amplitude_color_map(amplitudes, dark_mode)

    for color in colors.values():
        item = pg.PlotDataItem(
            [0.0, 1.0, 2.0], [0.0, 1.0, 2.0],
            pen=None, symbol='o', symbolPen=color, symbolBrush=color)
        assert len(item.scatter.data) == 3
