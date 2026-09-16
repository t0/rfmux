"""The decision-mark labels sit on a wash of the plot's own background."""

import pyqtgraph as pg
import pytest

from test.pulse_capture.test_decision_bands import _capture, _panel_showing


def _label_fills(plot):
    return [it.label.fill.color().getRgb()[:3]
            for it in plot.getPlotItem().items
            if isinstance(it, pg.InfiniteLine) and it.label]


def test_mark_labels_take_the_plot_background(qt_app):
    s, d = _capture(0.0)
    panel = _panel_showing(qt_app, d, s.noise_stats[1])
    try:
        fills = _label_fills(panel.pulse_plot_i)
        assert fills and all(f == (255, 255, 255) for f in fills)

        panel.apply_theme(True)
        fills = _label_fills(panel.pulse_plot_i)
        assert fills and all(f == (0, 0, 0) for f in fills)
    finally:
        panel.close()
