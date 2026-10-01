"""Axis labels in SI notation: the unit in a label's brackets is the
axis's own, prefixed to its tick scale, never a trailing scale factor."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
import pyqtgraph as pg  # noqa: E402

from rfmux.tools.periscope.utils import set_axis_label  # noqa: E402


@pytest.fixture
def plot(qt_app):
    w = pg.PlotWidget()
    w.resize(400, 300)
    yield w
    w.close()


def test_an_si_unit_is_prefixed_to_the_scale_not_given_a_factor(plot, qt_app):
    plot.plot(np.arange(10.0), np.linspace(0.0, 4e-3, 10))
    set_axis_label(plot, "left", "I (V)")
    axis = plot.getPlotItem().getAxis("left")
    assert (axis.labelText, axis.labelUnits) == ("I", "V")
    plot.show()
    plot.setYRange(0.0, 4e-3)
    qt_app.processEvents()
    assert "mV" in axis.labelString() and "x0.001" not in axis.labelString()


def test_text_after_the_unit_stays_with_the_name(plot):
    set_axis_label(plot, "left", "I (V) − baseline")
    axis = plot.getPlotItem().getAxis("left")
    assert (axis.labelText, axis.labelUnits) == ("I − baseline", "V")


def test_other_units_keep_their_text_and_their_scale(plot, qt_app):
    plot.plot(np.arange(10.0), np.linspace(0.0, 4e3, 10))
    plot.show()
    plot.setYRange(0.0, 4e3)
    for text in ("amplitude (counts)", "duration (ms)", "SNR (σ)", "count"):
        set_axis_label(plot, "left", text)
        axis = plot.getPlotItem().getAxis("left")
        assert (axis.labelText, axis.labelUnits) == (text, "")
        qt_app.processEvents()
        assert "x1000" not in axis.labelString()
