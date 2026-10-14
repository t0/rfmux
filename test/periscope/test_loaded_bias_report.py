"""Test restoration of the tuned catalog and calibration from saved bias reports."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog  # noqa: E402
from rfmux.tuning import AmplitudeSchedule, BiasReport, store, tuning_rows  # noqa: E402
from rfmux.tuning.sweep_results import pack_multisweep  # noqa: E402
from rfmux.tools.periscope.multisweep_panel import MultisweepPanel  # noqa: E402

MODULE_ID = "board1.2"


def _tuned_catalog(name: str):
    """The swept resonator, now carrying the IQ derivatives read off its
    sweep, so its df calibration is a number rather than None.

    Named after the sweep it came from, as ``find_bias_points``' own catalog
    is: the panel draws by name.
    """
    bias = BiasPoint(frequency_hz=1.0e9, amplitude=0.01,
                     dI_df=1e-9, dQ_df=2e-9)
    return ResonatorCatalog([Resonator(name=name, channel=1, bias=bias)],
                            module=2)


def _container():
    catalog = ResonatorCatalog.from_frequencies([1.0e9], module=2, amplitude=0.01)
    name = catalog.names()[0]
    frequencies = np.linspace(1.0e9 - 1e5, 1.0e9 + 1e5, 21)
    iq = np.ones(21, dtype=complex)
    entry = {"channel": 1, "frequencies": frequencies, "iq_counts": iq,
             "iq_volts": iq * 1e-6, "original_center_frequency": 1.0e9,
             "sweep_direction": "upward", "sweep_amplitude": 0.01}
    container = pack_multisweep(
        {0: {"upward": {name: entry}}}, module_id=MODULE_ID, module=2,
        amp_schedule=AmplitudeSchedule(), directions=["upward"], span_hz=2e5,
        npoints_per_sweep=21, nsamps=10, catalog=catalog)
    container[MODULE_ID]["bias_report"] = BiasReport(
        catalog=_tuned_catalog(name), findings=[]).to_dict()
    return container


def _panel(container):
    call_params = dict(container[MODULE_ID]["call_params"])
    call_params["catalog"] = ResonatorCatalog.from_dict(call_params["catalog"])
    call_params["amp"] = AmplitudeSchedule.from_dict(call_params["amp_schedule"])
    panel = MultisweepPanel(target_module=2, initial_params=call_params,
                            is_loaded_data=True)
    panel.show_measurement(2, container)
    return panel


def test_a_loaded_measurement_brings_its_bias_report_back(qt_app, tmp_path):
    container = store.load(store.save(_container(), "multisweep",
                                      directory=tmp_path))
    panel = _panel(container)
    try:
        assert panel.bias_report is not None
    finally:
        panel.close()


def test_the_catalog_a_loaded_measurement_applies_is_the_tuned_one(
        qt_app, tmp_path):
    container = store.load(store.save(_container(), "multisweep",
                                      directory=tmp_path))
    panel = _panel(container)
    try:
        # What Apply Bias would publish to the main window.
        rows = tuning_rows(panel.catalog)
        assert rows[1]["df_calibration"] is not None
    finally:
        panel.close()


def test_a_measurement_that_was_never_biased_has_no_report(qt_app, tmp_path):
    container = _container()
    del container[MODULE_ID]["bias_report"]
    panel = _panel(store.load(store.save(container, "multisweep",
                                         directory=tmp_path)))
    try:
        assert panel.bias_report is None
    finally:
        panel.close()


def test_derivative_plot_compares_pair_strength_to_both_thresholds(qt_app):
    import pyqtgraph as pg
    from rfmux.tuning import bifurcated_by_derivative
    from rfmux.tools.periscope.multisweep_grid_helpers import _plot_bifurcation

    entry = {"frequencies": np.arange(8.),
             "iq_counts": np.array([0, 1, 3, 4, 4.4, 4.7, 8, 9]) * (1 + 1j)}
    check = bifurcated_by_derivative({"upward": entry})
    widget = pg.PlotWidget()
    try:
        _plot_bifurcation(widget.getPlotItem(), [(0, "upward", 0.01, entry)],
                         "black", None, {})
        curves = widget.listDataItems()
        assert len(curves) == 3
        np.testing.assert_allclose([curve.yData[0] for curve in curves],
                                   [check.metric["pair_strength"],
                                    check.diagnostics["shape_threshold"],
                                    check.diagnostics["noise_threshold"]])
        assert all(curve.xData.tolist() == [0.01] for curve in curves)
        assert widget.getPlotItem().legend is None
    finally:
        widget.close()


def test_bias_plot_labels_are_shared_above_the_grids(qt_app):
    from PyQt6 import QtWidgets

    panel = _panel(_container())
    try:
        for tab, expected in (
                (panel.bias_sweeps_tab, ("Pair strength", "Shape threshold",
                                         "Noise threshold", "Selected amplitude")),
                (panel.freq_sweeps_tab, ("IQ arc speed", "dI/df", "dQ/df",
                                         "f_bias"))):
            strip = tab.findChild(QtWidgets.QWidget, "shared_plot_labels")
            assert strip is not None
            labels = " ".join(label.text() for label in
                              strip.findChildren(QtWidgets.QLabel))
            assert all(name in labels for name in expected)
            assert tab.layout().indexOf(strip) < tab.layout().indexOf(
                next(widget for widget in tab.findChildren(QtWidgets.QScrollArea)))
    finally:
        panel.close()


def test_derivative_shared_labels_point_to_the_bias_workbook(qt_app):
    from PyQt6 import QtWidgets

    panel = _panel(_container())
    try:
        strip = panel.bias_sweeps_tab.findChild(
            QtWidgets.QWidget, "shared_plot_labels")
        path = "rfmux/reference-notebooks/Demos/bias_finding.md"
        assert path in strip.toolTip()
        assert "Bifurcation detection: derivative method" in strip.toolTip()
        assert all(path in label.toolTip()
                   for label in strip.findChildren(QtWidgets.QLabel))
    finally:
        panel.close()


@pytest.mark.parametrize("foreground", ["k", "w"])
def test_frequency_speed_uses_foreground_without_highlight(qt_app, foreground):
    import pyqtgraph as pg
    from rfmux.tools.periscope.multisweep_grid_helpers import _plot_bias_frequency

    sweep = next(iter(_container()[MODULE_ID]["results"][0]["upward"].values()))
    sweep = {**sweep, "iq_counts": np.linspace(0, 1, 21) +
             1j * np.linspace(1, 0, 21)}
    widget = pg.PlotWidget()
    try:
        _plot_bias_frequency(widget.getPlotItem(),
                             [(0, "upward", 0.01, sweep)], foreground, None)
        curves = widget.listDataItems()
        assert len(curves) == 3
        assert curves[-1].opts["pen"].color() == pg.mkColor(foreground)
        assert all(not curve.property("bias_highlight") for curve in curves)
        assert widget.getPlotItem().legend is None
    finally:
        widget.close()


def test_bias_legend_uses_two_significant_figures_without_flags():
    from types import SimpleNamespace
    from rfmux.tools.periscope.multisweep_grid_helpers import bias_legend_label

    bias = SimpleNamespace(amplitude=0.0123456, flagged_kind="freq out of bounds")
    assert bias_legend_label(bias) == "f_bias<br>amp=0.012"


def test_fit_axis_summarizes_multiple_drawn_fits():
    from rfmux.tools.periscope.multisweep_grid_helpers import fit_axis_medians

    rows = [("upward", 0.01, {"fr": 1e9, "Qi": 1e5, "Qc": 2e5}),
            ("downward", 0.02, {"fr": 1.2e9, "Qi": 3e5, "Qc": 4e5})]
    assert fit_axis_medians(rows, "skewed") == (
        "med fr=1.1e+09 Qi=2.0e+05 Qc=3.0e+05")
