"""Per-resonator grids for saved noise measurements."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets

from rfmux.algorithms.measurement.noise_display import (
    noise_catalog, noise_display_products,
)
from .layouts import FlowLayout, labelled, WrappingLabel
from .multisweep_grid_helpers import arrange_plot_widgets, _new_subplot
from .utils import DEFAULT_SUBPLOTS, IQ_COLORS, LINE_WIDTH, ScreenshotMixin


class NoiseSpectrumPanel(QtWidgets.QWidget, ScreenshotMixin):
    def __init__(self, block: dict, parent=None, *, dark_mode: bool = False,
                 file_path: str = "") -> None:
        super().__init__(parent)
        self.block = block
        self.dark_mode = dark_mode
        self.names = list(block["results"]["resonators"])
        self._products = {}
        self._page = 0
        layout = QtWidgets.QVBoxLayout(self)
        title = WrappingLabel(file_path or "Noise measurement")
        layout.addWidget(title)
        toolbar = QtWidgets.QWidget()
        controls = FlowLayout(toolbar)
        self.units_combo = QtWidgets.QComboBox()
        self.units_combo.addItem("Volts", "volts")
        self.catalog = catalog = noise_catalog(block)
        if catalog and any(r.bias.df_calibration is not None for r in catalog):
            self.units_combo.addItem("df", "df")
        controls.addWidget(labelled("Units:", self.units_combo))
        self.stream_combo = QtWidgets.QComboBox()
        self.stream_combo.addItem("Slow", "slow")
        if any("pfb_data" in r for r in block["results"]["resonators"].values()):
            self.stream_combo.addItem("PFB", "pfb")
        self.stream_combo.setToolTip(
            "PFB times are relative to each sequential capture, not synchronized.")
        controls.addWidget(labelled("Stream:", self.stream_combo))
        self.mean_subtract = QtWidgets.QCheckBox("Subtract timestream mean")
        self.mean_subtract.setChecked(True)
        controls.addWidget(self.mean_subtract)
        self.prev_button = QtWidgets.QPushButton("Previous")
        self.next_button = QtWidgets.QPushButton("Next")
        self.page_label = QtWidgets.QLabel()
        for widget in (self.prev_button, self.page_label, self.next_button):
            controls.addWidget(widget)
        layout.addWidget(toolbar)
        self.plot_tabs = QtWidgets.QTabWidget()
        self.grids = []
        self.plots = [[], []]
        for label in ("Timestreams", "Power spectral densities"):
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            container = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(container)
            grid.setSpacing(10)
            scroll.setWidget(container)
            self.grids.append(grid)
            self.plot_tabs.addTab(scroll, label)
        layout.addWidget(self.plot_tabs)
        self.status = WrappingLabel("")
        layout.addWidget(self.status)
        self.units_combo.currentIndexChanged.connect(self._redraw)
        self.stream_combo.currentIndexChanged.connect(self._redraw)
        self.mean_subtract.toggled.connect(self._redraw)
        self.plot_tabs.currentChanged.connect(self._redraw)
        self.prev_button.clicked.connect(self._previous)
        self.next_button.clicked.connect(self._next)
        self._redraw()

    def _previous(self) -> None:
        self._page -= 1
        self._redraw()

    def _next(self) -> None:
        self._page += 1
        self._redraw()

    def _redraw(self) -> None:
        tab = self.plot_tabs.currentIndex()
        stream, units = self.stream_combo.currentData(), self.units_combo.currentData()
        names = self.names[self._page * DEFAULT_SUBPLOTS:
                           (self._page + 1) * DEFAULT_SUBPLOTS]
        pages = max(1, (len(self.names) + DEFAULT_SUBPLOTS - 1) // DEFAULT_SUBPLOTS)
        self.prev_button.setEnabled(self._page > 0)
        self.next_button.setEnabled(self._page + 1 < pages)
        self.page_label.setText(f"{self._page + 1} / {pages}")
        plots = self.plots[tab]
        while len(plots) < len(names):
            widget = _new_subplot(None)
            widget.setMinimumSize(220, 220)
            plots.append(widget)
        for widget in plots:
            widget.hide()
        background, foreground = ("k", "w") if self.dark_mode else ("w", "k")
        errors = []
        for name, widget in zip(names, plots):
            widget.show()
            widget.setBackground(background)
            plot = widget.getPlotItem()
            plot.clear()
            if plot.legend is None:
                plot.addLegend()
            else:
                plot.legend.clear()
            plot.legend.setLabelTextColor(foreground)
            record = self.block["results"]["resonators"][name]
            plot.setTitle(f"{name} · channel {record['channel']}", color=foreground)
            for axis in ("left", "bottom", "right", "top"):
                plot.getAxis(axis).setPen(foreground)
                plot.getAxis(axis).setTextPen(foreground)
            plot.setLogMode(x=bool(tab), y=bool(tab))
            plot.getAxis("left").enableAutoSIPrefix(not tab)
            plot.showGrid(x=True, y=True, alpha=0.3)
            label = "I/Q" if units == "volts" else "df/diss"
            unit = "V" if units == "volts" else "Hz"
            plot.setLabel("bottom", "Frequency" if tab else "Nominal elapsed time",
                          units="Hz" if tab else "s")
            plot.setLabel("left", f"{label} PSD" if tab else label,
                          units=f"{unit}²/Hz" if tab else unit)
            try:
                key = (name, stream, units, bool(tab))
                if key not in self._products:
                    self._products[key] = noise_display_products(
                        self.block, name, stream=stream, units=units,
                        include_psd=bool(tab), include_tod=not tab,
                        catalog=self.catalog)
                products = self._products[key]
            except (ValueError, KeyError) as exc:
                errors.append(f"{name}: {exc}")
                continue
            if tab:
                x = products["frequency_hz"]
                keep = x > 0
                x = x[keep]
                components = (products["psd_i"][keep], products["psd_q"][keep])
            else:
                x, iq = products["time_s"], products["iq"]
                if self.mean_subtract.isChecked():
                    iq = iq - np.mean(iq)
                components = (iq.real, iq.imag)
            labels = ("I", "Q") if units == "volts" else ("df", "diss")
            for values, component, color in zip(components, labels, IQ_COLORS.values()):
                if tab:
                    # The spectral helpers floor log(0) at machine-tiny.
                    values = np.where(values > 2 * np.finfo(float).tiny, values, np.nan)
                plot.plot(x, values, name=component,
                          pen=pg.mkPen(color, width=LINE_WIDTH))
            plot.enableAutoRange()
        arrange_plot_widgets(self.grids[tab], plots[:len(names)])
        self.status.setText("; ".join(errors))

    def apply_theme(self, dark_mode: bool) -> None:
        self.dark_mode = dark_mode
        self._redraw()
