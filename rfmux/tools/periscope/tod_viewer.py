"""One channel of a file's time-ordered data over time, drawn from at
most a few hundred bins: the whole run from the overview, and zooming
in reads the window's slice of the file until the samples themselves
show.  The reading is ``tod_window`` in
``rfmux.algorithms.measurement.tod``; this window only draws it."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from ...algorithms.measurement.tod import tod_extent, tod_window
from ...pulse_capture.channel_keys import channel_group, title_label
from ...streamer import epoch_to_utc
from .layouts import FlowLayout
from .utils import IQ_COLORS, ScreenshotMixin

#: Milliseconds after the last range change before the window is read.
SETTLE_MS = 60
#: The fast stream's curves, drawn under the slow ones in grey: its
#: noise is the wider of the two, and the slow keeps the I and Q colours.
FAST_COLOUR = (150, 150, 150, 170)


class TodViewer(QtWidgets.QWidget, ScreenshotMixin):
    """*key*'s time-ordered data in the file at *path*: its fast and
    slow streams, either or both, the fast drawn first."""

    def __init__(self, path, key, parent=None, *, dark_mode: bool = False):
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowType.Window)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.path, self.key = Path(path), key
        self.dark_mode = dark_mode
        self.f = h5py.File(self.path, "r")
        group = channel_group(key)
        # Fast first: its curves are added first, so the slow draw on top.
        self.streams = [s for s in ("fast", "slow")
                        if f"tod/{s}/{group}" in self.f]
        units = {str(self.f[f"tod/{s}/{group}"].attrs.get("stored_units", ""))
                 for s in self.streams}
        self.units = units.pop() if len(units) == 1 else ""
        extents = [tod_extent(self.f, s, key) for s in self.streams]
        firsts = [e[0] for e in extents if np.isfinite(e[0])]
        lasts = [e[1] for e in extents if np.isfinite(e[1])]
        #: Seconds of day the plots' zero is.
        self.origin = min(firsts) if firsts else 0.0
        self.span = (max(lasts) - self.origin) if lasts else 1.0
        self.setWindowTitle(f"{title_label(key)} time-ordered data: "
                            f"{self.path.name}")
        self._setup_ui()
        self.apply_theme(dark_mode)
        self.reset_view()

    # ── UI ────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QWidget()
        flow = FlowLayout(bar)
        self.stream_checks = {}
        for s in ("fast", "slow"):
            check = QtWidgets.QCheckBox(f"{s.capitalize()} stream")
            check.setChecked(s in self.streams)
            check.setEnabled(s in self.streams)
            check.toggled.connect(self._refresh)
            flow.addWidget(check)
            self.stream_checks[s] = check
        reset = QtWidgets.QPushButton("Whole run")
        reset.clicked.connect(self.reset_view)
        flow.addWidget(reset)
        self.info = QtWidgets.QLabel()
        flow.addWidget(self.info)
        outer.addWidget(bar)

        names = ("df", "dissipation") if self.units == "Hz" else ("I", "Q")
        unit = f" ({self.units})" if self.units else ""
        self.plots = []
        for name in names:
            plot = pg.PlotWidget()
            plot.setLabel("left", name + unit)
            plot.setMouseEnabled(x=True, y=False)
            plot.getPlotItem().enableAutoRange(axis="y")
            plot.getPlotItem().setAutoVisible(y=True)
            plot.addLegend(offset=(-10, 10))
            outer.addWidget(plot, 1)
            self.plots.append(plot)
        self.plots[1].setXLink(self.plots[0])
        self.plots[0].getPlotItem().getAxis("bottom").setStyle(
            showValues=False)
        epoch = self.f["metadata"].attrs.get("time_origin_epoch") \
            if "metadata" in self.f else None
        start = (epoch_to_utc(float(epoch) + self.origin) if epoch is not None
                 else f"{self.origin:.6f} s of day")
        self.plots[1].setLabel("bottom", f"time from {start}", units="s")

        colours = (IQ_COLORS["I"], IQ_COLORS["Q"])
        self.curves = {}
        for s in self.streams:
            for plot, colour in zip(self.plots, colours):
                c = FAST_COLOUR if s == "fast" else colour
                curve = plot.plot(pen=pg.mkPen(c, width=1 if s == "fast" else 1.5),
                                  symbolPen=None, symbolBrush=c, name=s)
                curve.setZValue(0 if s == "fast" else 1)
                self.curves.setdefault(s, []).append(curve)

        self._timer = QtCore.QTimer(self, singleShot=True, interval=SETTLE_MS)
        self._timer.timeout.connect(self._refresh)
        self.plots[0].sigXRangeChanged.connect(lambda *_: self._timer.start())
        self.resize(1100, 650)

    def apply_theme(self, dark_mode: bool) -> None:
        self.dark_mode = dark_mode
        bg, fg = ("k", "w") if dark_mode else ("w", "k")
        for plot in self.plots:
            plot.setBackground(bg)
            for ax in ("left", "bottom"):
                axis = plot.getPlotItem().getAxis(ax)
                axis.setPen(fg)
                axis.setTextPen(fg)

    # ── Drawing ───────────────────────────────────────────────────

    def reset_view(self) -> None:
        self.plots[0].setXRange(0.0, self.span, padding=0.01)
        self._refresh()

    def _refresh(self, *_) -> None:
        x0, x1 = self.plots[0].getPlotItem().viewRange()[0]
        t0, t1 = self.origin + x0, self.origin + x1
        parts = []
        for s in self.streams:
            curves = self.curves[s]
            if not self.stream_checks[s].isChecked():
                for curve in curves:
                    curve.setData([], [])
                continue
            view = tod_window(self.f, s, self.key, t0, t1)
            x, ys = _xy(view)
            # Each sample marked once the view is the samples themselves.
            symbol = "o" if view["kind"] == "raw" else None
            for curve, y in zip(curves, ys):
                curve.setData(x - self.origin, y, connect="finite",
                              symbol=symbol, symbolSize=4)
            parts.append(f"{s}: {view['samples']:,} samples, " + (
                "each drawn" if view["kind"] == "raw" else
                f"{len(view['t_first'])} bins from the {view['source']}"))
        self.info.setText("; ".join(parts))

    def closeEvent(self, event) -> None:
        self._timer.stop()
        if self.f is not None:
            self.f.close()
            self.f = None
        super().closeEvent(event)


def _xy(view: dict):
    """(x, (y_I, y_Q)) to draw a view: the samples, or each bin as a
    vertical stroke from its minimum to its maximum at its middle, so a
    one-sample spike in a bin of thousands still shows."""
    if view["kind"] == "raw":
        return view["time"], (view["I"], view["Q"])
    mid = 0.5 * (view["t_first"] + view["t_last"])
    x = np.repeat(mid, 2)
    ys = [np.column_stack([view[f"{a}_min"], view[f"{a}_max"]]).ravel()
          for a in ("i", "q")]
    return x, ys
