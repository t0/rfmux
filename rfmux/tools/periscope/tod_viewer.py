"""The Channel TOD View tab: one channel of a file's time-ordered data
over time, drawn from at most a few hundred bins.  The whole run comes
from the overview; zooming reads the window's slice of the file until
the samples themselves show.  The reading is ``tod_window`` in
``rfmux.algorithms.measurement.tod``; this widget only draws it."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from ...algorithms.measurement.tod import tod_extent, tod_window
from ...pulse_capture.channel_keys import channel_group, title_label
from ...streamer import epoch_to_utc
from .layouts import FlowLayout, labelled
from .utils import IQ_COLORS, ClickableViewBox

#: Milliseconds after the last range change before the window is read.
SETTLE_MS = 60
#: The fast stream's curves, drawn under the slow ones in grey: its
#: noise is the wider of the two, and the slow keeps the I and Q colours.
FAST_COLOUR = (150, 150, 150, 170)
#: What the tab says before a channel is chosen.
PROMPT = "Double-click a channel under Time-ordered data, or choose one here"


class TodViewer(QtWidgets.QWidget):
    """A channel of the time-ordered data in one file: its fast and slow
    streams, either or both, the fast drawn first.  Drag draws a zoom
    box, the wheel zooms time, Whole run returns."""

    def __init__(self, parent=None, *, dark_mode: bool = False):
        super().__init__(parent)
        self.dark_mode = dark_mode
        self.f: Optional[h5py.File] = None
        self.path: Optional[Path] = None
        self.key = None
        self.streams: List[str] = []
        self.curves = {}
        self.origin, self.span = 0.0, 1.0
        self._setup_ui()
        self.apply_theme(dark_mode)

    # ── UI ────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        bar = QtWidgets.QWidget()
        flow = FlowLayout(bar)
        self.btn_prev = QtWidgets.QPushButton("◀ Prev")
        self.btn_prev.clicked.connect(lambda: self.step(-1))
        self.btn_next = QtWidgets.QPushButton("Next ▶")
        self.btn_next.clicked.connect(lambda: self.step(+1))
        for btn in (self.btn_prev, self.btn_next):
            btn.setToolTip("The previous or next channel, over the same "
                           "time window")
            flow.addWidget(btn)
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.activated.connect(
            lambda i: self.show_channel(self.channel_combo.itemData(i)))
        flow.addWidget(labelled("Channel:", self.channel_combo))
        self.stream_checks = {}
        for s in ("fast", "slow"):
            check = QtWidgets.QCheckBox(f"{s.capitalize()} stream")
            check.setChecked(True)
            check.toggled.connect(self._refresh)
            flow.addWidget(check)
            self.stream_checks[s] = check
        self.reset_btn = QtWidgets.QPushButton("Whole run")
        self.reset_btn.setToolTip("The whole run, the vertical axis "
                                  "following the data again")
        self.reset_btn.clicked.connect(self.reset_view)
        flow.addWidget(self.reset_btn)
        outer.addWidget(bar)
        self.info = QtWidgets.QLabel(PROMPT)
        self.info.setWordWrap(True)
        outer.addWidget(self.info)

        self.plots = []
        for _ in range(2):
            plot = pg.PlotWidget(viewBox=ClickableViewBox())
            # The wheel zooms time, the vertical axis following what is in
            # view; a dragged box sets both.
            plot.setMouseEnabled(x=True, y=False)
            plot.getPlotItem().showGrid(x=True, y=True, alpha=0.3)
            plot.addLegend(offset=(-10, 10))
            outer.addWidget(plot, 1)
            self.plots.append(plot)
        self.plots[1].setXLink(self.plots[0])
        self.plots[0].getPlotItem().getAxis("bottom").setStyle(
            showValues=False)

        self._timer = QtCore.QTimer(self, singleShot=True, interval=SETTLE_MS)
        self._timer.timeout.connect(self._refresh)
        self.plots[0].sigXRangeChanged.connect(lambda *_: self._timer.start())

    def apply_theme(self, dark_mode: bool) -> None:
        self.dark_mode = dark_mode
        bg, fg = ("k", "w") if dark_mode else ("w", "k")
        for plot in self.plots:
            plot.setBackground(bg)
            for ax in ("left", "bottom"):
                axis = plot.getPlotItem().getAxis(ax)
                axis.setPen(fg)
                axis.setTextPen(fg)

    # ── The file and the channel ──────────────────────────────────

    def set_file(self, path, channels) -> None:
        """Browse the time-ordered data of *path*, *channels* offered;
        None closes the file."""
        self.close_file()
        if path is None:
            return
        self.path = Path(path)
        self.f = h5py.File(self.path, "r")
        for c in channels:
            self.channel_combo.addItem(title_label(c), c)

    def close_file(self) -> None:
        self._timer.stop()
        self._clear_curves()
        self.channel_combo.clear()
        self.key, self.streams = None, []
        if self.f is not None:
            self.f.close()
        self.f, self.path = None, None
        self.info.setText(PROMPT)

    def _clear_curves(self) -> None:
        for plot in self.plots:
            plot.getPlotItem().legend.clear()
            for curve in [c for cs in self.curves.values() for c in cs]:
                plot.removeItem(curve)
        self.curves = {}

    def step(self, step: int) -> None:
        """The channel *step* places along the Channel box, stopping at
        either end, over the time window in view."""
        n = self.channel_combo.count()
        if self.f is None or not n:
            return
        i = self.channel_combo.currentIndex() + step if self.key is not None \
            else 0
        i = max(0, min(n - 1, i))
        if self.channel_combo.itemData(i) != self.key:
            self.show_channel(self.channel_combo.itemData(i), keep_window=True)

    def show_channel(self, key, keep_window: bool = False) -> None:
        """Draw *key* over the whole run, or with *keep_window* over the
        time window in view."""
        if self.f is None:
            return
        window = None
        if keep_window and self.key is not None:
            x0, x1 = self.plots[0].getPlotItem().viewRange()[0]
            window = (self.origin + x0, self.origin + x1)
        self._clear_curves()
        self.key = key
        idx = self.channel_combo.findData(key)
        if idx >= 0:
            self.channel_combo.setCurrentIndex(idx)
        group = channel_group(key)
        # Fast first: its curves are added first, so the slow draw on top.
        self.streams = [s for s in ("fast", "slow")
                        if f"tod/{s}/{group}" in self.f]
        for s, check in self.stream_checks.items():
            check.setEnabled(s in self.streams)
        units = {str(self.f[f"tod/{s}/{group}"].attrs.get("stored_units", ""))
                 for s in self.streams}
        units = units.pop() if len(units) == 1 else ""
        extents = [tod_extent(self.f, s, key) for s in self.streams]
        firsts = [e[0] for e in extents if np.isfinite(e[0])]
        lasts = [e[1] for e in extents if np.isfinite(e[1])]
        #: Seconds of day the plots' zero is.
        self.origin = min(firsts) if firsts else 0.0
        self.span = (max(lasts) - self.origin) if lasts else 1.0

        names = ("df", "dissipation") if units == "Hz" else ("I", "Q")
        suffix = f" ({units})" if units else ""
        for plot, name in zip(self.plots, names):
            plot.setLabel("left", name + suffix)
        epoch = self.f["metadata"].attrs.get("time_origin_epoch") \
            if "metadata" in self.f else None
        start = (epoch_to_utc(float(epoch) + self.origin) if epoch is not None
                 else f"{self.origin:.6f} s of day")
        self.plots[1].setLabel("bottom", f"time from {start}", units="s")

        colours = (IQ_COLORS["I"], IQ_COLORS["Q"])
        for s in self.streams:
            for plot, colour in zip(self.plots, colours):
                c = FAST_COLOUR if s == "fast" else colour
                curve = plot.plot(pen=pg.mkPen(c, width=1 if s == "fast" else 1.5),
                                  symbolPen=None, symbolBrush=c, name=s)
                curve.setZValue(0 if s == "fast" else 1)
                self.curves.setdefault(s, []).append(curve)
        i = self.channel_combo.currentIndex()
        self.btn_prev.setEnabled(i > 0)
        self.btn_next.setEnabled(0 <= i < self.channel_combo.count() - 1)
        if window is None:
            self.reset_view()
            return
        # The time window stays; the levels are the new channel's.
        for plot in self.plots:
            plot.getPlotItem().enableAutoRange(axis="y")
            plot.getPlotItem().setAutoVisible(y=True)
        self.plots[0].setXRange(window[0] - self.origin,
                                window[1] - self.origin, padding=0)
        self._refresh()

    # ── Drawing ───────────────────────────────────────────────────

    def reset_view(self) -> None:
        if self.key is None:
            return
        for plot in self.plots:
            plot.getPlotItem().enableAutoRange(axis="y")
            plot.getPlotItem().setAutoVisible(y=True)
        self.plots[0].setXRange(0.0, self.span, padding=0.01)
        self._refresh()

    def _refresh(self, *_) -> None:
        if self.key is None or self.f is None:
            return
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
        self.info.setText(f"{title_label(self.key)}: " + "; ".join(parts))


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
