"""The array's fitted parameters, in aggregate.

The Fit Results tab shows one resonator at a time; this shows all of them at
once, which is the view that says whether an array is uniform, where its
outliers are, and what raising the drive did to it. One scatter of ``fr``
in ascending frequency order, and a histogram of each parameter worth binning.

Everything drawn comes from :func:`~rfmux.tuning.fits.collect_fit_params`, so
a notebook makes the same figures from the same rows. The tab holds no data of
its own -- only which model and which amplitudes it is showing.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from rfmux.tuning.fits import BIFURCATION_A, collect_fit_params

from .fit_display_toolbar import FitDisplayToolbar
from .fit_settings_panel import BIAS_AMPLITUDE
from .utils import LINE_WIDTH, TABLEAU10_COLORS

#: Which of a model's parameters get a histogram, in the order they are drawn.
#: ``fr`` is not among them: it spans the whole band, so a histogram of it is a
#: picture of where the tones were placed. It gets the scatter instead.
HISTOGRAM_PARAMS = {
    "skewed": ("Qr", "Qc", "Qi"),
    "nonlinear": ("Qr", "a"),
}

#: Parameters binned on a log axis, and sharing one set of bin edges with the
#: others so two quality factors side by side can be compared by eye.
LOG_PARAMS = frozenset({"Qr", "Qc", "Qi"})

#: How each parameter is labelled and what its numbers mean.
PARAM_TOOLTIPS = {
    "Qr": "Total quality factor",
    "Qc": "Coupling quality factor",
    "Qi": "Internal quality factor",
    "a": "Nonlinearity; at or past the marked line the model is multivalued "
         "and the tone is driving the resonator into bifurcation",
}

#: Bins for every histogram on the tab. Enough to show a second population on
#: a few hundred resonators without being noise on a few dozen.
NBINS = 30


class FitHistogramsTab(QtWidgets.QWidget):
    """A scatter of ``fr`` and a histogram per parameter, over a whole module.

    :meth:`show_sweeps` hands it the module block to read; it redraws itself
    when its toolbar's model or amplitude choice changes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sweeps = None
        self._qr_colorbar = None
        self._amplitude_to_color = {}
        self._bias_by_name: dict = {}
        #: ``{param: count}`` of fitted values no axis could hold -- a quality
        #: factor the fit made negative has no place on a log axis. Named on
        #: the plot it is missing from, and read by the tests that check the
        #: bins account for every fit.
        self.not_binned: dict = {}
        self._dark_mode = False
        self._plots: list[pg.PlotWidget] = []
        self._setup_ui()
        self.toolbar.display_changed.connect(self.redraw)

    # ── what it is showing ───────────────────────────────────────────────────

    def show_sweeps(self, sweeps, amplitude_to_color, dark_mode: bool,
                    bias_by_name: dict) -> None:
        """Draw *sweeps*, one module's block, with the grids' amplitude colours.

        *bias_by_name* is ``{name: BiasFinding}``, which is what the toolbar's
        "at bias" choice means: a different step for each resonator.
        """
        self._sweeps = sweeps
        self._amplitude_to_color = amplitude_to_color
        self._dark_mode = dark_mode
        self._bias_by_name = bias_by_name
        self.redraw()

    def redraw(self) -> None:
        """Rebuild every plot from the block, at the toolbar's choices."""
        model = self.toolbar.get_model()
        if self._sweeps is None or model not in HISTOGRAM_PARAMS:
            self._clear()
            self._status.setText("No fits to bin yet")
            return

        rows = self._at_chosen_amplitude(
            collect_fit_params(self._sweeps, model, iterations=self._iterations()))
        kept = [row for row in rows if row["failed_because"] is None]
        self._say_what_is_drawn(kept, rows, model)
        if not kept:
            self._clear()
            return

        params = HISTOGRAM_PARAMS[model]
        self.not_binned = {}
        self._lay_out(1 + len(params))
        self._draw_fr(self._plots[0], kept)
        edges = self._shared_log_edges(kept, params)
        for plot, param in zip(self._plots[1:], params):
            self._draw_histogram(plot, kept, param, edges)

    def _iterations(self):
        """The amplitude step to read, or None for every one of them."""
        choice = self.toolbar.get_amplitude()
        return choice if isinstance(choice, int) else None

    def _at_chosen_amplitude(self, rows: list) -> list:
        """Narrow *rows* to the "at bias" choice, which no step filter can express.

        Each resonator is biased at its own step, so this is a filter per row
        rather than per measurement -- and it selects nothing until a bias has
        been found.
        """
        if self.toolbar.get_amplitude() != BIAS_AMPLITUDE:
            return rows
        return [row for row in rows
                if row["name"] in self._bias_by_name
                and self._bias_by_name[row["name"]].iteration == row["iteration"]]

    def _say_what_is_drawn(self, kept: list, rows: list, model: str) -> None:
        """How many fits are on screen, and how many were left off.

        A fit the fitter rejected is not binned -- a parameter it converged on
        and then disowned would move the distribution without being a
        measurement of anything -- so the count says it is missing rather than
        the histogram quietly being short.
        """
        resonators = len({row["name"] for row in kept})
        message = (f"{len(kept)} {model} fits over {resonators} resonators"
                   if kept else f"No {model} fits to bin")
        rejected = len(rows) - len(kept)
        if rejected:
            message += f"; {rejected} rejected by the fitter, not binned"
        self._status.setText(message)

    # ── the plots ────────────────────────────────────────────────────────────

    def _draw_fr(self, plot, rows: list) -> None:
        """Frequency span in ascending order, coloured by each fit's Qr."""
        plot.clear()
        rows = sorted(rows, key=lambda row: row["params"]["fr"])
        qr = np.asarray([row["params"].get("Qr", np.nan) for row in rows])
        finite = np.isfinite(qr)
        cmap = pg.colormap.get("viridis")
        brushes = [pg.mkBrush(self._foreground()) for _ in rows]
        if self._qr_colorbar is None:
            self._qr_colorbar = pg.ColorBarItem(
                colorMap=cmap, label="Qr", interactive=False, colorMapMenu=False)
            self._qr_colorbar.setImageItem([], insert_in=plot.getPlotItem())
        self._qr_colorbar.setVisible(bool(finite.any()))
        if finite.any():
            low, high = float(qr[finite].min()), float(qr[finite].max())
            if low == high:
                low, high = low - 0.5, high + 0.5
            self._qr_colorbar.setLevels((low, high))
            for index in np.flatnonzero(finite):
                brushes[index] = pg.mkBrush(cmap.map(
                    (qr[index] - low) / (high - low), mode="qcolor"))
        for side in ("left", "right", "top", "bottom"):
            axis = self._qr_colorbar.getAxis(side)
            axis.setPen(self._foreground())
            axis.setTextPen(self._foreground())
        self._qr_colorbar.getAxis("left").setLabel("Qr", color=self._foreground())
        plot.addItem(pg.ScatterPlotItem(
            x=np.arange(len(rows)),
            y=[row["params"]["fr"] / 1e6 for row in rows],
            pen=None, brush=brushes, size=6,
        ))
        plot.setTitle("Resonant frequency", color=self._foreground())
        plot.setLabel("bottom", "Frequency rank")
        plot.setLabel("left", "fr", units="MHz")
        plot.setToolTip("Sorted fitted frequencies; colour shows Qr on a linear "
                        "scale. Missing or non-finite Qr uses the foreground colour.")

    def _draw_histogram(self, plot, rows: list, param: str, log_edges) -> None:
        """One parameter binned, one outline per drive amplitude.

        At bias, resonators can have different drive amplitudes; each drive
        retains its colour from the sweep grids.
        """
        logarithmic = param in LOG_PARAMS
        plot.clear()
        plot.setLogMode(x=logarithmic, y=False)
        plot.setLabel("bottom", param)
        plot.setLabel("left", "Resonators")
        plot.setToolTip(PARAM_TOOLTIPS.get(param, param))

        # A fit can converge on a negative quality factor, which no log axis
        # can hold. Counting those in the title is the difference between a
        # histogram that is short and a histogram that is wrong.
        placeable = self._placeable(rows, param, logarithmic)
        lost = len(self._values(rows, param)) - len(placeable)
        if lost:
            self.not_binned[param] = lost
        plot.setTitle(param if not lost else f"{param} ({lost} off the axis)",
                      color=self._foreground())

        edges = log_edges if logarithmic else self._edges(placeable)
        if edges is None:
            return

        for amplitude, of_amplitude in self._by_amplitude(rows).items():
            counts, _ = np.histogram(
                self._placeable(of_amplitude, param, logarithmic), bins=edges)
            plot.plot(edges, counts, stepMode="center",
                      pen=pg.mkPen(self._colour(amplitude), width=LINE_WIDTH))

        if param == "a":
            plot.addItem(pg.InfiniteLine(
                pos=BIFURCATION_A, angle=90, movable=False,
                pen=pg.mkPen(TABLEAU10_COLORS[3], width=LINE_WIDTH,
                             style=QtCore.Qt.PenStyle.DashLine),
                label=f"bifurcation, a={BIFURCATION_A:.3f}",
                labelOpts={"position": 0.9, "color": TABLEAU10_COLORS[3]}))

    def _shared_log_edges(self, rows: list, params) -> np.ndarray | None:
        """One set of log bins spanning every quality factor on the tab.

        Shared so that Qr, Qc and Qi are read against each other: on separate
        bins, three distributions of different widths all fill their axis and
        look alike.
        """
        values = np.concatenate(
            [self._placeable(rows, param, True)
             for param in params if param in LOG_PARAMS] or [np.empty(0)])
        if values.size < 2:
            return None
        low, high = float(values.min()), float(values.max())
        if low == high:
            low, high = low / 2, high * 2
        # geomspace rather than logspace over the logs: it sets the two end
        # edges to the values themselves, where 10**log10(x) rounds a hair
        # short of x and drops the largest value out of the last bin.
        return np.geomspace(low, high, NBINS + 1)

    @staticmethod
    def _edges(values: np.ndarray) -> np.ndarray | None:
        """Linear bins over *values*, or None if there is nothing to bin."""
        if values.size < 2:
            return None
        low, high = float(values.min()), float(values.max())
        if low == high:
            low, high = low - 0.5, high + 0.5
        return np.linspace(low, high, NBINS + 1)

    @classmethod
    def _placeable(cls, rows: list, param: str, logarithmic: bool) -> np.ndarray:
        """The values of *param* an axis can actually hold."""
        values = cls._values(rows, param)
        keep = np.isfinite(values)
        if logarithmic:
            keep &= values > 0
        return values[keep]

    @staticmethod
    def _values(rows: list, param: str) -> np.ndarray:
        return np.asarray([row["params"][param] for row in rows
                           if param in row["params"]], dtype=float)

    def _by_amplitude(self, rows: list) -> dict:
        """``{amplitude: rows}``, in the order the colour scale runs."""
        by_amplitude: dict = {}
        for row in rows:
            by_amplitude.setdefault(row["amplitude"], []).append(row)
        return {amplitude: by_amplitude[amplitude]
                for amplitude in sorted(by_amplitude, key=lambda a: (a is None, a))}

    def _colour(self, amplitude):
        """The colour the grids give this drive, or the foreground otherwise."""
        return self._amplitude_to_color.get(
            amplitude, "w" if self._dark_mode else "k")

    # ── construction ─────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.toolbar = FitDisplayToolbar(name="histograms", all_amplitudes=False)
        layout.addWidget(self.toolbar)

        self._status = QtWidgets.QLabel("No fits to bin yet")
        layout.addWidget(self._status)

        container = QtWidgets.QWidget()
        self._grid = QtWidgets.QGridLayout(container)
        self._grid.setSpacing(10)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        layout.addWidget(scroll)

    def _lay_out(self, count: int) -> None:
        """Show *count* plots in two columns, making any that do not exist yet."""
        while self._grid.count():
            self._grid.takeAt(0)
        while len(self._plots) < count:
            plot = pg.PlotWidget()
            plot.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                               QtWidgets.QSizePolicy.Policy.Expanding)
            self._plots.append(plot)
        for index, plot in enumerate(self._plots):
            plot.setVisible(index < count)
            self._theme(plot)
            if index < count:
                self._grid.addWidget(plot, index // 2, index % 2)

    def _clear(self) -> None:
        for plot in self._plots:
            plot.clear()
            plot.hide()

    def _foreground(self) -> str:
        return "w" if self._dark_mode else "k"

    def _theme(self, plot) -> None:
        """One plot's background and axes, in the current theme."""
        plot.setBackground("k" if self._dark_mode else "w")
        for axis in ("left", "bottom"):
            plot.getAxis(axis).setPen(self._foreground())
            plot.getAxis(axis).setTextPen(self._foreground())
