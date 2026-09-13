"""One resonator, in detail: every drive it was swept at, and the one it is biased at.

The grids answer a question about the array — which resonator looks wrong — and
cannot answer the next one, because a subplot a couple of hundred pixels wide
has no room for it. This tab is one resonator at the size of the panel: what it
did at every drive, what the drive it is biased at looks like with the fitted
model over it, and every number the fits and the bias search produced for it.

Nothing here is measured. Every number comes off the block the panel already
holds -- its sweeps, the fits in them and the bias report beside them -- through
the library's own readers, so a notebook draws the same figures from the same
numbers.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import pyqtSignal

from rfmux.tuning.fits import FIT_PARAMS, collect_fit_params

from .amplitude_colorbar import AmplitudeColorBar
from .fit_display_toolbar import FitDisplayToolbar
from .layouts import FlowLayout, labelled
from .multisweep_grid_helpers import (
    MODEL_LINE_WIDTH, iq_axis_labels, magnitude_axis_labels, model_in_counts,
    plot_iq, plot_magnitude, si)
from .utils import (
    AMPLITUDE_COLORMAP_THRESHOLD, ClickableViewBox, TABLEAU10_COLORS,
    UnitConverter, square_axes)

#: The model curve's colour on the bias plot. Not the foreground, as the Fit
#: Results grid draws it: there the model lies over traces of every drive,
#: here it lies over one, and the quiet end of a schedule is drawn in the
#: near-black bottom of the inferno colormap that a black line vanishes into.
#: Cyan is in neither that colormap nor the first three ``TABLEAU10_COLORS``,
#: which is what a drive is drawn in when there are few enough of them.
MODEL_COLOR = "#00B0F0"

#: How wide a row that is a sentence rather than a number may get before it
#: wraps. Long enough to read, short enough that one sentence does not set the
#: width of the column it is in.
SENTENCE_WRAP_PX = 320

#: How a fitted parameter and its error are written, and in what unit. The
#: error gets two significant figures, in the unit its value is written in:
#: more digits of an uncertainty is a claim about the claim. Anything not named
#: here is four significant figures in the units the fit reports it in.
PARAM_FORMATS = {
    "fr": (lambda value: f"{value / 1e6:.6f}", lambda e: f"{e / 1e6:.2g}", "MHz"),
    "Qr": (si, lambda e: si(e, 2), ""),
    "Qc": (si, lambda e: si(e, 2), ""),
    "Qi": (si, lambda e: si(e, 2), ""),
}

#: How everything else is written.
DEFAULT_FORMAT = (lambda value: f"{value:.4g}", lambda e: f"{e:.2g}", "")

#: What each parameter means, for the row that carries it.
PARAM_TOOLTIPS = {
    "fr": "Resonant frequency the model put the resonance at",
    "Qr": "Total quality factor",
    "Qc": "Coupling quality factor",
    "Qi": "Internal quality factor",
    "Qcre": "Real part of the complex coupling quality factor",
    "Qcim": "Imaginary part of the complex coupling quality factor",
    "A": "Overall scale of the skewed model",
    "a": "Nonlinearity; at or past 4√3/9 ≈ 0.77 the model is "
         "multivalued and the tone is driving the resonator into bifurcation",
    "amp": "Depth of the resonance in the nonlinear model",
    "phi": "Rotation of the resonance circle, in radians",
    "i0": "I offset of the nonlinear model",
    "q0": "Q offset of the nonlinear model",
}


class _Row(NamedTuple):
    """One ``parameter: value`` line of a column below the plots."""

    name: str
    value: str
    tooltip: str = ""
    warn: bool = False   # drawn in the flag colour
    wrap: bool = False   # a sentence, which wraps; a number never does


class _Column(NamedTuple):
    """One heading and the rows under it."""

    title: str
    rows: list


class DetectorDigestTab(QtWidgets.QWidget):
    """One resonator's sweeps, bias point and fits, on one page.

    The panel owns the measurement; this owns only which resonator and which
    model are on screen. It says so with :attr:`display_changed`, and the panel
    feeds it again through :meth:`show_resonator`.
    """

    #: Emitted when the resonator or the model changes.
    display_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._traces: list = []
        self._sweeps = None
        self._bias = None
        self._amplitude_to_color: dict = {}
        self._dark_mode = False
        self._unit_mode = "dbm"
        self._normalize = False
        self._dac_scale = None
        # So the arrow keys reach keyPressEvent: a plain QWidget takes no focus,
        # and the tab would never see them.
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._setup_ui()
        self.toolbar.display_changed.connect(self.display_changed)

    # ── which resonator ──────────────────────────────────────────────────────

    def resonator(self):
        """The resonator on screen, or None if nothing has been swept."""
        return self.name_combo.currentData()

    def set_resonators(self, names) -> None:
        """Offer *names*, in the catalog's order, keeping the one on screen.

        A measurement that no longer holds the resonator being shown falls back
        to its first, which is the only thing it can draw.
        """
        before = self.resonator()
        self.name_combo.blockSignals(True)
        self.name_combo.clear()
        for name in names:
            self.name_combo.addItem(name, name)
        self.name_combo.setCurrentIndex(max(0, self.name_combo.findData(before)))
        self.name_combo.blockSignals(False)
        self._say_where()
        if self.resonator() != before:
            self.display_changed.emit()

    def select(self, name: str) -> None:
        """Show *name*, if this measurement has it."""
        index = self.name_combo.findData(name)
        if index >= 0:
            self.name_combo.setCurrentIndex(index)

    def show_resonator(self, *, traces, sweeps, bias, amplitude_to_color,
                       dark_mode, unit_mode, normalize, dac_scale) -> None:
        """Draw the resonator on screen from what the panel holds.

        *traces* is ``[(step, direction, amplitude, sweep), ...]`` for this
        resonator alone, as the grids draw; *sweeps* the module block the fits
        are read out of, None while a call is still running; *bias* its
        :class:`~rfmux.tuning.bias.BiasFinding`, None until one is found.
        """
        self._traces = sorted(traces, key=lambda trace: (trace[0], trace[1]))
        self._sweeps = sweeps
        self._bias = bias
        self._amplitude_to_color = amplitude_to_color
        self._dark_mode = dark_mode
        self._unit_mode = unit_mode
        self._normalize = normalize
        self._dac_scale = dac_scale
        self.redraw()

    def redraw(self) -> None:
        """Rebuild the three plots and the columns under them."""
        for plot in self._plots:
            self._theme(plot)
        if self.resonator() is None or not self._traces:
            for plot in self._plots:
                plot.clear()
            self.colorbar.hide()
            self._show_columns([])
            return

        self._show_colorbar()
        self._draw_sweeps(self._plots[0])
        self._draw_loops(self._plots[1])
        self._draw_bias_sweep(self._plots[2], self.toolbar.get_model())
        self._show_columns([self._bias_column()] + self._fit_columns())

    def _bias_traces(self) -> list:
        """The sweeps taken at the amplitude step this resonator is biased at.

        Empty until a bias has been found: no step is the one it is biased at
        until something has chosen one.
        """
        if self._bias is None:
            return []
        return [trace for trace in self._traces if trace[0] == self._bias.iteration]

    # ── the plots ────────────────────────────────────────────────────────────

    def _draw_sweeps(self, plot) -> None:
        """The measurement at every drive, and which of them was chosen.

        The measurement alone: what the fit made of it is on the third plot,
        over the one sweep it is about.
        """
        plot_item = self._fresh(plot, self._sweep_title())
        self._legend(plot_item)
        plot_magnitude(plot_item, self._traces, self._amplitude_to_color,
                       self._foreground(), self._unit_mode, self._normalize,
                       self._legend_labels(), self._bias)
        magnitude_axis_labels(plot_item, self._unit_mode, self._normalize)

    def _sweep_title(self) -> str:
        """The resonator and where it was centred, as the grids title a subplot."""
        centre = self._traces[0][3]['original_center_frequency']
        return f"{self.resonator()} (f_central = {centre / 1e6:.4f} MHz)"

    def _draw_loops(self, plot) -> None:
        """The same sweeps as loops, with the tone's place on the biased one."""
        plot_item = self._fresh(plot, "IQ loops")
        self._legend(plot_item)
        plot_iq(plot_item, self._traces, self._amplitude_to_color,
                self._foreground(), self._unit_mode, self._normalize,
                self._legend_labels(), self._bias)
        iq_axis_labels(plot_item, self._unit_mode)
        square_axes(plot_item)

    def _draw_bias_sweep(self, plot, model) -> None:
        """The one sweep the operating point is on: its model, and its frequency.

        One drive rather than all of them, because this is the plot about the
        bias point: the model that was fitted to *that* sweep, and the line
        where the tone will actually sit.
        """
        traces = self._bias_traces()
        if not traces:
            plot_item = self._fresh(plot, "The bias point — none found yet")
            magnitude_axis_labels(plot_item, self._unit_mode, self._normalize)
            return

        drive = UnitConverter.format_probe_label(
            self._bias.amplitude, self._unit_mode, self._dac_scale)
        plot_item = self._fresh(plot, f"The bias point, at {drive}")
        self._legend(plot_item)
        plot_magnitude(plot_item, traces, self._amplitude_to_color,
                       self._foreground(), self._unit_mode, self._normalize,
                       self._labels_for(traces), self._bias)
        if model is not None:
            label = f"{model.capitalize()} fit"
            for _step, direction, _amplitude, sweep in traces:
                try:
                    offsets, curve = model_in_counts(sweep, model)
                except (ValueError, KeyError):
                    continue  # no fit of this model, or one that did not converge
                plot_item.plot(offsets, self._on_measurement_axis(np.abs(curve), sweep),
                               pen=self._model_pen(direction), name=label)
                # One entry for the model, however many of them are drawn: the
                # drives are already named, and the model is the other kind of
                # line rather than another sweep.
                label = None
        magnitude_axis_labels(plot_item, self._unit_mode, self._normalize)

    def _on_measurement_axis(self, magnitude, sweep):
        """A model's magnitude, in counts, put on the axis the sweep is drawn on.

        Converted with the measurement in front of it, so that a normalized
        plot divides the model by the measurement's own first point — the
        reference :func:`plot_magnitude` used — rather than by the model's.
        """
        counts = np.asarray(sweep['iq_counts'])
        both = UnitConverter.convert_amplitude(
            np.concatenate([np.abs(counts), magnitude]), counts,
            self._unit_mode, normalize=self._normalize)
        return both[len(counts):]

    def _legend_labels(self):
        """Drive labels for the two plots that draw every sweep.

        None once there are more drives than a legend can carry, which is when
        the colorbar above the plots takes them over -- the same rule, and the
        same threshold, as the grids.
        """
        if len(self._drives_drawn()) > AMPLITUDE_COLORMAP_THRESHOLD:
            return None
        return self._labels_for(self._traces)

    def _labels_for(self, traces) -> dict:
        """One entry per sweep: its drive, and which way it was swept."""
        return {
            (step, direction, amplitude):
                UnitConverter.format_probe_label(
                    amplitude, self._unit_mode, self._dac_scale)
                + (" (Down)" if direction == "downward" else " (Up)")
            for step, direction, amplitude, _sweep in traces
        }

    def _drives_drawn(self) -> set:
        """The drive amplitudes this resonator's traces were taken at."""
        return {amplitude for _step, _direction, amplitude, _sweep in self._traces}

    def _show_colorbar(self) -> None:
        """The drives as a scale, when there are too many of them to label.

        Over the measurement's whole scale rather than this resonator's, so the
        colours mean the same thing here as on the grids.
        """
        if len(self._drives_drawn()) <= AMPLITUDE_COLORMAP_THRESHOLD:
            self.colorbar.hide()
            return
        amplitudes = sorted(self._amplitude_to_color)
        self.colorbar.update_range(
            amplitudes[0], amplitudes[-1], self._dac_scale, self._unit_mode,
            self._dark_mode,
            any(direction == "downward"
                for _step, direction, _amplitude, _sweep in self._traces))
        self.colorbar.show()

    def _model_pen(self, direction):
        style = (QtCore.Qt.PenStyle.DotLine if direction == "downward"
                 else QtCore.Qt.PenStyle.SolidLine)
        return pg.mkPen(color=MODEL_COLOR, width=MODEL_LINE_WIDTH, style=style)

    # ── the columns ──────────────────────────────────────────────────────────

    def _bias_column(self) -> _Column:
        """How this resonator is biased, and what is wrong with it if anything.

        The flag lives here rather than over the plots: it is a property of the
        bias point, so it belongs beside the numbers that describe it.
        """
        if self._bias is None:
            return _Column("Bias point", [_Row(
                "", "No bias point yet. Find Bias chooses the drive and the "
                    "frequency this page is about.", wrap=True)])

        bias = self._bias
        rows = [
            _Row("Amplitude step", str(bias.iteration),
                 "Which step of the schedule the operating point came off"),
            _Row("Drive", UnitConverter.format_probe_label(
                bias.amplitude, self._unit_mode, self._dac_scale),
                 "In the units the panel is displaying; the plot's f_bias "
                 "legend carries it in normalized DAC units, which is what "
                 "goes back into a re-run"),
            _Row("Frequency", f"{bias.frequency_hz / 1e6:.6f} MHz",
                 "Where the tone goes, on the hardware's own frequency grid"),
            _Row("dI/df", f"{bias.dI_df:.4g} V/Hz",
                 "How fast I moves with frequency at the bias point"),
            _Row("dQ/df", f"{bias.dQ_df:.4g} V/Hz",
                 "How fast Q moves with frequency at the bias point"),
            _Row("|dIQ/df|", f"{np.hypot(bias.dI_df, bias.dQ_df):.4g} V/Hz",
                 "The responsivity the frequency search maximized"),
            _Row("Bifurcated at", self._bifurcated_at(),
                 "The quietest drive at which the bifurcation test fired"),
        ]
        if not bias.good:
            rows.append(_Row("Flagged", f"{bias.flagged_kind}: {bias.flagged_because}",
                             "The answer is a default rather than something the "
                             "amplitude steps established", warn=True, wrap=True))
        return _Column("Bias point", rows)

    def _bifurcated_at(self) -> str:
        """The drive bifurcation was first seen at, in the panel's own units."""
        if self._bias.bifurcated_at is None:
            return "not seen"
        return UnitConverter.format_probe_label(
            self._bias.bifurcated_at, self._unit_mode, self._dac_scale)

    def _fit_columns(self) -> list:
        """One column per fit of the sweep this resonator is biased at.

        The sweeps at the other drives were fitted too; what this page is about
        is the one the tone will sit on, and how the array's fits move with
        drive is the Fit Histograms tab's question.
        """
        if self._bias is None or self._sweeps is None:
            return []
        return [self._fit_column(model, row)
                for model in FIT_PARAMS
                for row in collect_fit_params(
                    self._sweeps, model, names=[self.resonator()],
                    iterations=[self._bias.iteration])]

    def _fit_column(self, model: str, row: dict) -> _Column:
        """One fit: every parameter it found, and what the fitter made of it."""
        rows = []
        for param in FIT_PARAMS[model]:
            value = row["params"].get(param)
            if value is None:
                continue
            error = (row["errors"] or {}).get(param)
            rows.append(_Row(param, self._parameter(param, value, error),
                             self._parameter_tooltip(param, value, error)))
        if row["failed_because"]:
            rows.append(_Row("Rejected", row["failed_because"],
                             "What it converged on is usually the clue, so the "
                             "parameters are here anyway", warn=True, wrap=True))
        return _Column(f"{model.capitalize()} fit — {row['direction']}", rows)

    @staticmethod
    def _parameter(param: str, value, error) -> str:
        """A fitted parameter and its error, in the unit the row is written in."""
        say, say_error, unit = PARAM_FORMATS.get(param, DEFAULT_FORMAT)
        text = say(value)
        if error is not None and np.isfinite(error):
            text += f" ± {say_error(error)}"
        return f"{text} {unit}".strip()

    @staticmethod
    def _parameter_tooltip(param: str, value, error) -> str:
        """What the parameter means, and the number the fit actually produced.

        float(), because a numpy scalar reprs as ``np.float64(...)`` and this
        is the place the full precision is readable.
        """
        said = f"{param} = {float(value)!r}"
        if error is not None:
            said += f" ± {float(error)!r}"
        meaning = PARAM_TOOLTIPS.get(param)
        return f"{meaning}\n{said}" if meaning else said

    # ── what it is saying ────────────────────────────────────────────────────

    def _say_where(self) -> None:
        count = self.name_combo.count()
        self.where_label.setText(
            f"{self.name_combo.currentIndex() + 1} of {count}" if count else "")

    # ── construction ─────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(self._navigation())

        self.colorbar = AmplitudeColorBar(self)
        self.colorbar.hide()
        layout.addWidget(self.colorbar)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(self._plot_row())
        splitter.addWidget(self._column_area())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        # A starting split as well as a stretch: the stretch alone divides the
        # space a resize adds, and the columns' own size hint is one row.
        splitter.setSizes([560, 300])
        layout.addWidget(splitter)

    def _plot_row(self) -> QtWidgets.QWidget:
        plots = QtWidgets.QWidget()
        side_by_side = QtWidgets.QHBoxLayout(plots)
        side_by_side.setContentsMargins(0, 0, 0, 0)
        self._plots = []
        for _ in range(3):
            plot = pg.PlotWidget(viewBox=ClickableViewBox())
            plot.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                               QtWidgets.QSizePolicy.Policy.Expanding)
            self._plots.append(plot)
            side_by_side.addWidget(plot)
        return plots

    def _column_area(self) -> QtWidgets.QWidget:
        """Where the columns of numbers go: wrapping, and scrolled if they must.

        A flow layout, so a narrow panel puts the later columns on a second row
        rather than off the edge.
        """
        self._columns_host = QtWidgets.QWidget()
        self._columns_layout = FlowLayout(self._columns_host, margin=0)

        self._nothing_yet = QtWidgets.QLabel("Nothing swept yet")
        self._columns_layout.addWidget(self._nothing_yet)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._columns_host)
        return scroll

    def _show_columns(self, columns) -> None:
        """Replace what is under the plots with *columns*."""
        while self._columns_layout.count():
            item = self._columns_layout.takeAt(0)
            widget = item.widget()
            if widget is not None and widget is not self._nothing_yet:
                widget.setParent(None)
                widget.deleteLater()
        self._nothing_yet.setVisible(not columns)
        if not columns:
            self._columns_layout.addWidget(self._nothing_yet)
            return
        for column in columns:
            self._columns_layout.addWidget(self._column_widget(column))

    def _column_widget(self, column: _Column) -> QtWidgets.QWidget:
        """One heading with its ``parameter: value`` rows under it."""
        box = QtWidgets.QGroupBox(column.title)
        grid = QtWidgets.QGridLayout(box)
        grid.setContentsMargins(8, 4, 8, 4)
        grid.setVerticalSpacing(2)
        for index, row in enumerate(column.rows):
            name = QtWidgets.QLabel(f"{row.name}:" if row.name else "")
            value = QtWidgets.QLabel(row.value)
            # A number keeps its line, however wide that makes the column:
            # "1001.610484 ± 0.000274 MHz" broken over two lines to save a
            # centimetre makes a column of them unreadable. A sentence wraps.
            if row.wrap:
                value.setWordWrap(True)
                value.setMaximumWidth(SENTENCE_WRAP_PX)
            # Selectable, because these are numbers people copy into a notebook.
            value.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            if row.warn:
                value.setStyleSheet(f"color: {TABLEAU10_COLORS[3]};")
            if row.tooltip:
                name.setToolTip(row.tooltip)
                value.setToolTip(row.tooltip)
            grid.addWidget(name, index, 0, QtCore.Qt.AlignmentFlag.AlignTop)
            grid.addWidget(value, index, 1, QtCore.Qt.AlignmentFlag.AlignTop)
        return box

    def _navigation(self) -> QtWidgets.QWidget:
        """The resonator this tab is on, and the fit it is drawing."""
        bar = QtWidgets.QWidget()
        row = FlowLayout(bar, margin=0)

        self.prev_btn = QtWidgets.QPushButton("◀")
        self.prev_btn.setToolTip("The resonator before this one (Left arrow)")
        self.prev_btn.clicked.connect(self._previous)

        self.name_combo = QtWidgets.QComboBox()
        self.name_combo.setToolTip(
            "Which resonator this page is about, in the catalog's own order")
        self.name_combo.currentIndexChanged.connect(self._name_changed)

        self.next_btn = QtWidgets.QPushButton("▶")
        self.next_btn.setToolTip("The resonator after this one (Right arrow)")
        self.next_btn.clicked.connect(self._next)

        row.addWidget(self.prev_btn)
        row.addWidget(labelled("Resonator:", self.name_combo))
        row.addWidget(self.next_btn)

        self.where_label = QtWidgets.QLabel("")
        row.addWidget(self.where_label)

        # The model only: this page is about one drive, so there is no drive
        # to choose.
        self.toolbar = FitDisplayToolbar(name="digest", amplitudes=False)
        row.addWidget(self.toolbar)
        return bar

    def _name_changed(self) -> None:
        self._say_where()
        self.display_changed.emit()

    def _previous(self) -> None:
        self._step_resonator(-1)

    def _next(self) -> None:
        self._step_resonator(1)

    def _step_resonator(self, delta: int) -> None:
        """Bound methods, not lambdas: a lambda slot on one of this tab's own
        buttons would own the tab, which is the reference cycle
        ``test_viewbox_lifetime.py`` exists to keep out."""
        index = self.name_combo.currentIndex() + delta
        if 0 <= index < self.name_combo.count():
            self.name_combo.setCurrentIndex(index)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Left and right walk the array, as the buttons do."""
        if event.key() == QtCore.Qt.Key.Key_Left:
            self._step_resonator(-1)
        elif event.key() == QtCore.Qt.Key.Key_Right:
            self._step_resonator(1)
        else:
            super().keyPressEvent(event)

    def enable_zoom_box(self, enable: bool) -> None:
        """Follow the panel's zoom box control, as its grids do."""
        for plot in self._plots:
            plot.getViewBox().enableZoomBoxMode(enable)

    def _fresh(self, plot, title: str):
        """One plot, emptied of the last resonator and titled for this one."""
        plot_item = plot.getPlotItem()
        plot_item.clear()
        if getattr(plot_item, 'legend', None):
            plot_item.legend.scene().removeItem(plot_item.legend)
            plot_item.legend = None
        plot_item.setTitle(title, color=self._foreground())
        plot_item.showGrid(x=True, y=True, alpha=0.3)
        return plot_item

    def _legend(self, plot_item) -> None:
        """A legend in the bottom left, as the grids draw one.

        ``addLegend`` hands back the one that is already there, so the plotters
        below add their entries to this rather than a second.
        """
        plot_item.addLegend(offset=(10, -10), labelTextColor=self._legend_colour())

    def _legend_colour(self) -> str:
        return '#CCCCCC' if self._dark_mode else '#333333'

    def _foreground(self) -> str:
        return "w" if self._dark_mode else "k"

    def _theme(self, plot) -> None:
        plot.setBackground("k" if self._dark_mode else "w")
        for axis in ("left", "bottom"):
            plot.getAxis(axis).setPen(self._foreground())
            plot.getAxis(axis).setTextPen(self._foreground())
