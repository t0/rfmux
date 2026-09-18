"""Render per-resonator grids from multisweep entries."""

import weakref

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from rfmux.core.transferfunctions import (convert_roc_to_volts,
                                          convert_dacunits_to_volts)
from rfmux.tuning.bias import (
    BiasFinding, bifurcated_by_derivative, iq_arc_speed, iq_derivatives, normalized_arc_speed,
    hysteresis_separation)
from rfmux.tuning.fits import nonlinear_model_iq, skewed_model_magnitude

from .utils import (
    LINE_WIDTH, TABLEAU10_COLORS, COLORMAP_CHOICES, AMPLITUDE_COLORMAP_THRESHOLD,
    UPWARD_SWEEP_STYLE, DOWNWARD_SWEEP_STYLE,
    ClickableViewBox, square_axes, UnitConverter,
    DEFAULT_SUBPLOT_COLUMNS, iq_axis_labels, iq_unit_mode,
    magnitude_axis_labels,
)


def offset_khz(sweep):
    """A sweep's frequencies as kHz either side of where it was centred."""
    return (sweep['frequencies'] - sweep['original_center_frequency']) / 1e3


def sweep_iq(sweep, unit_mode):
    """A sweep's IQ, in the units the panel is displaying.

    Counts scale to volts by one constant, so a sweep still being measured --
    which carries counts and not yet the volts the finished entry also holds --
    draws on the same axes as a finished one.
    """
    return (sweep['iq_counts'] if iq_unit_mode(unit_mode) == 'counts'
            else convert_roc_to_volts(sweep['iq_counts']))


def update_sweep_grid(grid_layout, traces_by_name, plot_type, current_batch, batch_size,
                      amplitude_to_color, dark_mode, unit_mode='dbm', normalize=False,
                      prev_btn=None, next_btn=None, batch_label=None, widget_cache=None,
                      dac_scale=None, show_legend=True, fit_model='skewed',
                      bias_by_name=None, bias_settings=None,
                      on_resonator_double_click=None,
                      columns=DEFAULT_SUBPLOT_COLUMNS):
    """
    Update a grid layout with one subplot per resonator.

    Args:
        grid_layout: QGridLayout to populate with plots
        traces_by_name: ``{name: [(step, direction, amplitude, sweep), ...]}``,
            in the order to draw them; *sweep* is one of multisweep's entries
        plot_type: 'magnitude', 'iq', 'fit', 'bias', 'hysteresis' or 'frequency'
        current_batch: Current batch index (0-based)
        batch_size: Number of resonators per batch
        columns: Subplots per row
        amplitude_to_color: Dict mapping drive amplitude to colour
        dark_mode: Boolean for theme
        unit_mode: Unit mode for magnitude display ('counts', 'dbm', 'volts')
        normalize: Whether to normalize traces
        prev_btn: Optional previous batch button to enable/disable
        next_btn: Optional next batch button to enable/disable
        batch_label: Optional label to update with batch info
        widget_cache: Optional list to cache plot widgets for reuse
        dac_scale: Optional DAC scale (dBm) for formatting legend labels
        show_legend: Draw per-subplot legends (off when a colorbar is shown)
        fit_model: which model the ``fit`` plot type draws over the measurement
        bias_by_name: Optional ``{name: BiasFinding}``, marking each
            resonator's operating point on the sweeps it was chosen from
        bias_settings: Optional ``find_bias_points`` arguments, for the bars
            the 'bias' plot type draws
        on_resonator_double_click: Optional ``callable(name)``, called when a
            subplot is double-clicked, with the resonator it is drawing
    """
    if not traces_by_name:
        return

    # Remove all items from grid without deleting widgets (we'll reuse them)
    while grid_layout.count():
        grid_layout.takeAt(0)

    names = list(traces_by_name)

    # Calculate batch range
    start_idx = current_batch * batch_size
    end_idx = min(start_idx + batch_size, len(names))
    batch_names = names[start_idx:end_idx]

    if not batch_names:
        return

    # A fixed number of columns, so a subplot is the same size whichever page
    # of a batch is on screen. Narrowed for a batch that does not fill a row,
    # which would otherwise be one plot beside four gaps.
    num_plots = len(batch_names)
    ncols = max(1, min(columns, num_plots))
    nrows = int(np.ceil(num_plots / ncols))

    # Theme colors
    bg_color, pen_color = ("k", "w") if dark_mode else ("w", "k")

    # Reset ALL existing stretch factors to 0 (clears stale rows/cols from
    # a previously-larger grid that would otherwise keep consuming space).
    for r in range(grid_layout.rowCount()):
        grid_layout.setRowStretch(r, 0)
    for c in range(grid_layout.columnCount()):
        grid_layout.setColumnStretch(c, 0)

    # Set uniform stretch factors for the active grid
    for r in range(nrows):
        grid_layout.setRowStretch(r, 1)
    for c in range(ncols):
        grid_layout.setColumnStretch(c, 1)

    # Ensure widget cache exists
    if widget_cache is None:
        widget_cache = []

    # Expand cache if needed
    while len(widget_cache) < num_plots:
        widget_cache.append(_new_subplot(on_resonator_double_click))

    # Hide all cached widgets first
    for widget in widget_cache:
        widget.hide()

    # One legend entry per sweep of the call, so the same drive reads the same
    # in every subplot. Direction is the line style, and the label says which.
    legend_labels = {}
    for traces in traces_by_name.values():
        for step, direction, amplitude, _sweep in traces:
            label = UnitConverter.format_probe_label(amplitude, unit_mode, dac_scale)
            suffix = " (Down)" if direction == "downward" else " (Up)"
            legend_labels[(step, direction, amplitude)] = (
                label if plot_type == "hysteresis" else label + suffix)

    # Populate grid
    for idx, name in enumerate(batch_names):
        row = idx // ncols
        col = idx % ncols

        plot_widget = widget_cache[idx]
        plot_widget.setBackground(bg_color)
        plot_item = plot_widget.getPlotItem()

        if plot_item:
            # Clear previous data and legend
            plot_item.clear()
            if hasattr(plot_item, 'legend') and plot_item.legend:
                plot_item.legend.scene().removeItem(plot_item.legend)
                plot_item.legend = None

            traces = traces_by_name[name]
            center_hz = traces[0][3]['original_center_frequency'] if traces else None

            if center_hz is not None:
                plot_item.setTitle(f"{name} (f_central = {center_hz / 1e6:.4f} MHz)",
                                   color=pen_color)
            else:
                plot_item.setTitle(name, color=pen_color)

            # Which resonator this subplot is currently drawing, for a
            # double-click to name. On the view box rather than captured in the
            # connection, because a cached widget draws a different resonator
            # on every batch and the connection is made once.
            plot_widget.getViewBox().resonator_name = name

            # Style axes
            for axis_name in ("left", "bottom", "right", "top"):
                ax = plot_item.getAxis(axis_name)
                if ax:
                    ax.setPen(pen_color)
                    ax.setTextPen(pen_color)

            # Plot data with legend labels (suppressed when colorbar is active)
            labels = legend_labels if (show_legend and legend_labels) else None

            bias = (bias_by_name or {}).get(name)
            # Why a point is flagged is a sentence; the legend says *that* it
            # is, and the sentence lives here rather than across a subplot.
            plot_widget.setToolTip(
                "" if bias is None or bias.good
                else f"{name} is flagged: {bias.flagged_because}")

            if plot_type == 'magnitude':
                plot_magnitude(plot_item, traces, amplitude_to_color,
                               pen_color, unit_mode, normalize, labels, bias,
                               dac_scale)
                magnitude_axis_labels(plot_item, unit_mode, normalize)
            elif plot_type == 'bias':
                _plot_bifurcation(plot_item, traces, amplitude_to_color,
                                  pen_color, bias, bias_settings or {}, labels)
                plot_item.setLabel('left', 'IQ-speed change (× threshold)')
                plot_item.setLabel('bottom', 'Frequency Offset', units='kHz')
            elif plot_type == 'hysteresis':
                _plot_hysteresis(plot_item, traces, amplitude_to_color, pen_color,
                                 bias, bias_settings or {}, labels)
                plot_item.setLabel('bottom', 'Frequency Offset', units='kHz')
            elif plot_type == 'frequency':
                _plot_bias_frequency(plot_item, traces, amplitude_to_color,
                                     pen_color, bias, labels)
                plot_item.setLabel('left', 'IQ arc speed', units='Counts/Hz')
                plot_item.setLabel('bottom', 'Frequency Offset', units='kHz')
            elif plot_type == 'fit':
                _plot_fit(plot_item, traces, amplitude_to_color, pen_color,
                          fit_model, labels)
                # The fitters' normalization, not the drive's: a fit works on
                # the trace divided by its own off-resonance level, and this
                # tab draws measurement and model together in those units.
                plot_item.setLabel('left', 'Magnitude / off-resonance level')
                plot_item.setLabel('bottom', 'Frequency Offset', units='kHz')
            else:  # IQ
                plot_iq(plot_item, traces, amplitude_to_color,
                        pen_color, unit_mode, normalize, labels, bias,
                        dac_scale)
                iq_axis_labels(plot_item, unit_mode, normalize)
                square_axes(plot_item)

            plot_item.showGrid(x=True, y=True, alpha=0.3)

        grid_layout.addWidget(plot_widget, row, col)
        plot_widget.show()

    # Update batch navigation
    total_batches = max(1, (len(names) + batch_size - 1) // batch_size)
    if prev_btn:
        prev_btn.setEnabled(current_batch > 0)
    if next_btn:
        next_btn.setEnabled(current_batch < total_batches - 1)
    if batch_label:
        batch_label.setText(f"{current_batch + 1} of {total_batches}")


# ---------------------------------------------------------------------------
# Per-resonator plotting helpers
# ---------------------------------------------------------------------------

def _new_subplot(on_resonator_double_click):
    """One grid subplot, wired to say which resonator was double-clicked.

    A :class:`~rfmux.tools.periscope.utils.ClickableViewBox` rather than the
    plain one, so the panel's zoom box control reaches these plots as it does
    every other plot in Periscope.
    """
    view_box = ClickableViewBox()
    plot_widget = pg.PlotWidget(viewBox=view_box)
    plot_widget.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Expanding,
        QtWidgets.QSizePolicy.Policy.Expanding,
    )
    if on_resonator_double_click is not None:
        callback = _weakly(on_resonator_double_click)
        view_box.doubleClickedEvent.connect(
            lambda event, vb=view_box: _named_double_click(vb, event, callback()))
    return plot_widget


def _weakly(callback):
    """``callback()`` again, without owning it when it is a bound method.

    The subplot belongs to the panel and the method it calls is the panel's, so
    a connection that held the method strongly would close the cycle
    ``test_viewbox_lifetime.py`` exists to keep out. A plain function closes no
    such cycle and is held as it is.
    """
    try:
        return weakref.WeakMethod(callback)
    except TypeError:
        return lambda: callback


def _named_double_click(view_box, event, callback):
    """Hand *callback* the resonator this subplot is drawing, and take the event.

    Accepting it is what stops the view box falling through to its coordinate
    readout: a double-click here means "show me this one", not "what is under
    the cursor".
    """
    name = getattr(view_box, 'resonator_name', None)
    if name is None or callback is None:
        return
    callback(name)
    event.accept()


def _add_legend(plot_item, pen_color):
    legend_color = '#CCCCCC' if pen_color in ('w', (255, 255, 255)) else '#333333'
    plot_item.addLegend(offset=(10, -10), labelTextColor=legend_color)


#: How much wider the sweep a resonator is biased at is drawn than the rest.
CHOSEN_TRACE_WIDTH = 2.5


def _trace_pen(amplitude, direction, amplitude_to_color, pen_color,
               chosen=False):
    """Colour says drive amplitude, line style says direction.

    Every trace is coloured by its drive, including the only one of a
    single-amplitude sweep, so a colour does not change meaning as the later
    steps of a schedule arrive. Width is free, so it says which step this
    resonator is biased at once something has chosen one.
    """
    color = amplitude_to_color.get(amplitude, pen_color)
    style = DOWNWARD_SWEEP_STYLE if direction == "downward" else UPWARD_SWEEP_STYLE
    width = LINE_WIDTH * CHOSEN_TRACE_WIDTH if chosen else LINE_WIDTH
    return pg.mkPen(color=color, width=width, style=style)


def _biased_at(bias, step) -> bool:
    """Is *step* the amplitude step this resonator is biased at?"""
    return bias is not None and step == bias.iteration


#: The bias frequency line's own style. Dashed, where a sweep is solid upward
#: and dotted downward, so the line is not read as a third direction.
BIAS_LINE_STYLE = QtCore.Qt.PenStyle.DashLine


def bias_legend_label(bias) -> str:
    """What the bias frequency line is called, over two lines.

    The drive in normalized DAC units, which is what a bias amplitude *is* and
    what goes back into a re-run -- the colorbar carries the same number in
    whatever the panel is displaying.

    A flagged point names its flag here, because this is the mark on the plot
    that a flag is about. The words are the library's
    (:data:`~rfmux.tuning.bias.FLAG_KINDS`), so the plot and a notebook call a
    flag the same thing; the sentence behind it goes in the subplot's tooltip.
    """
    flag = "" if bias.good else f" \u2014 {bias.flagged_kind}"
    return f"f_bias{flag}<br>bias amp. = {bias.amplitude:.4g}"


def _bias_frequency_line(plot_item, bias, sweep, amplitude_to_color, pen_color):
    """A vertical line where the tone goes, in the chosen drive's own colour.

    Named in the legend, because a bare vertical line on a magnitude plot says
    nothing about which of the drives on screen it belongs to or whether the
    point is one to trust.
    """
    offset = (bias.frequency_hz - sweep['original_center_frequency']) / 1e3
    color = amplitude_to_color.get(bias.amplitude, pen_color)
    pen = pg.mkPen(color=color, width=LINE_WIDTH, style=BIAS_LINE_STYLE)
    plot_item.addLine(x=offset, pen=pen)
    if plot_item.legend is not None:
        legend_key(plot_item, bias_legend_label(bias), pen)


def _bias_point_marker(plot_item, bias, sweep, i_vals, q_vals,
                       amplitude_to_color, pen_color):
    """Where on this loop the tone will sit, read off the drawn trace.

    Interpolated on the trace as displayed -- normalized or not, counts or
    volts -- so the marker is on the line rather than beside it.
    """
    frequencies = np.asarray(sweep['frequencies'])
    order = np.argsort(frequencies)
    color = amplitude_to_color.get(bias.amplitude, pen_color)
    plot_item.plot(
        [np.interp(bias.frequency_hz, frequencies[order], np.asarray(i_vals)[order])],
        [np.interp(bias.frequency_hz, frequencies[order], np.asarray(q_vals)[order])],
        pen=None, symbol='o', symbolSize=9,
        symbolPen=pg.mkPen(color=color, width=2), symbolBrush=None)


def plot_magnitude(plot_item, traces, amplitude_to_color, pen_color,
                   unit_mode='dbm', normalize=False, legend_labels=None,
                   bias=None, dac_scale=None):
    """Plot |S21| against frequency offset for one resonator.

    Args:
        plot_item: PyQtGraph PlotItem
        traces: ``[(step, direction, amplitude, sweep), ...]``
        amplitude_to_color: Dict {amplitude: color}
        pen_color: Fallback pen color
        unit_mode: 'counts', 'dbm', or 'volts'
        normalize: Whether to state each sweep against the drive it was taken
            at, rather than as the power that came back
        legend_labels: Optional {(step, direction, amplitude): label}
        bias: Optional BiasFinding; its step is drawn thick and its frequency
            gets a line
        dac_scale: the module's DAC full scale in dBm, which normalizing to
            volts or dB needs and normalizing to counts does not
    """
    # A legend for the bias line even when the colorbar is carrying the drives:
    # the line is the one thing on this plot that is not a measurement, and it
    # is also where a flagged point is marked.
    if legend_labels or bias is not None:
        _add_legend(plot_item, pen_color)

    drawn = None
    for step, direction, amplitude, sweep in traces:
        counts = sweep['iq_counts']
        if len(counts) == 0:
            continue
        magnitude = UnitConverter.convert_amplitude(
            np.abs(counts), unit_mode, normalize=normalize,
            drive=amplitude, dac_scale=dac_scale)
        pen = _trace_pen(amplitude, direction, amplitude_to_color, pen_color,
                         chosen=_biased_at(bias, step))
        name = legend_labels.get((step, direction, amplitude)) if legend_labels else None
        plot_item.plot(offset_khz(sweep), magnitude, pen=pen, name=name)
        drawn = sweep

    # Any drawn sweep will do: they are all centred on the same frequency, and
    # the line is a frequency.
    if bias is not None and drawn is not None:
        _bias_frequency_line(plot_item, bias, drawn, amplitude_to_color, pen_color)


#: Points per measured point when drawing a model curve. A fit evaluated on the
#: sweep's own frequencies is a polyline through the corners the fit saw, which
#: is not what the model looks like.
MODEL_OVERSAMPLE = 25


def _model_on_a_finer_grid(reader, sweep):
    """``(offsets_khz, model)`` from a reader, on a denser axis than was measured."""
    frequencies = np.linspace(
        sweep['frequencies'][0], sweep['frequencies'][-1],
        MODEL_OVERSAMPLE * len(sweep['frequencies']))
    finer = {**sweep, 'frequencies': frequencies}
    return offset_khz(finer), reader(finer)


#: How to read each model off a sweep, and what to divide its magnitude by so
#: it lands on the same axis as the normalized measurement. The skewed fit
#: already works in those units; the nonlinear model is in counts, on top of
#: ``iq_counts``.
FIT_READERS = {
    'skewed': (skewed_model_magnitude, lambda counts: 1.0),
    'nonlinear': (nonlinear_model_iq, lambda counts: 1.0 / np.abs(counts[-1])),
}


def model_in_counts(sweep, fit_model):
    """``(offsets_khz, model)`` for one fitted sweep, in that sweep's own counts.

    Each reader works in its own units -- the skewed fit normalized to the
    trace's last point, the nonlinear model already in counts -- and
    :func:`_plot_fit` puts the *measurement* into the fit's units to draw them
    together. A plot drawn in the panel's units needs the opposite, so the same
    pair of numbers is applied the other way round. The nonlinear model stays
    complex, which is what puts it on an IQ plane; the skewed model is a
    magnitude and has no loop to draw.

    Raises:
        ValueError: if this sweep has no fit of *fit_model*, or one that did
            not converge -- the readers' own message says which.
    """
    reader, scale_of = FIT_READERS[fit_model]
    counts = np.asarray(sweep['iq_counts'])
    offsets, model = _model_on_a_finer_grid(reader, sweep)
    return offsets, model * scale_of(counts) * np.abs(counts[-1])


#: How wide the model is drawn. Thinner than the measurement it lies on, so
#: that where the two agree the coloured line is still visible under it.
MODEL_LINE_WIDTH = 1

#: How faint the line at a fit's ``fr`` is. It marks where the model put the
#: resonance; it is not a thing that was measured, so it sits behind the two
#: lines that were.
FR_LINE_ALPHA = 120


def si(value: float, digits: int = 3) -> str:
    """A Q, short enough for a legend: ``29.6k``, ``1.24M``.

    *digits* is significant figures; two is what an uncertainty gets.
    """
    if abs(value) >= 1e6:
        return f"{value / 1e6:.{digits}g}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.{digits}g}k"
    return f"{value:.{digits}g}"


#: What a model's legend entry says it fitted, in the order it says it. The
#: headline numbers only: everything a fit learned is in the entry, and a
#: legend that listed it all would cover the plot it labels.
FIT_LEGEND_PARAMS = (
    ("fr", lambda value: f"fr {value / 1e6:.4f} MHz"),
    ("Qr", lambda value: f"Qr {si(value)}"),
    ("Qi", lambda value: f"Qi {si(value)}"),
    ("a", lambda value: f"a {value:.2f}"),
)


def fit_legend_label(sweep, fit_model) -> str:
    """What one model's line is labelled: its name, and what it fitted.

    A fit that has no parameters -- it did not converge -- is named and left
    at that; there is no curve of it on the plot to label anyway.
    """
    params = ((sweep.get('fits') or {}).get(fit_model) or {}).get('params') or {}
    said = [say(params[name]) for name, say in FIT_LEGEND_PARAMS
            if params.get(name) is not None]
    name = fit_model.capitalize()
    return f"{name}: {', '.join(said)}" if said else name


def _once(label: str, said: set):
    """*label* the first time it is asked for, and None after that."""
    if label in said:
        return None
    said.add(label)
    return label


def _plot_fit(plot_item, traces, amplitude_to_color, pen_color, fit_model,
              legend_labels=None):
    """One resonator's measured magnitude with one model fitted to it.

    Normalized to each trace's last point, because that is the skewed fit's own
    convention: the model comes back in those units, so the measurement is put
    into them rather than the model taken out of them.

    The measurement keeps the line it has on the other tabs -- coloured by its
    drive, styled by its direction -- and the fit is a thinner line in the
    foreground colour over it, so that at a glance the black or white line is
    the model and the coloured one is the data. One model at a time, which
    leaves line style free to mean direction here as it does everywhere else.

    A sweep with no fit of this model draws its measurement alone, and one that
    did not converge is simply absent: the count of what failed is on the
    toolbar.

    The legend is always drawn here, because on this tab it says which line is
    the measurement and which the model -- a distinction the colorbar cannot
    make. With *legend_labels* it says that per trace, with the drive on the
    measurement and the fitted numbers on the model; without them, which is
    when the colorbar is carrying the drives and there are more traces than
    rows to spare, it says it once for the pair.
    """
    if traces:
        _add_legend(plot_item, pen_color)

    reader, scale_of = FIT_READERS[fit_model]
    # Which of the pair the one-entry-each legend has already named. The first
    # *drawn* line of each kind takes the entry, not the first trace: a fit
    # that did not converge draws nothing to hang it on.
    said = set()
    for step, direction, amplitude, sweep in traces:
        counts = np.asarray(sweep['iq_counts'])
        if len(counts) == 0 or counts[-1] == 0:
            continue
        plot_item.plot(
            offset_khz(sweep), np.abs(counts / counts[-1]),
            pen=_trace_pen(amplitude, direction, amplitude_to_color, pen_color),
            name=(legend_labels.get((step, direction, amplitude)) if legend_labels
                  else _once("Measured", said)))

        try:
            offsets, model = _model_on_a_finer_grid(reader, sweep)
        except (ValueError, KeyError):
            continue    # no fit of this model, or one that did not converge
        style = DOWNWARD_SWEEP_STYLE if direction == 'downward' else UPWARD_SWEEP_STYLE
        plot_item.plot(
            offsets, np.abs(model) * scale_of(counts),
            pen=pg.mkPen(color=pen_color, width=MODEL_LINE_WIDTH, style=style),
            name=(fit_legend_label(sweep, fit_model) if legend_labels
                  else _once(f"{fit_model.capitalize()} fit", said)))
        _fr_line(plot_item, sweep, fit_model, amplitude, amplitude_to_color,
                 pen_color)


def _fr_line(plot_item, sweep, fit_model, amplitude, amplitude_to_color, pen_color):
    """A line where this model put the resonance, in its sweep's drive colour.

    The colour rather than the model's own, because the reading is how far
    ``fr`` moved between one drive and the next, and that is only legible if
    each line is paired with the trace it came off.
    """
    params = ((sweep.get('fits') or {}).get(fit_model) or {}).get('params') or {}
    if params.get('fr') is None:
        return
    colour = pg.mkColor(amplitude_to_color.get(amplitude, pen_color))
    colour.setAlpha(FR_LINE_ALPHA)
    plot_item.addLine(
        x=(params['fr'] - sweep['original_center_frequency']) / 1e3,
        pen=pg.mkPen(color=colour, width=1, style=DOWNWARD_SWEEP_STYLE))


#: How faint the bar that did not bind is drawn, against the one that did.
UNBINDING_BAR_ALPHA = 160

#: How solid the band inside a bar is filled. A bar is a region -- everything
#: inside it is "not a spike" -- and a filled region says that at a glance
#: where two horizontal lines leave the eye to do the work.
BAR_FILL_ALPHA = 26
UNBINDING_FILL_ALPHA = 30

#: Fraction of the range left clear around the bifurcation plot's contents.
#: The bar is the outermost thing on a subplot where nothing crossed it, and
#: pyqtgraph's own padding is too small to tell it from the frame.
BAR_PADDING = 0.12


def _derivative_bars(entry, settings) -> tuple[float, float]:
    """``(prominence bar, noise bar)`` for one trace, from the library itself.

    The two are prominences in the same units and the detector applies the
    higher, so it reports only that one. Switching each off in turn is how the
    detector is asked for them separately -- the same trick the docstring
    recommends for finding out which bar was binding, and the reason nothing is
    recomputed here.
    """
    prominence = bifurcated_by_derivative(
        {"one": entry},
        spike_prominence_factor=settings.get("spike_prominence_factor", 0.5),
        noise_gate_factor=0.0).threshold
    noise = bifurcated_by_derivative(
        {"one": entry}, spike_prominence_factor=0.0,
        noise_gate_factor=settings.get("noise_gate_factor", 50.0)).threshold
    return prominence, noise


def _plot_bifurcation(plot_item, traces, amplitude_to_color, pen_color,
                      bias, settings, legend_labels=None):
    """Plot changes in normalized IQ speed divided by each trace's cutoff.

    The cutoff is the larger of the noise and span thresholds, drawn at ±1.
    Show the unused threshold faintly for the bias step. The detector tests
    prominence and adjacency; a crossing alone does not establish bifurcation.
    """
    if traces:
        _add_legend(plot_item, pen_color)

    # Room above the bar, so that when nothing reaches it the line reads as a
    # threshold rather than as the top of the frame.
    plot_item.getViewBox().setDefaultPadding(BAR_PADDING)
    _bar_band(plot_item, 1.0, pen_color, BAR_FILL_ALPHA)
    plot_item.addLine(y=1.0, pen=pg.mkPen(color=pen_color, width=1))
    plot_item.addLine(y=-1.0, pen=pg.mkPen(color=pen_color, width=1))

    # The bar that did not bind, over the chosen step's traces. Collected
    # rather than drawn inside the loop: that step is swept in both directions,
    # each with a bar of its own, and two translucent bands one on top of the
    # other read as one darker band that means nothing.
    unbinding = []
    # Which of the two was in force, over every trace drawn. A set, because
    # different steps of a schedule can be held by different bars, and a legend
    # that named one of them would be wrong on the others.
    binding_kinds = set()
    lower_kinds = set()

    for step, direction, amplitude, sweep in traces:
        try:
            frequencies, speed = normalized_arc_speed(sweep)
            prominence_bar, noise_bar = _derivative_bars(sweep, settings)
        except (ValueError, KeyError):
            continue        # too short or too flat to difference
        bar = max(prominence_bar, noise_bar)
        if bar <= 0:
            continue

        # A difference belongs between the two samples it was taken from.
        midpoints = 0.5 * (frequencies[:-1] + frequencies[1:])
        offsets = (midpoints - sweep['original_center_frequency']) / 1e3
        chosen = _biased_at(bias, step)
        name = legend_labels.get((step, direction, amplitude)) if legend_labels else None
        plot_item.plot(
            offsets, np.diff(speed) / bar,
            pen=_trace_pen(amplitude, direction, amplitude_to_color, pen_color,
                           chosen=chosen),
            name=name)

        binding_kinds.add(_bar_kind(prominence_bar, noise_bar))
        if chosen:
            unbinding.append(min(prominence_bar, noise_bar) / bar)
            lower_kinds.add("Spike" if noise_bar >= prominence_bar else "Noise")

    if traces:
        _bar_legend(plot_item, pen_color, binding_kinds, lower_kinds)

    if unbinding:
        colour = pg.mkColor(pen_color)
        colour.setAlpha(UNBINDING_BAR_ALPHA)
        faint = pg.mkPen(color=colour, width=1, style=DOWNWARD_SWEEP_STYLE)
        # Shade once at the smallest unused cutoff to avoid overlapping bands.
        _bar_band(plot_item, min(unbinding), pen_color, UNBINDING_FILL_ALPHA)
        for other in unbinding:
            for sign in (1.0, -1.0):
                plot_item.addLine(y=sign * other, pen=faint)


#: Sources of the derivative cutoff: arc-speed span and noise estimate.
BAR_NAMES = ("Spike", "Noise")


def _bar_kind(prominence_bar: float, noise_bar: float) -> str:
    """Which of the two was in force on one trace, by name.

    The detector applies the higher, and reports only that one, so this is the
    same comparison it made.
    """
    return BAR_NAMES[1] if noise_bar >= prominence_bar else BAR_NAMES[0]


def _bar_legend(plot_item, pen_color, binding_kinds: set, lower_kinds: set) -> None:
    """Label the prominence reference and the smaller selected-drive gate."""
    named = (next(iter(binding_kinds)) if len(binding_kinds) == 1
             else "Larger")
    legend_key(plot_item, f"±1 × {named.lower()} prominence threshold",
               pg.mkPen(color=pen_color, width=1), pen_color, BAR_FILL_ALPHA)
    if lower_kinds:
        lower = next(iter(lower_kinds)) if len(lower_kinds) == 1 else "Lower"
        colour = pg.mkColor(pen_color)
        colour.setAlpha(UNBINDING_BAR_ALPHA)
        legend_key(
            plot_item, f"{lower} threshold (selected amp.)",
            pg.mkPen(color=colour, width=1, style=DOWNWARD_SWEEP_STYLE),
            pen_color, UNBINDING_FILL_ALPHA)


def legend_key(plot_item, name: str, pen, fill_color=None, fill_alpha: int = 0) -> None:
    """A legend row for something that is not a plotted curve.

    ``ItemSample`` paints from an item's ``opts``, so a detached
    ``PlotDataItem`` carrying a pen -- and, for a band, the fill under it --
    draws that thing's own swatch without a second copy of it going onto the
    plot.
    """
    fill = {}
    if fill_color is not None:
        colour = pg.mkColor(fill_color)
        colour.setAlpha(fill_alpha)
        fill = {"fillLevel": 0, "fillBrush": pg.mkBrush(colour)}
    plot_item.legend.addItem(pg.PlotDataItem(pen=pen, **fill), name)


def _bar_band(plot_item, bar: float, pen_color, alpha: int) -> None:
    """Fill the band a bar encloses, behind everything drawn on top of it.

    Nested where both bars are shown: the inner band is the gate that did not
    bind, so the two shades together say how much of the bar in force is the
    noise gate and how much the prominence.
    """
    colour = pg.mkColor(pen_color)
    colour.setAlpha(alpha)
    band = pg.LinearRegionItem(
        values=(-bar, bar), orientation='horizontal', movable=False,
        brush=pg.mkBrush(colour), pen=pg.mkPen(None))
    band.setZValue(-10)
    plot_item.addItem(band)


def _plot_hysteresis(
    plot_item: pg.PlotItem, traces: list[tuple], amplitude_to_color: dict,
    pen_color: str, bias: BiasFinding | None, settings: dict,
    legend_labels: dict | None = None,
) -> None:
    """Draw the hysteresis detector's separation curve for each amplitude."""
    compare = settings.get("compare", "magnitude")
    limit = settings.get("max_discrepancy", 0.1)
    divisor = limit if limit > 0 else 1.0
    units = "dip depth" if compare == "magnitude" else "loop radius"
    ylabel = "Up/down difference / limit" if limit > 0 else f"Up/down difference / {units}"
    plot_item.setLabel('left', ylabel)
    _add_legend(plot_item, pen_color)
    pairs = {}
    for step, direction, amplitude, sweep in traces:
        pairs.setdefault(step, {})[direction] = sweep
    drawn = False
    for step, pair in pairs.items():
        try:
            frequencies, separation = hysteresis_separation(pair, compare=compare)
        except (ValueError, KeyError):
            continue
        sweep = pair["upward"]
        amplitude = sweep["sweep_amplitude"]
        label = legend_labels.get((step, "upward", amplitude)) if legend_labels else None
        plot_item.plot(
            (frequencies - sweep["original_center_frequency"]) / 1e3,
            separation / divisor,
            pen=_trace_pen(amplitude, "upward", amplitude_to_color, pen_color,
                           chosen=_biased_at(bias, step)), name=label)
        drawn = True
    pen = pg.mkPen(color=pen_color, width=1, style=DOWNWARD_SWEEP_STYLE)
    plot_item.addLine(y=limit / divisor, pen=pen)
    legend_key(plot_item, f"Limit: {limit:g} × {units}", pen)
    if not drawn:
        note = pg.TextItem("No usable up/down pairs", color=pen_color)
        plot_item.addItem(note)
    plot_item.getViewBox().setLimits(yMin=0)


#: The two components of the arc speed, under the line whose magnitude they
#: make. Green and red rather than the panel's usual I/Q blue and orange: those
#: two mean measured I and Q on the IQ tab, and these are their derivatives.
#: Neither is a ``TABLEAU10_COLORS`` entry, which is what a drive is drawn in
#: when there are few enough of them to have distinct colours -- the green in
#: that list is the third drive's -- and inferno, which carries the drives
#: above that, has no green at all and no red this bright.
DERIVATIVE_COLORS = {"dI/df": "#00B050", "dQ/df": "#FF2D2D"}

#: How wide a component is drawn, against the arc speed over it. Thin, because
#: the speed is the quantity the method maximizes and the components are what
#: it is made of.
DERIVATIVE_LINE_WIDTH = 1


def _plot_bias_frequency(plot_item, traces, amplitude_to_color, pen_color,
                         bias, legend_labels=None):
    """What choosing the bias frequency looked at, at the drive it was chosen at.

    :func:`~rfmux.tuning.bias.iq_arc_speed` is the quantity the default
    ``"iq_derivative"`` method maximizes -- how far the IQ trace moves per hertz
    -- so its peak is the answer and the line is where the tone will actually
    go, after ``BiasPoint`` puts it on the hardware grid. The gap between the
    two is that quantization, which is the reading this tab exists for.

    ``dI/df`` and ``dQ/df`` are drawn under it, thin, because which of them
    carries the response is the other half of the reading: a speed that comes
    almost entirely from one component is an IQ loop that is not oriented the
    way it was assumed to be.

    Only the step the resonator is biased at is drawn. The other steps chose
    nothing, and a grid of them would bury the one that did.
    """
    if traces:
        _add_legend(plot_item, pen_color)

    # Which of the components the legend has already named. One entry each,
    # not one per direction: the colour means the component, and the pair of
    # them is the same pair on every trace of the subplot.
    said = set()
    for step, direction, amplitude, sweep in traces:
        try:
            frequencies, dI_df, dQ_df = iq_derivatives(sweep)
            _same, speed = iq_arc_speed(sweep)
        except (ValueError, KeyError):
            continue        # too short or too degenerate to differentiate
        offsets = (frequencies - sweep['original_center_frequency']) / 1e3
        style = DOWNWARD_SWEEP_STYLE if direction == 'downward' else UPWARD_SWEEP_STYLE
        for label, values in (("dI/df", dI_df), ("dQ/df", dQ_df)):
            plot_item.plot(
                offsets, values,
                pen=pg.mkPen(color=DERIVATIVE_COLORS[label],
                             width=DERIVATIVE_LINE_WIDTH, style=style),
                name=_once(label, said))
        plot_item.plot(
            offsets, speed,
            pen=_trace_pen(amplitude, direction, amplitude_to_color, pen_color,
                           chosen=True),
            name=(legend_labels.get((step, direction, amplitude)) if legend_labels
                  else _once("IQ arc speed", said)))

    if bias is not None and traces:
        _bias_frequency_line(plot_item, bias, traces[0][3], amplitude_to_color,
                             pen_color)


def plot_iq(plot_item, traces, amplitude_to_color, pen_color,
            unit_mode='dbm', normalize=False, legend_labels=None, bias=None,
            dac_scale=None):
    """Plot the IQ loops of one resonator.

    Args:
        plot_item: PyQtGraph PlotItem
        traces: ``[(step, direction, amplitude, sweep), ...]``
        amplitude_to_color: Dict {amplitude: color}
        pen_color: Fallback pen color
        unit_mode: 'counts' draws raw IQ, anything else the entry's volts
        normalize: Whether to divide each loop by the drive it was taken at
        legend_labels: Optional {(step, direction, amplitude): label}
        bias: Optional BiasFinding; its step is drawn thick and the point the
            tone sits at is marked on it
        dac_scale: the module's DAC full scale in dBm, which normalizing volts
            needs and normalizing counts does not

    A loop's axis is I, not frequency, so the bias frequency cannot be a
    vertical line here as it is on the magnitude plot. It is the point of the
    loop the tone will sit on, which is what the marker is.

    Every loop is divided by its own drive, so the loops keep their sizes
    relative to each other: a resonator driven harder really does return a
    bigger loop, and dividing each by its own peak -- which is what a plot
    normalized to itself does -- would throw away the comparison the grid is
    drawn for.
    """
    if legend_labels:
        _add_legend(plot_item, pen_color)

    for step, direction, amplitude, sweep in traces:
        iq = sweep_iq(sweep, unit_mode)
        if len(iq) == 0:
            continue

        i_vals, q_vals = np.real(iq), np.imag(iq)
        if normalize and UnitConverter.can_normalize(
                iq_unit_mode(unit_mode), amplitude, dac_scale):
            divisor = (amplitude if iq_unit_mode(unit_mode) == 'counts'
                       else convert_dacunits_to_volts(amplitude, dac_scale))
            i_vals, q_vals = i_vals / divisor, q_vals / divisor

        chosen = _biased_at(bias, step)
        pen = _trace_pen(amplitude, direction, amplitude_to_color, pen_color,
                         chosen=chosen)
        name = legend_labels.get((step, direction, amplitude)) if legend_labels else None
        plot_item.plot(i_vals, q_vals, pen=pen, name=name)

        if chosen:
            _bias_point_marker(plot_item, bias, sweep, i_vals, q_vals,
                               amplitude_to_color, pen_color)


def create_amplitude_color_map(amplitude_values, dark_mode):
    """
    Create a color mapping for drive amplitudes.

    Uses TABLEAU10_COLORS for few amplitudes, colormap for many.

    Args:
        amplitude_values: Iterable of amplitude values
        dark_mode: Boolean for theme

    Returns:
        Dict mapping amplitude values to colors
    """
    sorted_amplitudes = sorted(set(amplitude_values))
    num_amps = len(sorted_amplitudes)

    if num_amps == 0:
        return {}

    amplitude_to_color = {}
    cmap_name = COLORMAP_CHOICES.get("AMPLITUDE_SWEEP", "inferno")
    use_cmap = pg.colormap.get(cmap_name) if cmap_name else None

    for amp_idx, amp_val in enumerate(sorted_amplitudes):
        if num_amps <= AMPLITUDE_COLORMAP_THRESHOLD:
            color = TABLEAU10_COLORS[amp_idx % len(TABLEAU10_COLORS)]
        else:
            if use_cmap:
                # As a QColor rather than the RGBA array the colormap hands
                # back by default: a scatter plot reads an array of colour
                # components as one pen per point.
                normalized_idx = amp_idx / max(1, num_amps - 1)
                if dark_mode:
                    map_value = 0.3 + normalized_idx * 0.7
                else:
                    map_value = normalized_idx * 0.75
                color = use_cmap.map(map_value, mode=pg.ColorMap.QCOLOR)
            else:
                color = TABLEAU10_COLORS[amp_idx % len(TABLEAU10_COLORS)]

        amplitude_to_color[amp_val] = color

    return amplitude_to_color
