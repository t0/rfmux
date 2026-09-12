"""
Helper functions for rendering per-resonator grid plots in multisweep panels.

The grids render what ``multisweep`` returned: a trace is one of its sweep
entries, read here and not copied. The caller says which resonators to draw and
hands over their sweeps; everything on a subplot comes off the entry.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets

from rfmux.core.transferfunctions import convert_roc_to_volts
from rfmux.tuning.bias import (
    bifurcated_by_derivative, iq_arc_speed, normalized_arc_speed)
from rfmux.tuning.fits import nonlinear_model_iq, skewed_model_magnitude

from .utils import (
    LINE_WIDTH, TABLEAU10_COLORS, COLORMAP_CHOICES, AMPLITUDE_COLORMAP_THRESHOLD,
    UPWARD_SWEEP_STYLE, DOWNWARD_SWEEP_STYLE,
    square_axes, UnitConverter,
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
    return sweep['iq_counts'] if unit_mode == 'counts' else convert_roc_to_volts(sweep['iq_counts'])


def update_sweep_grid(grid_layout, traces_by_name, plot_type, current_batch, batch_size,
                      amplitude_to_color, dark_mode, unit_mode='dbm', normalize=False,
                      prev_btn=None, next_btn=None, batch_label=None, widget_cache=None,
                      dac_scale=None, show_legend=True, fit_model='skewed',
                      bias_by_name=None, bias_settings=None):
    """
    Update a grid layout with one subplot per resonator.

    Args:
        grid_layout: QGridLayout to populate with plots
        traces_by_name: ``{name: [(step, direction, amplitude, sweep), ...]}``,
            in the order to draw them; *sweep* is one of multisweep's entries
        plot_type: 'magnitude', 'iq', 'fit', 'bias' or 'frequency'
        current_batch: Current batch index (0-based)
        batch_size: Number of resonators per batch
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

    # Calculate grid dimensions — use ceil(sqrt(n)) for a balanced grid
    num_plots = len(batch_names)
    ncols = max(1, int(np.ceil(np.sqrt(num_plots))))
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
        plot_widget = pg.PlotWidget()
        plot_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        widget_cache.append(plot_widget)

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
            legend_labels[(step, direction, amplitude)] = label + suffix

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

            # Style axes
            for axis_name in ("left", "bottom", "right", "top"):
                ax = plot_item.getAxis(axis_name)
                if ax:
                    ax.setPen(pen_color)
                    ax.setTextPen(pen_color)

            # Plot data with legend labels (suppressed when colorbar is active)
            labels = legend_labels if (show_legend and legend_labels) else None

            bias = (bias_by_name or {}).get(name)

            if plot_type == 'magnitude':
                _plot_magnitude(plot_item, traces, amplitude_to_color,
                                pen_color, unit_mode, normalize, labels, bias)
                # Y-axis label
                if normalize:
                    units = 'dB' if unit_mode == "dbm" else ''
                    plot_item.setLabel('left', 'Normalized Magnitude', units=units)
                else:
                    if unit_mode == "counts":
                        plot_item.setLabel('left', 'Magnitude', units='Counts')
                    elif unit_mode == "dbm":
                        plot_item.setLabel('left', 'Power', units='dBm')
                    elif unit_mode == "volts":
                        plot_item.setLabel('left', 'Magnitude', units='V')
                plot_item.setLabel('bottom', 'Frequency Offset', units='kHz')
            elif plot_type == 'bias':
                _plot_bifurcation(plot_item, traces, amplitude_to_color,
                                  pen_color, bias, bias_settings or {}, labels)
                plot_item.setLabel('left', 'Change in arc speed / bar')
                plot_item.setLabel('bottom', 'Frequency Offset', units='kHz')
            elif plot_type == 'frequency':
                _plot_bias_frequency(plot_item, traces, amplitude_to_color,
                                     pen_color, bias, labels)
                plot_item.setLabel('left', 'IQ arc speed', units='Counts/Hz')
                plot_item.setLabel('bottom', 'Frequency Offset', units='kHz')
            elif plot_type == 'fit':
                _plot_fit(plot_item, traces, amplitude_to_color, pen_color,
                          fit_model, labels)
                plot_item.setLabel('left', 'Normalized Magnitude')
                plot_item.setLabel('bottom', 'Frequency Offset', units='kHz')
            else:  # IQ
                _plot_iq(plot_item, traces, amplitude_to_color,
                         pen_color, unit_mode, normalize, labels, bias)
                iq_units = 'Counts' if unit_mode == 'counts' else 'V'
                plot_item.setLabel('left', 'Q (Imaginary)', units=iq_units)
                plot_item.setLabel('bottom', 'I (Real)', units=iq_units)
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


def _bias_frequency_line(plot_item, bias, sweep, amplitude_to_color, pen_color):
    """A vertical line where the tone goes, in the chosen drive's own colour.

    The line and the thickened trace are the whole of what a bias report puts
    on these plots. Everything it has to say in words -- how many were biased,
    which were flagged and why -- is on the status line, so a measurement plot
    stays a measurement plot.
    """
    offset = (bias.frequency_hz - sweep['original_center_frequency']) / 1e3
    color = amplitude_to_color.get(bias.amplitude, pen_color)
    plot_item.addLine(x=offset, pen=pg.mkPen(color=color, width=LINE_WIDTH))


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


def _plot_magnitude(plot_item, traces, amplitude_to_color, pen_color,
                    unit_mode='dbm', normalize=False, legend_labels=None,
                    bias=None):
    """Plot |S21| against frequency offset for one resonator.

    Args:
        plot_item: PyQtGraph PlotItem
        traces: ``[(step, direction, amplitude, sweep), ...]``
        amplitude_to_color: Dict {amplitude: color}
        pen_color: Fallback pen color
        unit_mode: 'counts', 'dbm', or 'volts'
        normalize: Whether to normalize traces
        legend_labels: Optional {(step, direction, amplitude): label}
        bias: Optional BiasFinding; its step is drawn thick and its frequency
            gets a line
    """
    if legend_labels:
        _add_legend(plot_item, pen_color)

    drawn = None
    for step, direction, amplitude, sweep in traces:
        counts = sweep['iq_counts']
        if len(counts) == 0:
            continue
        magnitude = UnitConverter.convert_amplitude(
            np.abs(counts), counts, unit_mode, normalize=normalize)
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

#: How wide the model is drawn. Thinner than the measurement it lies on, so
#: that where the two agree the coloured line is still visible under it.
MODEL_LINE_WIDTH = 1

#: How faint the line at a fit's ``fr`` is. It marks where the model put the
#: resonance; it is not a thing that was measured, so it sits behind the two
#: lines that were.
FR_LINE_ALPHA = 120


def _si(value: float) -> str:
    """A Q, short enough for a legend: ``29.6k``, ``1.24M``."""
    if abs(value) >= 1e6:
        return f"{value / 1e6:.3g}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.3g}k"
    return f"{value:.3g}"


#: What a model's legend entry says it fitted, in the order it says it. The
#: headline numbers only: everything a fit learned is in the entry, and a
#: legend that listed it all would cover the plot it labels.
FIT_LEGEND_PARAMS = (
    ("fr", lambda value: f"fr {value / 1e6:.4f} MHz"),
    ("Qr", lambda value: f"Qr {_si(value)}"),
    ("Qi", lambda value: f"Qi {_si(value)}"),
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
    """What the derivative test looks at, in units of the bar it applied.

    The quantity is the point-to-point change in the normalized arc speed, and
    a spike in it above the bar -- with a spike the other way beside it -- is
    what the test calls a bifurcation. Every trace is divided by its own
    threshold, so a step driven a thousandth as hard is still visible beside
    the loud one and the bar is the same line for all of them: ``±1``.

    For the step a resonator is biased at, the bar that did *not* bind is drawn
    too, faintly. Below ±1 it is the noise gate that decided; at ±1 the two
    coincide. That is the question the detector otherwise answers by being run
    again with one of the two switched off.
    """
    if legend_labels:
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

        if chosen:
            unbinding.append(min(prominence_bar, noise_bar) / bar)

    if unbinding:
        colour = pg.mkColor(pen_color)
        colour.setAlpha(UNBINDING_BAR_ALPHA)
        faint = pg.mkPen(color=colour, width=1, style=DOWNWARD_SWEEP_STYLE)
        # One band, at the lowest of them: below that line the noise gate is
        # what decides, whichever direction the sweep was taken in.
        _bar_band(plot_item, min(unbinding), pen_color, UNBINDING_FILL_ALPHA)
        for other in unbinding:
            for sign in (1.0, -1.0):
                plot_item.addLine(y=sign * other, pen=faint)


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


def _plot_bias_frequency(plot_item, traces, amplitude_to_color, pen_color,
                         bias, legend_labels=None):
    """What choosing the bias frequency looked at, at the drive it was chosen at.

    :func:`~rfmux.tuning.bias.iq_arc_speed` is the quantity the default
    ``"iq_derivative"`` method maximizes -- how far the IQ trace moves per hertz
    -- so its peak is the answer and the line is where the tone will actually
    go, after ``BiasPoint`` puts it on the hardware grid. The gap between the
    two is that quantization, which is the reading this tab exists for.

    Only the step the resonator is biased at is drawn. The other steps chose
    nothing, and a grid of them would bury the one that did.
    """
    if legend_labels:
        _add_legend(plot_item, pen_color)

    for step, direction, amplitude, sweep in traces:
        try:
            frequencies, speed = iq_arc_speed(sweep)
        except (ValueError, KeyError):
            continue        # too short or too degenerate to differentiate
        plot_item.plot(
            (frequencies - sweep['original_center_frequency']) / 1e3, speed,
            pen=_trace_pen(amplitude, direction, amplitude_to_color, pen_color,
                           chosen=True),
            name=legend_labels.get((step, direction, amplitude)) if legend_labels else None)

    if bias is not None and traces:
        _bias_frequency_line(plot_item, bias, traces[0][3], amplitude_to_color,
                             pen_color)


def _plot_iq(plot_item, traces, amplitude_to_color, pen_color,
             unit_mode='dbm', normalize=False, legend_labels=None, bias=None):
    """Plot the IQ loops of one resonator.

    Args:
        plot_item: PyQtGraph PlotItem
        traces: ``[(step, direction, amplitude, sweep), ...]``
        amplitude_to_color: Dict {amplitude: color}
        pen_color: Fallback pen color
        unit_mode: 'counts' draws raw IQ, anything else the entry's volts
        normalize: Whether to normalize IQ by max magnitude
        legend_labels: Optional {(step, direction, amplitude): label}
        bias: Optional BiasFinding; its step is drawn thick and the point the
            tone sits at is marked on it

    A loop's axis is I, not frequency, so the bias frequency cannot be a
    vertical line here as it is on the magnitude plot. It is the point of the
    loop the tone will sit on, which is what the marker is.
    """
    if legend_labels:
        _add_legend(plot_item, pen_color)

    for step, direction, amplitude, sweep in traces:
        iq = sweep_iq(sweep, unit_mode)
        if len(iq) == 0:
            continue

        i_vals, q_vals = np.real(iq), np.imag(iq)
        if normalize:
            peak = np.max(np.abs(iq))
            if peak > 0:
                i_vals, q_vals = i_vals / peak, q_vals / peak

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
