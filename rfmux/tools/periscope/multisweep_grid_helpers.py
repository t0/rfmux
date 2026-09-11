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
                      dac_scale=None, show_legend=True, fit_model='skewed'):
    """
    Update a grid layout with one subplot per resonator.

    Args:
        grid_layout: QGridLayout to populate with plots
        traces_by_name: ``{name: [(step, direction, amplitude, sweep), ...]}``,
            in the order to draw them; *sweep* is one of multisweep's entries
        plot_type: 'magnitude' or 'iq'
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

            if plot_type == 'magnitude':
                _plot_magnitude(plot_item, traces, amplitude_to_color,
                                pen_color, unit_mode, normalize, labels)
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
            elif plot_type == 'fit':
                _plot_fit(plot_item, traces, amplitude_to_color, pen_color,
                          fit_model, labels)
                plot_item.setLabel('left', 'Normalized Magnitude')
                plot_item.setLabel('bottom', 'Frequency Offset', units='kHz')
            else:  # IQ
                _plot_iq(plot_item, traces, amplitude_to_color,
                         pen_color, unit_mode, normalize, labels)
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


def _trace_pen(amplitude, direction, amplitude_to_color, pen_color):
    """Colour says drive amplitude, line style says direction.

    Every trace is coloured by its drive, including the only one of a
    single-amplitude sweep, so a colour does not change meaning as the later
    steps of a schedule arrive.
    """
    color = amplitude_to_color.get(amplitude, pen_color)
    style = DOWNWARD_SWEEP_STYLE if direction == "downward" else UPWARD_SWEEP_STYLE
    return pg.mkPen(color=color, width=LINE_WIDTH, style=style)


def _plot_magnitude(plot_item, traces, amplitude_to_color, pen_color,
                    unit_mode='dbm', normalize=False, legend_labels=None):
    """Plot |S21| against frequency offset for one resonator.

    Args:
        plot_item: PyQtGraph PlotItem
        traces: ``[(step, direction, amplitude, sweep), ...]``
        amplitude_to_color: Dict {amplitude: color}
        pen_color: Fallback pen color
        unit_mode: 'counts', 'dbm', or 'volts'
        normalize: Whether to normalize traces
        legend_labels: Optional {(step, direction, amplitude): label}
    """
    if legend_labels:
        _add_legend(plot_item, pen_color)

    for step, direction, amplitude, sweep in traces:
        counts = sweep['iq_counts']
        if len(counts) == 0:
            continue
        magnitude = UnitConverter.convert_amplitude(
            np.abs(counts), counts, unit_mode, normalize=normalize)
        pen = _trace_pen(amplitude, direction, amplitude_to_color, pen_color)
        name = legend_labels.get((step, direction, amplitude)) if legend_labels else None
        plot_item.plot(offset_khz(sweep), magnitude, pen=pen, name=name)


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


def _plot_fit(plot_item, traces, amplitude_to_color, pen_color, fit_model,
              legend_labels=None):
    """One resonator's measured magnitude with one model fitted to it.

    Normalized to each trace's last point, because that is the skewed fit's own
    convention: the model comes back in those units, so the measurement is put
    into them rather than the model taken out of them.

    The measurement keeps the colour it has on the other tabs -- its drive --
    and the fit is drawn in the foreground colour, so that at a glance the
    black or white line is the model and the coloured points are the data. One
    model at a time, which leaves line style free to mean direction here as it
    does everywhere else.

    A sweep with no fit of this model draws its measurement alone, and one that
    did not converge is simply absent: the count of what failed is on the
    toolbar.
    """
    if legend_labels:
        _add_legend(plot_item, pen_color)

    reader, scale_of = FIT_READERS[fit_model]
    for step, direction, amplitude, sweep in traces:
        counts = np.asarray(sweep['iq_counts'])
        if len(counts) == 0 or counts[-1] == 0:
            continue
        color = amplitude_to_color.get(amplitude, pen_color)
        name = legend_labels.get((step, direction, amplitude)) if legend_labels else None
        plot_item.plot(
            offset_khz(sweep), np.abs(counts / counts[-1]),
            pen=None, symbol='x' if direction == 'downward' else 'o',
            symbolSize=4, symbolPen=color, symbolBrush=color, name=name)

        try:
            offsets, model = _model_on_a_finer_grid(reader, sweep)
        except (ValueError, KeyError):
            continue    # no fit of this model, or one that did not converge
        style = DOWNWARD_SWEEP_STYLE if direction == 'downward' else UPWARD_SWEEP_STYLE
        plot_item.plot(
            offsets, np.abs(model) * scale_of(counts),
            pen=pg.mkPen(color=pen_color, width=LINE_WIDTH, style=style))


def _plot_iq(plot_item, traces, amplitude_to_color, pen_color,
             unit_mode='dbm', normalize=False, legend_labels=None):
    """Plot the IQ loops of one resonator.

    Args:
        plot_item: PyQtGraph PlotItem
        traces: ``[(step, direction, amplitude, sweep), ...]``
        amplitude_to_color: Dict {amplitude: color}
        pen_color: Fallback pen color
        unit_mode: 'counts' draws raw IQ, anything else the entry's volts
        normalize: Whether to normalize IQ by max magnitude
        legend_labels: Optional {(step, direction, amplitude): label}
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

        pen = _trace_pen(amplitude, direction, amplitude_to_color, pen_color)
        name = legend_labels.get((step, direction, amplitude)) if legend_labels else None
        plot_item.plot(i_vals, q_vals, pen=pen, name=name)


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
                normalized_idx = amp_idx / max(1, num_amps - 1)
                if dark_mode:
                    map_value = 0.3 + normalized_idx * 0.7
                else:
                    map_value = normalized_idx * 0.75
                color = use_cmap.map(map_value)
            else:
                color = TABLEAU10_COLORS[amp_idx % len(TABLEAU10_COLORS)]

        amplitude_to_color[amp_val] = color

    return amplitude_to_color
