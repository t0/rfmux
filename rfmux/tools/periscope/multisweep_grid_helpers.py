"""
Helper functions for rendering per-detector grid plots in multisweep panels.

This module provides reusable plotting functions used by MultisweepPanel
to create grids of per-detector sweep plots.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore

from .utils import (
    LINE_WIDTH, TABLEAU10_COLORS, COLORMAP_CHOICES, AMPLITUDE_COLORMAP_THRESHOLD,
    UPWARD_SWEEP_STYLE, DOWNWARD_SWEEP_STYLE,
    square_axes, UnitConverter, mag_axis_label,
)
from rfmux.core.transferfunctions import convert_roc_to_volts


def update_sweep_grid(grid_layout, data_by_detector, plot_type, current_batch, batch_size,
                      amplitude_to_color, dark_mode, unit_mode='dbm', normalize=False,
                      prev_btn=None, next_btn=None, batch_label=None, widget_cache=None,
                      dac_scale=None, show_legend=True):
    """
    Update a grid layout with per-detector sweep plots.

    Args:
        grid_layout: QGridLayout to populate with plots
        data_by_detector: Dict {detector_id: {(amp, direction): {freq, iq, ...}}}
        plot_type: 'magnitude' or 'iq'
        current_batch: Current batch index (0-based)
        batch_size: Number of detectors per batch
        amplitude_to_color: Dict mapping amplitude values to colors
        dark_mode: Boolean for theme
        unit_mode: Unit mode for magnitude display ('counts', 'dbm', 'volts')
        normalize: Whether to normalize traces
        prev_btn: Optional previous batch button to enable/disable
        next_btn: Optional next batch button to enable/disable
        batch_label: Optional label to update with batch info
        widget_cache: Optional list to cache plot widgets for reuse
        dac_scale: Optional DAC scale (dBm) for formatting legend labels
    """
    if not data_by_detector:
        return

    # Remove all items from grid without deleting widgets (we'll reuse them)
    while grid_layout.count():
        grid_layout.takeAt(0)

    # Get sorted detector IDs
    detector_ids = sorted(data_by_detector.keys())

    # Calculate batch range
    start_idx = current_batch * batch_size
    end_idx = min(start_idx + batch_size, len(detector_ids))
    batch_detectors = detector_ids[start_idx:end_idx]

    if not batch_detectors:
        return

    # Calculate grid dimensions — use ceil(sqrt(n)) for a balanced grid
    num_plots = len(batch_detectors)
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

    # Collect the unique (amp, direction) sweep keys across all detectors in this batch
    all_sweep_keys = set()
    for det_id in batch_detectors:
        all_sweep_keys.update(data_by_detector.get(det_id, {}).keys())

    # Build sweep labels once (used for legends on every subplot)
    sweep_labels = {}
    for (amp_val, direction) in sorted(all_sweep_keys):
        label = UnitConverter.format_probe_label(amp_val, unit_mode, dac_scale)
        dir_suffix = " (Down)" if direction == "downward" else " (Up)"
        sweep_labels[(amp_val, direction)] = label + dir_suffix

    # Populate grid
    for idx, detector_id in enumerate(batch_detectors):
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

            # Get detector data
            detector_data = data_by_detector.get(detector_id, {})

            # Extract center frequency from first available entry
            center_freq_hz = None
            if detector_data:
                first_entry = next(iter(detector_data.values()), {})
                center_freq_hz = first_entry.get('original_center_frequency')

            # Title
            if center_freq_hz is not None:
                title = f"{detector_id}: {center_freq_hz / 1e6:.3f} MHz"
            else:
                title = f"Detector {detector_id}"
            plot_item.setTitle(title, color=pen_color)

            # Style axes
            for axis_name in ("left", "bottom", "right", "top"):
                ax = plot_item.getAxis(axis_name)
                if ax:
                    ax.setPen(pen_color)
                    ax.setTextPen(pen_color)

            # Plot data with legend labels (suppressed when colorbar is active)
            labels = sweep_labels if (show_legend and len(sweep_labels) > 0) else None

            if plot_type == 'magnitude':
                _plot_detector_magnitude(plot_item, detector_data, amplitude_to_color,
                                         pen_color, unit_mode, normalize, labels, dac_scale)
                lbl, units = mag_axis_label(unit_mode, normalize)
                plot_item.setLabel('left', lbl, units=units)
                plot_item.setLabel('bottom', 'Frequency Offset', units='kHz')
            else:  # IQ
                _plot_detector_iq(plot_item, detector_data, amplitude_to_color,
                                  pen_color, normalize, labels, unit_mode=unit_mode)
                iq_units = 'V' if unit_mode == 'volts' else 'Counts'
                plot_item.setLabel('left', 'Q (Imaginary)', units=iq_units)
                plot_item.setLabel('bottom', 'I (Real)', units=iq_units)
                square_axes(plot_item)

            plot_item.showGrid(x=True, y=True, alpha=0.3)

        # Store detector ID on widget for double-click navigation
        plot_widget._detector_id = detector_id
        grid_layout.addWidget(plot_widget, row, col)
        plot_widget.show()

    # Update batch navigation
    total_batches = max(1, (len(detector_ids) + batch_size - 1) // batch_size)
    if prev_btn:
        prev_btn.setEnabled(current_batch > 0)
    if next_btn:
        next_btn.setEnabled(current_batch < total_batches - 1)
    if batch_label:
        batch_label.setText(f"{current_batch + 1} of {total_batches}")


# ---------------------------------------------------------------------------
# Per-detector plotting helpers
# ---------------------------------------------------------------------------

def _plot_detector_magnitude(plot_item, detector_data, amplitude_to_color,
                             pen_color, unit_mode='dbm', normalize=False,
                             sweep_labels=None, dac_scale=None):
    """Plot S21 magnitude sweeps for a single detector.

    Args:
        plot_item: PyQtGraph PlotItem
        detector_data: Dict {(amp, direction): {freq, iq, ...}}
        amplitude_to_color: Dict {amp_value: color}
        pen_color: Fallback pen color
        unit_mode: 'counts', 'dbm', or 'volts'
        normalize: Whether to normalize traces
        sweep_labels: Optional dict {(amp, direction): label} for legend names.
        dac_scale: DAC full-scale in dBm (for computing probe_amp_dbm when normalizing)
    """
    sorted_keys = sorted(detector_data.keys())
    single_sweep = len(sorted_keys) == 1

    # Add legend in lower-left corner (usually free space)
    if sweep_labels:
        legend_color = '#CCCCCC' if amplitude_to_color else '#333333'
        # Infer dark mode from pen_color
        if pen_color == 'w' or pen_color == (255, 255, 255):
            legend_color = '#CCCCCC'
        else:
            legend_color = '#333333'
        plot_item.addLegend(offset=(10, -10), labelTextColor=legend_color)

    for (amp_val, direction) in sorted_keys:
        entry = detector_data[(amp_val, direction)]
        freqs = entry.get('freq')
        iq = entry.get('iq')
        if freqs is None or iq is None or len(freqs) == 0:
            continue

        mag = np.abs(iq)
        # Compute probe_amp_dbm for dBm normalization if dac_scale available
        probe_amp_dbm = None
        if normalize and unit_mode == 'dbm' and dac_scale is not None:
            probe_amp_dbm = UnitConverter.normalize_to_dbm(amp_val, dac_scale)
        mag_converted = UnitConverter.convert_amplitude(mag, iq, unit_mode, normalize=normalize, probe_amp_dbm=probe_amp_dbm)

        # Color from amplitude, line style from direction
        if single_sweep:
            pen = pg.mkPen(color=pen_color, width=LINE_WIDTH)
        else:
            color = amplitude_to_color.get(amp_val, pen_color)
            line_style = DOWNWARD_SWEEP_STYLE if direction == "downward" else UPWARD_SWEEP_STYLE
            pen = pg.mkPen(color=color, width=LINE_WIDTH, style=line_style)

        freqs_rel_khz = 1e-3 * (freqs - np.mean(freqs))
        name = sweep_labels.get((amp_val, direction)) if sweep_labels else None
        plot_item.plot(freqs_rel_khz, mag_converted, pen=pen, name=name)


def _plot_detector_iq(plot_item, detector_data, amplitude_to_color,
                      pen_color, normalize=False, sweep_labels=None, unit_mode='counts'):
    """Plot IQ circles for a single detector.

    Args:
        plot_item: PyQtGraph PlotItem
        detector_data: Dict {(amp, direction): {freq, iq, ...}}
        amplitude_to_color: Dict {amp_value: color}
        pen_color: Fallback pen color
        normalize: Whether to normalize IQ by max magnitude
        sweep_labels: Optional dict {(amp, direction): label} for legend names.
        unit_mode: ``'counts'``, ``'volts'``, or ``'dbm'``.  When ``'volts'``
            the I and Q arrays are converted via ``convert_roc_to_volts``
            before plotting.  For ``'counts'`` and ``'dbm'`` the raw ROC
            counts are used unchanged (dBm has no meaningful per-component
            interpretation for complex IQ data).
    """
    sorted_keys = sorted(detector_data.keys())
    single_sweep = len(sorted_keys) == 1

    if sweep_labels:
        if pen_color == 'w' or pen_color == (255, 255, 255):
            legend_color = '#CCCCCC'
        else:
            legend_color = '#333333'
        plot_item.addLegend(offset=(10, -10), labelTextColor=legend_color)

    for (amp_val, direction) in sorted_keys:
        entry = detector_data[(amp_val, direction)]
        iq = entry.get('iq')
        if iq is None or len(iq) == 0:
            continue

        i_vals = np.real(iq)
        q_vals = np.imag(iq)

        # Convert to volts when requested.  dBm is skipped — it has no
        # meaningful per-component interpretation for complex IQ data.
        if unit_mode == 'volts':
            i_vals = convert_roc_to_volts(i_vals)
            q_vals = convert_roc_to_volts(q_vals)

        if normalize:
            mag = np.abs(iq) if unit_mode != 'volts' else np.sqrt(i_vals**2 + q_vals**2)
            if len(mag) > 0 and np.max(mag) > 0:
                i_vals = i_vals / np.max(mag)
                q_vals = q_vals / np.max(mag)

        if single_sweep:
            pen = pg.mkPen(color=pen_color, width=LINE_WIDTH)
        else:
            color = amplitude_to_color.get(amp_val, pen_color)
            line_style = DOWNWARD_SWEEP_STYLE if direction == "downward" else UPWARD_SWEEP_STYLE
            pen = pg.mkPen(color=color, width=LINE_WIDTH, style=line_style)

        name = sweep_labels.get((amp_val, direction)) if sweep_labels else None
        plot_item.plot(i_vals, q_vals, pen=pen, name=name)


# ---------------------------------------------------------------------------
# Public helpers (kept for backward compatibility & use by other modules)
# ---------------------------------------------------------------------------

plot_detector_magnitude = _plot_detector_magnitude
plot_detector_iq = _plot_detector_iq


def create_amplitude_color_map(amplitude_values, dark_mode):
    """
    Create a color mapping for amplitude values.

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
