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
                      dac_scale=None, show_legend=True, res_info_dict=None):
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
        res_info_dict: Optional resonator registry {code: {...}} from MultisweepPanel.
            When provided, bias overlays (thickened chosen-amplitude trace,
            vertical line / IQ marker at bias frequency) are drawn for each
            detector whose ``bias_found`` flag is True.
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
                title = f"{detector_id} (<i>f</i><sub>central</sub> = {center_freq_hz / 1e6:.3f} MHz)"
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

            # Look up bias overlay info for this detector from the resonator registry
            det_bias_info = (res_info_dict or {}).get(detector_id, {})
            bias_amp = (
                det_bias_info.get('bias_amplitude')
                if det_bias_info.get('bias_found') else None
            )
            bias_freq_hz = (
                det_bias_info.get('bias_frequency')
                if det_bias_info.get('bias_found') else None
            )

            if plot_type == 'magnitude':
                _plot_detector_magnitude(plot_item, detector_data, amplitude_to_color,
                                         pen_color, unit_mode, normalize, labels, dac_scale,
                                         bias_amplitude=bias_amp,
                                         bias_frequency_hz=bias_freq_hz)
                lbl, units = mag_axis_label(unit_mode, normalize)
                plot_item.setLabel('left', lbl, units=units)
                plot_item.setLabel('bottom', '<i>f</i> − <i>f</i><sub>central</sub> [kHz]')
            else:  # IQ
                _plot_detector_iq(plot_item, detector_data, amplitude_to_color,
                                  pen_color, normalize, labels, unit_mode=unit_mode,
                                  bias_amplitude=bias_amp,
                                  bias_frequency_hz=bias_freq_hz,
                                  dac_scale=dac_scale)
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
                             sweep_labels=None, dac_scale=None,
                             bias_amplitude=None, bias_frequency_hz=None):
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
        bias_amplitude: Chosen bias amplitude from Find Bias (or None).  The
            trace for this amplitude is rendered with double line width.
        bias_frequency_hz: Refined bias frequency in Hz from Find Bias (or
            None).  A solid vertical line is drawn at this frequency when
            provided.
    """
    sorted_keys = sorted(detector_data.keys())
    single_sweep = len(sorted_keys) == 1

    # When colorbar mode is active (sweep_labels is None) but Find Bias has run
    # and the nonlinear fit succeeded for the chosen amplitude, prepare a one-entry
    # legend label so the 'a' value is still visible on each subplot.
    chosen_a_label = None
    if not sweep_labels and bias_amplitude is not None:
        for (av, dir_) in sorted_keys:
            if av == bias_amplitude:
                _entry = detector_data[(av, dir_)]
                if _entry.get('nonlinear_fit_success') and _entry.get('nonlinear_fit_params'):
                    _a_val = _entry['nonlinear_fit_params'].get('a')
                    if _a_val is not None:
                        _amp_str = UnitConverter.format_probe_label(bias_amplitude, unit_mode, dac_scale)
                        chosen_a_label = f"★ {_amp_str}, a={_a_val:.3f}"
                        break

    # Add legend: always when sweep_labels are provided (legend mode), or when
    # colorbar mode is active but we have a chosen-amplitude 'a' value to show.
    if sweep_labels or chosen_a_label:
        # Infer dark mode from pen_color
        if pen_color == 'w' or pen_color == (255, 255, 255):
            legend_color = '#CCCCCC'
        else:
            legend_color = '#333333'
        plot_item.addLegend(offset=(10, -10), labelTextColor=legend_color)

    # Track the frequency array and color for the chosen amplitude so we can
    # use them after the loop when drawing the bias vertical line.
    chosen_freqs_for_vline = None
    chosen_color_for_vline = pen_color

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

        # Determine whether this sweep is the chosen bias amplitude
        is_chosen = (bias_amplitude is not None) and (amp_val == bias_amplitude)
        curve_width = LINE_WIDTH * 2 if is_chosen else LINE_WIDTH

        # Color from amplitude_to_color when an entry exists (even when only one
        # sweep has arrived so far), falling back to pen_color only when the
        # amplitude truly has no colour mapping.
        if amp_val in amplitude_to_color:
            color = amplitude_to_color[amp_val]
            line_style = DOWNWARD_SWEEP_STYLE if direction == "downward" else UPWARD_SWEEP_STYLE
            pen = pg.mkPen(color=color, width=curve_width, style=line_style)
            if is_chosen:
                chosen_color_for_vline = color
        else:
            # Genuine fallback: amplitude not in the colour map at all
            pen = pg.mkPen(color=pen_color, width=curve_width)

        freqs_rel_khz = 1e-3 * (freqs - np.mean(freqs))

        # Append ★ and nonlinearity parameter 'a' to the legend name for the chosen amplitude.
        # In colorbar mode (sweep_labels is None), use the pre-computed chosen_a_label for the
        # chosen sweep so the 'a' value still appears in the mini one-entry legend.
        name = None
        if sweep_labels:
            base_name = sweep_labels.get((amp_val, direction))
            if is_chosen and base_name:
                nl_suffix = ""
                if entry.get('nonlinear_fit_success') and entry.get('nonlinear_fit_params'):
                    a_val = entry['nonlinear_fit_params'].get('a')
                    if a_val is not None:
                        nl_suffix = f", a={a_val:.3f}"
                name = base_name + " ★" + nl_suffix
            else:
                name = base_name
        elif is_chosen and chosen_a_label:
            name = chosen_a_label

        plot_item.plot(freqs_rel_khz, mag_converted, pen=pen, name=name)

        # Record this sweep's frequency axis reference for the vertical line
        if is_chosen:
            chosen_freqs_for_vline = freqs

    # --- Bias frequency vertical line ---
    # Draw a thin solid vertical line at the refined bias frequency using the
    # same colour as the chosen-amplitude traces.  The position is expressed
    # in the same kHz-offset coordinate as the traces.
    if bias_frequency_hz is not None and chosen_freqs_for_vline is not None:
        freq_mean_hz = float(np.mean(chosen_freqs_for_vline))
        bias_offset_khz = 1e-3 * (bias_frequency_hz - freq_mean_hz)
        vline_pen = pg.mkPen(
            chosen_color_for_vline,
            style=QtCore.Qt.PenStyle.SolidLine,
            width=1,
        )
        plot_item.addItem(pg.InfiniteLine(
            pos=bias_offset_khz, angle=90,
            pen=vline_pen, movable=False,
        ))


def _plot_detector_iq(plot_item, detector_data, amplitude_to_color,
                      pen_color, normalize=False, sweep_labels=None, unit_mode='counts',
                      bias_amplitude=None, bias_frequency_hz=None, dac_scale=None):
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
        bias_amplitude: Chosen bias amplitude from Find Bias (or None).  The
            trace for this amplitude is rendered with double line width.
        bias_frequency_hz: Refined bias frequency in Hz from Find Bias (or
            None).  A ✕ scatter marker is drawn at the IQ point closest to
            this frequency on the chosen-amplitude trace.
        dac_scale: DAC full-scale in dBm (for formatting the amplitude in the
            colorbar-mode mini legend label).
    """
    sorted_keys = sorted(detector_data.keys())
    single_sweep = len(sorted_keys) == 1

    # When colorbar mode is active (sweep_labels is None) but Find Bias has run
    # and the nonlinear fit succeeded for the chosen amplitude, prepare a one-entry
    # legend label so the 'a' value is still visible on each subplot.
    chosen_a_label = None
    if not sweep_labels and bias_amplitude is not None:
        for (av, dir_) in sorted_keys:
            if av == bias_amplitude:
                _entry = detector_data[(av, dir_)]
                if _entry.get('nonlinear_fit_success') and _entry.get('nonlinear_fit_params'):
                    _a_val = _entry['nonlinear_fit_params'].get('a')
                    if _a_val is not None:
                        _amp_str = UnitConverter.format_probe_label(bias_amplitude, unit_mode, dac_scale)
                        chosen_a_label = f"★ {_amp_str}, a={_a_val:.3f}"
                        break

    # Add legend: always when sweep_labels are provided (legend mode), or when
    # colorbar mode is active but we have a chosen-amplitude 'a' value to show.
    if sweep_labels or chosen_a_label:
        if pen_color == 'w' or pen_color == (255, 255, 255):
            legend_color = '#CCCCCC'
        else:
            legend_color = '#333333'
        plot_item.addLegend(offset=(10, -10), labelTextColor=legend_color)

    # Accumulate the IQ data for the chosen amplitude sweep so that we can
    # add the bias-point scatter marker after all traces are drawn.
    chosen_i_for_marker = None
    chosen_q_for_marker = None
    chosen_freqs_for_marker = None
    chosen_color_for_marker = pen_color

    for (amp_val, direction) in sorted_keys:
        entry = detector_data[(amp_val, direction)]
        freqs = entry.get('freq')
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

        # Determine whether this sweep is the chosen bias amplitude
        is_chosen = (bias_amplitude is not None) and (amp_val == bias_amplitude)
        curve_width = LINE_WIDTH * 2 if is_chosen else LINE_WIDTH

        # Color from amplitude_to_color when an entry exists (even when only one
        # sweep has arrived so far), falling back to pen_color only when the
        # amplitude truly has no colour mapping.
        if amp_val in amplitude_to_color:
            color = amplitude_to_color[amp_val]
            line_style = DOWNWARD_SWEEP_STYLE if direction == "downward" else UPWARD_SWEEP_STYLE
            pen = pg.mkPen(color=color, width=curve_width, style=line_style)
            if is_chosen:
                chosen_color_for_marker = color
        else:
            # Genuine fallback: amplitude not in the colour map at all
            pen = pg.mkPen(color=pen_color, width=curve_width)

        # Append ★ and nonlinearity parameter 'a' to the legend name for the chosen amplitude.
        # In colorbar mode (sweep_labels is None), use the pre-computed chosen_a_label for the
        # chosen sweep so the 'a' value still appears in the mini one-entry legend.
        name = None
        if sweep_labels:
            base_name = sweep_labels.get((amp_val, direction))
            if is_chosen and base_name:
                nl_suffix = ""
                if entry.get('nonlinear_fit_success') and entry.get('nonlinear_fit_params'):
                    a_val = entry['nonlinear_fit_params'].get('a')
                    if a_val is not None:
                        nl_suffix = f", a={a_val:.3f}"
                name = base_name + " ★" + nl_suffix
            else:
                name = base_name
        elif is_chosen and chosen_a_label:
            name = chosen_a_label

        plot_item.plot(i_vals, q_vals, pen=pen, name=name)

        # Store the converted IQ values for the chosen trace so we can place
        # the bias-point marker after the loop.
        if is_chosen:
            chosen_i_for_marker = i_vals
            chosen_q_for_marker = q_vals
            chosen_freqs_for_marker = freqs

    # --- Bias-point IQ marker ---
    # Find the sweep sample whose frequency is closest to bias_frequency_hz
    # and draw a ✕ scatter symbol there.
    if (bias_frequency_hz is not None
            and chosen_freqs_for_marker is not None
            and chosen_i_for_marker is not None):
        freq_arr = np.asarray(chosen_freqs_for_marker)
        idx_closest = int(np.argmin(np.abs(freq_arr - bias_frequency_hz)))
        i_bias = float(chosen_i_for_marker[idx_closest])
        q_bias = float(chosen_q_for_marker[idx_closest])
        scatter = pg.ScatterPlotItem(
            x=[i_bias], y=[q_bias],
            symbol='x',
            size=18,
            pen=pg.mkPen(chosen_color_for_marker, width=2),
            brush=pg.mkBrush(chosen_color_for_marker),
        )
        plot_item.addItem(scatter)


# ---------------------------------------------------------------------------
# IQ-derivative grid (shown after Find Bias)
# ---------------------------------------------------------------------------

def update_derivative_grid(
    grid_layout,
    bias_finding_by_detector,
    current_batch,
    batch_size,
    dark_mode,
    prev_btn=None,
    next_btn=None,
    batch_label=None,
    widget_cache=None,
    unit_mode='dbm',
    dac_scale=None,
):
    """
    Update a grid layout with per-detector IQ-derivative plots.

    Only detectors that have a ``bias_finding`` key in at least one of their
    sweep entries are included.  These are the resonators for which Find Bias
    has been run and the derivative/spline data saved.

    Parameters
    ----------
    grid_layout : QGridLayout
        Layout to populate.
    bias_finding_by_detector : dict
        ``{detector_id: bias_finding_dict}`` — pre-extracted by the panel's
        ``_redraw_derivative_grid`` method.
    current_batch : int
        Zero-based batch index.
    batch_size : int
        Number of detectors per batch.
    dark_mode : bool
        Theme flag.
    prev_btn, next_btn : QPushButton or None
        Batch navigation buttons to enable/disable.
    batch_label : QLabel or None
        Label to update with "N of M" text.
    widget_cache : list or None
        Reusable ``pg.PlotWidget`` instances.
    unit_mode : str
        Unit mode for amplitude display (``'counts'``, ``'dbm'``, or
        ``'volts'``).  Passed to :func:`UnitConverter.format_probe_label`
        when formatting the bias amplitude in each subplot title.
    dac_scale : float or None
        DAC full-scale in dBm used by :func:`UnitConverter.format_probe_label`
        to convert raw counts to physical units.
    """
    if not bias_finding_by_detector:
        return

    # Remove all items from grid without destroying widgets
    while grid_layout.count():
        grid_layout.takeAt(0)

    detector_ids = sorted(bias_finding_by_detector.keys())

    start_idx = current_batch * batch_size
    end_idx = min(start_idx + batch_size, len(detector_ids))
    batch_detectors = detector_ids[start_idx:end_idx]

    if not batch_detectors:
        return

    num_plots = len(batch_detectors)
    ncols = max(1, int(np.ceil(np.sqrt(num_plots))))
    nrows = int(np.ceil(num_plots / ncols))

    bg_color, pen_color = ("k", "w") if dark_mode else ("w", "k")

    # Reset stale stretch factors
    for r in range(grid_layout.rowCount()):
        grid_layout.setRowStretch(r, 0)
    for c in range(grid_layout.columnCount()):
        grid_layout.setColumnStretch(c, 0)
    for r in range(nrows):
        grid_layout.setRowStretch(r, 1)
    for c in range(ncols):
        grid_layout.setColumnStretch(c, 1)

    if widget_cache is None:
        widget_cache = []

    while len(widget_cache) < num_plots:
        pw = pg.PlotWidget()
        pw.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        widget_cache.append(pw)

    for widget in widget_cache:
        widget.hide()

    for idx, detector_id in enumerate(batch_detectors):
        row = idx // ncols
        col = idx % ncols

        plot_widget = widget_cache[idx]
        plot_widget.setBackground(bg_color)
        plot_item = plot_widget.getPlotItem()

        if plot_item:
            plot_item.clear()
            if hasattr(plot_item, 'legend') and plot_item.legend:
                plot_item.legend.scene().removeItem(plot_item.legend)
                plot_item.legend = None

            bf = bias_finding_by_detector[detector_id]

            # Title
            bias_amp = bf.get('bias_amplitude')
            bias_freq = bf.get('bias_frequency')
            if bias_freq is not None:
                amp_str = (
                    UnitConverter.format_probe_label(bias_amp, unit_mode, dac_scale)
                    if bias_amp is not None else None
                )
                title = (
                    f"{detector_id}  "
                    f"(<i>f</i><sub>bias</sub> = {bias_freq * 1e-6:.3f} MHz"
                    + (f", amp = {amp_str}" if amp_str is not None else "")
                    + ")"
                )
            else:
                title = f"Detector {detector_id}"
            plot_item.setTitle(title, color=pen_color)

            for axis_name in ("left", "bottom", "right", "top"):
                ax = plot_item.getAxis(axis_name)
                if ax:
                    ax.setPen(pen_color)
                    ax.setTextPen(pen_color)

            if pen_color == 'w' or pen_color == (255, 255, 255):
                legend_color = '#CCCCCC'
            else:
                legend_color = '#333333'
            plot_item.addLegend(offset=(10, -10), labelTextColor=legend_color)

            _plot_detector_derivatives(plot_item, bf, pen_color)

            plot_item.setLabel('bottom', '<i>f</i> − <i>f</i><sub>center</sub> [kHz]')
            plot_item.setLabel('left', 'Normalised IQ speed  [1/Hz]')
            plot_item.showGrid(x=True, y=True, alpha=0.3)

        plot_widget._detector_id = detector_id
        grid_layout.addWidget(plot_widget, row, col)
        plot_widget.show()

    total_batches = max(1, (len(detector_ids) + batch_size - 1) // batch_size)
    if prev_btn:
        prev_btn.setEnabled(current_batch > 0)
    if next_btn:
        next_btn.setEnabled(current_batch < total_batches - 1)
    if batch_label:
        batch_label.setText(f"{current_batch + 1} of {total_batches}")


def _plot_detector_derivatives(plot_item, bias_finding, pen_color):
    """
    Plot IQ-derivative data for a single detector using its ``bias_finding`` dict.

    Three quantities are plotted on a shared kHz-offset x-axis:

    * **I_speed** — normalised per-step I increments ``diff(I/I_range)``
      (discrete points, color A), with the normalised cubic-spline derivative
      ``I_speed_spline`` overlaid as a smooth curve (same color A).
    * **Q_speed** — same treatment in color B.
    * **arc_length_speed** — overall IQ arc-length speed per Hz (color C).
    * A thin dashed red vertical line at the refined ``bias_frequency``.

    Parameters
    ----------
    plot_item : pg.PlotItem
        The plot to draw on.
    bias_finding : dict
        The ``bias_finding`` sub-dict attached to the selected sweep entry by
        :func:`rfmux.algorithms.measurement.bias_kids.find_bias_points`.
    pen_color : str or tuple
        Fallback axis/text color (``'w'`` dark mode, ``'k'`` light mode).
    """
    freq_sorted = bias_finding.get('frequencies_sorted')
    freq_arc_speed = bias_finding.get('freq_arc_speed')
    I_speed = bias_finding.get('I_speed')
    Q_speed = bias_finding.get('Q_speed')
    arc_length_speed = bias_finding.get('arc_length_speed')
    I_speed_sp = bias_finding.get('I_speed_spline')
    Q_speed_sp = bias_finding.get('Q_speed_spline')
    bias_freq = bias_finding.get('bias_frequency')

    # All arrays must be present and non-empty for a meaningful plot
    if (freq_sorted is None or freq_arc_speed is None
            or I_speed is None or Q_speed is None
            or len(freq_sorted) < 2 or len(freq_arc_speed) == 0):
        return

    freq_sorted = np.asarray(freq_sorted)
    freq_arc_speed = np.asarray(freq_arc_speed)
    I_speed = np.asarray(I_speed)
    Q_speed = np.asarray(Q_speed)

    # I_speed and Q_speed from detect_bifurcation_derivative are dimensionless
    # differences (diff(I/I_range)), not per-Hz quantities.  Divide by the
    # per-step frequency interval so that all three curves share the same
    # 1/Hz units as arc_length_speed and the spline derivatives.
    freq_steps_hz = bias_finding.get('freq_steps_hz')
    if freq_steps_hz is not None and len(freq_steps_hz) == len(I_speed):
        _df = np.abs(np.asarray(freq_steps_hz))
        # Guard against zero-length steps (shouldn't happen but be safe)
        _df = np.where(_df > 0, _df, np.nan)
        I_speed = I_speed / _df
        Q_speed = Q_speed / _df

    # Reference frequency for kHz offset (mean of the full sorted array)
    freq_ref = float(np.mean(freq_sorted))

    def to_khz(f):
        return 1e-3 * (np.asarray(f) - freq_ref)

    freq_arc_khz = to_khz(freq_arc_speed)

    # Colors: A = blue (I), B = orange (Q), C = green (arc_length_speed)
    color_I   = TABLEAU10_COLORS[0]
    color_Q   = TABLEAU10_COLORS[1]
    color_arc = TABLEAU10_COLORS[2]

    alpha_discrete = 140  # semi-transparent for discrete points

    def make_rgba(color, alpha):
        """Convert color spec to (R, G, B, alpha) tuple."""
        if isinstance(color, str):
            # Named colors like 'b', 'r' etc.
            c = pg.mkColor(color)
            return (c.red(), c.green(), c.blue(), alpha)
        if isinstance(color, (list, tuple)) and len(color) >= 3:
            return (int(color[0]), int(color[1]), int(color[2]), alpha)
        return color

    # ── I_speed discrete points ('.-': line + circle symbols) ───────────────
    pen_I_disc = pg.mkPen(make_rgba(color_I, alpha_discrete), width=LINE_WIDTH)
    plot_item.plot(
        freq_arc_khz, I_speed,
        pen=pen_I_disc,
        symbol='o', symbolSize=4,
        symbolPen=pg.mkPen(make_rgba(color_I, alpha_discrete)),
        symbolBrush=pg.mkBrush(make_rgba(color_I, alpha_discrete)),
        name='I speed',
    )

    # ── Q_speed discrete points ('.-': line + circle symbols) ───────────────
    pen_Q_disc = pg.mkPen(make_rgba(color_Q, alpha_discrete), width=LINE_WIDTH)
    plot_item.plot(
        freq_arc_khz, Q_speed,
        pen=pen_Q_disc,
        symbol='o', symbolSize=4,
        symbolPen=pg.mkPen(make_rgba(color_Q, alpha_discrete)),
        symbolBrush=pg.mkBrush(make_rgba(color_Q, alpha_discrete)),
        name='Q speed',
    )

    # ── arc_length_speed: smooth curve derived from stored I/Q splines ───────
    # Use the normalised I and Q derivative splines (already stored in
    # bias_finding) to compute |dI/df + j·dQ/df| over a fine frequency grid.
    # This is identical to what the bias-finding algorithm used, so the curve
    # faithfully represents the quantity that was maximised to locate the bias
    # frequency.  Fall back to the raw discrete points if the splines are absent.
    pen_arc = pg.mkPen(color_arc, width=LINE_WIDTH)
    if I_speed_sp is not None and Q_speed_sp is not None:
        try:
            f_fine = np.linspace(float(freq_sorted[0]), float(freq_sorted[-1]), 500)
            arc_sp_vals = np.sqrt(I_speed_sp(f_fine) ** 2 + Q_speed_sp(f_fine) ** 2)
            plot_item.plot(to_khz(f_fine), arc_sp_vals, pen=pen_arc, name='Arc length speed')
        except Exception:
            if arc_length_speed is not None and len(arc_length_speed) == len(freq_arc_speed):
                plot_item.plot(freq_arc_khz, np.asarray(arc_length_speed),
                               pen=pen_arc, name='Arc length speed')
    elif arc_length_speed is not None and len(arc_length_speed) == len(freq_arc_speed):
        plot_item.plot(freq_arc_khz, np.asarray(arc_length_speed),
                       pen=pen_arc, name='Arc length speed')

    # ── Bias frequency vertical line ─────────────────────────────────────────
    if bias_freq is not None:
        bias_khz = 1e-3 * (float(bias_freq) - freq_ref)
        vline_pen = pg.mkPen('r', style=QtCore.Qt.PenStyle.DashLine, width=1)
        plot_item.addItem(pg.InfiniteLine(
            pos=bias_khz, angle=90, pen=vline_pen, movable=False,
            label=f'f_bias', labelOpts={'color': 'r', 'position': 0.92},
        ))


# ---------------------------------------------------------------------------
# Fit Results grid (Tab 6 — overlays fitted model curves on measured data)
# ---------------------------------------------------------------------------

def update_fit_results_grid(
    grid_layout,
    data_by_detector,
    res_info_dict,
    display_mode,
    display_amplitude_index,
    show_skewed,
    show_nonlinear,
    current_batch,
    batch_size,
    dark_mode,
    unit_mode='dbm',
    normalize=True,
    prev_btn=None,
    next_btn=None,
    batch_label=None,
    widget_cache=None,
    dac_scale=None,
):
    """
    Update a grid layout with per-detector fit-result plots.

    Each subplot shows the measured sweep magnitude (normalised) for the
    selected amplitude, with the skewed Lorentzian and/or nonlinear IQ
    fitted model overlaid as separate coloured curves.

    Parameters
    ----------
    grid_layout : QGridLayout
        Layout to populate.
    data_by_detector : dict
        ``MultisweepPanel.results_by_detector`` —
        ``{code: {iter_idx: entry}}``.
    res_info_dict : dict or None
        Resonator registry ``{code: {bias_amplitude, ...}}``.  Required
        when *display_mode* is ``'bias'``.
    display_mode : str
        ``'index'`` — pick the iteration at sorted amplitude position N;
        ``'bias'``  — pick the iteration whose amplitude matches each
                      resonator's ``bias_amplitude``.
    display_amplitude_index : int
        0-based index used when *display_mode* is ``'index'``.
    show_skewed : bool
        Draw the skewed Lorentzian model overlay.
    show_nonlinear : bool
        Draw the nonlinear IQ model overlay.
    current_batch, batch_size : int
        Batch navigation.
    dark_mode : bool
        Theme flag.
    unit_mode : str
        Not used for rendering (Fit Results always shows normalised
        magnitude), kept for API consistency.
    normalize : bool
        Not used directly (Fit Results always normalises), kept for API
        consistency.
    prev_btn, next_btn : QPushButton or None
        Batch navigation buttons.
    batch_label : QLabel or None
        ``"N of M"`` label.
    widget_cache : list or None
        Reusable ``pg.PlotWidget`` instances.
    dac_scale : float or None
        DAC full-scale in dBm (for subplot title amplitude labels).
    """
    if not data_by_detector:
        return

    # Remove all items from the grid without destroying widgets
    while grid_layout.count():
        grid_layout.takeAt(0)

    detector_ids = sorted(data_by_detector.keys())

    start_idx = current_batch * batch_size
    end_idx = min(start_idx + batch_size, len(detector_ids))
    batch_detectors = detector_ids[start_idx:end_idx]

    if not batch_detectors:
        return

    num_plots = len(batch_detectors)
    ncols = max(1, int(np.ceil(np.sqrt(num_plots))))
    nrows = int(np.ceil(num_plots / ncols))

    bg_color, pen_color = ("k", "w") if dark_mode else ("w", "k")

    # Reset stale stretch factors
    for r in range(grid_layout.rowCount()):
        grid_layout.setRowStretch(r, 0)
    for c in range(grid_layout.columnCount()):
        grid_layout.setColumnStretch(c, 0)
    for r in range(nrows):
        grid_layout.setRowStretch(r, 1)
    for c in range(ncols):
        grid_layout.setColumnStretch(c, 1)

    if widget_cache is None:
        widget_cache = []

    while len(widget_cache) < num_plots:
        pw = pg.PlotWidget()
        pw.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        widget_cache.append(pw)

    for widget in widget_cache:
        widget.hide()

    for idx, detector_id in enumerate(batch_detectors):
        row = idx // ncols
        col = idx % ncols

        plot_widget = widget_cache[idx]
        plot_widget.setBackground(bg_color)
        plot_item = plot_widget.getPlotItem()

        if plot_item:
            plot_item.clear()
            if hasattr(plot_item, 'legend') and plot_item.legend:
                plot_item.legend.scene().removeItem(plot_item.legend)
                plot_item.legend = None

            iter_dict = data_by_detector.get(detector_id, {})

            # ── Amplitude selection ───────────────────────────────────────────
            entry = None
            selected_amp = None

            if display_mode == 'bias':
                bias_amp = (res_info_dict or {}).get(detector_id, {}).get('bias_amplitude')
                if bias_amp is not None:
                    for it_entry in iter_dict.values():
                        if it_entry.get('sweep_amplitude') == bias_amp:
                            entry = it_entry
                            selected_amp = bias_amp
                            break
            else:  # 'index'
                if iter_dict:
                    sorted_items = sorted(
                        iter_dict.items(),
                        key=lambda kv: kv[1].get('sweep_amplitude', 0.0),
                    )
                    target_pos = min(display_amplitude_index, len(sorted_items) - 1)
                    _iter_idx, entry = sorted_items[target_pos]
                    selected_amp = entry.get('sweep_amplitude')

            # ── Title ─────────────────────────────────────────────────────────
            center_freq_hz = None
            if entry is not None:
                center_freq_hz = (
                    entry.get('original_center_frequency')
                    or entry.get('sweep_center_frequency')
                )
            elif iter_dict:
                first_entry = next(iter(iter_dict.values()))
                center_freq_hz = (
                    first_entry.get('original_center_frequency')
                    or first_entry.get('sweep_center_frequency')
                )

            amp_str = (
                UnitConverter.format_probe_label(selected_amp, unit_mode, dac_scale)
                if selected_amp is not None else "?"
            )
            if center_freq_hz is not None:
                title = (
                    f"{detector_id}  "
                    f"(<i>f</i><sub>central</sub> = {center_freq_hz / 1e6:.3f} MHz, "
                    f"amp = {amp_str})"
                )
            else:
                title = f"Detector {detector_id}  (amp = {amp_str})"
            plot_item.setTitle(title, color=pen_color)

            for axis_name in ("left", "bottom", "right", "top"):
                ax = plot_item.getAxis(axis_name)
                if ax:
                    ax.setPen(pen_color)
                    ax.setTextPen(pen_color)

            if entry is None:
                # No data available for the requested amplitude
                _add_centred_text(plot_item, "No data for amplitude", color='#FFA500')
            else:
                _plot_fit_result(
                    plot_item, entry,
                    show_skewed=show_skewed,
                    show_nonlinear=show_nonlinear,
                    pen_color=pen_color,
                    dark_mode=dark_mode,
                )

            plot_item.setLabel('bottom', '<i>f</i> − <i>f</i><sub>central</sub> [kHz]')
            plot_item.setLabel('left', '|S₂₁| (normalised)')
            plot_item.showGrid(x=True, y=True, alpha=0.3)

        plot_widget._detector_id = detector_id
        grid_layout.addWidget(plot_widget, row, col)
        plot_widget.show()

    total_batches = max(1, (len(detector_ids) + batch_size - 1) // batch_size)
    if prev_btn:
        prev_btn.setEnabled(current_batch > 0)
    if next_btn:
        next_btn.setEnabled(current_batch < total_batches - 1)
    if batch_label:
        batch_label.setText(f"{current_batch + 1} of {total_batches}")


def _add_centred_text(plot_item, message, color='#FFA500'):
    """Add a centred amber text annotation to a plot (used for empty-state messages)."""
    text = pg.TextItem(message, color=color, anchor=(0.5, 0.5))
    plot_item.addItem(text)
    text.setPos(0, 0.5)


def _plot_fit_result(plot_item, entry, show_skewed, show_nonlinear, pen_color, dark_mode):
    """
    Plot measured sweep data + fitted model overlays for a single detector entry.

    All curves are shown on a normalised magnitude scale with a kHz-offset
    x-axis centred on the sweep's mean frequency.

    Curves
    ------
    * **Measured** (grey/white): ``|iq_counts| / |iq_counts[-1]|``
    * **Skewed fit** (red, optional): ``entry['skewed_model_mag']`` —
      already normalised by the skewed fitter (A ≈ 1).
      A dashed ``InfiniteLine`` marks ``fit_params['fr']``.
    * **Nonlinear fit** (green, optional):
      ``|nonlinear_model_iq| / mean(|nonlinear_model_iq[[0, -1]]|)``
      A dashed ``InfiniteLine`` marks ``nonlinear_fit_params['fr']``.
    * **Q-value text** (top-left): Qr, Qi, Qc from whichever fits succeeded.
    * **"No fit data" text** (amber, centred): shown when neither fit
      succeeded for this entry.

    Parameters
    ----------
    plot_item : pg.PlotItem
    entry : dict
        One iteration entry from ``results_by_detector[code][iter_idx]``.
    show_skewed, show_nonlinear : bool
    pen_color : str or tuple
        Fallback axis/text colour.
    dark_mode : bool
    """
    freqs = entry.get('frequencies')
    iq = entry.get('iq_counts')

    if freqs is None or iq is None or len(freqs) == 0 or len(iq) == 0:
        _add_centred_text(plot_item, "No sweep data", color='#FFA500')
        return

    freqs = np.asarray(freqs)
    iq = np.asarray(iq)

    # kHz offset from the sweep mean frequency
    freq_ref = float(np.mean(freqs))
    freqs_khz = 1e-3 * (freqs - freq_ref)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_color = '#CCCCCC' if dark_mode else '#333333'
    plot_item.addLegend(offset=(10, 10), labelTextColor=legend_color)

    # ── Measured magnitude (normalised to last point ≈ off-resonance) ────────
    mag_raw = np.abs(iq)
    norm_ref = float(np.abs(iq[-1])) if len(iq) > 0 else 1.0
    if norm_ref == 0.0:
        norm_ref = 1.0
    mag_norm = mag_raw / norm_ref

    meas_color = (180, 180, 180) if dark_mode else (100, 100, 100)  # grey
    plot_item.plot(
        freqs_khz, mag_norm,
        pen=pg.mkPen(meas_color, width=LINE_WIDTH),
        name='Measured',
    )

    skewed_ok = entry.get('skewed_fit_success', False)
    nonlinear_ok = entry.get('nonlinear_fit_success', False)

    # ── Skewed Lorentzian overlay ─────────────────────────────────────────────
    if show_skewed and skewed_ok:
        skewed_mag = entry.get('skewed_model_mag')
        fit_p = entry.get('fit_params') or {}
        fr_skew = fit_p.get('fr')

        if skewed_mag is not None and fr_skew is not None:
            # Build legend name — fr on line 1, Qr/Qi on line 2, Qc on line 3
            fr_str = f"fr={float(fr_skew) * 1e-6:.3f} MHz"
            qr = fit_p.get('Qr')
            qi = fit_p.get('Qi')
            qc = fit_p.get('Qc')
            skewed_name = "Skewed  " + fr_str
            qr_qi_parts = []
            if qr is not None:
                qr_qi_parts.append(f"Qr={qr:.0f}")
            if qi is not None:
                qr_qi_parts.append(f"Qi={qi:.0f}")
            if qr_qi_parts:
                skewed_name += "<br>" + "  ".join(qr_qi_parts)
            if qc is not None:
                skewed_name += f"<br>Qc={qc:.0f}"

            plot_item.plot(
                freqs_khz, np.asarray(skewed_mag),
                pen=pg.mkPen('r', width=LINE_WIDTH),
                name=skewed_name,
            )
            fr_offset_khz = 1e-3 * (float(fr_skew) - freq_ref)
            plot_item.addItem(pg.InfiniteLine(
                pos=fr_offset_khz, angle=90,
                pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DashLine, width=1),
                movable=False,
            ))

    # ── Nonlinear IQ overlay ──────────────────────────────────────────────────
    if show_nonlinear and nonlinear_ok:
        nl_iq = entry.get('nonlinear_model_iq')
        nl_p = entry.get('nonlinear_fit_params') or {}
        fr_nl = nl_p.get('fr')

        if nl_iq is not None and fr_nl is not None:
            nl_iq = np.asarray(nl_iq)
            nl_mag = np.abs(nl_iq)
            # Normalise to off-resonance average (first + last point)
            off_res_avg = float(np.mean(np.abs(nl_iq[[0, -1]])))
            if off_res_avg == 0.0:
                off_res_avg = 1.0
            nl_mag_norm = nl_mag / off_res_avg

            # Build legend name — fr on line 1, Qr/Qi on line 2, Qc/a on line 3
            fr_str_nl = f"fr={float(fr_nl) * 1e-6:.3f} MHz"
            qr = nl_p.get('Qr')
            qi = nl_p.get('Qi')
            qc = nl_p.get('Qc')
            a_val = nl_p.get('a')
            nl_name = "NL  " + fr_str_nl
            qr_qi_parts_nl = []
            if qr is not None:
                qr_qi_parts_nl.append(f"Qr={qr:.0f}")
            if qi is not None:
                qr_qi_parts_nl.append(f"Qi={qi:.0f}")
            if qr_qi_parts_nl:
                nl_name += "<br>" + "  ".join(qr_qi_parts_nl)
            qc_a_parts_nl = []
            if qc is not None:
                qc_a_parts_nl.append(f"Qc={qc:.0f}")
            if a_val is not None:
                qc_a_parts_nl.append(f"a={a_val:.3f}")
            if qc_a_parts_nl:
                nl_name += "<br>" + "  ".join(qc_a_parts_nl)

            plot_item.plot(
                freqs_khz, nl_mag_norm,
                pen=pg.mkPen('g', width=LINE_WIDTH),
                name=nl_name,
            )
            fr_nl_offset_khz = 1e-3 * (float(fr_nl) - freq_ref)
            plot_item.addItem(pg.InfiniteLine(
                pos=fr_nl_offset_khz, angle=90,
                pen=pg.mkPen('g', style=QtCore.Qt.PenStyle.DashLine, width=1),
                movable=False,
            ))

    # ── "No fit data" message (shown when both fits are unavailable) ──────────
    if (show_skewed or show_nonlinear) and not skewed_ok and not nonlinear_ok:
        _add_centred_text(plot_item, "No fit data", color='#FFA500')


# ---------------------------------------------------------------------------
# Public helpers (kept for backward compatibility & use by other modules)
# ---------------------------------------------------------------------------

plot_detector_magnitude = _plot_detector_magnitude
plot_detector_iq = _plot_detector_iq


def create_amplitude_color_map(amplitude_values, dark_mode, reference_amplitudes=None):
    """
    Create a color mapping for amplitude values.

    Uses TABLEAU10_COLORS for few amplitudes, colormap for many.

    Args:
        amplitude_values: Iterable of amplitude values to assign colors to (the
            amplitudes that have actually been collected so far).
        dark_mode: Boolean for theme.
        reference_amplitudes: Optional iterable of *all* amplitude values that
            will ever be used in this sweep (across all sections and iterations).
            When provided, the threshold check and colour-position normalisation
            are based on this full reference set rather than the currently
            observed values.  This ensures that colours remain stable throughout
            a live measurement: the first iteration uses the same colour it will
            have when all iterations are complete, the legend-vs-colorbar
            decision is made once up-front, and the inferno scale extremes
            always correspond to the global min/max amplitude over the whole
            array.  Amplitudes present in *reference_amplitudes* but not yet in
            *amplitude_values* are simply skipped (no entry is added to the
            returned dict for them).

    Returns:
        Dict mapping amplitude values (from *amplitude_values*) to colors.
    """
    # Determine the reference sorted list: use the full expected set when
    # provided, otherwise fall back to what has been observed so far.
    if reference_amplitudes is not None:
        ref_sorted = sorted(set(reference_amplitudes))
    else:
        ref_sorted = sorted(set(amplitude_values))

    num_amps = len(ref_sorted)
    if num_amps == 0:
        return {}

    observed = set(amplitude_values)
    amplitude_to_color = {}
    cmap_name = COLORMAP_CHOICES.get("AMPLITUDE_SWEEP", "inferno")
    use_cmap = pg.colormap.get(cmap_name) if cmap_name else None

    for amp_idx, amp_val in enumerate(ref_sorted):
        # Skip amplitudes that haven't arrived yet — but keep amp_idx stable
        # so that colours for later amplitudes don't shift as data arrives.
        if amp_val not in observed:
            continue

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
