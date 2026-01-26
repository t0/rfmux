"""
Helper functions for rendering per-detector grid plots in multisweep panels.

This module provides reusable plotting functions used by MultisweepPanel
to create grids of per-detector sweep plots.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets

from .utils import LINE_WIDTH, TABLEAU10_COLORS, COLORMAP_CHOICES, AMPLITUDE_COLORMAP_THRESHOLD, square_axes, UnitConverter


def update_sweep_grid(grid_layout, data_by_detector, plot_type, current_batch, batch_size,
                      amplitude_to_color, dark_mode, unit_mode='dbm', normalize=False, 
                      prev_btn=None, next_btn=None, batch_label=None, widget_cache=None):
    """
    Update a grid layout with per-detector sweep plots.
    
    Args:
        grid_layout: QGridLayout to populate with plots
        data_by_detector: Dict {detector_id: {amp: {freq, iq, ...}}}
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
        
    Returns:
        None (updates grid_layout in place)
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
    
    # Calculate grid dimensions
    num_plots = len(batch_detectors)
    
    # Set minimum columns to 4 for better aspect ratios with small numbers of plots
    MIN_COLS = 4
    
    if num_plots <= MIN_COLS:
        # For few plots, use all columns in a single row with empty space
        ncols = MIN_COLS
        nrows = 1
    else:
        # For many plots, use the square root approach with minimum of 4 columns
        ncols = max(MIN_COLS, int(np.ceil(np.sqrt(num_plots))))
        nrows = int(np.ceil(num_plots / ncols))
    
    # Theme colors
    bg_color, pen_color = ("k", "w") if dark_mode else ("w", "k")
    
    # Ensure widget cache exists if provided
    if widget_cache is None:
        widget_cache = []
    
    # Expand cache if needed
    while len(widget_cache) < num_plots:
        plot_widget = pg.PlotWidget()
        widget_cache.append(plot_widget)
    
    # Reuse widgets from cache
    for idx, detector_id in enumerate(batch_detectors):
        row = idx // ncols
        col = idx % ncols
        
        # Reuse widget from cache
        plot_widget = widget_cache[idx]
        plot_widget.setBackground(bg_color)
        plot_item = plot_widget.getPlotItem()
        
        if plot_item:
            # Clear previous data
            plot_item.clear()
            
            title = f"Detector {detector_id}"
            plot_item.setTitle(title, color=pen_color)
            
            # Style axes
            for axis_name in ("left", "bottom", "right", "top"):
                ax = plot_item.getAxis(axis_name)
                if ax:
                    ax.setPen(pen_color)
                    ax.setTextPen(pen_color)
            
            # Plot data
            detector_data = data_by_detector.get(detector_id, {})
            
            if plot_type == 'magnitude':
                plot_detector_magnitude(plot_item, detector_data, amplitude_to_color, pen_color, unit_mode, normalize)
                
                # Set Y-axis label based on unit_mode and normalize
                if normalize:
                    if unit_mode == "dbm":
                        plot_item.setLabel('left', 'Normalized Magnitude', units='dB')
                    else:
                        plot_item.setLabel('left', 'Normalized Magnitude', units='')
                else:
                    if unit_mode == "counts":
                        plot_item.setLabel('left', 'Magnitude', units='Counts')
                    elif unit_mode == "dbm":
                        plot_item.setLabel('left', 'Power', units='dBm')
                    elif unit_mode == "volts":
                        plot_item.setLabel('left', 'Magnitude', units='V')
                
                plot_item.setLabel('bottom', 'Frequency Offset', units='kHz')
            else:  # IQ
                plot_detector_iq(plot_item, detector_data, amplitude_to_color, pen_color, normalize)
                plot_item.setLabel('left', 'Q (Imaginary)')
                plot_item.setLabel('bottom', 'I (Real)')
                # Lock aspect ratio to 1:1 to keep circles circular
                square_axes(plot_item)
            
            plot_item.showGrid(x=True, y=True, alpha=0.3)
        
        grid_layout.addWidget(plot_widget, row, col)
    
    # Update batch navigation
    total_batches = max(1, (len(detector_ids) + batch_size - 1) // batch_size)
    
    if prev_btn:
        prev_btn.setEnabled(current_batch > 0)
    if next_btn:
        next_btn.setEnabled(current_batch < total_batches - 1)
    if batch_label:
        batch_label.setText(f"{current_batch + 1} of {total_batches}")


def plot_detector_magnitude(plot_item, detector_data, amplitude_to_color, pen_color, unit_mode='dbm', normalize=False):
    """
    Plot S21 magnitude sweeps for a single detector across multiple amplitudes.
    
    Args:
        plot_item: PyQtGraph PlotItem to draw on
        detector_data: Dict {amplitude: {'freq': [...], 'iq': [...]}}
        amplitude_to_color: Dict mapping amplitude values to colors
        pen_color: Fallback pen color for single amplitude
        unit_mode: Unit mode for magnitude display ('counts', 'dbm', 'volts')
        normalize: Whether to normalize traces
    """
    amplitudes = sorted(detector_data.keys())
    
    for amp in amplitudes:
        amp_data = detector_data.get(amp, {})
        
        freqs = amp_data.get('freq')
        iq = amp_data.get('iq')
        
        if freqs is None or iq is None or len(freqs) == 0:
            continue
        
        # Calculate magnitude using UnitConverter (matches combined plot logic)
        mag = np.abs(iq)
        mag_converted = UnitConverter.convert_amplitude(mag, iq, unit_mode, normalize=normalize)
        
        # Get color for this amplitude
        if len(amplitudes) == 1:
            pen = pg.mkPen(color=pen_color, width=LINE_WIDTH)
        else:
            color = amplitude_to_color.get(amp, pen_color)
            pen = pg.mkPen(color=color, width=LINE_WIDTH)
        
        # Plot relative to mean frequency (in kHz)
        freqs_rel_khz = 1e-3 * (freqs - np.mean(freqs))
        
        plot_item.plot(freqs_rel_khz, mag_converted, pen=pen)


def plot_detector_iq(plot_item, detector_data, amplitude_to_color, pen_color, normalize=False):
    """
    Plot IQ circles for a single detector across multiple amplitudes.
    
    Args:
        plot_item: PyQtGraph PlotItem to draw on
        detector_data: Dict {amplitude: {'freq': [...], 'iq': [...]}}
        amplitude_to_color: Dict mapping amplitude values to colors
        pen_color: Fallback pen color for single amplitude
        normalize: Whether to normalize IQ data by max magnitude
    """
    amplitudes = sorted(detector_data.keys())
    
    for amp in amplitudes:
        amp_data = detector_data.get(amp, {})
        
        iq = amp_data.get('iq')
        
        if iq is None or len(iq) == 0:
            continue
        
        # Extract I and Q
        i_vals = np.real(iq)
        q_vals = np.imag(iq)
        
        # Optionally normalize by max magnitude
        if normalize:
            mag = np.abs(iq)
            if len(mag) > 0 and np.max(mag) > 0:
                i_vals = i_vals / np.max(mag)
                q_vals = q_vals / np.max(mag)
        
        # Get color for this amplitude
        if len(amplitudes) == 1:
            pen = pg.mkPen(color=pen_color, width=LINE_WIDTH)
        else:
            color = amplitude_to_color.get(amp, pen_color)
            pen = pg.mkPen(color=color, width=LINE_WIDTH)
        
        plot_item.plot(i_vals, q_vals, pen=pen)


def create_amplitude_color_map(amplitude_values, dark_mode):
    """
    Create a color mapping for amplitude values.
    
    Uses TABLEAU10_COLORS for few amplitudes, colormap for many.
    This matches the color scheme used in the combined plots.
    
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
    
    # Use colormap if many amplitudes
    cmap_name = COLORMAP_CHOICES.get("AMPLITUDE_SWEEP", "inferno")
    use_cmap = pg.colormap.get(cmap_name) if cmap_name else None
    
    for amp_idx, amp_val in enumerate(sorted_amplitudes):
        if num_amps <= AMPLITUDE_COLORMAP_THRESHOLD:
            # Use distinct colors for few amplitudes
            color = TABLEAU10_COLORS[amp_idx % len(TABLEAU10_COLORS)]
        else:
            # Use colormap for many amplitudes
            if use_cmap:
                normalized_idx = amp_idx / max(1, num_amps - 1)
                if dark_mode:
                    # For dark mode, map to [0.3, 1.0]
                    map_value = 0.3 + normalized_idx * 0.7
                else:
                    # For light mode, map to [0.0, 0.75]
                    map_value = normalized_idx * 0.75
                color = use_cmap.map(map_value)
            else:
                # Fallback if colormap unavailable
                color = TABLEAU10_COLORS[amp_idx % len(TABLEAU10_COLORS)]
        
        amplitude_to_color[amp_val] = color
    
    return amplitude_to_color


def prepare_detector_data_from_iterations(results_by_iteration):
    """
    Convert iteration-based results to detector-based format for grid plotting.
    
    Args:
        results_by_iteration: Dict {iteration: {amplitude, direction, data: {detector_id: {...}}}}
        
    Returns:
        Dict {detector_id: {amplitude: {'freq': array, 'iq': array, ...}}}
    """
    detector_data = {}
    
    # Group by detector and amplitude, keeping latest iteration for each (amp, direction) pair
    amplitude_direction_to_iteration = {}
    for iter_idx, iter_data in results_by_iteration.items():
        amp_val = iter_data["amplitude"]
        direction = iter_data["direction"]
        key = (amp_val, direction)
        
        # Keep the latest iteration for this amplitude+direction
        if key not in amplitude_direction_to_iteration or iter_idx > amplitude_direction_to_iteration[key]:
            amplitude_direction_to_iteration[key] = iter_idx
    
    # Extract data from the kept iterations
    for (amp_val, direction), iter_idx in amplitude_direction_to_iteration.items():
        iter_data = results_by_iteration[iter_idx]
        res_data = iter_data.get("data", {})
        
        for detector_id, sweep_data in res_data.items():
            if detector_id not in detector_data:
                detector_data[detector_id] = {}
            
            # Store just what we need for plotting
            freqs = sweep_data.get('frequencies', np.array([]))
            iq_complex = sweep_data.get('iq_complex_volts', np.array([]))
            
            if len(freqs) > 0 and len(iq_complex) > 0:
                detector_data[detector_id][amp_val] = {
                    'freq': freqs,
                    'iq': iq_complex,
                    'amplitude': amp_val,
                    'direction': direction
                }
    
    return detector_data
