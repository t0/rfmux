"""Panel for displaying multisweep aggregate plots (dockable)."""
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
import pyqtgraph as pg

# Imports from within the 'periscope' subpackage
from .utils import (
    ScreenshotMixin, find_parent_with_attr, TABLEAU10_COLORS, LINE_WIDTH, square_axes, SquarePlotWidget
)
from .aggregate_data_adapter import (
    extract_multisweep_data, calculate_grid_size
)


class MultisweepAggregatePanel(QtWidgets.QWidget, ScreenshotMixin):
    """
    A dockable panel for displaying aggregate multisweep plots.
    
    Shows either S21 magnitude or IQ circle plots for multiple resonators
    in a grid layout, with batch navigation for large datasets.
    
    Features:
    - Grid layout of sweep plots (up to batch_size resonators at once)
    - Two plot types: S21 magnitude and IQ circles
    - Color-coded by amplitude using 'gnuplot' colormap
    - Batch navigation for datasets with many resonators
    - Colorbar showing amplitude scale
    - Dark mode support
    """
    
    def __init__(self, parent=None, multisweep_panel=None, plot_type="magnitude", 
                 batch_size=50, amplitude_filter=None, dark_mode=False):
        """
        Initialize the aggregate panel.
        
        Args:
            parent: Parent widget
            multisweep_panel: Reference to the MultisweepPanel with data
            plot_type: "magnitude" or "iq" for plot type
            batch_size: Number of resonators to show per batch
            amplitude_filter: Optional list of amplitudes to include (None = all)
            dark_mode: Whether to use dark mode theme
        """
        super().__init__(parent)
        
        self.multisweep_panel = multisweep_panel
        self.plot_type = plot_type
        self.batch_size = batch_size
        self.amplitude_filter = amplitude_filter
        self.dark_mode = dark_mode
        
        # Data storage
        self.plot_data = None
        self.current_batch = 0
        self.total_batches = 0
        self.detector_ids = []
        
        # Plot widgets storage
        self.plot_widgets = []
        self.grid_layout = None
        
        self._setup_ui()
        self._load_and_plot_data()
        
    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create toolbar
        self._setup_toolbar(main_layout)
        
        # Create plot area with scroll
        self._setup_plot_area(main_layout)
        
    def _setup_toolbar(self, layout):
        """Create toolbar with navigation and controls."""
        toolbar = QtWidgets.QWidget()
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)
        
        # Batch navigation
        self.prev_batch_btn = QtWidgets.QPushButton("◀ Previous")
        self.prev_batch_btn.setToolTip("Show previous batch of resonators")
        self.prev_batch_btn.clicked.connect(self._prev_batch)
        toolbar_layout.addWidget(self.prev_batch_btn)
        
        self.batch_label = QtWidgets.QLabel("Batch 1 of 1")
        self.batch_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        toolbar_layout.addWidget(self.batch_label)
        
        self.next_batch_btn = QtWidgets.QPushButton("Next ▶")
        self.next_batch_btn.setToolTip("Show next batch of resonators")
        self.next_batch_btn.clicked.connect(self._next_batch)
        toolbar_layout.addWidget(self.next_batch_btn)
        
        toolbar_layout.addStretch(1)
        
        # Plot type selector
        toolbar_layout.addWidget(QtWidgets.QLabel("Plot Type:"))
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(["S21 Magnitude", "IQ Circles"])
        self.type_combo.setCurrentText("S21 Magnitude" if self.plot_type == "magnitude" else "IQ Circles")
        self.type_combo.currentTextChanged.connect(self._change_plot_type)
        toolbar_layout.addWidget(self.type_combo)
        
        # Screenshot button
        screenshot_btn = QtWidgets.QPushButton("📷")
        screenshot_btn.setToolTip("Export a screenshot of this panel")
        screenshot_btn.clicked.connect(self._export_screenshot)
        toolbar_layout.addWidget(screenshot_btn)
        
        layout.addWidget(toolbar)
        
    def _setup_plot_area(self, layout):
        """Set up the scrollable plot grid area."""
        # Create scroll area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Container for plots
        self.plot_container = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.plot_container)
        self.grid_layout.setSpacing(10)
        
        scroll.setWidget(self.plot_container)
        layout.addWidget(scroll)
        
    def _load_and_plot_data(self):
        """Load data from multisweep panel and create plots."""
        if not self.multisweep_panel:
            return
            
        # Extract data using adapter
        try:
            self.plot_data = extract_multisweep_data(self.multisweep_panel)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Data Error", 
                f"Failed to extract multisweep data: {str(e)}"
            )
            return
            
        if not self.plot_data or not self.plot_data.get('data'):
            QtWidgets.QMessageBox.information(
                self, "No Data", 
                "No multisweep data available to plot."
            )
            return
        
        # Get sorted detector IDs
        self.detector_ids = sorted(self.plot_data['data'].keys())
        
        # Calculate batches
        self.total_batches = max(1, (len(self.detector_ids) + self.batch_size - 1) // self.batch_size)
        self.current_batch = 0
        
        # Create initial batch
        self._update_batch_display()
        
    def _update_batch_display(self):
        """Update the display for the current batch."""
        # Clear existing plots
        self._clear_plots()
        
        # Get detectors for current batch
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.detector_ids))
        batch_detectors = self.detector_ids[start_idx:end_idx]
        
        if not batch_detectors:
            return
        
        # Calculate grid dimensions
        nrows, ncols = calculate_grid_size(len(batch_detectors))
        
        # Set up colormap for amplitude coding
        amplitudes = self._get_all_amplitudes()
        if not amplitudes:
            return
            
        ampmin, ampmax = min(amplitudes), max(amplitudes)
        ampmin = 1e-5 # avoid hitting 0 = black on black background
        # cmnorm = mpl.colors.Normalize(vmin=0, vmax=1)
        cmnorm = mpl.colors.LogNorm(vmin=ampmin, vmax=ampmax)
        colormap = cm.get_cmap('gnuplot')
        
        # Create plots
        bg_color, pen_color = ("k", "w") if self.dark_mode else ("w", "k")
        
        for idx, detector_id in enumerate(batch_detectors):
            row = idx // ncols
            col = idx % ncols
            
            # Create plot widget - use SquarePlotWidget for IQ plots to maintain aspect ratio
            if self.plot_type == "iq":
                plot_widget = SquarePlotWidget()
            else:
                plot_widget = pg.PlotWidget()
            plot_widget.setBackground(bg_color)
            plot_item = plot_widget.getPlotItem()
            
            if plot_item:
                # Set title
                title = f"Detector {detector_id}"
                plot_item.setTitle(title, color=pen_color)
                
                # Style axes
                for axis_name in ("left", "bottom", "right", "top"):
                    ax = plot_item.getAxis(axis_name)
                    if ax:
                        ax.setPen(pen_color)
                        ax.setTextPen(pen_color)
                
                # Plot data for this detector
                detector_data = self.plot_data['data'].get(detector_id, {})
                
                if self.plot_type == "magnitude":
                    self._plot_magnitude(plot_item, detector_data, amplitudes, colormap, cmnorm, pen_color)
                    plot_item.setLabel('left', 'S21 Magnitude', units='dB')
                    plot_item.setLabel('bottom', 'Frequency Offset', units='kHz')
                else:  # IQ
                    self._plot_iq(plot_item, detector_data, amplitudes, colormap, cmnorm, pen_color)
                    plot_item.setLabel('left', 'Q (Imaginary)')
                    plot_item.setLabel('bottom', 'I (Real)')
                
                plot_item.showGrid(x=True, y=True, alpha=0.3)
            
            # Add to grid
            self.grid_layout.addWidget(plot_widget, row, col)
            self.plot_widgets.append(plot_widget)
        
        # Update navigation buttons
        self.prev_batch_btn.setEnabled(self.current_batch > 0)
        self.next_batch_btn.setEnabled(self.current_batch < self.total_batches - 1)
        self.batch_label.setText(f"Batch {self.current_batch + 1} of {self.total_batches}")
        
    def _plot_magnitude(self, plot_item, detector_data, amplitudes, colormap, cmnorm, pen_color):
        """Plot S21 magnitude for a detector."""
        for amp in amplitudes:
            if self.amplitude_filter and amp not in self.amplitude_filter:
                continue
                
            amp_data = detector_data.get(amp)
            if not amp_data:
                continue
                
            freqs = amp_data.get('freq')
            iq = amp_data.get('iq')
            
            if freqs is None or iq is None or len(freqs) == 0:
                continue
            
            # Calculate magnitude in dB
            mag = np.abs(iq)
            mag_db = 20 * np.log10(mag + 1e-12)  # Avoid log(0)
            
            # Get color for this amplitude
            color_rgba = colormap(cmnorm(amp))
            color_rgb = tuple(int(c * 255) for c in color_rgba[:3])
            
            # Plot - use relative frequency in kHz from center
            freqs_rel_khz = 1e-3 * (freqs - np.mean(freqs))
            
            # Use visible color - if only one amplitude, use default pen_color
            if len(amplitudes) == 1:
                pen = pg.mkPen(color=pen_color, width=LINE_WIDTH)
            else:
                pen = pg.mkPen(color=color_rgb, width=LINE_WIDTH)
            
            plot_item.plot(freqs_rel_khz, mag_db, pen=pen)
            
    def _plot_iq(self, plot_item, detector_data, amplitudes, colormap, cmnorm, pen_color):
        """Plot IQ circles for a detector."""
        for amp in amplitudes:
            if self.amplitude_filter and amp not in self.amplitude_filter:
                continue
                
            amp_data = detector_data.get(amp)
            if not amp_data:
                continue
                
            iq = amp_data.get('iq')
            
            if iq is None or len(iq) == 0:
                continue
            
            # Extract I and Q
            i_vals = np.real(iq)
            q_vals = np.imag(iq)
            
            # Normalize by magnitude for cleaner display
            mag = np.abs(iq)
            if len(mag) > 0 and np.max(mag) > 0:
                i_vals = i_vals / np.max(mag)
                q_vals = q_vals / np.max(mag)
            
            # Get color for this amplitude
            color_rgba = colormap(cmnorm(amp))
            color_rgb = tuple(int(c * 255) for c in color_rgba[:3])
            
            # Use visible color - if only one amplitude, use default pen_color
            if len(amplitudes) == 1:
                pen = pg.mkPen(color=pen_color, width=LINE_WIDTH)
            else:
                pen = pg.mkPen(color=color_rgb, width=LINE_WIDTH)
            
            plot_item.plot(i_vals, q_vals, pen=pen)
        
        # Make the plot square with equal scaling (important for IQ circles)
        square_axes(plot_item)
            
    def _get_all_amplitudes(self):
        """Get list of all amplitudes in the data."""
        if not self.plot_data or not self.plot_data.get('data'):
            return []
        
        amplitudes = set()
        for detector_data in self.plot_data['data'].values():
            amplitudes.update(detector_data.keys())
        
        return sorted(amplitudes)
        
    def _clear_plots(self):
        """Clear all plot widgets from the grid."""
        # Remove all widgets from grid
        for plot_widget in self.plot_widgets:
            self.grid_layout.removeWidget(plot_widget)
            plot_widget.deleteLater()
        
        self.plot_widgets.clear()
        
    def _prev_batch(self):
        """Show previous batch of resonators."""
        if self.current_batch > 0:
            self.current_batch -= 1
            self._update_batch_display()
            
    def _next_batch(self):
        """Show next batch of resonators."""
        if self.current_batch < self.total_batches - 1:
            self.current_batch += 1
            self._update_batch_display()
            
    def _change_plot_type(self, text):
        """Change the plot type and redraw."""
        new_type = "magnitude" if text == "S21 Magnitude" else "iq"
        if new_type != self.plot_type:
            self.plot_type = new_type
            self._update_batch_display()
            
    def apply_theme(self, dark_mode: bool):
        """Apply theme and redraw plots."""
        self.dark_mode = dark_mode
        self._update_batch_display()
