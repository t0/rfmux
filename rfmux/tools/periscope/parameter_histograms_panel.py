"""Panel for displaying multisweep parameter histograms (dockable)."""
import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
import pyqtgraph as pg

# Imports from within the 'periscope' subpackage
from .utils import (
    ScreenshotMixin, find_parent_with_attr, TABLEAU10_COLORS, LINE_WIDTH
)
from .aggregate_data_adapter import (
    extract_multisweep_data, get_parameter_arrays, filter_failed_fits
)


class ParameterHistogramsPanel(QtWidgets.QWidget, ScreenshotMixin):
    """
    A dockable panel for displaying parameter statistics and histograms.
    
    Shows a 4-panel layout with:
    - Frequency locations (scatter plot)
    - Qr distribution (log-binned histogram)
    - Qc distribution (log-binned histogram)
    - Qi distribution (log-binned histogram)
    
    Features:
    - Amplitude selector to filter data
    - Adjustable bin count for histograms
    - Statistics overlay (mean, median, std dev)
    - Yield count display
    - Dark mode support
    """
    
    def __init__(self, parent=None, multisweep_panel=None, amplitude_idx=None, 
                 nbins=30, dark_mode=False):
        """
        Initialize the histograms panel.
        
        Args:
            parent: Parent widget
            multisweep_panel: Reference to the MultisweepPanel with data
            amplitude_idx: Index of amplitude to display (None = use last/highest)
            nbins: Number of bins for histograms
            dark_mode: Whether to use dark mode theme
        """
        super().__init__(parent)
        
        self.multisweep_panel = multisweep_panel
        self.amplitude_idx = amplitude_idx
        self.nbins = nbins
        self.dark_mode = dark_mode
        
        # Data storage
        self.plot_data = None
        self.available_amplitudes = []
        
        # Plot widgets
        self.freq_plot = None
        self.qr_plot = None
        self.qc_plot = None
        self.qi_plot = None
        
        self._setup_ui()
        self._load_and_plot_data()
        
    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create toolbar
        self._setup_toolbar(main_layout)
        
        # Create plot area with 2x2 grid
        self._setup_plot_area(main_layout)
        
    def _setup_toolbar(self, layout):
        """Create toolbar with controls."""
        toolbar = QtWidgets.QWidget()
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)
        
        # Amplitude selector
        toolbar_layout.addWidget(QtWidgets.QLabel("Amplitude:"))
        self.amp_combo = QtWidgets.QComboBox()
        self.amp_combo.setMinimumWidth(150)
        self.amp_combo.currentIndexChanged.connect(self._amplitude_changed)
        toolbar_layout.addWidget(self.amp_combo)
        
        toolbar_layout.addStretch(1)
        
        # Bin count control
        toolbar_layout.addWidget(QtWidgets.QLabel("Bins:"))
        self.bins_spin = QtWidgets.QSpinBox()
        self.bins_spin.setRange(10, 100)
        self.bins_spin.setValue(self.nbins)
        self.bins_spin.valueChanged.connect(self._bins_changed)
        toolbar_layout.addWidget(self.bins_spin)
        
        # Screenshot button
        screenshot_btn = QtWidgets.QPushButton("📷")
        screenshot_btn.setToolTip("Export a screenshot of this panel")
        screenshot_btn.clicked.connect(self._export_screenshot)
        toolbar_layout.addWidget(screenshot_btn)
        
        layout.addWidget(toolbar)
        
    def _setup_plot_area(self, layout):
        """Set up the 2x2 grid of plots."""
        # Create container widget with grid layout
        plot_container = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(plot_container)
        grid.setSpacing(10)
        
        bg_color, pen_color = ("k", "w") if self.dark_mode else ("w", "k")
        
        # Frequency scatter plot (top left)
        self.freq_plot = pg.PlotWidget()
        self.freq_plot.setBackground(bg_color)
        freq_item = self.freq_plot.getPlotItem()
        if freq_item:
            freq_item.setTitle("Resonance Frequencies", color=pen_color)
            freq_item.setLabel('left', 'Frequency', units='Hz')
            freq_item.setLabel('bottom', 'Detector ID')
            freq_item.showGrid(x=True, y=True, alpha=0.3)
            self._style_axes(freq_item, pen_color)
        grid.addWidget(self.freq_plot, 0, 0)
        
        # Qr histogram (top right)
        self.qr_plot = pg.PlotWidget()
        self.qr_plot.setBackground(bg_color)
        qr_item = self.qr_plot.getPlotItem()
        if qr_item:
            qr_item.setTitle("Qr Distribution", color=pen_color)
            qr_item.setLabel('left', 'Count')
            qr_item.setLabel('bottom', 'Qr')
            qr_item.showGrid(x=True, y=True, alpha=0.3)
            self._style_axes(qr_item, pen_color)
        grid.addWidget(self.qr_plot, 0, 1)
        
        # Qc histogram (bottom left)
        self.qc_plot = pg.PlotWidget()
        self.qc_plot.setBackground(bg_color)
        qc_item = self.qc_plot.getPlotItem()
        if qc_item:
            qc_item.setTitle("Qc Distribution", color=pen_color)
            qc_item.setLabel('left', 'Count')
            qc_item.setLabel('bottom', 'Qc')
            qc_item.showGrid(x=True, y=True, alpha=0.3)
            self._style_axes(qc_item, pen_color)
        grid.addWidget(self.qc_plot, 1, 0)
        
        # Qi histogram (bottom right)
        self.qi_plot = pg.PlotWidget()
        self.qi_plot.setBackground(bg_color)
        qi_item = self.qi_plot.getPlotItem()
        if qi_item:
            qi_item.setTitle("Qi Distribution", color=pen_color)
            qi_item.setLabel('left', 'Count')
            qi_item.setLabel('bottom', 'Qi')
            qi_item.showGrid(x=True, y=True, alpha=0.3)
            self._style_axes(qi_item, pen_color)
        grid.addWidget(self.qi_plot, 1, 1)
        
        layout.addWidget(plot_container)
        
    def _style_axes(self, plot_item, pen_color):
        """Apply consistent axis styling."""
        for axis_name in ("left", "bottom", "right", "top"):
            ax = plot_item.getAxis(axis_name)
            if ax:
                ax.setPen(pen_color)
                ax.setTextPen(pen_color)
                
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
            
        if not self.plot_data or not self.plot_data.get('fit_params'):
            QtWidgets.QMessageBox.information(
                self, "No Data", 
                "No fit parameters available to plot."
            )
            return
        
        # Get available amplitudes from the data
        self._populate_amplitude_selector()
        
        # Plot data for initial amplitude
        self._update_plots()
        
    def _populate_amplitude_selector(self):
        """Populate amplitude selector with available amplitudes."""
        if not self.plot_data or not self.plot_data.get('data'):
            return
        
        # Get all unique amplitudes from any detector
        amplitudes_set = set()
        for detector_data in self.plot_data['data'].values():
            amplitudes_set.update(detector_data.keys())
        
        self.available_amplitudes = sorted(amplitudes_set)
        
        # Populate combo box
        self.amp_combo.clear()
        for amp in self.available_amplitudes:
            self.amp_combo.addItem(f"{amp:.6f}")
        
        # Select the amplitude index (default to last/highest)
        if self.amplitude_idx is not None and self.amplitude_idx < len(self.available_amplitudes):
            self.amp_combo.setCurrentIndex(self.amplitude_idx)
        else:
            self.amp_combo.setCurrentIndex(len(self.available_amplitudes) - 1)
            
    def _amplitude_changed(self, index):
        """Handle amplitude selection change."""
        if index >= 0:
            self.amplitude_idx = index
            self._update_plots()
            
    def _bins_changed(self, value):
        """Handle bin count change."""
        self.nbins = value
        self._update_plots()
        
    def _update_plots(self):
        """Update all plots with current settings."""
        if not self.plot_data or self.amplitude_idx is None:
            print("[DEBUG] _update_plots: No plot_data or amplitude_idx is None")
            return
            
        if self.amplitude_idx >= len(self.available_amplitudes):
            print(f"[DEBUG] _update_plots: amplitude_idx {self.amplitude_idx} >= len(available_amplitudes) {len(self.available_amplitudes)}")
            return
        
        amp = self.available_amplitudes[self.amplitude_idx]
        fit_params = self.plot_data.get('fit_params', {})
        
        print(f"\n[DEBUG] _update_plots: Processing amplitude index {self.amplitude_idx} (value: {amp})")
        print(f"[DEBUG] Number of detectors in fit_params: {len(fit_params)}")
        
        # Extract parameter arrays (pass amplitude_idx, not amplitude value)
        try:
            fr_values, fr_ids = get_parameter_arrays(fit_params, 'fr', self.amplitude_idx)
            qr_values, qr_ids = get_parameter_arrays(fit_params, 'Qr', self.amplitude_idx)
            qc_values, qc_ids = get_parameter_arrays(fit_params, 'Qc', self.amplitude_idx)
            qi_values, qi_ids = get_parameter_arrays(fit_params, 'Qi', self.amplitude_idx)
            
            print(f"[DEBUG] Extracted fr: {len(fr_values)} values, detector_ids: {fr_ids}")
            print(f"[DEBUG] Extracted Qr: {len(qr_values)} values, detector_ids: {qr_ids}")
            print(f"[DEBUG] Extracted Qc: {len(qc_values)} values, detector_ids: {qc_ids}")
            print(f"[DEBUG] Extracted Qi: {len(qi_values)} values, detector_ids: {qi_ids}")
        except Exception as e:
            print(f"[DEBUG] Error extracting parameters: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Convert to dictionaries for filtering
        fr_dict = dict(zip(fr_ids, fr_values))
        qr_dict = dict(zip(qr_ids, qr_values))
        qc_dict = dict(zip(qc_ids, qc_values))
        qi_dict = dict(zip(qi_ids, qi_values))
        
        print(f"[DEBUG] Created dicts - fr: {len(fr_dict)}, Qr: {len(qr_dict)}, Qc: {len(qc_dict)}, Qi: {len(qi_dict)}")
        
        # Filter out failed fits (NaN values)
        fr_valid = filter_failed_fits(fr_dict)
        qr_valid = filter_failed_fits(qr_dict)
        qc_valid = filter_failed_fits(qc_dict)
        qi_valid = filter_failed_fits(qi_dict)
        
        print(f"[DEBUG] After filtering - fr: {len(fr_valid)}, Qr: {len(qr_valid)}, Qc: {len(qc_valid)}, Qi: {len(qi_valid)}")
        if fr_valid:
            print(f"[DEBUG] Sample fr data: {list(fr_valid.items())[:3]}")
        if qr_valid:
            print(f"[DEBUG] Sample Qr data: {list(qr_valid.items())[:3]}")
        
        # Update frequency scatter plot
        self._plot_frequency_scatter(fr_valid)
        
        # Update Q histograms
        self._plot_q_histogram(self.qr_plot, qr_valid, "Qr")
        self._plot_q_histogram(self.qc_plot, qc_valid, "Qc")
        self._plot_q_histogram(self.qi_plot, qi_valid, "Qi")
        
    def _plot_frequency_scatter(self, fr_dict):
        """Plot resonance frequency scatter."""
        if not self.freq_plot:
            return
        
        freq_item = self.freq_plot.getPlotItem()
        if not freq_item:
            return
            
        # Clear previous plot
        freq_item.clear()
        
        if not fr_dict:
            return
        
        # Extract detector IDs and frequencies
        detector_ids = sorted(fr_dict.keys())
        freqs = [fr_dict[det_id] for det_id in detector_ids]
        
        # Create scatter plot
        color = TABLEAU10_COLORS[0]
        scatter = pg.ScatterPlotItem(
            x=detector_ids,
            y=freqs,
            size=8,
            pen=pg.mkPen(color, width=1),
            brush=pg.mkBrush(color)
        )
        freq_item.addItem(scatter)
        
        # Add yield count to title
        bg_color, pen_color = ("k", "w") if self.dark_mode else ("w", "k")
        total = len(detector_ids)
        title = f"Resonance Frequencies (Yield: {total})"
        freq_item.setTitle(title, color=pen_color)
        
        # Auto range
        self.freq_plot.autoRange()
        
    def _plot_q_histogram(self, plot_widget, q_dict, param_name):
        """Plot Q factor histogram with log binning."""
        if not plot_widget:
            return
        
        plot_item = plot_widget.getPlotItem()
        if not plot_item:
            return
            
        # Clear previous plot
        plot_item.clear()
        
        if not q_dict:
            return
        
        # Extract Q values
        q_values = np.array(list(q_dict.values()))
        
        if len(q_values) == 0:
            return
        
        # Use log-spaced bins for Q factors
        q_min = np.min(q_values)
        q_max = np.max(q_values)
        
        if q_min <= 0 or q_max <= 0:
            # Fall back to linear bins if we have non-positive values
            bins = np.linspace(q_min, q_max, self.nbins + 1)
        else:
            # Log-spaced bins
            bins = np.logspace(np.log10(q_min), np.log10(q_max), self.nbins + 1)
        
        # Create histogram
        counts, edges = np.histogram(q_values, bins=bins)
        
        # Plot as bar graph
        x = (edges[:-1] + edges[1:]) / 2  # Bin centers
        width = edges[1:] - edges[:-1]
        
        color = TABLEAU10_COLORS[1]
        bar_graph = pg.BarGraphItem(
            x=x,
            height=counts,
            width=width,
            brush=color,
            pen=pg.mkPen(color, width=1)
        )
        plot_item.addItem(bar_graph)
        
        # Calculate statistics
        mean_q = np.mean(q_values)
        median_q = np.median(q_values)
        std_q = np.std(q_values)
        
        # Add statistics text
        bg_color, pen_color = ("k", "w") if self.dark_mode else ("w", "k")
        stats_text = (
            f"{param_name} Distribution (N={len(q_values)})\n"
            f"Mean: {mean_q:.2e}\n"
            f"Median: {median_q:.2e}\n"
            f"Std: {std_q:.2e}"
        )
        plot_item.setTitle(stats_text, color=pen_color)
        
        # Auto range
        plot_widget.autoRange()
        
    def apply_theme(self, dark_mode: bool):
        """Apply theme and redraw plots."""
        self.dark_mode = dark_mode
        
        # Update plot backgrounds and axes
        bg_color, pen_color = ("k", "w") if dark_mode else ("w", "k")
        
        for plot_widget in [self.freq_plot, self.qr_plot, self.qc_plot, self.qi_plot]:
            if plot_widget:
                plot_widget.setBackground(bg_color)
                plot_item = plot_widget.getPlotItem()
                if plot_item:
                    self._style_axes(plot_item, pen_color)
        
        # Redraw with updated theme
        self._update_plots()
