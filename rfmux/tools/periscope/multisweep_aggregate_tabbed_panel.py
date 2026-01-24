"""Unified tabbed panel for multisweep aggregate plots."""
import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
import pyqtgraph as pg

# Imports from within the 'periscope' subpackage
from .utils import ScreenshotMixin
from .aggregate_data_adapter import extract_multisweep_data


class MultisweepAggregateTabbedPanel(QtWidgets.QWidget, ScreenshotMixin):
    """
    A unified dockable panel with tabs for different aggregate plot types.
    
    Contains three tabs:
    - Magnitude vs Frequency sweeps
    - IQ Circle sweeps
    - Parameter Histograms
    
    Features:
    - Automatic updates as new amplitude data arrives
    - Toolbar with controls (batch size, amplitude filter, bins)
    - Dark mode support
    - Incremental plotting of new amplitude sweeps
    """
    
    def __init__(self, parent=None, multisweep_panel=None, dark_mode=False, 
                 is_auto=False):
        """
        Initialize the tabbed aggregate panel.
        
        Args:
            parent: Parent widget
            multisweep_panel: Reference to the MultisweepPanel with data
            dark_mode: Whether to use dark mode theme
            is_auto: Whether this is the auto-generated panel (vs manual)
        """
        super().__init__(parent)
        
        self.multisweep_panel = multisweep_panel
        self.dark_mode = dark_mode
        self.is_auto = is_auto
        
        # Track which amplitudes have been plotted
        self.plotted_amplitudes = set()
        
        # Default settings
        self.batch_size = 50
        self.nbins = 30
        self.current_batch = 0
        
        # Tab-specific plot storage
        self.mag_plots = {}  # {detector_id: plot_widget}
        self.iq_plots = {}   # {detector_id: plot_widget}
        self.hist_plots = {}  # {plot_name: plot_widget}
        
        # Grid layouts for sweep tabs
        self.mag_grid = None
        self.iq_grid = None
        
        self._setup_ui()
        
        # Initial plot if data already available
        if multisweep_panel and multisweep_panel.results_by_iteration:
            self._full_replot()
        
    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create toolbar
        self._setup_toolbar(main_layout)
        
        # Create tabbed widget
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        
        # Create tabs and store grid references
        self.mag_tab, self.mag_grid = self._create_sweep_tab()
        self.iq_tab, self.iq_grid = self._create_sweep_tab()
        self.hist_tab = self._create_histogram_tab()
        
        print(f"[DEBUG] After creating tabs:")
        print(f"[DEBUG]   self.mag_grid = {self.mag_grid}")
        print(f"[DEBUG]   self.iq_grid = {self.iq_grid}")
        print(f"[DEBUG]   mag_grid type: {type(self.mag_grid)}")
        
        self.tab_widget.addTab(self.mag_tab, "Magnitude Sweeps")
        self.tab_widget.addTab(self.iq_tab, "IQ Circles")
        self.tab_widget.addTab(self.hist_tab, "Histograms")
        
        main_layout.addWidget(self.tab_widget)
        
        # Update toolbar visibility based on initial tab
        self._update_toolbar_visibility()
        
    def _setup_toolbar(self, layout):
        """Create toolbar with controls."""
        toolbar = QtWidgets.QWidget()
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)
        
        # Batch size control (for sweep tabs)
        toolbar_layout.addWidget(QtWidgets.QLabel("Batch Size:"))
        self.batch_size_spin = QtWidgets.QSpinBox()
        self.batch_size_spin.setRange(10, 200)
        self.batch_size_spin.setValue(self.batch_size)
        self.batch_size_spin.setSingleStep(10)
        self.batch_size_spin.valueChanged.connect(self._batch_size_changed)
        toolbar_layout.addWidget(self.batch_size_spin)
        
        # Batch navigation
        self.prev_batch_btn = QtWidgets.QPushButton("◀ Previous")
        self.prev_batch_btn.clicked.connect(self._prev_batch)
        toolbar_layout.addWidget(self.prev_batch_btn)
        
        self.batch_label = QtWidgets.QLabel("Batch 1 of 1")
        self.batch_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        toolbar_layout.addWidget(self.batch_label)
        
        self.next_batch_btn = QtWidgets.QPushButton("Next ▶")
        self.next_batch_btn.clicked.connect(self._next_batch)
        toolbar_layout.addWidget(self.next_batch_btn)
        
        toolbar_layout.addStretch(1)
        
        # Amplitude filter (for histogram tab)
        toolbar_layout.addWidget(QtWidgets.QLabel("Amplitude:"))
        self.amp_combo = QtWidgets.QComboBox()
        self.amp_combo.setMinimumWidth(150)
        self.amp_combo.currentIndexChanged.connect(self._amplitude_changed)
        toolbar_layout.addWidget(self.amp_combo)
        
        # Bins control (for histogram tab)
        toolbar_layout.addWidget(QtWidgets.QLabel("Bins:"))
        self.bins_spin = QtWidgets.QSpinBox()
        self.bins_spin.setRange(10, 100)
        self.bins_spin.setValue(self.nbins)
        self.bins_spin.valueChanged.connect(self._bins_changed)
        toolbar_layout.addWidget(self.bins_spin)
        
        # Screenshot button
        screenshot_btn = QtWidgets.QPushButton("📷")
        screenshot_btn.setToolTip("Export a screenshot of current tab")
        screenshot_btn.clicked.connect(self._export_screenshot)
        toolbar_layout.addWidget(screenshot_btn)
        
        layout.addWidget(toolbar)
        
    def _create_sweep_tab(self):
        """Create a tab for sweep plots (magnitude or IQ). Returns (tab, grid_layout)."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Scroll area for plots
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        
        # Container for grid
        container = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(container)
        grid.setSpacing(10)
        
        scroll.setWidget(container)
        layout.addWidget(scroll)
        
        return tab, grid
        
    def _create_histogram_tab(self):
        """Create the histogram tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create 2x2 grid of plots
        grid = QtWidgets.QGridLayout()
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
        
        layout.addLayout(grid)
        
        return tab
        
    def _style_axes(self, plot_item, pen_color):
        """Apply consistent axis styling."""
        for axis_name in ("left", "bottom", "right", "top"):
            ax = plot_item.getAxis(axis_name)
            if ax:
                ax.setPen(pen_color)
                ax.setTextPen(pen_color)
                
    def _on_tab_changed(self, index):
        """Handle tab change to update toolbar visibility."""
        self._update_toolbar_visibility()
        
    def _update_toolbar_visibility(self):
        """Update which toolbar controls are visible based on current tab."""
        current_index = self.tab_widget.currentIndex()
        
        # Batch controls visible for sweep tabs (0, 1)
        is_sweep_tab = current_index in (0, 1)
        self.batch_size_spin.setVisible(is_sweep_tab)
        self.prev_batch_btn.setVisible(is_sweep_tab)
        self.batch_label.setVisible(is_sweep_tab)
        self.next_batch_btn.setVisible(is_sweep_tab)
        
        # Histogram controls visible for histogram tab (2)
        is_hist_tab = current_index == 2
        self.amp_combo.setVisible(is_hist_tab)
        self.bins_spin.setVisible(is_hist_tab)
        
        # Find labels and hide/show them too
        toolbar = self.batch_size_spin.parent()
        if toolbar:
            for i in range(toolbar.layout().count()):
                item = toolbar.layout().itemAt(i)
                if item and item.widget() and isinstance(item.widget(), QtWidgets.QLabel):
                    label = item.widget()
                    text = label.text()
                    if text in ("Batch Size:", "Amplitude:", "Bins:"):
                        if text == "Batch Size:":
                            label.setVisible(is_sweep_tab)
                        else:
                            label.setVisible(is_hist_tab)
                            
    def _batch_size_changed(self, value):
        """Handle batch size change."""
        self.batch_size = value
        self.current_batch = 0
        self._update_sweep_tabs()
        
    def _prev_batch(self):
        """Show previous batch."""
        if self.current_batch > 0:
            self.current_batch -= 1
            self._update_sweep_tabs()
            
    def _next_batch(self):
        """Show next batch."""
        # Calculate max batch for current tab
        current_index = self.tab_widget.currentIndex()
        if not self.multisweep_panel or not self.multisweep_panel.results_by_iteration:
            return
            
        data = extract_multisweep_data(self.multisweep_panel)
        if not data or not data.get('data'):
            return
            
        num_detectors = len(data['data'])
        total_batches = max(1, (num_detectors + self.batch_size - 1) // self.batch_size)
        
        if self.current_batch < total_batches - 1:
            self.current_batch += 1
            self._update_sweep_tabs()
            
    def _amplitude_changed(self, index):
        """Handle amplitude selection change for histograms."""
        self._update_histogram_tab()
        
    def _bins_changed(self, value):
        """Handle bins change for histograms."""
        self.nbins = value
        self._update_histogram_tab()
        
    def update_with_new_data(self):
        """
        Update plots with new amplitude data from multisweep panel.
        Called when new amplitude sweep completes.
        """
        if not self.multisweep_panel:
            return
            
        # For incremental updates, just redraw everything
        # (more sophisticated approach would track what changed)
        self._full_replot()
        
    def _full_replot(self):
        """Fully replot all tabs with current data."""
        print(f"[DEBUG] _full_replot called")
        if self.multisweep_panel:
            print(f"[DEBUG] multisweep_panel exists, results_by_iteration has {len(self.multisweep_panel.results_by_iteration)} items")
        self._update_sweep_tabs()
        self._update_histogram_tab()
        self._populate_amplitude_selector()
        
    def _update_sweep_tabs(self):
        """Update both magnitude and IQ sweep tabs."""
        print(f"[DEBUG] _update_sweep_tabs called")
        print(f"[DEBUG]   self.mag_grid in _update_sweep_tabs = {self.mag_grid}")
        print(f"[DEBUG]   self.iq_grid in _update_sweep_tabs = {self.iq_grid}")
        
        if not self.multisweep_panel:
            return
            
        try:
            data = extract_multisweep_data(self.multisweep_panel)
        except Exception as e:
            print(f"Error extracting multisweep data: {e}")
            return
            
        if not data or not data.get('data'):
            return
            
        # Update magnitude tab
        print(f"[DEBUG] About to call _update_sweep_tab with mag_grid = {self.mag_grid}")
        self._update_sweep_tab(self.mag_grid, data, plot_type='magnitude')
        
        # Update IQ tab
        print(f"[DEBUG] About to call _update_sweep_tab with iq_grid = {self.iq_grid}")
        self._update_sweep_tab(self.iq_grid, data, plot_type='iq')
        
    def _update_sweep_tab(self, grid_layout, data, plot_type='magnitude'):
        """Update a sweep tab with current batch of plots."""
        print(f"[DEBUG] _update_sweep_tab called for {plot_type}")
        print(f"[DEBUG] grid_layout = {grid_layout}")
        print(f"[DEBUG] grid_layout is None: {grid_layout is None}")
        
        if grid_layout is None:
            print(f"[DEBUG] grid_layout is None, returning")
            return
            
        # Clear existing plots
        while grid_layout.count():
            item = grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        # Get detector IDs
        detector_ids = sorted(data['data'].keys())
        print(f"[DEBUG] Found {len(detector_ids)} detectors: {detector_ids[:5]}...")
        if not detector_ids:
            print(f"[DEBUG] No detector_ids, returning")
            return
            
        # Calculate batch range
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(detector_ids))
        batch_detectors = detector_ids[start_idx:end_idx]
        
        # Calculate grid dimensions
        num_plots = len(batch_detectors)
        ncols = int(np.ceil(np.sqrt(num_plots)))
        nrows = int(np.ceil(num_plots / ncols))
        
        # Get colormap setup
        amplitudes = self._get_all_amplitudes(data)
        if not amplitudes:
            return
            
        import matplotlib.cm as cm
        import matplotlib as mpl
        
        # Handle single amplitude case
        if len(amplitudes) == 1:
            # For single amplitude, we don't need colormap normalization
            ampmin = amplitudes[0]
            ampmax = amplitudes[0] * 1.1  # Slightly larger to avoid vmin==vmax
            cmnorm = mpl.colors.Normalize(vmin=ampmin, vmax=ampmax)
        else:
            ampmin = max(1e-5, min(amplitudes))
            ampmax = max(amplitudes)
            cmnorm = mpl.colors.LogNorm(vmin=ampmin, vmax=ampmax)
        
        colormap = cm.get_cmap('gnuplot')
        
        bg_color, pen_color = ("k", "w") if self.dark_mode else ("w", "k")
        
        # Create plots
        for idx, detector_id in enumerate(batch_detectors):
            row = idx // ncols
            col = idx % ncols
            
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground(bg_color)
            plot_item = plot_widget.getPlotItem()
            
            if plot_item:
                title = f"Detector {detector_id}"
                plot_item.setTitle(title, color=pen_color)
                
                self._style_axes(plot_item, pen_color)
                
                # Plot data
                detector_data = data['data'].get(detector_id, {})
                
                if plot_type == 'magnitude':
                    self._plot_magnitude(plot_item, detector_data, amplitudes, 
                                       colormap, cmnorm, pen_color)
                    plot_item.setLabel('left', 'S21 Magnitude', units='dB')
                    plot_item.setLabel('bottom', 'Frequency Offset', units='kHz')
                else:  # IQ
                    self._plot_iq(plot_item, detector_data, amplitudes, 
                                colormap, cmnorm, pen_color)
                    plot_item.setLabel('left', 'Q (Imaginary)')
                    plot_item.setLabel('bottom', 'I (Real)')
                
                plot_item.showGrid(x=True, y=True, alpha=0.3)
            
            grid_layout.addWidget(plot_widget, row, col)
            
        # Update batch navigation
        total_batches = max(1, (len(detector_ids) + self.batch_size - 1) // self.batch_size)
        self.prev_batch_btn.setEnabled(self.current_batch > 0)
        self.next_batch_btn.setEnabled(self.current_batch < total_batches - 1)
        self.batch_label.setText(f"Batch {self.current_batch + 1} of {total_batches}")
        
    def _plot_magnitude(self, plot_item, detector_data, amplitudes, colormap, cmnorm, pen_color):
        """Plot S21 magnitude for a detector."""
        from .utils import LINE_WIDTH
        
        for amp in amplitudes:
            amp_data = detector_data.get(amp)
            if not amp_data:
                continue
                
            freqs = amp_data.get('freq')
            iq = amp_data.get('iq')
            
            if freqs is None or iq is None or len(freqs) == 0:
                continue
            
            # Calculate magnitude in dB
            mag = np.abs(iq)
            mag_db = 20 * np.log10(mag + 1e-12)
            
            # Get color
            color_rgba = colormap(cmnorm(amp))
            color_rgb = tuple(int(c * 255) for c in color_rgba[:3])
            
            # Plot
            freqs_rel_khz = 1e-3 * (freqs - np.mean(freqs))
            
            if len(amplitudes) == 1:
                pen = pg.mkPen(color=pen_color, width=LINE_WIDTH)
            else:
                pen = pg.mkPen(color=color_rgb, width=LINE_WIDTH)
            
            plot_item.plot(freqs_rel_khz, mag_db, pen=pen)
            
    def _plot_iq(self, plot_item, detector_data, amplitudes, colormap, cmnorm, pen_color):
        """Plot IQ circles for a detector."""
        from .utils import LINE_WIDTH
        
        for amp in amplitudes:
            amp_data = detector_data.get(amp)
            if not amp_data:
                continue
                
            iq = amp_data.get('iq')
            
            if iq is None or len(iq) == 0:
                continue
            
            # Extract I and Q
            i_vals = np.real(iq)
            q_vals = np.imag(iq)
            
            # Normalize
            mag = np.abs(iq)
            if len(mag) > 0 and np.max(mag) > 0:
                i_vals = i_vals / np.max(mag)
                q_vals = q_vals / np.max(mag)
            
            # Get color
            color_rgba = colormap(cmnorm(amp))
            color_rgb = tuple(int(c * 255) for c in color_rgba[:3])
            
            if len(amplitudes) == 1:
                pen = pg.mkPen(color=pen_color, width=LINE_WIDTH)
            else:
                pen = pg.mkPen(color=color_rgb, width=LINE_WIDTH)
            
            plot_item.plot(i_vals, q_vals, pen=pen)
            
    def _update_histogram_tab(self):
        """Update histogram tab with current amplitude selection."""
        if not self.multisweep_panel:
            return
            
        try:
            data = extract_multisweep_data(self.multisweep_panel)
        except Exception:
            return
            
        if not data or not data.get('fit_params'):
            return
            
        # Get selected amplitude index
        amp_idx = self.amp_combo.currentIndex()
        if amp_idx < 0:
            return
            
        from .aggregate_data_adapter import get_parameter_arrays, filter_failed_fits
        from .utils import TABLEAU10_COLORS
        
        # Extract parameters
        fit_params = data.get('fit_params', {})
        
        fr_values, fr_ids = get_parameter_arrays(fit_params, 'fr', amp_idx)
        qr_values, qr_ids = get_parameter_arrays(fit_params, 'Qr', amp_idx)
        qc_values, qc_ids = get_parameter_arrays(fit_params, 'Qc', amp_idx)
        qi_values, qi_ids = get_parameter_arrays(fit_params, 'Qi', amp_idx)
        
        # Convert to dicts and filter
        fr_valid = filter_failed_fits(dict(zip(fr_ids, fr_values)))
        qr_valid = filter_failed_fits(dict(zip(qr_ids, qr_values)))
        qc_valid = filter_failed_fits(dict(zip(qc_ids, qc_values)))
        qi_valid = filter_failed_fits(dict(zip(qi_ids, qi_values)))
        
        # Update plots
        self._plot_frequency_scatter(fr_valid)
        self._plot_q_histogram(self.qr_plot, qr_valid, "Qr")
        self._plot_q_histogram(self.qc_plot, qc_valid, "Qc")
        self._plot_q_histogram(self.qi_plot, qi_valid, "Qi")
        
    def _plot_frequency_scatter(self, fr_dict):
        """Plot resonance frequency scatter."""
        from .utils import TABLEAU10_COLORS
        
        if not self.freq_plot:
            return
        
        freq_item = self.freq_plot.getPlotItem()
        if not freq_item:
            return
            
        freq_item.clear()
        
        if not fr_dict:
            return
        
        detector_ids = sorted(fr_dict.keys())
        freqs = [fr_dict[det_id] for det_id in detector_ids]
        
        color = TABLEAU10_COLORS[0]
        scatter = pg.ScatterPlotItem(
            x=detector_ids,
            y=freqs,
            size=8,
            pen=pg.mkPen(color, width=1),
            brush=pg.mkBrush(color)
        )
        freq_item.addItem(scatter)
        
        bg_color, pen_color = ("k", "w") if self.dark_mode else ("w", "k")
        total = len(detector_ids)
        title = f"Resonance Frequencies (Yield: {total})"
        freq_item.setTitle(title, color=pen_color)
        
        self.freq_plot.autoRange()
        
    def _plot_q_histogram(self, plot_widget, q_dict, param_name):
        """Plot Q factor histogram with log binning."""
        from .utils import TABLEAU10_COLORS
        
        if not plot_widget:
            return
        
        plot_item = plot_widget.getPlotItem()
        if not plot_item:
            return
            
        plot_item.clear()
        
        if not q_dict:
            return
        
        q_values = np.array(list(q_dict.values()))
        
        if len(q_values) == 0:
            return
        
        q_min = np.min(q_values)
        q_max = np.max(q_values)
        
        if q_min <= 0 or q_max <= 0:
            bins = np.linspace(q_min, q_max, self.nbins + 1)
        else:
            bins = np.logspace(np.log10(q_min), np.log10(q_max), self.nbins + 1)
        
        counts, edges = np.histogram(q_values, bins=bins)
        
        x = (edges[:-1] + edges[1:]) / 2
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
        
        mean_q = np.mean(q_values)
        median_q = np.median(q_values)
        std_q = np.std(q_values)
        
        bg_color, pen_color = ("k", "w") if self.dark_mode else ("w", "k")
        stats_text = (
            f"{param_name} Distribution (N={len(q_values)})\n"
            f"Mean: {mean_q:.2e}\n"
            f"Median: {median_q:.2e}\n"
            f"Std: {std_q:.2e}"
        )
        plot_item.setTitle(stats_text, color=pen_color)
        
        plot_widget.autoRange()
        
    def _populate_amplitude_selector(self):
        """Populate amplitude selector with available amplitudes."""
        if not self.multisweep_panel:
            return
            
        try:
            data = extract_multisweep_data(self.multisweep_panel)
        except Exception:
            return
            
        if not data or not data.get('data'):
            return
        
        # Get all unique amplitudes
        amplitudes_set = set()
        for detector_data in data['data'].values():
            amplitudes_set.update(detector_data.keys())
        
        available_amplitudes = sorted(amplitudes_set)
        
        # Remember current selection
        current_amp = None
        if self.amp_combo.currentIndex() >= 0 and self.amp_combo.count() > 0:
            current_amp = self.amp_combo.currentText()
        
        # Populate combo
        self.amp_combo.blockSignals(True)
        self.amp_combo.clear()
        for amp in available_amplitudes:
            self.amp_combo.addItem(f"{amp:.6f}")
        
        # Restore selection or default to last
        if current_amp:
            idx = self.amp_combo.findText(current_amp)
            if idx >= 0:
                self.amp_combo.setCurrentIndex(idx)
            else:
                self.amp_combo.setCurrentIndex(len(available_amplitudes) - 1)
        else:
            self.amp_combo.setCurrentIndex(len(available_amplitudes) - 1)
            
        self.amp_combo.blockSignals(False)
        
    def _get_all_amplitudes(self, data):
        """Get list of all amplitudes in the data."""
        if not data or not data.get('data'):
            return []
        
        amplitudes = set()
        for detector_data in data['data'].values():
            amplitudes.update(detector_data.keys())
        
        return sorted(amplitudes)
        
    def apply_theme(self, dark_mode: bool):
        """Apply theme and redraw plots."""
        self.dark_mode = dark_mode
        self._full_replot()
