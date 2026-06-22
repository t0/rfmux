"""Panel for displaying multisweep parameter histograms (dockable)."""
import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
import pyqtgraph as pg

# Imports from within the 'periscope' subpackage
from .utils import (

    ScreenshotMixin, find_parent_with_attr, TABLEAU10_COLORS, LINE_WIDTH,
    UnitConverter, AMPLITUDE_COLORMAP_THRESHOLD
)
from .multisweep_grid_helpers import create_amplitude_color_map
from .amplitude_colorbar import AmplitudeColorBar


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
        self.available_sweep_keys = []  # List of (amplitude, direction) tuples
        
        # Histogram cache for performance
        # Structure: {sweep_idx: {fr_dict, qr_dict, ...}} or 'all_sweeps'
        self.histogram_cache = {}
        
        # Global Q-factor ranges for consistent x-axis scaling
        self.global_q_ranges = {'Qr': None, 'Qc': None, 'Qi': None}
        
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
        
        # Amplitude colorbar (shown when inferno colormap is active)
        self.colorbar = AmplitudeColorBar(self)
        main_layout.addWidget(self.colorbar)
        
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
            freq_item.setLabel('bottom', 'Frequency Rank')
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
        

    @staticmethod
    def _color_to_rgb(color):
        """Convert a color (hex string, tuple, list, or numpy array) to an (R, G, B) tuple."""
        if isinstance(color, str) and color.startswith('#'):
            color = color.lstrip('#')
            return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16))
        if isinstance(color, (tuple, list, np.ndarray)) and len(color) >= 3:
            return (int(color[0]), int(color[1]), int(color[2]))
        return (100, 100, 255)  # fallback

    def _style_axes(self, plot_item, pen_color):
        """Apply consistent axis styling."""
        for axis_name in ("left", "bottom", "right", "top"):
            ax = plot_item.getAxis(axis_name)
            if ax:
                ax.setPen(pen_color)
                ax.setTextPen(pen_color)
                
    def _load_and_plot_data(self):
        """Load data directly from the multisweep panel's results."""
        if not self.multisweep_panel:
            return
            
        results = getattr(self.multisweep_panel, 'results', {})
        if not results:
            return
        
        # Collect available (amplitude, direction) pairs and check for fit data
        sweep_keys_set = set()
        has_fit_data = False
        for iter_dict in results.values():
            for entry in iter_dict.values():
                amp = entry.get('sweep_amplitude_normalized')
                direction = entry.get('sweep_direction', 'upward')
                if amp is not None:
                    sweep_keys_set.add((amp, direction))
                if entry.get('fits', {}).get('skewed', {}).get('fit_params') or \
                   entry.get('fits', {}).get('nonlinear', {}).get('nonlinear_fit_params'):
                    has_fit_data = True
        
        if not has_fit_data:
            return
        
        # Sort by amplitude first, then direction (so "downward" < "upward" alphabetically)
        self.available_sweep_keys = sorted(sweep_keys_set, key=lambda x: (x[0], x[1]))
        
        # Invalidate cache on data reload
        self.histogram_cache.clear()
        
        # Populate amplitude selector and plot
        self._populate_amplitude_selector()
        self._update_plots()
        
    def _format_sweep_label(self, amp_value, direction):
        """Format a human-readable label for a sweep.
        
        Produces labels like '-23.45 dBm (Up)' or '152.3 µVpk (Down)'.
        
        Args:
            amp_value: Normalized amplitude value
            direction: Sweep direction string ('upward' or 'downward')
            
        Returns:
            str: Formatted label with power and direction
        """
        # Get unit mode and DAC scale from the multisweep panel
        unit_mode = getattr(self.multisweep_panel, 'unit_mode', 'dbm')
        dac_scales = getattr(self.multisweep_panel, 'dac_scales', {})
        active_module = getattr(self.multisweep_panel, 'active_module_for_dac', None)
        dac_scale = dac_scales.get(active_module) if active_module is not None else None
        
        label = UnitConverter.format_probe_label(amp_value, unit_mode, dac_scale)
        direction_suffix = " (Down)" if direction == "downward" else " (Up)"
        return label + direction_suffix
    
    def _populate_amplitude_selector(self):
        """Populate amplitude selector with available sweep keys."""
        if not self.available_sweep_keys:
            return
        
        # Populate combo box
        self.amp_combo.clear()
        
        # Add "All Sweeps" as first option
        self.amp_combo.addItem("All Sweeps")
        
        # Add individual sweeps with power + direction labels
        for amp_value, direction in self.available_sweep_keys:
            label = self._format_sweep_label(amp_value, direction)
            self.amp_combo.addItem(label)
        
        # Set default to "All Sweeps" (index 0)
        self.amp_combo.setCurrentIndex(0)
            
    def _amplitude_changed(self, index):
        """Handle amplitude selection change."""
        if index >= 0:
            # Index 0 is "All Sweeps", indices 1+ are specific sweeps
            if index == 0:
                self.amplitude_idx = None  # None means show all
            else:
                self.amplitude_idx = index - 1  # Adjust for "All Sweeps" offset
            self._update_plots()
            
    def _bins_changed(self, value):
        """Handle bin count change."""
        self.nbins = value
        # Invalidate cache since bin count changed
        self.histogram_cache.clear()
        self._update_plots()
        
    def _extract_params_for_sweep(self, amplitude, direction):
        """Extract fit parameters for all detectors at a given (amplitude, direction).

        Reads directly from self.multisweep_panel.results (new {iter_idx: {code: entry}}
        format), searching all entries for matching amplitude and direction.

        Args:
            amplitude: The amplitude value to match
            direction: The sweep direction to match ('upward' or 'downward')

        Returns:
            dict: {detector_id: {param_name: value, ...}} for detectors with fits.
        """
        results = self.multisweep_panel.results
        params_by_detector = {}

        for iter_idx, code_dict in results.items():
            for detector_id, entry in code_dict.items():
                if (entry.get('sweep_amplitude_normalized') != amplitude
                        or entry.get('sweep_direction') != direction):
                    continue
                # Skip if we already found a fit for this detector at this amplitude
                if detector_id in params_by_detector:
                    continue
                # Prefer nonlinear fit params, fall back to skewed
                _nl = entry.get('fits', {}).get('nonlinear', {})
                _sk = entry.get('fits', {}).get('skewed', {})
                fit_params = None
                if _nl.get('nonlinear_fit_params'):
                    fit_params = dict(_nl['nonlinear_fit_params'])
                elif _sk.get('fit_params'):
                    fit_params = dict(_sk['fit_params'])
                if fit_params:
                    params_by_detector[detector_id] = fit_params

        return params_by_detector

    @staticmethod
    def _extract_param_values(params_by_detector, param_name):
        """Extract values and IDs for a specific parameter, filtering NaN."""
        values = []
        ids = []
        for det_id, params in params_by_detector.items():
            val = params.get(param_name)
            if val is None or val == 'nan':
                continue
            if isinstance(val, float) and np.isnan(val):
                continue
            values.append(float(val))
            ids.append(det_id)
        return dict(zip(ids, values))

    def _update_plots(self):
        """Update all plots with current settings."""
        if not self.multisweep_panel or not self.multisweep_panel.results:
            return
        
        if self.amplitude_idx is None:
            self._update_plots_all_sweeps()
        else:
            # Single amplitude selected — hide colorbar, use per-plot color
            self.colorbar.hide()
            if self.amplitude_idx >= len(self.available_sweep_keys):
                return
            
            amp, direction = self.available_sweep_keys[self.amplitude_idx]
            
            # Check cache first
            cache_key = self.amplitude_idx
            if cache_key in self.histogram_cache:
                cached_data = self.histogram_cache[cache_key]
                self._plot_frequency_scatter(cached_data['fr_dict'])
                self._plot_q_histogram(self.qr_plot, cached_data['qr_dict'], "Qr")
                self._plot_q_histogram(self.qc_plot, cached_data['qc_dict'], "Qc")
                self._plot_q_histogram(self.qi_plot, cached_data['qi_dict'], "Qi")
                return
            
            # Extract parameters for this specific (amplitude, direction) pair
            params_by_det = self._extract_params_for_sweep(amp, direction)
            
            fr_valid = self._extract_param_values(params_by_det, 'fr')
            qr_valid = self._extract_param_values(params_by_det, 'Qr')
            qc_valid = self._extract_param_values(params_by_det, 'Qc')
            qi_valid = self._extract_param_values(params_by_det, 'Qi')
            
            # Cache
            self.histogram_cache[cache_key] = {
                'fr_dict': fr_valid,
                'qr_dict': qr_valid,
                'qc_dict': qc_valid,
                'qi_dict': qi_valid
            }
            
            # Plot
            self._plot_frequency_scatter(fr_valid)
            self._plot_q_histogram(self.qr_plot, qr_valid, "Qr")
            self._plot_q_histogram(self.qc_plot, qc_valid, "Qc")
            self._plot_q_histogram(self.qi_plot, qi_valid, "Qi")
        
    def _plot_frequency_scatter(self, fr_dict):
        """Plot resonance frequency scatter. fr_dict: {detector_id: freq_hz}.

        The x-axis shows frequency rank (0-based index sorted by ascending frequency)
        rather than raw detector IDs, so that string codes and integer keys both work.
        """
        if not self.freq_plot:
            return
        
        freq_item = self.freq_plot.getPlotItem()
        if not freq_item:
            return
            
        # Clear previous plot
        freq_item.clear()
        
        if not fr_dict:
            return
        
        # Sort detectors by frequency value; use rank as x-axis
        sorted_ids = sorted(fr_dict, key=lambda k: fr_dict[k])
        freqs = [fr_dict[det_id] for det_id in sorted_ids]
        x_rank = list(range(len(sorted_ids)))
        
        # Create scatter plot
        color = TABLEAU10_COLORS[0]
        scatter = pg.ScatterPlotItem(
            x=x_rank,
            y=freqs,
            size=8,
            pen=pg.mkPen(color, width=1),
            brush=pg.mkBrush(color)
        )
        freq_item.addItem(scatter)
        
        # Add yield count to title
        bg_color, pen_color = ("k", "w") if self.dark_mode else ("w", "k")
        total = len(sorted_ids)
        title = f"Resonance Frequencies (Yield: {total})"
        freq_item.setTitle(title, color=pen_color)
        
        # Auto range
        self.freq_plot.autoRange()
        
    def _update_plots_all_sweeps(self):
        """Update plots showing all sweeps (stacked histograms)."""
        if not self.multisweep_panel or not self.multisweep_panel.results:
            return
        
        # Determine colorbar vs legend based on number of unique amplitudes
        unique_amps = sorted(set(ak[0] for ak in self.available_sweep_keys))
        num_amps = len(unique_amps)
        has_downward = any(d == 'downward' for _, d in self.available_sweep_keys)
        
        if num_amps > AMPLITUDE_COLORMAP_THRESHOLD:
            # Get DAC scale and unit mode for colorbar labels
            unit_mode = getattr(self.multisweep_panel, 'unit_mode', 'dbm')
            dac_scales = getattr(self.multisweep_panel, 'dac_scales', {})
            active_module = getattr(self.multisweep_panel, 'active_module_for_dac', None)
            dac_scale = dac_scales.get(active_module) if active_module is not None else None
            self.colorbar.update_range(unique_amps[0], unique_amps[-1],
                                       dac_scale, unit_mode,
                                       self.dark_mode, has_downward)
            self.colorbar.show()
            self._use_legend = False
        else:
            self.colorbar.hide()
            self._use_legend = True
        
        # Check if cached
        cache_key = 'all_sweeps'
        if cache_key in self.histogram_cache:
            cached_data = self.histogram_cache[cache_key]
            self._plot_frequency_scatter_all_sweeps(cached_data['fr_by_sweep'])
            self._plot_stacked_q_histogram(self.qr_plot, cached_data['qr_by_sweep'], "Qr")
            self._plot_stacked_q_histogram(self.qc_plot, cached_data['qc_by_sweep'], "Qc")
            self._plot_stacked_q_histogram(self.qi_plot, cached_data['qi_by_sweep'], "Qi")
            return
        
        # Extract data for all sweeps directly from results_by_detector
        fr_by_sweep = {}
        qr_by_sweep = {}
        qc_by_sweep = {}
        qi_by_sweep = {}
        
        for sweep_idx, (amp, direction) in enumerate(self.available_sweep_keys):
            params_by_det = self._extract_params_for_sweep(amp, direction)
            
            fr_dict = self._extract_param_values(params_by_det, 'fr')
            qr_dict = self._extract_param_values(params_by_det, 'Qr')
            qc_dict = self._extract_param_values(params_by_det, 'Qc')
            qi_dict = self._extract_param_values(params_by_det, 'Qi')
            
            if fr_dict: fr_by_sweep[sweep_idx] = fr_dict
            if qr_dict: qr_by_sweep[sweep_idx] = qr_dict
            if qc_dict: qc_by_sweep[sweep_idx] = qc_dict
            if qi_dict: qi_by_sweep[sweep_idx] = qi_dict
        
        # Compute and store global Q-factor ranges
        self._compute_global_q_ranges(qr_by_sweep, qc_by_sweep, qi_by_sweep)
        
        # Cache the data
        self.histogram_cache[cache_key] = {
            'fr_by_sweep': fr_by_sweep,
            'qr_by_sweep': qr_by_sweep,
            'qc_by_sweep': qc_by_sweep,
            'qi_by_sweep': qi_by_sweep
        }
        
        # Plot
        self._plot_frequency_scatter_all_sweeps(fr_by_sweep)
        self._plot_stacked_q_histogram(self.qr_plot, qr_by_sweep, "Qr")
        self._plot_stacked_q_histogram(self.qc_plot, qc_by_sweep, "Qc")
        self._plot_stacked_q_histogram(self.qi_plot, qi_by_sweep, "Qi")
    
    def _compute_global_q_ranges(self, qr_by_amp, qc_by_amp, qi_by_amp):
        """Compute and store global Q-factor ranges for consistent x-axis scaling.

        Iterates over the three Q-factor dictionaries and stores the (min, max)
        range for each parameter so that all per-amplitude histograms share the
        same x-axis extent.
        """
        for param_name, by_amp in [('Qr', qr_by_amp), ('Qc', qc_by_amp), ('Qi', qi_by_amp)]:
            all_vals = [v for amp_data in by_amp.values() for v in amp_data.values()]
            if all_vals:
                arr = np.array(all_vals)
                vmin, vmax = np.min(arr), np.max(arr)
                if vmin > 0 and vmax > 0:
                    self.global_q_ranges[param_name] = (vmin, vmax)
    
    def _plot_frequency_scatter_all_sweeps(self, fr_by_sweep):
        """Plot resonance frequencies for all sweeps with color coding."""
        if not self.freq_plot:
            return
        
        freq_item = self.freq_plot.getPlotItem()
        if not freq_item:
            return
        
        # Clear previous plot
        freq_item.clear()
        
        if not fr_by_sweep:
            return
        
        # Get amplitude color mapping (extract amp values from sweep keys)
        amp_values = [self.available_sweep_keys[idx][0] for idx in sorted(fr_by_sweep.keys())]
        amplitude_to_color = create_amplitude_color_map(amp_values, self.dark_mode)
        
        # Plot each sweep with its color, using frequency rank as x-axis
        total_count = 0
        for sweep_idx in sorted(fr_by_sweep.keys()):
            fr_dict = fr_by_sweep[sweep_idx]
            if not fr_dict:
                continue
            
            amp_value = self.available_sweep_keys[sweep_idx][0]
            color = amplitude_to_color.get(amp_value, TABLEAU10_COLORS[0])
            
            # Sort by frequency value; use rank as x so string keys work
            sorted_ids = sorted(fr_dict, key=lambda k: fr_dict[k])
            freqs = [fr_dict[det_id] for det_id in sorted_ids]
            x_rank = list(range(len(sorted_ids)))
            
            scatter = pg.ScatterPlotItem(
                x=x_rank,
                y=freqs,
                size=8,
                pen=pg.mkPen(color, width=1),
                brush=pg.mkBrush(color)
            )
            freq_item.addItem(scatter)
            total_count += len(sorted_ids)
        
        # Add title
        bg_color, pen_color = ("k", "w") if self.dark_mode else ("w", "k")
        title = f"Resonance Frequencies - All Sweeps"
        freq_item.setTitle(title, color=pen_color)
        
        # Auto range
        self.freq_plot.autoRange()
    
    def _plot_stacked_q_histogram(self, plot_widget, q_by_sweep, param_name):
        """Plot stacked Q factor histograms for all sweeps."""
        if not plot_widget:
            return
        
        plot_item = plot_widget.getPlotItem()
        if not plot_item:
            return
        
        # Clear previous plot and legend
        plot_item.clear()
        if hasattr(plot_item, 'legend') and plot_item.legend:
            plot_item.legend.scene().removeItem(plot_item.legend)
            plot_item.legend = None
        
        if not q_by_sweep:
            return
        
        # Collect all Q values to determine global bins
        all_q_values = []
        for sweep_idx in q_by_sweep:
            all_q_values.extend(list(q_by_sweep[sweep_idx].values()))
        
        if len(all_q_values) == 0:
            return
        
        all_q_values = np.array(all_q_values)
        q_min = np.min(all_q_values)
        q_max = np.max(all_q_values)
        
        # Create shared bins
        if q_min <= 0 or q_max <= 0:
            bins = np.linspace(q_min, q_max, self.nbins + 1)
        else:
            bins = np.logspace(np.log10(q_min), np.log10(q_max), self.nbins + 1)
        
        # Get bin centers and widths
        x = (bins[:-1] + bins[1:]) / 2
        width = bins[1:] - bins[:-1]
        
        # Get amplitude color mapping (extract amp values from sweep keys)
        amp_values = [self.available_sweep_keys[idx][0] for idx in sorted(q_by_sweep.keys())]
        amplitude_to_color = create_amplitude_color_map(amp_values, self.dark_mode)
        
        # Calculate opacity gradient (back to front)
        num_sweeps = len(q_by_sweep)
        alphas = np.linspace(0.4, 0.8, num_sweeps)
        
        # Plot from highest amplitude to lowest (back to front for stacking)
        sorted_sweep_indices = sorted(q_by_sweep.keys(), reverse=True)
        
        # Only add legend when colorbar is not active
        bg_color, pen_color = ("k", "w") if self.dark_mode else ("w", "k")
        use_legend = getattr(self, '_use_legend', True)
        legend = None
        if use_legend:
            legend_color = '#CCCCCC' if self.dark_mode else '#333333'
            legend = plot_item.addLegend(offset=(-10, 10), labelTextColor=legend_color)
        
        for i, sweep_idx in enumerate(sorted_sweep_indices):
            q_dict = q_by_sweep[sweep_idx]
            if not q_dict:
                continue
            
            q_values = np.array(list(q_dict.values()))
            amp_value, direction = self.available_sweep_keys[sweep_idx]
            
            # Compute histogram with shared bins
            counts, _ = np.histogram(q_values, bins=bins)
            
            # Get color and alpha
            color = amplitude_to_color.get(amp_value, TABLEAU10_COLORS[1])
            alpha = alphas[num_sweeps - 1 - i]  # Reverse alpha for front-to-back
            
            # Convert color to rgba with alpha
            rgb = self._color_to_rgb(color)
            rgba = (*rgb, int(alpha * 255))
            
            # Create bar graph with transparency
            bar_graph = pg.BarGraphItem(
                x=x,
                height=counts,
                width=width,
                brush=pg.mkBrush(*rgba),
                pen=pg.mkPen(color, width=1)
            )
            plot_item.addItem(bar_graph)
            
            # Add to legend with power + direction label (only when legend is active)
            if legend:
                legend.addItem(bar_graph, self._format_sweep_label(amp_value, direction))
        
        # Add title
        bg_color, pen_color = ("k", "w") if self.dark_mode else ("w", "k")
        total_count = sum(len(q_by_sweep[idx]) for idx in q_by_sweep)
        title = f"{param_name} Distribution - All Sweeps (N={total_count})"
        plot_item.setTitle(title, color=pen_color)
        
        # Auto range
        plot_widget.autoRange()
    
    def _plot_q_histogram(self, plot_widget, q_dict, param_name):
        """Plot Q factor histogram with log binning and consistent transparency."""
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
        

        # Use global Q ranges if available for consistent x-axis
        global_range = self.global_q_ranges.get(param_name)
        if global_range:
            q_min, q_max = global_range
        else:
            q_min = np.min(q_values)
            q_max = np.max(q_values)
        
        # Create bins
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
        
        # Calculate consistent transparency based on sweep index
        if self.amplitude_idx is not None and len(self.available_sweep_keys) > 0:
            num_sweeps = len(self.available_sweep_keys)
            alphas = np.linspace(0.4, 0.8, num_sweeps)
            # Reverse so that higher amplitudes (later in sorted list) get higher alpha
            alpha = alphas[num_sweeps - 1 - self.amplitude_idx]
            
            # Get amplitude color mapping for consistency
            amp_values = [ak[0] for ak in self.available_sweep_keys]
            amplitude_to_color = create_amplitude_color_map(amp_values, self.dark_mode)
            amp_value = self.available_sweep_keys[self.amplitude_idx][0]
            color = amplitude_to_color.get(amp_value, TABLEAU10_COLORS[1])
            
            # Convert color to rgba with alpha
            rgb = self._color_to_rgb(color)
            rgba = (*rgb, int(alpha * 255))
        else:
            # Fallback for single color
            color = TABLEAU10_COLORS[1]
            rgb = self._color_to_rgb(color)
            rgba = (*rgb, 200)
        
        bar_graph = pg.BarGraphItem(
            x=x,
            height=counts,
            width=width,

            brush=pg.mkBrush(*rgba),
            pen=pg.mkPen(color, width=1)
        )
        plot_item.addItem(bar_graph)
        
        # Apply fixed x-axis range if available
        if global_range:
            plot_item.setXRange(q_min, q_max, padding=0.02)
        
        # Calculate statistics
        mean_q = np.mean(q_values)
        median_q = np.median(q_values)
        std_q = np.std(q_values)

        # Clean title with just parameter name and count
        bg_color, pen_color = ("k", "w") if self.dark_mode else ("w", "k")
        plot_item.setTitle(f"{param_name} Distribution (N={len(q_values)})", color=pen_color)
        
        # Add statistics as an in-plot text box (upper-right corner)
        legend_color = '#CCCCCC' if self.dark_mode else '#333333'
        stats_text = (
            f"Mean: {mean_q:.2e}\n"
            f"Median: {median_q:.2e}\n"
            f"Std: {std_q:.2e}"
        )
        text_item = pg.TextItem(text=stats_text, color=legend_color, anchor=(1, 0))
        plot_item.addItem(text_item, ignoreBounds=True)
        # Position in top-right of the view after auto-range
        vb = plot_item.getViewBox()
        if vb:
            view_range = vb.viewRange()
            text_item.setPos(view_range[0][1], view_range[1][1])
        
        # Auto range for y-axis only
        if not global_range:
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
