"""Panel for displaying multisweep analysis results (dockable)."""
import datetime
import pickle
import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg
import asyncio
import traceback
import time

# Imports from within the 'periscope' subpackage
from .utils import (
    LINE_WIDTH, UnitConverter, ClickableViewBox, QtWidgets, QtCore, pg,
    TABLEAU10_COLORS, COLORMAP_CHOICES, AMPLITUDE_COLORMAP_THRESHOLD, UPWARD_SWEEP_STYLE, DOWNWARD_SWEEP_STYLE,
    ScreenshotMixin, mag_axis_label
)
from .detector_digest_panel import DetectorDigestPanel
from .noise_spectrum_panel import NoiseSpectrumPanel
from .noise_spectrum_dialog import NoiseSpectrumDialog
from .parameter_histograms_panel import ParameterHistogramsPanel
from .amplitude_colorbar import AmplitudeColorBar
from .multisweep_grid_helpers import create_amplitude_color_map
from rfmux.core.transferfunctions import PFB_SAMPLING_FREQ
# from rfmux.algorithms.measurement import py_get_samples


class MultisweepPanel(QtWidgets.QWidget, ScreenshotMixin):
    """
    A dockable panel for displaying and interacting with multisweep analysis results.

    This panel visualizes S21 magnitude and phase data for multiple resonances
    across various probe amplitudes. It provides controls for data export,
    re-running sweeps, unit conversion, normalization, and plot interaction.
    Can be docked, floated, or tabbed within the main Periscope window.
    """
    
    # Signal emitted when bias_kids algorithm completes with df_calibration data
    df_calibration_ready = pyqtSignal(int, dict)  # module, {detector_idx: df_calibration}
    
    # Signal for session auto-export
    data_ready = pyqtSignal(str, str, dict)  # type, identifier, data
    def __init__(self, parent=None, target_module=None, initial_params=None, dac_scales=None, dark_mode=False, loaded_bias=False, is_loaded_data=False):
        """
        Initializes the MultisweepWindow.

        Args:
            parent: The parent widget.
            target_module (int, optional): The specific hardware module this window is for.
            initial_params (dict, optional): Initial parameters used for the multisweep.
                                             Defaults to an empty dict.
            dac_scales (dict, optional): DAC scaling factors for unit conversion.
                                         Defaults to an empty dict.
            dark_mode (bool, optional): Whether to use dark mode for plots.
                                         Defaults to False.
            loaded_bias (bool, optional): Whether bias/noise data is available from loaded file.
            is_loaded_data (bool, optional): Whether this panel is from loaded data (for naming).
        """
        super().__init__(parent)
        self.target_module = target_module
        self.initial_params = initial_params or {}  # Store initial parameters for potential re-runs
        self.dac_scales = dac_scales or {}          # DAC scales for unit conversions
        self.dark_mode = dark_mode                 # Store dark mode setting
        self.bias_data_avail = loaded_bias
        self.is_loaded_data = is_loaded_data       # Track if this is from loaded data
        self.samples_taken = False
        self.noise_data = {}
        self.spectrum_noise_data = {}

        self.debug_noise_data = {}
        self.debug_phase_data = []

        
        # Track open detector digest and noise spectrum windows to prevent garbage collection
        self.detector_digest_windows = []
        self.noise_spectrum_windows = []
        self.digest_window_count = 0  # Counter for naming digest tabs
        self.noise_panel_count = 0    # Counter for naming noise tabs
        
        # Amps used for the last configured/completed run, to compare if settings changed.
        # Use amp_arrays (new format) if present, falling back to amps (old format).
        _init_amp_arrays = self.initial_params.get('amp_arrays', [])
        if _init_amp_arrays:
            self.current_run_amps: list[float] = [arr[0] for arr in _init_amp_arrays if arr]
        else:
            self.current_run_amps: list[float] = list(self.initial_params.get('amps', []))
        # probe_amplitudes is used for progress display, should reflect current_run_amps
        self.probe_amplitudes = list(self.current_run_amps) # Ensure it's a copy and reflects current run

        self.setWindowTitle(f"Multisweep Results - Module {self.target_module}")


        # Data storage and state — detector-based format:
        # {detector_code: {iteration_index: {all_detector_data_fields + amplitude, direction, iteration metadata}}}
        self.results_by_detector = {}
        # Lightweight resonator registry {code: {bias_frequency, bias_amplitude, channel_number}}
        self.res_info_dict: dict = {}
        self.current_amplitude_being_processed = None # Tracks the amplitude currently being processed

        self.current_iteration_being_processed = None # Tracks the current iteration
        self.unit_mode = "dbm"  # Current unit for magnitude display ("counts", "dbm", "volts")
        self.normalize_traces = True  # Flag to normalize trace plots (magnitude and phase)
        self.zoom_box_mode = True  # Flag for enabling/disabling pyqtgraph's zoom box
        
        # Intermediate update data storage
        self._current_intermediate_data = {}  # Stores intermediate data during sweep
        self._intermediate_curves_mag = {}   # Stores {cf: PlotDataItem} for intermediate magnitude curves
        self._intermediate_curves_phase = {} # Stores {cf: PlotDataItem} for intermediate phase curves

        # Plot objects and related attributes
        self.combined_mag_plot = None
        self.combined_phase_plot = None
        self.mag_legend = None
        self.phase_legend = None
        self.curves_mag = {}  # Stores {amplitude: {cf: PlotDataItem_mag}}
        self.curves_phase = {} # Stores {amplitude: {cf: PlotDataItem_phase}}
        
        # Module context for DAC scale lookup (can be different from target_module if needed)
        self.active_module_for_dac = self.target_module

        # Center frequency line display
        self.show_cf_lines_cb = None # Checkbox for toggling CF lines
        self.cf_lines_mag = {}  # Stores {amplitude: [InfiniteLine_mag]}
        self.cf_lines_phase = {} # Stores {amplitude: [InfiniteLine_phase]}
        
        # Bias KIDs output storage
        self.bias_kids_output = None  # Stores the output from bias_kids algorithm
        self.nco_frequency_hz = None  # NCO frequency used when biasing (stored for export)

        # Initialize batch tracking for sweep tabs (before _setup_ui)
        self.current_batch = 0

        self.batch_size = 8
        
        # Storage for sweep grid plots - cached to avoid recreating widgets
        self.mag_sweep_plots_cache = []  # List of plot widgets for magnitude tab
        self.iq_sweep_plots_cache = []   # List of plot widgets for IQ tab
        
        # Histogram panel (created lazily in _setup_plot_area)
        self.histogram_panel = None
        self.histograms_generated = False  # Track if histograms have been generated

        self._setup_ui()
        
        # Set reasonable minimum size but allow flexible sizing
        self.setMinimumSize(600, 400)
        # Preferred size policy - adapt to dock size without forcing window resize
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Preferred
        )

    def _setup_ui(self):
        """Sets up the main UI layout, toolbar, and plot area."""
        # Main layout for the panel (no central widget for QWidget)
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self._setup_toolbar(main_layout)
        self._setup_progress_bar(main_layout)
        self._setup_plot_area(main_layout)


    def _setup_toolbar(self, layout):
        """Creates and configures the toolbar with controls (using QWidget instead of QToolBar)."""
        # Use QWidget container instead of QToolBar for QWidget compatibility
        toolbar = QtWidgets.QWidget()
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)

        # Export Data Button
        self.export_btn = QtWidgets.QPushButton("💾")
        self.export_btn.setToolTip("Export data")
        self.export_btn.clicked.connect(self._export_data)
        toolbar_layout.addWidget(self.export_btn)
        
        # Re-run Multisweep Button
        self.rerun_btn = QtWidgets.QPushButton("Re-run Multisweep")
        self.rerun_btn.clicked.connect(self._rerun_multisweep)
        toolbar_layout.addWidget(self.rerun_btn)
        
        # Bias KIDs Button
        self.bias_kids_btn = QtWidgets.QPushButton("Bias KIDs")
        self.bias_kids_btn.clicked.connect(self._bias_kids)
        self.bias_kids_btn.setToolTip("Bias detectors at optimal operating points based on multisweep results")
        toolbar_layout.addWidget(self.bias_kids_btn)

        self.noise_spectrum_btn = QtWidgets.QPushButton("Get Noise Spectrum")
        if self.bias_data_avail:
            self.noise_spectrum_btn.setEnabled(True)
        else:
            self.noise_spectrum_btn.setEnabled(False)
        self.noise_spectrum_btn.setToolTip("Open a dialog to configure and get the noise spectrum, will only work if KIDS is biased.")
        self.noise_spectrum_btn.clicked.connect(self._open_noise_spectrum_dialog)
        toolbar_layout.addWidget(self.noise_spectrum_btn)
        
        # Spacer to push subsequent items to the right
        toolbar_layout.addStretch(1)
        
        # Batch navigation controls (for sweep tabs)
        self.batch_label = QtWidgets.QLabel("Batch:")
        toolbar_layout.addWidget(self.batch_label)
        
        self.prev_batch_btn = QtWidgets.QPushButton("◀")
        self.prev_batch_btn.setToolTip("Previous batch")

        self.prev_batch_btn.setMaximumWidth(30)  # Shrink to 1/3 width

        self.prev_batch_btn.clicked.connect(self._prev_batch)
        toolbar_layout.addWidget(self.prev_batch_btn)
        
        self.batch_info_label = QtWidgets.QLabel("1 of 1")
        self.batch_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.batch_info_label.setMinimumWidth(40)

        toolbar_layout.addWidget(self.batch_info_label)
        
        self.next_batch_btn = QtWidgets.QPushButton("▶")
        self.next_batch_btn.setToolTip("Next batch")

        self.next_batch_btn.setMaximumWidth(30)  # Shrink to 1/3 width
        self.next_batch_btn.clicked.connect(self._next_batch)
        toolbar_layout.addWidget(self.next_batch_btn)
        
        self.batch_size_label = QtWidgets.QLabel("Subplots:")
        toolbar_layout.addWidget(self.batch_size_label)
        
        self.batch_size_spin = QtWidgets.QSpinBox()
        self.batch_size_spin.setRange(1, 200)
        self.batch_size_spin.setValue(self.batch_size)
        self.batch_size_spin.setSingleStep(1)
        self.batch_size_spin.setToolTip("Detectors per batch (press Update to apply)")
        # Note: No longer connects to immediate redraw
        toolbar_layout.addWidget(self.batch_size_spin)
        
        self.batch_update_btn = QtWidgets.QPushButton("Update")
        self.batch_update_btn.setToolTip("Apply new batch size and regenerate plots")
        self.batch_update_btn.clicked.connect(self._apply_batch_size)
        toolbar_layout.addWidget(self.batch_update_btn)

        # Normalization Checkbox
        self.normalize_checkbox = QtWidgets.QCheckBox("Normalize Traces")
        self.normalize_checkbox.setChecked(self.normalize_traces)
        self.normalize_checkbox.toggled.connect(self._toggle_trace_normalization)
        toolbar_layout.addWidget(self.normalize_checkbox)

        # Show Center Frequencies Checkbox
        self.show_cf_lines_cb = QtWidgets.QCheckBox("Show Center Frequencies")
        self.show_cf_lines_cb.setChecked(False) # Default to off
        self.show_cf_lines_cb.toggled.connect(self._toggle_cf_lines_visibility)
        toolbar_layout.addWidget(self.show_cf_lines_cb)

        self._setup_unit_controls(toolbar_layout)
        self._setup_zoom_box_control(toolbar_layout)

        # Screenshot button
        screenshot_btn = QtWidgets.QPushButton("📷")
        screenshot_btn.setToolTip("Export a screenshot of this panel to the session folder (or choose location)")
        screenshot_btn.clicked.connect(self._export_screenshot)
        toolbar_layout.addWidget(screenshot_btn)

        layout.addWidget(toolbar)

    def _setup_unit_controls(self, toolbar_layout):
        """Sets up radio buttons for selecting magnitude units."""
        unit_group = QtWidgets.QWidget() # Group for unit radio buttons
        unit_layout = QtWidgets.QHBoxLayout(unit_group)
        unit_layout.setContentsMargins(0, 0, 0, 0) # Compact layout
        unit_layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.rb_counts = QtWidgets.QRadioButton("Counts")
        self.rb_dbm = QtWidgets.QRadioButton("dBm")
        self.rb_volts = QtWidgets.QRadioButton("Volts")
        self.rb_dbm.setChecked(True) # Default to dBm
        
        unit_layout.addWidget(QtWidgets.QLabel("Units:"))
        unit_layout.addWidget(self.rb_counts)
        unit_layout.addWidget(self.rb_dbm)
        unit_layout.addWidget(self.rb_volts)
        
        # Connect signals for unit changes
        self.rb_counts.toggled.connect(lambda: self._update_unit_mode("counts"))
        self.rb_dbm.toggled.connect(lambda: self._update_unit_mode("dbm"))
        self.rb_volts.toggled.connect(lambda: self._update_unit_mode("volts"))
        
        unit_group.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        toolbar_layout.addWidget(unit_group)

    def _setup_zoom_box_control(self, toolbar_layout):
        """Sets up the checkbox to toggle zoom box mode for plots."""
        self.zoom_box_cb = QtWidgets.QCheckBox("Zoom Box Mode")
        self.zoom_box_cb.setChecked(self.zoom_box_mode)
        self.zoom_box_cb.toggled.connect(self._toggle_zoom_box_mode)
        toolbar_layout.addWidget(self.zoom_box_cb)

    def _setup_plot_area(self, layout):
        """Sets up the tabbed plot area with aggregate and combined views."""
        # Create tab widget
        self.plot_tabs = QtWidgets.QTabWidget()
        self.plot_tabs.currentChanged.connect(self._on_plot_tab_changed)
        
        # Tab 0: Magnitude Sweeps (per-detector grid)

        self.mag_sweeps_tab, self.mag_sweeps_grid, self.mag_colorbar = self._create_sweep_tab()
        self.plot_tabs.addTab(self.mag_sweeps_tab, "Magnitude Sweeps")
        
        # Tab 1: IQ Circles (per-detector grid)
        self.iq_sweeps_tab, self.iq_sweeps_grid, self.iq_colorbar = self._create_sweep_tab()
        self.plot_tabs.addTab(self.iq_sweeps_tab, "IQ Circles")
        
        # Tab 2: Combined Plots (original combined view)
        self.combined_tab = self._create_combined_tab()
        self.plot_tabs.addTab(self.combined_tab, "Combined Plots")
        
        # Tab 3: Histograms (parameter distributions)
        self.histogram_tab = self._create_histogram_tab()
        self.plot_tabs.addTab(self.histogram_tab, "Histograms")
        

        # Tab 4: Detector Digest (single-detector detail view)
        self.digest_tab = self._create_digest_tab()
        self.plot_tabs.addTab(self.digest_tab, "Detector Digest")
        
        # Set default tab to Magnitude Sweeps
        self.plot_tabs.setCurrentIndex(0)
        
        layout.addWidget(self.plot_tabs)
        
    def _create_sweep_tab(self):

        """Create a tab for sweep plots (magnitude or IQ). Returns (tab, grid_layout, colorbar)."""
        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.setContentsMargins(5, 5, 5, 5)
        
        # Amplitude colorbar (shown for >5 sweeps, hidden otherwise)
        colorbar = AmplitudeColorBar(tab)
        tab_layout.addWidget(colorbar)
        
        # Scroll area for plots
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        
        # Container for grid
        container = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(container)
        grid.setSpacing(10)
        
        scroll.setWidget(container)
        tab_layout.addWidget(scroll)
        
        return tab, grid, colorbar
        
    def _create_combined_tab(self):
        """Create the combined plots tab (original magnitude + phase plots)."""
        tab = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(tab)
        plot_layout.setContentsMargins(5, 5, 5, 5)

        # Amplitude colorbar (shown for >5 sweeps)
        self.combined_colorbar = AmplitudeColorBar(tab)
        plot_layout.addWidget(self.combined_colorbar)

        # Magnitude Plot
        vb_mag = ClickableViewBox()
        vb_mag.parent_window = self
        self.combined_mag_plot = pg.PlotWidget(viewBox=vb_mag)
        bg_color, pen_color = ("k", "w") if self.dark_mode else ("w", "k")
        plot_item_mag = self.combined_mag_plot.getPlotItem()
        if plot_item_mag:
            plot_item_mag.setTitle("Combined S21 Magnitude (All Resonances)", color=pen_color)
            plot_item_mag.setLabel('bottom', 'Frequency', units='Hz')
            plot_item_mag.showGrid(x=True, y=True, alpha=0.3)
            legend_color = '#CCCCCC' if self.dark_mode else '#333333'
            self.mag_legend = plot_item_mag.addLegend(offset=(10,-50),labelTextColor=legend_color)
        self._update_mag_plot_label()
        plot_layout.addWidget(self.combined_mag_plot)

        # Phase Plot
        vb_phase = ClickableViewBox()
        vb_phase.parent_window = self
        self.combined_phase_plot = pg.PlotWidget(viewBox=vb_phase)
        plot_item_phase = self.combined_phase_plot.getPlotItem()
        if plot_item_phase:
            plot_item_phase.setTitle("Combined S21 Phase (All Resonances)", color=pen_color)
            plot_item_phase.setLabel('bottom', 'Frequency', units='Hz')
            plot_item_phase.setLabel('left', 'Phase', units='deg')
            plot_item_phase.showGrid(x=True, y=True, alpha=0.3)
            legend_color = '#CCCCCC' if self.dark_mode else '#333333'
            self.phase_legend = plot_item_phase.addLegend(offset=(10,-50),labelTextColor=legend_color)
        plot_layout.addWidget(self.combined_phase_plot)
        
        # Link X-axes for synchronized zooming/panning
        if self.combined_phase_plot and self.combined_mag_plot:
            self.combined_phase_plot.setXLink(self.combined_mag_plot)
        self._apply_zoom_box_mode()
        
        # Apply theme
        if self.combined_mag_plot:
            self.combined_mag_plot.setBackground(bg_color)
            plot_item_mag_for_axes = self.combined_mag_plot.getPlotItem()
            if plot_item_mag_for_axes:
                for axis_name in ("left", "bottom", "right", "top"):
                    ax = plot_item_mag_for_axes.getAxis(axis_name)
                    if ax:
                        ax.setPen(pen_color)
                        ax.setTextPen(pen_color)
                    
        if self.combined_phase_plot:
            self.combined_phase_plot.setBackground(bg_color)
            plot_item_phase_for_axes = self.combined_phase_plot.getPlotItem()
            if plot_item_phase_for_axes:
                for axis_name in ("left", "bottom", "right", "top"):
                    ax = plot_item_phase_for_axes.getAxis(axis_name)
                    if ax:
                        ax.setPen(pen_color)
                        ax.setTextPen(pen_color)

        # Connect double click handlers
        if self.combined_mag_plot:
            view_box_mag = self.combined_mag_plot.getViewBox()
            if isinstance(view_box_mag, ClickableViewBox):
                view_box_mag.doubleClickedEvent.connect(self._handle_multisweep_plot_double_click)
                
        if self.combined_phase_plot:
            view_box_phase = self.combined_phase_plot.getViewBox()
            if isinstance(view_box_phase, ClickableViewBox):
                view_box_phase.doubleClickedEvent.connect(self._handle_multisweep_plot_double_click)
        
        return tab
    
    def _create_histogram_tab(self):
        """Create the histograms tab containing the ParameterHistogramsPanel."""
        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create a placeholder - we'll create the actual panel when data is available
        placeholder = QtWidgets.QLabel("Histogram plots will appear here when multisweep data is available.")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: gray; font-style: italic;")
        tab_layout.addWidget(placeholder)
        
        return tab
    

    def _create_digest_tab(self):
        """Create the detector digest tab (single-detector detail view, lazily populated)."""
        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        
        # Placeholder until data is available and a detector is selected
        self._digest_placeholder = QtWidgets.QLabel(
            "Detector digest will appear here when multisweep data is available.\n"
            "Double-click a resonance in the Combined or grid plots to view its digest."
        )
        self._digest_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._digest_placeholder.setStyleSheet("color: gray; font-style: italic;")
        tab_layout.addWidget(self._digest_placeholder)
        
        # The actual DetectorDigestPanel (created lazily)
        self.digest_panel = None
        
        return tab
    
    def _generate_histograms(self):
        """
        Generate histogram plots once when multisweep data is complete.
        This is called from all_sweeps_completed() to create the plots when fit data is ready.
        """
        if not self.results_by_detector:
            return
        
        # Check if we have any fit data
        has_fit_data = False
        for amp_dir_dict in self.results_by_detector.values():
            for det_data in amp_dir_dict.values():
                if 'fit_params' in det_data or 'nonlinear_fit_params' in det_data:
                    has_fit_data = True
                    break
            if has_fit_data:
                break
        
        if not has_fit_data:
            print("Note: No fit data available for histogram generation")
            return
        
        # Create the histogram panel if it doesn't exist
        if self.histogram_panel is None:
            if hasattr(self, 'histogram_tab') and self.histogram_tab:
                # Clear placeholder
                layout = self.histogram_tab.layout()
                if layout:
                    while layout.count():
                        item = layout.takeAt(0)
                        if item.widget():
                            item.widget().deleteLater()
                    
                    # Create the actual histogram panel with data
                    self.histogram_panel = ParameterHistogramsPanel(
                        parent=self.histogram_tab,
                        multisweep_panel=self,
                        amplitude_idx=None,  # Will use last/highest amplitude by default
                        nbins=30,
                        dark_mode=self.dark_mode
                    )
                    layout.addWidget(self.histogram_panel)
        else:
            # Panel exists, just reload data (for re-run scenario)
            if hasattr(self.histogram_panel, '_load_and_plot_data'):
                self.histogram_panel._load_and_plot_data()
    
    def _ensure_histogram_panel(self):
        """Ensure the histogram panel exists - no longer regenerates plots on tab open."""
        # If histograms haven't been generated yet, just return
        # They will be generated when all_sweeps_completed() is called
        if not self.histograms_generated:
            return
        
        # If we get here and panel doesn't exist but should, create it
        # This handles edge cases like theme changes
        if self.histogram_panel is None and self.results_by_detector:
            self._generate_histograms()


    def _on_plot_tab_changed(self, index):
        """Handle plot tab changes - show/hide batch controls appropriately."""
        # Batch controls visible for sweep tabs (0, 1), hidden for combined tab (2)
        is_sweep_tab = index in (0, 1)
        
        self.batch_label.setVisible(is_sweep_tab)
        self.prev_batch_btn.setVisible(is_sweep_tab)
        self.batch_info_label.setVisible(is_sweep_tab)
        self.next_batch_btn.setVisible(is_sweep_tab)

        self.batch_size_label.setVisible(is_sweep_tab)
        self.batch_size_spin.setVisible(is_sweep_tab)
        self.batch_update_btn.setVisible(is_sweep_tab)
        
        # Redraw the active tab's plots if we have data
        if self.results_by_detector:
            self._redraw_plots()
    
    def _apply_batch_size(self):
        """Apply the batch size from the spin box and regenerate plots."""
        new_batch_size = self.batch_size_spin.value()
        if new_batch_size != self.batch_size:
            self.batch_size = new_batch_size
            self.current_batch = 0
            self._redraw_plots()
    
    def _prev_batch(self):
        """Show previous batch."""
        if self.current_batch > 0:
            self.current_batch -= 1
            self._redraw_plots()
    
    def _next_batch(self):
        """Show next batch."""
        # Get current tab to determine which data to use
        current_tab_idx = self.plot_tabs.currentIndex()
        if current_tab_idx not in (0, 1):  # Only sweep tabs have batches
            return
            
        if not self.results_by_detector:
            return
        
        num_detectors = len(self.results_by_detector)
        total_batches = max(1, (num_detectors + self.batch_size - 1) // self.batch_size)
        
        if self.current_batch < total_batches - 1:
            self.current_batch += 1
            self._redraw_plots()

    def _toggle_trace_normalization(self, checked):
        """
        Slot for the 'Normalize Traces' checkbox.
        Updates normalization state for both magnitude and phase, and redraws plots.
        """
        self.normalize_traces = checked
        self._update_mag_plot_label() # Y-axis label might change
        self._redraw_plots()

    def _update_unit_mode(self, mode):
        """
        Slot for unit selection radio buttons.
        Updates unit mode and redraws plots if the mode changed.
        """
        if self.unit_mode != mode:
            self.unit_mode = mode
            self._update_mag_plot_label() # Y-axis label will change
            self._redraw_plots()
            
    def _update_mag_plot_label(self):
        """Updates the Y-axis label of the magnitude plot based on current unit and normalization settings."""
        if not self.combined_mag_plot:
            return
        label, units = mag_axis_label(self.unit_mode, self.normalize_traces)
        self.combined_mag_plot.setLabel('left', label, units=units)

    def _toggle_zoom_box_mode(self, enable):
        """
        Slot for the 'Zoom Box Mode' checkbox.
        Updates zoom box mode state and applies it to plots.
        """
        self.zoom_box_mode = enable
        self._apply_zoom_box_mode()

    def _apply_zoom_box_mode(self):
        """Applies the current zoom_box_mode state to both magnitude and phase plot viewboxes."""
        if self.combined_mag_plot and isinstance(self.combined_mag_plot.getViewBox(), ClickableViewBox):
            self.combined_mag_plot.getViewBox().enableZoomBoxMode(self.zoom_box_mode)
        if self.combined_phase_plot and isinstance(self.combined_phase_plot.getViewBox(), ClickableViewBox):
            self.combined_phase_plot.getViewBox().enableZoomBoxMode(self.zoom_box_mode)

    def _setup_progress_bar(self, layout):
        """Set up progress bar in a separate group, similar to NetworkAnalysisWindow."""
        self.progress_group = QtWidgets.QGroupBox("Analysis Progress")
        progress_layout = QtWidgets.QVBoxLayout(self.progress_group)
        
        # Main progress layout
        hlayout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(f"Module {self.target_module}:")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        hlayout.addWidget(label)
        hlayout.addWidget(self.progress_bar)
        
        progress_layout.addLayout(hlayout)

        # Current amplitude label below progress bar
        self.current_amp_label = QtWidgets.QLabel()
        num_amplitudes = len(self.probe_amplitudes)
        
        # Calculate total iterations based on sweep direction
        sweep_direction = self.initial_params.get('sweep_direction', 'upward')
        self.total_iterations = num_amplitudes * (2 if sweep_direction == "both" else 1)
        
        if num_amplitudes > 0:
            # Initial message showing what's about to happen
            # When sweep_direction is "both", MultisweepTask does upward first
            # Normalize sweep_direction to handle potential case or whitespace issues
            sweep_direction_norm = sweep_direction.lower().strip() if sweep_direction else ""
            
            if sweep_direction_norm == "downward":
                direction_text = "Down"
            elif sweep_direction_norm == "upward" or sweep_direction_norm == "both":
                direction_text = "Up"
            else:
                # Fallback for unexpected sweep_direction values
                direction_text = "Unknown"
                print(f"WARNING: Unexpected sweep_direction value: '{sweep_direction}'")
                
            self.current_amp_label.setText(f"Iteration 1/{self.total_iterations} ({direction_text})")
        else:
            self.current_amp_label.setText("No sweeps defined. (Waiting...)")
        self.current_amp_label.setAlignment(Qt.AlignmentFlag.AlignCenter) # Center the text
        progress_layout.addWidget(self.current_amp_label)
        
        layout.addWidget(self.progress_group)

    def _hide_progress_bars(self):
        """Hide the entire Analysis Progress group."""
        if self.progress_group:
            self.progress_group.hide()

    def update_progress(self, module, progress_percentage):
        """
        Updates the progress bar if the update is for the target module.

        Args:
            module (int): The module reporting progress.
            progress_percentage (float): The progress percentage (0-100).
        """
        if module == self.target_module:
            self.progress_bar.setValue(int(progress_percentage))
            # Show progress group if it was hidden
            if hasattr(self, 'progress_group') and not self.progress_group.isVisible():
                self.progress_group.setVisible(True)

    def handle_starting_iteration(self, module: int, iteration: int, amplitude: float, direction: str):
        """
        Handler for the starting_iteration signal. Updates the status bar at the start of a sweep.
        
        Args:
            module (int): The module reporting the start.
            iteration (int): The iteration index (0-based).
            amplitude (float): The probe amplitude for this iteration.
            direction (str): The sweep direction ("upward" or "downward").
        """
        if module != self.target_module: return
        
        # Convert direction to user-friendly format
        direction_text = "Down" if direction.lower().strip() == "downward" else "Up"
        
        # Convert to 1-based for display
        current_display_iteration = iteration + 1
        
        # Make sure total_iterations is at least as large as the current iteration
        self.total_iterations = max(self.total_iterations, current_display_iteration)
        
        # Set the status message BEFORE the sweep starts
        status_message = f"Iteration {current_display_iteration}/{self.total_iterations} ({direction_text})"
        self.current_amp_label.setText(status_message)

    def handle_fitting_progress(self, module: int, status_message: str):
        """
        Handler for the fitting_progress signal. Updates the status bar with fitting progress.
        
        Args:
            module (int): The module reporting fitting progress.
            status_message (str): The fitting status message.
        """
        if module != self.target_module: return
        
        # Get the current status base (iteration info)
        current_text = self.current_amp_label.text()
        
        # Split the text to separate iteration info from any fitting status
        if " - " in current_text:
            # Keep only the iteration part
            base_text = current_text.split(" - ")[0]
        else:
            base_text = current_text
        
        # Extract just the fitting status part from the message
        if "Fitting in progress: " in status_message:
            fitting_status = status_message.replace("Fitting in progress: ", "")
            updated_text = f"{base_text} - {fitting_status}"
            self.current_amp_label.setText(updated_text)
        elif status_message == "Fitting Completed":
            # When fitting is completed, just show the base text
            self.current_amp_label.setText(base_text)
        
    def update_data(self, module: int, iteration: int, amplitude: float, direction: str,
                    multisweep_data_dict: dict, res_info_dict: dict):
        """
        Receives final data for a completed iteration of a multisweep for the target module.
        Stores the sweep data dict for plotting and updates the resonator registry.

        Args:
            module (int): The module reporting data.
            iteration (int): The current iteration index.
            amplitude (float): The probe amplitude for which data is provided.
            direction (str): The sweep direction ("upward" or "downward").
            multisweep_data_dict (dict): Sweep data keyed by resonator code.
            res_info_dict (dict): Resonator registry keyed by code
                                  ({code: {bias_frequency, bias_amplitude, channel_number}}).
        """
        if module != self.target_module: return

        self.current_amplitude_being_processed = amplitude
        self.current_iteration_being_processed = iteration

        # Store data in detector-based structure, keyed by iteration index.
        # The amplitude and direction are stored inside each entry so that all
        # detectors share the same iteration indices regardless of per-section amplitudes.
        if multisweep_data_dict:
            for detector_id, det_data in multisweep_data_dict.items():
                if detector_id not in self.results_by_detector:
                    self.results_by_detector[detector_id] = {}
                entry = dict(det_data)
                entry['iteration'] = iteration
                self.results_by_detector[detector_id][iteration] = entry

        # --- Update the live resonator registry ---
        if res_info_dict:
            self.res_info_dict = res_info_dict

        # Invalidate digest panel so it gets recreated with fresh data
        # (the panel takes a snapshot at creation time and doesn't track live changes)
        if self.digest_panel is not None:
            self.digest_panel = None
        
        # Invalidate histogram cache so plots reflect the latest iteration
        if self.histogram_panel is not None:
            self.histogram_panel.histogram_cache.clear()
        
        self._redraw_plots() # Refresh plots with the new data
        
        # Note: We now update the status in handle_starting_iteration() instead of here

    def _redraw_plots(self):
        """
        Redraws plots based on the currently active tab.
        For sweep tabs (0, 1), uses grid plotting. For combined tab (2), uses original logic.
        For histogram tab (3), updates histogram panel.
        """
        # Early return if no data

        if not self.results_by_detector:
            # Clear any existing plots
            if hasattr(self, 'combined_mag_plot') and self.combined_mag_plot:
                self.combined_mag_plot.clear()
            if hasattr(self, 'combined_phase_plot') and self.combined_phase_plot:
                self.combined_phase_plot.clear()
            return
        
        # Get current tab
        current_tab_idx = self.plot_tabs.currentIndex()
        
        # Tabs 0 and 1: Grid sweep plots (magnitude and IQ)
        if current_tab_idx in (0, 1):
            self._redraw_sweep_grid(current_tab_idx)
        # Tab 2: Combined plots (original view)
        elif current_tab_idx == 2:
            self._redraw_combined_plots()
        # Tab 3: Histograms (parameter distributions)
        elif current_tab_idx == 3:

            self._generate_histograms()  # Creates or reloads histogram panel with latest data
            self.histograms_generated = True
        # Tab 4: Detector Digest (recreate if invalidated by new data)
        elif current_tab_idx == 4:
            if self.digest_panel is None and self.results_by_detector:
                first_key = next(iter(self.results_by_detector))
                self._open_detector_digest_for_index(first_key, switch_to_tab=False)
    
    def _redraw_sweep_grid(self, tab_idx):
        """Redraw the sweep grid plots for magnitude (tab 0) or IQ (tab 1)."""
        from .multisweep_grid_helpers import (

            create_amplitude_color_map,
            update_sweep_grid
        )
        
        # Prepare detector data for grid plotting directly from results_by_detector
        # Key by (amp_val, direction) for the grid helpers
        detector_data = {}
        for detector_id, iter_dict in self.results_by_detector.items():
            detector_data[detector_id] = {}
            for det_entry in iter_dict.values():
                amp_val = det_entry.get('sweep_amplitude')
                direction = det_entry.get('sweep_direction', 'upward')
                freqs = det_entry.get('frequencies', np.array([]))
                # Use raw counts when in counts mode, otherwise voltage-converted data
                if self.unit_mode == "counts":
                    iq_complex = det_entry.get('iq_complex', np.array([]))
                else:
                    iq_complex = det_entry.get('iq_complex', np.array([]))
                if amp_val is not None and len(freqs) > 0 and len(iq_complex) > 0:
                    detector_data[detector_id][(amp_val, direction)] = {
                        'freq': freqs,
                        'iq': iq_complex,
                        'amplitude': amp_val,
                        'direction': direction,
                        'original_center_frequency': det_entry.get('original_center_frequency')
                    }
        
        if not detector_data:
            return
        

        # Get all amplitudes for color mapping (unique amplitude values only)
        all_amps = set()
        for det_data in detector_data.values():
            for (amp_val, _direction) in det_data.keys():
                all_amps.add(amp_val)
        
        # Create amplitude color mapping (matches combined plot colors)
        amplitude_to_color = create_amplitude_color_map(all_amps, self.dark_mode)
        

        # Get DAC scale for label formatting
        dac_scale = self.dac_scales.get(self.active_module_for_dac)
        
        # Determine plot type, grid, cache, and colorbar based on tab
        if tab_idx == 0:
            plot_type = 'magnitude'
            grid_layout = self.mag_sweeps_grid
            widget_cache = self.mag_sweep_plots_cache
            colorbar = self.mag_colorbar
        else:  # tab_idx == 1
            plot_type = 'iq'
            grid_layout = self.iq_sweeps_grid
            widget_cache = self.iq_sweep_plots_cache
            colorbar = self.iq_colorbar
        
        # Count unique (amp, direction) pairs to decide legend vs colorbar
        all_sweep_keys = set()
        for det_data in detector_data.values():
            all_sweep_keys.update(det_data.keys())
        num_sweeps = len(all_sweep_keys)
        has_downward = any(d == 'downward' for _, d in all_sweep_keys)
        
        # Show colorbar when the inferno colormap is active (num_amps > threshold),
        # otherwise use per-plot legends with TABLEAU10 colors.
        num_amps = len(all_amps)
        if num_amps > AMPLITUDE_COLORMAP_THRESHOLD:
            sorted_amps = sorted(all_amps)
            colorbar.update_range(sorted_amps[0], sorted_amps[-1],
                                  dac_scale, self.unit_mode,
                                  self.dark_mode, has_downward)
            colorbar.show()
            use_legend = False  # colorbar replaces per-plot legends
        else:
            colorbar.hide()
            use_legend = True  # show per-plot legends
        
        # Update the grid with widget caching
        update_sweep_grid(
            grid_layout=grid_layout,
            data_by_detector=detector_data,
            plot_type=plot_type,
            current_batch=self.current_batch,
            batch_size=self.batch_size,
            amplitude_to_color=amplitude_to_color,
            dark_mode=self.dark_mode,
            unit_mode=self.unit_mode,
            normalize=self.normalize_traces,
            prev_btn=self.prev_batch_btn,
            next_btn=self.next_batch_btn,
            batch_label=self.batch_info_label,
            widget_cache=widget_cache,
            dac_scale=dac_scale,
            show_legend=use_legend
        )
        
        # Install double-click event filter on grid plot widgets
        # Must be AFTER update_sweep_grid so newly created widgets are included
        for pw in widget_cache:
            pw.installEventFilter(self)
    
    def _redraw_combined_plots(self):
        """Redraw the combined magnitude and phase plots (original view)."""
        if not self.combined_mag_plot or not self.combined_phase_plot:
            # Plots haven't been initialized yet
            return

        # Clear existing plot items and legends
        if self.mag_legend: self.mag_legend.clear()
        if self.phase_legend: self.phase_legend.clear()

        # Remove all existing data curves from plots
        for item in self.combined_mag_plot.listDataItems(): self.combined_mag_plot.removeItem(item)
        for item in self.combined_phase_plot.listDataItems(): self.combined_phase_plot.removeItem(item)
        
        self.curves_mag.clear() # Clear stored references to magnitude curves
        self.curves_phase.clear() # Clear stored references to phase curves

        # Remove existing center frequency (CF) lines
        for amp_val_lines in self.cf_lines_mag.values():
            for line in amp_val_lines: self.combined_mag_plot.removeItem(line)
        self.cf_lines_mag.clear()
        for amp_val_lines in self.cf_lines_phase.values():
            for line in amp_val_lines: self.combined_phase_plot.removeItem(line)
        self.cf_lines_phase.clear()

        if not self.results_by_detector:
            self.combined_mag_plot.autoRange(); self.combined_phase_plot.autoRange()
            return
        

        # Collect unique (amplitude, direction) pairs and amplitude values from entries
        amplitude_values = set()
        amp_dir_pairs = set()
        for iter_dict in self.results_by_detector.values():
            for entry in iter_dict.values():
                amp = entry.get('sweep_amplitude')
                direction = entry.get('sweep_direction', 'upward')
                if amp is not None:
                    amplitude_values.add(amp)
                    amp_dir_pairs.add((amp, direction))
        num_amps = len(amplitude_values)
            
        # Create a mapping for unique amplitude values to colors
        sorted_amplitudes = sorted(amplitude_values)
        amplitude_to_color = create_amplitude_color_map(amplitude_values, self.dark_mode)
        
        # Colorbar vs legend: show colorbar only when inferno colormap is active
        has_downward = any(d == 'downward' for _, d in amp_dir_pairs)
        dac_scale_for_module = self.dac_scales.get(self.active_module_for_dac)
        
        if num_amps > AMPLITUDE_COLORMAP_THRESHOLD:
            self.combined_colorbar.update_range(
                sorted_amplitudes[0], sorted_amplitudes[-1],
                dac_scale_for_module, self.unit_mode,
                self.dark_mode, has_downward)
            self.combined_colorbar.show()
            show_combined_legend = False
        else:
            self.combined_colorbar.hide()
            show_combined_legend = True
        
        legend_items_mag = {} # To avoid duplicate legend entries for the same amplitude/direction combination
        legend_items_phase = {}
        

        # Iterate through each (amplitude, direction) pair across all detectors
        for (amp_val, direction) in sorted(amp_dir_pairs):
            # Get color for this amplitude
            color = amplitude_to_color[amp_val]
            
            # Set line style based on direction using constants from utils.py
            line_style = DOWNWARD_SWEEP_STYLE if direction == "downward" else UPWARD_SWEEP_STYLE
            pen = pg.mkPen(color, width=LINE_WIDTH, style=line_style)
            
            # --- Prepare Legend Entry for this Amplitude (only if legends active) ---
            if show_combined_legend:
                legend_name_amp = UnitConverter.format_probe_label(amp_val, self.unit_mode, dac_scale_for_module)
                direction_suffix = " (Down)" if direction == "downward" else " (Up)"
                full_legend_name = legend_name_amp + direction_suffix
                
                legend_key = (amp_val, direction)
                
                if legend_key not in legend_items_mag and self.mag_legend:
                    dummy_mag_curve_for_legend = pg.PlotDataItem(pen=pen) 
                    self.mag_legend.addItem(dummy_mag_curve_for_legend, full_legend_name)
                    legend_items_mag[legend_key] = dummy_mag_curve_for_legend
                if legend_key not in legend_items_phase and self.phase_legend:
                    dummy_phase_curve_for_legend = pg.PlotDataItem(pen=pen)
                    self.phase_legend.addItem(dummy_phase_curve_for_legend, full_legend_name)
                    legend_items_phase[legend_key] = dummy_phase_curve_for_legend

            # --- Plot data for each detector at this (amplitude, direction) ---
            for res_idx, iter_dict in self.results_by_detector.items():
                # Find the entry matching this amplitude and direction
                data = None
                for entry in iter_dict.values():
                    if entry.get('sweep_amplitude') == amp_val and entry.get('sweep_direction', 'upward') == direction:
                        data = entry
                        break
                if data is None:
                    continue

                freqs_hz = data.get('frequencies', np.array([]))
                iq_complex = data.get('iq_complex', np.array([]))

                if freqs_hz is None or iq_complex is None or len(freqs_hz) == 0 or len(iq_complex) == 0:
                    continue
                
                # Calculate magnitude and phase
                s21_mag_raw = np.abs(iq_complex)
                # Compute probe_amp_dbm for dBm normalization if DAC scale available
                probe_amp_dbm = None
                if self.normalize_traces and self.unit_mode == 'dbm' and dac_scale_for_module is not None:
                    probe_amp_dbm = UnitConverter.normalize_to_dbm(amp_val, dac_scale_for_module)
                s21_mag_processed = UnitConverter.convert_amplitude(
                    s21_mag_raw, iq_complex, self.unit_mode, 
                    normalize=self.normalize_traces, probe_amp_dbm=probe_amp_dbm
                )
                # Use pre-calculated phase if available, otherwise calculate from IQ
                phase_deg = data.get('phase_degrees', np.degrees(np.angle(iq_complex))) 
                
                if self.normalize_traces and len(phase_deg) > 0:
                    first_phase_val = phase_deg[0]
                    if np.isfinite(first_phase_val):
                        phase_deg = phase_deg - first_phase_val
                
                # Plot magnitude curve
                mag_curve = self.combined_mag_plot.plot(pen=pen)
                mag_curve.setData(freqs_hz, s21_mag_processed)
                if amp_val not in self.curves_mag: self.curves_mag[amp_val] = {}
                self.curves_mag[amp_val][res_idx] = mag_curve

                # Plot phase curve
                phase_curve = self.combined_phase_plot.plot(pen=pen)
                phase_curve.setData(freqs_hz, phase_deg)
                if amp_val not in self.curves_phase: self.curves_phase[amp_val] = {}
                self.curves_phase[amp_val][res_idx] = phase_curve

                # Add center frequency (CF) lines if enabled
                if self.show_cf_lines_cb and self.show_cf_lines_cb.isChecked():
                    bias_freq = data.get('bias_frequency', data.get('original_center_frequency'))
                    if bias_freq is not None:
                        cf_line_pen = pg.mkPen(color, style=QtCore.Qt.PenStyle.DashLine, width=LINE_WIDTH/2)
                        
                        mag_cf_line = pg.InfiniteLine(pos=bias_freq, angle=90, pen=cf_line_pen, movable=False)
                        self.combined_mag_plot.addItem(mag_cf_line)
                        self.cf_lines_mag.setdefault(amp_val, []).append(mag_cf_line)

                        phase_cf_line = pg.InfiniteLine(pos=bias_freq, angle=90, pen=cf_line_pen, movable=False)
                        self.combined_phase_plot.addItem(phase_cf_line)
                        self.cf_lines_phase.setdefault(amp_val, []).append(phase_cf_line)
        
        # Adjust plot ranges to fit all data
        self.combined_mag_plot.autoRange()
        self.combined_phase_plot.autoRange()

    def completed_amplitude_sweep(self, module, amplitude):
        """
        Slot called when a sweep for a single amplitude is completed.
        Updates the progress bar.

        Args:
            module (int): The module that completed the sweep.
            amplitude (float): The amplitude for which the sweep was completed.
        """
        if module == self.target_module:
            self.progress_bar.setValue(100) # Mark as 100% for this specific amplitude

    def all_sweeps_completed(self):
        """
        Slot called when all amplitudes in the multisweep have been processed.
        Updates UI elements to reflect completion and auto-opens detector digest for first detector.
        """
        self._check_all_complete()
        self.current_amp_label.setText("All Amplitudes Processed")
        
        # Emit data_ready signal for session auto-export
        if self.results_by_detector:
            export_data = self._prepare_export_data()
            identifier = f"module{self.target_module}"
            self.data_ready.emit("multisweep", identifier, export_data)
        
        # Generate histogram plots once when all data is complete
        if self.results_by_detector and not self.histograms_generated:
            self._generate_histograms()
            self.histograms_generated = True
        
        # Auto-populate detector digest for the first detector in insertion order.
        # Don't switch focus — user should stay on the current tab (magnitude sweeps)
        if self.results_by_detector:
            first_key = next(iter(self.results_by_detector))
            self._open_detector_digest_for_index(first_key, switch_to_tab=False)
        
    def _check_all_complete(self):
        """
        Check if all progress is at 100% and hide the progress group when analysis is complete.
        This mimics the behavior in NetworkAnalysisWindow for consistency.
        """
        # Check if we have a parent window to determine the state of our tasks
        parent = self.parent()
        if not parent:
            # If no parent, just use the progress bar value as our indicator
            if self.progress_bar.value() == 100:
                self.progress_group.setVisible(False)
            return
            
        # If we have a parent, look for multisweep tasks related to this window
        window_has_active_tasks = False
        
        # If parent has multisweep_tasks, check if any are for this window
        if hasattr(parent, 'multisweep_tasks'):
            for task_key, task in parent.multisweep_tasks.items(): # type: ignore
                if hasattr(task, 'target_window') and task.target_window == self:
                    if not task.is_completed():
                        window_has_active_tasks = True
                        break
                        
        # Hide the progress group if there are no active tasks and progress is at 100%
        if not window_has_active_tasks and self.progress_bar.value() == 100:
            self.progress_group.setVisible(False)

    def handle_error(self, module, amplitude, error_msg):
        """
        Handles errors reported during the multisweep process.
        Displays an error message.

        Args:
            module (int): The module where the error occurred, or -1 for a general error.
            amplitude (float): The amplitude being processed when the error occurred, or -1.
            error_msg (str): The error message.
        """
        if module == self.target_module or module == -1: # -1 can indicate a general non-amplitude-specific error
            amp_str = f"for amplitude {amplitude:.4f}" if amplitude != -1 else "general"
            QtWidgets.QMessageBox.critical(
                self, 
                "Multisweep Error", 
                f"Error {amp_str} on Module {self.target_module}:\n{error_msg}"
            )
            self.progress_group.setVisible(False) # Hide progress bar on error

    def _export_data(self):
        """
        Exports the collected multisweep results to a pickle file using a non-blocking dialog.
        """
        # Thread marshalling - ensure we're on the main GUI thread
        app_instance = QtWidgets.QApplication.instance()
        if app_instance and QtCore.QThread.currentThread() != app_instance.thread():
            QtCore.QMetaObject.invokeMethod(self, "_export_data",
                                        QtCore.Qt.ConnectionType.QueuedConnection)
            return
            
        if not self.results_by_detector:
            QtWidgets.QMessageBox.warning(self, "No Data", "No data to export.")
            return
        
        # 1. Disable updates on graphics views
        if hasattr(self, 'combined_mag_plot') and self.combined_mag_plot:
            self.combined_mag_plot.setUpdatesEnabled(False)
        if hasattr(self, 'combined_phase_plot') and self.combined_phase_plot:
            self.combined_phase_plot.setUpdatesEnabled(False)
            
        # 2. Create a non-blocking file dialog
        dlg = QtWidgets.QFileDialog(self, "Export Multisweep Data")
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setNameFilters(["Pickle Files (*.pkl)", "All Files (*)"])
        dlg.setDefaultSuffix("pkl")
        
        # 3. Connect signals for handling dialog completion
        dlg.fileSelected.connect(self._handle_export_file_selected)
        dlg.finished.connect(self._resume_updates_after_export_dialog)
        
        # 4. Show the dialog non-modally
        dlg.open()  # Returns immediately, doesn't block
    
    def _resume_updates_after_export_dialog(self, result):
        """Resume updates after export dialog closes, regardless of the result."""
        # Re-enable updates on graphics views
        if hasattr(self, 'combined_mag_plot') and self.combined_mag_plot:
            self.combined_mag_plot.setUpdatesEnabled(True)
        if hasattr(self, 'combined_phase_plot') and self.combined_phase_plot:
            self.combined_phase_plot.setUpdatesEnabled(True)
    
    def _prepare_export_data(self) -> dict:
        """
        Prepare data dictionary for export.
        
        Returns:
            Dictionary containing all multisweep data for export
        """
        # Handle lack of noise data more gracefully
        if self.spectrum_noise_data:
            spectrum_data = self.spectrum_noise_data
        else:
            spectrum_data = None
        
        # Prepare initial_parameters with section_amplitudes if available
        export_initial_params = self.initial_params.copy()
        
        # Extract per-section amplitudes from detector data if not already stored
        if 'section_amplitudes' not in export_initial_params and self.results_by_detector:
            section_amps = []
            for det_idx in sorted(self.results_by_detector.keys()):
                if isinstance(det_idx, (int, np.integer)):
                    iter_dict = self.results_by_detector[det_idx]
                    if iter_dict:
                        first_entry = iter_dict[min(iter_dict.keys())]
                        amp = first_entry.get('sweep_amplitude')
                        if amp is not None:
                            section_amps.append(amp)

            if section_amps:
                export_initial_params['section_amplitudes'] = section_amps
            
        return {
            'timestamp': datetime.datetime.now().isoformat(),
            'target_module': self.target_module,
            'initial_parameters': export_initial_params,
            'dac_scales_used': self.dac_scales,
            'res_info_dict': self.res_info_dict,        # Lightweight resonator registry
            'results_by_detector': self.results_by_detector,
            'bias_kids_output': self.bias_kids_output,  # Include bias_kids results if available
            'nco_frequency_hz': self.nco_frequency_hz,  # NCO frequency used for biasing
            'noise_data': spectrum_data
        }
    
    def _handle_export_file_selected(self, filename):
        """Handle the file selection from the non-blocking dialog."""
        if not filename:
            return
            
        try:
            export_content = self._prepare_export_data()
            with open(filename, 'wb') as f: # Write in binary mode for pickle
                pickle.dump(export_content, f)
            QtWidgets.QMessageBox.information(self, "Export Complete", f"Data exported to {filename}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Error exporting data: {str(e)}")

    def _get_fit_frequencies(self) -> list[float]:
        """Get fitted resonance frequencies, ordered by channel_number.

        Iterates through res_info_dict codes (sorted by channel_number) and looks up
        the best available fit frequency from results_by_detector.  Falls back to
        bias_frequency when no fit result is present.

        Returns:
            List of frequencies in Hz, sorted by channel_number.
        """
        if not self.res_info_dict:
            return []

        sorted_codes = sorted(
            self.res_info_dict,
            key=lambda c: self.res_info_dict[c].get('channel_number', 0)
        )

        ref_freqs = []
        for code in sorted_codes:
            fallback = self.res_info_dict[code].get('bias_frequency', 0.0)
            iter_dict = self.results_by_detector.get(code)
            if not iter_dict:
                ref_freqs.append(fallback)
                continue
            first_entry = next(iter(iter_dict.values()))
            if first_entry.get('skewed_fit_success') and first_entry.get('fit_params'):
                ref_freqs.append(first_entry['fit_params']['fr'])
            elif first_entry.get('nonlinear_fit_success') and first_entry.get('nonlinear_fit_params'):
                ref_freqs.append(first_entry['nonlinear_fit_params']['fr'])
            else:
                ref_freqs.append(first_entry.get('bias_frequency', fallback))
        return ref_freqs
    
    
    def _rerun_multisweep(self):
        """
        Re-run the multisweep analysis, seeding the dialog from the live res_info_dict.
        Opens a MultisweepDialog for the user to adjust parameters before re-running.
        """
        from .dialogs import MultisweepDialog

        if not self.res_info_dict and not self.results_by_detector:
            QtWidgets.QMessageBox.warning(self, "Cannot Re-run",
                                          "No detector data available to re-run from.")
            return

        # Build ordered list of current bias frequencies (sorted by channel_number)
        sorted_codes = sorted(
            self.res_info_dict,
            key=lambda c: self.res_info_dict[c].get('channel_number', 0)
        ) if self.res_info_dict else []

        dialog_seed_frequencies = [
            self.res_info_dict[c]['bias_frequency'] for c in sorted_codes
        ]

        # Fall back to sweep_center_frequencies (or legacy resonance_frequencies) from
        # initial_params if res_info_dict is empty (e.g. loaded data without res_info_dict)
        if not dialog_seed_frequencies:
            dialog_seed_frequencies = list(
                self.initial_params.get('sweep_center_frequencies')
                or self.initial_params.get('resonance_frequencies', [])
            )

        if not dialog_seed_frequencies:
            QtWidgets.QMessageBox.warning(self, "Cannot Re-run",
                                          "No center frequencies are known to periscope to run on.")
            return

        # Fit frequencies (for user's "use fit frequency" option in dialog)
        fit_freqs = self._get_fit_frequencies() or None

        dialog = MultisweepDialog(
            parent=self,
            section_center_frequencies=dialog_seed_frequencies,
            dac_scales=self.dac_scales,
            current_module=self.target_module,
            initial_params=self.initial_params.copy(),
            load_multisweep=False,
            fit_frequencies=fit_freqs,
        )

        if not dialog.exec():
            return

        self.is_loaded_data = False

        # Update dock title
        periscope = self._get_periscope_parent()
        if periscope:
            my_dock = periscope.dock_manager.find_dock_for_widget(self)
            if my_dock:
                my_dock.setWindowTitle(my_dock.windowTitle().replace(" (Loaded)", ""))

        self.noise_spectrum_btn.setEnabled(False)
        new_params = dialog.get_parameters()
        if not new_params:
            return

        # Update current_run_amps
        _new_amp_arrays = new_params.get('amp_arrays', [])
        if _new_amp_arrays:
            self.current_run_amps = [arr[0] for arr in _new_amp_arrays if arr]
        else:
            self.current_run_amps = list(new_params.get('amps', []))
        self.probe_amplitudes = list(self.current_run_amps)

        # Apply parameters and inject res_info_dict so MultisweepTask re-uses existing codes
        self.initial_params.update(new_params)
        if self.res_info_dict:
            self.initial_params['res_info_dict'] = self.res_info_dict

        # Reset window state for the new sweep
        self.results_by_detector.clear()
        self.digest_panel = None
        self.histograms_generated = False
        self._redraw_plots()
        if self.progress_bar: self.progress_bar.setValue(0)
        if self.progress_group: self.progress_group.setVisible(True)

        num_amplitudes = len(self.probe_amplitudes)
        sweep_direction = self.initial_params.get('sweep_direction', 'upward')
        self.total_iterations = num_amplitudes * (2 if sweep_direction == "both" else 1)
        sweep_direction_norm = sweep_direction.lower().strip() if sweep_direction else ""
        direction_text = "Down" if sweep_direction_norm == "downward" else "Up"

        if self.current_amp_label:
            if num_amplitudes > 0:
                self.current_amp_label.setText(f"Iteration 1/{self.total_iterations} ({direction_text})")
            else:
                self.current_amp_label.setText("No sweeps defined. (Waiting...)")

        parent_widget = self._get_periscope_parent()
        if parent_widget and hasattr(parent_widget, '_start_multisweep_analysis_for_window'):
            parent_widget._start_multisweep_analysis_for_window(self, self.initial_params)  # type: ignore
        else:
            QtWidgets.QMessageBox.warning(self, "Error",
                                          "Cannot trigger re-run. Parent linkage or method missing.")

    def closeEvent(self, event: pg.QtGui.QCloseEvent):
        """
        Overrides QWidget.closeEvent.
        Notifies the parent/controller to stop any ongoing tasks associated with this window
        before closing.
        """
        parent_widget = self.parent()
        # Check if the parent object has a method to stop tasks for this window
        if parent_widget and hasattr(parent_widget, 'stop_multisweep_task_for_window'):
            parent_widget.stop_multisweep_task_for_window(self) # type: ignore
        super().closeEvent(event) # Proceed with the standard close event handling

    def _toggle_cf_lines_visibility(self, checked):
        """
        Slot for the 'Show Center Frequencies' checkbox.
        Shows or hides CF lines without a full redraw.
        Creates lines if they don't exist when showing.

        Args:
            checked (bool): The new state of the checkbox.
        """
        if not self.combined_mag_plot or not self.combined_phase_plot:
            return # Plots not ready

        if checked:
            # Show lines. Create them if they don't exist.
            # Get unique amplitudes from iterations
            amplitude_values = set()

            for iter_dict in self.results_by_detector.values():
                for entry in iter_dict.values():
                    amp = entry.get('sweep_amplitude')
                    if amp is not None:
                        amplitude_values.add(amp)
            num_amps = len(amplitude_values)
            
            if num_amps == 0:
                return

            # Use the canonical color mapping function
            amplitude_to_color = create_amplitude_color_map(amplitude_values, self.dark_mode)
            sorted_amplitudes = sorted(amplitude_values)


            for amp_val in sorted_amplitudes:
                color = amplitude_to_color[amp_val]
                cf_line_pen = pg.mkPen(color, style=QtCore.Qt.PenStyle.DashLine, width=LINE_WIDTH/2)

                # Ensure lists for this amplitude exist in cf_lines_mag/phase
                self.cf_lines_mag.setdefault(amp_val, [])
                self.cf_lines_phase.setdefault(amp_val, [])

                # Create dictionaries for quick lookup of existing lines by their X-position (CF)
                # This avoids iterating through the list of lines repeatedly for each CF.
                existing_mag_lines_for_amp = {line.pos().x(): line for line in self.cf_lines_mag[amp_val]}
                existing_phase_lines_for_amp = {line.pos().x(): line for line in self.cf_lines_phase[amp_val]}

                for res_idx, iter_dict in self.results_by_detector.items():
                    # Get detector data for this amplitude (any direction)
                    data = None
                    for entry in iter_dict.values():
                        if entry.get('sweep_amplitude') == amp_val:
                            data = entry
                            break
                    if data is None:
                        continue
                    # Get the actual bias frequency for CF line
                    bias_freq = data.get('bias_frequency', data.get('original_center_frequency'))
                    if bias_freq is None:
                        continue
                        
                    # Magnitude plot CF line
                    if bias_freq in existing_mag_lines_for_amp:
                        existing_mag_lines_for_amp[bias_freq].setVisible(True)
                    else:
                        mag_cf_line = pg.InfiniteLine(pos=bias_freq, angle=90, pen=cf_line_pen, movable=False)
                        self.combined_mag_plot.addItem(mag_cf_line)
                        self.cf_lines_mag[amp_val].append(mag_cf_line)
                        # mag_cf_line.setVisible(True) # Already visible by default when added

                    # Phase plot CF line
                    if bias_freq in existing_phase_lines_for_amp:
                        existing_phase_lines_for_amp[bias_freq].setVisible(True)
                    else:
                        phase_cf_line = pg.InfiniteLine(pos=bias_freq, angle=90, pen=cf_line_pen, movable=False)
                        self.combined_phase_plot.addItem(phase_cf_line)
                        self.cf_lines_phase[amp_val].append(phase_cf_line)
                        # phase_cf_line.setVisible(True) # Already visible by default when added
        else:
            # Hide all existing CF lines
            for amp_lines_list in self.cf_lines_mag.values():
                for line in amp_lines_list:
                    line.setVisible(False)
            for amp_lines_list in self.cf_lines_phase.values():
                for line in amp_lines_list:
                    line.setVisible(False)
        
        # Note: No call to self._redraw_plots() here, to preserve zoom.
        # The _redraw_plots method will still handle full reconstruction of lines
        # if it's called for other reasons (data update, unit change, etc.),
        # respecting the checkbox state at that time.


    def _take_noise_samps(self):
        """Collect a short noise sample from the CRS for diagnostic purposes.

        Takes 100 samples from all channels on the target module using
        ``crs.get_samples`` and stores them in ``self.noise_data``.

        Returns:
            The collected samples, or None if the CRS is unavailable.
        """
        total = 100
        # self.take_samp_btn.setEnabled(False)

        periscope = self._get_periscope_parent()
        if not periscope or periscope.crs is None:
            QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available")
            return None

        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Launch async sampler
        samples = loop.run_until_complete(periscope.crs.get_samples(total, average=False, channel=None, module=self.target_module))
        self.noise_data = samples
        loop.close()
        # self.take_samp_btn.setEnabled(True)
        self.samples_taken = True

        return self.noise_data


    def _open_noise_spectrum_dialog(self):
        num_res = len(self.res_info_dict) if self.res_info_dict else len(self.results_by_detector)
        periscope = self._get_periscope_parent()
        
        if not periscope or periscope.crs is None: 
            QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available") 
            return
        
        crs = periscope.crs
        noise_dialog = NoiseSpectrumDialog(self, num_res, crs)
        if noise_dialog.exec():
            params = noise_dialog.get_parameters()
            self._get_spectrum(params)

    def _set_decimation(self, crs, decimation):
        print("Setting decimation to", decimation)

        if decimation > 4:
            asyncio.run(crs.set_decimation(decimation , short = False))
        elif decimation == 4:
            asyncio.run(crs.set_decimation(decimation , module = self.target_module, short = False))
        else:
            
            asyncio.run(crs.set_decimation(decimation, module = self.target_module, short = True))

    def _get_spectrum(self, params, use_loaded_noise = False):
        """Acquire or load a noise spectrum and open the NoiseSpectrumPanel.

        When *use_loaded_noise* is ``True`` the method expects pre-collected
        data inside *params* (keys ``noise_parameters`` and ``data``) and
        simply opens the panel.  Otherwise it drives the CRS to collect slow
        and (optionally) PFB spectrum data, stores the results in
        ``self.spectrum_noise_data``, emits the ``data_ready`` signal for
        session auto-export, and opens the panel.

        Args:
            params: Dictionary of acquisition parameters (from NoiseSpectrumDialog)
                    or loaded noise data when *use_loaded_noise* is True.
            use_loaded_noise: If True, skip acquisition and use data already in *params*.
        """
        if use_loaded_noise:
            print(f"[Bias] Plotting noise data taken at decimation {params['noise_parameters']['decimation']}")
            self.spectrum_noise_data['noise_parameters'] = params['noise_parameters']
            self.spectrum_noise_data['data'] = params['data']
            # Open the noise spectrum panel for loaded noise data
            self._open_noise_spectrum_panel(1)
        else:
            periscope = self._get_periscope_parent()
            if not periscope or periscope.crs is None: 
                QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available") 
                return
            
            crs = periscope.crs
    
            time_taken = params['time_taken']
            pfb_enabled = params['pfb_enabled']
            
            if pfb_enabled:
                pfb_time_taken = params['pfb_time']
            else:
                pfb_time_taken = 0
                
            t = time.time() + time_taken + pfb_time_taken
            formatted_time = time.strftime("%H:%M:%S", time.localtime(t))
            # Show a progress dialog
            progress = QtWidgets.QProgressDialog(f"Getting noise spectrum...\n\nEstimated Completion Time {formatted_time}", None, 0, 0, self)
            progress.setWindowTitle("Please wait")
            progress.setCancelButton(None)
            progress.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
            progress.show()
            
            QtWidgets.QApplication.processEvents()# Show a simple "busy" message and spinner cursor)
    
            try:
                decimation = params['decimation']
                num_samples = params['num_samples']
                num_segments = params['num_segments']
                reference = params['reference']
                spec_lim = params['spectrum_limit']
                module = self.target_module
                curr_decimation = asyncio.run(crs.get_decimation())
    
                if pfb_enabled:
                    overlap = params['overlap']
                    pfb_samples = params['pfb_samples']
        
                if curr_decimation != decimation:
                    self._set_decimation(crs, decimation)
        
                spectrum_data  = asyncio.run(crs.py_get_samples(num_samples, 
                                                                return_spectrum=True, 
                                                                scaling='psd', 
                                                                reference=reference, 
                                                                nsegments=num_segments, 
                                                                spectrum_cutoff=spec_lim,
                                                                channel=None, 
                                                                module=module))
    
    
                self.spectrum_noise_data['noise_parameters'] = params
                num_res = len(self.res_info_dict) if self.res_info_dict else len(self.results_by_detector)
    
                amplitudes = []
                dac_scale_for_module = self.dac_scales.get(self.active_module_for_dac)
    
                pfb_psd_i = []
                pfb_psd_q = []
                pfb_dual = []
                pfb_i = []
                pfb_q = []
                pfb_freq_iq = []
                pfb_freq_dsb = []
    
                for i in range(num_res):
                    amp = asyncio.run(crs.get_amplitude(channel=i+1, module = module))
                    amp_dmb = UnitConverter.normalize_to_dbm(amp, dac_scale_for_module)
                    amplitudes.append(amp_dmb)
    
                    #### Also running pfb_samples ####
                    if pfb_enabled:
                        pfb_data = asyncio.run(crs.py_get_pfb_samples(pfb_samples,
                                                                      channel = i + 1,
                                                                      module = module,
                                                                      binlim = 1e6,
                                                                      trim = False,
                                                                      nsegments = num_segments,
                                                                      reference = reference,
                                                                      reset_NCO = False))
        
                        psd_i = pfb_data.spectrum.psd_i
                        pfb_psd_i.append(psd_i)
                        
                        psd_q = pfb_data.spectrum.psd_q
                        pfb_psd_q.append(psd_q)
                        
                        I = pfb_data.i
                        pfb_i.append(I)
                        
                        Q = pfb_data.q
                        pfb_q.append(Q)
                        
                        dual = pfb_data.spectrum.psd_dual_sideband
                        pfb_dual.append(dual)
                        
                        freq_iq = pfb_data.spectrum.freq_iq
                        pfb_freq_iq.append(freq_iq)
                        
                        freq_dsb = pfb_data.spectrum.freq_dsb
                        pfb_freq_dsb.append(freq_dsb)
    
                    #### Getting pfb time stamps for plotting #####
    
                if pfb_enabled:
                    total_time = (1/PFB_SAMPLING_FREQ) * pfb_samples #### 2.44 MSS is the rate 
                    ts_pfb = list(np.linspace(0, total_time, pfb_samples))
                    
                
                slow_freq = max(spectrum_data.spectrum.freq_iq)/spec_lim
                fast_freq = PFB_SAMPLING_FREQ/2   
    
    
                
                data = {}
                data['reference'] = reference
                data['ts'] = spectrum_data.ts
                data['I'] = spectrum_data.i[0:num_res]
                data['Q'] = spectrum_data.q[0:num_res]
                data['freq_iq'] = spectrum_data.spectrum.freq_iq
                data['single_psd_i'] = spectrum_data.spectrum.psd_i[0:num_res]
                data['single_psd_q'] = spectrum_data.spectrum.psd_q[0:num_res]
                data['freq_dsb'] = spectrum_data.spectrum.freq_dsb
                data['dual_psd'] = spectrum_data.spectrum.psd_dual_sideband[0:num_res]
                data['amplitudes_dbm'] = amplitudes
                data['slow_freq_hz'] = slow_freq
                data['fast_freq_hz'] = fast_freq
    
    
                ##### pfb data ####
                if pfb_enabled:
                    data['pfb_enabled'] = True
                    data['pfb_ts'] = ts_pfb
                    data['pfb_I'] = pfb_i
                    data['pfb_Q'] = pfb_q
                    data['pfb_freq_iq'] = pfb_freq_iq
                    data['pfb_psd_i'] = pfb_psd_i
                    data['pfb_psd_q'] = pfb_psd_q
                    data['pfb_freq_dsb'] = pfb_freq_dsb
                    data['pfb_dual_psd'] = pfb_dual
                    data['overlap'] = overlap
    
                else:
                    data['pfb_enabled'] = False
                
                self.spectrum_noise_data['data'] = data  
                
                # Emit data_ready signal for session auto-export
                export_data = self._prepare_export_data()
                identifier = f"module{module}_noise"
                self.data_ready.emit("noise", identifier, export_data)
                
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", str(e))
                traceback.print_exc()
                raise
            finally:
                progress.close()
                
            # If we successfully got data, open the noise spectrum panel
            if self.spectrum_noise_data.get('data'):
                # Default to opening for the first detector
                self._open_noise_spectrum_panel(1)

    def _open_noise_spectrum_panel(self, detector_idx: int = 1):
        """
        Open a NoiseSpectrumPanel for a specific detector index.
        
        Args:
            detector_idx: Detector index (1-based) to open panel for
        """
        # Get spectrum data
        spectrum_data = self.spectrum_noise_data.get('data')
        if not spectrum_data:
            print("Warning: No noise spectrum data available")
            return
            
        # Get resonance frequency for this detector, keyed by channel_number.
        # Build a {channel_number: bias_frequency} map from res_info_dict.
        ch_to_freq = {
            info.get('channel_number', 0): info.get('bias_frequency')
            for info in self.res_info_dict.values()
        } if self.res_info_dict else {}
        conceptual_resonance_base_freq_hz = ch_to_freq.get(detector_idx)
        if conceptual_resonance_base_freq_hz is None:
            print(f"Warning: Detector channel {detector_idx} not found in res_info_dict.")
            return

        # Gather data for ALL detectors (channel_number → freq) to enable navigation
        all_detectors_data = {}
        for code, info in self.res_info_dict.items():
            ch = info.get('channel_number', 0)
            freq = info.get('bias_frequency')
            if ch and freq is not None:
                all_detectors_data[ch] = {'conceptual_freq_hz': freq}
            
        # Find Periscope parent to create docked panel
        periscope = self._get_periscope_parent()
        if not periscope:
            print("ERROR: Could not find Periscope parent for noise spectrum panel")
            return
            
        # Create panel
        panel = NoiseSpectrumPanel(
            parent=self,
            detector_id=detector_idx,
            resonance_frequency_ghz=conceptual_resonance_base_freq_hz / 1e9,
            dark_mode=self.dark_mode,
            all_detectors_data=all_detectors_data,
            initial_detector_idx=detector_idx,
            spectrum_data=spectrum_data
        )
        
        # Store direct reference to this MultisweepPanel (if needed)
        panel.multisweep_panel_ref = self
        
        # Increment counter and use for tab name
        self.noise_panel_count += 1
        loaded_suffix = " (Loaded)" if self.is_loaded_data else ""
        dock_title = f"Noise Spectrum #{self.noise_panel_count}{loaded_suffix}"
        dock_id = f"noise_{self.noise_panel_count}_{int(time.time())}"
        
        # Create dock
        dock = periscope.dock_manager.create_dock(panel, dock_title, dock_id)
        
        # Track panel reference
        self.noise_spectrum_windows.append(panel)
        
        # Try to tabify with existing digest panel if available, else with multisweep panel
        target_dock = None
        if self.detector_digest_windows:
            # Tabify with the most recently created digest window
            last_digest = self.detector_digest_windows[-1]
            target_dock = periscope.dock_manager.find_dock_for_widget(last_digest)
        
        if not target_dock:
            # Fallback to multisweep panel
            target_dock = periscope.dock_manager.find_dock_for_widget(self)
            
        if target_dock:
            periscope.tabifyDockWidget(target_dock, dock)
        
        # Show and activate the dock
        dock.show()
        dock.raise_()

    def eventFilter(self, obj, event):
        """Catch double-clicks on grid subplot widgets to navigate the digest panel."""
        if event.type() == QtCore.QEvent.Type.MouseButtonDblClick:
            detector_id = getattr(obj, '_detector_id', None)
            if detector_id is not None:
                self._navigate_digest_to_detector(detector_id)
                return True
        return super().eventFilter(obj, event)
    
    def _navigate_digest_to_detector(self, detector_id: int):
        """Navigate the embedded digest panel to the given detector, or create it."""
        # If the digest panel already exists in the tab, just navigate it
        if self.digest_panel is not None:
            if hasattr(self.digest_panel, '_switch_to_detector') and hasattr(self.digest_panel, 'all_detectors_data'):
                if detector_id in self.digest_panel.all_detectors_data:
                    try:
                        self.digest_panel.current_detector_index_in_list = self.digest_panel.detector_indices.index(detector_id)
                    except ValueError:
                        pass
                    self.digest_panel._switch_to_detector(detector_id)
                    # Switch to the Detector Digest tab
                    self.plot_tabs.setCurrentWidget(self.digest_tab)
                    return
        # No existing digest panel — create one in the tab
        self._open_detector_digest_for_index(detector_id)
    
    @QtCore.pyqtSlot(object)
    def _handle_multisweep_plot_double_click(self, ev):
        """
        Handles a double-click event on the multisweep magnitude plot.
        Identifies the clicked resonance and opens detector digest for it.
        """
        if not self.results_by_detector:
            return

        # Get click coordinates in view space from the event's scenePos
        if not self.combined_mag_plot:
            return
        view_box = self.combined_mag_plot.getViewBox()
        if not view_box:
            return

        mouse_point = view_box.mapSceneToView(ev.scenePos())
        x_coord = mouse_point.x()

        # Find the resonance whose center/bias frequency is closest to the click
        resonance_centers = {}  # {res_idx: center_freq}
        
        for res_idx, amp_dir_dict in self.results_by_detector.items():
            # Use first available entry to get center frequency
            for det_data in amp_dir_dict.values():
                center_freq = det_data.get('bias_frequency', det_data.get('original_center_frequency'))
                if center_freq is not None:
                    resonance_centers[res_idx] = center_freq
                    break
        
        # Find the resonance with center frequency closest to the click
        min_distance = np.inf
        clicked_res_idx = None
        
        for res_idx, center_freq in resonance_centers.items():
            distance = abs(center_freq - x_coord)
            if distance < min_distance:
                min_distance = distance
                clicked_res_idx = res_idx

        if clicked_res_idx is not None:
            # Accept the event and open the detector digest
            ev.accept()
            self._open_detector_digest_for_index(clicked_res_idx)

    def _open_detector_digest_for_index(self, detector_idx: int, switch_to_tab: bool = True):
        """
        Open or update the detector digest panel for a specific detector index.
        
        The digest panel is embedded as a sub-tab (Tab 4) within this MultisweepPanel,
        following the same lazy-initialization pattern as the Histograms tab.
        If the panel already exists, it navigates to the requested detector.
        If not, it creates the panel and adds it to the digest tab.
        
        Args:
            detector_idx: Detector index (1-based) to open digest for
            switch_to_tab: If True (default), switch focus to the Detector Digest tab.
                          If False, populate the tab without switching focus (used by auto-populate on completion).
        """
        if not self.results_by_detector:
            return
        
        # If digest panel already exists, just navigate to the requested detector
        if self.digest_panel is not None:
            if hasattr(self.digest_panel, '_switch_to_detector') and hasattr(self.digest_panel, 'all_detectors_data'):
                if detector_idx in self.digest_panel.all_detectors_data:
                    try:
                        self.digest_panel.current_detector_index_in_list = self.digest_panel.detector_indices.index(detector_idx)
                    except ValueError:
                        pass
                    self.digest_panel._switch_to_detector(detector_idx)
                    # Switch to the Detector Digest tab
                    self.plot_tabs.setCurrentWidget(self.digest_tab)
                    return
        
        # --- First time: create the digest panel and embed it in the tab ---
        
        # Get debug data from Periscope parent
        periscope = self.parent()
        while periscope and not hasattr(periscope, 'get_test_noise'):
            periscope = periscope.parent()
        
        if periscope:
            self.debug_noise_data = periscope.get_test_noise()
            self.debug_phase_data = periscope.get_phase_shift()
        else:
            self.debug_noise_data = {}
            self.debug_phase_data = []
        
        # Get noise data if available
        noise_data = self.noise_data if self.samples_taken else None
        
        # Get reference frequency for this detector.
        # New format: string codes — read bias_frequency from data.
        # Old format: integer indices — use conceptual_section_frequencies.
        if isinstance(detector_idx, str):
            first_entry = next(iter(self.results_by_detector.get(detector_idx, {}).values()), {})
            conceptual_resonance_base_freq_hz = (
                first_entry.get('bias_frequency')
                or first_entry.get('sweep_center_frequency')
            )
            if conceptual_resonance_base_freq_hz is None:
                print(f"Warning: Could not determine frequency for detector {detector_idx!r}")
                return
        else:
            # Legacy integer keys (old format) — read frequency from sweep data directly
            first_entry = next(iter(self.results_by_detector.get(detector_idx, {}).values()), {})
            conceptual_resonance_base_freq_hz = first_entry.get(
                'bias_frequency', first_entry.get('original_center_frequency')
            )
            if conceptual_resonance_base_freq_hz is None:
                print(f"Warning: Detector index {detector_idx!r} not found or has no frequency.")
                return

        # Gather data for this specific detector across all amplitudes and directions
        section_data_for_digest = {}
        

        if detector_idx in self.results_by_detector:
            for det_entry in self.results_by_detector[detector_idx].values():
                amp_val = det_entry.get('sweep_amplitude')
                direction = det_entry.get('sweep_direction', 'upward')
                actual_cf_for_this_amp = det_entry.get('bias_frequency',
                                                       det_entry.get('original_center_frequency'))
                combo_key = f"{amp_val}:{direction}"
                section_data_for_digest[combo_key] = {
                    'data': det_entry,
                    'actual_cf_hz': actual_cf_for_this_amp,
                    'direction': direction,
                    'amplitude': amp_val
                }
        
        if not section_data_for_digest:
            print(f"Warning: No data for detector {detector_idx}")
            return
        
        # Gather data for ALL detectors to enable navigation
        all_detectors_data = {}
        

        for det_idx in sorted(self.results_by_detector.keys(), key=lambda k: str(k)):
            # Determine reference frequency from sweep data (works for both string codes and legacy ints)
            _entry = next(iter(self.results_by_detector[det_idx].values()), {})
            if isinstance(det_idx, str):
                conceptual_freq_hz = (
                    _entry.get('bias_frequency')
                    or _entry.get('sweep_center_frequency')
                    or _entry.get('original_center_frequency')
                )
            else:
                # Legacy integer keys — read directly from the sweep entry
                conceptual_freq_hz = _entry.get('bias_frequency', _entry.get('original_center_frequency'))
            
            if conceptual_freq_hz is None:
                continue
            
            detector_resonance_data = {}

            for det_entry in self.results_by_detector[det_idx].values():
                amp_val = det_entry.get('sweep_amplitude')
                direction = det_entry.get('sweep_direction', 'upward')
                actual_cf = det_entry.get('bias_frequency', det_entry.get('original_center_frequency'))
                combo_key = f"{amp_val}:{direction}"
                detector_resonance_data[combo_key] = {
                    'data': det_entry,
                    'actual_cf_hz': actual_cf,
                    'direction': direction,
                    'amplitude': amp_val
                }
            
            if detector_resonance_data:
                all_detectors_data[det_idx] = {
                    'resonance_data': detector_resonance_data,
                    'conceptual_freq_hz': conceptual_freq_hz
                }
        
        # Create the DetectorDigestPanel
        panel = DetectorDigestPanel(
            parent=self,
            resonance_data_for_digest=section_data_for_digest,
            detector_id=detector_idx,
            resonance_frequency_ghz=conceptual_resonance_base_freq_hz / 1e9,
            dac_scales=self.dac_scales,
            zoom_box_mode=self.zoom_box_mode,
            target_module=self.target_module,
            normalize_plot3=self.normalize_traces,
            dark_mode=self.dark_mode,
            all_detectors_data=all_detectors_data,
            initial_detector_idx=detector_idx,
            noise_data=noise_data,
            debug_noise_data=self.debug_noise_data,
            debug_phase_data=self.debug_phase_data,
            debug=False
        )
        
        # Store direct reference to this MultisweepPanel for noise sampling
        panel.multisweep_panel_ref = self
        
        # Embed the panel into the digest tab (replacing placeholder)
        layout = self.digest_tab.layout()
        if layout:
            # Remove placeholder
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            # Add the actual digest panel
            layout.addWidget(panel)
        
        # Store the panel reference
        self.digest_panel = panel
        # Also keep backward-compatible list reference
        self.detector_digest_windows = [panel]
        

        # Switch to the Detector Digest tab (unless suppressed, e.g. auto-populate on completion)
        if switch_to_tab:
            self.plot_tabs.setCurrentWidget(self.digest_tab)

    def _get_periscope_parent(self):
        """Find and return the Periscope parent window.

        Walks up the widget hierarchy looking for the main Periscope instance,
        identified by the ``dock_manager`` attribute.  This intentionally
        overrides :meth:`ScreenshotMixin._get_periscope_parent` (which
        searches for ``crs``) because panel operations need the dock manager,
        and the Periscope instance always carries both attributes.
        """
        parent = self.parent()
        while parent:
            if hasattr(parent, 'dock_manager'):
                return parent
            parent = parent.parent()
        return None
    
    def apply_theme(self, dark_mode: bool):
        """Apply the dark/light theme to all plots in this window."""
        self.dark_mode = dark_mode
        
        bg_color, pen_color = ("k", "w") if dark_mode else ("w", "k")
        
        # Apply to magnitude plot
        if self.combined_mag_plot:
            self.combined_mag_plot.setBackground(bg_color)
            plot_item_mag = self.combined_mag_plot.getPlotItem()
            if plot_item_mag:
                title_text_mag = plot_item_mag.titleLabel.text if plot_item_mag.titleLabel else "Combined S21 Magnitude (All Resonances)"
                plot_item_mag.setTitle(title_text_mag, color=pen_color)
                for axis_name in ("left", "bottom", "right", "top"):
                    ax = plot_item_mag.getAxis(axis_name)
                    if ax:
                        ax.setPen(pen_color)
                        ax.setTextPen(pen_color)
            if self.mag_legend:
                try:
                    legend_color = '#CCCCCC' if dark_mode else '#333333'
                    self.mag_legend.setLabelTextColor(legend_color)
                except Exception as e:
                    print(f"Error updating magnitude legend text color: {e}")

        # Apply to phase plot
        if self.combined_phase_plot:
            self.combined_phase_plot.setBackground(bg_color)
            plot_item_phase = self.combined_phase_plot.getPlotItem()
            if plot_item_phase:
                title_text_phase = plot_item_phase.titleLabel.text if plot_item_phase.titleLabel else "Combined S21 Phase (All Resonances)"
                plot_item_phase.setTitle(title_text_phase, color=pen_color)
                for axis_name in ("left", "bottom", "right", "top"):
                    ax = plot_item_phase.getAxis(axis_name)
                    if ax:
                        ax.setPen(pen_color)
                        ax.setTextPen(pen_color)
            if self.phase_legend:
                try:
                    legend_color = '#CCCCCC' if dark_mode else '#333333'
                    self.phase_legend.setLabelTextColor(legend_color)
                except Exception as e:
                    print(f"Error updating phase legend text color: {e}")
        
        # Redraw plots which will now use the updated legend text colors
        self._redraw_plots()
            
        # Propagate to histogram panel
        if self.histogram_panel and hasattr(self.histogram_panel, 'apply_theme'):
            self.histogram_panel.apply_theme(dark_mode)
            
        # Also propagate dark mode to any open detector digest windows
        for digest_window in self.detector_digest_windows:
            if hasattr(digest_window, 'apply_theme'):
                digest_window.apply_theme(dark_mode)
        
        # Propagate to noise spectrum windows
        for noise_window in self.noise_spectrum_windows:
            if hasattr(noise_window, 'apply_theme'):
                noise_window.apply_theme(dark_mode)
    
    def _bias_kids(self):
        """
        Run the bias_kids algorithm on the current multisweep results.
        Programs detectors at optimal operating points and stores calibration data.
        """
        # Check prerequisites
        if not self.results_by_detector:
            QtWidgets.QMessageBox.warning(self, "No Data", 
                                        "No multisweep data available. Please run a multisweep first.")
            return
        
        # Get Periscope parent
        periscope = self._get_periscope_parent()
        if not periscope:
            QtWidgets.QMessageBox.warning(self, "Parent Not Available", 
                                        "Parent window not available. Cannot access CRS object.")
            return
            
        if periscope.crs is None:
            QtWidgets.QMessageBox.warning(self, "CRS Not Available", 
                                        "CRS object is None. Cannot bias detectors.")
            return
        # Import the dialog
        from .bias_kids_dialog import BiasKidsDialog
        
        # Show dialog to get parameters
        dialog = BiasKidsDialog(self, self.target_module)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return  # User cancelled
        
        # Get parameters from dialog
        bias_params = dialog.get_parameters()
        
        # Pass detector-indexed format directly to bias_kids
        gui_format_results = {
            'results_by_detector': self.results_by_detector
        }
        
        # Import BiasKidsTask and BiasKidsSignals from tasks module
        from .tasks import BiasKidsTask, BiasKidsSignals
        
        # Create signals for communication with the task
        self.bias_kids_signals = BiasKidsSignals()
        self.bias_kids_signals.progress.connect(self._bias_kids_progress)
        self.bias_kids_signals.completed.connect(self._bias_kids_completed)
        self.bias_kids_signals.error.connect(self._bias_kids_error)
        
        # Ensure we have a valid module number
        if self.target_module is None:
            QtWidgets.QMessageBox.warning(self, "Module Not Set", 
                                        "Target module is not set. Cannot bias detectors.")
            return
        # Create and start the task with dialog parameters
        self.bias_kids_task = BiasKidsTask(
            periscope.crs,
            self.target_module,
            gui_format_results,
            self.bias_kids_signals,
            bias_params  # Pass the dialog parameters
        )
        
        # Update UI to show operation in progress
        self.bias_kids_btn.setEnabled(False)
        self.bias_kids_btn.setText("Biasing...")
        
        # Start the task
        self.bias_kids_task.start()

    def _bias_kids_progress(self, module, progress):
        """Handle progress updates from the bias_kids task."""
        # Could update a progress indicator if desired
        pass
    
    def _bias_kids_completed(self, module, biased_results, df_calibrations, nco_frequency_hz):
        """Handle completion of the bias_kids task."""
        # Store the output
        self.bias_kids_output = biased_results
        
        # Store the NCO frequency used during biasing
        self.nco_frequency_hz = nco_frequency_hz
        
        # Emit signal with df_calibration data
        if df_calibrations:
            self.df_calibration_ready.emit(module, df_calibrations)
        
        # Emit data_ready signal for session auto-export
        if biased_results:
            export_data = self._prepare_export_data()
            identifier = f"module{module}"
            self.data_ready.emit("bias", identifier, export_data)
        
        # Show success dialog
        num_biased = len(biased_results)
        total_detectors = len(self.res_info_dict) if self.res_info_dict else len(self.results_by_detector)

        msg = f"Successfully biased {num_biased} out of {total_detectors} detectors.\n\n"
        
        if num_biased > 0:
            msg += "The detectors have been programmed at their optimal operating points."
            if df_calibrations:
                msg += "\n\nFrequency shift calibration data has been loaded into the main window."
        else:
            msg += "No detectors met the criteria for biasing."
        
        QtWidgets.QMessageBox.information(self, "Bias KIDs Complete", msg)
        
        # Reset UI
        self.bias_kids_btn.setEnabled(True)
        self.noise_spectrum_btn.setEnabled(True)
        self.bias_kids_btn.setText("Bias KIDs")
        
        # Clean up the task
        self.bias_kids_task = None
    
    def _bias_kids_error(self, error_msg):
        """Handle errors from the bias_kids task."""
        QtWidgets.QMessageBox.critical(self, "Bias KIDs Error", error_msg)
        
        # Reset UI
        self.bias_kids_btn.setEnabled(True)
        self.bias_kids_btn.setText("Bias KIDs")
        
        # Clean up the task
        self.bias_kids_task = None
