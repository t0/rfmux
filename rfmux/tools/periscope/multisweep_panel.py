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
    ScreenshotMixin, mag_axis_label, make_tab_title,
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

    # Signal emitted when IQ rotation angles have been computed after Apply Bias
    iq_rotation_ready = pyqtSignal(int, dict)   # module, {code: angle_radians}

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

        # "Show Bias Info" checkbox — enabled/checked when bias data is available
        self.show_bias_info_cb = None  # Checkbox for toggling bias overlays

        # Bias frequency overlay lines (shown after Find Bias completes)
        # These are InfiniteLine items added to the combined plots to mark the
        # refined bias frequency for each detector.  Stored so they can be
        # removed when _redraw_combined_plots is next called.
        self.bias_freq_lines_mag = []    # InfiniteLine items on combined mag plot
        self.bias_freq_lines_phase = []  # InfiniteLine items on combined phase plot
        
        # Bias task storage
        self.bias_kids_output = None     # Stores the output from apply_bias (for export)
        self.nco_frequency_hz = None     # NCO frequency used when biasing (stored for export)
        self.find_bias_task = None       # Active FindBiasTask, if any
        self.find_bias_signals = None    # FindBiasSignals instance
        self.apply_bias_task = None      # Active ApplyBiasTask, if any
        self.apply_bias_signals = None   # ApplyBiasSignals instance
        self.iq_rotation_task = None     # Active ComputeIQRotationTask, if any
        self.iq_rotation_signals = None  # IQRotationSignals instance
        self.bias_settings_panel = None  # Lazy-initialized BiasSettingsPanel instance

        # Fit task storage
        self.run_fits_task    = None     # Active RunFitsTask, if any
        self.run_fits_signals = None     # RunFitsSignals instance
        self.fit_settings_panel = None   # Lazy-initialized FitSettingsPanel instance

        # Initialize batch tracking for sweep tabs (before _setup_ui)
        self.current_batch = 0

        self.batch_size = 8

        # Sort order for aggregate panel grids: "frequency" (default) or "name"
        self.sort_order = "frequency"
        
        # Storage for sweep grid plots - cached to avoid recreating widgets
        self.mag_sweep_plots_cache = []        # List of plot widgets for magnitude tab
        self.iq_sweep_plots_cache = []         # List of plot widgets for IQ tab
        self.derivative_plots_cache = []       # List of plot widgets for IQ Derivatives tab

        # Fit Results tab (Tab 6) — optional, enabled via Fit Settings
        self.fit_results_tab = None
        self.fit_results_grid = None
        self.fit_results_plots_cache = []
        self.fit_results_tab_index = 6
        # Tab 6 display controls (initialized in _setup_plot_area)
        self._fit_display_mode_rb_index = None
        self._fit_display_mode_rb_bias = None
        self._fit_display_index_combo = None
        self._fit_show_skewed_cb = None
        self._fit_show_nonlinear_cb = None

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
        """Creates and configures the two-row toolbar with controls (using QWidget instead of QToolBar).

        Top row — workflow actions (run, bias, export/screenshot).
        Bottom row — view and navigation controls (batch, units, normalization, zoom).
        """
        # Outer container holds both rows stacked vertically
        toolbar = QtWidgets.QWidget()
        toolbar_vbox = QtWidgets.QVBoxLayout(toolbar)
        toolbar_vbox.setContentsMargins(5, 3, 5, 3)
        toolbar_vbox.setSpacing(2)

        # ── Row 1: Workflow / Action buttons ──────────────────────────────────
        row1 = QtWidgets.QWidget()
        row1_layout = QtWidgets.QHBoxLayout(row1)
        row1_layout.setContentsMargins(0, 0, 0, 0)
        row1_layout.setSpacing(4)

        # Export Data Button
        self.export_btn = QtWidgets.QPushButton("💾")
        self.export_btn.setToolTip("Export data")
        self.export_btn.clicked.connect(self._export_data)
        row1_layout.addWidget(self.export_btn)

        # Screenshot button (kept next to Export — both save output)
        screenshot_btn = QtWidgets.QPushButton("📷")
        screenshot_btn.setToolTip("Export a screenshot of this panel to the session folder (or choose location)")
        screenshot_btn.clicked.connect(self._export_screenshot)
        row1_layout.addWidget(screenshot_btn)

        # Re-run Multisweep Button
        self.rerun_btn = QtWidgets.QPushButton("Re-run Multisweep")
        self.rerun_btn.clicked.connect(self._rerun_multisweep)
        row1_layout.addWidget(self.rerun_btn)

        # Run Fit button — runs the selected fitting algorithms on existing data
        self.run_fit_btn = QtWidgets.QPushButton("Run Fit")
        self.run_fit_btn.clicked.connect(self._run_fit)
        self.run_fit_btn.setToolTip(
            "Re-run resonator fitting algorithms on the existing multisweep data.\n"
            "Useful when the original sweep was run without fitting, or to re-fit\n"
            "with different settings.  Configure which fits to apply via ⚙ Fit Settings."
        )
        row1_layout.addWidget(self.run_fit_btn)

        # Fit Settings button — opens persistent FitSettingsPanel
        self.fit_settings_btn = QtWidgets.QPushButton("⚙ Fit Settings")
        self.fit_settings_btn.clicked.connect(self._show_fit_settings)
        self.fit_settings_btn.setToolTip("Open the fitting settings panel")
        row1_layout.addWidget(self.fit_settings_btn)

        # Find Bias button
        self.find_bias_btn = QtWidgets.QPushButton("Find Bias")
        self.find_bias_btn.clicked.connect(self._find_bias)
        self.find_bias_btn.setToolTip(
            "Analyse multisweep data to identify the optimal bias amplitude\n"
            "and frequency for each resonator (updates res_info_dict).\n"
            "Inspect the results before applying to hardware."
        )
        row1_layout.addWidget(self.find_bias_btn)

        # Apply Bias button — disabled until Find Bias has completed
        self.apply_bias_btn = QtWidgets.QPushButton("Apply Bias")
        self.apply_bias_btn.clicked.connect(self._apply_bias)
        self.apply_bias_btn.setEnabled(False)
        self.apply_bias_btn.setToolTip(
            "Programme the CRS hardware channels with the bias conditions\n"
            "found by Find Bias.  Only available after Find Bias succeeds."
        )
        row1_layout.addWidget(self.apply_bias_btn)

        # Bias Settings button — opens persistent settings panel
        self.bias_settings_btn = QtWidgets.QPushButton("⚙ Bias Settings")
        self.bias_settings_btn.clicked.connect(self._show_bias_settings)
        self.bias_settings_btn.setToolTip("Open the bias-finding settings panel")
        row1_layout.addWidget(self.bias_settings_btn)

        # Get Noise Spectrum button
        self.noise_spectrum_btn = QtWidgets.QPushButton("Get Noise Spectrum")
        self.noise_spectrum_btn.setEnabled(self.bias_data_avail)
        self.noise_spectrum_btn.setToolTip(
            "Open a dialog to configure and get the noise spectrum, will only work if KIDS is biased."
        )
        self.noise_spectrum_btn.clicked.connect(self._open_noise_spectrum_dialog)
        row1_layout.addWidget(self.noise_spectrum_btn)

        # Transient "Find Bias" status label — hidden until Find Bias completes,
        # then shown briefly before auto-hiding after 5 s.
        self._bias_status_label = QtWidgets.QLabel("✓ Bias found")
        self._bias_status_label.setStyleSheet(
            "color: #2a8a2a; font-weight: bold; padding: 2px 6px;"
        )
        self._bias_status_label.hide()
        row1_layout.addWidget(self._bias_status_label)

        # Transient "Apply Bias" status label — hidden until Apply Bias completes,
        # then shown briefly before auto-hiding after 5 s.
        self._apply_bias_status_label = QtWidgets.QLabel("✓ Bias applied")
        self._apply_bias_status_label.setStyleSheet(
            "color: #2a8a2a; font-weight: bold; padding: 2px 6px;"
        )
        self._apply_bias_status_label.hide()
        row1_layout.addWidget(self._apply_bias_status_label)

        # Transient "Fits complete" status label — hidden until Run Fit completes,
        # then shown briefly before auto-hiding after 5 s.
        self._fits_status_label = QtWidgets.QLabel("✓ Fits complete")
        self._fits_status_label.setStyleSheet(
            "color: #2a8a2a; font-weight: bold; padding: 2px 6px;"
        )
        self._fits_status_label.hide()
        row1_layout.addWidget(self._fits_status_label)

        # Transient "IQ rotation computed" status label — shown briefly after
        # ComputeIQRotationTask completes following Apply Bias.
        self._iq_rotation_status_label = QtWidgets.QLabel("✓ IQ rotation computed")
        self._iq_rotation_status_label.setStyleSheet(
            "color: #2a8a2a; font-weight: bold; padding: 2px 6px;"
        )
        self._iq_rotation_status_label.hide()
        row1_layout.addWidget(self._iq_rotation_status_label)

        row1_layout.addStretch(1)
        toolbar_vbox.addWidget(row1)

        # ── Horizontal separator ──────────────────────────────────────────────
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        toolbar_vbox.addWidget(separator)

        # ── Row 2: View / Navigation controls ────────────────────────────────
        row2 = QtWidgets.QWidget()
        row2_layout = QtWidgets.QHBoxLayout(row2)
        row2_layout.setContentsMargins(0, 0, 0, 0)
        row2_layout.setSpacing(4)

        # Batch navigation controls (for sweep tabs)
        self.batch_label = QtWidgets.QLabel("Batch:")
        row2_layout.addWidget(self.batch_label)

        self.prev_batch_btn = QtWidgets.QPushButton("◀")
        self.prev_batch_btn.setToolTip("Previous batch")
        self.prev_batch_btn.setMaximumWidth(30)
        self.prev_batch_btn.clicked.connect(self._prev_batch)
        row2_layout.addWidget(self.prev_batch_btn)

        self.batch_info_label = QtWidgets.QLabel("1 of 1")
        self.batch_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.batch_info_label.setMinimumWidth(40)
        row2_layout.addWidget(self.batch_info_label)

        self.next_batch_btn = QtWidgets.QPushButton("▶")
        self.next_batch_btn.setToolTip("Next batch")
        self.next_batch_btn.setMaximumWidth(30)
        self.next_batch_btn.clicked.connect(self._next_batch)
        row2_layout.addWidget(self.next_batch_btn)

        self.batch_size_label = QtWidgets.QLabel("Subplots:")
        row2_layout.addWidget(self.batch_size_label)

        self.batch_size_spin = QtWidgets.QSpinBox()
        self.batch_size_spin.setRange(1, 200)
        self.batch_size_spin.setValue(self.batch_size)
        self.batch_size_spin.setSingleStep(1)
        self.batch_size_spin.setToolTip("Detectors per batch (press Update to apply)")
        row2_layout.addWidget(self.batch_size_spin)

        self.batch_update_btn = QtWidgets.QPushButton("Update")
        self.batch_update_btn.setToolTip("Apply new batch size and regenerate plots")
        self.batch_update_btn.clicked.connect(self._apply_batch_size)
        row2_layout.addWidget(self.batch_update_btn)

        # Sort order selector — placed next to "Subplots" controls
        self.sort_order_label = QtWidgets.QLabel("Sort:")
        row2_layout.addWidget(self.sort_order_label)

        self.sort_order_combo = QtWidgets.QComboBox()
        self.sort_order_combo.addItem("By Frequency", "frequency")
        self.sort_order_combo.addItem("By Name", "name")
        self.sort_order_combo.setCurrentIndex(0)   # Default: by frequency
        self.sort_order_combo.setToolTip(
            "Sort order for resonator subplots in the aggregate panel pages.\n"
            "'By Frequency' orders subplots by ascending central frequency.\n"
            "'By Name' uses alphabetical ordering by resonator code."
        )
        self.sort_order_combo.currentIndexChanged.connect(self._on_sort_order_changed)
        row2_layout.addWidget(self.sort_order_combo)

        # Spacer to push view toggles to the right
        row2_layout.addStretch(1)

        # Normalization Checkbox
        self.normalize_checkbox = QtWidgets.QCheckBox("Normalize Traces")
        self.normalize_checkbox.setChecked(self.normalize_traces)
        self.normalize_checkbox.toggled.connect(self._toggle_trace_normalization)
        row2_layout.addWidget(self.normalize_checkbox)

        # Show Bias Info Checkbox — disabled until bias data is available
        self.show_bias_info_cb = QtWidgets.QCheckBox("Show Bias Info")
        self.show_bias_info_cb.setChecked(False)
        self.show_bias_info_cb.setEnabled(False)
        self.show_bias_info_cb.setToolTip(
            "Toggle bias overlays: highlighted bias amplitude sweep, bias frequency\n"
            "vertical lines, and ★/a-value legend details.\n"
            "Available after Find Bias has been run (or when bias info is loaded from file)."
        )
        self.show_bias_info_cb.toggled.connect(self._toggle_bias_info_visibility)
        row2_layout.addWidget(self.show_bias_info_cb)

        # Show Rotated IQ Checkbox — disabled until IQ rotation angles are available.
        # Only affects the IQ Circles tab (Tab 1); has no effect on other tabs.
        self.show_rotated_iq_cb = QtWidgets.QCheckBox("Show Rotated IQ")
        self.show_rotated_iq_cb.setChecked(False)
        self.show_rotated_iq_cb.setEnabled(False)
        self.show_rotated_iq_cb.setToolTip(
            "Rotate the IQ sweep circles so that the signal-sensitive direction\n"
            "aligns with the Q axis.  Only visible on the IQ Circles tab.\n"
            "Available after Apply Bias has been run (or when a file with stored\n"
            "IQ rotation angles is loaded)."
        )
        self.show_rotated_iq_cb.toggled.connect(self._on_show_rotated_iq_changed)
        row2_layout.addWidget(self.show_rotated_iq_cb)

        self._setup_unit_controls(row2_layout)
        self._setup_zoom_box_control(row2_layout)

        toolbar_vbox.addWidget(row2)

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
        self.plot_tabs.addTab(self.mag_sweeps_tab, "Mag vs Freq")
        
        # Tab 1: IQ Circles (per-detector grid)
        self.iq_sweeps_tab, self.iq_sweeps_grid, self.iq_colorbar = self._create_sweep_tab()
        self.plot_tabs.addTab(self.iq_sweeps_tab, "IQ Circles")
        
        # Tab 2: Combined Plots (original combined view)
        self.combined_tab = self._create_combined_tab()
        self.plot_tabs.addTab(self.combined_tab, "Mag, Phase Overview")
        
        # Tab 3: Histograms (parameter distributions)
        self.histogram_tab = self._create_histogram_tab()
        self.plot_tabs.addTab(self.histogram_tab, "Histograms")
        

        # Tab 4: Detector Digest (single-detector detail view)
        self.digest_tab = self._create_digest_tab()
        self.plot_tabs.addTab(self.digest_tab, "Detector Digest")

        # Tab 5: IQ Derivatives (shown after Find Bias)
        self.derivative_tab, self.derivative_grid = self._create_derivative_tab()
        self.plot_tabs.addTab(self.derivative_tab, "IQ Derivatives")

        # Tab 6: Fit Results (always visible; shows placeholder until fits have been run)
        self.fit_results_tab, self.fit_results_grid = self._create_fit_results_tab()
        self.plot_tabs.addTab(self.fit_results_tab, "Fit Results")

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
        placeholder = QtWidgets.QLabel("Histograms will appear once fits have been run on the selected multisweep file.")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: gray; font-style: italic;")
        tab_layout.addWidget(placeholder)
        
        return tab
    

    def _create_derivative_tab(self):
        """Create the IQ Derivatives tab (populated after Find Bias runs).

        Returns
        -------
        (tab, grid_layout)
            The QWidget tab and its inner QGridLayout for derivative subplots.
        """
        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.setContentsMargins(5, 5, 5, 5)

        # Placeholder shown before Find Bias has been run
        self._derivative_placeholder = QtWidgets.QLabel(
            "IQ derivative plots will appear here after Find Bias has been run.\n"
            "Each subplot shows I speed, Q speed, and arc-length speed for the\n"
            "selected bias amplitude, with the cubic-spline fit overlaid."
        )
        self._derivative_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._derivative_placeholder.setStyleSheet("color: gray; font-style: italic;")
        tab_layout.addWidget(self._derivative_placeholder)

        # Scroll area + grid (identical structure to sweep tabs)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(container)
        grid.setSpacing(10)
        scroll.setWidget(container)

        # The scroll area is hidden until Find Bias populates it
        scroll.hide()
        self._derivative_scroll = scroll
        tab_layout.addWidget(scroll)

        return tab, grid

    def _create_fit_results_tab(self):
        """Create the Fit Results tab (Tab 6).

        Shows per-detector magnitude sweeps overlaid with fitted model curves.
        Optional: enabled via the "Show Fit Results Tab" checkbox in Fit Settings.
        Hidden by default until both enabled and fit data are present.

        Returns
        -------
        (tab, grid_layout)
            The QWidget tab and its inner QGridLayout for fit-result subplots.
        """
        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.setContentsMargins(5, 5, 5, 5)
        tab_layout.setSpacing(4)

        # ── Controls row ──────────────────────────────────────────────────────
        controls_row = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls_row)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        # Amplitude display mode
        amp_label = QtWidgets.QLabel("Show amplitude:")
        controls_layout.addWidget(amp_label)

        self._fit_display_mode_rb_index = QtWidgets.QRadioButton("Index:")
        self._fit_display_mode_rb_index.setChecked(True)
        self._fit_display_mode_rb_index.setToolTip(
            "Show the amplitude at sorted position N (0 = lowest)."
        )
        controls_layout.addWidget(self._fit_display_mode_rb_index)

        self._fit_display_index_combo = QtWidgets.QComboBox()
        self._fit_display_index_combo.setMinimumWidth(60)
        self._fit_display_index_combo.setToolTip(
            "Select an amplitude index (0 = lowest).\n"
            "Only indices that have fit data are shown."
        )
        self._fit_display_index_combo.addItem("—")  # placeholder until data arrives
        self._fit_display_index_combo.setEnabled(True)
        controls_layout.addWidget(self._fit_display_index_combo)

        self._fit_display_mode_rb_bias = QtWidgets.QRadioButton("Bias amplitude")
        self._fit_display_mode_rb_bias.setToolTip(
            "Show the bias-amplitude iteration (requires Find Bias to have run)."
        )
        controls_layout.addWidget(self._fit_display_mode_rb_bias)

        # Vertical separator
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        sep.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        controls_layout.addWidget(sep)

        # Model overlay toggles
        overlay_label = QtWidgets.QLabel("Show:")
        controls_layout.addWidget(overlay_label)

        self._fit_show_skewed_cb = QtWidgets.QCheckBox("Skewed fit")
        self._fit_show_skewed_cb.setChecked(True)
        controls_layout.addWidget(self._fit_show_skewed_cb)

        self._fit_show_nonlinear_cb = QtWidgets.QCheckBox("Nonlinear fit")
        self._fit_show_nonlinear_cb.setChecked(True)
        controls_layout.addWidget(self._fit_show_nonlinear_cb)

        controls_layout.addStretch(1)
        tab_layout.addWidget(controls_row)

        # ── Connect controls → redraw ──────────────────────────────────────────
        self._fit_display_mode_rb_index.toggled.connect(self._on_fit_results_controls_changed)
        self._fit_display_mode_rb_bias.toggled.connect(self._on_fit_results_controls_changed)
        self._fit_display_index_combo.currentIndexChanged.connect(self._on_fit_results_controls_changed)
        self._fit_show_skewed_cb.toggled.connect(self._on_fit_results_controls_changed)
        self._fit_show_nonlinear_cb.toggled.connect(self._on_fit_results_controls_changed)

        # ── Placeholder shown before any fits have been run ───────────────────
        self._fit_results_placeholder = QtWidgets.QLabel(
            "Plots will appear here once fits have been run on the selected multisweep file."
        )
        self._fit_results_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._fit_results_placeholder.setStyleSheet("color: gray; font-style: italic;")
        tab_layout.addWidget(self._fit_results_placeholder)

        # ── Scroll area + grid ────────────────────────────────────────────────
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(container)
        grid.setSpacing(10)
        scroll.setWidget(container)

        # Hidden until fit data is available
        scroll.hide()
        self._fit_results_scroll = scroll
        tab_layout.addWidget(scroll)

        return tab, grid

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
                if det_data.get('fits'):
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
        # Batch controls visible for sweep tabs (0, 1), IQ Derivatives (5), and Fit Results (6)
        is_sweep_tab = index in (0, 1, 5, 6)
        
        self.batch_label.setVisible(is_sweep_tab)
        self.prev_batch_btn.setVisible(is_sweep_tab)
        self.batch_info_label.setVisible(is_sweep_tab)
        self.next_batch_btn.setVisible(is_sweep_tab)

        self.batch_size_label.setVisible(is_sweep_tab)
        self.batch_size_spin.setVisible(is_sweep_tab)
        self.batch_update_btn.setVisible(is_sweep_tab)

        # Sort order selector is only relevant for the aggregate grid tabs
        self.sort_order_label.setVisible(is_sweep_tab)
        self.sort_order_combo.setVisible(is_sweep_tab)
        
        # When switching to the Fit Results tab, refresh the amplitude index dropdown
        # so it reflects whatever fit data is currently present.
        if index == 6 and self.results_by_detector:
            self._populate_fit_amplitude_combo()

        # Redraw the active tab's plots if we have data
        if self.results_by_detector:
            self._redraw_plots()

    def _on_sort_order_changed(self, index: int):
        """Slot for the sort order combo box.

        Updates ``self.sort_order``, resets to the first batch (so the
        newly-ordered pages start from page 1), and triggers a full redraw of
        the currently visible aggregate grid tab.
        """
        self.sort_order = self.sort_order_combo.currentData()
        self.current_batch = 0  # Reset to first page after re-sorting
        self._redraw_plots()

    def _get_sorted_detector_ids(self, detector_ids_iterable, freq_lookup_func) -> list:
        """Return detector IDs ordered according to ``self.sort_order``.

        Parameters
        ----------
        detector_ids_iterable:
            Iterable of detector ID keys (strings or integers).
        freq_lookup_func:
            Callable ``(detector_id) -> float | None`` — returns the central
            frequency in Hz for the given detector, or ``None`` when the
            frequency is not yet known.  Detectors with no frequency are
            appended at the end, sorted alphabetically among themselves.

        Returns
        -------
        list
            Detector IDs in the chosen order.
        """
        ids = list(detector_ids_iterable)

        if self.sort_order == "name":
            return sorted(ids, key=lambda k: str(k))

        # sort_order == "frequency": ascending by central frequency
        with_freq = []
        without_freq = []
        for det_id in ids:
            freq = freq_lookup_func(det_id)
            if freq is not None:
                with_freq.append((float(freq), det_id))
            else:
                without_freq.append(det_id)

        with_freq.sort(key=lambda t: t[0])
        without_freq.sort(key=lambda k: str(k))
        return [det_id for _, det_id in with_freq] + without_freq
    
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
        if current_tab_idx not in (0, 1, 5, 6):  # Only sweep tabs, IQ Derivatives, and Fit Results have batches
            return
            
        if not self.results_by_detector:
            return
        
        # For the derivative tab, count only detectors that have bias_finding data
        if current_tab_idx == 5:
            num_detectors = sum(
                1 for iter_dict in self.results_by_detector.values()
                if any('bias_finding' in entry for entry in iter_dict.values())
            )
        else:
            # Tabs 0, 1, 6: all detectors
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
        """Applies the current zoom_box_mode state to all plot viewboxes.

        Covers:
        - The Combined Plots tab (combined_mag_plot / combined_phase_plot), which use
          ClickableViewBox and have an ``enableZoomBoxMode`` helper.
        - All cached grid plot widgets (Magnitude Sweeps, IQ Circles, IQ Derivatives
          tabs), which use a plain ``pg.ViewBox`` and are updated via ``setMouseMode``
          directly.
        """
        mode = pg.ViewBox.RectMode if self.zoom_box_mode else pg.ViewBox.PanMode

        # Combined Plots tab — ClickableViewBox
        if self.combined_mag_plot and isinstance(self.combined_mag_plot.getViewBox(), ClickableViewBox):
            self.combined_mag_plot.getViewBox().enableZoomBoxMode(self.zoom_box_mode)
        if self.combined_phase_plot and isinstance(self.combined_phase_plot.getViewBox(), ClickableViewBox):
            self.combined_phase_plot.getViewBox().enableZoomBoxMode(self.zoom_box_mode)

        # Grid plot caches (tabs 0, 1, 5, 6) — plain pg.ViewBox
        all_caches = (
            self.mag_sweep_plots_cache
            + self.iq_sweep_plots_cache
            + self.derivative_plots_cache
            + self.fit_results_plots_cache
        )
        for pw in all_caches:
            vb = pw.getViewBox()
            if vb is not None:
                vb.setMouseMode(mode)

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
            dac_scale = self.dac_scales.get(self.target_module)
            for detector_id, det_data in multisweep_data_dict.items():
                if detector_id not in self.results_by_detector:
                    self.results_by_detector[detector_id] = {}
                entry = dict(det_data)
                entry['iteration'] = iteration
                # Inject sweep_power_dbm so the live dict is self-contained.
                # iq_volts is already present (returned by the multisweep algorithm).
                norm_amp = entry.get('sweep_amplitude_normalized')
                entry['sweep_power_dbm'] = (
                    UnitConverter.normalize_to_dbm(norm_amp, dac_scale)
                    if norm_amp is not None and dac_scale is not None
                    else None
                )
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

    def _get_expected_amplitudes(self) -> set:
        """Return the full set of amplitude values expected for this sweep.

        Flattens every per-section amplitude array from ``initial_params`` so
        that the colour map and colourbar threshold are computed against the
        complete global range — from the smallest probe amplitude used on any
        section to the largest — right from the very first iteration.

        Falls back to the flat ``amps`` list for sweeps that use a common
        amplitude across all sections, and absorbs any already-observed values
        for robustness (e.g. panels loaded from file where ``initial_params``
        may not contain the original ``amp_arrays``).
        """
        expected: set = set()

        # Flatten all values from all per-section amplitude arrays.
        # amp_arrays[i] is the full sequence of amplitudes swept for section i.
        for arr in self.initial_params.get('amp_arrays', []):
            try:
                expected.update(arr)
            except TypeError:
                pass  # guard against non-iterable entries

        # Flat-list fallback (used when all sections share the same amplitude schedule)
        if not expected:
            expected.update(self.initial_params.get('amps', []))

        # Also absorb any already-observed amplitudes so loaded/re-run panels
        # always produce a correct reference set even when initial_params is stale.
        for iter_dict in self.results_by_detector.values():
            for entry in iter_dict.values():
                amp = entry.get('sweep_amplitude_normalized')
                if amp is not None:
                    expected.add(amp)

        return expected

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
        # Tab 5: IQ Derivatives (populated after Find Bias)
        elif current_tab_idx == 5:
            self._redraw_derivative_grid()
        # Tab 6: Fit Results (optional, enabled via Fit Settings)
        elif current_tab_idx == 6:
            self._redraw_fit_results_grid()
    
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
                amp_val = det_entry.get('sweep_amplitude_normalized')
                direction = det_entry.get('sweep_direction', 'upward')
                freqs = det_entry.get('frequencies', np.array([]))
                # Use raw counts when in counts mode, otherwise voltage-converted data
                if self.unit_mode == "counts":
                    iq_counts = det_entry.get('iq_counts', np.array([]))
                else:
                    iq_counts = det_entry.get('iq_counts', np.array([]))
                if amp_val is not None and len(freqs) > 0 and len(iq_counts) > 0:
                    detector_data[detector_id][(amp_val, direction)] = {
                        'freq': freqs,
                        'iq': iq_counts,
                        'amplitude': amp_val,
                        'direction': direction,
                        'original_center_frequency': (
                            det_entry.get('original_center_frequency')
                            or det_entry.get('sweep_center_frequency')
                        ),
                        'nonlinear_fit_success': det_entry.get('fits', {}).get('nonlinear', {}).get('nonlinear_fit_success', False),
                        'nonlinear_fit_params': det_entry.get('fits', {}).get('nonlinear', {}).get('nonlinear_fit_params'),
                    }
        
        if not detector_data:
            return
        

        # Get all amplitudes for color mapping (unique amplitude values only)
        all_amps = set()
        for det_data in detector_data.values():
            for (amp_val, _direction) in det_data.keys():
                all_amps.add(amp_val)
        
        # Build the full expected amplitude set (known from initial_params) so that
        # the colour map and colourbar threshold are stable across all iterations.
        expected_amps = self._get_expected_amplitudes()

        # Create amplitude color mapping — pass expected_amps as the reference so
        # colour positions are assigned based on the total sweep, not just what has
        # arrived so far.
        amplitude_to_color = create_amplitude_color_map(
            all_amps, self.dark_mode, reference_amplitudes=expected_amps
        )

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
        has_downward = any(d == 'downward' for _, d in all_sweep_keys)
        
        # Use total expected amplitude count for the threshold so that the
        # legend-vs-colourbar decision is made once up-front and doesn't flip
        # mid-sweep as more iterations arrive.
        num_amps_expected = len(expected_amps)
        if num_amps_expected > AMPLITUDE_COLORMAP_THRESHOLD:
            # Colourbar range is based on the expected global min/max.
            sorted_amps = sorted(expected_amps)
            colorbar.update_range(sorted_amps[0], sorted_amps[-1],
                                  dac_scale, self.unit_mode,
                                  self.dark_mode, has_downward)
            colorbar.show()
            use_legend = False  # colorbar replaces per-plot legends
        else:
            colorbar.hide()
            use_legend = True  # show per-plot legends
        
        # Pass res_info_dict only when the "Show Bias Info" checkbox is checked;
        # otherwise pass None so the grid helpers suppress all bias overlays.
        show_bias_info = (
            self.show_bias_info_cb is not None and self.show_bias_info_cb.isChecked()
        )

        # Build IQ rotation angles dict for the grid helper (IQ tab only).
        # Only pass angles when the "Show Rotated IQ" checkbox is active.
        iq_rotation_angles_for_grid = None
        if (tab_idx == 1
                and hasattr(self, 'show_rotated_iq_cb')
                and self.show_rotated_iq_cb is not None
                and self.show_rotated_iq_cb.isChecked()
                and self.res_info_dict):
            iq_rotation_angles_for_grid = {
                code: info['iq_rotation_angle']
                for code, info in self.res_info_dict.items()
                if 'iq_rotation_angle' in info
            } or None  # Return None rather than empty dict

        # Compute detector order according to the current sort preference.
        # For the sweep grid the center frequency is stored inside detector_data.
        def _sweep_center_freq(det_id):
            det_d = detector_data.get(det_id, {})
            if not det_d:
                return None
            first_entry = next(iter(det_d.values()), {})
            return first_entry.get('original_center_frequency')

        sorted_ids = self._get_sorted_detector_ids(detector_data.keys(), _sweep_center_freq)

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
            show_legend=use_legend,
            res_info_dict=self.res_info_dict if show_bias_info else None,
            iq_rotation_angles=iq_rotation_angles_for_grid,
            sorted_detector_ids=sorted_ids,
        )
        
        # Install double-click event filter on grid plot widgets
        # Must be AFTER update_sweep_grid so newly created widgets are included
        for pw in widget_cache:
            pw.installEventFilter(self)

        # Apply the current zoom box mode to all (potentially new) grid widgets.
        # update_sweep_grid may have created fresh pg.PlotWidget instances that
        # default to pyqtgraph's built-in PanMode; we must set them explicitly.
        mode = pg.ViewBox.RectMode if self.zoom_box_mode else pg.ViewBox.PanMode
        for pw in widget_cache:
            vb = pw.getViewBox()
            if vb is not None:
                vb.setMouseMode(mode)
    
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

        # Remove existing bias frequency overlay lines
        for line in self.bias_freq_lines_mag:
            self.combined_mag_plot.removeItem(line)
        self.bias_freq_lines_mag.clear()
        for line in self.bias_freq_lines_phase:
            self.combined_phase_plot.removeItem(line)
        self.bias_freq_lines_phase.clear()

        if not self.results_by_detector:
            self.combined_mag_plot.autoRange(); self.combined_phase_plot.autoRange()
            return

        # Bias overlay visibility is gated on the "Show Bias Info" checkbox.
        show_bias_info = (
            self.show_bias_info_cb is not None and self.show_bias_info_cb.isChecked()
        )

        # --- Determine which amplitudes are "chosen" by Find Bias ---
        # An amplitude is "chosen" if at least one detector has bias_found=True
        # and its bias_amplitude matches.  Only populated when bias overlays are on.
        chosen_amplitudes: set = set()
        if show_bias_info:
            for info in self.res_info_dict.values():
                if info.get('bias_found') and info.get('bias_amplitude') is not None:
                    chosen_amplitudes.add(info['bias_amplitude'])

        # Collect unique (amplitude, direction) pairs and amplitude values from entries
        amplitude_values = set()
        amp_dir_pairs = set()
        for iter_dict in self.results_by_detector.values():
            for entry in iter_dict.values():
                amp = entry.get('sweep_amplitude_normalized')
                direction = entry.get('sweep_direction', 'upward')
                if amp is not None:
                    amplitude_values.add(amp)
                    amp_dir_pairs.add((amp, direction))
        num_amps = len(amplitude_values)
            
        # Build the full expected amplitude set so colour positions and the
        # colourbar threshold are stable from the very first iteration.
        expected_amps = self._get_expected_amplitudes()

        # Create a mapping for unique amplitude values to colors
        amplitude_to_color = create_amplitude_color_map(
            amplitude_values, self.dark_mode, reference_amplitudes=expected_amps
        )

        # Colorbar vs legend: base the decision on the total expected count so
        # the display mode doesn't flip mid-sweep as iterations arrive.
        has_downward = any(d == 'downward' for _, d in amp_dir_pairs)
        dac_scale_for_module = self.dac_scales.get(self.active_module_for_dac)
        num_amps_expected = len(expected_amps)

        if num_amps_expected > AMPLITUDE_COLORMAP_THRESHOLD:
            # Colourbar range spans the expected global min/max.
            sorted_expected = sorted(expected_amps)
            self.combined_colorbar.update_range(
                sorted_expected[0], sorted_expected[-1],
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

            # --- Prepare Legend Entry for this Amplitude (only if legends active) ---
            if show_combined_legend:
                legend_name_amp = UnitConverter.format_probe_label(amp_val, self.unit_mode, dac_scale_for_module)
                direction_suffix = " (Down)" if direction == "downward" else " (Up)"
                # Append chosen indicator to the legend label when this amplitude was selected
                chosen_suffix = " ★" if amp_val in chosen_amplitudes else ""
                full_legend_name = legend_name_amp + direction_suffix + chosen_suffix

                # Use the chosen line width for the legend swatch too
                legend_width = LINE_WIDTH * 2 if amp_val in chosen_amplitudes else LINE_WIDTH
                legend_pen = pg.mkPen(color, width=legend_width, style=line_style)

                legend_key = (amp_val, direction)
                
                if legend_key not in legend_items_mag and self.mag_legend:
                    dummy_mag_curve_for_legend = pg.PlotDataItem(pen=legend_pen)
                    self.mag_legend.addItem(dummy_mag_curve_for_legend, full_legend_name)
                    legend_items_mag[legend_key] = dummy_mag_curve_for_legend
                if legend_key not in legend_items_phase and self.phase_legend:
                    dummy_phase_curve_for_legend = pg.PlotDataItem(pen=legend_pen)
                    self.phase_legend.addItem(dummy_phase_curve_for_legend, full_legend_name)
                    legend_items_phase[legend_key] = dummy_phase_curve_for_legend

            # --- Plot data for each detector at this (amplitude, direction) ---
            for res_idx, iter_dict in self.results_by_detector.items():
                # Find the entry matching this amplitude and direction
                data = None
                for entry in iter_dict.values():
                    if entry.get('sweep_amplitude_normalized') == amp_val and entry.get('sweep_direction', 'upward') == direction:
                        data = entry
                        break
                if data is None:
                    continue

                freqs_hz = data.get('frequencies', np.array([]))
                iq_counts = data.get('iq_counts', np.array([]))

                if freqs_hz is None or iq_counts is None or len(freqs_hz) == 0 or len(iq_counts) == 0:
                    continue

                # Thicken the line for this detector's chosen bias amplitude
                # (only when bias overlays are enabled via the checkbox)
                det_info = self.res_info_dict.get(res_idx, {})
                is_chosen_for_det = (
                    show_bias_info
                    and det_info.get('bias_found', False)
                    and det_info.get('bias_amplitude') == amp_val
                )
                curve_width = LINE_WIDTH * 2 if is_chosen_for_det else LINE_WIDTH
                pen = pg.mkPen(color, width=curve_width, style=line_style)
                
                # Calculate magnitude and phase
                s21_mag_raw = np.abs(iq_counts)
                # Compute probe_amp_dbm for dBm normalization if DAC scale available
                probe_amp_dbm = None
                if self.normalize_traces and self.unit_mode == 'dbm' and dac_scale_for_module is not None:
                    probe_amp_dbm = UnitConverter.normalize_to_dbm(amp_val, dac_scale_for_module)
                s21_mag_processed = UnitConverter.convert_amplitude(
                    s21_mag_raw, iq_counts, self.unit_mode,
                    normalize=self.normalize_traces, probe_amp_dbm=probe_amp_dbm
                )
                # Use pre-calculated phase if available, otherwise calculate from IQ
                phase_deg = data.get('phase_degrees', np.degrees(np.angle(iq_counts)))
                
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

                # --- Bias frequency overlay line (shown when Find Bias has run) ---
                # One solid vertical line per detector at its refined bias frequency,
                # using the same color as the chosen amplitude's curves.
                if is_chosen_for_det:
                    bias_freq_refined = det_info.get('bias_frequency')
                    if bias_freq_refined is not None:
                        # Thin dashed red line for the combined plots so the
                        # indicator is clearly distinguishable from the sweep
                        # traces regardless of their color.
                        bias_line_pen = pg.mkPen(
                            'r',
                            style=QtCore.Qt.PenStyle.DashLine,
                            width=1,
                        )
                        mag_bias_line = pg.InfiniteLine(
                            pos=bias_freq_refined, angle=90,
                            pen=bias_line_pen, movable=False,
                        )
                        self.combined_mag_plot.addItem(mag_bias_line)
                        self.bias_freq_lines_mag.append(mag_bias_line)

                        phase_bias_line = pg.InfiniteLine(
                            pos=bias_freq_refined, angle=90,
                            pen=bias_line_pen, movable=False,
                        )
                        self.combined_phase_plot.addItem(phase_bias_line)
                        self.bias_freq_lines_phase.append(phase_bias_line)
        
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

    def _get_filename_override(self) -> str | None:
        """Return a ``_filename_override`` value derived from the measurement name.

        Reads ``measurement_name`` from ``initial_params`` (set when the
        MultisweepDialog is accepted).  Returns ``None`` when no custom name
        was specified so the session manager falls back to its own filename
        generation / de-duplication logic.
        """
        name = self.initial_params.get('measurement_name')
        return f"{name}.pkl" if name else None

    def all_sweeps_completed(self):
        """
        Slot called when all amplitudes in the multisweep have been processed.
        Updates UI elements to reflect completion and auto-opens detector digest for first detector.
        """
        self._check_all_complete()
        self.current_amp_label.setText("All Amplitudes Processed")
        
        # Emit data_ready signal for session auto-export.
        # The session_manager.handle_data_ready will automatically overwrite any
        # existing file for this (data_type, identifier) pair so that Find Bias
        # and Run Fit update the same file instead of creating timestamped duplicates.
        if self.results_by_detector:
            identifier = f"module{self.target_module}"
            export_data = self._prepare_export_data()
            filename_override = self._get_filename_override()
            if filename_override:
                export_data['_filename_override'] = filename_override
            self.data_ready.emit("multisweep", identifier, export_data)
        
        # Generate histogram plots once when all data is complete
        if self.results_by_detector and not self.histograms_generated:
            self._generate_histograms()
            self.histograms_generated = True
        
        # Update the "Show Bias Info" checkbox state in case loaded data already
        # contains bias results (e.g. a file saved after Find Bias was run).
        self._update_bias_info_checkbox_state()

        # Show or hide the Fit Results tab based on settings and whether the loaded
        # data already contains fit results (e.g. file saved after Run Fit was run).
        self._update_fit_results_tab_visibility()

        # Auto-populate detector digest for the first detector in insertion order.
        # Don't switch focus — user should stay on the current tab (magnitude sweeps)
        if self.results_by_detector:
            first_key = next(iter(self.results_by_detector))
            self._open_detector_digest_for_index(first_key, switch_to_tab=False)

        # Auto-run Find Bias if the user requested it in the multisweep dialog.
        # Deferred via QTimer so all GUI completion signals are processed first.
        if self.initial_params.get('run_find_bias', False):
            QtCore.QTimer.singleShot(0, self._find_bias)
        
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
                        amp = first_entry.get('sweep_amplitude_normalized')
                        if amp is not None:
                            section_amps.append(amp)

            if section_amps:
                export_initial_params['section_amplitudes'] = section_amps
            
        return {
            'measurement_type': 'multisweep',
            'timestamp': datetime.datetime.now().isoformat(),
            'target_module': self.target_module,
            'initial_parameters': export_initial_params,
            # Only store the DAC scale for the module that actually ran the measurement.
            'dac_scales_used': self.dac_scales.get(self.target_module),
            'res_info_dict': self.res_info_dict,        # Lightweight resonator registry
            'results': self.results_by_detector,
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

    def _get_fit_result_frequencies(self) -> list[float] | None:
        """Return the fitted resonance frequencies (fr) from fit results, or None if no
        fit results exist in the data.

        Iterates through res_info_dict codes (sorted by channel_number) and looks up
        ``fit_params`` or ``nonlinear_fit_params`` for each detector in
        ``results_by_detector``.  Falls back to ``bias_frequency`` for individual
        detectors that have no fit result, but only if at least ONE detector in the
        dataset has a real fit result.  Returns ``None`` when no fit results exist at
        all, so callers can distinguish "fits ran but gave fallback values" from "no
        fitter was run".

        Returns:
            List of frequencies in Hz (one per detector, ordered by channel_number),
            or ``None`` if no ``fit_params`` / ``nonlinear_fit_params`` entries are
            present anywhere in ``results_by_detector``.
        """
        if not self.res_info_dict:
            return None

        # First pass: check whether any detector has real fit results
        has_any_fit = any(
            entry.get('fits', {}).get('skewed', {}).get('fit_params')
            or entry.get('fits', {}).get('nonlinear', {}).get('nonlinear_fit_params')
            for iter_dict in self.results_by_detector.values()
            if iter_dict
            for entry in [next(iter(iter_dict.values()))]
        )
        if not has_any_fit:
            return None

        # Second pass: build the frequency list, using bias_frequency as a per-entry
        # fallback for the rare detectors that failed to fit
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
            _skewed = first_entry.get('fits', {}).get('skewed', {})
            _nl     = first_entry.get('fits', {}).get('nonlinear', {})
            if _skewed.get('skewed_fit_success') and _skewed.get('fit_params'):
                ref_freqs.append(_skewed['fit_params']['fr'])
            elif _nl.get('nonlinear_fit_success') and _nl.get('nonlinear_fit_params'):
                ref_freqs.append(_nl['nonlinear_fit_params']['fr'])
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

        # Fit frequencies: only offered in the dialog when real fit results exist.
        # Pass an empty list when no fitter was run so the dialog still enters
        # re-run mode (shows the combo + editable field), but omits the fit option.
        fit_freqs = self._get_fit_result_frequencies() or []

        # Original sweep-center frequencies from initial_params (used by "Use previous
        # multisweep central frequencies" option in the dialog).
        original_sweep_centers = list(
            self.initial_params.get('sweep_center_frequencies')
            or self.initial_params.get('resonance_frequencies', [])
        ) or dialog_seed_frequencies

        # Bias frequencies: the refined bias_frequency values stored in res_info_dict
        # (only meaningful after Find Bias has run; before that they equal the sweep centres).
        # Only offer bias frequencies in the re-run dialog if Find Bias has actually been
        # run on the current data (i.e. at least one entry has bias_found == True).
        # Before Find Bias runs, res_info_dict["bias_frequency"] just holds the sweep
        # centre frequencies, which are already offered as "previous multisweep central
        # frequencies" — showing them again as "bias frequencies" would be misleading.
        has_bias_found = any(
            info.get('bias_found', False) for info in self.res_info_dict.values()
        )
        bias_freqs_for_dialog = (
            dialog_seed_frequencies if (dialog_seed_frequencies and has_bias_found) else None
        )

        # Pass res_info_dict to the dialog so the "Multiplicative scaling" validator
        # can find it in self.params immediately on the very first re-run of a session.
        # Without this, res_info_dict is only injected into self.initial_params *after*
        # the dialog closes, so the first-ever re-run dialog never sees it.
        dialog_params = self.initial_params.copy()
        if self.res_info_dict:
            dialog_params['res_info_dict'] = self.res_info_dict

        dialog = MultisweepDialog(
            parent=self,
            section_center_frequencies=original_sweep_centers,
            dac_scales=self.dac_scales,
            current_module=self.target_module,
            initial_params=dialog_params,
            load_multisweep=False,
            fit_frequencies=fit_freqs,
            bias_frequencies=bias_freqs_for_dialog,
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

        # Clear stale Find Bias results from the previous run so that the new sweep
        # starts without any bias overlays.  The res_info_dict codes, channel numbers,
        # and bias_amplitude are preserved (bias_amplitude is required by the multisweep
        # algorithm's Option B path to know what amplitude to sweep at).  Only
        # bias_found is cleared because all overlay logic is gated on that flag:
        #   is_chosen_for_det = det_info.get('bias_found', False) and ...
        for info in self.res_info_dict.values():
            info.pop('bias_found', None)

        # Apply Bias is no longer valid until Find Bias is re-run on the new data.
        self.apply_bias_btn.setEnabled(False)

        # Apply parameters and inject res_info_dict so MultisweepTask re-uses existing codes
        self.initial_params.update(new_params)
        if self.res_info_dict:
            self.initial_params['res_info_dict'] = self.res_info_dict

        # Reset window state for the new sweep.
        # Reset export-file tracking so the new sweep creates a fresh session file
        # rather than overwriting the previous sweep's file.
        periscope_for_reset = self._get_periscope_parent()
        if periscope_for_reset and hasattr(periscope_for_reset, 'session_manager'):
            periscope_for_reset.session_manager.reset_export_tracking(
                "multisweep", f"module{self.target_module}"
            )
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

    def _toggle_bias_info_visibility(self, checked):
        """Slot for the 'Show Bias Info' checkbox.

        Triggers a full redraw so all tabs pick up the new overlay state.
        The checkbox controls:
          - Highlighted (thickened) traces for the chosen bias amplitude
          - Vertical bias-frequency lines on Combined and grid plots
          - ★ and nonlinearity-parameter (a) labels in legends

        Args:
            checked (bool): New checked state of the checkbox.
        """
        self._redraw_plots()

    def _update_bias_info_checkbox_state(self):
        """Enable/disable and auto-check the 'Show Bias Info' checkbox.

        Should be called whenever ``res_info_dict`` may have gained new
        ``bias_found`` entries (i.e. after Find Bias completes or after
        loading data that already contains bias results).

        Behaviour:
          - If at least one detector has ``bias_found=True``:
              enable the checkbox and check it (so overlays appear immediately).
          - Otherwise: uncheck and disable the checkbox.
        """
        if self.show_bias_info_cb is None:
            return
        has_bias = any(
            info.get('bias_found', False) for info in self.res_info_dict.values()
        )
        # Block signals while we update the checkbox state to avoid a spurious
        # redraw triggered by the programmatic setChecked call.
        self.show_bias_info_cb.blockSignals(True)
        self.show_bias_info_cb.setEnabled(has_bias)
        if has_bias:
            self.show_bias_info_cb.setChecked(True)
        else:
            self.show_bias_info_cb.setChecked(False)
        self.show_bias_info_cb.blockSignals(False)


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
                export_data['measurement_type'] = 'noise'
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
        
        # Increment counter and use for tab name — prefix with parent measurement name if available
        self.noise_panel_count += 1
        loaded_suffix = " (Loaded)" if self.is_loaded_data else ""
        parent_mname = self.initial_params.get('measurement_name', '')
        if parent_mname:
            _nname = f"Noise: {parent_mname}{loaded_suffix}"
        else:
            _nname = f"Noise Spectrum #{self.noise_panel_count}{loaded_suffix}"
        dock_title = make_tab_title(_nname)
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
                amp_val = det_entry.get('sweep_amplitude_normalized')
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
                amp_val = det_entry.get('sweep_amplitude_normalized')
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
    
    # ── Bias Settings ────────────────────────────────────────────────────────

    def _show_bias_settings(self):
        """Show (or create) the persistent BiasSettingsPanel window."""
        from .bias_kids_dialog import BiasSettingsPanel
        if self.bias_settings_panel is None:
            self.bias_settings_panel = BiasSettingsPanel(parent=None)
        self.bias_settings_panel.show()
        self.bias_settings_panel.raise_()
        self.bias_settings_panel.activateWindow()

    # ── Find Bias ─────────────────────────────────────────────────────────────

    def _find_bias(self):
        """
        Run find_bias_points on the current multisweep results in a background thread.

        Reads algorithm settings from the BiasSettingsPanel (using defaults when
        the panel has not been opened yet), then starts a FindBiasTask.
        """
        # Safety guard: silently reject if either task is still running.
        # Normally unreachable because both buttons are disabled while a task
        # runs, but guards against any edge-case re-entry.
        if (self.find_bias_task is not None and self.find_bias_task.isRunning()) or \
           (self.run_fits_task is not None and self.run_fits_task.isRunning()):
            return

        if not self.results_by_detector:
            QtWidgets.QMessageBox.warning(
                self, "No Data",
                "No multisweep data available. Please run a multisweep first."
            )
            return

        if not self.res_info_dict:
            QtWidgets.QMessageBox.warning(
                self, "No Resonator Registry",
                "res_info_dict is empty — cannot determine channel assignments."
            )
            return

        # Collect settings from the panel (or use defaults)
        if self.bias_settings_panel is not None:
            bias_settings = self.bias_settings_panel.get_settings()
        else:
            bias_settings = {}  # find_bias_points will use its own defaults

        # Resolve "fraction of sweep bandwidth" → absolute Hz before forwarding
        # to FindBiasTask (which only understands max_deriv_distance_hz).
        if bias_settings.get('max_deriv_distance_mode') == 'fraction':
            span_hz = self.initial_params.get('span_hz', 0.0)
            fraction = bias_settings.get('max_deriv_distance_fraction', 0.5)
            if span_hz > 0:
                bias_settings['max_deriv_distance_hz'] = fraction * span_hz
            else:
                # span_hz unavailable — fall back to the absolute kHz value
                import warnings
                warnings.warn(
                    "BiasSettingsPanel: 'fraction of sweep bandwidth' mode selected "
                    "but span_hz is not available in initial_params; falling back to "
                    "the absolute kHz value."
                )

        from .tasks import FindBiasTask, FindBiasSignals

        self.find_bias_signals = FindBiasSignals()
        self.find_bias_signals.progress.connect(self._find_bias_progress)
        self.find_bias_signals.completed.connect(self._find_bias_completed)
        self.find_bias_signals.error.connect(self._find_bias_error)

        self.find_bias_task = FindBiasTask(
            module=self.target_module,
            results_by_detector=self.results_by_detector,
            res_info_dict=self.res_info_dict,
            signals=self.find_bias_signals,
            bias_settings=bias_settings,
        )

        # Disable both Find Bias and Run Fit for the duration of this task.
        self.find_bias_btn.setEnabled(False)
        self.find_bias_btn.setText("Finding…")
        self.run_fit_btn.setEnabled(False)
        self.apply_bias_btn.setEnabled(False)
        self.find_bias_task.start()

    def _find_bias_progress(self, module: int, progress: float):
        """Handle progress updates from FindBiasTask (reserved for future use)."""
        pass

    def _find_bias_completed(self, module: int, updated_res_info_dict: dict):
        """
        Handle successful completion of FindBiasTask.

        Stores the updated res_info_dict, enables Apply Bias, and triggers
        a redraw of CF lines so they move to the refined bias frequencies.
        """
        # Merge the updated fields back into the live res_info_dict
        for code, info in updated_res_info_dict.items():
            if code in self.res_info_dict:
                self.res_info_dict[code].update(info)
            else:
                self.res_info_dict[code] = info

        num_found = sum(
            1 for info in self.res_info_dict.values() if info.get('bias_found', False)
        )
        total = len(self.res_info_dict)

        # Wait for the thread to fully finish before dropping the reference.
        # run() has already returned (it emitted this signal), so wait() is
        # nearly instantaneous — it just lets Qt complete its internal join so
        # the C++ QThread object is in the Finished state before Python may GC
        # the wrapper.  Without this, dropping the last Python reference while
        # Qt's thread machinery is still winding down can cause a core dump.
        if self.find_bias_task is not None:
            self.find_bias_task.wait()

        # Re-enable both Find Bias and Run Fit buttons
        self.find_bias_btn.setEnabled(True)
        self.find_bias_btn.setText("Find Bias")
        self.run_fit_btn.setEnabled(True)
        self.find_bias_task = None

        # Enable Apply Bias only if at least one bias point was found
        self.apply_bias_btn.setEnabled(num_found > 0)

        # Enable and auto-check the "Show Bias Info" checkbox so overlays appear
        # immediately after Find Bias completes.
        self._update_bias_info_checkbox_state()

        # Redraw all active plot tabs so bias overlays (chosen-amplitude highlights,
        # bias-frequency vertical lines, and IQ markers) appear immediately.
        self._redraw_plots()

        # Auto-save the updated res_info_dict (with bias results) to the session file
        # so that the bias points are preserved when the file is reloaded.
        # session_manager.handle_data_ready will automatically overwrite the existing
        # multisweep file rather than creating a new timestamped duplicate.
        export_data = self._prepare_export_data()
        self.data_ready.emit("multisweep", f"module{self.target_module}", export_data)

        # Show a transient "✓ Bias found" label in the toolbar that auto-hides after 5 s
        self._bias_status_label.show()
        QtCore.QTimer.singleShot(5000, self._bias_status_label.hide)

    def _find_bias_error(self, error_msg: str):
        """Handle errors from FindBiasTask."""
        QtWidgets.QMessageBox.critical(self, "Find Bias Error", error_msg)
        if self.find_bias_task is not None:
            self.find_bias_task.wait()
        self.find_bias_btn.setEnabled(True)
        self.find_bias_btn.setText("Find Bias")
        self.run_fit_btn.setEnabled(True)
        self.find_bias_task = None

    # ── Apply Bias ────────────────────────────────────────────────────────────

    def _apply_bias(self):
        """
        Run apply_bias on the hardware using the bias conditions stored in res_info_dict.

        Requires Find Bias to have completed successfully (i.e. at least one
        res_info_dict entry has bias_found == True).
        """
        periscope = self._get_periscope_parent()
        if not periscope or periscope.crs is None:
            QtWidgets.QMessageBox.warning(
                self, "CRS Not Available",
                "CRS object is not available.  Cannot programme hardware."
            )
            return

        if self.target_module is None:
            QtWidgets.QMessageBox.warning(
                self, "Module Not Set",
                "Target module is not set.  Cannot programme hardware."
            )
            return

        num_ready = sum(
            1 for info in self.res_info_dict.values() if info.get('bias_found', False)
        )
        if num_ready == 0:
            QtWidgets.QMessageBox.warning(
                self, "No Bias Points",
                "No bias points are available.  Please run Find Bias first."
            )
            return

        from .tasks import ApplyBiasTask, ApplyBiasSignals

        self.apply_bias_signals = ApplyBiasSignals()
        self.apply_bias_signals.progress.connect(self._apply_bias_progress)
        self.apply_bias_signals.completed.connect(self._apply_bias_completed)
        self.apply_bias_signals.error.connect(self._apply_bias_error)

        self.apply_bias_task = ApplyBiasTask(
            crs=periscope.crs,
            module=self.target_module,
            res_info_dict=self.res_info_dict,
            signals=self.apply_bias_signals,
        )

        self.apply_bias_btn.setEnabled(False)
        self.apply_bias_btn.setText("Applying…")
        self.apply_bias_task.start()

    def _apply_bias_progress(self, module: int, progress: float):
        """Handle progress updates from ApplyBiasTask."""
        pass

    def _apply_bias_completed(self, module: int, apply_report: dict):
        """
        Handle successful completion of ApplyBiasTask.

        Stores apply_report for export, emits the df_calibration_ready signal
        so Periscope's df-unit mode picks up the new calibration data, and
        enables the Noise Spectrum button.
        """
        self.bias_kids_output = apply_report  # kept for export compat

        # Extract per-channel df_calibration from res_info_dict
        # (keyed by channel_number so app_runtime can use it directly)
        df_calibrations: dict = {}
        for code, info in self.res_info_dict.items():
            cal = info.get('df_calibration')
            ch = info.get('channel_number')
            if cal is not None and ch is not None:
                import cmath
                if not cmath.isnan(cal):
                    df_calibrations[ch] = cal

        if df_calibrations:
            self.df_calibration_ready.emit(module, df_calibrations)

        # Emit for session auto-export — update the existing multisweep pickle in-place
        # rather than creating a separate bias_HHMMSS.pkl.  The session manager's
        # overwrite-tracking mechanism handles the in-place update automatically.
        export_data = self._prepare_export_data()
        self.data_ready.emit("multisweep", f"module{self.target_module}", export_data)

        # Re-enable buttons
        self.apply_bias_btn.setEnabled(True)
        self.apply_bias_btn.setText("Apply Bias")
        self.noise_spectrum_btn.setEnabled(True)
        self.apply_bias_task = None

        # Show a transient "✓ Bias applied" label in the toolbar that auto-hides after 5 s
        self._apply_bias_status_label.show()
        QtCore.QTimer.singleShot(5000, self._apply_bias_status_label.hide)

        # Kick off background IQ rotation computation immediately after bias is applied.
        # The task runs non-blocking; completion is handled by _iq_rotation_completed.
        periscope = self._get_periscope_parent()
        if periscope and periscope.crs is not None and self.res_info_dict:
            from .tasks import ComputeIQRotationTask, IQRotationSignals
            self.iq_rotation_signals = IQRotationSignals()
            self.iq_rotation_signals.completed.connect(self._iq_rotation_completed)
            self.iq_rotation_signals.error.connect(self._iq_rotation_error)
            self.iq_rotation_task = ComputeIQRotationTask(
                crs=periscope.crs,
                module=module,
                res_info_dict=self.res_info_dict,
                signals=self.iq_rotation_signals,
            )
            self.iq_rotation_task.start()

    def _apply_bias_error(self, error_msg: str):
        """Handle errors from ApplyBiasTask."""
        QtWidgets.QMessageBox.critical(self, "Apply Bias Error", error_msg)
        self.apply_bias_btn.setEnabled(True)
        self.apply_bias_btn.setText("Apply Bias")
        self.apply_bias_task = None

    # ── IQ Plane Rotation ─────────────────────────────────────────────────────

    def _iq_rotation_completed(self, module: int, angles: dict):
        """Handle successful completion of ComputeIQRotationTask.

        Stores per-channel rotation angles in ``res_info_dict``, emits
        ``iq_rotation_ready`` (picked up by ``app.py`` to enable the Rotated
        IQ radio button), re-saves the updated pickle, and shows a transient
        status label.
        """
        if not angles:
            print("[MultisweepPanel] IQ rotation: no angles computed.", flush=True)
            if self.iq_rotation_task is not None:
                self.iq_rotation_task.wait()
            self.iq_rotation_task = None
            return

        # Store angle in res_info_dict for each code that was computed
        for code, theta in angles.items():
            if code in self.res_info_dict:
                self.res_info_dict[code]['iq_rotation_angle'] = float(theta)

        # Emit so app.py can store calibrations and enable rb_rotated_iq
        self.iq_rotation_ready.emit(module, angles)

        # Re-save pickle so angles survive session reload
        export_data = self._prepare_export_data()
        self.data_ready.emit("multisweep", f"module{self.target_module}", export_data)

        # Wait for the thread to fully finish before dropping the reference
        # (near-instantaneous — same rationale as FindBiasTask)
        if self.iq_rotation_task is not None:
            self.iq_rotation_task.wait()
        self.iq_rotation_task = None

        # Show transient "✓ IQ rotation computed" label
        self._iq_rotation_status_label.show()
        QtCore.QTimer.singleShot(5000, self._iq_rotation_status_label.hide)

        # Enable and auto-check the "Show Rotated IQ" checkbox so the IQ tab
        # immediately shows the rotated circles when the user switches to it.
        self._update_rotated_iq_checkbox_state()

    def _on_show_rotated_iq_changed(self, checked: bool):
        """Slot for the 'Show Rotated IQ' checkbox.

        Triggers a redraw of the IQ Circles tab (Tab 1) so the rotation is
        applied or removed.  Has no effect when a different tab is active —
        the next visit to Tab 1 will pick up the new state automatically.

        Args:
            checked (bool): New checked state of the checkbox.
        """
        # Only redraw if the IQ Circles tab is currently active; otherwise the
        # next tab switch will trigger a redraw naturally.
        if hasattr(self, 'plot_tabs') and self.plot_tabs.currentIndex() == 1:
            self._redraw_sweep_grid(1)

    def _update_rotated_iq_checkbox_state(self):
        """Enable/disable and auto-check the 'Show Rotated IQ' checkbox.

        Should be called whenever ``res_info_dict`` may have gained
        ``iq_rotation_angle`` entries (i.e. after
        :meth:`_iq_rotation_completed` is called, or after loading a session
        file that contains pre-computed rotation angles).

        Behaviour:
          - If at least one detector has an ``iq_rotation_angle``:
              enable the checkbox and check it.
          - Otherwise: uncheck and disable the checkbox.
        """
        if not hasattr(self, 'show_rotated_iq_cb') or self.show_rotated_iq_cb is None:
            return
        has_angles = any(
            'iq_rotation_angle' in info
            for info in self.res_info_dict.values()
        )
        self.show_rotated_iq_cb.blockSignals(True)
        self.show_rotated_iq_cb.setEnabled(has_angles)
        if has_angles:
            self.show_rotated_iq_cb.setChecked(True)
        else:
            self.show_rotated_iq_cb.setChecked(False)
        self.show_rotated_iq_cb.blockSignals(False)

    def _iq_rotation_error(self, error_msg: str):
        """Handle errors from ComputeIQRotationTask.

        Non-fatal — the IQ rotation is an enhancement on top of Apply Bias.
        Log the error but do not show a modal dialog.
        """
        import sys
        print(f"[MultisweepPanel] IQ rotation error (non-fatal): {error_msg}",
              file=sys.stderr, flush=True)
        if self.iq_rotation_task is not None:
            self.iq_rotation_task.wait()
        self.iq_rotation_task = None

    # ── IQ Derivatives tab ────────────────────────────────────────────────────

    def _redraw_derivative_grid(self):
        """Redraw the IQ Derivatives tab (Tab 5).

        Collects the ``bias_finding`` sub-dict from the selected-amplitude
        entry for each detector (set by ``find_bias_points``) and delegates
        rendering to :func:`multisweep_grid_helpers.update_derivative_grid`.

        If no ``bias_finding`` keys exist (Find Bias has not run yet) the
        placeholder label is shown and the scroll area is kept hidden.
        """
        from .multisweep_grid_helpers import update_derivative_grid

        # Build {detector_id: bias_finding_dict} for all detectors that have
        # a 'bias_finding' key in at least one of their sweep entries.
        bias_finding_by_detector: dict = {}
        for det_id, iter_dict in self.results_by_detector.items():
            for entry in iter_dict.values():
                bf = entry.get('bias_finding')
                if bf is not None:
                    bias_finding_by_detector[det_id] = bf
                    break  # Only one entry per detector will have bias_finding

        if not bias_finding_by_detector:
            # Find Bias hasn't run yet — show the placeholder
            if hasattr(self, '_derivative_placeholder'):
                self._derivative_placeholder.show()
            if hasattr(self, '_derivative_scroll'):
                self._derivative_scroll.hide()
            return

        # Data is ready — hide placeholder, show scroll area
        if hasattr(self, '_derivative_placeholder'):
            self._derivative_placeholder.hide()
        if hasattr(self, '_derivative_scroll'):
            self._derivative_scroll.show()

        # Compute detector order: use center frequency from the main sweep data.
        def _deriv_center_freq(det_id):
            iter_dict = self.results_by_detector.get(det_id, {})
            if not iter_dict:
                return None
            first_entry = next(iter(iter_dict.values()))
            return (
                first_entry.get('original_center_frequency')
                or first_entry.get('sweep_center_frequency')
            )

        sorted_ids = self._get_sorted_detector_ids(
            bias_finding_by_detector.keys(), _deriv_center_freq
        )

        update_derivative_grid(
            grid_layout=self.derivative_grid,
            bias_finding_by_detector=bias_finding_by_detector,
            current_batch=self.current_batch,
            batch_size=self.batch_size,
            dark_mode=self.dark_mode,
            prev_btn=self.prev_batch_btn,
            next_btn=self.next_batch_btn,
            batch_label=self.batch_info_label,
            widget_cache=self.derivative_plots_cache,
            unit_mode=self.unit_mode,
            dac_scale=self.dac_scales.get(self.active_module_for_dac),
            sorted_detector_ids=sorted_ids,
        )

        # Install double-click event filter so clicking a derivative subplot
        # navigates the Detector Digest to that detector.
        for pw in self.derivative_plots_cache:
            pw.installEventFilter(self)

        # Apply the current zoom box mode to all (potentially new) derivative widgets.
        mode = pg.ViewBox.RectMode if self.zoom_box_mode else pg.ViewBox.PanMode
        for pw in self.derivative_plots_cache:
            vb = pw.getViewBox()
            if vb is not None:
                vb.setMouseMode(mode)

    # ── Fit Results tab (Tab 6) ───────────────────────────────────────────────

    def _populate_fit_amplitude_combo(self):
        """Populate the amplitude index dropdown in the Fit Results tab.

        Scans ``results_by_detector`` to find which sorted amplitude positions
        (0-based) have at least one successful fit (skewed or nonlinear) across
        any detector.  Those positions become the combo items (displayed as
        plain integers: ``"0"``, ``"1"``, etc.).

        The current selection is preserved when possible; if the previously
        selected index no longer exists in the new set, the first available
        index is selected.  Signals are blocked during the rebuild to avoid
        spurious redraws.
        """
        if self._fit_display_index_combo is None:
            return
        if not self.results_by_detector:
            return

        # Collect valid positions: for each detector sort its iterations by
        # sweep_amplitude and record the 0-based positions that have a fit.
        valid_indices: set = set()
        for iter_dict in self.results_by_detector.values():
            sorted_items = sorted(
                iter_dict.items(),
                key=lambda kv: kv[1].get('sweep_amplitude_normalized', 0.0),
            )
            for pos, (_iter_idx, entry) in enumerate(sorted_items):
                if (entry.get('fits', {}).get('skewed', {}).get('skewed_fit_success')
                        or entry.get('fits', {}).get('nonlinear', {}).get('nonlinear_fit_success')):
                    valid_indices.add(pos)

        if not valid_indices:
            # No fit data yet — nothing to show
            return

        sorted_valid = sorted(valid_indices)

        # Block signals while rebuilding so we don't trigger redundant redraws
        self._fit_display_index_combo.blockSignals(True)
        prev_data = self._fit_display_index_combo.currentData()

        self._fit_display_index_combo.clear()
        for idx in sorted_valid:
            self._fit_display_index_combo.addItem(str(idx), userData=idx)

        # Restore previous selection if it still exists; otherwise select first
        restore_row = 0
        if prev_data is not None:
            for row in range(self._fit_display_index_combo.count()):
                if self._fit_display_index_combo.itemData(row) == prev_data:
                    restore_row = row
                    break
        self._fit_display_index_combo.setCurrentIndex(restore_row)

        self._fit_display_index_combo.blockSignals(False)

    def _on_fit_results_controls_changed(self):
        """Slot for any Tab-6 display control changing — triggers a redraw."""
        if self.plot_tabs.currentIndex() == 6:
            self._redraw_fit_results_grid()

    def _redraw_fit_results_grid(self):
        """Redraw the Fit Results tab (Tab 6).

        Reads the tab's display controls and calls
        :func:`multisweep_grid_helpers.update_fit_results_grid` to render
        per-detector magnitude sweeps with fitted model overlays.  Shows a
        placeholder when no fit data is present yet.
        """
        from .multisweep_grid_helpers import update_fit_results_grid

        if not self.results_by_detector:
            return
        if self._fit_display_mode_rb_index is None:
            return  # Tab not yet created

        # Check whether any fit data is present; if not, show the placeholder.
        has_fit_data = any(
            entry.get('fits', {}).get('skewed', {}).get('skewed_fit_success')
            or entry.get('fits', {}).get('nonlinear', {}).get('nonlinear_fit_success')
            for iter_dict in self.results_by_detector.values()
            for entry in iter_dict.values()
        )

        if not has_fit_data:
            if hasattr(self, '_fit_results_placeholder'):
                self._fit_results_placeholder.show()
            if hasattr(self, '_fit_results_scroll'):
                self._fit_results_scroll.hide()
            return

        # Fit data is present — hide placeholder, show scroll area
        if hasattr(self, '_fit_results_placeholder'):
            self._fit_results_placeholder.hide()
        if hasattr(self, '_fit_results_scroll'):
            self._fit_results_scroll.show()

        # Read controls
        display_mode = (
            'bias' if self._fit_display_mode_rb_bias.isChecked() else 'index'
        )
        # Read amplitude index from the combo (item data stores the actual 0-based index)
        combo_data = self._fit_display_index_combo.currentData()
        display_amp_index = combo_data if combo_data is not None else 0
        show_skewed    = self._fit_show_skewed_cb.isChecked()
        show_nonlinear = self._fit_show_nonlinear_cb.isChecked()
        dac_scale = self.dac_scales.get(self.active_module_for_dac)

        # Compute detector order for the Fit Results grid.
        def _fit_center_freq(det_id):
            iter_dict = self.results_by_detector.get(det_id, {})
            if not iter_dict:
                return None
            first_entry = next(iter(iter_dict.values()))
            return (
                first_entry.get('original_center_frequency')
                or first_entry.get('sweep_center_frequency')
            )

        sorted_ids = self._get_sorted_detector_ids(
            self.results_by_detector.keys(), _fit_center_freq
        )

        update_fit_results_grid(
            grid_layout=self.fit_results_grid,
            data_by_detector=self.results_by_detector,
            res_info_dict=self.res_info_dict,
            display_mode=display_mode,
            display_amplitude_index=display_amp_index,
            show_skewed=show_skewed,
            show_nonlinear=show_nonlinear,
            current_batch=self.current_batch,
            batch_size=self.batch_size,
            dark_mode=self.dark_mode,
            unit_mode=self.unit_mode,
            normalize=self.normalize_traces,
            prev_btn=self.prev_batch_btn,
            next_btn=self.next_batch_btn,
            batch_label=self.batch_info_label,
            widget_cache=self.fit_results_plots_cache,
            dac_scale=dac_scale,
            sorted_detector_ids=sorted_ids,
        )

        # Install double-click event filter and apply zoom box mode
        for pw in self.fit_results_plots_cache:
            pw.installEventFilter(self)
        mode = pg.ViewBox.RectMode if self.zoom_box_mode else pg.ViewBox.PanMode
        for pw in self.fit_results_plots_cache:
            vb = pw.getViewBox()
            if vb is not None:
                vb.setMouseMode(mode)

    def _update_fit_results_tab_visibility(self):
        """Refresh Tab 6 state after fits may have been added or removed.

        The Fit Results tab is always visible.  This method populates the
        amplitude index dropdown when fit data is present and triggers a
        redraw if Tab 6 is currently active.
        """
        has_fit_data = any(
            entry.get('fits', {}).get('skewed', {}).get('skewed_fit_success')
            or entry.get('fits', {}).get('nonlinear', {}).get('nonlinear_fit_success')
            for iter_dict in self.results_by_detector.values()
            for entry in iter_dict.values()
        ) if self.results_by_detector else False

        if has_fit_data:
            # Populate the amplitude index dropdown now that fit data is available
            self._populate_fit_amplitude_combo()

        if hasattr(self, 'plot_tabs') and self.plot_tabs is not None:
            if self.plot_tabs.currentIndex() == 6:
                self._redraw_fit_results_grid()

    # ── Fit Settings ─────────────────────────────────────────────────────────

    def _show_fit_settings(self):
        """Show (or create) the persistent FitSettingsPanel window."""
        from .fit_settings_panel import FitSettingsPanel
        if self.fit_settings_panel is None:
            self.fit_settings_panel = FitSettingsPanel(parent=None)
        self.fit_settings_panel.show()
        self.fit_settings_panel.raise_()
        self.fit_settings_panel.activateWindow()

    # ── Run Fit ───────────────────────────────────────────────────────────────

    def _run_fit(self):
        """Run the selected fitting algorithms on the current multisweep data.

        Reads fit settings from :class:`FitSettingsPanel` (or loads saved
        defaults when the panel has not been opened yet), then starts a
        :class:`~rfmux.tools.periscope.tasks.RunFitsTask` in a background
        thread.  The task operates on a deep copy of
        ``self.results_by_detector`` and emits the updated dict on completion;
        this method's completion handler merges the fit keys back into the
        live dict.
        """
        if not self.results_by_detector:
            QtWidgets.QMessageBox.warning(
                self, "No Data",
                "No multisweep data available.  Please run a multisweep first."
            )
            return

        # Collect settings — prefer the open panel, fall back to QSettings defaults.
        if self.fit_settings_panel is not None:
            fit_settings = self.fit_settings_panel.get_settings()
        else:
            from . import settings as periscope_settings
            fit_settings = periscope_settings.get_fit_defaults()

        # Guard: at least one fit type must be enabled
        if not fit_settings.get('apply_skewed_fit') and not fit_settings.get('apply_nonlinear_fit'):
            QtWidgets.QMessageBox.warning(
                self, "No Fits Selected",
                "No fit types are enabled.\n"
                "Open ⚙ Fit Settings and enable at least one fit."
            )
            return

        from .tasks import RunFitsTask, RunFitsSignals

        self.run_fits_signals = RunFitsSignals()
        self.run_fits_signals.progress.connect(self._run_fits_progress)
        self.run_fits_signals.completed.connect(self._run_fits_completed)
        self.run_fits_signals.error.connect(self._run_fits_error)

        self.run_fits_task = RunFitsTask(
            module=self.target_module,
            results_by_detector=self.results_by_detector,
            fit_settings=fit_settings,
            signals=self.run_fits_signals,
            res_info_dict=self.res_info_dict,
        )

        # Disable both Run Fit and Find Bias for the duration of this task.
        self.run_fit_btn.setEnabled(False)
        self.run_fit_btn.setText("Fitting…")
        self.find_bias_btn.setEnabled(False)
        self.run_fits_task.start()

    def _run_fits_progress(self, module: int, progress: float):
        """Handle progress updates from RunFitsTask (reserved for future use)."""
        pass

    def _run_fits_completed(self, module: int, updated_results_by_detector: dict):
        """Handle successful completion of RunFitsTask.

        Merges the fit-related keys from the updated dict back into
        ``self.results_by_detector``, invalidates cached views that depend on
        fit data (histograms, detector digest), emits ``data_ready`` for
        session auto-export, and shows a transient completion label.

        No broad plot redraw is triggered — lazy redraw on the next tab
        navigation handles everything without blocking the GUI.
        """
        # Merge only the fit-related keys back into the live dict.
        # All other data (sweep data, IQ, metadata) is untouched.
        from .tasks import RunFitsTask
        fit_keys = RunFitsTask._FIT_KEYS

        for code, iter_dict in updated_results_by_detector.items():
            if code not in self.results_by_detector:
                continue
            for iter_idx, updated_entry in iter_dict.items():
                if iter_idx not in self.results_by_detector[code]:
                    continue
                target = self.results_by_detector[code][iter_idx]
                for key in fit_keys:
                    if key in updated_entry:
                        target[key] = updated_entry[key]

        # Invalidate histogram panel so it regenerates with the new fit data
        # on the next visit to the Histograms tab.
        if self.histogram_panel is not None and hasattr(self.histogram_panel, 'histogram_cache'):
            self.histogram_panel.histogram_cache.clear()
        self.histograms_generated = False

        # Invalidate detector digest so it recreates with updated fit data
        # on the next visit to the Detector Digest tab.
        self.digest_panel = None

        # Emit for session auto-export so the pickle file is updated immediately.
        # session_manager.handle_data_ready will automatically overwrite the existing
        # multisweep file rather than creating a new timestamped duplicate.
        export_data = self._prepare_export_data()
        self.data_ready.emit("multisweep", f"module{self.target_module}", export_data)

        # Wait for the thread to fully finish before dropping the reference
        # (same rationale as FindBiasTask — prevents native crash on GC).
        if self.run_fits_task is not None:
            self.run_fits_task.wait()

        # Re-enable both Run Fit and Find Bias buttons.
        self.run_fit_btn.setEnabled(True)
        self.run_fit_btn.setText("Run Fit")
        self.find_bias_btn.setEnabled(True)
        self.run_fits_task = None

        # Update Fit Results tab visibility — now that new fit data is present,
        # the tab should appear if the user has enabled it in Fit Settings.
        self._update_fit_results_tab_visibility()

        # Show a transient "✓ Fits complete" label that auto-hides after 5 s.
        self._fits_status_label.show()
        QtCore.QTimer.singleShot(5000, self._fits_status_label.hide)

    def _run_fits_error(self, error_msg: str):
        """Handle errors from RunFitsTask."""
        QtWidgets.QMessageBox.critical(self, "Run Fit Error", error_msg)
        if self.run_fits_task is not None:
            self.run_fits_task.wait()
        self.run_fit_btn.setEnabled(True)
        self.run_fit_btn.setText("Run Fit")
        self.find_bias_btn.setEnabled(True)
        self.run_fits_task = None
