"""Panel for displaying multisweep analysis results (dockable)."""
import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg
import asyncio
import traceback
import time

# Imports from within the 'periscope' subpackage
from .layouts import FlowLayout, grouped
from .utils import (
    LINE_WIDTH, UnitConverter, ClickableViewBox, QtWidgets, QtCore, pg,
    AMPLITUDE_COLORMAP_THRESHOLD, UPWARD_SWEEP_STYLE, DOWNWARD_SWEEP_STYLE,
    STATUS_MESSAGE_MS, TABLEAU10_COLORS, ScreenshotMixin
)
from .noise_spectrum_panel import NoiseSpectrumPanel
from .noise_spectrum_dialog import NoiseSpectrumDialog
from .amplitude_colorbar import AmplitudeColorBar
from .multisweep_grid_helpers import create_amplitude_color_map
from .fit_settings_panel import (
    ALL_AMPLITUDES, BIAS_AMPLITUDE, MODELS as FIT_MODELS, FitSettingsPanel)
from .bias_settings_panel import BiasSettingsPanel
from .tasks import (
    ApplyBiasSignals, ApplyBiasTask, FindBiasSignals, FindBiasTask,
    RunFitsSignals, RunFitsTask)
from rfmux.core.resonators import ResonatorCatalog
from rfmux.tuning import AmplitudeSchedule, collect_amplitude_iterations_for, store
from rfmux.core.transferfunctions import PFB_SAMPLING_FREQ
# from rfmux.algorithms.measurement import py_get_samples

# A data callback arrives per sweep point; a grid of subplots takes longer to
# draw than a point takes to measure, so live redraws are coalesced to this.
LIVE_REDRAW_INTERVAL_MS = 100


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
    # Emitted once the call has returned and the panel holds it, so the session
    # can write the file where the panel, not the task, decides when.
    sweep_finished = pyqtSignal()
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
        # A file taken on another module: shown, but not something to sweep from.
        self.is_foreign_module = False
        self.spectrum_noise_data = {}

        self.debug_noise_data = {}
        self.debug_phase_data = []

        
        # Track open noise spectrum windows to prevent garbage collection
        self.noise_spectrum_windows = []
        self.noise_panel_count = 0    # Counter for naming noise tabs
        
        # Stores {amp: {conceptual_idx: output_cf}}, for the legacy lane only
        self.last_output_cfs_by_amp_and_conceptual_idx: dict[float, dict[int, float]] = {}

        self.setWindowTitle(f"Multisweep Results - Module {self.target_module}")

        # What the measurement is: multisweep's container, this module's block
        # out of it, and the array that was swept.
        self.multisweep_container = None
        self.module_sweeps = None
        self.catalog = self.initial_params.get('catalog')

        # Points measured so far, {name: {(step, direction): sweep}}, held only
        # while the call is running. Dropped when the block arrives.
        self._live = {}
        # {step: {name: amplitude}} for the whole call, resolved before the
        # first point so a trace's colour does not shift as sweeps land.
        self._step_amplitudes = {}
        self._set_amplitude_scale(self.initial_params.get('amp'))

        # The legacy bias and noise lane's shape, filled only by update_data on
        # the legacy load path and read only by that lane. Nothing draws it.
        self.results_by_detector = {}
        self.current_amplitude_being_processed = None # Tracks the amplitude currently being processed
        self.current_iteration_being_processed = None # Tracks the current iteration
        self.unit_mode = "dbm"  # Current unit for magnitude display ("counts", "dbm", "volts")
        self.normalize_traces = True  # Flag to normalize trace plots (magnitude and phase)
        self.zoom_box_mode = True  # Flag for enabling/disabling pyqtgraph's zoom box
        
        # Module context for DAC scale lookup (can be different from target_module if needed)
        self.active_module_for_dac = self.target_module

        # Bias KIDs output storage
        self.bias_kids_output = None  # Stores the output from bias_kids algorithm
        self.nco_frequency_hz = None  # NCO frequency used when biasing (stored for export)

        # Initialize batch tracking for sweep tabs (before _setup_ui)
        self.current_batch = 0
        self.batch_size = 8
        
        # Storage for sweep grid plots - cached to avoid recreating widgets
        self.mag_sweep_plots_cache = []  # List of plot widgets for magnitude tab
        self.iq_sweep_plots_cache = []   # List of plot widgets for IQ tab
        self.fit_sweep_plots_cache = []  # List of plot widgets for the fit tab
        self.bias_sweep_plots_cache = []  # List of plot widgets for the bias tab

        # The fitters' settings outlive any one fit, and are shared by nothing
        # else: one window per panel, as the measurement is one panel's.
        self.fit_settings = FitSettingsPanel(self)
        self.fit_settings.display_model_changed.connect(self._redraw_plots)

        # Bias finding's settings, the same way, and what the last run
        # concluded. The report's catalog becomes this panel's, so what is
        # applied and what is re-swept are one thing.
        self.bias_settings = BiasSettingsPanel(self)
        self.bias_report = None

        self._fit_status_timer = QtCore.QTimer(self)
        self._fit_status_timer.setSingleShot(True)

        self._bias_status_timer = QtCore.QTimer(self)
        self._bias_status_timer.setSingleShot(True)

        self._live_redraw_timer = QtCore.QTimer(self)
        self._live_redraw_timer.setSingleShot(True)
        self._live_redraw_timer.timeout.connect(self._redraw_plots)

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
        # Wraps into rows as the panel narrows; the batch controls and
        # the subplot controls each stay together.
        toolbar_layout = FlowLayout(toolbar)

        # Save Button
        self.export_btn = QtWidgets.QPushButton("💾")
        self.export_btn.setToolTip(
            "Save this multisweep to the session folder, or overwrite the file "
            "it was already saved to")
        self.export_btn.clicked.connect(self._save_multisweep_action)
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

        # Fitting: the button, its settings, and what it is doing.
        self.run_fit_btn = QtWidgets.QPushButton("Run Fit")
        self.run_fit_btn.setToolTip(
            "Fit resonator models to these sweeps, as the fit settings ask")
        self.run_fit_btn.clicked.connect(self._run_fits)
        fit_settings_btn = QtWidgets.QPushButton("⚙")
        fit_settings_btn.setMaximumWidth(30)
        fit_settings_btn.setToolTip("Which models to fit, and which sweeps")
        fit_settings_btn.clicked.connect(self._show_fit_settings)
        self.fit_status_label = QtWidgets.QLabel("")
        self.fit_status_label.setMinimumWidth(110)
        # The label's own slot, not a lambda over self: Qt drops a connection
        # to a destroyed receiver, where a closure would keep this panel's
        # Python wrapper alive and fire into a deleted widget.
        self._fit_status_timer.timeout.connect(self.fit_status_label.clear)
        # The fit controls wrap as one item, so the button keeps its settings.
        self.fit_controls = grouped(
            self.run_fit_btn, fit_settings_btn, self.fit_status_label)
        toolbar_layout.addWidget(self.fit_controls)

        self._populate_fit_models()
        self._populate_fit_amplitudes()

        # Bias finding: the button, its settings, and what it is doing --
        # shaped like the fit controls beside it, because it is the same
        # gesture over a different call.
        self.find_bias_btn = QtWidgets.QPushButton("Find Bias")
        self.find_bias_btn.setToolTip(
            "Choose an operating amplitude and frequency for every resonator "
            "in these sweeps, as the bias settings ask")
        self.find_bias_btn.clicked.connect(self._find_bias)
        bias_settings_btn = QtWidgets.QPushButton("\u2699")
        bias_settings_btn.setMaximumWidth(30)
        bias_settings_btn.setToolTip(
            "Which bifurcation test, and where in a sweep the tone goes")
        bias_settings_btn.clicked.connect(self._show_bias_settings)
        self.bias_status_label = QtWidgets.QLabel("")
        self.bias_status_label.setMinimumWidth(110)
        # The label's own slot, for the reason the fit line's is.
        self._bias_status_timer.timeout.connect(self.bias_status_label.clear)
        self.apply_bias_btn = QtWidgets.QPushButton("Apply Bias")
        self.apply_bias_btn.setToolTip(
            "Park a tone on every resonator, at the frequency and amplitude "
            "this panel's catalog carries")
        self.apply_bias_btn.clicked.connect(self._apply_bias)
        self.bias_controls = grouped(
            self.find_bias_btn, bias_settings_btn, self.apply_bias_btn,
            self.bias_status_label)
        toolbar_layout.addWidget(self.bias_controls)

        self.noise_spectrum_btn = QtWidgets.QPushButton("Get Noise Spectrum")
        if self.bias_data_avail:
            self.noise_spectrum_btn.setEnabled(True)
        else:
            self.noise_spectrum_btn.setEnabled(False)
        self.noise_spectrum_btn.setToolTip("Open a dialog to configure and get the noise spectrum, will only work if KIDS is biased.")
        self.noise_spectrum_btn.clicked.connect(self._open_noise_spectrum_dialog)
        toolbar_layout.addWidget(self.noise_spectrum_btn)
        
        # Batch navigation controls (for sweep tabs)
        self.batch_label = QtWidgets.QLabel("Batch:")
        
        self.prev_batch_btn = QtWidgets.QPushButton("◀")
        self.prev_batch_btn.setToolTip("Previous batch")
        self.prev_batch_btn.setMaximumWidth(30)  # Shrink to 1/3 width
        self.prev_batch_btn.clicked.connect(self._prev_batch)
        
        self.batch_info_label = QtWidgets.QLabel("1 of 1")
        self.batch_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.batch_info_label.setMinimumWidth(40)
        
        self.next_batch_btn = QtWidgets.QPushButton("▶")
        self.next_batch_btn.setToolTip("Next batch")
        self.next_batch_btn.setMaximumWidth(30)  # Shrink to 1/3 width
        self.next_batch_btn.clicked.connect(self._next_batch)
        # The batch controls wrap as one item, and hide as one.
        self.batch_nav = grouped(self.batch_label, self.prev_batch_btn,
                                 self.batch_info_label, self.next_batch_btn)
        toolbar_layout.addWidget(self.batch_nav)

        self.batch_size_label = QtWidgets.QLabel("Subplots:")
        
        self.batch_size_spin = QtWidgets.QSpinBox()
        self.batch_size_spin.setRange(1, 200)
        self.batch_size_spin.setValue(self.batch_size)
        self.batch_size_spin.setSingleStep(1)
        self.batch_size_spin.setToolTip("Detectors per batch (press Update to apply)")
        
        self.batch_update_btn = QtWidgets.QPushButton("Update")
        self.batch_update_btn.setToolTip("Apply new batch size and regenerate plots")
        self.batch_update_btn.clicked.connect(self._apply_batch_size)
        self.subplot_controls = grouped(
            self.batch_size_label, self.batch_size_spin, self.batch_update_btn)
        toolbar_layout.addWidget(self.subplot_controls)

        # Normalization Checkbox
        self.normalize_checkbox = QtWidgets.QCheckBox("Normalize Traces")
        self.normalize_checkbox.setChecked(self.normalize_traces)
        self.normalize_checkbox.toggled.connect(self._toggle_trace_normalization)
        toolbar_layout.addWidget(self.normalize_checkbox)

        # Show Center Frequencies Checkbox

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
        
        # Bound-method slots: a lambda closing over self would make the
        # button own the panel, and a panel in a reference cycle is torn
        # down by the cyclic collector, which crashes in Qt.
        self._unit_buttons = {self.rb_counts: "counts", self.rb_dbm: "dbm",
                              self.rb_volts: "volts"}
        for rb in self._unit_buttons:
            rb.toggled.connect(self._on_unit_toggled)

        unit_group.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        toolbar_layout.addWidget(unit_group)

    def _on_unit_toggled(self, checked: bool):
        if checked:
            self._update_unit_mode(self._unit_buttons[self.sender()])

    def _setup_zoom_box_control(self, toolbar_layout):
        """Sets up the checkbox to toggle zoom box mode for plots."""
        self.zoom_box_cb = QtWidgets.QCheckBox("Zoom Box Mode")
        self.zoom_box_cb.setChecked(self.zoom_box_mode)
        self.zoom_box_cb.toggled.connect(self._toggle_zoom_box_mode)
        toolbar_layout.addWidget(self.zoom_box_cb)

    def _setup_plot_area(self, layout):
        """Sets up the tabbed plot area: one grid per view."""
        # Create tab widget
        self.plot_tabs = QtWidgets.QTabWidget()
        self.plot_tabs.currentChanged.connect(self._on_plot_tab_changed)
        
        # Tab 0: Magnitude Sweeps (per-detector grid)
        self.mag_sweeps_tab, self.mag_sweeps_grid, self.mag_colorbar = self._create_sweep_tab()
        self.plot_tabs.addTab(self.mag_sweeps_tab, "Magnitude Sweeps")
        
        # Tab 1: IQ Circles (per-detector grid)
        self.iq_sweeps_tab, self.iq_sweeps_grid, self.iq_colorbar = self._create_sweep_tab()
        self.plot_tabs.addTab(self.iq_sweeps_tab, "IQ Circles")
        
        # Tab 2: Fit Results (per-detector grid, models over the measurement)
        self.fit_sweeps_tab, self.fit_sweeps_grid, self.fit_colorbar = self._create_sweep_tab()
        self.plot_tabs.addTab(self.fit_sweeps_tab, "Fit Results")

        # Tab 3: what the derivative bifurcation test looks at
        self.bias_sweeps_tab, self.bias_sweeps_grid, self.bias_colorbar = self._create_sweep_tab()
        self.plot_tabs.addTab(self.bias_sweeps_tab, "Bias Diagnostics")
        self.plot_tabs.setTabToolTip(
            3, "The point-to-point change in each sweep's normalized arc "
               "speed, in units of the bar the derivative test applied to it. "
               "A spike past \u00b11 with one the other way beside it is what "
               "that test calls a bifurcation.")

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
        
    def _on_plot_tab_changed(self, index):
        """Handle plot tab changes."""
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
        names = self._selected_names()
        total_batches = max(1, (len(names) + self.batch_size - 1) // self.batch_size)
        if self.current_batch < total_batches - 1:
            self.current_batch += 1
            self._redraw_plots()

    def _toggle_trace_normalization(self, checked):
        """
        Slot for the 'Normalize Traces' checkbox.
        Updates normalization state for both magnitude and phase, and redraws plots.
        """
        self.normalize_traces = checked
        self._redraw_plots()

    def _update_unit_mode(self, mode):
        """
        Slot for unit selection radio buttons.
        Updates unit mode and redraws plots if the mode changed.
        """
        if self.unit_mode != mode:
            self.unit_mode = mode
            self._redraw_plots()
            
    def _toggle_zoom_box_mode(self, enable):
        """
        Slot for the 'Zoom Box Mode' checkbox.
        Updates zoom box mode state and applies it to plots.
        """
        self.zoom_box_mode = enable
        self._apply_zoom_box_mode()

    def _apply_zoom_box_mode(self):
        """Applies the current zoom_box_mode state to the grid subplots."""
        for widget in (self.mag_sweep_plots_cache + self.iq_sweep_plots_cache
                       + self.fit_sweep_plots_cache):
            view_box = widget.getViewBox()
            if isinstance(view_box, ClickableViewBox):
                view_box.enableZoomBoxMode(self.zoom_box_mode)

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

        # What the call is about to do, until its first sweep reports back
        self.current_amp_label = QtWidgets.QLabel(self._planned_sweeps_text())
        self.current_amp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.current_amp_label)
        
        layout.addWidget(self.progress_group)

    def _hide_progress_bars(self):
        """Hide the entire Analysis Progress group."""
        if self.progress_group:
            self.progress_group.hide()

    def connect_task_signals(self, signals):
        """Route one task's signals to this panel's slots.

        Each task carries its own signals object, so several panels can sweep
        at once.
        """
        queued = QtCore.Qt.ConnectionType.QueuedConnection
        signals.progress.connect(self.update_progress, queued)
        signals.partial_data.connect(self.add_partial_sweep, queued)
        signals.sweep_completed.connect(self.handle_sweep_completed, queued)
        signals.completed.connect(self.complete_multisweep, queued)
        signals.error.connect(self.handle_error, queued)

    def add_partial_sweep(self, module: int, partial: dict, step: int, direction: str):
        """The points measured so far, for the region being swept.

        The driver resends each sweep whole, from its first point, so this
        replaces per resonator rather than appending; resonators in regions it
        has already finished keep the last it sent. Redraws are coalesced,
        because a callback arrives per point and a grid takes longer to draw
        than a point takes to measure.
        """
        if module != self.target_module:
            return
        for name, sweep in partial.items():
            self._live.setdefault(name, {})[(step, direction)] = sweep
        if not self._live_redraw_timer.isActive():
            self._live_redraw_timer.start(LIVE_REDRAW_INTERVAL_MS)

    def handle_sweep_completed(self, record: dict):
        """One sweep of the call is finished: say which, and how far in."""
        self.current_amp_label.setText(
            f"Sweep {record['completed']}/{record['total']}: "
            f"step {record['step']}, {record['direction']}")

    def show_measurement(self, module: int, container: dict):
        """Hold a multisweep and draw it, however it arrived.

        The catalog and the amplitude schedule are read back out of the
        measurement rather than kept beside it: a sweep records both, so a file
        opened an hour later knows the array it swept and the drives it walked
        without being told. One sweep off the board and one off a file reach
        the panel the same way and through here.
        """
        self.multisweep_container = container
        self.module_sweeps = next(
            block for block in container.values() if block['module'] == module)
        call_params = self.module_sweeps['call_params']
        self.catalog = ResonatorCatalog.from_dict(call_params['catalog'])
        self._live_redraw_timer.stop()
        self._live.clear()
        self._set_amplitude_scale(
            AmplitudeSchedule.from_dict(call_params['amp_schedule']))
        self._populate_fit_amplitudes()
        self._populate_fit_models()
        self.bias_settings.set_directions_swept(call_params.get('directions'))
        self.bias_report = None
        self._redraw_plots()

    def complete_multisweep(self, module: int, container: dict):
        """The call has returned: hold it, put the progress report away, and
        say it is ready to be saved."""
        self.show_measurement(module, container)
        self.progress_bar.setValue(100)
        self.current_amp_label.setText(
            f"{len(self.catalog.names())} resonators swept")
        self._hide_progress_bars()
        self.sweep_finished.emit()

    def save_multisweep(self) -> Optional[Path]:
        """Write the measurement through ``store``, and return where it went.

        The container as the driver returned it, so it opens in a notebook with
        ``store.load``. Saving the same panel twice overwrites the same file:
        the container carries the path it was written to.
        """
        if not self.multisweep_container:
            return None
        return store.save(self.multisweep_container, "multisweep",
                          label=self.initial_params.get("label"))

    def _save_multisweep_action(self):
        """The Save button: write the file, say where, and dialog only on failure."""
        try:
            path = self.save_multisweep()
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self, "Save Error", f"Could not save this multisweep:\n{e}")
            return
        self.current_amp_label.setText(
            f"Saved {path.name}" if path else "Nothing measured yet, so nothing to save.")

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
        
    def update_data(self, module: int, iteration: int, amplitude: float, direction: str, results_for_plotting: dict, results_for_history: dict):
        """
        Receives final data for a completed iteration of a multisweep for the target module.
        Stores the data for plotting and updates the CF history.

        Args:
            module (int): The module reporting data.
            iteration (int): The current iteration index.
            amplitude (float): The probe amplitude for which data is provided.
            direction (str): The sweep direction ("upward" or "downward").
            results_for_plotting (dict): Data for plotting, format: {output_cf: data_dict_val}.
            results_for_history (dict): Data for history, format: {conceptual_idx: output_cf_key}.
        """
        if module != self.target_module: return
        
        self.current_amplitude_being_processed = amplitude
        self.current_iteration_being_processed = iteration

        
        # Store data in detector-based structure, keyed by iteration index.
        # The amplitude and direction are stored inside each entry, not as keys,
        # so that all detectors share the same iteration indices even if they
        # use different amplitudes in the future.
        if results_for_plotting:
            for detector_id, det_data in results_for_plotting.items():
                if detector_id not in self.results_by_detector:
                    self.results_by_detector[detector_id] = {}
                entry = dict(det_data)
                entry['amplitude'] = amplitude
                entry['direction'] = direction
                entry['iteration'] = iteration
                self.results_by_detector[detector_id][iteration] = entry

        # --- Update CF history using the pre-mapped results_for_history ---
        if results_for_history:
            self.last_output_cfs_by_amp_and_conceptual_idx.setdefault(amplitude, {}).update(results_for_history)

        self._redraw_plots() # Refresh plots with the new data

    @property
    def conceptual_section_frequencies(self) -> list[float]:
        """Where each resonator sits, for the noise panel's channel mapping.

        Off the catalog rather than off a list beside it, so it cannot go stale
        when the array does.
        """
        if self.catalog is None:
            return []
        return [self.catalog[name].bias.frequency_hz
                for name in self.catalog.names()]

    def _set_amplitude_scale(self, amp):
        """Resolve the schedule's amplitudes for the catalog being swept.

        ``resolve_steps`` answers without a board, so every amplitude the call
        will produce is known before its first point and the colour scale is
        settled from the start.
        """
        if self.catalog is None:
            self._step_amplitudes = {}
            return
        schedule = amp if isinstance(amp, AmplitudeSchedule) else AmplitudeSchedule(amp)
        self._step_amplitudes = {
            step.step: dict(step.amplitudes)
            for step in schedule.resolve_steps(self.catalog)
        }

    def _amplitude_of(self, step: int, name: str, sweep: dict) -> float:
        """What one sweep is driven at, in normalized DAC units.

        A finished sweep records it. One still being measured does not, so the
        answer is what the schedule resolved for that step and resonator, which
        is the number the driver will write into it.
        """
        if 'sweep_amplitude' in sweep:
            return float(sweep['sweep_amplitude'])
        return self._step_amplitudes[step][name]

    def _selected_names(self) -> list[str]:
        """The resonators the grids draw, in the order they are drawn."""
        return list(self.catalog.names()) if self.catalog is not None else []

    def _collect_traces(self, names) -> dict:
        """``{name: [(step, direction, amplitude, sweep), ...]}`` to draw.

        One walk, over the live buffer while a call is running and over the
        block once it has returned -- the live buffer is only ever non-empty in
        between, and completion clears it. Nothing is copied: a sweep here is
        the entry the driver wrote, read at draw time and thrown away after.
        """
        collected = {}
        for name in names:
            traces = []
            if self._live:
                for (step, direction), sweep in self._live.get(name, {}).items():
                    traces.append((step, direction,
                                   self._amplitude_of(step, name, sweep), sweep))
            elif self.module_sweeps is not None:
                measured = collect_amplitude_iterations_for(self.module_sweeps, name)
                for step, by_direction in measured.items():
                    for direction, sweep in by_direction.items():
                        traces.append((step, direction,
                                       self._amplitude_of(step, name, sweep), sweep))
            if traces:
                collected[name] = traces
        return collected

    def _amplitudes_drawn(self) -> list[float]:
        """Every drive amplitude the call produces, for the colour scale."""
        return sorted({a for step in self._step_amplitudes.values()
                       for a in step.values()})

    def _redraw_plots(self):
        """Redraw the grid on the active tab."""
        if self.module_sweeps is None and not self._live:
            return
        self._redraw_sweep_grid(self.plot_tabs.currentIndex())
    
    def _redraw_sweep_grid(self, tab_idx):
        """Redraw one tab's grid: magnitude (0), IQ (1), fits (2), bias (3)."""
        from .multisweep_grid_helpers import update_sweep_grid

        names = self._selected_names()
        traces_by_name = self._collect_traces(names)
        if tab_idx == 2:
            traces_by_name = {
                name: [t for t in traces_by_name.get(name, []) if t[3].get('fits')]
                for name in names}
        if not traces_by_name:
            return

        amplitudes = self._amplitudes_drawn()
        amplitude_to_color = create_amplitude_color_map(amplitudes, self.dark_mode)

        # Get DAC scale for label formatting
        dac_scale = self.dac_scales.get(self.active_module_for_dac)

        # Determine plot type, grid, cache, and colorbar based on tab
        if tab_idx == 0:
            plot_type = 'magnitude'
            grid_layout = self.mag_sweeps_grid
            widget_cache = self.mag_sweep_plots_cache
            colorbar = self.mag_colorbar
        elif tab_idx == 2:
            plot_type = 'fit'
            grid_layout = self.fit_sweeps_grid
            widget_cache = self.fit_sweep_plots_cache
            colorbar = self.fit_colorbar
        elif tab_idx == 3:
            plot_type = 'bias'
            grid_layout = self.bias_sweeps_grid
            widget_cache = self.bias_sweep_plots_cache
            colorbar = self.bias_colorbar
        else:  # tab_idx == 1
            plot_type = 'iq'
            grid_layout = self.iq_sweeps_grid
            widget_cache = self.iq_sweep_plots_cache
            colorbar = self.iq_colorbar

        has_downward = any(direction == 'downward'
                           for traces in traces_by_name.values()
                           for _step, direction, _amp, _sweep in traces)

        # Show colorbar when the colormap is active (num_amps > threshold),
        # otherwise use per-plot legends with TABLEAU10 colors.
        if len(amplitudes) > AMPLITUDE_COLORMAP_THRESHOLD:
            colorbar.update_range(amplitudes[0], amplitudes[-1],
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
            traces_by_name=traces_by_name,
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
            fit_model=self.fit_settings.get_display_model() or 'skewed',
            bias_by_name=self._bias_by_name(),
            bias_settings=self.bias_settings.get_parameters(),
        )

    def _bias_by_name(self) -> dict:
        """``{name: BiasFinding}`` for the grids to mark, empty until a
        report exists."""
        if self.bias_report is None:
            return {}
        return {f.name: f for f in self.bias_report.findings}

    # ── fitting ──────────────────────────────────────────────────────────────

    def _show_fit_settings(self):
        """Raise the fitters' settings window; it outlives any one fit."""
        self.fit_settings.show()
        self.fit_settings.raise_()
        self.fit_settings.activateWindow()

    def _populate_fit_amplitudes(self):
        """The amplitude choices, from the steps this measurement actually has.

        Rebuilt whenever a measurement arrives, because a schedule's steps are
        the choice: "step 3" means nothing until something has been swept.
        """
        dac_scale = self.dac_scales.get(self.active_module_for_dac)
        self.fit_settings.set_amplitude_choices(
            [("All amplitudes", ALL_AMPLITUDES),
             ("At bias amplitude", BIAS_AMPLITUDE)]
            + [(f"Step {step}: {self._step_label(step, dac_scale)}", step)
               for step in sorted(self._step_amplitudes)])

    def _populate_fit_models(self):
        """Offer the models this measurement has fits for, keeping the choice.

        What was fitted, not what the settings ask for: a block loaded from a
        file was fitted by whatever fitted it, and one whose fits are still
        being run has none of them yet.
        """
        self.fit_settings.set_models_fitted(self._models_fitted())

    def _models_fitted(self) -> list:
        """The models the sweeps carry fits for, in the order the tab lists them."""
        if self.module_sweeps is None:
            return []
        present = {model
                   for name in self._selected_names()
                   for by_direction in
                   collect_amplitude_iterations_for(self.module_sweeps, name).values()
                   for sweep in by_direction.values()
                   for model in (sweep.get('fits') or {})}
        return [model for model in FIT_MODELS if model in present]

    def _step_label(self, step: int, dac_scale) -> str:
        """One step's drive, as a range when the resonators differ.

        A relative schedule drives every resonator at its own amplitude, so a
        step is a set of numbers rather than one; saying so is the difference
        between choosing a step and guessing at it.
        """
        amplitudes = sorted(self._step_amplitudes[step].values())
        low = UnitConverter.format_probe_label(amplitudes[0], self.unit_mode, dac_scale)
        if amplitudes[0] == amplitudes[-1]:
            return low
        high = UnitConverter.format_probe_label(amplitudes[-1], self.unit_mode, dac_scale)
        return f"{low} to {high}"

    def _run_fits(self):
        """Fit the chosen sweeps, off the GUI thread."""
        if self.module_sweeps is None:
            self._show_fit_status("Nothing swept yet", ok=False)
            return

        parameters = self.fit_settings.get_parameters()
        if not parameters["models"]:
            self._show_fit_status("No models to fit", ok=False)
            return

        self._set_analysis_enabled(False)
        self._show_fit_status("Fitting...", transient=False)

        signals = RunFitsSignals()
        signals.progress.connect(self._fits_progress)
        signals.completed.connect(self._fits_completed)
        signals.error.connect(self._fits_error)
        # Held so the thread is not collected while it runs.
        self._run_fits_task = RunFitsTask(
            self.module_sweeps, parameters["models"],
            parameters["amplitude_choice"], signals)
        self._run_fits_task.start()

    def _show_fit_status(self, message: str, *, ok: bool = True,
                         transient: bool = True) -> None:
        """Say what the fits are doing, and stop saying it after a while.

        Green for done, red for a failure, as the netanal panel's status line
        reads. Progress is not transient: a timer that cleared it mid-fit would
        leave a dead button with nothing next to it.
        """
        self.fit_status_label.setText(message)
        colour = TABLEAU10_COLORS[2] if ok else TABLEAU10_COLORS[3]
        self.fit_status_label.setStyleSheet(f"color: {colour};")
        self._fit_status_timer.stop()
        if transient:
            self._fit_status_timer.start(STATUS_MESSAGE_MS)

    def _fits_progress(self, completed: int, total: int):
        self._show_fit_status(
            f"Fitting... {100 * completed // max(1, total)}%", transient=False)

    def _fits_completed(self, report):
        """The fits are in the sweeps the panel holds: draw them, and re-save."""
        self._fits_done()
        message = f"{len(report.fitted)}/{len(report)} fitted"
        if report.failed:
            message += f", {len(report.failed)} failed"
        # The fits went into the block, so a file that exists is now out of
        # date by exactly this much. A panel never saved keeps the Save button.
        if store.saved_path(self.multisweep_container):
            try:
                message += f" -- saved to {self.save_multisweep().name}"
            except Exception as e:                      # noqa: BLE001 - reported
                traceback.print_exc()
                self._show_fit_status(f"{message}, but the save failed: {e}", ok=False)
                self._redraw_plots()
                return
        self._show_fit_status(message)
        self._redraw_plots()

    def _fits_error(self, message: str):
        self._fits_done()
        self._show_fit_status(message, ok=False)

    def _fits_done(self):
        self._set_analysis_enabled(True)
        self._populate_fit_models()

    # ── bias finding ─────────────────────────────────────────────────────────

    def _show_bias_settings(self):
        """Raise bias finding's settings window; it outlives any one run."""
        self.bias_settings.show()
        self.bias_settings.raise_()
        self.bias_settings.activateWindow()

    def _find_bias(self):
        """Choose an operating point for every resonator, off the GUI thread."""
        if self.module_sweeps is None:
            self._show_bias_status("Nothing swept yet", ok=False)
            return

        span_hz = self.module_sweeps['call_params'].get('span_hz')
        parameters = self.bias_settings.get_parameters(span_hz=span_hz)

        self._set_analysis_enabled(False)
        self._show_bias_status("Finding bias...", transient=False)

        signals = FindBiasSignals()
        signals.completed.connect(self._bias_found)
        signals.error.connect(self._bias_error)
        # Held so the thread is not collected while it runs.
        self._find_bias_task = FindBiasTask(
            self.module_sweeps, parameters, signals)
        self._find_bias_task.start()

    def _show_bias_status(self, message: str, *, ok: bool = True,
                          transient: bool = True) -> None:
        """Say what bias finding is doing, and stop saying it after a while."""
        self.bias_status_label.setText(message)
        colour = TABLEAU10_COLORS[2] if ok else TABLEAU10_COLORS[3]
        self.bias_status_label.setStyleSheet(f"color: {colour};")
        self._bias_status_timer.stop()
        if transient:
            self._bias_status_timer.start(STATUS_MESSAGE_MS)

    def _bias_found(self, report):
        """The report's catalog is the array now: hold it, draw it, re-save."""
        self._set_analysis_enabled(True)
        self.bias_report = report
        self.catalog = report.catalog

        # The report's own words: every resonator gets a bias point, and a flag
        # says that one is a fallback rather than a measurement.
        message = f"{len(report)} biased"
        if report.flagged:
            names = ", ".join(f.name for f in report.flagged[:3])
            if len(report.flagged) > 3:
                names += f", +{len(report.flagged) - 3} more"
            message += f", {len(report.flagged)} flagged: {names}"
        # The report went into the block, so a file that exists is now out of
        # date by exactly this much.
        if store.saved_path(self.multisweep_container):
            try:
                message += f" -- saved to {self.save_multisweep().name}"
            except Exception as e:                      # noqa: BLE001 - reported
                traceback.print_exc()
                self._show_bias_status(
                    f"{message}, but the save failed: {e}", ok=False)
                self._redraw_plots()
                return
        # A flag is the thing to read before applying anything, so it stays on
        # screen; a clean run says so and gets out of the way.
        self._show_bias_status(message, ok=not report.flagged,
                               transient=not report.flagged)
        self._redraw_plots()

    def _bias_error(self, message: str):
        self._set_analysis_enabled(True)
        self._show_bias_status(message, ok=False)

    # ── applying it ──────────────────────────────────────────────────────────

    def _apply_bias(self):
        """Park a tone on every resonator, off the GUI thread.

        The catalog is the whole of the instruction. Which NCO carries it, and
        putting the frequencies on the tone grid, are ``apply_bias``'s -- this
        panel does neither.
        """
        if self.catalog is None or len(self.catalog) == 0:
            self._show_bias_status("Nothing to bias", ok=False)
            return
        periscope = self._get_periscope_parent()
        if periscope is None or periscope.crs is None:
            self._show_bias_status("No board to bias", ok=False)
            return

        self.apply_bias_btn.setEnabled(False)
        self._show_bias_status("Applying bias...", transient=False)

        signals = ApplyBiasSignals()
        signals.completed.connect(self._bias_applied)
        signals.error.connect(self._apply_bias_error)
        # Held so the thread is not collected while it runs.
        self._apply_bias_task = ApplyBiasTask(
            periscope.crs, self.catalog, signals)
        self._apply_bias_task.start()

    def _bias_applied(self):
        """The tones are on the air: publish what reads them in hertz."""
        self.apply_bias_btn.setEnabled(True)
        calibrations = {r.channel: r.bias.df_calibration for r in self.catalog
                        if r.bias.df_calibration is not None}
        if calibrations:
            self.df_calibration_ready.emit(self.target_module, calibrations)
        self.bias_data_avail = True
        self.noise_spectrum_btn.setEnabled(True)
        self._show_bias_status("Bias applied")

    def _apply_bias_error(self, message: str):
        self.apply_bias_btn.setEnabled(True)
        self._show_bias_status(message, ok=False)

    def _set_analysis_enabled(self, enabled: bool) -> None:
        """Fitting and bias finding both walk every sweep this panel holds, so
        one runs at a time."""
        self.run_fit_btn.setEnabled(enabled)
        self.find_bias_btn.setEnabled(enabled)

    def handle_error(self, error_msg: str):
        """Say what went wrong where the sweep's progress is reported.

        A modal here is opened from a signal handler, which never returns on a
        headless run and takes the window away from the operator on any other.
        """
        self.current_amp_label.setText(error_msg)
        self.progress_bar.setValue(0)

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
            
        return {
            'timestamp': datetime.datetime.now().isoformat(),
            'target_module': self.target_module,
            'initial_parameters': self.initial_params,
            'dac_scales_used': self.dac_scales,
            'results_by_detector': self.results_by_detector,
            'bias_kids_output': self.bias_kids_output,  # Include bias_kids results if available
            'nco_frequency_hz': self.nco_frequency_hz,  # NCO frequency used for biasing
            'noise_data': spectrum_data
        }
    
    def _rerun_multisweep(self):
        """Sweep the panel's array again, with settings the dialog can change.

        The catalog is the seed: it carries where each resonator is and what it
        is driven at, so a re-run centres on wherever the array is now without
        any history of previous sweeps to consult. After Find Bias that catalog
        is the report's, which is what makes the sweep iterative.
        """
        from .dialogs import MultisweepDialog

        if self.catalog is None:
            self.current_amp_label.setText("Nothing to sweep: no array in this panel.")
            return

        dialog = MultisweepDialog(parent=self, catalog=self.catalog,
                                  dac_scales=self.dac_scales,
                                  initial_params=self.initial_params.copy())
        if not dialog.exec():
            return

        params = dialog.get_parameters()
        if not params:
            return

        self.is_loaded_data = False
        periscope = self._get_periscope_parent()
        if periscope:
            my_dock = periscope.dock_manager.find_dock_for_widget(self)
            if my_dock:
                my_dock.setWindowTitle(
                    my_dock.windowTitle().replace(" (Loaded)", ""))
        self.noise_spectrum_btn.setEnabled(False)

        self.initial_params.update(params)
        self.catalog = params['catalog']
        self._set_amplitude_scale(params['amp'])

        self.multisweep_container = None
        self.module_sweeps = None
        self._live.clear()
        self._redraw_plots()

        self.progress_bar.setValue(0)
        self.progress_group.setVisible(True)
        self.current_amp_label.setText(self._planned_sweeps_text())

        parent_widget = self._get_periscope_parent()
        if parent_widget is None:
            self.current_amp_label.setText(
                "Cannot re-run: no Periscope window to start the sweep from.")
            return
        parent_widget._start_multisweep_analysis_for_window(self, self.initial_params)

    def mark_foreign_module(self, file_module: int) -> None:
        """Say this file was taken on another module, and stop offering to sweep.

        Nothing about the file or the session is rewritten; what goes away is
        the control that would start a measurement from it.
        """
        self.is_foreign_module = True
        self.rerun_btn.setEnabled(False)
        self.apply_bias_btn.setEnabled(False)
        self.apply_bias_btn.setToolTip(
            f"This file was taken on module {file_module}, and this Periscope "
            f"controls module {self.target_module}.")
        self.rerun_btn.setToolTip(
            f"This file was taken on module {file_module}, and this Periscope "
            f"controls module {self.target_module}.")
        self.current_amp_label.setText(
            f"Module {file_module} measurement, shown but not re-runnable here.")

    def _planned_sweeps_text(self) -> str:
        """How many sweeps the configured call will take, before it starts."""
        directions = self.initial_params.get('sweep_direction', 'upward')
        n_directions = 1 if isinstance(directions, str) else len(directions)
        return f"{len(self._step_amplitudes) * n_directions} sweeps to take..."

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

    def _open_noise_spectrum_dialog(self):
        num_res = len(self.conceptual_section_frequencies)
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
                num_res = len(self.conceptual_section_frequencies)
    
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
            
        # Get conceptual frequency for this detector
        if detector_idx <= len(self.conceptual_section_frequencies) and detector_idx > 0:
            conceptual_resonance_base_freq_hz = self.conceptual_section_frequencies[detector_idx - 1]
        else:
            print(f"Warning: Detector index {detector_idx} exceeds conceptual frequencies list length.")
            return
            
        # Gather data for ALL detectors to enable navigation
        all_detectors_data = {}
        # We need conceptual frequencies for navigation
        for i, freq in enumerate(self.conceptual_section_frequencies):
            det_id = i + 1
            all_detectors_data[det_id] = {
                'conceptual_freq_hz': freq
            }
            
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
        
        target_dock = periscope.dock_manager.find_dock_for_widget(self)
        if target_dock:
            periscope.tabifyDockWidget(target_dock, dock)
        
        # Show and activate the dock
        dock.show()
        dock.raise_()

    def _get_closest_remembered_cf(self, conceptual_idx: int, target_amp: float) -> float | None:
        """
        Finds the remembered output CF for a given conceptual section index,
        for the amplitude in history closest to target_amp.

        Args:
            conceptual_idx: Index in self.conceptual_section_frequencies.
            target_amp: The amplitude we are trying to find a historical match for.

        Returns:
            The remembered output CF (float) or None if no suitable history found.
        """
        min_abs_amp_diff = np.inf
        best_cf_found = None

        if not self.last_output_cfs_by_amp_and_conceptual_idx:
            return None

        for amp_in_history, cfs_at_this_amp in self.last_output_cfs_by_amp_and_conceptual_idx.items():
            if conceptual_idx in cfs_at_this_amp:
                remembered_cf = cfs_at_this_amp[conceptual_idx]
                current_diff = abs(amp_in_history - target_amp)

                if current_diff < min_abs_amp_diff:
                    min_abs_amp_diff = current_diff
                    best_cf_found = remembered_cf
                elif current_diff == min_abs_amp_diff:
                    pass
        
        return best_cf_found
    
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
        
        # Redraw plots which will now use the updated legend text colors
        self._redraw_plots()
        
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
        dialog = BiasKidsDialog(self, self.target_module,
                                fits_present=self._fits_present())
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
    
    def _fits_present(self) -> set:
        """Which resonance fits the current results carry, for the Bias
        KIDs dialog to preselect from."""
        from rfmux.algorithms.measurement.df_calibration import fits_present
        return fits_present(entry for iterations in self.results_by_detector.values()
                            for entry in iterations.values())

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
        total_detectors = len(self.conceptual_section_frequencies)
        
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
