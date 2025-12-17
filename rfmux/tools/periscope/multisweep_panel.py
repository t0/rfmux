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
    ScreenshotMixin
)
from .detector_digest_panel import DetectorDigestPanel
from .noise_spectrum_panel import NoiseSpectrumPanel
from .noise_spectrum_dialog import NoiseSpectrumDialog
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
        
        # Stores the initial/base CFs for detector ID and fallback. Order is important.
        self.conceptual_resonance_frequencies: list[float] = list(self.initial_params.get('resonance_frequencies', []))
        # Stores {amp: {conceptual_idx: output_cf}}
        self.last_output_cfs_by_amp_and_conceptual_idx: dict[float, dict[int, float]] = {}
        # Amps used for the last configured/completed run, to compare if settings changed.
        self.current_run_amps: list[float] = list(self.initial_params.get('amps', []))
        # probe_amplitudes is used for progress display, should reflect current_run_amps
        self.probe_amplitudes = list(self.current_run_amps) # Ensure it's a copy and reflects current run

        self.setWindowTitle(f"Multisweep Results - Module {self.target_module}")

        # Data storage and state
        self.results_by_iteration = {}  # Stores {iteration: {"amplitude": amp_val, "direction": direction, "data": {cf: data_dict}}}
        self.current_amplitude_being_processed = None # Tracks the amplitude currently being processed
        self.current_iteration_being_processed = None # Tracks the current iteration
        self.unit_mode = "dbm"  # Current unit for magnitude display ("counts", "dbm", "volts")
        self.normalize_traces = False  # Flag to normalize trace plots (magnitude and phase)
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

        self._setup_ui()
        self.resize(1200, 800) # Default window size

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
        """Sets up the magnitude and phase plot widgets."""
        plot_container = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_container)

        # Magnitude Plot
        vb_mag = ClickableViewBox() # Custom viewbox for potential extra click interactions
        vb_mag.parent_window = self # Link back to this window if needed by viewbox
        self.combined_mag_plot = pg.PlotWidget(viewBox=vb_mag)
        bg_color, pen_color = ("k", "w") if self.dark_mode else ("w", "k")
        plot_item_mag = self.combined_mag_plot.getPlotItem()
        if plot_item_mag: # Add check for None
            plot_item_mag.setTitle("Combined S21 Magnitude (All Resonances)", color=pen_color)
            plot_item_mag.setLabel('bottom', 'Frequency', units='Hz')
            plot_item_mag.showGrid(x=True, y=True, alpha=0.3)
            self.mag_legend = plot_item_mag.addLegend(offset=(10,-50),labelTextColor=pen_color)
        self._update_mag_plot_label() # Set initial Y-axis label based on unit mode
        plot_layout.addWidget(self.combined_mag_plot)

        # Phase Plot
        vb_phase = ClickableViewBox()
        vb_phase.parent_window = self
        self.combined_phase_plot = pg.PlotWidget(viewBox=vb_phase)
        plot_item_phase = self.combined_phase_plot.getPlotItem()
        if plot_item_phase: # Add check for None
            plot_item_phase.setTitle("Combined S21 Phase (All Resonances)", color=pen_color)
            plot_item_phase.setLabel('bottom', 'Frequency', units='Hz')
            plot_item_phase.setLabel('left', 'Phase', units='deg')
            plot_item_phase.showGrid(x=True, y=True, alpha=0.3)
            self.phase_legend = plot_item_phase.addLegend(offset=(10,-50),labelTextColor=pen_color)
        plot_layout.addWidget(self.combined_phase_plot)
        
        layout.addWidget(plot_container)
        
        # Link X-axes of magnitude and phase plots for synchronized zooming/panning
        if self.combined_phase_plot and self.combined_mag_plot: # Add check for None
            self.combined_phase_plot.setXLink(self.combined_mag_plot)
        self._apply_zoom_box_mode() # Apply initial zoom box mode state
        
        # Apply initial theme based on dark_mode setting
        # bg_color, pen_color are already defined
        
        if self.combined_mag_plot:
            self.combined_mag_plot.setBackground(bg_color)
            plot_item_mag_for_axes = self.combined_mag_plot.getPlotItem()
            if plot_item_mag_for_axes: # Add check for None
                for axis_name in ("left", "bottom", "right", "top"):
                    ax = plot_item_mag_for_axes.getAxis(axis_name)
                    if ax:
                        ax.setPen(pen_color)
                        ax.setTextPen(pen_color)
                    
        if self.combined_phase_plot:
            self.combined_phase_plot.setBackground(bg_color)
            plot_item_phase_for_axes = self.combined_phase_plot.getPlotItem()
            if plot_item_phase_for_axes: # Add check for None
                for axis_name in ("left", "bottom", "right", "top"):
                    ax = plot_item_phase_for_axes.getAxis(axis_name)
                    if ax:
                        ax.setPen(pen_color)
                        ax.setTextPen(pen_color)

        # Connect double click signal to both magnitude and phase plot viewboxes
        if self.combined_mag_plot: # Add check for None
            view_box_mag = self.combined_mag_plot.getViewBox()
            if isinstance(view_box_mag, ClickableViewBox):
                view_box_mag.doubleClickedEvent.connect(self._handle_multisweep_plot_double_click)
                
        # Also connect the phase plot's viewbox to the same handler
        if self.combined_phase_plot: # Add check for None
            view_box_phase = self.combined_phase_plot.getViewBox()
            if isinstance(view_box_phase, ClickableViewBox):
                view_box_phase.doubleClickedEvent.connect(self._handle_multisweep_plot_double_click)


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
        if not self.combined_mag_plot: return

        if self.normalize_traces:
            label = "Normalized Magnitude" # Label for magnitude part of the trace
            # Normalized dBm is still in dB, other normalized units are unitless or relative.
            units = "dB" if self.unit_mode == "dbm" else "" 
        else:
            if self.unit_mode == "counts":
                label, units = "Magnitude", "Counts"
            elif self.unit_mode == "dbm":
                label, units = "Power", "dBm"
            elif self.unit_mode == "volts":
                label, units = "Magnitude", "V"
            else: # Fallback, should not ideally be reached if UI is constrained
                label, units = "Magnitude", ""
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
                
            self.current_amp_label.setText(f"Iteration 1/{self.total_iterations}: Amplitude {self.probe_amplitudes[0]:.4f} ({direction_text})")
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
        status_message = f"Iteration {current_display_iteration}/{self.total_iterations}: Amplitude {amplitude:.4f} ({direction_text})"
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

        
        # Store data in iteration-based structure
        self.results_by_iteration[iteration] = {
            "amplitude": amplitude,
            "direction": direction,
            "data": results_for_plotting
        }

        # --- Update CF history using the pre-mapped results_for_history ---
        if results_for_history:
            self.last_output_cfs_by_amp_and_conceptual_idx.setdefault(amplitude, {}).update(results_for_history)

        self._redraw_plots() # Refresh plots with the new data
        
        # Note: We now update the status in handle_starting_iteration() instead of here

    def _redraw_plots(self):
        """
        Clears and redraws all plot items (magnitude and phase curves, legends, CF lines)
        based on the current `results_by_amplitude` and UI settings (unit, normalization).
        """
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

        num_iterations = len(self.results_by_iteration)
        if num_iterations == 0: # No data to plot
            self.combined_mag_plot.autoRange(); self.combined_phase_plot.autoRange() # Ensure plots are reset
            return
        
        # Define color schemes for plotting multiple amplitudes
        # Use a distinct color family for few amplitudes, switch to a colormap for many
        cmap_name = COLORMAP_CHOICES.get("AMPLITUDE_SWEEP", "inferno") # Fallback if key missing
        use_cmap = pg.colormap.get(cmap_name) if cmap_name else None
        
        # Get unique amplitudes to determine color scheme
        amplitude_values = set()
        for iteration_data in self.results_by_iteration.values():
            amplitude_values.add(iteration_data["amplitude"])
        num_amps = len(amplitude_values)
            
        # Create a mapping for unique amplitude values to colors
        sorted_amplitudes = sorted(amplitude_values)
        amplitude_to_color = {}
        for amp_idx, amp_val in enumerate(sorted_amplitudes):
            # Determine color for this amplitude's curves
            if num_amps <= AMPLITUDE_COLORMAP_THRESHOLD: # Use distinct colors if few amplitudes
                color = TABLEAU10_COLORS[amp_idx % len(TABLEAU10_COLORS)]
            else: # Use a colormap for many amplitudes
                if use_cmap:
                    normalized_idx = amp_idx / max(1, num_amps - 1) # Normalize index for colormap base [0,1]
                    if self.dark_mode:
                        # For dark mode, map to [0.3, 1.0]
                        map_value = 0.3 + normalized_idx * 0.7
                    else:
                        # For light mode, map to [0.0, 0.75]
                        map_value = normalized_idx * 0.75
                    color = use_cmap.map(map_value)
                else: # Fallback if colormap is somehow None
                    color = TABLEAU10_COLORS[amp_idx % len(TABLEAU10_COLORS)]
            amplitude_to_color[amp_val] = color
        
        legend_items_mag = {} # To avoid duplicate legend entries for the same amplitude/direction combination
        legend_items_phase = {}
        
        # Iterate through each iteration's results
        for iteration, iteration_data in self.results_by_iteration.items():
            amp_val = iteration_data["amplitude"]
            direction = iteration_data["direction"]
            amp_results = iteration_data["data"]
            
            # Get color for this amplitude
            color = amplitude_to_color[amp_val]
            
            # Set line style based on direction using constants from utils.py
            line_style = DOWNWARD_SWEEP_STYLE if direction == "downward" else UPWARD_SWEEP_STYLE
            pen = pg.mkPen(color, width=LINE_WIDTH, style=line_style)
            
            # --- Prepare Legend Entry for this Amplitude ---
            legend_name_amp = ""
            dac_scale_for_module = self.dac_scales.get(self.active_module_for_dac)

            if self.unit_mode == "dbm":
                if dac_scale_for_module is not None:
                    dbm_val = UnitConverter.normalize_to_dbm(amp_val, dac_scale_for_module)
                    legend_name_amp = f"Probe: {dbm_val:.2f} dBm"
                else: # Fallback if DAC scale is missing
                    legend_name_amp = f"Probe: {amp_val:.3e} (Norm)" 
            elif self.unit_mode == "volts":
                if dac_scale_for_module is not None:
                    dbm_val = UnitConverter.normalize_to_dbm(amp_val, dac_scale_for_module)
                    power_watts = 10**((dbm_val - 30)/10) # Convert dBm to Watts
                    resistance = 50.0 # Standard impedance assumption
                    voltage_rms = np.sqrt(power_watts * resistance)
                    voltage_peak_uv = voltage_rms * np.sqrt(2) * 1e6 # Convert RMS to peak uV
                    legend_name_amp = f"Probe: {voltage_peak_uv:.1f} uVpk"
                else: # Fallback if DAC scale is missing
                    legend_name_amp = f"Probe: {amp_val:.3e} (Norm)"
            else: # Counts or other modes
                legend_name_amp = f"Probe: {amp_val:.3e} Norm"

            # Add direction to legend name
            direction_suffix = " (Down)" if direction == "downward" else " (Up)"
            full_legend_name = legend_name_amp + direction_suffix
            
            # Create a unique key for this amplitude+direction combination
            legend_key = (amp_val, direction)
            
            # Add legend item once per amplitude+direction combination
            if legend_key not in legend_items_mag and self.mag_legend:
                # Create a dummy item for the legend (actual curves are added later)
                dummy_mag_curve_for_legend = pg.PlotDataItem(pen=pen) 
                self.mag_legend.addItem(dummy_mag_curve_for_legend, full_legend_name)
                legend_items_mag[legend_key] = dummy_mag_curve_for_legend # Mark as added
            if legend_key not in legend_items_phase and self.phase_legend:
                dummy_phase_curve_for_legend = pg.PlotDataItem(pen=pen)
                self.phase_legend.addItem(dummy_phase_curve_for_legend, full_legend_name)
                legend_items_phase[legend_key] = dummy_phase_curve_for_legend

            # --- Plot Data for Each Resonance (by index) at this Amplitude ---
            for res_idx, data in amp_results.items():
                freqs_hz = data.get('frequencies')
                iq_complex = data.get('iq_complex')

                if freqs_hz is None or iq_complex is None or len(freqs_hz) == 0:
                    # Skip if essential data is missing or empty
                    continue
                
                # Calculate magnitude and phase
                s21_mag_raw = np.abs(iq_complex)
                s21_mag_processed = UnitConverter.convert_amplitude(
                    s21_mag_raw, iq_complex, self.unit_mode, 
                    normalize=self.normalize_traces
                )
                # Use pre-calculated phase if available, otherwise calculate from IQ
                phase_deg = data.get('phase_degrees', np.degrees(np.angle(iq_complex))) 
                
                if self.normalize_traces and len(phase_deg) > 0: # New phase normalization
                    first_phase_val = phase_deg[0]
                    if np.isfinite(first_phase_val): # Avoid issues with NaN/inf
                        phase_deg = phase_deg - first_phase_val
                
                # Plot magnitude curve
                mag_curve = self.combined_mag_plot.plot(pen=pen)
                mag_curve.setData(freqs_hz, s21_mag_processed)
                if amp_val not in self.curves_mag: self.curves_mag[amp_val] = {}
                self.curves_mag[amp_val][res_idx] = mag_curve # Store reference by index

                # Plot phase curve
                phase_curve = self.combined_phase_plot.plot(pen=pen)
                phase_curve.setData(freqs_hz, phase_deg)
                if amp_val not in self.curves_phase: self.curves_phase[amp_val] = {}
                self.curves_phase[amp_val][res_idx] = phase_curve # Store reference by index

                # Add center frequency (CF) lines if enabled
                if self.show_cf_lines_cb and self.show_cf_lines_cb.isChecked():
                    # Get the actual bias frequency for CF line
                    bias_freq = data.get('bias_frequency', data.get('original_center_frequency'))
                    if bias_freq is not None:
                        cf_line_pen = pg.mkPen(color, style=QtCore.Qt.PenStyle.DashLine, width=LINE_WIDTH/2)
                        
                        mag_cf_line = pg.InfiniteLine(pos=bias_freq, angle=90, pen=cf_line_pen, movable=False)
                        self.combined_mag_plot.addItem(mag_cf_line)
                        self.cf_lines_mag.setdefault(amp_val, []).append(mag_cf_line) # Store reference

                        phase_cf_line = pg.InfiniteLine(pos=bias_freq, angle=90, pen=cf_line_pen, movable=False)
                        self.combined_phase_plot.addItem(phase_cf_line)
                        self.cf_lines_phase.setdefault(amp_val, []).append(phase_cf_line) # Store reference
        
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
        if self.results_by_iteration:
            export_data = self._prepare_export_data()
            identifier = f"module{self.target_module}"
            self.data_ready.emit("multisweep", identifier, export_data)
        
        # Auto-open detector digest for the first detector (lowest frequency)
        if self.results_by_iteration:
            self._open_detector_digest_for_index(1)
        
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
            
        if not self.results_by_iteration:
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
            
        return {
            'timestamp': datetime.datetime.now().isoformat(),
            'target_module': self.target_module,
            'initial_parameters': self.initial_params,
            'dac_scales_used': self.dac_scales,
            'results_by_iteration': self.results_by_iteration,
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

    def _get_fit_frequencies(self, freqs):
        ref_freqs = []
        for i in range(len(freqs)):
            if self.results_by_iteration[0]['data'][i+1]['skewed_fit_success']:
                ref_freqs.append(self.results_by_iteration[0]['data'][i+1]['fit_params']['fr'])
            elif self.results_by_iteration[0]['data'][i+1]['nonlinear_fit_success']:
                ref_freqs.append(self.results_by_iteration[0]['data'][i+1]['nonlinear_fit_params']['fr'])
            else:
                ref_freqs.append(self.results_by_iteration[0]['data'][i+1]['bias_frequency'])
                
        ref_freqs.sort()
        return ref_freqs
    
    
    def _rerun_multisweep(self):
        """
        Allows the user to re-run the multisweep analysis, potentially with modified parameters.
        Opens a MultisweepDialog to gather new parameters.
        """
        # Ensure MultisweepDialog is available (local import to avoid circular dependencies if any)
        from .dialogs import MultisweepDialog

        if not self.conceptual_resonance_frequencies: # Check conceptual frequencies
            QtWidgets.QMessageBox.warning(self, "Cannot Re-run",
                                          "No center frequencies are known to periscope to run on.")
            return

        # --- Determine frequencies to seed the dialog ---
        dialog_seed_frequencies = list(self.conceptual_resonance_frequencies) # Start with conceptual

        ##### Getting the fit values for updating in re-run ######

        fit_freqs = self._get_fit_frequencies(dialog_seed_frequencies)

        
        if self.current_run_amps: # If there was a previous/current run configuration
            # Use a representative amplitude from the current/last run to seed the dialog
            # For simplicity, let's use the first amplitude from the current_run_amps.
            representative_amp_for_seeding = self.current_run_amps[0]
            for idx, conceptual_cf in enumerate(self.conceptual_resonance_frequencies):
                remembered_cf = self._get_closest_remembered_cf(idx, representative_amp_for_seeding)
                if remembered_cf is not None:
                    dialog_seed_frequencies[idx] = remembered_cf
        
        
        # Prepare other parameters for the dialog
        dialog_initial_params = self.initial_params.copy() # Use a copy of the window's last run parameters
        # The 'resonance_frequencies' in dialog_initial_params will be overwritten by dialog_seed_frequencies
        # when creating the dialog instance if MultisweepDialog uses its 'initial_params' argument
        # to populate its own 'resonance_frequencies' field.
        # However, MultisweepDialog takes 'resonance_frequencies' as a direct argument.

        dialog = MultisweepDialog(
            parent=self,
            resonance_frequencies=dialog_seed_frequencies, # Seed with potentially updated CFs
            dac_scales=self.dac_scales,
            current_module=self.target_module,
            initial_params=dialog_initial_params, # Pass other existing params
            load_multisweep = False,
            fit_frequencies = fit_freqs
        )

        if dialog.exec(): # True if user clicked OK
            # Reset the loaded data flag since we're now generating fresh data
            self.is_loaded_data = False
            
            # Update the dock title to remove "(Loaded)" suffix
            periscope = self._get_periscope_parent()
            if periscope:
                my_dock = periscope.dock_manager.find_dock_for_widget(self)
                if my_dock:
                    # Get current title and remove " (Loaded)" if present
                    current_title = my_dock.windowTitle()
                    new_title = current_title.replace(" (Loaded)", "")
                    my_dock.setWindowTitle(new_title)
            
            self.noise_spectrum_btn.setEnabled(False)
            new_params_from_dialog = dialog.get_parameters()
            if not new_params_from_dialog:
                return # Dialog returned None, likely due to validation error

            new_amps_for_this_run = list(new_params_from_dialog.get('amps', []))
            # resonance_frequencies_from_dialog are the ones dialog was seeded with, as it doesn't change them.
            resonance_frequencies_from_dialog = list(new_params_from_dialog.get('resonance_frequencies', []))

            # --- Determine the final input CFs for the new sweep task ---
            # This list will be passed to the MultisweepTask as its baseline.
            # The task itself will then refine this per amplitude.
            final_baseline_cfs_for_new_task = list(self.conceptual_resonance_frequencies)

            if not new_amps_for_this_run: # No amplitudes specified, fall back or warn
                 QtWidgets.QMessageBox.warning(self, "Configuration Error", "No amplitudes specified for the new sweep.")
                 # Default to conceptual, or could use resonance_frequencies_from_dialog
                 final_baseline_cfs_for_new_task = resonance_frequencies_from_dialog if resonance_frequencies_from_dialog else list(self.conceptual_resonance_frequencies)
            elif new_amps_for_this_run == self.current_run_amps:
                # Amplitudes haven't changed from the last run configuration.
                # Use the frequencies that were in the dialog (which were seeded from history).
                final_baseline_cfs_for_new_task = resonance_frequencies_from_dialog if resonance_frequencies_from_dialog else list(self.conceptual_resonance_frequencies)
            else:
                # Amplitudes have changed. For each conceptual resonance,
                # find the best historical CF based on the *new* representative amplitude.
                # If no history, use what was in the dialog (which was seeded based on old rep. amp or conceptual).
                if new_amps_for_this_run: # Ensure there's at least one new amp
                    representative_new_amp = new_amps_for_this_run[0]
                    for idx, conceptual_cf in enumerate(self.conceptual_resonance_frequencies):
                        remembered_cf = self._get_closest_remembered_cf(idx, representative_new_amp)
                        if remembered_cf is not None:
                            final_baseline_cfs_for_new_task[idx] = remembered_cf
                        else:
                            # Fallback to what was in the dialog for this index if no better history for new amp
                            if idx < len(resonance_frequencies_from_dialog):
                                 final_baseline_cfs_for_new_task[idx] = resonance_frequencies_from_dialog[idx]
                            # Else it remains the conceptual_cf (already initialized)
                else: # Should be caught by the "No amplitudes specified" case, but as a safeguard
                    final_baseline_cfs_for_new_task = resonance_frequencies_from_dialog if resonance_frequencies_from_dialog else list(self.conceptual_resonance_frequencies)


            # Update the 'resonance_frequencies' in new_params_from_dialog to be this chosen baseline
            new_params_from_dialog['resonance_frequencies'] = final_baseline_cfs_for_new_task
            
            # Store the parameters that will actually be used for this run
            self.initial_params.update(new_params_from_dialog)
            self.current_run_amps = new_amps_for_this_run # Update current run amps
            self.probe_amplitudes = list(self.current_run_amps) # For progress display

            # Reset window state for the new sweep
            self.results_by_iteration.clear()
            self._redraw_plots() # Clear plots
            if self.progress_bar: self.progress_bar.setValue(0)
            if self.progress_group: self.progress_group.setVisible(True)
            
            # Re-calculate total iterations based on new sweep direction
            num_amplitudes = len(self.probe_amplitudes)
            sweep_direction = self.initial_params.get('sweep_direction', 'upward')
            self.total_iterations = num_amplitudes * (2 if sweep_direction == "both" else 1)
            
            # Determine initial direction text - consistent with our other direction text logic
            sweep_direction_norm = sweep_direction.lower().strip() if sweep_direction else ""
            direction_text = "Down" if sweep_direction_norm == "downward" else "Up"

            
            
            if self.current_amp_label:
                if num_amplitudes > 0:
                    first_amplitude = self.probe_amplitudes[0]
                    self.current_amp_label.setText(f"Iteration 1/{self.total_iterations}: Amplitude {first_amplitude:.4f} ({direction_text})")
                else:
                    self.current_amp_label.setText("No sweeps defined. (Waiting...)")
            
            parent_widget = self._get_periscope_parent()
            if parent_widget and hasattr(parent_widget, '_start_multisweep_analysis_for_window'):
                # Pass self.initial_params which now contains the correctly determined baseline CFs
                parent_widget._start_multisweep_analysis_for_window(self, self.initial_params) # type: ignore
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
            for iteration_data in self.results_by_iteration.values():
                amplitude_values.add(iteration_data["amplitude"])
            num_amps = len(amplitude_values)
            
            if num_amps == 0:
                return

            # Color definitions (consistent with _redraw_plots)
            cmap_name = COLORMAP_CHOICES.get("AMPLITUDE_SWEEP", "inferno")
            use_cmap = pg.colormap.get(cmap_name) if cmap_name else None
            sorted_amplitudes = sorted(amplitude_values)

            for amp_idx, amp_val in enumerate(sorted_amplitudes):
                # Find the latest iteration for this amplitude
                latest_iter_idx = None
                latest_iter_data = None
                for iter_idx, iter_data in self.results_by_iteration.items():
                    if iter_data["amplitude"] == amp_val:
                        if latest_iter_idx is None or iter_idx > latest_iter_idx:
                            latest_iter_idx = iter_idx
                            latest_iter_data = iter_data
                
                if latest_iter_data is None:
                    continue
                    
                amp_results = latest_iter_data.get("data", {})
                
                # Determine color for this amplitude's lines
                if num_amps <= AMPLITUDE_COLORMAP_THRESHOLD: # Use consistent threshold from utils.py
                    color = TABLEAU10_COLORS[amp_idx % len(TABLEAU10_COLORS)]
                else: # Use a colormap for many amplitudes
                    if use_cmap:
                        normalized_idx = amp_idx / max(1, num_amps - 1) # Normalize index for colormap base [0,1]
                        if self.dark_mode:
                            # For dark mode, map to [0.3, 1.0]
                            map_value = 0.3 + normalized_idx * 0.7
                        else:
                            # For light mode, map to [0.0, 0.75]
                            map_value = normalized_idx * 0.75
                        color = use_cmap.map(map_value)
                    else: # Fallback if colormap is somehow None
                        color = TABLEAU10_COLORS[amp_idx % len(TABLEAU10_COLORS)]
                
                # Define pen for CF lines (consistent with _redraw_plots)
                # Note: LINE_WIDTH should be available from 'from .utils import LINE_WIDTH'
                cf_line_pen = pg.mkPen(color, style=QtCore.Qt.PenStyle.DashLine, width=LINE_WIDTH/2)

                # Ensure lists for this amplitude exist in cf_lines_mag/phase
                self.cf_lines_mag.setdefault(amp_val, [])
                self.cf_lines_phase.setdefault(amp_val, [])

                # Create dictionaries for quick lookup of existing lines by their X-position (CF)
                # This avoids iterating through the list of lines repeatedly for each CF.
                existing_mag_lines_for_amp = {line.pos().x(): line for line in self.cf_lines_mag[amp_val]}
                existing_phase_lines_for_amp = {line.pos().x(): line for line in self.cf_lines_phase[amp_val]}

                for res_idx, data in amp_results.items(): # Iterate through resonances by index
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
        num_res = len(self.conceptual_resonance_frequencies)
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
                num_res = len(self.conceptual_resonance_frequencies)
    
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
                raise
                traceback.print_exc()
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
        if detector_idx <= len(self.conceptual_resonance_frequencies) and detector_idx > 0:
            conceptual_resonance_base_freq_hz = self.conceptual_resonance_frequencies[detector_idx - 1]
        else:
            print(f"Warning: Detector index {detector_idx} exceeds conceptual frequencies list length.")
            return
            
        # Gather data for ALL detectors to enable navigation (similar logic to digest panel)
        all_detectors_data = {}
        # We need conceptual frequencies for navigation
        for i, freq in enumerate(self.conceptual_resonance_frequencies):
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

    @QtCore.pyqtSlot(object)
    def _handle_multisweep_plot_double_click(self, ev):
        """
        Handles a double-click event on the multisweep magnitude plot.
        Identifies the clicked resonance and opens detector digest for it.
        """
        if not self.results_by_iteration:
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
        
        for iteration_data in self.results_by_iteration.values():
            res_data = iteration_data.get("data", {})
            for res_idx, data in res_data.items():
                center_freq = data.get('bias_frequency', data.get('original_center_frequency'))
                if center_freq is not None:
                    resonance_centers[res_idx] = center_freq
        
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

    def _open_detector_digest_for_index(self, detector_idx: int):
        """
        Open a detector digest panel for a specific detector index.
        
        This method extracts and reuses the panel creation logic from the double-click handler,
        making it callable programmatically (e.g., for auto-opening on completion).
        
        Args:
            detector_idx: Detector index (1-based) to open digest for
        """
        if not self.results_by_iteration:
            return
        
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
        
        # Get conceptual frequency for this detector
        if detector_idx <= len(self.conceptual_resonance_frequencies) and detector_idx > 0:
            conceptual_resonance_base_freq_hz = self.conceptual_resonance_frequencies[detector_idx - 1]
        else:
            print(f"Warning: Detector index {detector_idx} exceeds conceptual frequencies list length.")
            return
        
        # Gather data for this specific detector across all amplitudes and directions
        resonance_data_for_digest = {}
        
        # Group iterations by amplitude and direction
        amplitude_direction_to_iteration = {}
        for iter_idx, iter_data in self.results_by_iteration.items():
            amp_val = iter_data["amplitude"]
            direction = iter_data["direction"]
            key = (amp_val, direction)
            
            if key not in amplitude_direction_to_iteration or iter_idx > amplitude_direction_to_iteration[key]:
                amplitude_direction_to_iteration[key] = iter_idx
        
        # Process data for the target detector
        for (amp_val, direction), iter_idx in amplitude_direction_to_iteration.items():
            iter_data = self.results_by_iteration[iter_idx]
            res_data = iter_data["data"]
            
            if not res_data or detector_idx not in res_data:
                continue
            
            resonance_sweep_data = res_data[detector_idx]
            actual_cf_for_this_amp = resonance_sweep_data.get('bias_frequency', 
                                                              resonance_sweep_data.get('original_center_frequency'))
            
            combo_key = f"{amp_val}:{direction}"
            resonance_data_for_digest[combo_key] = {
                'data': resonance_sweep_data,
                'actual_cf_hz': actual_cf_for_this_amp,
                'direction': direction,
                'amplitude': amp_val
            }
        
        if not resonance_data_for_digest:
            print(f"Warning: No data for detector {detector_idx}")
            return
        
        # Gather data for ALL detectors to enable navigation
        all_detectors_data = {}
        all_detector_indices = set()
        for iter_data in self.results_by_iteration.values():
            res_data = iter_data.get("data", {})
            all_detector_indices.update(res_data.keys())
        
        for det_idx in sorted(all_detector_indices):
            if det_idx <= len(self.conceptual_resonance_frequencies) and det_idx > 0:
                conceptual_freq_hz = self.conceptual_resonance_frequencies[det_idx - 1]
            else:
                conceptual_freq_hz = None
                for iter_data in self.results_by_iteration.values():
                    res_data = iter_data.get("data", {})
                    if det_idx in res_data:
                        det_data = res_data[det_idx]
                        conceptual_freq_hz = det_data.get('bias_frequency', det_data.get('original_center_frequency'))
                        if conceptual_freq_hz:
                            break
            
            if conceptual_freq_hz is None:
                continue
            
            detector_resonance_data = {}
            amplitude_direction_to_iteration_det = {}
            for iter_idx, iter_data in self.results_by_iteration.items():
                amp_val = iter_data["amplitude"]
                direction = iter_data["direction"]
                key = (amp_val, direction)
                
                if key not in amplitude_direction_to_iteration_det or iter_idx > amplitude_direction_to_iteration_det[key]:
                    amplitude_direction_to_iteration_det[key] = iter_idx
            
            for (amp_val, direction), iter_idx in amplitude_direction_to_iteration_det.items():
                iter_data = self.results_by_iteration[iter_idx]
                res_data = iter_data["data"]
                
                if not res_data or det_idx not in res_data:
                    continue
                
                resonance_sweep_data = res_data[det_idx]
                actual_cf_for_this_amp = resonance_sweep_data.get('bias_frequency', 
                                                                  resonance_sweep_data.get('original_center_frequency'))
                
                combo_key = f"{amp_val}:{direction}"
                detector_resonance_data[combo_key] = {
                    'data': resonance_sweep_data,
                    'actual_cf_hz': actual_cf_for_this_amp,
                    'direction': direction,
                    'amplitude': amp_val
                }
            
            if detector_resonance_data:
                all_detectors_data[det_idx] = {
                    'resonance_data': detector_resonance_data,
                    'conceptual_freq_hz': conceptual_freq_hz
                }
        
        # Find Periscope parent to create docked panel
        periscope = self._get_periscope_parent()
        if not periscope:
            print("ERROR: Could not find Periscope parent for detector digest panel")
            return
        
        # Create panel
        panel = DetectorDigestPanel(
            parent=self,
            resonance_data_for_digest=resonance_data_for_digest,
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
        
        # Increment counter and use for tab name
        self.digest_window_count += 1
        loaded_suffix = " (Loaded)" if self.is_loaded_data else ""
        dock_title = f"Detector Digest #{self.digest_window_count}{loaded_suffix}"
        dock_id = f"digest_{self.digest_window_count}_{int(time.time())}"
        dock = periscope.dock_manager.create_dock(panel, dock_title, dock_id)
        
        # Track panel reference to prevent garbage collection
        self.detector_digest_windows.append(panel)
        
        # Tabify with this multisweep panel's dock
        my_dock = periscope.dock_manager.find_dock_for_widget(self)
        if my_dock:
            periscope.tabifyDockWidget(my_dock, dock)
        
        # Show and activate the dock
        dock.show()
        dock.raise_()

    def _get_closest_remembered_cf(self, conceptual_idx: int, target_amp: float) -> float | None:
        """
        Finds the remembered output CF for a given conceptual resonance index,
        for the amplitude in history closest to target_amp.

        Args:
            conceptual_idx: Index in self.conceptual_resonance_frequencies.
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
                    self.mag_legend.setLabelTextColor(pen_color)
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
                    self.phase_legend.setLabelTextColor(pen_color)
                except Exception as e:
                    print(f"Error updating phase legend text color: {e}")
        
        # Redraw plots which will now use the updated legend text colors
        self._redraw_plots()
            
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
        if not self.results_by_iteration:
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
        
        # Prepare data in GUI format expected by bias_kids
        gui_format_results = {
            'results_by_iteration': list(self.results_by_iteration.values())
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
        total_detectors = len(self.conceptual_resonance_frequencies)
        
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
