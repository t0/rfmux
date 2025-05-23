"""Window for displaying multisweep analysis results."""
import datetime
import pickle
import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
import pyqtgraph as pg

# Imports from within the 'periscope' subpackage
from .utils import (
    LINE_WIDTH, UnitConverter, ClickableViewBox, QtWidgets, QtCore, pg,
    TABLEAU10_COLORS, COLORMAP_CHOICES, AMPLITUDE_COLORMAP_THRESHOLD, UPWARD_SWEEP_STYLE, DOWNWARD_SWEEP_STYLE
)
from .detector_digest_dialog import DetectorDigestWindow

class MultisweepWindow(QtWidgets.QMainWindow):
    """
    A QMainWindow subclass for displaying and interacting with multisweep analysis results.

    This window visualizes S21 magnitude and phase data for multiple resonances
    across various probe amplitudes. It provides controls for data export,
    re-running sweeps, unit conversion, normalization, and plot interaction.
    """
    def __init__(self, parent=None, target_module=None, initial_params=None, dac_scales=None, dark_mode=False):
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
        """
        super().__init__(parent)
        self.target_module = target_module
        self.initial_params = initial_params or {}  # Store initial parameters for potential re-runs
        self.dac_scales = dac_scales or {}          # DAC scales for unit conversions
        self.dark_mode = dark_mode                 # Store dark mode setting
        
        # Track open detector digest windows to prevent garbage collection
        self.detector_digest_windows = []
        
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

        self._setup_ui()
        self.resize(1200, 800) # Default window size

    def _setup_ui(self):
        """Sets up the main UI layout, toolbar, and plot area."""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        self._setup_toolbar(main_layout) # main_layout is not used by _setup_toolbar
        self._setup_progress_bar(main_layout)
        self._setup_plot_area(main_layout)


    def _setup_toolbar(self, layout): # layout parameter is not used
        """Creates and configures the main toolbar with controls."""
        toolbar = QtWidgets.QToolBar("Multisweep Controls")
        toolbar.setMovable(False) # Prevent toolbar from being moved by the user
        self.addToolBar(toolbar)

        # Export Data Button
        self.export_btn = QtWidgets.QPushButton("Export Data")
        self.export_btn.clicked.connect(self._export_data)
        toolbar.addWidget(self.export_btn)

        # Re-run Multisweep Button
        self.rerun_btn = QtWidgets.QPushButton("Re-run Multisweep")
        self.rerun_btn.clicked.connect(self._rerun_multisweep)
        toolbar.addWidget(self.rerun_btn)
        
        toolbar.addSeparator()

        # Spacer to push subsequent items to the right
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        # Normalization Checkbox
        self.normalize_checkbox = QtWidgets.QCheckBox("Normalize Traces")
        self.normalize_checkbox.setChecked(self.normalize_traces)
        self.normalize_checkbox.toggled.connect(self._toggle_trace_normalization)
        toolbar.addWidget(self.normalize_checkbox)

        # Show Center Frequencies Checkbox
        self.show_cf_lines_cb = QtWidgets.QCheckBox("Show Center Frequencies")
        self.show_cf_lines_cb.setChecked(False) # Default to off
        self.show_cf_lines_cb.toggled.connect(self._toggle_cf_lines_visibility)
        toolbar.addWidget(self.show_cf_lines_cb)

        self._setup_unit_controls(toolbar)
        self._setup_zoom_box_control(toolbar)

    def _setup_unit_controls(self, toolbar):
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
        toolbar.addSeparator()
        toolbar.addWidget(unit_group)

    def _setup_zoom_box_control(self, toolbar):
        """Sets up the checkbox to toggle zoom box mode for plots."""
        self.zoom_box_cb = QtWidgets.QCheckBox("Zoom Box Mode")
        self.zoom_box_cb.setChecked(self.zoom_box_mode)
        self.zoom_box_cb.toggled.connect(self._toggle_zoom_box_mode)
        toolbar.addWidget(self.zoom_box_cb)

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
            self.mag_legend = plot_item_mag.addLegend(offset=(30,10),labelTextColor=pen_color)
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
            self.phase_legend = plot_item_phase.addLegend(offset=(30,10), labelTextColor=pen_color)
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
        
    def handle_intermediate_data_update(self, module: int, iteration: int, amplitude: float, direction: str, intermediate_data: dict):
        """
        Handler for intermediate data updates during multisweep analysis.
        TEMPORARILY DISABLED for performance reasons - intermediate plot updates were causing severe slowdown.
        
        Args:
            module (int): The module reporting the intermediate data.
            iteration (int): The current iteration index.
            amplitude (float): The probe amplitude for this iteration.
            direction (str): The sweep direction ("upward" or "downward").
            intermediate_data (dict): Intermediate data in the same format as final data.
        """
        if module != self.target_module: 
            return
            
        # TEMPORARILY DISABLED - intermediate updates were causing 1000x slowdown
        # Store intermediate data but don't update plots
        self._current_intermediate_data = intermediate_data
        # TODO: Implement throttled intermediate updates with proper performance optimization

    def _update_plots_with_intermediate_data(self, amplitude: float, direction: str, intermediate_data: dict):
        """
        Updates plots with intermediate data during an ongoing sweep.
        Creates semi-transparent curves to show current progress.
        
        Args:
            amplitude (float): The current amplitude being swept.
            direction (str): The sweep direction.
            intermediate_data (dict): Intermediate sweep data.
        """
        if not self.combined_mag_plot or not self.combined_phase_plot:
            return
            
        # Clear any existing intermediate curves
        self._clear_intermediate_curves()
        
        # Choose styling for intermediate curves
        # Make them semi-transparent and use a consistent intermediate color
        intermediate_alpha = 0.6
        intermediate_color = (255, 165, 0, int(255 * intermediate_alpha))  # Orange with alpha
        
        # Create intermediate curves for each center frequency
        for cf, data in intermediate_data.items():
            freqs_hz = data.get('frequencies')
            iq_complex = data.get('iq_complex')
            
            if freqs_hz is None or iq_complex is None or len(freqs_hz) == 0:
                continue
                
            # Calculate magnitude and phase for intermediate data
            s21_mag_raw = np.abs(iq_complex)
            s21_mag_processed = UnitConverter.convert_amplitude(
                s21_mag_raw, iq_complex, self.unit_mode, 
                normalize=self.normalize_traces
            )
            phase_deg = data.get('phase_degrees', np.degrees(np.angle(iq_complex)))
            
            if self.normalize_traces and len(phase_deg) > 0:
                first_phase_val = phase_deg[0]
                if np.isfinite(first_phase_val):
                    phase_deg = phase_deg - first_phase_val
            
            # Create intermediate magnitude curve
            intermediate_pen_mag = pg.mkPen(intermediate_color, width=LINE_WIDTH, style=QtCore.Qt.PenStyle.DashLine)
            mag_curve = self.combined_mag_plot.plot(pen=intermediate_pen_mag)
            mag_curve.setData(freqs_hz, s21_mag_processed)
            self._intermediate_curves_mag[cf] = mag_curve
            
            # Create intermediate phase curve
            intermediate_pen_phase = pg.mkPen(intermediate_color, width=LINE_WIDTH, style=QtCore.Qt.PenStyle.DashLine)
            phase_curve = self.combined_phase_plot.plot(pen=intermediate_pen_phase)
            phase_curve.setData(freqs_hz, phase_deg)
            self._intermediate_curves_phase[cf] = phase_curve

    def _clear_intermediate_curves(self):
        """Clear all intermediate curves from the plots."""
        # Remove magnitude intermediate curves
        for curve in self._intermediate_curves_mag.values():
            if curve and self.combined_mag_plot:
                self.combined_mag_plot.removeItem(curve)
        self._intermediate_curves_mag.clear()
        
        # Remove phase intermediate curves
        for curve in self._intermediate_curves_phase.values():
            if curve and self.combined_phase_plot:
                self.combined_phase_plot.removeItem(curve)
        self._intermediate_curves_phase.clear()

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

        # Clear intermediate curves when final data arrives
        self._clear_intermediate_curves()
        
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

            # --- Plot Data for Each Resonance (Center Frequency) at this Amplitude ---
            for cf, data in amp_results.items():
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
                self.curves_mag[amp_val][cf] = mag_curve # Store reference

                # Plot phase curve
                phase_curve = self.combined_phase_plot.plot(pen=pen)
                phase_curve.setData(freqs_hz, phase_deg)
                if amp_val not in self.curves_phase: self.curves_phase[amp_val] = {}
                self.curves_phase[amp_val][cf] = phase_curve # Store reference

                # Add center frequency (CF) lines if enabled
                if self.show_cf_lines_cb and self.show_cf_lines_cb.isChecked():
                    cf_line_pen = pg.mkPen(color, style=QtCore.Qt.PenStyle.DashLine, width=LINE_WIDTH/2)
                    
                    mag_cf_line = pg.InfiniteLine(pos=cf, angle=90, pen=cf_line_pen, movable=False)
                    self.combined_mag_plot.addItem(mag_cf_line)
                    self.cf_lines_mag.setdefault(amp_val, []).append(mag_cf_line) # Store reference

                    phase_cf_line = pg.InfiniteLine(pos=cf, angle=90, pen=cf_line_pen, movable=False)
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
        Updates UI elements to reflect completion.
        """
        self._check_all_complete()
        self.current_amp_label.setText("All Amplitudes Processed")
        
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
    
    def _handle_export_file_selected(self, filename):
        """Handle the file selection from the non-blocking dialog."""
        if not filename:
            return
            
        try:
            # Prepare data for export
            export_content = {
                'timestamp': datetime.datetime.now().isoformat(),
                'target_module': self.target_module,
                'initial_parameters': self.initial_params,
                'dac_scales_used': self.dac_scales,
                'results_by_iteration': self.results_by_iteration
            }
            with open(filename, 'wb') as f: # Write in binary mode for pickle
                pickle.dump(export_content, f)
            QtWidgets.QMessageBox.information(self, "Export Complete", f"Data exported to {filename}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Error exporting data: {str(e)}")

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
            initial_params=dialog_initial_params # Pass other existing params
        )

        if dialog.exec(): # True if user clicked OK
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
            
            parent_widget = self.parent()
            parent_widget = self.parent()
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

                for cf in amp_results.keys(): # Iterate through center frequencies for this amplitude
                    # Magnitude plot CF line
                    if cf in existing_mag_lines_for_amp:
                        existing_mag_lines_for_amp[cf].setVisible(True)
                    else:
                        mag_cf_line = pg.InfiniteLine(pos=cf, angle=90, pen=cf_line_pen, movable=False)
                        self.combined_mag_plot.addItem(mag_cf_line)
                        self.cf_lines_mag[amp_val].append(mag_cf_line)
                        # mag_cf_line.setVisible(True) # Already visible by default when added

                    # Phase plot CF line
                    if cf in existing_phase_lines_for_amp:
                        existing_phase_lines_for_amp[cf].setVisible(True)
                    else:
                        phase_cf_line = pg.InfiniteLine(pos=cf, angle=90, pen=cf_line_pen, movable=False)
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

    @QtCore.pyqtSlot(object)
    def _handle_multisweep_plot_double_click(self, ev):
        """
        Handles a double-click event on the multisweep magnitude plot.
        Identifies the clicked resonance and prepares data for the DetectorDigestDialog.
        """
        if not self.results_by_iteration:
            return

        # Get click coordinates in view space from the event's scenePos
        # The event 'ev' is a QGraphicsSceneMouseEvent from pyqtgraph
        if not self.combined_mag_plot: return
        view_box = self.combined_mag_plot.getViewBox()
        if not view_box: return

        mouse_point = view_box.mapSceneToView(ev.scenePos())
        x_coord = mouse_point.x()
        # y_coord = mouse_point.y() # y_coord is not used for selecting the CF

        # --- Find the conceptual resonance closest to the click's X-coordinate ---
        min_horizontal_dist = float('inf')
        # This will be the CF of an actual measured trace that is horizontally closest to the click's X.
        # It serves as the initial seed to find the conceptual resonance.
        seed_cf_hz_for_conceptual_match = None

        # Iterate through all known center frequencies from all measurements
        # to find which one is horizontally closest to the click.
        all_measured_cfs = set()
        if self.results_by_iteration: # Ensure there's data
            for iteration_data in self.results_by_iteration.values():
                cf_map = iteration_data.get("data", {})
                all_measured_cfs.update(cf_map.keys()) # These keys are the output CFs

        if not all_measured_cfs: # No CFs to check against
            return # Or handle error appropriately

        for cf_candidate in all_measured_cfs:
            horizontal_dist = abs(cf_candidate - x_coord)
            if horizontal_dist < min_horizontal_dist:
                min_horizontal_dist = horizontal_dist
                seed_cf_hz_for_conceptual_match = cf_candidate
        
        if seed_cf_hz_for_conceptual_match is not None:
            # Now, seed_cf_hz_for_conceptual_match is the CF of an actual trace's *output key*
            # that is horizontally closest to the click.
            clicked_output_cf_hz = seed_cf_hz_for_conceptual_match

            # --- Determine the conceptual resonance and its ID ---
            # Use self.conceptual_resonance_frequencies for stable identification
            if not self.conceptual_resonance_frequencies:
                print("Warning: Conceptual resonance frequencies not available. Cannot reliably determine conceptual resonance for digest.")
                # Fallback: use clicked_output_cf_hz as the conceptual base, Detector ID might be less meaningful
                conceptual_resonance_base_freq_hz = clicked_output_cf_hz
                detector_id = -1 # Cannot reliably determine ID
            else:
                # Find which conceptual_resonance_frequency is "responsible" for the clicked_output_cf_hz.
                # This requires checking which conceptual resonance, when processed, resulted in clicked_output_cf_hz.
                # This is complex if multiple conceptual resonances map to similar output CFs.
                # A simpler approach: find the conceptual_resonance_frequency closest to the clicked_output_cf_hz.
                closest_conceptual_idx = np.abs(np.array(self.conceptual_resonance_frequencies) - clicked_output_cf_hz).argmin()
                conceptual_resonance_base_freq_hz = self.conceptual_resonance_frequencies[closest_conceptual_idx]
                detector_id = closest_conceptual_idx # This is the "Detector ID"

            # --- Gather all data for this conceptual resonance across all amplitudes and directions ---
            # Create a structure that preserves both amplitude and direction information
            resonance_data_for_digest = {}
            
            # Group iterations by amplitude and direction
            amplitude_direction_to_iteration = {}  # Maps (amplitude, direction) to iteration
            for iter_idx, iter_data in self.results_by_iteration.items():
                amp_val = iter_data["amplitude"]
                direction = iter_data["direction"]
                key = (amp_val, direction)
                
                # Keep the latest iteration for each amplitude+direction combination
                if key not in amplitude_direction_to_iteration or iter_idx > amplitude_direction_to_iteration[key]:
                    amplitude_direction_to_iteration[key] = iter_idx
            
            # Now process the appropriate iteration for each amplitude+direction combination
            for (amp_val, direction), iter_idx in amplitude_direction_to_iteration.items():
                iter_data = self.results_by_iteration[iter_idx]
                cf_map = iter_data["data"]
                
                if not cf_map:  # No CFs measured for this amplitude+direction
                    continue
                
                # Find the CF in cf_map closest to conceptual_resonance_base_freq_hz
                available_cfs = np.array(list(cf_map.keys()))
                closest_cf_idx = np.abs(available_cfs - conceptual_resonance_base_freq_hz).argmin()
                actual_cf_for_this_amp = available_cfs[closest_cf_idx]
                
                # Create a unique key that combines amplitude and direction
                # Format: "amp_val:direction" (e.g., "0.5:upward")
                combo_key = f"{amp_val}:{direction}"
                
                # Store with combined key to preserve both directions per amplitude
                resonance_data_for_digest[combo_key] = {
                    'data': cf_map[actual_cf_for_this_amp],
                    'actual_cf_hz': actual_cf_for_this_amp,  # Store the actual CF for this sweep
                    'direction': direction,  # Include direction information
                    'amplitude': amp_val  # Store the raw amplitude for reference
                }

            if not resonance_data_for_digest:
                print(f"Warning: No data gathered for conceptual resonance near {conceptual_resonance_base_freq_hz / 1e9:.6f} GHz for digest.")
                return

            # At this point, we have:
            # - clicked_cf_hz
            # - detector_id
            # - resonance_data_for_digest (dict of {amp_raw: sweep_data_dict})
            # - self.dac_scales
            # - self.zoom_box_mode

            # Accept the event to prevent further processing (e.g., ClickableViewBox's own message box)
            ev.accept()

            # Create a non-modal window for the detector digest
            digest_window = DetectorDigestWindow(
                parent=self, # type: ignore # parent is QWidget, DetectorDigestWindow expects QWidget
                resonance_data_for_digest=resonance_data_for_digest,
                detector_id=detector_id,
                resonance_frequency_ghz=conceptual_resonance_base_freq_hz / 1e9,
                dac_scales=self.dac_scales,
                zoom_box_mode=self.zoom_box_mode,
                target_module=self.target_module, # type: ignore # target_module is int | None, DetectorDigestWindow expects int
                normalize_plot3=self.normalize_traces,
                dark_mode=self.dark_mode
            )
            
            # Add to tracking list to prevent garbage collection
            self.detector_digest_windows.append(digest_window)
            
            # Show the window
            digest_window.show()
        else:
            # No curve found near click, event not accepted, default ClickableViewBox behavior might occur
            pass

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
        min_abs_amp_diff = float('inf')
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
