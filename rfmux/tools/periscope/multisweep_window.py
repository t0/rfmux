"""Window for displaying multisweep analysis results.""" # Simplified docstring to match others
import datetime
import pickle
import numpy as np
from PyQt6 import QtCore, QtWidgets # Keep direct Qt imports if used directly and not via utils.*
from PyQt6.QtCore import Qt
import pyqtgraph as pg

# Imports from within the 'periscope' subpackage
from .utils import LINE_WIDTH, UnitConverter, ClickableViewBox, QtWidgets, QtCore, pg # Ensure these are available

class MultisweepWindow(QtWidgets.QMainWindow):
    """
    A QMainWindow subclass for displaying and interacting with multisweep analysis results.

    This window visualizes S21 magnitude and phase data for multiple resonances
    across various probe amplitudes. It provides controls for data export,
    re-running sweeps, unit conversion, normalization, and plot interaction.
    """
    def __init__(self, parent=None, target_module=None, initial_params=None, dac_scales=None):
        """
        Initializes the MultisweepWindow.

        Args:
            parent: The parent widget.
            target_module (int, optional): The specific hardware module this window is for.
            initial_params (dict, optional): Initial parameters used for the multisweep.
                                             Defaults to an empty dict.
            dac_scales (dict, optional): DAC scaling factors for unit conversion.
                                         Defaults to an empty dict.
        """
        super().__init__(parent)
        self.target_module = target_module
        self.initial_params = initial_params or {}  # Store initial parameters for potential re-runs
        self.dac_scales = dac_scales or {}          # DAC scales for unit conversions
        self.probe_amplitudes = self.initial_params.get('amps', []) # Store for progress display

        self.setWindowTitle(f"Multisweep Results - Module {self.target_module}")

        # Data storage and state
        self.results_by_amplitude = {}  # Stores {amplitude: {cf: data_dict}}
        self.current_amplitude_being_processed = None # Tracks the amplitude currently being processed
        self.unit_mode = "dbm"  # Current unit for magnitude display ("counts", "dbm", "volts")
        self.normalize_magnitudes = False  # Flag to normalize magnitude plots
        self.zoom_box_mode = True  # Flag for enabling/disabling pyqtgraph's zoom box

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
        self.normalize_checkbox = QtWidgets.QCheckBox("Normalize Magnitudes")
        self.normalize_checkbox.setChecked(self.normalize_magnitudes)
        self.normalize_checkbox.toggled.connect(self._toggle_normalization)
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
        self.combined_mag_plot = pg.PlotWidget(viewBox=vb_mag, title="Combined S21 Magnitude (All Resonances)")
        self.combined_mag_plot.setLabel('bottom', 'Frequency', units='Hz')
        self._update_mag_plot_label() # Set initial Y-axis label based on unit mode
        self.combined_mag_plot.showGrid(x=True, y=True, alpha=0.3)
        self.mag_legend = self.combined_mag_plot.addLegend(offset=(30,10))
        plot_layout.addWidget(self.combined_mag_plot)

        # Phase Plot
        vb_phase = ClickableViewBox()
        vb_phase.parent_window = self
        self.combined_phase_plot = pg.PlotWidget(viewBox=vb_phase, title="Combined S21 Phase (All Resonances)")
        self.combined_phase_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.combined_phase_plot.setLabel('left', 'Phase', units='deg')
        self.combined_phase_plot.showGrid(x=True, y=True, alpha=0.3)
        self.phase_legend = self.combined_phase_plot.addLegend(offset=(30,10))
        plot_layout.addWidget(self.combined_phase_plot)
        
        layout.addWidget(plot_container)
        
        # Link X-axes of magnitude and phase plots for synchronized zooming/panning
        self.combined_phase_plot.setXLink(self.combined_mag_plot)
        self._apply_zoom_box_mode() # Apply initial zoom box mode state


    def _toggle_normalization(self, checked):
        """
        Slot for the 'Normalize Magnitudes' checkbox.
        Updates normalization state and redraws plots.
        """
        self.normalize_magnitudes = checked
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

        if self.normalize_magnitudes:
            label = "Normalized Magnitude"
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
        total_sweeps = len(self.probe_amplitudes)
        if total_sweeps > 0:
            first_amplitude = self.probe_amplitudes[0]
            self.current_amp_label.setText(f"Amplitude 1/{total_sweeps} ({first_amplitude:.4f})")
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

    def update_intermediate_data(self, module, amplitude, intermediate_results):
        """
        Placeholder for handling intermediate data during a sweep.
        Currently does nothing. Could be used for live plotting of partial results.

        Args:
            module (int): The module reporting data.
            amplitude (float): The current probe amplitude.
            intermediate_results: The intermediate data.
        """
        if module != self.target_module: return
        # This method is a placeholder. If intermediate data needs to be processed
        # (e.g., for live plotting during a long sweep for a single amplitude),
        # the logic would go here. For now, it's a no-op.
        pass

    def update_data(self, module, amplitude, final_results_for_amplitude):
        """
        Receives final data for a completed amplitude sweep for the target module.
        Stores the data and triggers a plot redraw.

        Args:
            module (int): The module reporting data.
            amplitude (float): The probe amplitude for which data is provided.
            final_results_for_amplitude (dict): The processed S21 data for this amplitude.
                                                Expected format: {center_freq: data_dict}
        """
        if module != self.target_module: return
        
        self.current_amplitude_being_processed = amplitude # This is the one whose data we just received
        self.results_by_amplitude[amplitude] = final_results_for_amplitude
        self._redraw_plots() # Refresh plots with the new data

        # Now, update the label for the *next* anticipated sweep
        total_sweeps = len(self.probe_amplitudes)
        if not self.probe_amplitudes:
            self.current_amp_label.setText("No sweeps defined.") # Should not happen if update_data is called
            return

        try:
            current_amp_index_in_list = self.probe_amplitudes.index(amplitude)
        except ValueError:
            # This amplitude wasn't in our list, which is unexpected.
            # Fallback to a generic message or log an error.
            print(f"Error: Processed amplitude {amplitude} not found in configured probe_amplitudes for Module {self.target_module}.")
            self.current_amp_label.setText(f"Processed: {amplitude:.4f} (Error finding next)")
            return

        if current_amp_index_in_list + 1 < total_sweeps:
            # There are more sweeps to go
            next_amplitude_index_in_list = current_amp_index_in_list + 1
            next_amplitude_value = self.probe_amplitudes[next_amplitude_index_in_list]
            # Display index is 1-based
            self.current_amp_label.setText(f"Amplitude {next_amplitude_index_in_list + 1}/{total_sweeps} ({next_amplitude_value:.4f})")
        else:
            # This was the last amplitude in the list
            # The all_sweeps_completed signal should handle the final message,
            # but we can set it here too as a fallback or intermediate state.
            # self.current_amp_label.setText(f"All {total_sweeps} Amplitudes Processed")
            # Let all_sweeps_completed handle the final message for clarity.
            # If this is the last one, the progress bar for this amp sweep should be 100%.
            # The overall "All Amplitudes Processed" will be set by all_sweeps_completed.
            pass

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

        num_amps = len(self.results_by_amplitude)
        if num_amps == 0: # No data to plot
            self.combined_mag_plot.autoRange(); self.combined_phase_plot.autoRange() # Ensure plots are reset
            return
        
        # Define color schemes for plotting multiple amplitudes
        # Use a distinct color family for few amplitudes, switch to a colormap for many
        channel_families = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"] # Tableau10
        viridis_cmap = pg.colormap.get("viridis")
        
        sorted_amplitudes = sorted(self.results_by_amplitude.keys())
        legend_items_mag = {} # To avoid duplicate legend entries for the same amplitude
        legend_items_phase = {}

        # Iterate through each amplitude's results
        for amp_idx, amp_val in enumerate(sorted_amplitudes):
            amp_results = self.results_by_amplitude[amp_val]
            
            # Determine color for this amplitude's curves
            if num_amps <= len(channel_families): # Use distinct colors if few amplitudes
                color = channel_families[amp_idx % len(channel_families)]
            else: # Use a colormap for many amplitudes
                color = viridis_cmap.map(amp_idx / max(1, num_amps - 1)) # Normalize index for colormap
            pen = pg.mkPen(color, width=LINE_WIDTH)
            
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

            # Add legend item only once per amplitude
            if amp_val not in legend_items_mag:
                # Create a dummy item for the legend (actual curves are added later)
                dummy_mag_curve_for_legend = pg.PlotDataItem(pen=pen) 
                self.mag_legend.addItem(dummy_mag_curve_for_legend, legend_name_amp)
                legend_items_mag[amp_val] = dummy_mag_curve_for_legend # Mark as added
            if amp_val not in legend_items_phase:
                dummy_phase_curve_for_legend = pg.PlotDataItem(pen=pen)
                self.phase_legend.addItem(dummy_phase_curve_for_legend, legend_name_amp)
                legend_items_phase[amp_val] = dummy_phase_curve_for_legend

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
                    normalize=self.normalize_magnitudes
                )
                # Use pre-calculated phase if available, otherwise calculate from IQ
                phase_deg = data.get('phase_degrees', np.degrees(np.angle(iq_complex))) 
                
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
            for task_key, task in parent.multisweep_tasks.items():
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
        Exports the collected multisweep results to a pickle file.
        Opens a file dialog for the user to choose the save location.
        """
        if not self.results_by_amplitude:
            QtWidgets.QMessageBox.warning(self, "No Data", "No data to export.")
            return

        dialog = QtWidgets.QFileDialog(self)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilters(["Pickle Files (*.pkl)", "All Files (*)"])
        dialog.setDefaultSuffix("pkl")
        
        if dialog.exec(): # True if user selected a file and clicked Save
            filename = dialog.selectedFiles()[0]
            try:
                # Prepare data for export
                export_content = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'target_module': self.target_module,
                    'initial_parameters': self.initial_params,
                    'dac_scales_used': self.dac_scales,
                    'results_by_amplitude': self.results_by_amplitude
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

        if not self.initial_params.get('resonance_frequencies'):
            QtWidgets.QMessageBox.warning(self, "Cannot Re-run", 
                                          "Initial resonance frequencies not available for re-run.")
            return
        
        # Prepare parameters for the dialog, preserving current user choices if possible
        current_recalc_cf_state = self.initial_params.get('recalculate_center_frequencies', True)
        current_perform_fits_state = self.initial_params.get('perform_fits', True)
        
        dialog_params = self.initial_params.copy() # Start with a copy of original params
        dialog_params['recalculate_center_frequencies'] = current_recalc_cf_state
        dialog_params['perform_fits'] = current_perform_fits_state
        
        # Create and show the dialog
        dialog = MultisweepDialog(
            parent=self, 
            resonance_frequencies=self.initial_params['resonance_frequencies'],
            dac_scales=self.dac_scales, 
            current_module=self.target_module, 
            initial_params=dialog_params # Pass potentially modified params to dialog
        )
        
        if dialog.exec(): # True if user clicked OK in the dialog
            new_params = dialog.get_parameters()
            if new_params:
                self.initial_params.update(new_params) # Store the new parameters
                
                # Reset window state for the new sweep
                self.results_by_amplitude.clear()
                self._redraw_plots() # Clear plots
                self.progress_bar.setValue(0)
                self.progress_group.setVisible(True)
                
                # Update probe_amplitudes list and reset label
                self.probe_amplitudes = self.initial_params.get('amps', [])
                total_sweeps = len(self.probe_amplitudes)
                if total_sweeps > 0:
                    first_amplitude = self.probe_amplitudes[0]
                    self.current_amp_label.setText(f"Amplitude 1/{total_sweeps} ({first_amplitude:.4f})")
                else:
                    self.current_amp_label.setText("No sweeps defined. (Waiting...)")
                
                # Trigger the new multisweep analysis via the parent window/controller
                # This assumes the parent object (likely the main application window)
                # has a method to start/restart the analysis for this specific window.
                if hasattr(self.parent(), '_start_multisweep_analysis_for_window'):
                    self.parent()._start_multisweep_analysis_for_window(self, new_params)
                else: 
                    # This case should ideally not be reached if the application is structured correctly.
                    QtWidgets.QMessageBox.warning(self, "Error", 
                                                  "Cannot trigger re-run. Parent linkage or method missing.")

    def closeEvent(self, event):
        """
        Overrides QWidget.closeEvent.
        Notifies the parent/controller to stop any ongoing tasks associated with this window
        before closing.
        """
        # Check if the parent object has a method to stop tasks for this window
        if hasattr(self.parent(), 'stop_multisweep_task_for_window'):
            self.parent().stop_multisweep_task_for_window(self)
        super().closeEvent(event) # Proceed with the standard close event handling

    def _toggle_cf_lines_visibility(self, checked):
        """
        Slot for the 'Show Center Frequencies' checkbox.
        Redraws plots to show or hide CF lines.

        Args:
            checked (bool): The new state of the checkbox.
        """
        # The actual logic for showing/hiding lines is within _redraw_plots,
        # which checks self.show_cf_lines_cb.isChecked().
        # We just need to trigger a redraw.
        self._redraw_plots()
