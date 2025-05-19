"""Window class for network analysis results."""
import datetime # Added for MultisweepWindow export
import pickle   # Added for MultisweepWindow export
import os 
import csv

# Imports from within the 'periscope' subpackage
from .utils import * # This will bring in QtWidgets, QtCore, pg, np, ClickableViewBox, UnitConverter, fitting, etc.
# from .tasks import * # Not directly used by this class, dialogs will import what they need.

# Dialogs are now imported from .dialogs within the same package
from .dialogs import NetworkAnalysisParamsDialog, FindResonancesDialog, MultisweepDialog

class NetworkAnalysisWindow(QtWidgets.QMainWindow):
    """
    Window for displaying network analysis results with real units support.
    """
    def __init__(self, parent=None, modules=None, dac_scales=None):
        super().__init__(parent)
        self.setWindowTitle("Network Analysis Results")
        self.modules = modules or []
        self.data = {}  # module -> amplitude data dictionary
        self.raw_data = {}  # Store the raw IQ data for unit conversion
        self.unit_mode = "dbm"  # Default to dBm instead of counts
        self.normalize_magnitudes = False  # Add this flag to track normalization state
        self.first_setup = True  # Flag to track initial setup
        self.zoom_box_mode = True  # Default to zoom box mode ON
        self.plots = {}  # Initialize plots dictionary early
        self.original_params = {}  # Initial parameters
        self.current_params = {}   # Most recently used parameters
        self.dac_scales = dac_scales or {}  # Store DAC scales
        self.resonance_lines_mag = {} # Store resonance lines for magnitude plots {module: [line_item, ...]}
        self.resonance_lines_phase = {} # Store resonance lines for phase plots {module: [line_item, ...]}
        self.resonance_freqs = {}  # Store resonance frequencies per module
        self.add_subtract_mode = False
        self.module_cable_lengths = {} # For Requirement 2
        self.faux_resonance_legend_items_mag = {} # For Req 3
        self.faux_resonance_legend_items_phase = {} # For Req 3
        
        # Setup the UI components
        self._setup_ui()
        # Set initial size only on creation
        self.resize(1000, 800)

    def _setup_ui(self):
        """Set up the user interface for the window."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        
        # Create toolbar
        self._setup_toolbar(layout)
        
        # Create progress bars
        self._setup_progress_bars(layout)
        
        # Create plot area
        self._setup_plot_area(layout)

    def _setup_toolbar(self, layout):
        """Set up the toolbars with controls."""
        # Toolbar 1: Global Controls (Top Row)
        toolbar_global = QtWidgets.QToolBar("Global Controls")
        toolbar_global.setMovable(False)
        self.addToolBar(toolbar_global)

        # Export button
        export_btn = QtWidgets.QPushButton("Export Data")
        export_btn.clicked.connect(self._export_data)
        toolbar_global.addWidget(export_btn)

        # Edit Other Parameters button (renamed to Re-run Analysis)
        edit_params_btn = QtWidgets.QPushButton("Re-run analysis") # Renamed
        edit_params_btn.clicked.connect(self._edit_parameters) # Kept self._edit_parameters as per user
        toolbar_global.addWidget(edit_params_btn)

        
        toolbar_global.addSeparator()

        # Show/Hide resonances checkbox
        self.show_resonances_cb = QtWidgets.QCheckBox("Show Resonances") # Count removed as per req 3
        self.show_resonances_cb.setChecked(True)
        self.show_resonances_cb.toggled.connect(self._toggle_resonances_visible)
        toolbar_global.addWidget(self.show_resonances_cb)

        # Add/Subtract mode
        self.edit_resonances_cb = QtWidgets.QCheckBox("Add/Subtract Resonances")
        self.edit_resonances_cb.setToolTip(
            "When enabled, double-click adds a resonance;\n"
            "Shift + double-click removes the nearest resonance."
        )
        self.edit_resonances_cb.toggled.connect(self._toggle_resonance_edit_mode)
        toolbar_global.addWidget(self.edit_resonances_cb)
        
        toolbar_global.addSeparator()

        # Add spacer to push the unit controls to the far right of the global toolbar
        spacer_global = QtWidgets.QWidget()
        spacer_global.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        toolbar_global.addWidget(spacer_global)

        # Normalize Magnitudes checkbox
        self.normalize_checkbox = QtWidgets.QCheckBox("Normalize Magnitudes")
        self.normalize_checkbox.setChecked(False)
        self.normalize_checkbox.setToolTip("Normalize all magnitude curves to their first data point")
        self.normalize_checkbox.toggled.connect(self._toggle_normalization)
        toolbar_global.addWidget(self.normalize_checkbox)

        # Add unit controls
        self._setup_unit_controls(toolbar_global) # Pass the global toolbar

        # Add zoom box mode checkbox
        self._setup_zoom_box_control(toolbar_global) # Pass the global toolbar

        # Toolbar 2: Module-Specific Controls (Bottom Row)
        toolbar_module = QtWidgets.QToolBar("Module Controls")
        toolbar_module.setMovable(False)
        self.addToolBarBreak(QtCore.Qt.ToolBarArea.TopToolBarArea) # Ensures this toolbar is on a new row
        self.addToolBar(toolbar_module)

        # Cable length control
        self.cable_length_label = QtWidgets.QLabel("Cable Length (m):")
        self.cable_length_spin = QtWidgets.QDoubleSpinBox()
        self.cable_length_spin.setRange(0.0, 1000.0)
        self.cable_length_spin.setValue(DEFAULT_CABLE_LENGTH)
        self.cable_length_spin.setSingleStep(0.05)
        self.cable_length_spin.valueChanged.connect(self._on_cable_length_changed)
        toolbar_module.addWidget(self.cable_length_label)
        toolbar_module.addWidget(self.cable_length_spin)

        # Unwrap Cable Delay button
        unwrap_button = QtWidgets.QPushButton("Unwrap Cable Delay")
        unwrap_button.setToolTip("Fit phase slope, calculate cable length, and apply compensation for the active module.")
        unwrap_button.clicked.connect(self._unwrap_cable_delay_action)
        toolbar_module.addWidget(unwrap_button)

        # Find Resonances button
        find_res_btn = QtWidgets.QPushButton("Find Resonances")
        find_res_btn.setToolTip("Identify resonance frequencies from the current sweep data for the active module.")
        find_res_btn.clicked.connect(self._show_find_resonances_dialog)
        toolbar_module.addWidget(find_res_btn)

        # Take Multisweep button
        self.take_multisweep_btn = QtWidgets.QPushButton("Take Multisweep")
        self.take_multisweep_btn.setToolTip("Perform a multisweep using identified resonance frequencies for the active module.")
        self.take_multisweep_btn.clicked.connect(self._show_multisweep_dialog)
        self.take_multisweep_btn.setEnabled(False) # Initially disabled
        toolbar_module.addWidget(self.take_multisweep_btn)

    def _setup_unit_controls(self, toolbar):
        """Set up the unit selection controls and add them to the specified toolbar."""
        unit_group = QtWidgets.QWidget()
        unit_layout = QtWidgets.QHBoxLayout(unit_group)
        unit_layout.setContentsMargins(0, 0, 0, 0)
        unit_layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.rb_counts = QtWidgets.QRadioButton("Counts")
        self.rb_dbm = QtWidgets.QRadioButton("dBm")
        self.rb_volts = QtWidgets.QRadioButton("Volts")
        self.rb_dbm.setChecked(True)  # Changed from rb_counts to rb_dbm
        
        unit_layout.addWidget(QtWidgets.QLabel("Units:"))
        unit_layout.addWidget(self.rb_counts)
        unit_layout.addWidget(self.rb_dbm)
        unit_layout.addWidget(self.rb_volts)
        
        # Connect signals
        self.rb_counts.toggled.connect(lambda: self._update_unit_mode("counts"))
        self.rb_dbm.toggled.connect(lambda: self._update_unit_mode("dbm"))
        self.rb_volts.toggled.connect(lambda: self._update_unit_mode("volts"))
        
        # Set fixed size policy to make alignment more predictable
        unit_group.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, 
                                QtWidgets.QSizePolicy.Policy.Preferred)
        
        # Now add the unit controls at the far right
        toolbar.addSeparator()
        toolbar.addWidget(unit_group)
        
    def _setup_zoom_box_control(self, toolbar):
        """Set up the zoom box mode control."""
        zoom_box_cb = QtWidgets.QCheckBox("Zoom Box Mode")
        zoom_box_cb.setChecked(self.zoom_box_mode)  # Default to ON
        zoom_box_cb.setToolTip("When enabled, left-click drag creates a zoom box. When disabled, left-click drag pans.")
        zoom_box_cb.toggled.connect(self._toggle_zoom_box)
        
        # Store reference to the checkbox
        self.zoom_box_cb = zoom_box_cb
        
        toolbar.addWidget(zoom_box_cb)

    def _setup_progress_bars(self, layout):
        """Set up progress bars for each module."""
        self.progress_group = None
        if self.modules:
            self.progress_group = QtWidgets.QGroupBox("Analysis Progress")
            progress_layout = QtWidgets.QVBoxLayout(self.progress_group)
            
            self.progress_bars = {}
            self.progress_labels = {}  # Add labels to show amplitude progress
            for module in self.modules:
                vlayout = QtWidgets.QVBoxLayout()
                
                # Main progress layout
                hlayout = QtWidgets.QHBoxLayout()
                label = QtWidgets.QLabel(f"Module {module}:")
                pbar = QtWidgets.QProgressBar()
                pbar.setRange(0, 100)
                pbar.setValue(0)
                hlayout.addWidget(label)
                hlayout.addWidget(pbar)
                vlayout.addLayout(hlayout)
                
                # Amplitude progress label
                amp_label = QtWidgets.QLabel("")
                amp_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                vlayout.addWidget(amp_label)
                
                progress_layout.addLayout(vlayout)
                self.progress_bars[module] = pbar
                self.progress_labels[module] = amp_label
                
            layout.addWidget(self.progress_group)
        else:
            self.progress_bars = {}
            self.progress_labels = {}

    def _setup_plot_area(self, layout):
        """Set up the plot area with tabs for each module."""
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.currentChanged.connect(self._on_active_module_changed) # For Requirement 2
        layout.addWidget(self.tabs)
        
        self.plots = {}
        for module in self.modules:
            tab = QtWidgets.QWidget()
            tab_layout = QtWidgets.QVBoxLayout(tab)

            # Create amplitude and phase plots with ClickableViewBox
            vb_amp = ClickableViewBox()
            vb_amp.parent_window = self
            vb_amp.module_id = module
            vb_amp.plot_role = 'amp'
            amp_plot = pg.PlotWidget(viewBox=vb_amp, title=f"Module {module} - Magnitude")
            self._update_amplitude_labels(amp_plot)
            amp_plot.setLabel('bottom', 'Frequency', units='Hz')
            amp_plot.showGrid(x=True, y=True, alpha=0.3)

            vb_phase = ClickableViewBox()
            vb_phase.parent_window = self
            vb_phase.module_id = module
            vb_phase.plot_role = 'phase'
            phase_plot = pg.PlotWidget(viewBox=vb_phase, title=f"Module {module} - Phase")
            phase_plot.setLabel('left', 'Phase', units='deg')
            phase_plot.setLabel('bottom', 'Frequency', units='Hz')
            phase_plot.showGrid(x=True, y=True, alpha=0.3)
            
            # Add legends for multiple amplitude plots
            amp_legend = amp_plot.addLegend(offset=(30, 10))
            phase_legend = phase_plot.addLegend(offset=(30, 10))

            # Create curves with periscope color scheme - but don't add data yet
            amp_curve = amp_plot.plot([], [], pen=pg.mkPen('#ff7f0e', width=LINE_WIDTH))  # Empty data
            phase_curve = phase_plot.plot([], [], pen=pg.mkPen('#1f77b4', width=LINE_WIDTH))  # Empty data

            tab_layout.addWidget(amp_plot)
            tab_layout.addWidget(phase_plot)
            self.tabs.addTab(tab, f"Module {module}")
            
            self.plots[module] = {
                'amp_plot': amp_plot,
                'phase_plot': phase_plot,
                'amp_curve': amp_curve,
                'phase_curve': phase_curve,
                'amp_legend': amp_legend,
                'phase_legend': phase_legend,
                'amp_curves': {},  # Will store multiple curves for different amplitudes
                'phase_curves': {},  # Will store multiple curves for different amplitudes
                'resonance_lines_mag': [], # For storing magnitude resonance lines
                'resonance_lines_phase': [] # For storing phase resonance lines
            }
            
            # Apply zoom box mode
            self._apply_zoom_box_mode()            

            # Link the x-axis of amplitude and phase plots for synchronized zooming
            phase_plot.setXLink(amp_plot)

    def clear_plots(self):
        """Clear all plots, curves, and legends."""
        for module_id_iter in self.plots: 
            plot_info = self.plots[module_id_iter]
            amp_plot = plot_info['amp_plot']
            phase_plot = plot_info['phase_plot']
            
            plot_info['amp_legend'].clear()
            plot_info['phase_legend'].clear()
            
            for amp, curve in list(plot_info['amp_curves'].items()):
                amp_plot.removeItem(curve)
            for amp, curve in list(plot_info['phase_curves'].items()):
                phase_plot.removeItem(curve)
            
            plot_info['amp_curves'].clear()
            plot_info['phase_curves'].clear()
            
            plot_info['amp_curve'].setData([], [])
            plot_info['phase_curve'].setData([], [])

            self._remove_faux_resonance_legend_entry(module_id_iter)
            self._update_multisweep_button_state(module_id_iter) 


    def update_amplitude_progress(self, module: int, current_amp: int, total_amps: int, amplitude: float):
        """Update the amplitude progress display for a module."""
        if hasattr(self, 'progress_labels') and module in self.progress_labels:
            self.progress_labels[module].setText(f"Amplitude {current_amp}/{total_amps} ({amplitude})")

    def _toggle_normalization(self, checked):
        """Toggle normalization of magnitude plots."""
        self.normalize_magnitudes = checked
        
        for module in self.plots:
            self._update_amplitude_labels(self.plots[module]['amp_plot'])
        
        for module in self.plots:
            if len(self.plots[module]['amp_curves']) > 0:
                self.plots[module]['amp_curve'].setData([], [])
                self.plots[module]['phase_curve'].setData([], [])
        
        for module_id in self.plots: 
            self._update_amplitude_labels(self.plots[module_id]['amp_plot'])

            if module_id in self.raw_data:
                plot_info = self.plots[module_id]
                
                for amp_key, raw_entry_tuple in self.raw_data[module_id].items():
                    amplitude_val, freqs, amps_raw, _, iq_data_raw = self._extract_data_from_tuple(amp_key, raw_entry_tuple)
                    
                    if freqs is None or amps_raw is None or iq_data_raw is None:
                        continue

                    converted_amps = UnitConverter.convert_amplitude(
                        amps_raw, iq_data_raw, self.unit_mode, normalize=self.normalize_magnitudes
                    )
                    
                    if amp_key != 'default':
                        if amplitude_val in plot_info['amp_curves']:
                            plot_info['amp_curves'][amplitude_val].setData(freqs, converted_amps)
                    else: 
                        if not plot_info['amp_curves']:
                            plot_info['amp_curve'].setData(freqs, converted_amps)
                
                plot_info['amp_plot'].autoRange() 
        

    def _toggle_zoom_box(self, enable):
        """Toggle zoom box mode for all plots."""
        self.zoom_box_mode = enable
        self._apply_zoom_box_mode()
        
    def _apply_zoom_box_mode(self):
        """Apply the current zoom box mode setting to all plots."""
        for module in self.plots:
            for plot_type in ['amp_plot', 'phase_plot']:
                viewbox = self.plots[module][plot_type].getViewBox()
                if isinstance(viewbox, ClickableViewBox):
                    viewbox.enableZoomBoxMode(self.zoom_box_mode)

    def _toggle_resonances_visible(self, checked: bool):
        """Show or hide all resonance indicator lines."""
        for module_id in self.plots.keys():
            for line in self.plots[module_id]['resonance_lines_mag']:
                line.setVisible(checked)
            for line in self.plots[module_id]['resonance_lines_phase']:
                line.setVisible(checked)
            self._update_resonance_legend_entry(module_id)

    def _toggle_resonance_edit_mode(self, checked: bool):
        """Enable or disable double-click add/subtract mode."""
        self.add_subtract_mode = checked

    def _update_resonance_checkbox_text(self, module: int):
        """Update the show-resonances checkbox label (count removed)."""
        self.show_resonances_cb.setText("Show Resonances")

    def _add_resonance(self, module: int, freq_hz: float):
        """Add a resonance line at the given frequency."""
        if module not in self.plots:
            return
        plot_info = self.plots[module]
        line_pen = pg.mkPen('r', style=QtCore.Qt.PenStyle.DashLine)
        line_mag = pg.InfiniteLine(pos=freq_hz, angle=90, movable=False, pen=line_pen)
        line_phase = pg.InfiniteLine(pos=freq_hz, angle=90, movable=False, pen=line_pen)
        plot_info['amp_plot'].addItem(line_mag)
        plot_info['phase_plot'].addItem(line_phase)
        plot_info['resonance_lines_mag'].append(line_mag)
        plot_info['resonance_lines_phase'].append(line_phase)
        self.resonance_freqs.setdefault(module, []).append(freq_hz)
        self._update_resonance_checkbox_text(module) 
        self._update_resonance_legend_entry(module) 
        self._toggle_resonances_visible(self.show_resonances_cb.isChecked())
        self._update_multisweep_button_state(module)


    def _remove_resonance(self, module: int, freq_hz: float):
        """Remove the nearest resonance line."""
        if module not in self.plots or module not in self.resonance_freqs:
            return
        freqs = self.resonance_freqs[module]
        if not freqs:
            self._update_multisweep_button_state(module) 
            return
        freqs_arr = np.array(freqs)
        idx = int(np.argmin(np.abs(freqs_arr - freq_hz)))
        line_mag = self.plots[module]['resonance_lines_mag'].pop(idx)
        line_phase = self.plots[module]['resonance_lines_phase'].pop(idx)
        self.plots[module]['amp_plot'].removeItem(line_mag)
        self.plots[module]['phase_plot'].removeItem(line_phase)
        freqs.pop(idx)
        self._update_resonance_checkbox_text(module) 
        self._update_resonance_legend_entry(module) 
        self._update_multisweep_button_state(module)

    def _update_unit_mode(self, mode):
        """Update unit mode and redraw only amplitude plots."""
        if mode != self.unit_mode:
            self.unit_mode = mode
            
            for module_id in self.plots: 
                self._update_amplitude_labels(self.plots[module_id]['amp_plot'])

                if module_id in self.raw_data:
                    plot_info = self.plots[module_id]
                    
                    for amp_key, raw_entry_tuple in self.raw_data[module_id].items():
                        amplitude_val, freqs, amps_raw, _, iq_data_raw = self._extract_data_from_tuple(amp_key, raw_entry_tuple)
                        
                        if freqs is None or amps_raw is None or iq_data_raw is None: 
                            continue

                        converted_amps = UnitConverter.convert_amplitude(
                            amps_raw, iq_data_raw, self.unit_mode, normalize=self.normalize_magnitudes
                        )
                        
                        if amp_key != 'default': 
                            if amplitude_val in plot_info['amp_curves']:
                                plot_info['amp_curves'][amplitude_val].setData(freqs, converted_amps)
                        else: 
                            if not plot_info['amp_curves']: 
                                plot_info['amp_curve'].setData(freqs, converted_amps)
                    
                    plot_info['amp_plot'].autoRange() 
            
            self._update_legends_for_unit_mode()

    def _update_legends_for_unit_mode(self):
        """Update the legend entries to reflect the current unit mode."""
        for module in self.plots:
            self.plots[module]['amp_legend'].clear()
            self.plots[module]['phase_legend'].clear()
            
            for amplitude, curve in self.plots[module]['amp_curves'].items():
                label = "" 
                if self.unit_mode == "dbm":
                    if module not in self.dac_scales:
                        print(f"Warning: No DAC scale available for module {module}, cannot display accurate probe power in dBm.")
                        self.rb_counts.setChecked(True)  
                        return  
                    
                    dac_scale = self.dac_scales[module]
                    dbm_value = UnitConverter.normalize_to_dbm(amplitude, dac_scale)
                    label = f"Probe: {dbm_value:.2f} dBm"
                elif self.unit_mode == "volts":
                    if module not in self.dac_scales:
                        print(f"Warning: No DAC scale available for module {module}, cannot display accurate probe amplitude in Volts.")
                        self.rb_counts.setChecked(True)  
                        return  
                    else:
                        dac_scale = self.dac_scales[module]
                        dbm_value = UnitConverter.normalize_to_dbm(amplitude, dac_scale)
                        
                        power_watts = 10**((dbm_value - 30)/10)
                        resistance = 50.0  
                        voltage_rms = np.sqrt(power_watts * resistance)
                        voltage_peak = voltage_rms * np.sqrt(2)
                        voltage_peak = voltage_peak*1e6
                        
                        label = f"Probe: {voltage_peak:.1f} uV (peak)"
                else:  # "counts"
                    label = f"Probe: {amplitude} Normalized Units"
                
                self.plots[module]['amp_legend'].addItem(curve, label)
                
                phase_curve = self.plots[module]['phase_curves'].get(amplitude)
                if phase_curve:
                    self.plots[module]['phase_legend'].addItem(phase_curve, label)         
    
    def _update_amplitude_labels(self, plot):
        """Update plot labels based on current unit mode and normalization state."""
        if self.normalize_magnitudes:
            if self.unit_mode == "dbm":
                plot.setLabel('left', 'Normalized Power', units='dB') 
            else:
                plot.setLabel('left', 'Normalized Magnitude', units='')
        else:
            if self.unit_mode == "counts":
                plot.setLabel('left', 'Magnitude', units='Counts')
            elif self.unit_mode == "dbm":
                plot.setLabel('left', 'Power', units='dBm')
            elif self.unit_mode == "volts":
                plot.setLabel('left', 'Magnitude', units='V')

    def _redraw_all_plots(self):
        """Redraw all plots with current unit mode."""
        for module_id_iter_redraw in self.raw_data: 
            if module_id_iter_redraw in self.plots:
                has_amp_curves = len(self.plots[module_id_iter_redraw]['amp_curves']) > 0
                
                for amp_key, data_tuple in self.raw_data[module_id_iter_redraw].items():
                    if amp_key != 'default':
                        amplitude, freqs, amps, phases, iq_data = self._extract_data_from_tuple(amp_key, data_tuple)
                        
                        if amplitude in self.plots[module_id_iter_redraw]['amp_curves']:
                            converted_amps = UnitConverter.convert_amplitude(
                                amps, iq_data, self.unit_mode, normalize=self.normalize_magnitudes)
                            self.plots[module_id_iter_redraw]['amp_curves'][amplitude].setData(freqs, converted_amps)
                            self.plots[module_id_iter_redraw]['phase_curves'][amplitude].setData(freqs, phases)
                
                if 'default' in self.raw_data[module_id_iter_redraw] and not has_amp_curves:
                    freqs, amps, phases, iq_data = self.raw_data[module_id_iter_redraw]['default']
                    converted_amps = UnitConverter.convert_amplitude(
                        amps, iq_data, self.unit_mode, normalize=self.normalize_magnitudes)
                    self.plots[module_id_iter_redraw]['amp_curve'].setData(freqs, converted_amps)
                    self.plots[module_id_iter_redraw]['phase_curve'].setData(freqs, phases)
                else:
                    self.plots[module_id_iter_redraw]['amp_curve'].setData([], [])
                    self.plots[module_id_iter_redraw]['phase_curve'].setData([], [])
                
                self.plots[module_id_iter_redraw]['amp_plot'].autoRange()
        
        self._update_legends_for_unit_mode()              
    
    def _extract_data_from_tuple(self, amp_key, data_tuple):
        """Extract amplitude and data from a data tuple."""
        if len(data_tuple) == 5:
            freqs, amps, phases, iq_data, amplitude = data_tuple
        else:
            freqs, amps, phases, iq_data = data_tuple
            try:
                amplitude = float(amp_key.split('_')[-1])
            except (ValueError, IndexError):
                amplitude = DEFAULT_AMPLITUDE
        return amplitude, freqs, amps, phases, iq_data

    def closeEvent(self, event):
        """Handle window close event by cleaning up resources."""
        parent = self.parent()
        
        if parent and hasattr(parent, 'netanal_windows'):
            window_id = None
            for w_id, w_data in parent.netanal_windows.items():
                if w_data['window'] == self:
                    window_id = w_id
                    break
            
            if window_id:
                if hasattr(parent, 'netanal_tasks'):
                    for task_key in list(parent.netanal_tasks.keys()):
                        if task_key.startswith(f"{window_id}_"):
                            task = parent.netanal_tasks.pop(task_key)
                            task.stop()
                
                parent.netanal_windows.pop(window_id, None)
        
        super().closeEvent(event)

    def _show_find_resonances_dialog(self):
        """Show the dialog to configure and run find_resonances."""
        current_tab_index = self.tabs.currentIndex()
        if current_tab_index < 0:
            QtWidgets.QMessageBox.warning(self, "No Module Selected", "Please select a module tab to analyze.")
            return
        
        active_module_text = self.tabs.tabText(current_tab_index)
        try:
            active_module = int(active_module_text.split(" ")[1])
        except (IndexError, ValueError):
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not determine active module from tab: {active_module_text}")
            return

        if not self.raw_data or active_module not in self.raw_data or not self.raw_data[active_module]:
            QtWidgets.QMessageBox.information(self, "No Data", f"No sweep data available for Module {active_module} to find resonances.")
            return

        dialog = FindResonancesDialog(self)
        if dialog.exec():
            params = dialog.get_parameters()
            if params:
                self._run_and_plot_resonances(active_module, params)

    def _run_and_plot_resonances(self, active_module: int, find_resonances_params: dict):
        """Run find_resonances and plot the results on the active module's plots."""
        module_sweeps = self.raw_data.get(active_module)
        if not module_sweeps:
            QtWidgets.QMessageBox.warning(self, "No Data", f"No data found for module {active_module}.")
            self._update_multisweep_button_state(active_module) 
            return

        target_sweep_key = None
        ordered_amplitudes_run = self.original_params.get('amps', [])
        if ordered_amplitudes_run:
            for amp_setting in reversed(ordered_amplitudes_run):
                sweep_key = f"{active_module}_{amp_setting}"
                if sweep_key in module_sweeps:
                    target_sweep_key = sweep_key
                    break
        
        if target_sweep_key is None and 'default' in module_sweeps:
            target_sweep_key = 'default'

        if target_sweep_key is None:
            if module_sweeps:
                target_sweep_key = list(module_sweeps.keys())[-1]


        if target_sweep_key is None:
            QtWidgets.QMessageBox.warning(self, "No Data", f"Could not determine which sweep to analyze for module {active_module}.")
            self._update_multisweep_button_state(active_module) 
            return

        data_tuple = module_sweeps[target_sweep_key]
        
        if len(data_tuple) == 5: 
            frequencies, _, _, iq_complex, _ = data_tuple
        elif len(data_tuple) == 4: 
            frequencies, _, _, iq_complex = data_tuple
        else:
            QtWidgets.QMessageBox.critical(self, "Data Error", "Unexpected data format for the selected sweep.")
            self._update_multisweep_button_state(active_module) 
            return

        if len(frequencies) == 0 or len(iq_complex) == 0:
            QtWidgets.QMessageBox.information(self, "No Data", "Selected sweep has no frequency or IQ data.")
            self._update_multisweep_button_state(active_module) 
            return
            
        try:
            resonance_results = fitting.find_resonances(
                frequencies=frequencies,
                iq_complex=iq_complex,
                module_identifier=f"Module {active_module} (Sweep: {target_sweep_key})",
                **find_resonances_params
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Resonance Finding Error", f"Error calling find_resonances: {str(e)}")
            traceback.print_exc()
            self._update_multisweep_button_state(active_module) 
            return

        if active_module in self.plots:
            plot_info = self.plots[active_module]
            amp_plot_item = plot_info['amp_plot'].getPlotItem()
            phase_plot_item = plot_info['phase_plot'].getPlotItem()

            for line in plot_info.get('resonance_lines_mag', []):
                amp_plot_item.removeItem(line)
            plot_info['resonance_lines_mag'] = []

            for line in plot_info.get('resonance_lines_phase', []):
                phase_plot_item.removeItem(line)
            plot_info['resonance_lines_phase'] = []

            self.resonance_freqs[active_module] = []
            
            res_freqs_hz = resonance_results.get('resonance_frequencies', [])

            if not res_freqs_hz:
                QtWidgets.QMessageBox.information(self, "No Resonances Found", 
                                                  f"No resonances were identified for Module {active_module} with the given parameters.")
                self._update_multisweep_button_state(active_module) 
                return 

            line_pen = pg.mkPen('r', style=QtCore.Qt.PenStyle.DashLine)
            for res_freq_hz in res_freqs_hz:
                line_mag = pg.InfiniteLine(pos=res_freq_hz, angle=90, movable=False, pen=line_pen)
                amp_plot_item.addItem(line_mag)
                plot_info['resonance_lines_mag'].append(line_mag)

                line_phase = pg.InfiniteLine(pos=res_freq_hz, angle=90, movable=False, pen=line_pen)
                phase_plot_item.addItem(line_phase)
                plot_info['resonance_lines_phase'].append(line_phase)

            self.resonance_freqs[active_module] = res_freqs_hz
            self._update_resonance_checkbox_text(active_module) 
            self._update_resonance_legend_entry(active_module) 
            self._toggle_resonances_visible(self.show_resonances_cb.isChecked()) 
        self._update_multisweep_button_state(active_module)


    def _remove_faux_resonance_legend_entry(self, module_id: int):
        """Removes the faux resonance legend entry for a module."""
        if module_id in self.plots:
            plot_info = self.plots[module_id]
            amp_legend = plot_info['amp_legend']
            phase_legend = plot_info['phase_legend']

            if module_id in self.faux_resonance_legend_items_mag:
                try:
                    amp_legend.removeItem(self.faux_resonance_legend_items_mag[module_id])
                except Exception: 
                    pass 
                del self.faux_resonance_legend_items_mag[module_id]
            
            if module_id in self.faux_resonance_legend_items_phase:
                try:
                    phase_legend.removeItem(self.faux_resonance_legend_items_phase[module_id])
                except Exception:
                    pass
                del self.faux_resonance_legend_items_phase[module_id]

    def _update_resonance_legend_entry(self, module_id: int):
        """Adds or updates the faux legend entry for resonance count."""
        if module_id not in self.plots:
            return

        self._remove_faux_resonance_legend_entry(module_id) 

        plot_info = self.plots[module_id]
        amp_legend = plot_info['amp_legend']
        phase_legend = plot_info['phase_legend']
        
        count = len(self.resonance_freqs.get(module_id, []))

        if self.show_resonances_cb.isChecked() and count > 0:
            dummy_pen = pg.mkPen('r', style=QtCore.Qt.PenStyle.DashLine)
            
            legend_sample_item_mag = pg.PlotDataItem(pen=dummy_pen)
            amp_legend.addItem(legend_sample_item_mag, f"{count} resonances")
            self.faux_resonance_legend_items_mag[module_id] = legend_sample_item_mag 

            legend_sample_item_phase = pg.PlotDataItem(pen=dummy_pen)
            phase_legend.addItem(legend_sample_item_phase, f"{count} resonances")
            self.faux_resonance_legend_items_phase[module_id] = legend_sample_item_phase


    def _check_all_complete(self):
        """
        Check if all progress bars are at 100% and hide the progress group 
        when all analyses are complete.
        """
        if not self.progress_group:
            return
            
        parent = self.parent()
        window_id = None
        
        if parent and hasattr(parent, 'netanal_windows'):
            for w_id, w_data in parent.netanal_windows.items():
                if w_data['window'] == self:
                    window_id = w_id
                    break
        
        if not window_id:
            return
            
        window_data = parent.netanal_windows[window_id]
        
        no_pending_amplitudes = True
        for module in window_data['amplitude_queues']:
            if window_data['amplitude_queues'][module]:
                no_pending_amplitudes = False
                break
        
        all_complete = all(pbar.value() == 100 for pbar in self.progress_bars.values())
        if all_complete and no_pending_amplitudes:
            self.progress_group.setVisible(False)

    def _edit_parameters(self):
        """Open dialog to edit parameters. Re-runs analysis using per-module cable lengths."""
        params_for_dialog = self.current_params.copy()
        params_for_dialog.pop('module_cable_lengths', None)
        params_for_dialog.pop('cable_length', None) 

        dialog = NetworkAnalysisParamsDialog(self, params_for_dialog)
        if dialog.exec():
            updated_general_params = dialog.get_parameters() 
            
            if updated_general_params:
                params_for_rerun = updated_general_params.copy()
                params_for_rerun['module_cable_lengths'] = self.module_cable_lengths.copy()
                params_for_rerun.pop('cable_length', None)
                self.current_params = params_for_rerun.copy()
                
                if self.progress_group:
                    self.progress_group.setVisible(True)
                
                self.parent()._rerun_network_analysis(self.current_params)

    def _rerun_analysis(self):
        """Re-run the analysis with potentially updated parameters."""
        if hasattr(self.parent(), '_rerun_network_analysis'):
            params = self.current_params.copy()
            params['module_cable_lengths'] = self.module_cable_lengths.copy()
            params.pop('cable_length', None)

            if self.progress_group:
                self.progress_group.setVisible(True)
            self.parent()._rerun_network_analysis(params)
    
    def set_params(self, params):
        """Set parameters for analysis."""
        self.original_params = params.copy()  
        self.current_params = params.copy()   

        default_cable_length_for_all = params.get('cable_length', DEFAULT_CABLE_LENGTH)
        for mod_id in self.modules: 
            self.module_cable_lengths[mod_id] = params.get('module_cable_lengths', {}).get(mod_id, default_cable_length_for_all)

        if self.tabs.count() > 0:
            self._on_active_module_changed(self.tabs.currentIndex())
        elif self.modules: 
            first_module_id = self.modules[0]
            initial_cable_length = self.module_cable_lengths.get(first_module_id, DEFAULT_CABLE_LENGTH)
            self.cable_length_spin.blockSignals(True)
            self.cable_length_spin.setValue(initial_cable_length)
            self.cable_length_spin.blockSignals(False)
        
        if not hasattr(self, 'plots') or not self.plots:
            return
            
        fmin = params.get('fmin', DEFAULT_MIN_FREQ)
        fmax = params.get('fmax', DEFAULT_MAX_FREQ)
        for module in self.plots:
            self.plots[module]['amp_plot'].setXRange(fmin, fmax)
            self.plots[module]['phase_plot'].setXRange(fmin, fmax)
            self.plots[module]['amp_plot'].enableAutoRange(pg.ViewBox.XAxis, False)
            self.plots[module]['phase_plot'].enableAutoRange(pg.ViewBox.XAxis, False)
            self.plots[module]['amp_plot'].enableAutoRange(pg.ViewBox.YAxis, True)
            self.plots[module]['phase_plot'].enableAutoRange(pg.ViewBox.YAxis, True)
    
    def update_data_with_amp(self, module: int, freqs: np.ndarray, amps: np.ndarray, phases: np.ndarray, amplitude: float):
        """Update the plot data for a specific module and amplitude."""
        iq_data = amps * np.exp(1j * np.radians(phases))  
        key = f"{module}_{amplitude}"
        
        if module not in self.raw_data: self.raw_data[module] = {}
        if module not in self.data: self.data[module] = {}
        
        self.raw_data[module][key] = (freqs, amps, phases, iq_data, amplitude)
        self.data[module][key] = (freqs, amps, phases)
        
        if module in self.plots:
            if len(self.plots[module]['amp_curves']) == 0:
                self.plots[module]['amp_curve'].setData([], [])
                self.plots[module]['phase_curve'].setData([], [])
            
            converted_amps = UnitConverter.convert_amplitude(
                amps, iq_data, self.unit_mode, normalize=self.normalize_magnitudes)
            
            amps_list = self.original_params.get('amps', [amplitude])
            if amplitude in amps_list: amp_index = amps_list.index(amplitude)
            else: amp_index = 0
                
            channel_families = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
            color = channel_families[amp_index % len(channel_families)]
            
            is_new_curve = False
            if amplitude not in self.plots[module]['amp_curves']:
                is_new_curve = True
                self.plots[module]['amp_curves'][amplitude] = self.plots[module]['amp_plot'].plot(
                    pen=pg.mkPen(color, width=LINE_WIDTH), name=f"Amp: {amplitude}")
                self.plots[module]['phase_curves'][amplitude] = self.plots[module]['phase_plot'].plot(
                    pen=pg.mkPen(color, width=LINE_WIDTH), name=f"Amp: {amplitude}")
            
            self.plots[module]['amp_curves'][amplitude].setData(freqs, converted_amps)
            self.plots[module]['phase_curves'][amplitude].setData(freqs, phases)
            
            if is_new_curve: self._update_legends_for_unit_mode()
        self._update_multisweep_button_state(module) 

    def update_data(self, module: int, freqs: np.ndarray, amps: np.ndarray, phases: np.ndarray):
        """Update the plot data for a specific module."""
        iq_data = amps * np.exp(1j * np.radians(phases))  
        
        if module not in self.raw_data: self.raw_data[module] = {}
        if module not in self.data: self.data[module] = {}
        
        self.raw_data[module]['default'] = (freqs, amps, phases, iq_data)
        self.data[module]['default'] = (freqs, amps, phases)
        
        if module in self.plots:
            if len(self.plots[module]['amp_curves']) == 0:
                converted_amps = UnitConverter.convert_amplitude(
                    amps, iq_data, self.unit_mode, normalize=self.normalize_magnitudes)
                
                freq_ghz = freqs # Keep as freqs, units are handled by axis label
                
                self.plots[module]['amp_curve'].setData(freq_ghz, converted_amps)
                self.plots[module]['phase_curve'].setData(freq_ghz, phases)
        self._update_multisweep_button_state(module) 
    
    def update_progress(self, module: int, progress: float):
        """Update the progress bar for a specific module."""
        if module in self.progress_bars:
            self.progress_bars[module].setValue(int(progress))
    
    def complete_analysis(self, module: int):
        """Mark analysis as complete for a module."""
        if module in self.progress_bars:
            self.progress_bars[module].setValue(100)
            self._check_all_complete()
    
    def _export_data(self):
        """Export the collected data with all unit conversions and metadata."""
        if not self.data:
            QtWidgets.QMessageBox.warning(self, "No Data", "No data to export yet.")
            return
        
        dialog = QtWidgets.QFileDialog(self)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilters(["Pickle Files (*.pkl)", "CSV Files (*.csv)", "All Files (*)"])
        dialog.setDefaultSuffix("pkl")
        
        if dialog.exec():
            filename = dialog.selectedFiles()[0]
            
            try:
                if filename.endswith('.pkl'): self._export_to_pickle(filename)
                elif filename.endswith('.csv'): self._export_to_csv(filename)
                else: self._export_to_pickle(filename)
                QtWidgets.QMessageBox.information(self, "Export Complete", f"Data exported to {filename}")
            except Exception as e:
                traceback.print_exc() 
                QtWidgets.QMessageBox.critical(self, "Export Error", f"Error exporting data: {str(e)}")
    
    def _export_to_pickle(self, filename):
        """Export data to a pickle file."""
        export_data = {'timestamp': datetime.datetime.now().isoformat(),
                       'parameters': self.current_params.copy() if hasattr(self, 'current_params') else {},
                       'modules': {}}
        
        for module, data_dict in self.raw_data.items():
            export_data['modules'][module] = {}; meas_idx = 0
            for key, data_tuple in data_dict.items():
                amplitude, freqs, amps, phases, iq_data = self._extract_data_for_export(key, data_tuple)
                counts = amps; volts = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="volts")
                dbm = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="dbm")
                counts_norm = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="counts", normalize=True)
                volts_norm = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="volts", normalize=True)
                dbm_norm = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="dbm", normalize=True)
                
                export_data['modules'][module][meas_idx] = {
                    'sweep_amplitude': amplitude,
                    'frequency': {'values': freqs.tolist(), 'unit': 'Hz'},
                    'magnitude': {'counts': {'raw': counts.tolist(), 'normalized': counts_norm.tolist(), 'unit': 'counts'},
                                  'volts': {'raw': volts.tolist(), 'normalized': volts_norm.tolist(), 'unit': 'V'},
                                  'dbm': {'raw': dbm.tolist(), 'normalized': dbm_norm.tolist(), 'unit': 'dBm'}},
                    'phase': {'values': phases.tolist(), 'unit': 'degrees'},
                    'complex': {'real': iq_data.real.tolist(), 'imag': iq_data.imag.tolist()}}
                meas_idx += 1
            export_data['modules'][module]['resonances_hz'] = self.resonance_freqs.get(module, [])
        
        with open(filename, 'wb') as f: pickle.dump(export_data, f)
    
    def _export_to_csv(self, filename):
        """Export data to CSV files."""
        base, ext = os.path.splitext(filename)
        meta_filename = f"{base}_metadata{ext}"
        with open(meta_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Parameter', 'Value']); writer.writerow(['Export Date', datetime.datetime.now().isoformat()])
            if hasattr(self, 'current_params'):
                writer.writerow(['', '']); writer.writerow(['Measurement Parameters', ''])
                for param, value in self.current_params.items():
                    if param in ['fmin', 'fmax', 'max_span'] and isinstance(value, (int, float)):
                        writer.writerow([param, f"{value/1e6} MHz"])
                    else: writer.writerow([param, value])
            if self.resonance_freqs:
                writer.writerow(['', '']); writer.writerow(['Resonances (Hz)', ''])
                for module, freqs in self.resonance_freqs.items():
                    writer.writerow([f'Module {module}', ','.join(map(str, freqs))])
        
        for module, data_dict in self.raw_data.items():
            idx = 0 
            for key, data_tuple in data_dict.items():
                amplitude, freqs, amps, phases, iq_data = self._extract_data_for_export(key, data_tuple)
                for unit_mode in ["counts", "volts", "dbm"]:
                    converted_amps = UnitConverter.convert_amplitude(amps, iq_data, unit_mode=unit_mode)
                    unit_label = "dBm" if unit_mode == "dbm" else ("V" if unit_mode == "volts" else unit_mode)
                    csv_filename = f"{base}_module{module}_idx{idx}_{unit_mode}{ext}"
                    with open(csv_filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['# Amplitude:', f"{amplitude}" if 'amplitude' in locals() else "Unknown"])
                        header = ['Frequency (Hz)', f'Power ({unit_label})' if unit_mode == "dbm" else f'Amplitude ({unit_label})', 'Phase (deg)']
                        writer.writerow(header)
                        for freq, amp, phase in zip(freqs, converted_amps, phases):
                            writer.writerow([freq, amp, phase])
                idx += 1
    
    def _extract_data_for_export(self, key, data_tuple):
        """Extract and prepare data for export from a data tuple."""
        amplitude = DEFAULT_AMPLITUDE 
        if key != 'default':
            if len(data_tuple) >= 5: freqs, amps, phases, iq_data, amplitude = data_tuple
            else:
                freqs, amps, phases, iq_data = data_tuple
                try: amplitude = float(key.split('_')[-1])
                except (ValueError, IndexError): pass
        else: freqs, amps, phases, iq_data = data_tuple
        return amplitude, freqs, amps, phases, iq_data

    def _unwrap_cable_delay_action(self):
        """
        Fits the phase data of the first curve in the active module's plot,
        calculates the corresponding cable length, updates the phase curves,
        and adjusts the cable length spinner.
        """
        if not self.raw_data: QtWidgets.QMessageBox.information(self, "No Data", "No data to process."); return
        current_tab_index = self.tabs.currentIndex()
        if current_tab_index < 0: QtWidgets.QMessageBox.warning(self, "No Module", "Select a module tab."); return
        active_module_text = self.tabs.tabText(current_tab_index)
        try: active_module = int(active_module_text.split(" ")[1])
        except (IndexError, ValueError): QtWidgets.QMessageBox.critical(self, "Error", f"Invalid module tab: {active_module_text}"); return
        if active_module not in self.raw_data or not self.raw_data[active_module]:
            QtWidgets.QMessageBox.information(self, "No Data", f"No data for Module {active_module}."); return
        module_data_dict = self.raw_data[active_module]; target_key = None
        if 'amps' in self.original_params and self.original_params['amps']:
            first_amplitude_setting = self.original_params['amps'][0]
            potential_key = f"{active_module}_{first_amplitude_setting}"
            if potential_key in module_data_dict: target_key = potential_key
        if target_key is None and 'default' in module_data_dict: target_key = 'default'
        if target_key is None:
            if module_data_dict: target_key = next(iter(module_data_dict))
            else: QtWidgets.QMessageBox.information(self, "No Data", f"No sweep data for Module {active_module}."); return
        data_tuple = module_data_dict[target_key]
        if len(data_tuple) == 5: freqs_active, _, phases_displayed_active_deg, _, _ = data_tuple
        elif len(data_tuple) == 4: freqs_active, _, phases_displayed_active_deg, _ = data_tuple
        else: QtWidgets.QMessageBox.critical(self, "Error", "Unexpected data format."); return
        if len(freqs_active) == 0: QtWidgets.QMessageBox.information(self, "No Data", "Selected sweep has no frequency data."); return
        L_old_physical = self.current_params.get('cable_length', DEFAULT_CABLE_LENGTH)
        try:
            tau_additional = fit_cable_delay(freqs_active, phases_displayed_active_deg)
            L_new_physical = calculate_new_cable_length(L_old_physical, tau_additional)
        except Exception as e: QtWidgets.QMessageBox.critical(self, "Calc Error", f"Cable delay calc error: {str(e)}"); traceback.print_exc(); return
        if active_module in self.plots:
            plot_info = self.plots[active_module]
            for amp_key_iter, curve_item in plot_info['phase_curves'].items():
                raw_data_key_for_curve = f"{active_module}_{amp_key_iter}"
                if raw_data_key_for_curve in module_data_dict:
                    data_tuple_curve = module_data_dict[raw_data_key_for_curve]
                    if len(data_tuple_curve) == 5: freqs_curve, _, phases_deg_current_display_curve, _, _ = data_tuple_curve
                    elif len(data_tuple_curve) == 4: freqs_curve, _, phases_deg_current_display_curve, _ = data_tuple_curve
                    else: continue
                    if len(freqs_curve) > 0:
                        new_phases_deg_for_curve = recalculate_displayed_phase(freqs_curve, phases_deg_current_display_curve, L_old_physical, L_new_physical)
                        if len(new_phases_deg_for_curve) > 0:
                            first_point_phase = new_phases_deg_for_curve[0]
                            new_phases_deg_for_curve = new_phases_deg_for_curve - first_point_phase
                        new_phases_deg_for_curve = ((new_phases_deg_for_curve + 180) % 360) - 180
                        curve_item.setData(freqs_curve, new_phases_deg_for_curve)
            if not plot_info['phase_curves'] and 'default' in module_data_dict:
                main_curve_item = plot_info['phase_curve']
                data_tuple_main = module_data_dict['default']
                if len(data_tuple_main) == 4:
                    freqs_main, _, phases_deg_current_display_main, _ = data_tuple_main
                    if len(freqs_main) > 0:
                        new_phases_deg_for_main = recalculate_displayed_phase(freqs_main, phases_deg_current_display_main, L_old_physical, L_new_physical)
                        if len(new_phases_deg_for_main) > 0:
                            first_point_phase = new_phases_deg_for_main[0]
                            new_phases_deg_for_main = new_phases_deg_for_main - first_point_phase
                        new_phases_deg_for_main = ((new_phases_deg_for_main + 180) % 360) - 180
                        main_curve_item.setData(freqs_main, new_phases_deg_for_main)
            plot_info['phase_plot'].enableAutoRange(pg.ViewBox.YAxis, True)
        self.module_cable_lengths[active_module] = L_new_physical
        self.cable_length_spin.blockSignals(True); self.cable_length_spin.setValue(L_new_physical); self.cable_length_spin.blockSignals(False)
        QtWidgets.QMessageBox.information(self, "Cable Delay Updated", f"Cable length for Module {active_module} updated to {L_new_physical:.3f} m.")

    def _on_active_module_changed(self, index: int):
        """Update UI elements when the active module tab changes."""
        if index < 0 or not self.modules or index >= len(self.modules): self._update_multisweep_button_state(None); return
        active_module_id = self.modules[index]
        if active_module_id in self.module_cable_lengths:
            self.cable_length_spin.blockSignals(True); self.cable_length_spin.setValue(self.module_cable_lengths[active_module_id]); self.cable_length_spin.blockSignals(False)
        else:
            self.cable_length_spin.blockSignals(True); self.cable_length_spin.setValue(self.current_params.get('cable_length', DEFAULT_CABLE_LENGTH)); self.cable_length_spin.blockSignals(False)
        self._update_multisweep_button_state(active_module_id)

    def _on_cable_length_changed(self, new_length: float):
        """Handle changes to the cable length spinner."""
        current_tab_index = self.tabs.currentIndex()
        if current_tab_index < 0 or not self.modules or current_tab_index >= len(self.modules): return
        active_module_id = self.modules[current_tab_index]
        self.module_cable_lengths[active_module_id] = new_length
        self._update_multisweep_button_state(active_module_id)

    def _update_multisweep_button_state(self, module_id: int | None = None):
        """Enable or disable the Take Multisweep button based on found resonances for the given module."""
        if not hasattr(self, 'take_multisweep_btn'): return
        if module_id is None:
            current_tab_index = self.tabs.currentIndex()
            if current_tab_index < 0: self.take_multisweep_btn.setEnabled(False); return
            active_module_text = self.tabs.tabText(current_tab_index)
            try: module_id = int(active_module_text.split(" ")[1])
            except (IndexError, ValueError): self.take_multisweep_btn.setEnabled(False); return
        has_resonances = bool(self.resonance_freqs.get(module_id))
        self.take_multisweep_btn.setEnabled(has_resonances)

    def _show_multisweep_dialog(self):
        """Show the dialog to configure and run multisweep."""
        current_tab_index = self.tabs.currentIndex()
        if current_tab_index < 0: QtWidgets.QMessageBox.warning(self, "No Module", "Select a module tab."); return
        active_module_text = self.tabs.tabText(current_tab_index)
        try: active_module = int(active_module_text.split(" ")[1])
        except (IndexError, ValueError): QtWidgets.QMessageBox.critical(self, "Error", f"Invalid module tab: {active_module_text}"); return
        resonances = self.resonance_freqs.get(active_module, [])
        if not resonances: QtWidgets.QMessageBox.information(self, "No Resonances", f"No resonances for Module {active_module}. Run 'Find Resonances'."); return
        dac_scales_for_dialog = {}
        if hasattr(self.parent(), 'dac_scales'): dac_scales_for_dialog = self.parent().dac_scales
        elif hasattr(self, 'dac_scales'): dac_scales_for_dialog = self.dac_scales
        
        dialog = MultisweepDialog(parent=self, resonance_frequencies=resonances, dac_scales=dac_scales_for_dialog, current_module=active_module)
        if dialog.exec():
            params = dialog.get_parameters()
            if params and hasattr(self.parent(), '_start_multisweep_analysis'):
                self.parent()._start_multisweep_analysis(params)
            elif not hasattr(self.parent(), '_start_multisweep_analysis'):
                 QtWidgets.QMessageBox.critical(self, "Error", "Cannot start multisweep: Parent integration missing.")
