"""Window class for network analysis results."""
import datetime # Added for MultisweepWindow export
import pickle   # Added for MultisweepWindow export
from .periscope_utils import *
from .periscope_tasks import *
from .periscope_dialogs import NetworkAnalysisParamsDialog, FindResonancesDialog, MultisweepDialog # Added MultisweepDialog

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
        for module_id_iter in self.plots: # Use a different variable name to avoid conflict
            plot_info = self.plots[module_id_iter]
            amp_plot = plot_info['amp_plot']
            phase_plot = plot_info['phase_plot']
            
            # Clear legends
            plot_info['amp_legend'].clear()
            plot_info['phase_legend'].clear()
            
            # Remove all amplitude-specific curves from plots
            for amp, curve in list(plot_info['amp_curves'].items()):
                amp_plot.removeItem(curve)
            for amp, curve in list(plot_info['phase_curves'].items()):
                phase_plot.removeItem(curve)
            
            # Clear curve dictionaries
            plot_info['amp_curves'].clear()
            plot_info['phase_curves'].clear()
            
            # Make sure main curves have no data 
            plot_info['amp_curve'].setData([], [])
            plot_info['phase_curve'].setData([], [])

            # Clear faux resonance legend items (Requirement 3)
            self._remove_faux_resonance_legend_entry(module_id_iter)
            self._update_multisweep_button_state(module_id_iter) # Update button state


    def update_amplitude_progress(self, module: int, current_amp: int, total_amps: int, amplitude: float):
        """Update the amplitude progress display for a module."""
        if hasattr(self, 'progress_labels') and module in self.progress_labels:
            self.progress_labels[module].setText(f"Amplitude {current_amp}/{total_amps} ({amplitude})")

    def _toggle_normalization(self, checked):
        """Toggle normalization of magnitude plots."""
        self.normalize_magnitudes = checked
        
        # Update axis labels
        for module in self.plots:
            self._update_amplitude_labels(self.plots[module]['amp_plot'])
        
        # Make sure main curves are cleared if we have amplitude-specific curves
        for module in self.plots:
            if len(self.plots[module]['amp_curves']) > 0:
                # Clear main curves if we have amplitude-specific curves
                self.plots[module]['amp_curve'].setData([], [])
                self.plots[module]['phase_curve'].setData([], [])
        
        # Redraw only amplitude plots with normalization applied
        for module_id in self.plots: # Iterate using module_id from self.plots.keys()
            # Update Y-axis label of the amplitude plot
            self._update_amplitude_labels(self.plots[module_id]['amp_plot'])

            # Redraw amplitude data for this module_id
            if module_id in self.raw_data:
                plot_info = self.plots[module_id]
                
                # Update amplitude-specific curves
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
                
                plot_info['amp_plot'].autoRange() # Auto-range based on enabled axes (Y is enabled)
        
        # Legends might need updating if normalization changes how amplitudes are displayed (e.g. if it affects units in legend)
        # However, current legend text is based on unit_mode and absolute amplitude, not normalization state.
        # If normalization were to change legend text (e.g. "Normalized Probe: ..."), then this would be needed:
        # self._update_legends_for_unit_mode() 


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
            # Update legend entry visibility (Requirement 3)
            self._update_resonance_legend_entry(module_id)

    def _toggle_resonance_edit_mode(self, checked: bool):
        """Enable or disable double-click add/subtract mode."""
        self.add_subtract_mode = checked

    def _update_resonance_checkbox_text(self, module: int):
        """Update the show-resonances checkbox label (count removed)."""
        # count = len(self.resonance_freqs.get(module, [])) # Count is no longer displayed here
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
        self._update_resonance_checkbox_text(module) # Though count is removed, keep for consistency if needed later
        self._update_resonance_legend_entry(module) # Requirement 3
        self._toggle_resonances_visible(self.show_resonances_cb.isChecked())
        self._update_multisweep_button_state(module)


    def _remove_resonance(self, module: int, freq_hz: float):
        """Remove the nearest resonance line."""
        if module not in self.plots or module not in self.resonance_freqs:
            return
        freqs = self.resonance_freqs[module]
        if not freqs:
            self._update_multisweep_button_state(module) # Update even if no freqs to remove
            return
        freqs_arr = np.array(freqs)
        idx = int(np.argmin(np.abs(freqs_arr - freq_hz)))
        # Remove lines
        line_mag = self.plots[module]['resonance_lines_mag'].pop(idx)
        line_phase = self.plots[module]['resonance_lines_phase'].pop(idx)
        self.plots[module]['amp_plot'].removeItem(line_mag)
        self.plots[module]['phase_plot'].removeItem(line_phase)
        freqs.pop(idx)
        self._update_resonance_checkbox_text(module) # Though count is removed, keep for consistency
        self._update_resonance_legend_entry(module) # Requirement 3
        self._update_multisweep_button_state(module)

    def _update_unit_mode(self, mode):
        """Update unit mode and redraw only amplitude plots."""
        if mode != self.unit_mode:
            self.unit_mode = mode
            
            for module_id in self.plots: # Iterate using module_id from self.plots.keys()
                # Update Y-axis label of the amplitude plot
                self._update_amplitude_labels(self.plots[module_id]['amp_plot'])

                # Redraw amplitude data for this module_id
                if module_id in self.raw_data:
                    plot_info = self.plots[module_id]
                    
                    # Update amplitude-specific curves
                    for amp_key, raw_entry_tuple in self.raw_data[module_id].items():
                        # amp_key is like "module_amplitude" or "default"
                        # raw_entry_tuple is (freqs, amps, phases, iq_data, optional_amplitude_val)
                        
                        amplitude_val, freqs, amps_raw, _, iq_data_raw = self._extract_data_from_tuple(amp_key, raw_entry_tuple)
                        
                        if freqs is None or amps_raw is None or iq_data_raw is None: # Should not happen with valid data
                            continue

                        converted_amps = UnitConverter.convert_amplitude(
                            amps_raw, iq_data_raw, self.unit_mode, normalize=self.normalize_magnitudes
                        )
                        
                        if amp_key != 'default': # Amplitude specific curve
                            # The amplitude_val from _extract_data_from_tuple is the actual float value
                            if amplitude_val in plot_info['amp_curves']:
                                plot_info['amp_curves'][amplitude_val].setData(freqs, converted_amps)
                        else: # Default curve (only if no amp-specific curves exist for this plot)
                            if not plot_info['amp_curves']: # Check if amp_curves dict is empty
                                plot_info['amp_curve'].setData(freqs, converted_amps)
                    
                    plot_info['amp_plot'].autoRange() # Auto-range based on enabled axes (Y is enabled)
            
            # Update legends because amplitude labels might change (e.g. "Probe: X dBm" vs "Probe: Y Norm Units")
            self._update_legends_for_unit_mode()

    def _update_legends_for_unit_mode(self):
        """Update the legend entries to reflect the current unit mode."""
        for module in self.plots:
            # Clear existing legends
            self.plots[module]['amp_legend'].clear()
            self.plots[module]['phase_legend'].clear()
            
            # Re-add curves with updated labels
            for amplitude, curve in self.plots[module]['amp_curves'].items():
                # Format amplitude according to current unit mode
                if self.unit_mode == "dbm":
                    # Check if we have a DAC scale for this module
                    if module not in self.dac_scales:
                        # Issue warning and switch to counts mode
                        print(f"Warning: No DAC scale available for module {module}, cannot display accurate probe power in dBm.")
                        self.rb_counts.setChecked(True)  # Switch to counts mode
                        return  # Exit and let _update_unit_mode call us again
                    
                    # Use the actual DAC scale without fallback
                    dac_scale = self.dac_scales[module]
                    dbm_value = UnitConverter.normalize_to_dbm(amplitude, dac_scale)
                    label = f"Probe: {dbm_value:.2f} dBm"
                elif self.unit_mode == "volts":
                    # Properly convert normalized amplitude to voltage through power calculation
                    # First get dBm value
                    if module not in self.dac_scales:
                        print(f"Warning: No DAC scale available for module {module}, cannot display accurate probe amplitude in Volts.")
                        self.rb_counts.setChecked(True)  # Switch to counts mode
                        return  # Exit and let _update_unit_mode call us again
                    else:
                        dac_scale = self.dac_scales[module]
                        dbm_value = UnitConverter.normalize_to_dbm(amplitude, dac_scale)
                        
                        # Convert dBm to watts: P = 10^((dBm - 30)/10)
                        power_watts = 10**((dbm_value - 30)/10)
                        
                        # Convert watts to peak voltage: V = sqrt(P * R)
                        resistance = 50.0  # Ohms
                        voltage_rms = np.sqrt(power_watts * resistance)
                        
                        # Convert RMS to peak voltage
                        voltage_peak = voltage_rms * np.sqrt(2)
                        voltage_peak = voltage_peak*1e6
                        
                        label = f"Probe: {voltage_peak:.1f} uV (peak)"
                else:  # "counts"
                    label = f"Probe: {amplitude} Normalized Units"
                
                # Add to legend with new label
                self.plots[module]['amp_legend'].addItem(curve, label)
                
                # Also update the phase legend for consistency
                phase_curve = self.plots[module]['phase_curves'].get(amplitude)
                if phase_curve:
                    self.plots[module]['phase_legend'].addItem(phase_curve, label)         
    
    def _update_amplitude_labels(self, plot):
        """Update plot labels based on current unit mode and normalization state."""
        if self.normalize_magnitudes:
            if self.unit_mode == "dbm":
                plot.setLabel('left', 'Normalized Power', units='dB') # Corrected units to dB
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
        for module_id_iter_redraw in self.raw_data: # Renamed to avoid conflict
            if module_id_iter_redraw in self.plots:
                # Check if we have amplitude-specific curves
                has_amp_curves = len(self.plots[module_id_iter_redraw]['amp_curves']) > 0
                
                # Update amplitude-specific curves first
                for amp_key, data_tuple in self.raw_data[module_id_iter_redraw].items():
                    if amp_key != 'default':
                        # Extract amplitude and data
                        amplitude, freqs, amps, phases, iq_data = self._extract_data_from_tuple(amp_key, data_tuple)
                        
                        # Update the curve if it exists
                        if amplitude in self.plots[module_id_iter_redraw]['amp_curves']:
                            converted_amps = UnitConverter.convert_amplitude(
                                amps, iq_data, self.unit_mode, normalize=self.normalize_magnitudes)
                            # Keep frequencies in Hz for plotting, as axes are in Hz
                            self.plots[module_id_iter_redraw]['amp_curves'][amplitude].setData(freqs, converted_amps)
                            self.plots[module_id_iter_redraw]['phase_curves'][amplitude].setData(freqs, phases)
                
                # Now handle the default curve - ONLY if there are no amplitude-specific curves
                if 'default' in self.raw_data[module_id_iter_redraw] and not has_amp_curves:
                    freqs, amps, phases, iq_data = self.raw_data[module_id_iter_redraw]['default']
                    converted_amps = UnitConverter.convert_amplitude(
                        amps, iq_data, self.unit_mode, normalize=self.normalize_magnitudes)
                    # Keep frequencies in Hz for plotting
                    self.plots[module_id_iter_redraw]['amp_curve'].setData(freqs, converted_amps)
                    self.plots[module_id_iter_redraw]['phase_curve'].setData(freqs, phases)
                else:
                    # Make sure default curves have no data if we have amplitude-specific curves
                    self.plots[module_id_iter_redraw]['amp_curve'].setData([], [])
                    self.plots[module_id_iter_redraw]['phase_curve'].setData([], [])
                
                # Enable auto range to fit new data
                self.plots[module_id_iter_redraw]['amp_plot'].autoRange()
        
        # Update legends after redrawing curves
        self._update_legends_for_unit_mode()              
    
    def _extract_data_from_tuple(self, amp_key, data_tuple):
        """Extract amplitude and data from a data tuple."""
        if len(data_tuple) == 5:
            # New format with amplitude included
            freqs, amps, phases, iq_data, amplitude = data_tuple
        else:
            # Old format, extract amplitude from key
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
            # Find our window ID
            window_id = None
            for w_id, w_data in parent.netanal_windows.items():
                if w_data['window'] == self:
                    window_id = w_id
                    break
            
            if window_id:
                # Stop all tasks for this window
                if hasattr(parent, 'netanal_tasks'):
                    for task_key in list(parent.netanal_tasks.keys()):
                        if task_key.startswith(f"{window_id}_"):
                            task = parent.netanal_tasks.pop(task_key)
                            task.stop()
                
                # Remove from windows dictionary
                parent.netanal_windows.pop(window_id, None)
        
        # Call parent implementation
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
            self._update_multisweep_button_state(active_module) # Update button state
            return

        target_sweep_key = None
        # Try to find the last run amplitude sweep
        ordered_amplitudes_run = self.original_params.get('amps', [])
        if ordered_amplitudes_run:
            for amp_setting in reversed(ordered_amplitudes_run):
                sweep_key = f"{active_module}_{amp_setting}"
                if sweep_key in module_sweeps:
                    target_sweep_key = sweep_key
                    break
        
        # Fallback to 'default' if no amplitude-specific sweep found or if 'amps' was empty
        if target_sweep_key is None and 'default' in module_sweeps:
            target_sweep_key = 'default'

        if target_sweep_key is None:
            # Fallback to the very last key added to the dictionary if all else fails
            if module_sweeps:
                target_sweep_key = list(module_sweeps.keys())[-1]


        if target_sweep_key is None:
            QtWidgets.QMessageBox.warning(self, "No Data", f"Could not determine which sweep to analyze for module {active_module}.")
            self._update_multisweep_button_state(active_module) # Update button state
            return

        # Extract data from the chosen sweep
        data_tuple = module_sweeps[target_sweep_key]
        
        if len(data_tuple) == 5: # New format with amplitude
            frequencies, _, _, iq_complex, _ = data_tuple
        elif len(data_tuple) == 4: # Old format or default
            frequencies, _, _, iq_complex = data_tuple
        else:
            QtWidgets.QMessageBox.critical(self, "Data Error", "Unexpected data format for the selected sweep.")
            self._update_multisweep_button_state(active_module) # Update button state
            return

        if len(frequencies) == 0 or len(iq_complex) == 0:
            QtWidgets.QMessageBox.information(self, "No Data", "Selected sweep has no frequency or IQ data.")
            self._update_multisweep_button_state(active_module) # Update button state
            return
            
        # Run find_resonances
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
            self._update_multisweep_button_state(active_module) # Update button state
            return

        # Plot the results
        if active_module in self.plots:
            plot_info = self.plots[active_module]
            amp_plot_item = plot_info['amp_plot'].getPlotItem()
            phase_plot_item = plot_info['phase_plot'].getPlotItem()
            # amp_legend = plot_info['amp_legend'] # Not directly used here
            # phase_legend = plot_info['phase_legend'] # Not directly used here

            # 1. Clear ALL previous resonance lines from plots and stored lists
            for line in plot_info.get('resonance_lines_mag', []):
                amp_plot_item.removeItem(line)
            plot_info['resonance_lines_mag'] = []

            for line in plot_info.get('resonance_lines_phase', []):
                phase_plot_item.removeItem(line)
            plot_info['resonance_lines_phase'] = []

            # 2. Clear old resonance frequency data
            self.resonance_freqs[active_module] = []
            
            res_freqs_hz = resonance_results.get('resonance_frequencies', [])

            if not res_freqs_hz:
                QtWidgets.QMessageBox.information(self, "No Resonances Found", 
                                                  f"No resonances were identified for Module {active_module} with the given parameters.")
                self._update_multisweep_button_state(active_module) # Update button state
                return # Important to return if no resonances, so no legend item is added

            # 3. Add new lines (all initially visible)
            line_pen = pg.mkPen('r', style=QtCore.Qt.PenStyle.DashLine)
            for res_freq_hz in res_freqs_hz:
                line_mag = pg.InfiniteLine(pos=res_freq_hz, angle=90, movable=False, pen=line_pen)
                amp_plot_item.addItem(line_mag)
                plot_info['resonance_lines_mag'].append(line_mag)

                line_phase = pg.InfiniteLine(pos=res_freq_hz, angle=90, movable=False, pen=line_pen)
                phase_plot_item.addItem(line_phase)
                plot_info['resonance_lines_phase'].append(line_phase)

            # 4. Store resonance frequencies and update checkbox/legend
            self.resonance_freqs[active_module] = res_freqs_hz
            self._update_resonance_checkbox_text(active_module) # Checkbox text itself doesn't change with count anymore
            self._update_resonance_legend_entry(active_module) # Requirement 3: Update legend
            self._toggle_resonances_visible(self.show_resonances_cb.isChecked()) # Ensure lines visibility matches checkbox
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
                except Exception: # Broad except as removeItem might fail if item already gone
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

        self._remove_faux_resonance_legend_entry(module_id) # Clear existing first

        plot_info = self.plots[module_id]
        amp_legend = plot_info['amp_legend']
        phase_legend = plot_info['phase_legend']
        
        count = len(self.resonance_freqs.get(module_id, []))

        if self.show_resonances_cb.isChecked() and count > 0:
            # Create a dummy PlotDataItem to get the sample swatch in the legend
            # This item itself won't be added to the plot, only its legend representation
            dummy_pen = pg.mkPen('r', style=QtCore.Qt.PenStyle.DashLine)
            
            # For Magnitude Plot Legend
            # We need to add an actual item to the legend.
            # A simple way is to create a PlotDataItem, add it to the legend, then make it invisible
            # or ensure it has no data.
            # However, LegendItem.addItem takes a PlotDataItem and a name.
            
            # Create a new PlotDataItem for the legend entry
            # This item will not be added to the plot itself, only to the legend.
            legend_sample_item_mag = pg.PlotDataItem(pen=dummy_pen)
            amp_legend.addItem(legend_sample_item_mag, f"{count} resonances")
            self.faux_resonance_legend_items_mag[module_id] = legend_sample_item_mag # Store the item used for legend

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
            
        # Find our window data in the parent
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
        
        # Check if there are any pending amplitudes
        no_pending_amplitudes = True
        for module in window_data['amplitude_queues']:
            if window_data['amplitude_queues'][module]:
                no_pending_amplitudes = False
                break
        
        # Hide progress when all bars are at 100% and no pending amplitudes
        all_complete = all(pbar.value() == 100 for pbar in self.progress_bars.values())
        if all_complete and no_pending_amplitudes:
            self.progress_group.setVisible(False)

    def _edit_parameters(self):
        """Open dialog to edit parameters. Re-runs analysis using per-module cable lengths."""
        # Prepare parameters for the dialog: use current_params but exclude any cable length info,
        # as the dialog is for other settings.
        params_for_dialog = self.current_params.copy()
        params_for_dialog.pop('module_cable_lengths', None)
        params_for_dialog.pop('cable_length', None) # Ensure no single cable_length is passed to dialog context

        dialog = NetworkAnalysisParamsDialog(self, params_for_dialog)
        if dialog.exec():
            # Get updated general parameters from the dialog
            updated_general_params = dialog.get_parameters() # These don't include cable lengths
            
            if updated_general_params:
                # Start with the updated general parameters
                params_for_rerun = updated_general_params.copy()

                # Add the authoritative per-module cable lengths from the window state
                params_for_rerun['module_cable_lengths'] = self.module_cable_lengths.copy()
                
                # Ensure no single 'cable_length' key conflicts; module_cable_lengths is primary
                params_for_rerun.pop('cable_length', None)

                # Update the window's current_params to reflect the full set for the new run
                self.current_params = params_for_rerun.copy()
                
                # Make progress group visible again
                if self.progress_group:
                    self.progress_group.setVisible(True)
                
                # Call the parent's rerun method with the fully formed params
                self.parent()._rerun_network_analysis(self.current_params)

    def _rerun_analysis(self):
        """Re-run the analysis with potentially updated parameters."""
        if hasattr(self.parent(), '_rerun_network_analysis'):
            # Get current parameters and add the per-module cable lengths
            params = self.current_params.copy()
            # Instead of a single cable_length, pass the dictionary
            params['module_cable_lengths'] = self.module_cable_lengths.copy()
            # Remove the old single 'cable_length' if it exists, to avoid confusion
            params.pop('cable_length', None)

            # Make progress group visible again
            if self.progress_group:
                self.progress_group.setVisible(True)
            self.parent()._rerun_network_analysis(params)
    
    def set_params(self, params):
        """Set parameters for analysis."""
        self.original_params = params.copy()  # Keep original for reference
        self.current_params = params.copy()   # Keep current for dialog

        # Initialize per-module cable lengths (Requirement 2)
        default_cable_length_for_all = params.get('cable_length', DEFAULT_CABLE_LENGTH)
        # If params already contains per-module lengths (e.g. from a future advanced re-run), use it
        # For now, assume all modules start with the same cable length from params
        for mod_id in self.modules: # self.modules is set in __init__
            self.module_cable_lengths[mod_id] = params.get('module_cable_lengths', {}).get(mod_id, default_cable_length_for_all)

        # Update spinner for the initially active module (if any)
        # This will be handled by _on_active_module_changed if tabs are already created,
        # or when the first tab becomes current.
        # If tabs exist, trigger an update for the current one.
        if self.tabs.count() > 0:
            self._on_active_module_changed(self.tabs.currentIndex())
        elif self.modules: # If modules are defined but no tabs yet (e.g. during init)
            # Set the spinner to the first module's cable length as a sensible default
            # before any tab is explicitly selected.
            first_module_id = self.modules[0]
            initial_cable_length = self.module_cable_lengths.get(first_module_id, DEFAULT_CABLE_LENGTH)
            self.cable_length_spin.blockSignals(True)
            self.cable_length_spin.setValue(initial_cable_length)
            self.cable_length_spin.blockSignals(False)
        
        # Only try to set plot ranges if plots exist
        if not hasattr(self, 'plots') or not self.plots:
            return
            
        # Set initial plot ranges based on frequency parameters
        fmin = params.get('fmin', DEFAULT_MIN_FREQ)
        fmax = params.get('fmax', DEFAULT_MAX_FREQ)
        for module in self.plots:
            self.plots[module]['amp_plot'].setXRange(fmin, fmax)
            self.plots[module]['phase_plot'].setXRange(fmin, fmax)
            # Disable auto range on X axis but keep Y auto range
            self.plots[module]['amp_plot'].enableAutoRange(pg.ViewBox.XAxis, False)
            self.plots[module]['phase_plot'].enableAutoRange(pg.ViewBox.XAxis, False)
            self.plots[module]['amp_plot'].enableAutoRange(pg.ViewBox.YAxis, True)
            self.plots[module]['phase_plot'].enableAutoRange(pg.ViewBox.YAxis, True)
    
    def update_data_with_amp(self, module: int, freqs: np.ndarray, amps: np.ndarray, phases: np.ndarray, amplitude: float):
        """Update the plot data for a specific module and amplitude."""
        # Store the raw data for unit conversion
        iq_data = amps * np.exp(1j * np.radians(phases))  # Reconstruct complex data
        
        # Use a unique key that includes the amplitude
        key = f"{module}_{amplitude}"
        
        # Initialize dictionaries if needed
        if module not in self.raw_data:
            self.raw_data[module] = {}
        if module not in self.data:
            self.data[module] = {}
        
        # Store the data with amplitude
        self.raw_data[module][key] = (freqs, amps, phases, iq_data, amplitude)
        self.data[module][key] = (freqs, amps, phases)
        
        if module in self.plots:
            # Hide the main curve if this is the first amplitude-specific curve
            if len(self.plots[module]['amp_curves']) == 0:
                # Set main curves to empty data to effectively hide them
                self.plots[module]['amp_curve'].setData([], [])
                self.plots[module]['phase_curve'].setData([], [])
            
            # Convert amplitude to selected units
            converted_amps = UnitConverter.convert_amplitude(
                amps, iq_data, self.unit_mode, normalize=self.normalize_magnitudes)
            
            # Generate a color based on the amplitude index in the list of amplitudes
            amps_list = self.original_params.get('amps', [amplitude])
            if amplitude in amps_list:
                amp_index = amps_list.index(amplitude)
            else:
                amp_index = 0
                
            # Use the same color families as the main application
            channel_families = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
            ]
            color = channel_families[amp_index % len(channel_families)]
            
            # Create or update curves for this amplitude
            is_new_curve = False
            if amplitude not in self.plots[module]['amp_curves']:
                is_new_curve = True
                self.plots[module]['amp_curves'][amplitude] = self.plots[module]['amp_plot'].plot(
                    pen=pg.mkPen(color, width=LINE_WIDTH), name=f"Amp: {amplitude}")
                self.plots[module]['phase_curves'][amplitude] = self.plots[module]['phase_plot'].plot(
                    pen=pg.mkPen(color, width=LINE_WIDTH), name=f"Amp: {amplitude}")
            
            # Update the curves
            self.plots[module]['amp_curves'][amplitude].setData(freqs, converted_amps)
            self.plots[module]['phase_curves'][amplitude].setData(freqs, phases)
            
            # If this is a new curve, update legends for proper unit display
            if is_new_curve:
                self._update_legends_for_unit_mode()
        self._update_multisweep_button_state(module) # Update button state when data changes

    def update_data(self, module: int, freqs: np.ndarray, amps: np.ndarray, phases: np.ndarray):
        """Update the plot data for a specific module."""
        # Store the raw data for unit conversion
        iq_data = amps * np.exp(1j * np.radians(phases))  # Reconstruct complex data
        
        # Initialize dictionaries if needed
        if module not in self.raw_data:
            self.raw_data[module] = {}
        if module not in self.data:
            self.data[module] = {}
        
        # Store under 'default' key
        self.raw_data[module]['default'] = (freqs, amps, phases, iq_data)
        self.data[module]['default'] = (freqs, amps, phases)
        
        if module in self.plots:
            # Only show default curve if no amplitude-specific curves exist yet
            if len(self.plots[module]['amp_curves']) == 0:
                # Convert amplitude to selected units
                converted_amps = UnitConverter.convert_amplitude(
                    amps, iq_data, self.unit_mode, normalize=self.normalize_magnitudes)
                
                # Convert frequency to GHz for display
                freq_ghz = freqs
                
                # Update plots
                self.plots[module]['amp_curve'].setData(freq_ghz, converted_amps)
                self.plots[module]['phase_curve'].setData(freq_ghz, phases)
        self._update_multisweep_button_state(module) # Update button state when data changes
    
    def update_progress(self, module: int, progress: float):
        """Update the progress bar for a specific module."""
        if module in self.progress_bars:
            self.progress_bars[module].setValue(int(progress))
    
    def complete_analysis(self, module: int):
        """Mark analysis as complete for a module."""
        if module in self.progress_bars:
            self.progress_bars[module].setValue(100)
            # Check if all modules are complete to hide progress bars
            self._check_all_complete()
    
    def _export_data(self):
        """Export the collected data with all unit conversions and metadata."""
        if not self.data:
            QtWidgets.QMessageBox.warning(self, "No Data", "No data to export yet.")
            return
        
        dialog = QtWidgets.QFileDialog(self)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilters([
            "Pickle Files (*.pkl)",
            "CSV Files (*.csv)",
            "All Files (*)"
        ])
        dialog.setDefaultSuffix("pkl")
        
        if dialog.exec():
            filename = dialog.selectedFiles()[0]
            
            try:
                if filename.endswith('.pkl'):
                    self._export_to_pickle(filename)
                elif filename.endswith('.csv'):
                    self._export_to_csv(filename)
                else:
                    # Default to pickle with same comprehensive format
                    self._export_to_pickle(filename)
                    
                QtWidgets.QMessageBox.information(self, "Export Complete", 
                                                f"Data exported to {filename}")
                
            except Exception as e:
                traceback.print_exc()  # Print stack trace to console
                QtWidgets.QMessageBox.critical(self, "Export Error", 
                                            f"Error exporting data: {str(e)}")
    
    def _export_to_pickle(self, filename):
        """Export data to a pickle file."""
        # Create comprehensive data structure with all units and metadata
        export_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'parameters': self.current_params.copy() if hasattr(self, 'current_params') else {},
            'modules': {}
        }
        
        # Process each module's data
        for module, data_dict in self.raw_data.items():
            export_data['modules'][module] = {}
            
            # Track measurement index for each module
            meas_idx = 0
            
            for key, data_tuple in data_dict.items():
                # Extract data and amplitude
                amplitude, freqs, amps, phases, iq_data = self._extract_data_for_export(key, data_tuple)
                
                # Convert to all unit types
                counts = amps  # Already in counts
                volts = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="volts")
                dbm = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="dbm")
                
                # Also include normalized versions
                counts_norm = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="counts", normalize=True)
                volts_norm = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="volts", normalize=True)
                dbm_norm = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="dbm", normalize=True)
                
                # Use iteration number as key instead of formatted amplitude
                export_data['modules'][module][meas_idx] = {
                    'sweep_amplitude': amplitude,
                    'frequency': {
                        'values': freqs.tolist(),
                        'unit': 'Hz'
                    },
                    'magnitude': {
                        'counts': {
                            'raw': counts.tolist(),
                            'normalized': counts_norm.tolist(),
                            'unit': 'counts'
                        },
                        'volts': {
                            'raw': volts.tolist(),
                            'normalized': volts_norm.tolist(),
                            'unit': 'V'
                        },
                        'dbm': {
                            'raw': dbm.tolist(),
                            'normalized': dbm_norm.tolist(),
                            'unit': 'dBm'
                        }
                    },
                    'phase': {
                        'values': phases.tolist(),
                        'unit': 'degrees'
                    },
                    'complex': {
                        'real': iq_data.real.tolist(),
                        'imag': iq_data.imag.tolist()
                    }
                }
                meas_idx += 1

            export_data['modules'][module]['resonances_hz'] = self.resonance_freqs.get(module, [])
        
        # Save the enhanced data structure
        with open(filename, 'wb') as f:
            pickle.dump(export_data, f)
    
    def _export_to_csv(self, filename):
        """Export data to CSV files."""
        base, ext = os.path.splitext(filename)
        
        # First create a metadata CSV with the parameters
        meta_filename = f"{base}_metadata{ext}"
        with open(meta_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Parameter', 'Value'])
            writer.writerow(['Export Date', datetime.datetime.now().isoformat()])
            
            # Add all parameters
            if hasattr(self, 'current_params'):
                writer.writerow(['', ''])
                writer.writerow(['Measurement Parameters', ''])
                for param, value in self.current_params.items():
                    # Convert Hz to MHz for frequency parameters
                    if param in ['fmin', 'fmax', 'max_span'] and isinstance(value, (int, float)):
                        writer.writerow([param, f"{value/1e6} MHz"])
                    else:
                        writer.writerow([param, value])

            if self.resonance_freqs:
                writer.writerow(['', ''])
                writer.writerow(['Resonances (Hz)', ''])
                for module, freqs in self.resonance_freqs.items():
                    freq_str = ','.join(str(f) for f in freqs)
                    writer.writerow([f'Module {module}', freq_str])
        
        # Now export each module's data
        for module, data_dict in self.raw_data.items():
            idx = 0  # Track measurement index
            for key, data_tuple in data_dict.items():
                # Extract data and amplitude
                amplitude, freqs, amps, phases, iq_data = self._extract_data_for_export(key, data_tuple)
                
                # Export for each unit type
                for unit_mode in ["counts", "volts", "dbm"]:
                    converted_amps = UnitConverter.convert_amplitude(amps, iq_data, unit_mode=unit_mode)
                    
                    unit_label = unit_mode
                    if unit_mode == "dbm":
                        unit_label = "dBm"
                    elif unit_mode == "volts":
                        unit_label = "V"
                    
                    csv_filename = f"{base}_module{module}_idx{idx}_{unit_mode}{ext}"
                    with open(csv_filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['# Amplitude:', f"{amplitude}" if 'amplitude' in locals() else "Unknown"])
                        if unit_mode == "dbm":
                            writer.writerow(['Frequency (Hz)', f'Power ({unit_label})', 'Phase (deg)'])
                        else:
                            writer.writerow(['Frequency (Hz)', f'Amplitude ({unit_label})', 'Phase (deg)'])
                        
                        for freq, amp, phase in zip(freqs, converted_amps, phases):
                            writer.writerow([freq, amp, phase])
                idx += 1
    
    def _extract_data_for_export(self, key, data_tuple):
        """Extract and prepare data for export from a data tuple."""
        if key != 'default':
            if len(data_tuple) >= 5:  # New format with amplitude included
                freqs, amps, phases, iq_data, amplitude = data_tuple
            else:
                # Try to extract amplitude from key format "module_amp"
                freqs, amps, phases, iq_data = data_tuple
                try:
                    amplitude = float(key.split('_')[-1])
                except (ValueError, IndexError):
                    amplitude = DEFAULT_AMPLITUDE
        else:
            amplitude = DEFAULT_AMPLITUDE  # Default for non-amplitude-specific data
            freqs, amps, phases, iq_data = data_tuple
            
        return amplitude, freqs, amps, phases, iq_data

    def _unwrap_cable_delay_action(self):
        """
        Fits the phase data of the first curve in the active module's plot,
        calculates the corresponding cable length, updates the phase curves,
        and adjusts the cable length spinner.
        """
        if not self.raw_data:
            QtWidgets.QMessageBox.information(self, "No Data", "No network analysis data available to process.")
            return

        current_tab_index = self.tabs.currentIndex()
        if current_tab_index < 0:
            QtWidgets.QMessageBox.warning(self, "No Module Selected", "Please select a module tab.")
            return
        
        active_module_text = self.tabs.tabText(current_tab_index)
        try:
            active_module = int(active_module_text.split(" ")[1])
        except (IndexError, ValueError):
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not determine active module from tab: {active_module_text}")
            return

        if active_module not in self.raw_data or not self.raw_data[active_module]:
            QtWidgets.QMessageBox.information(self, "No Data", f"No data for Module {active_module}.")
            return
        
        module_data_dict = self.raw_data[active_module]
        
        target_key = None
        if 'amps' in self.original_params and self.original_params['amps']:
            first_amplitude_setting = self.original_params['amps'][0]
            potential_key = f"{active_module}_{first_amplitude_setting}"
            if potential_key in module_data_dict:
                target_key = potential_key
        
        if target_key is None and 'default' in module_data_dict:
            target_key = 'default'
        
        if target_key is None:
            if module_data_dict:
                target_key = next(iter(module_data_dict))
            else:
                QtWidgets.QMessageBox.information(self, "No Data", f"No sweep data found for Module {active_module} to process.")
                return

        # --- Extract data for the chosen key ---
        # data_tuple: (freqs, amps, phases_displayed_deg, iq_data, amplitude_setting) OR (freqs, amps, phases_displayed_deg, iq_data)
        data_tuple = module_data_dict[target_key]
        
        if len(data_tuple) == 5:
            freqs_active, _, phases_displayed_active_deg, _, _ = data_tuple
        elif len(data_tuple) == 4:
            freqs_active, _, phases_displayed_active_deg, _ = data_tuple
        else:
            QtWidgets.QMessageBox.critical(self, "Error", "Unexpected data format for selected sweep.")
            return

        if len(freqs_active) == 0:
            QtWidgets.QMessageBox.information(self, "No Data", "Selected sweep has no frequency data.")
            return

        # --- Determine the old cable length used for this specific sweep ---
        # This is tricky. For now, assume current_params['cable_length'] was used for the "first" sweep.
        # A more robust solution would store the cable_length with each sweep in self.raw_data.
        L_old_physical = self.current_params.get('cable_length', DEFAULT_CABLE_LENGTH)

        # --- Perform Fit and Calculation ---
        try:
            tau_additional = fit_cable_delay(freqs_active, phases_displayed_active_deg)
            L_new_physical = calculate_new_cable_length(L_old_physical, tau_additional)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Calculation Error", f"Error during cable delay calculation: {str(e)}")
            traceback.print_exc()
            return

        # --- Update All Phase Curves for the Active Module ---
        if active_module in self.plots:
            plot_info = self.plots[active_module]
            
            # Update amplitude-specific phase curves
            for amp_key_iter, curve_item in plot_info['phase_curves'].items():
                # amp_key_iter is the amplitude value (float)
                # We need to find the corresponding full key in raw_data to get original displayed phases
                raw_data_key_for_curve = f"{active_module}_{amp_key_iter}"
                if raw_data_key_for_curve in module_data_dict:
                    data_tuple_curve = module_data_dict[raw_data_key_for_curve]
                    if len(data_tuple_curve) == 5:
                        freqs_curve, _, phases_deg_current_display_curve, _, _ = data_tuple_curve
                    elif len(data_tuple_curve) == 4: # Should not happen for amp_specific curves but handle
                        freqs_curve, _, phases_deg_current_display_curve, _ = data_tuple_curve
                    else:
                        continue # Skip malformed data

                    if len(freqs_curve) > 0:
                        # Assume this curve was also compensated with L_old_physical.
                        # This is an approximation if the user changed cable length between sweeps
                        # without a full re-run.
                        # First get the unwrapped phase data
                        new_phases_deg_for_curve = recalculate_displayed_phase(
                            freqs_curve, phases_deg_current_display_curve, L_old_physical, L_new_physical
                        )
                        
                        # Recenter the unwrapped phase data so the first point begins at 0 phase
                        if len(new_phases_deg_for_curve) > 0:
                            first_point_phase = new_phases_deg_for_curve[0]
                            new_phases_deg_for_curve = new_phases_deg_for_curve - first_point_phase
                            
                        # Now re-wrap phases to [-180, 180] range
                        new_phases_deg_for_curve = ((new_phases_deg_for_curve + 180) % 360) - 180
                            
                        curve_item.setData(freqs_curve, new_phases_deg_for_curve)
            
            # Update the main/default phase curve (if it holds data)
            # The main curve 'phase_curve' should only have data if no amp-specific curves exist.
            if not plot_info['phase_curves'] and 'default' in module_data_dict:
                main_curve_item = plot_info['phase_curve']
                data_tuple_main = module_data_dict['default']
                if len(data_tuple_main) == 4:
                    freqs_main, _, phases_deg_current_display_main, _ = data_tuple_main
                    if len(freqs_main) > 0:
                        # First get the unwrapped phase data
                        new_phases_deg_for_main = recalculate_displayed_phase(
                            freqs_main, phases_deg_current_display_main, L_old_physical, L_new_physical
                        )
                        
                        # Recenter the unwrapped phase data so the first point begins at 0 phase
                        if len(new_phases_deg_for_main) > 0:
                            first_point_phase = new_phases_deg_for_main[0]
                            new_phases_deg_for_main = new_phases_deg_for_main - first_point_phase
                            
                        # Now re-wrap phases to [-180, 180] range
                        new_phases_deg_for_main = ((new_phases_deg_for_main + 180) % 360) - 180
                        
                        main_curve_item.setData(freqs_main, new_phases_deg_for_main)

            plot_info['phase_plot'].enableAutoRange(pg.ViewBox.YAxis, True)


        # --- Update Cable Length Spinner and Stored Parameters ---
        self.module_cable_lengths[active_module] = L_new_physical # Update specific module's length
        self.cable_length_spin.blockSignals(True)
        self.cable_length_spin.setValue(L_new_physical)
        self.cable_length_spin.blockSignals(False)
        # self.current_params['cable_length'] will be updated if a global re-run happens,
        # or we'll need a more sophisticated way to pass per-module lengths to re-run.
        # For now, current_params reflects the global setting, while module_cable_lengths holds specifics.
        
        QtWidgets.QMessageBox.information(self, "Cable Delay Updated", 
                                          f"Cable length for Module {active_module} updated to {L_new_physical:.3f} m based on phase fit.")

    def _on_active_module_changed(self, index: int):
        """Update UI elements when the active module tab changes."""
        if index < 0 or not self.modules or index >= len(self.modules):
            self._update_multisweep_button_state(None) # Disable if no valid tab
            return

        active_module_id = self.modules[index] # Assuming self.modules order matches tab order

        # Update cable length spinner (Requirement 2)
        if active_module_id in self.module_cable_lengths:
            self.cable_length_spin.blockSignals(True)
            self.cable_length_spin.setValue(self.module_cable_lengths[active_module_id])
            self.cable_length_spin.blockSignals(False)
        else:
            # Fallback if somehow not set, though set_params should handle it
            self.cable_length_spin.blockSignals(True)
            self.cable_length_spin.setValue(self.current_params.get('cable_length', DEFAULT_CABLE_LENGTH))
            self.cable_length_spin.blockSignals(False)
        
        self._update_multisweep_button_state(active_module_id) # Update button for new active module

        # Future: Update other per-module UI elements here if any

    def _on_cable_length_changed(self, new_length: float):
        """Handle changes to the cable length spinner."""
        current_tab_index = self.tabs.currentIndex()
        if current_tab_index < 0 or not self.modules or current_tab_index >= len(self.modules):
            return

        active_module_id = self.modules[current_tab_index]
        self.module_cable_lengths[active_module_id] = new_length
        # Note: This change only affects the current module's stored value.
        # A re-run or re-analysis would be needed to apply this to the plots
        # unless _unwrap_cable_delay_action is called, which re-calculates phase.
        self._update_multisweep_button_state(active_module_id)

    def _update_multisweep_button_state(self, module_id: int | None = None):
        """Enable or disable the Take Multisweep button based on found resonances for the given module."""
        if not hasattr(self, 'take_multisweep_btn'): # Button might not be initialized yet
            return

        if module_id is None: # If no specific module, check current tab
            current_tab_index = self.tabs.currentIndex()
            if current_tab_index < 0:
                self.take_multisweep_btn.setEnabled(False)
                return
            active_module_text = self.tabs.tabText(current_tab_index)
            try:
                module_id = int(active_module_text.split(" ")[1])
            except (IndexError, ValueError):
                self.take_multisweep_btn.setEnabled(False)
                return
        
        has_resonances = bool(self.resonance_freqs.get(module_id))
        self.take_multisweep_btn.setEnabled(has_resonances)

    def _show_multisweep_dialog(self):
        """Show the dialog to configure and run multisweep."""
        current_tab_index = self.tabs.currentIndex()
        if current_tab_index < 0:
            QtWidgets.QMessageBox.warning(self, "No Module Selected", "Please select a module tab.")
            return
        
        active_module_text = self.tabs.tabText(current_tab_index)
        try:
            active_module = int(active_module_text.split(" ")[1])
        except (IndexError, ValueError):
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not determine active module from tab: {active_module_text}")
            return

        resonances = self.resonance_freqs.get(active_module, [])
        if not resonances:
            QtWidgets.QMessageBox.information(self, "No Resonances", 
                                              f"No resonances found for Module {active_module}. Please run 'Find Resonances' first.")
            return

        # Ensure dac_scales are available for the dialog
        # self.parent() is the Periscope main application instance
        dac_scales_for_dialog = {}
        if hasattr(self.parent(), 'dac_scales'):
            dac_scales_for_dialog = self.parent().dac_scales
        elif hasattr(self, 'dac_scales'): # Fallback to own dac_scales if parent doesn't have them (less likely)
             dac_scales_for_dialog = self.dac_scales


        dialog = MultisweepDialog(
            parent=self, 
            resonance_frequencies=resonances,
            dac_scales=dac_scales_for_dialog, 
            current_module=active_module
        )

        if dialog.exec():
            params = dialog.get_parameters()
            if params:
                # Parameters for MultisweepTask: crs, resonance_frequencies, params, signals
                # The Periscope main app will handle creating the task and new window.
                # We need to call a method on self.parent() (which is Periscope instance)
                if hasattr(self.parent(), '_start_multisweep_analysis'):
                    # params already includes 'module', 'resonance_frequencies', 'amps', 'span_hz', etc.
                    self.parent()._start_multisweep_analysis(params)
                else:
                    QtWidgets.QMessageBox.critical(self, "Error", "Cannot start multisweep: Parent integration missing.")

# ───────────────────────── Multisweep Window ─────────────────────────
class MultisweepWindow(QtWidgets.QMainWindow):
    """
    Window for displaying multisweep analysis results.
    Plots combined magnitude and phase for all resonances for each amplitude.
    """
    def __init__(self, parent=None, target_module=None, initial_params=None, dac_scales=None):
        super().__init__(parent)
        self.target_module = target_module
        self.initial_params = initial_params or {} # amps, span_hz, npoints_per_sweep, perform_fits etc.
        self.dac_scales = dac_scales or {}
        
        self.setWindowTitle(f"Multisweep Results - Module {self.target_module}")
        self.results_by_amplitude = {} # {amp: {cf: data_dict, ...}, ...}
        self.current_amplitude_being_processed = None
        self.unit_mode = "dbm"
        self.normalize_magnitudes = False
        self.zoom_box_mode = True # Default to zoom box mode ON

        self.combined_mag_plot = None
        self.combined_phase_plot = None
        self.mag_legend = None
        self.phase_legend = None
        self.curves_mag = {} # {amp: {cf: curve_item}}
        self.curves_phase = {} # {amp: {cf: curve_item}}
        
        self.active_module_for_dac = self.target_module # For UnitConverter consistency

        # For showing center frequency lines
        self.show_cf_lines_cb = None
        self.cf_lines_mag = {} # {amp_val: [line_item, ...]}
        self.cf_lines_phase = {}

        self._setup_ui()
        self.resize(1200, 800)

    def _setup_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        self._setup_toolbar(main_layout)
        self._setup_plot_area(main_layout)
        self._setup_status_bar()


    def _setup_toolbar(self, layout):
        toolbar = QtWidgets.QToolBar("Multisweep Controls")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.export_btn = QtWidgets.QPushButton("Export Data")
        self.export_btn.clicked.connect(self._export_data)
        toolbar.addWidget(self.export_btn)

        self.rerun_btn = QtWidgets.QPushButton("Re-run Multisweep")
        self.rerun_btn.clicked.connect(self._rerun_multisweep)
        # self.rerun_btn.setEnabled(False) # Enable when all_completed
        toolbar.addWidget(self.rerun_btn)
        
        toolbar.addSeparator()

        # Progress bar for current amplitude
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True) # Initially visible
        toolbar.addWidget(QtWidgets.QLabel("Current Sweep Progress:"))
        toolbar.addWidget(self.progress_bar)
        
        # Label for current amplitude
        self.current_amp_label = QtWidgets.QLabel("Current Amplitude: N/A")
        toolbar.addWidget(self.current_amp_label)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self.normalize_checkbox = QtWidgets.QCheckBox("Normalize Magnitudes")
        self.normalize_checkbox.setChecked(self.normalize_magnitudes)
        self.normalize_checkbox.toggled.connect(self._toggle_normalization)
        toolbar.addWidget(self.normalize_checkbox)

        self.show_cf_lines_cb = QtWidgets.QCheckBox("Show Center Frequencies")
        self.show_cf_lines_cb.setChecked(False) # Default to not showing
        self.show_cf_lines_cb.toggled.connect(self._toggle_cf_lines_visibility)
        toolbar.addWidget(self.show_cf_lines_cb)

        self._setup_unit_controls(toolbar)
        self._setup_zoom_box_control(toolbar)

    def _setup_unit_controls(self, toolbar):
        unit_group = QtWidgets.QWidget()
        unit_layout = QtWidgets.QHBoxLayout(unit_group)
        unit_layout.setContentsMargins(0, 0, 0, 0)
        unit_layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.rb_counts = QtWidgets.QRadioButton("Counts")
        self.rb_dbm = QtWidgets.QRadioButton("dBm")
        self.rb_volts = QtWidgets.QRadioButton("Volts")
        self.rb_dbm.setChecked(True)
        
        unit_layout.addWidget(QtWidgets.QLabel("Units:"))
        unit_layout.addWidget(self.rb_counts)
        unit_layout.addWidget(self.rb_dbm)
        unit_layout.addWidget(self.rb_volts)
        
        self.rb_counts.toggled.connect(lambda: self._update_unit_mode("counts"))
        self.rb_dbm.toggled.connect(lambda: self._update_unit_mode("dbm"))
        self.rb_volts.toggled.connect(lambda: self._update_unit_mode("volts"))
        
        unit_group.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        toolbar.addSeparator()
        toolbar.addWidget(unit_group)

    def _setup_zoom_box_control(self, toolbar):
        self.zoom_box_cb = QtWidgets.QCheckBox("Zoom Box Mode")
        self.zoom_box_cb.setChecked(self.zoom_box_mode)
        self.zoom_box_cb.toggled.connect(self._toggle_zoom_box_mode)
        toolbar.addWidget(self.zoom_box_cb)

    def _setup_plot_area(self, layout):
        plot_container = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_container)

        # Combined Magnitude Plot
        vb_mag = ClickableViewBox() # Use ClickableViewBox for zoom/pan
        vb_mag.parent_window = self # For click handling if needed later
        self.combined_mag_plot = pg.PlotWidget(viewBox=vb_mag, title="Combined S21 Magnitude (All Resonances)")
        self.combined_mag_plot.setLabel('bottom', 'Frequency', units='Hz')
        self._update_mag_plot_label() # Set initial Y label
        self.combined_mag_plot.showGrid(x=True, y=True, alpha=0.3)
        self.mag_legend = self.combined_mag_plot.addLegend(offset=(30,10))
        plot_layout.addWidget(self.combined_mag_plot)

        # Combined Phase Plot
        vb_phase = ClickableViewBox()
        vb_phase.parent_window = self
        self.combined_phase_plot = pg.PlotWidget(viewBox=vb_phase, title="Combined S21 Phase (All Resonances)")
        self.combined_phase_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.combined_phase_plot.setLabel('left', 'Phase', units='deg')
        self.combined_phase_plot.showGrid(x=True, y=True, alpha=0.3)
        self.phase_legend = self.combined_phase_plot.addLegend(offset=(30,10))
        plot_layout.addWidget(self.combined_phase_plot)
        
        layout.addWidget(plot_container)
        
        # Link X axes
        self.combined_phase_plot.setXLink(self.combined_mag_plot)
        self._apply_zoom_box_mode()


    def _setup_status_bar(self):
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

    def _toggle_normalization(self, checked):
        self.normalize_magnitudes = checked
        self._update_mag_plot_label()
        self._redraw_plots()

    def _update_unit_mode(self, mode):
        if self.unit_mode != mode:
            self.unit_mode = mode
            self._update_mag_plot_label()
            self._redraw_plots()
            
    def _update_mag_plot_label(self):
        if not self.combined_mag_plot: return
        if self.normalize_magnitudes:
            label = "Normalized Magnitude"
            units = "dB" if self.unit_mode == "dbm" else ""
        else:
            if self.unit_mode == "counts":
                label, units = "Magnitude", "Counts"
            elif self.unit_mode == "dbm":
                label, units = "Power", "dBm"
            elif self.unit_mode == "volts":
                label, units = "Magnitude", "V"
            else:
                label, units = "Magnitude", ""
        self.combined_mag_plot.setLabel('left', label, units=units)

    def _toggle_zoom_box_mode(self, enable):
        self.zoom_box_mode = enable
        self._apply_zoom_box_mode()

    def _apply_zoom_box_mode(self):
        if self.combined_mag_plot and isinstance(self.combined_mag_plot.getViewBox(), ClickableViewBox):
            self.combined_mag_plot.getViewBox().enableZoomBoxMode(self.zoom_box_mode)
        if self.combined_phase_plot and isinstance(self.combined_phase_plot.getViewBox(), ClickableViewBox):
            self.combined_phase_plot.getViewBox().enableZoomBoxMode(self.zoom_box_mode)

    def update_progress(self, module, progress_percentage):
        if module == self.target_module:
            self.progress_bar.setValue(int(progress_percentage))

    def update_intermediate_data(self, module, amplitude, intermediate_results):
        if module != self.target_module: return
        # intermediate_results is {cf: {'frequencies': ..., 'iq_complex': ...}}
        # This can be used for live plotting if desired, but for now,
        # we'll primarily update on final data_update per amplitude.
        # For simplicity, this is a placeholder for potential live updates.
        pass

    def update_data(self, module, amplitude, final_results_for_amplitude):
        if module != self.target_module: return
        
        self.current_amplitude_being_processed = amplitude
        self.current_amp_label.setText(f"Processing Amp: {amplitude:.4f}")
        self.results_by_amplitude[amplitude] = final_results_for_amplitude
        self._redraw_plots()

    def _redraw_plots(self):
        if not self.combined_mag_plot or not self.combined_phase_plot:
            return

        # Clear previous plot items and legends
        if self.mag_legend: self.mag_legend.clear()
        if self.phase_legend: self.phase_legend.clear()

        for item in self.combined_mag_plot.listDataItems():
            self.combined_mag_plot.removeItem(item)
        for item in self.combined_phase_plot.listDataItems():
            self.combined_phase_plot.removeItem(item)
        
        # self.curves_mag and self.curves_phase are not strictly needed anymore with this redraw logic,
        # but can be kept if direct curve manipulation is added later. For now, clear them.
        self.curves_mag.clear()
        self.curves_phase.clear()

        # Clear existing CF lines from plots and storage
        for amp_val_lines in self.cf_lines_mag.values():
            for line in amp_val_lines:
                self.combined_mag_plot.removeItem(line)
        self.cf_lines_mag.clear()

        for amp_val_lines in self.cf_lines_phase.values():
            for line in amp_val_lines:
                self.combined_phase_plot.removeItem(line)
        self.cf_lines_phase.clear()

        num_amps = len(self.results_by_amplitude)
        if num_amps == 0:
            return
        
        # Define the color scheme similar to NetworkAnalysisWindow for distinct colors per amplitude
        channel_families = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        viridis_cmap = pg.colormap.get("viridis")
        
        sorted_amplitudes = sorted(self.results_by_amplitude.keys())

        # Create legend items once per amplitude
        legend_items_mag = {}
        legend_items_phase = {}

        for amp_idx, amp_val in enumerate(sorted_amplitudes):
            amp_results = self.results_by_amplitude[amp_val]
            
            # Determine color based on the index of the amplitude and total number of amplitudes
            if num_amps <= 5:
                color = channel_families[amp_idx % len(channel_families)]
            else:
                # Use viridis for more than 5 amplitudes, mapping amp_idx to [0,1] for colormap
                color = viridis_cmap.map(amp_idx / max(1, num_amps - 1)) # max(1, num_amps-1) to avoid div by zero for single amp
            
            pen = pg.mkPen(color, width=LINE_WIDTH)
            
            # Format legend name based on unit_mode
            if self.unit_mode == "dbm":
                dac_scale = self.dac_scales.get(self.active_module_for_dac)
                if dac_scale is not None:
                    dbm_value = UnitConverter.normalize_to_dbm(amp_val, dac_scale)
                    legend_name_amp = f"Probe: {dbm_value:.2f} dBm"
                else:
                    legend_name_amp = f"Probe: {amp_val:.3e} (Norm)" # Fallback if DAC scale unknown
            elif self.unit_mode == "volts":
                dac_scale = self.dac_scales.get(self.active_module_for_dac)
                if dac_scale is not None:
                    dbm_value = UnitConverter.normalize_to_dbm(amp_val, dac_scale)
                    power_watts = 10**((dbm_value - 30)/10)
                    resistance = 50.0  # Ohms
                    voltage_rms = np.sqrt(power_watts * resistance)
                    voltage_peak_uv = voltage_rms * np.sqrt(2) * 1e6
                    legend_name_amp = f"Probe: {voltage_peak_uv:.1f} uVpk"
                else:
                    legend_name_amp = f"Probe: {amp_val:.3e} (Norm)" # Fallback
            else: # counts or other
                legend_name_amp = f"Probe: {amp_val:.3e} Norm"

            # Add to magnitude legend only once per amplitude
            if amp_val not in legend_items_mag:
                # Create a dummy item for the legend, or use the first curve of this amp
                # For simplicity, we'll add the first curve and then subsequent curves of the same amp won't add to legend.
                # This requires a flag or checking if legend item already exists.
                # A cleaner way: add a dummy plot item just for the legend.
                dummy_mag_curve_for_legend = pg.PlotDataItem(pen=pen) # Invisible item
                self.mag_legend.addItem(dummy_mag_curve_for_legend, legend_name_amp)
                legend_items_mag[amp_val] = dummy_mag_curve_for_legend # Mark as added

            if amp_val not in legend_items_phase:
                dummy_phase_curve_for_legend = pg.PlotDataItem(pen=pen) # Invisible item
                self.phase_legend.addItem(dummy_phase_curve_for_legend, legend_name_amp)
                legend_items_phase[amp_val] = dummy_phase_curve_for_legend


            for cf, data in amp_results.items():
                freqs_hz = data.get('frequencies')
                iq_complex = data.get('iq_complex')
                
                if freqs_hz is None or iq_complex is None or len(freqs_hz) == 0:
                    continue

                # Magnitude
                s21_mag_raw = np.abs(iq_complex)
                dac_scale_val = self.dac_scales.get(self.active_module_for_dac)
                
                s21_mag_processed = UnitConverter.convert_amplitude(
                    s21_mag_raw, # Pass raw magnitude
                    iq_complex,  # Pass complex IQ for dBm/Volts conversion
                    self.unit_mode,
                    normalize=self.normalize_magnitudes
                )

                # Phase
                phase_deg = data.get('phase_degrees', np.degrees(np.angle(iq_complex)))
                
                # Plot magnitude curve for this (amp, cf)
                mag_curve = self.combined_mag_plot.plot(pen=pen) # No name, legend handled per-amplitude
                mag_curve.setData(freqs_hz, s21_mag_processed)
                # Store if needed for other interactions, though not for legend here
                if amp_val not in self.curves_mag: self.curves_mag[amp_val] = {}
                self.curves_mag[amp_val][cf] = mag_curve

                # Plot phase curve for this (amp, cf)
                phase_curve = self.combined_phase_plot.plot(pen=pen) # No name
                phase_curve.setData(freqs_hz, phase_deg)
                if amp_val not in self.curves_phase: self.curves_phase[amp_val] = {}
                self.curves_phase[amp_val][cf] = phase_curve

                # If "Show Center Frequencies" is checked, add vertical lines for each cf_key
                if self.show_cf_lines_cb and self.show_cf_lines_cb.isChecked():
                    # The key 'cf' here is the center frequency for this specific sweep data
                    # (which could be original or recalculated based on how multisweep algorithm was run)
                    cf_line_pen = pg.mkPen(color, style=QtCore.Qt.PenStyle.DashLine, width=LINE_WIDTH/2) # Thinner dashed line
                    
                    mag_cf_line = pg.InfiniteLine(pos=cf, angle=90, pen=cf_line_pen, movable=False)
                    self.combined_mag_plot.addItem(mag_cf_line)
                    self.cf_lines_mag.setdefault(amp_val, []).append(mag_cf_line)

                    phase_cf_line = pg.InfiniteLine(pos=cf, angle=90, pen=cf_line_pen, movable=False)
                    self.combined_phase_plot.addItem(phase_cf_line)
                    self.cf_lines_phase.setdefault(amp_val, []).append(phase_cf_line)
        
        self.combined_mag_plot.autoRange()
        self.combined_phase_plot.autoRange()

    def completed_amplitude_sweep(self, module, amplitude):
        if module == self.target_module:
            self.statusBar.showMessage(f"Completed sweep for amplitude: {amplitude:.4f}")
            self.progress_bar.setValue(100) # Mark current amp as done

    def all_sweeps_completed(self):
        self.statusBar.showMessage("All multisweep amplitudes completed.")
        self.progress_bar.setVisible(False)
        self.current_amp_label.setText("All Amplitudes Processed")
        # self.rerun_btn.setEnabled(True)

    def handle_error(self, module, amplitude, error_msg):
        if module == self.target_module or module == -1: # -1 for general task error
            amp_str = f"for amplitude {amplitude:.4f}" if amplitude != -1 else "general"
            QtWidgets.QMessageBox.critical(self, "Multisweep Error", f"Error {amp_str} on Module {self.target_module}:\n{error_msg}")
            self.statusBar.showMessage(f"Error during multisweep {amp_str}.")
            self.progress_bar.setVisible(False) # Hide on error
            # self.rerun_btn.setEnabled(True)


    def _export_data(self):
        if not self.results_by_amplitude:
            QtWidgets.QMessageBox.warning(self, "No Data", "No multisweep data to export.")
            return

        dialog = QtWidgets.QFileDialog(self)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilters(["Pickle Files (*.pkl)", "All Files (*)"])
        dialog.setDefaultSuffix("pkl")

        if dialog.exec():
            filename = dialog.selectedFiles()[0]
            try:
                export_content = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'target_module': self.target_module,
                    'initial_parameters': self.initial_params,
                    'dac_scales_used': self.dac_scales, # Store DAC scales used by the window
                    'results_by_amplitude': self.results_by_amplitude
                }
                with open(filename, 'wb') as f:
                    pickle.dump(export_content, f)
                QtWidgets.QMessageBox.information(self, "Export Complete", f"Data exported to {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Export Error", f"Error exporting data: {str(e)}")

    def _rerun_multisweep(self):
        # This would typically involve getting resonance frequencies again if they might change,
        # or using the ones from initial_params. For now, assume initial_params has them.
        if not self.initial_params.get('resonance_frequencies'):
            QtWidgets.QMessageBox.warning(self, "Cannot Re-run", "Initial resonance frequencies not available for re-run.")
            return
        
        current_recalc_cf_state = self.initial_params.get('recalculate_center_frequencies', True)
        current_perform_fits_state = self.initial_params.get('perform_fits', True)

        dialog_params = self.initial_params.copy()
        dialog_params['recalculate_center_frequencies'] = current_recalc_cf_state
        dialog_params['perform_fits'] = current_perform_fits_state
        
        dialog = MultisweepDialog(
            parent=self,
            resonance_frequencies=self.initial_params['resonance_frequencies'],
            dac_scales=self.dac_scales,
            current_module=self.target_module,
            initial_params=dialog_params
        )

        if dialog.exec():
            new_params = dialog.get_parameters()
            if new_params:
                # Update initial_params for next potential re-run
                self.initial_params.update(new_params) 
                
                # Clear old results and UI state
                self.results_by_amplitude.clear()
                self._redraw_plots()
                self.progress_bar.setValue(0)
                self.progress_bar.setVisible(True)
                self.current_amp_label.setText("Current Amplitude: N/A")
                self.statusBar.showMessage("Starting new multisweep...")
                # self.rerun_btn.setEnabled(False)

                # Trigger the new sweep via the parent (Periscope main window)
                if hasattr(self.parent(), '_start_multisweep_analysis_for_window'):
                    self.parent()._start_multisweep_analysis_for_window(self, new_params)
                else:
                    QtWidgets.QMessageBox.warning(self, "Error", "Cannot trigger re-run. Parent linkage missing.")


    def closeEvent(self, event):
        # Signal the parent (Periscope main window) to stop the task associated with this window
        if hasattr(self.parent(), 'stop_multisweep_task_for_window'):
            self.parent().stop_multisweep_task_for_window(self)
        super().closeEvent(event)

    def _toggle_cf_lines_visibility(self, checked):
        """Toggle visibility of center frequency lines or redraw plots."""
        # Simplest is to just trigger a full redraw, which will respect the checkbox state
        self._redraw_plots()
