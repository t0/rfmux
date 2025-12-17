"""Panel class for network analysis results (dockable)."""
import datetime # Added for MultisweepWindow export
import pickle   # Added for MultisweepWindow export
import os 
import csv

# Imports from within the 'periscope' subpackage
from .utils import *
from .tasks import SetCableLengthSignals # Added import
# from .tasks import * # Not directly used by this class, dialogs will import what they need.

# Dialogs are now imported from .dialogs within the same package
from .dialogs import NetworkAnalysisParamsDialog, FindResonancesDialog, MultisweepDialog
from .network_analysis_export import NetworkAnalysisExportMixin

class NetworkAnalysisPanel(QtWidgets.QWidget, NetworkAnalysisExportMixin, ScreenshotMixin):
    """
    Dockable panel for displaying network analysis results with real units support.

    This panel can be wrapped in a QDockWidget for tabbed/floating display within
    the main Periscope window. All functionality from the original NetworkAnalysisWindow
    is preserved.
    
    Signals:
        data_ready: Emitted when analysis completes with full export data dict
    """
    
    # Signal for session auto-export
    data_ready = QtCore.pyqtSignal(dict)  # Full export data dictionary
    
    def __init__(self, parent=None, modules=None, dac_scales=None, dark_mode=False, is_loaded_data=False):
        super().__init__(parent)
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
        self.dark_mode = dark_mode  # Store dark mode setting
        self.is_loaded_data = is_loaded_data  # Track if this is from loaded data
        
        # Track last session export filename for overwriting
        self._last_export_filename: Optional[str] = None

        # Initialize signals for SetCableLengthTask
        self.set_cable_length_signals = SetCableLengthSignals()
        # Optionally, connect these signals to handlers for user feedback
        self.set_cable_length_signals.error.connect(self._handle_set_cable_length_error)
        
        # Setup the UI components
        self._setup_ui()
        # Set initial size only on creation
        self.resize(1000, 800)

    def _setup_ui(self):
        """Set up the user interface for the panel."""
        # Create main layout for the panel (no central widget needed for QWidget)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # No margins for cleaner docking
        
        # Create toolbar
        self._setup_toolbar(layout)
        
        # Create progress bars
        self._setup_progress_bars(layout)
        
        # Create plot area
        self._setup_plot_area(layout)

    def _setup_toolbar(self, layout):
        """Set up the toolbars with controls."""
        # Toolbar 1: Global Controls (Top Row)
        # Use QWidget container instead of QToolBar for compatibility with QWidget base class
        toolbar_global = QtWidgets.QWidget()
        toolbar_global_layout = QtWidgets.QHBoxLayout(toolbar_global)
        toolbar_global_layout.setContentsMargins(5, 5, 5, 5)

        # Export button
        export_btn = QtWidgets.QPushButton("💾")
        export_btn.setToolTip("Export data")
        export_btn.clicked.connect(self._export_data)
        toolbar_global_layout.addWidget(export_btn)

        # Edit Other Parameters button (renamed to Re-run Analysis)
        edit_params_btn = QtWidgets.QPushButton("Re-run analysis")
        edit_params_btn.clicked.connect(self._edit_parameters)
        toolbar_global_layout.addWidget(edit_params_btn)

        # Show/Hide resonances checkbox
        self.show_resonances_cb = QtWidgets.QCheckBox("Show Resonances")
        self.show_resonances_cb.setChecked(True)
        self.show_resonances_cb.toggled.connect(self._toggle_resonances_visible)
        toolbar_global_layout.addWidget(self.show_resonances_cb)

        # Add/Subtract mode
        self.edit_resonances_cb = QtWidgets.QCheckBox("Add/Subtract Resonances")
        self.edit_resonances_cb.setToolTip(
            "When enabled, double-click adds a resonance;\n"
            "Shift + double-click removes the nearest resonance."
        )
        self.edit_resonances_cb.toggled.connect(self._toggle_resonance_edit_mode)
        toolbar_global_layout.addWidget(self.edit_resonances_cb)

        # Add spacer to push the unit controls to the far right
        toolbar_global_layout.addStretch(1)

        # Normalize Magnitudes checkbox
        self.normalize_checkbox = QtWidgets.QCheckBox("Normalize Magnitudes")
        self.normalize_checkbox.setChecked(False)
        self.normalize_checkbox.setToolTip("Normalize all magnitude curves to their first data point")
        self.normalize_checkbox.toggled.connect(self._toggle_normalization)
        toolbar_global_layout.addWidget(self.normalize_checkbox)


        # Add unit controls
        self._setup_unit_controls(toolbar_global_layout)

        # Add zoom box mode checkbox
        self._setup_zoom_box_control(toolbar_global_layout)

        # Screenshot button
        screenshot_btn = QtWidgets.QPushButton("📷")
        screenshot_btn.setToolTip("Export a screenshot of this panel to the session folder (or choose location)")
        screenshot_btn.clicked.connect(self._export_screenshot)
        toolbar_global_layout.addWidget(screenshot_btn)
        
        layout.addWidget(toolbar_global)

        # Toolbar 2: Module-Specific Controls (Bottom Row)
        toolbar_module = QtWidgets.QWidget()
        toolbar_module_layout = QtWidgets.QHBoxLayout(toolbar_module)
        toolbar_module_layout.setContentsMargins(5, 5, 5, 5)

        # Cable length control
        self.cable_length_label = QtWidgets.QLabel("Cable Length (m):")
        self.cable_length_spin = QtWidgets.QDoubleSpinBox()
        self.cable_length_spin.setRange(0.0, 1000.0)
        self.cable_length_spin.setValue(DEFAULT_CABLE_LENGTH)
        self.cable_length_spin.setSingleStep(0.05)
        self.cable_length_spin.valueChanged.connect(self._on_cable_length_changed)
        toolbar_module_layout.addWidget(self.cable_length_label)
        toolbar_module_layout.addWidget(self.cable_length_spin)

        # Unwrap Cable Delay button
        unwrap_button = QtWidgets.QPushButton("Unwrap Cable Delay")
        unwrap_button.setToolTip("Fit phase slope, calculate cable length, and apply compensation for the active module.")
        unwrap_button.clicked.connect(self._unwrap_cable_delay_action)
        toolbar_module_layout.addWidget(unwrap_button)

        # Find Resonances button
        find_res_btn = QtWidgets.QPushButton("Find Resonances")
        find_res_btn.setToolTip("Identify resonance frequencies from the current sweep data for the active module.")
        find_res_btn.clicked.connect(self._show_find_resonances_dialog)
        toolbar_module_layout.addWidget(find_res_btn)

        # Take Multisweep button
        self.take_multisweep_btn = QtWidgets.QPushButton("Take Multisweep")
        self.take_multisweep_btn.setToolTip("Perform a multisweep using identified resonance frequencies for the active module.")
        self.take_multisweep_btn.clicked.connect(self._show_multisweep_dialog)
        self.take_multisweep_btn.setEnabled(False) # Initially disabled
        toolbar_module_layout.addWidget(self.take_multisweep_btn)
        
        toolbar_module_layout.addStretch(1)
        
        layout.addWidget(toolbar_module)

    def _setup_unit_controls(self, toolbar_layout):
        """Set up the unit selection controls and add them to the specified toolbar layout."""
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
        
        # Connect signals
        self.rb_counts.toggled.connect(lambda: self._update_unit_mode("counts"))
        self.rb_dbm.toggled.connect(lambda: self._update_unit_mode("dbm"))
        self.rb_volts.toggled.connect(lambda: self._update_unit_mode("volts"))
        
        # Set fixed size policy to make alignment more predictable
        unit_group.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, 
                                QtWidgets.QSizePolicy.Policy.Preferred)
        
        # Add the unit controls to the layout
        toolbar_layout.addWidget(unit_group)
        
    def _setup_zoom_box_control(self, toolbar_layout):
        """Set up the zoom box mode control."""
        zoom_box_cb = QtWidgets.QCheckBox("Zoom Box Mode")
        zoom_box_cb.setChecked(self.zoom_box_mode)
        zoom_box_cb.setToolTip("When enabled, left-click drag creates a zoom box. When disabled, left-click drag pans.")
        zoom_box_cb.toggled.connect(self._toggle_zoom_box)
        
        # Store reference to the checkbox
        self.zoom_box_cb = zoom_box_cb
        
        toolbar_layout.addWidget(zoom_box_cb)

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

    def _hide_progress_bars(self):
        """Hide the entire Analysis Progress group."""
        if self.progress_group:
            self.progress_group.hide()

    def _show_progress_bars(self, reset=False):
        """Show the Analysis Progress group again.
           If reset=True, reset progress bars and labels to defaults.
        """
        if self.progress_group:
            self.progress_group.show()
            if reset:
                for module, pbar in self.progress_bars.items():
                    pbar.setValue(0)  # reset progress
                for module, label in self.progress_labels.items():
                    label.clear()     # clear amplitude text
    
    def _setup_plot_area(self, layout):
        """Set up the plot area with tabs for each module."""
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.currentChanged.connect(self._on_active_module_changed) # For Requirement 2
        layout.addWidget(self.tabs)
        
        self.plots = {}
        # Initialize amp_plot and phase_plot to None or a default PlotWidget
        # to ensure they are bound before setXLink is called.
        last_amp_plot: Optional[pg.PlotWidget] = None
        last_phase_plot: Optional[pg.PlotWidget] = None

        for module in self.modules:
            tab = QtWidgets.QWidget()
            tab_layout = QtWidgets.QVBoxLayout(tab)

            # Create amplitude and phase plots with ClickableViewBox
            vb_amp = ClickableViewBox()
            vb_amp.parent_window = self
            vb_amp.module_id = module
            vb_amp.plot_role = 'amp'
            amp_plot = pg.PlotWidget(viewBox=vb_amp, title=f"Module {module} - Magnitude")
            plot_item_amp = amp_plot.getPlotItem()
            if plot_item_amp:
                self._update_amplitude_labels(amp_plot) # amp_plot is PlotWidget, _update_amplitude_labels expects PlotWidget
                plot_item_amp.setLabel('bottom', 'Frequency', units='Hz')
                plot_item_amp.showGrid(x=True, y=True, alpha=0.3)

            vb_phase = ClickableViewBox()
            vb_phase.parent_window = self
            vb_phase.module_id = module
            vb_phase.plot_role = 'phase'
            phase_plot = pg.PlotWidget(viewBox=vb_phase, title=f"Module {module} - Phase")
            plot_item_phase = phase_plot.getPlotItem()
            if plot_item_phase:
                plot_item_phase.setLabel('left', 'Phase', units='deg')
                plot_item_phase.setLabel('bottom', 'Frequency', units='Hz')
                plot_item_phase.showGrid(x=True, y=True, alpha=0.3)
            
            # Add legends for multiple amplitude plots with proper text color
            bg_color, pen_color = ("k", "w") if self.dark_mode else ("w", "k")
            amp_legend = plot_item_amp.addLegend(offset=(30, 10), labelTextColor=pen_color) if plot_item_amp else None
            phase_legend = plot_item_phase.addLegend(offset=(30, 10), labelTextColor=pen_color) if plot_item_phase else None

            # Create curves with periscope color scheme - but don't add data yet
            amp_curve = plot_item_amp.plot([], [], pen=pg.mkPen(TABLEAU10_COLORS[1], width=LINE_WIDTH)) if plot_item_amp else None
            phase_curve = plot_item_phase.plot([], [], pen=pg.mkPen(TABLEAU10_COLORS[0], width=LINE_WIDTH)) if plot_item_phase else None

            tab_layout.addWidget(amp_plot)
            tab_layout.addWidget(phase_plot)
            self.tabs.addTab(tab, f"Module {module}")
            
            self.plots[module] = {
                'amp_plot': amp_plot, # amp_plot is PlotWidget here
                'phase_plot': phase_plot, # phase_plot is PlotWidget here
                'amp_curve': amp_curve,
                'phase_curve': phase_curve,
                'amp_legend': amp_legend,
                'phase_legend': phase_legend,
                'amp_curves': {},  # Will store multiple curves for different amplitudes
                'phase_curves': {},  # Will store multiple curves for different amplitudes
                'resonance_lines_mag': [], # For storing magnitude resonance lines
                'resonance_lines_phase': [] # For storing phase resonance lines
            }
            last_amp_plot = amp_plot
            last_phase_plot = phase_plot
            
        # Apply zoom box mode
        self._apply_zoom_box_mode()

        # Link the x-axis of the last created amplitude and phase plots for synchronized zooming
        if last_phase_plot and last_amp_plot:
            last_phase_plot.setXLink(last_amp_plot)
        
        # Apply initial theme based on dark_mode setting
        if last_amp_plot: self._apply_theme_to_plot(last_amp_plot)
        if last_phase_plot: self._apply_theme_to_plot(last_phase_plot)
        
    def _apply_theme_to_plot(self, plot_widget):
        """Apply the current theme to a specific plot widget."""
        bg_color, pen_color = ("k", "w") if self.dark_mode else ("w", "k")
        plot_widget.setBackground(bg_color)
        
        # Update plot title color using a more direct approach
        plot_item = plot_widget.getPlotItem()
        if plot_item:
            # Set the title explicitly with the color parameter
            title_text = plot_item.titleLabel.text if plot_item.titleLabel else ""
            plot_item.setTitle(title_text, color=pen_color)
            
        # Update axes colors
        for axis_name in ("left", "bottom", "right", "top"):
            ax = plot_widget.getPlotItem().getAxis(axis_name)
            if ax:
                ax.setPen(pen_color)
                ax.setTextPen(pen_color)

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
        line_pen = pg.mkPen(RESONANCE_LINE_COLOR, style=QtCore.Qt.PenStyle.DashLine)
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
            raise
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
        
        # Emit data_ready signal for session auto-export after finding resonances
        # Include filename override if this panel has exported before (to update existing file)
        export_data = self.build_export_dict()  # Use inherited method from mixin
        if self._last_export_filename:
            export_data['_filename_override'] = self._last_export_filename
        self.data_ready.emit(export_data)
        
    def _use_loaded_resonances(self, active_module: int, load_resonance_freqs: list):
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
            
            res_freqs_hz = load_resonance_freqs

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


    def _get_periscope_parent(self):
        """
        Get the Periscope parent instance by walking up the parent hierarchy.
        
        Returns:
            The Periscope instance or None if not found
        """
        return find_parent_with_attr(self, 'netanal_windows')
    
    def _check_all_complete(self):
        """
        Check if all progress bars are at 100% and hide the progress group 
        when all analyses are complete.
        """
        if not self.progress_group:
            return
        
        # Walk up parent hierarchy to find Periscope instance
        # (panel may be wrapped in QDockWidget, so parent() might not be Periscope directly)
        parent = self._get_periscope_parent()
        
        if not parent:
            return
            
        window_id = None
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
                
                params_for_rerun = updated_general_params.copy()
                params_for_rerun['module_cable_lengths'] = self.module_cable_lengths.copy()
                params_for_rerun.pop('cable_length', None)
                self.current_params = params_for_rerun.copy()
                
                if self.progress_group:
                    self.progress_group.setVisible(True)
                
                parent_widget = self._get_periscope_parent()
                if parent_widget and hasattr(parent_widget, '_rerun_network_analysis'):
                    parent_widget._rerun_network_analysis(self.current_params) # type: ignore

    def _rerun_analysis(self):
        """Re-run the analysis with potentially updated parameters."""
        parent_widget = self._get_periscope_parent()
        if parent_widget and hasattr(parent_widget, '_rerun_network_analysis'):
            params = self.current_params.copy()
            params['module_cable_lengths'] = self.module_cable_lengths.copy()
            params.pop('cable_length', None)

            if self.progress_group:
                self.progress_group.setVisible(True)
            parent_widget._rerun_network_analysis(params) # type: ignore
    
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
            if amplitude in amps_list:
                amp_index = amps_list.index(amplitude)
            else:
                # Fallback if amplitude is not in the predefined list (should ideally not happen if params are consistent)
                # Treat as a new, distinct amplitude for color indexing if it's an unexpected one.
                # This might happen if data is updated with an amplitude not in original_params.
                # For simplicity, let's use a modulo of existing curves if this case is hit,
                # or default to the first color if it's the very first one.
                amp_index = len(self.plots[module]['amp_curves']) % len(TABLEAU10_COLORS)

            num_amps_total = len(amps_list)
            color = None

            if num_amps_total <= 5:
                color = TABLEAU10_COLORS[amp_index % len(TABLEAU10_COLORS)]
            else:
                use_cmap = pg.colormap.get(COLORMAP_CHOICES["AMPLITUDE_SWEEP"])
                # Ensure amp_index is valid for normalization if num_amps_total is 1
                normalized_idx = amp_index / max(1, num_amps_total - 1) if num_amps_total > 1 else 0.0
                if self.dark_mode:
                    # For dark mode, map to [0.3, 1.0]
                    map_value = 0.3 + normalized_idx * 0.7
                else:
                    # For light mode, map to [0.0, 0.75]
                    map_value = normalized_idx * 0.75
                color = use_cmap.map(map_value) if use_cmap else TABLEAU10_COLORS[amp_index % len(TABLEAU10_COLORS)] # Fallback
            
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
        """Mark analysis as complete for a module and emit data_ready signal."""
        if module in self.progress_bars:
            self.progress_bars[module].setValue(100)
            self._check_all_complete()
            
        # Emit data_ready signal for session auto-export
        if self._all_modules_complete():
            export_data = self.build_export_dict()  # Use inherited method from mixin
            self.data_ready.emit(export_data)
    
    def _all_modules_complete(self) -> bool:
        """Check if all modules have completed analysis."""
        if not self.progress_bars:
            return False
        return all(pbar.value() == 100 for pbar in self.progress_bars.values())
            
    def apply_theme(self, dark_mode: bool):
        """Apply the dark/light theme to all plots in this window."""
        self.dark_mode = dark_mode
        
        # Apply theme to all plots
        for module in self.plots:
            plot_info = self.plots[module]
            bg_color, pen_color = ("k", "w") if dark_mode else ("w", "k")
            
            # Apply to amplitude plot
            amp_plot = plot_info['amp_plot']
            amp_plot.setBackground(bg_color)
            
            # Update plot title color - use more direct approach
            amp_plot_item = amp_plot.getPlotItem()
            if amp_plot_item and hasattr(amp_plot_item, 'titleLabel'):
                title_text = amp_plot_item.titleLabel.text if amp_plot_item.titleLabel.text else f"Module {module} - Magnitude"
                amp_plot_item.setTitle(title_text, color=pen_color)
                
            # Update axes colors
            for axis_name in ("left", "bottom", "right", "top"):
                ax = amp_plot_item.getAxis(axis_name) if amp_plot_item else None
                if ax: 
                    ax.setPen(pen_color)
                    ax.setTextPen(pen_color)
            
            # Update legend text color for amplitude plot using the proper API
            amp_legend = plot_info['amp_legend']
            if amp_legend:
                try:
                    amp_legend.setLabelTextColor(pen_color)
                    amp_legend.update()
                except Exception as e:
                    print(f"Error updating amp_legend colors: {e}")
            
            # Apply to phase plot
            phase_plot = plot_info['phase_plot']
            phase_plot.setBackground(bg_color)
            
            # Update plot title color - use more direct approach
            phase_plot_item = phase_plot.getPlotItem()
            if phase_plot_item and hasattr(phase_plot_item, 'titleLabel'):
                title_text = phase_plot_item.titleLabel.text if phase_plot_item.titleLabel.text else f"Module {module} - Phase"
                phase_plot_item.setTitle(title_text, color=pen_color)
                
            # Update axes colors
            for axis_name in ("left", "bottom", "right", "top"):
                ax = phase_plot_item.getAxis(axis_name) if phase_plot_item else None
                if ax:
                    ax.setPen(pen_color)
                    ax.setTextPen(pen_color)
            
            # Update legend text color for phase plot using the proper API
            phase_legend = plot_info['phase_legend']
            if phase_legend:
                try:
                    phase_legend.setLabelTextColor(pen_color)
                    # No need to call phase_legend.update() here, setLabelTextColor should suffice
                except Exception as e:
                    print(f"Error updating phase_legend colors: {e}")
            
            # Redraw plots to ensure all legend items are updated correctly
            self._redraw_all_plots()
            
            # Update resonance legend entry to apply new theme colors
            self._update_resonance_legend_entry(module)

    def _handle_set_cable_length_error(self, module_id: int, error_message: str):
        """Handles error during setting of cable length."""
        QtWidgets.QMessageBox.warning(self, "Set Cable Length Error", 
                                      f"Module {module_id}: {error_message}")
