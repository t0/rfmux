"""Dialog and window classes for network analysis in Periscope."""

from .periscope_utils import *
from .periscope_tasks import *

class NetworkAnalysisDialogBase(QtWidgets.QDialog):
    """Base class for network analysis dialogs with shared functionality."""
    def __init__(self, parent=None, params=None, modules=None, dac_scales=None):
        super().__init__(parent)
        self.params = params or {}
        self.modules = modules or [1, 2, 3, 4]
        self.dac_scales = dac_scales or {m: None for m in self.modules}  # Default to None (unknown)
        self.currently_updating = False  # Flag to prevent circular updates
        
    def setup_amplitude_group(self, layout):
        """Setup the amplitude settings group with normalized and dBm inputs."""
        # Create the group box with an empty title (we'll add a title in the form layout)
        amp_group = QtWidgets.QGroupBox()
        amp_layout = QtWidgets.QFormLayout(amp_group)
        
        # Get amplitude values
        amps = self.params.get('amps', [self.params.get('amp', DEFAULT_AMPLITUDE)])
        amp_str = ','.join(str(a) for a in amps) if amps else str(DEFAULT_AMPLITUDE)
        
        # Normalized amplitude input
        self.amp_edit = QtWidgets.QLineEdit(amp_str)
        self.amp_edit.setToolTip("Enter a single value or comma-separated list (e.g., 0.001,0.01,0.1)")
        amp_layout.addRow("Normalized Amplitude:", self.amp_edit)
        
        # dBm input 
        self.dbm_edit = QtWidgets.QLineEdit()
        self.dbm_edit.setToolTip("Enter a single value or comma-separated list in dBm (e.g., -30,-20,-10)")
        amp_layout.addRow("Power (dBm):", self.dbm_edit)
        
        # DAC scale information
        self.dac_scale_info = QtWidgets.QLabel("Fetching DAC scales...")
        self.dac_scale_info.setWordWrap(True)
        amp_layout.addRow("DAC Scale (dBm):", self.dac_scale_info)
        
        # Connect signals for updating between normalized and dBm
        # Only update values during typing, but don't validate
        self.amp_edit.textChanged.connect(self._update_dbm_from_normalized_no_validate)
        self.dbm_edit.textChanged.connect(self._update_normalized_from_dbm_no_validate)
        
        # Add validation when editing is finished
        self.amp_edit.editingFinished.connect(self._validate_normalized_values)
        self.dbm_edit.editingFinished.connect(self._validate_dbm_values)
        
        # Add the group to the main layout with a title in the left column
        layout.addRow("Amplitude Settings:", amp_group)
        
        return amp_group
        
    def _update_dbm_from_normalized_no_validate(self):
        """Update dBm values based on normalized amplitude inputs, without validation warnings."""
        if self.currently_updating:
            return  # Prevent recursive updates
            
        # Skip if dBm field is disabled (unknown DAC scale)
        if not self.dbm_edit.isEnabled():
            return
            
        self.currently_updating = True
        try:
            amp_text = self.amp_edit.text().strip()
            if not amp_text:
                self.dbm_edit.setText("")
                return
                
            # Get DAC scale for selected modules
            dac_scale = self._get_selected_dac_scale()
            if dac_scale is None:
                # If scale becomes unknown, disable the field
                self.dbm_edit.setEnabled(False)
                self.dbm_edit.setToolTip("Unable to query current DAC scale - dBm input disabled")
                self.dbm_edit.clear()
                return
                
            # Parse normalized values
            normalized_values = self._parse_amplitude_values(amp_text)
            
            # Convert to dBm
            dbm_values = []
            for norm in normalized_values:
                dbm = UnitConverter.normalize_to_dbm(norm, dac_scale)
                dbm_values.append(f"{dbm:.2f}")
            
            # Update dBm field
            self.dbm_edit.setText(", ".join(dbm_values))
        finally:
            self.currently_updating = False

    def _update_normalized_from_dbm_no_validate(self):
        """Update normalized amplitude values based on dBm inputs, without validation warnings."""
        if self.currently_updating:
            return  # Prevent recursive updates
            
        # Skip if dBm field is disabled (unknown DAC scale)
        if not self.dbm_edit.isEnabled():
            return
            
        self.currently_updating = True
        try:
            dbm_text = self.dbm_edit.text().strip()
            if not dbm_text:
                self.amp_edit.setText("")
                return
                
            # Get DAC scale for selected modules
            dac_scale = self._get_selected_dac_scale()
            if dac_scale is None:
                # If scale becomes unknown, disable the field
                self.dbm_edit.setEnabled(False)
                self.dbm_edit.setToolTip("Unable to query current DAC scale - dBm input disabled")
                self.dbm_edit.clear()
                return
                
            # Parse dBm values
            dbm_values = self._parse_dbm_values(dbm_text)
            
            # Convert to normalized amplitude
            normalized_values = []
            for dbm in dbm_values:
                norm = UnitConverter.dbm_to_normalize(dbm, dac_scale)
                normalized_values.append(f"{norm:.6f}")
            
            # Update normalized field
            self.amp_edit.setText(", ".join(normalized_values))
        finally:
            self.currently_updating = False

    def _validate_normalized_values(self):
        """Validate normalized amplitude values when editing is finished."""
        amp_text = self.amp_edit.text().strip()
        if not amp_text:
            return
            
        dac_scale = self._get_selected_dac_scale()
        if dac_scale is None:
            return
            
        normalized_values = self._parse_amplitude_values(amp_text)
        
        warnings = []
        for norm in normalized_values:
            if norm > 1.0:
                warnings.append(f"Warning: Normalized amplitude {norm:.6f} > 1.0 (maximum)")
            elif norm < 1e-4:
                warnings.append(f"Warning: Normalized amplitude {norm:.6f} < 1e-4 (minimum recommended)")
        
        if warnings:
            self._show_warning_dialog("Amplitude Warning", warnings)

    def _validate_dbm_values(self):
        """Validate dBm values when editing is finished."""
        dbm_text = self.dbm_edit.text().strip()
        if not dbm_text:
            return
            
        dac_scale = self._get_selected_dac_scale()
        if dac_scale is None:
            return
            
        dbm_values = self._parse_dbm_values(dbm_text)
        
        warnings = []
        for dbm in dbm_values:
            if dbm > dac_scale:
                warnings.append(f"Warning: {dbm:.2f} dBm > {dac_scale:+.2f} dBm (DAC max)")
            
            norm = UnitConverter.dbm_to_normalize(dbm, dac_scale)
            if norm > 1.0:
                warnings.append(f"Warning: {dbm:.2f} dBm gives normalized amplitude > 1.0")
            elif norm < 1e-4:
                warnings.append(f"Warning: {dbm:.2f} dBm gives normalized amplitude < 1e-4")
        
        if warnings:
            self._show_warning_dialog("Amplitude Warning", warnings)

    def _show_warning_dialog(self, title, warnings):
        """Show a warning dialog with the given messages."""
        QtWidgets.QMessageBox.warning(self, title, "\n".join(warnings))
            
    def _parse_amplitude_values(self, amp_text):
        """Parse comma-separated amplitude values."""
        normalized_values = []
        for part in amp_text.split(','):
            part = part.strip()
            if part:
                try:
                    value = float(eval(part))
                    normalized_values.append(value)
                except (ValueError, SyntaxError, NameError):
                    continue
        return normalized_values
        
    def _parse_dbm_values(self, dbm_text):
        """Parse comma-separated dBm values."""
        dbm_values = []
        for part in dbm_text.split(','):
            part = part.strip()
            if part:
                try:
                    value = float(eval(part))
                    dbm_values.append(value)
                except (ValueError, SyntaxError, NameError):
                    continue
        return dbm_values

    def _update_dac_scale_info(self):
        """Update the DAC scale information label based on selected modules."""
        selected_modules = self._get_selected_modules()
        
        # Check if any selected module has a known DAC scale
        has_known_scale = False
        scales = []
        
        for m in selected_modules:
            dac_scale = self.dac_scales.get(m)
            if dac_scale is not None:
                has_known_scale = True
                scales.append(f"Module {m}: {dac_scale:+.2f} dBm")
            else:
                scales.append(f"Module {m}: Unknown")
        
        # Update the info text
        if not selected_modules:
            text = "Unknown (no modules selected)"
        else:
            text = "\n".join(scales)
        
        self.dac_scale_info.setText(text)
        
        # Enable/disable dBm input based on whether we have known scales
        if has_known_scale:
            self.dbm_edit.setEnabled(True)
            self.dbm_edit.setToolTip("Enter a single value or comma-separated list in dBm (e.g., -30,-20,-10)")
            self._update_dbm_from_normalized()  # Update conversion
        else:
            self.dbm_edit.setEnabled(False)
            self.dbm_edit.setToolTip("Unable to query current DAC scale - dBm input disabled")
            self.dbm_edit.clear()  # Clear any existing text
    
    def _get_selected_modules(self):
        """
        Get the list of currently selected modules.
        This method should be overridden by subclasses.
        """
        return []
        
    def _get_selected_dac_scale(self):
        """Get the DAC scale for the currently selected module(s)."""
        selected_modules = self._get_selected_modules()
        
        if not selected_modules:
            return None  # No modules selected
            
        # Look for the first module with a known scale
        for module in selected_modules:
            dac_scale = self.dac_scales.get(module)
            if dac_scale is not None:
                return dac_scale
                
        # If no selected module has a known scale, return None
        return None
    
    def _update_dbm_from_normalized(self):
        """Update dBm values based on normalized amplitude inputs."""
        if self.currently_updating:
            return  # Prevent recursive updates
            
        # Skip if dBm field is disabled (unknown DAC scale)
        if not self.dbm_edit.isEnabled():
            return
            
        self.currently_updating = True
        try:
            amp_text = self.amp_edit.text().strip()
            if not amp_text:
                self.dbm_edit.setText("")
                return
                
            # Get DAC scale for selected modules
            dac_scale = self._get_selected_dac_scale()
            if dac_scale is None:
                # If scale becomes unknown, disable the field
                self.dbm_edit.setEnabled(False)
                self.dbm_edit.setToolTip("Unable to query current DAC scale - dBm input disabled")
                self.dbm_edit.clear()
                return
                
            # Parse normalized values
            normalized_values = self._parse_amplitude_values(amp_text)
            
            # Convert to dBm
            dbm_values = []
            warnings = []
            
            for norm in normalized_values:
                if norm > 1.0:
                    warnings.append(f"Warning: Normalized amplitude {norm:.6f} > 1.0 (maximum)")
                elif norm < 1e-4:
                    warnings.append(f"Warning: Normalized amplitude {norm:.6f} < 1e-4 (minimum recommended)")
                    
                dbm = UnitConverter.normalize_to_dbm(norm, dac_scale)
                dbm_values.append(f"{dbm:.2f}")
            
            # Update dBm field
            self.dbm_edit.setText(", ".join(dbm_values))
            
            # Show warnings if any
            if warnings:
                self._show_warning_dialog("Amplitude Warning", warnings)
        finally:
            self.currently_updating = False
    
    def _update_normalized_from_dbm(self):
        """Update normalized amplitude values based on dBm inputs."""
        if self.currently_updating:
            return  # Prevent recursive updates
            
        # Skip if dBm field is disabled (unknown DAC scale)
        if not self.dbm_edit.isEnabled():
            return
            
        self.currently_updating = True
        try:
            dbm_text = self.dbm_edit.text().strip()
            if not dbm_text:
                self.amp_edit.setText("")
                return
                
            # Get DAC scale for selected modules
            dac_scale = self._get_selected_dac_scale()
            if dac_scale is None:
                # If scale becomes unknown, disable the field
                self.dbm_edit.setEnabled(False)
                self.dbm_edit.setToolTip("Unable to query current DAC scale - dBm input disabled")
                self.dbm_edit.clear()
                return
                
            # Parse dBm values
            dbm_values = self._parse_dbm_values(dbm_text)
            
            # Convert to normalized amplitude
            normalized_values = []
            warnings = []
            
            for dbm in dbm_values:
                if dbm > dac_scale:
                    warnings.append(f"Warning: {dbm:.2f} dBm > {dac_scale:.2f} dBm (DAC max)")
                
                norm = UnitConverter.dbm_to_normalize(dbm, dac_scale)
                
                if norm > 1.0:
                    warnings.append(f"Warning: {dbm:.2f} dBm gives normalized amplitude > 1.0")
                elif norm < 1e-4:
                    warnings.append(f"Warning: {dbm:.2f} dBm gives normalized amplitude < 1e-4")
                    
                normalized_values.append(f"{norm:.6f}")
            
            # Update normalized field
            self.amp_edit.setText(", ".join(normalized_values))
            
            # Show warnings if any
            if warnings:
                self._show_warning_dialog("Amplitude Warning", warnings)
        finally:
            self.currently_updating = False

class NetworkAnalysisDialog(NetworkAnalysisDialogBase):
    """Dialog for configuring network analysis parameters with dBm support."""
    def __init__(self, parent=None, modules=None, dac_scales=None):
        super().__init__(parent, None, modules, dac_scales)
        self.setWindowTitle("Network Analysis Configuration")
        self.setModal(False)
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the user interface for the dialog."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Parameters group
        param_group = QtWidgets.QGroupBox("Analysis Parameters")
        param_layout = QtWidgets.QFormLayout(param_group)
        
        # Module selection
        self.module_entry = QtWidgets.QLineEdit("All")
        self.module_entry.setToolTip("Enter module numbers (e.g., '1,2,5' or '1-4' or 'All')")
        self.module_entry.textChanged.connect(self._update_dac_scale_info)
        param_layout.addRow("Modules:", self.module_entry)
        
        # Frequency range (in MHz instead of Hz)
        self.fmin_edit = QtWidgets.QLineEdit(str(DEFAULT_MIN_FREQ / 1e6))
        self.fmax_edit = QtWidgets.QLineEdit(str(DEFAULT_MAX_FREQ / 1e6))
        param_layout.addRow("Min Frequency (MHz):", self.fmin_edit)
        param_layout.addRow("Max Frequency (MHz):", self.fmax_edit)

        # Cable length
        self.cable_length_edit = QtWidgets.QLineEdit(str(DEFAULT_CABLE_LENGTH))
        param_layout.addRow("Cable Length (m):", self.cable_length_edit)        
        
        # Add amplitude settings group
        self.setup_amplitude_group(param_layout)
        
        # Remainder of the original UI
        self.points_edit = QtWidgets.QLineEdit(str(DEFAULT_NPOINTS))
        param_layout.addRow("Number of Points:", self.points_edit)
        
        self.samples_edit = QtWidgets.QLineEdit(str(DEFAULT_NSAMPLES))
        param_layout.addRow("Samples to Average:", self.samples_edit)
        
        self.max_chans_edit = QtWidgets.QLineEdit(str(DEFAULT_MAX_CHANNELS))
        param_layout.addRow("Max Channels:", self.max_chans_edit)
        
        self.max_span_edit = QtWidgets.QLineEdit(str(DEFAULT_MAX_SPAN / 1e6))
        param_layout.addRow("Max Span (MHz):", self.max_span_edit)
        
        self.clear_channels_cb = QtWidgets.QCheckBox("Clear all channels first")
        self.clear_channels_cb.setChecked(True)
        param_layout.addRow("", self.clear_channels_cb)
        
        layout.addWidget(param_group)
        
        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start Analysis")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        
        # Connect buttons
        self.start_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        
        # Initialize dBm field based on default normalized value
        self._update_dbm_from_normalized()
        self.setMinimumSize(500, 600)  # Width, height in pixels
        
    def _get_selected_modules(self):
        """Parse module entry to determine which modules are selected."""
        module_text = self.module_entry.text().strip()
        selected_modules = []
        
        if module_text.lower() == 'all':
            selected_modules = list(range(1, 9))  # All modules
        else:
            # Parse comma-separated values and ranges
            for part in module_text.split(','):
                part = part.strip()
                if '-' in part:
                    try:
                        start, end = map(int, part.split('-'))
                        selected_modules.extend(range(start, end + 1))
                    except ValueError:
                        continue
                elif part:
                    try:
                        selected_modules.append(int(part))
                    except ValueError:
                        continue
        
        return selected_modules

    def get_parameters(self):
        """Get the configured parameters."""
        try:
            # Parse module entry
            module_text = self.module_entry.text().strip()
            selected_module = None
            if module_text.lower() != 'all':
                selected_modules = self._get_selected_modules()
                if selected_modules:
                    selected_module = selected_modules
            
            # Parse amplitude values
            amp_text = self.amp_edit.text().strip()
            amps = self._parse_amplitude_values(amp_text)
            if not amps:
                amps = [DEFAULT_AMPLITUDE]  # Default amplitude if none provided
            
            params = {
                'amps': amps,
                'module': selected_module,
                'fmin': float(eval(self.fmin_edit.text())) * 1e6,  # Convert MHz to Hz
                'fmax': float(eval(self.fmax_edit.text())) * 1e6,  # Convert MHz to Hz
                'cable_length': float(self.cable_length_edit.text()),
                'npoints': int(self.points_edit.text()),
                'nsamps': int(self.samples_edit.text()),
                'max_chans': int(self.max_chans_edit.text()),
                'max_span': float(eval(self.max_span_edit.text())) * 1e6,  # Convert MHz to Hz
                'clear_channels': self.clear_channels_cb.isChecked()
            }
            return params
        except Exception as e:
            traceback.print_exc()  # Print full stacktrace to console
            QtWidgets.QMessageBox.critical(self, "Error", f"Invalid parameter: {str(e)}")
            return None

class NetworkAnalysisParamsDialog(NetworkAnalysisDialogBase):
    """Dialog for editing network analysis parameters with dBm support."""
    def __init__(self, parent=None, params=None):
        super().__init__(parent, params)
        self.setWindowTitle("Edit Network Analysis Parameters")
        self.setModal(True)
        self._setup_ui()
        
        if parent and hasattr(parent, 'parent') and parent.parent() is not None:
            parent_main = parent.parent()
            if hasattr(parent_main, 'crs') and parent_main.crs is not None:
                self._fetch_dac_scales(parent_main.crs)
        
    def _fetch_dac_scales(self, crs):
        """Fetch DAC scales for all modules."""
        # Create a fetcher thread
        self.fetcher = DACScaleFetcher(crs)
        self.fetcher.dac_scales_ready.connect(self._on_dac_scales_ready)
        self.fetcher.start()
    
    def _on_dac_scales_ready(self, scales):
        """Handle fetched DAC scales."""
        self.dac_scales = scales
        self._update_dac_scale_info()
        self._update_dbm_from_normalized()
    
    def _setup_ui(self):
        """Set up the user interface for the dialog."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Parameters form
        form = QtWidgets.QFormLayout()
        
        # Frequency range (in MHz instead of Hz)
        fmin_mhz = str(self.params.get('fmin', DEFAULT_MIN_FREQ) / 1e6)
        fmax_mhz = str(self.params.get('fmax', DEFAULT_MAX_FREQ) / 1e6)
        self.fmin_edit = QtWidgets.QLineEdit(fmin_mhz)
        self.fmax_edit = QtWidgets.QLineEdit(fmax_mhz)
        form.addRow("Min Frequency (MHz):", self.fmin_edit)
        form.addRow("Max Frequency (MHz):", self.fmax_edit)
        
        # Add amplitude settings group
        self.setup_amplitude_group(form)
        
        # Number of points
        self.points_edit = QtWidgets.QLineEdit(str(self.params.get('npoints', DEFAULT_NPOINTS)))
        form.addRow("Number of Points:", self.points_edit)
        
        # Number of samples to average
        self.samples_edit = QtWidgets.QLineEdit(str(self.params.get('nsamps', DEFAULT_NSAMPLES)))
        form.addRow("Samples to Average:", self.samples_edit)
        
        # Max channels
        self.max_chans_edit = QtWidgets.QLineEdit(str(self.params.get('max_chans', DEFAULT_MAX_CHANNELS)))
        form.addRow("Max Channels:", self.max_chans_edit)
        
        # Max span
        max_span_mhz = str(self.params.get('max_span', DEFAULT_MAX_SPAN) / 1e6)
        self.max_span_edit = QtWidgets.QLineEdit(max_span_mhz)
        form.addRow("Max Span (MHz):", self.max_span_edit)
        
        # Add checkbox for clearing channels
        self.clear_channels_cb = QtWidgets.QCheckBox("Clear all channels first")
        self.clear_channels_cb.setChecked(self.params.get('clear_channels', True))
        form.addRow("", self.clear_channels_cb)
        
        layout.addLayout(form)
        
        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        
        # Connect buttons
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        
        # Initialize dBm field based on default normalized value
        self._update_dbm_from_normalized()

        self.setMinimumSize(500, 600)  # Width, height in pixels

    def _get_selected_modules(self):
        """Get selected modules from params."""
        # Get selected modules from params
        selected_module = self.params.get('module')
        if selected_module is None:
            # Using all modules
            return list(range(1, 9))
        elif isinstance(selected_module, list):
            # Multiple specific modules
            return selected_module
        else:
            # Single module
            return [selected_module]
    
    def get_parameters(self):
        """Get the updated parameters."""
        try:
            # Parse amplitude values
            amp_text = self.amp_edit.text().strip()
            amps = self._parse_amplitude_values(amp_text)
            if not amps:
                amps = [DEFAULT_AMPLITUDE]  # Default amplitude if none provided
            
            params = self.params.copy()
            params.update({
                'amps': amps,  # Store as list in 'amps'
                'amp': amps[0],  # Also store first value in 'amp' for backward compatibility
                'fmin': float(eval(self.fmin_edit.text())) * 1e6,  # Convert MHz to Hz
                'fmax': float(eval(self.fmax_edit.text())) * 1e6,  # Convert MHz to Hz
                'npoints': int(self.points_edit.text()),
                'nsamps': int(self.samples_edit.text()),
                'max_chans': int(self.max_chans_edit.text()),
                'max_span': float(eval(self.max_span_edit.text())) * 1e6,  # Convert MHz to Hz
                'clear_channels': self.clear_channels_cb.isChecked()
            })
            return params
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Invalid parameter: {str(e)}")
            return None

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
        """Set up the toolbar with controls."""
        toolbar = QtWidgets.QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Cable length control for quick adjustments
        self.cable_length_label = QtWidgets.QLabel("Cable Length (m):")
        self.cable_length_spin = QtWidgets.QDoubleSpinBox()
        self.cable_length_spin.setRange(0.0, 1000.0)
        self.cable_length_spin.setValue(DEFAULT_CABLE_LENGTH)
        self.cable_length_spin.setSingleStep(0.05)
        toolbar.addWidget(self.cable_length_label)
        toolbar.addWidget(self.cable_length_spin)

        # Add Unwrap Cable Delay button
        unwrap_button = QtWidgets.QPushButton("Unwrap Cable Delay")
        unwrap_button.setToolTip("Fit phase slope, calculate cable length, and apply compensation.")
        unwrap_button.clicked.connect(self._unwrap_cable_delay_action)
        toolbar.addWidget(unwrap_button)

        # Add edit parameters button to toolbar
        edit_params_btn = QtWidgets.QPushButton("Edit Other Parameters")
        edit_params_btn.clicked.connect(self._edit_parameters)
        toolbar.addWidget(edit_params_btn)

        # Re-run analysis button
        rerun_btn = QtWidgets.QPushButton("Re-run Analysis")
        rerun_btn.clicked.connect(self._rerun_analysis)
        toolbar.addWidget(rerun_btn)

        # Find Resonances button
        find_res_btn = QtWidgets.QPushButton("Find Resonances")
        find_res_btn.setToolTip("Identify resonance frequencies from the current sweep data.")
        find_res_btn.clicked.connect(self._show_find_resonances_dialog)
        toolbar.addWidget(find_res_btn)

        # Show/Hide resonances checkbox
        self.show_resonances_cb = QtWidgets.QCheckBox("Show Resonances (0)")
        self.show_resonances_cb.setChecked(True)
        self.show_resonances_cb.toggled.connect(self._toggle_resonances_visible)
        toolbar.addWidget(self.show_resonances_cb)

        # Add/Subtract mode and tolerance controls
        self.edit_resonances_cb = QtWidgets.QCheckBox("Add/Subtract Resonances")
        self.edit_resonances_cb.setToolTip(
            "When enabled, double-click adds a resonance;\n"
            "Shift + double-click removes the nearest resonance."
        )
        self.edit_resonances_cb.toggled.connect(self._toggle_resonance_edit_mode)
        toolbar.addWidget(self.edit_resonances_cb)


        # Export button
        export_btn = QtWidgets.QPushButton("Export Data")
        export_btn.clicked.connect(self._export_data)
        toolbar.addWidget(export_btn)
        
        # Add spacer to push the unit controls to the far right
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, 
                            QtWidgets.QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        
        # Normalize Magnitudes checkbox
        self.normalize_checkbox = QtWidgets.QCheckBox("Normalize Magnitudes")
        self.normalize_checkbox.setChecked(False)
        self.normalize_checkbox.setToolTip("Normalize all magnitude curves to their first data point")
        self.normalize_checkbox.toggled.connect(self._toggle_normalization)
        toolbar.addWidget(self.normalize_checkbox)

        # Add unit controls
        self._setup_unit_controls(toolbar)
        
        # Add zoom box mode checkbox
        self._setup_zoom_box_control(toolbar)

    def _setup_unit_controls(self, toolbar):
        """Set up the unit selection controls."""
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
        for module in self.plots:
            plot_info = self.plots[module]
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
        
        # Redraw plots with normalization applied
        self._redraw_all_plots()

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
        for module in self.plots:
            for line in self.plots[module]['resonance_lines_mag']:
                line.setVisible(checked)
            for line in self.plots[module]['resonance_lines_phase']:
                line.setVisible(checked)

    def _toggle_resonance_edit_mode(self, checked: bool):
        """Enable or disable double-click add/subtract mode."""
        self.add_subtract_mode = checked

    def _update_resonance_checkbox_text(self, module: int):
        """Update the show-resonances checkbox label with count."""
        count = len(self.resonance_freqs.get(module, []))
        self.show_resonances_cb.setText(f"Show Resonances ({count})")

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
        self._toggle_resonances_visible(self.show_resonances_cb.isChecked())

    def _remove_resonance(self, module: int, freq_hz: float):
        """Remove the nearest resonance line."""
        if module not in self.plots or module not in self.resonance_freqs:
            return
        freqs = self.resonance_freqs[module]
        if not freqs:
            return
        freqs_arr = np.array(freqs)
        idx = int(np.argmin(np.abs(freqs_arr - freq_hz)))
        # Remove lines
        line_mag = self.plots[module]['resonance_lines_mag'].pop(idx)
        line_phase = self.plots[module]['resonance_lines_phase'].pop(idx)
        self.plots[module]['amp_plot'].removeItem(line_mag)
        self.plots[module]['phase_plot'].removeItem(line_phase)
        freqs.pop(idx)
        self._update_resonance_checkbox_text(module)

    def _update_unit_mode(self, mode):
        """Update unit mode and redraw plots."""
        if mode != self.unit_mode:
            self.unit_mode = mode
            # Update all plot labels
            for module in self.plots:
                self._update_amplitude_labels(self.plots[module]['amp_plot'])
            
            # Redraw with new units
            self._redraw_all_plots()

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
        for module in self.raw_data:
            if module in self.plots:
                # Check if we have amplitude-specific curves
                has_amp_curves = len(self.plots[module]['amp_curves']) > 0
                
                # Update amplitude-specific curves first
                for amp_key, data_tuple in self.raw_data[module].items():
                    if amp_key != 'default':
                        # Extract amplitude and data
                        amplitude, freqs, amps, phases, iq_data = self._extract_data_from_tuple(amp_key, data_tuple)
                        
                        # Update the curve if it exists
                        if amplitude in self.plots[module]['amp_curves']:
                            converted_amps = UnitConverter.convert_amplitude(
                                amps, iq_data, self.unit_mode, normalize=self.normalize_magnitudes)
                            freq_ghz = freqs / 1e9
                            self.plots[module]['amp_curves'][amplitude].setData(freq_ghz, converted_amps)
                            self.plots[module]['phase_curves'][amplitude].setData(freq_ghz, phases)
                
                # Now handle the default curve - ONLY if there are no amplitude-specific curves
                if 'default' in self.raw_data[module] and not has_amp_curves:
                    freqs, amps, phases, iq_data = self.raw_data[module]['default']
                    converted_amps = UnitConverter.convert_amplitude(
                        amps, iq_data, self.unit_mode, normalize=self.normalize_magnitudes)
                    freq_ghz = freqs / 1e9
                    self.plots[module]['amp_curve'].setData(freq_ghz, converted_amps)
                    self.plots[module]['phase_curve'].setData(freq_ghz, phases)
                else:
                    # Make sure default curves have no data if we have amplitude-specific curves
                    self.plots[module]['amp_curve'].setData([], [])
                    self.plots[module]['phase_curve'].setData([], [])
                
                # Enable auto range to fit new data
                self.plots[module]['amp_plot'].autoRange()
        
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
            return

        # Extract data from the chosen sweep
        data_tuple = module_sweeps[target_sweep_key]
        
        if len(data_tuple) == 5: # New format with amplitude
            frequencies, _, _, iq_complex, _ = data_tuple
        elif len(data_tuple) == 4: # Old format or default
            frequencies, _, _, iq_complex = data_tuple
        else:
            QtWidgets.QMessageBox.critical(self, "Data Error", "Unexpected data format for the selected sweep.")
            return

        if len(frequencies) == 0 or len(iq_complex) == 0:
            QtWidgets.QMessageBox.information(self, "No Data", "Selected sweep has no frequency or IQ data.")
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
            return

        # Plot the results
        if active_module in self.plots:
            plot_info = self.plots[active_module]
            amp_plot_item = plot_info['amp_plot'].getPlotItem()
            phase_plot_item = plot_info['phase_plot'].getPlotItem()
            amp_legend = plot_info['amp_legend']
            phase_legend = plot_info['phase_legend']

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

            # 4. Store resonance frequencies and update checkbox
            self.resonance_freqs[active_module] = res_freqs_hz
            self._update_resonance_checkbox_text(active_module)
            self._toggle_resonances_visible(self.show_resonances_cb.isChecked())


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
        """Open dialog to edit parameters besides cable length."""
        dialog = NetworkAnalysisParamsDialog(self, self.current_params)
        if dialog.exec():
            # Get updated parameters
            params = dialog.get_parameters()
            if params:
                # Keep the current cable length
                params['cable_length'] = self.cable_length_spin.value()
                # Update current parameters
                self.current_params = params.copy()
                # Make progress group visible again
                if self.progress_group:
                    self.progress_group.setVisible(True)
                self.parent()._rerun_network_analysis(params)

    def _rerun_analysis(self):
        """Re-run the analysis with potentially updated parameters."""
        if hasattr(self.parent(), '_rerun_network_analysis'):
            # Get current parameters and update cable length
            params = self.current_params.copy()
            params['cable_length'] = self.cable_length_spin.value()
            # Make progress group visible again
            if self.progress_group:
                self.progress_group.setVisible(True)
            self.parent()._rerun_network_analysis(params)
    
    def set_params(self, params):
        """Set parameters for analysis."""
        self.original_params = params.copy()  # Keep original for reference
        self.current_params = params.copy()   # Keep current for dialog
        
        # Set cable length spinner to match value
        self.cable_length_spin.setValue(params.get('cable_length', DEFAULT_CABLE_LENGTH))
        
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
        self.cable_length_spin.setValue(L_new_physical)
        self.current_params['cable_length'] = L_new_physical
        
        QtWidgets.QMessageBox.information(self, "Cable Delay Updated", 
                                          f"Cable length updated to {L_new_physical:.3f} m based on phase fit.")

class InitializeCRSDialog(QtWidgets.QDialog):
    """Dialog for CRS initialization options."""
    def __init__(self, parent=None, crs_obj=None):
        super().__init__(parent)
        self.crs = crs_obj 
        self.setWindowTitle("Initialize CRS Board")
        self.setModal(True)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # IRIG Time Source Group
        irig_group = QtWidgets.QGroupBox("IRIG Time Source")
        irig_layout = QtWidgets.QVBoxLayout(irig_group)
        
        self.rb_backplane = QtWidgets.QRadioButton("BACKPLANE")
        self.rb_test = QtWidgets.QRadioButton("TEST")
        self.rb_sma = QtWidgets.QRadioButton("SMA")
        
        # Default selection (e.g., TEST)
        self.rb_test.setChecked(True) 
        
        irig_layout.addWidget(self.rb_backplane)
        irig_layout.addWidget(self.rb_test)
        irig_layout.addWidget(self.rb_sma)
        layout.addWidget(irig_group)
        
        # Clear Channels Checkbox
        self.cb_clear_channels = QtWidgets.QCheckBox("Clear all channels on this module")
        self.cb_clear_channels.setChecked(True) # Default to checked
        layout.addWidget(self.cb_clear_channels)
        
        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addStretch()
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(btn_layout)
        
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def get_selected_irig_source(self):
        if self.crs is None: # Should not happen if button is disabled correctly
            return None 
            
        if self.rb_backplane.isChecked():
            return self.crs.TIMESTAMP_PORT.BACKPLANE
        elif self.rb_test.isChecked():
            return self.crs.TIMESTAMP_PORT.TEST
        elif self.rb_sma.isChecked():
            return self.crs.TIMESTAMP_PORT.SMA
        return None # Should have one selected

    def get_clear_channels_state(self) -> bool:
        return self.cb_clear_channels.isChecked()

# ───────────────────────── Find Resonances Dialog ─────────────────────────
class FindResonancesDialog(QtWidgets.QDialog):
    """Dialog for configuring parameters for fitting.find_resonances."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find Resonances Parameters")
        self.setModal(True)
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QFormLayout(self)

        # expected_resonances (int or None)
        self.expected_resonances_edit = QtWidgets.QLineEdit()
        self.expected_resonances_edit.setPlaceholderText("Optional (e.g., 10)")
        # Validator for integer or empty string
        regex_int_or_empty = QRegularExpression("^$|^[1-9][0-9]*$") # Empty or positive integer
        self.expected_resonances_edit.setValidator(QRegularExpressionValidator(regex_int_or_empty, self))
        layout.addRow("Expected Resonances:", self.expected_resonances_edit)

        # min_dip_depth_db (float)
        self.min_dip_depth_db_edit = QtWidgets.QLineEdit(str(1.0)) # Default from find_resonances
        self.min_dip_depth_db_edit.setValidator(QDoubleValidator(0.01, 100.0, 2, self)) # Min, Max, Decimals
        layout.addRow("Min Dip Depth (dB):", self.min_dip_depth_db_edit)

        # min_Q (float)
        self.min_Q_edit = QtWidgets.QLineEdit(str(1e4)) # Default
        self.min_Q_edit.setValidator(QDoubleValidator(1.0, 1e9, 0, self))
        layout.addRow("Min Q:", self.min_Q_edit)

        # max_Q (float)
        self.max_Q_edit = QtWidgets.QLineEdit(str(1e7)) # Default
        self.max_Q_edit.setValidator(QDoubleValidator(1.0, 1e9, 0, self))
        layout.addRow("Max Q:", self.max_Q_edit)

        # min_resonance_separation_mhz (float, will be converted to Hz)
        self.min_resonance_separation_mhz_edit = QtWidgets.QLineEdit(str(0.1)) # Default 100e3 Hz = 0.1 MHz
        self.min_resonance_separation_mhz_edit.setValidator(QDoubleValidator(0.001, 1000.0, 3, self))
        layout.addRow("Min Separation (MHz):", self.min_resonance_separation_mhz_edit)
        
        # data_exponent (float)
        self.data_exponent_edit = QtWidgets.QLineEdit(str(2.0)) # Default
        self.data_exponent_edit.setValidator(QDoubleValidator(0.1, 10.0, 2, self))
        layout.addRow("Data Exponent:", self.data_exponent_edit)

        # Buttons
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addRow(self.button_box)

    def get_parameters(self) -> dict | None:
        params = {}
        try:
            # Expected Resonances (optional int)
            expected_text = self.expected_resonances_edit.text().strip()
            if expected_text:
                params['expected_resonances'] = int(expected_text)
            else:
                params['expected_resonances'] = None

            params['min_dip_depth_db'] = float(self.min_dip_depth_db_edit.text())
            params['min_Q'] = float(self.min_Q_edit.text())
            params['max_Q'] = float(self.max_Q_edit.text())
            
            min_sep_mhz = float(self.min_resonance_separation_mhz_edit.text())
            params['min_resonance_separation_hz'] = min_sep_mhz * 1e6 # Convert MHz to Hz
            
            params['data_exponent'] = float(self.data_exponent_edit.text())
            
            # Basic validation
            if params['min_Q'] >= params['max_Q']:
                QtWidgets.QMessageBox.warning(self, "Validation Error", "Min Q must be less than Max Q.")
                return None
            if params['min_dip_depth_db'] <=0:
                QtWidgets.QMessageBox.warning(self, "Validation Error", "Min Dip Depth must be positive.")
                return None

            return params
        except ValueError as e:
            QtWidgets.QMessageBox.critical(self, "Input Error", f"Invalid input: {str(e)}")
            return None

# ───────────────────────── Main Application ─────────────────────────
