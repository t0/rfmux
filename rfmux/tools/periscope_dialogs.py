"""Dialog classes for network analysis."""

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

# ───────────────────────── Multisweep Dialog ─────────────────────────
class MultisweepDialog(NetworkAnalysisDialogBase):
    """Dialog for configuring multisweep parameters."""
    def __init__(self, parent=None, resonance_frequencies: list[float] | None = None, dac_scales=None, current_module=None, initial_params=None):
        super().__init__(parent, params=initial_params, dac_scales=dac_scales) # Pass initial_params
        self.resonance_frequencies = resonance_frequencies or []
        self.current_module = current_module # Store the module this dialog is for

        self.setWindowTitle("Multisweep Configuration")
        self.setModal(True)
        self._setup_ui()

        # Fetch DAC scales if CRS is available from parent
        # The NetworkAnalysisDialogBase expects dac_scales to be populated for dBm conversion.
        # If parent is NetworkAnalysisWindow, it has a crs object.
        if parent and hasattr(parent, 'parent') and parent.parent() is not None:
            main_periscope_window = parent.parent() # This should be the Periscope main window instance
            if hasattr(main_periscope_window, 'crs') and main_periscope_window.crs is not None:
                # If dac_scales were not passed or are empty, fetch them.
                if not self.dac_scales and hasattr(self, '_fetch_dac_scales_for_dialog'):
                     self._fetch_dac_scales_for_dialog(main_periscope_window.crs)
                elif self.dac_scales: # If passed, just update UI
                    self._update_dac_scale_info()
                    self._update_dbm_from_normalized()


    def _fetch_dac_scales_for_dialog(self, crs_obj):
        """Helper to fetch DAC scales specifically for this dialog context."""
        self.fetcher = DACScaleFetcher(crs_obj) # DACScaleFetcher is in periscope_tasks
        self.fetcher.dac_scales_ready.connect(self._on_dac_scales_ready_dialog)
        self.fetcher.start()

    def _on_dac_scales_ready_dialog(self, scales):
        """Handle fetched DAC scales for the dialog."""
        self.dac_scales = scales
        self._update_dac_scale_info() # Update the label in the amplitude group
        self._update_dbm_from_normalized() # Update dBm field based on current normalized


    def _get_selected_modules(self):
        """Override to return the current module for DAC scale calculations."""
        if self.current_module is not None:
            return [self.current_module]
        return [] # Fallback, though current_module should always be set

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Resonance Info Group
        res_info_group = QtWidgets.QGroupBox("Target Resonances")
        res_info_layout = QtWidgets.QVBoxLayout(res_info_group)
        
        num_resonances = len(self.resonance_frequencies)
        res_label_text = f"Number of resonances to sweep: {num_resonances}"
        if num_resonances > 0:
            res_freq_mhz_str = ", ".join([f"{f / 1e6:.3f}" for f in self.resonance_frequencies[:5]])
            if num_resonances > 5:
                res_freq_mhz_str += ", ..."
            res_label_text += f"\nFrequencies (MHz): {res_freq_mhz_str}"
        
        self.resonances_info_label = QtWidgets.QLabel(res_label_text)
        self.resonances_info_label.setWordWrap(True)
        res_info_layout.addWidget(self.resonances_info_label)
        layout.addWidget(res_info_group)

        # Parameters Group
        param_group = QtWidgets.QGroupBox("Sweep Parameters")
        param_form_layout = QtWidgets.QFormLayout(param_group)

        # Use self.params (which is initial_params from super call) to set defaults
        default_span_khz = self.params.get('span_hz', 100000.0) / 1e3
        self.span_khz_edit = QtWidgets.QLineEdit(str(default_span_khz))
        self.span_khz_edit.setValidator(QDoubleValidator(0.1, 10000.0, 2, self)) # Min, Max, Decimals
        param_form_layout.addRow("Span per Resonance (kHz):", self.span_khz_edit)

        default_npoints = self.params.get('npoints_per_sweep', 101)
        self.npoints_edit = QtWidgets.QLineEdit(str(default_npoints))
        self.npoints_edit.setValidator(QIntValidator(2, 10000, self))
        param_form_layout.addRow("Number of Points per Sweep:", self.npoints_edit)
        
        default_nsamps = self.params.get('nsamps', DEFAULT_NSAMPLES)
        self.nsamps_edit = QtWidgets.QLineEdit(str(default_nsamps))
        self.nsamps_edit.setValidator(QIntValidator(1, 10000, self))
        param_form_layout.addRow("Samples to Average (nsamps):", self.nsamps_edit)

        # Add amplitude settings using the base class method
        # This will add "Normalized Amplitude", "Power (dBm)", and "DAC Scale (dBm)"
        # setup_amplitude_group already uses self.params to get 'amps'
        self.setup_amplitude_group(param_form_layout)

        default_perform_fits = self.params.get('perform_fits', True)
        self.perform_fits_cb = QtWidgets.QCheckBox("Perform fits after sweep")
        self.perform_fits_cb.setChecked(default_perform_fits)
        param_form_layout.addRow("", self.perform_fits_cb)
        
        layout.addWidget(param_group)

        # Buttons
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        # Initialize dBm field based on default normalized value from base class
        # This needs dac_scales to be populated.
        if self.dac_scales:
             self._update_dbm_from_normalized()
        
        self.setMinimumWidth(500)


    def get_parameters(self) -> dict | None:
        params = {}
        try:
            # Amplitudes from base class
            amp_text = self.amp_edit.text().strip()
            amps = self._parse_amplitude_values(amp_text)
            if not amps:
                # Use default from base class if not provided or invalid
                amps = [self.params.get('amp', DEFAULT_AMPLITUDE)]
            params['amps'] = amps

            params['span_hz'] = float(self.span_khz_edit.text()) * 1e3 # Convert kHz to Hz
            params['npoints_per_sweep'] = int(self.npoints_edit.text())
            params['nsamps'] = int(self.nsamps_edit.text())
            params['perform_fits'] = self.perform_fits_cb.isChecked()
            params['resonance_frequencies'] = self.resonance_frequencies # Pass along the target frequencies
            params['module'] = self.current_module # Pass the module

            if params['span_hz'] <= 0:
                QtWidgets.QMessageBox.warning(self, "Validation Error", "Span must be positive.")
                return None
            if params['npoints_per_sweep'] < 2:
                QtWidgets.QMessageBox.warning(self, "Validation Error", "Number of points must be at least 2.")
                return None
            if params['nsamps'] < 1:
                QtWidgets.QMessageBox.warning(self, "Validation Error", "Samples to average must be at least 1.")
                return None

            return params
        except ValueError as e:
            QtWidgets.QMessageBox.critical(self, "Input Error", f"Invalid input: {str(e)}")
            return None
        except Exception as e: # Catch any other unexpected errors during parsing
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not parse parameters: {str(e)}")
            return None
