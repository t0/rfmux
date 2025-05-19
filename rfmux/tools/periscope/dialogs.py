"""Dialog classes for Periscope."""

# Imports from within the 'periscope' subpackage
from .utils import (
    QtWidgets, QtCore, QRegularExpression, QRegularExpressionValidator,
    QDoubleValidator, QIntValidator,
    DEFAULT_AMPLITUDE, DEFAULT_MIN_FREQ, DEFAULT_MAX_FREQ, DEFAULT_CABLE_LENGTH,
    DEFAULT_NPOINTS, DEFAULT_NSAMPLES, DEFAULT_MAX_CHANNELS, DEFAULT_MAX_SPAN,
    UnitConverter, traceback # Added traceback
)
from .tasks import DACScaleFetcher # For fetching DAC scales

class NetworkAnalysisDialogBase(QtWidgets.QDialog):
    """
    Base class for network analysis dialogs, providing shared functionality
    for amplitude input (normalized and dBm), DAC scale handling, and
    parameter parsing.
    """
    def __init__(self, parent: QtWidgets.QWidget = None, params: dict = None,
                 modules: list[int] = None, dac_scales: dict[int, float] = None):
        """
        Initializes the base dialog.

        Args:
            parent: The parent widget.
            params: Dictionary of existing parameters to populate fields.
            modules: List of module numbers relevant to this dialog.
            dac_scales: Dictionary mapping module numbers to their DAC scales in dBm.
        """
        super().__init__(parent)
        self.params = params or {}  # Store initial parameters, default to empty dict
        self.modules = modules or [1, 2, 3, 4] # Default or passed-in modules
        # Initialize DAC scales for relevant modules, defaulting to None (unknown)
        self.dac_scales = dac_scales or {module_idx: None for module_idx in self.modules}
        self.currently_updating = False # Flag to prevent recursive updates between amp/dBm fields
        
    def setup_amplitude_group(self, layout: QtWidgets.QFormLayout) -> QtWidgets.QGroupBox:
        """
        Sets up the QGroupBox for amplitude settings (Normalized and dBm).

        This includes input fields for normalized amplitude and power in dBm,
        and a display for DAC scale information. Connections are made to
        synchronize and validate these fields.

        Args:
            layout: The QFormLayout to add the amplitude group to.

        Returns:
            The created QGroupBox containing amplitude settings.
        """
        amp_group = QtWidgets.QGroupBox("Amplitude Settings") # Group box title
        amp_layout = QtWidgets.QFormLayout(amp_group)
        
        # Determine initial amplitude: use 'amps' list if available, else 'amp', else default.
        amps_list = self.params.get('amps', [self.params.get('amp', DEFAULT_AMPLITUDE)])
        amp_str = ','.join(map(str, amps_list)) if amps_list else str(DEFAULT_AMPLITUDE)
        
        self.amp_edit = QtWidgets.QLineEdit(amp_str)
        self.amp_edit.setToolTip("Enter a single value or comma-separated list of normalized amplitudes (e.g., 0.001,0.01,0.1). Expressions like '1/1000' are allowed.")
        amp_layout.addRow("Normalized Amplitude:", self.amp_edit)
        
        self.dbm_edit = QtWidgets.QLineEdit()
        self.dbm_edit.setToolTip("Enter a single value or comma-separated list of power in dBm (e.g., -30,-20,-10). Expressions are allowed.")
        amp_layout.addRow("Power (dBm):", self.dbm_edit)
        
        self.dac_scale_info = QtWidgets.QLabel("Fetching DAC scales...")
        self.dac_scale_info.setWordWrap(True)
        amp_layout.addRow("DAC Scale (dBm):", self.dac_scale_info)
        
        # Connect signals for live updates and validation
        self.amp_edit.textChanged.connect(self._update_dbm_from_normalized_no_validate) # Live update dBm field
        self.dbm_edit.textChanged.connect(self._update_normalized_from_dbm_no_validate) # Live update normalized amp field
        self.amp_edit.editingFinished.connect(self._validate_normalized_values) # Validate on finishing edit
        self.dbm_edit.editingFinished.connect(self._validate_dbm_values)       # Validate on finishing edit
        
        layout.addRow("Amplitude Settings:", amp_group)
        return amp_group
        
    def _update_dbm_from_normalized_no_validate(self):
        """
        Updates the dBm field based on the normalized amplitude field during text input.
        This version does not trigger immediate pop-up validation warnings.
        It relies on a known DAC scale for conversion.
        """
        if self.currently_updating or not self.dbm_edit.isEnabled():
            return # Avoid recursion or updates if dBm field is disabled
        self.currently_updating = True
        try:
            amp_text = self.amp_edit.text().strip()
            if not amp_text:
                self.dbm_edit.setText("")
                return
            
            dac_scale = self._get_selected_dac_scale()
            if dac_scale is None:
                # Disable dBm editing if DAC scale is unknown
                self.dbm_edit.setEnabled(False)
                self.dbm_edit.setToolTip("Unable to query DAC scale for conversion.")
                self.dbm_edit.clear()
                return
            
            normalized_values = self._parse_amplitude_values(amp_text)
            dbm_values = [f"{UnitConverter.normalize_to_dbm(norm_val, dac_scale):.2f}" for norm_val in normalized_values]
            self.dbm_edit.setText(", ".join(dbm_values))
        finally:
            self.currently_updating = False

    def _update_normalized_from_dbm_no_validate(self):
        """
        Updates the normalized amplitude field based on the dBm field during text input.
        This version does not trigger immediate pop-up validation warnings.
        It relies on a known DAC scale for conversion.
        """
        if self.currently_updating or not self.dbm_edit.isEnabled(): # Check dbm_edit's enabled state as it's the source
            return # Avoid recursion or updates if dBm field is disabled (e.g. no DAC scale)
        self.currently_updating = True
        try:
            dbm_text = self.dbm_edit.text().strip()
            if not dbm_text:
                self.amp_edit.setText("")
                return

            dac_scale = self._get_selected_dac_scale()
            if dac_scale is None:
                # This case should ideally be handled by disabling dbm_edit,
                # but as a safeguard:
                self.amp_edit.setToolTip("Unable to query DAC scale for conversion.")
                # self.amp_edit.clear() # Don't clear amp_edit if dBm is being typed without DAC scale
                return

            dbm_values = self._parse_dbm_values(dbm_text)
            normalized_values = [f"{UnitConverter.dbm_to_normalize(dbm_val, dac_scale):.6f}" for dbm_val in dbm_values]
            self.amp_edit.setText(", ".join(normalized_values))
        finally:
            self.currently_updating = False

    def _validate_normalized_values(self):
        """
        Validates the entered normalized amplitude values after editing is finished.
        Shows a warning dialog if values are outside typical ranges (e.g., > 1.0 or < 1e-4).
        """
        amp_text = self.amp_edit.text().strip()
        if not amp_text:
            return

        # DAC scale is not strictly needed for validating normalized amplitude against 0-1 range,
        # but good to have for consistency if other checks were dac_scale dependent.
        # dac_scale = self._get_selected_dac_scale()
        # if dac_scale is None: return # Or proceed with partial validation

        warnings_list = []
        normalized_values = self._parse_amplitude_values(amp_text)
        for norm_val in normalized_values:
            if norm_val > 1.0:
                warnings_list.append(f"Warning: Normalized amplitude {norm_val:.6f} > 1.0 (maximum)")
            elif norm_val < 1e-4: # Arbitrary small value warning threshold
                warnings_list.append(f"Warning: Normalized amplitude {norm_val:.6f} < 1e-4 (minimum recommended)")
        
        if warnings_list:
            self._show_warning_dialog("Normalized Amplitude Warning", warnings_list)

    def _validate_dbm_values(self):
        """
        Validates the entered dBm values after editing is finished.
        Shows a warning dialog if values exceed DAC scale or result in problematic
        normalized amplitudes.
        """
        dbm_text = self.dbm_edit.text().strip()
        if not dbm_text:
            return

        dac_scale = self._get_selected_dac_scale()
        if dac_scale is None:
            # Cannot validate dBm against DAC scale if unknown
            self._show_warning_dialog("DAC Scale Unknown", ["Cannot validate dBm values without a known DAC scale."])
            return

        warnings_list = []
        dbm_values = self._parse_dbm_values(dbm_text)
        for dbm_val in dbm_values:
            if dbm_val > dac_scale:
                warnings_list.append(f"Warning: {dbm_val:.2f} dBm > {dac_scale:+.2f} dBm (DAC maximum)")
            
            # Check corresponding normalized amplitude
            norm_val = UnitConverter.dbm_to_normalize(dbm_val, dac_scale)
            if norm_val > 1.0:
                warnings_list.append(f"Warning: {dbm_val:.2f} dBm results in normalized amplitude > 1.0 ({norm_val:.6f})")
            elif norm_val < 1e-4: # Arbitrary small value warning threshold
                warnings_list.append(f"Warning: {dbm_val:.2f} dBm results in normalized amplitude < 1e-4 ({norm_val:.6f})")
        
        if warnings_list:
            self._show_warning_dialog("dBm Amplitude Warning", warnings_list)

    def _show_warning_dialog(self, title: str, warnings_list: list[str]):
        """Displays a warning message box with a list of warnings."""
        QtWidgets.QMessageBox.warning(self, title, "\n".join(warnings_list))
            
    def _parse_amplitude_values(self, amp_text: str) -> list[float]:
        """
        Parses a comma-separated string of amplitude values.
        Each part can be an expression evaluatable by `eval()`.
        Invalid parts are silently skipped.

        Args:
            amp_text: The string containing amplitude values.

        Returns:
            A list of parsed float values.
        """
        values = []
        for part in amp_text.split(','):
            part = part.strip()
            if part:
                try:
                    # Using eval allows for simple expressions like "1/1000".
                    # Caution: eval can execute arbitrary code if input is not controlled.
                    # In this GUI context, user inputs values for their own use.
                    values.append(float(eval(part)))
                except (ValueError, SyntaxError, NameError, TypeError): # Added TypeError
                    # Silently skip parts that cannot be evaluated to a float
                    continue
        return values
        
    def _parse_dbm_values(self, dbm_text: str) -> list[float]:
        """
        Parses a comma-separated string of dBm values.
        Each part can be an expression evaluatable by `eval()`.
        Invalid parts are silently skipped.

        Args:
            dbm_text: The string containing dBm values.

        Returns:
            A list of parsed float values.
        """
        values = []
        for part in dbm_text.split(','):
            part = part.strip()
            if part:
                try:
                    # Using eval allows for simple expressions.
                    # Caution: eval can execute arbitrary code if input is not controlled.
                    values.append(float(eval(part)))
                except (ValueError, SyntaxError, NameError, TypeError): # Added TypeError
                    # Silently skip parts that cannot be evaluated to a float
                    continue
        return values

    def _update_dac_scale_info(self):
        """
        Updates the DAC scale information label based on selected modules
        and their known DAC scales. Also enables/disables the dBm input field
        accordingly and triggers an update of dBm from normalized amplitude.
        """
        selected_modules = self._get_selected_modules()
        has_known_scale = False
        scales_text_list = []

        for module_idx in selected_modules:
            dac_scale = self.dac_scales.get(module_idx)
            if dac_scale is not None:
                has_known_scale = True
                scales_text_list.append(f"Module {module_idx}: {dac_scale:+.2f} dBm")
            else:
                scales_text_list.append(f"Module {module_idx}: Unknown")
        
        text_to_display = "\n".join(scales_text_list) if selected_modules else "Unknown (no modules selected)"
        self.dac_scale_info.setText(text_to_display)
        
        if has_known_scale:
            self.dbm_edit.setEnabled(True)
            self.dbm_edit.setToolTip("Enter dBm values (e.g., -30,-20,-10). Expressions are allowed.")
            self._update_dbm_from_normalized() # Update dBm field now that scale might be known/changed
        else:
            self.dbm_edit.setEnabled(False)
            self.dbm_edit.setToolTip("DAC scale unknown for selected module(s) - dBm input disabled.")
            self.dbm_edit.clear()
    
    def _get_selected_modules(self) -> list[int]:
        """
        Placeholder method to get the list of currently selected modules.
        Subclasses must override this to provide actual module selection logic.

        Returns:
            An empty list. Subclasses should return a list of integer module IDs.
        """
        # This method must be implemented by subclasses
        return [] 
        
    def _get_selected_dac_scale(self) -> float | None:
        """
        Retrieves the DAC scale for the currently selected module(s).
        If multiple modules are selected, it returns the DAC scale of the first
        module in the selection that has a known DAC scale.

        Returns:
            The DAC scale in dBm as a float, or None if no scale is known
            for any selected module or if no modules are selected.
        """
        selected_modules = self._get_selected_modules()
        if not selected_modules:
            return None
        
        for module_idx in selected_modules:
            dac_scale = self.dac_scales.get(module_idx)
            if dac_scale is not None:
                return dac_scale # Return the first known DAC scale
        return None # No known DAC scale for any of the selected modules
    
    def _update_dbm_from_normalized(self):
        """
        Updates the dBm field based on the normalized amplitude field.
        This version is typically called when DAC scales change or UI is initialized.
        It performs validation and may show a warning dialog if values are problematic
        and the normalized amplitude field does not have focus.
        """
        if self.currently_updating or not self.dbm_edit.isEnabled():
            return
        self.currently_updating = True
        try:
            amp_text = self.amp_edit.text().strip()
            warnings_list = []
            if not amp_text:
                self.dbm_edit.setText("")
                return

            dac_scale = self._get_selected_dac_scale()
            if dac_scale is None:
                self.dbm_edit.setEnabled(False)
                self.dbm_edit.setToolTip("Unable to query DAC scale for conversion.")
                self.dbm_edit.clear()
                return

            normalized_values = self._parse_amplitude_values(amp_text)
            dbm_values = []
            for norm_val in normalized_values:
                # Validate normalized amplitude
                if norm_val > 1.0:
                    warnings_list.append(f"Warning: Normalized amplitude {norm_val:.6f} > 1.0 (maximum)")
                elif norm_val < 1e-4:
                    warnings_list.append(f"Warning: Normalized amplitude {norm_val:.6f} < 1e-4 (minimum recommended)")
                dbm_values.append(f"{UnitConverter.normalize_to_dbm(norm_val, dac_scale):.2f}")
            
            self.dbm_edit.setText(", ".join(dbm_values))
            
            # Show warnings only if the source field (amp_edit) is not currently being edited,
            # to avoid interrupting user input.
            if warnings_list and not self.amp_edit.hasFocus():
                self._show_warning_dialog("Normalized Amplitude Warning", warnings_list)
        finally:
            self.currently_updating = False
    
    def _update_normalized_from_dbm(self):
        """
        Updates the normalized amplitude field based on the dBm field.
        This version is typically called when DAC scales change or UI is initialized.
        It performs validation and may show a warning dialog if values are problematic
        and the dBm field does not have focus.
        """
        if self.currently_updating or not self.dbm_edit.isEnabled():
            return
        self.currently_updating = True
        try:
            dbm_text = self.dbm_edit.text().strip()
            warnings_list = []
            if not dbm_text:
                self.amp_edit.setText("")
                return

            dac_scale = self._get_selected_dac_scale()
            if dac_scale is None:
                # This state should ideally be prevented by disabling dbm_edit.
                # If somehow reached, cannot convert.
                self.amp_edit.setToolTip("Unable to query DAC scale for conversion.")
                # self.amp_edit.clear() # Avoid clearing if user is typing in dBm field
                return

            dbm_values = self._parse_dbm_values(dbm_text)
            normalized_values = []
            for dbm_val in dbm_values:
                # Validate dBm value against DAC scale
                if dbm_val > dac_scale:
                    warnings_list.append(f"Warning: {dbm_val:.2f} dBm > {dac_scale:+.2f} dBm (DAC maximum)")
                
                norm_val = UnitConverter.dbm_to_normalize(dbm_val, dac_scale)
                # Validate resulting normalized amplitude
                if norm_val > 1.0:
                    warnings_list.append(f"Warning: {dbm_val:.2f} dBm results in normalized amplitude > 1.0 ({norm_val:.6f})")
                elif norm_val < 1e-4:
                    warnings_list.append(f"Warning: {dbm_val:.2f} dBm results in normalized amplitude < 1e-4 ({norm_val:.6f})")
                normalized_values.append(f"{norm_val:.6f}")
            
            self.amp_edit.setText(", ".join(normalized_values))

            # Show warnings only if the source field (dbm_edit) is not currently being edited.
            if warnings_list and not self.dbm_edit.hasFocus():
                self._show_warning_dialog("dBm Amplitude Warning", warnings_list)
        finally:
            self.currently_updating = False

class NetworkAnalysisDialog(NetworkAnalysisDialogBase):
    """
    Dialog for configuring and initiating a new Network Analysis.
    It allows users to specify parameters like frequency range, amplitude,
    modules to scan, and other analysis settings.
    """
    def __init__(self, parent: QtWidgets.QWidget = None, modules: list[int] = None,
                 dac_scales: dict[int, float] = None):
        """
        Initializes the Network Analysis configuration dialog.

        Args:
            parent: The parent widget.
            modules: List of available module numbers.
            dac_scales: Pre-fetched DAC scales for the modules.
        """
        super().__init__(parent, params=None, modules=modules, dac_scales=dac_scales)
        self.setWindowTitle("Network Analysis Configuration")
        self.setModal(False) # Modeless dialog
        self._setup_ui()
        
    def _setup_ui(self):
        """Sets up the user interface elements for the dialog."""
        layout = QtWidgets.QVBoxLayout(self)
        
        param_group = QtWidgets.QGroupBox("Analysis Parameters")
        param_layout = QtWidgets.QFormLayout(param_group)
        
        self.module_entry = QtWidgets.QLineEdit("All")
        self.module_entry.setToolTip("Specify modules to analyze (e.g., '1,2,5', '1-4', or 'All' for modules 1-8).")
        self.module_entry.textChanged.connect(self._update_dac_scale_info) # Update DAC info when module selection changes
        param_layout.addRow("Modules:", self.module_entry)
        
        self.fmin_edit = QtWidgets.QLineEdit(str(DEFAULT_MIN_FREQ / 1e6)) # Default in MHz
        self.fmax_edit = QtWidgets.QLineEdit(str(DEFAULT_MAX_FREQ / 1e6)) # Default in MHz
        param_layout.addRow("Min Frequency (MHz):", self.fmin_edit)
        param_layout.addRow("Max Frequency (MHz):", self.fmax_edit)

        self.cable_length_edit = QtWidgets.QLineEdit(str(DEFAULT_CABLE_LENGTH)) # Default cable length
        param_layout.addRow("Cable Length (m):", self.cable_length_edit)
        
        self.setup_amplitude_group(param_layout) # Add shared amplitude settings
        
        self.points_edit = QtWidgets.QLineEdit(str(DEFAULT_NPOINTS))
        param_layout.addRow("Number of Points:", self.points_edit)
        
        self.samples_edit = QtWidgets.QLineEdit(str(DEFAULT_NSAMPLES))
        param_layout.addRow("Samples to Average:", self.samples_edit)
        
        self.max_chans_edit = QtWidgets.QLineEdit(str(DEFAULT_MAX_CHANNELS))
        param_layout.addRow("Max Channels:", self.max_chans_edit)
        
        self.max_span_edit = QtWidgets.QLineEdit(str(DEFAULT_MAX_SPAN / 1e6)) # Default in MHz
        param_layout.addRow("Max Span (MHz):", self.max_span_edit)
        
        self.clear_channels_cb = QtWidgets.QCheckBox("Clear all channels first")
        self.clear_channels_cb.setChecked(True) # Default to clearing channels
        param_layout.addRow("", self.clear_channels_cb)
        
        layout.addWidget(param_group)
        
        # Buttons for starting or canceling the analysis
        btn_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start Analysis")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        
        self.start_btn.clicked.connect(self.accept) # Connect to QDialog's accept slot
        self.cancel_btn.clicked.connect(self.reject) # Connect to QDialog's reject slot
        
        self._update_dbm_from_normalized() # Initial update of dBm field based on default amplitude
        self.setMinimumSize(500, 600) # Set a reasonable minimum size
        
    def _get_selected_modules(self) -> list[int]:
        """
        Parses the module entry text to determine the list of selected modules.
        Supports "All", comma-separated values, and ranges (e.g., "1-4").

        Returns:
            A list of integer module IDs. Returns modules 1-8 if "All" is specified.
        """
        module_text = self.module_entry.text().strip()
        selected_modules = []
        if module_text.lower() == 'all':
            selected_modules = list(range(1, 9)) # "All" implies modules 1 through 8
        else:
            for part in module_text.split(','):
                part = part.strip()
                if '-' in part: # Handle ranges like "1-4"
                    try:
                        start, end = map(int, part.split('-'))
                        if start <= end: # Ensure valid range
                            selected_modules.extend(range(start, end + 1))
                    except ValueError:
                        continue # Skip malformed range parts
                elif part: # Handle single numbers
                    try:
                        selected_modules.append(int(part))
                    except ValueError:
                        continue # Skip non-integer parts
        # Remove duplicates and sort, though current logic might not produce duplicates if input is clean.
        return sorted(list(set(selected_modules)))

    def get_parameters(self) -> dict | None:
        """
        Retrieves and validates the network analysis parameters from the UI fields.

        Returns:
            A dictionary of parameters if valid, otherwise None.
            Shows an error message on invalid input.
        """
        try:
            module_text = self.module_entry.text().strip()
            selected_module_param = None # Parameter for 'module' key
            if module_text.lower() != 'all':
                parsed_modules = self._get_selected_modules()
                if parsed_modules:
                    selected_module_param = parsed_modules
            
            amp_text = self.amp_edit.text().strip()
            # Use parsed amplitude values, or default if input is empty
            amps_list = self._parse_amplitude_values(amp_text) or [DEFAULT_AMPLITUDE]
            
            # Construct parameters dictionary
            # Using eval for frequency and span allows expressions, but ensure inputs are numbers.
            # Consider replacing eval with direct float conversion if expressions aren't strictly needed.
            params_dict = {
                'amps': amps_list,
                'module': selected_module_param, # Can be None (interpreted as all by backend) or list of modules
                'fmin': float(eval(self.fmin_edit.text())) * 1e6,  # Convert MHz to Hz
                'fmax': float(eval(self.fmax_edit.text())) * 1e6,  # Convert MHz to Hz
                'cable_length': float(self.cable_length_edit.text()),
                'npoints': int(self.points_edit.text()),
                'nsamps': int(self.samples_edit.text()),
                'max_chans': int(self.max_chans_edit.text()),
                'max_span': float(eval(self.max_span_edit.text())) * 1e6, # Convert MHz to Hz
                'clear_channels': self.clear_channels_cb.isChecked()
            }
            # Basic validation for frequency range
            if params_dict['fmin'] >= params_dict['fmax']:
                QtWidgets.QMessageBox.warning(self, "Input Error", "Min Frequency must be less than Max Frequency.")
                return None
            return params_dict
        except Exception as e:
            traceback.print_exc() # Log the full traceback for debugging
            QtWidgets.QMessageBox.critical(self, "Error Parsing Parameters", f"Invalid parameter input: {str(e)}")
            return None

class NetworkAnalysisParamsDialog(NetworkAnalysisDialogBase):
    """
    Dialog for editing existing Network Analysis parameters.
    This dialog is typically modal and pre-filled with current analysis parameters.
    It fetches DAC scales asynchronously if a CRS object is available from its parent.
    """
    def __init__(self, parent: QtWidgets.QWidget = None, params: dict = None):
        """
        Initializes the dialog for editing network analysis parameters.

        Args:
            parent: The parent widget.
            params: Dictionary of existing parameters to populate the fields.
        """
        super().__init__(parent, params=params) # Pass params to base class
        self.setWindowTitle("Edit Network Analysis Parameters")
        self.setModal(True) # Modal dialog
        self._setup_ui()

        # Attempt to fetch DAC scales if CRS is available from the main window hierarchy
        if parent and hasattr(parent, 'parent') and parent.parent() is not None:
            # Assuming parent.parent() is the main Periscope window
            main_periscope_window = parent.parent() 
            if hasattr(main_periscope_window, 'crs') and main_periscope_window.crs is not None:
                self._fetch_dac_scales(main_periscope_window.crs)
        
    def _fetch_dac_scales(self, crs_obj):
        """
        Initiates asynchronous fetching of DAC scales using DACScaleFetcher.

        Args:
            crs_obj: The CRS (Control and Readout System) object to query for scales.
        """
        self.fetcher = DACScaleFetcher(crs_obj)
        self.fetcher.dac_scales_ready.connect(self._on_dac_scales_ready)
        self.fetcher.start() # Start the QThread for fetching
    
    @QtCore.pyqtSlot(dict)
    def _on_dac_scales_ready(self, scales_dict: dict[int, float]):
        """
        Slot to handle the reception of fetched DAC scales.
        Updates the internal DAC scales and refreshes relevant UI elements.

        Args:
            scales_dict: Dictionary mapping module ID to DAC scale (dBm).
        """
        self.dac_scales = scales_dict
        self._update_dac_scale_info() # Update the DAC scale display label
        self._update_dbm_from_normalized() # Recalculate dBm based on new scales
    
    def _setup_ui(self):
        """Sets up the user interface elements for the dialog."""
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        
        # Populate fields with existing parameters or defaults
        fmin_mhz = str(self.params.get('fmin', DEFAULT_MIN_FREQ) / 1e6)
        fmax_mhz = str(self.params.get('fmax', DEFAULT_MAX_FREQ) / 1e6)
        self.fmin_edit = QtWidgets.QLineEdit(fmin_mhz)
        self.fmax_edit = QtWidgets.QLineEdit(fmax_mhz)
        form.addRow("Min Frequency (MHz):", self.fmin_edit)
        form.addRow("Max Frequency (MHz):", self.fmax_edit)
        
        self.setup_amplitude_group(form) # Add shared amplitude settings
        
        self.points_edit = QtWidgets.QLineEdit(str(self.params.get('npoints', DEFAULT_NPOINTS)))
        form.addRow("Number of Points:", self.points_edit)
        
        self.samples_edit = QtWidgets.QLineEdit(str(self.params.get('nsamps', DEFAULT_NSAMPLES)))
        form.addRow("Samples to Average:", self.samples_edit)
        
        self.max_chans_edit = QtWidgets.QLineEdit(str(self.params.get('max_chans', DEFAULT_MAX_CHANNELS)))
        form.addRow("Max Channels:", self.max_chans_edit)
        
        max_span_mhz = str(self.params.get('max_span', DEFAULT_MAX_SPAN) / 1e6)
        self.max_span_edit = QtWidgets.QLineEdit(max_span_mhz)
        form.addRow("Max Span (MHz):", self.max_span_edit)
        
        self.clear_channels_cb = QtWidgets.QCheckBox("Clear all channels first")
        self.clear_channels_cb.setChecked(self.params.get('clear_channels', True))
        form.addRow("", self.clear_channels_cb)
        
        layout.addLayout(form)
        
        # OK and Cancel buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        
        # Initial update of dBm field, especially if DAC scales were passed in constructor
        # or if _fetch_dac_scales is not called (e.g., no CRS object).
        self._update_dac_scale_info() # Call this first to set up dac_scale_info label correctly
        self._update_dbm_from_normalized() 
        self.setMinimumSize(500, 600)

    def _get_selected_modules(self) -> list[int]:
        """
        Determines the modules relevant for DAC scale display in this dialog.
        Uses the 'module' parameter passed during initialization. If 'module'
        is None or not specified, it defaults to all modules (1-8).

        Returns:
            A list of integer module IDs.
        """
        selected_module_param = self.params.get('module')
        if selected_module_param is None:
            # If no specific module(s) defined in params, assume all for DAC display purposes
            return list(range(1, 9)) 
        # Ensure it's a list, even if a single int was passed in params
        return selected_module_param if isinstance(selected_module_param, list) else [selected_module_param]
    
    def get_parameters(self) -> dict | None:
        """
        Retrieves and validates the edited network analysis parameters.

        Returns:
            A dictionary of parameters if valid, otherwise None.
            Shows an error message on invalid input.
        """
        try:
            amp_text = self.amp_edit.text().strip()
            amps_list = self._parse_amplitude_values(amp_text) or [self.params.get('amp', DEFAULT_AMPLITUDE)]
            
            # Start with a copy of existing params and update with UI values
            params_dict = self.params.copy() 
            params_dict.update({
                'amps': amps_list,
                'amp': amps_list[0] if amps_list else DEFAULT_AMPLITUDE, # Update single 'amp' for compatibility
                'fmin': float(eval(self.fmin_edit.text())) * 1e6,
                'fmax': float(eval(self.fmax_edit.text())) * 1e6,
                'npoints': int(self.points_edit.text()),
                'nsamps': int(self.samples_edit.text()),
                'max_chans': int(self.max_chans_edit.text()),
                'max_span': float(eval(self.max_span_edit.text())) * 1e6,
                'clear_channels': self.clear_channels_cb.isChecked()
            })
            # Basic validation for frequency range
            if params_dict['fmin'] >= params_dict['fmax']:
                QtWidgets.QMessageBox.warning(self, "Input Error", "Min Frequency must be less than Max Frequency.")
                return None
            return params_dict
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error Parsing Parameters", f"Invalid parameter input: {str(e)}")
            return None

class InitializeCRSDialog(QtWidgets.QDialog):
    """
    Dialog for initializing a CRS (Control and Readout System) board.
    Allows selection of the IRIG time source and an option to clear existing channels.
    """
    def __init__(self, parent: QtWidgets.QWidget = None, crs_obj=None):
        """
        Initializes the CRS initialization dialog.

        Args:
            parent: The parent widget.
            crs_obj: The CRS object, used to access timestamp port enums.
                     This is optional but required for `get_selected_irig_source`
                     to return meaningful values.
        """
        super().__init__(parent)
        self.crs = crs_obj # Store the CRS object
        self.setWindowTitle("Initialize CRS Board")
        self.setModal(True) # Modal dialog

        layout = QtWidgets.QVBoxLayout(self)

        # IRIG Time Source selection
        irig_group = QtWidgets.QGroupBox("IRIG Time Source")
        irig_layout = QtWidgets.QVBoxLayout(irig_group)
        self.rb_backplane = QtWidgets.QRadioButton("BACKPLANE")
        self.rb_test = QtWidgets.QRadioButton("TEST")
        self.rb_sma = QtWidgets.QRadioButton("SMA")
        self.rb_test.setChecked(True) # Default to TEST
        irig_layout.addWidget(self.rb_backplane)
        irig_layout.addWidget(self.rb_test)
        irig_layout.addWidget(self.rb_sma)
        layout.addWidget(irig_group)

        # Option to clear channels
        self.cb_clear_channels = QtWidgets.QCheckBox("Clear all channels on this module")
        self.cb_clear_channels.setChecked(True) # Default to clearing channels
        layout.addWidget(self.cb_clear_channels)

        # OK and Cancel buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addStretch() # Push buttons to the right
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def get_selected_irig_source(self):
        """
        Gets the selected IRIG time source based on the radio button state.

        Returns:
            The corresponding CRS timestamp port enum value if `crs_obj` was provided
            and a selection is made, otherwise None.
        """
        if self.crs is None:
            # Log a warning or handle this case if crs_obj is critical for operation
            print("Warning: CRS object not provided to InitializeCRSDialog. Cannot determine IRIG source enum.")
            return None 
        if self.rb_backplane.isChecked():
            return self.crs.TIMESTAMP_PORT.BACKPLANE
        if self.rb_test.isChecked():
            return self.crs.TIMESTAMP_PORT.TEST
        if self.rb_sma.isChecked():
            return self.crs.TIMESTAMP_PORT.SMA
        return None # Should not happen if one is always checked

    def get_clear_channels_state(self) -> bool:
        """
        Gets the state of the 'Clear all channels' checkbox.

        Returns:
            True if the checkbox is checked, False otherwise.
        """
        return self.cb_clear_channels.isChecked()

class FindResonancesDialog(QtWidgets.QDialog):
    """
    Dialog for configuring parameters used in the 'Find Resonances' process.
    Allows setting criteria like expected number of resonances, dip depth, Q values, etc.
    """
    def __init__(self, parent: QtWidgets.QWidget = None):
        """
        Initializes the Find Resonances parameter dialog.

        Args:
            parent: The parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Find Resonances Parameters")
        self.setModal(True)
        self._setup_ui()

    def _setup_ui(self):
        """Sets up the user interface elements for the dialog."""
        layout = QtWidgets.QFormLayout(self)

        # Expected number of resonances (optional integer)
        self.expected_resonances_edit = QtWidgets.QLineEdit()
        self.expected_resonances_edit.setPlaceholderText("Optional (e.g., 10)")
        # Validator for empty string or positive integer
        regex_int_or_empty = QRegularExpression("^$|^[1-9][0-9]*$") 
        self.expected_resonances_edit.setValidator(QRegularExpressionValidator(regex_int_or_empty, self))
        layout.addRow("Expected Resonances:", self.expected_resonances_edit)

        # Minimum dip depth in dB (positive float)
        self.min_dip_depth_db_edit = QtWidgets.QLineEdit(str(1.0)) # Default 1.0 dB
        self.min_dip_depth_db_edit.setValidator(QDoubleValidator(0.01, 100.0, 2, self)) # Min 0.01, Max 100, 2 decimals
        layout.addRow("Min Dip Depth (dB):", self.min_dip_depth_db_edit)

        # Minimum Q (float)
        self.min_Q_edit = QtWidgets.QLineEdit(str(1e4)) # Default 10,000
        self.min_Q_edit.setValidator(QDoubleValidator(1.0, 1e9, 0, self)) # Allow scientific notation input
        layout.addRow("Min Q:", self.min_Q_edit)

        # Maximum Q (float)
        self.max_Q_edit = QtWidgets.QLineEdit(str(1e7)) # Default 10,000,000
        self.max_Q_edit.setValidator(QDoubleValidator(1.0, 1e9, 0, self))
        layout.addRow("Max Q:", self.max_Q_edit)

        # Minimum resonance separation in MHz (float)
        self.min_resonance_separation_mhz_edit = QtWidgets.QLineEdit(str(0.1)) # Default 0.1 MHz
        self.min_resonance_separation_mhz_edit.setValidator(QDoubleValidator(0.001, 1000.0, 3, self))
        layout.addRow("Min Separation (MHz):", self.min_resonance_separation_mhz_edit)

        # Data exponent for fitting (float)
        self.data_exponent_edit = QtWidgets.QLineEdit(str(2.0)) # Default 2.0
        self.data_exponent_edit.setValidator(QDoubleValidator(0.1, 10.0, 2, self))
        layout.addRow("Data Exponent:", self.data_exponent_edit)

        # Standard OK and Cancel buttons
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addRow(self.button_box)

    def get_parameters(self) -> dict | None:
        """
        Retrieves and validates the parameters for finding resonances.

        Returns:
            A dictionary of parameters if valid, otherwise None.
            Shows an error message on invalid input or validation failure.
        """
        params_dict = {}
        try:
            expected_text = self.expected_resonances_edit.text().strip()
            params_dict['expected_resonances'] = int(expected_text) if expected_text else None
            
            params_dict['min_dip_depth_db'] = float(self.min_dip_depth_db_edit.text())
            params_dict['min_Q'] = float(self.min_Q_edit.text())
            params_dict['max_Q'] = float(self.max_Q_edit.text())
            params_dict['min_resonance_separation_hz'] = float(self.min_resonance_separation_mhz_edit.text()) * 1e6 # Convert MHz to Hz
            params_dict['data_exponent'] = float(self.data_exponent_edit.text())

            # Perform basic validation
            if params_dict['min_Q'] >= params_dict['max_Q']:
                QtWidgets.QMessageBox.warning(self, "Validation Error", "Min Q must be less than Max Q.")
                return None
            if params_dict['min_dip_depth_db'] <= 0:
                QtWidgets.QMessageBox.warning(self, "Validation Error", "Min Dip Depth must be positive.")
                return None
            # Add other validations as necessary, e.g., min_resonance_separation_hz > 0

            return params_dict
        except ValueError as e: # Handles errors from float() or int() conversion
            QtWidgets.QMessageBox.critical(self, "Input Error", f"Invalid numerical input: {str(e)}")
            return None
        except Exception as e: # Catch any other unexpected errors
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not parse parameters: {str(e)}")
            return None

class MultisweepDialog(NetworkAnalysisDialogBase):
    """
    Dialog for configuring a Multisweep operation.
    Inherits from NetworkAnalysisDialogBase for amplitude settings and DAC scale handling.
    Allows specifying parameters for sweeping multiple pre-identified resonances.
    """
    def __init__(self, parent: QtWidgets.QWidget = None, 
                 resonance_frequencies: list[float] | None = None, 
                 dac_scales: dict[int, float] = None, 
                 current_module: int | None = None, 
                 initial_params: dict | None = None):
        """
        Initializes the Multisweep configuration dialog.

        Args:
            parent: The parent widget.
            resonance_frequencies: List of resonance frequencies (in Hz) to target.
            dac_scales: Pre-fetched DAC scales.
            current_module: The module ID on which the multisweep will be performed.
            initial_params: Dictionary of initial parameters to populate fields.
        """
        super().__init__(parent, params=initial_params, dac_scales=dac_scales)
        self.resonance_frequencies = resonance_frequencies or []
        self.current_module = current_module # Store the current module for DAC scale and params

        self.setWindowTitle("Multisweep Configuration")
        self.setModal(True)
        self._setup_ui()

        # Asynchronously fetch DAC scales if not provided and CRS is available
        # This is similar to NetworkAnalysisParamsDialog logic
        if parent and hasattr(parent, 'parent') and parent.parent() is not None:
            main_periscope_window = parent.parent()
            if hasattr(main_periscope_window, 'crs') and main_periscope_window.crs is not None:
                # Only fetch if dac_scales weren't passed in and we have a method to do so
                if not self.dac_scales and hasattr(self, '_fetch_dac_scales_for_dialog'):
                     self._fetch_dac_scales_for_dialog(main_periscope_window.crs)
                elif self.dac_scales: # If scales were provided, update UI
                    self._update_dac_scale_info()
                    self._update_dbm_from_normalized()

    def _fetch_dac_scales_for_dialog(self, crs_obj):
        """
        Initiates asynchronous fetching of DAC scales for this dialog.
        This method is specific to MultisweepDialog if its DAC fetching needs
        to be handled differently or if it's called from a different context.
        Currently, it's similar to the one in NetworkAnalysisParamsDialog.

        Args:
            crs_obj: The CRS object to query.
        """
        self.fetcher = DACScaleFetcher(crs_obj)
        self.fetcher.dac_scales_ready.connect(self._on_dac_scales_ready_dialog)
        self.fetcher.start()

    @QtCore.pyqtSlot(dict)
    def _on_dac_scales_ready_dialog(self, scales_dict: dict[int, float]):
        """
        Slot to handle received DAC scales specifically for this dialog instance.
        Updates DAC scales and refreshes relevant UI parts.

        Args:
            scales_dict: Dictionary of module ID to DAC scale (dBm).
        """
        self.dac_scales = scales_dict
        self._update_dac_scale_info()
        self._update_dbm_from_normalized()

    def _get_selected_modules(self) -> list[int]:
        """
        Returns the module relevant for this multisweep dialog.
        For multisweep, it's typically a single, pre-determined module.

        Returns:
            A list containing the current_module ID if set, otherwise an empty list.
        """
        return [self.current_module] if self.current_module is not None else []

    def _setup_ui(self):
        """Sets up the user interface elements for the Multisweep dialog."""
        layout = QtWidgets.QVBoxLayout(self)

        # Display information about target resonances
        res_info_group = QtWidgets.QGroupBox("Target Resonances")
        res_info_layout = QtWidgets.QVBoxLayout(res_info_group)
        num_resonances = len(self.resonance_frequencies)
        res_label_text = f"Number of resonances to sweep: {num_resonances}"
        if num_resonances > 0:
            # Show first few resonance frequencies for quick reference
            res_freq_mhz_str = ", ".join([f"{f / 1e6:.3f}" for f in self.resonance_frequencies[:5]])
            if num_resonances > 5:
                res_freq_mhz_str += ", ..." # Indicate more frequencies exist
            res_label_text += f"\nFrequencies (MHz): {res_freq_mhz_str}"
        self.resonances_info_label = QtWidgets.QLabel(res_label_text)
        self.resonances_info_label.setWordWrap(True)
        res_info_layout.addWidget(self.resonances_info_label)
        layout.addWidget(res_info_group)

        # Sweep parameters group
        param_group = QtWidgets.QGroupBox("Sweep Parameters")
        param_form_layout = QtWidgets.QFormLayout(param_group)

        # Span per resonance (kHz)
        default_span_khz = self.params.get('span_hz', 100000.0) / 1e3 # Default 100 kHz
        self.span_khz_edit = QtWidgets.QLineEdit(str(default_span_khz))
        self.span_khz_edit.setValidator(QDoubleValidator(0.1, 10000.0, 2, self)) # Min 0.1 kHz, Max 10 MHz
        param_form_layout.addRow("Span per Resonance (kHz):", self.span_khz_edit)

        # Number of points per sweep
        default_npoints = self.params.get('npoints_per_sweep', 101) # Default 101 points
        self.npoints_edit = QtWidgets.QLineEdit(str(default_npoints))
        self.npoints_edit.setValidator(QIntValidator(2, 10000, self)) # Min 2 points
        param_form_layout.addRow("Number of Points per Sweep:", self.npoints_edit)

        # Samples to average (nsamps)
        default_nsamps = self.params.get('nsamps', DEFAULT_NSAMPLES)
        self.nsamps_edit = QtWidgets.QLineEdit(str(default_nsamps))
        self.nsamps_edit.setValidator(QIntValidator(1, 10000, self)) # Min 1 sample
        param_form_layout.addRow("Samples to Average (nsamps):", self.nsamps_edit)

        self.setup_amplitude_group(param_form_layout) # Shared amplitude settings

        # Option to perform fits
        default_perform_fits = self.params.get('perform_fits', True)
        self.perform_fits_cb = QtWidgets.QCheckBox("Perform fits after sweep")
        self.perform_fits_cb.setChecked(default_perform_fits)
        param_form_layout.addRow("", self.perform_fits_cb)

        # Option to recalculate center frequencies
        default_recalc_cf = self.params.get('recalculate_center_frequencies', True)
        self.recalculate_cf_cb = QtWidgets.QCheckBox("Recalculate Center Frequencies (S21 min)")
        self.recalculate_cf_cb.setChecked(default_recalc_cf)
        self.recalculate_cf_cb.setToolTip(
            "If checked, the minimum of S21 magnitude will be used as the center "
            "frequency for subsequent results display and plotting, potentially "
            "overriding the initially provided resonance frequency for that sweep."
        )
        param_form_layout.addRow("", self.recalculate_cf_cb)
        layout.addWidget(param_group)

        # Standard OK and Cancel buttons
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        # Initial update of dBm field if DAC scales are already known
        if self.dac_scales: # Check if dac_scales were passed or fetched synchronously before UI setup
            self._update_dac_scale_info() # Ensure info label is also up-to-date
            self._update_dbm_from_normalized()
        
        self.setMinimumWidth(500) # Ensure dialog is wide enough

    def get_parameters(self) -> dict | None:
        """
        Retrieves and validates the parameters for the multisweep operation.

        Returns:
            A dictionary of parameters if valid, otherwise None.
            Shows an error message on invalid input or validation failure.
        """
        params_dict = {}
        try:
            amp_text = self.amp_edit.text().strip()
            # Use parsed amplitude values, or default from initial params or global default
            default_amp_val = self.params.get('amp', DEFAULT_AMPLITUDE) if self.params else DEFAULT_AMPLITUDE
            amps_list = self._parse_amplitude_values(amp_text) or [default_amp_val]
            
            params_dict['amps'] = amps_list
            params_dict['span_hz'] = float(self.span_khz_edit.text()) * 1e3 # Convert kHz to Hz
            params_dict['npoints_per_sweep'] = int(self.npoints_edit.text())
            params_dict['nsamps'] = int(self.nsamps_edit.text())
            params_dict['perform_fits'] = self.perform_fits_cb.isChecked()
            params_dict['recalculate_center_frequencies'] = self.recalculate_cf_cb.isChecked()
            
            # Include the essential context for the multisweep
            params_dict['resonance_frequencies'] = self.resonance_frequencies
            params_dict['module'] = self.current_module

            # Basic validation
            if params_dict['span_hz'] <= 0:
                QtWidgets.QMessageBox.warning(self, "Validation Error", "Span must be positive.")
                return None
            if params_dict['npoints_per_sweep'] < 2:
                QtWidgets.QMessageBox.warning(self, "Validation Error", "Number of points per sweep must be at least 2.")
                return None
            if params_dict['nsamps'] < 1:
                QtWidgets.QMessageBox.warning(self, "Validation Error", "Samples to average must be at least 1.")
                return None
            if not self.resonance_frequencies:
                 QtWidgets.QMessageBox.warning(self, "Configuration Error", "No target resonances specified for multisweep.")
                 return None


            return params_dict
        except ValueError as e: # Handles errors from float() or int() conversion
            QtWidgets.QMessageBox.critical(self, "Input Error", f"Invalid numerical input: {str(e)}")
            return None
        except Exception as e: # Catch any other unexpected errors
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not parse parameters: {str(e)}")
            return None
