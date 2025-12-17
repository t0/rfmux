"""Base dialog for network analysis parameter entry."""

from .utils import (
    QtWidgets, QtCore, QRegularExpression, QRegularExpressionValidator,
    QDoubleValidator, QIntValidator,
    DEFAULT_AMPLITUDE, DEFAULT_MIN_FREQ, DEFAULT_MAX_FREQ, DEFAULT_CABLE_LENGTH,
    DEFAULT_NPOINTS, DEFAULT_NSAMPLES, DEFAULT_MAX_CHANNELS, DEFAULT_MAX_SPAN,
    DEFAULT_AMP_START, DEFAULT_AMP_STOP, DEFAULT_AMP_ITERATIONS,
    UnitConverter, traceback
)
from .tasks import DACScaleFetcher
import numpy as np  # For linspace

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
        
        # Connect signals for live updates (without validation) and validation on edit finish
        self.amp_edit.textChanged.connect(lambda: self._update_dbm_from_normalized(validate=False)) # Live update dBm field
        self.dbm_edit.textChanged.connect(lambda: self._update_normalized_from_dbm(validate=False)) # Live update normalized amp field
        self.amp_edit.editingFinished.connect(self._validate_normalized_values) # Validate on finishing edit
        self.dbm_edit.editingFinished.connect(self._validate_dbm_values)       # Validate on finishing edit

        # Linspace generator UI
        linspace_group = QtWidgets.QGroupBox("Generate Amplitude List")
        linspace_layout = QtWidgets.QFormLayout(linspace_group)

        self.start_amp_edit = QtWidgets.QLineEdit(f"{DEFAULT_AMP_START}")
        self.start_amp_edit.setValidator(QDoubleValidator(self))
        self.start_amp_edit.setToolTip("Start value for linspace generation.")
        linspace_layout.addRow("Start:", self.start_amp_edit)

        self.stop_amp_edit = QtWidgets.QLineEdit(f"{DEFAULT_AMP_STOP}")
        self.stop_amp_edit.setValidator(QDoubleValidator(self))
        self.stop_amp_edit.setToolTip("Stop value for linspace generation.")
        linspace_layout.addRow("Stop:", self.stop_amp_edit)

        self.iterations_amp_edit = QtWidgets.QLineEdit(f"{DEFAULT_AMP_ITERATIONS}")
        self.iterations_amp_edit.setValidator(QIntValidator(2, 1000, self)) # Min 2 points for linspace
        self.iterations_amp_edit.setToolTip("Number of points for linspace generation (min 2).")
        linspace_layout.addRow("Iterations:", self.iterations_amp_edit)
        
        button_layout = QtWidgets.QHBoxLayout()
        self.fill_amp_button = QtWidgets.QPushButton("Fill Normalized Amplitude")
        self.fill_amp_button.clicked.connect(self._on_fill_amplitude_clicked)
        button_layout.addWidget(self.fill_amp_button)

        self.fill_dbm_button = QtWidgets.QPushButton("Fill Power (dBm)")
        self.fill_dbm_button.clicked.connect(self._on_fill_dbm_clicked)
        button_layout.addWidget(self.fill_dbm_button)
        
        linspace_layout.addRow(button_layout)
        amp_layout.addRow(linspace_group) # Add this subgroup to the main amplitude layout
        
        layout.addRow("Amplitude Settings:", amp_group)
        return amp_group

    def _on_fill_amplitude_clicked(self):
        """Handles the 'Fill Amplitude' button click."""
        try:
            start = float(self.start_amp_edit.text())
            stop = float(self.stop_amp_edit.text())
            iterations = int(self.iterations_amp_edit.text())

            if iterations < 2:
                QtWidgets.QMessageBox.warning(self, "Input Error", "Iterations must be at least 2.")
                return

            values = np.linspace(start, stop, iterations)
            # Use a general format, good for typical normalized amplitudes
            self.amp_edit.setText(", ".join([f"{v:.6g}" for v in values])) 
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Input Error", "Invalid input for Start, Stop, or Iterations.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not generate amplitude list: {str(e)}")

    def _on_fill_dbm_clicked(self):
        """Handles the 'Fill dBm' button click."""
        try:
            start = float(self.start_amp_edit.text())
            stop = float(self.stop_amp_edit.text())
            iterations = int(self.iterations_amp_edit.text())

            if iterations < 2:
                QtWidgets.QMessageBox.warning(self, "Input Error", "Iterations must be at least 2.")
                return

            values = np.linspace(start, stop, iterations)
            # Use a format suitable for dBm values
            self.dbm_edit.setText(", ".join([f"{v:.2f}" for v in values]))
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Input Error", "Invalid input for Start, Stop, or Iterations.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not generate dBm list: {str(e)}")
        


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
            
    def _parse_numeric_values(self, text: str) -> list[float]:
        """
        Parses a comma-separated string of numeric values.
        Each part can be an expression evaluatable by `eval()`.
        Invalid parts are silently skipped.

        Args:
            text: The string containing numeric values (amplitude or dBm).

        Returns:
            A list of parsed float values.
        """
        values = []
        for part in text.split(','):
            part = part.strip()
            if part:
                try:
                    # Using eval allows for simple expressions like "1/1000".
                    # Caution: eval can execute arbitrary code if input is not controlled.
                    # In this GUI context, user inputs values for their own use.
                    values.append(float(eval(part)))
                except (ValueError, SyntaxError, NameError, TypeError):
                    # Silently skip parts that cannot be evaluated to a float
                    continue
        return values
    
    def _parse_amplitude_values(self, amp_text: str) -> list[float]:
        """Parse comma-separated amplitude values (delegates to _parse_numeric_values)."""
        return self._parse_numeric_values(amp_text)
        
    def _parse_dbm_values(self, dbm_text: str) -> list[float]:
        """Parse comma-separated dBm values (delegates to _parse_numeric_values)."""
        return self._parse_numeric_values(dbm_text)

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
    
    def _update_dbm_from_normalized(self, validate: bool = True):
        """
        Updates the dBm field based on the normalized amplitude field.
        
        Args:
            validate: If True, validate values and show warnings (if field doesn't have focus).
                      If False, perform conversion without validation (for live updates).
        """
        if self.currently_updating or not self.dbm_edit.isEnabled():
            return
        self.currently_updating = True
        try:
            amp_text = self.amp_edit.text().strip()
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
            warnings_list = []
            
            for norm_val in normalized_values:
                # Collect validation warnings only if validate=True
                if validate:
                    if norm_val > 1.0:
                        warnings_list.append(f"Warning: Normalized amplitude {norm_val:.6f} > 1.0 (maximum)")
                    elif norm_val < 1e-4:
                        warnings_list.append(f"Warning: Normalized amplitude {norm_val:.6f} < 1e-4 (minimum recommended)")
                dbm_values.append(f"{UnitConverter.normalize_to_dbm(norm_val, dac_scale):.2f}")
            
            self.dbm_edit.setText(", ".join(dbm_values))
            
            # Show warnings only if validate=True and field doesn't have focus
            if validate and warnings_list and not self.amp_edit.hasFocus():
                self._show_warning_dialog("Normalized Amplitude Warning", warnings_list)
        finally:
            self.currently_updating = False
    
    def _update_normalized_from_dbm(self, validate: bool = True):
        """
        Updates the normalized amplitude field based on the dBm field.
        
        Args:
            validate: If True, validate values and show warnings (if field doesn't have focus).
                      If False, perform conversion without validation (for live updates).
        """
        if self.currently_updating or not self.dbm_edit.isEnabled():
            return
        self.currently_updating = True
        try:
            dbm_text = self.dbm_edit.text().strip()
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
            warnings_list = []
            
            for dbm_val in dbm_values:
                # Collect validation warnings only if validate=True
                if validate:
                    if dbm_val > dac_scale:
                        warnings_list.append(f"Warning: {dbm_val:.2f} dBm > {dac_scale:+.2f} dBm (DAC maximum)")
                
                norm_val = UnitConverter.dbm_to_normalize(dbm_val, dac_scale)
                
                # Validate resulting normalized amplitude only if validate=True
                if validate:
                    if norm_val > 1.0:
                        warnings_list.append(f"Warning: {dbm_val:.2f} dBm results in normalized amplitude > 1.0 ({norm_val:.6f})")
                    elif norm_val < 1e-4:
                        warnings_list.append(f"Warning: {dbm_val:.2f} dBm results in normalized amplitude < 1e-4 ({norm_val:.6f})")
                normalized_values.append(f"{norm_val:.6f}")
            
            self.amp_edit.setText(", ".join(normalized_values))

            # Show warnings only if validate=True and field doesn't have focus
            if validate and warnings_list and not self.dbm_edit.hasFocus():
                self._show_warning_dialog("dBm Amplitude Warning", warnings_list)
        finally:
            self.currently_updating = False
