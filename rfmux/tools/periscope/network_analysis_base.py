"""Base dialog for network analysis parameter entry."""

from .utils import (
    QtWidgets, QDoubleValidator, QIntValidator,
    DEFAULT_AMPLITUDE, DEFAULT_AMP_START, DEFAULT_AMP_STOP, DEFAULT_AMP_ITERATIONS,
    UnitConverter
)
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
        Sets up the QGroupBox for amplitude settings.

        Amplitude is entered in normalized DAC units, which is what the drivers
        take. The DAC scale is shown beside it, so the operator can see what
        full scale is on this board without the dialog converting anything.

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
        
        self.dac_scale_info = QtWidgets.QLabel("Fetching DAC scales...")
        self.dac_scale_info.setWordWrap(True)
        amp_layout.addRow("DAC Scale (dBm):", self.dac_scale_info)
        
        self.amp_edit.editingFinished.connect(self._validate_normalized_values)

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
        
        self.fill_amp_button = QtWidgets.QPushButton("Fill Normalized Amplitude")
        self.fill_amp_button.setAutoDefault(False)  # Prevent this button from capturing Enter key
        self.fill_amp_button.clicked.connect(self._on_fill_amplitude_clicked)
        linspace_layout.addRow(self.fill_amp_button)
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
        
    def _update_dac_scale_info(self):
        """What full scale is on the selected modules, as the board reports it."""
        selected_modules = self._get_selected_modules()
        scales_text_list = [
            f"Module {module_idx}: {scale:+.2f} dBm" if (
                scale := self.dac_scales.get(module_idx)) is not None
            else f"Module {module_idx}: Unknown"
            for module_idx in selected_modules
        ]
        self.dac_scale_info.setText(
            "\n".join(scales_text_list) if selected_modules
            else "Unknown (no modules selected)")
    
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
    
