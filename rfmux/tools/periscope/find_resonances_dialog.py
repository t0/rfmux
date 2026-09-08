"""Dialog for find_resonances parameters."""

from typing import Optional # Import Optional
from .utils import (
    QtWidgets, QRegularExpression, QRegularExpressionValidator,
    QDoubleValidator, DEFAULT_EXPECTED_RESONANCES, DEFAULT_MIN_DIP_DEPTH_DB, DEFAULT_MIN_Q,
    DEFAULT_MAX_Q, DEFAULT_MIN_RESONANCE_SEPARATION_HZ, DEFAULT_DATA_EXPONENT,
    DEFAULT_REQUIRE_ISOLATION
)

class FindResonancesDialog(QtWidgets.QDialog):
    """
    Dialog for configuring parameters used in the 'Find Resonances' process.
    Allows setting criteria like expected number of resonances, dip depth, Q values, etc.
    """
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
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
        # If DEFAULT_EXPECTED_RESONANCES is None, placeholder will be used. Otherwise, set the text.
        if DEFAULT_EXPECTED_RESONANCES is not None:
            self.expected_resonances_edit.setText(str(DEFAULT_EXPECTED_RESONANCES))
        else:
            self.expected_resonances_edit.setPlaceholderText("Optional (e.g., 10)")
        # Validator for empty string or positive integer
        regex_int_or_empty = QRegularExpression("^$|^[1-9][0-9]*$") 
        self.expected_resonances_edit.setValidator(QRegularExpressionValidator(regex_int_or_empty, self))
        layout.addRow("Expected Resonances:", self.expected_resonances_edit)

        # Minimum dip depth in dB (positive float)
        self.min_dip_depth_db_edit = QtWidgets.QLineEdit(str(DEFAULT_MIN_DIP_DEPTH_DB))
        self.min_dip_depth_db_edit.setValidator(QDoubleValidator(0.01, 100.0, 2, self)) # Min 0.01, Max 100, 2 decimals
        layout.addRow("Min Dip Depth (dB):", self.min_dip_depth_db_edit)

        # Minimum Q (float)
        self.min_Q_edit = QtWidgets.QLineEdit(str(DEFAULT_MIN_Q))
        self.min_Q_edit.setValidator(QDoubleValidator(1.0, 1e10, -1, self)) # Allow scientific notation input with any decimal places
        layout.addRow("Min Q:", self.min_Q_edit)

        # Maximum Q (float)
        self.max_Q_edit = QtWidgets.QLineEdit(str(DEFAULT_MAX_Q))
        self.max_Q_edit.setValidator(QDoubleValidator(1.0, 1e10, -1, self))
        layout.addRow("Max Q:", self.max_Q_edit)

        # Minimum resonance separation in MHz (float)
        # DEFAULT_MIN_RESONANCE_SEPARATION_HZ is in Hz, convert to MHz for display
        min_sep_khz = DEFAULT_MIN_RESONANCE_SEPARATION_HZ / 1e3
        self.min_resonance_separation_khz_edit = QtWidgets.QLineEdit(str(min_sep_khz))
        self.min_resonance_separation_khz_edit.setValidator(QDoubleValidator(0.00001, 100000.0, 3, self))
        layout.addRow("Min Separation (KHz):", self.min_resonance_separation_khz_edit)

        # What the separation above means.  Unchecked it thins: of any group
        # closer than it, the most prominent is kept -- so a kept resonance
        # can still have a real one beside it, the one that was discarded.
        # Checked it drops every member of such a group, so what comes back
        # has nothing within the separation.
        self.require_isolation_check = QtWidgets.QCheckBox(
            "Drop both when two are closer than the separation")
        self.require_isolation_check.setChecked(DEFAULT_REQUIRE_ISOLATION)
        self.require_isolation_check.setToolTip(
            "Off: of a crowded group, keep the most prominent. The list obeys "
            "the separation, but a kept resonance may still have a neighbour "
            "just under it -- the one that was dropped.\n\n"
            "On: drop the whole group. Every resonance returned is one "
            "nothing else is close to, so none of them collide.")
        layout.addRow("Isolation:", self.require_isolation_check)

        # Data exponent for fitting (float)
        self.data_exponent_edit = QtWidgets.QLineEdit(str(DEFAULT_DATA_EXPONENT))
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
            params_dict['min_resonance_separation_hz'] = float(self.min_resonance_separation_khz_edit.text()) * 1e3 # Convert KHz to Hz
            params_dict['require_isolation'] = self.require_isolation_check.isChecked()
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
