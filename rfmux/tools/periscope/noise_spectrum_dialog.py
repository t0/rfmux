"""
Noise Spectrum Configuration Dialog
===================================

Interactive dialog that allows users to configure parameters for
noise spectrum analysis. Parameters are interdependent — changing one
updates others automatically.
"""

from PyQt6 import QtWidgets, QtCore, QtGui


class NoiseSpectrumDialog(QtWidgets.QDialog):
    """
    Dialog for configuring noise spectrum parameters with live dependency updates.
    """

    def __init__(self, parent=None, get_decimation_func=None):
        """
        Args:
            parent: Parent QWidget.
            get_decimation_func: Optional callable returning highest frequency
                                 based on decimation (e.g., from hardware config).
        """
        super().__init__(parent)
        self.setWindowTitle("Noise Spectrum Configuration")
        self.setModal(True)

        # Internal state
        self._updating = False  # Prevent feedback loops during recalculations

        self._setup_ui()
        self._connect_signals()
        self._update_dependent_values()

    # ------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------
    def _setup_ui(self):
        layout = QtWidgets.QFormLayout(self)

        # Number of Samples
        self.samples_edit = QtWidgets.QLineEdit("10000")
        self.samples_edit.setValidator(QtGui.QIntValidator(1, 1_000_000))
        layout.addRow("Number of Samples:", self.samples_edit)


        # Spectrum Limit (0–1)
        self.spectrum_limit_edit = QtWidgets.QLineEdit("0.9")
        double_validator = QtGui.QDoubleValidator(0.0, 1.0, 3)
        double_validator.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)
        self.spectrum_limit_edit.setValidator(double_validator)
        layout.addRow("Spectrum Limit:", self.spectrum_limit_edit)

        # Number of Segments
        self.segments_edit = QtWidgets.QLineEdit("20")
        self.segments_edit.setValidator(QtGui.QIntValidator(1, 500_000))
        layout.addRow("Number of Segments:", self.segments_edit)

        # Decimation (0–6)
        self.decimation_input = QtWidgets.QSpinBox()
        self.decimation_input.setRange(0, 6)
        self.decimation_input.setValue(6)
        layout.addRow("Decimation:", self.decimation_input)

        # Time Taken (s) — Read-only label
        self.time_taken_label = QtWidgets.QLabel("0.00 s")
        layout.addRow("Estimated Time:", self.time_taken_label)

        self.highest_freq_label = QtWidgets.QLabel("268.22 Hz")
        layout.addRow("Highest Frequency:", self.highest_freq_label)

        self.freq_resolution_label = QtWidgets.QLabel("0 Hz")
        layout.addRow("Frequency Resolution:", self.freq_resolution_label)

        # Dialog Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addRow(button_box)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

    # ------------------------------------------------------
    # Signal Connections
    # ------------------------------------------------------
    def _connect_signals(self):
        self.samples_edit.textChanged.connect(self._update_dependent_values)
        self.spectrum_limit_edit.textChanged.connect(self._update_dependent_values)
        self.segments_edit.textChanged.connect(self._update_dependent_values)
        self.decimation_input.valueChanged.connect(self._update_dependent_values)

    # ------------------------------------------------------
    # Core Logic
    # ------------------------------------------------------
    def _update_dependent_values(self):
        """Recalculate dependent values (frequency and time)."""
        if self._updating:
            return
        self._updating = True

        # --- Read inputs safely ---
        samples = int(self.samples_edit.text())
        segments = int(self.segments_edit.text())
        spectrum_limit = float(self.spectrum_limit_edit.text())
        decimation = self.decimation_input.value()

        # --- Compute base & effective highest frequency ---
        base_highest_freq = self._get_frequency(decimation)
        effective_highest_freq = base_highest_freq * spectrum_limit
        if effective_highest_freq > 19_072:
            effective_highest_freq = 19_072.0  # Cap to limit

        # --- Compute estimated time (example heuristic) ---
        time_taken = samples/(base_highest_freq*2)

        nperseg = samples // segments
        freq_resolution = (base_highest_freq*2)/nperseg

        # --- Update UI ---
        self.highest_freq_label.setText(f"{effective_highest_freq:.2f} Hz")
        self.time_taken_label.setText(f"{time_taken:.2f} s")
        self.freq_resolution_label.setText(f"{freq_resolution:.4f} Hz")

        self._updating = False

    # ------------------------------------------------------
    # Default Decimation → Frequency Mapping
    # ------------------------------------------------------
    @staticmethod
    def _get_frequency(decimation_level: int) -> float:
        """Maps decimation level to max measurable frequency (Hz)."""
        slow_fs = 625e6 / (256 * 64 * (2 ** decimation_level))
        freq = slow_fs/2
        return freq


    # ------------------------------------------------------
    # Get User Parameters
    # ------------------------------------------------------
    # def get_parameters(self):
    #     """Return all configuration parameters."""
    #     return {
    #         "num_samples": int(self.samples_edit.text()),
    #         "spectrum_limit": float(self.spectrum_limit_edit.text()),
    #         "num_segments": int(self.segments_edit.text()),
    #         "decimation": self.decimation_input.value(),
    #         "effective_highest_freq": float(self.highest_freq_label.text()),
    #         "time_taken": float(self.time_taken_label.text()),
    #         "freq_resolution" : float(self.freq_resolution_label.text()),
    #     }
