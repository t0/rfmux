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
        self.samples_edit.setToolTip("Total number of samples to use in the analysis (1–1,000,000).")
        layout.addRow("Number of Samples:", self.samples_edit)


        # Spectrum Limit (0–1)
        self.spectrum_limit_input = QtWidgets.QDoubleSpinBox()
        self.spectrum_limit_input.setRange(0.05, 1.0)
        self.spectrum_limit_input.setDecimals(2)
        self.spectrum_limit_input.setSingleStep(0.05)
        self.spectrum_limit_input.setValue(0.9)
        self.spectrum_limit_input.setToolTip("Fraction of maximum spectrum to analyze (0.05–1.0).")
        layout.addRow("Spectrum Limit:", self.spectrum_limit_input)

        # Number of Segments
        self.segments_edit = QtWidgets.QLineEdit("20")
        self.segments_edit.setValidator(QtGui.QIntValidator(1, 500_000))
        self.segments_edit.setToolTip("Number of data segments used to compute the average spectrum.")
        layout.addRow("Number of Segments:", self.segments_edit)

        # Decimation (0–6)
        self.decimation_input = QtWidgets.QSpinBox()
        self.decimation_input.setRange(0, 6)
        self.decimation_input.setValue(6)
        self.decimation_input.setToolTip("Reduces the sample rate by 2^N. Lower values mean higher data rate.")
        layout.addRow("Decimation:", self.decimation_input)

        self.reference_input = QtWidgets.QComboBox()
        self.reference_input.addItems(["relative", "absolute"])
        self.reference_input.setCurrentText("relative")
        layout.addRow("Reference:", self.reference_input)

        # Time Taken (s) — Read-only label
        self.time_taken_label = QtWidgets.QLabel("0.00 s")
        self.time_taken_label.setToolTip("Displays estimated capture time (in seconds), provided no failures.")
        layout.addRow("Estimated Time:", self.time_taken_label)

        self.highest_freq_label = QtWidgets.QLabel("268.22 Hz")
        self.highest_freq_label.setToolTip("Displays the highest measurable frequency. Go lower in decimation for higher value.")
        layout.addRow("Highest Frequency:", self.highest_freq_label)

        self.freq_resolution_label = QtWidgets.QLabel("0 Hz")
        self.freq_resolution_label.setToolTip("Shows frequency resolution in Hz.")
        layout.addRow("Frequency Resolution:", self.freq_resolution_label)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("background-color: #fff3cd; color: #856404; padding: 5px; border-radius: 6px;")
        status_label = QtWidgets.QLabel("Status:")
        status_label.setToolTip("Shows warnings or important runtime messages related to configuration.")
        layout.addRow(status_label, self.status_label)

        # Dialog Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addRow(button_box)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        self.resize(480, 420)

    # ------------------------------------------------------
    # Signal Connections
    # ------------------------------------------------------
    def _connect_signals(self):
        self.samples_edit.textChanged.connect(self._update_dependent_values)
        self.spectrum_limit_input.textChanged.connect(self._update_dependent_values)
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
        samples = self._safe_int(self.samples_edit.text(), 10000)
        segments = self._safe_int(self.segments_edit.text(), 20)
        spectrum_limit = self.spectrum_limit_input.value()
        decimation = self.decimation_input.value()

        # --- Compute base & effective highest frequency ---
        base_highest_freq = self._get_frequency(decimation)
        effective_highest_freq = base_highest_freq * spectrum_limit
        if effective_highest_freq > 19_072:
            effective_highest_freq = 19_072.0  # Cap to limit

        # --- Compute estimated time (example heuristic) ---
        time_taken = samples/(base_highest_freq*2)

        nperseg = self._safe_int(samples // segments, 500)
        freq_resolution = (base_highest_freq*2)/nperseg

        # --- Update UI ---
        self.highest_freq_label.setText(f"{effective_highest_freq:.2f} Hz")
        self.time_taken_label.setText(f"{time_taken:.2f} s")
        self.freq_resolution_label.setText(f"{freq_resolution:.4f} Hz")

        if decimation <= 1:
            self.status_label.setText(
                "Decimation ≤ 1: You will drop packets in Mac and Windows, increase buffer in Linux (see Help).\nOnly 128 channels available."
            )
            self.status_label.setStyleSheet("background-color: #f8d7da; color: #721c24; padding: 5px; border-radius: 6px;")
        elif 1 < decimation <=3:
            self.status_label.setText("Decimation = 2 or 3: 128 channels but only for the current module.")
            self.status_label.setStyleSheet("background-color: #fff3cd; color: #856404; padding: 5px; border-radius: 6px;")
        elif decimation == 4:
            self.status_label.setText("Decimation = 4: 1024 channels but only for the current module.")
            self.status_label.setStyleSheet("background-color: #fff3cd; color: #856404; padding: 5px; border-radius: 6px;")
        else:
            self.status_label.setText("")
            self.status_label.setStyleSheet("")

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

    #### To avoid none errors, it assumes the default values #####
    def _safe_int(self, text, default=1):
        try:
            return max(1, int(text))
        except ValueError:
            return default

    def _safe_float(self, text, default=0.9):
        try:
            return float(text)
        except ValueError:
            return default


    # ------------------------------------------------------
    # Get User Parameters
    # ------------------------------------------------------
    def get_parameters(self):
        """Return all configuration parameters."""
        
        highest_freq = self.highest_freq_label.text().split()[0]
        time_taken = self.time_taken_label.text().split()[0]
        freq_res = self.freq_resolution_label.text().split()[0]
        
        return {
            "num_samples": self._safe_int(self.samples_edit.text(), 10000),
            "spectrum_limit": self.spectrum_limit_input.value(),
            "num_segments": self._safe_int(self.segments_edit.text(), 20),
            "decimation": self.decimation_input.value(),
            "reference" : self.reference_input.currentText(),
            "effective_highest_freq": float(highest_freq),
            "time_taken": float(time_taken),
            "freq_resolution" : float(freq_res),
        }
