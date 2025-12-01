"""
Noise Spectrum Configuration Dialog
===================================

Interactive dialog that allows users to configure parameters for
noise spectrum analysis. Parameters are interdependent — changing one
updates others automatically.
"""

from PyQt6 import QtWidgets, QtCore, QtGui
from rfmux.core.transferfunctions import decimation_to_sampling


class NoiseSpectrumDialog(QtWidgets.QDialog):
    """
    Dialog for configuring noise spectrum parameters with live dependency updates.
    """

    def __init__(self, parent=None, num_resonances = 0, crs = None): #### Remmove the decimation function
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
        
        self.nres = num_resonances

        if crs is not None:
            self.crs = crs
        else:
            print("Error no crs object found!")

        self._setup_ui()
        self._connect_signals()
        self._update_dependent_values()



        ##### Add here the number of resonances/frequencies here ####
        

    # ------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------
    def _setup_ui(self):
        layout = QtWidgets.QFormLayout(self)

        self.number_res = QtWidgets.QLabel(str(self.nres))
        self.number_res.setToolTip("Number of resonances")
        layout.addRow("Number of resonances:", self.number_res)

        # Number of Samples
        self.samples_edit = QtWidgets.QLineEdit("10000")
        self.samples_edit.setValidator(QtGui.QIntValidator(1, 10_000_000))
        self.samples_edit.setToolTip("Total number of samples to use in the analysis (1–10,000,000).")
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
        self.segments_edit = QtWidgets.QLineEdit("10")
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
        self.reference_input.addItems(["dBc", "dBm"])
        self.reference_input.setCurrentText("dBm")
        layout.addRow("Units:", self.reference_input)

        # Time Taken (s) — Read-only label
        self.time_taken_label = QtWidgets.QLabel("0.00 s")
        self.time_taken_label.setToolTip("Displays estimated capture time (in seconds), provided no failures.")
        layout.addRow("Estimated Slow Time:", self.time_taken_label)

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


        # ------------------------------------------------------
        # PFB Section (hidden under checkbox)
        # ------------------------------------------------------
        self.pfb_checkbox = QtWidgets.QCheckBox("Enable PFB spectrum")
        self.pfb_checkbox.setToolTip("Get PFB spectrum as well.")
        if self.crs.serial == "0000": ### Don't take pfb for mock mode
            self.pfb_checkbox.hide()
        layout.addRow(self.pfb_checkbox)
        
        # Create a sub-layout for PFB-related info
        self.pfb_group = QtWidgets.QWidget()
        pfb_layout = QtWidgets.QFormLayout(self.pfb_group)
        
        # PFB Calculations
        pfb_samples = 210_000 ### Change time_pfb as well if you change the number of samples
        time_pfb = 0.4 * self.nres  # Example computation, was timed in a notebook takes around 0.28 seconds for 100000 samples
        
        self.pfb_time_taken_label = QtWidgets.QLabel(f"{time_pfb:.3f} s")
        self.pfb_time_taken_label.setToolTip("Displays estimated capture time (in seconds) for PFB, provided no failures.")
        pfb_layout.addRow("Estimated PFB Time:", self.pfb_time_taken_label)
        
        self.pfb_samples = QtWidgets.QLabel(str(pfb_samples))
        self.pfb_samples.setToolTip("Number of PFB samples.")
        pfb_layout.addRow("PFB Samples:", self.pfb_samples)
        
        self.overlap_sample = QtWidgets.QLabel("0")
        self.overlap_sample.setToolTip("Internally estimated overlapping frequency samples. Reduce decimation or segments for more overlap.")
        pfb_layout.addRow("Overlapping Samples:", self.overlap_sample)

        
        # Hide by default
        self.pfb_group.setVisible(False)
        layout.addRow(self.pfb_group)


        # Dialog Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addRow(button_box)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        self.pfb_checkbox.toggled.connect(self._toggle_pfb_section)

        self.resize(500, 500)

    # ------------------------------------------------------
    # Signal Connections
    # ------------------------------------------------------
    def _connect_signals(self):
        self.samples_edit.textChanged.connect(self._update_dependent_values)
        self.spectrum_limit_input.valueChanged.connect(self._update_dependent_values)
        self.segments_edit.textChanged.connect(self._update_dependent_values)
        self.decimation_input.valueChanged.connect(self._update_dependent_values)

    def _toggle_pfb_section(self, checked: bool):
        """Show or hide the PFB configuration section."""
        self.pfb_group.setVisible(checked)


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
        segments = self._safe_int(self.segments_edit.text(), 10)
        spectrum_limit = self.spectrum_limit_input.value()
        decimation = self.decimation_input.value()
        pfb_samps = self.pfb_samples.text()

        # --- Compute base & effective highest frequency ---
        base_highest_freq = self._get_frequency(decimation)
        effective_highest_freq = base_highest_freq * spectrum_limit

        # --- Compute estimated time (example heuristic) ---
        time_taken = samples/(base_highest_freq*2)

        nperseg = self._safe_int(samples // segments, 500)
        freq_resolution = (base_highest_freq*2)/nperseg

        #### Calculating overlapping samples #######
        #### 7840 - Number of pfb frequency samples at segmentation 1, less than the maximum frequency at decimation 0. Determined manually
        #### This was estimated for million samples, hence we calculate a ratio
        #### 0.6 - keeping 60% of overlapping sample

        samp_ratio = 1000000/int(pfb_samps)

        overlap = ((7840 * spectrum_limit)/(2**decimation * segments * samp_ratio)) * 0.6            

        # --- Update UI ---
        self.highest_freq_label.setText(f"{effective_highest_freq:.2f} Hz")
        self.time_taken_label.setText(f"{time_taken:.2f} s")
        self.freq_resolution_label.setText(f"{freq_resolution:.4f} Hz")
        self.overlap_sample.setText(f"{int(overlap)}")

        if decimation <= 1:
            self.status_label.setText(
                "Decimation ≤ 1: You will drop packets in Mac and Windows, increase UDP buffer in Linux (see Help).\nOnly 128 channels available."
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
        slow_fs = decimation_to_sampling(decimation_level)
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
        
        pfb_samps = self.pfb_samples.text()
        pfb_time = self.pfb_time_taken_label.text().split()[0]
        overlap_samps = self.overlap_sample.text()
        
        # Map UI units back to backend reference mode
        # dBc -> relative, dBm -> absolute
        ref_text = self.reference_input.currentText()
        reference_mode = "relative" if ref_text == "dBc" else "absolute"
        
        params = {
            "num_samples": self._safe_int(self.samples_edit.text(), 10000),
            "spectrum_limit": self.spectrum_limit_input.value(),
            "num_segments": self._safe_int(self.segments_edit.text(), 10),
            "decimation": self.decimation_input.value(),
            "reference" : reference_mode,
            "effective_highest_freq": float(highest_freq),
            "time_taken": float(time_taken),
            "freq_resolution" : float(freq_res)
        }
        if self.pfb_checkbox.isChecked():
            params["pfb_enabled"] = True
            params["pfb_samples"] = self._safe_int(pfb_samps, 1000000)
            params["pfb_time"] = float(pfb_time)
            params["overlap"] = int(overlap_samps)
        else:
            params["pfb_enabled"] = False

        return params
