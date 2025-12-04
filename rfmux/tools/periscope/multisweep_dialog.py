"""Dialog for multisweep settings."""

from .utils import (
    QtWidgets, QtCore, QRegularExpression, QRegularExpressionValidator,
    QDoubleValidator, QIntValidator,
    DEFAULT_AMPLITUDE, DEFAULT_MIN_FREQ, DEFAULT_MAX_FREQ, DEFAULT_CABLE_LENGTH,
    DEFAULT_NPOINTS, DEFAULT_NSAMPLES, DEFAULT_MAX_CHANNELS, DEFAULT_MAX_SPAN,
    MULTISWEEP_DEFAULT_AMPLITUDE, MULTISWEEP_DEFAULT_SPAN_HZ, MULTISWEEP_DEFAULT_NPOINTS, 
    MULTISWEEP_DEFAULT_NSAMPLES, DEFAULT_AMP_START, DEFAULT_AMP_STOP, DEFAULT_AMP_ITERATIONS,
    UnitConverter, traceback
)
from .network_analysis_base import NetworkAnalysisDialogBase
from .tasks import DACScaleFetcher # Import DACScaleFetcher from tasks.py
import pickle
import numpy as np
from PyQt6.QtCore import Qt

def load_multisweep_payload(parent: QtWidgets.QWidget, file_path: str | None = None):
    """
    Loads a multisweep payload from a pickle file.

    If file_path is None, it prompts for a file using a blocking dialog (fallback).
    Otherwise, it loads directly from file_path.
    """
    if file_path is None:
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.Option.DontUseNativeDialog
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent,
            "Load Network Analysis Parameters",
            "",
            "Pickle Files (*.pkl *.pickle);;All Files (*)",
            options=options,
        )

    if not file_path:
        return None

    try:
        with open(file_path, "rb") as fh:
            payload = pickle.load(fh)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(
            parent,
            "Load Failed",
            f"Could not read '{file_path}':\n{exc}",
        )
        return None

    if (
        isinstance(payload, dict)
        and isinstance(payload.get("initial_parameters"), dict)
        and isinstance(payload.get("results_by_iteration"), dict)
    ):
        return payload

    QtWidgets.QMessageBox.warning(
        parent,
        "Invalid File",
        "The selected file does not contain Multisweep parameters.",
    )
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
                 initial_params: dict | None = None, load_multisweep = False, fit_frequencies: list[float] = None):
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
        self.load_multisweep = load_multisweep
        self.fit_frequencies = fit_frequencies

        self.use_data_from_file = False
        self._load_data = {}
        self.reso_count = 0

        self.setWindowTitle("Multisweep Configuration")
        self.setModal(True)
        self.use_raw_frequencies = True

        
        # if self.load_multisweep:
        #     self._setup_load_ui()
        # else:
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


    def _update_resonance_count(self, text):
        """Update label with resonance count based on QLineEdit content."""
        text = text.strip()
        if not text:
            self.resonances_info_label.setText("No data. Enter manually if desired.")
            self.start_btn.setEnabled(False)
            self.load_btn.setEnabled(False)
            return
        self.start_btn.setEnabled(True)
        # Split on commas, ignore empty pieces
        parts = [p.strip() for p in text.split(",") if p.strip()]
        self.reso_count = len(parts)
        self.resonances_info_label.setText(f"Loaded {self.reso_count} resonance(s).")    
    
    def _setup_ui(self):
        """Sets up the user interface elements for the Multisweep dialog."""
        layout = QtWidgets.QVBoxLayout(self)

        # Display information about target resonances
        if self.load_multisweep:
            self.import_button = QtWidgets.QPushButton("Import Sweep File")
            self.import_button.clicked.connect(self._import_file)
            layout.addWidget(self.import_button)
    
            # --- Resonances Section ---
            res_info_group = QtWidgets.QGroupBox("Target Resonances")
            res_info_layout = QtWidgets.QVBoxLayout(res_info_group)
            
            res_label_layout = QtWidgets.QHBoxLayout()
            self.resonances_info_label = QtWidgets.QLabel("No file loaded. Enter manually if desired.")
            self.resonances_info_label.setWordWrap(True)
            res_label_layout.addWidget(self.resonances_info_label, stretch=1)
            
            self.res_freq_combo = QtWidgets.QComboBox()
            self.res_freq_combo.setEnabled(False)
            self.res_freq_combo.addItems(["Use multisweep central frequency", "Use resonance fit frequency"])
            self.res_freq_combo.setToolTip("Select which resonance frequency type to use.")
            self.res_freq_combo.currentIndexChanged.connect(self._scroll_resonance)
            res_label_layout.addWidget(self.res_freq_combo)
            
            res_info_layout.addLayout(res_label_layout)
    
            # Manual input fallback (comma-separated resonances in MHz)
            self.resonances_edit = QtWidgets.QLineEdit()
            self.resonances_edit.setPlaceholderText("Enter resonance frequencies (MHz, comma separated)")
            self.resonances_edit.textChanged.connect(self._update_resonance_count)
            res_info_layout.addWidget(self.resonances_edit)
            layout.addWidget(res_info_group)

        elif self.fit_frequencies is not None:
            res_info_group = QtWidgets.QGroupBox("Target Resonances")
            res_info_layout = QtWidgets.QVBoxLayout(res_info_group)
            
            res_label_layout = QtWidgets.QHBoxLayout()
            self.resonances_info_label = QtWidgets.QLabel("Resonance Frequencies for re-run")
            self.resonances_info_label.setWordWrap(True)
            res_label_layout.addWidget(self.resonances_info_label, stretch=1)
            
            self.res_freq_combo = QtWidgets.QComboBox()
            self.res_freq_combo.addItems(["Use multisweep central frequency", "Use resonance fit frequency"])
            self.res_freq_combo.setToolTip("Select which resonance frequency type to use.")
            self.res_freq_combo.currentIndexChanged.connect(self._scroll_rerun_resonance)
            self.res_freq_combo.setCurrentIndex(0)
            res_label_layout.addWidget(self.res_freq_combo)
            
            res_info_layout.addLayout(res_label_layout)
    
            # Default input fallback (comma-separated resonances in MHz)
            self.resonances_edit = QtWidgets.QLineEdit()
            res_freq_rerun = ", ".join([f"{f / 1e6:.9f}" for f in self.resonance_frequencies])
            self.resonances_edit.setText(res_freq_rerun)
            self.resonances_edit.textChanged.connect(self._update_resonance_count)
            res_info_layout.addWidget(self.resonances_edit)
            layout.addWidget(res_info_group)
            
        else:
            res_info_group = QtWidgets.QGroupBox("Target Resonances")
            res_info_layout = QtWidgets.QVBoxLayout(res_info_group)
            num_resonances = len(self.resonance_frequencies)
            res_label_text = f"Number of resonances to sweep: {num_resonances}"
            if num_resonances > 0:
                # Show first few resonance frequencies for quick reference
                res_freq_mhz_str = ", ".join([f"{f / 1e6:.3f}" for f in self.resonance_frequencies[:5]])
                if num_resonances > 5:
                    res_freq_mhz_str += ", ..."  # Indicate more frequencies exist
                res_label_text += f"\nFrequencies (MHz): {res_freq_mhz_str}"
    
            
            self.resonances_info_label = QtWidgets.QLabel(res_label_text)
            self.resonances_info_label.setWordWrap(True)
            res_info_layout.addWidget(self.resonances_info_label)
            layout.addWidget(res_info_group)

        # Sweep parameters group
        param_group = QtWidgets.QGroupBox("Sweep Parameters")
        param_form_layout = QtWidgets.QFormLayout(param_group)

        # Span per resonance (kHz)
        default_span_khz = self.params.get('span_hz', MULTISWEEP_DEFAULT_SPAN_HZ) / 1e3
        self.span_khz_edit = QtWidgets.QLineEdit(str(default_span_khz))
        self.span_khz_edit.setValidator(QDoubleValidator(0.1, 10000.0, 2, self)) # Min 0.1 kHz, Max 10 MHz
        param_form_layout.addRow("Span per Resonance (kHz):", self.span_khz_edit)

        # Number of points per sweep
        default_npoints = self.params.get('npoints_per_sweep', MULTISWEEP_DEFAULT_NPOINTS)
        self.npoints_edit = QtWidgets.QLineEdit(str(default_npoints))
        self.npoints_edit.setValidator(QIntValidator(2, 10000, self)) # Min 2 points
        param_form_layout.addRow("Number of Points per Sweep:", self.npoints_edit)

        # Samples to average (nsamps)
        default_nsamps = self.params.get('nsamps', MULTISWEEP_DEFAULT_NSAMPLES)
        self.nsamps_edit = QtWidgets.QLineEdit(str(default_nsamps))
        self.nsamps_edit.setValidator(QIntValidator(1, 10000, self)) # Min 1 sample
        param_form_layout.addRow("Samples to Average (nsamps):", self.nsamps_edit)

        
        self.setup_amplitude_group(param_form_layout) # Shared amplitude settings

        
        # Option to recalculate center frequencies
        self.recalc_cf_combo = QtWidgets.QComboBox()
        self.recalc_cf_combo.addItems(["max-dIQ","min-S21","None"])
        
        # Set initial value for bias_frequency_method
        default_recalc_setting = self.params.get('bias_frequency_method', "max-diq")

        if default_recalc_setting == "min-s21":
            self.recalc_cf_combo.setCurrentText("min-S21")
        elif default_recalc_setting == "max-diq":
            self.recalc_cf_combo.setCurrentText("max-dIQ")
        else: # Covers None or any other unexpected string
            self.recalc_cf_combo.setCurrentText("None")
            
        self.recalc_cf_combo.setToolTip(
            "Determines how/if center frequencies are recalculated for biasing:\n"
            "- max-dIQ [Often Optimal]: Recalculates to max IQ velocity |d(I+jQ)/df|. Finds where the IQ trajectory moves fastest.\n"
            "- min-S21: Recalculates to min |S21|.\n"
            "- None: No recalculation. Use original center frequency.\n"
        )
        param_form_layout.addRow("Bias Frequency Method:", self.recalc_cf_combo)
        
        # Option to rotate saved data
        self.rotate_saved_data_checkbox = QtWidgets.QCheckBox("Rotate Saved Data")
        self.rotate_saved_data_checkbox.setChecked(self.params.get('rotate_saved_data', False)) # Default to False for df calibration
        self.rotate_saved_data_checkbox.setToolTip(
            "Whether to rotate sweep data based on TOD analysis.\n"
            "When checked and bias frequency method is not None:\n"
            "- For min-S21: Rotates to minimize the I component of the TOD's mean.\n"
            "- For max-dIQ: Rotates to align the principal component of the TOD with the I-axis.\n"
            "Note: Should be unchecked when using df calibration to ensure consistency."
        )
        param_form_layout.addRow(self.rotate_saved_data_checkbox)
        
        # Sweep direction selection
        self.sweep_direction_combo = QtWidgets.QComboBox()
        self.sweep_direction_combo.addItems(["Upward", "Downward", "Both"])
        
        # Set initial value for sweep_direction
        default_direction = self.params.get('sweep_direction', 'upward')
        if default_direction == "upward":
            self.sweep_direction_combo.setCurrentText("Upward")
        elif default_direction == "downward":
            self.sweep_direction_combo.setCurrentText("Downward")
        elif default_direction == "both":
            self.sweep_direction_combo.setCurrentText("Both")
        else: # Default to Upward for any unexpected value
            self.sweep_direction_combo.setCurrentText("Upward")
            
        self.sweep_direction_combo.setToolTip(
            "Direction of frequency sweep:\n"
            "- Upward: Sweep from lower to higher frequencies.\n"
            "- Downward: Sweep from higher to lower frequencies.\n"
            "- Both: Perform both sweep directions sequentially."
        )
        param_form_layout.addRow("Sweep Direction:", self.sweep_direction_combo)
        
        # Fitting options
        self.apply_skewed_fit_checkbox = QtWidgets.QCheckBox("Apply Skewed Fit")
        self.apply_skewed_fit_checkbox.setChecked(self.params.get('apply_skewed_fit', True)) # Default to True
        param_form_layout.addRow(self.apply_skewed_fit_checkbox)

        self.apply_nonlinear_fit_checkbox = QtWidgets.QCheckBox("Apply Nonlinear Fit")
        self.apply_nonlinear_fit_checkbox.setChecked(self.params.get('apply_nonlinear_fit', True)) # Default to True
        param_form_layout.addRow(self.apply_nonlinear_fit_checkbox)
        
        layout.addWidget(param_group)



        if self.load_multisweep:
            btn_layout = QtWidgets.QHBoxLayout()
            self.start_btn = QtWidgets.QPushButton("Start Multisweep")
            self.start_btn.setEnabled(False)
            self.load_btn = QtWidgets.QPushButton("Load Multisweep")
            self.load_btn.setEnabled(False) ### Will enable once file is available.
            self.cancel_btn = QtWidgets.QPushButton("Cancel")
            btn_layout.addWidget(self.start_btn)
            btn_layout.addWidget(self.load_btn)
            btn_layout.addWidget(self.cancel_btn)
            layout.addLayout(btn_layout)
        
            self.start_btn.clicked.connect(self.accept) # Connect to QDialog's accept slot
    
            self.load_btn.clicked.connect(self._load_data_avail) 
            
            self.cancel_btn.clicked.connect(self.reject) # Connect to QDialog's reject slot

        else:
            btn_layout = QtWidgets.QHBoxLayout()
            self.start_btn = QtWidgets.QPushButton("Start Multisweep")
            self.load_btn = QtWidgets.QPushButton("Load Multisweep")
            self.load_btn.hide()
            self.cancel_btn = QtWidgets.QPushButton("Cancel")
            btn_layout.addWidget(self.start_btn)
            btn_layout.addWidget(self.cancel_btn)
            layout.addLayout(btn_layout)
        
            self.start_btn.clicked.connect(self.accept) # Connect to QDialog's accept slot            
            self.cancel_btn.clicked.connect(self.reject) # Connect to QDialog's reject slot
            

        # Initial update of dBm field if DAC scales are already known
        if self.dac_scales: # Check if dac_scales were passed or fetched synchronously before UI setup
            self._update_dac_scale_info() # Ensure info label is also up-to-date
            self._update_dbm_from_normalized()
        
        self.setMinimumWidth(500) # Ensure dialog is wide enough

    def _load_data_avail(self):
        """Mark that data should be loaded from file and accept the dialog."""
        self.use_data_from_file = True
        self.accept()

    def _scroll_resonance(self):
        """Handle resonance selection changes and update displayed frequencies accordingly. Choice between fit or sweep frequencies"""
        selected = self.res_freq_combo.currentText().lower()
        self.resonances_edit.clear()

        if "fit" in selected:
            self.load_btn.setEnabled(False)
            self.use_raw_frequencies = False
            freqs = self._get_frequencies(self._load_data, self.use_raw_frequencies)
        else:
            self.use_raw_frequencies = True
            self.load_btn.setEnabled(True)
            freqs = self._get_frequencies(self._load_data, self.use_raw_frequencies)

        self.resonances_edit.setText(",".join([f"{f/1e6:.9f}" for f in freqs]))
        self.resonances_info_label.setText(f"Loaded {len(freqs)} resonances from file.")

    def _scroll_rerun_resonance(self):
        """Refresh resonance display when rerunning choice between fit or sweep frequencies."""
        selected = self.res_freq_combo.currentText().lower()
        self.resonances_edit.clear()

        if "fit" in selected:
            freqs = self.fit_frequencies
        else:
            freqs = self.resonance_frequencies

        self.resonances_edit.setText(",".join([f"{f/1e6:.9f}" for f in freqs]))
        self.resonances_info_label.setText(f"Loaded {len(freqs)} resonances from file.")
        
    
    def _import_file(self):
        """
        Trigger non-blocking async file dialog instead of blocking getOpenFileName.
        """
        QtCore.QTimer.singleShot(0, self._open_file_dialog_async)

    def _open_file_dialog_async(self):
        """Open a non-blocking file dialog to select a multisweep parameter file."""
        if not hasattr(self, "_file_dialog") or self._file_dialog is None:
            self._file_dialog = QtWidgets.QFileDialog(self, "Load Multisweep Parameters")
            self._file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
            self._file_dialog.setNameFilters([
                "Pickle Files (*.pkl *.pickle)",
                "All Files (*)",
            ])
            self._file_dialog.setOptions(
                QtWidgets.QFileDialog.Option.DontUseNativeDialog
                | QtWidgets.QFileDialog.Option.ReadOnly
            )
            self._file_dialog.setModal(False)
            self._file_dialog.fileSelected.connect(self._on_file_selected)
            self._file_dialog.rejected.connect(self._on_file_dialog_closed)

        self._file_dialog.open()


    @QtCore.pyqtSlot(str)
    def _on_file_selected(self, path: str):
        """Load selected multisweep data file, extract parameters, and populate the UI fields."""
        payload = load_multisweep_payload(self, file_path=path)
        if payload is None:
            return
    
        self.load_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self._load_data = payload.copy()
        self.res_freq_combo.setEnabled(True)
        
        try:
            params = payload['initial_parameters']
    
            freqs = self._get_frequencies(payload, self.use_raw_frequencies)
            
            self.resonances_edit.setText(",".join([f"{f/1e6:.9f}" for f in freqs]))
            self.resonances_info_label.setText(f"Loaded {len(freqs)} resonances from file.")
        
            # Span per Resonance (Hz -> kHz)
            span_khz = params['span_hz'] / 1e3
            self.span_khz_edit.setText(str(span_khz))
        
            self.npoints_edit.setText(str(params['npoints_per_sweep']))
            self.nsamps_edit.setText(str(params['nsamps']))
        
            idx = self.recalc_cf_combo.findText(params['bias_frequency_method'], Qt.MatchFlag.MatchFixedString)
            if idx >= 0:
                self.recalc_cf_combo.setCurrentIndex(idx)
        
            self.rotate_saved_data_checkbox.setChecked(params['rotate_saved_data'])
        
            idx = self.sweep_direction_combo.findText(params['sweep_direction'].capitalize(),
                                                      Qt.MatchFlag.MatchFixedString)
            if idx >= 0:
                self.sweep_direction_combo.setCurrentIndex(idx)
        
            self.apply_skewed_fit_checkbox.setChecked(params['apply_skewed_fit'])
            self.apply_nonlinear_fit_checkbox.setChecked(params['apply_nonlinear_fit'])
        
            amps = params.get("amps") or ([params["amp"]] if "amp" in params else None)
            if amps:
                try:
                    amp_text = ", ".join(f"{float(amp):g}" for amp in amps)
                except (TypeError, ValueError):
                    amp_text = ", ".join(str(amp) for amp in amps)
                self.amp_edit.setText(amp_text)
        except KeyError as e:
            missing = e.args[0]
            msg = (
                f"Key '{missing}' is missing in the payload.\n"
                "Default value will be used where possible."
            )
            QtWidgets.QMessageBox.warning(self, "Missing Key", msg)
    
    
    def _get_frequencies(self, payload, raw_resonance = True):
        """Extract resonance frequencies from payload, optionally using fitted or sweep data."""
        
        params = payload['initial_parameters']
        freqs = params['resonance_frequencies']
        
        if raw_resonance:
            return freqs

        else:
            ref_freqs = []
    
            if params['apply_skewed_fit']:
                for i in range(len(freqs)):
                    ref_freqs.append(payload['results_by_iteration'][0]['data'][i+1]['fit_params']['fr'])
                ref_freqs.sort()
                return ref_freqs
    
            if params['apply_nonlinear_fit']:
                for i in range(len(freqs)):
                    ref_freqs.append(payload['results_by_iteration'][0]['data'][i+1]['nonlinear_fit_params']['fr'])
                ref_freqs.sort()
                return ref_freqs
    
            for i in range(len(freqs)):
                ref_freqs.append(payload['results_by_iteration'][0]['data'][i+1]['bias_frequency'])
            ref_freqs.sort()
                
            return ref_freqs
        
    
    @QtCore.pyqtSlot()
    def _on_file_dialog_closed(self):
        """Handle closure of the file dialog without file selection."""
        pass  # Optional: keep or clear dialog

    
    def get_parameters(self) -> dict | None:
        """
        Retrieves and validates the parameters for the multisweep operation.

        Returns:
            A dictionary of parameters if valid, otherwise None.
            Shows an error message on invalid input or validation failure.
        """
        params_dict = {}
        try:
            if self.use_data_from_file:
                return self._load_data
            else:
                amp_text = self.amp_edit.text().strip()
                # Parse the text from the amplitude edit field.
                # _parse_amplitude_values returns a list of floats.
                amps_list = self._parse_amplitude_values(amp_text) 
    
                # If amp_text was empty or unparsable, _parse_amplitude_values returns an empty list.
                # In this case, we must provide a default list of amplitudes for the task.
                if not amps_list:
                    # Use a list containing a single default amplitude.
                    # Prioritize default from initial params if available, else global default.
                    initial_amp_setting = self.params.get('amp', DEFAULT_AMPLITUDE) # Could be from 'amp' or 'amps'[0] via setup_amplitude_group
                    
                    # Ensure initial_amp_setting is a single float value
                    if isinstance(initial_amp_setting, list):
                        single_default = initial_amp_setting[0] if initial_amp_setting else DEFAULT_AMPLITUDE
                    else:
                        single_default = initial_amp_setting
    
                    amps_list = [single_default]
    
                # Store the full list of amplitudes (parsed or defaulted) for the MultisweepTask.
                params_dict['amps'] = amps_list
    
                # Store the first amplitude under the singular 'amp' key for potential compatibility
                # or for display purposes elsewhere. The MultisweepTask itself iterates over 'amps'.
                # This assumes amps_list is now guaranteed to be non-empty.
                params_dict['amp'] = amps_list[0]
                
                params_dict['span_hz'] = float(self.span_khz_edit.text()) * 1e3 # Convert kHz to Hz
                params_dict['npoints_per_sweep'] = int(self.npoints_edit.text())
                params_dict['nsamps'] = int(self.nsamps_edit.text())
                
                recalc_method_text = self.recalc_cf_combo.currentText()
                if recalc_method_text == "None":
                    params_dict['bias_frequency_method'] = None
                elif recalc_method_text == "min-S21":
                    params_dict['bias_frequency_method'] = "min-s21"
                elif recalc_method_text == "max-dIQ":
                    params_dict['bias_frequency_method'] = "max-diq"
                else: # Should not happen with QComboBox
                    params_dict['bias_frequency_method'] = None
                
                # Get rotate saved data setting
                params_dict['rotate_saved_data'] = self.rotate_saved_data_checkbox.isChecked()
                
                # Get sweep direction
                sweep_direction_text = self.sweep_direction_combo.currentText()
                if sweep_direction_text == "Upward":
                    params_dict['sweep_direction'] = "upward"
                elif sweep_direction_text == "Downward":
                    params_dict['sweep_direction'] = "downward"
                elif sweep_direction_text == "Both":
                    params_dict['sweep_direction'] = "both"
                else: # Should not happen with QComboBox
                    params_dict['sweep_direction'] = "upward"
                    
                # Include the essential context for the multisweep
                if self.load_multisweep:
                    params_dict['resonance_frequencies'] = []
                    freqs = self.resonances_edit.text().split(',')
                    for f in freqs:
                        params_dict['resonance_frequencies'].append(np.float64(f) * 1e6)
                else:
                    params_dict['resonance_frequencies'] = self.resonance_frequencies
                params_dict['module'] = self.current_module
                
                # Get fitting parameters
                params_dict['apply_skewed_fit'] = self.apply_skewed_fit_checkbox.isChecked()
                params_dict['apply_nonlinear_fit'] = self.apply_nonlinear_fit_checkbox.isChecked()
    
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
                if len(params_dict['resonance_frequencies']) < 1:
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
