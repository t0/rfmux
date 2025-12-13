"""Dialog for configuring Bias KIDs algorithm parameters."""

from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QVBoxLayout, QFormLayout,
                             QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QLabel)
from PyQt6.QtCore import Qt
import numpy as np

from .utils import (QtWidgets, QtCore, QRegularExpression, QRegularExpressionValidator, QDoubleValidator, QIntValidator)
import pickle

def load_bias_payload(parent: QtWidgets.QWidget, file_path: str | None = None):
    """
    Loads a bias payload from a pickle file.

    If file_path is None, it prompts the user (blocking fallback).
    Otherwise loads directly from the given path.
    """
    if file_path is None:
        # Fallback blocking dialog if no path is passed
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.Option.DontUseNativeDialog
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent,
            "Load Bias Parameters",
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
        and (payload.get("bias_kids_output") is not None)
    ):
        return payload

    QtWidgets.QMessageBox.warning(
        parent,
        "Invalid File",
        "The selected file does not contain Bias parameters.",
    )
    return None



class BiasKidsDialog(QDialog):
    """Dialog for configuring Bias KIDs algorithm parameters."""
    
    def __init__(self, parent=None, active_module : int | None = None, load_bias = False, loaded_data: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Bias KIDs Configuration")
        self.setModal(True)
        self.setMinimumWidth(400)
        self._load_data = loaded_data or {}  # Pre-populate if provided
        self.use_load_file = False
        self.current_module = active_module

        if load_bias:
            self.load_ui()
        else:
            self.setup_ui()


    def setup_ui(self):
        # Create layout
        layout = QVBoxLayout(self)
        
        # Basic parameters group
        basic_group = QGroupBox("Basic Parameters")
        basic_layout = QFormLayout()
        
        # Nonlinear threshold
        self.nonlinear_threshold_spin = QDoubleSpinBox()
        self.nonlinear_threshold_spin.setRange(0.1, 2.0)
        self.nonlinear_threshold_spin.setValue(0.77)
        self.nonlinear_threshold_spin.setSingleStep(0.01)
        self.nonlinear_threshold_spin.setDecimals(2)
        self.nonlinear_threshold_spin.setToolTip(
            "Maximum acceptable nonlinear parameter 'a' for biasing.\n"
            "Detectors with 'a' above this threshold will use fallback amplitude."
        )
        basic_layout.addRow("Nonlinear Threshold:", self.nonlinear_threshold_spin)
        
        # Fallback to lowest checkbox
        self.fallback_checkbox = QCheckBox("Fallback to Lowest Amplitude")
        self.fallback_checkbox.setChecked(True)
        self.fallback_checkbox.setToolTip(
            "If no suitable amplitude is found (all bifurcated or above nonlinear threshold),\n"
            "use the lowest available amplitude. If unchecked, such detectors are skipped."
        )
        basic_layout.addRow("", self.fallback_checkbox)
        
        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)
        
        # Phase optimization group
        phase_group = QGroupBox("Phase Optimization")
        phase_layout = QFormLayout()
        
        # Optimize phase checkbox
        self.optimize_phase_checkbox = QCheckBox("Enable Phase Optimization")
        self.optimize_phase_checkbox.setChecked(False)
        self.optimize_phase_checkbox.setToolTip(
            "Scan through ADC phases to find the phase that maximizes\n"
            "the variance in the bandpass-filtered Q timestream."
        )
        phase_layout.addRow("", self.optimize_phase_checkbox)
        
        # Bandpass filter parameters
        filter_label = QLabel("Bandpass Filter:")
        filter_label.setStyleSheet("font-weight: bold;")
        phase_layout.addRow(filter_label, QLabel())
        
        # Low cutoff frequency
        self.lowcut_spin = QDoubleSpinBox()
        self.lowcut_spin.setRange(0.1, 100.0)
        self.lowcut_spin.setValue(5.0)
        self.lowcut_spin.setSingleStep(0.5)
        self.lowcut_spin.setDecimals(1)
        self.lowcut_spin.setSuffix(" Hz")
        self.lowcut_spin.setToolTip("Low cutoff frequency for bandpass filter")
        phase_layout.addRow("Low Cutoff:", self.lowcut_spin)
        
        # High cutoff frequency
        self.highcut_spin = QDoubleSpinBox()
        self.highcut_spin.setRange(1.0, 500.0)
        self.highcut_spin.setValue(20.0)
        self.highcut_spin.setSingleStep(1.0)
        self.highcut_spin.setDecimals(1)
        self.highcut_spin.setSuffix(" Hz")
        self.highcut_spin.setToolTip("High cutoff frequency for bandpass filter")
        phase_layout.addRow("High Cutoff:", self.highcut_spin)
        
        # Sampling frequency
        self.fs_spin = QDoubleSpinBox()
        self.fs_spin.setRange(100.0, 10000.0)
        self.fs_spin.setValue(597.0)
        self.fs_spin.setSingleStep(1.0)
        self.fs_spin.setDecimals(1)
        self.fs_spin.setSuffix(" Hz")
        self.fs_spin.setToolTip("Sampling frequency for bandpass filter")
        phase_layout.addRow("Sampling Frequency:", self.fs_spin)
        
        # Apply bandpass filter checkbox
        self.apply_bandpass_checkbox = QCheckBox("Apply Bandpass Filter")
        self.apply_bandpass_checkbox.setChecked(True)
        self.apply_bandpass_checkbox.setToolTip(
            "Apply a bandpass filter to the Q timestream before calculating variance.\n"
            "This can help isolate the signal of interest from noise."
        )
        phase_layout.addRow("", self.apply_bandpass_checkbox)
        
        # Phase optimization parameters
        opt_label = QLabel("Optimization:")
        opt_label.setStyleSheet("font-weight: bold;")
        phase_layout.addRow(opt_label, QLabel())
        
        # Number of samples per phase
        self.num_samples_spin = QSpinBox()
        self.num_samples_spin.setRange(50, 1000)
        self.num_samples_spin.setValue(300)
        self.num_samples_spin.setSingleStep(50)
        self.num_samples_spin.setToolTip(
            "Number of samples to collect at each phase\n"
            "for variance calculation"
        )
        phase_layout.addRow("Samples per Phase:", self.num_samples_spin)
        
        # Phase step size
        self.phase_step_spin = QSpinBox()
        self.phase_step_spin.setRange(1, 45)
        self.phase_step_spin.setValue(5)
        self.phase_step_spin.setSingleStep(1)
        self.phase_step_spin.setSuffix("°")
        self.phase_step_spin.setToolTip(
            "Phase step size in degrees for optimization scan.\n"
            "Smaller steps give finer resolution but take longer."
        )
        phase_layout.addRow("Phase Step:", self.phase_step_spin)
        
        phase_group.setLayout(phase_layout)
        layout.addWidget(phase_group)
        
        # Connect checkbox to enable/disable phase optimization controls
        self.optimize_phase_checkbox.toggled.connect(self._update_phase_controls)
        self._update_phase_controls(False)
        
        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def load_ui(self):
        """Initialize and load the user interface for the Load Bias dialog, 
        including input fields, buttons, and layout configuration."""
        
        self.setWindowTitle("Load Bias")
        layout = QtWidgets.QVBoxLayout(self)

        # --- Top: Import File Button ---
        self.import_button = QtWidgets.QPushButton("Import File")
        self.import_button.clicked.connect(self._import_file)
        layout.addWidget(self.import_button, alignment=QtCore.Qt.AlignTop)

        # --- Tones ---
        tones_layout = QtWidgets.QHBoxLayout()
        tones_label = QtWidgets.QLabel("Tones (MHz):")
        self.tones_edit = QtWidgets.QLineEdit()
        tones_layout.addWidget(tones_label)
        tones_layout.addWidget(self.tones_edit)
        layout.addLayout(tones_layout)

        # --- Sweep Amplitude ---
        amp_layout = QtWidgets.QHBoxLayout()
        amp_label = QtWidgets.QLabel("Sweep Amplitude (normalized):")
        self.amp_edit = QtWidgets.QLineEdit()
        amp_layout.addWidget(amp_label)
        amp_layout.addWidget(self.amp_edit)
        layout.addLayout(amp_layout)

        # span_layout = QtWidgets.QHBoxLayout()
        # span_label = QtWidgets.QLabel("Span (kHz)")
        # self.span_khz_edit = QtWidgets.QLineEdit(str(50.0))
        # self.span_khz_edit.setValidator(QDoubleValidator(0.1, 10000.0, 2, self)) # Min 0.1 kHz, Max 10 MHz
        # span_layout.addWidget(span_label)
        # span_layout.addWidget(self.span_khz_edit)
        # layout.addLayout(span_layout)

        # --- Phase ---
        # phase_layout = QtWidgets.QHBoxLayout()
        # phase_label = QtWidgets.QLabel("Rotational Phases (degrees):")
        # self.phase_edit = QtWidgets.QLineEdit("0.0")  # default value
        # phase_layout.addWidget(phase_label)
        # phase_layout.addWidget(self.phase_edit)
        # layout.addLayout(phase_layout)

        # --- Bottom Row: Set Bias / Set and Plot Bias ---
        bias_layout = QtWidgets.QHBoxLayout()
        self.set_bias_btn = QtWidgets.QPushButton("Set Bias")
        self.plot_bias_btn = QtWidgets.QPushButton("Set + Plot Bias")
        self.plot_bias_btn.setEnabled(False)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        bias_layout.addWidget(self.set_bias_btn, alignment=QtCore.Qt.AlignLeft)
        bias_layout.addWidget(self.plot_bias_btn, alignment=QtCore.Qt.AlignCenter)
        bias_layout.addWidget(self.cancel_btn, alignment=QtCore.Qt.AlignRight)
        layout.addLayout(bias_layout)

        # --- Connect actions ---
        self.set_bias_btn.clicked.connect(self.accept)
        self.plot_bias_btn.clicked.connect(self._on_set_and_plot_bias)
        self.cancel_btn.clicked.connect(self.reject)

        self.setMinimumWidth(500) # Ensure dialog is wide enough
        
        # If data was pre-loaded (from double-click), populate fields immediately
        if self._load_data:
            self._populate_fields_from_data(self._load_data)
            self.plot_bias_btn.setEnabled(True)
            self.use_load_file = True  # Mark as using loaded file

    def _populate_fields_from_data(self, payload: dict):
        """Populate UI fields from loaded payload data."""
        params = payload['initial_parameters']
        bias_output = payload['bias_kids_output']
    
        bias_freqs = []
        amplitudes = []
        phases = []
    
        for det_idx, det_data in bias_output.items():
            if not det_data.get("bias_successful", True):
                print("Bias was successful")
    
            channel = int(det_data.get("bias_channel", det_idx))
            bias_freq = det_data.get("bias_frequency") or det_data.get("original_center_frequency")
            bias_freqs.append(bias_freq)
    
            amplitude = det_data.get("sweep_amplitude")
            amplitudes.append(amplitude)
    
            phase = det_data.get("optimal_phase_degrees", 0)
            phases.append(phase)
    
        self.tones_edit.setText(",".join([f"{f/1e6:.6f}" for f in bias_freqs]))
        self.amp_edit.setText(",".join([f"{a:.3f}" for a in amplitudes]))

    def _on_set_and_plot_bias(self):
        """Set flag to use the loaded file and accept the dialog, 
        triggering both bias setting and plotting actions."""
        self.use_load_file = True
        self.accept()

    def _import_file(self):
        # Use a deferred call so the dialog opens outside the blocked event loop
        QtCore.QTimer.singleShot(0, self._open_file_dialog_async)


    def _open_file_dialog_async(self):
        """Open a non-blocking file dialog for selecting a bias parameter file."""
        if not hasattr(self, "_file_dialog") or self._file_dialog is None:
            self._file_dialog = QtWidgets.QFileDialog(self, "Load Bias Parameters")
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
        """Handle file selection, load bias data, and populate the UI fields with file contents."""
        payload = load_bias_payload(self, file_path=path)
        if payload is None:
            return
    
        self.plot_bias_btn.setEnabled(True)
        self._load_data = payload.copy()
        
        # Use the helper method to populate fields
        self._populate_fields_from_data(payload)
    
    
    @QtCore.pyqtSlot()
    def _on_file_dialog_closed(self):
        """Handle the event when the file dialog is closed without selection."""
        pass


        
    def get_load_param(self) -> dict | None:
        """Retrieve and validate user input or loaded data, returning parameters as a dictionary.
        Performs input validation and displays warnings for invalid configurations."""
        params_dict = {}
        try:
            if self.use_load_file:
                return self._load_data
            else:
                amp_text = [x.strip() for x in self.amp_edit.text().split(',')]
                tone_text = [t.strip() for t in self.tones_edit.text().split(',')]
                # phase_text = [p.strip() for p in self.phase_edit.text().split(',')]
    
                params_dict['module'] = self.current_module
                params_dict['phases'] = []
                params_dict['amplitudes'] = []
                params_dict['bias_frequencies'] = []
    
                for i in range(len(amp_text)):
                    params_dict['amplitudes'].append(float(amp_text[i]))
                for i in range(len(tone_text)):    
                    params_dict['bias_frequencies'].append(float(tone_text[i])*1e6)
    
                # span_hz = float(self.span_khz_edit.text()) * 1e3
                # params_dict['span_hz'] = span_hz
    
                len_amps = len(params_dict['amplitudes'])
                len_bias = len(params_dict['bias_frequencies'])
    
                # if len(phase_text) == 1: ##### In case user wants to set the same phase value for all the tones ####
                for i in range(len_amps):
                    params_dict['phases'].append(float(0)) ## default phase a 0
                # else:
                #     for i in range(len(phase_text)):    
                #         params_dict['phases'].append(float(phase_text[i]))
    
                # if params_dict['span_hz'] <= 0:
                #     QtWidgets.QMessageBox.warning(self, "Validation Error", "Span must be positive.")
                #     return None
                if params_dict['module'] is None:
                    QtWidgets.QMessageBox.warning(self, "Validation Error", "No module identified.")
                    return None
                if len_amps < 1:
                    QtWidgets.QMessageBox.warning(self, "Validation Error", "No amplitudes provided.")
                    return None
                if (len_amps != len_bias):
                    QtWidgets.QMessageBox.warning(self, "Validation Error", "Number of amplitudes and frequencies are not the same")
                    return None
                if len_bias < 1:
                    QtWidgets.QMessageBox.warning(self, "Configuration Error", "No frequencies provided.")
                    return None
    
                return params_dict

        
        except ValueError as e: # Handles errors from float() or int() conversion
            QtWidgets.QMessageBox.critical(self, "Input Error", f"Invalid numerical input: {str(e)}")
            return None
        except Exception as e: # Catch any other unexpected errors
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not parse parameters: {str(e)}")
            return None
    
    def _update_phase_controls(self, enabled):
        """Enable/disable phase optimization controls based on checkbox state."""
        self.apply_bandpass_checkbox.setEnabled(enabled)
        self.lowcut_spin.setEnabled(enabled and self.apply_bandpass_checkbox.isChecked())
        self.highcut_spin.setEnabled(enabled and self.apply_bandpass_checkbox.isChecked())
        self.fs_spin.setEnabled(enabled and self.apply_bandpass_checkbox.isChecked())
        self.num_samples_spin.setEnabled(enabled)
        self.phase_step_spin.setEnabled(enabled)
        
        # Connect the bandpass checkbox to update filter controls
        if enabled:
            self.apply_bandpass_checkbox.toggled.connect(self._update_filter_controls)
        else:
            try:
                self.apply_bandpass_checkbox.toggled.disconnect(self._update_filter_controls)
            except TypeError:
                pass  # Not connected
    
    def _update_filter_controls(self, enabled):
        """Enable/disable bandpass filter controls based on checkbox state."""
        phase_opt_enabled = self.optimize_phase_checkbox.isChecked()
        self.lowcut_spin.setEnabled(phase_opt_enabled and enabled)
        self.highcut_spin.setEnabled(phase_opt_enabled and enabled)
        self.fs_spin.setEnabled(phase_opt_enabled and enabled)
        
    def get_parameters(self):
        """Get the configured parameters as a dictionary."""
        params = {
            'nonlinear_threshold': self.nonlinear_threshold_spin.value(),
            'fallback_to_lowest': self.fallback_checkbox.isChecked(),
            'optimize_phase': self.optimize_phase_checkbox.isChecked(),
            'num_phase_samples': self.num_samples_spin.value(),
            'phase_step': self.phase_step_spin.value()
        }
        
        # Only include bandpass parameters if phase optimization is enabled
        if params['optimize_phase']:
            params['bandpass_params'] = {
                'apply_bandpass': self.apply_bandpass_checkbox.isChecked(),
                'lowcut': self.lowcut_spin.value(),
                'highcut': self.highcut_spin.value(),
                'fs': self.fs_spin.value()
            }
        
        return params
