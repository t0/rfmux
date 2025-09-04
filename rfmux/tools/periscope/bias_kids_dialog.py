"""Dialog for configuring Bias KIDs algorithm parameters."""

from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QVBoxLayout, QFormLayout,
                             QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QLabel)
from PyQt6.QtCore import Qt
import numpy as np


class BiasKidsDialog(QDialog):
    """Dialog for configuring Bias KIDs algorithm parameters."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bias KIDs Configuration")
        self.setModal(True)
        self.setMinimumWidth(400)
        
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
