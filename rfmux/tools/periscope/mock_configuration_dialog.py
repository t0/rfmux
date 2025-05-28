"""
Mock Configuration Dialog for Periscope
======================================

This dialog allows users to configure simulation parameters when running
Periscope in mock mode (connected to localhost/127.0.0.1).
"""

from .utils import QtWidgets, QtCore, QtGui, QDoubleValidator
import re
from rfmux.core import mock_constants


class ScientificDoubleValidator(QDoubleValidator):
    """Validator that accepts scientific notation."""
    
    def validate(self, string, pos):
        # Allow scientific notation
        if re.match(r'^-?(\d+\.?\d*|\d*\.\d+)([eE][+-]?\d+)?$', string) or string == '' or string == '-':
            return (QDoubleValidator.State.Acceptable, string, pos)
        return (QDoubleValidator.State.Invalid, string, pos)


class MockConfigurationDialog(QtWidgets.QDialog):
    """
    Dialog for configuring mock CRS simulation parameters.
    
    Provides UI controls for adjusting:
    - Basic parameters: number of resonators and frequency range
    - Advanced parameters: kinetic inductance, power dependence, Q factors, etc.
    """
    
    def __init__(self, parent=None, current_config=None):
        """
        Initialize the mock configuration dialog.
        
        Args:
            parent: Parent widget
            current_config: Dictionary of current configuration values (optional)
        """
        super().__init__(parent)
        self.setWindowTitle("Mock Mode Configuration")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        # Store current configuration or use defaults
        self.current_config = current_config or {}
        
        # Create UI
        self._create_ui()
        
        # Load current values
        self._load_current_values()
        
    def _create_ui(self):
        """Create the dialog UI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Warning message with theme-appropriate background
        warning_label = QtWidgets.QLabel(
            "⚠️ <b>Periscope is running in MOCK MODE</b><br>"
            "This simulates a CRS board with virtual KID resonators."
        )
        # Use theme-appropriate background color
        palette = self.palette()
        bg_color = palette.color(QtGui.QPalette.ColorRole.Window)
        is_dark = bg_color.lightness() < 128
        
        if is_dark:
            # Dark theme - use a darker orange/amber
            warning_style = "QLabel { background-color: #664422; color: #FFE4B5; padding: 10px; border-radius: 5px; }"
        else:
            # Light theme - use a light amber
            warning_style = "QLabel { background-color: #FFF4E0; color: #663300; padding: 10px; border-radius: 5px; }"
        warning_label.setStyleSheet(warning_style)
        layout.addWidget(warning_label)
        
        # Add spacing
        layout.addSpacing(10)
        
        # Basic parameters (always visible)
        layout.addWidget(self._create_basic_group())
        
        # Add spacing
        layout.addSpacing(10)
        
        # Advanced parameters (collapsible)
        self.advanced_widget = self._create_advanced_widget()
        layout.addWidget(self.advanced_widget)
        
        # Add stretch to push buttons to bottom
        layout.addStretch()
        
        # Dialog buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        # Reset to defaults button
        self.reset_button = QtWidgets.QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self._reset_to_defaults)
        button_layout.addWidget(self.reset_button)
        
        # Add stretch to push OK/Cancel to the right
        button_layout.addStretch()
        
        # OK/Cancel buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._validate_and_accept)
        button_box.rejected.connect(self.reject)
        button_layout.addWidget(button_box)
        
        layout.addLayout(button_layout)
        
    def _create_basic_group(self) -> QtWidgets.QGroupBox:
        """Create the basic parameters group (always visible)."""
        group = QtWidgets.QGroupBox("Basic Parameters")
        layout = QtWidgets.QGridLayout(group)
        
        row = 0
        
        # Number of resonances
        layout.addWidget(QtWidgets.QLabel("Number of Resonances:"), row, 0)
        self.num_resonances_spin = QtWidgets.QSpinBox()
        self.num_resonances_spin.setRange(1, 1000)
        self.num_resonances_spin.setSingleStep(10)
        self.num_resonances_spin.setValue(mock_constants.DEFAULT_NUM_RESONANCES)
        self.num_resonances_spin.setToolTip("Total number of resonances to generate")
        layout.addWidget(self.num_resonances_spin, row, 1)
        
        row += 1
        
        # Frequency range
        layout.addWidget(QtWidgets.QLabel("Frequency Start (GHz):"), row, 0)
        self.freq_start_spin = QtWidgets.QDoubleSpinBox()
        self.freq_start_spin.setRange(0.001, 100.0)
        self.freq_start_spin.setSingleStep(0.001)
        self.freq_start_spin.setDecimals(3)
        self.freq_start_spin.setValue(mock_constants.DEFAULT_FREQ_START / 1e9)  # Convert Hz to GHz
        self.freq_start_spin.setToolTip("Starting frequency for resonance generation")
        layout.addWidget(self.freq_start_spin, row, 1)
        
        layout.addWidget(QtWidgets.QLabel("Frequency End (GHz):"), row, 2)
        self.freq_end_spin = QtWidgets.QDoubleSpinBox()
        self.freq_end_spin.setRange(0.001, 100.0)
        self.freq_end_spin.setSingleStep(0.001)
        self.freq_end_spin.setDecimals(3)
        self.freq_end_spin.setValue(mock_constants.DEFAULT_FREQ_END / 1e9)  # Convert Hz to GHz
        self.freq_end_spin.setToolTip("Ending frequency for resonance generation")
        layout.addWidget(self.freq_end_spin, row, 3)
        
        return group
        
    def _create_advanced_widget(self) -> QtWidgets.QWidget:
        """Create the collapsible advanced parameters widget."""
        # Create a collapsible widget
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Toggle button
        self.advanced_toggle = QtWidgets.QPushButton("▶ Advanced Parameters")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setFlat(True)
        self.advanced_toggle.setStyleSheet("QPushButton { text-align: left; padding: 5px; }")
        self.advanced_toggle.clicked.connect(self._toggle_advanced)
        layout.addWidget(self.advanced_toggle)
        
        # Container for advanced parameters
        self.advanced_container = QtWidgets.QWidget()
        self.advanced_container.setVisible(False)  # Hidden by default
        advanced_layout = QtWidgets.QVBoxLayout(self.advanced_container)
        
        # Add advanced parameter groups
        advanced_layout.addWidget(self._create_kinetic_group())
        advanced_layout.addWidget(self._create_resonance_group())
        advanced_layout.addWidget(self._create_noise_group())
        
        layout.addWidget(self.advanced_container)
        
        return widget
        
    def _toggle_advanced(self):
        """Toggle visibility of advanced parameters."""
        is_visible = self.advanced_container.isVisible()
        self.advanced_container.setVisible(not is_visible)
        
        # Update button text
        if is_visible:
            self.advanced_toggle.setText("▶ Advanced Parameters")
        else:
            self.advanced_toggle.setText("▼ Advanced Parameters")
        
    def _create_kinetic_group(self) -> QtWidgets.QGroupBox:
        """Create the kinetic inductance and power dependence parameters group."""
        group = QtWidgets.QGroupBox("Kinetic Inductance and Power Dependence")
        layout = QtWidgets.QGridLayout(group)
        
        row = 0
        
        # Kinetic inductance fraction
        layout.addWidget(QtWidgets.QLabel("Kinetic Inductance Fraction:"), row, 0)
        self.kinetic_fraction_spin = QtWidgets.QDoubleSpinBox()
        self.kinetic_fraction_spin.setRange(0.0, 1.0)
        self.kinetic_fraction_spin.setSingleStep(0.01)
        self.kinetic_fraction_spin.setDecimals(2)
        self.kinetic_fraction_spin.setValue(mock_constants.DEFAULT_KINETIC_INDUCTANCE_FRACTION)
        self.kinetic_fraction_spin.setToolTip("Fraction of total inductance that's kinetic (0-1)")
        layout.addWidget(self.kinetic_fraction_spin, row, 1)
        
        # Kinetic inductance variation
        layout.addWidget(QtWidgets.QLabel("Kinetic Inductance Variation:"), row, 2)
        self.kinetic_variation_spin = QtWidgets.QDoubleSpinBox()
        self.kinetic_variation_spin.setRange(0.0, 1.0)
        self.kinetic_variation_spin.setSingleStep(0.01)
        self.kinetic_variation_spin.setDecimals(2)
        self.kinetic_variation_spin.setValue(mock_constants.KINETIC_INDUCTANCE_VARIATION)
        self.kinetic_variation_spin.setToolTip("±Variation between resonators (0-1)")
        layout.addWidget(self.kinetic_variation_spin, row, 3)
        
        row += 1
        
        # Frequency shift power law
        layout.addWidget(QtWidgets.QLabel("Frequency Shift Power Law:"), row, 0)
        self.freq_shift_power_spin = QtWidgets.QDoubleSpinBox()
        self.freq_shift_power_spin.setRange(0.1, 5.0)
        self.freq_shift_power_spin.setSingleStep(0.1)
        self.freq_shift_power_spin.setDecimals(1)
        self.freq_shift_power_spin.setValue(mock_constants.FREQUENCY_SHIFT_POWER_LAW)
        self.freq_shift_power_spin.setToolTip("Power law exponent: Δf ∝ P^α")
        layout.addWidget(self.freq_shift_power_spin, row, 1)
        
        # Frequency shift magnitude
        layout.addWidget(QtWidgets.QLabel("Frequency Shift Magnitude:"), row, 2)
        self.freq_shift_mag_edit = QtWidgets.QLineEdit(str(mock_constants.FREQUENCY_SHIFT_MAGNITUDE))
        self.freq_shift_mag_edit.setValidator(ScientificDoubleValidator())
        self.freq_shift_mag_edit.setToolTip("Base magnitude: Δf/f₀ = magnitude * (P/P₀)^α")
        layout.addWidget(self.freq_shift_mag_edit, row, 3)
        
        row += 1
        
        # Power normalization
        layout.addWidget(QtWidgets.QLabel("Power Normalization:"), row, 0)
        self.power_norm_edit = QtWidgets.QLineEdit(str(mock_constants.POWER_NORMALIZATION))
        self.power_norm_edit.setValidator(ScientificDoubleValidator())
        self.power_norm_edit.setToolTip("Reference power level for normalization")
        layout.addWidget(self.power_norm_edit, row, 1)
        
        # Enable bifurcation
        self.bifurcation_check = QtWidgets.QCheckBox("Enable Bifurcation")
        self.bifurcation_check.setChecked(mock_constants.ENABLE_BIFURCATION)
        self.bifurcation_check.setToolTip("Enable self-consistent frequency shift")
        layout.addWidget(self.bifurcation_check, row, 2, 1, 2)
        
        row += 1
        
        # Bifurcation parameters (enabled/disabled based on checkbox)
        layout.addWidget(QtWidgets.QLabel("Bifurcation Iterations:"), row, 0)
        self.bifurcation_iter_spin = QtWidgets.QSpinBox()
        self.bifurcation_iter_spin.setRange(100, 10000)
        self.bifurcation_iter_spin.setSingleStep(100)
        self.bifurcation_iter_spin.setValue(mock_constants.BIFURCATION_ITERATIONS)
        self.bifurcation_iter_spin.setToolTip("Number of iterations for self-consistency")
        layout.addWidget(self.bifurcation_iter_spin, row, 1)
        
        layout.addWidget(QtWidgets.QLabel("Convergence Tolerance:"), row, 2)
        self.bifurcation_tol_edit = QtWidgets.QLineEdit(str(mock_constants.BIFURCATION_CONVERGENCE_TOLERANCE))
        self.bifurcation_tol_edit.setValidator(ScientificDoubleValidator())
        self.bifurcation_tol_edit.setToolTip("Convergence criterion for frequency")
        layout.addWidget(self.bifurcation_tol_edit, row, 3)
        
        row += 1
        
        # Bifurcation damping
        layout.addWidget(QtWidgets.QLabel("Bifurcation Damping Factor:"), row, 0)
        self.bifurcation_damp_spin = QtWidgets.QDoubleSpinBox()
        self.bifurcation_damp_spin.setRange(0.1, 1.0)
        self.bifurcation_damp_spin.setSingleStep(0.1)
        self.bifurcation_damp_spin.setDecimals(1)
        self.bifurcation_damp_spin.setValue(mock_constants.BIFURCATION_DAMPING_FACTOR)
        self.bifurcation_damp_spin.setToolTip("Damping factor to prevent oscillation (0.1-1.0)")
        layout.addWidget(self.bifurcation_damp_spin, row, 1)
        
        row += 1
        
        # Saturation parameters
        layout.addWidget(QtWidgets.QLabel("Saturation Power:"), row, 0)
        self.saturation_power_spin = QtWidgets.QDoubleSpinBox()
        self.saturation_power_spin.setRange(0.001, 1.0)
        self.saturation_power_spin.setSingleStep(0.001)
        self.saturation_power_spin.setDecimals(3)
        self.saturation_power_spin.setValue(mock_constants.SATURATION_POWER)
        self.saturation_power_spin.setToolTip("Power level where nonlinearity saturates")
        layout.addWidget(self.saturation_power_spin, row, 1)
        
        layout.addWidget(QtWidgets.QLabel("Saturation Sharpness:"), row, 2)
        self.saturation_sharp_spin = QtWidgets.QDoubleSpinBox()
        self.saturation_sharp_spin.setRange(0.5, 10.0)
        self.saturation_sharp_spin.setSingleStep(0.1)
        self.saturation_sharp_spin.setDecimals(1)
        self.saturation_sharp_spin.setValue(mock_constants.SATURATION_SHARPNESS)
        self.saturation_sharp_spin.setToolTip("Sharpness of saturation transition")
        layout.addWidget(self.saturation_sharp_spin, row, 3)
        
        # Connect bifurcation checkbox to enable/disable related fields
        self.bifurcation_check.toggled.connect(self._toggle_bifurcation_fields)
        
        return group
        
    def _create_resonance_group(self) -> QtWidgets.QGroupBox:
        """Create the resonance physical parameters group."""
        group = QtWidgets.QGroupBox("Resonance Physical Parameters")
        layout = QtWidgets.QGridLayout(group)
        
        row = 0
        
        # Q factor range
        layout.addWidget(QtWidgets.QLabel("Q Factor Min:"), row, 0)
        self.q_min_spin = QtWidgets.QSpinBox()
        self.q_min_spin.setRange(1000, 1000000)
        self.q_min_spin.setSingleStep(1000)
        self.q_min_spin.setValue(mock_constants.DEFAULT_Q_MIN)
        self.q_min_spin.setToolTip("Minimum Q factor for resonances")
        layout.addWidget(self.q_min_spin, row, 1)
        
        layout.addWidget(QtWidgets.QLabel("Q Factor Max:"), row, 2)
        self.q_max_spin = QtWidgets.QSpinBox()
        self.q_max_spin.setRange(1000, 1000000)
        self.q_max_spin.setSingleStep(1000)
        self.q_max_spin.setValue(mock_constants.DEFAULT_Q_MAX)
        self.q_max_spin.setToolTip("Maximum Q factor for resonances")
        layout.addWidget(self.q_max_spin, row, 3)
        
        row += 1
        
        # Q variation
        layout.addWidget(QtWidgets.QLabel("Q Variation:"), row, 0)
        self.q_variation_spin = QtWidgets.QDoubleSpinBox()
        self.q_variation_spin.setRange(0.0, 1.0)
        self.q_variation_spin.setSingleStep(0.01)
        self.q_variation_spin.setDecimals(2)
        self.q_variation_spin.setValue(mock_constants.Q_VARIATION)
        self.q_variation_spin.setToolTip("±Variation between resonators (0-1)")
        layout.addWidget(self.q_variation_spin, row, 1)
        
        row += 1
        
        # Coupling range
        layout.addWidget(QtWidgets.QLabel("Coupling Min:"), row, 0)
        self.coupling_min_spin = QtWidgets.QDoubleSpinBox()
        self.coupling_min_spin.setRange(0.0, 1.0)
        self.coupling_min_spin.setSingleStep(0.01)
        self.coupling_min_spin.setDecimals(2)
        self.coupling_min_spin.setValue(mock_constants.DEFAULT_COUPLING_MIN)
        self.coupling_min_spin.setToolTip("Minimum coupling (affects resonance depth)")
        layout.addWidget(self.coupling_min_spin, row, 1)
        
        layout.addWidget(QtWidgets.QLabel("Coupling Max:"), row, 2)
        self.coupling_max_spin = QtWidgets.QDoubleSpinBox()
        self.coupling_max_spin.setRange(0.0, 1.0)
        self.coupling_max_spin.setSingleStep(0.01)
        self.coupling_max_spin.setDecimals(2)
        self.coupling_max_spin.setValue(mock_constants.DEFAULT_COUPLING_MAX)
        self.coupling_max_spin.setToolTip("Maximum coupling (affects resonance depth)")
        layout.addWidget(self.coupling_max_spin, row, 3)
        
        return group
        
    def _create_noise_group(self) -> QtWidgets.QGroupBox:
        """Create the amplitude and noise parameters group."""
        group = QtWidgets.QGroupBox("Amplitude and Noise Parameters")
        layout = QtWidgets.QGridLayout(group)
        
        row = 0
        
        # Base noise level
        layout.addWidget(QtWidgets.QLabel("Multiplicative Noise Level:"), row, 0)
        self.base_noise_spin = QtWidgets.QDoubleSpinBox()
        self.base_noise_spin.setRange(0.0, 0.1)
        self.base_noise_spin.setSingleStep(0.0001)
        self.base_noise_spin.setDecimals(4)
        self.base_noise_spin.setValue(mock_constants.BASE_NOISE_LEVEL)
        self.base_noise_spin.setToolTip("Base noise level (relative to S21)")
        layout.addWidget(self.base_noise_spin, row, 1)
        
        # Amplitude noise coupling
        layout.addWidget(QtWidgets.QLabel("Amplitude Noise Coupling:"), row, 2)
        self.amp_noise_spin = QtWidgets.QDoubleSpinBox()
        self.amp_noise_spin.setRange(0.0, 0.1)
        self.amp_noise_spin.setSingleStep(0.001)
        self.amp_noise_spin.setDecimals(3)
        self.amp_noise_spin.setValue(mock_constants.AMPLITUDE_NOISE_COUPLING)
        self.amp_noise_spin.setToolTip("How noise scales with amplitude")
        layout.addWidget(self.amp_noise_spin, row, 3)
        
        row += 1
        
        # UDP noise level
        layout.addWidget(QtWidgets.QLabel("Additive Noise Level:"), row, 0)
        self.udp_noise_spin = QtWidgets.QDoubleSpinBox()
        self.udp_noise_spin.setRange(0.0, 100.0)
        self.udp_noise_spin.setSingleStep(1.0)
        self.udp_noise_spin.setDecimals(1)
        self.udp_noise_spin.setValue(mock_constants.UDP_NOISE_LEVEL)
        self.udp_noise_spin.setToolTip("ADC noise level (in counts)")
        layout.addWidget(self.udp_noise_spin, row, 1)
        
        return group
        
    def _toggle_bifurcation_fields(self, enabled: bool):
        """Enable/disable bifurcation-related fields based on checkbox state."""
        self.bifurcation_iter_spin.setEnabled(enabled)
        self.bifurcation_tol_edit.setEnabled(enabled)
        self.bifurcation_damp_spin.setEnabled(enabled)
        
    def _reset_to_defaults(self):
        """Reset all parameters to default values from mock_constants."""
        # Basic parameters
        self.num_resonances_spin.setValue(mock_constants.DEFAULT_NUM_RESONANCES)
        self.freq_start_spin.setValue(mock_constants.DEFAULT_FREQ_START / 1e9)  # Convert Hz to GHz
        self.freq_end_spin.setValue(mock_constants.DEFAULT_FREQ_END / 1e9)  # Convert Hz to GHz
        
        # Kinetic inductance parameters
        self.kinetic_fraction_spin.setValue(mock_constants.DEFAULT_KINETIC_INDUCTANCE_FRACTION)
        self.kinetic_variation_spin.setValue(mock_constants.KINETIC_INDUCTANCE_VARIATION)
        
        # Power dependence parameters
        self.freq_shift_power_spin.setValue(mock_constants.FREQUENCY_SHIFT_POWER_LAW)
        self.freq_shift_mag_edit.setText(str(mock_constants.FREQUENCY_SHIFT_MAGNITUDE))
        self.power_norm_edit.setText(str(mock_constants.POWER_NORMALIZATION))
        
        # Bifurcation parameters
        self.bifurcation_check.setChecked(mock_constants.ENABLE_BIFURCATION)
        self.bifurcation_iter_spin.setValue(mock_constants.BIFURCATION_ITERATIONS)
        self.bifurcation_tol_edit.setText(str(mock_constants.BIFURCATION_CONVERGENCE_TOLERANCE))
        self.bifurcation_damp_spin.setValue(mock_constants.BIFURCATION_DAMPING_FACTOR)
        
        # Saturation parameters
        self.saturation_power_spin.setValue(mock_constants.SATURATION_POWER)
        self.saturation_sharp_spin.setValue(mock_constants.SATURATION_SHARPNESS)
        
        # Q factor parameters
        self.q_min_spin.setValue(mock_constants.DEFAULT_Q_MIN)
        self.q_max_spin.setValue(mock_constants.DEFAULT_Q_MAX)
        self.q_variation_spin.setValue(mock_constants.Q_VARIATION)
        
        # Coupling parameters
        self.coupling_min_spin.setValue(mock_constants.DEFAULT_COUPLING_MIN)
        self.coupling_max_spin.setValue(mock_constants.DEFAULT_COUPLING_MAX)
        
        # Noise parameters
        self.base_noise_spin.setValue(mock_constants.BASE_NOISE_LEVEL)
        self.amp_noise_spin.setValue(mock_constants.AMPLITUDE_NOISE_COUPLING)
        self.udp_noise_spin.setValue(mock_constants.UDP_NOISE_LEVEL)
        
    def _load_current_values(self):
        """Load current configuration values into the UI."""
        if not self.current_config:
            return
            
        # Basic parameters
        if 'num_resonances' in self.current_config:
            self.num_resonances_spin.setValue(self.current_config['num_resonances'])
        if 'freq_start' in self.current_config:
            self.freq_start_spin.setValue(self.current_config['freq_start'] / 1e9)  # Convert to GHz
        if 'freq_end' in self.current_config:
            self.freq_end_spin.setValue(self.current_config['freq_end'] / 1e9)  # Convert to GHz
            
        # Kinetic inductance parameters
        if 'kinetic_inductance_fraction' in self.current_config:
            self.kinetic_fraction_spin.setValue(self.current_config['kinetic_inductance_fraction'])
        if 'kinetic_inductance_variation' in self.current_config:
            self.kinetic_variation_spin.setValue(self.current_config['kinetic_inductance_variation'])
            
        # Power dependence parameters
        if 'frequency_shift_power_law' in self.current_config:
            self.freq_shift_power_spin.setValue(self.current_config['frequency_shift_power_law'])
        if 'frequency_shift_magnitude' in self.current_config:
            self.freq_shift_mag_edit.setText(str(self.current_config['frequency_shift_magnitude']))
        if 'power_normalization' in self.current_config:
            self.power_norm_edit.setText(str(self.current_config['power_normalization']))
            
        # Bifurcation parameters
        if 'enable_bifurcation' in self.current_config:
            self.bifurcation_check.setChecked(self.current_config['enable_bifurcation'])
        if 'bifurcation_iterations' in self.current_config:
            self.bifurcation_iter_spin.setValue(self.current_config['bifurcation_iterations'])
        if 'bifurcation_convergence_tolerance' in self.current_config:
            self.bifurcation_tol_edit.setText(str(self.current_config['bifurcation_convergence_tolerance']))
        if 'bifurcation_damping_factor' in self.current_config:
            self.bifurcation_damp_spin.setValue(self.current_config['bifurcation_damping_factor'])
            
        # Saturation parameters
        if 'saturation_power' in self.current_config:
            self.saturation_power_spin.setValue(self.current_config['saturation_power'])
        if 'saturation_sharpness' in self.current_config:
            self.saturation_sharp_spin.setValue(self.current_config['saturation_sharpness'])
            
        # Q factor parameters
        if 'q_min' in self.current_config:
            self.q_min_spin.setValue(self.current_config['q_min'])
        if 'q_max' in self.current_config:
            self.q_max_spin.setValue(self.current_config['q_max'])
        if 'q_variation' in self.current_config:
            self.q_variation_spin.setValue(self.current_config['q_variation'])
            
        # Coupling parameters
        if 'coupling_min' in self.current_config:
            self.coupling_min_spin.setValue(self.current_config['coupling_min'])
        if 'coupling_max' in self.current_config:
            self.coupling_max_spin.setValue(self.current_config['coupling_max'])
            
        # Noise parameters
        if 'base_noise_level' in self.current_config:
            self.base_noise_spin.setValue(self.current_config['base_noise_level'])
        if 'amplitude_noise_coupling' in self.current_config:
            self.amp_noise_spin.setValue(self.current_config['amplitude_noise_coupling'])
        if 'udp_noise_level' in self.current_config:
            self.udp_noise_spin.setValue(self.current_config['udp_noise_level'])
            
    def _validate_and_accept(self):
        """Validate inputs before accepting the dialog."""
        # Validate frequency range
        if self.freq_start_spin.value() >= self.freq_end_spin.value():
            QtWidgets.QMessageBox.warning(self, "Invalid Input", 
                               "Frequency Start must be less than Frequency End.")
            return
            
        # Validate Q factor range
        if self.q_min_spin.value() >= self.q_max_spin.value():
            QtWidgets.QMessageBox.warning(self, "Invalid Input", 
                               "Q Factor Min must be less than Q Factor Max.")
            return
            
        # Validate coupling range
        if self.coupling_min_spin.value() >= self.coupling_max_spin.value():
            QtWidgets.QMessageBox.warning(self, "Invalid Input", 
                               "Coupling Min must be less than Coupling Max.")
            return
            
        # Validate scientific notation fields
        try:
            float(self.freq_shift_mag_edit.text())
            float(self.power_norm_edit.text())
            float(self.bifurcation_tol_edit.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", 
                               "Please enter valid numeric values for all fields.")
            return
            
        self.accept()
        
    def get_configuration(self) -> dict:
        """
        Return a dictionary of all parameter values.
        
        Returns:
            dict: Configuration parameters with keys matching mock_constants.py
        """
        return {
            # Basic parameters
            'num_resonances': self.num_resonances_spin.value(),
            'freq_start': self.freq_start_spin.value() * 1e9,  # Convert GHz to Hz
            'freq_end': self.freq_end_spin.value() * 1e9,      # Convert GHz to Hz
            
            # Kinetic inductance parameters
            'kinetic_inductance_fraction': self.kinetic_fraction_spin.value(),
            'kinetic_inductance_variation': self.kinetic_variation_spin.value(),
            
            # Power dependence parameters
            'frequency_shift_power_law': self.freq_shift_power_spin.value(),
            'frequency_shift_magnitude': float(self.freq_shift_mag_edit.text()),
            'power_normalization': float(self.power_norm_edit.text()),
            
            # Bifurcation parameters
            'enable_bifurcation': self.bifurcation_check.isChecked(),
            'bifurcation_iterations': self.bifurcation_iter_spin.value(),
            'bifurcation_convergence_tolerance': float(self.bifurcation_tol_edit.text()),
            'bifurcation_damping_factor': self.bifurcation_damp_spin.value(),
            
            # Saturation parameters
            'saturation_power': self.saturation_power_spin.value(),
            'saturation_sharpness': self.saturation_sharp_spin.value(),
            
            # Q factor parameters
            'q_min': self.q_min_spin.value(),
            'q_max': self.q_max_spin.value(),
            'q_variation': self.q_variation_spin.value(),
            
            # Coupling parameters
            'coupling_min': self.coupling_min_spin.value(),
            'coupling_max': self.coupling_max_spin.value(),
            
            # Noise parameters
            'base_noise_level': self.base_noise_spin.value(),
            'amplitude_noise_coupling': self.amp_noise_spin.value(),
            'udp_noise_level': self.udp_noise_spin.value()
        }
