"""
Mock Configuration Dialog for Periscope (Unified SoT)
=====================================================

This dialog exposes only the parameters defined in rfmux.mock.config
(Single Source of Truth). It returns a configuration dict that can be passed
directly to MockCRS.generate_resonators().

Groups:
- Basic: number of resonances, frequency range (GHz), random seed
- Bias: automatic bias enable and bias amplitude (normalized)
- Advanced (collapsible):
  - Physics: T (K), Popt (W)
  - Circuit: Lg (H), Cc (F), L_junk (H), C_variation, Cc_variation
  - Readout: Vin (V), input_atten_dB, system_termination (Ω), ZLNA (Ω), GLNA (dB)
  - Noise: nqp_noise_enabled, nqp_noise_std_factor, udp_noise_level (counts)
  - Convergence: convergence_tolerance
"""

from .utils import QtWidgets, QtCore, QtGui
import re
import math
import numpy as np
from rfmux.mock import config as mc
from rfmux.mr_resonator.mr_complex_resonator import MR_complex_resonator
from rfmux.mr_resonator.mr_lekid import MR_LEKID
from . import settings
from .custom_material_dialog import CustomMaterialDialog, ManageCustomMaterialsDialog


class ScientificDoubleValidator(QtGui.QDoubleValidator):
    """Validator that accepts scientific notation."""
    def validate(self, string, pos):
        if re.match(r'^-?(\d+\.?\d*|\d*\.\d+)([eE][+-]?\d+)?$', string) or string == '' or string == '-':
            return (QtGui.QValidator.State.Acceptable, string, pos)
        return (QtGui.QValidator.State.Invalid, string, pos)


class MockConfigurationDialog(QtWidgets.QDialog):
    """
    Dialog for configuring mock CRS simulation parameters based on SoT (mock_config).
    """

    def __init__(self, parent=None, current_config=None):
        super().__init__(parent)
        self.setWindowTitle("Mock Mode Configuration")
        self.setModal(True)
        self.setMinimumWidth(1000)

        # Store current configuration or use defaults
        self.current_config = current_config or {}

        # Create UI
        self._create_ui()

        # Load current or default values
        self._load_current_values()

        # Connect signals for interactions
        self._connect_signals()

        # Trigger initial derived parameter update
        self._update_all_derived()

    def _connect_signals(self):
        """Connect signals for live update of derived parameters."""
        # Physics mode inputs
        for widget in [self.T_edit, self.Popt_edit, self.width_edit, 
                       self.thickness_edit, self.length_edit]:
            widget.textChanged.connect(self._update_all_derived)
        
        # Circuit design
        for widget in [self.Lg_edit, self.L_junk_edit, self.Cc_edit,
                       self.Vin_edit, self.input_atten_edit, self.ZLNA_edit]:
            widget.textChanged.connect(self._update_all_derived)
            
        self.freq_start_spin.valueChanged.connect(self._update_all_derived)

    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Warning message with theme-appropriate background
        warning_label = QtWidgets.QLabel(
            "⚠️ <b>Periscope is running in MOCK MODE</b><br>"
            "This simulates a CRS board with virtual KID resonators."
        )
        palette = self.palette()
        bg_color = palette.color(QtGui.QPalette.ColorRole.Window)
        is_dark = bg_color.lightness() < 128
        if is_dark:
            style = "QLabel { background-color: #664422; color: #FFE4B5; padding: 10px; border-radius: 5px; }"
        else:
            style = "QLabel { background-color: #FFF4E0; color: #663300; padding: 10px; border-radius: 5px; }"
        warning_label.setStyleSheet(style)
        layout.addWidget(warning_label)

        layout.addSpacing(8)

        # Basic and Bias parameters
        layout.addWidget(self._create_basic_group())
        layout.addSpacing(8)
        layout.addWidget(self._create_bias_group())

        layout.addSpacing(8)

        # Advanced parameters (collapsible)
        self.advanced_widget = self._create_advanced_widget()
        layout.addWidget(self.advanced_widget)

        layout.addStretch()

        # Buttons row
        button_layout = QtWidgets.QHBoxLayout()
        self.reset_button = QtWidgets.QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self._reset_to_defaults)
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._validate_and_accept)
        button_box.rejected.connect(self.reject)
        button_layout.addWidget(button_box)
        layout.addLayout(button_layout)

    def _create_basic_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Basic")
        layout = QtWidgets.QGridLayout(group)
        row = 0

        # Number of resonances
        layout.addWidget(QtWidgets.QLabel("Number of Resonances:"), row, 0)
        self.num_resonances_spin = QtWidgets.QSpinBox()
        self.num_resonances_spin.setRange(1, 10000)
        self.num_resonances_spin.setSingleStep(1)
        self.num_resonances_spin.setToolTip("Total number of resonances to generate")
        layout.addWidget(self.num_resonances_spin, row, 1)

        # Frequency start/end (GHz)
        layout.addWidget(QtWidgets.QLabel("Frequency Start (GHz):"), row, 2)
        self.freq_start_spin = QtWidgets.QDoubleSpinBox()
        self.freq_start_spin.setRange(0.001, 100.0)
        self.freq_start_spin.setSingleStep(0.001)
        self.freq_start_spin.setDecimals(3)
        self.freq_start_spin.setToolTip("Start frequency for resonance generation [GHz]")
        layout.addWidget(self.freq_start_spin, row, 3)

        row += 1
        layout.addWidget(QtWidgets.QLabel("Frequency End (GHz):"), row, 0)
        self.freq_end_spin = QtWidgets.QDoubleSpinBox()
        self.freq_end_spin.setRange(0.001, 100.0)
        self.freq_end_spin.setSingleStep(0.001)
        self.freq_end_spin.setDecimals(3)
        self.freq_end_spin.setToolTip("End frequency for resonance generation [GHz]")
        layout.addWidget(self.freq_end_spin, row, 1)

        # Random seed (optional)
        layout.addWidget(QtWidgets.QLabel("Random Seed:"), row, 2)
        self.random_seed_edit = QtWidgets.QLineEdit()
        self.random_seed_edit.setValidator(QtGui.QIntValidator(0, 2**31 - 1, self.random_seed_edit))
        self.random_seed_edit.setToolTip("Set a seed for reproducible layout (leave blank for random)")
        layout.addWidget(self.random_seed_edit, row, 3)

        return group

    def _create_bias_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Bias")
        layout = QtWidgets.QGridLayout(group)
        row = 0

        self.auto_bias_check = QtWidgets.QCheckBox("Automatically bias KIDs")
        self.auto_bias_check.setToolTip("Configure tones on generated resonator frequencies")
        layout.addWidget(self.auto_bias_check, row, 0, 1, 2)

        layout.addWidget(QtWidgets.QLabel("Bias Amplitude (normalized):"), row, 2)
        self.bias_amplitude_spin = QtWidgets.QDoubleSpinBox()
        self.bias_amplitude_spin.setRange(0.0001, 1.0)
        self.bias_amplitude_spin.setSingleStep(0.001)
        self.bias_amplitude_spin.setDecimals(4)
        self.bias_amplitude_spin.setToolTip("Amplitude for automatic biasing (e.g., 0.01 ≈ -40 dBm)")
        layout.addWidget(self.bias_amplitude_spin, row, 3)

        self.auto_bias_check.toggled.connect(self.bias_amplitude_spin.setEnabled)
        return group

    def _create_advanced_widget(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(widget)
        vbox.setContentsMargins(0, 0, 0, 0)

        self.advanced_toggle = QtWidgets.QPushButton("▶ Advanced Parameters")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setFlat(True)
        self.advanced_toggle.setStyleSheet("QPushButton { text-align: left; padding: 5px; }")
        self.advanced_toggle.setToolTip("Show or hide advanced physics, circuit, readout, noise, and solver settings.")
        self.advanced_toggle.clicked.connect(self._toggle_advanced)
        vbox.addWidget(self.advanced_toggle)

        self.advanced_container = QtWidgets.QWidget()
        self.advanced_container.setVisible(False)
        grid = QtWidgets.QHBoxLayout(self.advanced_container)

        # Left column
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self._create_mkids_group())
        left.addStretch()

        # Right column
        right = QtWidgets.QVBoxLayout()
        right.addWidget(self._create_readout_group())
        right.addWidget(self._create_pulse_injection_group())
        right.addWidget(self._create_noise_group())
        right.addWidget(self._create_simulation_realism_group())
        right.addStretch()

        grid.addLayout(left)
        grid.addLayout(right)
        vbox.addWidget(self.advanced_container)
        return widget

    def _toggle_advanced(self):
        vis = self.advanced_container.isVisible()
        self.advanced_container.setVisible(not vis)
        self.advanced_toggle.setText("▼ Advanced Parameters" if not vis else "▶ Advanced Parameters")

    def _create_mkids_group(self) -> QtWidgets.QGroupBox:
        """Create unified MKIDs section with physics-driven parameters."""
        group = QtWidgets.QGroupBox("MKIDs (Microwave Kinetic Inductance Detectors)")
        layout = QtWidgets.QVBoxLayout(group)
        
        # Physics-driven content
        physics_page = self._create_physics_driven_page()
        layout.addWidget(physics_page)
        
        # Circuit Design Parameters (shared between both modes, always visible)
        circuit_group = QtWidgets.QGroupBox("Circuit Design Parameters")
        circuit_layout = QtWidgets.QGridLayout(circuit_group)
        
        circuit_layout.addWidget(QtWidgets.QLabel("Lg (geometric, nH):"), 0, 0)
        self.Lg_edit = QtWidgets.QLineEdit()
        self.Lg_edit.setValidator(ScientificDoubleValidator())
        self.Lg_edit.setToolTip("Geometric inductance in nanohenries. Fixed by fabrication geometry.")
        circuit_layout.addWidget(self.Lg_edit, 0, 1)
        
        circuit_layout.addWidget(QtWidgets.QLabel("L_junk (nH):"), 0, 2)
        self.L_junk_edit = QtWidgets.QLineEdit()
        self.L_junk_edit.setValidator(ScientificDoubleValidator())
        self.L_junk_edit.setToolTip("Parasitic inductance in nH. Dilutes α_k = Lk/(Lk+Lg+L_junk).")
        circuit_layout.addWidget(self.L_junk_edit, 0, 3)
        
        circuit_layout.addWidget(QtWidgets.QLabel("Cc (fF):"), 1, 0)
        self.Cc_edit = QtWidgets.QLineEdit()
        self.Cc_edit.setValidator(ScientificDoubleValidator())
        self.Cc_edit.setToolTip("Coupling capacitor in femtofarads. Controls coupling strength.")
        circuit_layout.addWidget(self.Cc_edit, 1, 1)
        
        circuit_layout.addWidget(QtWidgets.QLabel("C variation:"), 1, 2)
        self.C_variation_edit = QtWidgets.QLineEdit()
        self.C_variation_edit.setValidator(ScientificDoubleValidator())
        self.C_variation_edit.setToolTip("Fractional σ of capacitance variation (e.g., 0.01 = 1%).")
        circuit_layout.addWidget(self.C_variation_edit, 1, 3)
        
        circuit_layout.addWidget(QtWidgets.QLabel("Cc variation:"), 2, 0)
        self.Cc_variation_edit = QtWidgets.QLineEdit()
        self.Cc_variation_edit.setValidator(ScientificDoubleValidator())
        self.Cc_variation_edit.setToolTip("Fractional σ of coupling capacitor variation.")
        circuit_layout.addWidget(self.Cc_variation_edit, 2, 1)
        
        layout.addWidget(circuit_group)
        
        # Derived Properties - combined two-column layout
        derived_group = QtWidgets.QGroupBox("Derived Properties (live-updated)")
        derived_main_layout = QtWidgets.QHBoxLayout(derived_group)
        
        # Left column - physics derived values
        left_form = QtWidgets.QFormLayout()
        left_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        
        self.nqp_label = QtWidgets.QLabel("—")
        self.nqp_label.setStyleSheet("color: #1E90FF; font-family: monospace;")
        self.nqp_label.setToolTip("Quasiparticle density. Increases with T and Popt.")
        left_form.addRow("nqp (µm⁻³):", self.nqp_label)
        
        self.sigma1_label = QtWidgets.QLabel("—")
        self.sigma1_label.setStyleSheet("color: #1E90FF; font-family: monospace;")
        self.sigma1_label.setToolTip("Real part of complex conductivity (related to loss).")
        left_form.addRow("σ₁ (S/m):", self.sigma1_label)
        
        self.sigma2_label = QtWidgets.QLabel("—")
        self.sigma2_label.setStyleSheet("color: #1E90FF; font-family: monospace;")
        self.sigma2_label.setToolTip("Imaginary part of complex conductivity (related to kinetic inductance).")
        left_form.addRow("σ₂ (S/m):", self.sigma2_label)
        
        self.lk_square_label = QtWidgets.QLabel("—")
        self.lk_square_label.setStyleSheet("color: #1E90FF; font-family: monospace;")
        self.lk_square_label.setToolTip("Kinetic inductance per square. Material property at given T, Popt.")
        left_form.addRow("Lk□ (pH/□):", self.lk_square_label)
        
        self.lk_label = QtWidgets.QLabel("—")
        self.lk_label.setStyleSheet("color: #1E90FF; font-family: monospace;")
        self.lk_label.setToolTip("Total kinetic inductance = Lk□ × (length/width).")
        left_form.addRow("Lk (nH):", self.lk_label)
        
        self.r_label = QtWidgets.QLabel("—")
        self.r_label.setStyleSheet("color: #1E90FF; font-family: monospace;")
        self.r_label.setToolTip("Total series resistance scaled from surface resistance.")
        left_form.addRow("R (mΩ):", self.r_label)
        
        # Right column - circuit derived values
        right_form = QtWidgets.QFormLayout()
        right_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        
        self.ltotal_label = QtWidgets.QLabel("—")
        self.ltotal_label.setStyleSheet("color: #1E90FF; font-family: monospace;")
        self.ltotal_label.setToolTip("Total inductance = Lk + Lg + L_junk. Determines resonance frequency.")
        right_form.addRow("L_total (nH):", self.ltotal_label)
        
        self.alpha_k_label = QtWidgets.QLabel("—")
        self.alpha_k_label.setStyleSheet("color: #1E90FF; font-family: monospace;")
        self.alpha_k_label.setToolTip("Kinetic inductance fraction = Lk / L_total. Determines responsivity.")
        right_form.addRow("α_k:", self.alpha_k_label)
        
        self.qr_label = QtWidgets.QLabel("—")
        self.qr_label.setStyleSheet("color: #1E90FF; font-family: monospace;")
        self.qr_label.setToolTip("Loaded quality factor. Limited by both internal loss and coupling.")
        right_form.addRow("Q_r (loaded):", self.qr_label)
        
        self.qi_label = QtWidgets.QLabel("—")
        self.qi_label.setStyleSheet("color: #1E90FF; font-family: monospace;")
        self.qi_label.setToolTip("Internal quality factor. Limited by loss only.")
        right_form.addRow("Q_i (internal):", self.qi_label)
        
        self.qc_label = QtWidgets.QLabel("—")
        self.qc_label.setStyleSheet("color: #1E90FF; font-family: monospace;")
        self.qc_label.setToolTip("Coupling quality factor. Higher Qc = weaker coupling.")
        right_form.addRow("Q_c (coupling):", self.qc_label)
        
        # Add columns to main layout
        derived_main_layout.addLayout(left_form)
        derived_main_layout.addLayout(right_form)
        
        layout.addWidget(derived_group)
        
        return group
    
    def _create_physics_driven_page(self) -> QtWidgets.QWidget:
        """Create physics-driven mode page with geometry inputs and derived properties."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        
        # Material & Operating Point subsection
        mat_group = QtWidgets.QGroupBox("Material and Operating Point")
        mat_layout = QtWidgets.QGridLayout(mat_group)
        
        # Material combo with manage button
        mat_layout.addWidget(QtWidgets.QLabel("Material:"), 0, 0)
        
        material_h_layout = QtWidgets.QHBoxLayout()
        self.material_combo = QtWidgets.QComboBox()
        self.material_combo.setToolTip("Superconductor material. Determines Tc, N₀, τ₀, σN constants.")
        self.material_combo.currentTextChanged.connect(self._on_material_changed)
        material_h_layout.addWidget(self.material_combo, 1)
        
        # Manage materials button
        self.manage_materials_btn = QtWidgets.QPushButton("⚙ Manage...")
        self.manage_materials_btn.setToolTip("Add, edit, or delete custom materials")
        self.manage_materials_btn.clicked.connect(self._manage_materials)
        self.manage_materials_btn.setMaximumWidth(100)
        material_h_layout.addWidget(self.manage_materials_btn)
        
        mat_layout.addLayout(material_h_layout, 0, 1, 1, 3)
        
        # Material constants label
        self.mat_constants_label = QtWidgets.QLabel("Tc=1.2K, N₀=1.72×10¹⁰ µm⁻³eV⁻¹, τ₀=438ns")
        self.mat_constants_label.setStyleSheet("color: gray; font-size: 9pt;")
        self.mat_constants_label.setToolTip(
            "<b>Material Constants:</b><br>"
            "• <b>Tc</b> = Critical temperature<br>"
            "• <b>N₀</b> = Density of states at Fermi level<br>"
            "• <b>τ₀</b> = Quasiparticle recombination time at Tc"
        )
        mat_layout.addWidget(self.mat_constants_label, 1, 0, 1, 4)
        
        # T and Popt
        mat_layout.addWidget(QtWidgets.QLabel("Temperature (K):"), 2, 0)
        self.T_edit = QtWidgets.QLineEdit()
        self.T_edit.setValidator(ScientificDoubleValidator())
        self.T_edit.setToolTip("Operating temperature in Kelvin. Lower T → lower nqp → higher Lk → higher Q.")
        mat_layout.addWidget(self.T_edit, 2, 1)
        
        mat_layout.addWidget(QtWidgets.QLabel("Optical Power (W):"), 2, 2)
        self.Popt_edit = QtWidgets.QLineEdit()
        self.Popt_edit.setValidator(ScientificDoubleValidator())
        self.Popt_edit.setToolTip("Incident optical power in Watts. Higher Popt → higher nqp → lower Q.")
        mat_layout.addWidget(self.Popt_edit, 2, 3)
        
        layout.addWidget(mat_group)
        
        # Load materials into combo box
        self._refresh_material_list()
        
        # Geometry subsection
        geom_group = QtWidgets.QGroupBox("Geometry")
        geom_layout = QtWidgets.QGridLayout(geom_group)
        
        # Width, thickness, length inputs
        geom_layout.addWidget(QtWidgets.QLabel("Width (µm):"), 0, 0)
        self.width_edit = QtWidgets.QLineEdit("2.0")
        self.width_edit.setValidator(ScientificDoubleValidator())
        self.width_edit.setToolTip("Strip width in micrometers. Affects number of squares.")
        geom_layout.addWidget(self.width_edit, 0, 1)
        
        geom_layout.addWidget(QtWidgets.QLabel("Thickness (nm):"), 0, 2)
        self.thickness_edit = QtWidgets.QLineEdit("30")
        self.thickness_edit.setValidator(ScientificDoubleValidator())
        self.thickness_edit.setToolTip("Film thickness in nanometers. Thicker films reduce surface impedance.")
        geom_layout.addWidget(self.thickness_edit, 0, 3)
        
        geom_layout.addWidget(QtWidgets.QLabel("Length (µm):"), 1, 0)
        self.length_edit = QtWidgets.QLineEdit("9000")
        self.length_edit.setValidator(ScientificDoubleValidator())
        self.length_edit.setToolTip("Strip length in micrometers. Longer → higher Lk and R.")
        geom_layout.addWidget(self.length_edit, 1, 1)
        
        # Volume display (calculated)
        self.volume_label = QtWidgets.QLabel("Volume: 540 µm³")
        self.volume_label.setStyleSheet("color: #1E90FF; font-weight: bold;")
        geom_layout.addWidget(self.volume_label, 1, 2, 1, 2)
        
        layout.addWidget(geom_group)
        
        return page
    
    def _refresh_material_list(self):
        """Populate material combo with built-in + custom materials."""
        current = self.material_combo.currentText()
        self.material_combo.blockSignals(True)  # Prevent triggering _on_material_changed during refresh
        self.material_combo.clear()
        
        # Built-in materials
        self.material_combo.addItem("Al (Aluminum)")
        
        # Custom materials
        customs = settings.get_custom_materials()
        for name in sorted(customs.keys()):
            self.material_combo.addItem(name)
        
        # Restore selection or default to Al
        idx = self.material_combo.findText(current)
        if idx >= 0:
            self.material_combo.setCurrentIndex(idx)
        else:
            self.material_combo.setCurrentIndex(0)
        
        self.material_combo.blockSignals(False)
        # Trigger update for current selection
        self._on_material_changed(self.material_combo.currentText())
    
    def _on_material_changed(self, material_text: str):
        """Update material constants label when material changes."""
        if not material_text:
            return
        
        # Extract material name (remove description if present)
        mat_name = material_text.split()[0] if material_text else "Al"
        
        try:
            props = settings.get_material_properties(mat_name)
            Tc = props['Tc']
            N0 = props['N0']
            tau0_ns = props['tau0'] * 1e9  # s to ns
            
            # Display Rs and thickness_ref if available
            Rs = props.get('Rs', None)
            thickness_ref = props.get('thickness_ref', None)
            
            if Rs is not None and thickness_ref is not None:
                self.mat_constants_label.setText(
                    f"Tc={Tc:.3g}K, N₀={N0:.3g} µm⁻³eV⁻¹, τ₀={tau0_ns:.3g}ns, "
                    f"Rs={Rs:.3g}Ω/□ @{thickness_ref:.0f}nm"
                )
            else:
                # Legacy materials without Rs
                self.mat_constants_label.setText(
                    f"Tc={Tc:.3g}K, N₀={N0:.3g} µm⁻³eV⁻¹, τ₀={tau0_ns:.3g}ns"
                )
        except ValueError:
            self.mat_constants_label.setText("Material properties not found")
        
        # Trigger recalculation
        self._update_all_derived()
    
    def _manage_materials(self):
        """Open dialog to manage custom materials."""
        dialog = ManageCustomMaterialsDialog(parent=self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            # Refresh material list after dialog closes
            self._refresh_material_list()
    
    def _update_all_derived(self):
        """Update all derived parameters based on physics-driven inputs."""
        try:
            # Physics mode: calculate from T, Popt, geometry
            try:
                T = float(self.T_edit.text())
                Popt = float(self.Popt_edit.text())
                width_um = float(self.width_edit.text())
                thickness_nm = float(self.thickness_edit.text())
                length_um = float(self.length_edit.text())
            except ValueError:
                return
            
            # Convert to SI
            width = width_um * 1e-6
            thickness = thickness_nm * 1e-9
            length = length_um * 1e-6
            
            # Calculate volume
            volume_um3 = width_um * (thickness_nm / 1000) * length_um
            self.volume_label.setText(f"Volume: {volume_um3:.1f} µm³")
            
            # Get frequency
            freq = self.freq_start_spin.value() * 1e9
            if freq <= 0:
                return
            
            try:
                Lg = float(self.Lg_edit.text()) * 1e-9
                L_junk = float(self.L_junk_edit.text()) * 1e-9
                Cc = float(self.Cc_edit.text()) * 1e-15
                Vin = float(self.Vin_edit.text())
                input_atten_dB = float(self.input_atten_edit.text())
                ZLNA = float(self.ZLNA_edit.text())
            except ValueError:
                return
            
            # Get material properties
            mat_text = self.material_combo.currentText()
            mat_name = mat_text.split()[0] if mat_text else "Al"
            try:
                mat_props = settings.get_material_properties(mat_name)
            except ValueError:
                # If material not found, use Al defaults
                mat_props = settings.get_material_properties("Al")
            
            # Create reference resonator with geometry and custom material properties
            # MR_complex_resonator now accepts explicit N0, tau0, Tc, sigmaN parameters
            # N0 unit conversion: mat_props['N0'] is in µm⁻³eV⁻¹, internal uses µm⁻³J⁻¹
            # Correct conversion: N0_J⁻¹ = N0_eV⁻¹ / (1.602e-19 J/eV)
            ref_params = {
                'T': T, 'Popt': Popt, 'C': 1e-12, 'Cc': Cc, 'fix_Lg': Lg,
                'L_junk': L_junk, 'Vin': Vin, 'input_atten_dB': input_atten_dB,
                'base_readout_f': freq, 'verbose': False,
                'ZLNA': complex(ZLNA, 0), 'GLNA': 1.0,
                'width': width, 'thickness': thickness, 'length': length,
                # Material properties - pass directly (no post-hoc override needed)
                'Tc': mat_props['Tc'],
                'N0': mat_props['N0'] / 1.602e-19,  # Convert µm⁻³eV⁻¹ → µm⁻³J⁻¹
                'tau0': mat_props['tau0'],
                'sigmaN': mat_props['sigmaN']
            }
            ref_res = MR_complex_resonator(**ref_params)
            
            # Extract intermediate values
            nqp = ref_res.calc_nqp()
            sigma1 = ref_res.sigma1_dark
            sigma2 = ref_res.sigma2_dark
            Lk = ref_res.lekid.Lk
            R = ref_res.lekid.R
            
            # Calculate Lk per square
            Lk_square = Lk / (length / width) if (length / width) > 0 else 0
            
            # Update derived property labels
            self.nqp_label.setText(f"{nqp:.3e}")
            self.sigma1_label.setText(f"{sigma1:.3e}")
            self.sigma2_label.setText(f"{sigma2:.3e}")
            self.lk_square_label.setText(f"{Lk_square*1e12:.2f}")
            self.lk_label.setText(f"{Lk*1e9:.3f}")
            self.r_label.setText(f"{R*1e6:.3f}")
            
            # Calculate final derived properties using already-parsed Lg, L_junk, Cc, etc.
            # (No second read needed - we already have these values from above)
            L_total = Lk + Lg + L_junk
            alpha_k = Lk / L_total if L_total > 0 else 0
            
            # Calculate Q values
            freq = self.freq_start_spin.value() * 1e9
            if freq > 0:
                C = 1.0 / ((2 * np.pi * freq)**2 * L_total) if L_total > 0 else 1e-12
                
                lekid = MR_LEKID(
                    C=C, Lk=Lk, Lg=Lg, R=R, Cc=Cc, 
                    L_junk=L_junk, Vin=Vin, 
                    system_termination=50.0, input_atten_dB=input_atten_dB,
                    ZLNA=complex(ZLNA, 0)
                )
                
                Qr, Qi, Qc = lekid.compute_Q_values()
            else:
                Qr = Qi = Qc = 0
            
            # Update final labels
            self.ltotal_label.setText(f"{L_total*1e9:.2f}")
            self.alpha_k_label.setText(f"{alpha_k:.4f}")
            self.qr_label.setText(f"{Qr:,.0f}")
            self.qi_label.setText(f"{Qi:,.0f}")
            self.qc_label.setText(f"{Qc:,.0f}")
            
            # Debug: print to console to verify values are updating
            # print(f"DEBUG _update_all_derived: Lg={Lg*1e9:.2f}nH, L_junk={L_junk*1e9:.2f}nH, Lk={Lk*1e9:.3f}nH -> L_total={L_total*1e9:.2f}nH, α_k={alpha_k:.4f}")
            
        except Exception as e:
            # Log the exception for debugging (was silently swallowed)
            print(f"DEBUG _update_all_derived exception: {type(e).__name__}: {e}")

    def _create_readout_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Readout")
        layout = QtWidgets.QGridLayout(group)
        row = 0

        layout.addWidget(QtWidgets.QLabel("Vin (V):"), row, 0)
        self.Vin_edit = QtWidgets.QLineEdit()
        self.Vin_edit.setValidator(ScientificDoubleValidator())
        self.Vin_edit.setToolTip("Input drive voltage at the source before attenuation (Volts).")
        layout.addWidget(self.Vin_edit, row, 1)

        layout.addWidget(QtWidgets.QLabel("Input Attenuation (dB):"), row, 2)
        self.input_atten_edit = QtWidgets.QLineEdit()
        self.input_atten_edit.setValidator(ScientificDoubleValidator())
        self.input_atten_edit.setToolTip("Input attenuation applied to Vin (dB).")
        layout.addWidget(self.input_atten_edit, row, 3)

        row += 1
        layout.addWidget(QtWidgets.QLabel("System Termination (Ω):"), row, 0)
        self.system_termination_edit = QtWidgets.QLineEdit()
        self.system_termination_edit.setValidator(ScientificDoubleValidator())
        self.system_termination_edit.setToolTip("Readout line termination impedance (Ohms).")
        layout.addWidget(self.system_termination_edit, row, 1)

        layout.addWidget(QtWidgets.QLabel("ZLNA (Ω):"), row, 2)
        self.ZLNA_edit = QtWidgets.QLineEdit()
        self.ZLNA_edit.setValidator(ScientificDoubleValidator())
        self.ZLNA_edit.setToolTip("LNA input impedance (Ohms). Typically 50 Ω.")
        layout.addWidget(self.ZLNA_edit, row, 3)

        row += 1
        layout.addWidget(QtWidgets.QLabel("GLNA (dB):"), row, 0)
        self.GLNA_db_spin = QtWidgets.QDoubleSpinBox()
        self.GLNA_db_spin.setRange(-60.0, 60.0)
        self.GLNA_db_spin.setSingleStep(0.1)
        self.GLNA_db_spin.setDecimals(1)
        self.GLNA_db_spin.setToolTip("Voltage gain of the LNA in dB (20·log10(Vout/Vin)). Internally converted to linear V/V.")
        layout.addWidget(self.GLNA_db_spin, row, 1)

        return group

    def _create_noise_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Noise")
        layout = QtWidgets.QGridLayout(group)
        row = 0

        self.nqp_noise_enabled_cb = QtWidgets.QCheckBox("Enable QP noise")
        self.nqp_noise_enabled_cb.setToolTip("Enable random fluctuations of quasiparticle density per evaluation.")
        layout.addWidget(self.nqp_noise_enabled_cb, row, 0, 1, 2)

        layout.addWidget(QtWidgets.QLabel("Quasiparticle Noise (frac):"), row, 2)
        self.nqp_noise_std_edit = QtWidgets.QLineEdit()
        self.nqp_noise_std_edit.setValidator(ScientificDoubleValidator())
        self.nqp_noise_std_edit.setToolTip("Std dev as a fraction of base quasiparticle density (e.g., 0.001 = 0.1%).")
        layout.addWidget(self.nqp_noise_std_edit, row, 3)

        row += 1
        layout.addWidget(QtWidgets.QLabel("ADC Noise (counts):"), row, 0)
        self.udp_noise_edit = QtWidgets.QLineEdit()
        self.udp_noise_edit.setValidator(ScientificDoubleValidator())
        self.udp_noise_edit.setToolTip("Additive ADC noise level in counts")
        layout.addWidget(self.udp_noise_edit, row, 1)

        return group

    def _create_pulse_injection_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Pulse Injection")
        layout = QtWidgets.QGridLayout(group)
        row = 0

        layout.addWidget(QtWidgets.QLabel("Period (s):"), row, 0)
        self.pulse_period_edit = QtWidgets.QLineEdit()
        self.pulse_period_edit.setValidator(ScientificDoubleValidator())
        self.pulse_period_edit.setToolTip("Pulse period in seconds (used in periodic mode).")
        layout.addWidget(self.pulse_period_edit, row, 1)

        layout.addWidget(QtWidgets.QLabel("Probability (/s):"), row, 2)
        self.pulse_probability_edit = QtWidgets.QLineEdit()
        self.pulse_probability_edit.setValidator(ScientificDoubleValidator())
        self.pulse_probability_edit.setToolTip("Per-resonator per-second probability (random mode).\nEffective per-update chance ≈ probability × dt.")
        layout.addWidget(self.pulse_probability_edit, row, 3)

        row += 1
        layout.addWidget(QtWidgets.QLabel("Tau rise (s):"), row, 0)
        self.pulse_tau_rise_edit = QtWidgets.QLineEdit()
        self.pulse_tau_rise_edit.setValidator(ScientificDoubleValidator())
        self.pulse_tau_rise_edit.setToolTip("Exponential rise time constant (seconds).")
        layout.addWidget(self.pulse_tau_rise_edit, row, 1)

        layout.addWidget(QtWidgets.QLabel("Tau decay (s):"), row, 2)
        self.pulse_tau_decay_edit = QtWidgets.QLineEdit()
        self.pulse_tau_decay_edit.setValidator(ScientificDoubleValidator())
        self.pulse_tau_decay_edit.setToolTip("Exponential decay time constant (seconds).")
        layout.addWidget(self.pulse_tau_decay_edit, row, 3)

        row += 1
        layout.addWidget(QtWidgets.QLabel("Amplitude (× base nqp):"), row, 0)
        self.pulse_amplitude_edit = QtWidgets.QLineEdit()
        self.pulse_amplitude_edit.setValidator(ScientificDoubleValidator())
        self.pulse_amplitude_edit.setToolTip("Multiplicative factor relative to base quasiparticle density.\nExample: 2.0 doubles base nqp.")
        layout.addWidget(self.pulse_amplitude_edit, row, 1)

        layout.addWidget(QtWidgets.QLabel("Resonators:"), row, 2)
        self.pulse_resonators_edit = QtWidgets.QLineEdit()
        self.pulse_resonators_edit.setToolTip('Target resonators: "all" or CSV of 0-based indices (e.g., 0,1,7).')
        layout.addWidget(self.pulse_resonators_edit, row, 3)

        # Random amplitude distribution (random mode)
        row += 1
        layout.addWidget(QtWidgets.QLabel("Random amplitude mode:"), row, 0)
        self.random_amp_mode_combo = QtWidgets.QComboBox()
        self.random_amp_mode_combo.addItems(["fixed", "uniform", "lognormal"])
        self.random_amp_mode_combo.setToolTip('Random pulse amplitude distribution in "random" mode.')
        layout.addWidget(self.random_amp_mode_combo, row, 1)

        row += 1
        layout.addWidget(QtWidgets.QLabel("Uniform min:"), row, 0)
        self.random_amp_min_edit = QtWidgets.QLineEdit()
        self.random_amp_min_edit.setValidator(ScientificDoubleValidator())
        self.random_amp_min_edit.setToolTip("Minimum amplitude (× base nqp) for uniform distribution (≥ 1.0).")
        layout.addWidget(self.random_amp_min_edit, row, 1)

        layout.addWidget(QtWidgets.QLabel("Uniform max:"), row, 2)
        self.random_amp_max_edit = QtWidgets.QLineEdit()
        self.random_amp_max_edit.setValidator(ScientificDoubleValidator())
        self.random_amp_max_edit.setToolTip("Maximum amplitude (× base nqp) for uniform distribution (≥ min).")
        layout.addWidget(self.random_amp_max_edit, row, 3)

        row += 1
        layout.addWidget(QtWidgets.QLabel("Lognormal mean (μ):"), row, 0)
        self.random_amp_logmean_edit = QtWidgets.QLineEdit()
        self.random_amp_logmean_edit.setValidator(ScientificDoubleValidator())
        self.random_amp_logmean_edit.setToolTip("Mean parameter μ for lognormal amplitude distribution.")
        layout.addWidget(self.random_amp_logmean_edit, row, 1)

        layout.addWidget(QtWidgets.QLabel("Lognormal sigma (σ):"), row, 2)
        self.random_amp_logsigma_edit = QtWidgets.QLineEdit()
        self.random_amp_logsigma_edit.setValidator(ScientificDoubleValidator())
        self.random_amp_logsigma_edit.setToolTip("Sigma parameter σ (≥ 0) for lognormal amplitude distribution.")
        layout.addWidget(self.random_amp_logsigma_edit, row, 3)

        # Connect mode change to toggling of fields
        self.random_amp_mode_combo.currentTextChanged.connect(self._on_random_amp_mode_changed)

        return group

    def _on_random_amp_mode_changed(self, mode: str):
        # Enable/disable distribution-specific fields
        mode = (mode or "").lower()
        is_uniform = mode == "uniform"
        is_lognormal = mode == "lognormal"
        # Uniform fields
        for w in (self.random_amp_min_edit, self.random_amp_max_edit):
            w.setEnabled(is_uniform)
        # Lognormal fields
        for w in (self.random_amp_logmean_edit, self.random_amp_logsigma_edit):
            w.setEnabled(is_lognormal)

    def _create_simulation_realism_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Simulation Realism")
        layout = QtWidgets.QGridLayout(group)

        # Convergence tolerance (solver accuracy)
        layout.addWidget(QtWidgets.QLabel("I^2 Convergence tolerance:"), 0, 0)
        self.conv_tol_edit = QtWidgets.QLineEdit()
        self.conv_tol_edit.setValidator(ScientificDoubleValidator())
        self.conv_tol_edit.setToolTip("Solver accuracy: lower = more accurate, slower (e.g., 1e-9 default).")
        layout.addWidget(self.conv_tol_edit, 0, 1)

        # Cache QP step (fraction)
        layout.addWidget(QtWidgets.QLabel("Cache QP step (frac):"), 1, 0)
        self.cache_qp_step_edit = QtWidgets.QLineEdit()
        self.cache_qp_step_edit.setValidator(ScientificDoubleValidator())
        self.cache_qp_step_edit.setToolTip("QP quantization as fraction of base QP (e.g., 0.001 = 0.1%). Larger = coarser = more reuse.")
        layout.addWidget(self.cache_qp_step_edit, 1, 1)

        return group

    def _reset_to_defaults(self):
        cfg = mc.defaults()
        # Basic
        self.num_resonances_spin.setValue(int(cfg["num_resonances"]))
        self.freq_start_spin.setValue(float(cfg["freq_start"]) / 1e9)
        self.freq_end_spin.setValue(float(cfg["freq_end"]) / 1e9)
        self.random_seed_edit.setText("" if cfg["resonator_random_seed"] is None else str(cfg["resonator_random_seed"]))
        # Bias
        self.auto_bias_check.setChecked(bool(cfg["auto_bias_kids"]))
        self.bias_amplitude_spin.setValue(float(cfg["bias_amplitude"]))
        # Physics-driven mode
        self.T_edit.setText(str(cfg["T"]))
        self.Popt_edit.setText(str(cfg["Popt"]))
        self.width_edit.setText(str(round(cfg.get("width", 2e-6) * 1e6, 6)))  # m to µm, round to 6 decimals
        self.thickness_edit.setText(str(round(cfg.get("thickness", 30e-9) * 1e9, 6)))  # m to nm, round to 6 decimals
        self.length_edit.setText(str(round(cfg.get("length", 9000e-6) * 1e6, 6)))  # m to µm, round to 6 decimals
        # Circuit
        self.Lg_edit.setText(str(cfg["Lg"] * 1e9))  # H to nH
        self.Cc_edit.setText(str(cfg["Cc"] * 1e15))  # F to fF
        self.L_junk_edit.setText(str(cfg["L_junk"] * 1e9))  # H to nH
        self.C_variation_edit.setText(str(cfg["C_variation"]))
        self.Cc_variation_edit.setText(str(cfg["Cc_variation"]))
        # Readout
        self.Vin_edit.setText(str(cfg["Vin"]))
        self.input_atten_edit.setText(str(cfg["input_atten_dB"]))
        self.system_termination_edit.setText(str(cfg["system_termination"]))
        self.ZLNA_edit.setText(str(cfg["ZLNA"]))
        try:
            glna_val = float(cfg["GLNA"])
            glna_db = 20.0 * math.log10(glna_val) if glna_val > 0 else 0.0
        except Exception:
            glna_db = 0.0
        self.GLNA_db_spin.setValue(glna_db)
        # Noise
        self.nqp_noise_enabled_cb.setChecked(bool(cfg["nqp_noise_enabled"]))
        self.nqp_noise_std_edit.setText(str(cfg["nqp_noise_std_factor"]))
        self.udp_noise_edit.setText(str(cfg["udp_noise_level"]))
        # Physics Realism & Cache
        self.conv_tol_edit.setText(str(cfg["convergence_tolerance"]))
        self.cache_qp_step_edit.setText(str(cfg["cache_qp_step"]))

        # QP Pulses
        self.pulse_period_edit.setText(str(cfg["pulse_period"]))
        self.pulse_probability_edit.setText(str(cfg["pulse_probability"]))
        self.pulse_tau_rise_edit.setText(str(cfg["pulse_tau_rise"]))
        self.pulse_tau_decay_edit.setText(str(cfg["pulse_tau_decay"]))
        self.pulse_amplitude_edit.setText(str(cfg["pulse_amplitude"]))
        resonators = cfg.get("pulse_resonators", "all")
        if isinstance(resonators, list):
            self.pulse_resonators_edit.setText(",".join(str(int(r)) for r in resonators))
        else:
            self.pulse_resonators_edit.setText(str(resonators))

        # Random amplitude distribution
        ram = str(cfg.get("pulse_random_amp_mode"))
        idx_ram = max(0, self.random_amp_mode_combo.findText(ram))
        self.random_amp_mode_combo.setCurrentIndex(idx_ram)
        self.random_amp_min_edit.setText(str(cfg.get("pulse_random_amp_min")))
        self.random_amp_max_edit.setText(str(cfg.get("pulse_random_amp_max")))
        self.random_amp_logmean_edit.setText(str(cfg.get("pulse_random_amp_logmean")))
        self.random_amp_logsigma_edit.setText(str(cfg.get("pulse_random_amp_logsigma")))
        self._on_random_amp_mode_changed(self.random_amp_mode_combo.currentText())
        
        # Trigger derived parameter update
        self._update_all_derived()

    def _load_current_values(self):
        if not self.current_config:
            self._reset_to_defaults()
            return
        cfg = {**mc.defaults(), **self.current_config}  # fill missing with defaults

        # Basic
        self.num_resonances_spin.setValue(int(cfg.get("num_resonances")))
        self.freq_start_spin.setValue(float(cfg.get("freq_start")) / 1e9)
        self.freq_end_spin.setValue(float(cfg.get("freq_end")) / 1e9)
        seed = cfg.get("resonator_random_seed", None)
        self.random_seed_edit.setText("" if seed in (None, "") else str(int(seed)))

        # Bias
        self.auto_bias_check.setChecked(bool(cfg.get("auto_bias_kids")))
        self.bias_amplitude_spin.setValue(float(cfg.get("bias_amplitude")))

        # Physics-driven mode
        self.T_edit.setText(str(cfg.get("T")))
        self.Popt_edit.setText(str(cfg.get("Popt")))
        self.width_edit.setText(str(round(cfg.get("width", 2e-6) * 1e6, 6)))  # m to µm, round to 6 decimals
        self.thickness_edit.setText(str(round(cfg.get("thickness", 30e-9) * 1e9, 6)))  # m to nm, round to 6 decimals
        self.length_edit.setText(str(round(cfg.get("length", 9000e-6) * 1e6, 6)))  # m to µm, round to 6 decimals
        
        # Material selection (if present in config)
        if 'material' in cfg:
            mat_name = cfg['material']
            # Find material in combo (might be "Al (Aluminum)" or just "Al" or custom "Nb")
            idx = -1
            for i in range(self.material_combo.count()):
                combo_text = self.material_combo.itemText(i)
                combo_mat = combo_text.split()[0]
                if combo_mat == mat_name:
                    idx = i
                    break
            if idx >= 0:
                self.material_combo.setCurrentIndex(idx)

        # Circuit
        self.Lg_edit.setText(str(cfg.get("Lg") * 1e9))  # H to nH
        self.Cc_edit.setText(str(cfg.get("Cc") * 1e15))  # F to fF
        self.L_junk_edit.setText(str(cfg.get("L_junk") * 1e9))  # H to nH
        self.C_variation_edit.setText(str(cfg.get("C_variation")))
        self.Cc_variation_edit.setText(str(cfg.get("Cc_variation")))

        # Readout
        self.Vin_edit.setText(str(cfg.get("Vin")))
        self.input_atten_edit.setText(str(cfg.get("input_atten_dB")))
        self.system_termination_edit.setText(str(cfg.get("system_termination")))
        self.ZLNA_edit.setText(str(cfg.get("ZLNA")))
        try:
            glna_val = float(cfg.get("GLNA"))
            glna_db = 20.0 * math.log10(glna_val) if glna_val > 0 else 0.0
        except Exception:
            glna_db = 0.0
        self.GLNA_db_spin.setValue(glna_db)

        # Noise
        self.nqp_noise_enabled_cb.setChecked(bool(cfg.get("nqp_noise_enabled")))
        self.nqp_noise_std_edit.setText(str(cfg.get("nqp_noise_std_factor")))
        self.udp_noise_edit.setText(str(cfg.get("udp_noise_level")))

        # Physics Realism & Cache
        self.conv_tol_edit.setText(str(cfg.get("convergence_tolerance")))
        self.cache_qp_step_edit.setText(str(cfg.get("cache_qp_step")))

        # QP Pulses
        self.pulse_period_edit.setText(str(cfg.get("pulse_period")))
        self.pulse_probability_edit.setText(str(cfg.get("pulse_probability")))
        self.pulse_tau_rise_edit.setText(str(cfg.get("pulse_tau_rise")))
        self.pulse_tau_decay_edit.setText(str(cfg.get("pulse_tau_decay")))
        self.pulse_amplitude_edit.setText(str(cfg.get("pulse_amplitude")))
        resonators = cfg.get("pulse_resonators", "all")
        if isinstance(resonators, list):
            self.pulse_resonators_edit.setText(",".join(str(int(r)) for r in resonators))
        else:
            self.pulse_resonators_edit.setText(str(resonators))

        # Random amplitude distribution
        ram = str(cfg.get("pulse_random_amp_mode"))
        idx_ram = max(0, self.random_amp_mode_combo.findText(ram))
        self.random_amp_mode_combo.setCurrentIndex(idx_ram)
        self.random_amp_min_edit.setText(str(cfg.get("pulse_random_amp_min")))
        self.random_amp_max_edit.setText(str(cfg.get("pulse_random_amp_max")))
        self.random_amp_logmean_edit.setText(str(cfg.get("pulse_random_amp_logmean")))
        self.random_amp_logsigma_edit.setText(str(cfg.get("pulse_random_amp_logsigma")))
        self._on_random_amp_mode_changed(self.random_amp_mode_combo.currentText())

    def _validate_and_accept(self):
        # Frequency range
        if self.freq_start_spin.value() >= self.freq_end_spin.value():
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Frequency Start must be less than Frequency End.")
            return

        # Validate numeric text fields that must parse
        try:
            # Physics-driven mode
            float(self.T_edit.text()); float(self.Popt_edit.text())
            float(self.width_edit.text()); float(self.thickness_edit.text()); float(self.length_edit.text())
            # Circuit
            float(self.Lg_edit.text()); float(self.Cc_edit.text()); float(self.L_junk_edit.text())
            float(self.C_variation_edit.text()); float(self.Cc_variation_edit.text())
            # Readout
            float(self.Vin_edit.text()); float(self.input_atten_edit.text())
            float(self.system_termination_edit.text()); float(self.ZLNA_edit.text()); float(self.GLNA_db_spin.value())
            # Noise and physics realism
            float(self.nqp_noise_std_edit.text()); float(self.udp_noise_edit.text()); float(self.conv_tol_edit.text())
            float(self.cache_qp_step_edit.text())
            # QP Pulses
            float(self.pulse_period_edit.text())
            float(self.pulse_probability_edit.text())
            float(self.pulse_tau_rise_edit.text())
            float(self.pulse_tau_decay_edit.text())
            float(self.pulse_amplitude_edit.text())
            # Random distribution fields (parse regardless; defaults present)
            float(self.random_amp_min_edit.text()); float(self.random_amp_max_edit.text())
            float(self.random_amp_logmean_edit.text()); float(self.random_amp_logsigma_edit.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter valid numeric values for all fields.")
            return

        self.accept()

    def get_configuration(self) -> dict:
        """
        Return a dictionary of all parameter values matching config.MOCK_DEFAULTS keys.
        """
        seed_text = self.random_seed_edit.text().strip()
        seed_value = int(seed_text) if seed_text != "" else None

        config_dict = {
            # Basic
            "num_resonances": int(self.num_resonances_spin.value()),
            "freq_start": float(self.freq_start_spin.value()) * 1e9,  # GHz -> Hz
            "freq_end": float(self.freq_end_spin.value()) * 1e9,      # GHz -> Hz
            "resonator_random_seed": seed_value,

            # Bias
            "auto_bias_kids": bool(self.auto_bias_check.isChecked()),
            "bias_amplitude": float(self.bias_amplitude_spin.value()),

            # Physics
            "T": float(self.T_edit.text()),
            "Popt": float(self.Popt_edit.text()),

            # Circuit
            "Lg": float(self.Lg_edit.text()) * 1e-9,  # nH -> H
            "Cc": float(self.Cc_edit.text()) * 1e-15,  # fF -> F
            "L_junk": float(self.L_junk_edit.text()) * 1e-9,  # nH -> H
            "C_variation": float(self.C_variation_edit.text()),
            "Cc_variation": float(self.Cc_variation_edit.text()),

            # Readout
            "Vin": float(self.Vin_edit.text()),
            "input_atten_dB": float(self.input_atten_edit.text()),
            "system_termination": float(self.system_termination_edit.text()),
            "ZLNA": float(self.ZLNA_edit.text()),
            "GLNA": float(10 ** (self.GLNA_db_spin.value() / 20.0)),

            # Noise
            "nqp_noise_enabled": bool(self.nqp_noise_enabled_cb.isChecked()),
            "nqp_noise_std_factor": float(self.nqp_noise_std_edit.text()),
            "udp_noise_level": float(self.udp_noise_edit.text()),

            # Physics Realism & Cache
            "convergence_tolerance": float(self.conv_tol_edit.text()),
            "cache_qp_step": float(self.cache_qp_step_edit.text()),

            # QP Pulses
            "pulse_period": float(self.pulse_period_edit.text()),
            "pulse_probability": float(self.pulse_probability_edit.text()),
            "pulse_tau_rise": float(self.pulse_tau_rise_edit.text()),
            "pulse_tau_decay": float(self.pulse_tau_decay_edit.text()),
            "pulse_amplitude": float(self.pulse_amplitude_edit.text()),
            "pulse_resonators": (lambda txt: "all" if txt.strip().lower() == "all" or txt.strip()=="" else [
                int(s) for s in [p.strip() for p in txt.split(",")] if s.isdigit()
            ])(self.pulse_resonators_edit.text()),

            # Random amplitude distribution (random mode)
            "pulse_random_amp_mode": str(self.random_amp_mode_combo.currentText()),
            "pulse_random_amp_min": float(self.random_amp_min_edit.text()),
            "pulse_random_amp_max": float(self.random_amp_max_edit.text()),
            "pulse_random_amp_logmean": float(self.random_amp_logmean_edit.text()),
            "pulse_random_amp_logsigma": float(self.random_amp_logsigma_edit.text()),
        }
        
        # Add geometry parameters
        config_dict['width'] = float(self.width_edit.text()) * 1e-6  # µm -> m
        config_dict['thickness'] = float(self.thickness_edit.text()) * 1e-9  # nm -> m
        config_dict['length'] = float(self.length_edit.text()) * 1e-6  # µm -> m
        
        # Add material name
        mat_text = self.material_combo.currentText()
        mat_name = mat_text.split()[0] if mat_text else "Al"
        config_dict['material'] = mat_name
        
        # If custom material, include properties for reproducibility
        if mat_name != "Al":
            try:
                props = settings.get_material_properties(mat_name)
                config_dict['material_Tc'] = props['Tc']
                config_dict['material_N0'] = props['N0']
                config_dict['material_tau0'] = props['tau0']
                if 'sigmaN' in props:
                    config_dict['material_sigmaN'] = props['sigmaN']
            except ValueError:
                pass
        
        return config_dict
