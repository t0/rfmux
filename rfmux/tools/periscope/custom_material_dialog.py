"""
Custom Material Dialog for Mock Configuration
==============================================

Dialog for defining custom superconductor materials with their physical properties.
Materials are persisted via QSettings and can be used in physics-based calculations.
"""

from .utils import QtWidgets, QtGui
from . import settings
import re


class ScientificDoubleValidator(QtGui.QValidator):
    """Validator that accepts scientific notation."""
    def validate(self, string, pos):
        # Accept empty string, minus sign, or valid scientific notation
        if string == '' or string == '-' or string == '+':
            return (QtGui.QValidator.State.Intermediate, string, pos)
        # Full scientific notation pattern
        if re.match(r'^[+-]?(\d+\.?\d*|\d*\.\d+)([eE][+-]?\d+)?$', string):
            return (QtGui.QValidator.State.Acceptable, string, pos)
        # Partial matches for typing
        if re.match(r'^[+-]?(\d+\.?|\d*\.)(\d*)?([eE][+-]?\d*)?$', string):
            return (QtGui.QValidator.State.Intermediate, string, pos)
        return (QtGui.QValidator.State.Invalid, string, pos)


class CustomMaterialDialog(QtWidgets.QDialog):
    """
    Dialog for adding or editing custom superconductor materials.
    """

    def __init__(self, parent=None, material_name=None):
        super().__init__(parent)
        self.setWindowTitle("Add Custom Material" if material_name is None else f"Edit Material: {material_name}")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        self.material_name = material_name
        self.is_edit_mode = material_name is not None
        
        self._create_ui()
        
        # Load existing values if editing
        if self.is_edit_mode:
            self._load_material(material_name)
    
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Info label
        info_label = QtWidgets.QLabel(
            "Define a custom superconductor material by providing its physical properties.\n"
            "These values are used in Mattis-Bardeen theory calculations."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addSpacing(10)
        
        # Form layout for inputs
        form_layout = QtWidgets.QFormLayout()
        
        # Material name
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("e.g., Nb, TiN, NbTiN")
        self.name_edit.setToolTip("Short name or chemical symbol for the material")
        if self.is_edit_mode:
            self.name_edit.setEnabled(False)  # Can't change name when editing
        form_layout.addRow("Material Name:", self.name_edit)
        
        # Critical temperature
        self.tc_edit = QtWidgets.QLineEdit()
        self.tc_edit.setValidator(ScientificDoubleValidator())
        self.tc_edit.setPlaceholderText("e.g., 1.2 for Al, 9.2 for Nb")
        self.tc_edit.setToolTip(
            "<b>Critical Temperature (Tc)</b><br>"
            "Temperature at which material transitions to superconducting state.<br>"
            "Examples: Al = 1.2 K, Nb = 9.2 K, TiN ≈ 4-5 K"
        )
        form_layout.addRow("Tc (K):", self.tc_edit)
        
        # Density of states
        self.n0_edit = QtWidgets.QLineEdit()
        self.n0_edit.setValidator(ScientificDoubleValidator())
        self.n0_edit.setPlaceholderText("e.g., 1.72e10 for Al")
        self.n0_edit.setToolTip(
            "<b>Density of States at Fermi Level (N₀)</b><br>"
            "Single-spin density of states per unit volume per unit energy.<br>"
            "Units: µm⁻³eV⁻¹<br>"
            "Example: Al = 1.72×10¹⁰ µm⁻³eV⁻¹"
        )
        form_layout.addRow("N₀ (µm⁻³eV⁻¹):", self.n0_edit)
        
        # Recombination time
        self.tau0_edit = QtWidgets.QLineEdit()
        self.tau0_edit.setValidator(ScientificDoubleValidator())
        self.tau0_edit.setPlaceholderText("e.g., 438e-9 for Al (438 ns)")
        self.tau0_edit.setToolTip(
            "<b>Quasiparticle Recombination Time (τ₀)</b><br>"
            "Characteristic time for quasiparticle recombination at Tc.<br>"
            "Also called electron-phonon interaction time.<br>"
            "Units: seconds<br>"
            "Example: Al = 438 ns = 438×10⁻⁹ s"
        )
        form_layout.addRow("τ₀ (s):", self.tau0_edit)
        
        # Sheet resistance
        self.rs_edit = QtWidgets.QLineEdit()
        self.rs_edit.setValidator(ScientificDoubleValidator())
        self.rs_edit.setPlaceholderText("e.g., 4.0 for thin Al")
        self.rs_edit.setToolTip(
            "<b>Sheet Resistance (Rs)</b><br>"
            "Surface resistance per square, measured at room temperature.<br>"
            "Units: Ω/□ (Ohms per square)<br>"
            "Example: Thin Al films typically 2-10 Ω/□<br>"
            "This is the value you measure directly from your fabricated film."
        )
        form_layout.addRow("Rs (Ω/□):", self.rs_edit)
        
        # Reference thickness
        self.thickness_ref_edit = QtWidgets.QLineEdit()
        self.thickness_ref_edit.setValidator(ScientificDoubleValidator())
        self.thickness_ref_edit.setPlaceholderText("e.g., 20 for 20nm Al")
        self.thickness_ref_edit.setToolTip(
            "<b>Reference Film Thickness</b><br>"
            "Film thickness where Rs was measured.<br>"
            "Units: nm (nanometers)<br>"
            "Used to calculate σN = 1/(Rs × thickness)"
        )
        form_layout.addRow("Thickness (nm):", self.thickness_ref_edit)
        
        layout.addLayout(form_layout)
        
        layout.addSpacing(10)
        
        # Example materials reference
        examples_label = QtWidgets.QLabel(
            "<small><b>Typical Values for Common Materials:</b><br>"
            "• <b>Al:</b> Tc=1.2K, N₀=1.72×10¹⁰ µm⁻³eV⁻¹, τ₀=438ns<br>"
            "• <b>Nb:</b> Tc≈9.2K (other properties vary by film quality)<br>"
            "• <b>TiN:</b> Tc≈4-5K (highly dependent on stoichiometry)</small>"
        )
        examples_label.setWordWrap(True)
        examples_label.setStyleSheet("color: gray;")
        layout.addWidget(examples_label)
        
        layout.addSpacing(10)
        
        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | 
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _load_material(self, name: str):
        """Load existing material properties for editing."""
        try:
            props = settings.get_material_properties(name)
            self.name_edit.setText(name)
            self.tc_edit.setText(str(props['Tc']))
            self.n0_edit.setText(str(props['N0']))
            self.tau0_edit.setText(str(props['tau0']))
            # Load Rs and thickness_ref if available
            if 'Rs' in props:
                self.rs_edit.setText(str(props['Rs']))
            if 'thickness_ref' in props:
                self.thickness_ref_edit.setText(str(props['thickness_ref']))
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Could not load material: {e}")
            self.reject()
    
    def _validate_and_accept(self):
        """Validate inputs before accepting."""
        # Check name
        name = self.name_edit.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter a material name.")
            return
        
        # Prevent overwriting built-in materials
        if not self.is_edit_mode and name == "Al":
            QtWidgets.QMessageBox.warning(
                self, "Invalid Name", 
                "Cannot create custom material named 'Al' - this is a built-in material."
            )
            return
        
        # Validate required numeric fields
        try:
            tc = float(self.tc_edit.text())
            if tc <= 0:
                raise ValueError("Tc must be positive")
        except (ValueError, AttributeError):
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter a valid positive value for Tc.")
            return
        
        try:
            n0 = float(self.n0_edit.text())
            if n0 <= 0:
                raise ValueError("N0 must be positive")
        except (ValueError, AttributeError):
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter a valid positive value for N₀.")
            return
        
        try:
            tau0 = float(self.tau0_edit.text())
            if tau0 <= 0:
                raise ValueError("tau0 must be positive")
        except (ValueError, AttributeError):
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter a valid positive value for τ₀.")
            return
        
        # Validate Rs and thickness_ref
        try:
            Rs = float(self.rs_edit.text())
            if Rs <= 0:
                raise ValueError("Rs must be positive")
        except (ValueError, AttributeError):
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter a valid positive value for Rs.")
            return
        
        try:
            thickness_ref = float(self.thickness_ref_edit.text())
            if thickness_ref <= 0:
                raise ValueError("thickness_ref must be positive")
        except (ValueError, AttributeError):
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter a valid positive value for thickness.")
            return
        
        # Save to QSettings
        try:
            settings.save_custom_material(name, tc, n0, tau0, Rs=Rs, thickness_ref_nm=thickness_ref)
            self.accept()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save material: {e}")
    
    def get_material_name(self) -> str:
        """Get the material name (for use after dialog accepts)."""
        return self.name_edit.text().strip()


class ManageCustomMaterialsDialog(QtWidgets.QDialog):
    """
    Dialog for managing (viewing, editing, deleting) custom materials.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Custom Materials")
        self.setModal(True)
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        
        self._create_ui()
        self._refresh_list()
    
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Info label
        info_label = QtWidgets.QLabel(
            "Manage your custom superconductor materials. "
            "Materials are persisted between sessions."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addSpacing(10)
        
        # Table for materials
        self.materials_table = QtWidgets.QTableWidget()
        self.materials_table.setColumnCount(5)
        self.materials_table.setHorizontalHeaderLabels(["Name", "Tc (K)", "N₀ (µm⁻³eV⁻¹)", "τ₀ (s)", "σN (S/m)"])
        self.materials_table.horizontalHeader().setStretchLastSection(True)
        self.materials_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.materials_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.materials_table)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.add_button = QtWidgets.QPushButton("Add New")
        self.add_button.clicked.connect(self._add_material)
        button_layout.addWidget(self.add_button)
        
        self.edit_button = QtWidgets.QPushButton("Edit")
        self.edit_button.clicked.connect(self._edit_material)
        button_layout.addWidget(self.edit_button)
        
        self.delete_button = QtWidgets.QPushButton("Delete")
        self.delete_button.clicked.connect(self._delete_material)
        button_layout.addWidget(self.delete_button)
        
        button_layout.addStretch()
        
        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
        # Update button states based on selection
        self.materials_table.itemSelectionChanged.connect(self._update_button_states)
        self._update_button_states()
    
    def _refresh_list(self):
        """Reload the materials table from QSettings."""
        self.materials_table.setRowCount(0)
        
        materials = settings.get_custom_materials()
        for name, props in sorted(materials.items()):
            row = self.materials_table.rowCount()
            self.materials_table.insertRow(row)
            
            self.materials_table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.materials_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{props['Tc']:.4g}"))
            self.materials_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{props['N0']:.4g}"))
            self.materials_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{props['tau0']:.4g}"))
            
            sigmaN = props.get('sigmaN', 1./(4*20e-9))
            self.materials_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{sigmaN:.4g}"))
    
    def _update_button_states(self):
        """Enable/disable buttons based on selection."""
        has_selection = len(self.materials_table.selectedItems()) > 0
        self.edit_button.setEnabled(has_selection)
        self.delete_button.setEnabled(has_selection)
    
    def _add_material(self):
        """Open dialog to add a new material."""
        dialog = CustomMaterialDialog(parent=self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._refresh_list()
    
    def _edit_material(self):
        """Edit the selected material."""
        selected_rows = self.materials_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        
        row = selected_rows[0].row()
        material_name = self.materials_table.item(row, 0).text()
        
        dialog = CustomMaterialDialog(parent=self, material_name=material_name)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._refresh_list()
    
    def _delete_material(self):
        """Delete the selected material."""
        selected_rows = self.materials_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        
        row = selected_rows[0].row()
        material_name = self.materials_table.item(row, 0).text()
        
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete material '{material_name}'?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            settings.delete_custom_material(material_name)
            self._refresh_list()
