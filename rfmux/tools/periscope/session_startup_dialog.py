"""
Unified Startup Dialog for Periscope
====================================

This dialog is shown when Periscope launches, prompting the user to configure
both connection mode (Hardware/Mock/Offline) and session management (New/Load/None).
"""

from PyQt6 import QtWidgets, QtCore, QtGui
import datetime
from . import settings


class UnifiedStartupDialog(QtWidgets.QDialog):
    """
    Unified startup dialog for connection and session configuration.
    
    Presents connection modes:
    1. Serial, Module - Connect to hardware CRS
    2. Mock - Connect to mock CRS
    3. Offline - Browse session data without CRS connection
    
    And session modes (conditional on connection):
    1. New Session - Create a new session folder for auto-exporting data
    2. Load Session - Load a previously created session
    3. No Session - Skip session management (disable auto-export)
    
    Note: In Offline mode, only "Load Session" is available.
    """
    
    # Connection mode results
    CONN_HARDWARE = 1
    CONN_MOCK = 2
    CONN_OFFLINE = 3
    
    # Session mode results
    SESS_NEW = 1
    SESS_LOAD = 2
    SESS_NONE = 3
    
    def __init__(self, parent=None, prefill: dict | None = None):
        """
        Initialize the unified startup dialog.
        
        Args:
            parent: Parent widget (typically the main Periscope window)
            prefill: Optional dict with pre-fill values:
                - connection_mode: CONN_HARDWARE, CONN_MOCK, or CONN_OFFLINE
                - crs_serial: CRS serial number string
                - module: Module number (1-8)
        """
        super().__init__(parent)
        
        self.connection_mode = None
        self.session_mode = None
        self.crs_serial = None
        self.module = None
        self.session_path = None
        self.session_folder_name = None
        
        # Store pre-fill values
        self._prefill = prefill or {}
        
        self.setWindowTitle("Periscope Configuration")
        self.setModal(True)
        
        # Prevent closing via X button
        self.setWindowFlags(
            self.windowFlags() & ~QtCore.Qt.WindowType.WindowCloseButtonHint
        )
        
        self._setup_ui()
        
        # Apply pre-fill values after UI is set up
        self._apply_prefill()
        
    def _setup_ui(self):
        """Create the dialog UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Title
        title_label = QtWidgets.QLabel("Welcome to Periscope")
        title_font = QtGui.QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Description
        desc_label = QtWidgets.QLabel(
            "Configure your connection and session settings to continue:"
        )
        desc_label.setWordWrap(True)
        desc_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc_label)
        
        layout.addSpacing(10)
        
        # Connection section
        layout.addWidget(self._create_connection_section())
        
        # Session section
        layout.addWidget(self._create_session_section())
        
        # OK/Cancel buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Set reasonable size
        self.setMinimumWidth(600)
        self.setMinimumHeight(550)
        
    def _create_connection_section(self) -> QtWidgets.QGroupBox:
        """Create the connection mode section."""
        group = QtWidgets.QGroupBox("Connection Mode")
        layout = QtWidgets.QVBoxLayout(group)
        
        # Connection radio buttons
        self.rb_hardware = QtWidgets.QRadioButton("Connect to Hardware (Serial, Module)")
        self.rb_hardware.setToolTip("Connect to a physical CRS board")
        self.rb_hardware.setChecked(True)  # Default
        
        self.rb_mock = QtWidgets.QRadioButton("Mock Mode")
        self.rb_mock.setToolTip("Use simulated CRS for testing")
        
        self.rb_offline = QtWidgets.QRadioButton("Offline Mode")
        self.rb_offline.setToolTip("Browse session data without CRS connection")
        
        layout.addWidget(self.rb_hardware)
        layout.addWidget(self.rb_mock)
        layout.addWidget(self.rb_offline)
        
        # Hardware connection details (Serial + Module)
        self.hardware_details = QtWidgets.QWidget()
        hardware_layout = QtWidgets.QFormLayout(self.hardware_details)
        hardware_layout.setContentsMargins(20, 10, 0, 10)
        
        self.serial_input = QtWidgets.QLineEdit()
        self.serial_input.setPlaceholderText("0042")
        self.serial_input.setToolTip("Serial number of the CRS board (e.g., 0042 for rfmux0042.local)")
        hardware_layout.addRow("CRS Serial:", self.serial_input)
        
        self.module_input = QtWidgets.QSpinBox()
        self.module_input.setRange(1, 8)
        self.module_input.setValue(1)
        self.module_input.setToolTip("Module number (1-8)")
        hardware_layout.addRow("Module:", self.module_input)
        
        layout.addWidget(self.hardware_details)
        
        # Connect signals to update UI state
        self.rb_hardware.toggled.connect(self._on_connection_mode_changed)
        self.rb_mock.toggled.connect(self._on_connection_mode_changed)
        self.rb_offline.toggled.connect(self._on_connection_mode_changed)
        
        # Initial state
        self._on_connection_mode_changed()
        
        return group
    
    def _create_session_section(self) -> QtWidgets.QGroupBox:
        """Create the session mode section."""
        group = QtWidgets.QGroupBox("Session Management")
        layout = QtWidgets.QVBoxLayout(group)
        
        # Session radio buttons
        self.rb_new_session = QtWidgets.QRadioButton("New Session")
        self.rb_new_session.setToolTip("Create a new session folder to auto-export data")
        self.rb_new_session.setChecked(True)  # Default
        
        self.rb_load_session = QtWidgets.QRadioButton("Load Existing Session")
        self.rb_load_session.setToolTip("Load a previously created session folder")
        
        self.rb_no_session = QtWidgets.QRadioButton("No Session")
        self.rb_no_session.setToolTip("Continue without session management (no auto-export)")
        
        layout.addWidget(self.rb_new_session)
        layout.addWidget(self.rb_load_session)
        layout.addWidget(self.rb_no_session)
        
        # New session details - just folder name input (path selected on OK click)
        self.new_session_details = QtWidgets.QWidget()
        new_session_layout = QtWidgets.QVBoxLayout(self.new_session_details)
        new_session_layout.setContentsMargins(20, 10, 0, 10)
        
        # Description
        new_desc = QtWidgets.QLabel("A folder selection dialog will open when you click OK.")
        new_desc.setStyleSheet("color: gray; font-style: italic;")
        new_session_layout.addWidget(new_desc)
        
        # Folder name input
        folder_name_layout = QtWidgets.QHBoxLayout()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.folder_name_input = QtWidgets.QLineEdit()
        self.folder_name_input.setText(f"session_{timestamp}")
        self.folder_name_input.setToolTip("Name for the new session folder")
        folder_name_layout.addWidget(QtWidgets.QLabel("Folder Name:"))
        folder_name_layout.addWidget(self.folder_name_input)
        new_session_layout.addLayout(folder_name_layout)
        
        layout.addWidget(self.new_session_details)
        
        # Load session details - just description (path selected on OK click)
        self.load_session_details = QtWidgets.QWidget()
        load_session_layout = QtWidgets.QVBoxLayout(self.load_session_details)
        load_session_layout.setContentsMargins(20, 10, 0, 10)
        
        load_desc = QtWidgets.QLabel("A folder selection dialog will open when you click OK.")
        load_desc.setStyleSheet("color: gray; font-style: italic;")
        load_session_layout.addWidget(load_desc)
        
        layout.addWidget(self.load_session_details)
        
        # Connect signals
        self.rb_new_session.toggled.connect(self._on_session_mode_changed)
        self.rb_load_session.toggled.connect(self._on_session_mode_changed)
        self.rb_no_session.toggled.connect(self._on_session_mode_changed)
        
        # Initial state
        self._on_session_mode_changed()
        
        return group
    
    def _on_connection_mode_changed(self):
        """Handle connection mode radio button changes."""
        # Show/hide hardware details
        is_hardware = self.rb_hardware.isChecked()
        self.hardware_details.setVisible(is_hardware)
        
        # Enable/disable session modes based on connection
        is_offline = self.rb_offline.isChecked()
        
        # In offline mode, only "Load Session" is allowed
        # Only update if session widgets exist (they're created after connection widgets)
        if hasattr(self, 'rb_new_session'):
            self.rb_new_session.setEnabled(not is_offline)
        if hasattr(self, 'rb_no_session'):
            self.rb_no_session.setEnabled(not is_offline)
        
        # If switching to offline and not on "Load Session", switch to it
        if is_offline and hasattr(self, 'rb_load_session') and not self.rb_load_session.isChecked():
            self.rb_load_session.setChecked(True)
    
    def _on_session_mode_changed(self):
        """Handle session mode radio button changes."""
        is_new = self.rb_new_session.isChecked()
        is_load = self.rb_load_session.isChecked()
        
        self.new_session_details.setVisible(is_new)
        self.load_session_details.setVisible(is_load)
    
    def _validate_and_accept(self):
        """Validate inputs and open file dialogs as needed before accepting."""
        # Determine connection mode
        if self.rb_hardware.isChecked():
            self.connection_mode = self.CONN_HARDWARE
            # Validate serial number
            if not self.serial_input.text().strip():
                QtWidgets.QMessageBox.warning(
                    self,
                    "Missing CRS Serial",
                    "Please enter a CRS serial number for hardware connection."
                )
                return
            self.crs_serial = self.serial_input.text().strip()
            self.module = self.module_input.value()
            
        elif self.rb_mock.isChecked():
            self.connection_mode = self.CONN_MOCK
            # Mock mode defaults
            self.crs_serial = None  # Mock doesn't need serial
            self.module = self.module_input.value() if hasattr(self, 'module_input') else 1
            
        elif self.rb_offline.isChecked():
            self.connection_mode = self.CONN_OFFLINE
            self.crs_serial = None
            self.module = None
        
        # Get last session directory for file dialogs
        last_session_dir = settings.get_last_session_directory()
        
        # Determine session mode and open appropriate file dialogs
        if self.rb_new_session.isChecked():
            self.session_mode = self.SESS_NEW
            
            # Validate folder name first
            if not self.folder_name_input.text().strip():
                QtWidgets.QMessageBox.warning(
                    self,
                    "Missing Folder Name",
                    "Please enter a folder name for the new session."
                )
                return
            self.session_folder_name = self.folder_name_input.text().strip()
            
            # Open folder selection dialog for base path
            # Start from last used directory if available
            start_dir = last_session_dir if last_session_dir else ""
            base_path = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Select Session Location",
                start_dir,
                QtWidgets.QFileDialog.Option.ShowDirsOnly | 
                QtWidgets.QFileDialog.Option.DontUseNativeDialog
            )
            
            # If user cancelled, stay in dialog
            if not base_path:
                return
            
            self.session_path = base_path
            # Save this directory for next time
            settings.set_last_session_directory(base_path)
            
        elif self.rb_load_session.isChecked():
            self.session_mode = self.SESS_LOAD
            
            # Open folder selection dialog for existing session
            # Start from last used directory if available
            start_dir = last_session_dir if last_session_dir else ""
            session_path = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Select Session Folder",
                start_dir,
                QtWidgets.QFileDialog.Option.ShowDirsOnly |
                QtWidgets.QFileDialog.Option.DontUseNativeDialog
            )
            
            # If user cancelled, stay in dialog
            if not session_path:
                return
            
            self.session_path = session_path
            self.session_folder_name = None
            # Save the parent directory for next time (not the session folder itself)
            import os
            parent_dir = os.path.dirname(session_path)
            if parent_dir:
                settings.set_last_session_directory(parent_dir)
            
        elif self.rb_no_session.isChecked():
            self.session_mode = self.SESS_NONE
            self.session_path = None
            self.session_folder_name = None
        
        # Save connection settings for next time
        if self.connection_mode == self.CONN_HARDWARE:
            settings.set_last_connection_mode("hardware")
            if self.crs_serial:
                settings.set_last_crs_serial(self.crs_serial)
            if self.module:
                settings.set_last_module(self.module)
        elif self.connection_mode == self.CONN_MOCK:
            settings.set_last_connection_mode("mock")
            if self.module:
                settings.set_last_module(self.module)
        elif self.connection_mode == self.CONN_OFFLINE:
            settings.set_last_connection_mode("offline")
        
        # All validation passed
        self.accept()
    
    def _apply_prefill(self):
        """
        Apply pre-fill values from CLI arguments or saved settings to the UI.
        
        Priority: CLI prefill values > saved settings > defaults
        """
        # Load saved settings if no prefill provided
        if not self._prefill:
            # Try to restore from saved settings
            saved_mode = settings.get_last_connection_mode()
            saved_serial = settings.get_last_crs_serial()
            saved_module = settings.get_last_module()
            
            # Apply saved connection mode
            if saved_mode == "hardware":
                self.rb_hardware.setChecked(True)
            elif saved_mode == "mock":
                self.rb_mock.setChecked(True)
            elif saved_mode == "offline":
                self.rb_offline.setChecked(True)
            
            # Apply saved serial and module
            if saved_serial:
                self.serial_input.setText(saved_serial)
            if saved_module:
                self.module_input.setValue(saved_module)
        else:
            # Apply prefill values from CLI (takes precedence over saved settings)
            conn_mode = self._prefill.get('connection_mode')
            if conn_mode == self.CONN_HARDWARE:
                self.rb_hardware.setChecked(True)
            elif conn_mode == self.CONN_MOCK:
                self.rb_mock.setChecked(True)
            elif conn_mode == self.CONN_OFFLINE:
                self.rb_offline.setChecked(True)
            
            # Apply CRS serial
            crs_serial = self._prefill.get('crs_serial')
            if crs_serial:
                self.serial_input.setText(str(crs_serial))
            
            # Apply module
            module = self._prefill.get('module')
            if module is not None:
                self.module_input.setValue(int(module))
        
        # Trigger UI updates
        self._on_connection_mode_changed()
    
    def get_configuration(self) -> dict:
        """
        Get the selected configuration.
        
        Returns:
            dict: Configuration with keys:
                - connection_mode: CONN_HARDWARE, CONN_MOCK, or CONN_OFFLINE
                - session_mode: SESS_NEW, SESS_LOAD, or SESS_NONE
                - crs_serial: CRS serial number (for hardware) or None
                - module: Module number (for hardware/mock) or None
                - session_path: Base path (for new session) or full path (for load)
                - session_folder_name: Folder name (for new session) or None
        """
        return {
            'connection_mode': self.connection_mode,
            'session_mode': self.session_mode,
            'crs_serial': self.crs_serial,
            'module': self.module,
            'session_path': self.session_path,
            'session_folder_name': self.session_folder_name,
        }
