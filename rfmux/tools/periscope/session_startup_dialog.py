"""
Session Startup Dialog for Periscope
=====================================

This dialog is shown when Periscope launches, prompting the user to either
start a new session, load an existing session, or continue without a session.
"""

from PyQt6 import QtWidgets, QtCore, QtGui


class SessionStartupDialog(QtWidgets.QDialog):
    """
    Startup dialog for session configuration.
    
    Presents three options:
    1. Start New Session - Create a new session folder for auto-exporting data
    2. Load Existing Session - Load a previously created session
    3. Continue Without Session - Skip session management (disable auto-export)
    
    This dialog is modal and requires the user to make a choice.
    """
    
    # Results
    START_NEW = 1
    LOAD_EXISTING = 2
    NO_SESSION = 3
    
    def __init__(self, parent=None):
        """
        Initialize the session startup dialog.
        
        Args:
            parent: Parent widget (typically the main Periscope window)
        """
        super().__init__(parent)
        
        self.choice = None  # Will store user's choice
        self.setWindowTitle("Session Management")
        self.setModal(True)  # Force user to make a choice
        
        # Prevent closing via X button (user must choose)
        self.setWindowFlags(
            self.windowFlags() & ~QtCore.Qt.WindowType.WindowCloseButtonHint
        )
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Create the dialog UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title/Description
        title_label = QtWidgets.QLabel("Welcome to Periscope")
        title_font = QtGui.QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        desc_label = QtWidgets.QLabel(
            "Sessions allow you to automatically save and organize your analysis data.\n"
            "Choose an option to continue:"
        )
        desc_label.setWordWrap(True)
        desc_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc_label)
        
        layout.addSpacing(10)
        
        # Button 1: Start New Session
        self.btn_new = QtWidgets.QPushButton("Start New Session")
        self.btn_new.setMinimumHeight(50)
        self.btn_new.setToolTip(
            "Create a new session folder to automatically export\n"
            "network analysis, multisweep, and other data"
        )
        icon_new = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileDialogNewFolder)
        self.btn_new.setIcon(icon_new)
        self.btn_new.clicked.connect(lambda: self._make_choice(self.START_NEW))
        layout.addWidget(self.btn_new)
        
        # Button 2: Load Existing Session
        self.btn_load = QtWidgets.QPushButton("Load Existing Session")
        self.btn_load.setMinimumHeight(50)
        self.btn_load.setToolTip(
            "Open a previously created session folder to\n"
            "continue exporting data and view past results"
        )
        icon_open = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DirOpenIcon)
        self.btn_load.setIcon(icon_open)
        self.btn_load.clicked.connect(lambda: self._make_choice(self.LOAD_EXISTING))
        layout.addWidget(self.btn_load)
        
        # Button 3: Continue Without Session
        self.btn_skip = QtWidgets.QPushButton("Continue Without Session")
        self.btn_skip.setMinimumHeight(50)
        self.btn_skip.setToolTip(
            "Skip session management and proceed without\n"
            "automatic data export (you can still export manually)"
        )
        self.btn_skip.clicked.connect(lambda: self._make_choice(self.NO_SESSION))
        layout.addWidget(self.btn_skip)
        
        # Set fixed width for consistent button appearance
        button_width = 350
        self.btn_new.setMinimumWidth(button_width)
        self.btn_load.setMinimumWidth(button_width)
        self.btn_skip.setMinimumWidth(button_width)
        
        # Set dialog size
        self.setFixedSize(400, 320)
    
    def _make_choice(self, choice: int):
        """
        Record the user's choice and close the dialog.
        
        Args:
            choice: One of START_NEW, LOAD_EXISTING, or NO_SESSION
        """
        self.choice = choice
        self.accept()
    
    def get_choice(self) -> int:
        """
        Get the user's choice.
        
        Returns:
            The chosen option (START_NEW, LOAD_EXISTING, or NO_SESSION)
        """
        return self.choice
