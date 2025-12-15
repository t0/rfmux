"""Jupyter Notebook panel for Periscope.

Provides a dockable panel that manages a Jupyter Lab server.
Notebooks are opened in the system browser and saved to the active session folder.
"""

import subprocess
import time
import socket
import os
import json
from pathlib import Path
from datetime import datetime
from PyQt6 import QtCore, QtWidgets, QtGui
from . import settings
from .utils import find_parent_with_attr
import sys


class JupyterServerManager(QtCore.QObject):
    """
    Manages a local Jupyter notebook server.
    
    Starts a JupyterLab server in a subprocess and monitors its status.
    The server is automatically stopped when the manager is destroyed.
    
    Signals:
        server_ready: Emitted with URL when server is ready to accept connections
        server_error: Emitted with error message on failure
    """
    server_ready = QtCore.pyqtSignal(str)  # url
    server_error = QtCore.pyqtSignal(str)  # error message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self.url = None
        self.base_url = None  # Base URL without notebook path
        self.notebook_dir = None
        self.port = None
        self._check_attempts = 0
        self._max_check_attempts = 30  # 30 seconds max wait time
    
    def start(self, notebook_dir: str, port: int = 8888):
        """
        Start Jupyter Lab server.
        
        Args:
            notebook_dir: Directory for notebooks (will be created if needed)
            port: Starting port to try (will increment if busy)
        """
        if self.process:
            return  # Already running
        
        self.notebook_dir = Path(notebook_dir)
        self.notebook_dir.mkdir(parents=True, exist_ok=True)
        
        # Find available port
        actual_port = self._find_available_port(port)
        if actual_port is None:
            self.server_error.emit("Could not find available port for Jupyter server")
            return
        
        self.port = actual_port
        
        # Start Jupyter Lab
        cmd = [
            'jupyter', 'lab',
            f'--notebook-dir={self.notebook_dir}',
            f'--port={actual_port}',
            '--no-browser',
            '--ServerApp.token=periscope',
            '--ServerApp.disable_check_xsrf=True',
        ]
        
        try:
            # Use CREATE_NO_WINDOW on Windows to hide console
            kwargs = {}
            if os.name == 'nt':
                # Windows: Hide console window and redirect output to DEVNULL
                # to prevent pipe buffer deadlock from verbose Jupyter startup output
                kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
                kwargs['stdout'] = subprocess.DEVNULL
                kwargs['stderr'] = subprocess.DEVNULL
            else:
                # Unix: Capture output for error diagnosis
                kwargs['stdout'] = subprocess.PIPE
                kwargs['stderr'] = subprocess.STDOUT
            self.process = subprocess.Popen(
                cmd,
                text=True,
                **kwargs
            )
            # Build URL and start checking
            self.url = f"http://localhost:{actual_port}/lab?token=periscope"
            self._check_attempts = 0
            QtCore.QTimer.singleShot(1000, self._check_ready)
            
        except FileNotFoundError:
            self.server_error.emit(
                "Jupyter is not installed.\n"
                "Install with: pip install jupyterlab"
            )
        except Exception as e:
            self.server_error.emit(f"Failed to start Jupyter: {e}")
    
    def _find_available_port(self, start_port: int) -> int | None:
        """Find an available port starting from start_port."""
        for port in range(start_port, start_port + 100):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('localhost', port))
                sock.close()
                return port
            except OSError:
                continue
        return None
    
    def _check_ready(self):
        """Check if server is ready and emit signal."""
        import urllib.request
        import urllib.error
        import traceback
    
        self._check_attempts += 1
    
        # If process exited, fail immediately
        if self.process and self.process.poll() is not None:
            # Process exited - read error output if available
            if self.process.stdout:
                output = self.process.stdout.read()
                self.server_error.emit(f"Jupyter server exited unexpectedly:\n{output[:500]}")
            else:
                # On Windows, stdout is not captured - suggest running manually for diagnosis
                self.server_error.emit(
                    "Jupyter server exited unexpectedly.\n"
                    "Run 'jupyter lab --no-browser' in a terminal for more details."
                )
            return
    
        # Probe lightweight API endpoint
        try:
            if self.url:
                urllib.request.urlopen(self.url, timeout=1)
                self.server_ready.emit(self.url)
    
        except Exception as e:
            # Print full diagnostics
            # print("[Notebook] Server not ready yet")
            # print(f"  Attempt: {self._check_attempts}")
            # print(f"  Exception type: {type(e)}")
            # print(f"  Exception value: {e}")
    
            # if hasattr(e, "reason"):
            #     print(f"  URLError.reason: {e.reason}")
            #     print(f"  reason type: {type(e.reason)}")
    
            # # if isinstance(e, OSError):
            #     print(f"  errno: {e.errno}")
            #     print(f"  winerror: {getattr(e, 'winerror', None)}")
            ### Keeping it here for debugging ####
    
            # Retry or fail
            if self._check_attempts < self._max_check_attempts:
                QtCore.QTimer.singleShot(1000, self._check_ready)
            else:
                self.server_error.emit(
                    "Jupyter server did not become ready within 30 seconds.\n"
                    "See console output for detailed startup errors."
                )

    
    def get_notebook_url(self, notebook_path: str) -> str | None:
        """
        Get the URL to open a specific notebook.
        
        Args:
            notebook_path: Full path to the .ipynb file
            
        Returns:
            URL to open the notebook in JupyterLab, or None if server not ready
        """
        if not self.port or not self.notebook_dir:
            return None
        
        # Calculate relative path from notebook_dir
        try:
            rel_path = Path(notebook_path).relative_to(self.notebook_dir)
            # URL encode the path
            import urllib.parse
            encoded_path = urllib.parse.quote(str(rel_path))
            return f"http://localhost:{self.port}/lab/tree/{encoded_path}?token=periscope"
        except ValueError:
            # Notebook not in notebook_dir
            return None
    
    def stop(self):
        """Stop the Jupyter server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            self.url = None


class NotebookPanel(QtWidgets.QWidget):
    """
    Jupyter notebook control panel.
    
    Manages a Jupyter Lab server and provides controls to open notebooks
    in the system browser. Notebooks are saved to the active session folder.
    
    The panel can either own its own server or use an external server reference.
    When using an external server, the panel does NOT stop the server on close.
    """
    
    def __init__(self, notebook_dir: str, parent=None, server: JupyterServerManager = None,
                 skip_initial_notebook: bool = False):
        """
        Initialize the notebook panel.
        
        Args:
            notebook_dir: Directory for notebooks (must be provided - typically the session folder).
            parent: Parent widget
            server: Optional external JupyterServerManager. If provided, the panel will
                   use this server instead of creating its own. The external server
                   will NOT be stopped when the panel is closed.
            skip_initial_notebook: If True, don't auto-create a template notebook on server ready.
                                  Use this when opening a specific existing notebook file.
        """
        super().__init__(parent)
        if not notebook_dir:
            raise ValueError("notebook_dir is required - notebooks must be saved to a session folder")
        self.notebook_dir: str = notebook_dir
        
        # Track whether we own the server (and should stop it) or use an external one
        self._owns_server = server is None
        self.server = server if server else JupyterServerManager(self)
        
        # Flag to skip auto-creating initial notebook (when opening a specific file)
        self._skip_initial_notebook = skip_initial_notebook
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Create the panel UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title and description
        title_label = QtWidgets.QLabel("Jupyter Lab Server")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title_label)
        
        info_label = QtWidgets.QLabel(
            "Jupyter Lab runs in your system browser.\n"
            "Notebooks are saved to the active session folder."
        )
        info_label.setStyleSheet("color: gray; font-size: 11px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addSpacing(10)
        
        # Status section
        status_group = QtWidgets.QGroupBox("Server Status")
        status_layout = QtWidgets.QVBoxLayout(status_group)
        
        self.status_label = QtWidgets.QLabel("⏳ Starting server...")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        status_layout.addWidget(self.status_label)
        
        # URL display (selectable/copyable)
        self.url_label = QtWidgets.QLineEdit("")
        self.url_label.setReadOnly(True)
        self.url_label.setPlaceholderText("Server URL will appear here...")
        self.url_label.setStyleSheet("font-size: 10px; background: transparent; border: none;")
        self.url_label.setToolTip("Server URL (select and copy)")
        status_layout.addWidget(self.url_label)
        
        self.path_label = QtWidgets.QLabel(f"📁 {self.notebook_dir}")
        self.path_label.setStyleSheet("font-size: 10px; color: gray;")
        self.path_label.setWordWrap(True)
        status_layout.addWidget(self.path_label)
        
        layout.addWidget(status_group)
        
        # Buttons
        button_layout = QtWidgets.QVBoxLayout()
        button_layout.setSpacing(8)
        
        self.open_browser_btn = QtWidgets.QPushButton("🌐 Open Jupyter Lab in Browser")
        self.open_browser_btn.setToolTip("Open Jupyter Lab in your default web browser")
        self.open_browser_btn.clicked.connect(self._open_in_browser)
        self.open_browser_btn.setEnabled(False)
        self.open_browser_btn.setMinimumHeight(40)
        button_layout.addWidget(self.open_browser_btn)
        
        self.create_notebook_btn = QtWidgets.QPushButton("➕ Create New Notebook")
        self.create_notebook_btn.setToolTip("Create a new notebook with pre-filled setup code")
        self.create_notebook_btn.clicked.connect(self._create_and_open_notebook)
        self.create_notebook_btn.setEnabled(False)
        button_layout.addWidget(self.create_notebook_btn)
        
        self.shutdown_btn = QtWidgets.QPushButton("🛑 Shutdown Server")
        self.shutdown_btn.setToolTip("Shutdown the Jupyter Lab server and close this panel")
        self.shutdown_btn.clicked.connect(self._shutdown_and_close)
        self.shutdown_btn.setEnabled(False)
        self.shutdown_btn.setStyleSheet("QPushButton { color: #d32f2f; }")
        button_layout.addWidget(self.shutdown_btn)
        
        layout.addLayout(button_layout)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        # Help text at bottom
        help_label = QtWidgets.QLabel(
            "💡 Tip: Double-click .ipynb files in the Session Browser "
            "to open them directly in Jupyter Lab."
        )
        help_label.setStyleSheet("font-size: 10px; color: gray; font-style: italic;")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
    
    def _connect_signals(self):
        """Connect signals."""
        self.server.server_ready.connect(self._on_server_ready)
        self.server.server_error.connect(self._on_server_error)
    
    def _get_default_user_library(self) -> Path:
        """Get the default user library path for this platform."""
        if os.name == 'nt':
            # Windows: ~/AppData/Local/rfmux/notebooks
            return Path.home() / "AppData" / "Local" / "rfmux" / "notebooks"
        else:
            # Linux/macOS: ~/.local/share/rfmux/notebooks
            return Path.home() / ".local" / "share" / "rfmux" / "notebooks"
    
    def _prompt_for_user_library(self) -> Path | None:
        """
        Prompt user for their notebook library directory.
        
        Uses saved path from previous sessions if available, otherwise uses platform default.
        
        Returns:
            Path to user library, or None if cancelled
        """
        # Get saved path from settings, or fall back to platform default
        saved_path = settings.get_user_library_path()
        if saved_path:
            default_path = Path(saved_path)
        else:
            default_path = self._get_default_user_library()
        
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("User Notebook Library")
        dialog.setMinimumWidth(500)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # Description
        desc_label = QtWidgets.QLabel(
            "Choose a directory for your personal notebook library.\n\n"
            "This library will be accessible across all Periscope sessions "
            "as '_user_notebooks' in Jupyter Lab."
        )
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        layout.addSpacing(10)
        
        # Path input - pre-populated with saved or default path
        path_layout = QtWidgets.QHBoxLayout()
        path_label = QtWidgets.QLabel("Library Path:")
        path_layout.addWidget(path_label)
        
        path_edit = QtWidgets.QLineEdit(str(default_path))
        path_edit.setMinimumWidth(300)
        path_layout.addWidget(path_edit, 1)
        
        browse_btn = QtWidgets.QPushButton("Browse...")
        browse_btn.clicked.connect(
            lambda: self._browse_for_directory(path_edit)
        )
        path_layout.addWidget(browse_btn)
        
        # Add "Reset to Default" button
        reset_btn = QtWidgets.QPushButton("Reset to Default")
        reset_btn.setToolTip("Reset to platform default location")
        reset_btn.clicked.connect(
            lambda: path_edit.setText(str(self._get_default_user_library()))
        )
        path_layout.addWidget(reset_btn)
        
        layout.addLayout(path_layout)
        
        layout.addSpacing(10)
        
        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # Show dialog
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            chosen_path = Path(path_edit.text().strip())
            if chosen_path:
                # Save the chosen path for next time
                settings.set_user_library_path(str(chosen_path))
                return chosen_path
        
        return None
    
    def _browse_for_directory(self, line_edit: QtWidgets.QLineEdit):
        """Browse for a directory and update the line edit."""
        current_path = line_edit.text()
        if not current_path:
            current_path = str(Path.home())
        
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Notebook Library Directory",
            current_path,
            QtWidgets.QFileDialog.Option.ShowDirsOnly |
            QtWidgets.QFileDialog.Option.DontUseNativeDialog
        )
        
        if directory:
            line_edit.setText(directory)
    
    def start(self):
        """Start the Jupyter Lab server."""
        # Prompt for user notebook library directory
        user_lib_path = self._prompt_for_user_library()
        if user_lib_path:
            self.user_library_path = user_lib_path
        else:
            self.user_library_path = None
        
        self.server.start(self.notebook_dir)
    
    def _get_parent_app_info(self) -> tuple[str | None, bool]:
        """
        Get CRS serial and mock mode from parent Periscope app.
        
        Walks up the parent hierarchy to find the Periscope main window
        and extracts CRS information from it.
        
        Returns:
            tuple: (crs_serial, is_mock)
                - crs_serial: CRS serial number as string, or None
                - is_mock: True if running in mock mode, False otherwise
        """
        parent_app = find_parent_with_attr(self, 'crs')
        
        crs_serial = None
        is_mock = False
        
        if parent_app:
            if hasattr(parent_app, 'crs') and parent_app.crs is not None:
                crs_serial = getattr(parent_app.crs, 'serial', None)
                if crs_serial:
                    crs_serial = str(crs_serial)
            
            if hasattr(parent_app, 'is_mock_mode'):
                is_mock = parent_app.is_mock_mode
        
        return crs_serial, is_mock
    
    def _on_server_ready(self, url: str):
        """Handle server ready."""
        self.status_label.setText("✅ Server running")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 12px; color: green;")
        
        # Display the server URL
        self.url_label.setText(url or self.server.url or "")
        
        self.open_browser_btn.setEnabled(True)
        self.create_notebook_btn.setEnabled(True)
        self.shutdown_btn.setEnabled(True)
        
        # Create symlinks to resources
        self._setup_notebook_resources()
        
        # Auto-create an analysis notebook and open it (unless skipped)
        if not self._skip_initial_notebook:
            QtCore.QTimer.singleShot(500, self._create_and_open_initial_notebook)
    
    def _create_and_open_initial_notebook(self):
        """Create an initial analysis notebook and open it in the browser."""
        # Get CRS info from parent app using helper method
        crs_serial, is_mock = self._get_parent_app_info()
        
        # Create the initial notebook
        notebook_path = self.create_notebook(
            crs_serial=crs_serial,
            session_path=self.notebook_dir,
            is_mock=is_mock
        )
        
        if notebook_path:
            # Open the specific notebook (not just the home page)
            url = self.server.get_notebook_url(notebook_path)
            if url:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))
        else:
            # Fallback to opening the home page
            self._open_in_browser()
    
    
    def create_dir_link(self, link_path: Path, target_path: Path):
        """
        Cross-platform directory linking:
        - Windows: directory junction
        - Unix: symlink
        """
        link_path = Path(link_path)
        target_path = Path(target_path)
        
        if link_path.exists():
            return
        
        if sys.platform == "win32":
            print("[Notebook] Identified as windows platform for the symlink")
            # Use mklink /J for directory junctions
            cmd = [
                "cmd",
                "/c",
                "mklink",
                "/J",
                str(link_path),
                str(target_path)
            ]
            subprocess.check_call(cmd, shell=False)
        else:
            link_path.symlink_to(target_path, target_is_directory=True)   
        
        
    def _setup_notebook_resources(self):
        """
        Create symlinks to shared notebook resources.
        
        Creates symlinks in the session folder to:
        - rfmux/home/ subfolders (Demos, Technical Documentation, Release Notes)
        - User notebook library (~/.local/share/rfmux/notebooks/)
        
        Symlinks use underscore prefix to group them at the top of file listings.
        """
        session_dir = Path(self.notebook_dir)
        
        # Find the rfmux package directory
        try:
            import rfmux
            rfmux_pkg_dir = Path(rfmux.__file__).parent
            home_dir = rfmux_pkg_dir.parent / "home"
            
            if home_dir.exists():                
                # Symlink specific home subfolders
                folders_to_link = ["Demos", "Release Notes", "Technical Documentation"]
                
                for folder_name in folders_to_link:
                    source = home_dir / folder_name
                    if source.exists() and source.is_dir():
                        # Use folder name as-is (no underscore prefix)
                        link_name = folder_name
                        link_path = session_dir / link_name
                        
                        # Create symlink if it doesn't exist
                        if not link_path.exists():
                            try:
                                # link_path.symlink_to(source, target_is_directory=True)
                                self.create_dir_link(link_path, source)
                                print(f"[Notebook] Linked: {link_name} -> {source}")
                            except OSError as e:
                                print(f"[Notebook] Could not create symlink {link_name}: {e}")
        except Exception as e:
            print(f"[Notebook] Could not setup home directory links: {e}")
        
        # Create and symlink user notebook library (if user selected one)
        if hasattr(self, 'user_library_path') and self.user_library_path:
            try:
                user_lib = Path(self.user_library_path)
                
                # Create user library directory
                user_lib.mkdir(parents=True, exist_ok=True)
                
                # Create symlink in session
                link_path = session_dir / "_user_notebooks"
                if not link_path.exists():
                    try:
                        link_path.symlink_to(user_lib, target_is_directory=True)
                        print(f"[Notebook] Linked: _user_notebooks -> {user_lib}")
                    except OSError as e:
                        print(f"[Notebook] Could not create user library symlink: {e}")
            except Exception as e:
                print(f"[Notebook] Could not setup user library: {e}")
    
    def _on_server_error(self, error: str):
        """Handle server error."""
        
        self.status_label.setText("❌ Server error")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 12px; color: red;")
        
        QtWidgets.QMessageBox.critical(
            self,
            "Jupyter Server Error",
            f"Failed to start Jupyter Lab:\n\n{error}"
        )
    
    def _open_in_browser(self):
        """Open Jupyter Lab in system browser."""
        if self.server.url:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(self.server.url))
    
    def _create_and_open_notebook(self):
        """Create a new notebook and open it in browser."""
        # Get CRS info from parent app using helper method
        crs_serial, is_mock = self._get_parent_app_info()
        
        # Create notebook
        notebook_path = self.create_notebook(
            crs_serial=crs_serial,
            session_path=self.notebook_dir,
            is_mock=is_mock
        )
        
        if notebook_path:
            # Open it in browser
            url = self.server.get_notebook_url(notebook_path)
            if url:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))
    
    def _shutdown_and_close(self):
        """Shutdown the Jupyter server and close the dock."""
        # Confirm with user
        reply = QtWidgets.QMessageBox.question(
            self,
            "Shutdown Jupyter Server",
            "Are you sure you want to shutdown the Jupyter Lab server?\n\n"
            "This will close all open notebooks in your browser.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No
        )
        
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            # Stop the server
            self.server.stop()
            
            # Find and close the dock widget
            parent_app = find_parent_with_attr(self, 'notebook_dock')
            
            if parent_app and hasattr(parent_app, 'notebook_dock'):
                # Clear the reference
                parent_app.notebook_dock = None
                
                # Find the actual dock widget
                dock = self.parentWidget()
                while dock and not isinstance(dock, QtWidgets.QDockWidget):
                    dock = dock.parentWidget()
                
                if dock:
                    dock.close()
                    dock.deleteLater()
    
    def closeEvent(self, event):
        """
        Handle panel close event.
        
        Only stops the server if this panel owns it (i.e., server was not
        passed in from external source). This allows the panel to be hidden/closed
        while keeping the server running for later reuse.
        """
        if self._owns_server:
            self.server.stop()
        super().closeEvent(event)
    
    def shutdown(self):
        """
        Explicitly shutdown the server.
        
        This should be called when the application is closing and the server
        should definitely be stopped, regardless of ownership.
        """
        self.server.stop()
    
    def is_server_running(self) -> bool:
        """
        Check if the Jupyter server is currently running.
        
        Returns:
            True if the server process is running, False otherwise
        """
        return self.server.process is not None and self.server.process.poll() is None
    
    def set_server(self, server: JupyterServerManager):
        """
        Set an external server to use.
        
        When using an external server, this panel will NOT stop the server
        when it is closed.
        
        Args:
            server: The JupyterServerManager to use
        """
        self.server = server
        self._owns_server = False
        
        # Reconnect signals
        self.server.server_ready.connect(self._on_server_ready)
        self.server.server_error.connect(self._on_server_error)
        
        # Update UI if server is already running
        if self.is_server_running():
            self._on_server_ready(self.server.url or "")
    
    def open_notebook(self, notebook_path: str):
        """
        Open a specific notebook in Jupyter Lab (in browser).
        
        Args:
            notebook_path: Full path to the .ipynb file
        """
        if not self.server.port:
            # Server not ready yet
            print(f"[Notebook] Server not ready, cannot open {notebook_path}")
            return
        
        url = self.server.get_notebook_url(notebook_path)
        if url:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))
        else:
            print(f"[Notebook] Cannot open notebook: {notebook_path}")
    
    def create_notebook(self, crs_serial: str | None = None, 
                       session_path: str | None = None,
                       is_mock: bool = False) -> str | None:
        """
        Create a new notebook with a pre-filled setup cell.
        
        Args:
            crs_serial: CRS board serial number (e.g., "0042")
            session_path: Path to the session directory (defaults to panel's notebook_dir)
            is_mock: Whether running in mock mode
            
        Returns:
            Path to the created notebook, or None on failure
        """
        # Always use the panel's notebook_dir (which is the session folder)
        nb_dir = Path(self.notebook_dir)
        # Use provided session_path for template, or default to notebook_dir
        if not session_path:
            session_path = self.notebook_dir
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{timestamp}.ipynb"
        notebook_path = nb_dir / filename
        
        # Build the setup cell content
        setup_lines = [
            "# Periscope Session Setup",
            "# This cell was auto-generated by Periscope",
            "",
            "import rfmux",
            "from rfmux import load_session, CRS",
            "import pickle",
            "from pathlib import Path",
            "",
        ]
        
        # Add session path if available
        if session_path:
            setup_lines.extend([
                f'# Session directory',
                f'session_path = Path("{session_path}")',
                "",
            ])
        
        # Add CRS connection code (different for mock vs real)
        if is_mock:
            setup_lines.extend([
                "# Note: Mock mode - CRS connection not available in notebook",
                "# Use the iPython console in Periscope for live CRS access,",
                "# or load saved .pkl files from the session directory.",
                "crs = None  # Not available in mock mode",
                "",
            ])
        elif crs_serial:
            setup_lines.extend([
                f'# Connect to CRS board',
                f'hwm = load_session(\'!HardwareMap [ !CRS {{ serial: "{crs_serial}" }} ]\')',
                f'crs = hwm.query(CRS).one()',
                f'await crs.resolve()',
                "",
            ])
        
        # Add helper code for loading session data
        setup_lines.extend([
            "# Helper: Load a pickle file from the session",
            "def load_data(filename):",
            "    \"\"\"Load data from a session pickle file.\"\"\"",
            "    filepath = session_path / filename if session_path else Path(filename)",
            "    with open(filepath, 'rb') as f:",
            "        return pickle.load(f)",
            "",
            "# Example: data = load_data('multisweep_xxx.pkl')",
        ])
        
        # For Jupyter notebook format, each line must end with \n (except last)
        setup_source = [line + "\n" for line in setup_lines[:-1]]
        if setup_lines:
            setup_source.append(setup_lines[-1])  # Last line without \n
        
        # Create the notebook JSON structure
        notebook_json = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": setup_source
                },
                {
                    "cell_type": "code", 
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": ["# Your analysis code here"]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.11"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 5
        }
        
        try:
            with open(notebook_path, 'w') as f:
                json.dump(notebook_json, f, indent=2)
            print(f"[Notebook] Created: {notebook_path}")
            return str(notebook_path)
        except Exception as e:
            print(f"[Notebook] Error creating notebook: {e}")
            return None
