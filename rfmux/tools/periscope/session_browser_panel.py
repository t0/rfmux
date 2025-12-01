"""
Session Browser Panel for Periscope Application
================================================

This module provides a dockable file browser panel for viewing and
loading session files. It displays the contents of the current session
folder using a QTreeView with QFileSystemModel.

Features:
- Displays session files in a tree view
- Double-click to load files into new panels
- Refresh button to update file list
- Open folder button to show in system file browser
- Filter by file type
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PyQt6 import QtCore, QtWidgets, QtGui

if TYPE_CHECKING:
    from .session_manager import SessionManager


class SessionBrowserPanel(QtWidgets.QWidget):
    """
    A dockable panel displaying session files with a tree view.
    
    This panel provides a file browser interface for viewing the contents
    of the current session folder. Users can double-click files to load
    them into new analysis panels.
    
    Signals:
        file_load_requested: Emitted when user double-clicks a file (file_path: str)
    
    Attributes:
        session_manager: Reference to the SessionManager instance
    """
    
    # Signal emitted when user wants to load a file
    file_load_requested = QtCore.pyqtSignal(str)  # file_path
    
    def __init__(self, session_manager: SessionManager, parent: Optional[QtWidgets.QWidget] = None):
        """
        Initialize the SessionBrowserPanel.
        
        Args:
            session_manager: The SessionManager instance to monitor
            parent: Optional parent widget
        """
        super().__init__(parent)
        
        self.session_manager = session_manager
        self._dark_mode = False
        
        self._setup_ui()
        self._connect_signals()
        self._update_session_display()
    
    def _setup_ui(self):
        """Create and configure the panel UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # ─────────────────────────────────────────────────────────────
        # Header section with session info
        # ─────────────────────────────────────────────────────────────
        
        header_group = QtWidgets.QGroupBox("Session")
        header_layout = QtWidgets.QVBoxLayout(header_group)
        header_layout.setContentsMargins(5, 5, 5, 5)
        
        # Session status label
        self.status_label = QtWidgets.QLabel("No active session")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(self.status_label)
        
        # Session path label (truncated if long)
        self.path_label = QtWidgets.QLabel("")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("font-size: 10px; color: gray;")
        self.path_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        header_layout.addWidget(self.path_label)
        
        # File count label
        self.count_label = QtWidgets.QLabel("")
        self.count_label.setStyleSheet("font-size: 10px;")
        header_layout.addWidget(self.count_label)
        
        layout.addWidget(header_group)
        
        # ─────────────────────────────────────────────────────────────
        # Filter section
        # ─────────────────────────────────────────────────────────────
        
        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.setContentsMargins(0, 0, 0, 0)
        
        filter_label = QtWidgets.QLabel("Filter:")
        filter_layout.addWidget(filter_label)
        
        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItem("All Files", "all")
        self.filter_combo.addItem("Network Analysis", "netanal")
        self.filter_combo.addItem("Multisweep", "multisweep")
        self.filter_combo.addItem("Bias KIDs", "bias")
        self.filter_combo.addItem("Noise Spectrum", "noise")
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_combo, 1)
        
        layout.addLayout(filter_layout)
        
        # ─────────────────────────────────────────────────────────────
        # File tree view
        # ─────────────────────────────────────────────────────────────
        
        self.tree_view = QtWidgets.QTreeView()
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setRootIsDecorated(False)
        self.tree_view.setSortingEnabled(True)
        self.tree_view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.tree_view.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        
        # Set up file system model (QFileSystemModel is in QtGui in this PyQt6 version)
        self.file_model = QtGui.QFileSystemModel()
        self.file_model.setNameFilters(["*.pkl"])
        self.file_model.setNameFilterDisables(False)
        self.file_model.setFilter(
            QtCore.QDir.Filter.Files | QtCore.QDir.Filter.NoDotAndDotDot
        )
        
        self.tree_view.setModel(self.file_model)
        
        # Hide unnecessary columns (keep Name, Size, Date Modified)
        self.tree_view.hideColumn(2)  # Hide Type column
        
        # Set column widths
        self.tree_view.setColumnWidth(0, 200)  # Name
        self.tree_view.setColumnWidth(1, 60)   # Size
        self.tree_view.setColumnWidth(3, 120)  # Date Modified
        
        # Sort by name descending (newest first due to timestamp naming)
        self.tree_view.sortByColumn(0, QtCore.Qt.SortOrder.DescendingOrder)
        
        # Connect double-click
        self.tree_view.doubleClicked.connect(self._on_file_double_clicked)
        self.tree_view.customContextMenuRequested.connect(self._show_context_menu)
        
        layout.addWidget(self.tree_view, 1)  # Stretch factor 1
        
        # ─────────────────────────────────────────────────────────────
        # Button row
        # ─────────────────────────────────────────────────────────────
        
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.setToolTip("Refresh the file list")
        self.refresh_btn.clicked.connect(self.refresh)
        button_layout.addWidget(self.refresh_btn)
        
        self.open_folder_btn = QtWidgets.QPushButton("Open Folder")
        self.open_folder_btn.setToolTip("Open session folder in file browser")
        self.open_folder_btn.clicked.connect(self._on_open_folder_clicked)
        button_layout.addWidget(self.open_folder_btn)
        
        layout.addLayout(button_layout)
        
        # ─────────────────────────────────────────────────────────────
        # Placeholder for when no session is active
        # ─────────────────────────────────────────────────────────────
        
        self.placeholder_label = QtWidgets.QLabel(
            "No session active.\n\n"
            "Use Session → Start New Session\n"
            "to begin auto-exporting data."
        )
        self.placeholder_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setStyleSheet("color: gray; padding: 20px;")
        self.placeholder_label.setWordWrap(True)
        
        # Initially show placeholder, hide tree
        self.tree_view.setVisible(False)
        self.filter_combo.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.open_folder_btn.setEnabled(False)
        
        layout.addWidget(self.placeholder_label)
    
    def _connect_signals(self):
        """Connect signals from the session manager."""
        self.session_manager.session_started.connect(self._on_session_started)
        self.session_manager.session_ended.connect(self._on_session_ended)
        self.session_manager.session_updated.connect(self.refresh)
        self.session_manager.file_exported.connect(self._on_file_exported)
    
    # ─────────────────────────────────────────────────────────────────
    # Public Methods
    # ─────────────────────────────────────────────────────────────────
    
    def refresh(self):
        """Refresh the file list display."""
        if not self.session_manager.is_active:
            return
        
        # Force the model to refresh
        root_path = str(self.session_manager.session_path)
        self.file_model.setRootPath(root_path)
        self.tree_view.setRootIndex(self.file_model.index(root_path))
        
        # Apply current filter
        self._apply_filter()
        
        # Update count label
        self._update_file_count()
    
    def apply_theme(self, dark_mode: bool):
        """
        Apply the dark/light theme to the panel.
        
        Args:
            dark_mode: True for dark theme, False for light theme
        """
        self._dark_mode = dark_mode
        
        if dark_mode:
            self.path_label.setStyleSheet("font-size: 10px; color: #aaaaaa;")
            self.placeholder_label.setStyleSheet("color: #888888; padding: 20px;")
        else:
            self.path_label.setStyleSheet("font-size: 10px; color: gray;")
            self.placeholder_label.setStyleSheet("color: gray; padding: 20px;")
    
    # ─────────────────────────────────────────────────────────────────
    # Session State Handlers
    # ─────────────────────────────────────────────────────────────────
    
    @QtCore.pyqtSlot(str)
    def _on_session_started(self, session_path: str):
        """Handle session start."""
        self._update_session_display()
        
        # Set the file model root to the session path
        self.file_model.setRootPath(session_path)
        self.tree_view.setRootIndex(self.file_model.index(session_path))
        
        # Show tree, hide placeholder
        self.tree_view.setVisible(True)
        self.placeholder_label.setVisible(False)
        
        # Enable controls
        self.filter_combo.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.open_folder_btn.setEnabled(True)
        
        self._update_file_count()
    
    @QtCore.pyqtSlot()
    def _on_session_ended(self):
        """Handle session end."""
        self._update_session_display()
        
        # Clear the tree
        self.file_model.setRootPath("")
        
        # Hide tree, show placeholder
        self.tree_view.setVisible(False)
        self.placeholder_label.setVisible(True)
        
        # Disable controls
        self.filter_combo.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.open_folder_btn.setEnabled(False)
    
    @QtCore.pyqtSlot(str, str)
    def _on_file_exported(self, file_path: str, data_type: str):
        """Handle new file export - refresh and highlight."""
        self.refresh()
        
        # Try to select the newly exported file
        index = self.file_model.index(file_path)
        if index.isValid():
            self.tree_view.setCurrentIndex(index)
            self.tree_view.scrollTo(index)
    
    def _update_session_display(self):
        """Update the session status labels."""
        if self.session_manager.is_active:
            self.status_label.setText(f"📁 {self.session_manager.session_name}")
            self.status_label.setStyleSheet("font-weight: bold; color: green;")
            
            path = str(self.session_manager.session_path)
            # Truncate long paths
            if len(path) > 40:
                path = "..." + path[-37:]
            self.path_label.setText(path)
            self.path_label.setToolTip(str(self.session_manager.session_path))
        else:
            self.status_label.setText("No active session")
            self.status_label.setStyleSheet("font-weight: bold;")
            self.path_label.setText("")
            self.path_label.setToolTip("")
            self.count_label.setText("")
    
    def _update_file_count(self):
        """Update the file count label."""
        if not self.session_manager.is_active:
            self.count_label.setText("")
            return
        
        summary = self.session_manager.get_session_summary()
        total = summary.get('total_files', 0)
        
        if total == 0:
            self.count_label.setText("No files yet")
        elif total == 1:
            self.count_label.setText("1 file")
        else:
            self.count_label.setText(f"{total} files")
    
    # ─────────────────────────────────────────────────────────────────
    # Filter Handling
    # ─────────────────────────────────────────────────────────────────
    
    def _on_filter_changed(self, index: int):
        """Handle filter combo box changes."""
        self._apply_filter()
    
    def _apply_filter(self):
        """Apply the current filter to the file model."""
        filter_type = self.filter_combo.currentData()
        
        if filter_type == "all":
            self.file_model.setNameFilters(["*.pkl"])
        else:
            # Filter by data type in filename
            self.file_model.setNameFilters([f"*_{filter_type}_*.pkl"])
    
    # ─────────────────────────────────────────────────────────────────
    # File Operations
    # ─────────────────────────────────────────────────────────────────
    
    def _on_file_double_clicked(self, index: QtCore.QModelIndex):
        """Handle double-click on a file."""
        if not index.isValid():
            return
        
        file_path = self.file_model.filePath(index)
        
        # Check it's a file (not a directory)
        if not Path(file_path).is_file():
            return
        
        # Emit signal to request file load
        self.file_load_requested.emit(file_path)
    
    def _on_open_folder_clicked(self):
        """Handle Open Folder button click."""
        self.session_manager.open_session_folder()
    
    def _show_context_menu(self, position: QtCore.QPoint):
        """Show context menu for file operations."""
        index = self.tree_view.indexAt(position)
        if not index.isValid():
            return
        
        file_path = self.file_model.filePath(index)
        if not Path(file_path).is_file():
            return
        
        menu = QtWidgets.QMenu(self)
        
        # Load action
        load_action = menu.addAction("Load File")
        load_action.triggered.connect(lambda: self.file_load_requested.emit(file_path))
        
        menu.addSeparator()
        
        # Copy path action
        copy_path_action = menu.addAction("Copy Path")
        copy_path_action.triggered.connect(
            lambda: QtWidgets.QApplication.clipboard().setText(file_path)
        )
        
        # Show in folder action
        show_in_folder_action = menu.addAction("Show in Folder")
        show_in_folder_action.triggered.connect(
            lambda: self._show_file_in_folder(file_path)
        )
        
        menu.addSeparator()
        
        # Delete action
        delete_action = menu.addAction("Delete File")
        delete_action.triggered.connect(lambda: self._delete_file(file_path))
        
        menu.exec(self.tree_view.viewport().mapToGlobal(position))
    
    def _show_file_in_folder(self, file_path: str):
        """Open the containing folder and select the file."""
        import subprocess
        import sys
        
        try:
            if sys.platform == 'darwin':  # macOS
                subprocess.run(['open', '-R', file_path], check=True)
            elif sys.platform == 'win32':  # Windows
                subprocess.run(['explorer', '/select,', file_path], check=True)
            else:  # Linux
                # xdg-open opens the folder, not selecting the file
                folder = str(Path(file_path).parent)
                subprocess.run(['xdg-open', folder], check=True)
        except Exception as e:
            print(f"[SessionBrowser] Error showing file in folder: {e}")
    
    def _delete_file(self, file_path: str):
        """Delete a file with confirmation."""
        filename = Path(file_path).name
        
        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete File",
            f"Are you sure you want to delete:\n{filename}?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No
        )
        
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            try:
                Path(file_path).unlink()
                self.refresh()
                print(f"[SessionBrowser] Deleted: {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Delete Error",
                    f"Could not delete file:\n{str(e)}"
                )
