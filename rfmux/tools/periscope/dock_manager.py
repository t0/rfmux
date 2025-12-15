"""
Dock Manager for Periscope Application
=======================================

This module provides centralized management of QDockWidget instances in the
Periscope application. It handles the creation, tracking, lifecycle, and
organization of dockable panels (NetworkAnalysis, Multisweep, DetectorDigest).

The DockManager enables:
- Creating dock widgets that wrap panel content
- Tracking active docks with unique identifiers
- Tabifying multiple docks together
- Tiling docks horizontally or vertically
- Floating/docking operations
- Cleanup and removal of docks
- Screenshot export of panel contents
"""

from PyQt6 import QtCore, QtWidgets, QtGui, sip
from PyQt6.QtCore import Qt
from typing import Dict, List, Optional
import datetime
import os


class RenamableDockWidget(QtWidgets.QDockWidget):
    """
    Custom QDockWidget that allows users to rename it via double-click on the title bar.
    
    Double-click on the title bar (when floating or docked alone) triggers the rename dialog.
    For tabified docks, the PeriscopeDockManager handles double-click on the tab bar.
    """
    
    def __init__(self, title: str, parent: QtWidgets.QWidget = None):
        super().__init__(title, parent)
        
    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent):
        """
        Handle double-click on the dock title bar to trigger rename.
        
        Only triggers rename if the click is on the title bar area (top ~30 pixels),
        not on the content area.
        """
        # Get the title bar height - approximate or use actual if available
        title_bar_height = 30  # Typical title bar height
        
        # Check if click is in the title bar area
        if event.position().y() < title_bar_height:
            self._show_rename_dialog()
        else:
            # Pass to parent for normal handling
            super().mouseDoubleClickEvent(event)
    
    def _show_rename_dialog(self):
        """Show a dialog to rename the dock widget."""
        current_title = self.windowTitle()
        
        # Show input dialog
        new_title, ok = QtWidgets.QInputDialog.getText(
            self,
            "Rename Tab",
            "Enter new name for this tab:",
            QtWidgets.QLineEdit.EchoMode.Normal,
            current_title
        )
        
        # Update title if user confirmed and provided non-empty text
        if ok and new_title.strip():
            self.setWindowTitle(new_title.strip())


class PeriscopeDockManager(QtCore.QObject):
    """
    Manages the lifecycle and organization of dock widgets in Periscope.
    
    This class provides a centralized interface for creating, tracking, and
    manipulating QDockWidget instances that contain analysis panels.
    
    Attributes:
        main_window: Reference to the main Periscope QMainWindow
        dock_widgets: Dictionary mapping unique dock IDs to QDockWidget instances
        dock_counter: Counter for generating unique dock IDs
        protected_docks: Set of dock IDs that should be hidden (not closed) when user clicks X
    """
    
    def __init__(self, main_window: QtWidgets.QMainWindow):
        """
        Initialize the DockManager.
        
        Args:
            main_window: The main Periscope QMainWindow instance
        """
        super().__init__(main_window)
        self.main_window = main_window
        self.dock_widgets: Dict[str, QtWidgets.QDockWidget] = {}
        self.dock_counter = 0
        self._monitored_tab_bars: set = set()
        self.protected_docks: set = set()  # Docks that should be hidden, not closed
        
        # Install event filter on main window to detect tab bar creation
        self.main_window.installEventFilter(self)
        
    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """
        Event filter to handle tab bar double-clicks for rename.
        
        Also monitors for child events to detect new tab bars.
        """
        # Handle tab bar double-click
        if isinstance(obj, QtWidgets.QTabBar):
            if event.type() == QtCore.QEvent.Type.MouseButtonDblClick:
                mouse_event = event
                tab_index = obj.tabAt(mouse_event.position().toPoint())
                if tab_index >= 0:
                    self._rename_tab_at_index(obj, tab_index)
                    return True
        
        # Monitor for new child widgets (to detect tab bars)
        if obj == self.main_window and event.type() == QtCore.QEvent.Type.ChildAdded:
            # Use a timer to check for tab bars after the event is processed
            QtCore.QTimer.singleShot(100, self._scan_for_tab_bars)
        
        return False
    
    def _scan_for_tab_bars(self):
        """Scan for any new QTabBar widgets in the main window and install event filter."""
        for tab_bar in self.main_window.findChildren(QtWidgets.QTabBar):
            if id(tab_bar) not in self._monitored_tab_bars:
                tab_bar.installEventFilter(self)
                self._monitored_tab_bars.add(id(tab_bar))

                tab_bar.setTabsClosable(True)
                # Use UniqueConnection to prevent duplicate signal connections
                # Wrap in try-except to handle edge cases where connection already exists
                try:
                    tab_bar.tabCloseRequested.connect(
                        self._on_tab_close_requested,
                        Qt.ConnectionType.UniqueConnection
                    )
                except TypeError:
                    # Connection already exists, silently continue
                    pass

    def _on_tab_close_requested(self, index: int):
        """
        Handle tab close request from the tab bar.
        
        For protected docks (like Main and Jupyter), hides instead of closes.
        For other docks, removes them completely.
        """
        tab_bar = self.sender()
        if not isinstance(tab_bar, QtWidgets.QTabBar):
            return
    
        tab_title = tab_bar.tabText(index)
        
        for dock_id, dock in list(self.dock_widgets.items()):
            # Check if the Qt C++ object has been deleted
            if sip.isdeleted(dock):
                # Clean up stale reference
                self.dock_widgets.pop(dock_id, None)
                continue
            
            if dock.windowTitle() == tab_title:
                if self.is_protected(dock_id):
                    # Hide protected docks instead of removing them
                    dock.hide()
                    print(f"[DockManager] Protected dock '{tab_title}' hidden (use View menu to show again)")
                else:
                    self.remove_dock(dock_id)
                break
    
    def protect_dock(self, dock_id: str) -> None:
        """
        Mark a dock as protected (will be hidden instead of closed).
        
        Protected docks are not removed when the user clicks the X button.
        Instead, they are hidden and can be shown again via the View menu.
        
        Args:
            dock_id: The unique identifier of the dock to protect
        """
        self.protected_docks.add(dock_id)
    
    def unprotect_dock(self, dock_id: str) -> None:
        """
        Remove protection from a dock.
        
        Args:
            dock_id: The unique identifier of the dock to unprotect
        """
        self.protected_docks.discard(dock_id)
    
    def is_protected(self, dock_id: str) -> bool:
        """
        Check if a dock is protected.
        
        Args:
            dock_id: The unique identifier of the dock
            
        Returns:
            True if the dock is protected, False otherwise
        """
        return dock_id in self.protected_docks
      
    
    def _rename_tab_at_index(self, tab_bar: QtWidgets.QTabBar, index: int):
        """
        Rename the dock widget corresponding to a tab at the given index.
        
        Args:
            tab_bar: The QTabBar containing the tab
            index: The index of the tab to rename
        """
        # Get the tab text (current title)
        current_title = tab_bar.tabText(index)
        
        # Show input dialog
        new_title, ok = QtWidgets.QInputDialog.getText(
            self.main_window,
            "Rename Tab",
            "Enter new name for this tab:",
            QtWidgets.QLineEdit.EchoMode.Normal,
            current_title
        )
        
        # Update title if user confirmed and provided non-empty text
        if ok and new_title.strip():
            new_title = new_title.strip()
            # Update the tab text
            tab_bar.setTabText(index, new_title)
            
            # Also update the corresponding dock widget's window title
            # Find the dock by matching the old title
            for dock in self.dock_widgets.values():
                if dock.windowTitle() == current_title:
                    dock.setWindowTitle(new_title)
                    break
        
    def create_dock(self, 
                   widget: QtWidgets.QWidget, 
                   title: str,
                   unique_id: Optional[str] = None,
                   area: Qt.DockWidgetArea = Qt.DockWidgetArea.BottomDockWidgetArea,
                   allowed_areas: Qt.DockWidgetArea = None) -> QtWidgets.QDockWidget:
        """
        Create a new dock widget containing the specified widget.
        
        Args:
            widget: The QWidget to be wrapped in the dock (e.g., NetworkAnalysisPanel)
            title: The title to display in the dock widget's title bar
            unique_id: Optional unique identifier for the dock. If None, one will be generated.
            area: The initial dock area where the dock will be placed
            allowed_areas: Bitwise OR of allowed dock areas. If None, allows Left, Right, and Bottom.
        
        Returns:
            The created QDockWidget instance
        """
        # Generate unique ID if not provided
        if unique_id is None:
            unique_id = f"dock_{self.dock_counter}"
            self.dock_counter += 1
        
        # Ensure unique ID is actually unique
        if unique_id in self.dock_widgets:
            # Append counter to make it unique
            base_id = unique_id
            counter = 1
            while f"{base_id}_{counter}" in self.dock_widgets:
                counter += 1
            unique_id = f"{base_id}_{counter}"
        
        # Create the renamable dock widget
        dock = RenamableDockWidget(title, self.main_window)
        dock.setObjectName(unique_id)  # Set object name for state persistence
        dock.setWidget(widget)
        
        # Configure allowed areas (default to Left, Right, Bottom)
        if allowed_areas is None:
            allowed_areas = (
                Qt.DockWidgetArea.LeftDockWidgetArea |
                Qt.DockWidgetArea.RightDockWidgetArea |
                Qt.DockWidgetArea.BottomDockWidgetArea
            )
        dock.setAllowedAreas(allowed_areas)
        
        # Configure dock features
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        
        # Connect close event to cleanup
        dock.visibilityChanged.connect(
            lambda visible: self._on_dock_visibility_changed(unique_id, visible)
        )
        
        # Add to main window
        self.main_window.addDockWidget(area, dock)
        
        # Store in tracking dictionary
        self.dock_widgets[unique_id] = dock
        
        # Scan for any new tab bars after adding dock
        QtCore.QTimer.singleShot(100, self._scan_for_tab_bars)
        
        return dock
    
    def get_dock(self, dock_id: str) -> Optional[QtWidgets.QDockWidget]:
        """
        Retrieve a dock widget by its unique ID.
        
        Args:
            dock_id: The unique identifier of the dock
            
        Returns:
            The QDockWidget if found, None otherwise
        """
        return self.dock_widgets.get(dock_id)
    
    def get_dock_widget(self, dock_id: str) -> Optional[QtWidgets.QWidget]:
        """
        Retrieve the widget contained within a dock.
        
        Args:
            dock_id: The unique identifier of the dock
            
        Returns:
            The contained QWidget (panel) if found, None otherwise
        """
        dock = self.get_dock(dock_id)
        return dock.widget() if dock else None
    
    def find_dock_for_widget(self, widget: QtWidgets.QWidget) -> Optional[QtWidgets.QDockWidget]:
        """
        Find the dock that contains the specified widget.
        
        Args:
            widget: The QWidget to search for
            
        Returns:
            The QDockWidget containing the widget, or None if not found
        """
        for dock in self.dock_widgets.values():
            if dock.widget() == widget:
                return dock
        return None
    
    def remove_dock(self, dock_id: str) -> bool:
        """
        Remove and clean up a dock widget.
        
        Args:
            dock_id: The unique identifier of the dock to remove
            
        Returns:
            True if the dock was found and removed, False otherwise
        """
        dock = self.dock_widgets.pop(dock_id, None)
        if dock:
            # Remove from main window
            self.main_window.removeDockWidget(dock)
            # Clean up the dock widget
            dock.close()
            dock.deleteLater()
            return True
        return False
    
    def tabify_docks(self, dock_ids: List[str]) -> bool:
        """
        Group multiple docks together as tabs.
        
        Args:
            dock_ids: List of dock IDs to tabify together
            
        Returns:
            True if successful, False if any docks were not found
        """
        if len(dock_ids) < 2:
            return False
        
        # Get all dock widgets
        docks = [self.get_dock(dock_id) for dock_id in dock_ids]
        
        # Check if all docks were found
        if None in docks:
            return False
        
        # Tabify them together (first dock becomes the base)
        for i in range(1, len(docks)):
            self.main_window.tabifyDockWidget(docks[0], docks[i])
        
        # Scan for any new tab bars after tabifying
        QtCore.QTimer.singleShot(100, self._scan_for_tab_bars)
        
        return True
    
    def tile_docks_horizontally(self, dock_ids: List[str]) -> bool:
        """
        Arrange docks in a horizontal layout.
        
        Args:
            dock_ids: List of dock IDs to tile
            
        Returns:
            True if successful, False otherwise
        """
        if len(dock_ids) < 2:
            return False
        
        # Get all dock widgets
        docks = [self.get_dock(dock_id) for dock_id in dock_ids]
        
        # Check if all docks were found
        if None in docks:
            return False
        
        # Split them horizontally (place side by side in bottom area)
        for dock in docks:
            self.main_window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        
        # Split them horizontally using Qt's split mechanism
        for i in range(1, len(docks)):
            self.main_window.splitDockWidget(docks[i-1], docks[i], 
                                            Qt.Orientation.Horizontal)
        
        return True
    
    def tile_docks_vertically(self, dock_ids: List[str]) -> bool:
        """
        Arrange docks in a vertical layout.
        
        Args:
            dock_ids: List of dock IDs to tile
            
        Returns:
            True if successful, False otherwise
        """
        if len(dock_ids) < 2:
            return False
        
        # Get all dock widgets
        docks = [self.get_dock(dock_id) for dock_id in dock_ids]
        
        # Check if all docks were found
        if None in docks:
            return False
        
        # Split them vertically
        for i in range(1, len(docks)):
            self.main_window.splitDockWidget(docks[i-1], docks[i], 
                                            Qt.Orientation.Vertical)
        
        return True
    
    def float_all_docks(self) -> None:
        """Float all docks (undock them into separate windows)."""
        for dock in self.dock_widgets.values():
            if not dock.isFloating():
                dock.setFloating(True)
    
    def dock_all_docks(self) -> None:
        """Dock all floating docks back into the main window."""
        for dock in self.dock_widgets.values():
            if dock.isFloating():
                dock.setFloating(False)
    
    def get_all_dock_ids(self) -> List[str]:
        """
        Get a list of all tracked dock IDs.
        
        Returns:
            List of dock unique identifiers
        """
        return list(self.dock_widgets.keys())
    
    def close_all_docks(self) -> None:
        """Close and remove all docks."""
        # Create a copy of keys to avoid dictionary modification during iteration
        dock_ids = list(self.dock_widgets.keys())
        for dock_id in dock_ids:
            self.remove_dock(dock_id)
    
    def export_screenshot(self, dock_id: str, filepath: str = None, 
                          session_manager=None, scale: int = 2) -> Optional[str]:
        """
        Export a screenshot of a dock widget's content as PNG with high DPI support.
        
        Args:
            dock_id: The unique identifier of the dock to screenshot
            filepath: Optional explicit filepath. If None and session_manager is provided,
                     auto-generates path in session folder. If both are None, prompts user.
            session_manager: Optional SessionManager for auto-export to session folder
            scale: Resolution scale factor (default 2 for 2x resolution)
            
        Returns:
            The filepath where the screenshot was saved, or None if user dialog is shown
            (async) or if cancelled/failed. When a dialog is shown, the screenshot is
            saved via callback and this method returns None immediately.
        """
        dock = self.get_dock(dock_id)
        if not dock:
            return None
        
        # Get the content widget
        widget = dock.widget()
        if not widget:
            return None
        
        # Generate filename based on dock title and timestamp
        dock_title = dock.windowTitle()
        # Clean the title for use as filename
        clean_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in dock_title)
        clean_title = clean_title.replace(" ", "_")
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        filename = f"screenshot_{clean_title}_{timestamp}.png"
        
        # Determine filepath
        if filepath is None:
            if session_manager is not None and session_manager.session_path:
                # Auto-save to session folder
                filepath = os.path.join(session_manager.session_path, filename)
            else:
                # Prompt user for location with non-blocking dialog
                # Store state for the callback
                self._pending_screenshot_widget = widget
                self._pending_screenshot_scale = scale
                
                # Create non-blocking file dialog (prevents hanging on Linux)
                dlg = QtWidgets.QFileDialog(self.main_window, "Save Screenshot")
                dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
                dlg.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
                dlg.setNameFilters(["PNG Images (*.png)", "All Files (*)"])
                dlg.setDefaultSuffix("png")
                dlg.selectFile(filename)
                
                # Connect signal for handling file selection
                dlg.fileSelected.connect(self._handle_screenshot_file_selected)
                
                # Show the dialog non-modally (returns immediately, doesn't block)
                dlg.open()
                return None  # Return immediately, save happens in callback
        
        # Capture the widget as a high-DPI pixmap (synchronous path)
        return self._save_screenshot(widget, filepath, scale, session_manager)
    
    def _handle_screenshot_file_selected(self, filepath: str):
        """
        Handle file selection from the non-blocking screenshot dialog.
        
        Args:
            filepath: The path selected by the user
        """
        if not filepath:
            return
        
        widget = getattr(self, '_pending_screenshot_widget', None)
        scale = getattr(self, '_pending_screenshot_scale', 2)
        
        if widget:
            self._save_screenshot(widget, filepath, scale)
        
        # Clean up state
        self._pending_screenshot_widget = None
        self._pending_screenshot_scale = 2
    
    def _save_screenshot(self, widget: QtWidgets.QWidget, filepath: str, scale: int, session_manager=None) -> Optional[str]:
        """
        Save a screenshot of a widget to a file.
        
        Args:
            widget: The widget to capture
            filepath: Path to save the PNG file
            scale: Resolution scale factor
            
        Returns:
            The filepath if successful, None on failure
        """
        try:
            size = widget.size()
            # Create a pixmap at the scaled resolution
            pixmap = QtGui.QPixmap(size.width() * scale, size.height() * scale)
            pixmap.setDevicePixelRatio(scale)
            pixmap.fill(Qt.GlobalColor.transparent)
            
            # Render the widget to the pixmap
            painter = QtGui.QPainter(pixmap)
            widget.render(painter)
            painter.end()
            
            pixmap.save(filepath, "PNG")
            if session_manager is not None:
                session_manager.register_screenshot(filepath)
            print(f"[Screenshot] Saved: {filepath} ({size.width() * scale}x{size.height() * scale} px)")
            return filepath
        except Exception as e:
            print(f"[Screenshot] Error saving screenshot: {e}")
            QtWidgets.QMessageBox.warning(
                self.main_window,
                "Screenshot Error",
                f"Failed to save screenshot:\n{str(e)}"
            )
            return None
    
    def _on_dock_visibility_changed(self, dock_id: str, visible: bool) -> None:
        """
        Handle dock visibility changes.
        
        This is called when a dock is shown or hidden. If a dock is closed
        (hidden), we may want to clean it up depending on the use case.
        For now, we keep it tracked even when hidden to allow re-showing.
        
        Args:
            dock_id: The unique identifier of the dock
            visible: True if dock became visible, False if hidden
        """
        # For now, we don't remove hidden docks automatically
        # They can be re-shown via the Window menu
        pass
    
    def save_state(self) -> QtCore.QByteArray:
        """
        Save the current dock layout state.
        
        Returns:
            QByteArray containing the serialized dock state
        """
        return self.main_window.saveState()
    
    def restore_state(self, state: QtCore.QByteArray) -> bool:
        """
        Restore a previously saved dock layout state.
        
        Args:
            state: QByteArray containing the serialized dock state
            
        Returns:
            True if restore was successful
        """
        return self.main_window.restoreState(state)
