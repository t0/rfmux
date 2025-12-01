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
"""

from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtCore import Qt
from typing import Dict, List, Optional


class RenamableDockWidget(QtWidgets.QDockWidget):
    """
    Custom QDockWidget that allows users to rename it via right-click context menu.
    """
    
    def __init__(self, title: str, parent: QtWidgets.QWidget = None):
        super().__init__(title, parent)
        
    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        """
        Handle right-click context menu on the dock title bar.
        
        Adds a "Rename" option to the context menu.
        """
        # Create context menu
        menu = QtWidgets.QMenu(self)
        
        # Add rename action
        rename_action = menu.addAction("Rename Tab...")
        
        # Show menu and get selected action
        action = menu.exec(event.globalPos())
        
        if action == rename_action:
            self._show_rename_dialog()
    
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


class PeriscopeDockManager:
    """
    Manages the lifecycle and organization of dock widgets in Periscope.
    
    This class provides a centralized interface for creating, tracking, and
    manipulating QDockWidget instances that contain analysis panels.
    
    Attributes:
        main_window: Reference to the main Periscope QMainWindow
        dock_widgets: Dictionary mapping unique dock IDs to QDockWidget instances
        dock_counter: Counter for generating unique dock IDs
    """
    
    def __init__(self, main_window: QtWidgets.QMainWindow):
        """
        Initialize the DockManager.
        
        Args:
            main_window: The main Periscope QMainWindow instance
        """
        self.main_window = main_window
        self.dock_widgets: Dict[str, QtWidgets.QDockWidget] = {}
        self.dock_counter = 0
        
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
