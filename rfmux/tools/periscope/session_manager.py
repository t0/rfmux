"""
Session Manager for Periscope Application
==========================================

This module provides centralized session management for automatic data export
in the Periscope application. It handles:
- Creating and managing session folders
- Pointing ``store`` at the session folder, so measurements save there
- Tracking session state and metadata

Usage:
    session_mgr = SessionManager(parent=main_window)
    session_mgr.start_session("/path/to/data", "my_session")
"""

from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6 import QtCore

from ...tuning import store
from rfmux.core import session_folder as _session


class SessionManager(QtCore.QObject):
    """
    Manages Periscope sessions for automatic data export.
    
    The SessionManager coordinates the export of analysis data to a
    user-selected session folder. It provides automatic filename generation
    with timestamps and supports multiple data types (network analysis,
    multisweep, bias, noise spectrum).
    
    Signals:
        session_started: Emitted when a new session begins (session_path: str)
        session_ended: Emitted when session is closed
        file_exported: Emitted after each file export (file_path: str, data_type: str)
        session_updated: Emitted when session contents change
    
    Attributes:
        session_path: Path to the current session folder, or None if no session
        auto_export_enabled: Whether auto-export is enabled (default True)
        session_metadata: Dictionary of session metadata
    """
    
    # Signals for session state changes
    session_started = QtCore.pyqtSignal(str)     # session_path
    session_ended = QtCore.pyqtSignal()
    file_exported = QtCore.pyqtSignal(str, str)  # file_path, data_type
    session_updated = QtCore.pyqtSignal()
    
    # Supported data types and their descriptions
    DATA_TYPES = {
        'netanal': 'Network Analysis',
        'multisweep': 'Multisweep Analysis',
        'noise': 'Noise Spectrum',
        'screenshot': 'Screenshot' ,
    }

    ARTIFACT_TYPES = {'screenshot': 'Screenshots',}
    
    def __init__(self, parent: Optional[QtCore.QObject] = None):
        """
        Initialize the SessionManager.
        
        Args:
            parent: Optional parent QObject (typically the main Periscope window)
        """
        super().__init__(parent)
        
        self._session_path: Optional[Path] = None
        self._auto_export_enabled: bool = True
        self._session_metadata: Dict[str, Any] = {}
        self._export_count: int = 0
        self._session_start_time: Optional[datetime.datetime] = None
    
    # ─────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────
    
    @property
    def session_path(self) -> Optional[Path]:
        """Get the current session folder path."""
        return self._session_path
    
    @property
    def is_active(self) -> bool:
        """Check if a session is currently active."""
        return self._session_path is not None
    
    @property
    def auto_export_enabled(self) -> bool:
        """Check if auto-export is enabled."""
        return self._auto_export_enabled
    
    @auto_export_enabled.setter
    def auto_export_enabled(self, enabled: bool):
        """Enable or disable auto-export."""
        self._auto_export_enabled = enabled
    
    @property
    def session_metadata(self) -> Dict[str, Any]:
        """Get the session metadata dictionary."""
        return self._session_metadata.copy()
    
    @property
    def export_count(self) -> int:
        """Get the number of files exported in this session."""
        return self._export_count
    
    @property
    def session_name(self) -> Optional[str]:
        """Get the session folder name."""
        return self._session_path.name if self._session_path else None
    
    # ─────────────────────────────────────────────────────────────────
    # Session Lifecycle Methods
    # ─────────────────────────────────────────────────────────────────
    
    def start_session(self, base_path: str, folder_name: Optional[str] = None) -> Path:
        """
        Start a new session in the specified base path.
        
        Creates a new session folder and initializes session state.
        If a session is already active, it will be ended first.
        
        Args:
            base_path: Parent directory where the session folder will be created
            folder_name: Name for the session folder. If None, generates a
                        timestamped name like 'session_20251201_092000'
        
        Returns:
            Path to the created session folder
        
        Raises:
            ValueError: If base_path doesn't exist or isn't a directory
            OSError: If folder creation fails
        """
        base = Path(base_path)
        
        if not base.exists():
            raise ValueError(f"Base path does not exist: {base_path}")
        if not base.is_dir():
            raise ValueError(f"Base path is not a directory: {base_path}")
        
        # End any existing session
        if self.is_active:
            self.end_session()
        
        # Generate folder name if not provided
        if folder_name is None:
            folder_name = _session.folder_name()
        
        # Create the session folder
        session_path = base / folder_name
        session_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize session state
        self._session_path = session_path
        # The library writes into the session folder, flat, as the notebooks do.
        store.set_output_directory(session_path)
        
        # Save the full session path so it's pre-selected next time
        from . import settings
        settings.set_last_session_path(str(session_path))
        self._export_count = 0
        self._session_start_time = datetime.datetime.now()
        self._session_metadata = _session.new_metadata(session_path)
        
        # Try to capture CRS info from parent
        if self.parent() and hasattr(self.parent(), 'crs'):
            crs = self.parent().crs
            if crs is not None:
                try:
                    self._session_metadata['crs_serial'] = str(getattr(crs, 'serial', 'unknown'))
                except Exception:
                    pass
            if hasattr(self.parent(), 'module'):
                self._session_metadata['module'] = self.parent().module
        
        # Save initial metadata
        self._save_metadata()
        
        # Emit signal
        self.session_started.emit(str(session_path))
        
        #print(f"[Session] Started new session: {session_path}")
        return session_path
    
    def load_session(self, session_path: str) -> bool:
        """
        Load an existing session folder.
        
        Sets the session path to an existing folder and loads any
        existing metadata.
        
        Args:
            session_path: Path to the existing session folder
        
        Returns:
            True if session was loaded successfully, False otherwise
        """
        path = Path(session_path)
        
        if not path.exists():
            print(f"[Session] Error: Path does not exist: {session_path}")
            return False
        if not path.is_dir():
            print(f"[Session] Error: Path is not a directory: {session_path}")
            return False
        
        # End any existing session
        if self.is_active:
            self.end_session()
        
        # Set the session path
        self._session_path = path
        store.set_output_directory(path)
        self._session_start_time = datetime.datetime.now()
        
        # Try to load existing metadata
        if _session.is_session(path):
            self._session_metadata = _session.load_metadata(path)
            if self._session_metadata:
                self._export_count = len(self._session_metadata.get('exports', []))
            else:
                print("[Session] Warning: Could not load metadata")
                self._session_metadata = {'loaded': datetime.datetime.now().isoformat()}
                self._export_count = len(list(path.glob('*.pkl')))
        else:
            self._session_metadata = {'loaded': datetime.datetime.now().isoformat()}
            self._export_count = len(list(path.glob('*.pkl')))
        
        # Emit signal
        self.session_started.emit(str(path))
        
        #print(f"[Session] Loaded session: {path}")
        return True
    
    def end_session(self):
        """
        End the current session.
        
        Saves final metadata and clears session state.
        Does nothing if no session is active.
        """
        if not self.is_active:
            return
        
        # Update and save final metadata
        self._session_metadata['ended'] = datetime.datetime.now().isoformat()
        self._session_metadata['total_exports'] = self._export_count
        self._save_metadata()
        
        session_name = self.session_name
        
        # Clear state
        store.set_output_directory(None)
        self._session_path = None
        self._export_count = 0
        self._session_start_time = None
        self._session_metadata = {}
        
        # Emit signal
        self.session_ended.emit()
        
        print(f"[Session] Ended session: {session_name}")
    
    # ─────────────────────────────────────────────────────────────────
    # Session Query Methods
    # ─────────────────────────────────────────────────────────────────
    
    def get_session_files(self) -> List[Path]:
        """
        Get a list of pickle files in the current session folder.
        
        Returns:
            List of Path objects for each .pkl file and screenshot, sorted by name (newest first)
        """
        if not self.is_active or self._session_path is None:
            return []
        
        files = list(self._session_path.glob('*.pkl'))
        files += list(self._session_path.glob('screenshot_*.png'))
        # Sort by name descending (newest timestamps first)
        files.sort(key=lambda p: p.name, reverse=True)
        return files
    
    def get_files_by_type(self, data_type: str) -> List[Path]:
        """
        Get session files filtered by data type.
        
        Args:
            data_type: Type to filter by (netanal, multisweep, bias, noise, screenshot)
        
        Returns:
            List of Path objects matching the data type
        """
        if not self.is_active or self._session_path is None:
            return []
        
        if data_type == 'screenshot':
            files = list(self._session_path.glob('screenshot_*.png'))
        else:
            pattern = f'*_{data_type}_*.pkl'
            files = list(self._session_path.glob(pattern))
        files.sort(key=lambda p: p.name, reverse=True)
        return files
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current session.
        
        Returns:
            Dictionary with session info (path, duration, file counts, etc.)
        """
        if not self.is_active:
            return {'active': False}
        
        files = self.get_session_files()
        
        # Count files by type
        type_counts = {}
        for data_type in self.DATA_TYPES:
            type_counts[data_type] = len(self.get_files_by_type(data_type))

        type_counts['screenshot'] = len(self.get_files_by_type('screenshot'))
        
        # Calculate session duration
        duration = None
        if self._session_start_time:
            duration = datetime.datetime.now() - self._session_start_time
        
        return {
            'active': True,
            'path': str(self._session_path),
            'name': self.session_name,
            'total_files': len(files),
            'export_count': self._export_count,
            'files_by_type': type_counts,
            'duration': str(duration) if duration else None,
            'auto_export_enabled': self._auto_export_enabled,
        }
    
    # ─────────────────────────────────────────────────────────────────
    # Mock Mode Configuration Methods
    # ─────────────────────────────────────────────────────────────────
    
    def save_mock_config(self, config: Dict[str, Any]):
        """
        Save mock mode configuration to session metadata.
        
        This preserves the mock simulation parameters so that when the session
        is loaded later, the same mock configuration can be restored.
        
        Args:
            config: Dictionary of mock configuration parameters from MockConfigurationDialog
        """
        if not self.is_active or self._session_path is None:
            return
        
        # Store in metadata
        self._session_metadata['mock_mode_config'] = config
        self._save_metadata()
        
        print(f"[Session] Saved mock mode configuration with {len(config)} parameters")
    
    def get_mock_config(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve mock mode configuration from session metadata.
        
        Returns:
            Dictionary of mock configuration parameters, or None if not present
        """
        if not self.is_active:
            return None
        
        return self._session_metadata.get('mock_mode_config')
    
    def has_mock_config(self) -> bool:
        """
        Check if the current session has mock mode configuration.
        
        Returns:
            True if session has mock config, False otherwise
        """
        return self.get_mock_config() is not None
    
    # ─────────────────────────────────────────────────────────────────
    # File Operations
    # ─────────────────────────────────────────────────────────────────
    
    def open_session_folder(self):
        """
        Open the session folder in the system file browser.
        """
        if not self.is_active:
            return
        
        path = str(self._session_path)
        
        try:
            if sys.platform == 'darwin':  # macOS
                subprocess.run(['open', path], check=True)
            elif sys.platform == 'win32':  # Windows
                subprocess.run(['explorer', path], check=True)
            else:  # Linux and others
                subprocess.run(['xdg-open', path], check=True)
        except Exception as e:
            print(f"[Session] Error opening folder: {e}")
    
    def load_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load data from a pickle file.

        Through ``store``, so a file that was moved is told where it now is and
        an analysis re-saves it in place rather than beside itself.

        Args:
            file_path: Path to the pickle file to load

        Returns:
            Dictionary of loaded data, or None if load fails
        """
        try:
            return store.load(file_path)
        except Exception as e:
            print(f"[Session] Error loading file {file_path}: {e}")
            return None
    
    def identify_file_type(self, file_path: str) -> Optional[str]:
        """
        Identify the data type of a session file by inspecting its content.
        
        This method loads the pickle file and checks for identifying metadata
        or data structures. This makes it robust against filename changes.
        
        Detection priority:
        1. Check store's file_metadata (for files written through store)
        2. Check '_session_export' metadata (for files exported by session manager)
        
        Args:
            file_path: Path to the pickle file
        
        Returns:
            Data type string (netanal, multisweep, bias, noise, pulse)
            or None if unknown
        """
        # HDF5 files are never pickles.  The pulse capture reader owns
        # the file layout: a file it opens with captured channels, or
        # with a trigger config, is one.
        if str(file_path).lower().endswith(('.h5', '.hdf5')):
            from rfmux.pulse_capture.hdf5 import (
                TRIGGER_CONFIG_ATTR, PulseHDF5Reader)
            try:
                with PulseHDF5Reader(file_path) as reader:
                    return ('pulse' if reader.channels
                            or TRIGGER_CONFIG_ATTR in reader.metadata
                            else None)
            except Exception:
                return None

        # Try to load the file
        data = self.load_file(file_path)
        if data is None or not isinstance(data, dict):
            return None
        
        # 1. A file written through store says what it is, whoever wrote it:
        # a container carries file_metadata in each module's block, and a
        # to_dict payload carries it at the top.
        for block in [data, *data.values()]:
            if isinstance(block, dict) and store.METADATA_KEY in block:
                measurement_type = block[store.METADATA_KEY].get('measurement_type')
                if measurement_type:
                    return measurement_type

        # 2. Files the session manager wrote itself, before store
        if '_session_export' in data:
            metadata = data['_session_export']
            if isinstance(metadata, dict) and 'data_type' in metadata:
                return metadata['data_type']
        
        # Unknown file type
        return None


    def register_external_file(self, filepath: str, data_type: str,
                               identifier: str = ""):
        """
        Register a file written directly into the session folder by a
        panel/task (e.g. a streamed pulse-capture HDF5).

        Records the file in the session metadata 'exports' list and emits
        file_exported / session_updated so the browser refreshes.
        """
        if not self.is_active or self._session_path is None:
            return

        path = Path(filepath)
        self._export_count += 1
        self._session_metadata.setdefault('exports', []).append(
            _session.export_entry(path.name, data_type, identifier))
        self._save_metadata()
        self.file_exported.emit(str(path), data_type)
        self.session_updated.emit()

    def register_screenshot(self, filepath: str):
        """
        Register a screenshot file saved into the session folder.
        """
        if not self.is_active or self._session_path is None:
            return
    
        path = Path(filepath)
        if not path.exists():
            return
    
        self._session_metadata.setdefault('screenshots', []).append({
            'filename': path.name,
            'timestamp': datetime.datetime.now().isoformat(),
        })
    
        self._save_metadata()
        self.session_updated.emit()

    # ─────────────────────────────────────────────────────────────────
    # Private Methods
    # ─────────────────────────────────────────────────────────────────
    
    def _save_metadata(self):
        """Save session metadata to JSON file."""
        if not self.is_active or self._session_path is None:
            return

        try:
            # Keep exports something else listed meanwhile (crs.apply_bias
            # lists its bias_catalog file itself, from a worker thread).
            with _session.METADATA_LOCK:
                on_disk = _session.load_metadata(self._session_path).get('exports', [])
                known = {e.get('filename') for e in
                         self._session_metadata.get('exports', [])}
                for entry in on_disk:
                    if entry.get('filename') not in known:
                        self._session_metadata.setdefault('exports', []).append(entry)
                _session.save_metadata(self._session_path, self._session_metadata)
        except Exception as e:
            print(f"[Session] Warning: Could not save metadata: {e}")
    
