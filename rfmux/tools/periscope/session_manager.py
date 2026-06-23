"""
Session Manager for Periscope Application
==========================================

This module provides centralized session management for automatic data export
in the Periscope application. It handles:
- Creating and managing session folders
- Auto-exporting data files with timestamped naming
- Tracking session state and metadata
- Coordinating with analysis panels via signals

Usage:
    session_mgr = SessionManager(parent=main_window)
    session_mgr.start_session("/path/to/data", "my_session")
    session_mgr.export_data("netanal", data_dict)
"""

from __future__ import annotations

import datetime
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6 import QtCore
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtCore import QUrl


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
        'bias': 'Bias KIDs',
        'noise': 'Noise Spectrum',
        'channel_noise':'Channel Noise',
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
        # Tracks the most-recently-written filename for each (data_type, identifier) pair.
        # Used by handle_data_ready to automatically overwrite the existing file on
        # subsequent emissions (e.g. Find Bias, Run Fit) rather than creating new
        # timestamped duplicates.
        self._last_exported_per_identifier: Dict[tuple, str] = {}
    
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
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"session_{timestamp}"
        
        # Create the session folder
        session_path = base / folder_name
        session_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize session state
        self._session_path = session_path
        
        # Save the full session path so it's pre-selected next time
        from . import settings
        settings.set_last_session_path(str(session_path))
        self._export_count = 0
        self._session_start_time = datetime.datetime.now()
        self._last_exported_per_identifier = {}  # Fresh session — no prior exports
        self._session_metadata = {
            'created': self._session_start_time.isoformat(),
            'folder_name': folder_name,
            'base_path': str(base),
            'exports': [],
            'screenshots':[],
        }
        
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
        self._session_start_time = datetime.datetime.now()
        
        # Try to load existing metadata
        metadata_file = path / 'session_metadata.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self._session_metadata = json.load(f)
                self._export_count = len(self._session_metadata.get('exports', []))
            except Exception as e:
                print(f"[Session] Warning: Could not load metadata: {e}")
                self._session_metadata = {'loaded': datetime.datetime.now().isoformat()}
                self._export_count = len(list(path.glob('*.pkl')))
        else:
            self._session_metadata = {'loaded': datetime.datetime.now().isoformat()}
            self._export_count = len(list(path.glob('*.pkl')))
        
        # Rebuild _last_exported_per_identifier from existing metadata so that
        # subsequent Find Bias / Run Fit calls on a loaded panel update the
        # already-existing session file rather than creating new duplicates.
        self._last_exported_per_identifier = {}
        for exp in self._session_metadata.get('exports', []):
            key = (exp.get('data_type', ''), exp.get('identifier', ''))
            filename = exp.get('filename', '')
            if filename and key[0] and key[1]:
                # Only track it if the file still exists on disk
                if (path / filename).exists():
                    self._last_exported_per_identifier[key] = filename

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
        self._session_path = None
        self._export_count = 0
        self._session_start_time = None
        self._session_metadata = {}
        
        # Emit signal
        self.session_ended.emit()
        
        print(f"[Session] Ended session: {session_name}")
    
    # ─────────────────────────────────────────────────────────────────
    # Data Export Methods
    # ─────────────────────────────────────────────────────────────────
    
    def export_data(self, data_type: str, identifier: str, data: Dict[str, Any], 
                    filename_override: Optional[str] = None) -> Optional[Path]:
        """
        Export data to the session folder with timestamped filename.
        
        This is the main method for exporting analysis data. It generates
        a timestamped filename and saves the data as a pickle file.
        
        If filename_override is provided, that filename is used instead of
        generating a new one. This enables natural file overwriting when
        a panel re-exports updated data (e.g., after Find Resonances).
        
        Args:
            data_type: Type of data being exported (netanal, multisweep, bias, noise)
            identifier: Internal tracking identifier (e.g., 'module1'). Not included
                        in the filename, but used to de-duplicate exports within a
                        session (so Find Bias / Run Fit overwrite the same file).
            data: Dictionary of data to export
            filename_override: If provided, use this filename instead of generating
                              a new timestamped one. Enables natural overwriting.
        
        Returns:
            Path to the exported file, or None if session not active or export disabled
        
        Example:
            >>> session_mgr.export_data('netanal', 'module1', netanal_data)
            Path('/data/session_20251201/netanal_092000.pkl')
            
            # Re-export with same filename to overwrite:
            >>> session_mgr.export_data('netanal', 'module1', updated_data, 
            ...                         filename_override='netanal_092000.pkl')
        """
        if not self.is_active or self._session_path is None:
            print(f"[Session] No active session - skipping export of {data_type}")
            return None
        
        if not self._auto_export_enabled:
            print(f"[Session] Auto-export disabled - skipping export of {data_type}")
            return None
        
        # Use override filename if provided, otherwise generate new one
        if filename_override:
            filename = filename_override
            is_overwrite = True
        else:
            filename = self.generate_filename(data_type, identifier)
            is_overwrite = False
        file_path = self._session_path / filename
        
        # Add export metadata to the data
        export_data = data.copy()
        export_data['_session_export'] = {
            'timestamp': datetime.datetime.now().isoformat(),
            'data_type': data_type,
            'identifier': identifier,
            'session_name': self.session_name,
            'export_number': self._export_count + 1,
        }
        
        try:
            # Write the pickle file
            with open(file_path, 'wb') as f:
                pickle.dump(export_data, f)
            
            # Update session state
            self._export_count += 1
            self._session_metadata.setdefault('exports', []).append({
                'filename': filename,
                'data_type': data_type,
                'identifier': identifier,
                'timestamp': datetime.datetime.now().isoformat(),
            })
            self._save_metadata()

            # Remember this filename so subsequent handle_data_ready calls for
            # the same (data_type, identifier) overwrite it instead of creating
            # a new timestamped duplicate.
            self._last_exported_per_identifier[(data_type, identifier)] = filename

            # Emit signals
            self.file_exported.emit(str(file_path), data_type)
            self.session_updated.emit()
            
            #print(f"[Session] Exported: {filename}")
            return file_path
            
        except Exception as e:
            print(f"[Session] Error exporting data: {e}")
            return None
    
    def generate_filename(self, data_type: str, identifier: str) -> str:
        """
        Generate a timestamped filename for data export.
        
        Uses time-only suffix since files are already in a timestamped session folder.
        The identifier is used only for internal de-duplication tracking and is not
        included in the filename.
        
        Args:
            data_type: Type of data (netanal, multisweep, bias, noise)
            identifier: Internal tracking identifier (not included in filename)
        
        Returns:
            Filename in format: <type>_HHMMSS.pkl
        
        Example:
            >>> session_mgr.generate_filename('netanal', 'module1')
            'netanal_092000.pkl'
        """
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        return f"{data_type}_{timestamp}.pkl"
    
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
            # New format: {data_type}_{timestamp}.pkl  (e.g. multisweep_092000.pkl)
            files = list(self._session_path.glob(f'{data_type}_*.pkl'))
            # Backwards compat: old format had the module identifier embedded,
            # e.g. multisweep_module1_092000.pkl  →  *_{data_type}_*.pkl
            files += list(self._session_path.glob(f'*_{data_type}_*.pkl'))
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
        
        Args:
            file_path: Path to the pickle file to load
        
        Returns:
            Dictionary of loaded data, or None if load fails
        """
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[Session] Error loading file {file_path}: {e}")
            return None
    
    def identify_file_type(self, file_path: str) -> Optional[str]:
        """
        Identify the data type of a session file by inspecting its content.
        
        This method loads the pickle file and checks for identifying metadata
        or data structures. This makes it robust against filename changes.
        
        Detection priority:
        1. Check '_session_export' metadata (for files exported by session manager)
        2. Inspect data structure (for older files or external files)
        
        Args:
            file_path: Path to the pickle file
        
        Returns:
            Data type string (netanal, multisweep, bias, noise) or None if unknown
        """
        # Try to load the file
        data = self.load_file(file_path)
        if data is None or not isinstance(data, dict):
            return None
        
        # 1. Top-level 'measurement_type' key — most reliable, present in all new files
        if 'measurement_type' in data:
            return data['measurement_type']

        # 2. Check for session export metadata (files saved before measurement_type was added)
        if '_session_export' in data:
            metadata = data['_session_export']
            if isinstance(metadata, dict) and 'data_type' in metadata:
                return metadata['data_type']
        
        # 3. Structure-based fallback for older files without either of the above keys.
        #    Priority order matters: bias and noise are subsets of multisweep.
        
        # Check for multisweep data (new 'results' key, old 'results_by_detector', or legacy 'results_by_iteration')
        has_multisweep = (
            'results' in data
            or 'results_by_detector' in data
            or 'results_by_iteration' in data
        )

        # Bias files: have both bias_kids_output AND multisweep data
        if 'bias_kids_output' in data and has_multisweep:
            return 'bias'

        # Noise files: have noise_data AND multisweep data
        if 'noise_data' in data and data['noise_data'] is not None and has_multisweep:
            return 'noise'

        # Multisweep files: have multisweep data (but not bias or noise)
        if has_multisweep:
            return 'multisweep'
        
        # Network analysis files: old schema ('parameters'+'modules') or
        # new harmonised schema ('initial_parameters'+'results'+'target_module')
        if ('parameters' in data and 'modules' in data) or \
           ('initial_parameters' in data and 'results' in data and 'target_module' in data):
            return 'netanal'

        if 'channel_noise_data' in data and data['channel_noise_data'] is not None:
            return 'channel_noise'
        
        # Unknown file type
        return None


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
        
        metadata_file = self._session_path / 'session_metadata.json'
        
        try:
            with open(metadata_file, 'w') as f:
                json.dump(self._session_metadata, f, indent=2)
        except Exception as e:
            print(f"[Session] Warning: Could not save metadata: {e}")
    
    # ─────────────────────────────────────────────────────────────────
    # Slot for Panel Signals
    # ─────────────────────────────────────────────────────────────────
    
    @QtCore.pyqtSlot(str, str, dict)
    def handle_data_ready(self, data_type: str, identifier: str, data: Dict[str, Any]):
        """
        Slot to handle data_ready signals from analysis panels.
        
        This slot should be connected to the data_ready signals emitted
        by NetworkAnalysisPanel, MultisweepPanel, and DetectorDigestPanel.
        
        If the *data* dict contains the special key ``'_filename_override'``,
        that value is extracted and forwarded to :meth:`export_data` as
        *filename_override*, causing the export to overwrite the previously
        created file rather than generating a new timestamped name.  This
        mirrors the pattern used by the network-analysis panel so that
        subsequent updates (Find Bias, Run Fits, …) always update the same
        multisweep pickle rather than littering the session folder with
        duplicate files.
        
        Args:
            data_type: Type of data (netanal, multisweep, bias, noise)
            identifier: Identifier string (e.g., module1)
            data: Dictionary of data to export
        """
        if self.is_active and self._auto_export_enabled:
            # 1. Allow the emitter to request an explicit in-place overwrite.
            filename_override = data.pop('_filename_override', None)

            # 2. If no explicit override was given, check whether we've already
            #    exported a file for this (data_type, identifier) pair during this
            #    session (including ones loaded from a previous session via
            #    load_session).  If so, overwrite that file so that Find Bias and
            #    Run Fit update the existing multisweep pickle rather than creating
            #    new timestamped duplicates.
            if filename_override is None:
                key = (data_type, identifier)
                existing = self._last_exported_per_identifier.get(key)
                if existing and self._session_path and (self._session_path / existing).exists():
                    filename_override = existing

            self.export_data(data_type, identifier, data, filename_override=filename_override)

    def reset_export_tracking(self, data_type: str, identifier: str) -> None:
        """Remove the tracked filename for a given (data_type, identifier) pair.

        Call this before starting a re-run sweep so that the new sweep creates
        a fresh timestamped file rather than overwriting the previous sweep's file.

        Args:
            data_type: Data type key (e.g. ``"multisweep"``).
            identifier: Identifier string (e.g. ``"module1"``).
        """
        self._last_exported_per_identifier.pop((data_type, identifier), None)

    def register_loaded_file(self, data_type: str, identifier: str, file_path: str) -> None:
        """Register a specific on-disk file as the export target for a loaded panel.

        When a user loads an existing session file into a new panel and then runs
        Find Bias or Run Fit, the session manager should overwrite *that* file
        rather than the most-recently-measured file for the same module.

        Call this immediately after creating a panel from loaded data and
        connecting its ``data_ready`` signal, passing the path of the loaded
        ``.pkl`` file.  Subsequent ``handle_data_ready`` calls for this
        ``(data_type, identifier)`` pair will overwrite that file.

        Args:
            data_type: Data type key (e.g. ``"multisweep"``).
            identifier: Identifier string (e.g. ``"module1"``).
            file_path: Full path (or basename) of the loaded file.  Only the
                       basename is stored; the file must reside in the active
                       session folder.
        """
        from pathlib import Path
        filename = Path(file_path).name
        if filename:
            self._last_exported_per_identifier[(data_type, identifier)] = filename
