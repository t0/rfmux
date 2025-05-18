"""Utility functions and helpers for the Periscope viewer."""

import argparse, textwrap
import math
import os
import threading
import queue
import socket
import sys, warnings, ctypes.util, ctypes, platform
import time
import random
import pickle
import traceback
import csv
import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import concurrent.futures

# Create session
import asyncio
from ..core.session import load_session
from ..core.schema import CRS

# ───────────────────────── Global Constants ─────────────────────────
# Display settings
LINE_WIDTH = 1.5
UI_FONT_SIZE = 12
DENSITY_GRID = 512
DENSITY_DOT_SIZE = 1
SMOOTH_SIGMA = 1.3
LOG_COMPRESS = True
SCATTER_POINTS = 1_000
SCATTER_SIZE = 5

# Network analysis defaults
DEFAULT_MIN_FREQ = 100e6  # 100 MHz
DEFAULT_MAX_FREQ = 2450e6  # 2.45 GHz
DEFAULT_CABLE_LENGTH = 10  # meters
DEFAULT_AMPLITUDE = 0.001
DEFAULT_MAX_CHANNELS = 1023
DEFAULT_MAX_SPAN = 500e6  # 500 MHz
DEFAULT_NPOINTS = 5000
DEFAULT_NSAMPLES = 10

# Sampling settings
BASE_SAMPLING = 625e6 / 256.0 / 64.0  # ≈38 147.46 Hz base for dec=0
DEFAULT_BUFFER_SIZE = 5_000
DEFAULT_REFRESH_MS = 33

# GUI update intervals
NETANAL_UPDATE_INTERVAL = 0.1  # seconds

ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icons', 'periscope-icon.svg')

# ───────────────────────── Utility Functions ─────────────────────────

def _check_xcb_cursor_runtime() -> bool:
    """
    Detects whether libxcb-cursor is available on the current system.
    Returns True if it can be opened, False otherwise (after emitting a warning).
    """
    if platform.system() != "Linux":        # macOS/Windows ship Qt plugins that do not need it
        return True

    libname = ctypes.util.find_library("xcb-cursor")   # returns None if not found
    if libname is None:
        warnings.warn(
            "System library 'libxcb-cursor0' is missing.\n"
            "Install it with:  sudo apt-get install libxcb-cursor0  (Debian/Ubuntu)\n"
            "or the equivalent package for your distribution.",
            RuntimeWarning,
        )
        return False
    try:
        ctypes.CDLL(libname)    # will raise OSError on broken symlink
        return True
    except OSError as exc:
        warnings.warn(
            f"libxcb-cursor was found ({libname}) but cannot be loaded: {exc}. Try reinstalling it.",
            RuntimeWarning,
        )
        return False

_check_xcb_cursor_runtime()

# Import PyQt only after checking XCB dependencies
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal, Qt, QRegularExpression
from PyQt6.QtGui import QFont, QIntValidator, QIcon, QDoubleValidator, QRegularExpressionValidator
import pyqtgraph as pg

# Imports for embedded iPython console
try:
    from qtconsole.rich_jupyter_widget import RichJupyterWidget
    from qtconsole.inprocess import QtInProcessKernelManager
    QTCONSOLE_AVAILABLE = True
except ImportError:
    QTCONSOLE_AVAILABLE = False
    warnings.warn(
        "qtconsole or ipykernel not found. Interactive session feature will be disabled.\n"
        "Install them with: pip install qtconsole ipykernel",
        RuntimeWarning,
    )

# Local imports
import rfmux # Ensure rfmux is available for the console
from .. import streamer
from ..awaitless import load_ipython_extension as load_awaitless_extension
from ..core.transferfunctions import (
    spectrum_from_slow_tod,
    convert_roc_to_volts,
    convert_roc_to_dbm,
    fit_cable_delay,
    calculate_new_cable_length,
    recalculate_displayed_phase,
    EFFECTIVE_PROPAGATION_SPEED, # For direct use if needed, though functions encapsulate it
)
from ..algorithms.measurement import fitting # Added for find_resonances
from ..core.hardware_map import macro
from ..core.schema import CRS

# Configure PyQtGraph
pg.setConfigOptions(useOpenGL=False, antialias=False)

# Try to import optional SciPy features
try:
    from scipy.ndimage import gaussian_filter, convolve
except ImportError:  # SciPy not installed – graceful degradation
    gaussian_filter = None
    convolve = None
    SMOOTH_SIGMA = 0.0

def pin_current_thread_to_core():
    """
    Pins the calling thread to a randomly selected CPU core from the set of
    cores this process is allowed to run on (Linux only). On other platforms
    or if unsupported, it emits a warning and does nothing.

    Notes
    -----
    - Uses os.sched_getaffinity(0) to retrieve the process's allowed CPU cores.
    - Picks one randomly (via random.choice).
    - Calls os.sched_setaffinity(tid, {chosen_core}) for the native thread ID.
    - Requires Python 3.8+ for threading.get_native_id().
    - If permission is denied or the environment doesn't support thread affinity,
      it emits a warning but continues.
    """
    if platform.system() != "Linux":
        warnings.warn(
            "Warning: Thread pinning is only supported on Linux. "
            "Performance may suffer\n"
        )
        return
    try:
        # Retrieve the set of CPU cores the current process is allowed to use
        allowed_cores = os.sched_getaffinity(0)
        if not allowed_cores:
            warnings.warn(
                "Warning: No cores found in sched_getaffinity(0). "
                "Not pinning.\n"
            )
            return

        # Select one core randomly
        chosen_core = random.choice(list(allowed_cores))

        # Get the OS thread ID (Python 3.8+)
        tid = threading.get_native_id()

        # Pin this thread to the chosen core
        os.sched_setaffinity(tid, {chosen_core})

    except (AttributeError, NotImplementedError, PermissionError) as e:
        # Possibly older Python or lacking permissions
        warnings.warn(f"Warning: Could not pin thread to a single CPU: {e}\n")

    except Exception as ex:
        warnings.warn(f"Warning: Unexpected error pinning thread to single CPU: {ex}\n")

def get_ipython():
    """
    Attempt to import and return IPython's get_ipython() if available.

    Returns
    -------
    IPython.core.interactiveshell.InteractiveShell or None
        The interactive IPython instance, or None if IPython is not present.
    """
    try:
        from IPython import get_ipython
        return get_ipython()
    except ImportError:
        return None

def is_qt_event_loop_running() -> bool:
    """
    Check if a Qt event loop is active in IPython.

    Returns
    -------
    bool
        True if a Qt event loop is running, False otherwise.
    """
    ip = get_ipython()
    return bool(ip and getattr(ip, "active_eventloop", None) == "qt")

def is_running_inside_ipython() -> bool:
    """
    Check if the current environment is an IPython shell.

    Returns
    -------
    bool
        True if running in IPython, otherwise False.
    """
    return get_ipython() is not None

def infer_dec_stage(fs: float) -> int:
    """
    Estimate an integer decimation stage from a measured sample rate.

    The decimation is derived by taking a log2 ratio of the base sampling
    frequency (~38.147 kHz) to fs, rounded to the nearest integer, clamped
    to [0..15].

    Parameters
    ----------
    fs : float
        Measured sample rate.

    Returns
    -------
    int
        Decimation stage in the range [0..15].
    """
    if fs < 1.0:
        return 0
    ratio = BASE_SAMPLING / fs
    dec_approx = np.log2(ratio) if ratio > 0 else 0
    dec_rounded = int(round(dec_approx))
    return max(0, min(15, dec_rounded))

def parse_channels_multich(txt: str) -> List[List[int]]:
    """
    Parse a comma-separated string of channel specifications, supporting '&'
    to group multiple channels on a single row.

    Examples
    --------
    "3&5" => [[3,5]]
    "3,5" => [[3],[5]]
    "2&4,7&9" => [[2,4],[7,9]]
    "1,2,3&8" => [[1],[2],[3,8]]

    Parameters
    ----------
    txt : str
        A comma-separated list of channels. Ampersand '&' merges them into one row.

    Returns
    -------
    list of list of int
        Each sub-list is a row containing one or more channels. If none found, returns [[1]].
    """
    out = []
    for token in txt.split(","):
        token = token.strip()
        if not token:
            continue
        if "&" in token:
            subs = token.split("&")
            group = []
            for sub in subs:
                sub = sub.strip()
                if sub.isdigit():
                    c = int(sub)
                    if 1 <= c <= streamer.NUM_CHANNELS:
                        group.append(c)
            if group:
                out.append(group)
        else:
            if token.isdigit():
                c = int(token)
                if 1 <= c <= streamer.NUM_CHANNELS:
                    out.append([c])
    if not out:
        return [[1]]
    return out

def mode_title(mode: str) -> str:
    """
    Provide a more user-friendly label for each plot mode.

    Parameters
    ----------
    mode : str
        One of {"T", "IQ", "F", "S", "D", "NA"}.

    Returns
    -------
    str
        A human-readable title segment for that mode.
    """
    mode_titles = {
        "T": "Time",
        "IQ": "IQ",
        "F": "Raw FFT",
        "S": "SSB PSD",
        "D": "DSB PSD",
        "NA": "Network Analysis"
    }
    return mode_titles.get(mode, mode)

# ───────────────────────── Unit Conversion ─────────────────────────
class UnitConverter:
    """
    Utility class for converting between different units.
    """
    
    @staticmethod
    def normalize_to_dbm(normalized_amplitude: float, dac_scale_dbm: float, resistance: float = 50.0) -> float:
        """
        Convert normalized amplitude (0-1) to dBm power.
        
        Parameters
        ----------
        normalized_amplitude : float
            Normalized amplitude value (0-1)
        dac_scale_dbm : float
            DAC scale in dBm
        resistance : float, optional
            Load resistance in ohms, default is 50.0
            
        Returns
        -------
        float
            Power in dBm
        """
        if normalized_amplitude <= 0:
            return float('-inf')  # -infinity dBm for zero amplitude
            
        # Calculate maximum voltage from dac_scale_dbm (peak voltage)
        power_max_mw = 10**(dac_scale_dbm/10)
        power_max_w = power_max_mw / 1000
        v_rms_max = np.sqrt(power_max_w * resistance)
        v_peak_max = v_rms_max * np.sqrt(2.0)
        
        # Calculate peak voltage from normalized amplitude
        v_peak = normalized_amplitude * v_peak_max
        
        # Convert to RMS for power calculation
        v_rms = v_peak / np.sqrt(2.0)
        
        # Calculate power and convert to dBm
        power_w = v_rms**2 / resistance
        power_mw = power_w * 1000
        dbm = 10 * np.log10(power_mw)
        
        return dbm

    @staticmethod
    def dbm_to_normalize(dbm: float, dac_scale_dbm: float, resistance: float = 50.0) -> float:
        """
        Convert dBm power to normalized amplitude (0-1).
        
        Parameters
        ----------
        dbm : float
            Power in dBm
        dac_scale_dbm : float
            DAC scale in dBm
        resistance : float, optional
            Load resistance in ohms, default is 50.0
            
        Returns
        -------
        float
            Normalized amplitude (0-1)
        """
        # Convert dBm to RMS voltage
        power_mw = 10**(dbm/10)
        power_w = power_mw / 1000
        v_rms = np.sqrt(power_w * resistance)
        
        # Convert RMS to peak voltage
        v_peak = v_rms * np.sqrt(2.0)
        
        # Calculate maximum peak voltage from dac_scale_dbm
        power_max_mw = 10**(dac_scale_dbm/10)
        power_max_w = power_max_mw / 1000
        v_rms_max = np.sqrt(power_max_w * resistance)
        v_peak_max = v_rms_max * np.sqrt(2.0)
        
        # Calculate normalized amplitude
        normalized_amplitude = v_peak / v_peak_max
        
        return normalized_amplitude

    @staticmethod
    def convert_amplitude(amps: np.ndarray, iq_data: np.ndarray, unit_mode: str = None, 
                          current_mode: str = "counts", normalize: bool = False) -> np.ndarray:
        """
        Convert amplitude values to the specified unit, with optional normalization.
        
        Parameters
        ----------
        amps : ndarray
            Raw amplitude values (magnitude of complex data)
        iq_data : ndarray
            Complex IQ data for additional conversions
        unit_mode : str, optional
            Unit mode to convert to: "counts", "volts", or "dbm"
            If None, uses current_mode
        current_mode : str, optional
            Current unit mode if no target is specified
        normalize : bool, optional
            Whether to normalize the values (default: False)
                
        Returns
        -------
        ndarray
            Converted amplitude values in the selected unit
        """
        # Use specified unit_mode or fall back to current_mode
        mode = unit_mode if unit_mode is not None else current_mode
        
        # First convert to the right units
        if mode == "counts":
            result = amps.copy()  # Raw counts
        elif mode == "volts":
            result = convert_roc_to_volts(amps)
        elif mode == "dbm":
            # For network analysis, amp values are already magnitudes
            result = convert_roc_to_dbm(amps)
        else:
            result = amps.copy()  # Default fallback
            
        # Then normalize if requested (and if there's data)
        if normalize and len(result) > 0:
            # For dBm, we subtract the first value instead of dividing
            if mode == "dbm":
                # Handle cases where the first value might be invalid (inf or nan)
                ref_val = result[0]
                if np.isfinite(ref_val):
                    result = result - ref_val
            else:
                # For counts or volts, we divide by the first value
                ref_val = result[0]
                if ref_val != 0 and np.isfinite(ref_val):
                    result = result / ref_val
        
        return result

# ───────────────────────── Lock‑Free Ring Buffer ─────────────────────────
class Circular:
    """
    A fixed-size lock-free ring buffer for time-series data.

    Parameters
    ----------
    size : int
        The maximum capacity of the buffer.
    dtype : data-type, optional
        Numpy data type for the underlying array (default float).
    """

    def __init__(self, size: int, dtype=float) -> None:
        self.N = size
        self.buf = np.zeros(size * 2, dtype=dtype)
        self.ptr = 0
        self.count = 0

    def add(self, value):
        """
        Add a single data point to the ring buffer.

        Parameters
        ----------
        value : scalar
            The new data value to store.
        """
        self.buf[self.ptr] = value
        self.buf[self.ptr + self.N] = value
        self.ptr = (self.ptr + 1) % self.N
        self.count = min(self.count + 1, self.N)

    def data(self) -> np.ndarray:
        """
        Retrieve the valid data from the buffer as a contiguous NumPy slice.

        Returns
        -------
        np.ndarray
            The valid contiguous section of the buffer.
        """
        if self.count < self.N:
            return self.buf[: self.count]
        return self.buf[self.ptr : self.ptr + self.N]

# ───────────────────────── Custom Plot Controls ─────────────────────────
class ClickableViewBox(pg.ViewBox):
    """
    A custom ViewBox that opens a coordinate readout dialog when double-clicked
    and supports rectangle selection zooming.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default to RectMode (box zoom)
        self.setMouseMode(pg.ViewBox.RectMode)  

    def enableZoomBoxMode(self, enable=True):
        """
        Enable or disable zoom box mode.
        
        Parameters
        ----------
        enable : bool
            If True, enables rectangle selection zooming. If False, returns to pan mode.
        """
        self.setMouseMode(pg.ViewBox.RectMode if enable else pg.ViewBox.PanMode)

    def mouseDoubleClickEvent(self, ev):
        """
        Handle double-click events to display the point's X/Y coordinates
        in a message box.

        Parameters
        ----------
        ev : QMouseEvent
            The mouse event instance.
        """
        window = getattr(self, 'parent_window', None)
        log_x, log_y = self.state["logMode"]
        pt_view = self.mapSceneToView(ev.scenePos())
        x_val = 10 ** pt_view.x() if log_x else pt_view.x()

        if window and getattr(window, 'add_subtract_mode', False):
            freq = x_val
            if ev.button() == QtCore.Qt.MouseButton.RightButton or ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                window._remove_resonance(self.module_id, freq)
            else:
                window._add_resonance(self.module_id, freq)
            ev.accept()
            return
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            super().mouseDoubleClickEvent(ev)
            return
        plot_item = self.parentItem()
        x_label = y_label = ""
        if isinstance(plot_item, pg.PlotItem):
            x_axis = plot_item.getAxis("bottom")
            y_axis = plot_item.getAxis("left")
            if x_axis and x_axis.label:
                x_label = x_axis.label.toPlainText().strip()
            if y_axis and y_axis.label:
                y_label = y_axis.label.toPlainText().strip()
        x_label = x_label or "X"
        y_label = y_label or "Y"

        y_val = 10 ** pt_view.y() if log_y else pt_view.y()

        parent_widget = self.scene().views()[0].window()
        box = QtWidgets.QMessageBox(parent_widget)
        box.setWindowTitle("Coordinates")
        box.setText(f"{y_label}: {y_val:.6g}\n{x_label}: {x_val:.6g}")
        box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Close)
        box.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        box.show()
        ev.accept()

# ───────────────────────── Network Data Processing ─────────────────────────
