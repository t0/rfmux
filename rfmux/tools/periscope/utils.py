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
# Adjusted imports for new location
from rfmux.core.session import load_session
from rfmux.core.schema import CRS

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
DEFAULT_MAX_CHANNELS = 1024
DEFAULT_MAX_SPAN = 500e6  # 500 MHz
DEFAULT_NPOINTS = 5000
DEFAULT_NSAMPLES = 10
DEFAULT_AMP_START = 0.001  # Default start for amplitude linspace
DEFAULT_AMP_STOP = 0.01     # Default stop for amplitude linspace
DEFAULT_AMP_ITERATIONS = 10  # Default number of iterations for linspace

# Multisweep defaults
MULTISWEEP_DEFAULT_AMPLITUDE = DEFAULT_AMPLITUDE  # Same as network analysis default
MULTISWEEP_DEFAULT_SPAN_HZ = 100000.0  # 100 kHz span per resonance
MULTISWEEP_DEFAULT_NPOINTS = 101  # Points per sweep
MULTISWEEP_DEFAULT_NSAMPLES = DEFAULT_NSAMPLES  # Samples to average (10)

# Sampling settings
BASE_SAMPLING = 625e6 / 256.0 / 64.0  # ≈38 147.46 Hz base for dec=0
DEFAULT_BUFFER_SIZE = 5_000
DEFAULT_REFRESH_MS = 33

# GUI update intervals
NETANAL_UPDATE_INTERVAL = 0.1  # seconds

# ICON_PATH needs to be relative to this file's new location
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
# pyqtSignal is already imported, no need for alias unless it's shadowed, which it isn't here.
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

# Local imports (adjusted for new location)
import rfmux # Ensure rfmux is available for the console
from rfmux import streamer # Adjusted import
from rfmux.awaitless import load_ipython_extension as load_awaitless_extension # Adjusted import
from rfmux.core.transferfunctions import ( # Adjusted import
    spectrum_from_slow_tod,
    convert_roc_to_volts,
    convert_roc_to_dbm,
    fit_cable_delay,
    calculate_new_cable_length,
    recalculate_displayed_phase,
    EFFECTIVE_PROPAGATION_SPEED, 
)
from rfmux.algorithms.measurement import fitting # Adjusted import
from rfmux.core.hardware_map import macro # Adjusted import
# CRS is already imported from rfmux.core.schema

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
    """
    if platform.system() != "Linux":
        warnings.warn(
            "Warning: Thread pinning is only supported on Linux. "
            "Performance may suffer\n"
        )
        return
    try:
        allowed_cores = os.sched_getaffinity(0)
        if not allowed_cores:
            warnings.warn(
                "Warning: No cores found in sched_getaffinity(0). "
                "Not pinning.\n"
            )
            return
        chosen_core = random.choice(list(allowed_cores))
        tid = threading.get_native_id()
        os.sched_setaffinity(tid, {chosen_core})
    except (AttributeError, NotImplementedError, PermissionError) as e:
        warnings.warn(f"Warning: Could not pin thread to a single CPU: {e}\n")
    except Exception as ex:
        warnings.warn(f"Warning: Unexpected error pinning thread to single CPU: {ex}\n")

def get_ipython():
    """
    Attempt to import and return IPython's get_ipython() if available.
    """
    try:
        from IPython import get_ipython
        return get_ipython()
    except ImportError:
        return None

def is_qt_event_loop_running() -> bool:
    """
    Check if a Qt event loop is active in IPython.
    """
    ip = get_ipython()
    return bool(ip and getattr(ip, "active_eventloop", None) == "qt")

def is_running_inside_ipython() -> bool:
    """
    Check if the current environment is an IPython shell.
    """
    return get_ipython() is not None

def infer_dec_stage(fs: float) -> int:
    """
    Estimate an integer decimation stage from a measured sample rate.
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
                    # streamer.NUM_CHANNELS is available from rfmux.streamer
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
    """
    mode_titles = {
        "T": "Time", "IQ": "IQ", "F": "Raw FFT",
        "S": "SSB PSD", "D": "DSB PSD", "NA": "Network Analysis"
    }
    return mode_titles.get(mode, mode)

# ───────────────────────── Unit Conversion ─────────────────────────
class UnitConverter:
    """
    Utility class for converting between different units.
    """
    @staticmethod
    def normalize_to_dbm(normalized_amplitude: float, dac_scale_dbm: float, resistance: float = 50.0) -> float:
        if normalized_amplitude <= 0: return float('-inf')
        power_max_mw = 10**(dac_scale_dbm/10)
        power_max_w = power_max_mw / 1000
        v_rms_max = np.sqrt(power_max_w * resistance)
        v_peak_max = v_rms_max * np.sqrt(2.0)
        v_peak = normalized_amplitude * v_peak_max
        v_rms = v_peak / np.sqrt(2.0)
        power_w = v_rms**2 / resistance
        power_mw = power_w * 1000
        return 10 * np.log10(power_mw)

    @staticmethod
    def dbm_to_normalize(dbm: float, dac_scale_dbm: float, resistance: float = 50.0) -> float:
        power_mw = 10**(dbm/10)
        power_w = power_mw / 1000
        v_rms = np.sqrt(power_w * resistance)
        v_peak = v_rms * np.sqrt(2.0)
        power_max_mw = 10**(dac_scale_dbm/10)
        power_max_w = power_max_mw / 1000
        v_rms_max = np.sqrt(power_max_w * resistance)
        v_peak_max = v_rms_max * np.sqrt(2.0)
        return v_peak / v_peak_max

    @staticmethod
    def convert_amplitude(amps: np.ndarray, iq_data: np.ndarray, unit_mode: str = None, 
                          current_mode: str = "counts", normalize: bool = False) -> np.ndarray:
        mode_to_use = unit_mode if unit_mode is not None else current_mode # Renamed mode
        if mode_to_use == "counts": result = amps.copy()
        elif mode_to_use == "volts": result = convert_roc_to_volts(amps) # from rfmux.core.transferfunctions
        elif mode_to_use == "dbm": result = convert_roc_to_dbm(amps)   # from rfmux.core.transferfunctions
        else: result = amps.copy()
            
        if normalize and len(result) > 0:
            ref_val = result[0]
            if mode_to_use == "dbm":
                if np.isfinite(ref_val): result = result - ref_val
            else:
                if ref_val != 0 and np.isfinite(ref_val): result = result / ref_val
        return result

# ───────────────────────── Lock‑Free Ring Buffer ─────────────────────────
class Circular:
    def __init__(self, size: int, dtype=float) -> None:
        self.N = size; self.buf = np.zeros(size * 2, dtype=dtype)
        self.ptr = 0; self.count = 0
    def add(self, value):
        self.buf[self.ptr] = value; self.buf[self.ptr + self.N] = value
        self.ptr = (self.ptr + 1) % self.N; self.count = min(self.count + 1, self.N)
    def data(self) -> np.ndarray:
        return self.buf[: self.count] if self.count < self.N else self.buf[self.ptr : self.ptr + self.N]

# ───────────────────────── Custom Plot Controls ─────────────────────────
class ClickableViewBox(pg.ViewBox):
    # Signal to emit the mouse event on double click, allowing connected slots to accept it.
    doubleClickedEvent = pyqtSignal(object) 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseMode(pg.ViewBox.RectMode)  

    def enableZoomBoxMode(self, enable=True):
        self.setMouseMode(pg.ViewBox.RectMode if enable else pg.ViewBox.PanMode)

    def mouseDoubleClickEvent(self, ev):
        window = getattr(self, 'parent_window', None)
        log_x, log_y = self.state["logMode"]
        # scenePos() is from the event 'ev', map it to view coordinates
        pt_view = self.mapSceneToView(ev.scenePos()) 
        x_val = 10 ** pt_view.x() if log_x else pt_view.x()
        y_val = 10 ** pt_view.y() if log_y else pt_view.y() # y_val needed for QMessageBox

        # 1. Handle window-specific modes first (e.g., add_subtract_mode for NetworkAnalysisWindow)
        if window and getattr(window, 'add_subtract_mode', False):
            freq = x_val
            # module_id should be set on the ViewBox instance by its parent window if needed
            module_id_to_use = getattr(self, 'module_id', None)
            # Fallback for NetworkAnalysisWindow if module_id isn't directly on ViewBox,
            # but NetworkAnalysisWindow itself has an active_module_for_dac.
            if module_id_to_use is None and hasattr(window, 'active_module_for_dac'):
                 module_id_to_use = window.active_module_for_dac
            
            # Original logic for add/remove: Right-click or Shift+Left-click for remove
            if ev.button() == QtCore.Qt.MouseButton.RightButton or \
               (ev.button() == QtCore.Qt.MouseButton.LeftButton and ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier):
                if hasattr(window, '_remove_resonance'):
                     window._remove_resonance(module_id_to_use, freq)
                ev.accept()
                return
            elif ev.button() == QtCore.Qt.MouseButton.LeftButton: # Normal left click for add
                if hasattr(window, '_add_resonance'):
                    window._add_resonance(module_id_to_use, freq)
                ev.accept()
                return
        
        # 2. Emit the generic doubleClickedEvent signal.
        #    This is intended for features like the Detector Digest in MultisweepWindow.
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.doubleClickedEvent.emit(ev) # Pass the original event object
            if ev.isAccepted():
                # If a slot connected to doubleClickedEvent accepted the event,
                # we assume it's fully handled.
                return

        # 3. Default behavior for left double-click: show coordinates QMessageBox
        #    This executes if not in add_subtract_mode and no slot accepted doubleClickedEvent.
        if ev.button() == QtCore.Qt.MouseButton.LeftButton and not ev.isAccepted():
            plot_item = self.parentItem()
            x_label_text = y_label_text = "" # Renamed to avoid conflict with x_val, y_val
            if isinstance(plot_item, pg.PlotItem):
                x_axis = plot_item.getAxis("bottom"); y_axis = plot_item.getAxis("left")
                if x_axis and x_axis.label: x_label_text = x_axis.label.toPlainText().strip()
                if y_axis and y_axis.label: y_label_text = y_axis.label.toPlainText().strip()
            x_label_text = x_label_text or "X"; y_label_text = y_label_text or "Y"
            
            parent_widget = None
            if self.scene() and self.scene().views(): # Ensure scene and views exist
                parent_widget = self.scene().views()[0].window()

            if parent_widget: # Only show if we have a valid parent widget
                box = QtWidgets.QMessageBox(parent_widget)
                box.setWindowTitle("Coordinates")
                box.setText(f"{y_label_text}: {y_val:.6g}\n{x_label_text}: {x_val:.6g}")
                box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Close)
                box.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
                box.show()
            ev.accept() # Accept the event after showing the message box
            return

        # 4. If not a left button double click and not handled by any of the above,
        #    pass to superclass for any other default ViewBox handling.
        if not ev.isAccepted():
            super().mouseDoubleClickEvent(ev)

# ───────────────────────── Network Data Processing ─────────────────────────
# (No functions here in the original, so nothing to adjust)
