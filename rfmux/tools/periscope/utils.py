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
LINE_WIDTH = 2
UI_FONT_SIZE = 12
DENSITY_GRID = 512
DENSITY_DOT_SIZE = 1
SMOOTH_SIGMA = 1.3
LOG_COMPRESS = True
SCATTER_POINTS = 1_000
SCATTER_SIZE = 5

# ───────────────────────── Plot Color Constants ─────────────────────────
# Core color palettes that work well in both dark and light themes
TABLEAU10_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# Rich color scheme for I/Q plotting - selected for contrast and accessibility
IQ_COLORS = {
    "I": "#3366CC",       # Clear blue
    "Q": "#CC6633",       # Burnt orange
    "MAGNITUDE": None     # None will be replaced with theme-appropriate default
}

# Standard colors for scatter plots
SCATTER_COLORS = {
    "DEFAULT": "#1f77b4",  # Blue
    "HIGHLIGHT": "#d62728", # Red 
    "SELECTION": "#2ca02c", # Green
    "NOISE": None          # None will be replaced with theme-appropriate default
}

# Default colormaps for various plot types
COLORMAP_CHOICES = {
    "SEQUENTIAL": "viridis",    # Good for sequential data
    "AMPLITUDE_SWEEP": "inferno", # Preserving current colormap for amplitude sweeps
    "DIVERGENT": "coolwarm",     # Good for data with positive/negative values
    "CATEGORICAL": "tab10"       # Good for categorical data
}

# Colors for distinct plot series (limited set - when we need distinct colors)
DISTINCT_PLOT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# Utility colors
RESONANCE_LINE_COLOR = "#d62728"  # Red color for resonance lines

# Colors for fitting overlays
FITTING_COLORS = {
    "SKEWED": "#2ca02c",    # Green for skewed fit
    "NONLINEAR": "#9467bd", # Purple for nonlinear fit
    "ERROR_BAND": "#ffbb78" # Light orange for error bands (if used)
}

# Threshold for using distinct colors vs. colormap in amplitude sweeps
AMPLITUDE_COLORMAP_THRESHOLD = 3  # Use distinct colors if num_amps <= this value

# Network analysis defaults
DEFAULT_MIN_FREQ = 1e9  # 1 GHz
DEFAULT_MAX_FREQ = 1.5e9  # 1.5 GHz
DEFAULT_CABLE_LENGTH = 10  # meters
DEFAULT_AMPLITUDE = 0.005
DEFAULT_MAX_CHANNELS = 1024
DEFAULT_MAX_SPAN = 500e6  # 500 MHz
DEFAULT_NPOINTS = 50000
DEFAULT_NSAMPLES = 10

# Default linspace settings for amplitude sweeps
DEFAULT_AMP_START = 0.001  # Default start for amplitude linspace
DEFAULT_AMP_STOP = 0.01     # Default stop for amplitude linspace
DEFAULT_AMP_ITERATIONS = 3  # Default number of iterations for linspace

# Multisweep defaults
MULTISWEEP_DEFAULT_AMPLITUDE = DEFAULT_AMPLITUDE  # Same as network analysis default
MULTISWEEP_DEFAULT_SPAN_HZ = 200000.0  # 200 kHz span per resonance
MULTISWEEP_DEFAULT_NPOINTS = 101  # Points per sweep
MULTISWEEP_DEFAULT_NSAMPLES = DEFAULT_NSAMPLES  # Samples to average (10)

# Find Resonances defaults
DEFAULT_EXPECTED_RESONANCES = None  # Optional
DEFAULT_MIN_DIP_DEPTH_DB = 2.0  # dB
DEFAULT_MIN_Q = 1e4
DEFAULT_MAX_Q = 1e7
DEFAULT_MIN_RESONANCE_SEPARATION_HZ = 1e4  # 10 KHz
DEFAULT_DATA_EXPONENT = 2.0

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

    libname = ctypes.util.find_library("xcb-cursor")
    if libname is None:
        warnings.warn(
            "System library 'libxcb-cursor0' is missing.\n"
            "Install it with:  sudo apt-get install libxcb-cursor0  (Debian/Ubuntu)\n"
            "or the equivalent package for your distribution.",
            RuntimeWarning,
        )
        return False
    try:
        # Ensure libname is not None before passing to CDLL
        if libname:
            ctypes.CDLL(libname)    # will raise OSError on broken symlink
            return True
        else: # Should have been caught by the earlier check, but as a safeguard
            return False 
    except OSError as exc:
        warnings.warn(
            f"libxcb-cursor was found ({libname}) but cannot be loaded: {exc}. Try reinstalling it.",
            RuntimeWarning,
        )
        return False

_check_xcb_cursor_runtime()

# Import PyQt only after checking XCB dependencies
from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal, Qt, QRegularExpression 
# pyqtSignal is already imported, no need for alias unless it's shadowed, which it isn't here.
from PyQt6.QtGui import QFont, QIntValidator, QIcon, QDoubleValidator, QRegularExpressionValidator
import pyqtgraph as pg

# Multisweep line styles (for differentiating sweep directions)
UPWARD_SWEEP_STYLE = Qt.PenStyle.SolidLine
DOWNWARD_SWEEP_STYLE = Qt.PenStyle.DotLine  # Dotted line for downward sweeps

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
        from IPython.core.getipython import get_ipython # Corrected import path
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
                    # streamer.LONG_PACKET_CHANNELS is available from rfmux.streamer
                    if 1 <= c <= streamer.LONG_PACKET_CHANNELS:
                        group.append(c)
            if group:
                out.append(group)
        else:
            if token.isdigit():
                c = int(token)
                if 1 <= c <= streamer.LONG_PACKET_CHANNELS:
                    out.append([c])
    if not out:
        return [[1]]
    return out

def find_parent_with_attr(widget: QtWidgets.QWidget, attr_name: str) -> Optional[QtWidgets.QWidget]:
    """
    Walk up the parent hierarchy to find a widget with the specified attribute.
    
    This is a common pattern for panels to find the Periscope main window instance,
    which can be identified by various attributes like 'crs', 'dac_scales', 'netanal_windows', etc.
    
    Args:
        widget: Starting widget (typically self)
        attr_name: Name of the attribute to search for
        
    Returns:
        The first parent widget that has the specified attribute, or None if not found
        
    Example:
        periscope = find_parent_with_attr(self, 'crs')
        periscope = find_parent_with_attr(self, 'dac_scales')
    """
    parent = widget.parent() if hasattr(widget, 'parent') else None
    while parent and not hasattr(parent, attr_name):
        parent = parent.parent()
    return parent

def mode_title(mode: str) -> str:
    """
    Provide a more user-friendly label for each plot mode.
    """
    mode_titles = {
        "T": "Timestream", "IQ": "IQ", "F": "Raw FFT",
        "S": "Single Sideband PSD", "D": "Dual Sideband PSD", "NA": "Network Analysis"
    }
    return mode_titles.get(mode, mode)

# ───────────────────────── Unit Conversion ─────────────────────────
class UnitConverter:
    """
    Utility class for converting between different units.
    """
    @staticmethod
    def normalize_to_dbm(normalized_amplitude: float, dac_scale_dbm: float, resistance: float = 50.0) -> float:
        if normalized_amplitude <= 0: return -np.inf
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
    # Declare attributes that are dynamically assigned elsewhere to satisfy Pylance
    parent_window: Optional[QtWidgets.QWidget] = None
    module_id: Optional[int] = None
    plot_role: Optional[str] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseMode(pg.ViewBox.RectMode)

    def enableZoomBoxMode(self, enable=True):
        self.setMouseMode(pg.ViewBox.RectMode if enable else pg.ViewBox.PanMode)

    def mouseDoubleClickEvent(self, event: Optional[QtWidgets.QGraphicsSceneMouseEvent]): # Matched base class type hint
        if event is None: # Handle None case if event can be None
            super().mouseDoubleClickEvent(event)
            return
            
        window = getattr(self, 'parent_window', None)
        log_x, log_y = self.state["logMode"]
        # scenePos() is from the event 'event', map it to view coordinates
        pt_view = self.mapSceneToView(event.scenePos()) 
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
            if event.button() == QtCore.Qt.MouseButton.RightButton or \
               (event.button() == QtCore.Qt.MouseButton.LeftButton and event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier):
                if hasattr(window, '_remove_resonance'):
                     window._remove_resonance(module_id_to_use, freq)
                event.accept()
                return
            elif event.button() == QtCore.Qt.MouseButton.LeftButton: # Normal left click for add
                if hasattr(window, '_add_resonance'):
                    window._add_resonance(module_id_to_use, freq)
                event.accept()
                return
        
        # 2. Emit the generic doubleClickedEvent signal.
        #    This is intended for features like the Detector Digest in MultisweepWindow.
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.doubleClickedEvent.emit(event) # Pass the original event object
            if event.isAccepted():
                # If a slot connected to doubleClickedEvent accepted the event,
                # we assume it's fully handled.
                return

        # 3. Default behavior for left double-click: show coordinates QMessageBox
        #    This executes if not in add_subtract_mode and no slot accepted doubleClickedEvent.
        if event.button() == QtCore.Qt.MouseButton.LeftButton and not event.isAccepted():
            plot_item = self.parentItem()
            x_label_text = y_label_text = "" # Renamed to avoid conflict with x_val, y_val
            if isinstance(plot_item, pg.PlotItem):
                x_axis = plot_item.getAxis("bottom"); y_axis = plot_item.getAxis("left")
                if x_axis and x_axis.label: x_label_text = x_axis.label.toPlainText().strip()
                if y_axis and y_axis.label: y_label_text = y_axis.label.toPlainText().strip()
            x_label_text = x_label_text or "X"; y_label_text = y_label_text or "Y"
            
            parent_widget = None
            current_scene = self.scene() # Store scene in a variable
            if current_scene and current_scene.views(): # Ensure scene and views exist
                parent_widget = current_scene.views()[0].window()

            if parent_widget: # Only show if we have a valid parent widget
                box = QtWidgets.QMessageBox(parent_widget)
                box.setWindowTitle("Coordinates")
                box.setText(f"{y_label_text}: {y_val:.6g}\n{x_label_text}: {x_val:.6g}")
                box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Close)
                box.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
                box.show()
            event.accept() # Accept the event after showing the message box
            return

        # 4. If not a left button double click and not handled by any of the above,
        #    pass to superclass for any other default ViewBox handling.
        if not event.isAccepted():
            super().mouseDoubleClickEvent(event)


# ───────────────────────── Screenshot Mixin ─────────────────────────
class ScreenshotMixin:
    """
    Mixin class that provides screenshot functionality for panel widgets.
    
    Any panel that inherits from this mixin will have access to screenshot
    export methods that integrate with the Periscope dock manager and session manager.
    
    Usage:
        class MyPanel(QtWidgets.QWidget, ScreenshotMixin):
            def __init__(self):
                ...
                # Create screenshot button
                self.screenshot_btn = QtWidgets.QPushButton("📷")
                self.screenshot_btn.setToolTip("Export a screenshot of this panel")
                self.screenshot_btn.clicked.connect(self._export_screenshot)
    """
    
    def _get_periscope_parent(self) -> Optional[QtWidgets.QWidget]:
        """
        Get the Periscope parent instance by walking up the parent hierarchy.
        
        Returns:
            The Periscope instance or None if not found
        """
        return find_parent_with_attr(self, 'crs')
    
    def _export_screenshot(self):
        """Export a screenshot of this panel to the session folder or user-chosen location."""
        periscope = self._get_periscope_parent()
        if not periscope:
            # No parent - just save with dialog
            self._export_screenshot_with_dialog()
            return
        
        # Find this panel's dock
        dock = periscope.dock_manager.find_dock_for_widget(self)
        if not dock:
            # No dock found - just save with dialog
            self._export_screenshot_with_dialog()
            return
        
        # Get dock ID
        dock_id = dock.objectName()
        
        # Get session manager
        session_manager = getattr(periscope, 'session_manager', None)
        
        # Use dock manager's export_screenshot method
        filepath = periscope.dock_manager.export_screenshot(
            dock_id,
            session_manager=session_manager
        )
        
        if filepath:
            # Refresh session browser if available
            if hasattr(periscope, 'session_browser') and periscope.session_browser:
                periscope.session_browser.refresh()
    
    def _export_screenshot_with_dialog(self):
        """Export screenshot with a non-blocking file dialog (fallback when no parent/dock).
        
        Uses DontUseNativeDialog option to prevent hanging on some Linux systems.
        """
        # Generate default filename
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        panel_name = getattr(self, 'objectName', lambda: 'panel')()
        if not panel_name:
            panel_name = self.__class__.__name__
        filename = f"screenshot_{panel_name}_{timestamp}.png"
        
        # Create a non-blocking file dialog (prevents hanging on Linux)
        dlg = QtWidgets.QFileDialog(self, "Save Screenshot")
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setNameFilters(["PNG Images (*.png)", "All Files (*)"])
        dlg.setDefaultSuffix("png")
        dlg.selectFile(filename)
        
        # Connect signal for handling file selection
        dlg.fileSelected.connect(self._handle_screenshot_file_selected)
        
        # Show the dialog non-modally (returns immediately, doesn't block)
        dlg.open()
    
    def _handle_screenshot_file_selected(self, filepath: str):
        """Handle the file selection from the screenshot dialog.
        
        Args:
            filepath: The path to the file selected by the user
        """
        if not filepath:
            return
        
        scale = 1  # 1x resolution
        size = self.size()
        pixmap = QtGui.QPixmap(size.width() * scale, size.height() * scale)
        pixmap.setDevicePixelRatio(scale)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        
        painter = QtGui.QPainter(pixmap)
        self.render(painter)
        painter.end()
        
        pixmap.save(filepath, "PNG")
        print(f"[Screenshot] Saved: {filepath}")
