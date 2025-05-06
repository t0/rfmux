#!/usr/bin/env -S uv run
"""
Periscope – Real‑Time Multi‑Pane Viewer
=======================================

Features:
- Time‑domain (TOD)
- IQ (density or scatter)
- FFT (pyqtgraph's fftMode=True)
- Single‑Sideband PSD (SSB)
- Dual‑Sideband PSD (DSB)
- Network Analysis (amplitude and phase vs frequency)

USAGE
-----
Command-Line Examples:
- Requires rfmux to be installed as a python package
    - `$ cd rfmux ; pip install -e .`
- `$ periscope rfmux0022.local --module 2 --channels "3&5,7"`
IPython / Jupyter: invoke directly from CRS object
- `>>> crs.raise_periscope(module=2, channels="3&5")`
- If in a non-blocking mode, you can still interact with your session concurrently.

ARCHITECTURAL SUMMARY
---------------------
This module implements the Periscope real-time multi-pane viewer using PyQt6. It
visualizes data from a CRS data streamer in multiple ways:
    - Time-domain (TOD)
    - IQ (density or scatter)
    - FFT (using pyqtgraph's fftMode=True)
    - Single-Sideband PSD (SSB)
    - Dual-Sideband PSD (DSB)
    - Network Analysis (amplitude and phase vs frequency)

Key Components & Concurrency:
    - A ring buffer (Circular) for each channel stores incoming data.
    - A UDPReceiver runs in its own QThread to receive streaming data asynchronously.
    - Separate IQTask and PSDTask workers offload expensive computations to a QThreadPool.
    - The main Periscope class orchestrates the UI, buffer management, and dispatch of tasks.
    - NetworkAnalysisTask runs network analyses in the background and sends updates to the GUI.

Performance Notes:
    - GPUs are hard, and Python doesn't love concurrency, so performance is dictated
      by the single-threaded nature of PyQtGraph's rendering engine. So any time you have
      very long buffers being plotted it will be pokey. If you want buttery smooth performance
      for many different channels at once, the best way to get this is to launch multiple
      instances.

--------------------------------------------------------------------------------
"""

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
from PyQt6.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QIntValidator
import pyqtgraph as pg

# Local imports
from .. import streamer
from ..core.transferfunctions import (
    spectrum_from_slow_tod,
    convert_roc_to_volts,
    convert_roc_to_dbm,
)
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

def _pin_current_thread_to_core():
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

def _get_ipython():
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

def _is_qt_event_loop_running() -> bool:
    """
    Check if a Qt event loop is active in IPython.

    Returns
    -------
    bool
        True if a Qt event loop is running, False otherwise.
    """
    ip = _get_ipython()
    return bool(ip and getattr(ip, "active_eventloop", None) == "qt")

def _is_running_inside_ipython() -> bool:
    """
    Check if the current environment is an IPython shell.

    Returns
    -------
    bool
        True if running in IPython, otherwise False.
    """
    return _get_ipython() is not None

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

def _parse_channels_multich(txt: str) -> List[List[int]]:
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

def _mode_title(mode: str) -> str:
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
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            super().mouseDoubleClickEvent(ev)
            return
        pt_view = self.mapSceneToView(ev.scenePos())
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

        log_x, log_y = self.state["logMode"]
        x_val = 10 ** pt_view.x() if log_x else pt_view.x()
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
class UDPReceiver(QtCore.QThread):
    """
    Receives multicast packets in a dedicated QThread and pushes them
    into a thread-safe queue.

    Parameters
    ----------
    host : str
        The multicast or UDP host address.
    module : int
        The module number to filter on.
    """

    def __init__(self, host: str, module: int) -> None:
        super().__init__()
        self.module_id = module
        self.queue = queue.Queue()
        self.sock = streamer.get_multicast_socket(host)
        self.sock.settimeout(0.2)

    def run(self):
        """
        Main loop that receives packets from the socket, parses them, and
        adds them to the internal queue. Exits upon thread interruption
        or socket error.
        """
        while not self.isInterruptionRequested():
            try:
                data = self.sock.recv(streamer.STREAMER_LEN)
                pkt = streamer.DfmuxPacket.from_bytes(data)
            except socket.timeout:
                continue
            except OSError:
                break
            if pkt.module == self.module_id - 1:
                self.queue.put(pkt)

    def stop(self):
        """
        Signal the thread to stop, and close the socket.
        """
        self.requestInterruption()
        try:
            self.sock.close()
        except OSError:
            pass

# ───────────────────────── IQ Task & Signals ─────────────────────────
class IQSignals(QObject):
    """
    Holds custom signals emitted by IQ tasks.
    """
    done = pyqtSignal(int, str, object)
    # Emitted with arguments: (row, mode_string, payload)
    #   row: row index
    #   mode_string: "density" or "scatter"
    #   payload: data from the computation

class IQTask(QRunnable):
    """
    Off-thread worker for computing IQ scatter or density histograms.

    Parameters
    ----------
    row : int
        Row index in the channel list (maps results back to the correct row).
    ch : int
        Actual channel number from which the data originates.
    I : np.ndarray
        Array of I samples.
    Q : np.ndarray
        Array of Q samples.
    dot_px : int
        Dot diameter in pixels for point dilation in density mode.
    mode : {"density", "scatter"}
        Determine whether to compute a 2D histogram or scatter points.
    signals : IQSignals
        An IQSignals instance for communicating results back to the GUI thread.
    """

    def __init__(self, row, ch, I, Q, dot_px, mode, signals: IQSignals):
        super().__init__()
        self.row = row
        self.ch = ch
        self.I = I.copy()
        self.Q = Q.copy()
        self.dot_px = dot_px
        self.mode = mode
        self.signals = signals

    def run(self):
        """
        Perform the required IQ computation off the main thread. Emit results
        via self.signals.done.
        """
        if len(self.I) < 2:
            self._handle_insufficient_data()
            return

        if self.mode == "density":
            payload = self._compute_density()
        else:  # scatter mode
            payload = self._compute_scatter()

        self.signals.done.emit(self.row, self.mode, payload)
        
    def _handle_insufficient_data(self):
        """Handle the edge case of insufficient data."""
        if self.mode == "density":
            empty = np.zeros((DENSITY_GRID, DENSITY_GRID), np.uint8)
            payload = (empty, (0, 1, 0, 1))
        else:
            payload = ([], [], [])
        self.signals.done.emit(self.row, self.mode, payload)
        
    def _compute_density(self):
        """Compute IQ density histogram."""
        g = DENSITY_GRID
        hist = np.zeros((g, g), np.uint32)

        Imin, Imax = self.I.min(), self.I.max()
        Qmin, Qmax = self.Q.min(), self.Q.max()
        if Imin == Imax or Qmin == Qmax:
            return (hist.astype(np.uint8), (Imin, Imax, Qmin, Qmax))

        # Map I/Q data to pixel indices
        ix = ((self.I - Imin) * (g - 1) / (Imax - Imin)).astype(np.intp)
        qy = ((self.Q - Qmin) * (g - 1) / (Qmax - Qmin)).astype(np.intp)

        # Base histogram
        np.add.at(hist, (qy, ix), 1)

        # Optional dot dilation
        if self.dot_px > 1:
            self._apply_dot_dilation(hist, ix, qy, g)

        # Optional smoothing & log-compression
        if gaussian_filter is not None and SMOOTH_SIGMA > 0:
            hist = gaussian_filter(hist.astype(np.float32), SMOOTH_SIGMA, mode="nearest")
        if LOG_COMPRESS:
            hist = np.log1p(hist, out=hist.astype(np.float32))

        # 8-bit normalization
        if hist.max() > 0:
            hist = (hist * (255.0 / hist.max())).astype(np.uint8)

        return (hist, (Imin, Imax, Qmin, Qmax))
        
    def _apply_dot_dilation(self, hist, ix, qy, g):
        """Apply dot dilation to the histogram."""
        r = self.dot_px // 2
        if convolve is not None:
            # Faster path if SciPy is available
            k = 2 * r + 1
            kernel = np.ones((k, k), dtype=np.uint8)
            hist = convolve(hist, kernel, mode="constant", cval=0)
        else:
            # Fallback
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    ys, xs = qy + dy, ix + dx
                    mask = ((0 <= ys) & (ys < g) &
                            (0 <= xs) & (xs < g))
                    np.add.at(hist, (ys[mask], xs[mask]), 1)
                    
    def _compute_scatter(self):
        """Compute IQ scatter points."""
        N = len(self.I)
        if N > SCATTER_POINTS:
            idx = np.linspace(0, N - 1, SCATTER_POINTS, dtype=np.intp)
        else:
            idx = np.arange(N, dtype=np.intp)
        xs, ys = self.I[idx], self.Q[idx]
        rel = idx / (idx.max() if idx.size else 1)
        colors = pg.colormap.get("turbo").map(
            rel.astype(np.float32), mode="byte"
        )
        return (xs, ys, colors)

# ───────────────────────── PSD Task & Signals ─────────────────────────
class PSDSignals(QObject):
    """
    Holds custom signals emitted by PSD tasks.
    """
    done = pyqtSignal(int, str, int, object)
    # Emitted with arguments: (row, mode_string, channel, payload)

class PSDTask(QRunnable):
    """
    Off‑thread worker for single or dual sideband PSD computation.

    Parameters
    ----------
    row : int
        Row index in the channel list (used to map results back to the correct UI row).
    ch : int
        Actual channel number (data source).
    I : np.ndarray
        Array of I samples.
    Q : np.ndarray
        Array of Q samples.
    mode : {"SSB", "DSB"}
        Determines the type of PSD computation.
    dec_stage : int
        Decimation stage for the spectrum_from_slow_tod() call.
    real_units : bool
        If True, convert PSD to dBm/dBc. Otherwise, keep as raw counts²/Hz.
    psd_absolute : bool
        If True and real_units is True, uses absolute (dBm) reference. Otherwise relative (dBc).
    segments : int
        Number of segments for Welch segmentation. Data is split by nperseg = data_len // segments.
    signals : PSDSignals
        A PSDSignals instance for communicating results back to the GUI thread.
    """

    def __init__(
        self,
        row: int,
        ch: int,
        I: np.ndarray,
        Q: np.ndarray,
        mode: str,
        dec_stage: int,
        real_units: bool,
        psd_absolute: bool,
        segments: int,
        signals: PSDSignals,
    ):
        super().__init__()
        self.row = row
        self.ch = ch
        self.I = I.copy()
        self.Q = Q.copy()
        self.mode = mode
        self.dec_stage = dec_stage
        self.real_units = real_units
        self.psd_absolute = psd_absolute
        self.segments = segments
        self.signals = signals

    def run(self):
        """
        Perform PSD computations off the main thread. Emit the resulting
        frequency array(s) and PSD(s) via signals.done.
        """
        data_len = len(self.I)
        if data_len < 2:
            self._handle_insufficient_data()
            return

        # Determine reference type
        ref = self._get_reference_type()
        
        # Calculate segment size
        nper = max(1, data_len // max(1, self.segments))

        # Compute spectrum based on mode
        if self.mode == "SSB":
            payload = self._compute_ssb_psd(ref, nper)
        else:  # "DSB"
            payload = self._compute_dsb_psd(ref, nper)

        self.signals.done.emit(self.row, self.mode, self.ch, payload)

    def _handle_insufficient_data(self):
        """Handle the edge case of insufficient data."""
        if self.mode == "SSB":
            payload = ([], [], [], [], [], [], 0.0)
        else:
            payload = ([], [])
        self.signals.done.emit(self.row, self.mode, self.ch, payload)
        
    def _get_reference_type(self) -> str:
        """Determine the reference type for the spectrum computation."""
        if not self.real_units:
            return "counts"
        return "absolute" if self.psd_absolute else "relative"
        
    def _compute_ssb_psd(self, ref, nper):
        """Compute single-sideband PSD."""
        # First compute IQ spectrum
        spec_iq = spectrum_from_slow_tod(
            i_data=self.I,
            q_data=self.Q,
            dec_stage=self.dec_stage,
            scaling="psd",
            reference=ref,
            nperseg=nper,
            spectrum_cutoff=0.9,
        )
        
        # Then compute magnitude spectrum
        M_data = np.sqrt(self.I**2 + self.Q**2)
        spec_m = spectrum_from_slow_tod(
            i_data=M_data,
            q_data=np.zeros_like(M_data),
            dec_stage=self.dec_stage,
            scaling="psd",
            reference=ref,
            nperseg=nper,
            spectrum_cutoff=0.9,
        )
        
        # Return combined results
        return (
            spec_iq["freq_iq"],
            spec_iq["psd_i"],
            spec_iq["psd_q"],
            spec_m["psd_i"],
            spec_m["freq_iq"],
            spec_m["psd_i"],
            float(self.dec_stage),
        )
        
    def _compute_dsb_psd(self, ref, nper):
        """Compute dual-sideband PSD."""
        # Compute IQ spectrum
        spec_iq = spectrum_from_slow_tod(
            i_data=self.I,
            q_data=self.Q,
            dec_stage=self.dec_stage,
            scaling="psd",
            reference=ref,
            nperseg=nper,
            spectrum_cutoff=0.9,
        )
        
        # Extract dual-sideband spectrum and sort by frequency
        freq_dsb = spec_iq["freq_dsb"]
        psd_dsb = spec_iq["psd_dual_sideband"]
        order = np.argsort(freq_dsb)
        
        return (freq_dsb[order], psd_dsb[order])

# ───────────────────────── Network Analysis ─────────────────────────
class NetworkAnalysisSignals(QObject):
    """
    Holds custom signals emitted by network analysis tasks.
    """
    progress = pyqtSignal(int, float)  # module, progress percentage
    data_update = pyqtSignal(int, np.ndarray, np.ndarray, np.ndarray)  # module, freqs, amps, phases
    data_update_with_amp = pyqtSignal(int, np.ndarray, np.ndarray, np.ndarray, float)  # module, freqs, amps, phases, amplitude
    completed = pyqtSignal(int)  # module
    error = pyqtSignal(str)  # error message

class DACScaleFetcher(QtCore.QThread):
    """Asynchronously fetch DAC scales for all modules."""
    dac_scales_ready = QtCore.pyqtSignal(dict)
    
    def __init__(self, crs):
        super().__init__()
        self.crs = crs
        
    def run(self):
        """Fetch DAC scales for all modules."""
        dac_scales = {}
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            self._fetch_all_dac_scales(loop, dac_scales)
        finally:
            loop.close()
            
        self.dac_scales_ready.emit(dac_scales)
        
    def _fetch_all_dac_scales(self, loop, dac_scales):
        """Fetch DAC scales for all modules using the provided event loop."""
        for module in range(1, 9):
            try:
                dac_scale = loop.run_until_complete(
                    self.crs.get_dac_scale('DBM', module=module)
                )
                dac_scales[module] = dac_scale - 1.5  # Compensation for the balun
            except Exception as e:
                # Silently handle the expected "Can't access module X with low analog banking" error
                if "Can't access module" in str(e) and "analog banking" in str(e):
                    # Just set to None without printing error
                    dac_scales[module] = None
                else:
                    # Still print unexpected ValueErrors
                    print(f"Error fetching DAC scale for module {module}: {e}")
                    dac_scales[module] = None

class NetworkAnalysisTask(QRunnable):
    """
    Off-thread worker for running network analysis.
    
    Parameters
    ----------
    crs : CRS
        CRS object from HardwareMap
    module : int
        Module number to run analysis on
    params : dict
        Network analysis parameters
    signals : NetworkAnalysisSignals
        Signals for communication with GUI thread
    amplitude : float, optional
        Specific amplitude value to use for this task
    """
    
    def __init__(self, crs: "CRS", module: int, params: dict, signals: NetworkAnalysisSignals, amplitude=None):
        super().__init__()
        self.crs = crs
        self.module = module
        self.params = params
        self.signals = signals
        self.amplitude = amplitude if amplitude is not None else params.get('amp', DEFAULT_AMPLITUDE)
        self._running = True
        self._last_update_time = 0
        self._update_interval = NETANAL_UPDATE_INTERVAL  # seconds
        self._task = None
        self._loop = None
        
    def stop(self):
        """
        Signal the task to stop and cancel any running asyncio tasks.
        """
        self._running = False
        
        # Cancel the async task if it exists
        if self._task and not self._task.done() and self._loop:
            # Schedule task cancellation in the event loop
            self._loop.call_soon_threadsafe(self._task.cancel)
            
            # Clean up channels
            if self._loop.is_running():
                try:
                    # Try to schedule cleanup in the event loop
                    cleanup_future = asyncio.run_coroutine_threadsafe(
                        self._cleanup_channels(), self._loop
                    )
                    # Wait for cleanup with a timeout
                    cleanup_future.result(timeout=2.0)
                except (asyncio.CancelledError, concurrent.futures.TimeoutError, Exception):
                    pass
    
    async def _cleanup_channels(self):
        """Clean up channels on the module being analyzed."""
        try:
            async with self.crs.tuber_context() as ctx:
                # Zero out amplitudes for all channels on this module
                for j in range(1, 1024):  # Assuming max channels is 1023
                    ctx.set_amplitude(0, channel=j, module=self.module)
                await ctx()
        except Exception:
            # Ignore errors during cleanup
            pass
    
    def run(self):
        """Execute the network analysis using the take_netanal macro."""
        # Create event loop for async operations
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._execute_network_analysis()
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            # Clean up resources
            self._cleanup_resources()
            
    def _execute_network_analysis(self):
        """Run the network analysis operation."""
        # Create callbacks that emit signals
        progress_cb = self._create_progress_callback()
        data_cb = self._create_data_callback()
        
        # Extract parameters
        params = self._extract_parameters()

        # Clear channels if requested
        if params['clear_channels'] and self._running:
            self._loop.run_until_complete(self.crs.clear_channels(module=self.module))

        # Set cable length before running the analysis
        if self._running:
            self._loop.run_until_complete(
                self.crs.set_cable_length(length=params['cable_length'], module=self.module)
            )

        # Create a task for the netanal operation
        if self._running:
            self._execute_netanal_task(params, progress_cb, data_cb)
            
    def _create_progress_callback(self):
        """Create a callback for progress updates."""
        def progress_cb(module, progress):
            if self._running:
                self.signals.progress.emit(module, progress)
        return progress_cb
        
    def _create_data_callback(self):
        """Create a callback for data updates."""
        def data_cb(module, freqs_raw, amps_raw, phases_raw):
            if self._running:
                # Sort data by frequency before displaying
                sort_idx = np.argsort(freqs_raw)
                freqs = freqs_raw[sort_idx]
                amps = amps_raw[sort_idx]
                phases = phases_raw[sort_idx]
                
                # Throttle updates to prevent GUI lag
                current_time = time.time()
                if current_time - self._last_update_time >= self._update_interval:
                    self._last_update_time = current_time                    
                
                # Emit standard update and amplitude-specific update
                self.signals.data_update.emit(module, freqs, amps, phases)
                self.signals.data_update_with_amp.emit(module, freqs, amps, phases, self.amplitude)
        return data_cb
    
    def _extract_parameters(self):
        """Extract parameters from the params dictionary with defaults."""
        return {
            'fmin': self.params.get('fmin', DEFAULT_MIN_FREQ),
            'fmax': self.params.get('fmax', DEFAULT_MAX_FREQ),
            'nsamps': self.params.get('nsamps', DEFAULT_NSAMPLES),
            'npoints': self.params.get('npoints', DEFAULT_NPOINTS),
            'max_chans': self.params.get('max_chans', DEFAULT_MAX_CHANNELS),
            'max_span': self.params.get('max_span', DEFAULT_MAX_SPAN),
            'cable_length': self.params.get('cable_length', DEFAULT_CABLE_LENGTH),
            'clear_channels': self.params.get('clear_channels', True)
        }
        
    def _execute_netanal_task(self, params, progress_cb, data_cb):
        """Execute the network analysis task and handle results."""
        netanal_coro = self.crs.take_netanal(
            amp=self.amplitude,  # Use the specific amplitude for this task
            fmin=params['fmin'],
            fmax=params['fmax'],
            nsamps=params['nsamps'],
            npoints=params['npoints'],
            max_chans=params['max_chans'],
            max_span=params['max_span'],
            module=self.module,
            progress_callback=progress_cb,
            data_callback=data_cb
        )
        
        # Create and store the task so it can be canceled
        self._task = self._loop.create_task(netanal_coro)
        
        try:
            # Run the task and get the result
            result = self._loop.run_until_complete(self._task)
            
            # Extract and emit final data if task completed successfully
            if self._running and result:
                fs_sorted, iq_sorted, phase_sorted = result
                amp_sorted = np.abs(iq_sorted)
                
                # Emit final data update
                self.signals.data_update.emit(self.module, fs_sorted, amp_sorted, phase_sorted)
                self.signals.data_update_with_amp.emit(
                    self.module, fs_sorted, amp_sorted, phase_sorted, self.amplitude
                )
                self.signals.completed.emit(self.module)
        except asyncio.CancelledError:
            # Task was canceled, emit error signal
            self.signals.error.emit(f"Analysis canceled for module {self.module}")
            
            # Make sure to clean up channels
            self._loop.run_until_complete(self._cleanup_channels())
            
    def _cleanup_resources(self):
        """Clean up task and event loop resources."""
        if self._task and not self._task.done():
            self._task.cancel()
            
        # Clean up the event loop
        if self._loop and self._loop.is_running():
            self._loop.stop()
        if self._loop:
            self._loop.close()
            self._loop = None

# ───────────────────────── Network Analysis UI ─────────────────────────
class NetworkAnalysisDialogBase(QtWidgets.QDialog):
    """Base class for network analysis dialogs with shared functionality."""
    def __init__(self, parent=None, params=None, modules=None, dac_scales=None):
        super().__init__(parent)
        self.params = params or {}
        self.modules = modules or [1, 2, 3, 4]
        self.dac_scales = dac_scales or {m: None for m in self.modules}  # Default to None (unknown)
        self.currently_updating = False  # Flag to prevent circular updates
        
    def setup_amplitude_group(self, layout):
        """Setup the amplitude settings group with normalized and dBm inputs."""
        # Create the group box with an empty title (we'll add a title in the form layout)
        amp_group = QtWidgets.QGroupBox()
        amp_layout = QtWidgets.QFormLayout(amp_group)
        
        # Get amplitude values
        amps = self.params.get('amps', [self.params.get('amp', DEFAULT_AMPLITUDE)])
        amp_str = ','.join(str(a) for a in amps) if amps else str(DEFAULT_AMPLITUDE)
        
        # Normalized amplitude input
        self.amp_edit = QtWidgets.QLineEdit(amp_str)
        self.amp_edit.setToolTip("Enter a single value or comma-separated list (e.g., 0.001,0.01,0.1)")
        amp_layout.addRow("Normalized Amplitude:", self.amp_edit)
        
        # dBm input 
        self.dbm_edit = QtWidgets.QLineEdit()
        self.dbm_edit.setToolTip("Enter a single value or comma-separated list in dBm (e.g., -30,-20,-10)")
        amp_layout.addRow("Power (dBm):", self.dbm_edit)
        
        # DAC scale information
        self.dac_scale_info = QtWidgets.QLabel("Fetching DAC scales...")
        self.dac_scale_info.setWordWrap(True)
        amp_layout.addRow("DAC Scale (dBm):", self.dac_scale_info)
        
        # Connect signals for updating between normalized and dBm
        # Only update values during typing, but don't validate
        self.amp_edit.textChanged.connect(self._update_dbm_from_normalized_no_validate)
        self.dbm_edit.textChanged.connect(self._update_normalized_from_dbm_no_validate)
        
        # Add validation when editing is finished
        self.amp_edit.editingFinished.connect(self._validate_normalized_values)
        self.dbm_edit.editingFinished.connect(self._validate_dbm_values)
        
        # Add the group to the main layout with a title in the left column
        layout.addRow("Amplitude Settings:", amp_group)
        
        return amp_group
        
    def _update_dbm_from_normalized_no_validate(self):
        """Update dBm values based on normalized amplitude inputs, without validation warnings."""
        if self.currently_updating:
            return  # Prevent recursive updates
            
        # Skip if dBm field is disabled (unknown DAC scale)
        if not self.dbm_edit.isEnabled():
            return
            
        self.currently_updating = True
        try:
            amp_text = self.amp_edit.text().strip()
            if not amp_text:
                self.dbm_edit.setText("")
                return
                
            # Get DAC scale for selected modules
            dac_scale = self._get_selected_dac_scale()
            if dac_scale is None:
                # If scale becomes unknown, disable the field
                self.dbm_edit.setEnabled(False)
                self.dbm_edit.setToolTip("Unable to query current DAC scale - dBm input disabled")
                self.dbm_edit.clear()
                return
                
            # Parse normalized values
            normalized_values = self._parse_amplitude_values(amp_text)
            
            # Convert to dBm
            dbm_values = []
            for norm in normalized_values:
                dbm = UnitConverter.normalize_to_dbm(norm, dac_scale)
                dbm_values.append(f"{dbm:.2f}")
            
            # Update dBm field
            self.dbm_edit.setText(", ".join(dbm_values))
        finally:
            self.currently_updating = False

    def _update_normalized_from_dbm_no_validate(self):
        """Update normalized amplitude values based on dBm inputs, without validation warnings."""
        if self.currently_updating:
            return  # Prevent recursive updates
            
        # Skip if dBm field is disabled (unknown DAC scale)
        if not self.dbm_edit.isEnabled():
            return
            
        self.currently_updating = True
        try:
            dbm_text = self.dbm_edit.text().strip()
            if not dbm_text:
                self.amp_edit.setText("")
                return
                
            # Get DAC scale for selected modules
            dac_scale = self._get_selected_dac_scale()
            if dac_scale is None:
                # If scale becomes unknown, disable the field
                self.dbm_edit.setEnabled(False)
                self.dbm_edit.setToolTip("Unable to query current DAC scale - dBm input disabled")
                self.dbm_edit.clear()
                return
                
            # Parse dBm values
            dbm_values = self._parse_dbm_values(dbm_text)
            
            # Convert to normalized amplitude
            normalized_values = []
            for dbm in dbm_values:
                norm = UnitConverter.dbm_to_normalize(dbm, dac_scale)
                normalized_values.append(f"{norm:.6f}")
            
            # Update normalized field
            self.amp_edit.setText(", ".join(normalized_values))
        finally:
            self.currently_updating = False

    def _validate_normalized_values(self):
        """Validate normalized amplitude values when editing is finished."""
        amp_text = self.amp_edit.text().strip()
        if not amp_text:
            return
            
        dac_scale = self._get_selected_dac_scale()
        if dac_scale is None:
            return
            
        normalized_values = self._parse_amplitude_values(amp_text)
        
        warnings = []
        for norm in normalized_values:
            if norm > 1.0:
                warnings.append(f"Warning: Normalized amplitude {norm:.6f} > 1.0 (maximum)")
            elif norm < 1e-4:
                warnings.append(f"Warning: Normalized amplitude {norm:.6f} < 1e-4 (minimum recommended)")
        
        if warnings:
            self._show_warning_dialog("Amplitude Warning", warnings)

    def _validate_dbm_values(self):
        """Validate dBm values when editing is finished."""
        dbm_text = self.dbm_edit.text().strip()
        if not dbm_text:
            return
            
        dac_scale = self._get_selected_dac_scale()
        if dac_scale is None:
            return
            
        dbm_values = self._parse_dbm_values(dbm_text)
        
        warnings = []
        for dbm in dbm_values:
            if dbm > dac_scale:
                warnings.append(f"Warning: {dbm:.2f} dBm > {dac_scale:+.2f} dBm (DAC max)")
            
            norm = UnitConverter.dbm_to_normalize(dbm, dac_scale)
            if norm > 1.0:
                warnings.append(f"Warning: {dbm:.2f} dBm gives normalized amplitude > 1.0")
            elif norm < 1e-4:
                warnings.append(f"Warning: {dbm:.2f} dBm gives normalized amplitude < 1e-4")
        
        if warnings:
            self._show_warning_dialog("Amplitude Warning", warnings)

    def _show_warning_dialog(self, title, warnings):
        """Show a warning dialog with the given messages."""
        QtWidgets.QMessageBox.warning(self, title, "\n".join(warnings))
            
    def _parse_amplitude_values(self, amp_text):
        """Parse comma-separated amplitude values."""
        normalized_values = []
        for part in amp_text.split(','):
            part = part.strip()
            if part:
                try:
                    value = float(eval(part))
                    normalized_values.append(value)
                except (ValueError, SyntaxError, NameError):
                    continue
        return normalized_values
        
    def _parse_dbm_values(self, dbm_text):
        """Parse comma-separated dBm values."""
        dbm_values = []
        for part in dbm_text.split(','):
            part = part.strip()
            if part:
                try:
                    value = float(eval(part))
                    dbm_values.append(value)
                except (ValueError, SyntaxError, NameError):
                    continue
        return dbm_values

    def _update_dac_scale_info(self):
        """Update the DAC scale information label based on selected modules."""
        selected_modules = self._get_selected_modules()
        
        # Check if any selected module has a known DAC scale
        has_known_scale = False
        scales = []
        
        for m in selected_modules:
            dac_scale = self.dac_scales.get(m)
            if dac_scale is not None:
                has_known_scale = True
                scales.append(f"Module {m}: {dac_scale:+.2f} dBm")
            else:
                scales.append(f"Module {m}: Unknown")
        
        # Update the info text
        if not selected_modules:
            text = "Unknown (no modules selected)"
        else:
            text = "\n".join(scales)
        
        self.dac_scale_info.setText(text)
        
        # Enable/disable dBm input based on whether we have known scales
        if has_known_scale:
            self.dbm_edit.setEnabled(True)
            self.dbm_edit.setToolTip("Enter a single value or comma-separated list in dBm (e.g., -30,-20,-10)")
            self._update_dbm_from_normalized()  # Update conversion
        else:
            self.dbm_edit.setEnabled(False)
            self.dbm_edit.setToolTip("Unable to query current DAC scale - dBm input disabled")
            self.dbm_edit.clear()  # Clear any existing text
    
    def _get_selected_modules(self):
        """
        Get the list of currently selected modules.
        This method should be overridden by subclasses.
        """
        return []
        
    def _get_selected_dac_scale(self):
        """Get the DAC scale for the currently selected module(s)."""
        selected_modules = self._get_selected_modules()
        
        if not selected_modules:
            return None  # No modules selected
            
        # Look for the first module with a known scale
        for module in selected_modules:
            dac_scale = self.dac_scales.get(module)
            if dac_scale is not None:
                return dac_scale
                
        # If no selected module has a known scale, return None
        return None
    
    def _update_dbm_from_normalized(self):
        """Update dBm values based on normalized amplitude inputs."""
        if self.currently_updating:
            return  # Prevent recursive updates
            
        # Skip if dBm field is disabled (unknown DAC scale)
        if not self.dbm_edit.isEnabled():
            return
            
        self.currently_updating = True
        try:
            amp_text = self.amp_edit.text().strip()
            if not amp_text:
                self.dbm_edit.setText("")
                return
                
            # Get DAC scale for selected modules
            dac_scale = self._get_selected_dac_scale()
            if dac_scale is None:
                # If scale becomes unknown, disable the field
                self.dbm_edit.setEnabled(False)
                self.dbm_edit.setToolTip("Unable to query current DAC scale - dBm input disabled")
                self.dbm_edit.clear()
                return
                
            # Parse normalized values
            normalized_values = self._parse_amplitude_values(amp_text)
            
            # Convert to dBm
            dbm_values = []
            warnings = []
            
            for norm in normalized_values:
                if norm > 1.0:
                    warnings.append(f"Warning: Normalized amplitude {norm:.6f} > 1.0 (maximum)")
                elif norm < 1e-4:
                    warnings.append(f"Warning: Normalized amplitude {norm:.6f} < 1e-4 (minimum recommended)")
                    
                dbm = UnitConverter.normalize_to_dbm(norm, dac_scale)
                dbm_values.append(f"{dbm:.2f}")
            
            # Update dBm field
            self.dbm_edit.setText(", ".join(dbm_values))
            
            # Show warnings if any
            if warnings:
                self._show_warning_dialog("Amplitude Warning", warnings)
        finally:
            self.currently_updating = False
    
    def _update_normalized_from_dbm(self):
        """Update normalized amplitude values based on dBm inputs."""
        if self.currently_updating:
            return  # Prevent recursive updates
            
        # Skip if dBm field is disabled (unknown DAC scale)
        if not self.dbm_edit.isEnabled():
            return
            
        self.currently_updating = True
        try:
            dbm_text = self.dbm_edit.text().strip()
            if not dbm_text:
                self.amp_edit.setText("")
                return
                
            # Get DAC scale for selected modules
            dac_scale = self._get_selected_dac_scale()
            if dac_scale is None:
                # If scale becomes unknown, disable the field
                self.dbm_edit.setEnabled(False)
                self.dbm_edit.setToolTip("Unable to query current DAC scale - dBm input disabled")
                self.dbm_edit.clear()
                return
                
            # Parse dBm values
            dbm_values = self._parse_dbm_values(dbm_text)
            
            # Convert to normalized amplitude
            normalized_values = []
            warnings = []
            
            for dbm in dbm_values:
                if dbm > dac_scale:
                    warnings.append(f"Warning: {dbm:.2f} dBm > {dac_scale:.2f} dBm (DAC max)")
                
                norm = UnitConverter.dbm_to_normalize(dbm, dac_scale)
                
                if norm > 1.0:
                    warnings.append(f"Warning: {dbm:.2f} dBm gives normalized amplitude > 1.0")
                elif norm < 1e-4:
                    warnings.append(f"Warning: {dbm:.2f} dBm gives normalized amplitude < 1e-4")
                    
                normalized_values.append(f"{norm:.6f}")
            
            # Update normalized field
            self.amp_edit.setText(", ".join(normalized_values))
            
            # Show warnings if any
            if warnings:
                self._show_warning_dialog("Amplitude Warning", warnings)
        finally:
            self.currently_updating = False

class NetworkAnalysisDialog(NetworkAnalysisDialogBase):
    """Dialog for configuring network analysis parameters with dBm support."""
    def __init__(self, parent=None, modules=None, dac_scales=None):
        super().__init__(parent, None, modules, dac_scales)
        self.setWindowTitle("Network Analysis Configuration")
        self.setModal(False)
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the user interface for the dialog."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Parameters group
        param_group = QtWidgets.QGroupBox("Analysis Parameters")
        param_layout = QtWidgets.QFormLayout(param_group)
        
        # Module selection
        self.module_entry = QtWidgets.QLineEdit("All")
        self.module_entry.setToolTip("Enter module numbers (e.g., '1,2,5' or '1-4' or 'All')")
        self.module_entry.textChanged.connect(self._update_dac_scale_info)
        param_layout.addRow("Modules:", self.module_entry)
        
        # Frequency range (in MHz instead of Hz)
        self.fmin_edit = QtWidgets.QLineEdit(str(DEFAULT_MIN_FREQ / 1e6))
        self.fmax_edit = QtWidgets.QLineEdit(str(DEFAULT_MAX_FREQ / 1e6))
        param_layout.addRow("Min Frequency (MHz):", self.fmin_edit)
        param_layout.addRow("Max Frequency (MHz):", self.fmax_edit)

        # Cable length
        self.cable_length_edit = QtWidgets.QLineEdit(str(DEFAULT_CABLE_LENGTH))
        param_layout.addRow("Cable Length (m):", self.cable_length_edit)        
        
        # Add amplitude settings group
        self.setup_amplitude_group(param_layout)
        
        # Remainder of the original UI
        self.points_edit = QtWidgets.QLineEdit(str(DEFAULT_NPOINTS))
        param_layout.addRow("Number of Points:", self.points_edit)
        
        self.samples_edit = QtWidgets.QLineEdit(str(DEFAULT_NSAMPLES))
        param_layout.addRow("Samples to Average:", self.samples_edit)
        
        self.max_chans_edit = QtWidgets.QLineEdit(str(DEFAULT_MAX_CHANNELS))
        param_layout.addRow("Max Channels:", self.max_chans_edit)
        
        self.max_span_edit = QtWidgets.QLineEdit(str(DEFAULT_MAX_SPAN / 1e6))
        param_layout.addRow("Max Span (MHz):", self.max_span_edit)
        
        self.clear_channels_cb = QtWidgets.QCheckBox("Clear all channels first")
        self.clear_channels_cb.setChecked(True)
        param_layout.addRow("", self.clear_channels_cb)
        
        layout.addWidget(param_group)
        
        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start Analysis")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        
        # Connect buttons
        self.start_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        
        # Initialize dBm field based on default normalized value
        self._update_dbm_from_normalized()
        self.setMinimumSize(500, 600)  # Width, height in pixels
        
    def _get_selected_modules(self):
        """Parse module entry to determine which modules are selected."""
        module_text = self.module_entry.text().strip()
        selected_modules = []
        
        if module_text.lower() == 'all':
            selected_modules = list(range(1, 9))  # All modules
        else:
            # Parse comma-separated values and ranges
            for part in module_text.split(','):
                part = part.strip()
                if '-' in part:
                    try:
                        start, end = map(int, part.split('-'))
                        selected_modules.extend(range(start, end + 1))
                    except ValueError:
                        continue
                elif part:
                    try:
                        selected_modules.append(int(part))
                    except ValueError:
                        continue
        
        return selected_modules

    def get_parameters(self):
        """Get the configured parameters."""
        try:
            # Parse module entry
            module_text = self.module_entry.text().strip()
            selected_module = None
            if module_text.lower() != 'all':
                selected_modules = self._get_selected_modules()
                if selected_modules:
                    selected_module = selected_modules
            
            # Parse amplitude values
            amp_text = self.amp_edit.text().strip()
            amps = self._parse_amplitude_values(amp_text)
            if not amps:
                amps = [DEFAULT_AMPLITUDE]  # Default amplitude if none provided
            
            params = {
                'amps': amps,
                'module': selected_module,
                'fmin': float(eval(self.fmin_edit.text())) * 1e6,  # Convert MHz to Hz
                'fmax': float(eval(self.fmax_edit.text())) * 1e6,  # Convert MHz to Hz
                'cable_length': float(self.cable_length_edit.text()),
                'npoints': int(self.points_edit.text()),
                'nsamps': int(self.samples_edit.text()),
                'max_chans': int(self.max_chans_edit.text()),
                'max_span': float(eval(self.max_span_edit.text())) * 1e6,  # Convert MHz to Hz
                'clear_channels': self.clear_channels_cb.isChecked()
            }
            return params
        except Exception as e:
            traceback.print_exc()  # Print full stacktrace to console
            QtWidgets.QMessageBox.critical(self, "Error", f"Invalid parameter: {str(e)}")
            return None

class NetworkAnalysisParamsDialog(NetworkAnalysisDialogBase):
    """Dialog for editing network analysis parameters with dBm support."""
    def __init__(self, parent=None, params=None):
        super().__init__(parent, params)
        self.setWindowTitle("Edit Network Analysis Parameters")
        self.setModal(True)
        self._setup_ui()
        
        if parent and hasattr(parent, 'parent') and parent.parent() is not None:
            parent_main = parent.parent()
            if hasattr(parent_main, 'crs') and parent_main.crs is not None:
                self._fetch_dac_scales(parent_main.crs)
        
    def _fetch_dac_scales(self, crs):
        """Fetch DAC scales for all modules."""
        # Create a fetcher thread
        self.fetcher = DACScaleFetcher(crs)
        self.fetcher.dac_scales_ready.connect(self._on_dac_scales_ready)
        self.fetcher.start()
    
    def _on_dac_scales_ready(self, scales):
        """Handle fetched DAC scales."""
        self.dac_scales = scales
        self._update_dac_scale_info()
        self._update_dbm_from_normalized()
    
    def _setup_ui(self):
        """Set up the user interface for the dialog."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Parameters form
        form = QtWidgets.QFormLayout()
        
        # Frequency range (in MHz instead of Hz)
        fmin_mhz = str(self.params.get('fmin', DEFAULT_MIN_FREQ) / 1e6)
        fmax_mhz = str(self.params.get('fmax', DEFAULT_MAX_FREQ) / 1e6)
        self.fmin_edit = QtWidgets.QLineEdit(fmin_mhz)
        self.fmax_edit = QtWidgets.QLineEdit(fmax_mhz)
        form.addRow("Min Frequency (MHz):", self.fmin_edit)
        form.addRow("Max Frequency (MHz):", self.fmax_edit)
        
        # Add amplitude settings group
        self.setup_amplitude_group(form)
        
        # Number of points
        self.points_edit = QtWidgets.QLineEdit(str(self.params.get('npoints', DEFAULT_NPOINTS)))
        form.addRow("Number of Points:", self.points_edit)
        
        # Number of samples to average
        self.samples_edit = QtWidgets.QLineEdit(str(self.params.get('nsamps', DEFAULT_NSAMPLES)))
        form.addRow("Samples to Average:", self.samples_edit)
        
        # Max channels
        self.max_chans_edit = QtWidgets.QLineEdit(str(self.params.get('max_chans', DEFAULT_MAX_CHANNELS)))
        form.addRow("Max Channels:", self.max_chans_edit)
        
        # Max span
        max_span_mhz = str(self.params.get('max_span', DEFAULT_MAX_SPAN) / 1e6)
        self.max_span_edit = QtWidgets.QLineEdit(max_span_mhz)
        form.addRow("Max Span (MHz):", self.max_span_edit)
        
        # Add checkbox for clearing channels
        self.clear_channels_cb = QtWidgets.QCheckBox("Clear all channels first")
        self.clear_channels_cb.setChecked(self.params.get('clear_channels', True))
        form.addRow("", self.clear_channels_cb)
        
        layout.addLayout(form)
        
        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        
        # Connect buttons
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        
        # Initialize dBm field based on default normalized value
        self._update_dbm_from_normalized()

        self.setMinimumSize(500, 600)  # Width, height in pixels

    def _get_selected_modules(self):
        """Get selected modules from params."""
        # Get selected modules from params
        selected_module = self.params.get('module')
        if selected_module is None:
            # Using all modules
            return list(range(1, 9))
        elif isinstance(selected_module, list):
            # Multiple specific modules
            return selected_module
        else:
            # Single module
            return [selected_module]
    
    def get_parameters(self):
        """Get the updated parameters."""
        try:
            # Parse amplitude values
            amp_text = self.amp_edit.text().strip()
            amps = self._parse_amplitude_values(amp_text)
            if not amps:
                amps = [DEFAULT_AMPLITUDE]  # Default amplitude if none provided
            
            params = self.params.copy()
            params.update({
                'amps': amps,  # Store as list in 'amps'
                'amp': amps[0],  # Also store first value in 'amp' for backward compatibility
                'fmin': float(eval(self.fmin_edit.text())) * 1e6,  # Convert MHz to Hz
                'fmax': float(eval(self.fmax_edit.text())) * 1e6,  # Convert MHz to Hz
                'npoints': int(self.points_edit.text()),
                'nsamps': int(self.samples_edit.text()),
                'max_chans': int(self.max_chans_edit.text()),
                'max_span': float(eval(self.max_span_edit.text())) * 1e6,  # Convert MHz to Hz
                'clear_channels': self.clear_channels_cb.isChecked()
            })
            return params
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Invalid parameter: {str(e)}")
            return None

class NetworkAnalysisWindow(QtWidgets.QMainWindow):
    """
    Window for displaying network analysis results with real units support.
    """
    def __init__(self, parent=None, modules=None, dac_scales=None):
        super().__init__(parent)
        self.setWindowTitle("Network Analysis Results")
        self.modules = modules or []
        self.data = {}  # module -> amplitude data dictionary
        self.raw_data = {}  # Store the raw IQ data for unit conversion
        self.unit_mode = "dbm"  # Default to dBm instead of counts
        self.normalize_magnitudes = False  # Add this flag to track normalization state
        self.first_setup = True  # Flag to track initial setup
        self.zoom_box_mode = True  # Default to zoom box mode ON
        self.plots = {}  # Initialize plots dictionary early
        self.original_params = {}  # Initial parameters
        self.current_params = {}   # Most recently used parameters
        self.dac_scales = dac_scales or {}  # Store DAC scales
        
        # Setup the UI components
        self._setup_ui()
        # Set initial size only on creation
        self.resize(1000, 800)

    def _setup_ui(self):
        """Set up the user interface for the window."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        
        # Create toolbar
        self._setup_toolbar(layout)
        
        # Create progress bars
        self._setup_progress_bars(layout)
        
        # Create plot area
        self._setup_plot_area(layout)

    def _setup_toolbar(self, layout):
        """Set up the toolbar with controls."""
        toolbar = QtWidgets.QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Cable length control for quick adjustments
        self.cable_length_label = QtWidgets.QLabel("Cable Length (m):")
        self.cable_length_spin = QtWidgets.QDoubleSpinBox()
        self.cable_length_spin.setRange(0.0, 1000.0)
        self.cable_length_spin.setValue(DEFAULT_CABLE_LENGTH)
        self.cable_length_spin.setSingleStep(0.05)
        toolbar.addWidget(self.cable_length_label)
        toolbar.addWidget(self.cable_length_spin)        

        # Add edit parameters button to toolbar
        edit_params_btn = QtWidgets.QPushButton("Edit Other Parameters")
        edit_params_btn.clicked.connect(self._edit_parameters)
        toolbar.addWidget(edit_params_btn)

        # Re-run analysis button
        rerun_btn = QtWidgets.QPushButton("Re-run Analysis")
        rerun_btn.clicked.connect(self._rerun_analysis)
        toolbar.addWidget(rerun_btn)        

        # Export button
        export_btn = QtWidgets.QPushButton("Export Data")
        export_btn.clicked.connect(self._export_data)
        toolbar.addWidget(export_btn) 
        
        # Add spacer to push the unit controls to the far right
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, 
                            QtWidgets.QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        
        # Normalize Magnitudes checkbox
        self.normalize_checkbox = QtWidgets.QCheckBox("Normalize Magnitudes")
        self.normalize_checkbox.setChecked(False)
        self.normalize_checkbox.setToolTip("Normalize all magnitude curves to their first data point")
        self.normalize_checkbox.toggled.connect(self._toggle_normalization)
        toolbar.addWidget(self.normalize_checkbox)

        # Add unit controls
        self._setup_unit_controls(toolbar)
        
        # Add zoom box mode checkbox
        self._setup_zoom_box_control(toolbar)

    def _setup_unit_controls(self, toolbar):
        """Set up the unit selection controls."""
        unit_group = QtWidgets.QWidget()
        unit_layout = QtWidgets.QHBoxLayout(unit_group)
        unit_layout.setContentsMargins(0, 0, 0, 0)
        unit_layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.rb_counts = QtWidgets.QRadioButton("Counts")
        self.rb_dbm = QtWidgets.QRadioButton("dBm")
        self.rb_volts = QtWidgets.QRadioButton("Volts")
        self.rb_dbm.setChecked(True)  # Changed from rb_counts to rb_dbm
        
        unit_layout.addWidget(QtWidgets.QLabel("Units:"))
        unit_layout.addWidget(self.rb_counts)
        unit_layout.addWidget(self.rb_dbm)
        unit_layout.addWidget(self.rb_volts)
        
        # Connect signals
        self.rb_counts.toggled.connect(lambda: self._update_unit_mode("counts"))
        self.rb_dbm.toggled.connect(lambda: self._update_unit_mode("dbm"))
        self.rb_volts.toggled.connect(lambda: self._update_unit_mode("volts"))
        
        # Set fixed size policy to make alignment more predictable
        unit_group.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, 
                                QtWidgets.QSizePolicy.Policy.Preferred)
        
        # Now add the unit controls at the far right
        toolbar.addSeparator()
        toolbar.addWidget(unit_group)
        
    def _setup_zoom_box_control(self, toolbar):
        """Set up the zoom box mode control."""
        zoom_box_cb = QtWidgets.QCheckBox("Zoom Box Mode")
        zoom_box_cb.setChecked(self.zoom_box_mode)  # Default to ON
        zoom_box_cb.setToolTip("When enabled, left-click drag creates a zoom box. When disabled, left-click drag pans.")
        zoom_box_cb.toggled.connect(self._toggle_zoom_box)
        
        # Store reference to the checkbox
        self.zoom_box_cb = zoom_box_cb
        
        toolbar.addWidget(zoom_box_cb)

    def _setup_progress_bars(self, layout):
        """Set up progress bars for each module."""
        self.progress_group = None
        if self.modules:
            self.progress_group = QtWidgets.QGroupBox("Analysis Progress")
            progress_layout = QtWidgets.QVBoxLayout(self.progress_group)
            
            self.progress_bars = {}
            self.progress_labels = {}  # Add labels to show amplitude progress
            for module in self.modules:
                vlayout = QtWidgets.QVBoxLayout()
                
                # Main progress layout
                hlayout = QtWidgets.QHBoxLayout()
                label = QtWidgets.QLabel(f"Module {module}:")
                pbar = QtWidgets.QProgressBar()
                pbar.setRange(0, 100)
                pbar.setValue(0)
                hlayout.addWidget(label)
                hlayout.addWidget(pbar)
                vlayout.addLayout(hlayout)
                
                # Amplitude progress label
                amp_label = QtWidgets.QLabel("")
                amp_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                vlayout.addWidget(amp_label)
                
                progress_layout.addLayout(vlayout)
                self.progress_bars[module] = pbar
                self.progress_labels[module] = amp_label
                
            layout.addWidget(self.progress_group)
        else:
            self.progress_bars = {}
            self.progress_labels = {}

    def _setup_plot_area(self, layout):
        """Set up the plot area with tabs for each module."""
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)
        
        self.plots = {}
        for module in self.modules:
            tab = QtWidgets.QWidget()
            tab_layout = QtWidgets.QVBoxLayout(tab)
            
            # Create amplitude and phase plots with ClickableViewBox
            vb_amp = ClickableViewBox()
            amp_plot = pg.PlotWidget(viewBox=vb_amp, title=f"Module {module} - Magnitude")
            self._update_amplitude_labels(amp_plot)
            amp_plot.setLabel('bottom', 'Frequency', units='Hz')
            amp_plot.showGrid(x=True, y=True, alpha=0.3)
            
            vb_phase = ClickableViewBox()
            phase_plot = pg.PlotWidget(viewBox=vb_phase, title=f"Module {module} - Phase")
            phase_plot.setLabel('left', 'Phase', units='deg')
            phase_plot.setLabel('bottom', 'Frequency', units='Hz')
            phase_plot.showGrid(x=True, y=True, alpha=0.3)
            
            # Add legends for multiple amplitude plots
            amp_legend = amp_plot.addLegend(offset=(30, 10))
            phase_legend = phase_plot.addLegend(offset=(30, 10))

            # Create curves with periscope color scheme - but don't add data yet
            amp_curve = amp_plot.plot([], [], pen=pg.mkPen('#ff7f0e', width=LINE_WIDTH))  # Empty data
            phase_curve = phase_plot.plot([], [], pen=pg.mkPen('#1f77b4', width=LINE_WIDTH))  # Empty data

            tab_layout.addWidget(amp_plot)
            tab_layout.addWidget(phase_plot)
            self.tabs.addTab(tab, f"Module {module}")
            
            self.plots[module] = {
                'amp_plot': amp_plot,
                'phase_plot': phase_plot,
                'amp_curve': amp_curve,
                'phase_curve': phase_curve,
                'amp_legend': amp_legend,
                'phase_legend': phase_legend,
                'amp_curves': {},  # Will store multiple curves for different amplitudes
                'phase_curves': {}  # Will store multiple curves for different amplitudes
            }
            
            # Apply zoom box mode
            self._apply_zoom_box_mode()            

            # Link the x-axis of amplitude and phase plots for synchronized zooming
            phase_plot.setXLink(amp_plot)

    def clear_plots(self):
        """Clear all plots, curves, and legends."""
        for module in self.plots:
            plot_info = self.plots[module]
            amp_plot = plot_info['amp_plot']
            phase_plot = plot_info['phase_plot']
            
            # Clear legends
            plot_info['amp_legend'].clear()
            plot_info['phase_legend'].clear()
            
            # Remove all amplitude-specific curves from plots
            for amp, curve in list(plot_info['amp_curves'].items()):
                amp_plot.removeItem(curve)
            for amp, curve in list(plot_info['phase_curves'].items()):
                phase_plot.removeItem(curve)
            
            # Clear curve dictionaries
            plot_info['amp_curves'].clear()
            plot_info['phase_curves'].clear()
            
            # Make sure main curves have no data 
            plot_info['amp_curve'].setData([], [])
            plot_info['phase_curve'].setData([], [])

    def update_amplitude_progress(self, module: int, current_amp: int, total_amps: int, amplitude: float):
        """Update the amplitude progress display for a module."""
        if hasattr(self, 'progress_labels') and module in self.progress_labels:
            self.progress_labels[module].setText(f"Amplitude {current_amp}/{total_amps} ({amplitude})")

    def _toggle_normalization(self, checked):
        """Toggle normalization of magnitude plots."""
        self.normalize_magnitudes = checked
        
        # Update axis labels
        for module in self.plots:
            self._update_amplitude_labels(self.plots[module]['amp_plot'])
        
        # Make sure main curves are cleared if we have amplitude-specific curves
        for module in self.plots:
            if len(self.plots[module]['amp_curves']) > 0:
                # Clear main curves if we have amplitude-specific curves
                self.plots[module]['amp_curve'].setData([], [])
                self.plots[module]['phase_curve'].setData([], [])
        
        # Redraw plots with normalization applied
        self._redraw_all_plots()

    def _toggle_zoom_box(self, enable):
        """Toggle zoom box mode for all plots."""
        self.zoom_box_mode = enable
        self._apply_zoom_box_mode()
        
    def _apply_zoom_box_mode(self):
        """Apply the current zoom box mode setting to all plots."""
        for module in self.plots:
            for plot_type in ['amp_plot', 'phase_plot']:
                viewbox = self.plots[module][plot_type].getViewBox()
                if isinstance(viewbox, ClickableViewBox):
                    viewbox.enableZoomBoxMode(self.zoom_box_mode)            

    def _update_unit_mode(self, mode):
        """Update unit mode and redraw plots."""
        if mode != self.unit_mode:
            self.unit_mode = mode
            # Update all plot labels
            for module in self.plots:
                self._update_amplitude_labels(self.plots[module]['amp_plot'])
            
            # Redraw with new units
            self._redraw_all_plots()

    def _update_legends_for_unit_mode(self):
        """Update the legend entries to reflect the current unit mode."""
        for module in self.plots:
            # Clear existing legends
            self.plots[module]['amp_legend'].clear()
            self.plots[module]['phase_legend'].clear()
            
            # Re-add curves with updated labels
            for amplitude, curve in self.plots[module]['amp_curves'].items():
                # Format amplitude according to current unit mode
                if self.unit_mode == "dbm":
                    # Check if we have a DAC scale for this module
                    if module not in self.dac_scales:
                        # Issue warning and switch to counts mode
                        print(f"Warning: No DAC scale available for module {module}, cannot display accurate probe power in dBm.")
                        self.rb_counts.setChecked(True)  # Switch to counts mode
                        return  # Exit and let _update_unit_mode call us again
                    
                    # Use the actual DAC scale without fallback
                    dac_scale = self.dac_scales[module]
                    dbm_value = UnitConverter.normalize_to_dbm(amplitude, dac_scale)
                    label = f"Probe: {dbm_value:.2f} dBm"
                elif self.unit_mode == "volts":
                    # Properly convert normalized amplitude to voltage through power calculation
                    # First get dBm value
                    if module not in self.dac_scales:
                        print(f"Warning: No DAC scale available for module {module}, cannot display accurate probe amplitude in Volts.")
                        self.rb_counts.setChecked(True)  # Switch to counts mode
                        return  # Exit and let _update_unit_mode call us again
                    else:
                        dac_scale = self.dac_scales[module]
                        dbm_value = UnitConverter.normalize_to_dbm(amplitude, dac_scale)
                        
                        # Convert dBm to watts: P = 10^((dBm - 30)/10)
                        power_watts = 10**((dbm_value - 30)/10)
                        
                        # Convert watts to peak voltage: V = sqrt(P * R)
                        resistance = 50.0  # Ohms
                        voltage_rms = np.sqrt(power_watts * resistance)
                        
                        # Convert RMS to peak voltage
                        voltage_peak = voltage_rms * np.sqrt(2)
                        voltage_peak = voltage_peak*1e6
                        
                        label = f"Probe: {voltage_peak:.1f} uV (peak)"
                else:  # "counts"
                    label = f"Probe: {amplitude} Normalized Units"
                
                # Add to legend with new label
                self.plots[module]['amp_legend'].addItem(curve, label)
                
                # Also update the phase legend for consistency
                phase_curve = self.plots[module]['phase_curves'].get(amplitude)
                if phase_curve:
                    self.plots[module]['phase_legend'].addItem(phase_curve, label)         
    
    def _update_amplitude_labels(self, plot):
        """Update plot labels based on current unit mode and normalization state."""
        if self.normalize_magnitudes:
            if self.unit_mode == "dbm":
                plot.setLabel('left', 'Normalized Power', units='dBm')
            else:
                plot.setLabel('left', 'Normalized Magnitude', units='')
        else:
            if self.unit_mode == "counts":
                plot.setLabel('left', 'Magnitude', units='Counts')
            elif self.unit_mode == "dbm":
                plot.setLabel('left', 'Power', units='dBm')
            elif self.unit_mode == "volts":
                plot.setLabel('left', 'Magnitude', units='V')

    def _redraw_all_plots(self):
        """Redraw all plots with current unit mode."""
        for module in self.raw_data:
            if module in self.plots:
                # Check if we have amplitude-specific curves
                has_amp_curves = len(self.plots[module]['amp_curves']) > 0
                
                # Update amplitude-specific curves first
                for amp_key, data_tuple in self.raw_data[module].items():
                    if amp_key != 'default':
                        # Extract amplitude and data
                        amplitude, freqs, amps, phases, iq_data = self._extract_data_from_tuple(amp_key, data_tuple)
                        
                        # Update the curve if it exists
                        if amplitude in self.plots[module]['amp_curves']:
                            converted_amps = UnitConverter.convert_amplitude(
                                amps, iq_data, self.unit_mode, normalize=self.normalize_magnitudes)
                            freq_ghz = freqs / 1e9
                            self.plots[module]['amp_curves'][amplitude].setData(freq_ghz, converted_amps)
                            self.plots[module]['phase_curves'][amplitude].setData(freq_ghz, phases)
                
                # Now handle the default curve - ONLY if there are no amplitude-specific curves
                if 'default' in self.raw_data[module] and not has_amp_curves:
                    freqs, amps, phases, iq_data = self.raw_data[module]['default']
                    converted_amps = UnitConverter.convert_amplitude(
                        amps, iq_data, self.unit_mode, normalize=self.normalize_magnitudes)
                    freq_ghz = freqs / 1e9
                    self.plots[module]['amp_curve'].setData(freq_ghz, converted_amps)
                    self.plots[module]['phase_curve'].setData(freq_ghz, phases)
                else:
                    # Make sure default curves have no data if we have amplitude-specific curves
                    self.plots[module]['amp_curve'].setData([], [])
                    self.plots[module]['phase_curve'].setData([], [])
                
                # Enable auto range to fit new data
                self.plots[module]['amp_plot'].autoRange()
        
        # Update legends after redrawing curves
        self._update_legends_for_unit_mode()              
    
    def _extract_data_from_tuple(self, amp_key, data_tuple):
        """Extract amplitude and data from a data tuple."""
        if len(data_tuple) == 5:
            # New format with amplitude included
            freqs, amps, phases, iq_data, amplitude = data_tuple
        else:
            # Old format, extract amplitude from key
            freqs, amps, phases, iq_data = data_tuple
            try:
                amplitude = float(amp_key.split('_')[-1])
            except (ValueError, IndexError):
                amplitude = DEFAULT_AMPLITUDE
        return amplitude, freqs, amps, phases, iq_data

    def closeEvent(self, event):
        """Handle window close event by cleaning up resources."""
        parent = self.parent()
        
        if parent and hasattr(parent, 'netanal_windows'):
            # Find our window ID
            window_id = None
            for w_id, w_data in parent.netanal_windows.items():
                if w_data['window'] == self:
                    window_id = w_id
                    break
            
            if window_id:
                # Stop all tasks for this window
                if hasattr(parent, 'netanal_tasks'):
                    for task_key in list(parent.netanal_tasks.keys()):
                        if task_key.startswith(f"{window_id}_"):
                            task = parent.netanal_tasks.pop(task_key)
                            task.stop()
                
                # Remove from windows dictionary
                parent.netanal_windows.pop(window_id, None)
        
        # Call parent implementation
        super().closeEvent(event)

    def _check_all_complete(self):
        """
        Check if all progress bars are at 100% and hide the progress group 
        when all analyses are complete.
        """
        if not self.progress_group:
            return
            
        # Find our window data in the parent
        parent = self.parent()
        window_id = None
        
        if parent and hasattr(parent, 'netanal_windows'):
            for w_id, w_data in parent.netanal_windows.items():
                if w_data['window'] == self:
                    window_id = w_id
                    break
        
        if not window_id:
            return
            
        window_data = parent.netanal_windows[window_id]
        
        # Check if there are any pending amplitudes
        no_pending_amplitudes = True
        for module in window_data['amplitude_queues']:
            if window_data['amplitude_queues'][module]:
                no_pending_amplitudes = False
                break
        
        # Hide progress when all bars are at 100% and no pending amplitudes
        all_complete = all(pbar.value() == 100 for pbar in self.progress_bars.values())
        if all_complete and no_pending_amplitudes:
            self.progress_group.setVisible(False)

    def _edit_parameters(self):
        """Open dialog to edit parameters besides cable length."""
        dialog = NetworkAnalysisParamsDialog(self, self.current_params)
        if dialog.exec():
            # Get updated parameters
            params = dialog.get_parameters()
            if params:
                # Keep the current cable length
                params['cable_length'] = self.cable_length_spin.value()
                # Update current parameters
                self.current_params = params.copy()
                # Make progress group visible again
                if self.progress_group:
                    self.progress_group.setVisible(True)
                self.parent()._rerun_network_analysis(params)

    def _rerun_analysis(self):
        """Re-run the analysis with potentially updated parameters."""
        if hasattr(self.parent(), '_rerun_network_analysis'):
            # Get current parameters and update cable length
            params = self.current_params.copy()
            params['cable_length'] = self.cable_length_spin.value()
            # Make progress group visible again
            if self.progress_group:
                self.progress_group.setVisible(True)
            self.parent()._rerun_network_analysis(params)
    
    def set_params(self, params):
        """Set parameters for analysis."""
        self.original_params = params.copy()  # Keep original for reference
        self.current_params = params.copy()   # Keep current for dialog
        
        # Set cable length spinner to match value
        self.cable_length_spin.setValue(params.get('cable_length', DEFAULT_CABLE_LENGTH))
        
        # Only try to set plot ranges if plots exist
        if not hasattr(self, 'plots') or not self.plots:
            return
            
        # Set initial plot ranges based on frequency parameters
        fmin = params.get('fmin', DEFAULT_MIN_FREQ)
        fmax = params.get('fmax', DEFAULT_MAX_FREQ)
        for module in self.plots:
            self.plots[module]['amp_plot'].setXRange(fmin, fmax)
            self.plots[module]['phase_plot'].setXRange(fmin, fmax)
            # Disable auto range on X axis but keep Y auto range
            self.plots[module]['amp_plot'].enableAutoRange(pg.ViewBox.XAxis, False)
            self.plots[module]['phase_plot'].enableAutoRange(pg.ViewBox.XAxis, False)
            self.plots[module]['amp_plot'].enableAutoRange(pg.ViewBox.YAxis, True)
            self.plots[module]['phase_plot'].enableAutoRange(pg.ViewBox.YAxis, True)
    
    def update_data_with_amp(self, module: int, freqs: np.ndarray, amps: np.ndarray, phases: np.ndarray, amplitude: float):
        """Update the plot data for a specific module and amplitude."""
        # Store the raw data for unit conversion
        iq_data = amps * np.exp(1j * np.radians(phases))  # Reconstruct complex data
        
        # Use a unique key that includes the amplitude
        key = f"{module}_{amplitude}"
        
        # Initialize dictionaries if needed
        if module not in self.raw_data:
            self.raw_data[module] = {}
        if module not in self.data:
            self.data[module] = {}
        
        # Store the data with amplitude
        self.raw_data[module][key] = (freqs, amps, phases, iq_data, amplitude)
        self.data[module][key] = (freqs, amps, phases)
        
        if module in self.plots:
            # Hide the main curve if this is the first amplitude-specific curve
            if len(self.plots[module]['amp_curves']) == 0:
                # Set main curves to empty data to effectively hide them
                self.plots[module]['amp_curve'].setData([], [])
                self.plots[module]['phase_curve'].setData([], [])
            
            # Convert amplitude to selected units
            converted_amps = UnitConverter.convert_amplitude(
                amps, iq_data, self.unit_mode, normalize=self.normalize_magnitudes)
            
            # Generate a color based on the amplitude index in the list of amplitudes
            amps_list = self.original_params.get('amps', [amplitude])
            if amplitude in amps_list:
                amp_index = amps_list.index(amplitude)
            else:
                amp_index = 0
                
            # Use the same color families as the main application
            channel_families = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
            ]
            color = channel_families[amp_index % len(channel_families)]
            
            # Create or update curves for this amplitude
            is_new_curve = False
            if amplitude not in self.plots[module]['amp_curves']:
                is_new_curve = True
                self.plots[module]['amp_curves'][amplitude] = self.plots[module]['amp_plot'].plot(
                    pen=pg.mkPen(color, width=LINE_WIDTH), name=f"Amp: {amplitude}")
                self.plots[module]['phase_curves'][amplitude] = self.plots[module]['phase_plot'].plot(
                    pen=pg.mkPen(color, width=LINE_WIDTH), name=f"Amp: {amplitude}")
            
            # Update the curves
            self.plots[module]['amp_curves'][amplitude].setData(freqs, converted_amps)
            self.plots[module]['phase_curves'][amplitude].setData(freqs, phases)
            
            # If this is a new curve, update legends for proper unit display
            if is_new_curve:
                self._update_legends_for_unit_mode()

    def update_data(self, module: int, freqs: np.ndarray, amps: np.ndarray, phases: np.ndarray):
        """Update the plot data for a specific module."""
        # Store the raw data for unit conversion
        iq_data = amps * np.exp(1j * np.radians(phases))  # Reconstruct complex data
        
        # Initialize dictionaries if needed
        if module not in self.raw_data:
            self.raw_data[module] = {}
        if module not in self.data:
            self.data[module] = {}
        
        # Store under 'default' key
        self.raw_data[module]['default'] = (freqs, amps, phases, iq_data)
        self.data[module]['default'] = (freqs, amps, phases)
        
        if module in self.plots:
            # Only show default curve if no amplitude-specific curves exist yet
            if len(self.plots[module]['amp_curves']) == 0:
                # Convert amplitude to selected units
                converted_amps = UnitConverter.convert_amplitude(
                    amps, iq_data, self.unit_mode, normalize=self.normalize_magnitudes)
                
                # Convert frequency to GHz for display
                freq_ghz = freqs
                
                # Update plots
                self.plots[module]['amp_curve'].setData(freq_ghz, converted_amps)
                self.plots[module]['phase_curve'].setData(freq_ghz, phases)
    
    def update_progress(self, module: int, progress: float):
        """Update the progress bar for a specific module."""
        if module in self.progress_bars:
            self.progress_bars[module].setValue(int(progress))
    
    def complete_analysis(self, module: int):
        """Mark analysis as complete for a module."""
        if module in self.progress_bars:
            self.progress_bars[module].setValue(100)
            # Check if all modules are complete to hide progress bars
            self._check_all_complete()
    
    def _export_data(self):
        """Export the collected data with all unit conversions and metadata."""
        if not self.data:
            QtWidgets.QMessageBox.warning(self, "No Data", "No data to export yet.")
            return
        
        dialog = QtWidgets.QFileDialog(self)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilters([
            "Pickle Files (*.pkl)",
            "CSV Files (*.csv)",
            "All Files (*)"
        ])
        dialog.setDefaultSuffix("pkl")
        
        if dialog.exec():
            filename = dialog.selectedFiles()[0]
            
            try:
                if filename.endswith('.pkl'):
                    self._export_to_pickle(filename)
                elif filename.endswith('.csv'):
                    self._export_to_csv(filename)
                else:
                    # Default to pickle with same comprehensive format
                    self._export_to_pickle(filename)
                    
                QtWidgets.QMessageBox.information(self, "Export Complete", 
                                                f"Data exported to {filename}")
                
            except Exception as e:
                traceback.print_exc()  # Print stack trace to console
                QtWidgets.QMessageBox.critical(self, "Export Error", 
                                            f"Error exporting data: {str(e)}")
    
    def _export_to_pickle(self, filename):
        """Export data to a pickle file."""
        # Create comprehensive data structure with all units and metadata
        export_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'parameters': self.current_params.copy() if hasattr(self, 'current_params') else {},
            'modules': {}
        }
        
        # Process each module's data
        for module, data_dict in self.raw_data.items():
            export_data['modules'][module] = {}
            
            # Track measurement index for each module
            meas_idx = 0
            
            for key, data_tuple in data_dict.items():
                # Extract data and amplitude
                amplitude, freqs, amps, phases, iq_data = self._extract_data_for_export(key, data_tuple)
                
                # Convert to all unit types
                counts = amps  # Already in counts
                volts = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="volts")
                dbm = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="dbm")
                
                # Also include normalized versions
                counts_norm = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="counts", normalize=True)
                volts_norm = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="volts", normalize=True)
                dbm_norm = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="dbm", normalize=True)
                
                # Use iteration number as key instead of formatted amplitude
                export_data['modules'][module][meas_idx] = {
                    'sweep_amplitude': amplitude,
                    'frequency': {
                        'values': freqs.tolist(),
                        'unit': 'Hz'
                    },
                    'magnitude': {
                        'counts': {
                            'raw': counts.tolist(),
                            'normalized': counts_norm.tolist(),
                            'unit': 'counts'
                        },
                        'volts': {
                            'raw': volts.tolist(),
                            'normalized': volts_norm.tolist(),
                            'unit': 'V'
                        },
                        'dbm': {
                            'raw': dbm.tolist(),
                            'normalized': dbm_norm.tolist(),
                            'unit': 'dBm'
                        }
                    },
                    'phase': {
                        'values': phases.tolist(),
                        'unit': 'degrees'
                    },
                    'complex': {
                        'real': iq_data.real.tolist(),
                        'imag': iq_data.imag.tolist()
                    }
                }
                meas_idx += 1
        
        # Save the enhanced data structure
        with open(filename, 'wb') as f:
            pickle.dump(export_data, f)
    
    def _export_to_csv(self, filename):
        """Export data to CSV files."""
        base, ext = os.path.splitext(filename)
        
        # First create a metadata CSV with the parameters
        meta_filename = f"{base}_metadata{ext}"
        with open(meta_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Parameter', 'Value'])
            writer.writerow(['Export Date', datetime.datetime.now().isoformat()])
            
            # Add all parameters
            if hasattr(self, 'current_params'):
                writer.writerow(['', ''])
                writer.writerow(['Measurement Parameters', ''])
                for param, value in self.current_params.items():
                    # Convert Hz to MHz for frequency parameters
                    if param in ['fmin', 'fmax', 'max_span'] and isinstance(value, (int, float)):
                        writer.writerow([param, f"{value/1e6} MHz"])
                    else:
                        writer.writerow([param, value])
        
        # Now export each module's data
        for module, data_dict in self.raw_data.items():
            idx = 0  # Track measurement index
            for key, data_tuple in data_dict.items():
                # Extract data and amplitude
                amplitude, freqs, amps, phases, iq_data = self._extract_data_for_export(key, data_tuple)
                
                # Export for each unit type
                for unit_mode in ["counts", "volts", "dbm"]:
                    converted_amps = UnitConverter.convert_amplitude(amps, iq_data, unit_mode=unit_mode)
                    
                    unit_label = unit_mode
                    if unit_mode == "dbm":
                        unit_label = "dBm"
                    elif unit_mode == "volts":
                        unit_label = "V"
                    
                    csv_filename = f"{base}_module{module}_idx{idx}_{unit_mode}{ext}"
                    with open(csv_filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['# Amplitude:', f"{amplitude}" if 'amplitude' in locals() else "Unknown"])
                        if unit_mode == "dbm":
                            writer.writerow(['Frequency (Hz)', f'Power ({unit_label})', 'Phase (deg)'])
                        else:
                            writer.writerow(['Frequency (Hz)', f'Amplitude ({unit_label})', 'Phase (deg)'])
                        
                        for freq, amp, phase in zip(freqs, converted_amps, phases):
                            writer.writerow([freq, amp, phase])
                idx += 1
    
    def _extract_data_for_export(self, key, data_tuple):
        """Extract and prepare data for export from a data tuple."""
        if key != 'default':
            if len(data_tuple) >= 5:  # New format with amplitude included
                freqs, amps, phases, iq_data, amplitude = data_tuple
            else:
                # Try to extract amplitude from key format "module_amp"
                freqs, amps, phases, iq_data = data_tuple
                try:
                    amplitude = float(key.split('_')[-1])
                except (ValueError, IndexError):
                    amplitude = DEFAULT_AMPLITUDE
        else:
            amplitude = DEFAULT_AMPLITUDE  # Default for non-amplitude-specific data
            freqs, amps, phases, iq_data = data_tuple
            
        return amplitude, freqs, amps, phases, iq_data

# ───────────────────────── Main Application ─────────────────────────
class Periscope(QtWidgets.QMainWindow):
    """
    Multi‑pane PyQt application for real-time data visualization of:
      - Time-domain waveforms (TOD)
      - IQ (density or scatter)
      - FFT
      - Single-sideband PSD (SSB)
      - Dual-sideband PSD (DSB)
      - Network Analysis (amplitude and phase vs frequency)

    Additional Features
    -------------------
    - Multi-channel grouping via '&'.
    - An "Auto Scale" checkbox for IQ/FFT/SSB/DSB (not TOD).
    - Global toggles to hide/show I, Q, and Mag lines for TOD, FFT, and SSB.
    - A "Help" button to display usage and interaction details.
    - Network Analysis functionality with real-time updates.

    Parameters
    ----------
    host : str
        The multicast/UDP host address for receiving packets.
    module : int
        The module index (1-based) used to filter incoming packets.
    chan_str : str, optional
        A comma-separated list of channels, possibly using '&' to group channels
        in one row. Defaults to "1".
    buf_size : int, optional
        Size of each ring buffer for storing incoming data (default is 5000).
    refresh_ms : int, optional
        GUI refresh interval in milliseconds (default 33).
    dot_px : int, optional
        Dot diameter in pixels for IQ density mode (default is 1).
    crs : CRS, optional
        CRS object for hardware communication (needed for network analysis).
    """

    def __init__(
        self,
        host: str,
        module: int,
        chan_str="1",
        buf_size=DEFAULT_BUFFER_SIZE,
        refresh_ms=DEFAULT_REFRESH_MS,
        dot_px=DENSITY_DOT_SIZE,
        crs=None,
    ):
        super().__init__()
        self.host = host
        self.module = module
        self.N = buf_size
        self.refresh_ms = refresh_ms
        self.dot_px = max(1, int(dot_px))
        self.crs = crs

        # Parse multi-channel format
        self.channel_list = _parse_channels_multich(chan_str)

        # State variables
        self.paused = False
        self.start_time = None
        self.frame_cnt = 0
        self.pkt_cnt = 0
        self.t_last = time.time()

        # Single decimation stage, updated from first channel's sample rate
        self.dec_stage = 6
        self.last_dec_update = 0.0

        # Display settings
        self.dark_mode = True
        self.real_units = False
        self.psd_absolute = True
        self.auto_scale_plots = True
        self.show_i = True
        self.show_q = True
        self.show_m = True

        # Initialize worker tracking
        self._init_workers()

        # Create color map
        self._init_colormap()

        # Start UDP receiver
        self._init_receiver()

        # Initialize thread pool
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(4)  # Allow multiple network analyses

        # Build UI
        self._build_ui(chan_str)
        self._init_buffers()
        self._build_layout()

        # Start the GUI update timer
        self._start_timer()

    def _init_workers(self):
        """Initialize worker tracking structures."""
        # IQ concurrency tracking
        self.iq_workers: Dict[int, bool] = {}
        self.iq_signals = IQSignals()
        self.iq_signals.done.connect(self._iq_done)

        # PSD concurrency tracking, per row -> "S"/"D" -> channel -> bool
        self.psd_workers: Dict[int, Dict[str, Dict[int, bool]]] = {}
        for row_i, group in enumerate(self.channel_list):
            self.psd_workers[row_i] = {"S": {}, "D": {}}
            for c in group:
                self.psd_workers[row_i]["S"][c] = False
                self.psd_workers[row_i]["D"][c] = False

        self.psd_signals = PSDSignals()
        self.psd_signals.done.connect(self._psd_done)

        # Network analysis tracking
        self.netanal_signals = NetworkAnalysisSignals()
        self.netanal_signals.progress.connect(self._netanal_progress)
        self.netanal_signals.data_update.connect(self._netanal_data_update)
        self.netanal_signals.data_update_with_amp.connect(self._netanal_data_update_with_amp)
        self.netanal_signals.completed.connect(self._netanal_completed)
        self.netanal_signals.error.connect(self._netanal_error)
        
        # Network analysis window tracking
        self.netanal_windows = {}  # Dictionary of windows indexed by unique ID
        self.netanal_window_count = 0  # Counter for window IDs
        self.netanal_tasks = {}  # Tasks dictionary with window-specific keys

    def _init_colormap(self):
        """Initialize the colormap for IQ density plots."""
        cmap = pg.colormap.get("turbo")
        lut_rgb = cmap.getLookupTable(0.0, 1.0, 255)
        self.lut = np.vstack([
            np.zeros((1, 4), np.uint8),
            np.hstack([lut_rgb, 255 * np.ones((255, 1), np.uint8)])
        ])

    def _init_receiver(self):
        """Initialize the UDP receiver."""
        self.receiver = UDPReceiver(self.host, self.module)
        self.receiver.start()

    def _start_timer(self):
        """Start the periodic GUI update timer."""
        self.timer = QtCore.QTimer(singleShot=False)
        self.timer.timeout.connect(self._update_gui)
        self.timer.start(self.refresh_ms)
        self.setWindowTitle("Periscope")

    # ───────────────────────── UI Construction ─────────────────────────
    def _build_ui(self, chan_str: str):
        """
        Create and configure all top-level widgets and layouts.
        
        Parameters
        ----------
        chan_str : str
            The user-supplied channel specification string.
        """
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_vbox = QtWidgets.QVBoxLayout(central)

        self._add_title(main_vbox)
        self._add_toolbar(main_vbox, chan_str)
        self._add_config_panel(main_vbox)
        self._add_plot_container(main_vbox)
        self._add_status_bar()

    def _add_title(self, layout):
        """Add the title to the layout."""
        title = QtWidgets.QLabel(f"CRS: {self.host}    Module: {self.module}")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        ft = title.font()
        ft.setPointSize(16)
        title.setFont(ft)
        layout.addWidget(title)

    def _add_toolbar(self, layout, chan_str):
        """Add the toolbar to the layout."""
        top_bar = QtWidgets.QWidget()
        top_h = QtWidgets.QHBoxLayout(top_bar)

        # Config toggle button
        self.btn_toggle_cfg = QtWidgets.QPushButton("Show Configuration")
        self.btn_toggle_cfg.setCheckable(True)
        self.btn_toggle_cfg.toggled.connect(self._toggle_config)

        # Channel input
        self.e_ch = QtWidgets.QLineEdit(chan_str)
        self.e_ch.setToolTip("Enter comma-separated channels or use '&' to group in one row.")
        self.e_ch.returnPressed.connect(self._update_channels)

        # Buffer size input
        self.e_buf = QtWidgets.QLineEdit(str(self.N))
        self.e_buf.setValidator(QIntValidator(10, 1_000_000, self))
        self.e_buf.setMaximumWidth(80)
        self.e_buf.setToolTip("Size of the ring buffer for each channel.")
        self.e_buf.editingFinished.connect(self._change_buffer)

        # Pause button
        self.b_pause = QtWidgets.QPushButton("Pause", clicked=self._toggle_pause)
        self.b_pause.setToolTip("Pause/resume real-time data.")

        # Real units checkbox
        self.cb_real = QtWidgets.QCheckBox("Real Units", checked=self.real_units)
        self.cb_real.setToolTip("Toggle raw 'counts' vs. real-voltage/dBm units.")
        self.cb_real.toggled.connect(self._toggle_real_units)

        # Mode checkboxes
        self.cb_time = QtWidgets.QCheckBox("TOD", checked=True)
        self.cb_iq = QtWidgets.QCheckBox("IQ", checked=False)
        self.cb_fft = QtWidgets.QCheckBox("FFT", checked=False)
        self.cb_ssb = QtWidgets.QCheckBox("Single Sideband PSD", checked=True)
        self.cb_dsb = QtWidgets.QCheckBox("Dual Sideband PSD", checked=False)
        for cb in (self.cb_time, self.cb_iq, self.cb_fft, self.cb_ssb, self.cb_dsb):
            cb.toggled.connect(self._build_layout)

        # Network Analysis button
        self.btn_netanal = QtWidgets.QPushButton("Network Analyzer")
        self.btn_netanal.setToolTip("Open network analysis configuration window.")
        self.btn_netanal.clicked.connect(self._show_netanal_dialog)
        if self.crs is None:
            self.btn_netanal.setEnabled(False)
            self.btn_netanal.setToolTip("CRS object not available - cannot run network analysis.")

        # Help button
        self.btn_help = QtWidgets.QPushButton("Help")
        self.btn_help.setToolTip("Show usage, interaction details, and examples.")
        self.btn_help.clicked.connect(self._show_help)

        # Add widgets to toolbar
        top_h.addWidget(QtWidgets.QLabel("Channels:"))
        top_h.addWidget(self.e_ch)
        top_h.addSpacing(20)
        top_h.addWidget(QtWidgets.QLabel("Buffer:"))
        top_h.addWidget(self.e_buf)
        top_h.addWidget(self.b_pause)
        top_h.addSpacing(30)
        top_h.addWidget(self.cb_real)
        top_h.addSpacing(30)
        for cb in (self.cb_time, self.cb_iq, self.cb_fft, self.cb_ssb, self.cb_dsb):
            top_h.addWidget(cb)
        top_h.addStretch(1)
        top_h.addWidget(self.btn_netanal)
        top_h.addWidget(self.btn_help)

        layout.addWidget(top_bar)

    def _add_config_panel(self, layout):
        """Add the configuration panel to the layout."""
        layout.addWidget(self.btn_toggle_cfg, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.ctrl_panel = QtWidgets.QGroupBox("Configuration")
        self.ctrl_panel.setVisible(False)
        cfg_hbox = QtWidgets.QHBoxLayout(self.ctrl_panel)

        # Show Curves group
        cfg_hbox.addWidget(self._create_show_curves_group())

        # IQ Mode group
        cfg_hbox.addWidget(self._create_iq_mode_group())

        # PSD Mode group
        cfg_hbox.addWidget(self._create_psd_mode_group())

        # General Display group
        cfg_hbox.addWidget(self._create_display_group())

        layout.addWidget(self.ctrl_panel)

    def _create_show_curves_group(self):
        """Create the Show Curves configuration group."""
        show_curves_g = QtWidgets.QGroupBox("Show Curves")
        show_curves_h = QtWidgets.QHBoxLayout(show_curves_g)
        
        self.cb_show_i = QtWidgets.QCheckBox("I", checked=True)
        self.cb_show_q = QtWidgets.QCheckBox("Q", checked=True)
        self.cb_show_m = QtWidgets.QCheckBox("Magnitude", checked=True)
        
        self.cb_show_i.toggled.connect(self._toggle_iqmag)
        self.cb_show_q.toggled.connect(self._toggle_iqmag)
        self.cb_show_m.toggled.connect(self._toggle_iqmag)
        
        show_curves_h.addWidget(self.cb_show_i)
        show_curves_h.addWidget(self.cb_show_q)
        show_curves_h.addWidget(self.cb_show_m)
        
        return show_curves_g

    def _create_iq_mode_group(self):
        """Create the IQ Mode configuration group."""
        iq_g = QtWidgets.QGroupBox("IQ Mode")
        iq_h = QtWidgets.QHBoxLayout(iq_g)
        
        self.rb_density = QtWidgets.QRadioButton("Density", checked=True)
        self.rb_density.setToolTip("2D histogram of I/Q values.")
        self.rb_scatter = QtWidgets.QRadioButton("Scatter")
        self.rb_scatter.setToolTip("Scatter of up to 1,000 I/Q points. CPU intensive.")
        
        rb_group = QtWidgets.QButtonGroup(iq_g)
        rb_group.addButton(self.rb_density)
        rb_group.addButton(self.rb_scatter)
        
        for rb in (self.rb_density, self.rb_scatter):
            rb.toggled.connect(self._build_layout)
            
        iq_h.addWidget(self.rb_density)
        iq_h.addWidget(self.rb_scatter)
        
        return iq_g

    def _create_psd_mode_group(self):
        """Create the PSD Mode configuration group."""
        psd_g = QtWidgets.QGroupBox("PSD Mode")
        psd_grid = QtWidgets.QGridLayout(psd_g)
        
        self.lbl_psd_scale = QtWidgets.QLabel("PSD Scale:")
        self.rb_psd_abs = QtWidgets.QRadioButton("Absolute (dBm)", checked=True)
        self.rb_psd_rel = QtWidgets.QRadioButton("Relative (dBc)")
        
        for rb in (self.rb_psd_abs, self.rb_psd_rel):
            rb.toggled.connect(self._psd_ref_changed)
        
        self.spin_segments = QtWidgets.QSpinBox()
        self.spin_segments.setRange(1, 256)
        self.spin_segments.setValue(1)
        self.spin_segments.setMaximumWidth(80)
        self.spin_segments.setToolTip("Number of segments for Welch PSD averaging.")
        
        psd_grid.addWidget(self.lbl_psd_scale, 0, 0)
        psd_grid.addWidget(self.rb_psd_abs, 0, 1)
        psd_grid.addWidget(self.rb_psd_rel, 0, 2)
        psd_grid.addWidget(QtWidgets.QLabel("Segments:"), 1, 0)
        psd_grid.addWidget(self.spin_segments, 1, 1)
        
        return psd_g

    def _create_display_group(self):
        """Create the General Display configuration group."""
        disp_g = QtWidgets.QGroupBox("General Display")
        disp_h = QtWidgets.QHBoxLayout(disp_g)

        # Add Zoom Box Mode checkbox
        self.cb_zoom_box = QtWidgets.QCheckBox("Zoom Box Mode", checked=True)
        self.cb_zoom_box.setToolTip("When enabled, left-click drag creates a zoom box. When disabled, left-click drag pans.")
        self.cb_zoom_box.toggled.connect(self._toggle_zoom_box_mode)
        disp_h.addWidget(self.cb_zoom_box)

        # Dark Mode Toggle
        self.cb_dark = QtWidgets.QCheckBox("Dark Mode", checked=self.dark_mode)
        self.cb_dark.setToolTip("Switch between dark/light themes.")
        self.cb_dark.toggled.connect(self._toggle_dark_mode)
        disp_h.addWidget(self.cb_dark)

        # Autoscale button
        self.cb_auto_scale = QtWidgets.QCheckBox("Auto Scale", checked=self.auto_scale_plots)
        self.cb_auto_scale.setToolTip("Enable/disable auto-range for IQ/FFT/SSB/DSB. Can improve display performance.")
        self.cb_auto_scale.toggled.connect(self._toggle_auto_scale)
        disp_h.addWidget(self.cb_auto_scale)
        
        return disp_g

    def _add_plot_container(self, layout):
        """Add the plot container to the layout."""
        self.container = QtWidgets.QWidget()
        layout.addWidget(self.container)
        self.grid = QtWidgets.QGridLayout(self.container)

    def _add_status_bar(self):
        """Add the status bar."""
        self.setStatusBar(QtWidgets.QStatusBar())

    def _show_help(self):
        """
        Show a dialog containing usage instructions, interaction details,
        and example commands.
        """
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Periscope Help")
        msg.setTextFormat(QtCore.Qt.TextFormat.MarkdownText)
        help_text = (
            "**Usage:**\n"
            "  - Multi-channel grouping: use '&' to display multiple channels in one row.\n"
            "    e.g., \"3&5\" for channels 3 and 5 in one row, \"3&5,7\" for that row plus a row with channel 7.\n\n"
            "**Standard PyQtGraph Interactions:**\n"
            "  - Pan: Left-click and drag (when Zoom Box Mode is disabled).\n"
            "  - Zoom Box: Left-click and drag to create a selection rectangle (when enabled).\n"
            "  - Zoom: Right-click/drag or mouse-wheel in most plots.\n"
            "  - Axis scaling (log vs lin) and auto-zooming: Individually configurable by right-clicking any plot.\n"
            "  - Double-click within a plot: Show the coordinates of the clicked position.\n"
            "  - Export plot to CSV, Image, Vector Graphics, or interactive Matplotlib window: Right-click -> Export\n"
            "  - See PyQtGraph docs for more.\n\n"
            "**Network Analysis:**\n"
            "  - Click the 'Network Analysis' button to open the configuration dialog.\n"
            "  - Configure parameters like frequency range, amplitude, and number of points.\n"
            "  - Multiple amplitudes: Enter comma-separated values to run analysis at multiple power levels.\n"
            "  - Unit conversion: Configure amplitudes in normalized values or dBm power.\n"
            "  - Cable length: Compensate for phase due to cables by estimating cable lengths.\n"
            "  - Analysis runs in a separate window with real-time updates.\n"
            "  - Use the Unit selector to view data in Counts, Volts, or dBm, or normalized versions to compare relative responses.\n"
            "  - Export data from Network Analysis with the 'Export Data' button. Pickle and CSV exports available.\n\n"
            "**Performance tips for higher FPS:**\n"
            "  - Disable auto-scaling in the configuration panel\n"
            "  - Use the density mode for IQ, smaller buffers, or enable only a subset of I,Q,M.\n"            
            "  - Periscope is limited by the single-thread renderer. Launching multiple instances can improve performance for viewing many channels.\n\n"
            "**Command-Line Examples:**\n"
            "  - `$ periscope rfmux0022.local --module 2 --channels \"3&5,7\"`\n\n"
            "**IPython / Jupyter:** invoke directly from CRS object\n"
            "  - `>>> crs.raise_periscope(module=2, channels=\"3&5\")`\n"
            "  - If in a non-blocking mode, you can still interact with your session concurrently.\n\n"
        )
        msg.setText(help_text)
        msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Close)
        msg.exec()

    def _toggle_zoom_box_mode(self, enable: bool):
        """
        Enable or disable zoom box mode for all plot viewboxes.

        Parameters
        ----------
        enable : bool
            If True, enables rectangle selection zooming. If False, returns to pan mode.
        """
        if not hasattr(self, "plots"):
            return
        self.zoom_box_mode = enable  # Store the state            

        for rowPlots in self.plots:
            for mode, plot_widget in rowPlots.items():
                viewbox = plot_widget.getViewBox()
                if isinstance(viewbox, ClickableViewBox):
                    viewbox.enableZoomBoxMode(enable)
        
        # Also update any network analysis plots if they exist
        for window_id, window_data in self.netanal_windows.items():
            window = window_data.get('window')
            if window:
                for module in window.plots:
                    for plot_type in ['amp_plot', 'phase_plot']:
                        viewbox = window.plots[module][plot_type].getViewBox()
                        if isinstance(viewbox, ClickableViewBox):
                            viewbox.enableZoomBoxMode(enable)

                # Also update the Network Analysis window's zoom box checkbox if it exists
                if hasattr(window, 'zoom_box_cb'):
                    window.zoom_box_cb.setChecked(enable)                        
                        
        # Show help message if enabling
        if enable:
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Zoom Box Mode Enabled")
            msg.setText(
                "Zoom Box Mode is now enabled:\n\n"
                "• Left-click and drag to draw a selection rectangle\n"
                "• Release to zoom to that selection\n"
                "• Right-click to zoom out\n\n"
                "You can toggle this feature off in the configuration panel."
            )
            msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            msg.exec()
        
    def _show_netanal_dialog(self):
        """Show the network analysis configuration dialog."""
        if self.crs is None:
            QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available")
            return
        
        # Create a dialog with default DAC scales
        default_dac_scales = {m: -0.5 for m in range(1, 9)}
        dialog = NetworkAnalysisDialog(self, modules=list(range(1, 9)), dac_scales=default_dac_scales)
        dialog.module_entry.setText(str(self.module))
        
        # Start fetching DAC scales in background
        fetcher = DACScaleFetcher(self.crs)
        fetcher.dac_scales_ready.connect(lambda scales: dialog.dac_scales.update(scales))
        fetcher.dac_scales_ready.connect(dialog._update_dac_scale_info)
        fetcher.dac_scales_ready.connect(dialog._update_dbm_from_normalized)
        
        # Store DAC scales in our main class when they're ready
        fetcher.dac_scales_ready.connect(lambda scales: setattr(self, 'dac_scales', scales))
        
        fetcher.start()
        
        # Show dialog
        if dialog.exec():
            # Store the dialog's DAC scales in our class when dialog is accepted
            self.dac_scales = dialog.dac_scales.copy()
            
            params = dialog.get_parameters()
            if params:
                self._start_network_analysis(params)

    def _start_network_analysis(self, params: dict):
        """Start network analysis on selected modules."""
        try:
            if self.crs is None:
                QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available")
                return
                    
            # Determine which modules to run
            selected_module = params.get('module')
            if selected_module is None:
                modules = list(range(1, 9))  # All modules
            elif isinstance(selected_module, list):
                modules = selected_module
            else:
                modules = [selected_module]

            # Verify DAC scales are available
            if not hasattr(self, 'dac_scales'):
                QtWidgets.QMessageBox.critical(self, "Error", 
                    "DAC scales are not available. Please run the network analysis configuration again.")
                return
            
            # Create a unique window ID
            window_id = f"window_{self.netanal_window_count}"
            self.netanal_window_count += 1
            
            # Create window-specific signal handlers
            window_signals = NetworkAnalysisSignals()
            
            # Use actual DAC scales, no defaults
            dac_scales = self.dac_scales.copy()
            
            # Create a new window with DAC scales
            window = NetworkAnalysisWindow(self, modules, dac_scales)
            window.set_params(params)
            window.window_id = window_id  # Attach ID to window
            
            # Store window in dictionary
            self.netanal_windows[window_id] = {
                'window': window,
                'signals': window_signals,
                'amplitude_queues': {},
                'current_amp_index': {}
            }
            
            # Connect signals for this specific window
            window_signals.progress.connect(
                lambda module, progress: window.update_progress(module, progress))
            window_signals.data_update.connect(
                lambda module, freqs, amps, phases: window.update_data(module, freqs, amps, phases))
            window_signals.data_update_with_amp.connect(
                lambda module, freqs, amps, phases, amplitude: 
                window.update_data_with_amp(module, freqs, amps, phases, amplitude))
            window_signals.completed.connect(
                lambda module: self._handle_analysis_completed(module, window_id))
            window_signals.error.connect(
                lambda error_msg: QtWidgets.QMessageBox.critical(window, "Network Analysis Error", error_msg))
            
            # Get amplitudes from params
            amplitudes = params.get('amps', [params.get('amp', DEFAULT_AMPLITUDE)])
            
            # Set up amplitude queues for this window
            window_data = self.netanal_windows[window_id]
            window_data['amplitude_queues'] = {module: list(amplitudes) for module in modules}
            window_data['current_amp_index'] = {module: 0 for module in modules}
            
            # Update progress displays
            for module in modules:
                window.update_amplitude_progress(module, 1, len(amplitudes), amplitudes[0])
                    
                # Start the first amplitude task
                self._start_next_amplitude_task(module, params, window_id)
            
            # Show the window
            window.show()
        except Exception as e:
            print(f"Error in _start_network_analysis: {e}")
            traceback.print_exc()
    
    def _handle_analysis_completed(self, module: int, window_id: str):
        """Handle completion of a network analysis task for a specific window."""
        try:
            if window_id not in self.netanal_windows:
                return  # Window was closed
                
            window_data = self.netanal_windows[window_id]
            window = window_data['window']
            
            # Update window
            window.complete_analysis(module)
            
            # Clean up tasks
            for task_key in list(self.netanal_tasks.keys()):
                if task_key.startswith(f"{window_id}_{module}_"):
                    self.netanal_tasks.pop(task_key, None)
            
            # Check for more amplitudes
            if module in window_data['amplitude_queues'] and window_data['amplitude_queues'][module]:
                # Update index
                window_data['current_amp_index'][module] += 1
                
                # Update display
                total_amps = len(window.original_params.get('amps', []))
                next_amp = window_data['amplitude_queues'][module][0]
                window.update_amplitude_progress(
                    module, 
                    window_data['current_amp_index'][module] + 1,
                    total_amps,
                    next_amp
                )
                
                # Reset progress bar
                if module in window.progress_bars:
                    window.progress_bars[module].setValue(0)
                    if window.progress_group:
                        window.progress_group.setVisible(True)
                
                # Start next task
                self._start_next_amplitude_task(module, window.original_params, window_id)
        except Exception as e:
            print(f"Error in _handle_analysis_completed: {e}")
            traceback.print_exc()    

    def _start_next_amplitude_task(self, module: int, params: dict, window_id: str):
        """Start the next amplitude task for a module in a specific window."""
        try:
            if window_id not in self.netanal_windows:
                return  # Window was closed
                
            window_data = self.netanal_windows[window_id]
            window = window_data['window']
            signals = window_data['signals']
            
            if module not in window_data['amplitude_queues'] or not window_data['amplitude_queues'][module]:
                return  # No more amplitudes to process
            
            # Get the next amplitude
            amplitude = window_data['amplitude_queues'][module][0]
            window_data['amplitude_queues'][module].pop(0)  # Remove it from the queue
            
            # Create task parameters
            task_params = params.copy()
            task_params['module'] = module
            
            # Create a unique task key
            task_key = f"{window_id}_{module}_amp_{amplitude}"
            
            # Create and start the task
            task = NetworkAnalysisTask(self.crs, module, task_params, signals, amplitude=amplitude)
            self.netanal_tasks[task_key] = task
            self.pool.start(task)
        except Exception as e:
            print(f"Error in _start_next_amplitude_task: {e}")
            traceback.print_exc()
    
    def _rerun_network_analysis(self, params: dict):
        """Rerun network analysis in the window that triggered this call."""
        try:
            if self.crs is None:
                QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available")
                return
            
            # Find which window triggered this
            sender = self.sender()
            source_window = None
            window_id = None
            
            if sender and hasattr(sender, 'window'):
                source_window = sender.window()
            
            # Find matching window in our dictionary
            for w_id, w_data in self.netanal_windows.items():
                if w_data['window'] == source_window:
                    window_id = w_id
                    break
                    
            if not window_id:
                # Could not determine which window called us
                return
                
            window_data = self.netanal_windows[window_id]
            window = window_data['window']
            signals = window_data['signals']
            
            # Update window
            window.data.clear()
            window.raw_data.clear()
            for module, pbar in window.progress_bars.items():
                pbar.setValue(0)
            window.clear_plots()
            window.set_params(params)
            
            # Determine modules
            selected_module = params.get('module')
            if selected_module is None:
                modules = list(range(1, 9))
            elif isinstance(selected_module, list):
                modules = selected_module
            else:
                modules = [selected_module]
            
            # Stop existing tasks for this window
            for task_key in list(self.netanal_tasks.keys()):
                if task_key.startswith(f"{window_id}_"):
                    task = self.netanal_tasks.pop(task_key)
                    task.stop()
            
            # Get amplitudes
            amplitudes = params.get('amps', [params.get('amp', DEFAULT_AMPLITUDE)])
            
            # Reset amplitude queues
            window_data['amplitude_queues'] = {module: list(amplitudes) for module in modules}
            window_data['current_amp_index'] = {module: 0 for module in modules}
            
            # Make progress visible
            if window.progress_group:
                window.progress_group.setVisible(True)
            
            # Start new tasks
            for module in modules:
                window.update_amplitude_progress(module, 1, len(amplitudes), amplitudes[0])
                self._start_next_amplitude_task(module, params, window_id)
        except Exception as e:
            print(f"Error in _rerun_network_analysis: {e}")
            traceback.print_exc()
    
    def _netanal_progress(self, module: int, progress: float):
        """Handle network analysis progress updates."""
        # This function is not used directly, as we route signals to specific windows
        pass
    
    def _netanal_data_update(self, module: int, freqs: np.ndarray, amps: np.ndarray, phases: np.ndarray):
        """Handle network analysis data updates."""
        # This function is not used directly, as we route signals to specific windows
        pass
    
    def _netanal_data_update_with_amp(self, module: int, freqs: np.ndarray, amps: np.ndarray, phases: np.ndarray, amplitude: float):
        """Handle network analysis data updates with amplitude information."""
        # This function is not used directly, as we route signals to specific windows
        pass
    
    def _netanal_completed(self, module: int):
        """Handle network analysis completion."""
        # This function is not used directly, as we route signals to specific windows
        pass
    
    def _netanal_error(self, error_msg: str):
        """Handle network analysis errors."""
        QtWidgets.QMessageBox.critical(self, "Network Analysis Error", error_msg)

    def _toggle_config(self, visible: bool):
        """
        Show or hide the advanced configuration panel.

        Parameters
        ----------
        visible : bool
            True to show, False to hide.
        """
        self.ctrl_panel.setVisible(visible)
        self.btn_toggle_cfg.setText("Hide Configuration" if visible else "Show Configuration")

    def _toggle_auto_scale(self, checked: bool):
        """
        Enable or disable auto-ranging for non-TOD plots.

        Parameters
        ----------
        checked : bool
            If True, IQ/FFT/SSB/DSB will auto-range. If False, they remain fixed.
        """
        self.auto_scale_plots = checked
        if hasattr(self, "plots"):
            for rowPlots in self.plots:
                for mode, pw in rowPlots.items():
                    if mode != "T":
                        pw.enableAutoRange(pg.ViewBox.XYAxes, checked)

    def _toggle_iqmag(self):
        """
        Globally hide or show I/Q/M lines in TOD, FFT, and SSB (DSB unaffected).
        """
        self.show_i = self.cb_show_i.isChecked()
        self.show_q = self.cb_show_q.isChecked()
        self.show_m = self.cb_show_m.isChecked()

        if not hasattr(self, "curves"):
            return
        for rowCurves in self.curves:
            for mode in ("T", "F", "S"):
                if mode in rowCurves:
                    subdict = rowCurves[mode]
                    for ch, cset in subdict.items():
                        if "I" in cset:
                            cset["I"].setVisible(self.show_i)
                        if "Q" in cset:
                            cset["Q"].setVisible(self.show_q)
                        if "Mag" in cset:
                            cset["Mag"].setVisible(self.show_m)

    def _init_buffers(self):
        """
        Recreate ring buffers for all unique channels referenced in self.channel_list.
        """
        unique_chs = set()
        for group in self.channel_list:
            for c in group:
                unique_chs.add(c)
        self.all_chs = sorted(unique_chs)
        self.buf = {}
        self.tbuf = {}
        for ch in self.all_chs:
            self.buf[ch] = {k: Circular(self.N) for k in ("I", "Q", "M")}
            self.tbuf[ch] = Circular(self.N)

    def _build_layout(self):
        """
        Construct the layout grid for each row in self.channel_list and each enabled mode.
        """
        self._clear_current_layout()
        modes = self._get_active_modes()
        self.plots = []
        self.curves = []

        # Create font for labels
        font = QFont()
        font.setPointSize(UI_FONT_SIZE)

        # Define color configurations
        single_colors = self._get_single_channel_colors()
        channel_families = self._get_channel_color_families()

        # Build the layout for each row
        for row_i, group in enumerate(self.channel_list):
            rowPlots, rowCurves = self._create_row_plots_and_curves(row_i, group, modes, font, single_colors, channel_families)
            self.plots.append(rowPlots)
            self.curves.append(rowCurves)

        # Re-enable auto-range after building
        self._restore_auto_range_settings()

        # Apply "Show Curves" toggles
        self._toggle_iqmag()

        # Apply zoom box mode to all plots
        if hasattr(self, "zoom_box_mode"):
            self._toggle_zoom_box_mode(self.zoom_box_mode)

    def _clear_current_layout(self):
        """Clear the current layout."""
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _get_active_modes(self):
        """Get the list of active visualization modes."""
        modes = []
        if self.cb_time.isChecked():
            modes.append("T")
        if self.cb_iq.isChecked():
            modes.append("IQ")
        if self.cb_fft.isChecked():
            modes.append("F")
        if self.cb_ssb.isChecked():
            modes.append("S")
        if self.cb_dsb.isChecked():
            modes.append("D")
        return modes

    def _get_single_channel_colors(self):
        """Get color definitions for single-channel plots."""
        return {
            "I": "#1f77b4",
            "Q": "#ff7f0e",
            "Mag": "#2ca02c",
            "DSB": "#bcbd22",
        }

    def _get_channel_color_families(self):
        """Get color families for multi-channel plots."""
        return [
            ("#1f77b4", "#4a8cc5", "#90bce0"),
            ("#ff7f0e", "#ffa64d", "#ffd2a3"),
            ("#2ca02c", "#63c063", "#a1d9a1"),
            ("#d62728", "#eb6a6b", "#f2aeae"),
            ("#9467bd", "#ae8ecc", "#d3c4e3"),
            ("#8c564b", "#b58e87", "#d9c3bf"),
            ("#e377c2", "#f0a8dc", "#f7d2ee"),
            ("#7f7f7f", "#aaaaaa", "#d3d3d3"),
            ("#bcbd22", "#cfd342", "#e2e795"),
            ("#17becf", "#51d2de", "#9ae8f2"),
        ]

    def _create_row_plots_and_curves(self, row_i, group, modes, font, single_colors, channel_families):
        """Create plots and curves for a single row."""
        rowPlots = {}
        rowCurves = {}

        row_title = "Ch " + ("&".join(map(str, group)) if len(group) > 1 else str(group[0]))

        for col, mode in enumerate(modes):
            vb = ClickableViewBox()
            pw = pg.PlotWidget(viewBox=vb, title=f"{_mode_title(mode)} – {row_title}")

            # Configure auto-range based on mode
            self._configure_plot_auto_range(pw, mode)
            
            # Set up plot labels and axes
            self._configure_plot_axes(pw, mode)

            # Apply theme and grid
            self._apply_plot_theme(pw)
            pw.showGrid(x=True, y=True, alpha=0.3)
            self.grid.addWidget(pw, row_i, col)
            rowPlots[mode] = pw

            # Apply font to axes
            self._configure_plot_fonts(pw, font)

            # Create curves or image items based on mode
            if mode == "IQ":
                rowCurves["IQ"] = self._create_iq_plot_item(pw)
            else:
                # Add legend
                legend = pw.addLegend(offset=(30, 10))
                
                # Create curves based on channel count
                if len(group) == 1:
                    ch = group[0]
                    rowCurves[mode] = self._create_single_channel_curves(pw, mode, ch, single_colors, legend)
                else:
                    rowCurves[mode] = self._create_multi_channel_curves(pw, mode, group, channel_families, legend)
                
                # Make legend entries clickable
                self._make_legend_clickable(legend)

        return rowPlots, rowCurves

    def _configure_plot_auto_range(self, pw, mode):
        """Configure plot auto-range behavior based on mode."""
        if mode == "T":
            pw.enableAutoRange(pg.ViewBox.XYAxes, True)
        else:
            pw.enableAutoRange(pg.ViewBox.XYAxes, self.auto_scale_plots)

    def _configure_plot_axes(self, pw, mode):
        """Configure plot axes based on mode."""
        if mode == "T":
            if not self.real_units:
                pw.setLabel("left", "Amplitude", units="Counts")
            else:
                pw.setLabel("left", "Amplitude", units="V")
        elif mode == "IQ":
            pw.getViewBox().setAspectLocked(True)
            if not self.real_units:
                pw.setLabel("bottom", "I", units="Counts")
                pw.setLabel("left",   "Q", units="Counts")
            else:
                pw.setLabel("bottom", "I", units="V")
                pw.setLabel("left",   "Q", units="V")
        elif mode == "F":
            pw.setLogMode(x=True, y=True)
            pw.setLabel("bottom", "Freq", units="Hz")
            if not self.real_units:
                pw.setLabel("left", "Amplitude", units="Counts")
            else:
                pw.setLabel("left", "Amplitude", units="V")
        elif mode == "S":
            pw.setLogMode(x=True, y=not self.real_units)
            pw.setLabel("bottom", "Freq", units="Hz")
            if not self.real_units:
                pw.setLabel("left", "PSD (Counts²/Hz)")
            else:
                lbl = "dBm/Hz" if self.psd_absolute else "dBc/Hz"
                pw.setLabel("left", f"PSD ({lbl})")
        else:  # "D"
            pw.setLogMode(x=False, y=not self.real_units)
            pw.setLabel("bottom", "Freq", units="Hz")
            if not self.real_units:
                pw.setLabel("left", "PSD (Counts²/Hz)")
            else:
                lbl = "dBm/Hz" if self.psd_absolute else "dBc/Hz"
                pw.setLabel("left", f"PSD ({lbl})")

    def _configure_plot_fonts(self, pw, font):
        """Configure plot font settings."""
        pi = pw.getPlotItem()
        for axis_name in ("left", "bottom", "right", "top"):
            axis = pi.getAxis(axis_name)
            if axis:
                axis.setTickFont(font)
                if axis.label:
                    axis.label.setFont(font)
        pi.titleLabel.setFont(font)

    def _create_iq_plot_item(self, pw):
        """Create an IQ plot item based on selected mode."""
        if self.rb_scatter.isChecked():
            sp = pg.ScatterPlotItem(pen=None, size=SCATTER_SIZE)
            pw.addItem(sp)
            return {"mode": "scatter", "item": sp}
        else:
            img = pg.ImageItem(axisOrder="row-major")
            img.setLookupTable(self.lut)
            pw.addItem(img)
            return {"mode": "density", "item": img}

    def _create_single_channel_curves(self, pw, mode, ch, single_colors, legend):
        """Create curves for a single-channel plot."""
        if mode == "T":
            cI = pw.plot(pen=pg.mkPen(single_colors["I"],   width=LINE_WIDTH), name="I")
            cQ = pw.plot(pen=pg.mkPen(single_colors["Q"],   width=LINE_WIDTH), name="Q")
            cM = pw.plot(pen=pg.mkPen(single_colors["Mag"], width=LINE_WIDTH), name="Mag")
            self._fade_hidden_entries(legend, ("I", "Q"))
            return {ch: {"I": cI, "Q": cQ, "Mag": cM}}
        elif mode == "F":
            cI = pw.plot(pen=pg.mkPen(single_colors["I"],   width=LINE_WIDTH), name="I")
            cI.setFftMode(True)
            cQ = pw.plot(pen=pg.mkPen(single_colors["Q"],   width=LINE_WIDTH), name="Q")
            cQ.setFftMode(True)
            cM = pw.plot(pen=pg.mkPen(single_colors["Mag"], width=LINE_WIDTH), name="Mag")
            cM.setFftMode(True)
            self._fade_hidden_entries(legend, ("I", "Q"))
            return {ch: {"I": cI, "Q": cQ, "Mag": cM}}
        elif mode == "S":
            cI = pw.plot(pen=pg.mkPen(single_colors["I"],   width=LINE_WIDTH), name="I")
            cQ = pw.plot(pen=pg.mkPen(single_colors["Q"],   width=LINE_WIDTH), name="Q")
            cM = pw.plot(pen=pg.mkPen(single_colors["Mag"], width=LINE_WIDTH), name="Mag")
            return {ch: {"I": cI, "Q": cQ, "Mag": cM}}
        else:  # "D"
            cD = pw.plot(pen=pg.mkPen(single_colors["DSB"], width=LINE_WIDTH), name="Complex DSB")
            return {ch: {"Cmplx": cD}}

    def _create_multi_channel_curves(self, pw, mode, group, channel_families, legend):
        """Create curves for a multi-channel plot."""
        mode_dict = {}
        for i, ch in enumerate(group):
            (colI, colQ, colM) = channel_families[i % len(channel_families)]
            
            if mode == "T":
                cI = pw.plot(pen=pg.mkPen(colI, width=LINE_WIDTH), name=f"ch{ch}-I")
                cQ = pw.plot(pen=pg.mkPen(colQ, width=LINE_WIDTH), name=f"ch{ch}-Q")
                cM = pw.plot(pen=pg.mkPen(colM, width=LINE_WIDTH), name=f"ch{ch}-Mag")
                mode_dict[ch] = {"I": cI, "Q": cQ, "Mag": cM}
            elif mode == "F":
                cI = pw.plot(pen=pg.mkPen(colI, width=LINE_WIDTH), name=f"ch{ch}-I")
                cI.setFftMode(True)
                cQ = pw.plot(pen=pg.mkPen(colQ, width=LINE_WIDTH), name=f"ch{ch}-Q")
                cQ.setFftMode(True)
                cM = pw.plot(pen=pg.mkPen(colM, width=LINE_WIDTH), name=f"ch{ch}-Mag")
                cM.setFftMode(True)
                mode_dict[ch] = {"I": cI, "Q": cQ, "Mag": cM}
            elif mode == "S":
                cI = pw.plot(pen=pg.mkPen(colI, width=LINE_WIDTH), name=f"ch{ch}-I")
                cQ = pw.plot(pen=pg.mkPen(colQ, width=LINE_WIDTH), name=f"ch{ch}-Q")
                cM = pw.plot(pen=pg.mkPen(colM, width=LINE_WIDTH), name=f"ch{ch}-Mag")
                mode_dict[ch] = {"I": cI, "Q": cQ, "Mag": cM}
            else:  # "D"
                cD = pw.plot(pen=pg.mkPen(colI, width=LINE_WIDTH), name=f"ch{ch}-DSB")
                mode_dict[ch] = {"Cmplx": cD}
                
        return mode_dict

    def _restore_auto_range_settings(self):
        """Restore auto-range settings after building layout."""
        self.auto_scale_plots = True
        self.cb_auto_scale.setChecked(True)
        for rowPlots in self.plots:
            for mode, pw in rowPlots.items():
                if mode != "T":
                    pw.enableAutoRange(pg.ViewBox.XYAxes, True)

    def _apply_plot_theme(self, pw: pg.PlotWidget):
        """
        Configure the plot widget's background and axis color
        based on self.dark_mode.

        Parameters
        ----------
        pw : pg.PlotWidget
            The plot widget to style.
        """
        if self.dark_mode:
            pw.setBackground("k")
            for axis_name in ("left", "bottom", "right", "top"):
                ax = pw.getPlotItem().getAxis(axis_name)
                if ax:
                    ax.setPen("w")
                    ax.setTextPen("w")
        else:
            pw.setBackground("w")
            for axis_name in ("left", "bottom", "right", "top"):
                ax = pw.getPlotItem().getAxis(axis_name)
                if ax:
                    ax.setPen("k")
                    ax.setTextPen("k")

    @staticmethod
    def _fade_hidden_entries(legend, hide_labels):
        """
        Fade out (gray) specific legend entries to indicate
        they are typically less interesting (like I and Q).
        """
        for sample, label in legend.items:
            txt = label.labelItem.toPlainText() if hasattr(label, "labelItem") else ""
            if txt in hide_labels:
                sample.setOpacity(0.3)
                label.setOpacity(0.3)

    @staticmethod
    def _make_legend_clickable(legend):
        """
        Make each legend entry clickable to toggle the associated curve's visibility.

        Parameters
        ----------
        legend : pg.LegendItem
            The legend container with (sample, label) items.
        """
        for sample, label in legend.items:
            curve = sample.item

            def toggle(evt, c=curve, s=sample, l=label):
                vis = not c.isVisible()
                c.setVisible(vis)
                op = 1.0 if vis else 0.3
                s.setOpacity(op)
                l.setOpacity(op)

            label.mousePressEvent = toggle
            sample.mousePressEvent = toggle

    def _toggle_dark_mode(self, checked: bool):
        """
        Switch the entire UI between dark or light color schemes
        and rebuild the plot layout.

        Parameters
        ----------
        checked : bool
            True if dark mode, False for light mode.
        """
        self.dark_mode = checked
        self._build_layout()

    def _toggle_real_units(self, checked: bool):
        """
        Toggle between raw counts and real-units (volts/dBm).

        Parameters
        ----------
        checked : bool
            True if real units, False if raw counts.
        """
        self.real_units = checked
        if checked:
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Real Units On")
            msg.setText(
                "Global conversion to real units (V, dBm) is approximate.\n"
                "All PSD plots are droop-corrected for the CIC1 and CIC2 decimation filters; the 'Raw FFT' is calculated from the raw TOD and not droop-corrected."
            )
            msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            msg.exec()
        self._build_layout()

    def _psd_ref_changed(self):
        """
        Switch between absolute (dBm) and relative (dBc) scaling for PSD plots
        when Real Units is enabled.
        """
        self.psd_absolute = self.rb_psd_abs.isChecked()
        self._build_layout()

    def _update_channels(self):
        """
        Parse the channel specification string from self.e_ch, supporting '&'
        to group multiple channels in one row. Re-init buffers/layout if changed.
        """
        new_parsed = _parse_channels_multich(self.e_ch.text())
        if new_parsed != self.channel_list:
            self.channel_list = new_parsed
            self.iq_workers.clear()
            self.psd_workers.clear()
            for row_i, group in enumerate(self.channel_list):
                self.psd_workers[row_i] = {"S": {}, "D": {}}
                for c in group:
                    self.psd_workers[row_i]["S"][c] = False
                    self.psd_workers[row_i]["D"][c] = False
            self._init_buffers()
            self._build_layout()

    def _change_buffer(self):
        """
        Update ring buffer size from self.e_buf if it differs from the current size,
        then re-init buffers.
        """
        try:
            n = int(self.e_buf.text())
        except ValueError:
            return
        if n != self.N:
            self.N = n
            self._init_buffers()

    def _toggle_pause(self):
        """
        Pause or resume real-time data updates. When paused, new packets
        are discarded to avoid stale accumulation.
        """
        self.paused = not self.paused
        self.b_pause.setText("Resume" if self.paused else "Pause")

    # ───────────────────────── Main GUI Update ─────────────────────────
    def _update_gui(self):
        """
        Periodic GUI update method (via QTimer):
          - Reads UDP packets into ring buffers
          - Updates decimation stage
          - Spawns background tasks (IQ, PSD)
          - Updates displayed lines/images
          - Logs FPS and PPS to status bar
        """
        if self.paused:
            self._discard_packets()
            return

        # Collect new packets and update buffers
        self._process_incoming_packets()

        # Update frame counter and time tracking
        self.frame_cnt += 1
        now = time.time()

        # Recompute dec_stage once per second
        if (now - self.last_dec_update) > 1.0:
            self._update_dec_stage()
            self.last_dec_update = now

        # Update plot data for each row
        self._update_plot_data()

        # Update FPS / PPS display
        self._update_performance_stats(now)

    def _discard_packets(self):
        """Discard all pending packets while paused."""
        while not self.receiver.queue.empty():
            self.receiver.queue.get()

    def _process_incoming_packets(self):
        """Process incoming data packets and update buffers."""
        while not self.receiver.queue.empty():
            pkt = self.receiver.queue.get()
            self.pkt_cnt += 1

            # Calculate relative timestamp
            t_rel = self._calculate_relative_timestamp(pkt)

            # Update ring buffers for all channels
            self._update_buffers(pkt, t_rel)

    def _calculate_relative_timestamp(self, pkt):
        """Calculate relative timestamp from packet timestamp."""
        ts = pkt.ts
        if ts.recent:
            ts.ss += int(0.02 * streamer.SS_PER_SECOND)
            ts.renormalize()
            t_now = ts.h * 3600 + ts.m * 60 + ts.s + ts.ss / streamer.SS_PER_SECOND
            if self.start_time is None:
                self.start_time = t_now
            t_rel = t_now - self.start_time
        else:
            t_rel = None
        return t_rel

    def _update_buffers(self, pkt, t_rel):
        """Update ring buffers with packet data."""
        for ch in self.all_chs:
            Ival = pkt.s[2 * (ch - 1)] / 256.0
            Qval = pkt.s[2 * (ch - 1) + 1] / 256.0
            self.buf[ch]["I"].add(Ival)
            self.buf[ch]["Q"].add(Qval)
            self.buf[ch]["M"].add(math.hypot(Ival, Qval))
            self.tbuf[ch].add(t_rel)

    def _update_plot_data(self):
        """Update plot data for all rows and channels."""
        for row_i, group in enumerate(self.channel_list):
            rowCurves = self.curves[row_i]
            
            # Update time-domain and FFT data for each channel
            for ch in group:
                self._update_channel_plot_data(ch, rowCurves)
            
            # Dispatch IQ computation tasks
            if "IQ" in rowCurves and not self.iq_workers.get(row_i, False):
                self._dispatch_iq_task(row_i, group, rowCurves)
            
            # Dispatch PSD computation tasks
            self._dispatch_psd_tasks(row_i, group)

    def _update_channel_plot_data(self, ch, rowCurves):
        """Update plot data for a specific channel."""
        # Grab ring buffer data
        rawI = self.buf[ch]["I"].data()
        rawQ = self.buf[ch]["Q"].data()
        rawM = self.buf[ch]["M"].data()
        tarr = self.tbuf[ch].data()

        # Apply unit conversion if enabled
        if self.real_units:
            I = convert_roc_to_volts(rawI)
            Q = convert_roc_to_volts(rawQ)
            M = convert_roc_to_volts(rawM)
        else:
            I, Q, M = rawI, rawQ, rawM

        # Update time-domain plots
        if "T" in rowCurves and ch in rowCurves["T"]:
            cset = rowCurves["T"][ch]
            if cset["I"].isVisible():
                cset["I"].setData(tarr, I)
            if cset["Q"].isVisible():
                cset["Q"].setData(tarr, Q)
            if cset["Mag"].isVisible():
                cset["Mag"].setData(tarr, M)

        # Update FFT plots
        if "F" in rowCurves and ch in rowCurves["F"]:
            cset = rowCurves["F"][ch]
            if cset["I"].isVisible():
                cset["I"].setData(tarr, I, fftMode=True)
            if cset["Q"].isVisible():
                cset["Q"].setData(tarr, Q, fftMode=True)
            if cset["Mag"].isVisible():
                cset["Mag"].setData(tarr, M, fftMode=True)

    def _dispatch_iq_task(self, row_i, group, rowCurves):
        """Dispatch an IQ computation task for a row."""
        mode = rowCurves["IQ"]["mode"]
        
        if len(group) == 1:
            # Single channel case
            c = group[0]
            rawI = self.buf[c]["I"].data()
            rawQ = self.buf[c]["Q"].data()
            self.iq_workers[row_i] = True
            task = IQTask(row_i, c, rawI, rawQ, self.dot_px, mode, self.iq_signals)
            self.pool.start(task)
        else:
            # Multi-channel case - combine data
            concatI = np.concatenate([self.buf[ch]["I"].data() for ch in group])
            concatQ = np.concatenate([self.buf[ch]["Q"].data() for ch in group])
            
            # Limit data size to avoid excessive processing
            big_size = concatI.size
            if big_size > 50000:
                stride = max(1, big_size // 50000)
                concatI = concatI[::stride]
                concatQ = concatQ[::stride]
            
            if concatI.size > 1:
                self.iq_workers[row_i] = True
                task = IQTask(row_i, 0, concatI, concatQ, self.dot_px, mode, self.iq_signals)
                self.pool.start(task)

    def _dispatch_psd_tasks(self, row_i, group):
        """Dispatch PSD computation tasks for a row."""
        # Single-sideband PSD tasks
        if "S" in self.curves[row_i]:
            for ch in group:
                if not self.psd_workers[row_i]["S"][ch]:
                    rawI = self.buf[ch]["I"].data()
                    rawQ = self.buf[ch]["Q"].data()
                    
                    # Apply voltage conversion if real units are enabled
                    if self.real_units:
                        I = convert_roc_to_volts(rawI)
                        Q = convert_roc_to_volts(rawQ)
                    else:
                        I, Q = rawI, rawQ
                    
                    self.psd_workers[row_i]["S"][ch] = True
                    task = PSDTask(
                        row=row_i,
                        ch=ch,
                        I=I,  # Use the converted values
                        Q=Q,  # Use the converted values
                        mode="SSB",
                        dec_stage=self.dec_stage,
                        real_units=self.real_units,
                        psd_absolute=self.psd_absolute,
                        segments=self.spin_segments.value(),
                        signals=self.psd_signals,
                    )
                    self.pool.start(task)

        # Dual-sideband PSD tasks
        if "D" in self.curves[row_i]:
            for ch in group:
                if not self.psd_workers[row_i]["D"][ch]:
                    rawI = self.buf[ch]["I"].data()
                    rawQ = self.buf[ch]["Q"].data()
                    
                    # Apply voltage conversion if real units are enabled
                    if self.real_units:
                        I = convert_roc_to_volts(rawI)
                        Q = convert_roc_to_volts(rawQ)
                    else:
                        I, Q = rawI, rawQ
                    
                    self.psd_workers[row_i]["D"][ch] = True
                    task = PSDTask(
                        row=row_i,
                        ch=ch,
                        I=I,  # Use the converted values
                        Q=Q,  # Use the converted values
                        mode="DSB",
                        dec_stage=self.dec_stage,
                        real_units=self.real_units,
                        psd_absolute=self.psd_absolute,
                        segments=self.spin_segments.value(),
                        signals=self.psd_signals,
                    )
                    self.pool.start(task)

    def _update_performance_stats(self, now):
        """Update FPS and packets-per-second display."""
        if (now - self.t_last) >= 1.0:
            fps = self.frame_cnt / (now - self.t_last)
            pps = self.pkt_cnt / (now - self.t_last)
            self.statusBar().showMessage(f"FPS {fps:.1f} | Packets/s {pps:.1f}")
            self.frame_cnt = 0
            self.pkt_cnt = 0
            self.t_last = now

    def _update_dec_stage(self):
        """
        Update the global decimation stage by measuring the sample rate
        from the first row's first channel.
        """
        if not self.channel_list:
            return
        first_group = self.channel_list[0]
        if not first_group:
            return
        ch = first_group[0]
        tarr = self.tbuf[ch].data()
        if len(tarr) < 2:
            return
        dt = (tarr[-1] - tarr[0]) / max(1, (len(tarr) - 1))
        fs = 1.0 / dt if dt > 0 else 1.0
        self.dec_stage = infer_dec_stage(fs)

    # ───────────────────────── IQ & PSD Slots ─────────────────────────
    @QtCore.pyqtSlot(int, str, object)
    def _iq_done(self, row: int, task_mode: str, payload):
        """
        Slot called when an off-thread IQTask finishes.

        Parameters
        ----------
        row : int
            Row index within self.channel_list.
        task_mode : {"density", "scatter"}
            The IQ plot mode for which data was computed.
        payload : object
            The result data. For density: (hist, (Imin,Imax,Qmin,Qmax)).
            For scatter: (xs, ys, colors).
        """
        self.iq_workers[row] = False
        if row >= len(self.curves) or "IQ" not in self.curves[row]:
            return
        pane = self.curves[row]["IQ"]
        if pane["mode"] != task_mode:
            return

        item = pane["item"]
        if task_mode == "density":
            self._update_density_image(item, payload)
        else:  # scatter
            self._update_scatter_plot(item, payload)

    def _update_density_image(self, item, payload):
        """Update a density image with new data."""
        hist, (Imin, Imax, Qmin, Qmax) = payload
        if self.real_units:
            Imin, Imax = convert_roc_to_volts(np.array([Imin, Imax], dtype=float))
            Qmin, Qmax = convert_roc_to_volts(np.array([Qmin, Qmax], dtype=float))
        item.setImage(hist, levels=(0, 255), autoLevels=False)
        item.setRect(
            QtCore.QRectF(
                float(Imin),
                float(Qmin),
                float(Imax - Imin),
                float(Qmax - Qmin),
            )
        )

    def _update_scatter_plot(self, item, payload):
        """Update a scatter plot with new data."""
        xs, ys, colors = payload
        if self.real_units:
            xs = convert_roc_to_volts(xs)
            ys = convert_roc_to_volts(ys)
        item.setData(xs, ys, brush=colors, pen=None, size=SCATTER_SIZE)

    @QtCore.pyqtSlot(int, str, int, object)
    def _psd_done(self, row: int, psd_mode: str, ch: int, payload):
        """
        Slot called when an off-thread PSDTask finishes for a particular channel.

        Parameters
        ----------
        row : int
            Row index in self.channel_list.
        psd_mode : {"SSB", "DSB"}
            The PSD mode that was computed.
        ch : int
            Which channel (within that row) was computed.
        payload : object
            For "SSB": (freq_i, psd_i, psd_q, psd_m, freq_m, psd_m, dec_stage_float).
            For "DSB": (freq_dsb, psd_dsb).
        """
        if row not in self.psd_workers:
            return
        key = psd_mode[0]  # 'S' or 'D'
        if key not in self.psd_workers[row]:
            return
        if ch not in self.psd_workers[row][key]:
            return

        self.psd_workers[row][key][ch] = False
        if row >= len(self.curves):
            return

        if psd_mode == "SSB":
            self._update_ssb_curves(row, ch, payload)
        else:  # DSB
            self._update_dsb_curve(row, ch, payload)

    def _update_ssb_curves(self, row, ch, payload):
        """Update single-sideband PSD curves."""
        if "S" not in self.curves[row]:
            return
        sdict = self.curves[row]["S"]
        if ch not in sdict:
            return
        
        freq_i, psd_i, psd_q, psd_m, _, _, _ = payload
        
        if sdict[ch]["I"].isVisible():
            sdict[ch]["I"].setData(freq_i, psd_i)
        if sdict[ch]["Q"].isVisible():
            sdict[ch]["Q"].setData(freq_i, psd_q)
        if sdict[ch]["Mag"].isVisible():
            sdict[ch]["Mag"].setData(freq_i, psd_m)

    def _update_dsb_curve(self, row, ch, payload):
        """Update dual-sideband PSD curve."""
        if "D" not in self.curves[row]:
            return
        ddict = self.curves[row]["D"]
        if ch not in ddict:
            return
        
        freq_dsb, psd_dsb = payload
        
        if ddict[ch]["Cmplx"].isVisible():
            ddict[ch]["Cmplx"].setData(freq_dsb, psd_dsb)

    def closeEvent(self, event):
        """
        Cleanly shut down the background receiver and stop the timer before closing.

        Parameters
        ----------
        event : QCloseEvent
            The close event instance.
        """
        self.timer.stop()
        self.receiver.stop()
        self.receiver.wait()
        
        # Stop any running network analysis tasks
        for task_key in list(self.netanal_tasks.keys()):
            task = self.netanal_tasks[task_key]
            task.stop()
            self.netanal_tasks.pop(task_key, None)
        
        super().closeEvent(event)
        event.accept()


def main():
    """
    Entry point for command-line usage. Supports multi-channel grouping via '&',
    auto-scaling, and global I/Q/M toggles. Launches Periscope in blocking mode.
    """
    ap = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent("""\
        Periscope — real-time CRS packet visualizer with network analysis.

        Connects to a UDP/multicast stream, filters a single module, and drives a
        PyQt6 GUI with up to six linked views per channel group:

          • Time-domain waveform (TOD)
          • IQ plane (density or scatter)
          • Raw FFT
          • Single-sideband PSD (SSB)  – CIC-corrected
          • Dual-sideband PSD (DSB)    – CIC-corrected
          • Network Analysis — amplitude and phase vs frequency

        Key options
        -----------
          • Comma-separated channel list with '&' to overlay channels on one row,
            e.g. "3&5,7" → two rows: {3,5} and {7}
          • -n / --num-samples   Ring-buffer length per channel (history / FFT depth)
          • -f / --fps           Maximum GUI refresh rate (frames s⁻¹) [typically system-limited anyway]
          • -d / --density-dot   Dot size in IQ-density mode (pixels) [not typically adjusted]
          • --enable-netanal     Create CRS object to enable network analysis

        Advanced features
        -----------------
          • Real-unit conversion (counts → V, dBm/Hz, dBc/Hz) with CIC droop compensation in the PSDs
          • Welch segmentation to average PSD noise floor
          • Network analysis with real-time amplitude and phase measurements

        Example
        -------
          $ periscope rfmux0022.local --module 2 --channels "3&5,7" --enable-netanal

        Run with -h / --help for the full option list.
    """))

    ap.add_argument("hostname")
    ap.add_argument("-m", "--module", type=int, default=1)
    ap.add_argument("-c", "--channels", default="1")
    ap.add_argument("-n", "--num-samples", type=int, default=DEFAULT_BUFFER_SIZE)
    ap.add_argument("-f", "--fps", type=float, default=30.0)
    ap.add_argument("-d", "--density-dot", type=int, default=DENSITY_DOT_SIZE)
    args = ap.parse_args()

    if args.fps <= 0:
        ap.error("FPS must be positive.")
    if args.fps > 30:
        warnings.warn("FPS>30 might be unnecessary", RuntimeWarning)
    if args.density_dot < 1:
        ap.error("Density-dot size must be ≥1 pixel.")

    refresh_ms = int(round(1000.0 / args.fps))
    app = QtWidgets.QApplication(sys.argv[:1])

    # Bind only the main GUI (this) thread to a single CPU to reduce scheduling jitter
    _pin_current_thread_to_core()

    # Create CRS object so we can tell it what to do
    crs = None
    try:
        # Extract serial number from hostname
        hostname = args.hostname
        if "rfmux" in hostname and ".local" in hostname:
            serial = hostname.replace("rfmux", "").replace(".local", "")
        else:
            # Default to hostname as serial
            serial = hostname
        
        # Create and resolve CRS in a synchronous way
        s = load_session(f'!HardwareMap [ !CRS {{ serial: "{serial}" }} ]')
        crs = s.query(CRS).one()
        
        # Resolve the CRS object
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(crs.resolve())
        
    except Exception as e:
        warnings.warn(f"Failed to create CRS object: {str(e)}\nNetwork analysis will be disabled.")
        crs = None

    win = Periscope(
        host=args.hostname,
        module=args.module,
        chan_str=args.channels,
        buf_size=args.num_samples,
        refresh_ms=refresh_ms,
        dot_px=args.density_dot,
        crs=crs  # Pass the CRS object
    )
    win.show()
    sys.exit(app.exec())

@macro(CRS, register=True)
async def raise_periscope(
    crs: CRS,
    *,
    module: int = 1,
    channels: str = "1",
    buf_size: int = DEFAULT_BUFFER_SIZE,
    fps: float = 30.0,
    density_dot: int = DENSITY_DOT_SIZE,
    blocking: bool | None = None,
):
    """
    Programmatic entry point for embedding or interactive usage of Periscope.

    This function either blocks (runs the Qt event loop) or returns control
    immediately, depending on whether a Qt event loop is already running or the
    'blocking' parameter is explicitly set.

    Parameters
    ----------
    crs : CRS object
        CRS object for hardware communication (needed for network analysis).
    module : int, optional
        Module number to filter data (default is 1).
    channels : str, optional
        A comma-separated list of channel indices, possibly with '&' to group
        multiple channels on one row (default "1").
    buf_size : int, optional
        Ring buffer size for each channel (default 5000).
    fps : float, optional
        Frames per second (default 30.0). Determines GUI update rate.
    density_dot : int, optional
        Dot dilation in pixels for IQ density mode (default 1).
    blocking : bool or None, optional
        If True, runs the Qt event loop until exit. If False, returns control
        immediately. If None, infers from the environment.

    Returns
    -------
    Periscope or (Periscope, QApplication)
        The Periscope instance, or (instance, QApplication) if non-blocking.
    """
    ip = _get_ipython()
    qt_loop = _is_qt_event_loop_running()

    if ip and not qt_loop:
        ip.run_line_magic("gui", "qt")
        qt_loop = True

    if blocking is None:
        blocking = not qt_loop

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])

    # Bind only the main GUI (this) thread to a single CPU to reduce scheduling jitter
    _pin_current_thread_to_core()

    refresh_ms = int(round(1000.0 / fps))

    viewer = Periscope(
        host=crs.tuber_hostname,
        module=module,
        chan_str=channels,
        buf_size=buf_size,
        refresh_ms=refresh_ms,
        dot_px=density_dot,
        crs=crs,  # Pass the CRS object for network analysis
    )
    viewer.show()

    if blocking:
        if _is_running_inside_ipython():
            app.exec()
        else:
            sys.exit(app.exec())
        return viewer
    else:
        return viewer, app


if __name__ == "__main__":
    main()