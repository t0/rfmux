#!/usr/bin/env -S uv run
"""
ARCHITECTURAL SUMMARY
---------------------
This module implements the Periscope real-time multi-pane viewer using PyQt6. It
visualizes data from a CRS data streamer in multiple ways:
    - Time-domain (TOD)
    - IQ (density or scatter)
    - FFT (using pyqtgraph’s fftMode=True)
    - Single-Sideband PSD (SSB)
    - Dual-Sideband PSD (DSB)

Key Components & Concurrency:
    - A ring buffer (Circular) for each channel stores incoming data.
    - A UDPReceiver runs in its own QThread to receive streaming data asynchronously.
    - Separate IQTask and PSDTask workers offload expensive computations to a QThreadPool.
    - The main Periscope class orchestrates the UI, buffer management, and dispatch of tasks.

Performance Notes:
    - Real-time updates rely on a QTimer to periodically pull new data from a thread-safe queue.
    - The code can optionally use SciPy for faster density/histogram manipulations.

--------------------------------------------------------------------------------
Periscope – Real‑Time Multi‑Pane Viewer
=======================================

Features:
- Time‑domain (TOD)
- IQ (density or scatter)
- FFT (pyqtgraph’s fftMode=True)
- Single‑Sideband PSD (SSB)
- Dual‑Sideband PSD (DSB)
"""

import argparse
import math
import os
import queue
import socket
import sys
import time
import warnings
from typing import Dict, List

import numpy as np

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QIntValidator
import pyqtgraph as pg

# Local imports
from .. import streamer
from ..core.transferfunctions import (
    spectrum_from_slow_tod,
    convert_roc_to_volts,
)

# ───────────────────────── Global Settings ─────────────────────────
pg.setConfigOptions(useOpenGL=False, antialias=False)

LINE_WIDTH       = 1.5
UI_FONT_SIZE     = 12
DENSITY_GRID     = 512
DENSITY_DOT_SIZE = 1
SMOOTH_SIGMA     = 1.3
LOG_COMPRESS     = True

SCATTER_POINTS   = 1_000
SCATTER_SIZE     = 5

try:
    from scipy.ndimage import gaussian_filter, convolve
except ImportError:  # SciPy not installed – graceful degradation
    gaussian_filter = None
    convolve = None
    SMOOTH_SIGMA = 0.0

BASE_SAMPLING = 625e6 / 256.0 / 64.0  # ≈38 147.46 Hz base for dec=0


# ───────────────────────── Utility helpers ─────────────────────────
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


class ClickableViewBox(pg.ViewBox):
    """
    A custom ViewBox that opens a coordinate readout dialog when double-clicked.
    """

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
            # Edge case: not enough data
            if self.mode == "density":
                empty = np.zeros((DENSITY_GRID, DENSITY_GRID), np.uint8)
                payload = (empty, (0, 1, 0, 1))
            else:
                payload = ([], [], [])
            self.signals.done.emit(self.row, self.mode, payload)
            return

        if self.mode == "density":
            g = DENSITY_GRID
            hist = np.zeros((g, g), np.uint32)

            Imin, Imax = self.I.min(), self.I.max()
            Qmin, Qmax = self.Q.min(), self.Q.max()
            if Imin == Imax or Qmin == Qmax:
                payload = (hist.astype(np.uint8), (Imin, Imax, Qmin, Qmax))
                self.signals.done.emit(self.row, self.mode, payload)
                return

            # Map I/Q data to pixel indices
            ix = ((self.I - Imin) * (g - 1) / (Imax - Imin)).astype(np.intp)
            qy = ((self.Q - Qmin) * (g - 1) / (Qmax - Qmin)).astype(np.intp)

            # Base histogram
            np.add.at(hist, (qy, ix), 1)

            # Optional dot dilation
            if self.dot_px > 1:
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

            # Optional smoothing & log-compression
            if gaussian_filter is not None and SMOOTH_SIGMA > 0:
                hist = gaussian_filter(hist.astype(np.float32), SMOOTH_SIGMA, mode="nearest")
            if LOG_COMPRESS:
                hist = np.log1p(hist, out=hist.astype(np.float32))

            # 8-bit normalization
            if hist.max() > 0:
                hist = (hist * (255.0 / hist.max())).astype(np.uint8)

            payload = (hist, (Imin, Imax, Qmin, Qmax))

        else:  # "scatter" mode
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
            payload = (xs, ys, colors)

        self.signals.done.emit(self.row, self.mode, payload)


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
            # Not enough data
            if self.mode == "SSB":
                payload = ([], [], [], [], [], [], 0.0)
            else:
                payload = ([], [])
            self.signals.done.emit(self.row, self.mode, self.ch, payload)
            return

        ref = (
            "counts"
            if not self.real_units
            else ("absolute" if self.psd_absolute else "relative")
        )
        nper = max(1, data_len // max(1, self.segments))

        spec_iq = spectrum_from_slow_tod(
            i_data=self.I,
            q_data=self.Q,
            dec_stage=self.dec_stage,
            scaling="psd",
            reference=ref,
            nperseg=nper,
            spectrum_cutoff=0.9,
        )

        if self.mode == "SSB":
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
            payload = (
                spec_iq["freq_iq"],
                spec_iq["psd_i"],
                spec_iq["psd_q"],
                spec_m["psd_i"],
                spec_m["freq_iq"],
                spec_m["psd_i"],
                float(self.dec_stage),
            )
        else:  # "DSB"
            freq_dsb = spec_iq["freq_dsb"]
            psd_dsb = spec_iq["psd_dual_sideband"]
            order = np.argsort(freq_dsb)
            payload = (freq_dsb[order], psd_dsb[order])

        self.signals.done.emit(self.row, self.mode, self.ch, payload)


# ───────────────────────── Multi-Channel Parsing ─────────────────────────
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
        One of {"T", "IQ", "F", "S", "D"}.

    Returns
    -------
    str
        A human-readable title segment for that mode.
    """
    if mode == "T":
        return "Time"
    if mode == "IQ":
        return "IQ"
    if mode == "F":
        return "Raw FFT"
    if mode == "S":
        return "SSB PSD"
    if mode == "D":
        return "DSB PSD"
    return mode


class Periscope(QtWidgets.QMainWindow):
    """
    Multi‑pane PyQt application for real-time data visualization of:
      - Time-domain waveforms (TOD)
      - IQ (density or scatter)
      - FFT
      - Single-Sideband PSD (SSB)
      - Dual-Sideband PSD (DSB)

    Additional Features
    -------------------
    - Multi-channel grouping via '&'.
    - An “Auto Scale” checkbox for IQ/FFT/SSB/DSB (not TOD).
    - Global toggles to hide/show I, Q, and Mag lines for TOD, FFT, and SSB.
    - A "Help" button to display usage and interaction details.

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
    """

    def __init__(
        self,
        host: str,
        module: int,
        chan_str="1",
        buf_size=5_000,
        refresh_ms=33,
        dot_px=DENSITY_DOT_SIZE,
    ):
        super().__init__()
        self.host = host
        self.module = module
        self.N = buf_size
        self.refresh_ms = refresh_ms
        self.dot_px = max(1, int(dot_px))

        # Parse multi-channel format
        self.channel_list = _parse_channels_multich(chan_str)

        self.paused = False
        self.start_time = None
        self.frame_cnt = 0
        self.pkt_cnt = 0
        self.t_last = time.time()

        # Single decimation stage, updated from first channel’s sample rate
        self.dec_stage = 6
        self.last_dec_update = 0.0

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

        # Density LUT
        cmap = pg.colormap.get("turbo")
        lut_rgb = cmap.getLookupTable(0.0, 1.0, 255)
        self.lut = np.vstack([
            np.zeros((1, 4), np.uint8),
            np.hstack([lut_rgb, 255 * np.ones((255, 1), np.uint8)])
        ])

        # Start UDP receiver
        self.receiver = UDPReceiver(self.host, self.module)
        self.receiver.start()

        # Thread pool
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(os.cpu_count() or 1)

        # Display settings
        self.dark_mode = True
        self.real_units = False
        self.psd_absolute = True

        # Auto Scale
        self.auto_scale_plots = True

        # Show I, Q, Mag lines
        self.show_i = True
        self.show_q = True
        self.show_m = True

        # Build UI
        self._build_ui(chan_str)
        self._init_buffers()
        self._build_layout()

        # Timer for periodic GUI updates
        self.timer = QtCore.QTimer(singleShot=False)
        self.timer.timeout.connect(self._update_gui)
        self.timer.start(self.refresh_ms)

        self.setWindowTitle("Periscope")

    # ───────────────────────── UI Construction ─────────────────────────
    def _build_ui(self, chan_str: str):
        """
        Create and configure all top-level widgets and layouts,
        preserving the original style, plus the new Help button.

        Parameters
        ----------
        chan_str : str
            The user-supplied channel specification string.
        """
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_vbox = QtWidgets.QVBoxLayout(central)

        # Title
        title = QtWidgets.QLabel(f"CRS: {self.host}    Module: {self.module}")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        ft = title.font()
        ft.setPointSize(16)
        title.setFont(ft)
        main_vbox.addWidget(title)

        # Top bar
        top_bar = QtWidgets.QWidget()
        top_h = QtWidgets.QHBoxLayout(top_bar)

        self.btn_toggle_cfg = QtWidgets.QPushButton("Show Configuration")
        self.btn_toggle_cfg.setCheckable(True)
        self.btn_toggle_cfg.toggled.connect(self._toggle_config)

        self.e_ch = QtWidgets.QLineEdit(chan_str)
        self.e_ch.setToolTip("Enter comma-separated channels or use '&' to group in one row.")
        self.e_ch.returnPressed.connect(self._update_channels)

        self.e_buf = QtWidgets.QLineEdit(str(self.N))
        self.e_buf.setValidator(QIntValidator(10, 1_000_000, self))
        self.e_buf.setMaximumWidth(80)
        self.e_buf.setToolTip("Size of the ring buffer for each channel.")
        self.e_buf.editingFinished.connect(self._change_buffer)

        self.b_pause = QtWidgets.QPushButton("Pause", clicked=self._toggle_pause)
        self.b_pause.setToolTip("Pause/resume real-time data.")

        self.cb_real = QtWidgets.QCheckBox("Real Units", checked=self.real_units)
        self.cb_real.setToolTip("Toggle raw 'counts' vs. real-voltage/dBm units.")
        self.cb_real.toggled.connect(self._toggle_real_units)

        self.cb_time = QtWidgets.QCheckBox("TOD", checked=True)
        self.cb_iq = QtWidgets.QCheckBox("IQ", checked=True)
        self.cb_fft = QtWidgets.QCheckBox("FFT", checked=True)
        self.cb_ssb = QtWidgets.QCheckBox("Single Sideband PSD", checked=False)
        self.cb_dsb = QtWidgets.QCheckBox("Dual Sideband PSD", checked=False)
        for cb in (self.cb_time, self.cb_iq, self.cb_fft, self.cb_ssb, self.cb_dsb):
            cb.toggled.connect(self._build_layout)

        # The new Help button
        self.btn_help = QtWidgets.QPushButton("Help")
        self.btn_help.setToolTip("Show usage, interaction details, and examples.")
        self.btn_help.clicked.connect(self._show_help)

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
        # Insert the Help button to the top bar
        top_h.addWidget(self.btn_help)

        main_vbox.addWidget(top_bar)

        # Show/hide config
        main_vbox.addWidget(self.btn_toggle_cfg, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.ctrl_panel = QtWidgets.QGroupBox("Configuration")
        self.ctrl_panel.setVisible(False)
        cfg_hbox = QtWidgets.QHBoxLayout(self.ctrl_panel)

        # Show Curves
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
        cfg_hbox.addWidget(show_curves_g)

        # IQ Mode
        self.rb_density = QtWidgets.QRadioButton("Density", checked=True)
        self.rb_density.setToolTip("2D histogram of I/Q values.")
        self.rb_scatter = QtWidgets.QRadioButton("Scatter")
        self.rb_scatter.setToolTip("Scatter of up to 1,000 I/Q points. CPU intensive.")
        rb_group = QtWidgets.QButtonGroup(self)
        rb_group.addButton(self.rb_density)
        rb_group.addButton(self.rb_scatter)
        for rb in (self.rb_density, self.rb_scatter):
            rb.toggled.connect(self._build_layout)

        iq_g = QtWidgets.QGroupBox("IQ Mode")
        iq_h = QtWidgets.QHBoxLayout(iq_g)
        iq_h.addWidget(self.rb_density)
        iq_h.addWidget(self.rb_scatter)
        cfg_hbox.addWidget(iq_g)

        # PSD Mode
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

        psd_g = QtWidgets.QGroupBox("PSD Mode")
        psd_grid = QtWidgets.QGridLayout(psd_g)
        psd_grid.addWidget(self.lbl_psd_scale, 0, 0)
        psd_grid.addWidget(self.rb_psd_abs, 0, 1)
        psd_grid.addWidget(self.rb_psd_rel, 0, 2)
        psd_grid.addWidget(QtWidgets.QLabel("Segments:"), 1, 0)
        psd_grid.addWidget(self.spin_segments, 1, 1)
        cfg_hbox.addWidget(psd_g)

        # General Display
        disp_g = QtWidgets.QGroupBox("General Display")
        disp_h = QtWidgets.QHBoxLayout(disp_g)
        self.cb_dark = QtWidgets.QCheckBox("Dark Mode", checked=self.dark_mode)
        self.cb_dark.setToolTip("Switch between dark/light themes.")
        self.cb_dark.toggled.connect(self._toggle_dark_mode)
        disp_h.addWidget(self.cb_dark)

        self.cb_auto_scale = QtWidgets.QCheckBox("Auto Scale", checked=self.auto_scale_plots)
        self.cb_auto_scale.setToolTip("Enable/disable auto-range for IQ/FFT/SSB/DSB. Can improve display performance.")
        self.cb_auto_scale.toggled.connect(self._toggle_auto_scale)
        disp_h.addWidget(self.cb_auto_scale)
        cfg_hbox.addWidget(disp_g)

        main_vbox.addWidget(self.ctrl_panel)

        # Plot container
        self.container = QtWidgets.QWidget()
        main_vbox.addWidget(self.container)
        self.grid = QtWidgets.QGridLayout(self.container)

        # Status bar
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
            "  - Pan: Left-click and drag.\n"
            "  - Zoom: Right-click/drag or mouse-wheel in most plots.\n"
            "  - Axis scaling (log vs lin) and auto-zooming: Individually configurable by right-clicking any plot.\n"
            "  - Double-click within a plot: Show the coordinates of the clicked position.\n"
            "  - Export plot to CSV, Image, Vector Graphics, or interactive Matplotlib window: Right-click -> Export\n"
            "  - See PyQtGraph docs for more.\n\n"
            "**Performance tips for higher FPS:**\n"
            "  - Disable auto-scaling in the configuration pain\n"
            "  - Use the density mode for IQ, smaller buffers, or enable a subset of I,Q,M.\n"            
            "  - Periscope is limited by the single-thread renderer. Lauching multiple instances can improve performance for viewing many channels.\n\n"
            "**Command-Line Examples:**\n"
            "  - `$ periscope.py rfmux0022.local --module 2 --channels \"3&5,7`\n"
            "  - `$ periscope.py rfmux0022.local -c \"1,2,3&4\"`\n\n"
            "**IPython / Jupyter:**\n"
            "  - `>>> from rfmux.core.tools import periscope as ps`\n"
            "  - `>>> ps.launch(\"rfmux0022.local\", module=2, channels=\"3&5\")`\n"
            "  - If in a non-blocking mode, you can still interact with your session concurrently.\n\n"
        )
        msg.setText(help_text)
        msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Close)
        msg.exec()

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
        This version supports multi-channel rows via &-grouping, auto scaling, and line toggles.
        """
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

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

        self.plots = []
        self.curves = []

        font = QFont()
        font.setPointSize(UI_FONT_SIZE)

        # Single-channel color set
        single_colors = {
            "I": "#1f77b4",
            "Q": "#ff7f0e",
            "Mag": "#2ca02c",
            "DSB": "#bcbd22",
        }

        # Families for multi-ch lines (I, Q, M)
        channel_families = [
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

        for row_i, group in enumerate(self.channel_list):
            rowPlots = {}
            rowCurves = {}

            row_title = (
                f"Ch {group[0]}" if len(group) == 1
                else "Ch " + "&".join(map(str, group))
            )

            for col, mode in enumerate(modes):
                vb = ClickableViewBox()
                pw = pg.PlotWidget(viewBox=vb, title=f"{_mode_title(mode)} – {row_title}")

                if mode == "T":
                    pw.enableAutoRange(pg.ViewBox.XYAxes, True)
                else:
                    pw.enableAutoRange(pg.ViewBox.XYAxes, self.auto_scale_plots)

                # Labeling
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

                # Theme & grid
                self._apply_plot_theme(pw)
                pw.showGrid(x=True, y=True, alpha=0.3)
                self.grid.addWidget(pw, row_i, col)
                rowPlots[mode] = pw

                # Font
                pi = pw.getPlotItem()
                for axis_name in ("left", "bottom", "right", "top"):
                    axis = pi.getAxis(axis_name)
                    if axis:
                        axis.setTickFont(font)
                        if axis.label:
                            axis.label.setFont(font)
                pi.titleLabel.setFont(font)

                # If IQ mode
                if mode == "IQ":
                    if self.rb_scatter.isChecked():
                        sp = pg.ScatterPlotItem(pen=None, size=SCATTER_SIZE)
                        pw.addItem(sp)
                        rowCurves["IQ"] = {"mode": "scatter", "item": sp}
                    else:
                        img = pg.ImageItem(axisOrder="row-major")
                        img.setLookupTable(self.lut)
                        pw.addItem(img)
                        rowCurves["IQ"] = {"mode": "density", "item": img}
                else:
                    # For T, F, S, D => lines + legend
                    legend = pw.addLegend(offset=(30, 10))

                    if len(group) == 1:
                        # Single channel => the original triple lines for T, F, S or single line for D
                        ch = group[0]
                        if mode == "T":
                            cI = pw.plot(pen=pg.mkPen(single_colors["I"],   width=LINE_WIDTH), name="I")
                            cQ = pw.plot(pen=pg.mkPen(single_colors["Q"],   width=LINE_WIDTH), name="Q")
                            cM = pw.plot(pen=pg.mkPen(single_colors["Mag"], width=LINE_WIDTH), name="Mag")
                            rowCurves["T"] = {ch: {"I": cI, "Q": cQ, "Mag": cM}}
                            self._fade_hidden_entries(legend, ("I", "Q"))
                        elif mode == "F":
                            cI = pw.plot(pen=pg.mkPen(single_colors["I"],   width=LINE_WIDTH), name="I")
                            cI.setFftMode(True)
                            cQ = pw.plot(pen=pg.mkPen(single_colors["Q"],   width=LINE_WIDTH), name="Q")
                            cQ.setFftMode(True)
                            cM = pw.plot(pen=pg.mkPen(single_colors["Mag"], width=LINE_WIDTH), name="Mag")
                            cM.setFftMode(True)
                            rowCurves["F"] = {ch: {"I": cI, "Q": cQ, "Mag": cM}}
                            self._fade_hidden_entries(legend, ("I", "Q"))
                        elif mode == "S":
                            cI = pw.plot(pen=pg.mkPen(single_colors["I"],   width=LINE_WIDTH), name="I")
                            cQ = pw.plot(pen=pg.mkPen(single_colors["Q"],   width=LINE_WIDTH), name="Q")
                            cM = pw.plot(pen=pg.mkPen(single_colors["Mag"], width=LINE_WIDTH), name="Mag")
                            rowCurves["S"] = {ch: {"I": cI, "Q": cQ, "Mag": cM}}
                        else:  # "D"
                            cD = pw.plot(pen=pg.mkPen(single_colors["DSB"], width=LINE_WIDTH),
                                         name="Complex DSB")
                            rowCurves["D"] = {ch: {"Cmplx": cD}}
                        self._make_legend_clickable(legend)
                    else:
                        # Multi-channel => color families
                        mode_dict = {}
                        for i, ch in enumerate(group):
                            (colI, colQ, colM) = channel_families[i % len(channel_families)]
                            if mode == "T":
                                cI = pw.plot(pen=pg.mkPen(colI, width=LINE_WIDTH),   name=f"ch{ch}-I")
                                cQ = pw.plot(pen=pg.mkPen(colQ, width=LINE_WIDTH),   name=f"ch{ch}-Q")
                                cM = pw.plot(pen=pg.mkPen(colM, width=LINE_WIDTH),   name=f"ch{ch}-Mag")
                                mode_dict[ch] = {"I": cI, "Q": cQ, "Mag": cM}
                            elif mode == "F":
                                cI = pw.plot(pen=pg.mkPen(colI, width=LINE_WIDTH),   name=f"ch{ch}-I")
                                cI.setFftMode(True)
                                cQ = pw.plot(pen=pg.mkPen(colQ, width=LINE_WIDTH),   name=f"ch{ch}-Q")
                                cQ.setFftMode(True)
                                cM = pw.plot(pen=pg.mkPen(colM, width=LINE_WIDTH),   name=f"ch{ch}-Mag")
                                cM.setFftMode(True)
                                mode_dict[ch] = {"I": cI, "Q": cQ, "Mag": cM}
                            elif mode == "S":
                                cI = pw.plot(pen=pg.mkPen(colI, width=LINE_WIDTH),   name=f"ch{ch}-I")
                                cQ = pw.plot(pen=pg.mkPen(colQ, width=LINE_WIDTH),   name=f"ch{ch}-Q")
                                cM = pw.plot(pen=pg.mkPen(colM, width=LINE_WIDTH),   name=f"ch{ch}-Mag")
                                mode_dict[ch] = {"I": cI, "Q": cQ, "Mag": cM}
                            else:  # "D"
                                cD = pw.plot(pen=pg.mkPen(colI, width=LINE_WIDTH),
                                             name=f"ch{ch}-DSB")
                                mode_dict[ch] = {"Cmplx": cD}
                        rowCurves[mode] = mode_dict
                        self._make_legend_clickable(legend)

            self.plots.append(rowPlots)
            self.curves.append(rowCurves)

        # Re-enable auto-range after building
        self.auto_scale_plots = True
        self.cb_auto_scale.setChecked(True)
        for rowPlots in self.plots:
            for mode, pw in rowPlots.items():
                if mode != "T":
                    pw.enableAutoRange(pg.ViewBox.XYAxes, True)

        # Apply “Show Curves” toggles
        self._toggle_iqmag()

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
            while not self.receiver.queue.empty():
                self.receiver.queue.get()
            return

        # Collect new packets
        while not self.receiver.queue.empty():
            pkt = self.receiver.queue.get()
            self.pkt_cnt += 1

            # Basic timestamp alignment
            ts = pkt.ts
            ts.ss += int(0.02 * streamer.SS_PER_SECOND)
            ts.renormalize()
            t_now = ts.h * 3600 + ts.m * 60 + ts.s + ts.ss / streamer.SS_PER_SECOND
            if self.start_time is None:
                self.start_time = t_now
            t_rel = t_now - self.start_time

            # Fill ring buffers
            for ch in self.all_chs:
                Ival = pkt.s[2 * (ch - 1)] / 256.0
                Qval = pkt.s[2 * (ch - 1) + 1] / 256.0
                self.buf[ch]["I"].add(Ival)
                self.buf[ch]["Q"].add(Qval)
                self.buf[ch]["M"].add(math.hypot(Ival, Qval))
                self.tbuf[ch].add(t_rel)

        self.frame_cnt += 1
        now = time.time()

        # Recompute dec_stage once per second
        if (now - self.last_dec_update) > 1.0:
            self._update_dec_stage()
            self.last_dec_update = now

        # Update row data
        for row_i, group in enumerate(self.channel_list):
            rowCurves = self.curves[row_i]
            for ch in group:
                # Grab ring buffer data
                rawI = self.buf[ch]["I"].data()
                rawQ = self.buf[ch]["Q"].data()
                rawM = self.buf[ch]["M"].data()
                tarr = self.tbuf[ch].data()

                # Real units if enabled
                if self.real_units:
                    I = convert_roc_to_volts(rawI)
                    Q = convert_roc_to_volts(rawQ)
                    M = convert_roc_to_volts(rawM)
                else:
                    I, Q, M = rawI, rawQ, rawM

                # TOD
                if "T" in rowCurves and ch in rowCurves["T"]:
                    cset = rowCurves["T"][ch]
                    if cset["I"].isVisible():
                        cset["I"].setData(tarr, I)
                    if cset["Q"].isVisible():
                        cset["Q"].setData(tarr, Q)
                    if cset["Mag"].isVisible():
                        cset["Mag"].setData(tarr, M)

                # FFT
                if "F" in rowCurves and ch in rowCurves["F"]:
                    cset = rowCurves["F"][ch]
                    if cset["I"].isVisible():
                        cset["I"].setData(tarr, I, fftMode=True)
                    if cset["Q"].isVisible():
                        cset["Q"].setData(tarr, Q, fftMode=True)
                    if cset["Mag"].isVisible():
                        cset["Mag"].setData(tarr, M, fftMode=True)

            # IQ tasks
            if "IQ" in rowCurves and not self.iq_workers.get(row_i, False):
                mode = rowCurves["IQ"]["mode"]
                if len(group) == 1:
                    c = group[0]
                    rawI = self.buf[c]["I"].data()
                    rawQ = self.buf[c]["Q"].data()
                    self.iq_workers[row_i] = True
                    task = IQTask(row_i, c, rawI, rawQ, self.dot_px, mode, self.iq_signals)
                    self.pool.start(task)
                else:
                    # Combine data from multiple channels
                    concatI = np.concatenate([self.buf[ch]["I"].data() for ch in group])
                    concatQ = np.concatenate([self.buf[ch]["Q"].data() for ch in group])
                    big_size = concatI.size
                    if big_size > 50000:
                        stride = max(1, big_size // 50000)
                        concatI = concatI[::stride]
                        concatQ = concatQ[::stride]
                    if concatI.size > 1:
                        self.iq_workers[row_i] = True
                        task = IQTask(row_i, 0, concatI, concatQ, self.dot_px, mode, self.iq_signals)
                        self.pool.start(task)

            # PSD tasks
            if "S" in rowCurves:
                for ch in group:
                    if not self.psd_workers[row_i]["S"][ch]:
                        rawI = self.buf[ch]["I"].data()
                        rawQ = self.buf[ch]["Q"].data()
                        self.psd_workers[row_i]["S"][ch] = True
                        task = PSDTask(
                            row=row_i,
                            ch=ch,
                            I=rawI,
                            Q=rawQ,
                            mode="SSB",
                            dec_stage=self.dec_stage,
                            real_units=self.real_units,
                            psd_absolute=self.psd_absolute,
                            segments=self.spin_segments.value(),
                            signals=self.psd_signals,
                        )
                        self.pool.start(task)

            if "D" in rowCurves:
                for ch in group:
                    if not self.psd_workers[row_i]["D"][ch]:
                        rawI = self.buf[ch]["I"].data()
                        rawQ = self.buf[ch]["Q"].data()
                        self.psd_workers[row_i]["D"][ch] = True
                        task = PSDTask(
                            row=row_i,
                            ch=ch,
                            I=rawI,
                            Q=rawQ,
                            mode="DSB",
                            dec_stage=self.dec_stage,
                            real_units=self.real_units,
                            psd_absolute=self.psd_absolute,
                            segments=self.spin_segments.value(),
                            signals=self.psd_signals,
                        )
                        self.pool.start(task)

        # FPS / PPS
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
        else:  # scatter
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
        else:  # DSB
            if "D" not in self.curves[row]:
                return
            ddict = self.curves[row]["D"]
            if ch not in ddict:
                return
            freq_dsb, psd_dsb = payload
            if ddict[ch]["Cmplx"].isVisible():
                ddict[ch]["Cmplx"].setData(freq_dsb, psd_dsb)

    # ───────────────────────── UI callbacks ─────────────────────────
    def _toggle_pause(self):
        """
        Pause or resume real-time data updates. When paused, new packets
        are discarded to avoid stale accumulation.
        """
        self.paused = not self.paused
        self.b_pause.setText("Resume" if self.paused else "Pause")

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

    def closeEvent(self, ev):
        """
        Cleanly shut down the background receiver and stop the timer before closing.

        Parameters
        ----------
        ev : QCloseEvent
            The close event instance.
        """
        self.timer.stop()
        self.receiver.stop()
        self.receiver.wait()
        super().closeEvent(ev)
        ev.accept()


def main():
    """
    Entry point for command-line usage. Supports multi-channel grouping via '&',
    auto-scaling, and global I/Q/M toggles. Launches Periscope in blocking mode.
    """
    ap = argparse.ArgumentParser(
        description=(
            "Periscope – Real-time multi-pane viewer with optional multi-channel grouping, "
            "auto-scaling, and I/Q/M toggles."
        )
    )
    ap.add_argument("hostname")
    ap.add_argument("-m", "--module", type=int, default=1)
    ap.add_argument("-c", "--channels", default="1")
    ap.add_argument("-n", "--num-samples", type=int, default=5_000)
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

    win = Periscope(
        host=args.hostname,
        module=args.module,
        chan_str=args.channels,
        buf_size=args.num_samples,
        refresh_ms=refresh_ms,
        dot_px=args.density_dot,
    )
    win.show()
    sys.exit(app.exec())


def launch(
    hostname: str,
    *,
    module: int = 1,
    channels: str = "1",
    buf_size: int = 5_000,
    fps: float = 30.0,
    density_dot: int = 1,
    blocking: bool | None = None,
):
    """
    Programmatic entry point for embedding or interactive usage of Periscope.

    This function either blocks (runs the Qt event loop) or returns control
    immediately, depending on whether a Qt event loop is already running or the
    'blocking' parameter is explicitly set.

    Parameters
    ----------
    hostname : str
        Multicast/UDP host address.
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
    refresh_ms = int(round(1000.0 / fps))

    viewer = Periscope(
        host=hostname,
        module=module,
        chan_str=channels,
        buf_size=buf_size,
        refresh_ms=refresh_ms,
        dot_px=density_dot,
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
