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
- Single‑Sideband PSD
- Dual‑Sideband PSD
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
from PyQt6.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal
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
DENSITY_DOT_SIZE = 1          # default pixel “diameter” used for density mode
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
        True if a Qt event loop is already running, otherwise False.
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


class IQSignals(QObject):
    """
    Holds custom signals emitted by IQ tasks.
    """
    done = pyqtSignal(int, str, object)
    # Emitted with arguments: (channel, mode_string, payload)
    #   channel: int
    #   mode_string: "density" or "scatter"
    #   payload: task-specific data


class IQTask(QRunnable):
    """
    Off-thread worker for computing IQ scatter or density histograms.

    Parameters
    ----------
    ch : int
        Channel index.
    I : np.ndarray
        Array of I samples.
    Q : np.ndarray
        Array of Q samples.
    dot_px : int
        Dot diameter in pixels for point dilation (density mode).
    mode : {"density", "scatter"}
        Determines whether to compute a 2D histogram or just scatter points.
    signals : IQSignals
        An IQSignals instance to emit results back to the GUI thread.
    """

    def __init__(self, ch, I, Q, dot_px, mode, signals: IQSignals):
        super().__init__()
        self.ch = ch
        self.I = I.copy()
        self.Q = Q.copy()
        self.dot_px = dot_px
        self.mode = mode
        self.signals = signals

    def run(self):
        """
        Perform the required computation off the main thread. Emit results
        via self.signals.done.
        """
        if len(self.I) < 2:
            # Edge case: not enough data
            if self.mode == "density":
                empty = np.zeros((DENSITY_GRID, DENSITY_GRID), np.uint8)
                payload = (empty, (0, 1, 0, 1))
            else:
                payload = ([], [], [])
            self.signals.done.emit(self.ch, self.mode, payload)
            return

        if self.mode == "density":
            g = DENSITY_GRID
            hist = np.zeros((g, g), np.uint32)

            Imin, Imax = self.I.min(), self.I.max()
            Qmin, Qmax = self.Q.min(), self.Q.max()
            if Imin == Imax or Qmin == Qmax:
                payload = (hist.astype(np.uint8), (Imin, Imax, Qmin, Qmax))
                self.signals.done.emit(self.ch, self.mode, payload)
                return

            ix = ((self.I - Imin) * (g - 1) / (Imax - Imin)).astype(np.intp)
            qy = ((self.Q - Qmin) * (g - 1) / (Qmax - Qmin)).astype(np.intp)

            # Base histogram (unit radius)
            np.add.at(hist, (qy, ix), 1)

            # Point dilation if requested
            if self.dot_px > 1:
                r = self.dot_px // 2
                if convolve is not None:
                    k = 2 * r + 1
                    kernel = np.ones((k, k), dtype=np.uint8)
                    hist = convolve(hist, kernel, mode="constant", cval=0)
                else:
                    # Fallback to original nested‑loop approach
                    for dy in range(-r, r + 1):
                        for dx in range(-r, r + 1):
                            ys, xs = qy + dy, ix + dx
                            mask = (
                                (0 <= ys) & (ys < g) &
                                (0 <= xs) & (xs < g)
                            )
                            np.add.at(hist, (ys[mask], xs[mask]), 1)

            # Optional smoothing & log‑compression
            if gaussian_filter is not None and SMOOTH_SIGMA > 0:
                hist = gaussian_filter(
                    hist.astype(np.float32), SMOOTH_SIGMA, mode="nearest"
                )
            if LOG_COMPRESS:
                hist = np.log1p(hist, out=hist.astype(np.float32))

            # 8‑bit normalisation
            if hist.max() > 0:
                hist = (hist * (255.0 / hist.max())).astype(np.uint8)

            payload = (hist, (Imin, Imax, Qmin, Qmax))

        else:  # scatter mode
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

        self.signals.done.emit(self.ch, self.mode, payload)


class PSDSignals(QObject):
    """
    Holds custom signals emitted by PSD tasks.
    """
    done = pyqtSignal(int, str, object)
    # Emitted with arguments: (channel, mode_string, payload)
    #   channel: int
    #   mode_string: "SSB" or "DSB"
    #   payload: task-specific data


class PSDTask(QRunnable):
    """
    Off‑thread worker for single or dual sideband PSD computation.

    Parameters
    ----------
    ch : int
        Channel index.
    I : np.ndarray
        Array of I samples.
    Q : np.ndarray
        Array of Q samples.
    mode : {"SSB", "DSB"}
        Determines the type of PSD computation.
    dec_stage : int
        Decimation stage, used for the spectrum_from_slow_tod() call.
    real_units : bool
        If True, convert PSD to dBm/dBc. Otherwise, keep as raw counts²/Hz.
    psd_absolute : bool
        If True and real_units is True, use 'absolute' reference (dBm).
        If False and real_units is True, use 'relative' reference (dBc).
    segments : int
        Number of noise segments for Welch segmentation. Data is split by
        nperseg = data_len // segments.
    signals : PSDSignals
        A PSDSignals instance to emit results back to the GUI thread.
    """

    def __init__(
        self,
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
            # Not enough data to do anything meaningful
            if self.mode == "SSB":
                payload = ([], [], [], [], [], [], 0.0)
            else:
                payload = ([], [])
            self.signals.done.emit(self.ch, self.mode, payload)
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
        else:
            freq_dsb = spec_iq["freq_dsb"]
            psd_dsb = spec_iq["psd_dual_sideband"]
            order = np.argsort(freq_dsb)  # sort ‑ve → +ve
            payload = (freq_dsb[order], psd_dsb[order])

        self.signals.done.emit(self.ch, self.mode, payload)


class Periscope(QtWidgets.QMainWindow):
    """
    Multi‑pane PyQt application for real-time data visualization of:
      - Time-domain waveforms (TOD)
      - IQ density or scatter
      - FFT
      - Single-Sideband PSD (SSB)
      - Dual-Sideband PSD (DSB)

    Parameters
    ----------
    host : str
        The multicast/UDP host address for receiving packets.
    module : int
        The module index (1-based) used to filter incoming packets.
    chan_str : str, optional
        Comma-separated string of channels to display. Defaults to "1".
    buf_size : int, optional
        Size of each ring buffer for storing incoming data. Defaults to 5000.
    refresh_ms : int, optional
        GUI refresh interval in milliseconds. Defaults to 33.
    dot_px : int, optional
        Dot diameter in pixels for IQ density dot dilation. Defaults to DENSITY_DOT_SIZE.
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
        self.channels = self._parse_channels(chan_str)

        self.paused = False
        self.start_time = None
        self.frame_cnt = 0
        self.pkt_cnt = 0
        self.t_last = time.time()

        self.dec_stages: Dict[int, int] = {ch: 6 for ch in self.channels}
        self.last_dec_update = 0.0

        # IQ concurrency tracking
        self.iq_workers: Dict[int, bool] = {}
        self.iq_signals = IQSignals()
        self.iq_signals.done.connect(self._iq_done)

        # PSD concurrency tracking
        self.psd_workers: Dict[int, Dict[str, bool]] = {}
        self.psd_signals = PSDSignals()
        self.psd_signals.done.connect(self._psd_done)

        # Color look-up table for density images
        cmap = pg.colormap.get("turbo")
        lut_rgb = cmap.getLookupTable(0.0, 1.0, 255)
        self.lut = np.vstack(
            [np.zeros((1, 4), np.uint8),
             np.hstack([lut_rgb, 255 * np.ones((255, 1), np.uint8)])]
        )

        # Start the background receiver thread
        self.receiver = UDPReceiver(host, module)
        self.receiver.start()

        # Thread pool for IQ and PSD tasks
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(os.cpu_count() or 1)

        # Initial display settings
        self.dark_mode = True
        self.real_units = False
        self.psd_absolute = True

        self._build_ui(chan_str)
        self._init_buffers()
        self._build_layout()

        # Periodic timer for GUI updates
        self.timer = QtCore.QTimer(singleShot=False)
        self.timer.timeout.connect(self._update_gui)
        self.timer.start(self.refresh_ms)

        self.setWindowTitle("Periscope")

    # ───────────────────────── UI construction ─────────────────────────
    def _build_ui(self, chan_str: str):
        """
        Create and configure all top-level widgets and layouts (excluding plots),
        then assemble them into the main window's layout.

        Parameters
        ----------
        chan_str : str
            Comma-separated channel specification for the user interface label.
        """

        # Instantiate controls ------------------------------------------------
        self.btn_toggle_cfg = QtWidgets.QPushButton("Show Configuration")
        self.btn_toggle_cfg.setCheckable(True)
        self.btn_toggle_cfg.toggled.connect(self._toggle_config)
        self.btn_toggle_cfg.setToolTip("Show or hide the advanced configuration panel.")

        # Channels & buffer
        self.e_ch = QtWidgets.QLineEdit(chan_str)
        self.e_ch.returnPressed.connect(self._update_channels)
        self.e_ch.setToolTip("Enter a comma-separated list of channel indices (1-based).")

        self.e_buf = QtWidgets.QLineEdit(str(self.N))
        self.e_buf.setValidator(QIntValidator(10, 1_000_000, self))
        self.e_buf.editingFinished.connect(self._change_buffer)
        self.e_buf.setMaximumWidth(80)
        self.e_buf.setToolTip("Size of the ring buffer for each channel.")

        self.b_pause = QtWidgets.QPushButton("Pause", clicked=self._toggle_pause)
        self.b_pause.setToolTip("Pause or resume the real-time data acquisition.")

        # Global “Real Units”
        self.cb_real = QtWidgets.QCheckBox("Real Units", checked=self.real_units)
        self.cb_real.toggled.connect(self._toggle_real_units)
        self.cb_real.setToolTip("Toggle between raw 'counts' units and real-voltage/dBm units.")

        # Plot‑mode checkboxes
        self.cb_time = QtWidgets.QCheckBox("TOD", checked=True)
        self.cb_time.setToolTip("Display time-domain waveforms.")
        self.cb_iq = QtWidgets.QCheckBox("IQ", checked=True)
        self.cb_iq.setToolTip("Display IQ (density or scatter) plots.")
        self.cb_fft = QtWidgets.QCheckBox("FFT", checked=True)
        self.cb_fft.setToolTip("Display FFT (log frequency vs. log amplitude).")
        self.cb_ssb = QtWidgets.QCheckBox("SSB PSD", checked=False)
        self.cb_ssb.setToolTip("Display Single-Sideband PSD.")
        self.cb_dsb = QtWidgets.QCheckBox("DSB PSD", checked=False)
        self.cb_dsb.setToolTip("Display Dual-Sideband PSD.")

        for cb in (self.cb_time, self.cb_iq, self.cb_fft, self.cb_ssb, self.cb_dsb):
            cb.toggled.connect(self._build_layout)

        # IQ‑mode radios
        self.rb_density = QtWidgets.QRadioButton("Density", checked=True)
        self.rb_density.setToolTip("Use a 2D histogram of I/Q values (CPU friendly).")
        self.rb_scatter = QtWidgets.QRadioButton("Scatter")
        self.rb_scatter.setToolTip("Plot a subset of individual I/Q samples color graded by time (CPU intensive).")
        rb_group = QtWidgets.QButtonGroup(self)
        for rb in (self.rb_density, self.rb_scatter):
            rb_group.addButton(rb)
            rb.toggled.connect(self._build_layout)

        # PSD‑scale controls
        self.lbl_psd_scale = QtWidgets.QLabel("PSD Scale:")
        self.rb_psd_abs = QtWidgets.QRadioButton("Absolute (dBm)", checked=True)
        self.rb_psd_abs.setToolTip("Compute PSD in absolute scale (dBm) if Real Units is enabled.")
        self.rb_psd_rel = QtWidgets.QRadioButton("Relative (dBc)")
        self.rb_psd_rel.setToolTip("Compute PSD in relative scale (dBc) if Real Units is enabled.")
        for rb in (self.rb_psd_abs, self.rb_psd_rel):
            rb.toggled.connect(self._psd_ref_changed)

        # Noise segments spinner
        self.spin_segments = QtWidgets.QSpinBox()
        self.spin_segments.setRange(1, 256)
        self.spin_segments.setValue(1)
        self.spin_segments.setMaximumWidth(80)
        self.spin_segments.setToolTip(
            "Number of segments for Welch PSD averaging. "
            "Data is split into 'segments' parts for PSD calculation."
        )

        # Display toggle
        self.cb_dark = QtWidgets.QCheckBox("Dark Mode", checked=self.dark_mode)
        self.cb_dark.setToolTip("Switch between dark and light UI themes.")
        self.cb_dark.toggled.connect(self._toggle_dark_mode)

        # Build main layout ---------------------------------------------------
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
        top_h.addWidget(QtWidgets.QLabel("Channels:"))
        top_h.addWidget(self.e_ch)
        top_h.addSpacing(20)
        top_h.addWidget(QtWidgets.QLabel("Buffer:"))
        top_h.addWidget(self.e_buf)
        top_h.addWidget(self.b_pause)
        top_h.addSpacing(40)
        top_h.addWidget(self.cb_real)
        top_h.addSpacing(20)
        for cb in (self.cb_time, self.cb_iq, self.cb_fft, self.cb_ssb, self.cb_dsb):
            top_h.addWidget(cb)
        top_h.addStretch(1)
        main_vbox.addWidget(top_bar)

        # Show/hide configuration toggle
        main_vbox.addWidget(
            self.btn_toggle_cfg,
            alignment=QtCore.Qt.AlignmentFlag.AlignRight,
        )

        # Configuration panel (hidden by default)
        self.ctrl_panel = QtWidgets.QGroupBox("Configuration")
        self.ctrl_panel.setVisible(False)
        cfg_hbox = QtWidgets.QHBoxLayout(self.ctrl_panel)

        # — IQ Mode —
        self.iq_g = QtWidgets.QGroupBox("IQ Mode")
        iq_h = QtWidgets.QHBoxLayout(self.iq_g)
        iq_h.addWidget(self.rb_density)
        iq_h.addWidget(self.rb_scatter)
        cfg_hbox.addWidget(self.iq_g)

        # — PSD Options —
        self.psd_g = QtWidgets.QGroupBox("PSD Options")
        psd_grid = QtWidgets.QGridLayout(self.psd_g)
        psd_grid.addWidget(self.lbl_psd_scale, 0, 0)
        psd_grid.addWidget(self.rb_psd_abs, 0, 1)
        psd_grid.addWidget(self.rb_psd_rel, 0, 2)
        psd_grid.addWidget(QtWidgets.QLabel("Noise Segments:"), 1, 0)
        psd_grid.addWidget(self.spin_segments, 1, 1)
        cfg_hbox.addWidget(self.psd_g)

        # — Display —
        disp_g = QtWidgets.QGroupBox("Display")
        disp_h = QtWidgets.QHBoxLayout(disp_g)
        disp_h.addWidget(self.cb_dark)
        cfg_hbox.addWidget(disp_g)

        main_vbox.addWidget(self.ctrl_panel)

        # Plot grid container
        self.container = QtWidgets.QWidget()
        main_vbox.addWidget(self.container)
        self.grid = QtWidgets.QGridLayout(self.container)

        # Status bar
        self.setStatusBar(QtWidgets.QStatusBar())

    def _toggle_config(self, visible: bool):
        """
        Toggle the visibility of the advanced configuration panel.

        Parameters
        ----------
        visible : bool
            If True, the configuration panel is made visible; otherwise hidden.
        """
        self.ctrl_panel.setVisible(visible)
        self.btn_toggle_cfg.setText("Hide Configuration" if visible else "Show Configuration")

    def _init_buffers(self):
        """
        Recreate the ring buffers for each selected channel
        to match the current buffer size self.N.
        """
        self.buf = {}
        self.tbuf = {}
        for ch in self.channels:
            self.buf[ch] = {k: Circular(self.N) for k in ("I", "Q", "M")}
            self.tbuf[ch] = Circular(self.N)
            self.psd_workers[ch] = {"S": False, "D": False}
            self.dec_stages[ch] = 6

    def _build_layout(self):
        """
        Rebuild the plot layout grid based on the currently enabled modes (TOD, IQ, FFT, SSB, DSB).
        Each channel is placed in a row, and each active mode is placed in a column.
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

        self.plots = {}
        self.curves = {}
        font = QFont()
        font.setPointSize(UI_FONT_SIZE)

        for row, ch in enumerate(self.channels):
            self.plots[ch] = {}
            self.curves[ch] = {}
            for col, mode in enumerate(modes):
                vb = ClickableViewBox()
                if mode == "T":
                    pw = pg.PlotWidget(viewBox=vb, title=f"Ch {ch} – Time")
                    if not self.real_units:
                        pw.setLabel("left", "Amplitude", units="Counts")
                    else:
                        pw.setLabel("left", "Amplitude", units="V")
                elif mode == "IQ":
                    pw = pg.PlotWidget(viewBox=vb, title=f"Ch {ch} – IQ")
                    if not self.real_units:
                        pw.setLabel("bottom", "I", units="Counts")
                        pw.setLabel("left", "Q", units="Counts")
                    else:
                        pw.setLabel("bottom", "I", units="V")
                        pw.setLabel("left", "Q", units="V")
                    pw.getViewBox().setAspectLocked(True)
                elif mode == "F":
                    pw = pg.PlotWidget(viewBox=vb, title=f"Ch {ch} – Raw FFT")
                    pw.setLogMode(x=True, y=True)
                    pw.setLabel("bottom", "Freq", units="Hz")
                    if not self.real_units:
                        pw.setLabel("left", "Amplitude", units="Counts")
                    else:
                        pw.setLabel("left", "Amplitude", units="V")
                elif mode == "S":
                    pw = pg.PlotWidget(viewBox=vb, title=f"Ch {ch} – SSB PSD")
                    pw.setLogMode(x=True, y=not self.real_units)
                    pw.setLabel("bottom", "Freq", units="Hz")
                    if not self.real_units:
                        pw.setLabel("left", "PSD (Counts²/Hz)")
                    else:
                        lbl = "dBm/Hz" if self.psd_absolute else "dBc/Hz"
                        pw.setLabel("left", f"PSD ({lbl})")
                else:  # "D"
                    pw = pg.PlotWidget(viewBox=vb, title=f"Ch {ch} – DSB PSD")
                    pw.setLogMode(x=False, y=not self.real_units)
                    pw.setLabel("bottom", "Freq", units="Hz")
                    if not self.real_units:
                        pw.setLabel("left", "PSD (Counts²/Hz)")
                    else:
                        lbl = "dBm/Hz" if self.psd_absolute else "dBc/Hz"
                        pw.setLabel("left", f"PSD ({lbl})")

                self._apply_plot_theme(pw)
                pw.showGrid(x=True, y=True, alpha=0.3)
                self.plots[ch][mode] = pw
                self.grid.addWidget(pw, row, col)

                # Axis font
                pi = pw.getPlotItem()
                for axis_name in ("left", "bottom", "right", "top"):
                    axis = pi.getAxis(axis_name)
                    if axis:
                        axis.setTickFont(font)
                        if axis.label:
                            axis.label.setFont(font)
                pi.titleLabel.setFont(font)

                if mode == "T":
                    legend = pw.addLegend(offset=(30, 10))
                    self.curves[ch]["T"] = {
                        k: pw.plot(pen=pg.mkPen(c, width=LINE_WIDTH), name=k)
                        for k, c in zip(("I", "Q", "Mag"), ("#1f77b4", "#ff7f0e", "#2ca02c"))
                    }
                    self._fade_hidden_entries(legend, hide_labels=("I", "Q"))
                    self._make_legend_clickable(legend)

                elif mode == "F":
                    legend = pw.addLegend(offset=(30, 10))
                    self.curves[ch]["F"] = {
                        k: pw.plot(pen=pg.mkPen(c, width=LINE_WIDTH), name=k)
                        for k, c in zip(("I", "Q", "Mag"), ("#1f77b4", "#ff7f0e", "#2ca02c"))
                    }
                    for curve in self.curves[ch]["F"].values():
                        curve.setFftMode(True)
                    self._fade_hidden_entries(legend, hide_labels=("I", "Q"))
                    self._make_legend_clickable(legend)

                elif mode == "IQ":
                    if self.rb_scatter.isChecked():
                        sp = pg.ScatterPlotItem(pen=None, size=SCATTER_SIZE)
                        pw.addItem(sp)
                        self.curves[ch]["IQ"] = {"mode": "scatter", "item": sp}
                    else:
                        img = pg.ImageItem(axisOrder="row-major")
                        img.setLookupTable(self.lut)
                        pw.addItem(img)
                        self.curves[ch]["IQ"] = {"mode": "density", "item": img}

                elif mode == "S":
                    legend = pw.addLegend(offset=(30, 10))
                    self.curves[ch]["S"] = {
                        k: pw.plot(pen=pg.mkPen(c, width=LINE_WIDTH), name=k)
                        for k, c in zip(("I", "Q", "Mag"), ("#1f77b4", "#ff7f0e", "#2ca02c"))
                    }
                    self._make_legend_clickable(legend)

                else:  # "D"
                    legend = pw.addLegend(offset=(30, 10))
                    curve = pw.plot(pen=pg.mkPen("#bcbd22", width=LINE_WIDTH), name="Complex DSB")
                    self.curves[ch]["D"] = {"Cmplx": curve}
                    self._make_legend_clickable(legend)

    def _apply_plot_theme(self, pw: pg.PlotWidget):
        """
        Configure the plot widget's background and axis color
        based on self.dark_mode.

        Parameters
        ----------
        pw : pg.PlotWidget
            The plot widget to be styled.
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
        Fade out (gray) specific legend entries to indicate they are typically
        optional or less interesting.

        Parameters
        ----------
        legend : pg.LegendItem
            The legend item containing curve references.
        hide_labels : tuple of str
            Legend labels whose entries should be faded.
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
            The legend item containing curve references.
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

    # ───────────────────────── Main GUI update ─────────────────────────
    def _update_gui(self):
        """
        Periodic GUI update method called by the QTimer. It:
          - Processes new UDP data packets from self.receiver's queue.
          - Fills channel ring buffers.
          - Handles decimation stage updates once per second.
          - Spawns background tasks for IQ/PSD if necessary.
          - Updates displayed curves/images with new data.
          - Tracks and displays frames/packets per second in the status bar.
        """
        if self.paused:
            while not self.receiver.queue.empty():
                self.receiver.queue.get()
            return

        # Ingest new packets
        while not self.receiver.queue.empty():
            pkt = self.receiver.queue.get()
            self.pkt_cnt += 1

            ts = pkt.ts
            ts.ss += int(0.02 * streamer.SS_PER_SECOND)
            ts.renormalize()
            t_now = ts.h * 3600 + ts.m * 60 + ts.s + ts.ss / streamer.SS_PER_SECOND
            if self.start_time is None:
                self.start_time = t_now
            t_rel = t_now - self.start_time

            for ch in self.channels:
                Ival = pkt.s[2 * (ch - 1)] / 256.0
                Qval = pkt.s[2 * (ch - 1) + 1] / 256.0
                self.buf[ch]["I"].add(Ival)
                self.buf[ch]["Q"].add(Qval)
                self.buf[ch]["M"].add(math.hypot(Ival, Qval))
                self.tbuf[ch].add(t_rel)

        self.frame_cnt += 1

        # Recompute dec_stage once per second
        now = time.time()
        if (now - self.last_dec_update) > 1.0:
            self._update_dec_stages()
            self.last_dec_update = now

        # Per‑channel updates
        for ch in self.channels:
            rawI = self.buf[ch]["I"].data()
            rawQ = self.buf[ch]["Q"].data()
            rawM = self.buf[ch]["M"].data()
            tarr = self.tbuf[ch].data()

            if self.real_units:
                I = convert_roc_to_volts(rawI)
                Q = convert_roc_to_volts(rawQ)
                M = convert_roc_to_volts(rawM)
            else:
                I, Q, M = rawI, rawQ, rawM

            # T
            if "T" in self.curves[ch]:
                c = self.curves[ch]["T"]
                if c["I"].isVisible():
                    c["I"].setData(tarr, I)
                if c["Q"].isVisible():
                    c["Q"].setData(tarr, Q)
                if c["Mag"].isVisible():
                    c["Mag"].setData(tarr, M)

            # IQ – off‑thread
            if "IQ" in self.curves[ch] and not self.iq_workers.get(ch, False):
                mode = self.curves[ch]["IQ"]["mode"]
                self.iq_workers[ch] = True
                task = IQTask(ch, rawI, rawQ, self.dot_px, mode, self.iq_signals)
                self.pool.start(task)

            # FFT
            if "F" in self.curves[ch]:
                c = self.curves[ch]["F"]
                c["I"].setData(tarr, I, fftMode=True)
                c["Q"].setData(tarr, Q, fftMode=True)
                c["Mag"].setData(tarr, M, fftMode=True)

            # PSD tasks
            dstage = self.dec_stages[ch]
            segments = self.spin_segments.value()

            if "S" in self.curves[ch] and not self.psd_workers[ch]["S"]:
                self.psd_workers[ch]["S"] = True
                task = PSDTask(
                    ch=ch,
                    I=rawI,
                    Q=rawQ,
                    mode="SSB",
                    dec_stage=dstage,
                    real_units=self.real_units,
                    psd_absolute=self.psd_absolute,
                    segments=segments,
                    signals=self.psd_signals,
                )
                self.pool.start(task)

            if "D" in self.curves[ch] and not self.psd_workers[ch]["D"]:
                self.psd_workers[ch]["D"] = True
                task = PSDTask(
                    ch=ch,
                    I=rawI,
                    Q=rawQ,
                    mode="DSB",
                    dec_stage=dstage,
                    real_units=self.real_units,
                    psd_absolute=self.psd_absolute,
                    segments=segments,
                    signals=self.psd_signals,
                )
                self.pool.start(task)

        # FPS / PPS in status‑bar
        if (now - self.t_last) >= 1.0:
            fps = self.frame_cnt / (now - self.t_last)
            pps = self.pkt_cnt / (now - self.t_last)
            self.statusBar().showMessage(f"FPS {fps:.1f} | Packets/s {pps:.1f}")
            self.frame_cnt = 0
            self.pkt_cnt = 0
            self.t_last = now

    def _update_dec_stages(self):
        """
        Update each channel's decimation stage by measuring the time buffer
        sample spacing and inferring the sample rate.
        """
        for ch in self.channels:
            tarr = self.tbuf[ch].data()
            if len(tarr) < 2:
                continue
            dt = (tarr[-1] - tarr[0]) / max(1, (len(tarr) - 1))
            fs = 1.0 / dt if dt > 0 else 1.0
            self.dec_stages[ch] = infer_dec_stage(fs)

    # ───────────────────────── Slots ─────────────────────────
    @QtCore.pyqtSlot(int, str, object)
    def _iq_done(self, ch: int, task_mode: str, payload):
        """
        Slot called when an off-thread IQTask finishes.

        Parameters
        ----------
        ch : int
            Channel index.
        task_mode : {"density", "scatter"}
            The IQ mode that was computed.
        payload : object
            Task result data. For "density", it's (hist2D, (Imin, Imax, Qmin, Qmax)).
            For "scatter", it's (xs, ys, color_array).
        """
        self.iq_workers[ch] = False
        if ch not in self.curves or "IQ" not in self.curves[ch]:
            return
        pane = self.curves[ch]["IQ"]
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

    @QtCore.pyqtSlot(int, str, object)
    def _psd_done(self, ch: int, psd_mode: str, payload):
        """
        Slot called when an off-thread PSDTask finishes.

        Parameters
        ----------
        ch : int
            Channel index.
        psd_mode : {"SSB", "DSB"}
            The PSD mode that was computed.
        payload : object
            Task result data. For "SSB", it's (freq_i, psd_i, psd_q, psd_mag, ...).
            For "DSB", it's (freq_dsb, psd_dsb).
        """
        self.psd_workers[ch][psd_mode[0]] = False
        if ch not in self.curves:
            return

        if psd_mode == "SSB":
            if "S" not in self.curves[ch]:
                return
            c = self.curves[ch]["S"]
            freq_i, psd_i, psd_q, psd_m, _, _, _ = payload
            if c["I"].isVisible():
                c["I"].setData(freq_i, psd_i)
            if c["Q"].isVisible():
                c["Q"].setData(freq_i, psd_q)
            if c["Mag"].isVisible():
                c["Mag"].setData(freq_i, psd_m)

        else:  # "DSB"
            if "D" not in self.curves[ch]:
                return
            c = self.curves[ch]["D"]
            freq_dsb, psd_dsb = payload
            c["Cmplx"].setData(freq_dsb, psd_dsb)

    # ───────────────────────── UI callbacks ─────────────────────────
    def _toggle_pause(self):
        """
        Pause or resume real-time data updates. When paused, new packets
        are discarded to prevent stale data accumulation.
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
            True if dark mode is desired, False for light mode.
        """
        self.dark_mode = checked
        self._build_layout()

    def _toggle_real_units(self, checked: bool):
        """
        Toggle between raw counts and real-units (volts/dBm/dBc).

        Parameters
        ----------
        checked : bool
            True if real units are enabled, False for raw counts.
        """
        self.real_units = checked
        if checked:
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Real Units On")
            msg.setText(
                "The global conversion to real units (Volts, dBm) is approximate and normalized to a typical value in the lower 1st Nyquist region.\n"
                "PSD plots are corrected for the local CIC1 and CIC2 passband droop and will match the output of py_get_samples. The 'RAW FFT' plot is not droop-corrected."
            )
            msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            msg.exec()
        self._build_layout()

    def _psd_ref_changed(self):
        """
        Switch between absolute (dBm) and relative (dBc) scaling for
        PSD plots when Real Units is enabled.
        """
        self.psd_absolute = self.rb_psd_abs.isChecked()
        self._build_layout()

    def _update_channels(self):
        """
        Parse the channel specification string from self.e_ch,
        re-initialize buffers and layout if channels changed.
        """
        new_ch = self._parse_channels(self.e_ch.text())
        if new_ch != self.channels:
            self.channels = new_ch
            self._init_buffers()
            self._build_layout()

    def _change_buffer(self):
        """
        Update the ring buffer size from self.e_buf if it differs
        from the current size, and re-initialize buffers.
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
        Cleanly shut down the background receiver and stop the GUI timer
        before closing the application.

        Parameters
        ----------
        ev : QCloseEvent
            The close event.
        """
        self.timer.stop()
        self.receiver.stop()
        self.receiver.wait()
        super().closeEvent(ev)
        ev.accept()

    # ───────────────────────── Static helpers ─────────────────────────
    @staticmethod
    def _parse_channels(txt: str) -> List[int]:
        """
        Convert a comma-separated channel string into a list of integers,
        clamped to the valid range of streamer.NUM_CHANNELS.

        Parameters
        ----------
        txt : str
            Comma-separated list of channel indices.

        Returns
        -------
        List[int]
            List of valid channels (default [1] if none parsed).
        """
        out = []
        for tok in txt.split(","):
            tok = tok.strip()
            if tok.isdigit():
                v = int(tok)
                if 1 <= v <= streamer.NUM_CHANNELS:
                    out.append(v)
        return out or [1]


# ───────────────────────── Command‑line entry‑points ─────────────────────────
def main():
    """
    Entry point for command-line usage. Parses CLI arguments,
    constructs a QApplication, and launches Periscope in blocking mode.
    """
    ap = argparse.ArgumentParser(
        description=(
            "Periscope – noise segments spinner (only if S or D) + PSD scale "
            "hidden if RealUnits=off"
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
        ap.error("Density‑dot size must be ≥1 pixel.")

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
    Programmatic entry point for embedding or interactive usage.

    Parameters
    ----------
    hostname : str
        Multicast/UDP host address.
    module : int, optional
        Module number, default is 1.
    channels : str, optional
        Comma-separated string of channels, default "1".
    buf_size : int, optional
        Ring buffer size, default 5000.
    fps : float, optional
        Frames per second (default 30.0).
    density_dot : int, optional
        Dot dilation in pixels for IQ density mode, default 1.
    blocking : bool or None, optional
        If None and in IPython with an active Qt loop, the method is non-blocking;
        otherwise blocking.

    Returns
    -------
    Periscope or (Periscope, QApplication)
        If blocking, returns the Periscope instance after the Qt event loop exits.
        If non-blocking, returns a tuple of (Periscope, QApplication).
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
