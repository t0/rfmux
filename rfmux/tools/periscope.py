#!/usr/bin/env -S uv run
"""
Periscope – Real‑Time Multi‑Pane Viewer
=======================================

A real-time data visualizer using Qt6 + pyqtgraph, off-loading heavy computations
to a bounded QThreadPool. It shows:
- Time-domain (T)
- IQ domain (density or scatter)
- FFT (F)
- Single-Sideband PSD (S) [optional]
- Double-Sideband PSD (D) [optional]

Key Features
------------
- IQ histogram/scatter handled in IQTask (thread pool).
- Built-in FFT mode from pyqtgraph, using (time, amplitude).
- Concurrency bounded by CPU core count.
- SSB PSD (I, Q, Mag) and DSB PSD (I + jQ) also off-loaded to worker threads,
  properly normalized for PSD in Counts²/Hz.

Dependencies
------------
- Python >= 3.9
- NumPy
- PyQt6 (official wheels on PyPI)
- pyqtgraph
- rfmux (in-tree package; provides `streamer`)

Optional
--------
- SciPy (for Gaussian smoothing in IQ density plots)

"""

import argparse
import math
import os
import queue
import socket
import sys
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import (
    QRunnable,
    QThreadPool,
    pyqtSignal,
    QObject
)
from PyQt6.QtGui import QFont, QIntValidator
import pyqtgraph as pg

from .. import streamer

# ────────────────────────── Global Settings ──────────────────────────
pg.setConfigOptions(useOpenGL=False, antialias=False)

LINE_WIDTH         = 2
UI_FONT_SIZE       = 12
DENSITY_GRID       = 512
DENSITY_DOT_SIZE   = 1
SMOOTH_SIGMA       = 1.3
LOG_COMPRESS       = True

SCATTER_POINTS     = 1_000
SCATTER_SIZE       = 5

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None
    SMOOTH_SIGMA = 0.0

# ───────────────────── IPython / Qt Helpers ─────────────────────────
def _get_ipython():
    """
    Attempt to get the active IPython instance (if any).
    """
    try:
        from IPython import get_ipython
        return get_ipython()
    except ImportError:
        return None

def _is_qt_event_loop_running() -> bool:
    """
    Check if an active IPython Qt event loop is running.
    """
    ip = _get_ipython()
    return bool(ip and getattr(ip, "active_eventloop", None) == "qt")

def _is_running_inside_ipython() -> bool:
    """
    Determine whether the code is running inside an IPython environment.
    """
    return _get_ipython() is not None

# ───────────────────────── Lock‑free Ring Buffer ─────────────────────
class Circular:
    """
    Fixed-size lock-free ring buffer using a contiguous NumPy slice.
    
    Parameters
    ----------
    size : int
        Capacity of the ring buffer.
    dtype : type
        NumPy data type for internal storage (default: float).
    """
    def __init__(self, size: int, dtype=float) -> None:
        self.N = size
        self.buf = np.zeros(size * 2, dtype=dtype)
        self.ptr = 0
        self.count = 0

    def add(self, value):
        """
        Add a single value to the ring buffer.
        """
        self.buf[self.ptr] = value
        self.buf[self.ptr + self.N] = value
        self.ptr = (self.ptr + 1) % self.N
        self.count = min(self.count + 1, self.N)

    def data(self) -> np.ndarray:
        """
        Return the contiguous slice of valid data.
        """
        if self.count < self.N:
            return self.buf[: self.count]
        return self.buf[self.ptr : self.ptr + self.N]


class ClickableViewBox(pg.ViewBox):
    """
    Custom ViewBox that opens a coordinate read-out dialog on double-click.
    """

    def mouseDoubleClickEvent(self, ev):
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
        x_val = 10**pt_view.x() if log_x else pt_view.x()
        y_val = 10**pt_view.y() if log_y else pt_view.y()

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
    Receives multicast packets and pushes them into a thread-safe queue.

    Parameters
    ----------
    host : str
        Multicast group or IP address.
    module : int
        Target module ID (1-based).
    """
    def __init__(self, host: str, module: int) -> None:
        super().__init__()
        self.module_id = module
        self.queue = queue.Queue()
        self.sock = streamer.get_multicast_socket(host)
        self.sock.settimeout(0.2)

    def run(self):
        """
        Continuously receive data from the multicast socket and enqueue.
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
        Signal the thread to stop and close the socket.
        """
        self.requestInterruption()
        try:
            self.sock.close()
        except OSError:
            pass


class IQSignals(QObject):
    """
    PyQt signals container for IQTask results (IQ scatter/density).
    """
    done = pyqtSignal(int, str, object)
    # (channel, mode, payload)


class IQTask(QRunnable):
    """
    Off-thread worker for IQ histogram or scatter computations in a QThreadPool.
    
    Parameters
    ----------
    ch : int
        Channel index.
    I, Q : np.ndarray
        Time-domain I/Q data.
    dot_px : int
        Dot size in pixels for density plots.
    mode : str
        Either 'density' or 'scatter'.
    signals : IQSignals
        Signal object for emitting computed results.
    """
    def __init__(self, ch, I, Q, dot_px, mode, signals: IQSignals):
        super().__init__()
        self.ch = ch
        self.dot_px = dot_px
        self.mode = mode  # "density" or "scatter"
        self.signals = signals
        # Copy data to ensure safe usage off-thread:
        self.I = I.copy()
        self.Q = Q.copy()

    def run(self):
        """
        Perform the requested IQ computation off-thread, then emit the result.
        """
        if len(self.I) < 2:
            # Not enough samples:
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
                # Degenerate case
                payload = (hist.astype(np.uint8), (Imin, Imax, Qmin, Qmax))
                self.signals.done.emit(self.ch, self.mode, payload)
                return

            ix = ((self.I - Imin) * (g - 1) / (Imax - Imin)).astype(np.intp)
            qy = ((self.Q - Qmin) * (g - 1) / (Qmax - Qmin)).astype(np.intp)

            r = self.dot_px // 2
            if self.dot_px == 1:
                # Direct
                np.add.at(hist, (qy, ix), 1)
            else:
                # Thicker point
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        ys, xs = qy + dy, ix + dx
                        mask = (0 <= ys) & (ys < g) & (0 <= xs) & (xs < g)
                        np.add.at(hist, (ys[mask], xs[mask]), 1)

            if gaussian_filter and SMOOTH_SIGMA > 0:
                hist = gaussian_filter(hist.astype(np.float32), SMOOTH_SIGMA, mode="nearest")
            if LOG_COMPRESS:
                hist = np.log1p(hist, out=hist.astype(np.float32))
            if hist.max() > 0:
                hist = (hist * (255.0 / hist.max())).astype(np.uint8)
            payload = (hist, (Imin, Imax, Qmin, Qmax))

        else:  # scatter
            N = len(self.I)
            if N > SCATTER_POINTS:
                idx = np.linspace(0, N - 1, SCATTER_POINTS, dtype=np.intp)
            else:
                idx = np.arange(N, dtype=np.intp)

            xs, ys = self.I[idx], self.Q[idx]
            rel = idx / (idx.max() if idx.size else 1)
            colors = pg.colormap.get("turbo").map(rel.astype(np.float32), mode="byte")
            payload = (xs, ys, colors)

        self.signals.done.emit(self.ch, self.mode, payload)


# ───────────────────── PSD Worker for SSB & DSB ─────────────────────
class PSDSignals(QObject):
    """
    PyQt signals container for PSDTask results (SSB or DSB PSD).
    """
    done = pyqtSignal(int, str, object)
    # (channel, mode, payload)


class PSDTask(QRunnable):
    """
    Off-thread worker to compute single-sideband (SSB) or double-sideband (DSB)
    PSD for a given channel's data. SSB PSD is done individually for I, Q, and M;
    DSB PSD is done once from the complex I + jQ.

    Parameters
    ----------
    ch : int
        Channel index.
    I, Q, M : np.ndarray
        Time-domain I/Q magnitude arrays.
    t : np.ndarray
        Time array (seconds).
    mode : str
        Either 'SSB' or 'DSB'.
    signals : PSDSignals
        Signal object for emitting computed results.
    """
    def __init__(self, ch: int, I: np.ndarray, Q: np.ndarray, M: np.ndarray,
                 t: np.ndarray, mode: str, signals: PSDSignals):
        super().__init__()
        self.ch = ch
        self.I = I.copy()
        self.Q = Q.copy()
        self.M = M.copy()
        self.t = t.copy()
        self.mode = mode  # "SSB" or "DSB"
        self.signals = signals

    def run(self):
        """
        Compute the PSD and emit results. 
        """
        # Not enough data check:
        if len(self.t) < 2 or len(self.I) < 2:
            if self.mode == "SSB":
                # Return empty for I, Q, M
                payload = ([], [], [], [], [], [], [])
            else:
                # Return empty for freq, PSD
                payload = ([], [])
            self.signals.done.emit(self.ch, self.mode, payload)
            return

        dt = np.mean(np.diff(self.t))
        fs = 1.0 / dt

        if self.mode == "SSB":
            # SSB PSD for I, Q, M -> real signals => single-sided
            fI, psdI = self._compute_ssb_psd(self.I, fs)
            fQ, psdQ = self._compute_ssb_psd(self.Q, fs)
            fM, psdM = self._compute_ssb_psd(self.M, fs)
            # All three freq arrays should be identical, so we can just return one.
            payload = (fI, psdI, psdQ, psdM, fQ, fM, fs)
            # We'll store them for clarity: 
            #   freq = fI
            #   PSD_I = psdI, PSD_Q = psdQ, PSD_M = psdM
        else:
            # DSB PSD from complex: c = I + jQ
            c = self.I + 1j*self.Q
            f, psd = self._compute_dsb_psd(c, fs)
            payload = (f, psd)

        self.signals.done.emit(self.ch, self.mode, payload)

    @staticmethod
    def _compute_ssb_psd(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute single-sided real PSD using a standard approach:
          PSD = (2/(fs*N)) * |FFT(rfft(x))|^2
        (except DC and possibly the Nyquist, which doesn't get doubled).
        """
        n = len(x)
        xf = np.fft.rfft(x, n=n)
        # raw power
        pwr = (np.abs(xf)**2) / (fs * n)
        # single-sided factor of 2 except DC or (Nyquist if n even)
        # freq array
        freq = np.fft.rfftfreq(n, d=1.0/fs)

        # Double everything except DC (index=0) and possibly last bin if n is even
        # (Nyquist freq is only one-sided).
        if n % 2 == 0:
            # even length -> last bin is f_nyquist
            pwr[1:-1] *= 2.0
        else:
            # odd length -> no exact Nyquist bin
            pwr[1:] *= 2.0

        return freq, pwr

    @staticmethod
    def _compute_dsb_psd(xc: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute double-sided PSD from a complex timeseries:
          PSD = (1/(fs*N)) * |FFT(xc)|^2, full frequency range
        Then shift to [-fs/2, +fs/2].
        """
        n = len(xc)
        xf = np.fft.fft(xc, n=n)
        pwr = (np.abs(xf)**2) / (fs * n)

        freq = np.fft.fftfreq(n, d=1.0/fs)
        # Shift so freq is ascending from negative to positive:
        freq = np.fft.fftshift(freq)
        pwr = np.fft.fftshift(pwr)

        return freq, pwr


class Periscope(QtWidgets.QMainWindow):
    """
    Multi-pane viewer for:
    - Time (T)
    - IQ (density or scatter)
    - FFT (F)
    - SSB PSD (S) [optional]
    - DSB PSD (D) [optional]

    The latter two are done off-thread with proper PSD normalization.

    Parameters
    ----------
    host : str
        Multicast group or IP address.
    module : int
        Target module ID (1-based).
    chan_str : str
        Comma-separated string of channel numbers.
    buf_size : int
        Number of samples to store in ring buffers.
    refresh_ms : int
        UI refresh interval in milliseconds.
    dot_px : int
        Dot size in pixels for IQ density plot.
    """
    def __init__(
        self,
        host,
        module,
        chan_str="1",
        buf_size=5_000,
        refresh_ms=33,
        dot_px=DENSITY_DOT_SIZE
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

        # IQ concurrency
        self.iq_workers: Dict[int, bool] = {}
        self.iq_signals = IQSignals()
        self.iq_signals.done.connect(self._iq_done)

        # PSD concurrency
        self.psd_workers: Dict[int, Dict[str, bool]] = {}
        # For each channel -> { "S": False, "D": False }
        self.psd_signals = PSDSignals()
        self.psd_signals.done.connect(self._psd_done)

        # Color map (turbo) for IQ density
        cmap = pg.colormap.get("turbo")
        lut_rgb = cmap.getLookupTable(0.0, 1.0, 255)
        self.lut = np.vstack([
            np.zeros((1, 4), np.uint8),
            np.hstack([lut_rgb, 255 * np.ones((255, 1), np.uint8)])
        ])

        # Background UDP receiver
        self.receiver = UDPReceiver(host, module)
        self.receiver.start()

        # Thread pool
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(os.cpu_count() or 1)

        # Dark mode preference
        self.dark_mode = True

        # Build UI
        self._build_ui(chan_str)
        self._init_buffers()
        self._build_layout()

        # Timer for periodic GUI updates
        self.timer = QtCore.QTimer(singleShot=False)
        self.timer.timeout.connect(self._update_gui)
        self.timer.start(self.refresh_ms)

        self.setWindowTitle("Periscope – Real‑Time Viewer (TOD, IQ, FFT, SSB PSD, DSB PSD)")

    def _build_ui(self, chan_str: str):
        """
        Create top-level UI elements and layout, including new PSD checkboxes.
        """
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        # Title
        title = QtWidgets.QLabel(f"CRS: {self.host}    Module: {self.module}")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        ft = title.font()
        ft.setPointSize(16)
        title.setFont(ft)
        vbox.addWidget(title)

        # Top bar
        bar = QtWidgets.QHBoxLayout()
        vbox.addLayout(bar)

        bar.addWidget(QtWidgets.QLabel("Channels:"))
        self.e_ch = QtWidgets.QLineEdit(chan_str)
        self.e_ch.returnPressed.connect(self._update_channels)
        bar.addWidget(self.e_ch)
        btn_update = QtWidgets.QPushButton("Update", clicked=self._update_channels)
        bar.addWidget(btn_update)

        bar.addWidget(QtWidgets.QLabel("Buffer:"))
        self.e_buf = QtWidgets.QLineEdit(str(self.N))
        self.e_buf.setValidator(QIntValidator(10, 1_000_000, self))
        self.e_buf.editingFinished.connect(self._change_buffer)
        bar.addWidget(self.e_buf)

        self.b_pause = QtWidgets.QPushButton("Pause", clicked=self._toggle_pause)
        bar.addWidget(self.b_pause)

        self.cb_time = QtWidgets.QCheckBox("TOD", checked=True)
        self.cb_iq   = QtWidgets.QCheckBox("IQ",   checked=True)
        self.cb_fft  = QtWidgets.QCheckBox("FFT",  checked=True)
        self.cb_ssb  = QtWidgets.QCheckBox("SSB PSD", checked=False)
        self.cb_dsb  = QtWidgets.QCheckBox("DSB PSD", checked=False)

        # Connect them to layout rebuild
        for cb in (self.cb_time, self.cb_iq, self.cb_fft, self.cb_ssb, self.cb_dsb):
            cb.toggled.connect(self._build_layout)

        self.rb_density = QtWidgets.QRadioButton("Density", checked=True)
        self.rb_scatter = QtWidgets.QRadioButton("Scatter")
        group = QtWidgets.QButtonGroup(self)
        for rb in (self.rb_density, self.rb_scatter):
            group.addButton(rb)
            rb.toggled.connect(self._build_layout)

        bar.addWidget(self.cb_time)
        bar.addWidget(self.cb_iq)
        bar.addWidget(self.rb_density)
        bar.addWidget(self.rb_scatter)
        bar.addWidget(self.cb_fft)
        bar.addWidget(self.cb_ssb)
        bar.addWidget(self.cb_dsb)

        # Dark Mode checkbox
        self.cb_dark = QtWidgets.QCheckBox("Dark Mode", checked=True)
        self.cb_dark.toggled.connect(self._toggle_dark_mode)
        bar.addWidget(self.cb_dark)

        # Hide scatter/density toggles if IQ disabled
        self.rb_density.setVisible(self.cb_iq.isChecked())
        self.rb_scatter.setVisible(self.cb_iq.isChecked())
        self.cb_iq.toggled.connect(self.rb_density.setVisible)
        self.cb_iq.toggled.connect(self.rb_scatter.setVisible)

        bar.addStretch()

        # Container for plots
        self.container = QtWidgets.QWidget()
        vbox.addWidget(self.container)
        self.grid = QtWidgets.QGridLayout(self.container)
        self.setStatusBar(QtWidgets.QStatusBar())

    def _init_buffers(self):
        """
        Initialize ring buffers for each channel (I, Q, M, t).
        Also init psd_workers[ch].
        """
        self.buf = {}
        self.tbuf = {}
        for ch in self.channels:
            self.buf[ch] = {k: Circular(self.N) for k in ("I", "Q", "M")}
            self.tbuf[ch] = Circular(self.N)
            # PSD concurrency flags
            self.psd_workers[ch] = {"S": False, "D": False}

    def _build_layout(self):
        """
        Build the plot layout for each selected channel and each enabled mode.
        """
        # Clear old layout
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        # Gather modes
        modes = []
        if self.cb_time.isChecked():
            modes.append("T")   # Time
        if self.cb_iq.isChecked():
            modes.append("IQ")  # IQ density/scatter
        if self.cb_fft.isChecked():
            modes.append("F")   # FFT
        if self.cb_ssb.isChecked():
            modes.append("S")   # SSB PSD
        if self.cb_dsb.isChecked():
            modes.append("D")   # DSB PSD

        self.plots  = {}
        self.curves = {}
        font = QFont()
        font.setPointSize(UI_FONT_SIZE)

        for row, ch in enumerate(self.channels):
            self.plots[ch]  = {}
            self.curves[ch] = {}

            for col, mode in enumerate(modes):
                vb = ClickableViewBox()
                if mode == "T":
                    pw = pg.PlotWidget(viewBox=vb, title=f"Ch {ch} – Time")
                    pw.setLabel("left", "Amplitude (Counts)")
                elif mode == "IQ":
                    pw = pg.PlotWidget(viewBox=vb, title=f"Ch {ch} – IQ")
                    pw.setLabel("bottom", "I")
                    pw.setLabel("left", "Q")
                    pw.getViewBox().setAspectLocked(True)
                elif mode == "F":
                    pw = pg.PlotWidget(viewBox=vb, title=f"Ch {ch} – FFT")
                    pw.setLogMode(x=True, y=True)
                    pw.setLabel("bottom", "Freq (Hz)")
                    pw.setLabel("left", "Raw FFT (Counts)")
                elif mode == "S":
                    pw = pg.PlotWidget(viewBox=vb, title=f"Ch {ch} – SSB PSD")
                    pw.setLogMode(x=True, y=True)
                    pw.setLabel("bottom", "Freq (Hz)")
                    pw.setLabel("left", "PSD (Counts²/Hz)")
                else:  # "D"
                    pw = pg.PlotWidget(viewBox=vb, title=f"Ch {ch} – DSB PSD")
                    pw.setLogMode(x=False, y=True)
                    pw.setLabel("bottom", "Freq (Hz)")
                    pw.setLabel("left", "PSD (Counts²/Hz)")

                self._apply_plot_theme(pw)
                pw.showGrid(x=True, y=True, alpha=0.3)
                self.plots[ch][mode] = pw
                self.grid.addWidget(pw, row, col)

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
                        for k, c in zip(
                            ("I", "Q", "Mag"),
                            ("#1f77b4", "#ff7f0e", "#2ca02c")
                        )
                    }
                    self._fade_hidden_entries(legend, hide_labels=("I", "Q"))
                    self._make_legend_clickable(legend)

                elif mode == "F":
                    legend = pw.addLegend(offset=(30, 10))
                    self.curves[ch]["F"] = {
                        k: pw.plot(pen=pg.mkPen(c, width=LINE_WIDTH), name=k)
                        for k, c in zip(
                            ("I", "Q", "Mag"),
                            ("#1f77b4", "#ff7f0e", "#2ca02c")
                        )
                    }
                    # Enable built-in FFT mode
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
                    # SSB PSD => 3 curves: I, Q, M
                    legend = pw.addLegend(offset=(30, 10))
                    self.curves[ch]["S"] = {
                        k: pw.plot(pen=pg.mkPen(c, width=LINE_WIDTH), name=k)
                        for k, c in zip(
                            ("I", "Q", "Mag"),
                            ("#1f77b4", "#ff7f0e", "#2ca02c")
                        )
                    }
                    self._make_legend_clickable(legend)

                else:  # "D" => DSB PSD => 1 curve: "Cmplx"
                    legend = pw.addLegend(offset=(30, 10))
                    curve = pw.plot(pen=pg.mkPen("#bcbd22", width=LINE_WIDTH), name="Complex Dual-Sideband")
                    self.curves[ch]["D"] = {"Cmplx": curve}
                    self._make_legend_clickable(legend)

    def _apply_plot_theme(self, pw: pg.PlotWidget):
        """
        Apply either a dark or light background (and matching axis pen) to PlotWidget.
        """
        if self.dark_mode:
            pw.setBackground("k")
            for axis in ("left", "bottom", "right", "top"):
                ax = pw.getPlotItem().getAxis(axis)
                if ax:
                    ax.setPen("w")
                    ax.setTextPen("w")
        else:
            pw.setBackground("w")
            for axis in ("left", "bottom", "right", "top"):
                ax = pw.getPlotItem().getAxis(axis)
                if ax:
                    ax.setPen("k")
                    ax.setTextPen("k")

    @staticmethod
    def _fade_hidden_entries(legend, hide_labels):
        """
        Dim specified legend entries (e.g. 'I', 'Q') to indicate they're optional.
        """
        for sample, label in legend.items:
            txt = label.labelItem.toPlainText() if hasattr(label, "labelItem") else ""
            if txt in hide_labels:
                sample.setOpacity(0.3)
                label.setOpacity(0.3)

    @staticmethod
    def _make_legend_clickable(legend):
        """
        Allow legend items to toggle curve visibility on click.
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

    def _update_gui(self):
        """
        Periodic UI refresh, reading new packets, updating time & FFT,
        scheduling off-thread IQ tasks and PSD tasks if needed.
        """
        if self.paused:
            # Discard queued packets when paused
            while not self.receiver.queue.empty():
                self.receiver.queue.get()
            return

        # Ingest new packets
        while not self.receiver.queue.empty():
            pkt = self.receiver.queue.get()
            self.pkt_cnt += 1

            ts = pkt.ts
            # Optional offset if desired:
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

        # Update each channel's time, IQ, FFT
        for ch in self.channels:
            I = self.buf[ch]["I"].data()
            Q = self.buf[ch]["Q"].data()
            M = self.buf[ch]["M"].data()
            t = self.tbuf[ch].data()

            # Time domain
            if "T" in self.curves[ch]:
                c = self.curves[ch]["T"]
                c["Mag"].setData(t, M)
                if c["I"].isVisible():
                    c["I"].setData(t, I)
                if c["Q"].isVisible():
                    c["Q"].setData(t, Q)

            # IQ (off-thread histogram/scatter)
            if "IQ" in self.curves[ch] and not self.iq_workers.get(ch, False):
                mode = self.curves[ch]["IQ"]["mode"]
                task = IQTask(ch, I, Q, self.dot_px, mode, self.iq_signals)
                self.iq_workers[ch] = True
                self.pool.start(task)

            # FFT (built-in)
            if "F" in self.curves[ch]:
                c = self.curves[ch]["F"]
                c["I"].setData(t, I, fftMode=True)
                c["Q"].setData(t, Q, fftMode=True)
                c["Mag"].setData(t, M, fftMode=True)

            # SSB PSD
            if "S" in self.curves[ch]:
                # Only run if not already in progress
                if not self.psd_workers[ch]["S"]:
                    task = PSDTask(ch, I, Q, M, t, "SSB", self.psd_signals)
                    self.psd_workers[ch]["S"] = True
                    self.pool.start(task)

            # DSB PSD
            if "D" in self.curves[ch]:
                if not self.psd_workers[ch]["D"]:
                    task = PSDTask(ch, I, Q, M, t, "DSB", self.psd_signals)
                    self.psd_workers[ch]["D"] = True
                    self.pool.start(task)

        # Status bar: show updated FPS/packets rate
        now = time.time()
        if now - self.t_last >= 1.0:
            fps = self.frame_cnt / (now - self.t_last)
            pps = self.pkt_cnt / (now - self.t_last)
            self.statusBar().showMessage(f"FPS {fps:.1f} | Packets/s {pps:.1f}")
            self.frame_cnt = 0
            self.pkt_cnt = 0
            self.t_last = now

    @QtCore.pyqtSlot(int, str, object)
    def _iq_done(self, ch: int, task_mode: str, payload):
        """
        Handle completion of off-thread IQ calculations.
        """
        self.iq_workers[ch] = False
        if ch not in self.curves or "IQ" not in self.curves[ch]:
            return

        pane = self.curves[ch]["IQ"]
        local_mode = pane["mode"]
        if local_mode != task_mode:
            return  # stale

        item = pane["item"]
        if task_mode == "density":
            hist, (Imin, Imax, Qmin, Qmax) = payload
            item.setImage(hist, levels=(0, 255), autoLevels=False)
            item.setRect(QtCore.QRectF(Imin, Qmin, Imax - Imin, Qmax - Qmin))
        else:
            xs, ys, colors = payload
            item.setData(xs, ys, brush=colors, pen=None, size=SCATTER_SIZE)

    @QtCore.pyqtSlot(int, str, object)
    def _psd_done(self, ch: int, psd_mode: str, payload):
        """
        Handle completion of off-thread PSD computations (SSB or DSB).
        """
        self.psd_workers[ch][psd_mode[0]] = False  # "S" or "D"
        if ch not in self.curves:
            return

        if psd_mode == "SSB":
            if "S" not in self.curves[ch]:
                return
            c = self.curves[ch]["S"]
            # payload = (freqI, psdI, psdQ, psdM, freqQ, freqM, fs)
            freqI, psdI, psdQ, psdM, _, _, _ = payload
            # Update curves
            c["I"].setData(freqI, psdI)
            if c["Q"].isVisible():
                c["Q"].setData(freqI, psdQ)
            if c["Mag"].isVisible():
                c["Mag"].setData(freqI, psdM)

        else:  # "DSB"
            if "D" not in self.curves[ch]:
                return
            c = self.curves[ch]["D"]
            # payload = (freq, psd)
            freq, psd = payload
            c["Cmplx"].setData(freq, psd)

    def _toggle_pause(self):
        """
        Toggle paused/resumed state.
        """
        self.paused = not self.paused
        self.b_pause.setText("Resume" if self.paused else "Pause")

    def _toggle_dark_mode(self, checked: bool):
        """
        Toggle dark vs. light theme.
        Rebuilds the layout so new PlotWidgets get the updated style.
        """
        self.dark_mode = checked
        self._build_layout()

    def _update_channels(self):
        """
        Update selected channels from user input.
        """
        new_channels = self._parse_channels(self.e_ch.text())
        if new_channels != self.channels:
            self.channels = new_channels
            self._init_buffers()
            self._build_layout()

    def _change_buffer(self):
        """
        Update the ring buffer size if user changes the buffer input.
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
        Cleanup on window close: stop timer, receiver thread, etc.
        """
        self.timer.stop()
        self.receiver.stop()
        self.receiver.wait()
        super().closeEvent(ev)
        ev.accept()

    @staticmethod
    def _parse_channels(txt: str) -> List[int]:
        """
        Parse a comma-separated channel spec into a list of valid channel indices.
        """
        out = []
        for tok in txt.split(","):
            tok = tok.strip()
            if tok.isdigit():
                v = int(tok)
                if 1 <= v <= streamer.NUM_CHANNELS:
                    out.append(v)
        return out or [1]


def main():
    """
    Entry point for command-line usage of Periscope.
    """
    ap = argparse.ArgumentParser(
        description="Periscope – real-time viewer with optional SSB/DSB PSD computations."
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
        warnings.warn("FPS > 30 may be unnecessary", RuntimeWarning)
    if args.density_dot < 1:
        ap.error("Density-dot size must be ≥ 1 pixel.")

    refresh_ms = int(round(1000.0 / args.fps))
    app = QtWidgets.QApplication(sys.argv[:1])

    win = Periscope(
        host=args.hostname,
        module=args.module,
        chan_str=args.channels,
        buf_size=args.num_samples,
        refresh_ms=refresh_ms,
        dot_px=args.density_dot
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
    blocking: bool | None = None
):
    """
    Launch Periscope in IPython or stand-alone mode.
    
    Parameters
    ----------
    hostname : str
        Host or multicast address.
    module : int
        Module ID (1-based).
    channels : str
        Comma-separated channel string.
    buf_size : int
        Number of samples to buffer.
    fps : float
        Frames per second for UI refresh.
    density_dot : int
        Dot size in pixels for IQ density mode.
    blocking : bool | None
        If None, it defaults to non-blocking if a Qt event loop is active in IPython.
    """
    ip = _get_ipython()
    qt_eventloop_active = _is_qt_event_loop_running()

    if ip and not qt_eventloop_active:
        ip.run_line_magic("gui", "qt")
        qt_eventloop_active = True

    if blocking is None:
        blocking = not qt_eventloop_active

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    refresh_ms = int(round(1000.0 / fps))

    viewer = Periscope(hostname, module, channels, buf_size, refresh_ms, density_dot)
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
