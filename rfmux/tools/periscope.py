#!/usr/bin/env -S uv run
"""
Periscope – real‑time multi‑pane viewer
======================================

Real‑time data visualiser using *Qt 6* + *pyqtgraph*.

High‑level architecture
-----------------------
    UDPReceiver   → background multicast, packets → queue
    Circular      → lock‑free ring buffers
    FFTWorker     → off‑thread amplitude‑spectral‑density
    ClickableView → double‑click coordinate read‑out
    MainWindow    → dynamic layout (Time | IQ | FFT)

Dependencies
------------
* Python ≥ 3.9
* numpy
* PyQt6  (official wheels on PyPI, no system Qt needed)
* pyqtgraph
* rfmux    (in‑tree package; provides ``streamer``)

Optional (adds features but not required):
* scipy      – Gaussian smoothing for IQ‑density maps
* IPython    – auto‑integration with Jupyter / IPython shells

Installation
------------
Using **pip** (wheel packages, no compiler required)::

    # core runtime
    pip install numpy PyQt6 pyqtgraph

    # optional extras
    pip install scipy ipython

Or everything in one line::

    pip install "numpy" "PyQt6" "pyqtgraph" "scipy" "ipython"

Using **conda** (Conda‑Forge channel)::

    conda install -c conda-forge pyqt pyqtgraph numpy scipy ipython

Quick‑start
-----------
1. Ensure the *rfmux* repository (this file) is on ``PYTHONPATH``::

       pip install -e .

2. Run from a shell::

       python -m rfmux.tools.periscope rfmux0009.local

   …or from a Python/Jupyter session::

       import rfmux.tools.periscope as ps
       ps.launch("rfmux0009.local")        # auto‑handles %gui qt

"""

# ──────────────────────── stdlib / third‑party imports ───────────────────────
from __future__ import annotations

import argparse
import math
import queue
import socket
import sys
import time
import warnings
from typing import Dict, List

import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QFont, QIntValidator
import pyqtgraph as pg
from rfmux import streamer

# ─────────────────────────────── global settings ─────────────────────────────
pg.setConfigOptions(useOpenGL=False, antialias=False)

# ────────────────────────────── module‑wide constants ────────────────────────
LINE_WIDTH       = 2      # stroke width for plots
UI_FONT_SIZE     = 12     # axis / title font (pt)

DENSITY_GRID     = 512    # IQ‑density heat‑map resolution
DENSITY_DOT_SIZE = 1      # dot radius (px) in density mode
SMOOTH_SIGMA     = 1.3    # Gaussian σ for density smoothing
LOG_COMPRESS     = True   # apply log‑compression to density map

FFT_THROTTLE     = 1      # GUI frames skipped between FFT updates
SCATTER_POINTS   = 1_000  # points plotted in scatter mode
SCATTER_SIZE     = 7      # scatter marker diameter (px)

# optional SciPy smooth
try:
    from scipy.ndimage import gaussian_filter
except ImportError:        # noqa: WPS440
    gaussian_filter = None
    SMOOTH_SIGMA = 0

# ────────────────────────────── IPython helpers ──────────────────────────────
def _get_ipython():
    try:
        from IPython import get_ipython
        return get_ipython()
    except Exception:  # pragma: no cover
        return None


def _qt_eventloop_running() -> bool:
    ip = _get_ipython()
    if ip is None:
        return False
    if getattr(ip, "active_eventloop", None) == "qt":
        return True
    return str(getattr(ip, "eventloop", "")).lower() == "qt"


def _running_inside_ipython() -> bool:
    return _get_ipython() is not None


# ────────────────────────────────── utilities ────────────────────────────────
class Circular:
    """Fixed‑size lock‑free ring buffer returning a contiguous NumPy slice."""

    def __init__(self, size: int, dtype=float) -> None:
        self.N = size
        self.buf = np.zeros(size * 2, dtype=dtype)
        self.ptr = 0
        self.count = 0

    # API ------------------------------------------------------------------
    def add(self, value):
        self.buf[self.ptr] = value
        self.buf[self.ptr + self.N] = value  # mirror for contiguous slice
        self.ptr = (self.ptr + 1) % self.N
        self.count = min(self.count + 1, self.N)

    def data(self) -> np.ndarray:
        return (
            self.buf[self.ptr : self.ptr + self.N]
            if self.count == self.N
            else self.buf[: self.count]
        )


class ClickableViewBox(pg.ViewBox):
    """ViewBox that opens a coordinate read‑out dialog on double‑click."""

    def mouseDoubleClickEvent(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            super().mouseDoubleClickEvent(ev)
            return

        pt_view = self.mapSceneToView(ev.scenePos())

        # axis titles
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

        # undo log scaling if needed
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


# ──────────────────────────── background workers ───────────────────────────
class UDPReceiver(QtCore.QThread):
    """Receive multicast packets and push them into a thread‑safe queue."""

    def __init__(self, host: str, module: int) -> None:
        super().__init__()
        self.module_id = module
        self.queue = queue.Queue()
        self.sock = streamer.get_multicast_socket(host)
        self.sock.settimeout(0.2)  # wake up regularly to check interruption

    def run(self):
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
        self.requestInterruption()
        try:
            self.sock.close()
        except OSError:
            pass


class FFTWorker(QtCore.QThread):
    """Compute ASD in a worker thread and emit the result."""

    done = QtCore.pyqtSignal(int, np.ndarray, np.ndarray, np.ndarray, np.ndarray)

    def __init__(self, ch, I_view, Q_view, M_view, t_view) -> None:
        super().__init__()
        self.ch, self.Iv, self.Qv, self.Mv, self.tv = ch, I_view, Q_view, M_view, t_view

    # static helpers -------------------------------------------------------
    @staticmethod
    def _asd(x: np.ndarray, fs: float) -> np.ndarray:
        """One‑sided amplitude‑spectral‑density (counts / √Hz)."""
        N = len(x)
        X = np.abs(np.fft.rfft(x))
        X *= np.sqrt(2.0 / fs) / N
        X[0] /= np.sqrt(2.0)
        if N % 2 == 0:
            X[-1] /= np.sqrt(2.0)
        return X

    def run(self):
        I = np.ascontiguousarray(self.Iv)
        Q = np.ascontiguousarray(self.Qv)
        M = np.ascontiguousarray(self.Mv)
        t = np.ascontiguousarray(self.tv)

        fs = 1.0 / np.mean(np.diff(t)) if len(t) > 1 else 625e6 / 256 / 64 / 64
        freqs = np.linspace(0.0, fs / 2, len(I) // 2 + 1)

        fi = self._asd(I, fs)
        fq = self._asd(Q, fs)
        fm = self._asd(M, fs)

        self.done.emit(self.ch, fi, fq, fm, freqs)


# ─────────────────────────────── main window ───────────────────────────────
class Periscope(QtWidgets.QMainWindow):
    """
    GUI window.  Rows = channels, columns = enabled plot types (Time | IQ | FFT).
    """

    # ───────────────────────────── initialisation ─────────────────────────
    def __init__(
        self,
        host: str,
        module: int,
        chan_str: str = "1",
        buf_size: int = 5_000,
        refresh_ms: int = 33,
        dot_px: int = DENSITY_DOT_SIZE,
    ):
        super().__init__()

        # ---- runtime state -------------------------------------------------
        self.host, self.module = host, module
        self.N = buf_size
        self.refresh_ms = refresh_ms
        self.dot_px = max(1, int(dot_px))
        self.channels = self._parse_channels(chan_str)

        self.paused = False
        self.start_time = None
        self.frame_cnt = self.pkt_cnt = 0
        self.t_last = time.time()
        self.fft_workers: Dict[int, FFTWorker | None] = {}

        # ---- colour map ----------------------------------------------------
        cmap = pg.colormap.get("turbo")
        lut_rgb = cmap.getLookupTable(0.0, 1.0, 255)
        self.cmap = cmap
        self.lut = np.vstack(
            [np.zeros((1, 4), np.uint8),
             np.hstack([lut_rgb, 255 * np.ones((255, 1), np.uint8)])]
        )

        # ---- background receiver ------------------------------------------
        self.receiver = UDPReceiver(host, module)
        self.receiver.start()

        # ---- GUI widgets ---------------------------------------------------
        self._build_ui(chan_str)
        self._init_buffers()
        self._build_layout()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_gui)
        self.timer.start(refresh_ms)

        self.setWindowTitle("Periscope – real‑time viewer")

    # ────────────────────────── UI construction helpers ───────────────────
    def _build_ui(self, chan_str: str) -> None:
        """Static widgets (title bar, channel selectors, check‑boxes)."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        # title -------------------------------------------------------------
        title = QtWidgets.QLabel(f"CRS: {self.host}    Module: {self.module}")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        font = title.font()
        font.setPointSize(16)
        title.setFont(font)
        vbox.addWidget(title)

        # top bar -----------------------------------------------------------
        bar = QtWidgets.QHBoxLayout()
        vbox.addLayout(bar)

        # channel selector
        bar.addWidget(QtWidgets.QLabel("Channels:"))
        self.e_ch = QtWidgets.QLineEdit(chan_str)
        self.e_ch.returnPressed.connect(self._update_channels)
        bar.addWidget(self.e_ch)
        bar.addWidget(QtWidgets.QPushButton("Update", clicked=self._update_channels))

        # buffer size
        bar.addWidget(QtWidgets.QLabel("Buffer:"))
        self.e_buf = QtWidgets.QLineEdit(str(self.N))
        self.e_buf.setValidator(QIntValidator(100, 100_000, self))
        self.e_buf.editingFinished.connect(self._change_buffer)
        bar.addWidget(self.e_buf)

        # pause button
        self.b_pause = QtWidgets.QPushButton("Pause", clicked=self._toggle_pause)
        bar.addWidget(self.b_pause)

        # plot‑type checkboxes
        self.cb_time = QtWidgets.QCheckBox("Time", checked=True)
        self.cb_iq = QtWidgets.QCheckBox("IQ", checked=True)
        self.cb_fft = QtWidgets.QCheckBox("FFT", checked=True)

        # IQ style radio buttons
        self.rb_density = QtWidgets.QRadioButton("Density", checked=True)
        self.rb_scatter = QtWidgets.QRadioButton("Scatter")
        self.rb_density.setToolTip(
            "Density heat‑map. Fastest for large buffers."
        )
        self.rb_scatter.setToolTip(
            "Scatter of 1 000 points coloured by age."
        )

        group = QtWidgets.QButtonGroup(self)
        for rb in (self.rb_density, self.rb_scatter):
            group.addButton(rb)
            rb.toggled.connect(self._build_layout)

        bar.addWidget(self.cb_time)
        bar.addWidget(self.cb_iq)
        bar.addWidget(self.rb_density)
        bar.addWidget(self.rb_scatter)
        bar.addWidget(self.cb_fft)

        for cb in (self.cb_time, self.cb_iq, self.cb_fft):
            cb.toggled.connect(self._build_layout)

        self.rb_density.setVisible(self.cb_iq.isChecked())
        self.rb_scatter.setVisible(self.cb_iq.isChecked())
        self.cb_iq.toggled.connect(self.rb_density.setVisible)
        self.cb_iq.toggled.connect(self.rb_scatter.setVisible)

        bar.addStretch()

        # central grid ------------------------------------------------------
        self.container = QtWidgets.QWidget()
        vbox.addWidget(self.container)
        self.grid = QtWidgets.QGridLayout(self.container)
        self.setStatusBar(QtWidgets.QStatusBar())

    # buffer initialisation -----------------------------------------------
    def _init_buffers(self) -> None:
        self.buf, self.tbuf = {}, {}
        for ch in self.channels:
            self.buf[ch] = {k: Circular(self.N) for k in "IQM"}
            self.tbuf[ch] = Circular(self.N)

    # dynamic grid of plots ------------------------------------------------
    def _build_layout(self) -> None:
        """Rebuild grid whenever channels or plot toggles change."""
        while self.grid.count():
            self.grid.takeAt(0).widget().deleteLater()

        modes = [
            m for m, cb in zip(("T", "IQ", "F"), (self.cb_time, self.cb_iq, self.cb_fft))
            if cb.isChecked()
        ]
        self.plots, self.curves = {}, {}

        font = QFont()
        font.setPointSize(UI_FONT_SIZE)

        # iterate rows = channels, cols = modes
        for row, ch in enumerate(self.channels):
            self.plots[ch], self.curves[ch] = {}, {}
            for col, mode in enumerate(modes):
                vb = ClickableViewBox()

                if mode == "F":       # FFT pane
                    pw = pg.PlotWidget(viewBox=vb, title=f"Ch {ch} – FFT")
                    pw.setLogMode(x=True, y=True)
                    pw.setLabel("bottom", "Freq (Hz)")
                    pw.setLabel("left", "ASD (Counts / √Hz)")
                else:
                    pane_title = "Time" if mode == "T" else "IQ"
                    pw = pg.PlotWidget(viewBox=vb, title=f"Ch {ch} – {pane_title}")
                    if mode == "T":
                        pw.setLabel("left", "Amplitude (Readout Counts)")

                pw.showGrid(x=True, y=True)
                self.plots[ch][mode] = pw
                self.grid.addWidget(pw, row, col)

                # common axis fonts
                pi = pw.getPlotItem()
                for ax_name in ("left", "bottom", "right", "top"):
                    ax = pi.getAxis(ax_name)
                    if ax is not None:
                        ax.setTickFont(font)
                        if ax.label is not None:
                            ax.label.setFont(font)
                pi.titleLabel.setFont(font)

                # --- mode‑specific content ---------------------------------
                if mode == "T":           # time‑series pane
                    pw.setLabel("bottom", "Time (s)")
                    legend = pw.addLegend(offset=(30, 10))
                    self.curves[ch]["T"] = {
                        k: pw.plot(pen=pg.mkPen(col, width=LINE_WIDTH), name=k)
                        for k, col in zip(("I", "Q", "Mag"),
                                          ("#1f77b4", "#ff7f0e", "#2ca02c"))
                    }
                    self._fade_hidden_entries(legend, hide_labels=("I", "Q"))
                    self._make_legend_clickable(legend)

                elif mode == "F":         # FFT pane
                    legend = pw.addLegend(offset=(30, 10))
                    self.curves[ch]["F"] = {
                        k: pw.plot(pen=pg.mkPen(col, width=LINE_WIDTH), name=k)
                        for k, col in zip(("I", "Q", "Mag"),
                                          ("#1f77b4", "#ff7f0e", "#2ca02c"))
                    }
                    self._fade_hidden_entries(legend, hide_labels=("I", "Q"))
                    self._make_legend_clickable(legend)

                else:                     # IQ pane
                    pw.setLabel("bottom", "I")
                    pw.setLabel("left", "Q")
                    pw.getViewBox().setAspectLocked(True)
                    if self.rb_scatter.isChecked():
                        sp = pg.ScatterPlotItem(pen=None, size=SCATTER_SIZE)
                        pw.addItem(sp)
                        self.curves[ch]["IQ"] = {"mode": "scatter", "item": sp}
                    else:
                        img = pg.ImageItem(axisOrder="row-major")
                        img.setLookupTable(self.lut)
                        pw.addItem(img)
                        self.curves[ch]["IQ"] = {"mode": "density", "item": img}

    # ───────────────────────── legend helpers ─────────────────────────────
    @staticmethod
    def _fade_hidden_entries(legend, hide_labels):
        for sample, label in legend.items:
            txt = label.labelItem.toPlainText() if hasattr(label, "labelItem") else ""
            if txt in hide_labels:
                sample.setOpacity(0.3)
                label.setOpacity(0.3)

    @staticmethod
    def _make_legend_clickable(legend):
        for sample, label in legend.items:
            curve = sample.item

            def toggle(evt, *, c=curve, s=sample, l=label):
                vis = not c.isVisible()
                c.setVisible(vis)
                op = 1.0 if vis else 0.3
                s.setOpacity(op)
                l.setOpacity(op)

            label.mousePressEvent = toggle
            sample.mousePressEvent = toggle

    # ─────────────────────────── GUI callbacks ────────────────────────────
    def _toggle_pause(self):
        self.paused = not self.paused
        self.b_pause.setText("Resume" if self.paused else "Pause")

    def _update_channels(self):
        new = self._parse_channels(self.e_ch.text())
        if new != self.channels:
            self.channels = new
            self._init_buffers()
            self._build_layout()

    def _change_buffer(self):
        try:
            n = int(self.e_buf.text())
        except ValueError:
            return
        if n != self.N:
            self.N = n
            self._init_buffers()

    # ───────────────────────── main update loop ──────────────────────────
    def _update_gui(self):
        # ---------- ingest new packets ------------------------------------
        if self.paused:
            while not self.receiver.queue.empty():
                self.receiver.queue.get()
            return

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
                I = pkt.s[2 * (ch - 1)] / 256.0
                Q = pkt.s[2 * (ch - 1) + 1] / 256.0
                self.buf[ch]["I"].add(I)
                self.buf[ch]["Q"].add(Q)
                self.buf[ch]["M"].add(math.hypot(I, Q))
                self.tbuf[ch].add(t_rel)

        # ---------- update plots ------------------------------------------
        for ch in self.channels:
            I = self.buf[ch]["I"].data()
            Q = self.buf[ch]["Q"].data()
            M = self.buf[ch]["M"].data()
            t = self.tbuf[ch].data()

            # time plot
            if "T" in self.curves[ch]:
                c = self.curves[ch]["T"]
                c["Mag"].setData(t, M)
                if c["I"].isVisible():
                    c["I"].setData(t, I)
                if c["Q"].isVisible():
                    c["Q"].setData(t, Q)

            # IQ pane
            if "IQ" in self.curves[ch] and len(I):
                pane = self.curves[ch]["IQ"]
                mode = pane["mode"]
                if mode == "density":
                    g = DENSITY_GRID
                    Imin, Imax = I.min(), I.max()
                    Qmin, Qmax = Q.min(), Q.max()
                    if Imin == Imax or Qmin == Qmax:
                        continue
                    ix = ((I - Imin) * (g - 1) / (Imax - Imin)).astype(np.intp)
                    qy = ((Q - Qmin) * (g - 1) / (Qmax - Qmin)).astype(np.intp)
                    hist = pane.get("hist_buf")
                    if hist is None or hist.shape != (g, g):
                        hist = np.zeros((g, g), np.uint32)
                        pane["hist_buf"] = hist
                    else:
                        hist.fill(0)
                    r = self.dot_px // 2
                    if self.dot_px == 1:
                        np.add.at(hist, (qy, ix), 1)
                    else:
                        for dy in range(-r, r + 1):
                            for dx in range(-r, r + 1):
                                ys = qy + dy
                                xs = ix + dx
                                mask = (
                                    (ys >= 0) & (ys < g) & (xs >= 0) & (xs < g)
                                )
                                np.add.at(hist, (ys[mask], xs[mask]), 1)
                    if SMOOTH_SIGMA and gaussian_filter is not None:
                        hist = gaussian_filter(hist.astype(np.float32),
                                               SMOOTH_SIGMA, mode="nearest")
                    if LOG_COMPRESS:
                        hist = np.log1p(hist, out=hist.astype(np.float32))
                    if hist.max() > 0:
                        hist *= 255.0 / hist.max()
                    img = pane["item"]
                    img.setImage(hist.astype(np.uint8),
                                 levels=(0, 255), autoLevels=False)
                    img.setRect(QtCore.QRectF(Imin, Qmin,
                                              Imax - Imin, Qmax - Qmin))
                else:  # scatter
                    idx = (
                        np.linspace(0, len(I) - 1, SCATTER_POINTS, dtype=np.intp)
                        if len(I) > SCATTER_POINTS else np.arange(len(I))
                    )
                    xs, ys = I[idx], Q[idx]
                    rel = idx / idx.max() if idx.size else idx
                    colors = self.cmap.map(rel.astype(np.float32), mode="byte")
                    pane["item"].setData(xs, ys, brush=colors,
                                         pen=None, size=SCATTER_SIZE)

            # FFT pane
            if (
                "F" in self.curves[ch]
                and self.buf[ch]["I"].count >= 2
                and self.frame_cnt % FFT_THROTTLE == 0
                and self.fft_workers.get(ch) is None
            ):
                w = FFTWorker(ch, I, Q, M, t)
                w.done.connect(self._fft_done)
                self.fft_workers[ch] = w
                w.start()

        # FPS / packet rate status line
        self.frame_cnt += 1
        now = time.time()
        if now - self.t_last >= 1.0:
            fps = self.frame_cnt / (now - self.t_last)
            pps = self.pkt_cnt / (now - self.t_last)
            self.statusBar().showMessage(f"FPS {fps:.1f} | Packets/s {pps:.1f}")
            self.frame_cnt = self.pkt_cnt = 0
            self.t_last = now

    # FFT callback ---------------------------------------------------------
    def _fft_done(self, ch, fi, fq, fm, freqs):
        if ch in self.curves and "F" in self.curves[ch]:
            curves = self.curves[ch]["F"]
            curves["Mag"].setData(freqs, fm)
            if curves["I"].isVisible():
                curves["I"].setData(freqs, fi)
            if curves["Q"].isVisible():
                curves["Q"].setData(freqs, fq)
        self.fft_workers[ch] = None

    # helpers / cleanup ----------------------------------------------------
    @staticmethod
    def _parse_channels(txt: str) -> List[int]:
        out = []
        for tok in txt.split(","):
            tok = tok.strip()
            if tok.isdigit():
                v = int(tok)
                if 1 <= v <= streamer.NUM_CHANNELS:
                    out.append(v)
        return out or [1]

    def closeEvent(self, ev):
        """Graceful shutdown of timers and worker threads."""
        self.timer.stop()
        self.receiver.stop()
        self.receiver.wait()
        for w in self.fft_workers.values():
            if w is not None:
                w.quit()
                w.wait()
        super().closeEvent(ev)
        ev.accept()


# ──────────────────────── command‑line entry‑point ─────────────────────────
def main():
    """CLI wrapper: ``python -m rfmux.tools.periscope <host> [options]``."""
    ap = argparse.ArgumentParser(description="Periscope – real‑time viewer")
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
        warnings.warn("FPS > 30 seems unnecessary", RuntimeWarning)
    if args.density_dot < 1:
        ap.error("Density‑dot size must be ≥1 pixel.")

    refresh_ms = int(round(1000.0 / args.fps))
    app = QtWidgets.QApplication(sys.argv[:1])
    win = Periscope(
        args.hostname,
        args.module,
        args.channels,
        args.num_samples,
        refresh_ms,
        args.density_dot,
    )
    win.show()
    sys.exit(app.exec())


# ─────────────────────────── one‑liner launcher ───────────────────────────
def launch(
    hostname: str,
    *,
    module: int = 1,
    channels: str = "1",
    buf_size: int = 5_000,
    fps: float = 30.0,
    density_dot: int = DENSITY_DOT_SIZE,
    blocking: bool | None = None,
):
    """
    Launch a **Periscope** window within a python environment.
    
    Behaviour
    ---------
    • **Plain Python / REPL** – blocks until the window closes.  
    • **IPython / Jupyter**   – enables ``%gui qt`` automatically and returns
      immediately (*non‑blocking*). Pass ``blocking=True`` to override.

    Parameters
    ----------
    hostname : str
        Multicast hostname, e.g. ``"rfmux0009.local"``.
    module : int, default=1
        1‑based module number recognised by the firmware.
    channels : str, default ``"1"``
        Comma‑separated channel list (whitespace ignored), e.g. ``"1,2,5"``.
    buf_size : int, default ``5000``
        Ring‑buffer length in *samples* per channel.
    fps : float, default ``30.0``
        GUI refresh rate. Values > 30 fps rarely help and raise CPU use.
    density_dot : int, default ``1``
        Pixel radius for IQ‑density dots (must be ≥ 1).
    blocking : bool | None
        • ``True``  — enter the Qt event‑loop *here* and block until the window  
          closes.  
        • ``False`` — return immediately; caller manages (or already runs) the  
          event‑loop.  
        • ``None``  — *auto*: blocks in a plain Python/REPL, **returns** inside  
          IPython/Jupyter when a Qt loop is active (or after enabling it).

    Returns
    -------
    viewer : :class:`Periscope`
        The viewer instance (always).
    app : QtWidgets.QApplication
        Shown only when the function does **not** block (so you can call
        ``app.exec()`` yourself if desired).

    Examples
    --------
    **1. Plain Python script / REPL (blocks):**

    >>> import rfmux.tools.periscope as ps
    >>> ps.launch("rfmux0009.local", module=1, channels="1,2")

    **2. IPython or Jupyter _without_ an active Qt loop (auto non‑blocking):**

    >>> import rfmux.tools.periscope as ps
    >>> viewer, app = ps.launch("rfmux0009.local")   # %gui qt injected

    **3. IPython / Jupyter _with_ ``%gui qt`` already enabled:**

    >>> %gui qt        # run once per session
    >>> import rfmux.tools.periscope as ps
    >>> viewer, app = ps.launch("rfmux0009.local", blocking=False)

    **4. Multiple viewers in one process (non‑blocking):**

    >>> import rfmux.tools.periscope as ps
    >>> viewers = [                                         # doctest: +SKIP
    ...     ps.launch("rfmux0009.local", module=1, blocking=False),
    ...     ps.launch("rfmux0010.local", module=2, blocking=False),
    ... ]


    """
    ip = _get_ipython()
    have_loop = _qt_eventloop_running()
    if ip and not have_loop:
        ip.run_line_magic("gui", "qt")
        have_loop = True

    if blocking is None:
        blocking = not have_loop

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    refresh_ms = int(round(1000.0 / fps))

    viewer = Periscope(hostname, module, channels, buf_size, refresh_ms, density_dot)
    viewer.show()

    if blocking:
        if _running_inside_ipython():
            app.exec()
        else:
            sys.exit(app.exec())
    return viewer, app


if __name__ == "__main__":
    main()
