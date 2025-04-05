#!/usr/bin/env -S uv run

import argparse
import array
import dataclasses
import math
import numpy as np
import queue
import select
import socket
import struct
import sys
import time
import warnings

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QIntValidator
import pyqtgraph as pg

# For our IRIG timestamp, we need the following constant:
SS_PER_SECOND = 125000000

# Configure PyQtGraph for performance.
pg.setConfigOptions(antialias=False)
pg.setConfigOptions(useOpenGL=False)

# Constants
STREAMER_PORT = 9876
STREAMER_HOST = "239.192.0.2"
STREAMER_LEN = 8240  # expected packet length in bytes
NUM_CHANNELS = 1024
HEADER_FORMAT = "<IHHBBBBI"  # header: 16 bytes total
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def get_local_ip(crs_hostname: str) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect((crs_hostname, 1))
            return s.getsockname()[0]
        except Exception:
            raise Exception("Could not determine local IP address!")


class CircularBuffer:
    def __init__(self, size, dtype=np.float64):
        self.size = size
        self.buffer = np.zeros(size * 2, dtype=dtype)
        self.ptr = 0
        self.count = 0

    def add(self, value):
        self.buffer[self.ptr] = value
        self.buffer[self.ptr + self.size] = value
        self.ptr = (self.ptr + 1) % self.size
        if self.count < self.size:
            self.count += 1

    def get_data(self):
        if self.count < self.size:
            return self.buffer[: self.count]
        else:
            # Return a contiguous block regardless of wrap-around.
            return self.buffer[self.ptr : self.ptr + self.size]


@dataclasses.dataclass(order=True)
class Timestamp:
    y: int  # Year (0-99)
    d: int  # Day (1-366)
    h: int  # Hour (0-23)
    m: int  # Minute (0-59)
    s: int  # Second (0-59)
    ss: int  # Sub-second (0 to SS_PER_SECOND-1)
    c: int  # Unused here
    sbs: int  # Unused here
    source: str = "GND"
    recent: bool = False

    def renormalize(self):
        carry, self.ss = divmod(self.ss, SS_PER_SECOND)
        self.s += carry
        carry, self.s = divmod(self.s, 60)
        self.m += carry
        carry, self.m = divmod(self.m, 60)
        self.h += carry
        carry, self.h = divmod(self.h, 24)
        self.d += carry

    @classmethod
    def from_bytes(cls, data: bytes):
        vals = struct.unpack("<8I", data)
        return cls(*vals, source="GND", recent=False)

    @classmethod
    def from_TuberResult(cls, ts):
        return cls(
            ts.y, ts.d, ts.h, ts.m, ts.s, ts.ss, ts.c, ts.sbs, ts.source, ts.recent
        )


class DfmuxPacket:
    def __init__(
        self,
        magic,
        version,
        serial,
        num_modules,
        block,
        fir_stage,
        module,
        seq,
        s,
        ts: Timestamp,
    ):
        self.magic = magic
        self.version = version
        self.serial = serial
        self.num_modules = num_modules
        self.block = block
        self.fir_stage = fir_stage
        self.module = module  # zero-indexed module number
        self.seq = seq
        self.s = s  # array of channel samples
        self.ts = ts  # Timestamp object

    @classmethod
    def from_bytes(cls, data: bytes):
        if len(data) != STREAMER_LEN:
            raise ValueError(f"Packet size {len(data)} != expected {STREAMER_LEN}")
        header = struct.Struct(HEADER_FORMAT)
        header_args = header.unpack(data[:HEADER_SIZE])
        body = array.array("i")
        bodysize = NUM_CHANNELS * 2 * body.itemsize
        body.frombytes(data[HEADER_SIZE : HEADER_SIZE + bodysize])
        ts_data = data[HEADER_SIZE + bodysize :]
        ts = Timestamp.from_bytes(ts_data)
        return cls(*header_args, s=body, ts=ts)


class UDPReceiver(QtCore.QThread):
    def __init__(self, crs_hostname: str, module: int, parent=None):
        super().__init__(parent)
        self.crs_hostname = crs_hostname
        self.module = module
        self.packet_queue = queue.Queue()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("", STREAMER_PORT))
        # Set a large receive buffer size
        rcvbuf = 16777216 * 8
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)

        actual = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        if sys.platform == "linux":
            actual /= 2
        if actual != rcvbuf:
            warnings.warn(
                f"Unable to set SO_RCVBUF to {rcvbuf} (got {actual}). Consider "
                "'sudo sysctl net.core.rmem_max=67108864' or similar."
            )

        local_ip = get_local_ip(crs_hostname)
        self.sock.setsockopt(
            socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(local_ip)
        )
        mreq = struct.pack(
            "4s4s", socket.inet_aton(STREAMER_HOST), socket.inet_aton(local_ip)
        )
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self.sock.setblocking(False)

    def run(self):
        while not self.isInterruptionRequested():
            r, _, _ = select.select([self.sock], [], [], 0.001)
            if self.sock in r:
                try:
                    while True:
                        data = self.sock.recv(STREAMER_LEN)
                        packet = DfmuxPacket.from_bytes(data)
                        if packet.module != (self.module - 1):
                            continue
                        self.packet_queue.put(packet)
                except BlockingIOError:
                    pass
        self.sock.close()


class FFTWorker(QtCore.QThread):
    fftComputed = QtCore.pyqtSignal(
        int, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    )  # channel, fft_I, fft_Q, fft_Mag, freq

    def __init__(self, channel: int, I_data, Q_data, Mag_data, x_data, parent=None):
        super().__init__(parent)
        self.channel = channel
        self.I_data = I_data
        self.Q_data = Q_data
        self.Mag_data = Mag_data
        self.x_data = x_data

    def run(self):
        # Infer sampling rate from x_data
        if len(self.x_data) > 1:
            dt = np.diff(self.x_data)
            mean_dt = np.mean(dt)
            sampling_rate_inferred = (
                (1.0 / mean_dt) if mean_dt > 0 else 625e6 / 256 / 64 / 64
            )
            sampling_rate_inferred *= 1.25
        else:
            sampling_rate_inferred = 625e6 / 256 / 64 / 64
        fft_I = np.abs(np.fft.rfft(self.I_data))
        fft_Q = np.abs(np.fft.rfft(self.Q_data))
        fft_Mag = np.abs(np.fft.rfft(self.Mag_data))
        freq = np.linspace(0, sampling_rate_inferred / 2, len(fft_I))
        self.fftComputed.emit(self.channel, fft_I, fft_Q, fft_Mag, freq)


class RealTimePlot(QtWidgets.QMainWindow):
    def __init__(
        self,
        crs_hostname: str,
        module: int,
        default_channels: str = "1",
        default_buffer_size: int = 5000,
        update_period: int = 33,
        parent=None,
    ):
        """
        :param update_period: Update period in milliseconds.
        """
        super().__init__(parent)
        self.crs_hostname = crs_hostname
        self.module = module
        self.buffer_size = default_buffer_size
        self.update_period = update_period

        # Time reference.
        self.start_time = None

        # State.
        self.resizing = False
        self.dark_mode = True
        self.fft_throttle = 1  # FFT update throttle count.
        self.paused = False

        # Metrics.
        self.last_metrics_update_time = time.time()
        self.frame_count = 0
        self.packets_processed = 0

        # Title label.
        self.title_label = QtWidgets.QLabel(f"CRS: {crs_hostname}    Module: {module}")
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title_font = self.title_label.font()
        title_font.setPointSize(16)
        self.title_label.setFont(title_font)

        # Initialize channel list and buffers.
        self.channels = self.parse_channels(default_channels)
        self.init_buffers()

        # FFT worker tracking per channel.
        self.fft_workers = {}

        # Set up UDP receiver.
        self.receiver = UDPReceiver(self.crs_hostname, self.module)
        self.receiver.start()

        # Build UI.
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.addWidget(self.title_label)

        # Control Panel.
        control_panel = QtWidgets.QHBoxLayout()
        control_panel.addWidget(QtWidgets.QLabel("Channels (comma-separated):"))
        self.channels_edit = QtWidgets.QLineEdit(default_channels)
        control_panel.addWidget(self.channels_edit)
        self.update_channels_btn = QtWidgets.QPushButton("Update Channels")
        self.update_channels_btn.clicked.connect(self.update_channels)
        control_panel.addWidget(self.update_channels_btn)
        control_panel.addWidget(QtWidgets.QLabel("Buffer Size:"))
        self.buffer_edit = QtWidgets.QLineEdit(str(self.buffer_size))
        self.buffer_edit.setValidator(QIntValidator(100, 100000, self))
        self.buffer_edit.editingFinished.connect(self.buffer_edit_changed)
        control_panel.addWidget(self.buffer_edit)

        self.dark_mode_checkbox = QtWidgets.QCheckBox("Dark Mode")
        self.dark_mode_checkbox.setChecked(True)
        self.dark_mode_checkbox.toggled.connect(self.toggle_theme)
        control_panel.addWidget(self.dark_mode_checkbox)

        self.pause_button = QtWidgets.QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        control_panel.addWidget(self.pause_button)

        # Plot mode checkboxes.
        self.time_checkbox = QtWidgets.QCheckBox("Time Domain")
        self.time_checkbox.setChecked(True)
        self.time_checkbox.toggled.connect(self.update_plot_layout)
        control_panel.addWidget(self.time_checkbox)
        self.iq_checkbox = QtWidgets.QCheckBox("IQ Scatter")
        self.iq_checkbox.setChecked(True)
        self.iq_checkbox.toggled.connect(self.update_plot_layout)
        control_panel.addWidget(self.iq_checkbox)
        self.fft_checkbox = QtWidgets.QCheckBox("FFT")
        self.fft_checkbox.setChecked(True)
        self.fft_checkbox.toggled.connect(self.update_plot_layout)
        control_panel.addWidget(self.fft_checkbox)
        control_panel.addStretch()
        main_layout.addLayout(control_panel)

        # Plot area.
        self.plots_container = QtWidgets.QWidget()
        self.plot_grid = QtWidgets.QGridLayout(self.plots_container)
        main_layout.addWidget(self.plots_container)

        # Status bar for real-time metrics.
        self.setStatusBar(QtWidgets.QStatusBar())

        # Dictionaries to hold plot widgets and curves.
        # Structure: self.plot_widgets[channel][mode] = PlotWidget; same for self.curves.
        self.plot_widgets = {}
        self.curves = {}
        self.update_plot_layout()

        self.setWindowTitle("Real-Time I/Q Data Plot")
        self.update_count = 0

        # Timer for GUI updates.
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(self.update_period)

    def init_buffers(self):
        """Initialize circular buffers for each channel."""
        self.data_buffer = {}
        self.x_buffer = {}
        for ch in self.channels:
            self.data_buffer[ch] = {
                "I": CircularBuffer(self.buffer_size),
                "Q": CircularBuffer(self.buffer_size),
                "Mag": CircularBuffer(self.buffer_size),
            }
            self.x_buffer[ch] = CircularBuffer(self.buffer_size)

    def parse_channels(self, channels_str):
        try:
            return [
                int(ch.strip())
                for ch in channels_str.split(",")
                if ch.strip() and 1 <= int(ch.strip()) <= NUM_CHANNELS
            ]
        except Exception:
            return [1]

    def update_channels(self):
        new_channels = self.parse_channels(self.channels_edit.text())
        if not new_channels:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Channels",
                "Please enter at least one valid channel number (1-1024).",
            )
            return
        if new_channels != self.channels:
            self.channels = new_channels
            self.init_buffers()
            self.update_plot_layout()

    def buffer_edit_changed(self):
        try:
            new_size = int(self.buffer_edit.text())
        except ValueError:
            return
        if new_size != self.buffer_size:
            self.buffer_size = new_size
            self.init_buffers()

    def update_plot_layout(self):
        """Rebuild the grid of plots based on active modes and channels."""
        # Determine active modes in fixed order.
        modes = []
        if self.time_checkbox.isChecked():
            modes.append("Time Domain")
        if self.iq_checkbox.isChecked():
            modes.append("IQ Scatter")
        if self.fft_checkbox.isChecked():
            modes.append("FFT")
        self.active_modes = modes

        # Clear existing plots from the grid.
        while self.plot_grid.count():
            item = self.plot_grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.plot_widgets = {}
        self.curves = {}
        # For each channel (row) and each active mode (column), create a PlotWidget.
        for row, ch in enumerate(self.channels):
            self.plot_widgets[ch] = {}
            self.curves[ch] = {}
            for col, mode in enumerate(self.active_modes):
                pw = pg.PlotWidget(title=f"Channel {ch} - {mode}")
                pw.showGrid(x=True, y=True)
                if mode == "Time Domain":
                    pw.setLabel("bottom", "Time (s)")
                    legend = pw.addLegend(offset=(10, 10))
                    curve_I = pw.plot(pen=pg.mkPen("#1f77b4", width=2), name="I")
                    curve_Q = pw.plot(pen=pg.mkPen("#ff7f0e", width=2), name="Q")
                    curve_Mag = pw.plot(pen=pg.mkPen("#2ca02c", width=2), name="Mag")
                    for curve in (curve_I, curve_Q, curve_Mag):
                        curve.setDownsampling(True, "peak")
                        curve.setClipToView(True)
                    self.curves[ch]["Time Domain"] = {
                        "I": curve_I,
                        "Q": curve_Q,
                        "Mag": curve_Mag,
                    }
                elif mode == "IQ Scatter":
                    pw.setLabel("bottom", "I")
                    pw.setLabel("left", "Q")
                    pw.getViewBox().setAspectLocked(True)
                    scatter = pg.ScatterPlotItem(
                        pen=pg.mkPen(None), brush=pg.mkBrush("#1f77b4"), size=5
                    )
                    pw.addItem(scatter)
                    self.curves[ch]["IQ Scatter"] = scatter
                elif mode == "FFT":
                    pw.setLabel("bottom", "Frequency (Hz)")
                    pw.setLogMode(x=True, y=True)
                    legend = pw.addLegend(offset=(10, 10))
                    fft_I = pw.plot(pen=pg.mkPen("#1f77b4", width=2), name="FFT I")
                    fft_Q = pw.plot(pen=pg.mkPen("#ff7f0e", width=2), name="FFT Q")
                    fft_Mag = pw.plot(pen=pg.mkPen("#2ca02c", width=2), name="FFT Mag")
                    for curve in (fft_I, fft_Q, fft_Mag):
                        curve.setDownsampling(True, "peak")
                        curve.setClipToView(True)
                    self.curves[ch]["FFT"] = {
                        "FFT I": fft_I,
                        "FFT Q": fft_Q,
                        "FFT Mag": fft_Mag,
                    }
                    # Add clickable legend items for FFT curves.
                    for sample, label in legend.items:

                        def make_mouse_press(curve, lbl):
                            def mousePressEvent(event):
                                curve.setVisible(not curve.isVisible())
                                lbl.setColor("w" if curve.isVisible() else "r")

                            return mousePressEvent

                        label.mousePressEvent = make_mouse_press(sample, label)
                self.plot_widgets[ch][mode] = pw
                self.plot_grid.addWidget(pw, row, col)
        self.apply_theme()

    def apply_theme(self):
        for ch, plots in self.plot_widgets.items():
            for mode, pw in plots.items():
                pw.setBackground("k" if self.dark_mode else "w")
        self.title_label.setStyleSheet(
            "color: white;" if self.dark_mode else "color: black;"
        )

    def toggle_theme(self, state):
        self.dark_mode = state
        self.apply_theme()

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.setText("Resume" if self.paused else "Pause")

    def onFFTComputed(self, channel, fft_I, fft_Q, fft_Mag, freq):
        if channel in self.curves and "FFT" in self.curves[channel]:
            fft_curves = self.curves[channel]["FFT"]
            fft_curves["FFT I"].setData(x=freq, y=fft_I)
            fft_curves["FFT Q"].setData(x=freq, y=fft_Q)
            fft_curves["FFT Mag"].setData(x=freq, y=fft_Mag)
        self.fft_workers[channel] = None

    def update_plot(self):
        # When paused, flush the UDP queue and skip updating plots.
        if self.paused:
            while not self.receiver.packet_queue.empty():
                self.receiver.packet_queue.get()
            return

        # Process all packets in the queue.
        while not self.receiver.packet_queue.empty():
            packet = self.receiver.packet_queue.get()
            self.packets_processed += 1
            ts = Timestamp(
                packet.ts.y,
                packet.ts.d,
                packet.ts.h,
                packet.ts.m,
                packet.ts.s,
                packet.ts.ss,
                packet.ts.c,
                packet.ts.sbs,
                packet.ts.source,
                packet.ts.recent,
            )
            ts.ss += int(0.02 * SS_PER_SECOND)
            ts.renormalize()
            current_time = ts.h * 3600 + ts.m * 60 + ts.s + ts.ss / SS_PER_SECOND
            if self.start_time is None:
                self.start_time = current_time
            rel_time = current_time - self.start_time

            for ch in self.channels:
                I_val = packet.s[2 * (ch - 1)] / 256.0
                Q_val = packet.s[2 * (ch - 1) + 1] / 256.0
                mag = math.sqrt(I_val**2 + Q_val**2)
                # Add new sample to each circular buffer.
                self.data_buffer[ch]["I"].add(I_val)
                self.data_buffer[ch]["Q"].add(Q_val)
                self.data_buffer[ch]["Mag"].add(mag)
                self.x_buffer[ch].add(rel_time)

        # For each channel, extract contiguous window of data.
        for ch in self.channels:
            I_data = self.data_buffer[ch]["I"].get_data()
            Q_data = self.data_buffer[ch]["Q"].get_data()
            Mag_data = self.data_buffer[ch]["Mag"].get_data()
            xdata = self.x_buffer[ch].get_data()

            # Update Time Domain and IQ Scatter plots.
            if "Time Domain" in self.active_modes:
                curves = self.curves[ch]["Time Domain"]
                curves["I"].setData(x=xdata, y=I_data)
                curves["Q"].setData(x=xdata, y=Q_data)
                curves["Mag"].setData(x=xdata, y=Mag_data)
            if "IQ Scatter" in self.active_modes:
                pts = [{"pos": [I, Q]} for I, Q in zip(I_data, Q_data)]
                self.curves[ch]["IQ Scatter"].setData(pts)

            # Offload FFT computation if enabled and not already running.
            if "FFT" in self.active_modes:
                if self.update_count % self.fft_throttle == 0:
                    # Only start a new FFT worker if sufficient data is available.
                    if self.data_buffer[ch]["I"].count >= self.buffer_size:
                        if self.fft_workers.get(ch) is None:
                            worker = FFTWorker(
                                ch,
                                I_data.copy(),
                                Q_data.copy(),
                                Mag_data.copy(),
                                xdata.copy(),
                            )
                            worker.fftComputed.connect(self.onFFTComputed)
                            self.fft_workers[ch] = worker
                            worker.start()

        self.update_count += 1
        self.frame_count += 1

        # Update real-time metrics (approx. once per second).
        current_time = time.time()
        if current_time - self.last_metrics_update_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_metrics_update_time)
            pkt_rate = self.packets_processed / (
                current_time - self.last_metrics_update_time
            )
            self.statusBar().showMessage(
                f"FPS: {fps:.2f} | Packets/s: {pkt_rate:.2f} | Buffer Size: {self.buffer_size}"
            )
            self.last_metrics_update_time = current_time
            self.frame_count = 0
            self.packets_processed = 0

    def resizeEvent(self, event):
        self.resizing = True
        QtCore.QTimer.singleShot(300, self.end_resize)
        super().resizeEvent(event)

    def end_resize(self):
        self.resizing = False
        for ch in self.channels:
            if "Time Domain" in self.plot_widgets[ch]:
                self.plot_widgets[ch]["Time Domain"].enableAutoRange(
                    axis=pg.ViewBox.YAxis, enable=True
                )
            if "FFT" in self.plot_widgets[ch]:
                self.plot_widgets[ch]["FFT"].enableAutoRange(
                    axis=pg.ViewBox.YAxis, enable=True
                )

    def closeEvent(self, event):
        self.receiver.requestInterruption()
        self.receiver.wait()
        # Optionally wait for any FFT workers to finish.
        for worker in self.fft_workers.values():
            if worker is not None:
                worker.wait()
        event.accept()


def main():
    parser = argparse.ArgumentParser(
        prog="periscope",
        description="Real-time, streaming visualization for the CRS",
    )
    parser.add_argument("hostname")
    parser.add_argument("--module", "-m", type=int, default=1)
    parser.add_argument("--channels", "-c", type=str, default="1")
    parser.add_argument("--num-samples", "-n", type=int, default=5000)
    parser.add_argument("--refresh-ms", "-r", type=int, default=100)
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv[:1])
    window = RealTimePlot(
        crs_hostname=args.hostname,
        module=args.module,
        default_channels=args.channels,
        default_buffer_size=args.num_samples,
        update_period=args.refresh_ms,
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
