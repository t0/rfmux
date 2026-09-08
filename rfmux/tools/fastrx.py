#!/usr/bin/env python3
"""
rfmux fastrx - tools for the fastrx AF_XDP capture path.

fastrx is an alternate, higher-performance capture path to rfmux.streamer's
socket receiver: fastrxd attaches an XDP program to a NIC and hands
validated packets to clients over shared memory, without a socket recv()
per packet.

Subcommands:

  hud     live I/Q + PSD viewer
"""

import click
import sys
import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from scipy.signal import welch

DEFAULT_DEPTH = 1024          # packets per grab


@click.group()
def cli():
    pass


PACKET_RATE = 625e6 / 256
REFRESH_FPS = 30
WATERFALL_ROWS = 200
DEPTH_MIN_LOG2 = 6
DEPTH_MAX_LOG2 = 16


class HUD(QMainWindow):
    def __init__(self, channel: int = 0, pipe: int = 1, depth: int = DEFAULT_DEPTH,
                 *, interface: str | None = None, socket: str | None = None):
        super().__init__()
        # Imported here, not at module level: the native extension may not be
        # built in a given environment, and everything above the `hud`
        # subcommand should stay usable without it.  The `hud` command
        # surfaces a failure here as a clean CLI error.
        from rfmux import fastrx
        self._fastrx = fastrx

        self.channel = channel
        self.pipe = pipe
        self.depth = depth
        self._consumer = fastrx.PacketCapture(interface=interface, socket=socket)
        self._n_grabs = 0

        self.setWindowTitle("fastrx HUD")
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # matplotlib canvas
        self._fig = Figure(figsize=(10, 6), tight_layout=True)
        self._ax_iq = self._fig.add_subplot(3, 1, 1)
        self._ax_fft = self._fig.add_subplot(3, 1, 2)
        self._ax_wf = self._fig.add_subplot(3, 1, 3)
        self._canvas = FigureCanvas(self._fig)
        # Stretch 1: the canvas absorbs all spare vertical space.  The control
        # row below is added with the default stretch of 0, so it keeps its
        # size-hint height however large the window gets.  This is why neither
        # needs a fixed pixel height -- Qt divides only the *extra* space, in
        # proportion to these factors.
        layout.addWidget(self._canvas, 1)

        dummy = np.zeros(1)
        self._line_i, = self._ax_iq.plot(dummy, label="I", color="C0", lw=0.7)
        self._line_q, = self._ax_iq.plot(dummy, label="Q", color="C1", lw=0.7)
        self._line_fft, = self._ax_fft.plot(dummy, color="C2", lw=0.7)

        self._mean_i_line = self._ax_iq.axhline(
            0, color="C0", linestyle="-.", lw=0.8, alpha=0.6)
        self._mean_q_line = self._ax_iq.axhline(
            0, color="C1", linestyle="-.", lw=0.8, alpha=0.6)

        # Waterfall: a rolling history of the same PSD drawn above it.
        #
        # Deliberately reuses pxx rather than recomputing: the FFT is the
        # expensive part of a frame and the two views want identical data, so a
        # discrepancy between them would be a bug rather than a feature.
        #
        # The image is allocated on the first frame, because its width is the PSD
        # length -- which depends on nperseg, and therefore on a slider.
        self._wf_data = None
        self._wf_image = None
        self._wf_bins = 0
        self._wf_clim = None

        # Blitting state.  The cached backgrounds hold everything static, so
        # a frame only has to re-render the three artists that actually change.
        # They go stale whenever an axis limit moves, which autoscaling does,
        # so _draw() re-caches when it detects that and blits otherwise.
        self._bg = None
        self._lims = None

        # Axis extents, set when they change rather than autoscaled per frame.
        self._iq_xmax = None
        self._fft_xlim = None

        self._ax_iq.legend(loc="upper right", fontsize=8)
        self._ax_iq.set_ylabel("ADC counts")
        self._ax_iq.set_xlabel("Packet")
        self._ax_fft.set_ylabel("PSD (dB/Hz)")
        self._ax_fft.set_xlabel("Frequency (MHz)")
        self._ax_wf.set_ylabel("Frames ago")
        self._ax_wf.set_xlabel("Frequency (MHz)")
        self._ax_iq.grid(True)
        self._ax_fft.grid(True)

        # Controls
        #
        # Held in their own widget rather than added to the root layout directly.
        # A layout has no size policy of its own, so the sliders inside it would
        # still grow vertically; a widget does, and Maximum pins its height to the
        # size hint no matter how tall the window becomes.
        ctrl_widget = QWidget()
        ctrl = QHBoxLayout(ctrl_widget)
        ctrl.setContentsMargins(0, 0, 0, 0)
        ctrl_widget.setSizePolicy(QSizePolicy.Policy.Preferred,
                                  QSizePolicy.Policy.Maximum)
        layout.addWidget(ctrl_widget)

        # Capture depth: packets per grab, and therefore the length of the time
        # series and the resolution of the PSD.  Stepped in powers of two so the
        # slider covers three decades without a useless amount of travel.
        self._depth_slider = self._make_slider(
            ctrl, "depth", DEPTH_MIN_LOG2, DEPTH_MAX_LOG2,
            max(DEPTH_MIN_LOG2, min(DEPTH_MAX_LOG2, depth.bit_length() - 1)),
            fmt=lambda v: f"depth: {1 << v}")
        self._depth_slider.valueChanged.connect(self._on_depth_changed)

        # Upper bound tracks the depth (see _on_depth_changed): a segment longer
        # than the capture is meaningless, though _refresh() also clamps against
        # the packets actually received, which may be fewer than asked for.
        self._nperseg_slider = self._make_slider(
            ctrl, "nperseg", 16, self.depth, min(256, self.depth))

        win_box = QVBoxLayout()
        win_box.addWidget(QLabel("Window"))
        self._window_combo = QComboBox()
        for name in ("hann", "hamming", "blackman", "flattop", "boxcar"):
            self._window_combo.addItem(name)
        win_box.addWidget(self._window_combo)
        ctrl.addLayout(win_box)

        # self.pipe is 1-indexed, like every pipeline identifier in the API, so
        # the spin box shows it as-is.  self.channel stays 0-indexed: it selects a
        # numpy column, and those are 0-based.
        pipe_box = QVBoxLayout()
        pipe_box.addWidget(QLabel("Pipe"))
        self._pipe_spin = QSpinBox()
        self._pipe_spin.setRange(1, fastrx.NUM_PIPELINES)
        self._pipe_spin.setValue(pipe)
        pipe_box.addWidget(self._pipe_spin)
        ctrl.addLayout(pipe_box)

        ch_box = QVBoxLayout()
        ch_box.addWidget(QLabel("Channel"))
        self._ch_spin = QSpinBox()
        self._ch_spin.setRange(1, fastrx.MAX_SAMPLES)
        self._ch_spin.setValue(channel + 1)
        ch_box.addWidget(self._ch_spin)
        ctrl.addLayout(ch_box)

        # Renormalize: the waterfall's colour scale is latched and held,
        # so a later shift in the signal's dynamic range leaves old rows
        # clipped or washed out against a scale set for a different regime.
        wf_box = QVBoxLayout()
        wf_box.addWidget(QLabel("Waterfall"))
        self._renorm_button = QPushButton("Renormalize")
        self._renorm_button.clicked.connect(self._on_renormalize)
        wf_box.addWidget(self._renorm_button)
        ctrl.addLayout(wf_box)

        self._status = QLabel("starting...")
        layout.addWidget(self._status)

        # Each tick grabs afresh, so both controls take effect on the next
        # frame with nothing to restart: channel reselects a column, pipe
        # changes which pipe the next get_samples() asks for.
        self._ch_spin.valueChanged.connect(self._on_channel_changed)
        self._pipe_spin.valueChanged.connect(self._on_pipe_changed)

        # The display rate bounds the work done. Otherwise, the HUD would grab
        # and redraw hundreds of times a second and progressively bog down as
        # the Qt event queue backed up.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(int(1000 / REFRESH_FPS))

    def _make_slider(self, parent_layout, label: str,
                     lo: int, hi: int, default: int, fmt=None) -> QSlider:
        """A labelled slider.  `fmt` renders the value when the position is not
        the value itself -- the depth slider is logarithmic, for instance."""
        fmt = fmt or (lambda v: f"{label}: {v}")
        box = QVBoxLayout()
        lbl = QLabel(fmt(default))
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setRange(lo, hi)
        sld.setValue(default)
        sld.setTickInterval(max(1, (hi - lo) // 10))
        sld.valueChanged.connect(lambda v: lbl.setText(fmt(v)))
        box.addWidget(lbl)
        box.addWidget(sld)
        parent_layout.addLayout(box)
        return sld

    def _on_channel_changed(self, value: int):
        self.channel = value - 1        # GUI is 1-indexed

    def _on_renormalize(self):
        """Drop the latched waterfall colour scale so the next row relatches."""
        self._wf_clim = None

    def _close_consumer(self):
        if self._consumer is not None:
            self._consumer.stop()
            self._consumer = None

    def resizeEvent(self, event):
        """Drop the blit cache: the cached bitmaps are the old geometry."""
        self._bg = None
        super().resizeEvent(event)

    def closeEvent(self, event):
        """Release the fastrxd client slot when the window goes away, rather than
        leaving it held until the interpreter exits."""
        self._timer.stop()
        self._close_consumer()
        super().closeEvent(event)

    def _on_depth_changed(self, value: int):
        """Slider is log2; nperseg must stay within the new depth or welch()
        raises rather than degrading."""
        self.depth = 1 << value
        self._nperseg_slider.setMaximum(self.depth)

    def _on_pipe_changed(self, value: int):
        self.pipe = value # both 1-indexed

    # Grab timeout.  Bounded by the frame interval rather than generous, because
    # _tick() runs inline on the GUI thread: a grab that waits longer than a frame
    # freezes the window for exactly as long as it waits.  A pipe the transmitter
    # is not sending would otherwise block for the full timeout on every tick,
    # which reads as the whole application hanging.
    GRAB_TIMEOUT = 1.0 / REFRESH_FPS

    def _tick(self):
        """Grab and render, inline on the GUI thread.

        get_samples() costs ~2 ms at these depths and a redraw ~2 ms, against a
        67 ms frame budget at 15 fps -- so there is no reason to grab on a
        separate thread.  Doing it here keeps the data path obvious and means
        the display rate, not the packet rate, sets how much work happens.

        The corollary is that nothing here may block for longer than a frame; see
        GRAB_TIMEOUT.
        """
        try:
            d = self._consumer.capture(
                    self.depth, pipe=self.pipe, timeout=self.GRAB_TIMEOUT)
        except Exception as exc:                # noqa: BLE001 - surfaced in the UI
            # No reconnect attempt: a failure here (e.g. fastrxd restarting)
            # is reported and the HUD needs to be relaunched to recover.
            self._status.setText(f"grab failed: {exc}")
            return

        if len(d["seq"]) == 0:
            # Distinguish "this pipe is not in the stream" from "nothing is
            # arriving at all" -- the remedies are completely different, and the
            # snapshot tells us which it is.
            snap = d.get("pipe_snapshot", 0)
            if snap and not (snap & (1 << (self.pipe - 1))):
                present = ", ".join(str(p + 1) for p in range(self._fastrx.NUM_PIPELINES)
                                    if snap & (1 << p))
                self._status.setText(
                    f"pipe {self.pipe} is not being transmitted "
                    f"(streaming: {present})")
            else:
                self._status.setText("no packets (is the transmitter streaming?)")
            return

        # Counted here, once, for a grab that returned packets.  _refresh() used
        # to bump it as well, double-counting -- and wrongly, since _refresh can
        # return early on an out-of-range channel or a too-short batch, neither of
        # which means the grab failed.
        self._n_grabs += 1
        self._refresh(d["i"], d["q"], d["seq"])

    def _refresh(self, i_arr, q_arr, seq):
        ch = self.channel
        if ch >= i_arr.shape[1]:
            return

        i_ch = i_arr[:, ch].astype(np.float32)
        q_ch = q_arr[:, ch].astype(np.float32)
        n = len(i_ch)
        if n < 4:
            return

        xs = np.arange(n)

        self._line_i.set_data(xs, i_ch)
        self._line_q.set_data(xs, q_ch)
        # x follows the capture length; y autoscales with hysteresis so the blit
        # cache survives a stationary signal (see _autoscale_y).
        if self._iq_xmax != n:
            self._iq_xmax = n
            self._ax_iq.set_xlim(0, n)
        self._autoscale_y(self._ax_iq,
                          float(min(i_ch.min(), q_ch.min())),
                          float(max(i_ch.max(), q_ch.max())))

        for ln, y in ((self._mean_i_line, i_ch.mean()),
                      (self._mean_q_line, q_ch.mean())):
            ln.set_ydata([y, y])

        cx = i_ch + 1j * q_ch
        nperseg = min(self._nperseg_slider.value(), n)
        freq, pxx = welch(cx, fs=PACKET_RATE,
                          window=self._window_combo.currentText(),
                          detrend=False,
                          nperseg=nperseg, return_onesided=False)
        freq = np.fft.fftshift(freq)
        pxx = np.fft.fftshift(pxx)
        db = 10 * np.log10(pxx + 1e-12)
        mhz = freq / 1e6
        self._line_fft.set_data(mhz, db)
        if self._fft_xlim != (mhz[0], mhz[-1]):
            self._fft_xlim = (mhz[0], mhz[-1])
            self._ax_fft.set_xlim(mhz[0], mhz[-1])
        self._autoscale_y(self._ax_fft, float(db.min()), float(db.max()))

        self._update_waterfall(mhz, db)
        self._draw()

        gaps = int(np.count_nonzero(np.diff(seq.astype(np.int64)) != 1))
        self._status.setText(
            f"pipe {self.pipe} ch {ch + 1}  packets={n}  "   # channel 1-indexed for display
            f"grabs={self._n_grabs}  seq={seq[0]}..{seq[-1]}  gaps={gaps}  "
            f"rms={np.hypot(i_ch, q_ch).std():.1f}")

    @staticmethod
    def _autoscale_y(ax, lo, hi):
        """Autoscale y, but reluctantly, so the blit cache mostly survives.

        Plain autoscale_view() moves the limits by a few counts on every frame
        even for a stationary signal, which invalidates the cached background
        every time and costs a full redraw -- slower than not blitting at all.

        Widen as soon as data leaves the view, so nothing is ever clipped, but
        shrink only once the data is comfortably inside it.  A settled signal then
        stops rescaling entirely while a level change still tracks."""
        pad = 0.1 * max(hi - lo, 1.0)
        want = (lo - pad, hi + pad)
        cur = ax.get_ylim()

        if lo < cur[0] or hi > cur[1]:
            ax.set_ylim(min(cur[0], want[0]), max(cur[1], want[1]))
        elif (hi - lo) < 0.4 * (cur[1] - cur[0]):
            ax.set_ylim(*want)          # data has shrunk well inside: tighten

    def _draw(self):
        """Blit the changing artists, re-caching the background when axes move.

        A full redraw re-renders every tick, label, spine and grid line on all
        three axes and costs ~63 ms -- more than a frame at any useful rate.
        Blitting restores a cached bitmap of that static furniture and redraws
        only the artists, at ~15 ms.

        The cache is valid only while the axes are unchanged, so the limits are
        compared first.  With hysteresis on the y ranges a settled signal rescales
        about once in seventy frames, so nearly every frame takes the fast path."""
        axes = (self._ax_iq, self._ax_fft, self._ax_wf)
        lims = tuple((a.get_xlim(), a.get_ylim()) for a in axes)

        if self._bg is None or lims != self._lims:
            self._recache(axes)
            self._lims = lims
            return

        for bg, (ax, arts) in zip(self._bg, self._animated()):
            self._canvas.restore_region(bg)
            for art in arts:
                ax.draw_artist(art)
            self._canvas.blit(ax.bbox)

    def _animated(self):
        """(axis, artists) pairs for everything that changes between frames."""
        return ((self._ax_iq, (self._line_i, self._line_q,
                               self._mean_i_line, self._mean_q_line)),
                (self._ax_fft, (self._line_fft,)),
                (self._ax_wf, (self._wf_image,)))

    def _recache(self, axes):
        """Redraw the static furniture and cache it, without the data in it.

        The artists must be hidden for this draw.  Capturing them into the
        background leaves that frame's trace painted into every subsequent
        blit -- a frozen ghost of the PSD sitting behind the live one, which is
        exactly what a stale background looks like."""
        animated = [art for _, arts in self._animated() for art in arts]
        for art in animated:
            art.set_visible(False)
        try:
            self._canvas.draw()
            self._bg = [self._canvas.copy_from_bbox(a.bbox) for a in axes]
        finally:
            for art in animated:
                art.set_visible(True)

        # Draw them once now, so the frame that pays for a re-cache still shows
        # current data rather than an empty pane.
        for bg, (ax, arts) in zip(self._bg, self._animated()):
            self._canvas.restore_region(bg)
            for art in arts:
                ax.draw_artist(art)
            self._canvas.blit(ax.bbox)

    def _update_waterfall(self, mhz, db):
        """Scroll one PSD row into the waterfall.

        Reallocates only when the PSD length changes, which happens when the
        nperseg or depth slider moves -- not per frame."""
        if self._wf_bins != len(db):
            self._wf_bins = len(db)
            self._wf_data = np.full((WATERFALL_ROWS, self._wf_bins),
                                    np.nan, dtype=np.float32)
            self._wf_clim = None
            if self._wf_image is not None:
                self._wf_image.remove()
            self._wf_image = self._ax_wf.imshow(
                self._wf_data, aspect="auto", origin="lower",
                interpolation="nearest", cmap="viridis",
                extent=(mhz[0], mhz[-1], 0, -WATERFALL_ROWS))
            # New image object and new extent: the cached backgrounds describe
            # the old one.
            self._bg = None

        # New data enters at the bottom and history scrolls upwards.
        self._wf_data[1:] = self._wf_data[:-1]
        self._wf_data[0] = db

        self._wf_image.set_data(self._wf_data)

        # Colour scale latched from the first row, then held.
        if self._wf_clim is None:
            lo = np.floor(np.percentile(db, 5) / 10.0) * 10.0
            hi = np.ceil(np.percentile(db, 99) / 10.0) * 10.0
            self._wf_clim = (lo, max(hi, lo + 10.0))
            self._wf_image.set_clim(*self._wf_clim)


@cli.command(name="hud")
@click.option("--channel", type=int, default=1, show_default=True,
              help="channel to plot (1-indexed)")
@click.option("--pipe", type=int, default=1, show_default=True,
              help="fastrxd pipeline to read from (1-indexed)")
@click.option("--depth", type=int, default=DEFAULT_DEPTH, show_default=True,
              help="packets per capture")
@click.option("--interface", type=str, default=None,
              help="NIC name whose fastrxd to attach to (default: the only "
                   "one running)")
@click.option("--socket", "socket_path", type=str, default=None,
              help="fastrxd socket path (for --socket-path daemons; "
                   "give --interface or this, not both)")
def hud(channel: int, pipe: int, depth: int,
        interface: str | None, socket_path: str | None):
    """Launch the HUD.  `channel` and `pipe` are 1-indexed, matching the GUI.

    Identify the daemon by `interface` or `socket`, as for fastrx.Consumer."""
    app = QApplication.instance() or QApplication(sys.argv)
    try:
        w = HUD(channel=channel - 1, pipe=pipe, depth=depth,
                interface=interface, socket=socket_path)
    except ImportError as exc:
        click.echo(
            "fastrx is not available in this build. It requires Linux, "
            "clang, and libxdp/libbpf at build time; see "
            f"rfmux/streamer/CMakeLists.txt. ({exc})",
            err=True)
        sys.exit(1)
    except ValueError as exc:
        # Raised by Consumer.resolve_socket() or the connection itself --
        # most commonly fastrxd not running yet.
        click.echo(str(exc), err=True)
        sys.exit(1)

    w.resize(1000, 700)
    w.show()
    app.exec()


if __name__ == "__main__":
    cli()
