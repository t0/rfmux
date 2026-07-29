"""
Dockable live pulse-capture panel.

Layout mirrors the SPIE2026 HUD mockup: a toolbar of capture controls,
a status strip (state / counts / rate + per-channel noise), a pulse
tree on the left (newest first, pileup flagged), and a tabbed viewer on
the right (single-pulse I/Q waveform with threshold bands, and a 2×2
grid of running histograms — SNR, peak amplitude, duration, derived τ).

All capture logic lives in
:class:`~rfmux.algorithms.measurement.pulse_capture_session.PulseCaptureSession`;
this panel only configures a session, bridges it through
:class:`~rfmux.tools.periscope.pulse_capture_task.PulseCaptureTask`,
and draws.
"""

from __future__ import annotations

import datetime
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from .utils import (
    ScreenshotMixin,
    IQ_COLORS,
    TABLEAU10_COLORS,
    LINE_WIDTH,
    find_parent_with_attr,
    theme_colors,
)
from .pulse_capture_task import PulseCaptureSignals, PulseCaptureTask
from .pulse_capture_settings_dialog import PulseCaptureSettingsDialog
from ...algorithms.measurement.pulse_capture_session import (
    PulseCaptureConfig,
    PulseCaptureSession,
)
from ...algorithms.measurement.pulse_hdf5 import PulseHDF5Reader
from ...algorithms.measurement.streamer_config import (
    PFB_SAMPLE_RATE,
    slow_sample_rate,
)

# Fast/PFB stream overlay colors (darker variants, HUD convention)
FAST_IQ_COLORS = {"I": "#24478F", "Q": "#8F4724"}

# Channel plot colors: ch1 = I-blue, ch2 = Q-orange (HUD convention),
# further channels from Tableau10.
def _channel_color(channel: int) -> str:
    if channel == 1:
        return IQ_COLORS["I"]
    if channel == 2:
        return IQ_COLORS["Q"]
    return TABLEAU10_COLORS[(channel - 3) % len(TABLEAU10_COLORS)]


_HIST_METRICS = [
    ("snr", "Signal-to-noise (σ)", "peak deviation (σ)"),
    ("amplitude", "Peak amplitude", "amplitude (counts)"),
    ("duration_ms", "Duration", "duration (ms)"),
    ("tau_ms", "Derived τ", "derived τ (ms)"),
]


class PulseCapturePanel(QtWidgets.QWidget, ScreenshotMixin):
    """Live pulse capture with pulse tree, waveform viewer, and histograms."""

    def __init__(
        self,
        parent=None,
        *,
        periscope=None,
        session_manager=None,
        dark_mode: bool = False,
        df_calibrations: Optional[Dict[int, float]] = None,
        module: int = 1,
    ):
        super().__init__(parent)
        self.periscope = periscope
        self.session_manager = session_manager
        self.dark_mode = dark_mode
        self.df_calibrations = df_calibrations

        self.capture_config = PulseCaptureConfig()
        self.task: Optional[PulseCaptureTask] = None
        self.signals: Optional[PulseCaptureSignals] = None
        self.reader: Optional[PulseHDF5Reader] = None
        self._review_mode = False
        self._registered_export = False
        self._tap_registered = False
        self.noise_stats: Dict[int, object] = {}
        self.hdf5_path: Optional[Path] = None
        self._browse_dir: Optional[str] = None

        # Panel-side pulse registry: arrival-ordered (channel, pulse_idx)
        self._pulse_order: List[Tuple[int, int]] = []
        self._pulse_summaries: Dict[Tuple[int, int], dict] = {}
        self._current_view: Optional[Tuple[int, int]] = None
        self._pending_fetch: Optional[Tuple] = None
        self._both_mode = False
        self._current_pair: Optional[Tuple[int, int]] = None
        self._pair_meta: Dict[Tuple[int, int], dict] = {}
        self._stream_counts: Dict[str, int] = {}
        self._noise_by_stream: Dict[str, dict] = {}

        # Follow-latest coalescing: at fast-mode pulse rates the queued
        # per-pulse redraws lag the worker and land on already-evicted
        # waveforms ("loading…" churn).  A short single-shot timer
        # draws only the newest item, which is always still cached.
        self._follow_timer = QtCore.QTimer(self)
        self._follow_timer.setSingleShot(True)
        self._follow_timer.setInterval(100)
        self._follow_timer.timeout.connect(self._show_latest)
        self._counts: Dict[int, int] = {}
        self._last_stats: dict = {}
        self._hist_data: dict = {}
        self._capture_start_wall: Optional[float] = None

        self._setup_ui()
        self._set_run_state(False)
        self.apply_theme(dark_mode)
        self.module_spin.setValue(module)

        # Default the channel list to what Periscope is actually
        # streaming — the tap only ever sees displayed channels, so any
        # other request would wait forever in noise estimation.
        streamed = getattr(periscope, "all_chs", None)
        if streamed:
            self.channels_edit.setText(
                ",".join(str(c) for c in sorted(set(streamed))))

    # ── UI construction ───────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._setup_toolbar(layout)
        self._setup_status(layout)
        self._setup_main_area(layout)

    def _setup_toolbar(self, layout: QtWidgets.QVBoxLayout) -> None:
        bar = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(5, 5, 5, 5)

        self.btn_start = QtWidgets.QPushButton("▶ Start")
        self.btn_start.clicked.connect(self._on_start_stop)
        h.addWidget(self.btn_start)

        self.btn_reestimate = QtWidgets.QPushButton("⟳ Re-estimate Noise")
        self.btn_reestimate.setToolTip(
            "Pause triggering, collect fresh noise statistics, resume")
        self.btn_reestimate.clicked.connect(self._on_reestimate)
        h.addWidget(self.btn_reestimate)

        h.addWidget(QtWidgets.QLabel("Mode:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["slow", "fast", "both"])
        self.mode_combo.setToolTip(
            "slow: ~kHz readout stream (taps the Periscope display "
            "stream)\nfast: ~1.22 MHz PFB stream (max 4 channels; "
            "configures the fast streamer automatically)\n"
            "both: concurrent slow+fast with live pulse matching — the "
            "tree lists matched pairs")
        h.addWidget(self.mode_combo)

        h.addWidget(QtWidgets.QLabel("Channels:"))
        self.channels_edit = QtWidgets.QLineEdit("1,2")
        self.channels_edit.setFixedWidth(70)
        self.channels_edit.setToolTip("Comma-separated 1-indexed channels")
        h.addWidget(self.channels_edit)

        h.addWidget(QtWidgets.QLabel("Module:"))
        self.module_spin = QtWidgets.QSpinBox()
        self.module_spin.setRange(1, 8)
        h.addWidget(self.module_spin)

        h.addWidget(QtWidgets.QLabel("Thresh σ:"))
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.5, 1000.0)
        self.threshold_spin.setValue(5.0)
        self.threshold_spin.setSingleStep(0.5)
        h.addWidget(self.threshold_spin)

        h.addWidget(QtWidgets.QLabel("End σ:"))
        self.end_spin = QtWidgets.QDoubleSpinBox()
        self.end_spin.setRange(0.1, 100.0)
        self.end_spin.setValue(1.5)
        self.end_spin.setSingleStep(0.1)
        self.end_spin.setToolTip(
            "Pulse end requires BOTH I and Q within this band.  Below "
            "~1.2σ the end condition confirms only by chance on Gaussian "
            "noise — 1.5σ recommended.")
        h.addWidget(self.end_spin)

        self.pileup_check = QtWidgets.QCheckBox("Pileup")
        self.pileup_check.setChecked(True)
        self.pileup_check.setToolTip("Split piled-up events on sharp "
                                     "re-triggers during the decay")
        h.addWidget(self.pileup_check)

        self.btn_settings = QtWidgets.QPushButton("Settings…")
        self.btn_settings.setToolTip(
            "All capture parameters: margins, min/max pulse length, "
            "noise training — with live ms → samples math")
        self.btn_settings.clicked.connect(self._on_capture_settings)
        h.addWidget(self.btn_settings)

        self.btn_streamer = QtWidgets.QPushButton("Streamer…")
        self.btn_streamer.setToolTip(
            "Configure decimation / packet format / PFB streaming")
        self.btn_streamer.clicked.connect(self._on_streamer_config)
        h.addWidget(self.btn_streamer)

        self.path_label = QtWidgets.QLabel("HDF5: (session)")
        self.path_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        h.addWidget(self.path_label)
        self.btn_browse = QtWidgets.QPushButton("…")
        self.btn_browse.setFixedWidth(28)
        self.btn_browse.setToolTip("Choose the output HDF5 file")
        self.btn_browse.clicked.connect(self._on_browse)
        h.addWidget(self.btn_browse)

        h.addStretch(1)

        screenshot_btn = QtWidgets.QPushButton("📷")
        screenshot_btn.setToolTip("Export a screenshot of this panel")
        screenshot_btn.clicked.connect(self._export_screenshot)
        h.addWidget(screenshot_btn)

        layout.addWidget(bar)

    def _setup_status(self, layout: QtWidgets.QVBoxLayout) -> None:
        strip = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(strip)
        v.setContentsMargins(8, 2, 8, 2)
        v.setSpacing(1)
        self.status_label = QtWidgets.QLabel()
        self.noise_label = QtWidgets.QLabel()
        f = self.noise_label.font()
        f.setPointSizeF(f.pointSizeF() * 0.9)
        self.noise_label.setFont(f)
        v.addWidget(self.status_label)
        v.addWidget(self.noise_label)
        layout.addWidget(strip)

    def _setup_main_area(self, layout: QtWidgets.QVBoxLayout) -> None:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        self.pulse_tree = QtWidgets.QTreeWidget()
        self.pulse_tree.setHeaderLabels(["Pulse", "Samples", "SNR"])
        self.pulse_tree.setColumnWidth(0, 110)
        self.pulse_tree.setColumnWidth(1, 60)
        self.pulse_tree.itemDoubleClicked.connect(self._on_tree_double_click)
        splitter.addWidget(self.pulse_tree)

        self.viewer_tabs = QtWidgets.QTabWidget()
        self.viewer_tabs.addTab(self._build_pulse_view(), "Pulse View")
        self.viewer_tabs.addTab(self._build_histograms_view(), "Histograms")
        splitter.addWidget(self.viewer_tabs)
        splitter.setSizes([260, 740])
        layout.addWidget(splitter, stretch=1)

    def _build_pulse_view(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)

        nav = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("◀ Prev")
        self.btn_prev.clicked.connect(lambda: self._navigate(-1))
        self.btn_next = QtWidgets.QPushButton("Next ▶")
        self.btn_next.clicked.connect(lambda: self._navigate(+1))
        self.follow_check = QtWidgets.QCheckBox("Follow latest")
        self.follow_check.setChecked(True)
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        nav.addWidget(self.follow_check)
        nav.addStretch(1)
        v.addLayout(nav)

        self.pulse_info = QtWidgets.QLabel("No pulse selected")
        self.pulse_info.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.pulse_info)

        # I and Q stacked vertically (x-linked), each with its own
        # baseline/threshold bands in its own quadrature's sigma.
        self.pulse_plot_i = pg.PlotWidget()
        self.pulse_plot_q = pg.PlotWidget()
        for plot, ylabel in ((self.pulse_plot_i, "I (counts)"),
                             (self.pulse_plot_q, "Q (counts)")):
            item = plot.getPlotItem()
            item.setLabel("left", ylabel)
            item.showGrid(x=True, y=True, alpha=0.3)
        self.pulse_plot_q.getPlotItem().setLabel("bottom", "time", units="s")
        self.pulse_plot_q.setXLink(self.pulse_plot_i)
        v.addWidget(self.pulse_plot_i, stretch=1)
        v.addWidget(self.pulse_plot_q, stretch=1)
        return w

    def _build_histograms_view(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)

        controls = QtWidgets.QHBoxLayout()
        self.log_check = QtWidgets.QCheckBox("Log y")
        self.log_check.toggled.connect(self._render_histograms)
        controls.addWidget(self.log_check)
        controls.addStretch(1)
        v.addLayout(controls)

        grid_holder = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(grid_holder)
        grid.setSpacing(8)
        self.hist_plots: Dict[str, pg.PlotWidget] = {}
        for i, (metric, title, xlabel) in enumerate(_HIST_METRICS):
            plot = pg.PlotWidget()
            item = plot.getPlotItem()
            item.setLabel("bottom", xlabel)
            item.setLabel("left", "count")
            item.showGrid(x=True, y=True, alpha=0.3)
            item.addLegend(offset=(-10, 10))
            self.hist_plots[metric] = plot
            grid.addWidget(plot, i // 2, i % 2)
        v.addWidget(grid_holder, stretch=1)
        return w

    # ── Capture lifecycle ─────────────────────────────────────────

    def _parse_channels(self) -> Optional[List[int]]:
        try:
            channels = sorted({int(tok) for tok in
                               self.channels_edit.text().replace(" ", "")
                               .split(",") if tok})
        except ValueError:
            channels = []
        if not channels or any(c < 1 for c in channels):
            QtWidgets.QMessageBox.warning(
                self, "Pulse Capture",
                "Channels must be a comma-separated list of 1-indexed "
                "channel numbers (e.g. \"1,2\").")
            return None
        return channels

    def _resolve_runtime(self):
        """The object owning register_pulse_tap (Periscope main window)."""
        if self.periscope is not None:
            return self.periscope
        return find_parent_with_attr(self, "register_pulse_tap")

    def _resolve_hdf5_path(self, module: int) -> Path:
        stamp = datetime.datetime.now().strftime("%H%M%S")
        name = f"pulse_module{module}_{stamp}.h5"
        if self.hdf5_path is not None:
            return self.hdf5_path
        sm = self.session_manager
        if sm is not None and getattr(sm, "is_active", False) \
                and sm.session_path is not None:
            return Path(sm.session_path) / name
        base = Path(self._browse_dir) if self._browse_dir else Path.home()
        return base / name

    def _sync_config_from_toolbar(self) -> None:
        self.capture_config.threshold_sigma = float(
            self.threshold_spin.value())
        self.capture_config.end_sigma = float(self.end_spin.value())
        self.capture_config.enable_pileup = self.pileup_check.isChecked()

    def _sync_toolbar_from_config(self) -> None:
        self.threshold_spin.setValue(self.capture_config.threshold_sigma)
        self.end_spin.setValue(self.capture_config.end_sigma)
        self.pileup_check.setChecked(self.capture_config.enable_pileup)

    def _current_sample_rate(self, mode: str) -> float:
        if mode == "fast":
            return PFB_SAMPLE_RATE
        runtime = self._resolve_runtime()
        dec = getattr(runtime, "actual_dec_stage", None)
        return slow_sample_rate(dec if dec is not None else 6)

    def _on_capture_settings(self) -> None:
        self._sync_config_from_toolbar()
        mode = self.mode_combo.currentText()
        channels = self._parse_channels() or [1]
        dlg = PulseCaptureSettingsDialog(
            self,
            config=self.capture_config,
            sample_rate=self._current_sample_rate(mode),
            mode=mode,
            n_channels=len(channels),
        )
        if dlg.exec():
            self.capture_config = dlg.get_config()
            self._sync_toolbar_from_config()

    def _on_streamer_config(self) -> None:
        runtime = self._resolve_runtime()
        handler = getattr(runtime, "_show_streamer_config_dialog", None)
        if handler is None:
            QtWidgets.QMessageBox.information(
                self, "Pulse Capture",
                "Streamer configuration is available from the main "
                "Periscope window.")
            return
        handler()

    def _on_browse(self) -> None:
        dlg = QtWidgets.QFileDialog(
            self, "Pulse capture output file",
            str(self._resolve_hdf5_path(int(self.module_spin.value()))),
            "HDF5 files (*.h5 *.hdf5)")
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dlg.setDefaultSuffix("h5")

        def _chosen():
            files = dlg.selectedFiles()
            if files:
                self.hdf5_path = Path(files[0])
                self._browse_dir = str(self.hdf5_path.parent)
                self.path_label.setText(f"HDF5: {self.hdf5_path}")

        dlg.accepted.connect(_chosen)
        dlg.open()  # non-modal — modal exec() can hang on Linux

    def _on_start_stop(self) -> None:
        if self.task is not None:
            self._on_stop()
        else:
            self._on_start()

    def _on_start(self) -> None:
        channels = self._parse_channels()
        if channels is None:
            return
        runtime = self._resolve_runtime()
        if runtime is None:
            QtWidgets.QMessageBox.warning(
                self, "Pulse Capture",
                "No running Periscope stream found to tap.")
            return
        mode = self.mode_combo.currentText()
        crs = getattr(runtime, "crs", None)
        host = None
        if mode in ("fast", "both"):
            if len(channels) > 4:
                QtWidgets.QMessageBox.warning(
                    self, "Pulse Capture",
                    "The PFB (fast) streamer supports at most 4 channels "
                    f"— {len(channels)} requested.")
                return
            host = getattr(crs, "tuber_hostname", None) \
                or getattr(runtime, "host", None)
            if crs is None or host in (None, "OFFLINE"):
                QtWidgets.QMessageBox.warning(
                    self, "Pulse Capture",
                    f"{mode} mode needs a CRS connection to configure "
                    "the PFB streamer.")
                return
        else:
            if getattr(runtime, "_pulse_tap", None) is not None:
                QtWidgets.QMessageBox.warning(
                    self, "Pulse Capture",
                    "Another pulse capture is already tapping the stream. "
                    "Stop it first.")
                return

            # Any channel the packet carries can be captured (display is
            # irrelevant) — but the packet width is a hard limit:
            # 128 channels in short-packet mode, 1024 in long mode.
            max_ch = 128 if getattr(runtime, "is_short_packet", False) \
                else 1024
            too_high = [c for c in channels if c > max_ch]
            if too_high:
                QtWidgets.QMessageBox.warning(
                    self, "Pulse Capture",
                    f"Channel(s) {too_high} exceed the current packet "
                    f"width ({max_ch} channels"
                    f"{' — short-packet mode' if max_ch == 128 else ''}).")
                return

        module = int(self.module_spin.value())
        path = self._resolve_hdf5_path(module)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QtWidgets.QMessageBox.warning(
                self, "Pulse Capture", f"Cannot create {path.parent}: {e}")
            return

        self._sync_config_from_toolbar()
        fs = self._current_sample_rate(mode)
        config_errors = [m for s, m in
                         self.capture_config.validate(fs) if s == "error"]
        if config_errors:
            QtWidgets.QMessageBox.warning(
                self, "Pulse Capture",
                "Invalid capture settings:\n- "
                + "\n- ".join(config_errors))
            return

        if mode == "both":
            from ...algorithms.measurement.pulse_capture_dual import (
                DualPulseCaptureSession,
            )
            session = DualPulseCaptureSession(
                channels=channels,
                module=module,
                slow_rate=self._current_sample_rate("slow"),
                fast_rate=PFB_SAMPLE_RATE,
                config=self.capture_config,
                hdf5_path=path,
                df_calibrations=self.df_calibrations,
            )
        else:
            session = PulseCaptureSession(
                channels=channels,
                module=module,
                streamer_mode=mode,
                hdf5_path=path,
                df_calibrations=self.df_calibrations,
                sample_rate=fs,
                **self.capture_config.session_kwargs(fs),
            )
        self.signals = PulseCaptureSignals()
        self.task = PulseCaptureTask(session, self.signals, mode=mode,
                                     crs=crs, host=host, module=module)
        conn = QtCore.Qt.ConnectionType.QueuedConnection
        self.signals.noise_estimated.connect(self._on_noise_estimated, conn)
        self.signals.noise_progress.connect(self._on_noise_progress, conn)
        self.signals.pulse_detected.connect(self._on_pulse_detected, conn)
        self.signals.pair_matched.connect(self._on_pair_matched, conn)
        self.signals.stats_updated.connect(self._on_stats, conn)
        self.signals.histograms_updated.connect(self._on_histograms, conn)
        self.signals.waveform_ready.connect(self._on_waveform_ready, conn)
        self.signals.error.connect(self._on_error, conn)
        self.signals.finished.connect(self._on_task_finished, conn)

        # A fresh capture replaces any file we were browsing
        if self.reader is not None:
            self.reader.close()
            self.reader = None

        self._both_mode = (mode == "both")
        self._reset_results(channels)
        self._registered_export = False
        self.path_label.setText(f"HDF5: {path}")
        self._capture_start_wall = time.time()

        self.task.start()
        if mode == "slow":
            runtime.register_pulse_tap(self.task.enqueue, channels)
            self._tap_registered = True
        self._set_run_state(True)
        self._set_status("● Estimating noise…", "#FFCC33")
        print(f"[PulseCapture] Start — channels {channels}, module "
              f"{module}, mode {mode}, "
              f"threshold {self.threshold_spin.value():g}σ, "
              f"end {self.end_spin.value():g}σ → {path}")

    def _on_stop(self) -> None:
        self._unregister_tap()
        if self.task is not None:
            self.task.request_stop()
        self.btn_start.setEnabled(False)  # until finished arrives

    def _on_task_finished(self) -> None:
        finalized_path = None
        if self.task is not None:
            finalized_path = self.task.session.hdf5_path
            self.task.wait(2000)
            self.task = None
        self.signals = None
        self._set_run_state(False)
        n = sum(self._counts.values())
        self._set_status(f"● Stopped — {n} pulses captured", "#9A9A9A")
        print(f"[PulseCapture] Stopped — {n} pulses captured")

        # Reopen the finalized file so every pulse stays browsable
        if finalized_path is not None and n > 0:
            try:
                self.reader = PulseHDF5Reader(finalized_path)
            except Exception as e:
                print(f"[PulseCapture] Could not reopen {finalized_path} "
                      f"for browsing: {e}")

    def _on_reestimate(self) -> None:
        if self.task is not None:
            self.task.request_noise_reestimate()
            self._set_status("● Re-estimating noise…", "#FFCC33")

    def _unregister_tap(self) -> None:
        if self._tap_registered:
            runtime = self._resolve_runtime()
            if runtime is not None:
                runtime.unregister_pulse_tap()
            self._tap_registered = False

    def closeEvent(self, event) -> None:
        self._unregister_tap()
        if self.task is not None:
            self.task.request_stop()
            self.task.wait(3000)
            self.task = None
        if self.reader is not None:
            self.reader.close()
            self.reader = None
        super().closeEvent(event)

    # ── Review mode (existing HDF5 file, no live capture) ─────────

    def load_from_hdf5(self, path) -> None:
        """Open an existing pulse-capture HDF5 for browsing."""
        self.reader = PulseHDF5Reader(path)
        meta = self.reader.metadata
        channels = [int(c) for c in self.reader.channels]

        # Restore capture parameters so bands/labels reflect the file
        if "streamer_mode" in meta:
            self.mode_combo.setCurrentText(str(meta["streamer_mode"]))
        if "threshold_sigma" in meta:
            self.threshold_spin.setValue(float(meta["threshold_sigma"]))
        if "end_sigma" in meta:
            self.end_spin.setValue(float(meta["end_sigma"]))
        if "module" in meta:
            self.module_spin.setValue(int(meta["module"]))
        self.channels_edit.setText(",".join(str(c) for c in channels))

        started = None
        if "capture_start" in meta:
            started = datetime.datetime.fromtimestamp(
                float(meta["capture_start"])).strftime("%H:%M:%S")
        self._both_mode = self.reader.dual
        self._reset_results(channels, started=started)

        if self.reader.dual:
            self._load_dual_review(channels, path)
            return

        self.noise_stats = {c: self.reader.noise_stats(c) for c in channels}
        parts = [f"Ch{c} I={ns.mean_I:.1f}±{ns.std_I:.2f}, "
                 f"Q={ns.mean_Q:.1f}±{ns.std_Q:.2f}"
                 for c, ns in sorted(self.noise_stats.items())]
        self.noise_label.setText("Noise:  " + "   |   ".join(parts))

        # Tree from lazy per-pulse metadata (ascending insert at row 0
        # → newest first), with fallbacks for files written before the
        # unified snr/peak_amp/tau_s attrs existed.
        for c in channels:
            for m in self.reader.iter_pulse_metadata(c):
                idx = int(m["pulse_idx"])
                snr = float(m.get("snr") or max(
                    m.get("peak_snr_I", 0.0), m.get("peak_snr_Q", 0.0)))
                summary = {
                    "n_samples": int(m.get("n_samples", 0)),
                    "pileup": bool(m.get("pileup", False)),
                    "peak_amp": float(m.get("peak_amp") or max(
                        m.get("peak_I", 0.0), m.get("peak_Q", 0.0))),
                    "snr": snr,
                    "duration_ms": float(m.get("duration_s", 0.0)) * 1e3,
                    "tau_ms": float(m.get("tau_s", float("nan"))) * 1e3,
                    "timestamp": float(m.get("timestamp", 0.0)),
                }
                self._pulse_order.append((c, idx))
                self._pulse_summaries[(c, idx)] = summary
                self._counts[c] = self._counts.get(c, 0) + 1
                parent = self._channel_items.get(c)
                if parent is not None:
                    label = ("⚠" if summary["pileup"] else "◆") \
                        + f" #{idx:06d}"
                    item = QtWidgets.QTreeWidgetItem(
                        [label, str(summary["n_samples"]),
                         f"{snr:.1f}σ"])
                    item.setData(0, QtCore.Qt.ItemDataRole.UserRole,
                                 ("pulse", c, idx))
                    if summary["pileup"]:
                        for col in range(3):
                            item.setBackground(col, QtGui.QColor(
                                "#3a3320" if self.dark_mode else "#fff3c2"))
                    parent.insertChild(0, item)
                    parent.setText(0, f"▤ Channel {c} ({self._counts[c]})")

        self._hist_data = self.reader.get_histograms()
        self._render_histograms()

        self._enter_review_state(path, f"{sum(self._counts.values())} "
                                       f"pulses")
        if self._pulse_order:
            self._show_pulse(*self._pulse_order[-1])

    def _load_dual_review(self, channels: List[int], path) -> None:
        """Populate the pair tree from a dual ('both' mode) file."""
        self.noise_stats = {
            c: self.reader.noise_stats(c, "slow") for c in channels}
        parts = []
        for stream in ("slow", "fast"):
            stats = {c: self.reader.noise_stats(c, stream)
                     for c in channels}
            self._noise_by_stream[stream] = stats
            parts.append(f"{stream}: " + " | ".join(
                f"Ch{c} σI={ns.std_I:.2f}"
                for c, ns in sorted(stats.items())))
        self.noise_label.setText("Noise:  " + "   —   ".join(parts))

        self.follow_check.setChecked(False)
        for c in channels:
            for pair in self.reader.iter_matches(c):
                lean = {
                    "channel": c,
                    "pair_idx": pair["pair_idx"],
                    "slow_idx": pair["slow_idx"],
                    "fast_idx": pair["fast_idx"],
                    "time_offset": (
                        pair["time_offset"]
                        if pair.get("time_offset") is not None
                        and np.isfinite(pair["time_offset"]) else None),
                    "slow_summary": (
                        self.reader.get_pulse_metadata(
                            c, pair["slow_idx"], "slow")
                        if pair["slow_idx"] else None),
                    "fast_summary": (
                        self.reader.get_pulse_metadata(
                            c, pair["fast_idx"], "fast")
                        if pair["fast_idx"] else None),
                    "has_slow_tod": "slow_tod" in pair,
                    "has_fast_tod": "fast_tod" in pair,
                }
                self._on_pair_matched(lean)

        self._hist_data = self.reader.get_histograms("slow")
        self._render_histograms()
        self._enter_review_state(
            path, f"{sum(self._counts.values())} pairs")
        if self._pulse_order:
            self._show_pair(*self._pulse_order[-1])

    def _enter_review_state(self, path, what: str) -> None:
        self._review_mode = True
        self._set_run_state(False)
        self.btn_start.setEnabled(False)
        self.btn_start.setToolTip("Review mode — this panel browses an "
                                  "existing capture file")
        for w in (self.mode_combo, self.channels_edit, self.module_spin,
                  self.threshold_spin, self.end_spin, self.pileup_check,
                  self.btn_browse):
            w.setEnabled(False)
        self.path_label.setText(f"HDF5: {path}")
        self._set_status(
            f"● Review Mode — {what} — {Path(path).name}", "#3366CC")
        self.follow_check.setChecked(False)

    def _set_run_state(self, running: bool) -> None:
        self.btn_start.setText("■ Stop" if running else "▶ Start")
        self.btn_start.setEnabled(True)
        self.btn_start.setStyleSheet(
            "background-color: #4CC38A; color: #10241B; font-weight: bold;"
            if running else "")
        self.btn_reestimate.setEnabled(running)
        for w in (self.mode_combo, self.channels_edit, self.module_spin,
                  self.threshold_spin, self.end_spin, self.pileup_check,
                  self.btn_browse, self.btn_streamer, self.btn_settings):
            w.setEnabled(not running)
        if not running:
            self._set_status("● Idle", "#9A9A9A")

    def _reset_results(self, channels: List[int],
                       started: Optional[str] = None) -> None:
        self._pulse_order.clear()
        self._pulse_summaries.clear()
        self._pair_meta.clear()
        self._stream_counts = {"slow": 0, "fast": 0}
        self._noise_by_stream = {}
        self._current_pair = None
        self._current_view = None
        self._counts = {c: 0 for c in channels}
        self._hist_data = {}
        self.noise_stats = {}
        self.pulse_tree.clear()
        self._channel_items: Dict[int, QtWidgets.QTreeWidgetItem] = {}
        for c in channels:
            item = QtWidgets.QTreeWidgetItem([f"▤ Channel {c} (0)", "", ""])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, ("channel", c))
            self.pulse_tree.addTopLevelItem(item)
            item.setExpanded(True)
            self._channel_items[c] = item
        meta = QtWidgets.QTreeWidgetItem(["▦ Metadata", "", ""])
        meta.addChild(QtWidgets.QTreeWidgetItem(
            [f"mode={self.mode_combo.currentText()}", "", ""]))
        meta.addChild(QtWidgets.QTreeWidgetItem(
            [f"σ={self.threshold_spin.value():g}  "
             f"end={self.end_spin.value():g}", "", ""]))
        meta.addChild(QtWidgets.QTreeWidgetItem(
            [f"started "
             f"{started or datetime.datetime.now().strftime('%H:%M:%S')}",
             "", ""]))
        self.pulse_tree.addTopLevelItem(meta)
        self.pulse_plot_i.clear()
        self.pulse_plot_q.clear()
        self.pulse_info.setText("No pulse selected")
        self._render_histograms()

    # ── Signal handlers (GUI thread) ──────────────────────────────

    def _on_noise_progress(self, progress: dict) -> None:
        collected = progress.get("collected", {})
        target = progress.get("target", 0)
        prefix = (f"[{progress['stream']}] "
                  if progress.get("stream") else "")
        parts = [f"Ch{c} {collected[c]}/{target}"
                 for c in sorted(collected)]
        self._set_status(f"● Estimating noise — {prefix}"
                         + " | ".join(parts), "#FFCC33")

    def _on_noise_estimated(self, noise_stats: dict) -> None:
        if "stream" in noise_stats and "stats" in noise_stats:
            stream = noise_stats["stream"]
            stats = noise_stats["stats"]
            self._noise_by_stream[stream] = stats
            if stream == "slow":
                self.noise_stats = stats
            parts = [f"{s}: " + " | ".join(
                f"Ch{c} σI={ns.std_I:.2f}"
                for c, ns in sorted(st.items()))
                for s, st in sorted(self._noise_by_stream.items())]
            self.noise_label.setText("Noise:  " + "   —   ".join(parts))
            print(f"[PulseCapture] Noise estimated ({stream})")
            self._refresh_status_line()
            return

        self.noise_stats = noise_stats
        parts = []
        for c in sorted(noise_stats):
            ns = noise_stats[c]
            parts.append(f"Ch{c} I={ns.mean_I:.1f}±{ns.std_I:.2f}, "
                         f"Q={ns.mean_Q:.1f}±{ns.std_Q:.2f}")
        self.noise_label.setText("Noise:  " + "   |   ".join(parts))
        print("[PulseCapture] Noise estimated: " + " | ".join(parts))
        self._refresh_status_line()
        # Show what the estimator saw (until the first pulse replaces it)
        if self.follow_check.isChecked() or self._current_view is None:
            self._show_noise_segment()

        # Register the streamed HDF5 with the session (once per capture)
        if (not self._registered_export and self.task is not None
                and self.session_manager is not None
                and getattr(self.session_manager, "is_active", False)):
            path = self.task.session.hdf5_path
            sp = self.session_manager.session_path
            if path is not None and sp is not None \
                    and Path(sp) in Path(path).parents:
                self.session_manager.register_external_file(
                    str(path), "pulse",
                    f"module{self.task.session.module}")
            self._registered_export = True

    def _on_pulse_detected(self, channel: int, pulse_idx: int,
                           summary: dict) -> None:
        if self._both_mode:
            stream = summary.get("stream", "slow")
            self._stream_counts[stream] = \
                self._stream_counts.get(stream, 0) + 1
            return  # both-mode tree shows matched pairs, not raw pulses
        key = (channel, pulse_idx)
        self._pulse_order.append(key)
        self._pulse_summaries[key] = summary
        self._counts[channel] = self._counts.get(channel, 0) + 1

        parent = self._channel_items.get(channel)
        if parent is not None:
            pileup = summary.get("pileup", False)
            label = ("⚠" if pileup else "◆") + f" #{pulse_idx:06d}"
            item = QtWidgets.QTreeWidgetItem(
                [label, str(summary.get("n_samples", "")),
                 f"{summary.get('snr', 0):.1f}σ"])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole,
                         ("pulse", channel, pulse_idx))
            if pileup:
                for col in range(3):
                    item.setBackground(col, QtGui.QColor(
                        "#3a3320" if self.dark_mode else "#fff3c2"))
            parent.insertChild(0, item)
            parent.setText(0, f"▤ Channel {channel} "
                              f"({self._counts[channel]})")

        if self.follow_check.isChecked() \
                and not self._follow_timer.isActive():
            self._follow_timer.start()

    def _on_stats(self, stats: dict) -> None:
        self._last_stats = stats
        self._refresh_status_line()

    def _on_pair_matched(self, pair: dict) -> None:
        ch = pair["channel"]
        pair_idx = pair["pair_idx"]
        key = (ch, pair_idx)
        self._pair_meta[key] = pair
        self._pulse_order.append(key)
        self._counts[ch] = self._counts.get(ch, 0) + 1

        matched = pair["slow_idx"] is not None \
            and pair["fast_idx"] is not None
        if matched:
            dt = pair.get("time_offset") or 0.0
            label = (f"◆ Pair #{pair_idx:04d}  "
                     f"s#{pair['slow_idx']}/f#{pair['fast_idx']}")
            detail = f"Δt={dt*1e6:+.0f}µs"
        else:
            side = "slow" if pair["slow_idx"] is not None else "fast"
            idx = pair["slow_idx"] or pair["fast_idx"]
            label = f"◐ Pair #{pair_idx:04d}  {side} only #{idx}"
            detail = ("+TOD" if pair.get("has_slow_tod")
                      or pair.get("has_fast_tod") else "")
        summ = pair.get("slow_summary") or pair.get("fast_summary") or {}
        parent = self._channel_items.get(ch)
        if parent is not None:
            item = QtWidgets.QTreeWidgetItem(
                [label, detail, f"{summ.get('snr', 0):.1f}σ"])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole,
                         ("pair", ch, pair_idx))
            if not matched:
                for col in range(3):
                    item.setBackground(col, QtGui.QColor(
                        "#33251c" if self.dark_mode else "#ffe8d9"))
            parent.insertChild(0, item)
            parent.setText(0, f"▤ Channel {ch} ({self._counts[ch]} pairs)")

        if self.follow_check.isChecked() \
                and not self._follow_timer.isActive():
            self._follow_timer.start()

    def _refresh_status_line(self) -> None:
        s = self._last_stats
        if self._both_mode and "pairs_matched" in s:
            slow_n = s.get("slow", {}).get("total_pulses", 0)
            fast_n = s.get("fast", {}).get("total_pulses", 0)
            self._set_status(
                f"● Capturing — slow {slow_n} | fast {fast_n} pulses — "
                f"{s['pairs_matched']} matched / "
                f"{s['pairs_unmatched']} one-sided pairs", "#4CC38A")
            return
        total = s.get("total_pulses", 0)
        rate = s.get("rate_per_min", 0.0)
        per_ch = s.get("per_channel", {})
        ch_str = " | ".join(f"Ch{c}: {n}" for c, n in sorted(per_ch.items()))
        elapsed = int(s.get("elapsed_s", 0))
        hh, rem = divmod(elapsed, 3600)
        mm, ss = divmod(rem, 60)
        dropped = s.get("dropped_invalid_ts", 0)
        drop_str = f" — {dropped} dropped (no timestamp)" if dropped else ""
        self._set_status(
            f"● Capturing — {total} pulses ({rate:.1f}/min) — {ch_str} — "
            f"{hh:02d}:{mm:02d}:{ss:02d}{drop_str}", "#4CC38A")

    def _on_histograms(self, data: dict) -> None:
        if "stream" in data and "data" in data:
            # both-mode: per-stream payloads; display the slow stream
            if data["stream"] != "slow":
                return
            self._hist_data = data["data"]
        else:
            self._hist_data = data
        self._render_histograms()

    def _on_error(self, message: str) -> None:
        self.noise_label.setToolTip(message)
        self._set_status(f"● Error: {message}", "#E5484D")
        print(f"[PulseCapture] ERROR: {message}")

    def _show_noise_segment(self) -> None:
        """Plot the noise-training segment with the fitted baselines and
        trigger/end bands — visual confirmation of what the estimator saw."""
        if self.task is None:
            return
        noise_data = getattr(self.task.session, "noise_data", {})
        channel = next((c for c in sorted(self.noise_stats)
                        if c in noise_data and len(noise_data[c])), None)
        if channel is None:
            return
        arr = noise_data[channel]
        ns = self.noise_stats[channel]
        thr = float(self.threshold_spin.value())
        end = float(self.end_spin.value())

        self._current_view = None
        self.pulse_info.setText(
            f"Noise training segment — Channel {channel} "
            f"({len(arr)} samples)\n"
            f"I = {ns.mean_I:.1f} ± {ns.std_I:.2f}   "
            f"Q = {ns.mean_Q:.1f} ± {ns.std_Q:.2f}\n"
            f"bands: ±{thr:g}σ trigger (dashed), ±{end:g}σ end (dotted)")

        x = np.arange(len(arr))
        self.pulse_plot_i.clear()
        self.pulse_plot_q.clear()
        self.pulse_plot_i.setTitle(f"Noise training — Channel {channel}")
        self.pulse_plot_q.setTitle(None)
        self.pulse_plot_q.getPlotItem().setLabel("bottom", "sample")
        for plot, data, color, mean, std in (
                (self.pulse_plot_i, arr.real, IQ_COLORS["I"],
                 ns.mean_I, ns.std_I),
                (self.pulse_plot_q, arr.imag, IQ_COLORS["Q"],
                 ns.mean_Q, ns.std_Q)):
            plot.plot(x, data, pen=pg.mkPen(color, width=1.0))
            plot.addLine(y=mean, pen=pg.mkPen(
                color, width=0.8, style=QtCore.Qt.PenStyle.DotLine))
            for sign in (+1, -1):
                plot.addLine(
                    y=mean + sign * thr * std,
                    pen=pg.mkPen("#888888", width=0.8,
                                 style=QtCore.Qt.PenStyle.DashLine))
                plot.addLine(
                    y=mean + sign * end * std,
                    pen=pg.mkPen("#666666", width=0.8,
                                 style=QtCore.Qt.PenStyle.DotLine))

    def _set_status(self, text: str, color: str) -> None:
        self.status_label.setText(
            f'<span style="color:{color}">●</span> <b>{text[2:]}</b>'
            if text.startswith("● ") else text)

    # ── Pulse view ────────────────────────────────────────────────

    def _show_latest(self) -> None:
        """Coalesced follow-latest redraw (newest pulse or pair)."""
        if not self.follow_check.isChecked() or not self._pulse_order:
            return
        latest = self._pulse_order[-1]
        if self._both_mode:
            self._show_pair(*latest)
        else:
            self._show_pulse(*latest)

    def _on_tree_double_click(self, item, column) -> None:
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not data:
            return
        if data[0] == "pulse":
            self.follow_check.setChecked(False)
            self.viewer_tabs.setCurrentIndex(0)
            self._show_pulse(data[1], data[2])
        elif data[0] == "pair":
            self.follow_check.setChecked(False)
            self.viewer_tabs.setCurrentIndex(0)
            self._show_pair(data[1], data[2])

    def _navigate(self, step: int) -> None:
        if not self._pulse_order:
            return
        self.follow_check.setChecked(False)
        current = (self._current_pair if self._both_mode
                   else self._current_view)
        if current in self._pulse_order:
            i = self._pulse_order.index(current) + step
        else:
            i = len(self._pulse_order) - 1
        i = max(0, min(len(self._pulse_order) - 1, i))
        if self._both_mode:
            self._show_pair(*self._pulse_order[i])
        else:
            self._show_pulse(*self._pulse_order[i])

    def _on_waveform_ready(self, channel: int, pulse_idx: int) -> None:
        """Worker warmed the cache (or failed) — redraw if still viewing."""
        if self._both_mode:
            if self._current_pair is not None:
                self._show_pair(*self._current_pair)
        elif self._current_view == (channel, pulse_idx):
            self._show_pulse(channel, pulse_idx)

    def current_hdf5_path(self) -> Optional[str]:
        """Resolved path of the file this panel is capturing/browsing."""
        path = None
        if self.task is not None:
            path = self.task.session.hdf5_path
        elif self.reader is not None:
            path = self.reader.path
        elif self.hdf5_path is not None:
            path = self.hdf5_path
        return str(Path(path).resolve()) if path is not None else None

    def _get_waveform(self, channel: int, pulse_idx: int,
                      stream: Optional[str] = None) -> Optional[dict]:
        if self.task is not None:
            wf = self.task.get_pulse(channel, pulse_idx, stream)
            if wf is not None:
                return wf
        if self.reader is not None:
            return self.reader.get_pulse(channel, pulse_idx,
                                         stream=stream)
        return None

    def _get_pair(self, channel: int, pair_idx: int) -> Optional[dict]:
        if self.task is not None:
            pair = self.task.get_pair(channel, pair_idx)
            if pair is not None:
                return pair
        if self.reader is not None and self.reader.dual:
            return self.reader.get_match(channel, pair_idx)
        return None

    def _show_pulse(self, channel: int, pulse_idx: int) -> None:
        summary = self._pulse_summaries.get((channel, pulse_idx))
        wf = self._get_waveform(channel, pulse_idx)
        self._current_view = (channel, pulse_idx)

        if summary is None:
            return
        pile = "[pileup]" if summary.get("pileup") else "[no pileup]"
        tau_ms = summary.get("tau_ms", float("nan"))
        tau_str = f"{tau_ms:.2f} ms" if np.isfinite(tau_ms) else "n/a"
        self.pulse_info.setText(
            f"Pulse #{pulse_idx:06d} — Channel {channel}   {pile}\n"
            f"{summary.get('n_samples', 0)} samples   "
            f"{summary.get('duration_ms', 0):.2f} ms   "
            f"peak {summary.get('peak_amp', 0):.0f} cts "
            f"({summary.get('snr', 0):.1f}σ)\n"
            f"derived τ = {tau_str}")

        for plot in (self.pulse_plot_i, self.pulse_plot_q):
            plot.clear()
            plot.setTitle(None)
        self.pulse_plot_q.getPlotItem().setLabel("bottom", "time", units="s")
        if wf is None:
            # Evicted from the live cache — fetch it from the HDF5 file
            # via the worker thread (waveform_ready redraws on arrival).
            key = (channel, pulse_idx)
            if self.task is not None and self._pending_fetch != key:
                self._pending_fetch = key
                self.task.request_waveform(channel, pulse_idx)
                self.pulse_plot_i.setTitle("loading waveform from file…")
            else:
                self.pulse_plot_i.setTitle("waveform not available")
            return
        self._pending_fetch = None

        t = np.asarray(wf["Time"], dtype=np.float64)
        finite = np.isfinite(t)
        t0 = t[finite][0] if np.any(finite) else 0.0
        t_rel = t - t0
        amp_I = np.asarray(wf["Amp_I"], dtype=np.float64)
        amp_Q = np.asarray(wf["Amp_Q"], dtype=np.float64)

        ns = self.noise_stats.get(channel)
        thr = float(self.threshold_spin.value())
        end = float(self.end_spin.value())
        for plot, data, color, mean, std in (
                (self.pulse_plot_i, amp_I, IQ_COLORS["I"],
                 getattr(ns, "mean_I", None), getattr(ns, "std_I", None)),
                (self.pulse_plot_q, amp_Q, IQ_COLORS["Q"],
                 getattr(ns, "mean_Q", None), getattr(ns, "std_Q", None))):
            plot.plot(t_rel, data, pen=pg.mkPen(color, width=LINE_WIDTH))
            if mean is None or std is None:
                continue
            plot.addLine(y=mean, pen=pg.mkPen(
                color, width=0.8, style=QtCore.Qt.PenStyle.DotLine))
            for sign in (+1, -1):
                plot.addLine(
                    y=mean + sign * thr * std,
                    pen=pg.mkPen("#888888", width=0.8,
                                 style=QtCore.Qt.PenStyle.DashLine))
                plot.addLine(
                    y=mean + sign * end * std,
                    pen=pg.mkPen("#666666", width=0.8,
                                 style=QtCore.Qt.PenStyle.DotLine))

    def _show_pair(self, channel: int, pair_idx: int) -> None:
        """Matched-pair overlay: dense fast trace under slow markers,
        per quadrature (HUD 'both' view)."""
        pair = self._get_pair(channel, pair_idx)
        meta = self._pair_meta.get((channel, pair_idx)) or pair
        if meta is None:
            return
        self._current_pair = (channel, pair_idx)
        self._current_view = None

        loading = False
        slow_wf = fast_wf = None
        if meta.get("slow_idx") is not None:
            slow_wf = self._get_waveform(channel, meta["slow_idx"],
                                         "slow")
            if slow_wf is None and self.task is not None:
                key = ("slow", channel, meta["slow_idx"])
                if self._pending_fetch != key:
                    self._pending_fetch = key
                    self.task.request_waveform(channel,
                                               meta["slow_idx"], "slow")
                loading = True
        elif pair is not None:
            slow_wf = pair.get("slow_tod")
        if meta.get("fast_idx") is not None:
            fast_wf = self._get_waveform(channel, meta["fast_idx"],
                                         "fast")
            if fast_wf is None and self.task is not None:
                key = ("fast", channel, meta["fast_idx"])
                if self._pending_fetch != key:
                    self._pending_fetch = key
                    self.task.request_waveform(channel,
                                               meta["fast_idx"], "fast")
                loading = True
        elif pair is not None:
            fast_wf = pair.get("fast_tod")

        matched = meta.get("slow_idx") is not None \
            and meta.get("fast_idx") is not None
        dt = meta.get("time_offset")
        summ = meta.get("slow_summary") or meta.get("fast_summary") or {}
        tau_ms = summ.get("tau_ms", float("nan"))
        self.pulse_info.setText(
            f"Pair #{pair_idx:04d} — Channel {channel}   "
            f"[{'matched' if matched else 'one-sided'}]\n"
            f"slow #{meta.get('slow_idx')} / fast #{meta.get('fast_idx')}"
            + (f"   Δt = {dt*1e6:+.0f} µs" if dt is not None else "")
            + (f"\nslow SNR {summ.get('snr', 0):.1f}σ, "
               f"τ = {tau_ms:.2f} ms"
               if np.isfinite(tau_ms) else ""))

        for plot in (self.pulse_plot_i, self.pulse_plot_q):
            plot.clear()
            plot.setTitle(None)
        self.pulse_plot_q.getPlotItem().setLabel("bottom", "time", units="s")
        if loading:
            self.pulse_plot_i.setTitle("loading waveforms from file…")
        if slow_wf is None and fast_wf is None:
            if not loading:
                self.pulse_plot_i.setTitle("waveforms not available")
            return

        # Shared clock → common time origin across both streams
        t0 = None
        for wf in (fast_wf, slow_wf):
            if wf is not None:
                t = np.asarray(wf["Time"], dtype=np.float64)
                finite = t[np.isfinite(t)]
                if len(finite):
                    t0 = (finite[0] if t0 is None
                          else min(t0, finite[0]))
        if t0 is None:
            t0 = 0.0

        for quad, plot in (("I", self.pulse_plot_i),
                           ("Q", self.pulse_plot_q)):
            if fast_wf is not None:
                t_rel = np.asarray(fast_wf["Time"], float) - t0
                plot.plot(t_rel,
                          np.asarray(fast_wf[f"Amp_{quad}"], float),
                          pen=pg.mkPen(FAST_IQ_COLORS[quad], width=1.0))
            if slow_wf is not None:
                t_rel = np.asarray(slow_wf["Time"], float) - t0
                data = np.asarray(slow_wf[f"Amp_{quad}"], float)
                plot.plot(t_rel, data, pen=pg.mkPen(
                    IQ_COLORS[quad], width=LINE_WIDTH * 0.6))
                plot.plot(t_rel, data, pen=None, symbol="o",
                          symbolSize=6,
                          symbolBrush=IQ_COLORS[quad],
                          symbolPen=pg.mkPen("w", width=0.6))

    # ── Histograms ────────────────────────────────────────────────

    def _render_histograms(self) -> None:
        log_y = self.log_check.isChecked()
        for metric, title, _xlabel in _HIST_METRICS:
            plot = self.hist_plots[metric]
            item = plot.getPlotItem()
            plot.clear()
            item.setTitle(title)
            item.setLogMode(y=log_y)

            edges = self._hist_data.get(f"{metric}_edges")
            if edges is None:
                continue
            edges = np.asarray(edges, dtype=np.float64)
            for ch in sorted(self._counts):
                counts = self._hist_data.get(f"{metric}_counts_ch{ch}")
                if counts is None:
                    continue
                counts = np.asarray(counts, dtype=np.float64)
                if log_y:
                    counts = np.where(counts > 0, counts, np.nan)
                color = _channel_color(ch)
                brush = QtGui.QColor(color)
                brush.setAlpha(110)
                plot.plot(
                    edges, counts,
                    stepMode="center",
                    fillLevel=0 if not log_y else None,
                    brush=brush,
                    pen=pg.mkPen(color, width=1.2),
                    name=f"Ch {ch} (n={int(np.nansum(counts))})",
                    connect="finite",
                )

    # ── Theme ─────────────────────────────────────────────────────

    def apply_theme(self, dark_mode: bool) -> None:
        self.dark_mode = dark_mode
        bg_color, pen_color = theme_colors(dark_mode)
        plots = [self.pulse_plot_i, self.pulse_plot_q] \
            + list(self.hist_plots.values())
        for plot in plots:
            plot.setBackground(bg_color)
            item = plot.getPlotItem()
            for side in ("left", "bottom", "top", "right"):
                ax = item.getAxis(side)
                ax.setPen(pen_color)
                ax.setTextPen(pen_color)
        self._render_histograms()
