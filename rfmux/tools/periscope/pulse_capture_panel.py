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
from ...algorithms.measurement.pulse_capture_session import PulseCaptureSession
from ...algorithms.measurement.pulse_hdf5 import PulseHDF5Reader

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
        self._counts: Dict[int, int] = {}
        self._last_stats: dict = {}
        self._hist_data: dict = {}
        self._capture_start_wall: Optional[float] = None

        self._setup_ui()
        self._set_run_state(False)
        self.apply_theme(dark_mode)
        self.module_spin.setValue(module)

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
        model = self.mode_combo.model()
        for row in (1, 2):  # fast/both arrive with PFB support (Phase D)
            item = model.item(row)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEnabled)
            item.setToolTip("PFB streaming support not wired into the "
                            "panel yet — use the trigger_capture macro")
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

        self.pulse_plot = pg.PlotWidget()
        self.pulse_plot.getPlotItem().addLegend(offset=(-10, 10))
        self.pulse_plot.getPlotItem().setLabel("bottom", "time (ms)")
        self.pulse_plot.getPlotItem().setLabel("left", "amplitude (counts)")
        self.pulse_plot.getPlotItem().showGrid(x=True, y=True, alpha=0.3)
        v.addWidget(self.pulse_plot, stretch=1)
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
        if getattr(runtime, "_pulse_tap", None) is not None:
            QtWidgets.QMessageBox.warning(
                self, "Pulse Capture",
                "Another pulse capture is already tapping the stream. "
                "Stop it first.")
            return

        module = int(self.module_spin.value())
        path = self._resolve_hdf5_path(module)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QtWidgets.QMessageBox.warning(
                self, "Pulse Capture", f"Cannot create {path.parent}: {e}")
            return

        session = PulseCaptureSession(
            channels=channels,
            module=module,
            streamer_mode=self.mode_combo.currentText(),
            threshold_sigma=float(self.threshold_spin.value()),
            end_sigma=float(self.end_spin.value()),
            enable_pileup=self.pileup_check.isChecked(),
            hdf5_path=path,
            df_calibrations=self.df_calibrations,
        )
        self.signals = PulseCaptureSignals()
        self.task = PulseCaptureTask(session, self.signals)
        conn = QtCore.Qt.ConnectionType.QueuedConnection
        self.signals.noise_estimated.connect(self._on_noise_estimated, conn)
        self.signals.pulse_detected.connect(self._on_pulse_detected, conn)
        self.signals.stats_updated.connect(self._on_stats, conn)
        self.signals.histograms_updated.connect(self._on_histograms, conn)
        self.signals.error.connect(self._on_error, conn)
        self.signals.finished.connect(self._on_task_finished, conn)

        self._reset_results(channels)
        self._registered_export = False
        self.path_label.setText(f"HDF5: {path}")
        self._capture_start_wall = time.time()

        self.task.start()
        runtime.register_pulse_tap(self.task.enqueue)
        self._tap_registered = True
        self._set_run_state(True)
        self._set_status("● Estimating noise…", "#FFCC33")

    def _on_stop(self) -> None:
        self._unregister_tap()
        if self.task is not None:
            self.task.request_stop()
        self.btn_start.setEnabled(False)  # until finished arrives

    def _on_task_finished(self) -> None:
        if self.task is not None:
            self.task.wait(2000)
            self.task = None
        self.signals = None
        self._set_run_state(False)
        n = sum(self._counts.values())
        self._set_status(f"● Stopped — {n} pulses captured", "#9A9A9A")

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
        self._reset_results(channels, started=started)

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
        total = sum(self._counts.values())
        self._set_status(
            f"● Review Mode — {total} pulses — {Path(path).name}",
            "#3366CC")

        self.follow_check.setChecked(False)
        if self._pulse_order:
            self._show_pulse(*self._pulse_order[-1])

    def _set_run_state(self, running: bool) -> None:
        self.btn_start.setText("■ Stop" if running else "▶ Start")
        self.btn_start.setEnabled(True)
        self.btn_start.setStyleSheet(
            "background-color: #4CC38A; color: #10241B; font-weight: bold;"
            if running else "")
        self.btn_reestimate.setEnabled(running)
        for w in (self.mode_combo, self.channels_edit, self.module_spin,
                  self.threshold_spin, self.end_spin, self.pileup_check,
                  self.btn_browse):
            w.setEnabled(not running)
        if not running:
            self._set_status("● Idle", "#9A9A9A")

    def _reset_results(self, channels: List[int],
                       started: Optional[str] = None) -> None:
        self._pulse_order.clear()
        self._pulse_summaries.clear()
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
        self.pulse_plot.clear()
        self.pulse_info.setText("No pulse selected")
        self._render_histograms()

    # ── Signal handlers (GUI thread) ──────────────────────────────

    def _on_noise_estimated(self, noise_stats: dict) -> None:
        self.noise_stats = noise_stats
        parts = []
        for c in sorted(noise_stats):
            ns = noise_stats[c]
            parts.append(f"Ch{c} I={ns.mean_I:.1f}±{ns.std_I:.2f}, "
                         f"Q={ns.mean_Q:.1f}±{ns.std_Q:.2f}")
        self.noise_label.setText("Noise:  " + "   |   ".join(parts))
        self._refresh_status_line()

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

        if self.follow_check.isChecked():
            self._show_pulse(channel, pulse_idx)

    def _on_stats(self, stats: dict) -> None:
        self._last_stats = stats
        self._refresh_status_line()

    def _refresh_status_line(self) -> None:
        s = self._last_stats
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
        self._hist_data = data
        self._render_histograms()

    def _on_error(self, message: str) -> None:
        self.noise_label.setToolTip(message)
        self._set_status(f"● Error: {message}", "#E5484D")

    def _set_status(self, text: str, color: str) -> None:
        self.status_label.setText(
            f'<span style="color:{color}">●</span> <b>{text[2:]}</b>'
            if text.startswith("● ") else text)

    # ── Pulse view ────────────────────────────────────────────────

    def _on_tree_double_click(self, item, column) -> None:
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if data and data[0] == "pulse":
            self.follow_check.setChecked(False)
            self.viewer_tabs.setCurrentIndex(0)
            self._show_pulse(data[1], data[2])

    def _navigate(self, step: int) -> None:
        if not self._pulse_order:
            return
        self.follow_check.setChecked(False)
        if self._current_view in self._pulse_summaries:
            i = self._pulse_order.index(self._current_view) + step
        else:
            i = len(self._pulse_order) - 1
        i = max(0, min(len(self._pulse_order) - 1, i))
        self._show_pulse(*self._pulse_order[i])

    def _get_waveform(self, channel: int, pulse_idx: int) -> Optional[dict]:
        if self.task is not None:
            wf = self.task.get_pulse(channel, pulse_idx)
            if wf is not None:
                return wf
        if self.reader is not None:
            return self.reader.get_pulse(channel, pulse_idx)
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

        self.pulse_plot.clear()
        if wf is None:
            self.pulse_plot.setTitle("waveform no longer cached")
            return
        self.pulse_plot.setTitle(None)

        t = np.asarray(wf["Time"], dtype=np.float64)
        finite = np.isfinite(t)
        t0 = t[finite][0] if np.any(finite) else 0.0
        t_ms = (t - t0) * 1e3
        amp_I = np.asarray(wf["Amp_I"], dtype=np.float64)
        amp_Q = np.asarray(wf["Amp_Q"], dtype=np.float64)

        self.pulse_plot.plot(t_ms, amp_I, pen=pg.mkPen(
            IQ_COLORS["I"], width=LINE_WIDTH), name="I")
        self.pulse_plot.plot(t_ms, amp_Q, pen=pg.mkPen(
            IQ_COLORS["Q"], width=LINE_WIDTH), name="Q")

        ns = self.noise_stats.get(channel)
        if ns is not None:
            thr = float(self.threshold_spin.value())
            end = float(self.end_spin.value())
            for mean, color in ((ns.mean_I, IQ_COLORS["I"]),
                                (ns.mean_Q, IQ_COLORS["Q"])):
                self.pulse_plot.addLine(y=mean, pen=pg.mkPen(
                    color, width=0.8, style=QtCore.Qt.PenStyle.DotLine))
            for sign in (+1, -1):
                self.pulse_plot.addLine(
                    y=ns.mean_I + sign * thr * ns.std_I,
                    pen=pg.mkPen("#888888", width=0.8,
                                 style=QtCore.Qt.PenStyle.DashLine))
                self.pulse_plot.addLine(
                    y=ns.mean_I + sign * end * ns.std_I,
                    pen=pg.mkPen("#666666", width=0.8,
                                 style=QtCore.Qt.PenStyle.DotLine))

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
        plots = [self.pulse_plot] + list(self.hist_plots.values())
        for plot in plots:
            plot.setBackground(bg_color)
            item = plot.getPlotItem()
            for side in ("left", "bottom", "top", "right"):
                ax = item.getAxis(side)
                ax.setPen(pen_color)
                ax.setTextPen(pen_color)
        self._render_histograms()
