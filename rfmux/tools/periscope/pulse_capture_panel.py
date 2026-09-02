"""
Dockable live pulse-capture panel.

The layout: a toolbar of capture controls, a status strip (state /
counts / rate + per-channel noise), a pulse tree on the left (newest
first, pileup flagged), and three tabs on the right — a single-pulse
I/Q waveform with threshold bands, a 2×2 grid of running histograms
(SNR, peak amplitude, duration, derived τ), and the trigger-aligned
template stack.

All capture logic lives in
:class:`~rfmux.pulse_capture.capture_session.PulseCaptureSession`;
this panel only configures a session, bridges it through
:class:`~rfmux.tools.periscope.pulse_capture_task.PulseCaptureTask`,
and draws.
"""

from __future__ import annotations

import asyncio
import csv
import datetime
import time
from pathlib import Path
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from .utils import (
    ClickableViewBox,
    ScreenshotMixin,
    IQ_COLORS,
    TABLEAU10_COLORS,
    LINE_WIDTH,
    find_parent_with_attr,
    theme_colors,
)
from .pulse_capture_task import PulseCaptureSignals, PulseCaptureTask
from .pulse_capture_settings_dialog import PulseCaptureSettingsDialog
from ...algorithms.measurement.channel_selection import (
    parse_channel_spec,
)
from ...pulse_capture.capture_session import (
    PulseCaptureConfig,
    PulseCaptureSession,
)
from ...pulse_capture.hdf5 import PulseHDF5Reader
from ...core.transferfunctions import (
    apply_iq_conversion,
    PFB_SAMPLING_FREQ,
    decimation_to_sampling,
)
from ...pulse_capture.detection import ChannelNoiseStats
from ...pulse_capture.analysis import (
    display_transform,
    storage_transform,
)

# The fast stream keeps the quadrature hue family but sits apart from
# the slow one on the same plot: blue/purple, orange/red.
FAST_IQ_COLORS = {"I": "#7A3FBF", "Q": "#CC3333"}

# Channel plot colors: ch1 = I-blue, ch2 = Q-orange (HUD convention),
# further channels from Tableau10.
#: Above this many channels the status strip summarises instead of
#: naming every channel.  200 channels of "Ch12 I=1.0±2.34, Q=…" is
#: several thousand characters on one line, and a QLabel that wide
#: drags the whole dock out with it.
MAX_LISTED_CHANNELS = 4


def _summarize(stats: dict, per_channel, summary) -> str:
    """List every channel while there are few, else summarise.

    ``per_channel(channel, ns)`` renders one entry; ``summary(stats)``
    renders the condensed form.
    """
    if not stats:
        return "—"
    if len(stats) <= MAX_LISTED_CHANNELS:
        return "   |   ".join(per_channel(c, ns)
                              for c, ns in sorted(stats.items()))
    return summary(stats)


def _spread(values) -> str:
    """min–max (median) for a run of numbers."""
    ordered = sorted(float(v) for v in values)
    return (f"{ordered[0]:.2f}–{ordered[-1]:.2f} "
            f"(median {ordered[len(ordered) // 2]:.2f})")


def _noise_line(stats: dict, names=("I", "Q"), unit: str = "") -> str:
    """Noise strip text for one stream, in the units on the axes.

    The names and the unit come from the caller because the strip sits
    above plots whose basis the user can change: printing I and Q with
    no unit described whatever happened to be on disk.  ``g`` rather
    than a fixed decimal count, because hertz and volts are many orders
    of magnitude apart and neither should overflow the strip.
    """
    a, b = names
    tail = f" {unit}" if unit else ""
    return _summarize(
        stats,
        lambda c, ns: (f"Ch{c} {a}={ns.mean_I:.4g}±{ns.std_I:.3g}, "
                       f"{b}={ns.mean_Q:.4g}±{ns.std_Q:.3g}{tail}"),
        lambda st: (f"{len(st)} ch — σ{a} {_spread(n.std_I for n in st.values())}"
                    f", σ{b} {_spread(n.std_Q for n in st.values())}{tail}"))


def _noise_line_sigma(stats: dict) -> str:
    """Compact per-stream variant used in both-mode."""
    return _summarize(
        stats,
        lambda c, ns: f"Ch{c} σI={ns.std_I:.2f}",
        lambda st: f"{len(st)} ch — σI {_spread(n.std_I for n in st.values())}")


def _legend_name(channel: int, count, n_channels: int):
    """Curve name, or None to keep it out of the legend.

    pyqtgraph draws one legend row per named curve, so a 128-channel
    capture buries the plot under its own key.  Past
    MAX_LISTED_CHANNELS the curves stay unnamed and the channel count
    goes in the title instead.
    """
    if n_channels > MAX_LISTED_CHANNELS:
        return None
    return f"Ch {channel} (n={count})"


def _noise_detail(stats: dict, names=("I", "Q"), unit: str = "") -> str:
    """Full per-channel listing, for the tooltip."""
    a, b = names
    tail = f" {unit}" if unit else ""
    return "\n".join(
        f"Ch{c}  {a}={ns.mean_I:.4g}±{ns.std_I:.3g}   "
        f"{b}={ns.mean_Q:.4g}±{ns.std_Q:.3g}{tail}"
        for c, ns in sorted(stats.items()))


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


#: The three views the panel offers, and the (basis, units) each means.
#: Kept in step with the main window's counts / real / df selector.
UNITS_COUNTS = "counts"
UNITS_VOLTS = "volts"
UNITS_DF = "df (Hz)"
_VIEW_STATES = {
    UNITS_COUNTS: ("iq", "counts"),
    UNITS_VOLTS: ("iq", "V"),
    UNITS_DF: ("df", "Hz"),
}


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
        self._hist_data_by_stream: Dict[str, dict] = {}
        self._template_data: dict = {}
        self._template_data_by_stream: Dict[str, dict] = {}

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

        #: Set once the user picks a view; the frequency-view default
        #: below is applied only until then.
        self._view_chosen = False
        self._setup_ui()
        self._apply_default_view()
        self._setup_shortcuts()
        self._set_run_state(False)
        self.apply_theme(dark_mode)
        self.module_spin.setValue(module)

        # Prefill with what Periscope is displaying: a useful starting
        # point, not a limit.  The slow packet carries every streamed
        # channel, so the tap can capture any of them (see
        # register_pulse_tap) -- and "all" selects by bias, not display.
        streamed = getattr(periscope, "all_chs", None)
        if streamed:
            self.channels_edit.setText(
                ",".join(str(c) for c in sorted(set(streamed))))

    def _setup_shortcuts(self) -> None:
        """Keyboard navigation, active while the panel has focus."""
        def add(seq, slot):
            action = QtGui.QAction(self)
            action.setShortcut(QtGui.QKeySequence(seq))
            action.setShortcutContext(
                QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            action.triggered.connect(slot)
            self.addAction(action)

        add(QtCore.Qt.Key.Key_Left, lambda: self._navigate(-1))
        add(QtCore.Qt.Key.Key_Right, lambda: self._navigate(+1))
        add(QtCore.Qt.Key.Key_Space, self._cycle_tab)
        add(QtCore.Qt.Key.Key_Home, lambda: self._navigate_end(first=True))
        add(QtCore.Qt.Key.Key_End, lambda: self._navigate_end(first=False))
        add("Ctrl+E", self._on_export)

    def _cycle_tab(self) -> None:
        count = self.viewer_tabs.count()
        self.viewer_tabs.setCurrentIndex(
            (self.viewer_tabs.currentIndex() + 1) % count)

    def _navigate_end(self, first: bool) -> None:
        if not self._pulse_order:
            return
        self.follow_check.setChecked(False)
        key = self._pulse_order[0 if first else -1]
        if self._both_mode:
            self._show_pair(*key)
        else:
            self._show_pulse(*key)

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
            "stream)\nfast: ~2.44 MHz PFB stream (max 4 channels; "
            "configures the fast streamer automatically)\n"
            "both: concurrent slow+fast with live pulse matching — the "
            "tree lists matched pairs")
        h.addWidget(self.mode_combo)

        h.addWidget(QtWidgets.QLabel("Channels:"))
        self.channels_edit = QtWidgets.QLineEdit("1,2")
        self.channels_edit.setFixedWidth(90)
        self.channels_edit.setToolTip(
            "1-indexed channels: \"1,2\", ranges \"2-19\", or a mix\n"
            "\"1,5-8,20\".  \"all\" takes every channel on this module\n"
            "that has a bias set.")
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

        h.addWidget(QtWidgets.QLabel("Units:"))
        self.units_combo = QtWidgets.QComboBox()
        # The same three the main window offers, and for the same reason:
        # basis and scale are not independent in any useful way.  Hertz is
        # a property of the frequency axis, so "df" means rotated *and*
        # scaled, and there is no such thing as volts on that axis.
        self.units_combo.addItems([UNITS_COUNTS, UNITS_VOLTS, UNITS_DF])
        self.units_combo.setCurrentText(UNITS_VOLTS)
        self.units_combo.setToolTip(
            "Units for waveforms, histograms and templates.\n\n"
            "counts: raw ADC.  volts: the readout scale.  df: rotated "
            "into frequency and dissipation, in hertz — needs a df "
            "calibration for the channel.")
        self.units_combo.currentTextChanged.connect(self._on_user_view_changed)
        h.addWidget(self.units_combo)

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

        self.btn_export = QtWidgets.QPushButton("Export…")
        self.btn_export.setToolTip(
            "Export the current pulse/pair, histograms or template to CSV")
        self.btn_export.clicked.connect(self._on_export)
        h.addWidget(self.btn_export)

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
        # Belt and braces with the summarising above: a label's size
        # hint otherwise becomes the dock's minimum width, so one long
        # line is enough to make the panel unusable.
        for label in (self.status_label, self.noise_label):
            label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                QtWidgets.QSizePolicy.Policy.Preferred)
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
        self.viewer_tabs.addTab(self._build_template_view(), "Template")
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
        self.pulse_plot_i = pg.PlotWidget(viewBox=ClickableViewBox())
        self.pulse_plot_q = pg.PlotWidget(viewBox=ClickableViewBox())
        for plot, ylabel in ((self.pulse_plot_i, "I (V)"),
                             (self.pulse_plot_q, "Q (V)")):
            item = plot.getPlotItem()
            item.setLabel("left", ylabel)
            item.showGrid(x=True, y=True, alpha=0.3)
            item.addLegend(offset=(-10, 10))
        self._set_pulse_x_axis("time", "s")
        self.pulse_plot_q.setXLink(self.pulse_plot_i)
        v.addWidget(self.pulse_plot_i, stretch=1)
        v.addWidget(self.pulse_plot_q, stretch=1)
        return w

    def _decision_noise(self, wf, ns):
        """The bands a record was decided against: (trigger, end).

        The live stats object is re-centred after every pulse, so a band
        drawn from it is not the band the engine tested.  Records carry
        their bands; older records fall back to the stats, and the end
        band to the trigger band.
        """
        if not isinstance(wf, dict) or "trigger_baseline_I" not in wf:
            return ns, ns
        try:
            v = {k: float(wf[k]) for k in (
                "trigger_baseline_I", "trigger_baseline_Q",
                "trigger_sigma_I", "trigger_sigma_Q")}
        except (KeyError, TypeError, ValueError):
            return ns, ns
        if not all(np.isfinite(x) for x in v.values()) \
                or v["trigger_sigma_I"] <= 0 or v["trigger_sigma_Q"] <= 0:
            return ns, ns
        base = ns if ns is not None else ChannelNoiseStats()
        trig = replace(base, mean_I=v["trigger_baseline_I"],
                       mean_Q=v["trigger_baseline_Q"],
                       std_I=v["trigger_sigma_I"], std_Q=v["trigger_sigma_Q"])
        end = trig
        try:
            e_I, e_Q = float(wf["end_baseline_I"]), float(wf["end_baseline_Q"])
            if np.isfinite(e_I) and np.isfinite(e_Q):
                end = replace(trig, mean_I=e_I, mean_Q=e_Q)
        except (KeyError, TypeError, ValueError):
            pass
        return trig, end

    def _annotate_noise_bands(self, plot, quad, ns, x0, x1, color,
                              prefix="", thr=None, end=None,
                              end_ns=None) -> None:
        """Draw the baseline and the ±σ bands on *plot*.

        Drawn as two-point curves rather than addLine(): an InfiniteLine
        is not a PlotDataItem and never shows up in the legend, so the
        bands were unlabelled wherever they did appear.
        """
        if ns is None:
            return
        mean = getattr(ns, f"mean_{quad}", None)
        std = getattr(ns, f"std_{quad}", None)
        if mean is None or std is None or not np.isfinite(std) or std <= 0:
            return
        # The record's own levels when it carries them (the spins can
        # have moved since the capture, and say nothing about a file).
        thr = float(self.threshold_spin.value()) if thr is None else float(thr)
        end = float(self.end_spin.value()) if end is None else float(end)
        end_mean = mean
        if end_ns is not None:
            end_mean = getattr(end_ns, f"mean_{quad}", mean)
            if end_mean is None or not np.isfinite(end_mean):
                end_mean = mean
        x = np.array([x0, x1], dtype=float)
        plot.plot(x, np.full(2, mean),
                  pen=pg.mkPen(color, width=1.0,
                               style=QtCore.Qt.PenStyle.DotLine),
                  name=f"{prefix}baseline")
        for centre, level, style, tag in (
                (mean, thr, QtCore.Qt.PenStyle.DashLine,
                 f"±{thr:g}σ trigger"),
                (end_mean, end, QtCore.Qt.PenStyle.DotLine,
                 f"±{end:g}σ end")):
            # One item per +/- pair, joined by a NaN gap, so one legend
            # entry hides and shows both lines.
            band = centre + level * std
            plot.plot(
                np.array([x0, x1, np.nan, x0, x1], dtype=float),
                np.array([band, band, np.nan,
                          2 * centre - band, 2 * centre - band],
                         dtype=float),
                connect="finite",
                pen=pg.mkPen(color, width=1.0, style=style),
                name=f"{prefix}{tag}")

    @staticmethod
    def _decision_text(wf) -> str:
        """One line describing how the capture was bounded."""
        if not isinstance(wf, dict) or "trigger_index" not in wf:
            return ""
        trig = wf.get("trigger_index")
        end = wf.get("end_index")
        below = wf.get("below_threshold_index")
        got = wf.get("end_confirm_samples")
        want = wf.get("end_confirm_target")
        parts = [f"trigger @ sample {trig}"]
        if wf.get("trigger_quad"):
            # Which quadrature started the above-threshold run: the
            # mark sits on both plots, and on the other one it can
            # precede any visible crossing.
            parts[0] += f" (on {wf['trigger_quad']})"
        if below is not None:
            parts.append(f"below threshold @ {below}")
        if end is not None:
            parts.append(f"end confirmed @ {end}")
        if got is not None and want is not None:
            parts.append(f"bucket {got}/{want}")
        return "\n" + "   ".join(parts)

    def _saves_full_tail(self) -> bool:
        """Whether captures keep samples to the end-of-pulse confirmation.

        Live, from the configured capture; in review, restored from the
        file's capture parameters when it was opened.
        """
        return bool(self.capture_config.save_to_end_confirmed)

    def _annotate_decisions(self, plot, wf, t0, quad,
                            prefix="") -> None:
        """Mark where the engine triggered and where the end condition
        confirmed the end.

        Vertical lines rather than markers on the trace: the end point
        normally sits PAST the last saved sample, because the window is
        trimmed back to where the signal returned to baseline rather
        than where the bucket finished confirming it.  A line still
        shows that, a data marker could not.
        """
        t = np.asarray(wf.get("Time"), dtype=np.float64)
        if not len(t):
            return

        def _t_at(idx_key, time_key):
            # Prefer the absolute time: when these marks were looked up
            # from a triggered record but drawn over a different
            # (union) window, its indices do not apply here.
            tv = wf.get(time_key)
            if tv is not None and np.isfinite(tv):
                return float(tv) - t0
            idx = wf.get(idx_key)
            if idx is None or not (0 <= int(idx) < len(t)):
                return None
            return float(t[int(idx)]) - t0

        marks = [
            ("trigger_index", "trigger_time", "#33CC66", "trigger"),
            ("below_threshold_index", "below_threshold_time", "#CCAA33",
             "below threshold"),
        ]
        # The confirmation instant is drawn only when the tail was kept
        # to it; otherwise it lies past the end of the saved data.
        if self._saves_full_tail():
            marks.append(("end_index", "end_time", "#CC3366",
                          "end confirmed"))
        for idx_key, time_key, color, label in marks:
            label = f"{prefix}{label}"
            x = _t_at(idx_key, time_key)
            if x is None:
                continue
            plot.addItem(pg.InfiniteLine(
                pos=x, angle=90,
                pen=pg.mkPen(color, width=1.4,
                             style=QtCore.Qt.PenStyle.DashLine),
                # Label only on the top plot; the two are x-linked, so
                # repeating it underneath is noise.
                label=label if quad == "I" else None,
                labelOpts={"position": 0.95, "color": color,
                           "fill": (0, 0, 0, 120), "movable": False}))

    def _set_pulse_x_axis(self, label: str, units: str | None = None) -> None:
        """Label BOTH stacked plots the same way.

        Setting ``units`` turns on pyqtgraph's SI-prefix autoscaling for
        that axis, so labelling only one of an x-linked pair leaves them
        showing the same data against differently scaled ticks (µs on
        one, raw seconds on the other).
        """
        for plot in (self.pulse_plot_i, self.pulse_plot_q):
            plot.getPlotItem().setLabel("bottom", label, units=units)

    def _build_histograms_view(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)

        controls = QtWidgets.QHBoxLayout()
        self.log_check = QtWidgets.QCheckBox("Log y")
        self.log_check.toggled.connect(self._render_histograms)
        controls.addWidget(self.log_check)
        self.hist_stream_combo = QtWidgets.QComboBox()
        self.hist_stream_combo.addItems(["slow", "fast"])
        self.hist_stream_combo.setToolTip(
            "Which stream's histograms to display (both mode)")
        self.hist_stream_combo.currentTextChanged.connect(
            self._on_hist_stream_changed)
        self.hist_stream_combo.setVisible(False)  # both-mode only
        controls.addWidget(self.hist_stream_combo)

        controls.addStretch(1)
        v.addLayout(controls)

        grid_holder = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(grid_holder)
        grid.setSpacing(8)
        self.hist_plots: Dict[str, pg.PlotWidget] = {}
        for i, (metric, title, xlabel) in enumerate(_HIST_METRICS):
            plot = pg.PlotWidget(viewBox=ClickableViewBox())
            item = plot.getPlotItem()
            item.setLabel("bottom", xlabel)
            item.setLabel("left", "count")
            item.showGrid(x=True, y=True, alpha=0.3)
            item.addLegend(offset=(-10, 10))
            self.hist_plots[metric] = plot
            grid.addWidget(plot, i // 2, i % 2)
        v.addWidget(grid_holder, stretch=1)
        return w

    def _build_template_view(self) -> QtWidgets.QWidget:
        """Trigger-aligned pulse stack: mean template ± residual RMS."""
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)

        controls = QtWidgets.QHBoxLayout()
        self.template_info = QtWidgets.QLabel("No pulses stacked yet")
        controls.addWidget(self.template_info)
        controls.addStretch(1)
        self.template_residual_check = QtWidgets.QCheckBox(
            "Show residual RMS")
        self.template_residual_check.setChecked(True)
        self.template_residual_check.setToolTip(
            "Shaded band: per-bin RMS spread of the stacked pulses "
            "about the mean template")
        self.template_residual_check.toggled.connect(
            self._render_templates)
        controls.addWidget(self.template_residual_check)
        v.addLayout(controls)

        self.template_plot_i = pg.PlotWidget(viewBox=ClickableViewBox())
        self.template_plot_q = pg.PlotWidget(viewBox=ClickableViewBox())
        for plot, ylabel in ((self.template_plot_i, "I (counts)"),
                             (self.template_plot_q, "Q (counts)")):
            item = plot.getPlotItem()
            item.setLabel("left", ylabel)
            item.showGrid(x=True, y=True, alpha=0.3)
            item.addLegend(offset=(-10, 10))
        for plot in (self.template_plot_i, self.template_plot_q):
            plot.getPlotItem().setLabel("bottom", "time from trigger",
                                        units="s")
        self.template_plot_q.setXLink(self.template_plot_i)
        v.addWidget(self.template_plot_i, stretch=1)
        v.addWidget(self.template_plot_q, stretch=1)
        return w

    def _on_templates(self, data: dict) -> None:
        if "stream" in data and "data" in data:
            self._template_data_by_stream[data["stream"]] = data["data"]
            if data["stream"] != self.hist_stream_combo.currentText():
                return
            self._template_data = data["data"]
        else:
            self._template_data = data
        self._render_templates()

    def _render_templates(self) -> None:
        for plot in (self.template_plot_i, self.template_plot_q):
            plot.clear()
        data = self._template_data
        if not data:
            self.template_info.setText("No pulses stacked yet")
            return

        show_band = self.template_residual_check.isChecked()
        totals = []
        # Track the data extent so the view fits the STACKED region:
        # bins outside it are NaN/empty and would otherwise stretch the
        # axes over the full pre/post grid.
        x_lo = x_hi = None
        y_lim = {"I": [None, None], "Q": [None, None]}
        for ch in sorted(self._counts):
            t = data.get(f"time_s_ch{ch}")
            counts = data.get(f"counts_ch{ch}")
            if t is None or counts is None:
                continue
            counts = np.asarray(counts)
            n_pulses = int(np.max(counts)) if len(counts) else 0
            totals.append((ch, n_pulses))
            t_arr = np.asarray(t, dtype=np.float64)
            populated = np.nonzero(counts > 0)[0]
            if len(populated):
                lo, hi = t_arr[populated[0]], t_arr[populated[-1]]
                x_lo = lo if x_lo is None else min(x_lo, lo)
                x_hi = hi if x_hi is None else max(x_hi, hi)
            color = _channel_color(ch)
            scale = self._amp_scale(ch)
            # Convert the pair together.  Averaging and the conversion are
            # both linear, so a rotated template equals the template of
            # rotated pulses -- but only if the two axes are transformed
            # as a pair.  Scaling them one at a time left the templates
            # unrotated under df/dissipation labels.
            view = self._view_coeffs(ch)
            means = {q: data.get(f"template_{q}_ch{ch}") for q in ("I", "Q")}
            rotated = (view is not None and means["I"] is not None
                       and means["Q"] is not None)
            if rotated:
                a, b = apply_iq_conversion(
                    np.asarray(means["I"], dtype=np.float64),
                    np.asarray(means["Q"], dtype=np.float64), view[0])
                means = {"I": a, "Q": b}
            for quad, plot in (("I", self.template_plot_i),
                               ("Q", self.template_plot_q)):
                mean = means.get(quad)
                if mean is None:
                    continue
                mean = np.asarray(mean, dtype=np.float64)
                if scale is not None and not rotated:
                    mean = mean * scale
                plot.plot(np.asarray(t, float), mean,
                          pen=pg.mkPen(color, width=2.2),
                          connect="finite",
                          name=_legend_name(ch, n_pulses,
                                            len(self._counts)))
                finite = np.isfinite(mean)
                if np.any(finite):
                    lo, hi = float(np.min(mean[finite])), \
                        float(np.max(mean[finite]))
                    cur = y_lim[quad]
                    cur[0] = lo if cur[0] is None else min(cur[0], lo)
                    cur[1] = hi if cur[1] is None else max(cur[1], hi)

                # The residual is a spread, not a signed pair: the
                # rotation mixes the quadratures, so only its length
                # carries over -- which is what `scale` already is.
                resid = data.get(f"residual_{quad}_ch{ch}")
                if show_band and resid is not None:
                    resid = np.asarray(resid, dtype=np.float64)
                    if scale is not None:
                        resid = resid * scale
                    both = np.isfinite(mean) & np.isfinite(resid)
                    if np.any(both):
                        cur = y_lim[quad]
                        cur[0] = min(cur[0], float(np.min(
                            (mean - resid)[both])))
                        cur[1] = max(cur[1], float(np.max(
                            (mean + resid)[both])))
                    band = QtGui.QColor(color)
                    band.setAlpha(60)
                    upper = pg.PlotDataItem(np.asarray(t, float),
                                            mean + resid,
                                            connect="finite")
                    lower = pg.PlotDataItem(np.asarray(t, float),
                                            mean - resid,
                                            connect="finite")
                    fill = pg.FillBetweenItem(upper, lower, brush=band)
                    plot.addItem(fill)

        first, second = self._axis_names(self._label_channel())
        for plot, quad, name in ((self.template_plot_i, "I", first),
                                 (self.template_plot_q, "Q", second)):
            item = plot.getPlotItem()
            # Same names the pulse view uses.  These were set here
            # separately and kept the old "I (Δf)" wording, overwriting
            # what the view had already chosen.
            item.setLabel("left", name)
            # Fit to the stacked data, not the whole pre/post grid
            if x_lo is not None and x_hi > x_lo:
                item.vb.setXRange(x_lo, x_hi, padding=0.02)
            lo, hi = y_lim[quad]
            if lo is not None and hi is not None:
                if hi > lo:
                    item.vb.setYRange(lo, hi, padding=0.08)
                else:  # flat trace — keep it visible, not a zero-height box
                    item.vb.setYRange(lo - 1.0, hi + 1.0, padding=0.0)
        if not totals:
            self.template_info.setText("No pulses stacked yet")
        elif len(totals) <= MAX_LISTED_CHANNELS:
            self.template_info.setText(
                "Trigger-aligned stack — "
                + ", ".join(f"Ch{c}: {n}" for c, n in totals))
        else:
            stacked = sum(n for _, n in totals)
            busiest = max(totals, key=lambda cn: (cn[1], -cn[0]))
            self.template_info.setText(
                f"Trigger-aligned stack — {len(totals)} ch, "
                f"{stacked:,} pulses stacked, "
                f"deepest Ch{busiest[0]}: {busiest[1]}")
        self.template_info.setToolTip(
            "\n".join(f"Ch{c}  {n}" for c, n in totals))

    # ── Capture lifecycle ─────────────────────────────────────────

    def _parse_channels(self, *, runtime=None,
                        quiet: bool = False) -> Optional[List[int]]:
        """Channels named by the toolbar field, or None if unusable.

        The syntax itself lives in the algorithms layer so notebooks
        accept the same strings; this only decides what to do with the
        wildcard and how to report a bad spec.
        """
        try:
            channels = parse_channel_spec(self.channels_edit.text())
        except ValueError as e:
            if not quiet:
                QtWidgets.QMessageBox.warning(self, "Pulse Capture", str(e))
            return None
        if channels is None:  # the "all" wildcard
            return self._resolve_biased_channels(runtime, quiet=quiet)
        return channels

    def _resolve_biased_channels(self, runtime=None, *, quiet: bool = False
                                 ) -> Optional[List[int]]:
        """Channels on the current module that are carrying a tone.

        Read off the board rather than remembered, so "all" tracks
        whatever is biased at the moment Start is pressed.
        """
        def warn(text: str) -> None:
            if not quiet:
                QtWidgets.QMessageBox.warning(self, "Pulse Capture", text)

        if runtime is None:
            runtime = self._resolve_runtime()
        crs = getattr(runtime, "crs", None)
        if crs is None:
            warn('"all" reads bias amplitudes off the board, which needs '
                 "a CRS connection. Enter explicit channel numbers "
                 "instead.")
            return None

        module = int(self.module_spin.value())
        # A channel the slow packet cannot carry cannot be captured,
        # however it is biased — so don't offer them.
        max_ch = 128 if getattr(runtime, "is_short_packet", False) else 1024

        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
        try:
            channels = asyncio.run(
                crs.get_biased_channels(module, max_channels=max_ch))
        except Exception as e:
            warn(f"Could not read bias amplitudes from module {module}:\n{e}")
            return None
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        if not channels:
            warn(f"No channel on module {module} has a bias set, so "
                 '"all" selects nothing. Bias some channels first.')
            return None
        return channels

    def _resolve_runtime(self):
        """The object owning register_pulse_tap (Periscope main window)."""
        if self.periscope is not None:
            return self.periscope
        return find_parent_with_attr(self, "register_pulse_tap")

    def _resolve_hdf5_path(self, module: int) -> Path:
        """Where this capture writes, unless one was set explicitly."""
        if self.hdf5_path is not None:
            return self.hdf5_path
        stamp = datetime.datetime.now().strftime("%H%M%S")
        return self._export_dir() / f"pulse_module{module}_{stamp}.h5"

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
            return PFB_SAMPLING_FREQ
        runtime = self._resolve_runtime()
        dec = getattr(runtime, "actual_dec_stage", None)
        return decimation_to_sampling(dec if dec is not None else 6)

    # ── Export ────────────────────────────────────────────────────

    def _export_dir(self) -> Path:
        sm = self.session_manager
        if sm is not None and getattr(sm, "is_active", False) \
                and sm.session_path is not None:
            return Path(sm.session_path)
        if self._browse_dir:
            return Path(self._browse_dir)
        return Path.home()

    def _on_export(self) -> None:
        """Export what the active tab is showing, as CSV."""
        tab = self.viewer_tabs.currentIndex()
        stamp = datetime.datetime.now().strftime("%H%M%S")
        try:
            if tab == 0:
                rows, name = self._export_waveform_rows(stamp)
            elif tab == 1:
                rows, name = self._export_histogram_rows(stamp)
            else:
                rows, name = self._export_template_rows(stamp)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Pulse Capture", f"Nothing to export:\n{e}")
            return
        if not rows:
            QtWidgets.QMessageBox.information(
                self, "Pulse Capture", "Nothing to export from this tab.")
            return

        path = self._export_dir() / name
        try:
            with open(path, "w", newline="") as fh:
                csv.writer(fh).writerows(rows)
        except OSError as e:
            QtWidgets.QMessageBox.warning(
                self, "Pulse Capture", f"Could not write {path}:\n{e}")
            return
        if self.session_manager is not None and \
                getattr(self.session_manager, "is_active", False):
            try:
                self.session_manager.register_external_file(
                    str(path), "pulse", "export")
            except Exception:
                pass
        print(f"[PulseCapture] Exported {path}")
        self._set_status(f"● Exported {path.name}", "#3366CC")

    def _export_waveform_rows(self, stamp):
        """Current pulse (or pair) waveform(s), one column set per stream."""
        if self._both_mode and self._current_pair is not None:
            ch, idx = self._current_pair
            pair = self._get_pair(ch, idx) or {}
            meta = self._pair_meta.get((ch, idx), {})
            slow = pair.get("slow_tod") or (
                self._get_waveform(ch, meta.get("slow_idx"), "slow")
                if meta.get("slow_idx") else None)
            fast = pair.get("fast_tod") or (
                self._get_waveform(ch, meta.get("fast_idx"), "fast")
                if meta.get("fast_idx") else None)
            rows = [["stream", "time_s", "Amp_I", "Amp_Q"]]
            for label, wf in (("slow", slow), ("fast", fast)):
                if not wf:
                    continue
                for t, i, q in zip(wf["Time"], wf["Amp_I"], wf["Amp_Q"]):
                    rows.append([label, float(t), float(i), float(q)])
            return rows, f"pulse_pair_ch{ch}_{idx:04d}_{stamp}.csv"

        if self._current_view is None:
            return [], ""
        ch, idx = self._current_view
        wf = self._get_waveform(ch, idx)
        if not wf:
            return [], ""
        rows = [["time_s", "Amp_I", "Amp_Q"]]
        for t, i, q in zip(wf["Time"], wf["Amp_I"], wf["Amp_Q"]):
            rows.append([float(t), float(i), float(q)])
        return rows, f"pulse_ch{ch}_{idx:06d}_{stamp}.csv"

    def _export_histogram_rows(self, stamp):
        data = self._hist_data
        if not data:
            return [], ""
        rows = [["metric", "channel", "bin_left", "bin_right", "count"]]
        for metric, _title, _x in _HIST_METRICS:
            edges = data.get(f"{metric}_edges")
            if edges is None:
                continue
            edges = np.asarray(edges, dtype=np.float64)
            for ch in sorted(self._counts):
                counts = data.get(f"{metric}_counts_ch{ch}")
                if counts is None:
                    continue
                for k, n in enumerate(np.asarray(counts)):
                    rows.append([metric, ch, float(edges[k]),
                                 float(edges[k + 1]), int(n)])
        return rows, f"pulse_histograms_{stamp}.csv"

    def _export_template_rows(self, stamp):
        data = self._template_data
        if not data:
            return [], ""
        rows = [["channel", "time_s", "template_I", "template_Q",
                 "residual_I", "residual_Q", "n_stacked"]]
        for ch in sorted(self._counts):
            t = data.get(f"time_s_ch{ch}")
            if t is None:
                continue
            ti = data.get(f"template_I_ch{ch}")
            tq = data.get(f"template_Q_ch{ch}")
            ri = data.get(f"residual_I_ch{ch}")
            rq = data.get(f"residual_Q_ch{ch}")
            counts = data.get(f"counts_ch{ch}")
            for k in range(len(t)):
                rows.append([
                    ch, float(t[k]),
                    float(ti[k]) if ti is not None else "",
                    float(tq[k]) if tq is not None else "",
                    float(ri[k]) if ri is not None else "",
                    float(rq[k]) if rq is not None else "",
                    int(counts[k]) if counts is not None else "",
                ])
        return rows, f"pulse_template_{stamp}.csv"

    def _on_capture_settings(self) -> None:
        self._sync_config_from_toolbar()
        mode = self.mode_combo.currentText()
        channels = self._parse_channels(quiet=True) or [1]
        dlg = PulseCaptureSettingsDialog(
            self,
            config=self.capture_config,
            sample_rate=self._current_sample_rate(mode),
            mode=mode,
            n_channels=len(channels),
            df_available=any(self._channel_cal(ch) is not None
                             for ch in channels),
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
        # Runtime first: "all" needs the CRS handle and the packet width
        # it carries to know which channels are even reachable.
        runtime = self._resolve_runtime()
        if runtime is None:
            QtWidgets.QMessageBox.warning(
                self, "Pulse Capture",
                "No running Periscope stream found to tap.")
            return
        channels = self._parse_channels(runtime=runtime)
        if channels is None:
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
                    f"{mode} mode needs a CRS connection to check the "
                    "PFB streamer.")
                return
        if mode in ("slow", "both"):
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
            from ...pulse_capture.capture_session import (
                DualPulseCaptureSession,
            )
            capture_session = DualPulseCaptureSession(
                channels=channels,
                module=module,
                slow_rate=self._current_sample_rate("slow"),
                fast_rate=PFB_SAMPLING_FREQ,
                config=self.capture_config,
                hdf5_path=path,
                df_calibrations=self._flat_df_calibrations(),
            )
        else:
            capture_session = PulseCaptureSession(
                channels=channels,
                module=module,
                streamer_mode=mode,
                hdf5_path=path,
                df_calibrations=self._flat_df_calibrations(),
                sample_rate=fs,
                **self.capture_config.session_kwargs(fs),
            )
        self.signals = PulseCaptureSignals()
        self.task = PulseCaptureTask(capture_session, self.signals, mode=mode,
                                     crs=crs, host=host, module=module)
        conn = QtCore.Qt.ConnectionType.QueuedConnection
        self.signals.noise_estimated.connect(self._on_noise_estimated, conn)
        self.signals.noise_progress.connect(self._on_noise_progress, conn)
        self.signals.pulse_detected.connect(self._on_pulse_detected, conn)
        self.signals.pair_matched.connect(self._on_pair_matched, conn)
        self.signals.stats_updated.connect(self._on_stats, conn)
        self.signals.histograms_updated.connect(self._on_histograms, conn)
        self.signals.templates_updated.connect(self._on_templates, conn)
        self.signals.waveform_ready.connect(self._on_waveform_ready, conn)
        self.signals.error.connect(self._on_error, conn)
        self.signals.failed.connect(self._on_failed, conn)
        self.signals.finished.connect(self._on_task_finished, conn)

        # A fresh capture replaces any file we were browsing
        if self.reader is not None:
            self.reader.close()
            self.reader = None

        self._both_mode = (mode == "both")
        self._reset_results(channels)
        self._apply_default_view()
        self._registered_export = False
        self.path_label.setText(f"HDF5: {path}")
        self._capture_start_wall = time.time()

        self.task.start()
        if mode in ("slow", "both"):
            runtime.register_pulse_tap(self.task.enqueue_packet,
                                       channels,
                                       on_frame_end=self.task.flush_tap)
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
        self._unregister_tap()   # however the worker ended
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
        # The marks drawn over a record depend on the policy that made
        # it, so an opened file sets it the way a live capture does.
        self.capture_config.save_to_end_confirmed = bool(
            meta.get("save_to_end_confirmed", True))
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
        self._apply_default_view()

        if self.reader.dual:
            self._load_dual_review(channels, path)
            return

        self.noise_stats = {c: self.reader.noise_stats(c) for c in channels}
        self._refresh_noise_label()

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
                    "truncated": bool(m.get("truncated", False)),
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
                self._add_pulse_row(c, idx, summary)

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
            parts.append(f"{stream}: " + _noise_line_sigma(stats))
        self.noise_label.setText("Noise:  " + "   —   ".join(parts))
        self.noise_label.setToolTip("\n\n".join(
            f"{stream}\n" + _noise_detail(st)
            for stream, st in sorted(self._noise_by_stream.items())))

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

        for stream in ("slow", "fast"):
            self._hist_data_by_stream[stream] = \
                self.reader.get_histograms(stream)
        self._hist_data = self._hist_data_by_stream.get(
            self.hist_stream_combo.currentText(), {})
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
        self._hist_data_by_stream = {}
        self._template_data = {}
        self._template_data_by_stream = {}
        self.hist_stream_combo.setVisible(self._both_mode)
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
        self._render_templates()

    # ── Signal handlers (GUI thread) ──────────────────────────────

    def _on_noise_progress(self, progress: dict) -> None:
        collected = progress.get("collected", {})
        target = progress.get("target", 0)
        prefix = (f"[{progress['stream']}] "
                  if progress.get("stream") else "")
        if len(collected) <= MAX_LISTED_CHANNELS:
            body = " | ".join(f"Ch{c} {collected[c]}/{target}"
                              for c in sorted(collected))
        else:
            done = sum(1 for n in collected.values() if n >= target)
            body = (f"{len(collected)} ch, {done} done, slowest "
                    f"{min(collected.values())}/{target}")
        self._set_status(f"● Estimating noise — {prefix}{body}", "#FFCC33")
        self.status_label.setToolTip(
            "\n".join(f"Ch{c}  {collected[c]}/{target}"
                       for c in sorted(collected)))

    def _baseline_summary(self) -> str:
        """The span the rolling baseline median covers."""
        sess = getattr(self.task, "session", None)
        if sess is None:
            return ""
        inners = ([("slow", sess.slow), ("fast", sess.fast)]
                  if hasattr(sess, "slow") and hasattr(sess, "fast")
                  else [("", sess)])
        parts = []
        for name, s in inners:
            n = getattr(s, "baseline_window", 0)
            if not n:
                continue
            rate = getattr(s, "sample_rate", None)
            span = f"{n / rate:,.3g} s" if rate else f"{n:,} samples"
            parts.append(f"{name + ' ' if name else ''}{span}")
        return ("   —   baseline median over:  " + ",  ".join(parts)
                if parts else "")

    def _on_noise_estimated(self, noise_stats: dict) -> None:
        if "stream" in noise_stats and "stats" in noise_stats:
            stream = noise_stats["stream"]
            stats = noise_stats["stats"]
            self._noise_by_stream[stream] = stats
            if stream == "slow":
                self.noise_stats = stats
            parts = [f"{s}: " + _noise_line_sigma(st)
                     for s, st in sorted(self._noise_by_stream.items())]
            self.noise_label.setText("Noise:  " + "   —   ".join(parts)
                                     + self._baseline_summary())
            self.noise_label.setToolTip("\n\n".join(
                f"{s}\n" + _noise_detail(st)
                for s, st in sorted(self._noise_by_stream.items())))
            print(f"[PulseCapture] Noise estimated ({stream})"
                  f"{self._baseline_summary()}")
            self._refresh_status_line()
            # Show what the estimator saw for this stream (most recent
            # estimate wins until the first pulse replaces it)
            if self.follow_check.isChecked() or self._current_view is None:
                self._show_noise_segment(stream)
            return

        self.noise_stats = noise_stats
        self._refresh_noise_label(self._baseline_summary())
        print("[PulseCapture] Noise estimated: " + _noise_line(noise_stats)
              + self._baseline_summary())
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

    def _add_pulse_row(self, channel: int, pulse_idx: int,
                       summary: dict) -> None:
        """Insert one pulse into the tree, newest first.

        Shared by the live handler and the review loader, so a row looks
        the same whether the pulse just arrived or came out of a file.
        """
        parent = self._channel_items.get(channel)
        if parent is None:
            return
        pileup = bool(summary.get("pileup", False))
        truncated = bool(summary.get("truncated", False))
        label = ("\u2298" if truncated else "\u26a0" if pileup else "\u25c6") \
            + f" #{pulse_idx:06d}"
        item = QtWidgets.QTreeWidgetItem(
            [label, str(summary.get("n_samples", "")),
             f"{summary.get('snr', 0):.1f}\u03c3"])
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole,
                     ("pulse", channel, pulse_idx))
        if truncated or pileup:
            colour = QtGui.QColor(
                ("#3a2222" if self.dark_mode else "#ffd9d2") if truncated
                else ("#3a3320" if self.dark_mode else "#fff3c2"))
            for col in range(3):
                item.setBackground(col, colour)
        parent.insertChild(0, item)
        parent.setText(0, f"\u25a4 Channel {channel} "
                          f"({self._counts[channel]})")

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

        self._add_pulse_row(channel, pulse_idx, summary)

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

        # A pair ALWAYS carries both streams' data when the rings allow:
        # "matched" vs "slow-trig"/"fast-trig" is trigger provenance,
        # not data presence.  Tint only when the complement window
        # could NOT be recovered (ring had slid past it).
        matched = pair["slow_idx"] is not None \
            and pair["fast_idx"] is not None
        complement_missing = False
        if matched:
            label = (f"◆ Pair #{pair_idx:04d}  "
                     f"s#{pair['slow_idx']}/f#{pair['fast_idx']}")
            dt = pair.get("time_offset") or 0.0
            detail = f"Δt(trig)={dt*1e6:+.0f}µs"
        else:
            side = "slow" if pair["slow_idx"] is not None else "fast"
            other = "fast" if side == "slow" else "slow"
            idx = pair["slow_idx"] or pair["fast_idx"]
            label = f"◆ Pair #{pair_idx:04d}  {side}-trig #{idx}"
            if pair.get(f"has_{other}_tod"):
                detail = f"+{other} data"
            else:
                detail = f"{other} n/a"
                complement_missing = True
        summ = pair.get("slow_summary") or pair.get("fast_summary") or {}
        parent = self._channel_items.get(ch)
        if parent is not None:
            item = QtWidgets.QTreeWidgetItem(
                [label, detail, f"{summ.get('snr', 0):.1f}σ"])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole,
                         ("pair", ch, pair_idx))
            if complement_missing:
                for col in range(3):
                    item.setBackground(col, QtGui.QColor(
                        "#33251c" if self.dark_mode else "#ffe8d9"))
            parent.insertChild(0, item)
            parent.setText(0, f"▤ Channel {ch} ({self._counts[ch]} pairs)")

        if self.follow_check.isChecked() \
                and not self._follow_timer.isActive():
            self._follow_timer.start()

    #: The fast engine shares the event loop with the slow one; when it
    #: cannot keep real time its ring drifts behind the slow ring, and
    #: past the ring overlap cross-stream windows go unavailable.  Amber
    #: at half the overlap, red once the lag exceeds it.
    _LAG_WARN_FRACTION = 0.5

    def _stream_lag_signal(self, stats: dict):
        """(status colour, (short, tooltip)) for the current stream lag.

        Returns the healthy green and ``None`` when the lag is fine, so
        the caller only appends a note when there is one.
        """
        lag = stats.get("stream_lag_s")
        overlap = stats.get("ring_overlap_s")
        if lag is None or overlap is None or overlap <= 0:
            return "#4CC38A", None
        lag = abs(lag)
        if lag < self._LAG_WARN_FRACTION * overlap:
            return "#4CC38A", None
        who = "fast" if (stats.get("stream_lag_s") or 0) >= 0 else "slow"
        short = f"⚠ {who} stream {lag:.1f}s behind"
        tip = (
            f"The {who} stream is {lag:.2f}s behind the other, and the "
            f"rings overlap only {overlap:.2f}s. When the lag exceeds the "
            "overlap the two streams no longer share a time span: "
            "cross-stream windows read 'unavailable' and pulses stop "
            "matching.\n\nThe engine is not keeping real time on the "
            "shared event loop — raise the threshold σ (fewer crossings "
            "to walk) or capture fewer channels.")
        colour = "#E5484D" if lag >= overlap else "#E5A23B"
        return colour, (short, tip)

    def _refresh_status_line(self) -> None:
        s = self._last_stats
        if self._both_mode and "pairs_matched" in s:
            slow_n = s.get("slow", {}).get("total_pulses", 0)
            fast_n = s.get("fast", {}).get("total_pulses", 0)
            text = (f"● Capturing — slow {slow_n} | fast {fast_n} pulses — "
                    f"{s['pairs_matched']} matched / "
                    f"{s['pairs_unmatched']} single-trigger pairs")
            dropped = {name: s.get(name, {}).get("dropped_invalid_ts", 0)
                       for name in ("slow", "fast")}
            if any(dropped.values()):
                text += " — " + " / ".join(
                    f"{name} {n} dropped (no timestamp)"
                    for name, n in dropped.items() if n)
            colour, tip = self._stream_lag_signal(s)
            if tip:                       # append the drift note when it bites
                text += f"  —  {tip[0]}"
            self._set_status(text, colour)
            tooltip = tip[1] if tip else ""
            skew, n = s.get("stream_skew_s"), s.get("stream_skew_n", 0)
            shift = s.get("slow_time_offset_s") or 0.0
            if shift:
                tooltip += (("\n\n" if tooltip else "")
                            + f"Slow timestamps shifted by {shift*1e3:+.3f} "
                              "ms to take out the decimated stream's CIC "
                              "group delay (the PFB clock is the "
                              "reference). The skew below is the residual.")
            if skew is not None and n:
                tooltip += (("\n\n" if tooltip else "")
                            + f"Stream clock skew: slow − fast trigger time "
                              f"= {skew*1e3:+.2f} ms (median of {n} matched "
                              "pairs). This is how far the two streams' "
                              "timestamps disagree on one event; in the pair "
                              "view it shows as one stream's marks sitting "
                              "off the other stream's trace by this much.")
            self.status_label.setToolTip(tooltip)
            return
        total = s.get("total_pulses", 0)
        rate = s.get("rate_per_min", 0.0)
        per_ch = s.get("per_channel", {})
        if len(per_ch) <= MAX_LISTED_CHANNELS:
            ch_str = " | ".join(f"Ch{c}: {n}"
                                for c, n in sorted(per_ch.items()))
        else:
            firing = {c: n for c, n in per_ch.items() if n}
            if firing:
                busiest = max(firing.items(), key=lambda kv: (kv[1], -kv[0]))
                ch_str = (f"{len(firing)}/{len(per_ch)} ch firing, "
                          f"busiest Ch{busiest[0]}: {busiest[1]}")
            else:
                ch_str = f"{len(per_ch)} ch, none firing yet"
        self.status_label.setToolTip(
            "\n".join(f"Ch{c}  {n}" for c, n in sorted(per_ch.items())))
        elapsed = int(s.get("elapsed_s", 0))
        hh, rem = divmod(elapsed, 3600)
        mm, ss = divmod(rem, 60)
        dropped = s.get("dropped_invalid_ts", 0)
        drop_str = f" — {dropped} dropped (no timestamp)" if dropped else ""
        self._set_status(
            f"● Capturing — {total} pulses ({rate:.1f}/min) — {ch_str} — "
            f"{hh:02d}:{mm:02d}:{ss:02d}{drop_str}", "#4CC38A")

    def _flat_df_calibrations(self) -> Dict[int, Any]:
        """Calibrations for the selected module, as {channel: calibration}.

        Periscope keeps one mapping per module, because its plots are
        per-module; PulseCaptureSession and PulseHDF5Writer take the flat
        per-channel mapping.  Flattening here keeps the storage layer
        from having to know about modules.  Accepts either shape, so a
        headless caller's flat mapping passes through unchanged.
        """
        cal = self.df_calibrations
        if not isinstance(cal, dict) or not cal:
            return {}
        per_module = cal.get(int(self.module_spin.value()))
        if isinstance(per_module, dict):
            return dict(per_module)
        # Already flat: a headless caller's {channel: calibration}.
        return {ch: v for ch, v in cal.items() if not isinstance(v, dict)}

    def _channel_cal(self, channel: int):
        """This channel's df calibration, from the session or the file.

        In review mode there is no Periscope session holding one, so the
        file is the only place it survives a capture.
        """
        cal = self._flat_df_calibrations().get(channel)
        if cal is None and self.reader is not None:
            try:
                cal = self.reader.df_calibration(channel)
            except Exception:
                cal = None
        return cal

    def _stored_state(self, channel: int) -> Tuple[str, str]:
        """(basis, units) the samples for *channel* are held in.

        A finished capture says so in the file; a live one follows the
        same storage_transform the session applies.  The units decide
        the basis: a channel is in hertz only once it has been rotated,
        so a df capture of an uncalibrated channel is quadratures in
        volts.
        """
        units = None
        if self.reader is not None:
            try:
                units = self.reader.stored_units(channel)
            except Exception:
                pass
        if units is None:
            _factor, units = storage_transform(
                self._flat_df_calibrations().get(channel),
                self.capture_config.trigger_basis)
        return ("df" if units == "Hz" else "iq"), units

    def _view_state(self) -> Tuple[str, str]:
        """(basis, units) for the selected view.

        Three choices, not a basis crossed with a scale: rotating without
        converting to hertz, or asking for hertz on the quadratures, are
        combinations with no meaning.
        """
        return _VIEW_STATES.get(self.units_combo.currentText(),
                                ("iq", "V"))

    def _view_coeffs(self, channel: int):
        """(factor, units) taking stored samples to the current view.

        None when the view cannot be produced -- hertz or a rotation with
        no calibration.  Callers show what is stored rather than putting
        an unscaled number under a label it does not have.
        """
        sb, su = self._stored_state(channel)
        vb, vu = self._view_state()
        return display_transform(self._channel_cal(channel), sb, su, vb, vu)

    def _amp_scale(self, channel: int) -> Optional[float]:
        """Factor for amplitude-like scalars, or None if unavailable.

        Amplitudes are magnitudes, so the rotation contributes only its
        length -- which is 1 -- and this reduces to the units ratio.
        """
        t = self._view_coeffs(channel)
        if t is None:
            return None
        return abs(t[0])

    def _units_label(self, channel: int) -> str:
        """Axis label for *channel* under the current view."""
        t = self._view_coeffs(channel)
        if t is not None:
            return t[1]
        return self._stored_state(channel)[1]

    def _refresh_pulse_plot(self) -> None:
        """Redraw for a changed view, labels included.

        The labels are set even with nothing selected, so an empty panel
        does not advertise units the next capture will not be in.
        """
        ch = self._label_channel()
        first, second = self._axis_names(ch)
        for plot, name in ((self.pulse_plot_i, first),
                           (self.pulse_plot_q, second),
                           (self.template_plot_i, first),
                           (self.template_plot_q, second)):
            plot.getPlotItem().setLabel("left", name)
        cur = self._current_view
        if cur is not None:
            try:
                self._show_pulse(*cur)
            except Exception:
                pass

    def _label_channel(self) -> int:
        """A channel to take axis labels from when none is selected."""
        cur = self._current_view
        if cur is not None:
            return cur[0]
        try:
            chans = self._parse_channels()
            if chans:
                return chans[0]
        except Exception:
            pass
        return 1

    def _axis_names(self, channel: int) -> Tuple[str, str]:
        """Axis labels for what is actually plotted.

        Both halves come from the same place.  Taking the basis from the
        request and the units from the fallback gave "I (Hz)" -- the
        quadrature names over samples stored as frequency and
        dissipation, which are not the quadratures at all.
        """
        view = self._view_coeffs(channel)
        if view is not None:
            basis, _ = self._view_state()
            unit = view[1]
        else:
            # Cannot produce what was asked for, so the stored samples
            # are drawn as they are: name those.
            basis, unit = self._stored_state(channel)
        if basis == "df":
            return f"df ({unit})", f"dissipation ({unit})"
        return f"I ({unit})", f"Q ({unit})"

    def _view_noise(self, channel: int, ns):
        """Noise statistics taken into the current view.

        The two halves do not transform alike.  The baseline is a
        signed position in the plane, so it rotates with the samples --
        scaling it alone left the baseline and the threshold lines
        drawn from volts while the trace above them was in hertz.  The
        spreads are magnitudes, and the rotation mixes the quadratures,
        so they carry over by its length only.
        """
        if ns is None:
            return ns
        view = self._view_coeffs(channel)
        if view is None or view[0] == 1.0:
            return ns
        factor = view[0]
        k = abs(factor)
        mean_I, mean_Q = apply_iq_conversion(ns.mean_I, ns.mean_Q, factor)
        return replace(
            ns, mean_I=mean_I, std_I=ns.std_I * k,
            mean_Q=mean_Q, std_Q=ns.std_Q * k,
            jump_std_I=ns.jump_std_I * k, jump_std_Q=ns.jump_std_Q * k)

    def _refresh_noise_label(self, extra: str = "") -> None:
        """Noise strip in the units on the axes, not the ones on disk.

        Set once when the statistics arrived, it kept naming I and Q in
        whatever the capture stored while the plots below it had been
        switched to another basis.  It is cheap, so it just follows the
        view.
        """
        stats = {c: self._view_noise(c, ns)
                 for c, ns in (self.noise_stats or {}).items()}
        first, second = self._axis_names(self._label_channel())
        names = tuple(n.split(" (")[0] for n in (first, second))
        unit = first.rpartition(" (")[2].rstrip(")")
        self.noise_label.setText(
            "Noise:  " + _noise_line(stats, names, unit) + extra)
        self.noise_label.setToolTip(_noise_detail(stats, names, unit))

    def _units_are_hz(self) -> bool:
        return self.units_combo.currentText() == UNITS_DF

    def _on_user_view_changed(self, _text: str = "") -> None:
        self._view_chosen = True
        self._on_view_changed()

    def _apply_default_view(self) -> None:
        """Default to the frequency view once a calibration makes it
        possible.

        Applied wherever a calibration can become known -- construction,
        capture start, opening a file -- and only until the user picks a
        view, which then stands.  Volts is the only default that can
        always be drawn, so it stays the fallback.
        """
        if self._view_chosen or self._units_are_hz():
            return
        if not self._any_channel_calibrated():
            return
        self.units_combo.blockSignals(True)
        self.units_combo.setCurrentText(UNITS_DF)
        self.units_combo.blockSignals(False)
        self._on_view_changed()

    def _on_view_changed(self, _text: str = "") -> None:
        if (self.units_combo.currentText() == UNITS_DF
                and not self._any_channel_calibrated()):
            # Say so rather than quietly drawing volts under a hertz
            # label, which is what falling back used to look like.
            QtWidgets.QMessageBox.warning(
                self, "df Calibration Not Available",
                "No df calibration for these channels, so frequency and "
                "dissipation cannot be separated.\n\n"
                "It comes from bias_kids — run a multisweep and click "
                "'Bias KIDs'. In mock mode, enabling auto_bias_kids "
                "measures one for each channel it tunes.\n\n"
                "A capture already holding one carries it in the file, "
                "so opening that capture is enough.")
            self.units_combo.blockSignals(True)
            self.units_combo.setCurrentText(UNITS_VOLTS)
            self.units_combo.blockSignals(False)
        self._render_histograms()
        self._render_templates()
        self._refresh_pulse_plot()
        self._refresh_noise_label()

    def _any_channel_calibrated(self) -> bool:
        """Whether any displayed channel has a df calibration.

        Any rather than all: a mixed capture should still offer the view
        for the channels that can show it.
        """
        channels = list(self._counts) or [self._label_channel()]
        return any(self._channel_cal(ch) is not None for ch in channels)

    def _on_hist_stream_changed(self, stream: str) -> None:
        if stream in self._hist_data_by_stream:
            self._hist_data = self._hist_data_by_stream[stream]
            self._render_histograms()
        if stream in self._template_data_by_stream:
            self._template_data = self._template_data_by_stream[stream]
            self._render_templates()

    def _on_histograms(self, data: dict) -> None:
        if "stream" in data and "data" in data:
            # both-mode: keep per-stream stores, render the selected one
            self._hist_data_by_stream[data["stream"]] = data["data"]
            if data["stream"] == self.hist_stream_combo.currentText():
                self._hist_data = data["data"]
                self._render_histograms()
        else:
            self._hist_data = data
            self._render_histograms()

    def _on_error(self, message: str) -> None:
        self.noise_label.setToolTip(message)
        self._set_status(f"● Error: {message}", "#E5484D")
        print(f"[PulseCapture] ERROR: {message}")

    def _on_failed(self, message: str) -> None:
        """The capture cannot run or has died: say so in a dialog, not
        only the status line.  finished follows and cleans up."""
        self._on_error(message)
        QtWidgets.QMessageBox.warning(self, "Pulse Capture", message)

    def _show_noise_segment(self, stream: str | None = None) -> None:
        """Plot the noise-training segment with the fitted baselines and
        trigger/end bands — visual confirmation of what the estimator saw.

        In "both" mode the dual session holds one training record per
        stream (``session.slow`` / ``session.fast``); *stream* selects
        which one, and the most recent estimate wins the view."""
        if self.task is None:
            return
        session = self.task.session
        stats = self.noise_stats
        tag = ""
        if stream is not None:
            session = getattr(session, stream, None)
            stats = self._noise_by_stream.get(stream, {})
            tag = f" ({stream})"
        if session is None:
            return
        noise_data = getattr(session, "noise_data", {})
        channel = next((c for c in sorted(stats)
                        if c in noise_data and len(noise_data[c])), None)
        if channel is None:
            return
        arr = noise_data[channel]
        ns = stats[channel]
        thr = float(self.threshold_spin.value())
        end = float(self.end_spin.value())

        self._current_view = None
        self.pulse_info.setText(
            f"Noise training segment{tag} — Channel {channel} "
            f"({len(arr)} samples)\n"
            f"I = {ns.mean_I:.1f} ± {ns.std_I:.2f}   "
            f"Q = {ns.mean_Q:.1f} ± {ns.std_Q:.2f}\n"
            f"bands: ±{thr:g}σ trigger (dashed), ±{end:g}σ end (dotted)")

        x = np.arange(len(arr))
        self.pulse_plot_i.clear()
        self.pulse_plot_q.clear()
        self.pulse_plot_i.setTitle(f"Noise training{tag} — Channel {channel}")
        self.pulse_plot_q.setTitle(None)
        self._set_pulse_x_axis("sample")
        x1 = float(len(arr) - 1) if len(arr) else 1.0
        for quad, plot, data in (("I", self.pulse_plot_i, arr.real),
                                 ("Q", self.pulse_plot_q, arr.imag)):
            plot.plot(x, data, pen=pg.mkPen(IQ_COLORS[quad], width=1.0),
                      name=f"{quad} (training)")
            self._annotate_noise_bands(plot, quad, ns, 0.0, x1, "#888888")

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
        if summary.get("truncated"):
            pile += "  [truncated at max pulse — hard stop]"
        tau_ms = summary.get("tau_ms", float("nan"))
        tau_str = f"{tau_ms:.2f} ms" if np.isfinite(tau_ms) else "n/a"
        self.pulse_info.setText(
            f"Pulse #{pulse_idx:06d} — Channel {channel}   {pile}\n"
            f"{summary.get('n_samples', 0)} samples   "
            f"{summary.get('duration_ms', 0):.2f} ms   "
            f"peak {summary.get('peak_amp', 0):.4g} "f"{self._units_label(channel)} "
            f"({summary.get('snr', 0):.1f}σ)\n"
            f"derived τ = {tau_str}"
            + self._decision_text(wf))

        for plot in (self.pulse_plot_i, self.pulse_plot_q):
            plot.clear()
            plot.setTitle(None)
        self._set_pulse_x_axis("time", "s")
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

        # Stored samples into the requested view.  Both axes rotate
        # together, so the noise bands drawn below have to come along.
        ns, ns_end = self._decision_noise(wf, self.noise_stats.get(channel))
        view = self._view_coeffs(channel)
        if view is not None:
            factor, _units = view
            amp_I, amp_Q = apply_iq_conversion(amp_I, amp_Q, factor)
            ns = self._view_noise(channel, ns)
            ns_end = self._view_noise(channel, ns_end)
        first, second = self._axis_names(channel)
        for plot, name in ((self.pulse_plot_i, first),
                           (self.pulse_plot_q, second)):
            plot.getPlotItem().setLabel("left", name)
        x0 = float(t_rel[0]) if len(t_rel) else 0.0
        x1 = float(t_rel[-1]) if len(t_rel) else 1.0
        # Series names follow the axes: "I (pulse)" over a plot labelled
        # df (Hz) names the wrong thing.  The colours stay keyed to the
        # underlying quadrature so a trace does not change colour when
        # the view does.
        series = [n.split(" (")[0] for n in (first, second)]
        for quad, label, plot, data in (
                ("I", series[0], self.pulse_plot_i, amp_I),
                ("Q", series[1], self.pulse_plot_q, amp_Q)):
            plot.plot(t_rel, data,
                      pen=pg.mkPen(IQ_COLORS[quad], width=LINE_WIDTH),
                      name=f"{label} (pulse)")
            self._annotate_noise_bands(
                plot, quad, ns, x0, x1, "#888888",
                thr=wf.get("threshold_sigma"), end=wf.get("end_sigma"),
                end_ns=ns_end)
            self._annotate_decisions(plot, wf, t0, quad)

    def _show_pair(self, channel: int, pair_idx: int) -> None:
        """Matched-pair overlay: dense fast trace under slow markers,
        per quadrature (HUD 'both' view)."""
        pair = self._get_pair(channel, pair_idx)
        meta = self._pair_meta.get((channel, pair_idx)) or pair
        if meta is None:
            return
        self._current_pair = (channel, pair_idx)
        self._current_view = None

        # Prefer the pair's UNION windows (both streams over the same
        # time span, extracted from the rings at pair time); fall back
        # to the per-stream triggered waveforms.
        loading = False
        slow_wf = fast_wf = None
        if pair is not None:
            slow_wf = pair.get("slow_tod")
            fast_wf = pair.get("fast_tod")
        if slow_wf is None and meta.get("slow_idx") is not None:
            slow_wf = self._get_waveform(channel, meta["slow_idx"],
                                         "slow")
            if slow_wf is None and self.task is not None:
                key = ("slow", channel, meta["slow_idx"])
                if self._pending_fetch != key:
                    self._pending_fetch = key
                    self.task.request_waveform(channel,
                                               meta["slow_idx"], "slow")
                loading = True
        if fast_wf is None and meta.get("fast_idx") is not None:
            fast_wf = self._get_waveform(channel, meta["fast_idx"],
                                         "fast")
            if fast_wf is None and self.task is not None:
                key = ("fast", channel, meta["fast_idx"])
                if self._pending_fetch != key:
                    self._pending_fetch = key
                    self.task.request_waveform(channel,
                                               meta["fast_idx"], "fast")
                loading = True

        matched = meta.get("slow_idx") is not None \
            and meta.get("fast_idx") is not None
        if matched:
            provenance = "both triggered"
        elif meta.get("slow_idx") is not None:
            provenance = "slow-triggered (fast shown from ring)" \
                if (fast_wf is not None) else \
                "slow-triggered (fast window unavailable)"
        else:
            provenance = "fast-triggered (slow shown from ring)" \
                if (slow_wf is not None) else \
                "fast-triggered (slow window unavailable)"
        dt = meta.get("time_offset")
        summ = meta.get("slow_summary") or meta.get("fast_summary") or {}
        tau_ms = summ.get("tau_ms", float("nan"))
        self.pulse_info.setText(
            f"Pair #{pair_idx:04d} — Channel {channel}   "
            f"[{provenance}]\n"
            f"slow #{meta.get('slow_idx')} / fast #{meta.get('fast_idx')}"
            + (f"   Δt(trigger) = {dt*1e6:+.0f} µs  (slow − fast: the "
               f"streams' clocks on this event)"
               if dt is not None and np.isfinite(dt) else "")
            + (f"\nSNR {summ.get('snr', 0):.1f}σ, "
               f"τ = {tau_ms:.2f} ms"
               if np.isfinite(tau_ms) else ""))

        for plot in (self.pulse_plot_i, self.pulse_plot_q):
            plot.clear()
            plot.setTitle(None)
        self._set_pulse_x_axis("time", "s")
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

        # Span of the drawn data, for the band lines below.
        x0 = x1 = None
        for wf in (fast_wf, slow_wf):
            if wf is None:
                continue
            tt = np.asarray(wf["Time"], float) - t0
            tt = tt[np.isfinite(tt)]
            if len(tt):
                x0 = tt[0] if x0 is None else min(x0, tt[0])
                x1 = tt[-1] if x1 is None else max(x1, tt[-1])
        if x0 is None:
            x0, x1 = 0.0, 1.0

        for quad, plot in (("I", self.pulse_plot_i),
                           ("Q", self.pulse_plot_q)):
            if fast_wf is not None:
                t_rel = np.asarray(fast_wf["Time"], float) - t0
                plot.plot(t_rel,
                          np.asarray(fast_wf[f"Amp_{quad}"], float),
                          pen=pg.mkPen(FAST_IQ_COLORS[quad], width=2.2),
                          name="fast (PFB, 2.44 MHz)")
            if slow_wf is not None:
                # Slow stream: dots only — sparse samples, no
                # interpolating line (Joshua's spec)
                t_rel = np.asarray(slow_wf["Time"], float) - t0
                data = np.asarray(slow_wf[f"Amp_{quad}"], float)
                plot.plot(t_rel, data, pen=None,
                          symbol="o", symbolSize=8,
                          symbolBrush=IQ_COLORS[quad],
                          symbolPen=pg.mkPen("w", width=0.8),
                          name="slow (readout)")

            # The two streams have different noise, so their bands are
            # at different levels — draw each in its own trace's colour
            # rather than one shared grey, or it is ambiguous which
            # threshold a given excursion had to clear.
            for stream, wf, tint in (
                    ("fast", fast_wf, FAST_IQ_COLORS[quad]),
                    ("slow", slow_wf, IQ_COLORS[quad])):
                if wf is None:
                    continue
                # The displayed window is usually the UNION ring
                # extract, which carries no decisions — those live on
                # the stream's own triggered record.  The marks are
                # absolute times, so they land correctly on the union
                # axis once looked up; the bands come from the same
                # record, so they are the bands those marks were
                # decided against.
                idx = meta.get(f"{stream}_idx")
                marks = wf if "trigger_index" in wf else None
                if marks is None and idx is not None:
                    marks = self._get_waveform(channel, idx, stream)
                stats = self._noise_by_stream.get(stream) or {}
                ns, ns_end = self._decision_noise(marks, stats.get(channel))
                levels = marks if isinstance(marks, dict) else {}
                self._annotate_noise_bands(
                    plot, quad, ns, x0, x1, tint, prefix=f"{stream} ",
                    thr=levels.get("threshold_sigma"),
                    end=levels.get("end_sigma"), end_ns=ns_end)
                if marks is not None:
                    self._annotate_decisions(plot, marks, t0, quad,
                                             prefix=f"{stream} ")

    # ── Histograms ────────────────────────────────────────────────

    def _render_histograms(self) -> None:
        log_y = self.log_check.isChecked()
        for metric, title, _xlabel in _HIST_METRICS:
            plot = self.hist_plots[metric]
            item = plot.getPlotItem()
            plot.clear()
            item.setTitle(
                title if len(self._counts) <= MAX_LISTED_CHANNELS
                else f"{title} — {len(self._counts)} ch")
            item.setLogMode(y=log_y)

            edges = self._hist_data.get(f"{metric}_edges")
            if edges is None:
                continue
            base_edges = np.asarray(edges, dtype=np.float64)
            # Amplitude is the only metric with ADC-count units; SNR,
            # duration and tau are dimensionless or times.
            scalable = (metric == "amplitude") and self._units_are_hz()
            any_scaled = False
            occupied_lo = occupied_hi = None
            for ch in sorted(self._counts):
                counts = self._hist_data.get(f"{metric}_counts_ch{ch}")
                if counts is None:
                    continue
                edges = base_edges
                if scalable:
                    scale = self._amp_scale(ch)
                    if scale is not None:
                        edges = base_edges * scale
                        any_scaled = True
                counts = np.asarray(counts, dtype=np.float64)
                nz = np.nonzero(counts > 0)[0]
                if len(nz):
                    lo, hi = edges[nz[0]], edges[nz[-1] + 1]
                    occupied_lo = lo if occupied_lo is None \
                        else min(occupied_lo, lo)
                    occupied_hi = hi if occupied_hi is None \
                        else max(occupied_hi, hi)
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
                    name=_legend_name(ch, int(np.nansum(counts)),
                                      len(self._counts)),
                    connect="finite",
                )
            if metric == "amplitude":
                # One unit, in the label.  pyqtgraph's units= appends a
                # second bracket, which read "amplitude (Δf) (Hz)".
                item.setLabel(
                    "bottom",
                    f"amplitude ({self._units_label(self._label_channel())})")
            # Fit x to the populated bins — auto-expanded ranges
            # otherwise leave the data huddled at one edge
            if occupied_lo is not None and occupied_hi > occupied_lo:
                plot.getPlotItem().vb.setXRange(
                    occupied_lo, occupied_hi, padding=0.08)

    # ── Theme ─────────────────────────────────────────────────────

    def apply_theme(self, dark_mode: bool) -> None:
        self.dark_mode = dark_mode
        bg_color, pen_color = theme_colors(dark_mode)
        plots = [self.pulse_plot_i, self.pulse_plot_q,
                 self.template_plot_i, self.template_plot_q] \
            + list(self.hist_plots.values())
        for plot in plots:
            plot.setBackground(bg_color)
            item = plot.getPlotItem()
            for side in ("left", "bottom", "top", "right"):
                ax = item.getAxis(side)
                ax.setPen(pen_color)
                ax.setTextPen(pen_color)
        self._render_histograms()
        self._render_templates()
