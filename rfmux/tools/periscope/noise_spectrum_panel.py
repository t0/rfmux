"""Dockable panel for displaying noise spectrum results for detectors.

This panel is the noise-only counterpart to DetectorDigestPanel.
It is opened only after the user runs "Get Noise Spectrum" from the multisweep panel.

The implementation is largely extracted from the former "Noise" tab that used to live
inside DetectorDigestPanel, but adapted to be a standalone QWidget.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets, QtGui

from rfmux.core.transferfunctions import (
    convert_roc_to_volts,
    exp_bin_noise_data,
    PFB_SAMPLING_FREQ,
)

from .utils import (
    ClickableViewBox,
    LINE_WIDTH,
    IQ_COLORS,
    ScreenshotMixin,
)


class NoiseSpectrumPanel(QtWidgets.QWidget, ScreenshotMixin):
    """Noise-only panel for a single detector, with navigation between detectors."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        detector_id: int = -1,
        resonance_frequency_ghz: float = 0.0,
        dark_mode: bool = False,
        all_detectors_data: dict | None = None,
        initial_detector_idx: int | None = None,
        spectrum_data: dict | None = None,
    ):
        super().__init__(parent)

        self.detector_id = detector_id
        self.resonance_frequency_ghz_title = resonance_frequency_ghz
        self.dark_mode = dark_mode

        # Navigation support (mirrors DetectorDigestPanel pattern)
        self.all_detectors_data = all_detectors_data or {}
        self.detector_indices = sorted(self.all_detectors_data.keys()) if self.all_detectors_data else []
        self.current_detector_index_in_list = 0

        if self.detector_indices and initial_detector_idx is not None:
            try:
                self.current_detector_index_in_list = self.detector_indices.index(initial_detector_idx)
            except ValueError:
                self.current_detector_index_in_list = 0

        self.mean_subtract_enabled = True
        self.exp_binning_enabled = True  # Default to exponential binning ON

        self.spectrum_data = spectrum_data
        self.reference = "absolute"
        self.show_fast_tod = False

        # Derived per-detector products (set by _load_detector_noise_products())
        self.single_psd_i = None
        self.single_psd_q = None
        self.tod_i = None
        self.tod_q = None
        self.tone_amp = None
        self.slow_freq = 0
        self.fast_freq = PFB_SAMPLING_FREQ / 2

        # Optional PFB products
        self.pfb_psd_i = None
        self.pfb_psd_q = None
        self.pfb_tod_i = None
        self.pfb_tod_q = None
        self.pfb_freq_iq = None
        self.pfb_freq_dsb = None
        self.pfb_dual_psd = None

        self._load_detector_noise_products()

        self._setup_ui()
        self._update_noise_plots()

        self.setWindowTitle(
            f"Noise Spectrum: Detector {self.detector_id} ({self.resonance_frequency_ghz_title * 1e3:.6f} MHz)"
        )
        self.setWindowFlags(QtCore.Qt.WindowType.Window)
        self.resize(1200, 800)

    # ---------------------------
    # Data loading / navigation
    # ---------------------------

    def _load_detector_noise_products(self) -> None:
        """Load per-detector arrays from spectrum_data for current detector."""
        self.show_fast_tod = False

        if not self.spectrum_data:
            return

        self.reference = self.spectrum_data.get("reference", "absolute")

        det_idx = self.detector_id - 1
        try:
            self.single_psd_i = self.spectrum_data["single_psd_i"][det_idx]
            self.single_psd_q = self.spectrum_data["single_psd_q"][det_idx]
            self.tod_i = self.spectrum_data["I"][det_idx]
            self.tod_q = self.spectrum_data["Q"][det_idx]
            self.tone_amp = self.spectrum_data["amplitudes_dbm"][det_idx]
            self.slow_freq = self.spectrum_data["slow_freq_hz"]

            if self.spectrum_data.get("pfb_enabled", False):
                self.show_fast_tod = True

            if self.show_fast_tod:
                self.pfb_psd_i = self.spectrum_data["pfb_psd_i"][det_idx]
                self.pfb_psd_q = self.spectrum_data["pfb_psd_q"][det_idx]
                self.pfb_tod_i = self.spectrum_data["pfb_I"][det_idx]
                self.pfb_tod_q = self.spectrum_data["pfb_Q"][det_idx]
                self.pfb_freq_iq = self.spectrum_data["pfb_freq_iq"][det_idx]
                self.pfb_freq_dsb = self.spectrum_data["pfb_freq_dsb"][det_idx]
                self.pfb_dual_psd = self.spectrum_data["pfb_dual_psd"][det_idx]
        except Exception as e:
            print(f"[NoiseSpectrumPanel] Error loading detector noise products: {e}")

    def _navigate_previous(self) -> None:
        if not self.detector_indices or len(self.detector_indices) <= 1:
            return
        self.current_detector_index_in_list = (self.current_detector_index_in_list - 1) % len(self.detector_indices)
        self._switch_to_detector(self.detector_indices[self.current_detector_index_in_list])

    def _navigate_next(self) -> None:
        if not self.detector_indices or len(self.detector_indices) <= 1:
            return
        self.current_detector_index_in_list = (self.current_detector_index_in_list + 1) % len(self.detector_indices)
        self._switch_to_detector(self.detector_indices[self.current_detector_index_in_list])

    def _switch_to_detector(self, new_detector_id: int) -> None:
        if new_detector_id not in self.all_detectors_data:
            return

        detector_data = self.all_detectors_data[new_detector_id]
        self.detector_id = new_detector_id
        self.resonance_frequency_ghz_title = detector_data["conceptual_freq_hz"] / 1e9

        self._load_detector_noise_products()

        title_text = f"Detector {self.detector_id} ({self.resonance_frequency_ghz_title * 1e3:.6f} MHz)"
        self.setWindowTitle(f"Noise Spectrum: {title_text}")
        if hasattr(self, "title_label"):
            self.title_label.setText(title_text)

        if hasattr(self, "detector_count_label") and self.detector_indices:
            count_text = f"({self.current_detector_index_in_list + 1} of {len(self.detector_indices)})"
            self.detector_count_label.setText(count_text)

        self._update_noise_plots()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key.Key_Left:
            self._navigate_previous()
        elif event.key() == QtCore.Qt.Key.Key_Right:
            self._navigate_next()
        else:
            super().keyPressEvent(event)

    # ---------------------------
    # UI
    # ---------------------------

    def _setup_ui(self) -> None:
        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar-ish controls: mean subtraction / exp binning
        checkbox_widget = QtWidgets.QWidget()
        checkbox_layout = QtWidgets.QHBoxLayout(checkbox_widget)
        checkbox_layout.setContentsMargins(5, 5, 5, 0)

        self.mean_subtract_checkbox = QtWidgets.QCheckBox("Mean Subtracted")
        self.mean_subtract_checkbox.setToolTip("If checked, TOD data will have its mean subtracted before plotting.")
        self.mean_subtract_checkbox.setChecked(True)
        self.mean_subtract_checkbox.stateChanged.connect(self._toggle_mean_subtraction)

        self.exp_binning_checkbox = QtWidgets.QCheckBox("Exponential Binning")
        self.exp_binning_checkbox.setToolTip("If checked, noise spectrum will use exponential binning.")
        self.exp_binning_checkbox.setChecked(True)  # Default to ON
        self.exp_binning_checkbox.toggled.connect(self._toggle_exp_binning)

        checkbox_layout.addStretch()
        checkbox_layout.addWidget(self.mean_subtract_checkbox)
        checkbox_layout.addSpacing(20)
        checkbox_layout.addWidget(self.exp_binning_checkbox)
        checkbox_layout.addStretch()

        outer_layout.addWidget(checkbox_widget)

        # Navigation bar
        nav_widget = QtWidgets.QWidget()
        nav_layout = QtWidgets.QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(5, 5, 5, 5)

        self.prev_button = QtWidgets.QPushButton("◀ Previous")
        self.prev_button.clicked.connect(self._navigate_previous)
        self.prev_button.setEnabled(len(self.detector_indices) > 1)
        nav_layout.addWidget(self.prev_button)

        title_color_str = "white" if self.dark_mode else "black"
        title_text = f"Detector {self.detector_id} ({self.resonance_frequency_ghz_title * 1e3:.6f} MHz)"
        self.title_label = QtWidgets.QLabel(title_text)
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        font = self.title_label.font()
        font.setPointSize(font.pointSize() + 2)
        self.title_label.setFont(font)
        self.title_label.setStyleSheet(f"QLabel {{ color: {title_color_str}; background-color: transparent; }}")
        nav_layout.addWidget(self.title_label, 1)

        self.next_button = QtWidgets.QPushButton("Next ▶")
        self.next_button.clicked.connect(self._navigate_next)
        self.next_button.setEnabled(len(self.detector_indices) > 1)
        nav_layout.addWidget(self.next_button)

        # Screenshot button
        self.screenshot_btn = QtWidgets.QPushButton("📷")
        self.screenshot_btn.setToolTip("Export a screenshot of this panel to the session folder (or choose location)")
        self.screenshot_btn.clicked.connect(self._export_screenshot)
        nav_layout.addWidget(self.screenshot_btn)

        detector_count_text = (
            f"({self.current_detector_index_in_list + 1} of {len(self.detector_indices)})" if self.detector_indices else ""
        )
        self.detector_count_label = QtWidgets.QLabel(detector_count_text)
        self.detector_count_label.setStyleSheet(f"QLabel {{ color: {title_color_str}; background-color: transparent; }}")
        nav_layout.addWidget(self.detector_count_label)

        outer_layout.addWidget(nav_widget)

        splitter_main = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter_top = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        plot_bg_color, plot_pen_color = ("k", "w") if self.dark_mode else ("w", "k")

        # TOD plot
        vb_time = ClickableViewBox()
        vb_time.parent_window = self
        self.plot_time_vs_mag = pg.PlotWidget(viewBox=vb_time, name="TimeVsAmplitude")
        self.plot_time_vs_mag.setBackground(plot_bg_color)
        self.plot_time_vs_mag.setLabel("left", "Amplitude", units="V")
        self.plot_time_vs_mag.setLabel("bottom", "Time", units="s")
        self.plot_time_vs_mag.showGrid(x=True, y=True, alpha=0.3)
        self.plot_time_vs_mag.setTitle("TOD", color=plot_pen_color)
        self.plot_time_vs_mag.addLegend(offset=(30, 10), labelTextColor=plot_pen_color)
        splitter_top.addWidget(self.plot_time_vs_mag)

        # Optional fast TOD
        self.plot_fast_tod = None
        if self.show_fast_tod:
            vb_fast_tod = ClickableViewBox()
            vb_fast_tod.parent_window = self
            self.plot_fast_tod = pg.PlotWidget(viewBox=vb_fast_tod, name="FastTOD")
            self.plot_fast_tod.setBackground(plot_bg_color)
            self.plot_fast_tod.setLabel("left", "Amplitude", units="V")
            self.plot_fast_tod.setLabel("bottom", "Time", units="s")
            self.plot_fast_tod.showGrid(x=True, y=True, alpha=0.2)
            self.plot_fast_tod.setTitle("Fast TOD", color=plot_pen_color)
            self.plot_fast_tod.addLegend(offset=(30, 10), labelTextColor=plot_pen_color)
            splitter_top.addWidget(self.plot_fast_tod)

        # Noise spectrum plot
        vb_spec = ClickableViewBox()
        vb_spec.parent_window = self
        self.plot_noise_spectrum = pg.PlotWidget(viewBox=vb_spec, name="NoiseSpectrum")
        self.plot_noise_spectrum.setBackground(plot_bg_color)
        if self.reference == "relative":
            self.plot_noise_spectrum.setLabel("left", "Amplitude", units="dBc/Hz")
        else:
            self.plot_noise_spectrum.setLabel("left", "Amplitude", units="dBm/Hz")
        self.plot_noise_spectrum.setLabel("bottom", "Frequency", units="Hz")
        self.plot_noise_spectrum.showGrid(x=True, y=True, alpha=0.2)
        self.plot_noise_spectrum.setTitle("Noise Spectrum", color=plot_pen_color)
        self.plot_noise_spectrum.setYRange(-180, 0)
        self.plot_noise_spectrum.addLegend(offset=(30, 10), labelTextColor=plot_pen_color)

        splitter_main.addWidget(splitter_top)
        splitter_main.addWidget(self.plot_noise_spectrum)
        splitter_main.setSizes([400, 400])

        self.noise_placeholder_label = QtWidgets.QLabel(
            "Data only populated if 'Get Noise Spectrum' has been run from the multisweep panel"
        )
        self.noise_placeholder_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.noise_placeholder_label.setStyleSheet("font-size: 12pt; color: gray;")

        outer_layout.addWidget(splitter_main)
        outer_layout.addWidget(self.noise_placeholder_label)

        self.noise_splitter = splitter_main

        self.apply_theme(self.dark_mode)

    # ---------------------------
    # Plotting
    # ---------------------------

    def _toggle_mean_subtraction(self, state):
        self.mean_subtract_enabled = (state == QtCore.Qt.CheckState.Checked.value)
        self._update_noise_plots()

    def _toggle_exp_binning(self, checked: bool):
        self.exp_binning_enabled = checked
        self._update_noise_plots()

    def _get_relative_timestamps(self, ts, num_samples):
        SS_PER_SECOND = 156250000
        first = ts[0].h * 3600 + ts[0].m * 60 + ts[0].s + ts[0].ss / SS_PER_SECOND
        last = ts[-1].h * 3600 + ts[-1].m * 60 + ts[-1].s + ts[-1].ss / SS_PER_SECOND
        total_time = last - first
        return list(np.linspace(0, total_time, num_samples))

    def _update_noise_plots(self):
        self.apply_theme(self.dark_mode)

        if not self.spectrum_data or self.single_psd_i is None or self.single_psd_q is None:
            if hasattr(self, "noise_splitter"):
                self.noise_splitter.hide()
            if hasattr(self, "noise_placeholder_label"):
                self.noise_placeholder_label.show()
            if hasattr(self, "plot_time_vs_mag"):
                self.plot_time_vs_mag.clear()
            if hasattr(self, "plot_noise_spectrum"):
                self.plot_noise_spectrum.clear()
            if self.plot_fast_tod is not None:
                self.plot_fast_tod.clear()
            return

        if hasattr(self, "noise_splitter"):
            self.noise_splitter.show()
        if hasattr(self, "noise_placeholder_label"):
            self.noise_placeholder_label.hide()

        exp_bins = 100

        self.plot_time_vs_mag.clear()
        self.plot_noise_spectrum.clear()
        if self.plot_fast_tod is not None:
            self.plot_fast_tod.clear()

        # TOD
        ts = self._get_relative_timestamps(self.spectrum_data["ts"], len(self.tod_i))
        if self.reference == "relative":
            tod_i_volts = convert_roc_to_volts(np.array(self.tod_i))
            tod_q_volts = convert_roc_to_volts(np.array(self.tod_q))
        else:
            tod_i_volts = np.array(self.tod_i)
            tod_q_volts = np.array(self.tod_q)

        if self.mean_subtract_enabled:
            tod_i_volts = tod_i_volts - np.mean(tod_i_volts)
            tod_q_volts = tod_q_volts - np.mean(tod_q_volts)

        self.plot_time_vs_mag.plot(ts, tod_i_volts, pen=pg.mkPen(IQ_COLORS["I"], width=LINE_WIDTH), name="I")
        self.plot_time_vs_mag.plot(ts, tod_q_volts, pen=pg.mkPen(IQ_COLORS["Q"], width=LINE_WIDTH), name="Q")
        self.plot_time_vs_mag.autoRange()

        # Fast TOD
        if self.show_fast_tod and self.plot_fast_tod is not None:
            pfb_ts = self.spectrum_data["pfb_ts"]
            tod_pfb_i_volts = convert_roc_to_volts(np.array(self.pfb_tod_i))
            tod_pfb_q_volts = convert_roc_to_volts(np.array(self.pfb_tod_q))

            if self.mean_subtract_enabled:
                tod_pfb_i_volts = tod_pfb_i_volts - np.mean(tod_pfb_i_volts)
                tod_pfb_q_volts = tod_pfb_q_volts - np.mean(tod_pfb_q_volts)

            self.plot_fast_tod.plot(
                pfb_ts,
                tod_pfb_i_volts,
                pen=pg.mkPen(IQ_COLORS["I"], width=LINE_WIDTH),
                name="PFB I",
            )
            self.plot_fast_tod.plot(
                pfb_ts,
                tod_pfb_q_volts,
                pen=pg.mkPen(IQ_COLORS["Q"], width=LINE_WIDTH),
                name="PFB Q",
            )
            self.plot_fast_tod.autoRange()

        # Spectrum
        frequencies = self.spectrum_data["freq_iq"]
        psd_i = self.single_psd_i[3:]
        psd_q = self.single_psd_q[3:]
        freq = frequencies[3:]

        # PFB data inclusion
        if self.show_fast_tod:
            overlap = self.spectrum_data["overlap"]
            if overlap < 2:
                overlap = 2
            pfb_psd_i = self.pfb_psd_i[overlap:]
            pfb_psd_q = self.pfb_psd_q[overlap:]
            pfb_freq = self.pfb_freq_iq[overlap:]

            freq_dsb = np.array(self.pfb_freq_dsb)
            psd_dsb = np.array(self.pfb_dual_psd)
            min_f_abs = abs(np.min(freq_dsb))
            max_f_abs = abs(np.max(freq_dsb))
            if max_f_abs >= min_f_abs:
                mask = freq_dsb >= 0
                freq_sel = freq_dsb[mask]
                psd_sel = psd_dsb[mask]
                h_label = "(I-Q)/2"
            else:
                mask = freq_dsb <= 0
                freq_sel = np.abs(freq_dsb[mask])
                psd_sel = psd_dsb[mask]
                h_label = "(I+Q)/2"

            sort_idx = np.argsort(freq_sel)
            freq_sel = freq_sel[sort_idx][2:]
            psd_sel = psd_sel[sort_idx][2:]
        else:
            pfb_freq = []
            pfb_psd_i = []
            pfb_psd_q = []
            psd_sel = []
            freq_sel = []
            h_label = ""

        full_freq = list(freq) + list(pfb_freq)
        log_freqs = np.log10(np.clip(full_freq, 1e-12, None))

        # Summary box
        amplitude = float(self.tone_amp)
        slow_freq = float(self.slow_freq)
        fast_freq_mhz = (self.fast_freq / 1e6) if self.fast_freq else 0.0
        summary_text = (
            f"<span style='font-size:12pt; color:grey;'>"
            f"Tone Amp: {amplitude:.1f} dBm <br>"
            f"Slow Bandwidth: {slow_freq:.0f} Hz <br>"
            f"Fast Bandwidth: {fast_freq_mhz:.2f} MHz </span>"
        )

        self.summary_label = pg.TextItem(html=summary_text, anchor=(0, 0), border="w", fill=(0, 0, 0, 150))
        self.plot_noise_spectrum.addItem(self.summary_label)
        self.summary_label.setPos(np.max(log_freqs) * 0.8, np.max(psd_i) * 0.9)

        self.plot_noise_spectrum.setLogMode(x=True, y=False)

        # Always compute binned arrays (needed for hover even when not displayed)
        f_bin_i, psd_i_bin = exp_bin_noise_data(freq, psd_i, exp_bins)
        f_bin_q, psd_q_bin = exp_bin_noise_data(freq, psd_q, exp_bins)
        
        # Initialize PFB binned arrays (will be populated if show_fast_tod)
        pfb_f_bin_i = []
        pfb_psd_i_bin = []
        pfb_psd_q_bin = []
        pfb_f_bin_mag = []
        pfb_psd_mag_bin = []
        
        if self.show_fast_tod:
            pfb_f_bin_i, pfb_psd_i_bin = exp_bin_noise_data(pfb_freq, pfb_psd_i, exp_bins)
            pfb_f_bin_q, pfb_psd_q_bin = exp_bin_noise_data(pfb_freq, pfb_psd_q, exp_bins)
            pfb_f_bin_mag, pfb_psd_mag_bin = exp_bin_noise_data(freq_sel, psd_sel, exp_bins)

        # Build full arrays for hover interpolation (non-binned)
        full_psd_i = list(psd_i) + list(pfb_psd_i)
        full_psd_q = list(psd_q) + list(pfb_psd_q)
        full_lin_comb = list(psd_sel)
        freq_lin_comb = list(freq_sel)

        if self.exp_binning_enabled:
            self.plot_noise_spectrum.plot(
                f_bin_i,
                psd_i_bin,
                pen=pg.mkPen(IQ_COLORS["I"], width=LINE_WIDTH),
                name="I",
            )
            self.plot_noise_spectrum.plot(
                f_bin_q,
                psd_q_bin,
                pen=pg.mkPen(IQ_COLORS["Q"], width=LINE_WIDTH),
                name="Q",
            )

            if self.show_fast_tod:
                self.plot_noise_spectrum.plot(
                    pfb_f_bin_i,
                    pfb_psd_i_bin,
                    pen=pg.mkPen(IQ_COLORS["I"], width=LINE_WIDTH, style=QtCore.Qt.DashLine),
                    name="PFB I",
                )
                self.plot_noise_spectrum.plot(
                    pfb_f_bin_q,
                    pfb_psd_q_bin,
                    pen=pg.mkPen(IQ_COLORS["Q"], width=LINE_WIDTH, style=QtCore.Qt.DashLine),
                    name="PFB Q",
                )
                self.plot_noise_spectrum.plot(
                    pfb_f_bin_mag,
                    pfb_psd_mag_bin,
                    pen=pg.mkPen(color=(255, 0, 0, 100), width=LINE_WIDTH, style=QtCore.Qt.DashLine),
                    name="wideband linear combination (I,Q)",
                )
        else:
            self.plot_noise_spectrum.plot(
                freq,
                psd_i,
                pen=pg.mkPen(IQ_COLORS["I"], width=LINE_WIDTH),
                name="I",
            )
            self.plot_noise_spectrum.plot(
                freq,
                psd_q,
                pen=pg.mkPen(IQ_COLORS["Q"], width=LINE_WIDTH),
                name="Q",
            )

            if self.show_fast_tod:
                self.plot_noise_spectrum.plot(
                    pfb_freq,
                    pfb_psd_i,
                    pen=pg.mkPen(IQ_COLORS["I"], width=LINE_WIDTH, style=QtCore.Qt.DashLine),
                    name="PFB I",
                )
                self.plot_noise_spectrum.plot(
                    pfb_freq,
                    pfb_psd_q,
                    pen=pg.mkPen(IQ_COLORS["Q"], width=LINE_WIDTH, style=QtCore.Qt.DashLine),
                    name="PFB Q",
                )
                self.plot_noise_spectrum.plot(
                    freq_sel,
                    psd_sel,
                    pen=pg.mkPen(color=(255, 0, 0, 100), width=LINE_WIDTH, style=QtCore.Qt.DashLine),
                    name="wideband linear combination (I,Q)",
                )

        self.plot_noise_spectrum.autoRange()

        # Hover label setup
        vb = self.plot_noise_spectrum.getViewBox()
        self.hover_label = pg.TextItem("", anchor=(0, 1), color="w")
        self.plot_noise_spectrum.addItem(self.hover_label)
        self.hover_label.hide()

        def on_mouse_move(evt):
            pos = evt[0]  # current mouse position in scene coordinates
            if self.plot_noise_spectrum.sceneBoundingRect().contains(pos):
                mouse_point = vb.mapSceneToView(pos)

                log_x = mouse_point.x()
                x = 10 ** log_x

                summary_rect = self.summary_label.boundingRect()
                summary_rect = self.summary_label.mapRectToScene(summary_rect)
                if summary_rect.contains(pos):
                    # Mouse is over text box — disable hover update
                    return

                if self.exp_binning_enabled:
                    # Use the binned frequency and PSD data
                    freq_i = np.log10(np.clip(np.concatenate((f_bin_i, pfb_f_bin_i if self.show_fast_tod else [])), 1e-12, None))
                    psd_i_vals = np.concatenate((psd_i_bin, pfb_psd_i_bin if self.show_fast_tod else []))
                    psd_q_vals = np.concatenate((psd_q_bin, pfb_psd_q_bin if self.show_fast_tod else []))
                    psd_dual_vals = pfb_psd_mag_bin
                    freq_l = pfb_f_bin_mag
                else:
                    # Use the original data
                    freq_i = log_freqs
                    psd_i_vals = full_psd_i
                    psd_q_vals = full_psd_q
                    psd_dual_vals = full_lin_comb
                    freq_l = freq_lin_comb


                if self.show_fast_tod:
                    if x > max(freq_l) if len(freq_l) > 0 else True:
                        self.hover_label.hide()
                        return


                if self.show_fast_tod:
                    if len(freq_i) > 0 and log_x < max(freq_i):
                        y_i = np.interp(log_x, freq_i, psd_i_vals)
                        y_q = np.interp(log_x, freq_i, psd_q_vals)
                        y_d = np.interp(log_x, np.log10(np.clip(freq_l, 1e-12, None)), psd_dual_vals) if len(freq_l) > 0 else 0
                        self.hover_label.setHtml(
                            f"<span style='color:{IQ_COLORS['I']}'>I: {y_i:.3f}</span><br>"
                            f"<span style='color:{IQ_COLORS['Q']}'>Q: {y_q:.3f}</span><br>"
                            f"<span style='color:red'>{h_label}: {y_d:.3f}</span><br>"
                            f"<span style='color:yellow'>Freq: {x:.3f} Hz</span>"
                        )
                        self.hover_label.setPos(mouse_point.x(), mouse_point.y())
                    else:
                        y_d = np.interp(log_x, np.log10(np.clip(freq_l, 1e-12, None)), psd_dual_vals) if len(freq_l) > 0 else 0
                        self.hover_label.setHtml(
                            f"<span style='color:red'>{h_label}: {y_d:.3f}</span><br>"
                            f"<span style='color:yellow'> Freq: {x:.3f} Hz</span>"
                        )
                        self.hover_label.setPos(mouse_point.x(), mouse_point.y())
                else:
                    if len(freq_i) > 0 and log_x < max(freq_i):
                        y_i = np.interp(log_x, freq_i, psd_i_vals)
                        y_q = np.interp(log_x, freq_i, psd_q_vals)
                        self.hover_label.setHtml(
                            f"<span style='color:{IQ_COLORS['I']}'>I: {y_i:.3f}</span><br>"
                            f"<span style='color:{IQ_COLORS['Q']}'>Q: {y_q:.3f}</span><br>"
                            f"<span style='color:yellow'>Freq: {x:.3f} Hz</span>"
                        )
                        self.hover_label.setPos(mouse_point.x(), mouse_point.y())

                self.hover_label.show()
            else:
                self.hover_label.hide()

        self.proxy = pg.SignalProxy(self.plot_noise_spectrum.scene().sigMouseMoved, rateLimit=60, slot=on_mouse_move)

    # ---------------------------
    # Theme
    # ---------------------------

    def apply_theme(self, dark_mode: bool):
        self.dark_mode = dark_mode

        bg_color_hex = "#1C1C1C" if dark_mode else "#FFFFFF"
        self.setStyleSheet(f"QWidget {{ background-color: {bg_color_hex}; }}")

        title_color_str = "white" if dark_mode else "black"
        plot_bg_color, plot_pen_color = ("k", "w") if dark_mode else ("w", "k")

        # Labels
        if hasattr(self, "title_label"):
            self.title_label.setStyleSheet(f"QLabel {{ color: {title_color_str}; background-color: transparent; }}")
        if hasattr(self, "detector_count_label"):
            self.detector_count_label.setStyleSheet(
                f"QLabel {{ color: {title_color_str}; background-color: transparent; }}"
            )

        # Plots
        if hasattr(self, "plot_time_vs_mag"):
            self.plot_time_vs_mag.setBackground(plot_bg_color)
            plot_item = self.plot_time_vs_mag.getPlotItem()
            plot_item.setTitle("TOD", color=plot_pen_color)
            for ax in ("left", "bottom"):
                axis = plot_item.getAxis(ax)
                axis.setPen(plot_pen_color)
                axis.setTextPen(plot_pen_color)

        if hasattr(self, "plot_noise_spectrum"):
            self.plot_noise_spectrum.setBackground(plot_bg_color)
            plot_item = self.plot_noise_spectrum.getPlotItem()
            plot_item.setTitle("Noise Spectrum", color=plot_pen_color)
            for ax in ("left", "bottom"):
                axis = plot_item.getAxis(ax)
                axis.setPen(plot_pen_color)
                axis.setTextPen(plot_pen_color)

        if self.plot_fast_tod is not None:
            self.plot_fast_tod.setBackground(plot_bg_color)
            plot_item = self.plot_fast_tod.getPlotItem()
            plot_item.setTitle("Fast TOD", color=plot_pen_color)
            for ax in ("left", "bottom"):
                axis = plot_item.getAxis(ax)
                axis.setPen(plot_pen_color)
                axis.setTextPen(plot_pen_color)

        # Buttons
        if dark_mode:
            button_style = """
                QPushButton {
                    background-color: #3C3C3C;
                    color: white;
                    border: 1px solid #555555;
                    padding: 5px 10px;
                    border-radius: 3px;
                }
                QPushButton:hover { background-color: #4C4C4C; }
                QPushButton:pressed { background-color: #2C2C2C; }
                QPushButton:disabled { background-color: #1C1C1C; color: #666666; }
            """
        else:
            button_style = """
                QPushButton {
                    background-color: #F0F0F0;
                    color: black;
                    border: 1px solid #CCCCCC;
                    padding: 5px 10px;
                    border-radius: 3px;
                }
                QPushButton:hover { background-color: #E0E0E0; }
                QPushButton:pressed { background-color: #D0D0D0; }
                QPushButton:disabled { background-color: #F8F8F8; color: #999999; }
            """

        for name in ("prev_button", "next_button", "screenshot_btn"):
            if hasattr(self, name):
                getattr(self, name).setStyleSheet(button_style)

        # Checkboxes
        if hasattr(self, "mean_subtract_checkbox"):
            self.mean_subtract_checkbox.setStyleSheet(f"QCheckBox {{ color: {title_color_str}; }}")
        if hasattr(self, "exp_binning_checkbox"):
            self.exp_binning_checkbox.setStyleSheet(f"QCheckBox {{ color: {title_color_str}; }}")
