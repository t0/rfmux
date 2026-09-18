"""
Capture Settings dialog for the Pulse Capture panel.

Thin view over
:class:`rfmux.pulse_capture.capture_session.PulseCaptureConfig`:
every derived number (ms → samples, auto-sized ring buffer, memory) and
every rule comes from the config object; the dialog renders them live
at the stream rate the capture will actually run at.
"""

from __future__ import annotations


from PyQt6 import QtCore, QtWidgets

from .utils import apply_issue_banner

from ...core.transferfunctions import decimation_to_sampling
from ...pulse_capture.capture_session import (
    PulseCaptureConfig,
)
from ...pulse_capture.detection import (
    EDGE_LOOKBACK_FRACTION,
)


def _ms(ms: float) -> str:
    return f"{ms/1000:.3g} s" if ms >= 1000 else f"{ms:.3g} ms"


def _rows_html(rows) -> str:
    """A label's worth of name/value pairs, one per line, names aligned."""
    body = "".join(
        f"<tr><td style='padding-right:10px; color:#777'>{name}</td>"
        f"<td>{value}</td></tr>" for name, value in rows)
    return f"<table cellspacing='0' cellpadding='1'>{body}</table>"


class PulseCaptureSettingsForm(QtWidgets.QWidget):
    """Edit a PulseCaptureConfig with live unit conversions.

    *gate* is a button enabled only while the settings validate;
    ``updated`` fires after every recomputation."""

    updated = QtCore.pyqtSignal()

    def __init__(self, parent=None, *,
                 config: PulseCaptureConfig | None = None,
                 sample_rate: float = decimation_to_sampling(6),
                 mode: str = "slow",
                 n_channels: int = 2,
                 df_available: bool = True,
                 gate: QtWidgets.QAbstractButton | None = None):
        super().__init__(parent)
        self.gate = gate
        self.sample_rate = float(sample_rate)
        self.mode = mode
        self.n_channels = max(1, n_channels)
        self.df_available = bool(df_available)
        config = config or PulseCaptureConfig()
        self._updating = False

        # Two decisions belong to the user — how selective the trigger
        # is, and how long a pulse can be.  Everything else is derived
        # from those, measured from the training record, or has a
        # defensible default, so it lives under Advanced.
        outer = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        outer.addLayout(form)

        rate_str = (f"{self.sample_rate/1e6:.2f} MHz" if
                    self.sample_rate >= 1e5
                    else f"{self.sample_rate:,.0f} Hz")
        form.addRow("Stream:", QtWidgets.QLabel(
            f"{mode} @ {rate_str}"))

        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.5, 1000.0)
        self.threshold_spin.setSingleStep(0.5)
        self.threshold_spin.setValue(config.threshold_sigma)
        self.threshold_spin.setToolTip(
            "Significance required by both the amplitude and edge triggers.")
        form.addRow("Threshold σ:", self.threshold_spin)

        self.max_pulse_spin = QtWidgets.QDoubleSpinBox()
        self.max_pulse_spin.setRange(0.1, 60_000.0)
        self.max_pulse_spin.setDecimals(1)
        self.max_pulse_spin.setValue(config.max_pulse_ms)
        # Tooltip is built in _update_dependent_values: its ratios come
        # from the config's constants.
        form.addRow("Max pulse (ms):", self.max_pulse_spin)

        self.pre_pulse_spin = QtWidgets.QDoubleSpinBox()
        self.post_pulse_spin = QtWidgets.QDoubleSpinBox()
        for spin, value in ((self.pre_pulse_spin, config.pre_pulse_ms),
                            (self.post_pulse_spin, config.post_pulse_ms)):
            spin.setRange(0.0, 60_000.0)
            spin.setDecimals(3)
            spin.setValue(value)
        self.pre_pulse_spin.setToolTip(
            "Time saved before the trigger, at least 2 samples.")
        self.post_pulse_spin.setToolTip(
            "Time saved after the pulse settles.")
        form.addRow("Pre-pulse time (ms):", self.pre_pulse_spin)
        form.addRow("Post-pulse time (ms):", self.post_pulse_spin)

        self.coincidence_spin = QtWidgets.QDoubleSpinBox()
        self.coincidence_spin.setRange(0.0, 60_000.0)
        self.coincidence_spin.setDecimals(3)
        self.coincidence_spin.setSpecialValueText("off")
        self.coincidence_spin.setValue(config.coincidence_window_ms)
        self.coincidence_spin.setToolTip(
            "Group channel triggers within this interval as one event.")
        form.addRow("Coincidence window (ms):", self.coincidence_spin)
        self.dump_check = QtWidgets.QCheckBox(
            "Save every channel with each event")
        self.dump_check.setChecked(config.dump_all_channels)
        self.dump_check.setToolTip(
            "Also save untriggered channels over each event's time span.")
        form.addRow(self.dump_check)

        self.noise_capture_spin = QtWidgets.QDoubleSpinBox()
        self.noise_capture_spin.setRange(0.0, 86_400.0)
        self.noise_capture_spin.setDecimals(3)
        self.noise_capture_spin.setSpecialValueText("off")
        self.noise_capture_spin.setValue(config.noise_capture_interval_s)
        self.noise_capture_spin.setToolTip(
            "Average interval between normally distributed noise samples.")
        form.addRow("Noise sample every (s):", self.noise_capture_spin)

        # The 1/f window is its own time scale, seconds whatever the
        # pulse length: the record fitted for sigma and the span of the
        # rolling baseline median.
        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(0.0, 600_000.0)
        self.window_spin.setDecimals(0)
        self.window_spin.setSingleStep(500.0)
        self.window_spin.setSpecialValueText(
            f"derived ({PulseCaptureConfig.NOISE_TRAIN_PULSES}× max pulse)")
        self.window_spin.setValue(config.noise_train_ms)
        self.window_spin.setToolTip(
            "Window used to estimate noise and the rolling baseline; 0 derives it.")
        form.addRow("1/f window (ms):", self.window_spin)
        self.noise_label = QtWidgets.QLabel()
        self.noise_label.setToolTip(
            "Effective noise window at the current sample rate.")
        form.addRow("Window at this rate:", self.noise_label)

        adv_box = QtWidgets.QGroupBox("Advanced")
        adv_box.setCheckable(True)
        adv_box.setChecked(False)
        adv = QtWidgets.QFormLayout(adv_box)
        form.addRow(adv_box)
        self.adv_box = adv_box
        adv_box.toggled.connect(
            lambda on: [adv.itemAt(i).widget().setVisible(on)
                        for i in range(adv.count())
                        if adv.itemAt(i).widget() is not None])

        self.trigger_spin = QtWidgets.QSpinBox()
        self.trigger_spin.setRange(0, 64)
        self.trigger_spin.setSpecialValueText("auto")
        self.trigger_spin.setValue(config.trigger_samples)
        self.trigger_spin.setToolTip(
            "Consecutive threshold crossings required; auto uses the stream rate.")
        adv.addRow("Trigger confirmation (samples):", self.trigger_spin)

        self.end_spin = QtWidgets.QDoubleSpinBox()
        self.end_spin.setRange(0.1, 100.0)
        self.end_spin.setSingleStep(0.1)
        self.end_spin.setValue(config.end_sigma)
        self.end_spin.setToolTip(
            "Pulse ends when BOTH I and Q stay within this band")
        adv.addRow("End σ:", self.end_spin)

        self.min_end_spin = QtWidgets.QSpinBox()
        self.min_end_spin.setRange(1, 100_000)
        self.min_end_spin.setValue(config.min_end_samples)
        self.min_end_spin.setToolTip(
            "Minimum settled-sample count for the end-confirmation bucket.")
        adv.addRow("End confirmation floor (samples):", self.min_end_spin)

        self.min_pulse_spin = QtWidgets.QDoubleSpinBox()
        self.min_pulse_spin.setRange(0.0, 10_000.0)
        self.min_pulse_spin.setDecimals(3)
        self.min_pulse_spin.setValue(config.min_pulse_ms)
        self.min_pulse_spin.setToolTip(
            "Discard shorter pulses as glitches; 0 keeps all pulses.")
        adv.addRow("Min pulse (ms):", self.min_pulse_spin)

        self.pileup_check = QtWidgets.QCheckBox(
            "Split piled-up events (edge re-trigger)")
        self.pileup_check.setToolTip(
            "Split a decaying pulse when a fresh trigger edge arrives.")
        self.pileup_check.setChecked(config.enable_pileup)
        adv.addRow(self.pileup_check)

        self.basis_combo = QtWidgets.QComboBox()
        self.basis_combo.addItems(["I/Q (quadratures)",
                                   "df/dissipation (rotated)"])
        self.basis_combo.setCurrentIndex(
            1 if config.trigger_basis == "df" else 0)
        if not self.df_available:
            # Nothing to rotate with: the item stays visible so the
            # option is known, but cannot be chosen.  A stored "df"
            # keeps showing (and reading back) as df, so the setting
            # survives the dialog and applies once a calibration exists;
            # a channel without one stays on the quadratures anyway.
            item = self.basis_combo.model().item(1)
            item.setEnabled(False)
            item.setToolTip(
                "Unavailable until these channels have a df calibration.")
        self.basis_combo.setToolTip(
            "Apply thresholds in raw I/Q or calibrated df/dissipation coordinates.")
        adv.addRow("Trigger basis:", self.basis_combo)

        # What each primary knob drives, at the actual stream rate —
        # the derivations live in PulseCaptureConfig, this only renders
        # them.
        self.pulse_derived_label = QtWidgets.QLabel()
        self.pulse_derived_label.setWordWrap(True)
        self.pulse_derived_label.setToolTip(
            "Derived buffer, timeout, lookback, and baseline time scales.")
        form.addRow("Time scales:", self.pulse_derived_label)

        self.sigma_derived_label = QtWidgets.QLabel()
        self.sigma_derived_label.setWordWrap(True)
        self.sigma_derived_label.setToolTip(
            "Derived amplitude, edge, and pileup thresholds.")
        form.addRow("Threshold σ sets:", self.sigma_derived_label)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        form.addRow(self.status_label)

        for w in (self.threshold_spin, self.end_spin, self.pre_pulse_spin,
                  self.post_pulse_spin, self.coincidence_spin,
                  self.noise_capture_spin,
                  self.min_pulse_spin, self.max_pulse_spin, self.window_spin,
                  self.trigger_spin, self.min_end_spin):
            w.valueChanged.connect(self._update_dependent_values)
        self.pileup_check.toggled.connect(self._update_dependent_values)
        self.dump_check.toggled.connect(self._update_dependent_values)
        adv_box.toggled.emit(False)
        self._update_dependent_values()

    def get_config(self) -> PulseCaptureConfig:
        return PulseCaptureConfig(
            threshold_sigma=float(self.threshold_spin.value()),
            end_sigma=float(self.end_spin.value()),
            pre_pulse_ms=float(self.pre_pulse_spin.value()),
            post_pulse_ms=float(self.post_pulse_spin.value()),
            coincidence_window_ms=float(self.coincidence_spin.value()),
            dump_all_channels=self.dump_check.isChecked(),
            noise_capture_interval_s=float(self.noise_capture_spin.value()),
            trigger_samples=int(self.trigger_spin.value()),
            min_pulse_ms=float(self.min_pulse_spin.value()),
            max_pulse_ms=float(self.max_pulse_spin.value()),
            noise_train_ms=float(self.window_spin.value()),
            enable_pileup=self.pileup_check.isChecked(),
            min_end_samples=int(self.min_end_spin.value()),
            trigger_basis=("df" if self.basis_combo.currentIndex() == 1
                           else "iq"),
        )

    def _update_dependent_values(self):
        if self._updating:
            return
        self._updating = True
        try:
            cfg = self.get_config()
            d = cfg.describe(self.sample_rate, self.n_channels)
            self.max_pulse_spin.setToolTip(
                "Longest pulse you expect — every time scale in pulse "
                "detection derives from this.  Estimate generously.\n"
                f"Sets the ring buffer ({cfg.BUFFER_SAFETY:g}×, plus the "
                "pre-pulse and post-pulse times), the hard stop that "
                "force-ends a stuck capture "
                f"({cfg.HARD_STOP_FACTOR:g}×, plus the post-pulse time), "
                f"the noise-training length ({cfg.NOISE_TRAIN_PULSES}×), "
                "the rolling-baseline median span, and the edge-detector "
                f"lookback ({EDGE_LOOKBACK_FRACTION:.0%}).")
            # The record is memory-bounded at fast rates and floored at
            # slow ones; the label says which length is actually used.
            span = d["noise_train_span_ms"]
            n_noise = d["noise_samples"]
            wanted = round(span * 1e-3 * self.sample_rate)
            if abs(n_noise - wanted) <= 1:
                note = f"{n_noise:,} samples"
            elif n_noise < wanted:
                note = f"capped at {n_noise:,} samples from {_ms(span)}"
            else:
                note = f"floor of {n_noise:,} samples over {_ms(span)}"
            self.noise_label.setText(
                f"{_ms(d['noise_train_actual_ms'])} ({note})")
            acc = d["accidental_per_min"]
            acc_str = (f"{acc:,.0f}/min" if acc >= 1 else
                       f"{acc*60:.2g}/hr" if acc >= 0.001 else "negligible")

            rows = [
                ("ring buffer", f"{d['buf_samples']:,} samples "
                                f"({d['buf_mb_per_channel']:.2f} MB/ch, "
                                f"{d['buf_mb_total']:.2f} MB total)"),
                ("saved around the pulse",
                 f"{d['pre_pulse_samples']:,} samples before the trigger, "
                 f"{d['post_pulse_samples']:,} after it settled"),
                ("hard stop", f"{_ms(d['max_capture_ms'])} "
                              f"({cfg.HARD_STOP_FACTOR:g}× + post-pulse; a "
                              "stuck capture is saved and flagged truncated)"),
                ("noise training", f"{_ms(d['noise_train_actual_ms'])} "
                                   f"({d['noise_samples']:,} samples)"),
                ("baseline median", _ms(d['baseline_window_ms'])),
                ("edge lookback", f"{_ms(d['edge_lookback_ms'])} "
                                  f"({d['edge_lookback']:,} samples)"),
                ("end floor", f"{d['min_end_samples']} samples "
                              f"({_ms(d['min_end_ms'])})"),
            ]
            if cfg.min_pulse_ms > 0:
                rows.append(("min pulse", f"{d['min_pulse_samples']} samples"))
            self.pulse_derived_label.setText(_rows_html(rows))
            n_conf = d['trigger_samples']
            self.sigma_derived_label.setText(_rows_html([
                ("confirmation", f"{n_conf} sample{'s' if n_conf != 1 else ''}; "
                                 f"accidentals ≈ {acc_str} per channel"),
                ("edge jump", f"> {cfg.threshold_sigma:g} jump-σ over the "
                              f"lookback (≈ {d['edge_floor_sigma']:.1f}σ "
                              "amplitude floor in white noise)"),
            ]))

            issues = cfg.validate(self.sample_rate)
            self.valid = apply_issue_banner(self.status_label, self.gate,
                                            issues)
        finally:
            self._updating = False
        self.updated.emit()


class PulseCaptureSettingsDialog(QtWidgets.QDialog):
    """The form with OK and Cancel; the form's controls are reachable
    on the dialog."""

    def __init__(self, parent=None, **form_kwargs):
        super().__init__(parent)
        self.setWindowTitle("Pulse Capture Settings")
        self.setModal(True)
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.form = PulseCaptureSettingsForm(
            self, gate=self.buttons.button(
                QtWidgets.QDialogButtonBox.StandardButton.Ok),
            **form_kwargs)
        outer = QtWidgets.QVBoxLayout(self)
        outer.addWidget(self.form)
        outer.addWidget(self.buttons)
        self.form.updated.connect(self._fit_height)
        self.setMinimumWidth(520)
        self._fit_height()

    def get_config(self) -> PulseCaptureConfig:
        return self.form.get_config()

    def __getattr__(self, name):
        return getattr(self.__dict__.get("form"), name)

    def _fit_height(self) -> None:
        """Tall enough for the derived tables, which wrap to the width
        and grow with the settings; never shorter than they need."""
        self.layout().activate()
        want = self.layout().totalHeightForWidth(self.width()) \
            if self.layout().hasHeightForWidth() else self.sizeHint().height()
        self.resize(self.width(), max(want, self.sizeHint().height()))
