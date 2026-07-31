"""
Capture Settings dialog for the Pulse Capture panel.

Thin view over
:class:`rfmux.algorithms.measurement.pulse_capture_session.PulseCaptureConfig`:
every derived number (ms → samples, auto-sized ring buffer, memory) and
every rule comes from the config object; the dialog renders them live
at the stream rate the capture will actually run at.
"""

from __future__ import annotations

from dataclasses import replace

from PyQt6 import QtWidgets

from ...algorithms.measurement.pulse_capture_session import (
    PulseCaptureConfig,
)

_BANNER_CSS = {
    "error": "background-color: #f8d7da; color: #721c24; "
             "padding: 5px; border-radius: 6px;",
    "warning": "background-color: #fff3cd; color: #856404; "
               "padding: 5px; border-radius: 6px;",
    "info": "background-color: #d1ecf1; color: #0c5460; "
            "padding: 5px; border-radius: 6px;",
}


class PulseCaptureSettingsDialog(QtWidgets.QDialog):
    """Edit a PulseCaptureConfig with live unit conversions."""

    def __init__(self, parent=None, *,
                 config: PulseCaptureConfig | None = None,
                 sample_rate: float = 596.0464477539062,
                 mode: str = "slow",
                 n_channels: int = 2):
        super().__init__(parent)
        self.setWindowTitle("Pulse Capture Settings")
        self.setModal(True)
        self.sample_rate = float(sample_rate)
        self.mode = mode
        self.n_channels = max(1, n_channels)
        config = config or PulseCaptureConfig()
        self._updating = False

        # Two decisions belong to the user — how selective the trigger
        # is, and how long they are willing to spend training.  The rest
        # is measured from the training record or has a defensible
        # default, so it lives under Advanced.
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
            "Trigger when EITHER I or Q deviates this many σ from "
            "baseline")
        form.addRow("Threshold σ:", self.threshold_spin)

        self.noise_spin = QtWidgets.QDoubleSpinBox()
        self.noise_spin.setRange(0.001, 300.0)
        self.noise_spin.setDecimals(3)
        self.noise_spin.setSingleStep(1.0)
        self.noise_spin.setValue(config.noise_train_ms / 1000.0)
        self.noise_spin.setToolTip(
            "How long to watch before capturing starts.\n"
            "This record is fitted for the noise level and for the 1/f "
            "knee that sets the baseline tracking window; the knee is "
            "only visible if training runs well past it.\n"
            "Robust estimators tolerate pulses in the window.")
        form.addRow("Noise training (s):", self.noise_spin)

        adv_box = QtWidgets.QGroupBox("Advanced")
        adv_box.setCheckable(True)
        adv_box.setChecked(False)
        adv = QtWidgets.QFormLayout(adv_box)
        outer.addWidget(adv_box)
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
            "Consecutive samples that must clear the threshold before a "
            "capture starts.  auto keeps accidental triggers under "
            "1/min per channel at this stream rate.\n"
            "How much evidence one sample is depends entirely on the "
            "rate: at 5σ noise alone crosses ~2.5 times per HOUR at "
            "596 Hz but ~1.4 times per SECOND on the PFB stream.  "
            "Forcing 2 everywhere would reject real pulses on a heavily "
            "decimated slow stream, where a fast pulse spans less than "
            "one sample.")
        adv.addRow("Trigger confirmation (samples):", self.trigger_spin)

        self.end_spin = QtWidgets.QDoubleSpinBox()
        self.end_spin.setRange(0.1, 100.0)
        self.end_spin.setSingleStep(0.1)
        self.end_spin.setValue(config.end_sigma)
        self.end_spin.setToolTip(
            "Pulse ends when BOTH I and Q stay within this band")
        adv.addRow("End σ:", self.end_spin)

        self.margin_spin = QtWidgets.QDoubleSpinBox()
        self.margin_spin.setRange(0.0, 1.0)
        self.margin_spin.setSingleStep(0.05)
        self.margin_spin.setValue(config.margin_fraction)
        self.margin_spin.setToolTip(
            "Fraction of the pulse length shown before the trigger, "
            "and the adaptive end-confirmation count")
        adv.addRow("Margin fraction:", self.margin_spin)

        self.min_pulse_spin = QtWidgets.QDoubleSpinBox()
        self.min_pulse_spin.setRange(0.0, 10_000.0)
        self.min_pulse_spin.setDecimals(3)
        self.min_pulse_spin.setValue(config.min_pulse_ms)
        self.min_pulse_spin.setToolTip(
            "Completed captures shorter than this are discarded as "
            "glitches (0 = keep everything)")
        adv.addRow("Min pulse (ms):", self.min_pulse_spin)

        self.max_pulse_spin = QtWidgets.QDoubleSpinBox()
        self.max_pulse_spin.setRange(0.1, 60_000.0)
        self.max_pulse_spin.setDecimals(1)
        self.max_pulse_spin.setValue(config.max_pulse_ms)
        self.max_pulse_spin.setToolTip(
            "Longest pulse the ring buffer must hold — captures that "
            "outlast the buffer lose their rising edge.\n"
            "Estimate generously: it also sets the floor under the "
            "baseline tracking window.")
        adv.addRow("Max pulse (ms):", self.max_pulse_spin)

        self.baseline_auto_check = QtWidgets.QCheckBox(
            "Measure from noise training")
        self.baseline_auto_check.setChecked(config.baseline_track_auto)
        self.baseline_auto_check.setToolTip(
            "Take the tracking window from the training data instead of "
            "setting it by hand.\n"
            "The usable window tops out at the 1/f knee — below it the "
            "average is only smoothing white noise, above it the "
            "baseline has already moved — and the training record "
            "measures exactly that.\n"
            "Floored at a multiple of the max pulse length so the "
            "tracker can never absorb a pulse tail.")
        adv.addRow("Baseline tracking:", self.baseline_auto_check)

        self.baseline_spin = QtWidgets.QDoubleSpinBox()
        self.baseline_spin.setRange(0.0, 600_000.0)
        self.baseline_spin.setDecimals(1)
        self.baseline_spin.setValue(config.baseline_track_ms)
        self.baseline_spin.setToolTip(
            "Follow baseline drift with an exponential moving average "
            "of the quiet samples (0 = frozen baseline).\n"
            "Needed under 1/f noise, where the true baseline wanders "
            "away from the training-time mean while sigma stays put — "
            "causing false triggers and, worse, an end condition that "
            "can never be satisfied.\n"
            "Choose pulse length << this << drift timescale.")
        adv.addRow("   … or fixed (ms):", self.baseline_spin)

        self.pileup_check = QtWidgets.QCheckBox(
            "Split piled-up events (derivative re-trigger)")
        self.pileup_check.setChecked(config.enable_pileup)
        adv.addRow(self.pileup_check)

        self.derived_label = QtWidgets.QLabel()
        self.derived_label.setWordWrap(True)
        form.addRow("At this rate:", self.derived_label)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        form.addRow(self.status_label)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        form.addRow(self.buttons)

        for w in (self.threshold_spin, self.end_spin, self.margin_spin,
                  self.min_pulse_spin, self.max_pulse_spin,
                  self.noise_spin, self.baseline_spin, self.trigger_spin):
            w.valueChanged.connect(self._update_dependent_values)
        for c in (self.pileup_check, self.baseline_auto_check):
            c.toggled.connect(self._update_dependent_values)
        adv_box.setChecked(False)
        adv_box.toggled.emit(False)
        self._update_dependent_values()
        self.resize(520, 400)

    def get_config(self) -> PulseCaptureConfig:
        return PulseCaptureConfig(
            threshold_sigma=float(self.threshold_spin.value()),
            end_sigma=float(self.end_spin.value()),
            margin_fraction=float(self.margin_spin.value()),
            trigger_samples=int(self.trigger_spin.value()),
            min_pulse_ms=float(self.min_pulse_spin.value()),
            max_pulse_ms=float(self.max_pulse_spin.value()),
            noise_train_ms=float(self.noise_spin.value()) * 1000.0,
            baseline_track_auto=self.baseline_auto_check.isChecked(),
            baseline_track_ms=float(self.baseline_spin.value()),
            enable_pileup=self.pileup_check.isChecked(),
        )

    def _update_dependent_values(self):
        if self._updating:
            return
        self._updating = True
        try:
            cfg = self.get_config()
            # Values that auto supersedes stay visible (they are the
            # fallbacks) but greyed, rather than leaving two
            # live-looking controls for one quantity.
            self.baseline_spin.setEnabled(not cfg.baseline_track_auto)
            d = cfg.describe(self.sample_rate, self.n_channels)
            acc = d["accidental_per_min"]
            acc_str = (f"{acc:,.0f}/min" if acc >= 1 else
                       f"{acc*60:.2g}/hr" if acc >= 0.001 else "negligible")
            self.derived_label.setText(
                f"noise training = {d['noise_samples']:,} samples "
                f"({d['noise_train_actual_ms']/1000:.3g} s) · "
                f"accidental triggers ≈ {acc_str} per channel · "
                f"buffer = {d['buf_samples']:,} samples "
                f"({d['buf_mb_per_channel']:.2f} MB/ch, "
                f"{d['buf_mb_total']:.2f} MB total) · "
                f"longest recordable ≈ {d['max_recordable_ms']:,.0f} ms · "
                f"min pulse = {d['min_pulse_samples']} samples"
                + (f" · baseline measured at training, no faster than "
                   f"{d['baseline_track_min_ms']:,.0f} ms "
                   f"({d['baseline_track_min_samples']:,} samples)"
                   if d['baseline_track_auto'] else
                   f" · baseline EMA = "
                   f"{d['baseline_track_samples']:,} samples"
                   if d['baseline_track_samples'] else
                   " · baseline frozen"))

            issues = cfg.validate(self.sample_rate)
            errors = [m for s, m in issues if s == "error"]
            worst = ("error" if errors else
                     "warning" if any(s == "warning" for s, _ in issues)
                     else "info" if issues else None)
            if worst:
                self.status_label.setText("\n".join(m for _, m in issues))
                self.status_label.setStyleSheet(_BANNER_CSS[worst])
            else:
                self.status_label.setText("")
                self.status_label.setStyleSheet("")
            self.buttons.button(
                QtWidgets.QDialogButtonBox.StandardButton.Ok
            ).setEnabled(not errors)
        finally:
            self._updating = False
