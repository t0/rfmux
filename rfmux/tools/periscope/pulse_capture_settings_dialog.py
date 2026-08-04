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
            "How significant an event must be, used by BOTH trigger "
            "tests:\n"
            "• amplitude — EITHER I or Q deviates this many σ from "
            "baseline, and\n"
            "• edge — the deviation GREW by this many jump-σ within "
            "the edge lookback.\n"
            "The edge test compares two raw samples, so the baseline "
            "cancels out of it: slow 1/f wander that drifts across the "
            "amplitude band cannot fake it.")
        form.addRow("Threshold σ:", self.threshold_spin)

        self.max_pulse_spin = QtWidgets.QDoubleSpinBox()
        self.max_pulse_spin.setRange(0.1, 60_000.0)
        self.max_pulse_spin.setDecimals(1)
        self.max_pulse_spin.setValue(config.max_pulse_ms)
        self.max_pulse_spin.setToolTip(
            "Longest pulse you expect — every time scale in the "
            "detector derives from this.  Estimate generously.\n"
            "Sets the ring buffer (1.5×), the hard stop that "
            "force-ends a stuck capture (1.2×), the noise-training "
            "length (20×), the rolling-baseline median span, and the "
            "edge-detector lookback (10%).")
        form.addRow("Max pulse (ms):", self.max_pulse_spin)

        # Training is derived, not chosen: what matters is that the
        # window is long compared with a pulse, and that ratio follows
        # the pulse length automatically.
        self.noise_label = QtWidgets.QLabel()
        self.noise_label.setToolTip(
            f"{PulseCaptureConfig.NOISE_TRAIN_PULSES}x the max pulse "
            "length.  The record is fitted for the noise level (sigma), "
            "and the same span is what the rolling baseline median "
            "covers — sigma is stationary and wants a long record, the "
            "mean drifts and wants recency.\n"
            "Robust estimators tolerate pulses in the window.")
        form.addRow("Noise training:", self.noise_label)

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
            "Fraction of the pulse length kept before the trigger and "
            "after the pulse drops below threshold (the saved tail), "
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

        self.pileup_check = QtWidgets.QCheckBox(
            "Split piled-up events (edge re-trigger)")
        self.pileup_check.setToolTip(
            "A fresh edge arriving while the current pulse is decaying "
            "splits the capture into separate events.  Uses the same "
            "edge detector as the trigger.")
        self.pileup_check.setChecked(config.enable_pileup)
        adv.addRow(self.pileup_check)

        # What each primary knob drives, at the actual stream rate —
        # the derivations live in PulseCaptureConfig, this only renders
        # them.
        self.pulse_derived_label = QtWidgets.QLabel()
        self.pulse_derived_label.setWordWrap(True)
        self.pulse_derived_label.setToolTip(
            "Everything with units of time derives from the max pulse "
            "length, so one setting works at any stream rate.")
        form.addRow("Max pulse sets:", self.pulse_derived_label)

        self.sigma_derived_label = QtWidgets.QLabel()
        self.sigma_derived_label.setWordWrap(True)
        self.sigma_derived_label.setToolTip(
            "Everything statistical derives from the threshold.  The "
            "edge jump-σ itself is measured from the training record "
            "at the lookback lag, so filter correlation and 1/f power "
            "are priced in automatically.")
        form.addRow("Threshold σ sets:", self.sigma_derived_label)

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
                  self.trigger_spin):
            w.valueChanged.connect(self._update_dependent_values)
        for c in (self.pileup_check,):
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
            enable_pileup=self.pileup_check.isChecked(),
        )

    def _update_dependent_values(self):
        if self._updating:
            return
        self._updating = True
        try:
            cfg = self.get_config()
            d = cfg.describe(self.sample_rate, self.n_channels)
            span = d["noise_train_span_ms"]
            self.noise_label.setText(
                f"{span/1000:.3g} s "
                f"({cfg.NOISE_TRAIN_PULSES}× the max pulse length)")
            acc = d["accidental_per_min"]
            acc_str = (f"{acc:,.0f}/min" if acc >= 1 else
                       f"{acc*60:.2g}/hr" if acc >= 0.001 else "negligible")

            def _ms(ms):
                return f"{ms/1000:.3g} s" if ms >= 1000 else f"{ms:.3g} ms"

            self.pulse_derived_label.setText(
                f"ring buffer {d['buf_samples']:,} samples "
                f"({d['buf_mb_per_channel']:.2f} MB/ch, "
                f"{d['buf_mb_total']:.2f} MB total) · "
                f"hard stop at {_ms(d['max_capture_ms'])} "
                f"(1.2× — a stuck capture is saved and flagged "
                f"truncated) · "
                f"noise training {_ms(d['noise_train_actual_ms'])} "
                f"({d['noise_samples']:,} samples) · "
                f"baseline median over {_ms(d['baseline_window_ms'])} · "
                f"edge lookback {_ms(d['edge_lookback_ms'])} "
                f"({d['edge_lookback']:,} samples)"
                + (f" · min pulse {d['min_pulse_samples']} samples"
                   if cfg.min_pulse_ms > 0 else ""))
            self.sigma_derived_label.setText(
                f"confirmation {d['trigger_samples']} sample"
                f"{'s' if d['trigger_samples'] != 1 else ''} "
                f"(accidentals ≈ {acc_str} per channel) · "
                f"edge jump > {cfg.threshold_sigma:g} jump-σ over the "
                f"lookback (≈ {d['edge_floor_sigma']:.1f}σ amplitude "
                f"floor in white noise)")

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
