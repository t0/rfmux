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
from ...pulse_capture.channel_keys import channel_arg, describe
from ...pulse_capture.events import NoiseSampler
from ...pulse_capture.detection import (
    EDGE_LOOKBACK_FRACTION,
    END_CONFIRM_FRACTION,
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
    ``updated`` fires after every recomputation.  Given *channels*, a
    table sets each one's trigger settings."""

    updated = QtCore.pyqtSignal()

    def __init__(self, parent=None, *,
                 config: PulseCaptureConfig | None = None,
                 sample_rate: float = decimation_to_sampling(6),
                 mode: str = "slow",
                 channels=None,
                 df_available: bool = True,
                 gate: QtWidgets.QAbstractButton | None = None):
        super().__init__(parent)
        self.gate = gate
        self.sample_rate = float(sample_rate)
        self.mode = mode
        self.channels = list(channels or [])
        self.n_channels = max(1, len(self.channels) if channels else 2)
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
            "Time saved after the pulse settled.\n\n"
            "The capture is released once this much has arrived after "
            "the settled sample.  The end confirmation runs at least "
            "this long, and the channel cannot trigger again within it.  "
            "A pulse arriving inside it is a pileup.")
        form.addRow("Pre-pulse time (ms):", self.pre_pulse_spin)
        form.addRow("Post-pulse time (ms):", self.post_pulse_spin)

        self.coincidence_spin = QtWidgets.QDoubleSpinBox()
        self.coincidence_spin.setRange(0.0, 60_000.0)
        self.coincidence_spin.setDecimals(3)
        self.coincidence_spin.setSpecialValueText("off")
        self.coincidence_spin.setValue(config.coincidence_window_ms)
        self.coincidence_spin.setToolTip(
            "Pulses on any channels that trigger within this of an "
            "event's first trigger are recorded as one event.  The "
            "pulses are stored under their channels either way; the "
            "events index them, and the pulse list can be grouped by "
            "either.\n\n"
            "Off records no coincident events.  The pulse list can still "
            "group a capture by events afterwards, from the trigger "
            "times.\n"
            "In both mode a channel's share of an event is its pair, "
            "whichever of the two streams triggered.")
        form.addRow("Coincidence window (ms):", self.coincidence_spin)
        self.dump_check = QtWidgets.QCheckBox(
            "Save every channel with each event")
        self.dump_check.setChecked(config.dump_all_channels)
        self.dump_check.setToolTip(
            "With each event, also save the same span of every channel "
            "that did not trigger, from both streams in both mode.  "
            "Only a capture can do this: those samples are gone once the "
            "ring buffer moves on.  rfmux record adds the 100G recording "
            "over the same span when it merges.\n\n"
            "With the coincidence window off, each pulse is an event of "
            "its own, unless two channels trigger on the same sample.  "
            "The file grows by the untriggered channels for every event.")
        form.addRow(self.dump_check)

        self.noise_capture_spin = QtWidgets.QDoubleSpinBox()
        self.noise_capture_spin.setRange(0.0, 86_400.0)
        self.noise_capture_spin.setDecimals(3)
        self.noise_capture_spin.setSpecialValueText("off")
        self.noise_capture_spin.setValue(config.noise_capture_interval_s)
        self.noise_capture_spin.setToolTip(
            "Take noise samples: every channel over one window, at random "
            "moments, whether or not a pulse is present, for the "
            "statistics of the noise.  Each is an event tagged as a noise "
            "sample; a pulse that happens to fall inside one is listed "
            "with it.\n\n"
            "The waits between samples are normally distributed about this "
            f"many seconds, {NoiseSampler.JITTER:.0%} of it wide.  A sample "
            "is as long as a typical pulse record: the median of the "
            f"latest {NoiseSampler.RECORDS_KEPT} saved.  Until "
            f"{NoiseSampler.MIN_RECORDS} records have been saved it is the "
            "pre-pulse time plus the max pulse plus the post-pulse time.")
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
            "The record the noise level (sigma) is fitted from, and the "
            "span the rolling baseline median covers.  It must be long "
            "compared with any pulse and with the 1/f knee, so it is "
            "seconds whatever the pulse length.  Below "
            f"{PulseCaptureConfig.MIN_WINDOW_MS / 1e3:g} s the baseline is "
            "refreshed often enough to fall behind the stream at many "
            "channels.\n"
            "Robust estimators tolerate pulses in the window.  0 derives "
            f"it as {PulseCaptureConfig.NOISE_TRAIN_PULSES}× the max pulse.")
        form.addRow("1/f window (ms):", self.window_spin)
        self.noise_label = QtWidgets.QLabel()
        self.noise_label.setToolTip(
            "The window as the capture will use it at this rate: the "
            "record is memory-bounded on the PFB stream and floored "
            "against the ring buffer.")
        form.addRow("Window at this rate:", self.noise_label)

        # Settings for channels not in the table travel through as they
        # came, so opening the dialog never drops them.
        self._other_channels = {k: dict(v) for k, v
                                in config.per_channel.items()
                                if k not in self.channels}
        self._bad_cells: list = []
        self.channel_table = None
        if self.channels:
            self.channel_table = self._channel_table(config)
            form.addRow(self.channel_table)

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
            "Consecutive samples that must clear the threshold before a "
            "capture starts.  auto keeps accidental triggers under "
            "1/min per channel at this stream rate.\n"
            "How much evidence one sample is depends entirely on the "
            "rate: at 5σ noise alone crosses ~2.5 times per HOUR at "
            "596 Hz but ~2.8 times per SECOND on the PFB stream.  "
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

        self.min_end_spin = QtWidgets.QSpinBox()
        self.min_end_spin.setRange(1, 100_000)
        self.min_end_spin.setValue(config.min_end_samples)
        self.min_end_spin.setToolTip(
            "Floor under the end-confirmation count.\n\n"
            "The end is decided by a leaky bucket: it fills by one for "
            "each sample with BOTH quadratures inside the end band, "
            "leaks by one for each sample outside it, and the capture "
            "ends when it exceeds the largest of this floor, "
            f"{END_CONFIRM_FRACTION:.0%} of the pulse's own length above "
            "threshold, and the post-pulse time.  The leak is what "
            "lets an isolated noisy sample pass without restarting the "
            "count.\n\n"
            "For a short pulse the floor is what ends it, so this sets "
            "how long after the pulse settles the capture is released.  "
            "The record itself ends the post-pulse time after the pulse "
            "settled.  It is a "
            "sample count: the same number is 17 ms at 596 Hz and 4 µs "
            "on the PFB stream."
            "\n\nAlso how far back the pileup test looks for the pulse's "
            "own recent level: a rise of threshold sigma over this many "
            "samples, after decay evidence, splits the capture.")
        adv.addRow("End confirmation floor (samples):", self.min_end_spin)

        self.min_pulse_spin = QtWidgets.QDoubleSpinBox()
        self.min_pulse_spin.setRange(0.0, 10_000.0)
        self.min_pulse_spin.setDecimals(3)
        self.min_pulse_spin.setValue(config.min_pulse_ms)
        self.min_pulse_spin.setToolTip(
            "Pulses shorter than this, trigger to settled, are discarded "
            "as glitches (0 = keep everything)")
        adv.addRow("Min pulse (ms):", self.min_pulse_spin)

        self.pileup_check = QtWidgets.QCheckBox(
            "Split piled-up events (edge re-trigger)")
        self.pileup_check.setToolTip(
            "A fresh edge arriving while the current pulse is decaying "
            "splits the capture into separate events.  Uses the same "
            "edge detector as the trigger.")
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
                "No df calibration for these channels.  Run a multisweep "
                "and Bias KIDs (or bias_kids headlessly), then reopen "
                "these settings.")
        self.basis_combo.setToolTip(
            "What the threshold is applied to.\n\n"
            "A pulse moves the resonance frequency, so it lies along one "
            "direction in the IQ plane — set by the bias point and cable "
            "delay, and unrelated to the I and Q axes.  Testing the raw "
            "quadratures therefore tests an arbitrary basis: at 45 "
            "degrees each one sees the pulse divided by root two while "
            "carrying the full noise.  Rotating first puts the signal in "
            "one axis.\n\n"
            "Needs a df calibration from bias_kids.  Channels without "
            "one cannot be rotated and stay on the quadratures.\n\n"
            "This makes the calibration matter for detection, not just "
            "for labelling an axis: a wrong one costs sensitivity.  It "
            "also sets the units the capture is stored in — hertz once "
            "rotated, volts otherwise.")
        adv.addRow("Trigger basis:", self.basis_combo)

        # What each primary knob drives, at the actual stream rate —
        # the derivations live in PulseCaptureConfig, this only renders
        # them.
        self.pulse_derived_label = QtWidgets.QLabel()
        self.pulse_derived_label.setWordWrap(True)
        self.pulse_derived_label.setToolTip(
            "The ring buffer, hard stop and edge lookback follow the max "
            "pulse; the training record and the baseline median follow "
            "the 1/f window, floored against the ring so the median never "
            "runs in a span a pulse could dominate.")
        form.addRow("Time scales:", self.pulse_derived_label)

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

        for w in (self.threshold_spin, self.end_spin, self.pre_pulse_spin,
                  self.post_pulse_spin, self.coincidence_spin,
                  self.noise_capture_spin,
                  self.min_pulse_spin, self.max_pulse_spin, self.window_spin,
                  self.trigger_spin, self.min_end_spin):
            w.valueChanged.connect(self._update_dependent_values)
        self.pileup_check.toggled.connect(self._update_dependent_values)
        self.dump_check.toggled.connect(self._update_dependent_values)
        if self.channel_table is not None:
            self.channel_table.itemChanged.connect(
                self._update_dependent_values)
        adv_box.toggled.emit(False)
        self._update_dependent_values()

    _SIGMA_COLUMNS = ((2, "threshold_sigma"), (3, "end_sigma"))

    def _channel_table(self, config: PulseCaptureConfig):
        table = QtWidgets.QTableWidget(len(self.channels), 4)
        table.setHorizontalHeaderLabels(
            ["Channel", "Trigger", "Threshold σ", "End σ"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch)
        # Up to six rows before it scrolls.
        table.setFixedHeight(
            table.horizontalHeader().sizeHint().height()
            + table.verticalHeader().defaultSectionSize()
            * min(len(self.channels), 6) + 2 * table.frameWidth())
        table.setToolTip(
            "Each channel's own trigger settings.\n\n"
            "Trigger: unchecked records the channel with every event and "
            "noise sample, without triggering on it.\n"
            "Threshold σ and End σ: blank takes the values above.")
        for row, key in enumerate(self.channels):
            setting = config.per_channel.get(key, {})
            item = QtWidgets.QTableWidgetItem(channel_arg(key))
            item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            table.setItem(row, 0, item)
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled
                          | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                QtCore.Qt.CheckState.Unchecked
                if setting.get("trigger", True) is False
                else QtCore.Qt.CheckState.Checked)
            table.setItem(row, 1, item)
            for col, name in self._SIGMA_COLUMNS:
                value = setting.get(name)
                table.setItem(row, col, QtWidgets.QTableWidgetItem(
                    "" if value is None else f"{value:g}"))
        return table

    def _per_channel(self) -> dict:
        """The table's settings, only those that differ from the
        capture's; unreadable cells are noted in ``_bad_cells``."""
        out = {k: dict(v) for k, v in self._other_channels.items()}
        self._bad_cells = []
        table = self.channel_table
        for row, key in enumerate(self.channels if table else []):
            setting = {}
            if (table.item(row, 1).checkState()
                    == QtCore.Qt.CheckState.Unchecked):
                setting["trigger"] = False
            for col, name in self._SIGMA_COLUMNS:
                text = table.item(row, col).text().strip()
                if not text:
                    continue
                try:
                    setting[name] = float(text)
                except ValueError:
                    self._bad_cells.append(
                        f"{describe(key)}: {text!r} is not a number.")
            if setting:
                out[key] = setting
        return out

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
            per_channel=self._per_channel(),
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

            issues = cfg.validate(self.sample_rate) + [
                ("error", message) for message in self._bad_cells]
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
