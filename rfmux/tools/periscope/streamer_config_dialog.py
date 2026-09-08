"""
Streamer Configuration dialog.

A thin GUI over :mod:`rfmux.algorithms.measurement.streamer_config`:
every derived number and every rule shown here comes from that module's
``describe()`` / ``validate()`` — the dialog just renders them live and
refuses OK while any error-tier issue exists.  Applying runs
``apply_streamer_config`` in a worker thread.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, List, Optional, Tuple

from PyQt6 import QtCore, QtWidgets

from .utils import apply_issue_banner
from PyQt6.QtCore import pyqtSignal

from ...algorithms.measurement.channel_selection import parse_channel_spec
from ...algorithms.measurement.streamer_config import (
    LINK_MBPS,
    StreamerConfig,
    describe,
    read_streamer_config,
    apply_streamer_config,
    validate,
)


class _CoroutineThread(QtCore.QThread):
    """Run one coroutine on its own event loop off the GUI thread."""

    success = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, coro: Awaitable[dict], parent=None):
        super().__init__(parent)
        self._coro = coro

    def run(self):
        loop = asyncio.new_event_loop()
        try:
            self.success.emit(loop.run_until_complete(self._coro))
        except Exception as e:
            self.error.emit(str(e))
        finally:
            loop.close()


class ApplyStreamerConfigTask(_CoroutineThread):
    """Apply a StreamerConfig off the GUI thread."""

    def __init__(self, crs, cfg: StreamerConfig, parent=None):
        super().__init__(apply_streamer_config(crs, cfg), parent)


class StreamerConfigDialog(QtWidgets.QDialog):
    """Configure the slow/fast streamers with live combination math."""

    def __init__(self, parent=None, *, crs=None,
                 current_dec: Optional[int] = None,
                 current_short: Optional[bool] = None,
                 module: int = 1):
        super().__init__(parent)
        self.setWindowTitle("Streamer Configuration")
        self.setModal(True)
        self.crs = crs
        self._updating = False
        self.fetcher: Optional[_CoroutineThread] = None

        self._setup_ui(current_dec, current_short, module)
        self._connect_signals()
        self._update_dependent_values()

        if crs is not None:
            self.fetcher = _CoroutineThread(
                read_streamer_config(crs, pfb_module=module), parent=self)
            self.fetcher.success.connect(self._on_state_ready)
            self.fetcher.error.connect(lambda _: self._on_state_ready({}))
            self.fetcher.start()

    # ── UI ────────────────────────────────────────────────────────

    def _setup_ui(self, current_dec, current_short, module):
        form = QtWidgets.QFormLayout(self)

        state = []
        if current_dec is not None:
            state.append(f"dec {current_dec}")
        if current_short is not None:
            state.append("short (128 ch)" if current_short
                         else "long (1024 ch)")
        self.current_label = QtWidgets.QLabel(
            ", ".join(state) if state else "querying…")
        form.addRow("Current stream:", self.current_label)

        self.dec_spin = QtWidgets.QSpinBox()
        self.dec_spin.setRange(0, 6)
        self.dec_spin.setValue(current_dec if current_dec is not None else 6)
        self.dec_spin.setToolTip(
            "Slow-stream decimation stage: rate = 625 MHz/256/64/2^N")
        form.addRow("Decimation stage:", self.dec_spin)

        self.short_check = QtWidgets.QCheckBox("128-channel short packets")
        self.short_check.setChecked(bool(current_short))
        self.short_check.setToolTip(
            "Short packets carry channels 1-128 at ~1/8 the bandwidth. "
            "Locked on at the stages where long packets would exceed "
            "the link.")
        form.addRow("Packet format:", self.short_check)

        self.modules_edit = QtWidgets.QLineEdit(str(module))
        self.modules_edit.setToolTip(
            "Modules to stream, e.g. \"1,2\" or \"1-4\"; blank or \"all\" = "
            "every module. Starts at the current module because below "
            "stage 5 streaming is validated one module at a time.")
        form.addRow("Modules:", self.modules_edit)

        self.pfb_check = QtWidgets.QCheckBox(
            "Enable fast (PFB) streamer — ~2.44 MHz")
        form.addRow(self.pfb_check)

        self.pfb_group = QtWidgets.QWidget()
        pfb_form = QtWidgets.QFormLayout(self.pfb_group)
        pfb_form.setContentsMargins(20, 0, 0, 0)
        self.pfb_channels_edit = QtWidgets.QLineEdit("1")
        self.pfb_channels_edit.setToolTip(
            "Up to 4 channels of one module, e.g. \"1,2\" or \"1-4\"")
        pfb_form.addRow("PFB channels:", self.pfb_channels_edit)
        self.pfb_module_spin = QtWidgets.QSpinBox()
        self.pfb_module_spin.setRange(1, 8)
        self.pfb_module_spin.setValue(module)
        pfb_form.addRow("PFB module:", self.pfb_module_spin)
        self.pfb_group.setVisible(False)
        form.addRow(self.pfb_group)

        self.rate_label = QtWidgets.QLabel()
        form.addRow("Slow sample rate:", self.rate_label)
        self.nyquist_label = QtWidgets.QLabel()
        form.addRow("Nyquist:", self.nyquist_label)
        self.width_label = QtWidgets.QLabel()
        form.addRow("Channels/module:", self.width_label)
        self.bandwidth_label = QtWidgets.QLabel()
        form.addRow("Link budget:", self.bandwidth_label)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        form.addRow(self.status_label)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        form.addRow(self.buttons)

        self.resize(520, 480)

    def _connect_signals(self):
        self.dec_spin.valueChanged.connect(self._update_dependent_values)
        self.short_check.toggled.connect(self._update_dependent_values)
        self.modules_edit.textChanged.connect(self._update_dependent_values)
        self.pfb_check.toggled.connect(self._on_pfb_toggled)
        self.pfb_channels_edit.textChanged.connect(
            self._update_dependent_values)
        self.pfb_module_spin.valueChanged.connect(
            self._update_dependent_values)

    def _on_pfb_toggled(self, checked: bool):
        self.pfb_group.setVisible(checked)
        self._update_dependent_values()

    def _on_state_ready(self, state: dict):
        parts = []
        if state.get("dec_stage") is not None:
            parts.append(f"dec {state['dec_stage']}")
        if state.get("pfb_channels"):
            parts.append(f"PFB on: ch {state['pfb_channels']}")
        current = self.current_label.text()
        if current == "querying…":
            self.current_label.setText(", ".join(parts) or "unknown")
        elif parts:
            self.current_label.setText(f"{current}  ({', '.join(parts)})")

    # ── Config assembly / live math ───────────────────────────────

    def _read_fields(self) -> Tuple[StreamerConfig, List[Tuple[str, str]]]:
        """The configuration the fields describe, plus an error-tier issue
        for each text field that does not parse.

        The dialog states the whole streamer configuration: an unchecked
        PFB box means no PFB streamer (``[]``, which apply disables).  A
        PFB field that does not parse leaves ``pfb_channels`` None while
        the error keeps OK disabled.
        """
        issues: List[Tuple[str, str]] = []
        modules = None
        if self.modules_edit.text().strip():
            try:
                modules = parse_channel_spec(self.modules_edit.text())
            except ValueError:
                issues.append(("error",
                               "Modules must be a list like \"1,2\" or "
                               "\"1-4\" (blank = all)."))
        pfb_channels: Optional[List[int]] = []
        if self.pfb_check.isChecked():
            pfb_channels = None
            try:
                pfb_channels = parse_channel_spec(
                    self.pfb_channels_edit.text())
                if pfb_channels is None:
                    raise ValueError("list up to 4 channels explicitly.")
            except ValueError as e:
                issues.append(("error", f"PFB channels: {e}"))
        cfg = StreamerConfig(
            dec_stage=int(self.dec_spin.value()),
            short_packets=self.short_check.isChecked(),
            modules=modules,
            pfb_channels=pfb_channels,
            pfb_module=int(self.pfb_module_spin.value()),
        )
        return cfg, issues

    def get_config(self) -> StreamerConfig:
        return self._read_fields()[0]

    @staticmethod
    def _long_packets_refused(stage: int) -> bool:
        # With zero modules the link budget is zero, so the packet-format
        # rule is the only error validate() can raise: the stage threshold
        # lives in streamer_config alone.
        probe = StreamerConfig(dec_stage=stage, short_packets=False,
                               modules=[])
        return any(sev == "error" for sev, _ in validate(probe))

    def _update_dependent_values(self):
        if self._updating:
            return
        self._updating = True
        try:
            # Where long packets are refused, short packets are forced + locked
            if self._long_packets_refused(self.dec_spin.value()):
                self.short_check.setChecked(True)
                self.short_check.setEnabled(False)
            else:
                self.short_check.setEnabled(True)

            cfg, issues = self._read_fields()
            d = describe(cfg)
            fs = d["sample_rate_hz"]
            self.rate_label.setText(
                f"{fs:,.1f} Hz" if fs < 1e5 else f"{fs/1e3:,.1f} kHz")
            self.nyquist_label.setText(f"{d['nyquist_hz']:,.1f} Hz")
            self.width_label.setText(str(d["channels_per_module"]))
            bw = (f"slow {d['slow_mbps']:.0f} Mbps"
                  + (f" + PFB {d['pfb_mbps']:.0f} Mbps" if d["pfb_mbps"]
                     else "")
                  + f" = {d['total_mbps']:.0f} / {LINK_MBPS:.0f} Mbps")
            self.bandwidth_label.setText(bw)

            issues += validate(cfg)
            apply_issue_banner(
                self.status_label,
                self.buttons.button(
                    QtWidgets.QDialogButtonBox.StandardButton.Ok),
                issues)
        finally:
            self._updating = False
