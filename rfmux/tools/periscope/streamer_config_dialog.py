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
from typing import Optional

from PyQt6 import QtCore, QtWidgets

from .utils import apply_issue_banner
from PyQt6.QtCore import pyqtSignal

from ...algorithms.measurement.streamer_config import (
    StreamerConfig,
    describe,
    read_streamer_config,
    apply_streamer_config,
    validate,
)


class StreamerStateFetcher(QtCore.QThread):
    """Async read of the current board streamer state (non-blocking)."""

    state_ready = pyqtSignal(dict)

    def __init__(self, crs, pfb_module: int = 1, parent=None):
        super().__init__(parent)
        self.crs = crs
        self.pfb_module = pfb_module

    def run(self):
        loop = asyncio.new_event_loop()
        try:
            state = loop.run_until_complete(
                read_streamer_config(self.crs, pfb_module=self.pfb_module))
        except Exception:
            state = {}
        finally:
            loop.close()
        self.state_ready.emit(state)


class ApplyStreamerConfigTask(QtCore.QThread):
    """Apply a StreamerConfig off the GUI thread."""

    success = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, crs, cfg: StreamerConfig, parent=None):
        super().__init__(parent)
        self.crs = crs
        self.cfg = cfg

    def run(self):
        loop = asyncio.new_event_loop()
        try:
            info = loop.run_until_complete(
                apply_streamer_config(self.crs, self.cfg))
            self.success.emit(info)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            loop.close()


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
        self.fetcher: Optional[StreamerStateFetcher] = None

        self._setup_ui(current_dec, current_short, module)
        self._connect_signals()
        self._update_dependent_values()

        if crs is not None:
            self.fetcher = StreamerStateFetcher(crs, pfb_module=module)
            self.fetcher.state_ready.connect(self._on_state_ready)
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
            "Mandatory below stage 3.")
        form.addRow("Packet format:", self.short_check)

        self.modules_edit = QtWidgets.QLineEdit(str(module))
        self.modules_edit.setToolTip(
            "Comma-separated modules to stream (blank = all)")
        form.addRow("Modules:", self.modules_edit)

        self.pfb_check = QtWidgets.QCheckBox(
            "Enable fast (PFB) streamer — ~2.44 MHz")
        form.addRow(self.pfb_check)

        self.pfb_group = QtWidgets.QWidget()
        pfb_form = QtWidgets.QFormLayout(self.pfb_group)
        pfb_form.setContentsMargins(20, 0, 0, 0)
        self.pfb_channels_edit = QtWidgets.QLineEdit("1")
        self.pfb_channels_edit.setToolTip("Up to 4 channels of one module")
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

    def _parse_int_list(self, text: str):
        try:
            return [int(t) for t in text.replace(" ", "").split(",") if t]
        except ValueError:
            return None

    def get_config(self) -> StreamerConfig:
        modules = self._parse_int_list(self.modules_edit.text())
        pfb_channels = None
        if self.pfb_check.isChecked():
            pfb_channels = self._parse_int_list(
                self.pfb_channels_edit.text()) or []
        return StreamerConfig(
            dec_stage=int(self.dec_spin.value()),
            short_packets=self.short_check.isChecked(),
            modules=modules if modules else None,
            pfb_channels=pfb_channels,
            pfb_module=int(self.pfb_module_spin.value()),
        )

    def _update_dependent_values(self):
        if self._updating:
            return
        self._updating = True
        try:
            # Below stage 3, short packets are mandatory — force + lock
            if self.dec_spin.value() < 3:
                self.short_check.setChecked(True)
                self.short_check.setEnabled(False)
            else:
                self.short_check.setEnabled(True)

            cfg = self.get_config()
            d = describe(cfg)
            fs = d["sample_rate_hz"]
            self.rate_label.setText(
                f"{fs:,.1f} Hz" if fs < 1e5 else f"{fs/1e3:,.1f} kHz")
            self.nyquist_label.setText(f"{d['nyquist_hz']:,.1f} Hz")
            self.width_label.setText(str(d["channels_per_module"]))
            bw = (f"slow {d['slow_mbps']:.0f} Mbps"
                  + (f" + PFB {d['pfb_mbps']:.0f} Mbps" if d["pfb_mbps"]
                     else "")
                  + f" = {d['total_mbps']:.0f} / 1000 Mbps")
            self.bandwidth_label.setText(bw)

            issues = validate(cfg)
            if self._parse_int_list(self.modules_edit.text()) is None \
                    and self.modules_edit.text().strip():
                issues.append(("error",
                               "Modules must be a comma-separated list "
                               "of integers."))
            apply_issue_banner(
                self.status_label,
                self.buttons.button(
                    QtWidgets.QDialogButtonBox.StandardButton.Ok),
                issues)
        finally:
            self._updating = False
