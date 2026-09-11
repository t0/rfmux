"""The dialog behind a bare ``rfmux record``: every option of the
command on one page, remembered per user, with the fastrxd check and
its start command in view."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from pathlib import Path
from typing import List, Optional

from PyQt6 import QtCore, QtWidgets

from ..algorithms.measurement.record_streams import (
    biased_channels, calibrated, fastrx_bytes_per_s, latest_bias_export)
from ..core.transferfunctions import decimation_to_sampling
from ..pulse_capture.capture_session import PulseCaptureConfig
from ..core.channels import parse_channel_spec, parse_module_channels
from .periscope.pulse_capture_settings_dialog import PulseCaptureSettingsForm
from .periscope.settings import APPLICATION, ORGANIZATION

_KEY = "record/"
_SHOW = ("periscope", "overlay", "none")


#: Mb/s at and above which an interface is a channel-stream (100G) one.
_FAST_MBPS = 100_000


def interface_speeds() -> dict:
    """{interface: negotiated Mb/s} for the host's interfaces, without
    loopback; None for one without a link or a reported speed."""
    speeds = {}
    try:
        names = sorted(n for n in os.listdir("/sys/class/net") if n != "lo")
    except OSError:
        return speeds
    for name in names:
        try:
            speed = int(Path("/sys/class/net", name, "speed").read_text())
        except (OSError, ValueError):
            speed = None
        speeds[name] = speed if speed and speed > 0 else None
    return speeds


def _label(name: str, speed) -> str:
    if speed is None:
        return f"{name} (no link)"
    return (f"{name} ({speed / 1000:g} Gb/s)" if speed >= 1000
            else f"{name} ({speed} Mb/s)")


def _combo_value(combo: QtWidgets.QComboBox) -> str:
    """The interface a combo names: the chosen item's, or typed text."""
    idx = combo.currentIndex()
    text = combo.currentText().strip()
    if idx >= 0 and combo.itemText(idx) == text:
        return combo.itemData(idx)
    return text


def _select(combo: QtWidgets.QComboBox, value: str) -> None:
    idx = combo.findData(value)
    if idx >= 0:
        combo.setCurrentIndex(idx)
    else:
        combo.setEditText(value)


def _fastrx():
    """The fastrx module, or None in a build without it."""
    try:
        from .. import fastrx
        return fastrx
    except ImportError:
        return None


class RecordDialog(QtWidgets.QDialog):

    def __init__(self, parent=None, *,
                 settings: Optional[QtCore.QSettings] = None):
        super().__init__(parent)
        self.setWindowTitle("rfmux record")
        self.settings = settings or QtCore.QSettings(ORGANIZATION, APPLICATION)
        outer = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        outer.addWidget(self.tabs)
        run_page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(run_page)
        add = form.addRow
        self.tabs.addTab(run_page, "Run")

        # ── Board ────────────────────────────────────────────────
        self.serial_edit = QtWidgets.QLineEdit()
        self.serial_edit.setPlaceholderText("0156")
        self.hostname_edit = QtWidgets.QLineEdit()
        self.hostname_edit.setPlaceholderText("only when not <serial>.local")
        self.modules_edit = QtWidgets.QLineEdit()
        self.modules_edit.setPlaceholderText(
            "1, or 2,3 for one RF line over several modules")
        add("CRS serial:", self.serial_edit)
        add("Hostname:", self.hostname_edit)
        add("Modules:", self.modules_edit)

        # ── Session ──────────────────────────────────────────────
        self.rb_existing = QtWidgets.QRadioButton("Existing folder")
        self.session_path_edit = QtWidgets.QLineEdit()
        self.rb_new = QtWidgets.QRadioButton("New session under")
        self.session_dir_edit = QtWidgets.QLineEdit()
        # Each pair is its own exclusive group; without one, Qt groups
        # every auto-exclusive radio under the page together.
        self.session_group = QtWidgets.QButtonGroup(self)
        self.session_group.addButton(self.rb_existing)
        self.session_group.addButton(self.rb_new)
        add("Session:", self._row(self.rb_existing, self.session_path_edit,
                                  self._browse(self.session_path_edit)))
        add("", self._row(self.rb_new, self.session_dir_edit,
                          self._browse(self.session_dir_edit)))

        # ── Channels and duration ────────────────────────────────
        self.rb_bias = QtWidgets.QRadioButton(
            "Biased channels of the session's newest bias export per module")
        self.bias_label = QtWidgets.QLabel()
        self.rb_ranges = QtWidgets.QRadioButton("Ranges")
        self.channels_edit = QtWidgets.QLineEdit()
        self.channels_group = QtWidgets.QButtonGroup(self)
        self.channels_group.addButton(self.rb_bias)
        self.channels_group.addButton(self.rb_ranges)
        self.channels_edit.setPlaceholderText(
            "1-88 or 1,5-10 on every module; 2:1-114,3:1-96 per module")
        add("Channels:", self.rb_bias)
        add("", self.bias_label)
        add("", self._row(self.rb_ranges, self.channels_edit))
        self.duration_spin = QtWidgets.QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 86400.0)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setToolTip("Seconds to record, after the "
                                      "capture's noise training")
        add("Duration:", self.duration_spin)

        # ── Products ─────────────────────────────────────────────
        self.capture_check = QtWidgets.QCheckBox(
            "Pulse capture of the slow stream")
        self.parser_check = QtWidgets.QCheckBox("Parser dirfile")
        self.parser_iface_combo = QtWidgets.QComboBox()
        self.parser_iface_combo.setEditable(True)
        self.parser_iface_combo.setToolTip(
            "1G interface for the parser; auto finds it from the board "
            "address")
        self.fastrx_check = QtWidgets.QCheckBox("fastrx recording")
        self.fastrx_iface_combo = QtWidgets.QComboBox()
        self.fastrx_iface_combo.setEditable(True)
        self.fastrx_iface_combo.setToolTip("100G interface fastrxd runs on")
        self.fastrx_status = QtWidgets.QLabel()
        self.fastrx_status.setWordWrap(True)
        self.fastrx_status.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.copy_btn = QtWidgets.QPushButton("Copy command")
        self.recheck_btn = QtWidgets.QPushButton("Check again")
        self.disk_label = QtWidgets.QLabel()
        self.merge_check = QtWidgets.QCheckBox(
            "Merge the recording into the pulse file after the run")
        self.show_combo = QtWidgets.QComboBox()
        self.show_combo.addItems(["Periscope in review mode",
                                  "the overlay viewer", "nothing"])
        add("Products:", self.capture_check)
        add("", self._row(self.parser_check, QtWidgets.QLabel("1G interface"),
                          self.parser_iface_combo))
        add("", self._row(self.fastrx_check,
                          QtWidgets.QLabel("100G interface"),
                          self.fastrx_iface_combo))
        add("", self.fastrx_status)
        add("", self._row(self.copy_btn, self.recheck_btn))
        add("", self.disk_label)
        add("", self.merge_check)
        add("After the run:", self.show_combo)

        # ── Pulse capture settings, on their own tab ─────────────
        capture_page = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(capture_page)
        self.capture_form = PulseCaptureSettingsForm(
            capture_page, config=self._saved_config(),
            sample_rate=decimation_to_sampling(6), mode="slow")
        box.addWidget(self.capture_form)
        stage_note = QtWidgets.QLabel(
            "Sample counts and time scales above are for decimation "
            "stage 6; the run derives them from the board's stage.")
        stage_note.setWordWrap(True)
        box.addWidget(stage_note)
        box.addStretch(1)
        self.tabs.addTab(capture_page, "Pulse capture")

        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.record_btn = self.buttons.addButton(
            "Record", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        outer.addWidget(self.buttons)

        self._load()
        for w in (self.serial_edit, self.session_path_edit,
                  self.session_dir_edit, self.channels_edit):
            w.textChanged.connect(self._refresh)
        for w in (self.rb_existing, self.rb_new, self.rb_bias, self.rb_ranges,
                  self.fastrx_check, self.parser_check, self.capture_check):
            w.toggled.connect(self._refresh)
        self.modules_edit.textChanged.connect(self._refresh)
        self.duration_spin.valueChanged.connect(self._refresh)
        self.fastrx_iface_combo.currentTextChanged.connect(self._refresh)
        self.capture_form.updated.connect(self._refresh)
        self.recheck_btn.clicked.connect(self._refresh)
        self.copy_btn.clicked.connect(
            lambda: QtWidgets.QApplication.clipboard().setText(
                self._start_command()))
        self._refresh()

    # ── Layout helpers ───────────────────────────────────────────

    @staticmethod
    def _row(*widgets) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        for x in widgets:
            lay.addWidget(x, 1 if isinstance(x, (QtWidgets.QLineEdit,
                                                  QtWidgets.QComboBox)) else 0)
        return w

    def _browse(self, edit: QtWidgets.QLineEdit) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton("Browse…")

        def pick():
            dlg = QtWidgets.QFileDialog(self, "Choose a folder", edit.text())
            dlg.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
            dlg.fileSelected.connect(edit.setText)
            dlg.open()
        btn.clicked.connect(pick)
        return btn

    # ── State ────────────────────────────────────────────────────

    def _session_folder(self) -> Optional[Path]:
        """The folder the run joins, if it exists already."""
        if self.rb_existing.isChecked():
            p = Path(self.session_path_edit.text()).expanduser()
            return p if p.is_dir() else None
        return None

    def _modules(self) -> Optional[List[int]]:
        try:
            return parse_channel_spec(self.modules_edit.text(), name="module",
                                      max_value=8, wildcard=False)
        except ValueError:
            return None

    def _channels(self):
        """(channels, note): the ``{module: channels}`` the options
        resolve to, or None with the reason.  Reading the bias exports
        costs a tenth of a second, so the answer is kept until an input
        changes."""
        key = (self.rb_ranges.isChecked(), self.channels_edit.text(),
               self._session_folder(), self.modules_edit.text())
        cached = getattr(self, "_channels_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        result = self._resolve_channels()
        self._channels_cache = (key, result)
        return result

    def _resolve_channels(self):
        modules = self._modules()
        if self.rb_ranges.isChecked():
            text = self.channels_edit.text()
            try:
                if ":" in text:
                    wanted = parse_module_channels(text, max_module=8,
                                                   max_channel=1024)
                else:
                    if modules is None:
                        return None, "name the modules, like 1 or 2,3"
                    chans = parse_channel_spec(text, max_value=1024,
                                               wildcard=False)
                    wanted = {m: chans for m in modules}
            except ValueError as e:
                return None, str(e)
            n = sum(len(c) for c in wanted.values())
            return wanted, (f"{n} channels on module(s) "
                            f"{', '.join(str(m) for m in wanted)}")
        if modules is None:
            return None, "name the modules, like 1 or 2,3"
        folder = self._session_folder()
        if folder is None:
            return None, "an existing session folder is needed"
        wanted, notes = {}, []
        for module in modules:
            bias = latest_bias_export(folder, module)
            if bias is None:
                return None, (f"no bias export for module {module} in the "
                              "session folder")
            chans, rows = biased_channels(bias)
            if not chans:
                return None, f"{bias.name} biased no channels"
            wanted[module] = chans
            notes.append(f"{bias.name}: {len(chans)} channels, "
                         f"{calibrated(rows)} calibrated")
        return wanted, "\n".join(notes)

    def _fill_interfaces(self, running) -> None:
        """The parser's list: interfaces under 100 Gb/s; fastrx's: the
        running daemons, then the 100 Gb/s interfaces, the one of them
        filled in when nothing was chosen.  Each shows its rate."""
        speeds = interface_speeds()
        slow = [n for n, v in speeds.items() if v is None or v < _FAST_MBPS]
        fast = running + [n for n, v in speeds.items()
                          if v is not None and v >= _FAST_MBPS
                          and n not in running]
        for combo, names, extra in (
                (self.parser_iface_combo, slow, [("auto", "auto")]),
                (self.fastrx_iface_combo, fast, [])):
            current = _combo_value(combo)
            combo.blockSignals(True)
            combo.clear()
            for text, value in extra:
                combo.addItem(text, value)
            for n in names:
                combo.addItem(_label(n, speeds.get(n)), n)
            if not current and combo is self.fastrx_iface_combo \
                    and len(fast) == 1:
                current = fast[0]
            _select(combo, current)
            combo.blockSignals(False)

    def _start_command(self) -> str:
        fx = _fastrx()
        iface = _combo_value(self.fastrx_iface_combo)
        return fx.start_command(iface) if fx and iface else ""

    def _refresh(self, *_) -> None:
        chans, note = self._channels()
        self.bias_label.setText(note if self.rb_bias.isChecked() else "")
        problems = []
        if not self.serial_edit.text().strip():
            problems.append("a CRS serial is needed")
        if chans is None:
            problems.append(note)

        fx = _fastrx()
        running = fx.running_interfaces() if fx else []
        if self.fastrx_iface_combo.count() == 0 or \
                self.sender() is self.recheck_btn:
            self._fill_interfaces(running)
        iface = _combo_value(self.fastrx_iface_combo)
        for w in (self.fastrx_iface_combo, self.fastrx_status, self.copy_btn,
                  self.recheck_btn, self.disk_label, self.merge_check):
            w.setEnabled(self.fastrx_check.isChecked())
        self.parser_iface_combo.setEnabled(self.parser_check.isChecked())
        if self.fastrx_check.isChecked():
            if fx is None:
                self.fastrx_status.setText(
                    "this rfmux build does not include fastrx")
                problems.append("fastrx is not built")
            elif iface in running:
                self.fastrx_status.setText(f"● fastrxd is running on {iface}")
            else:
                self.fastrx_status.setText(
                    f"✗ fastrxd is not running on {iface}. Start it in a "
                    f"terminal, then Check again:\n{self._start_command()}"
                    if iface else "✗ name the 100G interface fastrxd runs on")
                problems.append("fastrxd is not running")
            folder = self._session_folder() or \
                Path(self.session_dir_edit.text() or ".").expanduser()
            if chans and folder.is_dir():
                need = self.duration_spin.value() * fastrx_bytes_per_s(
                    max(c for chs in chans.values() for c in chs))
                free = shutil.disk_usage(folder).free
                self.disk_label.setText(
                    f"disk: {free / 1e9:.0f} GB free in {folder}, about "
                    f"{need / 1e9:.0f} GB needed")
                if free < need:
                    problems.append("not enough disk for the recording")
            else:
                self.disk_label.setText("")
        else:
            self.fastrx_status.setText("")
            self.disk_label.setText("")
        if not self.capture_form.valid:
            problems.append("the pulse capture settings do not validate")
        self.status_label.setText("\n".join(problems))
        self.record_btn.setEnabled(not problems)

    # ── Options ──────────────────────────────────────────────────

    def get_options(self) -> dict:
        """Keyword arguments for rfmux.tools.record._run."""
        existing = self.rb_existing.isChecked()
        parser_iface = _combo_value(self.parser_iface_combo)
        return {
            "serial": self.serial_edit.text().strip(),
            "hostname": self.hostname_edit.text().strip() or None,
            "modules": self._modules() or [1],
            "channels": (self.channels_edit.text().strip()
                         if self.rb_ranges.isChecked() else None),
            "duration": self.duration_spin.value(),
            "session": (self.session_path_edit.text().strip() or None
                        if existing else None),
            "session_dir": self.session_dir_edit.text().strip() or ".",
            "capture": self.capture_check.isChecked(),
            "parser": self.parser_check.isChecked(),
            "fastrx": self.fastrx_check.isChecked(),
            "parser_interface": (None if parser_iface in ("", "auto")
                                 else parser_iface),
            "fastrx_interface": _combo_value(self.fastrx_iface_combo) or None,
            "fastrx_socket": None,
            "merge_fastrx": self.merge_check.isChecked(),
            "show": _SHOW[self.show_combo.currentIndex()],
            "bias": None,
            "config": self.capture_form.get_config(),
        }

    def accept(self) -> None:
        self._save()
        super().accept()

    @classmethod
    def ask(cls, parent=None) -> Optional[dict]:
        """Run the dialog; the options, or None when cancelled."""
        app = QtWidgets.QApplication.instance() or \
            QtWidgets.QApplication([])
        dlg = cls(parent)
        return dlg.get_options() if dlg.exec() else None

    # ── Persistence ──────────────────────────────────────────────

    @staticmethod
    def newest_session(base: Path) -> Optional[Path]:
        """The newest session folder under *base*, by the stamp in its
        name, or None."""
        found = sorted(p for p in Path(base).expanduser().glob("session_*")
                       if p.is_dir() and (p / "session_metadata.json").exists())
        return found[-1] if found else None

    def _saved_config(self) -> PulseCaptureConfig:
        raw = self.settings.value(_KEY + "capture_config", "")
        try:
            return PulseCaptureConfig(**json.loads(raw)) if raw else \
                PulseCaptureConfig()
        except (TypeError, ValueError):
            return PulseCaptureConfig()

    def _load(self) -> None:
        s = self.settings
        v = lambda key, default: s.value(_KEY + key, default)
        self.serial_edit.setText(str(v("serial", "")))
        self.hostname_edit.setText(str(v("hostname", "")))
        self.modules_edit.setText(str(v("modules", "1")))
        (self.rb_existing if v("session_mode", "existing") == "existing"
         else self.rb_new).setChecked(True)
        self.session_dir_edit.setText(str(v("session_dir", str(Path.cwd()))))
        saved = str(v("session_path", ""))
        if not (saved and Path(saved).expanduser().is_dir()):
            # The newest session under the default path, when there is one.
            newest = self.newest_session(Path(self.session_dir_edit.text()))
            saved = str(newest) if newest else ""
        self.session_path_edit.setText(saved)
        (self.rb_bias if v("channels_mode", "bias") == "bias"
         else self.rb_ranges).setChecked(True)
        self.channels_edit.setText(str(v("channels", "")))
        self.duration_spin.setValue(float(v("duration", 20.0)))
        self.capture_check.setChecked(v("capture", "true") in (True, "true"))
        self.parser_check.setChecked(v("parser", "true") in (True, "true"))
        self.fastrx_check.setChecked(v("fastrx", "true") in (True, "true"))
        _select(self.parser_iface_combo, str(v("parser_interface", "auto")))
        _select(self.fastrx_iface_combo, str(v("fastrx_interface", "")))
        self.merge_check.setChecked(v("merge_fastrx", "true") in (True, "true"))
        show = str(v("show", "periscope"))
        self.show_combo.setCurrentIndex(
            _SHOW.index(show) if show in _SHOW else 0)

    def _save(self) -> None:
        s = self.settings
        o = self.get_options()
        for key, value in (
                ("serial", o["serial"]), ("hostname", o["hostname"] or ""),
                ("modules", self.modules_edit.text().strip()),
                ("session_mode", "existing" if self.rb_existing.isChecked()
                 else "new"),
                ("session_path", self.session_path_edit.text()),
                ("session_dir", o["session_dir"]),
                ("channels_mode", "ranges" if self.rb_ranges.isChecked()
                 else "bias"),
                ("channels", self.channels_edit.text()),
                ("duration", o["duration"]),
                ("capture", "true" if o["capture"] else "false"),
                ("parser", "true" if o["parser"] else "false"),
                ("fastrx", "true" if o["fastrx"] else "false"),
                ("parser_interface", o["parser_interface"] or "auto"),
                ("fastrx_interface", o["fastrx_interface"] or ""),
                ("merge_fastrx", "true" if o["merge_fastrx"] else "false"),
                ("show", o["show"]),
                ("capture_config",
                 json.dumps(dataclasses.asdict(o["config"])))):
            s.setValue(_KEY + key, value)
        s.sync()
