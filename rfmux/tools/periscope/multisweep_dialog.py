"""The Multisweep dialog: a view over ``crs.multisweep``'s own arguments.

Nothing here computes what a sweep will do. The amplitude group builds an
:class:`~rfmux.tuning.multisweep_amplitudes.AmplitudeSchedule` through the
constructor each radio names, and the schedule's own ``describe`` and
``validate`` fill the summary line and the status label. What the dialog emits
is the keyword arguments the driver takes, and nothing else.
"""

import inspect
import traceback

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator, QIntValidator

from rfmux.algorithms.measurement.multisweep import multisweep
from rfmux.core.resonators import ResonatorCatalog
from rfmux.tuning import AmplitudeSchedule, store

from .network_analysis_base import NetworkAnalysisDialogBase
from .utils import DEFAULT_AMPLITUDE
from .field_memory import remember_fields

# What the driver does when you say nothing. Read once, at import, so the
# dialog cannot offer a default the library does not have.
DEFAULTS = {
    name: parameter.default
    for name, parameter in inspect.signature(multisweep).parameters.items()
    if parameter.default is not inspect.Parameter.empty
}

#: Severity to icon, worst first — the order the status label picks from.
_SEVERITY = [("error", "✗"), ("warning", "⚠"), ("info", "✓")]
_SEVERITY_COLOR = {"error": "#CC3333", "warning": "#CC8833", "info": "#338833"}


def load_multisweep_container(parent: QtWidgets.QWidget, file_path: str):
    """Read a saved multisweep with ``store.load``, or say why it is not one.

    The same reader a notebook uses, so a file Periscope wrote opens there and
    one written there opens here. What comes back is the container the driver
    returned; the file says what it is through ``store``'s own metadata rather
    than by having its shape inspected.
    """
    try:
        container = store.load(file_path)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(
            parent, "Load Failed", f"Could not read '{file_path}':\n{exc}")
        return None

    blocks = list(container.values()) if isinstance(container, dict) else []
    if not blocks or blocks[0].get("measurement") != "multisweep":
        QtWidgets.QMessageBox.warning(
            parent, "Not a Multisweep",
            f"'{file_path}' does not hold a multisweep.")
        return None
    return container


class MultisweepDialog(NetworkAnalysisDialogBase):
    """Configure one ``crs.multisweep`` call over a :class:`ResonatorCatalog`."""

    def __init__(self, parent: QtWidgets.QWidget = None,
                 catalog: ResonatorCatalog | None = None,
                 dac_scales: dict[int, float] = None,
                 module: int | None = None,
                 initial_params: dict | None = None,
                 load_multisweep: bool = False):
        """
        Args:
            catalog: the array to sweep. A multisweep measures a catalog, so
                this is the dialog's subject: the schedule resolves against it,
                and the summary and validation are about it.
            dac_scales: pre-fetched DAC scales, for the power range in the
                summary and the full-scale label.
            module: this session's module, for the load and custom-frequency
                modes, where there is no catalog yet to read it off.
            initial_params: a previous call's arguments, to seed the fields.
            load_multisweep: offer Import and a Load button rather than a sweep.
        """
        super().__init__(parent, params=initial_params, dac_scales=dac_scales,
                         module=catalog.module if catalog is not None else module)
        self.catalog = catalog
        self.load_multisweep = load_multisweep

        self.use_data_from_file = False
        self.loaded_container = None

        self.setWindowTitle("Multisweep Configuration")
        self.setModal(True)

        self._setup_ui()

        if self.crs_for_dac_scales() is not None and not self.dac_scales:
            self._fetch_dac_scales_for_dialog(self.crs_for_dac_scales())
        self._update_dac_scale_info()
        remember_fields(self, restore=not self.params)
        self._refresh()

    # ── the board, for the DAC scale only ────────────────────────────────────

    def crs_for_dac_scales(self):
        """The board, if a Periscope is above us, so full scale can be shown."""
        widget = self.parent()
        while widget is not None:
            crs = getattr(widget, "crs", None)
            if crs is not None:
                return crs
            widget = widget.parent()
        return None

    def _fetch_dac_scales_for_dialog(self, crs_obj):
        from .tasks import DACScaleFetcher
        self._fetcher = DACScaleFetcher(crs_obj)
        self._fetcher.dac_scales_ready.connect(self._on_dac_scales_ready_dialog)
        self._fetcher.start()

    def _on_dac_scales_ready_dialog(self, scales_dict: dict[int, float]):
        self.dac_scales.update(scales_dict)
        self._update_dac_scale_info()
        self._refresh()

    # ── the UI ───────────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        if self.load_multisweep:
            self.import_button = QtWidgets.QPushButton("Import Sweep File")
            self.import_button.clicked.connect(self._import_file)
            layout.addWidget(self.import_button)

        layout.addWidget(self._sections_group())
        layout.addWidget(self._amplitude_group())
        layout.addWidget(self._parameters_group())
        layout.addWidget(self._output_group())

        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        layout.addLayout(self._buttons())

        for key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(self._start_if_valid)

        self.setMinimumWidth(520)

    def _sections_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Sweep sections")
        form = QtWidgets.QVBoxLayout(group)

        self.sections_info_label = QtWidgets.QLabel()
        self.sections_info_label.setWordWrap(True)
        form.addWidget(self.sections_info_label)

        self.custom_frequencies_cb = QtWidgets.QCheckBox("Custom frequencies")
        self.custom_frequencies_cb.setToolTip(
            "Sweep frequencies you type rather than the array you came from.\n"
            "A fresh catalog is built from them, which means new names and one\n"
            "amplitude for every resonator.")
        self.custom_frequencies_cb.toggled.connect(self._refresh)
        form.addWidget(self.custom_frequencies_cb)

        self.custom_widget = QtWidgets.QWidget()
        custom_form = QtWidgets.QFormLayout(self.custom_widget)
        custom_form.setContentsMargins(0, 0, 0, 0)
        self.sections_edit = QtWidgets.QLineEdit()
        self.sections_edit.setPlaceholderText(
            "Sweep centres (MHz, comma separated)")
        self.sections_edit.textChanged.connect(self._refresh)
        custom_form.addRow("Frequencies (MHz):", self.sections_edit)
        self.custom_amp_edit = QtWidgets.QLineEdit(str(DEFAULT_AMPLITUDE))
        self.custom_amp_edit.setValidator(QDoubleValidator(0.0, 1.0, 9, self))
        self.custom_amp_edit.textChanged.connect(self._refresh)
        custom_form.addRow("Amplitude for each:", self.custom_amp_edit)
        form.addWidget(self.custom_widget)

        return group

    def _amplitude_group(self) -> QtWidgets.QGroupBox:
        """One radio per ``AmplitudeSchedule`` constructor, each with its own
        fields; the selected one is the schedule."""
        group = QtWidgets.QGroupBox("Amplitude schedule")
        grid = QtWidgets.QGridLayout(group)
        self.schedule_buttons = QtWidgets.QButtonGroup(self)

        def radio(row, key, text, tip, fields=()):
            button = QtWidgets.QRadioButton(text)
            button.setToolTip(tip)
            self.schedule_buttons.addButton(button)
            button.setProperty("schedule_kind", key)
            grid.addWidget(button, row, 0)
            holder = QtWidgets.QWidget()
            line = QtWidgets.QHBoxLayout(holder)
            line.setContentsMargins(0, 0, 0, 0)
            for label, widget in fields:
                if label:
                    line.addWidget(QtWidgets.QLabel(label))
                line.addWidget(widget)
            line.addStretch(1)
            grid.addWidget(holder, row, 1)
            self._schedule_fields[key] = holder
            return button

        self._schedule_fields = {}

        self.catalog_radio = radio(
            0, "catalog", "Each resonator's own amplitude",
            "AmplitudeSchedule(): one pass, at the amplitude the catalog "
            "records for each resonator.")

        self.single_amp_edit = self._number(DEFAULT_AMPLITUDE)
        self.single_radio = radio(
            1, "single", "One amplitude",
            "AmplitudeSchedule(x): one pass, at this amplitude for everything.",
            [("", self.single_amp_edit)])

        self.list_amp_edit = QtWidgets.QLineEdit("0.001, 0.002, 0.004")
        self.list_amp_edit.textChanged.connect(self._refresh)
        self.list_radio = radio(
            2, "explicit", "A list of amplitudes",
            "AmplitudeSchedule.explicit([...]): these amplitudes, in this order.",
            [("", self.list_amp_edit)])

        self.ramp_start_edit = self._number(0.001)
        self.ramp_stop_edit = self._number(0.01)
        self.ramp_steps_edit = self._integer(5)
        self.ramp_spacing = self._spacing_combo()
        self.ramp_radio = radio(
            3, "ramp", "A ramp",
            "AmplitudeSchedule.ramp(start, stop, steps): absolute amplitudes, "
            "the same for every resonator.",
            [("from", self.ramp_start_edit), ("to", self.ramp_stop_edit),
             ("in", self.ramp_steps_edit), ("steps,", self.ramp_spacing)])

        self.factor_start_edit = self._number(0.5)
        self.factor_stop_edit = self._number(2.0)
        self.factor_steps_edit = self._integer(5)
        self.factor_spacing = self._spacing_combo()
        self.multiplicative_radio = radio(
            4, "multiplicative", "Multiples of each catalog amplitude",
            "AmplitudeSchedule.multiplicative(start, stop, steps): every "
            "resonator keeps its own scale and walks the same factors.",
            [("×", self.factor_start_edit), ("to ×", self.factor_stop_edit),
             ("in", self.factor_steps_edit), ("steps,", self.factor_spacing)])

        self.schedule_buttons.buttonToggled.connect(self._refresh)
        self._seed_schedule_from(self.params.get("amp"))
        return group

    def _number(self, default) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(f"{default:g}")
        edit.setValidator(QDoubleValidator(0.0, 1e6, 9, self))
        edit.setMaximumWidth(90)
        edit.textChanged.connect(self._refresh)
        return edit

    def _integer(self, default) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(str(default))
        edit.setValidator(QIntValidator(1, 1000, self))
        edit.setMaximumWidth(60)
        edit.textChanged.connect(self._refresh)
        return edit

    def _spacing_combo(self) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.addItems(["log", "linear"])
        combo.currentIndexChanged.connect(self._refresh)
        return combo

    def _seed_schedule_from(self, amp):
        """Select the radio that says what a previous call's ``amp`` was."""
        if isinstance(amp, AmplitudeSchedule):
            if amp.relative and amp.steps == (1.0,):
                self.catalog_radio.setChecked(True)
            elif amp.relative:
                self.multiplicative_radio.setChecked(True)
                self.factor_start_edit.setText(f"{amp.steps[0]:g}")
                self.factor_stop_edit.setText(f"{amp.steps[-1]:g}")
                self.factor_steps_edit.setText(str(len(amp.steps)))
            else:
                self.list_radio.setChecked(True)
                self.list_amp_edit.setText(
                    ", ".join(f"{v:g}" for v in amp.steps))
        elif isinstance(amp, (int, float)):
            self.single_radio.setChecked(True)
            self.single_amp_edit.setText(f"{float(amp):g}")
        else:
            self.catalog_radio.setChecked(True)

    def _parameters_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Sweep Parameters")
        form = QtWidgets.QFormLayout(group)

        span_khz = self.params.get("span_hz", DEFAULTS["span_hz"]) / 1e3
        self.span_khz_edit = QtWidgets.QLineEdit(f"{span_khz:g}")
        self.span_khz_edit.setValidator(QDoubleValidator(0.1, 10000.0, 3, self))
        self.span_khz_edit.textChanged.connect(self._refresh)
        form.addRow("Span per section (kHz):", self.span_khz_edit)

        self.npoints_edit = QtWidgets.QLineEdit(str(self.params.get(
            "npoints_per_sweep", DEFAULTS["npoints_per_sweep"])))
        self.npoints_edit.setValidator(QIntValidator(2, 10000, self))
        self.npoints_edit.textChanged.connect(self._refresh)
        form.addRow("Points per sweep:", self.npoints_edit)

        self.nsamps_edit = QtWidgets.QLineEdit(str(self.params.get(
            "nsamps", DEFAULTS["nsamps"])))
        self.nsamps_edit.setValidator(QIntValidator(1, 10000, self))
        self.nsamps_edit.textChanged.connect(self._refresh)
        form.addRow("Samples to average per point:", self.nsamps_edit)

        directions = self._seeded_directions()
        self.upward_cb = QtWidgets.QCheckBox("Upward")
        self.upward_cb.setChecked("upward" in directions)
        self.downward_cb = QtWidgets.QCheckBox("Downward")
        self.downward_cb.setChecked("downward" in directions)
        for box in (self.upward_cb, self.downward_cb):
            box.toggled.connect(self._refresh)
        direction_row = QtWidgets.QWidget()
        direction_layout = QtWidgets.QHBoxLayout(direction_row)
        direction_layout.setContentsMargins(0, 0, 0, 0)
        direction_layout.addWidget(self.upward_cb)
        direction_layout.addWidget(self.downward_cb)
        direction_layout.addStretch(1)
        form.addRow("Sweep direction:", direction_row)

        self.dac_scale_info = QtWidgets.QLabel("Unknown")
        self.dac_scale_info.setWordWrap(True)
        form.addRow("DAC full scale (dBm):", self.dac_scale_info)

        return group

    def _seeded_directions(self) -> tuple[str, ...]:
        seeded = self.params.get("sweep_direction",
                                 self.params.get("directions", "upward"))
        if isinstance(seeded, str):
            return (seeded,)
        return tuple(seeded)

    def _output_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Measurement")
        form = QtWidgets.QFormLayout(group)
        self.label_edit = QtWidgets.QLineEdit(self.params.get("label") or "")
        self.label_edit.setPlaceholderText("optional name for this measurement")
        self.label_edit.textChanged.connect(self._refresh)
        form.addRow("Name:", self.label_edit)
        self.filename_label = QtWidgets.QLabel()
        form.addRow("Saves as:", self.filename_label)
        return group

    def _buttons(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start Multisweep")
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self.accept)
        row.addWidget(self.start_btn)

        self.load_btn = QtWidgets.QPushButton("Load Multisweep")
        self.load_btn.setEnabled(False)
        self.load_btn.clicked.connect(self._load_data_avail)
        if not self.load_multisweep:
            self.load_btn.hide()
        row.addWidget(self.load_btn)

        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        row.addWidget(self.cancel_btn)
        return row

    # ── what the fields say ──────────────────────────────────────────────────

    def _schedule_kind(self) -> str:
        button = self.schedule_buttons.checkedButton()
        return button.property("schedule_kind") if button else "catalog"

    def _numbers(self, text: str) -> list[float]:
        return [float(part) for part in text.replace(",", " ").split()]

    def schedule(self) -> AmplitudeSchedule:
        """The schedule the selected radio names. Raises on unusable fields."""
        kind = self._schedule_kind()
        if kind == "single":
            return AmplitudeSchedule(float(self.single_amp_edit.text()))
        if kind == "explicit":
            return AmplitudeSchedule.explicit(
                self._numbers(self.list_amp_edit.text()))
        if kind == "ramp":
            return AmplitudeSchedule.ramp(
                float(self.ramp_start_edit.text()),
                float(self.ramp_stop_edit.text()),
                int(self.ramp_steps_edit.text()),
                spacing=self.ramp_spacing.currentText())
        if kind == "multiplicative":
            return AmplitudeSchedule.multiplicative(
                float(self.factor_start_edit.text()),
                float(self.factor_stop_edit.text()),
                int(self.factor_steps_edit.text()),
                spacing=self.factor_spacing.currentText())
        return AmplitudeSchedule()

    def directions(self):
        """One direction as a string, both as the sequence multisweep takes."""
        chosen = tuple(d for d, box in (("upward", self.upward_cb),
                                        ("downward", self.downward_cb))
                       if box.isChecked())
        return chosen[0] if len(chosen) == 1 else chosen

    def sweep_catalog(self) -> ResonatorCatalog | None:
        """The array to sweep: the one handed in, or one minted from typed
        frequencies, which needs an amplitude and invents names."""
        if not self.custom_frequencies_cb.isChecked():
            return self.catalog
        frequencies = [f * 1e6 for f in self._numbers(self.sections_edit.text())]
        if not frequencies or self.module is None:
            return None
        return ResonatorCatalog.from_frequencies(
            frequencies, module=self.module,
            amplitude=float(self.custom_amp_edit.text()))

    # ── the live preview ─────────────────────────────────────────────────────

    def _refresh(self, *_args):
        """Say what this call would do, and whether it can run at all.

        Every number shown is one the schedule computed: ``validate`` for the
        complaints and ``describe`` for the summary, so the dialog and the
        driver cannot disagree about what was asked for.
        """
        if not hasattr(self, "status_label"):
            return

        self.custom_widget.setVisible(self.custom_frequencies_cb.isChecked())
        for kind, holder in self._schedule_fields.items():
            holder.setEnabled(kind == self._schedule_kind())

        catalog = None
        issues = []
        try:
            catalog = self.sweep_catalog()
        except (ValueError, TypeError) as exc:
            issues.append(("error", str(exc)))

        if catalog is None and not issues:
            issues.append(("error", "No resonators to sweep."))
        self.sections_info_label.setText(
            f"{len(catalog.names())} resonators, "
            f"{catalog.names()[0]} to {catalog.names()[-1]}"
            if catalog is not None and catalog.names() else "No array loaded.")

        directions = self.directions()
        if not directions:
            issues.append(("error", "Pick at least one sweep direction."))
        n_directions = 1 if isinstance(directions, str) else len(directions)

        schedule = None
        if catalog is not None:
            try:
                schedule = self.schedule()
            except (ValueError, TypeError) as exc:
                issues.append(("error", str(exc)))
            else:
                issues.extend(schedule.validate(catalog, max(n_directions, 1)))

        issues.extend(self._parameter_issues())
        self._show_issues(issues)
        self._show_summary(schedule, catalog, n_directions)
        self.filename_label.setText(
            store._filename("multisweep", self.label_edit.text().strip() or None,
                            store._now()))

        blocked = any(severity == "error" for severity, _ in issues)
        self.start_btn.setEnabled(not blocked)

    def _parameter_issues(self) -> list[tuple[str, str]]:
        issues = []
        try:
            if float(self.span_khz_edit.text()) <= 0:
                issues.append(("error", "Span must be positive."))
        except ValueError:
            issues.append(("error", "Span is not a number."))
        try:
            if int(self.npoints_edit.text()) < 2:
                issues.append(("error", "A sweep needs at least two points."))
        except ValueError:
            issues.append(("error", "Points per sweep is not a whole number."))
        try:
            if int(self.nsamps_edit.text()) < 1:
                issues.append(("error", "Samples to average must be at least 1."))
        except ValueError:
            issues.append(("error", "Samples to average is not a whole number."))
        return issues

    def _show_issues(self, issues):
        for severity, icon in _SEVERITY:
            said = [message for level, message in issues if level == severity]
            if said:
                self.status_label.setText(f"{icon} {said[0]}")
                self.status_label.setStyleSheet(
                    f"color: {_SEVERITY_COLOR[severity]};")
                self.status_label.setToolTip("\n".join(
                    message for _, message in issues))
                return
        self.status_label.setText("")
        self.status_label.setToolTip("")

    def _show_summary(self, schedule, catalog, n_directions):
        if schedule is None or catalog is None:
            self.summary_label.setText("")
            return
        dac_scale = self.dac_scales.get(self.module)
        try:
            described = schedule.describe(catalog, max(n_directions, 1), dac_scale)
        except (ValueError, TypeError):
            self.summary_label.setText("")
            return
        sweeps, sections = described["n_sweeps"], described["n_sections"]
        text = (f"{sweeps} sweep{'' if sweeps == 1 else 's'} of "
                f"{sections} section{'' if sections == 1 else 's'}, "
                f"{described['amplitude_min']:.4g} to "
                f"{described['amplitude_max']:.4g} normalized")
        if "power_dbm_min" in described:
            text += (f" ({described['power_dbm_min']:+.1f} to "
                     f"{described['power_dbm_max']:+.1f} dBm)")
        self.summary_label.setText(text)

    def _start_if_valid(self):
        if self.start_btn.isEnabled():
            self.accept()

    # ── loading a previous sweep ─────────────────────────────────────────────

    def _load_data_avail(self):
        self.use_data_from_file = True
        self.accept()

    def _import_file(self):
        QtCore.QTimer.singleShot(0, self._open_file_dialog_async)

    def _open_file_dialog_async(self):
        if getattr(self, "_file_dialog", None) is None:
            self._file_dialog = QtWidgets.QFileDialog(
                self, "Load Multisweep Parameters")
            self._file_dialog.setFileMode(
                QtWidgets.QFileDialog.FileMode.ExistingFile)
            self._file_dialog.setNameFilters(
                ["Pickle Files (*.pkl *.pickle)", "All Files (*)"])
            self._file_dialog.setOptions(
                QtWidgets.QFileDialog.Option.DontUseNativeDialog
                | QtWidgets.QFileDialog.Option.ReadOnly)
            self._file_dialog.setModal(False)
            self._file_dialog.fileSelected.connect(self._on_file_selected)
        self._file_dialog.open()

    def _on_file_selected(self, path: str):
        """Fill the fields in from what the file records about the sweep."""
        container = load_multisweep_container(self, path)
        if container is None:
            return

        self.loaded_container = container
        self.load_btn.setEnabled(True)

        block = next(iter(container.values()))
        call_params = block["call_params"]
        self.catalog = ResonatorCatalog.from_dict(call_params["catalog"])
        self.module = self.catalog.module

        self.span_khz_edit.setText(f"{call_params['span_hz'] / 1e3:g}")
        self.npoints_edit.setText(str(call_params["npoints_per_sweep"]))
        self.nsamps_edit.setText(str(call_params["nsamps"]))
        directions = call_params["directions"]
        self.upward_cb.setChecked("upward" in directions)
        self.downward_cb.setChecked("downward" in directions)
        self._seed_schedule_from(
            AmplitudeSchedule.from_dict(call_params["amp_schedule"]))
        self._update_dac_scale_info()
        self._refresh()

    # ── what the task is given ───────────────────────────────────────────────

    def get_parameters(self) -> dict | None:
        """``crs.multisweep``'s keyword arguments, or the loaded container."""
        if self.use_data_from_file:
            return self.loaded_container
        try:
            catalog = self.sweep_catalog()
            if catalog is None:
                return None
            return {
                "catalog": catalog,
                "amp": self.schedule(),
                "span_hz": float(self.span_khz_edit.text()) * 1e3,
                "npoints_per_sweep": int(self.npoints_edit.text()),
                "nsamps": int(self.nsamps_edit.text()),
                "sweep_direction": self.directions(),
                "label": self.label_edit.text().strip() or None,
            }
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Could not read the sweep settings: {e}")
            return None
