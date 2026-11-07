"""The Tuning Catalog window: the catalog of a multisweep panel as a table.

The window edits a copy of the catalog. Apply gives the copy to the panel;
Close discards it. The collision cut and the re-sweep without some
resonators are in the same window.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
from PyQt6 import QtCore, QtWidgets

from rfmux.core.resonators import ResonatorCatalog
from rfmux.core.transferfunctions import (convert_dacunits_to_dbm,
                                          convert_dbm_to_amplitude)
from rfmux.tuning import (find_sweeps_with_nearby_resonances, load_catalog,
                          save_catalog)

#: (header, tooltip). Columns marked editable in EDITABLE take typed values.
COLUMNS = [
    ("Name", "The name of the resonator. Read-only: it is the key of the "
             "resonator in the saved sweeps."),
    ("Ch", "The hardware channel. Read-only: it is the binding of the "
           "resonator to the board."),
    ("Bias (MHz)", "The bias frequency. The catalog puts it on the tone grid."),
    ("Amplitude", "The bias amplitude, as a fraction of DAC full scale."),
    ("Power (dBm)", "The power of the amplitude with the DAC scale of the "
                    "module now, at the output."),
    ("Measured (dBm)", "The power at the time of the measurement that chose "
                       "the bias point."),
    ("|df cal| (Hz/V)", "The magnitude of the df calibration, "
                        "1 / (dI/df + j dQ/df)."),
    ("Bifurcated at", "The first amplitude where the resonator bifurcated."),
    ("Flag", "Why Find Bias doubted the bias point. The tooltip of a cell "
             "gives the reason."),
    ("Skewed fr (MHz)", "The resonance frequency of the skewed fit of the "
                        "bias sweep. Find Bias sets the bias sweep."),
    ("Skewed Qr", "The loaded quality factor of the skewed fit."),
    ("Skewed Qi", "The internal quality factor of the skewed fit."),
    ("Skewed Qc", "The coupling quality factor of the skewed fit."),
    ("Nonlinear fr (MHz)", "The resonance frequency of the nonlinear fit of "
                           "the bias sweep. Find Bias sets the bias sweep."),
    ("Nonlinear Qr", "The loaded quality factor of the nonlinear fit."),
    ("Nonlinear Qi", "The internal quality factor of the nonlinear fit."),
    ("Nonlinear Qc", "The coupling quality factor of the nonlinear fit."),
    ("a", "The nonlinearity parameter of the nonlinear fit. Bifurcation "
          "starts at 4√3/9 ≈ 0.77."),
    ("Notes", "Your comment about the resonator (notes['comment']). The "
              "tooltip of a cell shows other notes, read-only."),
]
NAME, CHANNEL, FREQUENCY, AMPLITUDE, POWER, MEASURED, DF_CAL, BIFURCATED, \
    FLAG, SKEWED_FR, SKEWED_QR, SKEWED_QI, SKEWED_QC, NONLINEAR_FR, \
    NONLINEAR_QR, NONLINEAR_QI, NONLINEAR_QC, NONLINEARITY, NOTES = range(len(COLUMNS))

#: The fit columns of each model: (column, parameter). A model's group shows
#: only when a bias sweep in the catalog has its fit. The column of ``a``
#: always shows.
FIT_COLUMNS = {
    "skewed": ((SKEWED_FR, "fr"), (SKEWED_QR, "Qr"), (SKEWED_QI, "Qi"),
               (SKEWED_QC, "Qc")),
    "nonlinear": ((NONLINEAR_FR, "fr"), (NONLINEAR_QR, "Qr"),
                  (NONLINEAR_QI, "Qi"), (NONLINEAR_QC, "Qc")),
}
EDITABLE = (FREQUENCY, AMPLITUDE, POWER, NOTES)

#: A change of frequency or amplitude clears the calibration of the
#: resonator (``Resonator.update_bias_point``).
_CLEARS = ("A change of frequency or amplitude clears the calibration, the "
           "fits and the flag of the resonator.")


def _text(value, fmt: str) -> str:
    return "" if value is None else format(value, fmt)


def _fit_params(bias, model: str) -> dict:
    """The parameters of the *model* fit of the bias sweep, or {}. Only Find
    Bias sets a bias sweep, so a catalog before Find Bias has no fits."""
    fits = (bias.bias_sweep or {}).get("fits") or {}
    return (fits.get(model) or {}).get("params") or {}


class TuningCatalogDialog(QtWidgets.QDialog):
    """The catalog of the panel as a table that you can edit."""

    run_requested = QtCore.pyqtSignal()           # Run Collision Cut
    remove_requested = QtCore.pyqtSignal(list)    # names to remove and re-sweep
    catalog_applied = QtCore.pyqtSignal(object)   # the edited ResonatorCatalog

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tuning Catalog")
        self.resize(1100, 800)
        self._catalog: ResonatorCatalog | None = None
        self._dac_scale = None
        self._read_only = False
        self._filling = False

        outer = QtWidgets.QVBoxLayout(self)
        self.summary = QtWidgets.QLabel()
        outer.addWidget(self.summary)

        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([h for h, _ in COLUMNS])
        for column, (_header, tip) in enumerate(COLUMNS):
            self.table.horizontalHeaderItem(column).setToolTip(tip)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._edited)
        self.table.sortByColumn(FREQUENCY, QtCore.Qt.SortOrder.AscendingOrder)
        outer.addWidget(self.table, 1)   # the table takes the extra height

        rows = QtWidgets.QHBoxLayout()
        self.delete_button = QtWidgets.QPushButton("Delete selected")
        self.delete_button.setToolTip(
            "Delete the selected resonators from the catalog. Apply keeps the "
            "change.")
        self.delete_button.clicked.connect(self._delete_selected)
        self.remove_button = QtWidgets.QPushButton("Remove selected and re-sweep")
        self.remove_button.setToolTip(
            "Open a new multisweep of the catalog without the selected "
            "resonators. The catalog of the panel does not change.")
        self.remove_button.clicked.connect(self._request_removal)
        self.export_button = QtWidgets.QPushButton("Export CSV…")
        self.export_button.setToolTip(
            "Save the name, channel, bias frequency and amplitude of each "
            "resonator as a CSV file.")
        self.export_button.clicked.connect(self._export_csv)
        self.import_button = QtWidgets.QPushButton("Import CSV…")
        self.import_button.setToolTip(
            "Read a CSV file with the columns name, channel, bias_frequency_hz "
            "and bias_amplitude. The imported catalog has no calibration. "
            "Apply keeps it.")
        self.import_button.clicked.connect(self._import_csv)
        self.save_button = QtWidgets.QPushButton("Save Catalog")
        self.save_button.setToolTip(
            "Save the catalog in the table, with its calibration, fits and "
            "notes, as a catalog file in the session folder.")
        self.save_button.clicked.connect(self.save)
        self.open_button = QtWidgets.QPushButton("Open Catalog…")
        self.open_button.setToolTip(
            "Read a catalog file or a bias_catalog file into the table. Apply "
            "keeps it.")
        self.open_button.clicked.connect(self._open)
        for button in (self.delete_button, self.remove_button,
                       self.save_button, self.open_button,
                       self.export_button, self.import_button):
            rows.addWidget(button)
        rows.addStretch(1)
        outer.addLayout(rows)

        outer.addWidget(self._collision_group())

        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Apply
            | QtWidgets.QDialogButtonBox.StandardButton.Close)
        self.apply_button = buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Apply)
        self.apply_button.setToolTip("Give the edited catalog to the panel. The "
                                     "next multisweep and Apply Bias use it.")
        self.apply_button.clicked.connect(self._apply)
        buttons.rejected.connect(self.hide)
        outer.addWidget(buttons)

    # ── the collision cut ────────────────────────────────────────────────────

    def _collision_group(self) -> QtWidgets.QGroupBox:
        collision = QtWidgets.QGroupBox("Collision Cut")
        layout = QtWidgets.QFormLayout(collision)
        self.separation = QtWidgets.QLineEdit("100")
        self.prominence = QtWidgets.QLineEdit("1.0")
        spacing_hz = inspect.signature(find_sweeps_with_nearby_resonances).parameters[
            "min_dip_spacing_hz"].default
        self.spacing = QtWidgets.QLineEdit(f"{spacing_hz / 1e3:g}")
        self.separation.setToolTip(
            "Flag sweeps with two dips separated by this distance or less.")
        self.spacing.setToolTip(
            "Minimum spacing between minima counted as separate dips.")
        layout.addRow("Collision threshold (kHz):", self.separation)
        explanation = QtWidgets.QLabel(
            "A sweep is flagged when two detected dips are separated by the "
            "threshold or less. To pass, their separation must be greater.")
        explanation.setWordWrap(True)
        layout.addRow(explanation)
        layout.addRow("Minimum dip prominence (dB):", self.prominence)
        layout.addRow("Dip detection spacing (kHz):", self.spacing)
        self.iteration = QtWidgets.QComboBox()
        self.direction = QtWidgets.QComboBox()
        self.iteration.setToolTip(
            "Choose which drive amplitudes the collision check uses.")
        layout.addRow("Amplitude:", self.iteration)
        layout.addRow("Direction:", self.direction)
        self.run_button = QtWidgets.QPushButton("Run Collision Cut")
        self.run_button.clicked.connect(self.run_requested)
        layout.addRow(self.run_button)
        return collision

    def set_measurement(self, block: dict) -> None:
        """Offer the steps and directions of *block* to the collision cut."""
        self.status.clear()
        self.iteration.clear()
        self.iteration.addItem("All amplitudes", None)
        for step in sorted(block['results']):
            self.iteration.addItem(f"Step {step}", step)
        self.direction.clear()
        self.direction.addItem("All directions", None)
        for direction in sorted({d for steps in block['results'].values()
                                 for d in steps}):
            self.direction.addItem(direction, direction)

    def get_parameters(self) -> dict:
        """The arguments of the collision cut. Raise ValueError if a field is
        not correct."""
        separation = float(self.separation.text()) * 1e3
        prominence = float(self.prominence.text())
        spacing = float(self.spacing.text()) * 1e3
        if math.isnan(separation) or separation < 0:
            raise ValueError("Collision separation must be >= 0 kHz or inf.")
        if not math.isfinite(prominence) or prominence <= 0:
            raise ValueError("Dip prominence must be finite and > 0 dB.")
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("Dip spacing must be finite and > 0 kHz.")
        return dict(min_separation_hz=separation,
                    min_prominence_db=prominence,
                    min_dip_spacing_hz=spacing,
                    iteration=self.iteration.currentData(),
                    direction=self.direction.currentData())

    # ── the catalog ──────────────────────────────────────────────────────────

    def set_catalog(self, catalog: ResonatorCatalog | None, dac_scale_dbm,
                    read_only: bool = False) -> None:
        """Show a copy of *catalog*. Edits that were not applied go."""
        self._catalog = None if catalog is None else catalog.copy()
        self._dac_scale = dac_scale_dbm
        self._read_only = read_only
        for button in (self.delete_button, self.import_button,
                       self.open_button, self.apply_button):
            button.setEnabled(not read_only)
        self._fill()

    def catalog(self) -> ResonatorCatalog | None:
        """The edited copy."""
        return self._catalog

    def _fill(self) -> None:
        self._filling = True
        self.table.setSortingEnabled(False)
        try:
            resonators = list(self._catalog) if self._catalog is not None else []
            self.table.setRowCount(len(resonators))
            for row, resonator in enumerate(resonators):
                self._fill_row(row, resonator)
        finally:
            self.table.setSortingEnabled(True)
            self._filling = False
        resonators = list(self._catalog) if self._catalog is not None else []
        for model, columns in FIT_COLUMNS.items():
            shown = any(_fit_params(r.bias, model) for r in resonators)
            for column, _name in columns:
                self.table.setColumnHidden(column, not shown)
        self.table.resizeColumnsToContents()
        if self._catalog is None:
            self.summary.setText("No catalog.")
        else:
            scale = ("unknown" if self._dac_scale is None
                     else f"{self._dac_scale:+.2f} dBm")
            self.summary.setText(
                f"{self._catalog.name}: {len(self._catalog)} resonators on "
                f"module {self._catalog.module}. DAC full scale at output: "
                f"{scale}. {_CLEARS}")

    def _fill_row(self, row: int, resonator) -> None:
        bias = resonator.bias
        df_cal = bias.df_calibration
        power = (None if self._dac_scale is None
                 else float(convert_dacunits_to_dbm(bias.amplitude, self._dac_scale)))
        cells = {
            NAME: (resonator.name, resonator.name),
            CHANNEL: (str(resonator.channel), resonator.channel),
            FREQUENCY: (f"{bias.frequency_hz / 1e6:.6f}", bias.frequency_hz),
            AMPLITUDE: (f"{bias.amplitude:.6g}", bias.amplitude),
            POWER: (_text(power, ".2f"), power),
            MEASURED: (_text(bias.bias_power_dbm, ".2f"), bias.bias_power_dbm),
            DF_CAL: (_text(None if df_cal is None else abs(df_cal), ".4g"),
                     None if df_cal is None else abs(df_cal)),
            BIFURCATED: (_text(bias.bifurcated_at, ".4g"), bias.bifurcated_at),
            FLAG: (bias.flagged_kind or "", bias.flagged_kind or ""),
            NONLINEARITY: (_text(_fit_params(bias, "nonlinear").get("a"), ".3g"),
                           _fit_params(bias, "nonlinear").get("a")),
            NOTES: (str(resonator.notes.get("comment", "")),
                    resonator.notes.get("comment", "")),
        }
        for model, columns in FIT_COLUMNS.items():
            params = _fit_params(bias, model)
            for column, name in columns:
                value = params.get(name)
                cells[column] = (
                    _text(None if value is None else
                          value / 1e6 if name == "fr" else value,
                          ".6f" if name == "fr" else ".4g"), value)
        for column, (text, value) in cells.items():
            item = _Item(text, value)
            item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, resonator.name)
            editable = (column in EDITABLE and not self._read_only
                        and not (column == POWER and self._dac_scale is None))
            if not editable:
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            if column == FLAG and bias.flagged_because:
                item.setToolTip(bias.flagged_because)
            if column == NOTES:
                # The table edits "comment". Other keys stay, read-only here.
                others = {k: v for k, v in resonator.notes.items()
                          if k != "comment"}
                if others:
                    item.setToolTip("Other notes (read-only here):\n" + "\n".join(
                        f"{k}: {v}" for k, v in others.items()))
            self.table.setItem(row, column, item)

    def _edited(self, item: QtWidgets.QTableWidgetItem) -> None:
        """Put a typed value into the copy, or refuse it."""
        if self._filling or self._catalog is None:
            return
        name = item.data(QtCore.Qt.ItemDataRole.UserRole + 1)
        resonator = self._catalog[name]
        column = item.column()
        try:
            if column == FREQUENCY:
                resonator.update_bias_point(frequency_hz=float(item.text()) * 1e6)
            elif column == AMPLITUDE:
                resonator.update_bias_point(amplitude=float(item.text()))
            elif column == POWER:
                resonator.update_bias_point(amplitude=float(
                    convert_dbm_to_amplitude(float(item.text()), self._dac_scale)))
            elif column == NOTES:
                resonator.notes["comment"] = item.text()
            self.status.setText(f"{name} changed. Apply keeps the change.")
        except (ValueError, TypeError) as exc:
            self.status.setText(f"{name}: {exc}")
        self._fill()

    def _selected_names(self) -> list[str]:
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        return [self.table.item(row, NAME).text() for row in rows]

    def _delete_selected(self) -> None:
        names = self._selected_names()
        if not names:
            self.status.setText("Select at least one resonator.")
            return
        if len(names) == len(self._catalog):
            self.status.setText("Keep at least one resonator.")
            return
        for name in names:
            self._catalog.remove(name)
        self._fill()
        self.status.setText(f"{len(names)} deleted. Apply keeps the change.")

    def _request_removal(self) -> None:
        self.remove_requested.emit(self._selected_names())

    def _apply(self) -> None:
        if self._catalog is not None and not self._read_only:
            self.catalog_applied.emit(self._catalog.copy())
            self.status.setText("Applied. The next multisweep and Apply Bias "
                                "use this catalog.")

    # ── the native catalog file ──────────────────────────────────────────────

    def save(self) -> None:
        """Save the catalog in the table as a catalog file."""
        if self._catalog is None:
            return
        path = save_catalog(self._catalog)
        self.status.setText(f"Saved {path}." if path else
                            "The catalog was not saved. The console says why.")

    def _open(self) -> None:
        dialog = QtWidgets.QFileDialog(self, "Open catalog", "",
                                       "Catalog files (*.pkl)")
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dialog.fileSelected.connect(self.open_catalog)
        dialog.open()

    def open_catalog(self, path: str) -> None:
        """Read a catalog file or a bias_catalog file into the table."""
        if self._catalog is None:
            return
        try:
            catalog = load_catalog(path)
        except Exception as exc:
            self.status.setText(f"Could not open {path}: {exc}")
            return
        if catalog.module != self._catalog.module:
            self.status.setText(f"{Path(path).name} is a catalog of module "
                                f"{catalog.module}, not of module "
                                f"{self._catalog.module}.")
            return
        self._catalog = catalog
        self._fill()
        self.status.setText(f"Opened {len(catalog)} resonators from "
                            f"{Path(path).name}. Apply keeps them.")

    # ── CSV ──────────────────────────────────────────────────────────────────

    def _export_csv(self) -> None:
        if self._catalog is None:
            return
        dialog = QtWidgets.QFileDialog(self, "Export catalog", "",
                                       "CSV files (*.csv)")
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setDefaultSuffix("csv")
        dialog.fileSelected.connect(self.export_csv)
        dialog.open()

    def export_csv(self, path: str) -> None:
        Path(path).write_text(self._catalog.to_csv())
        self.status.setText(f"Exported {len(self._catalog)} resonators to {path}.")

    def _import_csv(self) -> None:
        dialog = QtWidgets.QFileDialog(self, "Import catalog", "",
                                       "CSV files (*.csv)")
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dialog.fileSelected.connect(self.import_csv)
        dialog.open()

    def import_csv(self, path: str) -> None:
        if self._catalog is None:
            return
        try:
            self._catalog = ResonatorCatalog.from_csv(
                Path(path).read_text(), module=self._catalog.module)
        except (OSError, ValueError, KeyError) as exc:
            self.status.setText(f"Could not import {path}: {exc}")
            return
        self._fill()
        self.status.setText(f"Imported {len(self._catalog)} resonators from "
                            f"{path}. Apply keeps them.")


class _Item(QtWidgets.QTableWidgetItem):
    """A cell that sorts by its value, not its text. Cells with no value go
    last."""

    def __init__(self, text: str, value):
        super().__init__(text)
        self._value = value

    def __lt__(self, other) -> bool:
        a, b = self._value, getattr(other, "_value", None)
        if a is None or b is None:
            return a is not None
        if isinstance(a, (int, float, np.floating)) and isinstance(
                b, (int, float, np.floating)):
            return a < b
        return str(a) < str(b)
