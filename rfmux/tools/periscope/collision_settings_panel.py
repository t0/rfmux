"""Catalog editing and settings for the multisweep collision check."""

import inspect
import math

from PyQt6 import QtCore, QtWidgets

from rfmux.tuning import find_sweeps_with_nearby_resonances


class CollisionTask(QtCore.QThread):
    completed = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, ms_module_output: dict, parameters: dict,
                 parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.ms_module_output = ms_module_output
        self.parameters = parameters

    def run(self) -> None:
        try:
            names = find_sweeps_with_nearby_resonances(
                self.ms_module_output, **self.parameters)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.completed.emit(names)


class CatalogEditDialog(QtWidgets.QDialog):
    run_requested = QtCore.pyqtSignal()
    remove_requested = QtCore.pyqtSignal(list)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit catalog")
        outer = QtWidgets.QVBoxLayout(self)
        removal = QtWidgets.QGroupBox("Remove resonators by name")
        removal_layout = QtWidgets.QFormLayout(removal)
        self.names = QtWidgets.QPlainTextEdit()
        self.names.setPlaceholderText(
            "Exact names, one per line or separated by commas")
        self.names.setMaximumHeight(90)
        removal_layout.addRow("Resonator names:", self.names)
        self.remove_button = QtWidgets.QPushButton("Remove names and re-sweep")
        self.remove_button.clicked.connect(self._request_removal)
        removal_layout.addRow(self.remove_button)
        outer.addWidget(removal)
        collision = QtWidgets.QGroupBox("Collision Cut")
        layout = QtWidgets.QFormLayout(collision)
        outer.addWidget(collision)
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
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        close = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.hide)
        outer.addWidget(close)

    def _request_removal(self) -> None:
        entries = self.names.toPlainText().replace(",", "\n").splitlines()
        names = list(dict.fromkeys(
            name.strip() for name in entries
            if name.strip()))
        self.remove_requested.emit(names)

    def set_measurement(self, block: dict) -> None:
        self.names.clear()
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
