"""Settings for the headless multisweep collision check."""

import math

from PyQt6 import QtCore, QtWidgets

from rfmux.tuning import find_sweeps_with_nearby_resonances


class CollisionTask(QtCore.QThread):
    completed = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, block: dict, parameters: dict,
                 parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.block = block
        self.parameters = parameters

    def run(self) -> None:
        try:
            names = find_sweeps_with_nearby_resonances(
                self.block, **self.parameters)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.completed.emit(names)


class CollisionSettingsPanel(QtWidgets.QDialog):
    run_requested = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Multisweep Collision Cut")
        layout = QtWidgets.QFormLayout(self)
        self.separation = QtWidgets.QLineEdit("100")
        self.prominence = QtWidgets.QLineEdit("1.0")
        self.spacing = QtWidgets.QLineEdit("1.0")
        self.separation.setToolTip(
            "Reject a section with two dips this close or closer. "
            "Enter inf for any second dip in the window.")
        self.spacing.setToolTip(
            "Minimum spacing for resolving two dips; keep below the cut.")
        layout.addRow("Collision separation (kHz):", self.separation)
        layout.addRow("Minimum dip prominence (dB):", self.prominence)
        layout.addRow("Minimum dip spacing (kHz):", self.spacing)
        self.iteration = QtWidgets.QComboBox()
        self.direction = QtWidgets.QComboBox()
        self.iteration.setToolTip(
            "All amplitudes rejects on any collision. Select an early step "
            "if high-drive bifurcation produces false hits.")
        layout.addRow("Amplitude:", self.iteration)
        layout.addRow("Direction:", self.direction)
        self.run_button = QtWidgets.QPushButton("Run Collision Cut")
        self.run_button.clicked.connect(self.run_requested)
        layout.addRow(self.run_button)

    def set_measurement(self, block: dict) -> None:
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
