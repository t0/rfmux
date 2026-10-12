"""The thread that runs the multisweep collision check."""

from PyQt6 import QtCore

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
