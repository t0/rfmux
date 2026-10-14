"""Shared Apply and Close actions for analysis settings."""

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal


class AnalysisSettingsPanel(QtWidgets.QWidget):
    applied = pyqtSignal()

    def _setup_actions(self) -> None:
        self._applied = self._read_parameters()
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Apply
            | QtWidgets.QDialogButtonBox.StandardButton.Close)
        if hasattr(self, "_reset"):
            reset_button = buttons.addButton(
                "Reset to Defaults", QtWidgets.QDialogButtonBox.ButtonRole.ResetRole)
            reset_button.setToolTip("Restore defaults. Press Apply to save them.")
            reset_button.clicked.connect(self._reset)
        self.apply_button = buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Apply)
        self.apply_button.setToolTip("Save these settings for future runs.")
        self.apply_button.clicked.connect(self._apply)
        self.close_button = buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Close)
        self.close_button.setToolTip("Close and discard unapplied edits.")
        self.close_button.clicked.connect(self.close)
        self.layout().addWidget(buttons)

    def get_parameters(self) -> dict:
        """Settings accepted by Apply, ready for the next run."""
        return dict(self._applied)

    def _apply(self) -> None:
        self._applied = self._read_parameters()
        self._save()
        self.applied.emit()

    def closeEvent(self, event) -> None:
        self.set_parameters(self._applied)
        super().closeEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)
