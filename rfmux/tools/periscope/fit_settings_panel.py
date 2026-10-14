"""Fit settings; Apply saves edits for future runs."""

from __future__ import annotations

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from . import settings as periscope_settings
from .analysis_settings_panel import AnalysisSettingsPanel

#: The models the fitters are offered, in the order they are listed. The circle
#: fit is not among them: it fits the IQ loop, so it draws nothing on a
#: magnitude plot, and nothing here reads it yet.
MODELS = ("skewed", "nonlinear")

#: What the amplitude choice means, for the choices that are not a step number.
ALL_AMPLITUDES = None
BIAS_AMPLITUDE = "bias"


class FitSettingsPanel(AnalysisSettingsPanel):
    """The fitters' settings, remembered between fits.

    :meth:`get_parameters` returns ``{"models": (...), "amplitude_choice": ...}``
    — the models as :func:`~rfmux.tuning.fits.fit_sweeps` takes them, and the
    amplitude choice as :class:`~rfmux.tools.periscope.tasks.RunFitsTask` does.
    Nothing here changes what is on screen: it is all what the next fit is
    asked for.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fit Settings")
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self._setup_ui()
        self.set_parameters(periscope_settings.get_fit_parameters())
        self._setup_actions()

    # ── what the fitters are asked for ───────────────────────────────────────

    def _read_parameters(self) -> dict:
        """The settings, as the boxes have them."""
        return {
            "models": tuple(name for name, box in self._model_checks.items()
                            if box.isChecked()),
            "amplitude_choice": self.amplitude_combo.currentData(),
        }

    def set_parameters(self, parameters: dict) -> None:
        """Fill the boxes in. Anything absent keeps what it has."""
        models = parameters.get("models")
        if models is not None:
            for name, box in self._model_checks.items():
                box.blockSignals(True)
                box.setChecked(name in models)
                box.blockSignals(False)
        if "amplitude_choice" in parameters:
            self.set_amplitude_choice(parameters["amplitude_choice"])

    # ── the amplitudes on offer ──────────────────────────────────────────────

    def set_amplitude_choices(self, choices) -> None:
        """Offer *choices*, as ``[(label, value), ...]``, keeping the current one.

        The steps of a schedule are the choice, and a schedule is a property of
        a measurement, so the multisweep panel says what they are: "step 3"
        means nothing until something has been swept at it.
        """
        previous = self.amplitude_combo.currentData()
        self.amplitude_combo.blockSignals(True)
        self.amplitude_combo.clear()
        for label, value in choices:
            self.amplitude_combo.addItem(label, value)
        index = self.amplitude_combo.findData(previous)
        self.amplitude_combo.setCurrentIndex(max(0, index))
        self.amplitude_combo.blockSignals(False)
        if self.amplitude_combo.findData(self._applied["amplitude_choice"]) < 0:
            self._applied["amplitude_choice"] = self.amplitude_combo.itemData(0)

    def set_amplitude_choice(self, value) -> None:
        """Select *value*, or the first choice if this measurement has no such step."""
        self._select(self.amplitude_combo, value)

    @staticmethod
    def _select(combo, value) -> None:
        """Select *value* silently, falling back to the first choice."""
        combo.blockSignals(True)
        combo.setCurrentIndex(max(0, combo.findData(value)))
        combo.blockSignals(False)

    # ── construction ─────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        models_group = QtWidgets.QGroupBox("Which models to fit")
        models_layout = QtWidgets.QVBoxLayout(models_group)
        self._model_checks = {}
        for name, tip in (
            ("skewed", "A skewed Lorentzian over |S21|: fr, Qr, Qc, Qi"),
            ("nonlinear", "Fit the complex IQ trace, including nonlinearity."),
        ):
            box = QtWidgets.QCheckBox(name.capitalize())
            box.setToolTip(tip)
            box.setChecked(True)
            models_layout.addWidget(box)
            self._model_checks[name] = box
        layout.addWidget(models_group)

        amplitude_group = QtWidgets.QGroupBox("Which sweeps to fit")
        amplitude_layout = QtWidgets.QVBoxLayout(amplitude_group)
        self.amplitude_combo = QtWidgets.QComboBox()
        self.amplitude_combo.setToolTip(
            "Fit all amplitudes, each resonator at its bias amplitude, "
            "or one sweep step.")
        self.amplitude_combo.addItem("All amplitudes", ALL_AMPLITUDES)
        amplitude_layout.addWidget(self.amplitude_combo)
        layout.addWidget(amplitude_group)

        layout.addStretch()

    def _save(self):
        periscope_settings.set_fit_parameters(self.get_parameters())
