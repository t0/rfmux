"""Persistent settings for the resonator fitters.

A non-modal window over what :func:`rfmux.tuning.fits.fit_sweeps` is asked
for -- which models to run, and which of a schedule's amplitudes to run them
on -- and over which of the results the Fit Results tab draws. Open it from
the multisweep panel's ``⚙`` button, set it once, press Run Fit as many times
as you like. Values persist across Periscope sessions through
:mod:`~rfmux.tools.periscope.settings`.

Everything else the fitters take -- ``approx_Qr``, ``normalize``,
``fr_limit_hz``, ``n_extrema_points``, ``max_residual`` -- stays at the
library's default, which is one fewer place for a GUI value to drift from the
fitters' own. Expose one here when something asks for it.
"""

from __future__ import annotations

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal

from . import settings as periscope_settings

#: The models this panel offers, in the order it lists them. The circle fit is
#: not among them: it fits the IQ loop, so it draws nothing on a magnitude
#: plot, and nothing here reads it yet.
MODELS = ("skewed", "nonlinear")

#: What the amplitude choice means, for the choices that are not a step number.
ALL_AMPLITUDES = None
BIAS_AMPLITUDE = "bias"


class FitSettingsPanel(QtWidgets.QWidget):
    """The fitters' settings, remembered between fits.

    :meth:`get_parameters` returns ``{"models": (...), "amplitude_choice": ...}``
    — the models as :func:`~rfmux.tuning.fits.fit_sweeps` takes them, and the
    amplitude choice as :class:`~rfmux.tools.periscope.tasks.RunFitsTask` does.
    """

    #: Emitted when the model to draw changes, so the tab can redraw. Nothing
    #: else here changes what is on screen: the rest is what the next fit is
    #: asked for.
    display_model_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fit Settings")
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        # What a past session drew, held until a measurement says which models
        # it has fits for: the combo is empty until then, so it cannot hold it.
        self._wanted_display_model = None
        self._setup_ui()
        self.set_parameters(periscope_settings.get_fit_parameters())
        for box in self._model_checks.values():
            box.toggled.connect(self._save)
        self.amplitude_combo.currentIndexChanged.connect(self._save)
        self.display_combo.currentIndexChanged.connect(self._display_changed)

    # ── what the fitters are asked for ───────────────────────────────────────

    def get_parameters(self) -> dict:
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
        if parameters.get("display_model"):
            self._wanted_display_model = parameters["display_model"]
            self._select(self.display_combo, self._wanted_display_model)

    # ── which model is drawn ─────────────────────────────────────────────────

    def get_display_model(self):
        """The model the Fit Results tab should draw, or None if none is fitted."""
        return self.display_combo.currentData()

    def set_models_fitted(self, models) -> None:
        """Offer *models*, which are the ones the sweeps carry fits for.

        What was fitted, not what the checkboxes ask for: a measurement loaded
        from a file was fitted by whatever fitted it, and one not yet fitted
        has nothing to draw.
        """
        previous = self.display_combo.currentData() or self._wanted_display_model
        self.display_combo.blockSignals(True)
        self.display_combo.clear()
        for model in models:
            self.display_combo.addItem(model.capitalize(), model)
        self.display_combo.setCurrentIndex(
            max(0, self.display_combo.findData(previous)))
        self.display_combo.blockSignals(False)
        self.display_group.setEnabled(bool(models))
        if self.display_combo.currentData() != previous:
            self.display_model_changed.emit()

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
            ("nonlinear", "The complex trace after the readout gain is removed: "
                          "resonator parameters and the nonlinearity a"),
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
            "All of the sweeps, each resonator at the amplitude it is biased "
            "at, or one amplitude step of the schedule")
        self.amplitude_combo.addItem("All amplitudes", ALL_AMPLITUDES)
        amplitude_layout.addWidget(self.amplitude_combo)
        layout.addWidget(amplitude_group)

        self.display_group = QtWidgets.QGroupBox("Which model to draw")
        display_layout = QtWidgets.QVBoxLayout(self.display_group)
        self.display_combo = QtWidgets.QComboBox()
        self.display_combo.setToolTip(
            "Which fitted model the Fit Results tab draws over the measurement. "
            "One at a time, so a subplot carries one line over its points")
        self.display_group.setEnabled(False)
        display_layout.addWidget(self.display_combo)
        layout.addWidget(self.display_group)

        layout.addStretch()

    def _display_changed(self):
        self._wanted_display_model = self.get_display_model()
        self._save()
        self.display_model_changed.emit()

    def _save(self):
        periscope_settings.set_fit_parameters(
            {**self.get_parameters(), "display_model": self.get_display_model()})
