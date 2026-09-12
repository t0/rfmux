"""What a tab drawing fits draws: which model, and over which sweeps.

A view control rather than a fit setting, so it sits in the tab it changes
instead of in the fitters' settings window. The models on offer are the ones
the sweeps carry fits for and the amplitudes are the steps the measurement
walked, so the multisweep panel says what both are; the bias step is among
them once something has chosen one. The Fit Results and Fit Histograms tabs
each have one, chosen independently and each persisting across Periscope
sessions under its own *name*, through
:mod:`~rfmux.tools.periscope.settings`.
"""

from __future__ import annotations

from PyQt6 import QtWidgets
from PyQt6.QtCore import pyqtSignal

from . import settings as periscope_settings
from .fit_settings_panel import ALL_AMPLITUDES
from .layouts import FlowLayout, labelled


class FitDisplayToolbar(QtWidgets.QWidget):
    """A fit tab's own toolbar: one model, one set of amplitudes."""

    #: Emitted when either choice changes, so the tab can redraw.
    display_changed = pyqtSignal()

    def __init__(self, parent=None, *, name: str = "fits"):
        super().__init__(parent)
        self._name = name
        # What was last drawn, held apart from the combos: they carry only
        # what the measurement on screen has, and a choice it cannot honour is
        # picked up again by one that can.
        saved = periscope_settings.get_fit_display(name)
        self._wanted_model = saved.get("model")
        self._wanted_amplitude = saved.get("amplitude", ALL_AMPLITUDES)
        self._setup_ui()
        self.model_combo.currentIndexChanged.connect(self._changed)
        self.amplitude_combo.currentIndexChanged.connect(self._changed)

    # ── what is drawn ────────────────────────────────────────────────────────

    def get_model(self):
        """The model to draw, or None if these sweeps carry no fits."""
        return self.model_combo.currentData()

    def get_amplitude(self):
        """Which sweeps to draw: a step, ``BIAS_AMPLITUDE``, or ``ALL_AMPLITUDES``."""
        return self.amplitude_combo.currentData()

    def set_models_fitted(self, models) -> None:
        """Offer *models*, which are the ones the sweeps carry fits for.

        What was fitted, not what the fit settings ask for: a measurement
        loaded from a file was fitted by whatever fitted it, and one not yet
        fitted has nothing to draw.
        """
        self._refill(self.model_combo,
                     [(model.capitalize(), model) for model in models],
                     self._wanted_model)
        self.model_combo.setEnabled(bool(models))

    def set_amplitude_choices(self, choices) -> None:
        """Offer *choices*, as ``[(label, value), ...]``, keeping the current one.

        A step means nothing until something has been swept at it, and "at
        bias" nothing until something has chosen one, so the panel says which
        of them this measurement has.
        """
        self._refill(self.amplitude_combo, choices, self._wanted_amplitude)

    def _refill(self, combo, choices, wanted) -> None:
        """Rebuild *combo* on what a measurement has, back on *wanted*.

        The wanted choice is what was last asked for, and it outlives a
        measurement that cannot honour it: a step this one did not walk falls
        back to the first choice, and is picked up again by one that did.
        """
        before = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for label, value in choices:
            combo.addItem(label, value)
        combo.setCurrentIndex(max(0, combo.findData(wanted)))
        combo.blockSignals(False)
        if combo.currentData() != before:
            self.display_changed.emit()

    # ── construction ─────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = FlowLayout(self, margin=0)

        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setToolTip(
            "Which fitted model is drawn. One at a time, so a plot shows one "
            "model's answer rather than three overlaid")
        self.model_combo.setEnabled(False)
        layout.addWidget(labelled("Fit:", self.model_combo))

        self.amplitude_combo = QtWidgets.QComboBox()
        self.amplitude_combo.setToolTip(
            "Which sweeps are drawn: all of them, one amplitude step of the "
            "schedule, or -- once a bias has been found -- each resonator at "
            "the step it is biased at")
        self.amplitude_combo.addItem("All amplitudes", ALL_AMPLITUDES)
        layout.addWidget(labelled("Amplitude:", self.amplitude_combo))

    def _changed(self):
        self._wanted_model = self.get_model()
        self._wanted_amplitude = self.get_amplitude()
        periscope_settings.set_fit_display(
            {"model": self._wanted_model, "amplitude": self._wanted_amplitude},
            self._name)
        self.display_changed.emit()
