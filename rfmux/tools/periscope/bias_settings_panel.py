"""Persistent settings for bias finding.

A non-modal window over the keyword arguments of
:func:`rfmux.tuning.bias.find_bias_points`: open it from the multisweep
panel's ``⚙`` button, set the thresholds, press Find Bias as many times as you
like. Values persist across Periscope sessions through
:mod:`~rfmux.tools.periscope.settings`.

The settings are grouped by the test each one belongs to, so it is clear what
controls what: a bifurcation method decides an amplitude, and only the group
that method runs is live. The defaults come out of the finder's own signature,
which is what Reset to Defaults puts back.
"""

from __future__ import annotations

import inspect

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from . import settings as periscope_settings
from ...tuning.bias import (
    BIFURCATION_METHODS,
    FREQUENCY_METHODS,
    HYSTERESIS_COMPARISONS,
    NEEDS_BOTH_DIRECTIONS,
    find_bias_points,
)

# What the library does when you say nothing. Read once, at import. *save* and
# *label* are the panel's business, not a setting: the multisweep panel saves
# through store so the finder's autosave cannot put a second copy elsewhere.
DEFAULTS = {
    name: parameter.default
    for name, parameter in inspect.signature(find_bias_points).parameters.items()
    if parameter.default is not inspect.Parameter.empty
    and name not in ("save", "label")
}

#: How the distance guard is expressed. The library takes hertz; a fraction is
#: resolved against the sweep span at the press, so the same setting means the
#: same thing on a measurement swept at another span.
DISTANCE_MODES = ("none", "absolute", "fraction")

#: What the fraction field starts at. Half a span reaches the edge of the
#: sweep, where the guard stops constraining anything, so the useful range is
#: below that and this sits in the middle of it.
DEFAULT_DISTANCE_FRACTION = 0.25

#: What the absolute field starts at, in kilohertz.
DEFAULT_DISTANCE_KHZ = 10.0

#: The panel's own settings, beside the library's.
PANEL_DEFAULTS = {
    "max_distance_mode": "none",
    "max_distance_fraction": DEFAULT_DISTANCE_FRACTION,
    "max_distance_khz": DEFAULT_DISTANCE_KHZ,
}

#: What ``direction=None`` is called on screen.
_AUTOMATIC = "Automatic"


class BiasSettingsPanel(QtWidgets.QWidget):
    """Bias finding's arguments, remembered between runs.

    :meth:`get_parameters` returns them ready to splat into
    :func:`~rfmux.tuning.bias.find_bias_points`.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bias Settings")
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self._setup_ui()
        self.set_parameters(periscope_settings.get_bias_parameters())
        for widget in self._inputs:
            signal = (getattr(widget, "valueChanged", None)
                      or getattr(widget, "currentIndexChanged", None)
                      or widget.toggled)
            signal.connect(self._save)
        self._update_enabled()

    # ── the arguments ────────────────────────────────────────────────────────

    def get_parameters(self, span_hz: float | None = None) -> dict:
        """The finder's keyword arguments, as the boxes have them.

        Args:
            span_hz: the span the sweeps were taken over, which is what a
                fractional distance guard is a fraction of. Required only when
                that mode is selected; without it the guard is dropped, since
                a fraction of an unknown span is not a distance.
        """
        return {
            "amplitude_method": self.method_combo.currentData(),
            "frequency_method": self.frequency_combo.currentData(),
            "direction": self.direction_combo.currentData(),
            "spike_prominence_factor": self.prominence_spin.value(),
            "noise_gate_factor": self.noise_gate_spin.value(),
            "max_discrepancy": self.discrepancy_spin.value(),
            "compare": self.compare_combo.currentData(),
            "max_distance_hz": self._max_distance_hz(span_hz),
        }

    def set_parameters(self, parameters: dict) -> None:
        """Fill the boxes in, taking anything absent from the library."""
        values = {**DEFAULTS, **PANEL_DEFAULTS, **parameters}
        for widget in self._inputs:
            widget.blockSignals(True)
        self._select(self.method_combo, values["amplitude_method"])
        self._select(self.frequency_combo, values["frequency_method"])
        self._select(self.direction_combo, values["direction"])
        self.prominence_spin.setValue(values["spike_prominence_factor"])
        self.noise_gate_spin.setValue(values["noise_gate_factor"])
        self.discrepancy_spin.setValue(values["max_discrepancy"])
        self._select(self.compare_combo, values["compare"])
        self.fraction_spin.setValue(values["max_distance_fraction"])
        self.absolute_spin.setValue(values["max_distance_khz"])
        mode = values["max_distance_mode"]
        # A saved max_distance_hz with no mode beside it is a setting from
        # before the radios existed, or the library's None; either way the mode
        # is what says which field is live.
        self._distance_radios[
            mode if mode in DISTANCE_MODES else "none"].setChecked(True)
        for widget in self._inputs:
            widget.blockSignals(False)
        self._update_enabled()

    def _max_distance_hz(self, span_hz: float | None) -> float | None:
        """The guard in hertz, whichever way it was expressed."""
        mode = self._distance_mode()
        if mode == "absolute":
            return self.absolute_spin.value() * 1e3
        if mode == "fraction" and span_hz:
            return self.fraction_spin.value() * float(span_hz)
        return None

    def _distance_mode(self) -> str:
        for mode, radio in self._distance_radios.items():
            if radio.isChecked():
                return mode
        return "none"

    # ── what the measurement supports ────────────────────────────────────────

    def set_directions_swept(self, directions) -> None:
        """Offer only the methods and directions *directions* can answer for.

        Comparing two sweeps needs two sweeps: a one-direction measurement has
        no hysteresis to look at, and the default method runs that test. Rather
        than let the press fail, the choices that cannot work are taken away.
        """
        directions = list(directions or [])
        both = {"upward", "downward"} <= set(directions)
        for index in range(self.method_combo.count()):
            method = self.method_combo.itemData(index)
            self._set_item_enabled(
                self.method_combo, index,
                both or method not in NEEDS_BOTH_DIRECTIONS)
        if not both and self.method_combo.currentData() in NEEDS_BOTH_DIRECTIONS:
            self._select(self.method_combo, "derivative")
            self._save()

        for index in range(1, self.direction_combo.count()):  # 0 is Automatic
            self._set_item_enabled(
                self.direction_combo, index,
                self.direction_combo.itemData(index) in directions)
        if (self.direction_combo.currentData() is not None
                and self.direction_combo.currentData() not in directions):
            self._select(self.direction_combo, None)
            self._save()
        self._update_enabled()

    @staticmethod
    def _set_item_enabled(combo, index: int, enabled: bool) -> None:
        """Grey a combo entry out rather than removing it, so the list a user
        learned does not change shape under them."""
        item = combo.model().item(index)
        item.setEnabled(enabled)

    # ── construction ─────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        amplitude_group = QtWidgets.QGroupBox("Which amplitude to bias at")
        amplitude_form = QtWidgets.QFormLayout(amplitude_group)
        self.method_combo = QtWidgets.QComboBox()
        for method in BIFURCATION_METHODS:
            self.method_combo.addItem(method.capitalize(), method)
        self.method_combo.setToolTip(
            "Which test decides that an amplitude step has bifurcated. The\n"
            "step below the first one that has is the one biased at.\n\n"
            "Both: either test firing counts, the most sensitive of the three.\n"
            "Derivative: jumps in one sweep, and the only one a single-direction\n"
            "measurement can run.\n"
            "Hysteresis: the two sweep directions disagreeing."
        )
        self.method_combo.currentIndexChanged.connect(self._update_enabled)
        amplitude_form.addRow("Bifurcation test:", self.method_combo)
        layout.addWidget(amplitude_group)

        self.derivative_group = QtWidgets.QGroupBox("Derivative test: jumps in one sweep")
        derivative_form = QtWidgets.QFormLayout(self.derivative_group)

        self.prominence_spin = QtWidgets.QDoubleSpinBox()
        self.prominence_spin.setRange(0.0, 10.0)
        self.prominence_spin.setDecimals(3)
        self.prominence_spin.setSingleStep(0.05)
        self.prominence_spin.setToolTip(
            "How far a spike must stand out of its neighbourhood, as a\n"
            "fraction of the whole range of the arc speed. Larger is less\n"
            "sensitive. This bar alone cannot tell a jump from noise, which\n"
            "is what the noise gate is for."
        )
        derivative_form.addRow("Spike prominence factor:", self.prominence_spin)

        self.noise_gate_spin = QtWidgets.QDoubleSpinBox()
        self.noise_gate_spin.setRange(0.0, 1000.0)
        self.noise_gate_spin.setDecimals(1)
        self.noise_gate_spin.setSingleStep(5.0)
        self.noise_gate_spin.setSpecialValueText("Off")
        self.noise_gate_spin.setToolTip(
            "How far above the sweep's own scatter a spike must stand, in\n"
            "median absolute deviations. Larger is less sensitive; Off leaves\n"
            "the prominence bar on its own, which a quiet sweep clears by\n"
            "construction. Roughly 10 to 150 behaves."
        )
        derivative_form.addRow("Noise gate factor:", self.noise_gate_spin)
        layout.addWidget(self.derivative_group)

        self.hysteresis_group = QtWidgets.QGroupBox(
            "Hysteresis test: the two directions disagreeing")
        hysteresis_form = QtWidgets.QFormLayout(self.hysteresis_group)

        self.discrepancy_spin = QtWidgets.QDoubleSpinBox()
        self.discrepancy_spin.setRange(0.0, 10.0)
        self.discrepancy_spin.setDecimals(3)
        self.discrepancy_spin.setSingleStep(0.01)
        self.discrepancy_spin.setToolTip(
            "How far apart the upward and downward sweeps may run before the\n"
            "step counts as bifurcated, as a fraction of the dip depth or the\n"
            "loop radius. Larger is less sensitive."
        )
        hysteresis_form.addRow("Max discrepancy:", self.discrepancy_spin)

        self.compare_combo = QtWidgets.QComboBox()
        for comparison in HYSTERESIS_COMPARISONS:
            self.compare_combo.addItem(comparison.capitalize(), comparison)
        self.compare_combo.setToolTip(
            "What the two directions are compared in: their |S21| against\n"
            "frequency, or how far apart they run on the IQ plane."
        )
        hysteresis_form.addRow("Compare in:", self.compare_combo)
        layout.addWidget(self.hysteresis_group)

        frequency_group = QtWidgets.QGroupBox("Where in that sweep the tone goes")
        frequency_form = QtWidgets.QFormLayout(frequency_group)

        self.frequency_combo = QtWidgets.QComboBox()
        for method, label in (("iq_derivative", "IQ derivative"),
                              ("minimum", "Minimum |S21|")):
            self.frequency_combo.addItem(label, method)
        self.frequency_combo.setToolTip(
            "IQ derivative: where the IQ trace moves fastest per hertz, which\n"
            "is where a frequency shift reads largest.\n"
            "Minimum |S21|: the bottom of the dip."
        )
        frequency_form.addRow("Frequency method:", self.frequency_combo)

        self.direction_combo = QtWidgets.QComboBox()
        self.direction_combo.addItem(_AUTOMATIC, None)
        for direction in ("upward", "downward"):
            self.direction_combo.addItem(direction.capitalize(), direction)
        self.direction_combo.setToolTip(
            "Which direction's sweep the bias frequency and the calibration\n"
            "are measured on. Automatic takes upward where there is one. The\n"
            "amplitude search is unaffected: it sees every direction."
        )
        frequency_form.addRow("Measured on:", self.direction_combo)
        layout.addWidget(frequency_group)

        distance_group = QtWidgets.QGroupBox("How far the tone may move")
        distance_layout = QtWidgets.QGridLayout(distance_group)
        distance_group.setToolTip(
            "How far from the centre of its sweep a resonance may be found\n"
            "before the answer is disbelieved. Past it, the tone stays where\n"
            "the sweep was centred and the finding is flagged."
        )

        self._distance_radios = {
            "none": QtWidgets.QRadioButton("No limit"),
            "absolute": QtWidgets.QRadioButton("Absolute:"),
            "fraction": QtWidgets.QRadioButton("Fraction of span:"),
        }
        self._distance_radios["none"].setToolTip(
            "Believe anything. The answer is a point of the measured trace, so\n"
            "it is inside the span whatever happens.")

        self.absolute_spin = QtWidgets.QDoubleSpinBox()
        self.absolute_spin.setRange(0.001, 1e6)
        self.absolute_spin.setDecimals(3)
        self.absolute_spin.setSingleStep(1.0)
        self.absolute_spin.setSuffix(" kHz")

        self.fraction_spin = QtWidgets.QDoubleSpinBox()
        self.fraction_spin.setRange(0.001, 0.5)
        self.fraction_spin.setDecimals(3)
        self.fraction_spin.setSingleStep(0.01)
        self.fraction_spin.setToolTip(
            "Of the span each sweep covered, resolved when Find Bias is\n"
            "pressed. A sweep runs half a span either side of its centre, so\n"
            "0.5 reaches the edge and constrains nothing.")

        distance_layout.addWidget(self._distance_radios["none"], 0, 0, 1, 2)
        distance_layout.addWidget(self._distance_radios["absolute"], 1, 0)
        distance_layout.addWidget(self.absolute_spin, 1, 1)
        distance_layout.addWidget(self._distance_radios["fraction"], 2, 0)
        distance_layout.addWidget(self.fraction_spin, 2, 1)
        for radio in self._distance_radios.values():
            radio.toggled.connect(self._update_enabled)
        layout.addWidget(distance_group)

        # Every input, for blocking signals and for wiring the auto-save.
        self._inputs = (
            self.method_combo,
            self.prominence_spin,
            self.noise_gate_spin,
            self.discrepancy_spin,
            self.compare_combo,
            self.frequency_combo,
            self.direction_combo,
            self.absolute_spin,
            self.fraction_spin,
            *self._distance_radios.values(),
        )

        buttons = QtWidgets.QHBoxLayout()
        reset_btn = QtWidgets.QPushButton("Reset to Defaults")
        reset_btn.setToolTip("Back to what the library does when you say nothing.")
        reset_btn.clicked.connect(self._reset)
        buttons.addWidget(reset_btn)
        buttons.addStretch(1)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.hide)
        close_btn.setDefault(True)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

    @staticmethod
    def _select(combo, value) -> None:
        """Select *value* silently, falling back to the first choice."""
        combo.blockSignals(True)
        combo.setCurrentIndex(max(0, combo.findData(value)))
        combo.blockSignals(False)

    def _update_enabled(self):
        """Only the group the chosen test runs is live, and only the distance
        field its radio selects."""
        method = self.method_combo.currentData()
        self.derivative_group.setEnabled(method in ("both", "derivative"))
        self.hysteresis_group.setEnabled(method in ("both", "hysteresis"))
        self.absolute_spin.setEnabled(self._distance_radios["absolute"].isChecked())
        self.fraction_spin.setEnabled(self._distance_radios["fraction"].isChecked())

    # ── persistence ──────────────────────────────────────────────────────────

    def _saved_form(self) -> dict:
        """What goes to settings: the library's arguments as chosen, plus how
        the distance guard was expressed, which hertz alone cannot say."""
        return {
            **self.get_parameters(),
            "max_distance_mode": self._distance_mode(),
            "max_distance_fraction": self.fraction_spin.value(),
            "max_distance_khz": self.absolute_spin.value(),
        }

    def _save(self):
        periscope_settings.set_bias_parameters(self._saved_form())

    def _reset(self):
        self.set_parameters({**DEFAULTS, **PANEL_DEFAULTS})
        self._save()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.hide()
        else:
            super().keyPressEvent(event)
