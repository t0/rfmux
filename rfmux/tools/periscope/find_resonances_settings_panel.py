"""Persistent settings for the resonance finder.

A non-modal window over the keyword arguments of
:func:`rfmux.tuning.find_resonances.find_resonances`, and nothing else: open
it from the netanal panel's ``⚙`` button, set a threshold, press Find
Resonances as many times as you like. Values persist across Periscope
sessions through :mod:`~rfmux.tools.periscope.settings`.

The defaults come out of the finder's own signature, so the boxes cannot
drift from the library they call.
"""

from __future__ import annotations

import inspect

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from . import settings as periscope_settings
from ...tuning.find_resonances import find_resonances

# What the library does when you say nothing. Read once, at import.
DEFAULTS = {
    name: parameter.default
    for name, parameter in inspect.signature(find_resonances).parameters.items()
    if parameter.default is not inspect.Parameter.empty and name != "label"
}

# 0 in a box means "say nothing", which is not the same as a Q of 0.
_NO_LIMIT = "No limit"


class FindResonancesSettingsPanel(QtWidgets.QWidget):
    """The finder's arguments, remembered between searches.

    :meth:`get_parameters` returns them ready to splat into
    :func:`~rfmux.tuning.find_resonances.find_resonances_in_netanal`.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find Resonances Settings")
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self._setup_ui()
        self.set_parameters(periscope_settings.get_find_resonances_parameters())
        for widget in self._inputs:
            signal = getattr(widget, "valueChanged", None) or widget.toggled
            signal.connect(self._save)

    # ── the arguments ────────────────────────────────────────────────────────

    def get_parameters(self) -> dict:
        """The finder's keyword arguments, as the boxes have them."""
        expected = self.expected_resonances_spin.value()
        return {
            "min_dip_depth_db": self.min_dip_depth_spin.value(),
            "min_Q": self.min_q_spin.value() or None,
            "max_Q": self.max_q_spin.value() or None,
            "min_separation_hz": self.min_separation_spin.value() * 1e3,
            "require_isolation": self.require_isolation_check.isChecked(),
            "expected_resonances": expected or None,
        }

    def set_parameters(self, parameters: dict) -> None:
        """Fill the boxes in, taking anything absent from the library."""
        values = {**DEFAULTS, **parameters}
        for widget in self._inputs:
            widget.blockSignals(True)
        self.min_dip_depth_spin.setValue(values["min_dip_depth_db"])
        self.min_q_spin.setValue(values["min_Q"] or 0.0)
        self.max_q_spin.setValue(values["max_Q"] or 0.0)
        self.min_separation_spin.setValue((values["min_separation_hz"] or 0.0) / 1e3)
        self.require_isolation_check.setChecked(bool(values["require_isolation"]))
        self.expected_resonances_spin.setValue(values["expected_resonances"] or 0)
        for widget in self._inputs:
            widget.blockSignals(False)

    # ── construction ─────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        depth_group = QtWidgets.QGroupBox("How deep a dip counts")
        depth_form = QtWidgets.QFormLayout(depth_group)

        self.min_dip_depth_spin = QtWidgets.QDoubleSpinBox()
        self.min_dip_depth_spin.setRange(0.01, 100.0)
        self.min_dip_depth_spin.setDecimals(2)
        self.min_dip_depth_spin.setSingleStep(0.1)
        self.min_dip_depth_spin.setSuffix(" dB")
        self.min_dip_depth_spin.setToolTip(
            "Prominence floor, in true dB, for a dip to count.\n"
            "Lower it to 0.3-0.5 for shallow, overcoupled or low-Q resonators."
        )
        depth_form.addRow("Min dip depth:", self.min_dip_depth_spin)

        self.expected_resonances_spin = QtWidgets.QSpinBox()
        self.expected_resonances_spin.setRange(0, 10000)
        self.expected_resonances_spin.setSpecialValueText("Auto")
        self.expected_resonances_spin.setToolTip(
            "How many resonances the array has, if you know.\n"
            "The deepest this many are kept and the rest rejected; if fewer "
            "are found, the search says so.\nAuto imposes no count."
        )
        depth_form.addRow("Expected resonances:", self.expected_resonances_spin)

        layout.addWidget(depth_group)

        width_group = QtWidgets.QGroupBox("How wide a dip may be")
        width_form = QtWidgets.QFormLayout(width_group)

        self.min_q_spin = QtWidgets.QDoubleSpinBox()
        self.min_q_spin.setRange(0.0, 1e10)
        self.min_q_spin.setDecimals(0)
        self.min_q_spin.setSingleStep(1e3)
        self.min_q_spin.setSpecialValueText(_NO_LIMIT)
        self.min_q_spin.setToolTip(
            "Sets the widest dip accepted, as frequency / Q.\n"
            "Lower it for broad resonances; no limit leaves dips unbounded above.\n"
            "This is a screen on frequency / width, not a measurement of Q -- "
            "fitting a multisweep is what measures Q."
        )
        width_form.addRow("Min Q:", self.min_q_spin)

        self.max_q_spin = QtWidgets.QDoubleSpinBox()
        self.max_q_spin.setRange(0.0, 1e10)
        self.max_q_spin.setDecimals(0)
        self.max_q_spin.setSingleStep(1e5)
        self.max_q_spin.setSpecialValueText(_NO_LIMIT)
        self.max_q_spin.setToolTip(
            "Sets the narrowest dip accepted, as frequency / Q.\n"
            "This is what rejects single-sample noise spikes, so removing it "
            "is rarely what you want.\nAt netanal resolution neither Q bound "
            "usually bites."
        )
        width_form.addRow("Max Q:", self.max_q_spin)

        layout.addWidget(width_group)

        collision_group = QtWidgets.QGroupBox("Resonances too close to each other")
        collision_form = QtWidgets.QFormLayout(collision_group)

        self.min_separation_spin = QtWidgets.QDoubleSpinBox()
        self.min_separation_spin.setRange(0.0, 1e5)
        self.min_separation_spin.setDecimals(3)
        self.min_separation_spin.setSingleStep(1.0)
        self.min_separation_spin.setSuffix(" kHz")
        self.min_separation_spin.setSpecialValueText("Off")
        self.min_separation_spin.setToolTip(
            "Separation below which two resonances are treated as colliding.\n"
            "Off acts only on candidates at identical frequencies, so it "
            "touches nothing real.\n"
            "What happens to a close group is the switch below."
        )
        collision_form.addRow("Collision cut:", self.min_separation_spin)

        self.require_isolation_check = QtWidgets.QCheckBox(
            "Cut every member of a colliding group")
        self.require_isolation_check.setToolTip(
            "On: cut the whole group, so every resonance returned is one "
            "nothing else is near.\nA tone on either member of a collided "
            "pair still reads the other, which is why this is the default.\n\n"
            "Off: keep the deepest member and reject the rest. The list obeys "
            "the separation, but a survivor can still have a real resonance "
            "beside it -- the one that was cut.\n\n"
            "Either way, what was cut is drawn as a rejected candidate with "
            "its reason."
        )
        collision_form.addRow("", self.require_isolation_check)

        layout.addWidget(collision_group)

        # Every input, for blocking signals and for wiring the auto-save.
        self._inputs = (
            self.min_dip_depth_spin,
            self.expected_resonances_spin,
            self.min_q_spin,
            self.max_q_spin,
            self.min_separation_spin,
            self.require_isolation_check,
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

    # ── persistence ──────────────────────────────────────────────────────────

    def _save(self):
        periscope_settings.set_find_resonances_parameters(self.get_parameters())

    def _reset(self):
        self.set_parameters(DEFAULTS)
        self._save()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.hide()
        else:
            super().keyPressEvent(event)
