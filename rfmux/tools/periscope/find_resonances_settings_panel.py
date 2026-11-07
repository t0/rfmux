"""Find resonances settings; Apply saves edits for future runs."""

from __future__ import annotations

import inspect

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from . import settings as periscope_settings
from .analysis_settings_panel import AnalysisSettingsPanel
from ...tuning.find_resonances import find_resonances

# What the library does when you say nothing. Read once, at import.
DEFAULTS = {
    name: parameter.default
    for name, parameter in inspect.signature(find_resonances).parameters.items()
    if parameter.default is not inspect.Parameter.empty and name != "label"
}

# 0 in a box means "say nothing", which is not the same as a Q of 0.
_NO_LIMIT = "No limit"


class FindResonancesSettingsPanel(AnalysisSettingsPanel):
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
        self._setup_actions()

    # ── the arguments ────────────────────────────────────────────────────────

    def _read_parameters(self) -> dict:
        """The finder's keyword arguments, as the boxes have them."""
        expected = self.expected_resonances_spin.value()
        return {
            "min_dip_depth_db": self.min_dip_depth_spin.value(),
            "min_Q": self.min_q_spin.value() or None,
            "max_Q": self.max_q_spin.value() or None,
            "min_separation_hz": (None if self.disable_collision_check.isChecked()
                                  else self.min_separation_spin.value() * 1e3),
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
        separation = values["min_separation_hz"]
        self.disable_collision_check.setChecked(separation is None)
        if separation is not None:
            self.min_separation_spin.setValue(separation / 1e3)
        self.require_isolation_check.setChecked(bool(values["require_isolation"]))
        self.expected_resonances_spin.setValue(values["expected_resonances"] or 0)
        for widget in self._inputs:
            widget.blockSignals(False)
        self._update_collision_controls()

    def _update_collision_controls(self) -> None:
        self.collision_controls.setEnabled(
            not self.disable_collision_check.isChecked())

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
            "Minimum dip prominence in dB. Lower values detect shallower "
            "dips."
        )
        depth_form.addRow("Min dip depth:", self.min_dip_depth_spin)

        self.expected_resonances_spin = QtWidgets.QSpinBox()
        self.expected_resonances_spin.setRange(0, 10000)
        self.expected_resonances_spin.setSpecialValueText("Auto")
        self.expected_resonances_spin.setToolTip(
            "Keep up to this many of the deepest resonances. Auto keeps "
            "all matches."
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
            "Reject dips wider than frequency / Q. Lower values allow "
            "broader dips."
        )
        width_form.addRow("Min Q:", self.min_q_spin)

        self.max_q_spin = QtWidgets.QDoubleSpinBox()
        self.max_q_spin.setRange(0.0, 1e10)
        self.max_q_spin.setDecimals(0)
        self.max_q_spin.setSingleStep(1e5)
        self.max_q_spin.setSpecialValueText(_NO_LIMIT)
        self.max_q_spin.setToolTip(
            "Reject dips narrower than frequency / Q, including narrow "
            "noise spikes."
        )
        width_form.addRow("Max Q:", self.max_q_spin)

        layout.addWidget(width_group)

        collision_group = QtWidgets.QGroupBox("Resonances too close to each other")
        collision_layout = QtWidgets.QVBoxLayout(collision_group)
        self.disable_collision_check = QtWidgets.QCheckBox("Disable collision cut")
        self.disable_collision_check.setToolTip(
            "Skip the separation check entirely; other resonance filters still apply.")
        self.disable_collision_check.toggled.connect(self._update_collision_controls)
        collision_layout.addWidget(self.disable_collision_check)
        self.collision_controls = QtWidgets.QWidget()
        collision_form = QtWidgets.QFormLayout(self.collision_controls)
        collision_form.setContentsMargins(0, 0, 0, 0)
        collision_layout.addWidget(self.collision_controls)

        self.min_separation_spin = QtWidgets.QDoubleSpinBox()
        self.min_separation_spin.setRange(0.0, 1e5)
        self.min_separation_spin.setDecimals(3)
        self.min_separation_spin.setSingleStep(1.0)
        self.min_separation_spin.setSuffix(" kHz")
        self.min_separation_spin.setToolTip(
            "Reject resonance pairs separated by this distance or less."
        )
        collision_form.addRow("Collision threshold (kHz):", self.min_separation_spin)
        explanation = QtWidgets.QLabel(
            "Resonators collide when their frequency separation is at or below "
            "the threshold. To pass, separation must be greater.")
        explanation.setWordWrap(True)
        collision_form.addRow(explanation)

        self.require_isolation_check = QtWidgets.QCheckBox(
            "Cut every member of a colliding group")
        self.require_isolation_check.setToolTip(
            "Checked: reject every resonance in a close group. Unchecked: "
            "keep the deepest."
        )
        collision_form.addRow("", self.require_isolation_check)

        layout.addWidget(collision_group)

        # Inputs whose signals are blocked while restoring values.
        self._inputs = (
            self.min_dip_depth_spin,
            self.expected_resonances_spin,
            self.min_q_spin,
            self.max_q_spin,
            self.min_separation_spin,
            self.disable_collision_check,
            self.require_isolation_check,
        )

    # ── persistence ──────────────────────────────────────────────────────────

    def _save(self):
        periscope_settings.set_find_resonances_parameters(self.get_parameters())

    def _reset(self):
        self.set_parameters(DEFAULTS)
