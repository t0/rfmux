"""
Persistent settings panel for the Find Resonances algorithm.

``FindResonancesSettingsPanel`` is a non-modal ``QWidget`` that retains its
values between "Find Resonances" invocations and across Periscope sessions
(via QSettings).  Open it from the "⚙ Find Resonances Settings" button on
the Network Analysis panel toolbar; it appears as a small floating window and
can be dismissed without disrupting ongoing work.

``get_settings()`` returns a dict that can be passed directly to
``fitting.find_resonances()`` as keyword arguments.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QSpinBox, QPushButton, QHBoxLayout, QLabel,
    QRadioButton, QButtonGroup,
)
from PyQt6.QtCore import Qt

from . import settings as periscope_settings


class FindResonancesSettingsPanel(QWidget):
    """
    Persistent, non-modal settings panel for the Find Resonances algorithm.

    Intended to be instantiated once by ``NetworkAnalysisPanel`` and
    shown/raised when the user clicks "⚙ Find Resonances Settings".  Values
    persist across Periscope sessions via :mod:`tools.periscope.settings`.

    Call :meth:`get_settings` to retrieve the current parameter dict, which
    can be passed directly to ``fitting.find_resonances()`` as keyword
    arguments.

    Call :meth:`update_amplitude_count` whenever the loaded network analysis
    data changes so that the index spinbox range is clamped to the actual
    number of amplitude iterations available.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find Resonances Settings")
        # Float as a real top-level window
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowCloseButtonHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setMinimumWidth(340)
        self._setup_ui()
        self._load_settings()

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Amplitude iteration to search: (TOP) ──────────────────────────────
        amp_group = QGroupBox("Amplitude iteration to search:")
        amp_layout = QVBoxLayout()
        amp_layout.setSpacing(4)

        self._amp_mode_btn_group = QButtonGroup(self)

        # Option 1: Last amplitude (default — preserves existing behaviour)
        self.amp_last_rb = QRadioButton("Last amplitude")
        self.amp_last_rb.setChecked(True)
        self.amp_last_rb.setToolTip(
            "Use the last (highest-index) amplitude iteration from the\n"
            "network analysis file.  This is the default behaviour."
        )
        self._amp_mode_btn_group.addButton(self.amp_last_rb, 0)
        amp_layout.addWidget(self.amp_last_rb)

        # Option 2: By sorted index
        index_row = QWidget()
        index_row_layout = QHBoxLayout(index_row)
        index_row_layout.setContentsMargins(0, 0, 0, 0)
        index_row_layout.setSpacing(4)

        self.amp_index_rb = QRadioButton("Amplitude index:")
        self.amp_index_rb.setToolTip(
            "Use the amplitude iteration at the given 0-based position when\n"
            "amplitudes are sorted from lowest to highest.\n"
            "Index 0 = lowest amplitude, 1 = next lowest, etc.\n"
            "The maximum is constrained to the iterations present in the\n"
            "loaded network analysis file."
        )
        self._amp_mode_btn_group.addButton(self.amp_index_rb, 1)
        index_row_layout.addWidget(self.amp_index_rb)

        self.amp_index_spin = QSpinBox()
        self.amp_index_spin.setRange(0, 0)  # max updated via update_amplitude_count()
        self.amp_index_spin.setValue(0)
        self.amp_index_spin.setSuffix("  (0 = lowest)")
        self.amp_index_spin.setToolTip(
            "0-based index into the sorted amplitude list (0 = lowest amplitude).\n"
            "The maximum is updated to match the number of iterations in the\n"
            "loaded file each time the settings panel is opened."
        )
        self.amp_index_spin.setEnabled(False)  # only active when its radio button is checked
        index_row_layout.addWidget(self.amp_index_spin)
        index_row_layout.addStretch(1)
        amp_layout.addWidget(index_row)

        amp_group.setLayout(amp_layout)
        layout.addWidget(amp_group)

        # ── Detection criteria ────────────────────────────────────────────────
        detect_group = QGroupBox("Detection Criteria")
        detect_layout = QFormLayout()
        detect_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        # Expected resonances (optional — 0 means "not specified")
        self.expected_resonances_spin = QSpinBox()
        self.expected_resonances_spin.setRange(0, 10000)
        self.expected_resonances_spin.setSpecialValueText("Auto (not specified)")
        self.expected_resonances_spin.setValue(0)
        self.expected_resonances_spin.setToolTip(
            "Expected number of resonances to find.  Set to 0 (Auto) to let\n"
            "the algorithm decide.  When specified the algorithm will stop\n"
            "searching once this many resonances have been identified."
        )
        detect_layout.addRow("Expected Resonances:", self.expected_resonances_spin)

        self.min_dip_depth_spin = QDoubleSpinBox()
        self.min_dip_depth_spin.setRange(0.01, 100.0)
        self.min_dip_depth_spin.setSingleStep(0.5)
        self.min_dip_depth_spin.setDecimals(2)
        self.min_dip_depth_spin.setSuffix(" dB")
        self.min_dip_depth_spin.setValue(2.0)
        self.min_dip_depth_spin.setToolTip(
            "Minimum dip depth a candidate feature must have in the magnitude\n"
            "spectrum to be considered a resonance."
        )
        detect_layout.addRow("Min Dip Depth:", self.min_dip_depth_spin)

        self.min_separation_spin = QDoubleSpinBox()
        self.min_separation_spin.setRange(0.001, 100000.0)
        self.min_separation_spin.setSingleStep(10.0)
        self.min_separation_spin.setDecimals(3)
        self.min_separation_spin.setSuffix(" kHz")
        self.min_separation_spin.setValue(10.0)
        self.min_separation_spin.setToolTip(
            "Minimum frequency separation between two adjacent resonances\n"
            "(displayed in kHz; stored internally in Hz)."
        )
        detect_layout.addRow("Min Separation:", self.min_separation_spin)

        detect_group.setLayout(detect_layout)
        layout.addWidget(detect_group)

        # ── Quality factor bounds ─────────────────────────────────────────────
        q_group = QGroupBox("Quality Factor Bounds")
        q_layout = QFormLayout()
        q_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        self.min_q_spin = QDoubleSpinBox()
        self.min_q_spin.setRange(1.0, 1e10)
        self.min_q_spin.setSingleStep(1000.0)
        self.min_q_spin.setDecimals(0)
        self.min_q_spin.setValue(1e4)
        self.min_q_spin.setToolTip(
            "Minimum loaded quality factor Qr for a candidate resonance to\n"
            "be accepted."
        )
        q_layout.addRow("Min Q:", self.min_q_spin)

        self.max_q_spin = QDoubleSpinBox()
        self.max_q_spin.setRange(1.0, 1e10)
        self.max_q_spin.setSingleStep(100000.0)
        self.max_q_spin.setDecimals(0)
        self.max_q_spin.setValue(1e7)
        self.max_q_spin.setToolTip(
            "Maximum loaded quality factor Qr for a candidate resonance to\n"
            "be accepted."
        )
        q_layout.addRow("Max Q:", self.max_q_spin)

        q_group.setLayout(q_layout)
        layout.addWidget(q_group)

        # ── Algorithm tuning ──────────────────────────────────────────────────
        algo_group = QGroupBox("Algorithm Tuning")
        algo_layout = QFormLayout()
        algo_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        self.data_exponent_spin = QDoubleSpinBox()
        self.data_exponent_spin.setRange(0.1, 10.0)
        self.data_exponent_spin.setSingleStep(0.5)
        self.data_exponent_spin.setDecimals(2)
        self.data_exponent_spin.setValue(2.0)
        self.data_exponent_spin.setToolTip(
            "Exponent applied to the magnitude data before peak-finding.\n"
            "Higher values emphasise deep, narrow dips."
        )
        algo_layout.addRow("Data Exponent:", self.data_exponent_spin)

        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)

        # ── Auto-save on change ───────────────────────────────────────────────
        self.amp_last_rb.toggled.connect(self._on_amp_mode_changed)
        self.amp_index_rb.toggled.connect(self._on_amp_mode_changed)
        self.amp_index_spin.valueChanged.connect(self._save_settings)
        self.expected_resonances_spin.valueChanged.connect(self._save_settings)
        self.min_dip_depth_spin.valueChanged.connect(self._save_settings)
        self.min_separation_spin.valueChanged.connect(self._save_settings)
        self.min_q_spin.valueChanged.connect(self._save_settings)
        self.max_q_spin.valueChanged.connect(self._save_settings)
        self.data_exponent_spin.valueChanged.connect(self._save_settings)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_defaults)
        btn_layout.addWidget(reset_btn)
        btn_layout.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.hide)
        close_btn.setDefault(True)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    # ── Event handling ────────────────────────────────────────────────────────

    def _on_amp_mode_changed(self):
        """Enable/disable the index spinbox based on the active radio button."""
        self.amp_index_spin.setEnabled(self.amp_index_rb.isChecked())
        self._save_settings()

    def keyPressEvent(self, event):
        """Close the panel when Enter or Return is pressed."""
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.hide()
        else:
            super().keyPressEvent(event)

    # ── Public API ────────────────────────────────────────────────────────────

    def update_amplitude_count(self, n: int) -> None:
        """
        Update the index spinbox so the user can only select valid indices.

        Should be called by :class:`NetworkAnalysisPanel` every time the
        settings panel is shown, passing the number of amplitude iterations
        that are actually present in the loaded network analysis file for the
        currently active module.

        Args:
            n: Number of available amplitude iterations (≥ 0).  When 0 or 1
               the spinbox maximum is set to 0 so only index 0 is selectable.
        """
        max_index = max(0, n - 1)
        # Block signals so we don't fire _save_settings mid-update
        self.amp_index_spin.blockSignals(True)
        self.amp_index_spin.setMaximum(max_index)
        # Update suffix to show the valid range
        if n > 1:
            self.amp_index_spin.setSuffix(f"  (0–{max_index}; 0 = lowest)")
        else:
            self.amp_index_spin.setSuffix("  (0 = lowest)")
        # Clamp the stored value into the valid range
        if self.amp_index_spin.value() > max_index:
            self.amp_index_spin.setValue(max_index)
        self.amp_index_spin.blockSignals(False)

    def get_settings(self) -> dict:
        """
        Return the current settings as a dict suitable for passing to
        ``fitting.find_resonances()`` as keyword arguments (plus amplitude
        iteration control keys consumed by the panel caller before forwarding).

        Keys:

        * ``expected_resonances`` (int or None) — None when set to 0 / "Auto"
        * ``min_dip_depth_db`` (float)
        * ``min_Q`` (float)
        * ``max_Q`` (float)
        * ``min_resonance_separation_hz`` (float) — converted from the kHz spinbox
        * ``data_exponent`` (float)
        * ``find_resonances_amplitude_mode`` (str) — ``'last'`` or ``'index'``
        * ``find_resonances_amplitude_index`` (int) — 0-based sorted index
          (only relevant when mode is ``'index'``)
        """
        expected = self.expected_resonances_spin.value()
        mode = 'index' if self.amp_index_rb.isChecked() else 'last'
        return {
            'expected_resonances':               expected if expected > 0 else None,
            'min_dip_depth_db':                  self.min_dip_depth_spin.value(),
            'min_Q':                             self.min_q_spin.value(),
            'max_Q':                             self.max_q_spin.value(),
            'min_resonance_separation_hz':       self.min_separation_spin.value() * 1e3,
            'data_exponent':                     self.data_exponent_spin.value(),
            'find_resonances_amplitude_mode':    mode,
            'find_resonances_amplitude_index':   self.amp_index_spin.value(),
        }

    # Alias so that any generic parameter extractor can find get_parameters.
    get_parameters = get_settings

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_settings(self):
        """Load previously saved settings from QSettings."""
        saved = periscope_settings.get_find_resonances_defaults()

        # Block signals while restoring values to avoid triggering _save_settings
        for widget in (
            self.expected_resonances_spin,
            self.min_dip_depth_spin,
            self.min_separation_spin,
            self.min_q_spin,
            self.max_q_spin,
            self.data_exponent_spin,
            self.amp_last_rb,
            self.amp_index_rb,
            self.amp_index_spin,
        ):
            widget.blockSignals(True)

        # expected_resonances: None → 0 (special "Auto" value)
        expected = saved.get('expected_resonances')
        self.expected_resonances_spin.setValue(expected if expected is not None else 0)

        self.min_dip_depth_spin.setValue(saved.get('min_dip_depth_db', 2.0))

        # min_resonance_separation stored in Hz; display in kHz
        sep_hz = saved.get('min_resonance_separation_hz', 1e4)
        self.min_separation_spin.setValue(sep_hz / 1e3)

        self.min_q_spin.setValue(saved.get('min_Q', 1e4))
        self.max_q_spin.setValue(saved.get('max_Q', 1e7))
        self.data_exponent_spin.setValue(saved.get('data_exponent', 2.0))

        # Amplitude iteration mode
        amp_mode = saved.get('find_resonances_amplitude_mode', 'last')
        if amp_mode == 'index':
            self.amp_index_rb.setChecked(True)
        else:
            self.amp_last_rb.setChecked(True)
        # Note: setMaximum is NOT restored from settings here — it is updated
        # dynamically by update_amplitude_count() when the panel is shown.
        self.amp_index_spin.setValue(int(saved.get('find_resonances_amplitude_index', 0)))
        self.amp_index_spin.setEnabled(amp_mode == 'index')

        for widget in (
            self.expected_resonances_spin,
            self.min_dip_depth_spin,
            self.min_separation_spin,
            self.min_q_spin,
            self.max_q_spin,
            self.data_exponent_spin,
            self.amp_last_rb,
            self.amp_index_rb,
            self.amp_index_spin,
        ):
            widget.blockSignals(False)

    def _save_settings(self):
        """Persist current settings to QSettings (called on every control change)."""
        periscope_settings.set_find_resonances_defaults(self.get_settings())

    def _reset_defaults(self):
        """Restore all controls to their default values and persist."""
        defaults = periscope_settings._FIND_RESONANCES_DEFAULTS

        # Block signals during reset to avoid multiple intermediate saves
        for widget in (
            self.expected_resonances_spin,
            self.min_dip_depth_spin,
            self.min_separation_spin,
            self.min_q_spin,
            self.max_q_spin,
            self.data_exponent_spin,
            self.amp_last_rb,
            self.amp_index_rb,
            self.amp_index_spin,
        ):
            widget.blockSignals(True)

        self.expected_resonances_spin.setValue(
            defaults.get('expected_resonances') or 0
        )
        self.min_dip_depth_spin.setValue(defaults.get('min_dip_depth_db', 2.0))
        self.min_separation_spin.setValue(
            defaults.get('min_resonance_separation_hz', 1e4) / 1e3
        )
        self.min_q_spin.setValue(defaults.get('min_Q', 1e4))
        self.max_q_spin.setValue(defaults.get('max_Q', 1e7))
        self.data_exponent_spin.setValue(defaults.get('data_exponent', 2.0))

        # Amplitude iteration: default to 'last' / index 0
        self.amp_last_rb.setChecked(True)
        self.amp_index_spin.setValue(0)
        self.amp_index_spin.setEnabled(False)

        for widget in (
            self.expected_resonances_spin,
            self.min_dip_depth_spin,
            self.min_separation_spin,
            self.min_q_spin,
            self.max_q_spin,
            self.data_exponent_spin,
            self.amp_last_rb,
            self.amp_index_rb,
            self.amp_index_spin,
        ):
            widget.blockSignals(False)

        # One explicit save after all controls have been reset
        self._save_settings()
