"""
Persistent settings panel for the resonator fitting algorithms.

``FitSettingsPanel`` is a non-modal ``QWidget`` that retains its values between
"Run Fit" invocations and across Periscope sessions (via QSettings).  Open it
from the "⚙ Fit Settings" button on the MultisweepPanel toolbar; it appears as
a small floating window and can be dismissed without disrupting ongoing work.

``get_settings()`` returns a dict that can be passed directly to
``RunFitsTask`` as *fit_settings*.

**"Amplitude to Fit"** group: choose which amplitude iterations
``RunFitsTask`` should actually fit — all amplitudes (default), a
specific amplitude by sorted index, or the bias amplitude found by
Find Bias.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QCheckBox, QPushButton, QRadioButton, QSpinBox, QLabel,
    QButtonGroup,
)
from PyQt6.QtCore import Qt

from . import settings as periscope_settings


class FitSettingsPanel(QWidget):
    """
    Persistent, non-modal settings panel for the resonator fitting algorithms.

    Intended to be instantiated once by ``MultisweepPanel`` and shown/raised
    when the user clicks "⚙ Fit Settings".  Values persist across Periscope
    sessions via :mod:`tools.periscope.settings`.

    Call :meth:`get_settings` to retrieve the current parameter dict, which
    can be passed directly to :class:`~rfmux.tools.periscope.tasks.RunFitsTask`
    as *fit_settings*.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fit Settings")
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

        # ── Fit selection ─────────────────────────────────────────────────────
        fits_group = QGroupBox("Fits to Apply")
        fits_layout = QFormLayout()
        fits_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        self.skewed_fit_cb = QCheckBox("Apply Skewed Lorentzian Fit")
        self.skewed_fit_cb.setChecked(True)
        self.skewed_fit_cb.setToolTip(
            "Fit a skewed Lorentzian (asymmetric resonance model) to the\n"
            "magnitude of each sweep trace and extract fr, Qr, Qc, Qi."
        )
        fits_layout.addRow(self.skewed_fit_cb)

        self.nonlinear_fit_cb = QCheckBox("Apply Nonlinear IQ Fit")
        self.nonlinear_fit_cb.setChecked(True)
        self.nonlinear_fit_cb.setToolTip(
            "Fit the CITKID-style nonlinear resonator model to the complex\n"
            "IQ data of each sweep trace and extract fr, Qr, amp, phi, a."
        )
        fits_layout.addRow(self.nonlinear_fit_cb)

        fits_group.setLayout(fits_layout)
        layout.addWidget(fits_group)

        # ── Amplitude to Fit ──────────────────────────────────────────────────
        amp_group = QGroupBox("Amplitude to Fit")
        amp_layout = QVBoxLayout()
        amp_layout.setSpacing(4)

        self._amp_mode_btn_group = QButtonGroup(self)

        # Option 1: All amplitudes (default)
        self.amp_all_rb = QRadioButton("All amplitudes")
        self.amp_all_rb.setChecked(True)
        self.amp_all_rb.setToolTip(
            "Fit every amplitude iteration (current behaviour)."
        )
        self._amp_mode_btn_group.addButton(self.amp_all_rb, 0)
        amp_layout.addWidget(self.amp_all_rb)

        # Option 2: By sorted index
        index_row = QWidget()
        index_row_layout = QHBoxLayout(index_row)
        index_row_layout.setContentsMargins(0, 0, 0, 0)
        index_row_layout.setSpacing(4)

        self.amp_index_rb = QRadioButton("Amplitude index:")
        self.amp_index_rb.setToolTip(
            "Fit only the iteration at the given 0-based position when\n"
            "amplitudes are sorted from lowest to highest.\n"
            "Index 0 = lowest amplitude, 1 = next, etc."
        )
        self._amp_mode_btn_group.addButton(self.amp_index_rb, 1)
        index_row_layout.addWidget(self.amp_index_rb)

        self.amp_index_spin = QSpinBox()
        self.amp_index_spin.setRange(0, 99)
        self.amp_index_spin.setValue(0)
        self.amp_index_spin.setSuffix("  (0 = lowest)")
        self.amp_index_spin.setToolTip(
            "0-based index into the sorted amplitude list.\n"
            "Clamped to the last available index if out of range."
        )
        self.amp_index_spin.setEnabled(False)  # only active when its radio button is checked
        index_row_layout.addWidget(self.amp_index_spin)
        index_row_layout.addStretch(1)

        amp_layout.addWidget(index_row)

        # Option 3: Bias amplitude only
        self.amp_bias_rb = QRadioButton("Bias amplitude only")
        self.amp_bias_rb.setToolTip(
            "Fit only the iteration whose amplitude matches the bias\n"
            "amplitude stored in res_info_dict for each resonator.\n"
            "Requires Find Bias to have been run first."
        )
        self._amp_mode_btn_group.addButton(self.amp_bias_rb, 2)
        amp_layout.addWidget(self.amp_bias_rb)

        amp_group.setLayout(amp_layout)
        layout.addWidget(amp_group)

        # ── Auto-save on change ───────────────────────────────────────────────
        self.skewed_fit_cb.toggled.connect(self._save_settings)
        self.nonlinear_fit_cb.toggled.connect(self._save_settings)
        self.amp_all_rb.toggled.connect(self._on_amp_mode_changed)
        self.amp_index_rb.toggled.connect(self._on_amp_mode_changed)
        self.amp_bias_rb.toggled.connect(self._on_amp_mode_changed)
        self.amp_index_spin.valueChanged.connect(self._save_settings)

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

    def get_settings(self) -> dict:
        """
        Return the current settings as a dict suitable for passing to
        :class:`~rfmux.tools.periscope.tasks.RunFitsTask` as *fit_settings*.

        Keys
        ----
        * ``apply_skewed_fit`` (bool)
        * ``apply_nonlinear_fit`` (bool)
        * ``fit_run_amplitude_mode`` (str): ``'all'``, ``'index'``, or ``'bias'``
        * ``fit_run_amplitude_index`` (int): 0-based index (only relevant when
          ``fit_run_amplitude_mode == 'index'``)
        """
        if self.amp_index_rb.isChecked():
            mode = 'index'
        elif self.amp_bias_rb.isChecked():
            mode = 'bias'
        else:
            mode = 'all'

        return {
            'apply_skewed_fit':        self.skewed_fit_cb.isChecked(),
            'apply_nonlinear_fit':     self.nonlinear_fit_cb.isChecked(),
            'fit_run_amplitude_mode':  mode,
            'fit_run_amplitude_index': self.amp_index_spin.value(),
        }

    # Alias so that any generic parameter extractor can find get_parameters.
    get_parameters = get_settings

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_settings(self):
        """Load previously saved settings from QSettings."""
        saved = periscope_settings.get_fit_defaults()

        # Block all signals while we restore values to avoid triggering _save_settings
        for widget in (
            self.skewed_fit_cb, self.nonlinear_fit_cb,
            self.amp_all_rb, self.amp_index_rb, self.amp_bias_rb,
            self.amp_index_spin,
        ):
            widget.blockSignals(True)

        self.skewed_fit_cb.setChecked(saved.get('apply_skewed_fit', True))
        self.nonlinear_fit_cb.setChecked(saved.get('apply_nonlinear_fit', True))

        mode = saved.get('fit_run_amplitude_mode', 'all')
        if mode == 'index':
            self.amp_index_rb.setChecked(True)
        elif mode == 'bias':
            self.amp_bias_rb.setChecked(True)
        else:
            self.amp_all_rb.setChecked(True)

        self.amp_index_spin.setValue(int(saved.get('fit_run_amplitude_index', 0)))
        self.amp_index_spin.setEnabled(mode == 'index')

        for widget in (
            self.skewed_fit_cb, self.nonlinear_fit_cb,
            self.amp_all_rb, self.amp_index_rb, self.amp_bias_rb,
            self.amp_index_spin,
        ):
            widget.blockSignals(False)

    def _save_settings(self):
        """Persist current settings to QSettings (called on every control change)."""
        periscope_settings.set_fit_defaults(self.get_settings())

    def _reset_defaults(self):
        """Restore all controls to their default values and persist."""
        # Block signals during reset to avoid multiple intermediate saves
        for widget in (
            self.skewed_fit_cb, self.nonlinear_fit_cb,
            self.amp_all_rb, self.amp_index_rb, self.amp_bias_rb,
            self.amp_index_spin,
        ):
            widget.blockSignals(True)

        self.skewed_fit_cb.setChecked(True)
        self.nonlinear_fit_cb.setChecked(True)
        self.amp_all_rb.setChecked(True)
        self.amp_index_spin.setValue(0)
        self.amp_index_spin.setEnabled(False)

        for widget in (
            self.skewed_fit_cb, self.nonlinear_fit_cb,
            self.amp_all_rb, self.amp_index_rb, self.amp_bias_rb,
            self.amp_index_spin,
        ):
            widget.blockSignals(False)

        # One explicit save after all controls have been reset
        self._save_settings()
