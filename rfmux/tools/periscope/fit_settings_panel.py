"""
Persistent settings panel for the resonator fitting algorithms.

``FitSettingsPanel`` is a non-modal ``QWidget`` that retains its values between
"Run Fit" invocations and across Periscope sessions (via QSettings).  Open it
from the "⚙ Fit Settings" button on the MultisweepPanel toolbar; it appears as
a small floating window and can be dismissed without disrupting ongoing work.

``get_settings()`` returns a dict that can be passed directly to
``RunFitsTask`` as *fit_settings*.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QCheckBox, QPushButton, QHBoxLayout,
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
        self.setMinimumWidth(300)
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

        # ── Auto-save on change ───────────────────────────────────────────────
        # Settings are persisted immediately when a checkbox is toggled, so
        # they are preserved even when the panel is closed via the X button.
        self.skewed_fit_cb.toggled.connect(self._save_settings)
        self.nonlinear_fit_cb.toggled.connect(self._save_settings)

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

        Keys:

        * ``apply_skewed_fit`` (bool)
        * ``apply_nonlinear_fit`` (bool)
        """
        return {
            'apply_skewed_fit':    self.skewed_fit_cb.isChecked(),
            'apply_nonlinear_fit': self.nonlinear_fit_cb.isChecked(),
        }

    # Alias so that any generic parameter extractor can find get_parameters.
    get_parameters = get_settings

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_settings(self):
        """Load previously saved settings from QSettings."""
        saved = periscope_settings.get_fit_defaults()
        # Block signals while we restore values to avoid triggering _save_settings
        self.skewed_fit_cb.blockSignals(True)
        self.nonlinear_fit_cb.blockSignals(True)
        self.skewed_fit_cb.setChecked(saved.get('apply_skewed_fit', True))
        self.nonlinear_fit_cb.setChecked(saved.get('apply_nonlinear_fit', True))
        self.skewed_fit_cb.blockSignals(False)
        self.nonlinear_fit_cb.blockSignals(False)

    def _save_settings(self):
        """Persist current settings to QSettings (called on every checkbox toggle)."""
        periscope_settings.set_fit_defaults(self.get_settings())

    def _reset_defaults(self):
        """Restore all controls to their default values and persist."""
        self.skewed_fit_cb.setChecked(True)
        self.nonlinear_fit_cb.setChecked(True)
        # _save_settings is connected to toggled, so it fires automatically above.
        # Call explicitly in case neither checkbox actually changed state.
        self._save_settings()
