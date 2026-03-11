"""
Persistent settings panel for the bias-finding algorithm.

``BiasSettingsPanel`` is a non-modal ``QWidget`` (not a ``QDialog``) that
retains its values between "Find Bias" runs.  Open it from the "⚙ Bias
Settings" button on the MultisweepPanel toolbar; it appears as a small
floating window and can be dismissed without disrupting ongoing work.

``get_settings()`` returns a dict that can be passed directly to
``FindBiasTask`` as *bias_settings*.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QLabel,
    QDoubleSpinBox, QCheckBox, QButtonGroup, QRadioButton,
    QPushButton, QHBoxLayout,
)
from PyQt6.QtCore import Qt


class BiasSettingsPanel(QWidget):
    """
    Persistent, non-modal settings panel for the bias-finding algorithm.

    Intended to be instantiated once by ``MultisweepPanel`` and shown/raised
    when the user clicks "⚙ Bias Settings".  Values persist for the lifetime
    of the panel.

    Call :meth:`get_settings` to retrieve the current parameter dict, which
    can be passed directly to :class:`~rfmux.tools.periscope.tasks.FindBiasTask`
    as *bias_settings*.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bias Finding Settings")
        # Float as a real top-level window when shown without a parent holding it
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowCloseButtonHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setMinimumWidth(380)
        self._setup_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Bifurcation detection ─────────────────────────────────────────────
        bifurc_group = QGroupBox("Bifurcation Detection")
        bifurc_layout = QFormLayout()
        bifurc_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        self.spike_prominence_spin = QDoubleSpinBox()
        self.spike_prominence_spin.setRange(0.5, 20.0)
        self.spike_prominence_spin.setSingleStep(0.5)
        self.spike_prominence_spin.setDecimals(1)
        self.spike_prominence_spin.setValue(2.0)
        self.spike_prominence_spin.setToolTip(
            "Bifurcation spike prominence threshold =\n"
            "  range(dist) / spike_prominence_factor\n"
            "Larger → less sensitive."
        )
        bifurc_layout.addRow("Spike Prominence Factor:", self.spike_prominence_spin)

        self.spike_height_spin = QDoubleSpinBox()
        self.spike_height_spin.setRange(0.5, 20.0)
        self.spike_height_spin.setSingleStep(0.5)
        self.spike_height_spin.setDecimals(1)
        self.spike_height_spin.setValue(3.0)
        self.spike_height_spin.setToolTip(
            "Bifurcation spike height threshold =\n"
            "  spike_height_factor × std(distdiff)\n"
            "Larger → less sensitive."
        )
        bifurc_layout.addRow("Spike Height Factor:", self.spike_height_spin)

        self.fallback_cb = QCheckBox("Fallback to Highest Amplitude")
        self.fallback_cb.setChecked(True)
        self.fallback_cb.setToolTip(
            "If no bifurcation is found across all amplitudes,\n"
            "use the highest available amplitude as the bias point.\n"
            "If unchecked, resonators without a detected bifurcation\n"
            "are skipped (bias_found = False)."
        )
        bifurc_layout.addRow("", self.fallback_cb)

        bifurc_group.setLayout(bifurc_layout)
        layout.addWidget(bifurc_group)

        # ── Bias frequency refinement ─────────────────────────────────────────
        freq_group = QGroupBox("Bias Frequency Refinement")
        freq_layout = QFormLayout()
        freq_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        self.max_deriv_dist_spin = QDoubleSpinBox()
        self.max_deriv_dist_spin.setRange(1.0, 5000.0)
        self.max_deriv_dist_spin.setSingleStep(10.0)
        self.max_deriv_dist_spin.setDecimals(0)
        self.max_deriv_dist_spin.setValue(100.0)
        self.max_deriv_dist_spin.setSuffix(" kHz")
        self.max_deriv_dist_spin.setToolTip(
            "Maximum distance (kHz) that the max-derivative point\n"
            "may lie from the reference frequency before it is\n"
            "rejected as an outlier and the reference is used instead."
        )
        freq_layout.addRow("Max Deriv Distance:", self.max_deriv_dist_spin)

        # Reference frequency source radio buttons
        ref_label = QLabel("Reference Frequency Source:")
        freq_layout.addRow(ref_label)

        self._ref_freq_group = QButtonGroup(self)
        radio_layout = QVBoxLayout()
        radio_layout.setSpacing(2)

        self.rb_bias_freq = QRadioButton("Bias frequency (res_info_dict)")
        self.rb_bias_freq.setChecked(True)
        self.rb_bias_freq.setToolTip(
            "Use the existing bias_frequency stored in res_info_dict\n"
            "(i.e. the sweep-centre frequency set during the multisweep run)."
        )
        self._ref_freq_group.addButton(self.rb_bias_freq)
        radio_layout.addWidget(self.rb_bias_freq)

        self.rb_fit_fr = QRadioButton("Fitted resonance frequency (fr)")
        self.rb_fit_fr.setToolTip(
            "Use the fitted fr from fit_params or nonlinear_fit_params\n"
            "stored in the sweep entry. Falls back to sweep_center_frequency\n"
            "when no fit result is available."
        )
        self._ref_freq_group.addButton(self.rb_fit_fr)
        radio_layout.addWidget(self.rb_fit_fr)

        self.rb_sweep_center = QRadioButton("Sweep centre frequency")
        self.rb_sweep_center.setToolTip(
            "Always use sweep_center_frequency from the sweep entry\n"
            "(equivalent to the original multisweep centre)."
        )
        self._ref_freq_group.addButton(self.rb_sweep_center)
        radio_layout.addWidget(self.rb_sweep_center)

        freq_layout.addRow(radio_layout)
        freq_group.setLayout(freq_layout)
        layout.addWidget(freq_group)

        # ── Diagnostic fit ────────────────────────────────────────────────────
        fit_group = QGroupBox("Diagnostic Nonlinear Fit")
        fit_layout = QFormLayout()

        self.fit_selected_cb = QCheckBox("Fit Selected Amplitude")
        self.fit_selected_cb.setChecked(True)
        self.fit_selected_cb.setToolTip(
            "After selecting the optimal (non-bifurcated) amplitude,\n"
            "run the nonlinear resonator fitter on that sweep as a\n"
            "diagnostic.  The fit is skipped if the selected entry is\n"
            "itself bifurcated."
        )
        fit_layout.addRow("", self.fit_selected_cb)
        fit_group.setLayout(fit_layout)
        layout.addWidget(fit_group)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_defaults)
        btn_layout.addWidget(reset_btn)
        btn_layout.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.hide)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_settings(self) -> dict:
        """
        Return the current settings as a dict suitable for passing to
        :class:`~rfmux.tools.periscope.tasks.FindBiasTask` as *bias_settings*.

        Keys:

        * ``spike_prominence_factor`` (float)
        * ``spike_height_factor`` (float)
        * ``max_deriv_distance_hz`` (float) — converted from kHz to Hz
        * ``fallback_to_highest`` (bool)
        * ``reference_freq_source`` (str) — one of
          ``"bias_frequency"``, ``"fit_fr"``, ``"sweep_center"``
        * ``fit_selected_amplitude`` (bool)
        """
        if self.rb_fit_fr.isChecked():
            ref_source = "fit_fr"
        elif self.rb_sweep_center.isChecked():
            ref_source = "sweep_center"
        else:
            ref_source = "bias_frequency"

        return {
            'spike_prominence_factor': self.spike_prominence_spin.value(),
            'spike_height_factor':     self.spike_height_spin.value(),
            'max_deriv_distance_hz':   self.max_deriv_dist_spin.value() * 1e3,
            'fallback_to_highest':     self.fallback_cb.isChecked(),
            'reference_freq_source':   ref_source,
            'fit_selected_amplitude':  self.fit_selected_cb.isChecked(),
        }

    # Alias so that ParamKeyExtractor (which looks for get_parameters) can
    # validate the settings keys against the smoke-test mock dict.
    get_parameters = get_settings

    # ── Private helpers ───────────────────────────────────────────────────────

    def _reset_defaults(self):
        """Restore all controls to their default values."""
        self.spike_prominence_spin.setValue(2.0)
        self.spike_height_spin.setValue(3.0)
        self.max_deriv_dist_spin.setValue(100.0)
        self.fallback_cb.setChecked(True)
        self.rb_bias_freq.setChecked(True)
        self.fit_selected_cb.setChecked(True)


# ── Load Bias Dialog (legacy "Load Bias from file" feature) ───────────────────

import pickle

from PyQt6.QtWidgets import QDialog, QDialogButtonBox


def load_bias_payload(parent, file_path=None):
    """
    Load a bias payload dict from a pickle file.

    If *file_path* is None, opens a file chooser dialog.
    Returns the payload dict on success, or None on failure/cancel.
    """
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    if file_path is None:
        options = QFileDialog.Options()
        options |= QFileDialog.Option.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(
            parent,
            "Load Bias Parameters",
            "",
            "Pickle Files (*.pkl *.pickle);;All Files (*)",
            options=options,
        )

    if not file_path:
        return None

    try:
        with open(file_path, "rb") as fh:
            payload = pickle.load(fh)
    except Exception as exc:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(parent, "Load Failed",
                             f"Could not read '{file_path}':\n{exc}")
        return None

    if (
        isinstance(payload, dict)
        and isinstance(payload.get("initial_parameters"), dict)
        and payload.get("bias_kids_output") is not None
    ):
        return payload

    from PyQt6.QtWidgets import QMessageBox
    QMessageBox.warning(parent, "Invalid File",
                        "The selected file does not contain Bias parameters.")
    return None


class LoadBiasDialog(QDialog):
    """
    Dialog for loading a previously saved bias configuration from a pickle file
    and applying it directly to the hardware, bypassing the Find/Apply Bias workflow.

    This is a legacy "Load Bias" feature accessible from the main Periscope window.
    """

    def __init__(self, parent=None, active_module=None, loaded_data=None):
        super().__init__(parent)
        self.setWindowTitle("Load Bias")
        self.setModal(True)
        self.setMinimumWidth(500)
        self._load_data = loaded_data or {}
        self.use_load_file = False
        self.current_module = active_module
        self._setup_ui()

    def _setup_ui(self):
        from PyQt6.QtWidgets import (
            QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
        )
        from PyQt6.QtCore import Qt, QTimer

        layout = QVBoxLayout(self)

        self.import_button = QPushButton("Import File")
        self.import_button.clicked.connect(self._import_file)
        layout.addWidget(self.import_button, alignment=Qt.AlignmentFlag.AlignTop)

        tones_layout = QHBoxLayout()
        tones_layout.addWidget(QLabel("Tones (MHz):"))
        self.tones_edit = QLineEdit()
        tones_layout.addWidget(self.tones_edit)
        layout.addLayout(tones_layout)

        amp_layout = QHBoxLayout()
        amp_layout.addWidget(QLabel("Sweep Amplitude (normalized):"))
        self.amp_edit = QLineEdit()
        amp_layout.addWidget(self.amp_edit)
        layout.addLayout(amp_layout)

        bias_layout = QHBoxLayout()
        self.set_bias_btn = QPushButton("Set Bias")
        self.plot_bias_btn = QPushButton("Set + Plot Bias")
        self.plot_bias_btn.setEnabled(False)
        self.cancel_btn = QPushButton("Cancel")
        bias_layout.addWidget(self.set_bias_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        bias_layout.addWidget(self.plot_bias_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        bias_layout.addWidget(self.cancel_btn, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addLayout(bias_layout)

        self.set_bias_btn.clicked.connect(self.accept)
        self.plot_bias_btn.clicked.connect(self._on_set_and_plot_bias)
        self.cancel_btn.clicked.connect(self.reject)

        if self._load_data:
            self._populate_fields_from_data(self._load_data)
            self.plot_bias_btn.setEnabled(True)
            self.use_load_file = True

    def _populate_fields_from_data(self, payload):
        params = payload.get('initial_parameters', {})
        bias_output = payload.get('bias_kids_output', {})
        bias_freqs, amplitudes = [], []
        for det_data in bias_output.values():
            bias_freq = det_data.get("bias_frequency") or det_data.get("original_center_frequency")
            bias_freqs.append(bias_freq)
            amplitudes.append(det_data.get("sweep_amplitude"))
        self.tones_edit.setText(",".join([f"{f/1e6:.6f}" for f in bias_freqs if f]))
        self.amp_edit.setText(",".join([f"{a:.3f}" for a in amplitudes if a is not None]))

    def _on_set_and_plot_bias(self):
        self.use_load_file = True
        self.accept()

    def _import_file(self):
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self._open_file_dialog_async)

    def _open_file_dialog_async(self):
        from PyQt6.QtWidgets import QFileDialog
        from PyQt6.QtCore import Qt
        if not hasattr(self, "_file_dialog") or self._file_dialog is None:
            self._file_dialog = QFileDialog(self, "Load Bias Parameters")
            self._file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            self._file_dialog.setNameFilters(["Pickle Files (*.pkl *.pickle)", "All Files (*)"])
            self._file_dialog.setOptions(
                QFileDialog.Option.DontUseNativeDialog | QFileDialog.Option.ReadOnly
            )
            self._file_dialog.setModal(False)
            self._file_dialog.fileSelected.connect(self._on_file_selected)
        self._file_dialog.open()

    def _on_file_selected(self, path):
        payload = load_bias_payload(self, file_path=path)
        if payload is None:
            return
        self.plot_bias_btn.setEnabled(True)
        self._load_data = payload.copy()
        self._populate_fields_from_data(payload)

    def get_load_param(self):
        """Return the loaded payload dict or manually entered parameters."""
        if self.use_load_file:
            return self._load_data
        try:
            amp_text = [x.strip() for x in self.amp_edit.text().split(',')]
            tone_text = [t.strip() for t in self.tones_edit.text().split(',')]
            params = {
                'module': self.current_module,
                'phases': [0.0] * len(amp_text),
                'amplitudes': [float(a) for a in amp_text],
                'bias_frequencies': [float(t) * 1e6 for t in tone_text],
            }
            from PyQt6.QtWidgets import QMessageBox
            if params['module'] is None:
                QMessageBox.warning(self, "Validation Error", "No module identified.")
                return None
            if len(params['amplitudes']) != len(params['bias_frequencies']):
                QMessageBox.warning(self, "Validation Error",
                                    "Number of amplitudes and frequencies must match.")
                return None
            return params
        except (ValueError, Exception) as exc:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Input Error", f"Invalid input: {exc}")
            return None
