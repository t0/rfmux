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

        bifurc_group.setLayout(bifurc_layout)
        layout.addWidget(bifurc_group)

        # ── Bias frequency refinement ─────────────────────────────────────────
        freq_group = QGroupBox("Bias Frequency Refinement")
        freq_layout = QFormLayout()
        freq_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        # ── Max Deriv Distance: two radio+spinbox pairs ───────────────────
        deriv_container = QWidget()
        deriv_vbox = QVBoxLayout(deriv_container)
        deriv_vbox.setContentsMargins(0, 0, 0, 0)
        deriv_vbox.setSpacing(4)

        self._deriv_mode_group = QButtonGroup(self)

        # Row 1 — Absolute (kHz)
        abs_row = QHBoxLayout()
        abs_row.setContentsMargins(0, 0, 0, 0)
        self.rb_deriv_absolute = QRadioButton("Absolute (kHz)")
        self.rb_deriv_absolute.setChecked(True)
        self._deriv_mode_group.addButton(self.rb_deriv_absolute)
        abs_row.addWidget(self.rb_deriv_absolute)

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
        abs_row.addWidget(self.max_deriv_dist_spin)
        abs_row.addStretch(1)
        deriv_vbox.addLayout(abs_row)

        # Row 2 — Fraction of sweep bandwidth
        frac_row = QHBoxLayout()
        frac_row.setContentsMargins(0, 0, 0, 0)
        self.rb_deriv_fraction = QRadioButton("Fraction of sweep bandwidth")
        self._deriv_mode_group.addButton(self.rb_deriv_fraction)
        frac_row.addWidget(self.rb_deriv_fraction)

        self.max_deriv_frac_spin = QDoubleSpinBox()
        self.max_deriv_frac_spin.setRange(0.001, 1.0)
        self.max_deriv_frac_spin.setSingleStep(0.05)
        self.max_deriv_frac_spin.setDecimals(3)
        self.max_deriv_frac_spin.setValue(0.5)
        self.max_deriv_frac_spin.setEnabled(False)
        self.max_deriv_frac_spin.setToolTip(
            "Maximum distance as a fraction of the sweep bandwidth that\n"
            "the max-derivative point may lie from the reference frequency\n"
            "before it is rejected as an outlier and the reference is used\n"
            "instead.  E.g. 0.5 = ½ × span_hz."
        )
        frac_row.addWidget(self.max_deriv_frac_spin)
        frac_row.addStretch(1)
        deriv_vbox.addLayout(frac_row)

        # Wire radio buttons to enable/disable their spinboxes
        self.rb_deriv_absolute.toggled.connect(self._on_deriv_mode_changed)
        self.rb_deriv_fraction.toggled.connect(self._on_deriv_mode_changed)

        freq_layout.addRow("Max Deriv Distance:", deriv_container)

        # Reference frequency source radio buttons
        ref_label = QLabel("Reference Frequency Source:")
        freq_layout.addRow(ref_label)

        self._ref_freq_group = QButtonGroup(self)
        radio_layout = QVBoxLayout()
        radio_layout.setSpacing(2)

        self.rb_bias_freq = QRadioButton("Pre-existing bias frequency (res_info_dict)")
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
        :class:`~rfmux.tools.periscope.tasks.FindBiasTask` as *bias_settings*.

        Keys:

        * ``spike_prominence_factor`` (float)
        * ``spike_height_factor`` (float)
        * ``max_deriv_distance_mode`` (str) — ``"absolute"`` or ``"fraction"``
        * ``max_deriv_distance_hz`` (float) — set when mode is ``"absolute"``;
          converted from the kHz spin-box value to Hz
        * ``max_deriv_distance_fraction`` (float) — set when mode is
          ``"fraction"``; caller is responsible for multiplying by
          ``span_hz`` to produce the Hz value passed to
          :func:`~rfmux.algorithms.measurement.bias_kids.find_bias_points`
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

        if self.rb_deriv_fraction.isChecked():
            deriv_mode = "fraction"
            deriv_hz = self.max_deriv_dist_spin.value() * 1e3   # kept for compat; caller overrides
            deriv_fraction = self.max_deriv_frac_spin.value()
        else:
            deriv_mode = "absolute"
            deriv_hz = self.max_deriv_dist_spin.value() * 1e3
            deriv_fraction = self.max_deriv_frac_spin.value()   # stored but not used in abs mode

        return {
            'spike_prominence_factor':    self.spike_prominence_spin.value(),
            'spike_height_factor':        self.spike_height_spin.value(),
            'max_deriv_distance_mode':    deriv_mode,
            'max_deriv_distance_hz':      deriv_hz,
            'max_deriv_distance_fraction': deriv_fraction,
            'reference_freq_source':      ref_source,
            'fit_selected_amplitude':     self.fit_selected_cb.isChecked(),
        }

    # Alias so that ParamKeyExtractor (which looks for get_parameters) can
    # validate the settings keys against the smoke-test mock dict.
    get_parameters = get_settings

    # ── Private helpers ───────────────────────────────────────────────────────

    def _on_deriv_mode_changed(self):
        """Enable the spinbox that belongs to the active deriv-distance radio button."""
        absolute = self.rb_deriv_absolute.isChecked()
        self.max_deriv_dist_spin.setEnabled(absolute)
        self.max_deriv_frac_spin.setEnabled(not absolute)

    def _reset_defaults(self):
        """Restore all controls to their default values."""
        self.spike_prominence_spin.setValue(2.0)
        self.spike_height_spin.setValue(3.0)
        self.rb_deriv_absolute.setChecked(True)   # resets mode → absolute
        self.max_deriv_dist_spin.setValue(100.0)
        self.max_deriv_frac_spin.setValue(0.5)
        self.rb_bias_freq.setChecked(True)
        self.fit_selected_cb.setChecked(True)


# ── Bias KIDs Dialog ──────────────────────────────────────────────────────────

import csv
import os
import pickle

from PyQt6.QtWidgets import QDialog


class BiasKIDsDialog(QDialog):
    """
    Dialog for biasing KIDs directly from the main Periscope window.

    Three ways to specify bias parameters:

    1. **Load from multisweep file** — reads ``res_info_dict`` entries where
       ``bias_found = True`` (i.e. Find Bias has been run on the file).
    2. **Load from CSV** — two-column file: ``frequency_mhz``, ``amplitude``.
       Header rows (any row whose first column can't be parsed as a float)
       are silently skipped.
    3. **Manual entry** — type comma-separated values directly into the fields.

    "Apply + Plot" is enabled only when a multisweep pickle was loaded, because
    it requires the full sweep data to construct a :class:`MultisweepPanel`.
    """

    def __init__(self, parent=None, active_module=None):
        super().__init__(parent)
        self.setWindowTitle("Bias KIDs")
        self.setModal(True)
        self.setMinimumWidth(520)
        self.current_module = active_module
        self._load_params = None   # Full pkl payload (set after loading a multisweep file)
        self.use_plot = False      # True when user clicked "Apply + Plot"
        self._setup_ui()

    # ── UI construction ────────────────────────────────────────────────────────

    def _setup_ui(self):
        from PyQt6.QtWidgets import (
            QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
            QGroupBox,
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ── File loading ───────────────────────────────────────────────────────
        file_group = QGroupBox("Load from file")
        file_layout = QVBoxLayout(file_group)

        btn_row = QHBoxLayout()
        self._load_pkl_btn = QPushButton("Load Multisweep file…")
        self._load_pkl_btn.setToolTip(
            "Load bias parameters from a multisweep pickle file.\n"
            "Requires that Find Bias has already been run on the file."
        )
        self._load_pkl_btn.clicked.connect(self._on_load_pkl)
        btn_row.addWidget(self._load_pkl_btn)

        self._load_csv_btn = QPushButton("Load CSV…")
        self._load_csv_btn.setToolTip(
            "Load bias parameters from a CSV file.\n"
            "Expected columns: frequency_mhz, amplitude"
        )
        self._load_csv_btn.clicked.connect(self._on_load_csv)
        btn_row.addWidget(self._load_csv_btn)
        btn_row.addStretch(1)
        file_layout.addLayout(btn_row)

        self._file_status_label = QLabel("No file loaded")
        self._file_status_label.setStyleSheet("color: gray; font-style: italic;")
        file_layout.addWidget(self._file_status_label)

        layout.addWidget(file_group)

        # ── Manual / populated entry ───────────────────────────────────────────
        entry_group = QGroupBox("Bias parameters")
        entry_layout = QVBoxLayout(entry_group)

        entry_layout.addWidget(QLabel("Frequencies (MHz), comma-separated:"))
        self._freq_edit = QLineEdit()
        self._freq_edit.setPlaceholderText("e.g. 500.123, 501.456, 502.789")
        entry_layout.addWidget(self._freq_edit)

        entry_layout.addWidget(QLabel("Amplitudes (normalized), comma-separated:"))
        self._amp_edit = QLineEdit()
        self._amp_edit.setPlaceholderText("e.g. 0.5, 0.5, 0.5")
        entry_layout.addWidget(self._amp_edit)

        layout.addWidget(entry_group)

        # ── Action buttons ─────────────────────────────────────────────────────
        btn_layout = QHBoxLayout()

        self._apply_btn = QPushButton("Apply Bias")
        self._apply_btn.setDefault(True)
        self._apply_btn.setToolTip("Apply the bias parameters to hardware.")
        self._apply_btn.clicked.connect(self._on_apply)
        btn_layout.addWidget(self._apply_btn)

        self._apply_plot_btn = QPushButton("Apply + Plot")
        self._apply_plot_btn.setEnabled(False)
        self._apply_plot_btn.setToolTip(
            "Apply bias and open a MultisweepPanel showing the sweep data.\n"
            "Only available when a multisweep file has been loaded."
        )
        self._apply_plot_btn.clicked.connect(self._on_apply_and_plot)
        btn_layout.addWidget(self._apply_plot_btn)

        btn_layout.addStretch(1)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self._cancel_btn)

        layout.addLayout(btn_layout)

    # ── File loading helpers ───────────────────────────────────────────────────

    def _on_load_pkl(self):
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Multisweep File", "",
            "Pickle Files (*.pkl *.pickle);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            self._load_from_multisweep(path)

    def _load_from_multisweep(self, path: str):
        """Populate the frequency/amplitude fields from a multisweep pickle file."""
        from PyQt6.QtWidgets import QMessageBox

        try:
            with open(path, "rb") as fh:
                payload = pickle.load(fh)
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", f"Could not read '{path}':\n{exc}")
            return

        self._load_from_multisweep_payload(payload, label=os.path.basename(path))

    def _load_from_multisweep_payload(self, payload: dict, label: str = "session file"):
        """
        Populate the frequency/amplitude fields from an already-loaded payload dict.

        This is the internal counterpart to :meth:`_load_from_multisweep` for
        cases where the pickle has already been deserialized by the caller
        (e.g. the session browser pre-loads the data before opening this dialog).

        Parameters
        ----------
        payload :
            The deserialized pickle dict.
        label :
            Short human-readable name to show in the status line (e.g. filename).
        """
        from PyQt6.QtWidgets import QMessageBox

        if not isinstance(payload, dict) or 'initial_parameters' not in payload:
            QMessageBox.warning(
                self, "Invalid File",
                "This does not appear to be a multisweep file\n"
                "(missing 'initial_parameters' key)."
            )
            return

        res_info = payload.get('res_info_dict', {}) or {}
        biased = sorted(
            (
                (code, info) for code, info in res_info.items()
                if isinstance(info, dict) and info.get('bias_found') is True
            ),
            key=lambda ci: ci[1].get('channel_number', 0),
        )

        if not biased:
            QMessageBox.warning(
                self, "No Bias Data",
                "This file does not contain bias results.\n"
                "Please run Find Bias on this sweep first."
            )
            return

        freq_strs = [
            f"{info.get('bias_frequency', 0.0) / 1e6:.6f}"
            for _, info in biased
        ]
        amp_strs = [
            f"{info.get('bias_amplitude', 0.0):.4f}"
            for _, info in biased
        ]

        self._freq_edit.setText(", ".join(freq_strs))
        self._amp_edit.setText(", ".join(amp_strs))
        self._load_params = payload
        self._apply_plot_btn.setEnabled(True)

        self._file_status_label.setText(f"Loaded: {label} ({len(biased)} channels)")
        self._file_status_label.setStyleSheet("color: #2a8a2a;")

    def _on_load_csv(self):
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Load CSV File", "",
            "CSV Files (*.csv *.txt);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            self._load_from_csv(path)

    def _load_from_csv(self, path: str):
        """Populate the frequency/amplitude fields from a two-column CSV file."""
        from PyQt6.QtWidgets import QMessageBox

        freq_strs: list[str] = []
        amp_strs:  list[str] = []

        try:
            with open(path, newline='') as fh:
                reader = csv.reader(fh)
                for row in reader:
                    if len(row) < 2:
                        continue
                    try:
                        f = float(row[0].strip())
                        a = float(row[1].strip())
                        freq_strs.append(f"{f:.6f}")
                        amp_strs.append(f"{a:.4f}")
                    except ValueError:
                        # Skip header rows or unparseable lines silently
                        continue
        except Exception as exc:
            QMessageBox.critical(
                self, "Load Failed", f"Could not read CSV '{path}':\n{exc}"
            )
            return

        if not freq_strs:
            QMessageBox.warning(
                self, "Empty File",
                "No valid data rows found in the CSV file.\n"
                "Expected two columns: frequency_mhz, amplitude"
            )
            return

        self._freq_edit.setText(", ".join(freq_strs))
        self._amp_edit.setText(", ".join(amp_strs))
        # CSV does not carry sweep data — "Apply + Plot" is not available
        self._load_params = None
        self._apply_plot_btn.setEnabled(False)

        self._file_status_label.setText(
            f"Loaded CSV: {os.path.basename(path)} ({len(freq_strs)} channels)"
        )
        self._file_status_label.setStyleSheet("color: #2a8a2a;")

    # ── Button handlers ────────────────────────────────────────────────────────

    def _on_apply(self):
        self.use_plot = False
        self.accept()

    def _on_apply_and_plot(self):
        self.use_plot = True
        self.accept()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_result(self):
        """
        Return validated bias parameters entered or loaded by the user.

        Returns
        -------
        dict or None
            On success::

                {
                    'frequencies_hz': list[float],
                    'amplitudes':     list[float],
                    'load_params':    dict | None,
                }

            ``load_params`` is the full pickle payload when a multisweep file
            was loaded, or ``None`` otherwise (CSV / manual entry).

            Returns ``None`` when validation fails; an error message is shown
            to the user before returning.
        """
        from PyQt6.QtWidgets import QMessageBox

        freq_texts = [t.strip() for t in self._freq_edit.text().split(',') if t.strip()]
        amp_texts  = [t.strip() for t in self._amp_edit.text().split(',') if t.strip()]

        if not freq_texts:
            QMessageBox.warning(self, "Validation Error",
                                "Please enter at least one frequency.")
            return None

        if len(freq_texts) != len(amp_texts):
            QMessageBox.warning(
                self, "Validation Error",
                f"Number of frequencies ({len(freq_texts)}) and "
                f"amplitudes ({len(amp_texts)}) must match."
            )
            return None

        try:
            frequencies_hz = [float(f) * 1e6 for f in freq_texts]
        except ValueError as exc:
            QMessageBox.critical(self, "Input Error", f"Invalid frequency value: {exc}")
            return None

        try:
            amplitudes = [float(a) for a in amp_texts]
        except ValueError as exc:
            QMessageBox.critical(self, "Input Error", f"Invalid amplitude value: {exc}")
            return None

        return {
            'frequencies_hz': frequencies_hz,
            'amplitudes':     amplitudes,
            'load_params':    self._load_params,
        }
