"""
Bias-finding settings panel, reusable bias-entry widget, and Bias KIDs dialog.

``BiasSettingsPanel`` is a non-modal ``QWidget`` (not a ``QDialog``) that
retains its values between "Find Bias" runs.  Open it from the "⚙ Bias
Settings" button on the MultisweepPanel toolbar; it appears as a small
floating window and can be dismissed without disrupting ongoing work.

``get_settings()`` returns a dict that can be passed directly to
``FindBiasTask`` as *bias_settings*.

``CustomBiasWidget`` is a reusable widget (shared by ``BiasKIDsDialog`` and
``BiasSettingsPanel``) that provides file-load buttons and manual
frequency/amplitude entry fields.

``BiasKIDsDialog`` is the dialog launched by the "Bias KIDs" button on the
main Periscope window.
"""

import csv
import os
import pickle

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QLabel,
    QDoubleSpinBox, QCheckBox, QButtonGroup, QRadioButton,
    QPushButton, QHBoxLayout, QLineEdit, QDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal


# ── Shared bias-entry widget ──────────────────────────────────────────────────

class CustomBiasWidget(QWidget):
    """
    Reusable widget for entering or loading custom bias parameters
    (a list of frequencies in MHz and normalized amplitudes).

    Provides "Load Multisweep file…" and "Load CSV…" buttons together with
    manual comma-separated frequency and amplitude entry fields.

    Used by both :class:`BiasKIDsDialog` and the Custom Bias section of
    :class:`BiasSettingsPanel`.

    Signals
    -------
    pkl_data_available(bool)
        Emitted after any file-load or clear operation.  ``True`` when a
        multisweep pickle file is currently loaded (so callers can enable
        "Apply + Plot"); ``False`` after loading a CSV or clearing the widget.
    """

    pkl_data_available = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._load_params = None   # Full pkl payload (set after loading a multisweep file)
        self._setup_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # ── File loading ───────────────────────────────────────────────────────
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
        layout.addLayout(btn_row)

        self._file_status_label = QLabel("No file loaded")
        self._file_status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self._file_status_label)

        # ── Entry fields ───────────────────────────────────────────────────────
        layout.addWidget(QLabel("Frequencies (MHz), comma-separated:"))
        self._freq_edit = QLineEdit()
        self._freq_edit.setPlaceholderText("e.g. 500.123, 501.456, 502.789")
        layout.addWidget(self._freq_edit)

        layout.addWidget(QLabel("Amplitudes (normalized), comma-separated:"))
        self._amp_edit = QLineEdit()
        self._amp_edit.setPlaceholderText("e.g. 0.5, 0.5, 0.5")
        layout.addWidget(self._amp_edit)

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
        self.load_from_multisweep_payload(payload, label=os.path.basename(path))

    def load_from_multisweep_payload(self, payload: dict, label: str = "session file"):
        """
        Populate the frequency/amplitude fields from an already-loaded payload dict.

        This is the internal counterpart to :meth:`_load_from_multisweep` for
        cases where the pickle has already been deserialized by the caller
        (e.g. the session browser pre-loads the data before opening this widget).

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

        self._file_status_label.setText(f"Loaded: {label} ({len(biased)} channels)")
        self._file_status_label.setStyleSheet("color: #2a8a2a;")
        self.pkl_data_available.emit(True)

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
        self._load_params = None

        self._file_status_label.setText(
            f"Loaded CSV: {os.path.basename(path)} ({len(freq_strs)} channels)"
        )
        self._file_status_label.setStyleSheet("color: #2a8a2a;")
        self.pkl_data_available.emit(False)

    def clear(self):
        """Reset all fields to their blank initial state."""
        self._freq_edit.clear()
        self._amp_edit.clear()
        self._file_status_label.setText("No file loaded")
        self._file_status_label.setStyleSheet("color: gray; font-style: italic;")
        self._load_params = None
        self.pkl_data_available.emit(False)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def load_params(self):
        """The full pkl payload if a multisweep file was loaded, else ``None``."""
        return self._load_params

    @property
    def has_pkl_data(self) -> bool:
        """``True`` when a multisweep pickle is currently loaded."""
        return self._load_params is not None

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


# ── BiasSettingsPanel ─────────────────────────────────────────────────────────

class BiasSettingsPanel(QWidget):
    """
    Persistent, non-modal settings panel for the bias-finding algorithm.

    Intended to be instantiated once by ``MultisweepPanel`` and shown/raised
    when the user clicks "⚙ Bias Settings".  Values persist for the lifetime
    of the panel.

    Call :meth:`get_settings` to retrieve the current parameter dict, which
    can be passed directly to :class:`~rfmux.tools.periscope.tasks.FindBiasTask`
    as *bias_settings*.

    Signals
    -------
    apply_custom_bias_requested
        Emitted when the user clicks "Apply Custom Bias to Hardware" inside
        the panel.  :class:`~rfmux.tools.periscope.multisweep_panel.MultisweepPanel`
        connects this signal to trigger the full custom-bias → apply-hardware
        pipeline without the user having to click two separate toolbar buttons.
    """

    apply_custom_bias_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bias Finding Settings")
        # Float as a real top-level window when shown without a parent holding it
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowCloseButtonHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setMinimumWidth(420)
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

        # ── Bias Frequency Method ─────────────────────────────────────────────
        method_group = QGroupBox("Bias Frequency Method")
        method_layout = QVBoxLayout()
        method_layout.setSpacing(4)

        self._bias_method_group = QButtonGroup(self)

        self.rb_iq_derivative = QRadioButton("IQ arc-length speed  |dI/df + j·dQ/df|")
        self.rb_iq_derivative.setChecked(True)
        self.rb_iq_derivative.setToolTip(
            "Find the bias frequency by maximising the IQ arc-length speed\n"
            "|dI/df + j·dQ/df|.  This is the classic method: the peak\n"
            "of the arc-length speed corresponds to the resonance frequency\n"
            "for a standard resonator sweep."
        )
        self._bias_method_group.addButton(self.rb_iq_derivative)
        method_layout.addWidget(self.rb_iq_derivative)

        self.rb_log_iq_derivative = QRadioButton("Log-mag of derivative  log(|dI/df + j·dQ/df|)")
        self.rb_log_iq_derivative.setToolTip(
            "Find the bias frequency by maximising the log-magnitude of the\n"
            "IQ arc-length speed: log(|dI/df + j·dQ/df|).\n\n"
            "Taking the logarithm compresses the dynamic range of the\n"
            "resonance feature so that the true peak stands out more clearly\n"
            "in noisy data.  The IQ derivatives used for df_calibration are\n"
            "always computed from the standard (non-log) splines.\n\n"
            "The IQ Derivatives tab shows the log-mag curve when this\n"
            "method is active."
        )
        self._bias_method_group.addButton(self.rb_log_iq_derivative)
        method_layout.addWidget(self.rb_log_iq_derivative)

        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

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

        # ── Custom Bias ───────────────────────────────────────────────────────
        self._custom_bias_group = QGroupBox("Custom Bias (overrides Find Bias algorithm)")
        custom_layout = QVBoxLayout()
        custom_layout.setSpacing(6)

        # Enable checkbox
        self.enable_custom_bias_cb = QCheckBox(
            "Use Custom Bias instead of Find Bias algorithm"
        )
        self.enable_custom_bias_cb.setToolTip(
            "When checked, clicking 'Find Bias' on the MultisweepPanel toolbar\n"
            "will skip the automatic bias-finding algorithm and instead assign\n"
            "the frequencies and amplitudes provided below to the existing\n"
            "resonators (matched in ascending frequency order).\n\n"
            "The number of provided values must equal the number of resonators\n"
            "in the current multisweep panel."
        )
        custom_layout.addWidget(self.enable_custom_bias_cb)

        # Resonator count hint (updated by MultisweepPanel when the panel is opened)
        self._resonator_count_label = QLabel("")
        self._resonator_count_label.setStyleSheet("color: gray; font-style: italic;")
        custom_layout.addWidget(self._resonator_count_label)

        # Bias entry widget (file loading + text fields)
        self._custom_bias_widget = CustomBiasWidget(self)
        custom_layout.addWidget(self._custom_bias_widget)

        # Quantize checkbox
        self.quantize_custom_bias_cb = QCheckBox(
            "Quantize to hardware frequency grid (≈298 Hz steps)"
        )
        self.quantize_custom_bias_cb.setChecked(True)
        self.quantize_custom_bias_cb.setToolTip(
            "Round each bias frequency to the nearest hardware tone bin\n"
            "(multiples of 625 MHz / 2²¹ ≈ 298.023 Hz) before storing\n"
            "in res_info_dict.  This ensures the saved frequencies exactly\n"
            "match what will be programmed into the CRS hardware."
        )
        custom_layout.addWidget(self.quantize_custom_bias_cb)

        # "Apply Custom Bias to Hardware" shortcut button
        apply_custom_btn_row = QHBoxLayout()
        apply_custom_btn_row.addStretch(1)
        self._apply_custom_bias_btn = QPushButton("Apply Custom Bias to Hardware")
        self._apply_custom_bias_btn.setToolTip(
            "Populate res_info_dict with the custom bias values and\n"
            "immediately apply them to the CRS hardware in one step.\n"
            "Equivalent to clicking 'Find Bias' then 'Apply Bias' on\n"
            "the MultisweepPanel toolbar."
        )
        self._apply_custom_bias_btn.clicked.connect(self._on_apply_custom_bias)
        apply_custom_btn_row.addWidget(self._apply_custom_bias_btn)
        custom_layout.addLayout(apply_custom_btn_row)

        self._custom_bias_group.setLayout(custom_layout)
        layout.addWidget(self._custom_bias_group)

        # Wire the enable checkbox to enable/disable the controls inside
        self.enable_custom_bias_cb.toggled.connect(self._on_enable_custom_bias_changed)
        self._on_enable_custom_bias_changed(False)  # start disabled

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

    def set_resonator_count(self, n: int):
        """
        Display the number of resonators currently in the associated
        MultisweepPanel so the user knows how many custom values to provide.

        Parameters
        ----------
        n :
            Number of resonators in the active multisweep panel.
            Pass 0 to hide the hint.
        """
        if n > 0:
            self._resonator_count_label.setText(
                f"Active resonators in panel: {n}  "
                f"(provide exactly {n} frequencies and amplitudes)"
            )
        else:
            self._resonator_count_label.setText("")

    def get_settings(self) -> dict:
        """
        Return the current settings as a dict suitable for passing to
        :class:`~rfmux.tools.periscope.tasks.FindBiasTask` as *bias_settings*.

        Keys
        ----
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
        * ``bias_freq_method`` (str) — one of
          ``"iq_derivative"`` (default), ``"log_iq_derivative"``
        * ``custom_bias_enabled`` (bool)
        * ``custom_bias_frequencies_hz`` (list[float] | None) — only present
          when *custom_bias_enabled* is True and fields are non-empty
        * ``custom_bias_amplitudes`` (list[float] | None)
        * ``custom_bias_quantize`` (bool)
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

        bias_method = (
            "log_iq_derivative" if self.rb_log_iq_derivative.isChecked()
            else "iq_derivative"
        )

        settings = {
            'spike_prominence_factor':    self.spike_prominence_spin.value(),
            'spike_height_factor':        self.spike_height_spin.value(),
            'max_deriv_distance_mode':    deriv_mode,
            'max_deriv_distance_hz':      deriv_hz,
            'max_deriv_distance_fraction': deriv_fraction,
            'reference_freq_source':      ref_source,
            'fit_selected_amplitude':     self.fit_selected_cb.isChecked(),
            'bias_freq_method':           bias_method,
            'custom_bias_enabled':        self.enable_custom_bias_cb.isChecked(),
            'custom_bias_quantize':       self.quantize_custom_bias_cb.isChecked(),
        }

        # Include custom bias values when enabled and fields are non-empty
        if self.enable_custom_bias_cb.isChecked():
            freq_text = self._custom_bias_widget._freq_edit.text().strip()
            amp_text  = self._custom_bias_widget._amp_edit.text().strip()

            if freq_text and amp_text:
                try:
                    settings['custom_bias_frequencies_hz'] = [
                        float(t.strip()) * 1e6
                        for t in freq_text.split(',') if t.strip()
                    ]
                except ValueError:
                    settings['custom_bias_frequencies_hz'] = None

                try:
                    settings['custom_bias_amplitudes'] = [
                        float(t.strip())
                        for t in amp_text.split(',') if t.strip()
                    ]
                except ValueError:
                    settings['custom_bias_amplitudes'] = None
            else:
                settings['custom_bias_frequencies_hz'] = None
                settings['custom_bias_amplitudes']     = None

        return settings

    # Alias so that ParamKeyExtractor (which looks for get_parameters) can
    # validate the settings keys against the smoke-test mock dict.
    get_parameters = get_settings

    # ── Private helpers ───────────────────────────────────────────────────────

    def _on_deriv_mode_changed(self):
        """Enable the spinbox that belongs to the active deriv-distance radio button."""
        absolute = self.rb_deriv_absolute.isChecked()
        self.max_deriv_dist_spin.setEnabled(absolute)
        self.max_deriv_frac_spin.setEnabled(not absolute)

    def _on_enable_custom_bias_changed(self, checked: bool):
        """Enable/disable all controls inside the Custom Bias group."""
        self._custom_bias_widget.setEnabled(checked)
        self.quantize_custom_bias_cb.setEnabled(checked)
        self._apply_custom_bias_btn.setEnabled(checked)
        self._resonator_count_label.setEnabled(checked)

    def _on_apply_custom_bias(self):
        """Emit the apply_custom_bias_requested signal when the shortcut button is clicked."""
        self.apply_custom_bias_requested.emit()

    def _reset_defaults(self):
        """Restore all *algorithm* controls to their default values.

        The custom bias fields (frequencies and amplitudes) are intentionally
        preserved so the user does not accidentally lose data they have entered.
        The 'Use Custom Bias' checkbox is unchecked.
        """
        self.spike_prominence_spin.setValue(2.0)
        self.spike_height_spin.setValue(3.0)
        self.rb_deriv_absolute.setChecked(True)   # resets mode → absolute
        self.max_deriv_dist_spin.setValue(100.0)
        self.max_deriv_frac_spin.setValue(0.5)
        self.rb_bias_freq.setChecked(True)
        self.rb_iq_derivative.setChecked(True)    # resets method → classic IQ derivative
        self.fit_selected_cb.setChecked(True)
        # Uncheck custom bias (values are preserved)
        self.enable_custom_bias_cb.setChecked(False)
        self.quantize_custom_bias_cb.setChecked(True)


# ── Bias KIDs Dialog ──────────────────────────────────────────────────────────

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
        self.use_plot = False      # True when user clicked "Apply + Plot"
        self._setup_ui()

    # ── UI construction ────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ── Bias parameters (file-load + entry fields via CustomBiasWidget) ────
        entry_group = QGroupBox("Bias parameters")
        entry_layout = QVBoxLayout(entry_group)
        self._bias_entry = CustomBiasWidget(self)
        entry_layout.addWidget(self._bias_entry)
        layout.addWidget(entry_group)

        # Enable "Apply + Plot" only when a pkl was loaded
        self._bias_entry.pkl_data_available.connect(self._on_pkl_data_available)

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

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_pkl_data_available(self, available: bool):
        self._apply_plot_btn.setEnabled(available)

    def _on_apply(self):
        self.use_plot = False
        self.accept()

    def _on_apply_and_plot(self):
        self.use_plot = True
        self.accept()

    # ── Public API for pre-population (called by session browser) ──────────────

    def _load_from_multisweep_payload(self, payload: dict, label: str = "session file"):
        """
        Pre-populate the dialog from an already-loaded multisweep payload.

        This mirrors the old direct method on the dialog; now it simply
        delegates to :class:`CustomBiasWidget`.
        """
        self._bias_entry.load_from_multisweep_payload(payload, label=label)

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
        return self._bias_entry.get_result()
