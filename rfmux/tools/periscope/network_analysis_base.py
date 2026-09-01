"""Base dialog for network analysis parameter entry."""

from .utils import (
    QtWidgets, QtCore, QRegularExpression, QRegularExpressionValidator,
    QDoubleValidator, QIntValidator,
    DEFAULT_AMPLITUDE, DEFAULT_MIN_FREQ, DEFAULT_MAX_FREQ, DEFAULT_CABLE_LENGTH,
    DEFAULT_NPOINTS, DEFAULT_NSAMPLES, DEFAULT_MAX_CHANNELS, DEFAULT_MAX_SPAN,
    DEFAULT_AMP_START, DEFAULT_AMP_STOP, DEFAULT_AMP_ITERATIONS,
    UnitConverter, traceback
)
from .tasks import DACScaleFetcher
import datetime
import numpy as np  # For linspace

class NetworkAnalysisDialogBase(QtWidgets.QDialog):
    """
    Base class for network analysis dialogs, providing shared functionality
    for amplitude input (normalized and dBm), DAC scale handling, and
    parameter parsing.
    """
    def __init__(self, parent: QtWidgets.QWidget = None, params: dict = None,
                 modules: list[int] = None, dac_scales: dict[int, float] = None):
        """
        Initializes the base dialog.

        Args:
            parent: The parent widget.
            params: Dictionary of existing parameters to populate fields.
            modules: List of module numbers relevant to this dialog.
            dac_scales: Dictionary mapping module numbers to their DAC scales in dBm.
        """
        super().__init__(parent)
        self.params = params or {}  # Store initial parameters, default to empty dict
        self.modules = modules or [1, 2, 3, 4] # Default or passed-in modules
        # Initialize DAC scales for relevant modules, defaulting to None (unknown)
        self.dac_scales = dac_scales or {module_idx: None for module_idx in self.modules}
        self.currently_updating = False # Flag to prevent recursive updates between amp/dBm fields
        
    def setup_amplitude_group(self, layout: QtWidgets.QFormLayout) -> QtWidgets.QGroupBox:
        """
        Sets up the QGroupBox for amplitude settings.

        Presents two modes via radio buttons:
          • Single amplitude — one normalized field plus a paired dBm field
            with live bidirectional conversion.
          • Amplitude sweep (min → max) — start/stop normalized + dBm pairs
            with live bidirectional conversion, plus a step count.

        Args:
            layout: The QFormLayout to add the amplitude group to.

        Returns:
            The created QGroupBox containing amplitude settings.
        """
        amp_group = QtWidgets.QGroupBox("Amplitude Settings")
        amp_outer_layout = QtWidgets.QVBoxLayout(amp_group)

        # ── Radio buttons ──────────────────────────────────────────────
        self.single_amp_radio = QtWidgets.QRadioButton("Single amplitude")
        self.single_amp_radio.setChecked(True)
        self.sweep_amp_radio = QtWidgets.QRadioButton("Amplitude sweep")
        amp_outer_layout.addWidget(self.single_amp_radio)

        # ── Single amplitude sub-widget ────────────────────────────────
        self.single_amp_controls = QtWidgets.QWidget()
        single_form = QtWidgets.QFormLayout(self.single_amp_controls)
        single_form.setContentsMargins(20, 0, 0, 0)

        # Determine initial amplitude from saved params
        amps_list = self.params.get('amps', [self.params.get('amp', DEFAULT_AMPLITUDE)])
        amp_str = str(amps_list[0]) if amps_list else str(DEFAULT_AMPLITUDE)

        self.amp_edit = QtWidgets.QLineEdit(amp_str)
        self.amp_edit.setToolTip(
            "Normalized amplitude (0–1). "
            "Expressions like '1/1000' are allowed."
        )
        single_form.addRow("Amplitude (norm.):", self.amp_edit)

        self.dbm_edit = QtWidgets.QLineEdit()
        self.dbm_edit.setToolTip(
            "Equivalent power in dBm. "
            "Edit here to update the normalized field above."
        )
        single_form.addRow("Power (dBm):", self.dbm_edit)

        amp_outer_layout.addWidget(self.single_amp_controls)

        # ── Sweep radio + sub-widget ───────────────────────────────────
        amp_outer_layout.addWidget(self.sweep_amp_radio)

        self.sweep_amp_controls = QtWidgets.QWidget()
        sweep_form = QtWidgets.QFormLayout(self.sweep_amp_controls)
        sweep_form.setContentsMargins(20, 0, 0, 0)

        # Start row: normalized + dBm side-by-side
        self.start_amp_edit = QtWidgets.QLineEdit()
        self.start_amp_edit.setPlaceholderText("e.g., 0.001")
        self.start_amp_edit.setToolTip("Start normalized amplitude.")

        self.start_dbm_edit = QtWidgets.QLineEdit()
        self.start_dbm_edit.setPlaceholderText("dBm")
        self.start_dbm_edit.setToolTip(
            "Start power in dBm. "
            "Edit here to update the normalized field to the left."
        )

        start_row_widget = QtWidgets.QWidget()
        start_row = QtWidgets.QHBoxLayout(start_row_widget)
        start_row.setContentsMargins(0, 0, 0, 0)
        start_row.addWidget(QtWidgets.QLabel("norm:"))
        start_row.addWidget(self.start_amp_edit)
        start_row.addWidget(QtWidgets.QLabel("dBm:"))
        start_row.addWidget(self.start_dbm_edit)
        sweep_form.addRow("Start amplitude:", start_row_widget)

        # Stop row: normalized + dBm side-by-side
        self.stop_amp_edit = QtWidgets.QLineEdit()
        self.stop_amp_edit.setPlaceholderText("e.g., 0.1")
        self.stop_amp_edit.setToolTip("Stop normalized amplitude.")

        self.stop_dbm_edit = QtWidgets.QLineEdit()
        self.stop_dbm_edit.setPlaceholderText("dBm")
        self.stop_dbm_edit.setToolTip(
            "Stop power in dBm. "
            "Edit here to update the normalized field to the left."
        )

        stop_row_widget = QtWidgets.QWidget()
        stop_row = QtWidgets.QHBoxLayout(stop_row_widget)
        stop_row.setContentsMargins(0, 0, 0, 0)
        stop_row.addWidget(QtWidgets.QLabel("norm:"))
        stop_row.addWidget(self.stop_amp_edit)
        stop_row.addWidget(QtWidgets.QLabel("dBm:"))
        stop_row.addWidget(self.stop_dbm_edit)
        sweep_form.addRow("Stop amplitude:", stop_row_widget)

        # Steps
        self.num_steps_edit = QtWidgets.QLineEdit(
            str(self.params.get('amp_sweep_steps', 3))
        )
        self.num_steps_edit.setValidator(QIntValidator(2, 1000, self))
        self.num_steps_edit.setToolTip("Number of amplitude steps (minimum 2).")
        sweep_form.addRow("Number of steps:", self.num_steps_edit)

        self.sweep_amp_controls.setVisible(False)
        amp_outer_layout.addWidget(self.sweep_amp_controls)

        # ── Sweep tab order: norm fields chain together, dBm fields chain together
        QtWidgets.QWidget.setTabOrder(self.start_amp_edit, self.stop_amp_edit)
        QtWidgets.QWidget.setTabOrder(self.stop_amp_edit,  self.num_steps_edit)
        QtWidgets.QWidget.setTabOrder(self.num_steps_edit, self.start_dbm_edit)
        QtWidgets.QWidget.setTabOrder(self.start_dbm_edit, self.stop_dbm_edit)

        # ── Connections ────────────────────────────────────────────────
        self.single_amp_radio.toggled.connect(self._on_amplitude_mode_changed)
        self.sweep_amp_radio.toggled.connect(self._on_amplitude_mode_changed)

        # Single mode — bidirectional live sync
        self.amp_edit.textChanged.connect(
            lambda: self._update_dbm_from_normalized(validate=False)
        )
        self.dbm_edit.textChanged.connect(
            lambda: self._update_normalized_from_dbm(validate=False)
        )
        self.amp_edit.editingFinished.connect(self._validate_normalized_values)
        self.dbm_edit.editingFinished.connect(self._validate_dbm_values)

        # Sweep start — bidirectional live sync
        self.start_amp_edit.textChanged.connect(
            lambda: self._update_start_dbm_from_normalized(validate=False)
        )
        self.start_dbm_edit.textChanged.connect(
            lambda: self._update_start_normalized_from_dbm(validate=False)
        )

        # Sweep stop — bidirectional live sync
        self.stop_amp_edit.textChanged.connect(
            lambda: self._update_stop_dbm_from_normalized(validate=False)
        )
        self.stop_dbm_edit.textChanged.connect(
            lambda: self._update_stop_normalized_from_dbm(validate=False)
        )

        layout.addRow("Amplitude Settings:", amp_group)

        # Restore saved amplitude mode if available
        saved_mode = self.params.get('amplitude_mode', 'single')
        if saved_mode == 'sweep':
            self.sweep_amp_radio.setChecked(True)
            saved_start = self.params.get('amp_sweep_start')
            saved_stop = self.params.get('amp_sweep_stop')
            if saved_start is not None:
                self.start_amp_edit.setText(f"{saved_start:.6g}")
            if saved_stop is not None:
                self.stop_amp_edit.setText(f"{saved_stop:.6g}")

        return amp_group

    # ── Mode switching ─────────────────────────────────────────────────

    def _on_amplitude_mode_changed(self):
        """Show/hide the appropriate sub-widget when the radio selection changes."""
        is_single = self.single_amp_radio.isChecked()
        self.single_amp_controls.setVisible(is_single)
        self.sweep_amp_controls.setVisible(not is_single)

    # ── Public helper used by subclass get_parameters() ────────────────

    def _get_amplitude_list(self) -> list[float] | None:
        """
        Build and return the final amplitude list based on the selected mode.

        Single mode  → ``[amp]``
        Sweep mode   → ``np.linspace(start, stop, steps).tolist()``

        Returns None and shows a warning dialog on invalid input.
        """
        if self.single_amp_radio.isChecked():
            amp_text = self.amp_edit.text().strip()
            values = self._parse_amplitude_values(amp_text)
            if not values:
                QtWidgets.QMessageBox.warning(
                    self, "Validation Error",
                    "Please enter a valid amplitude value."
                )
                return None
            return values  # may be a single-element list or multi-value legacy list

        else:
            # Sweep mode
            start_text = self.start_amp_edit.text().strip()
            stop_text = self.stop_amp_edit.text().strip()
            steps_text = self.num_steps_edit.text().strip()

            if not start_text or not stop_text:
                QtWidgets.QMessageBox.warning(
                    self, "Validation Error",
                    "Please enter both a start and stop amplitude for the sweep."
                )
                return None

            try:
                start_val = float(eval(start_text))
                stop_val = float(eval(stop_text))
                steps_val = int(steps_text) if steps_text else 2
            except Exception:
                QtWidgets.QMessageBox.warning(
                    self, "Validation Error",
                    "Invalid numerical input in amplitude sweep fields."
                )
                return None

            if start_val <= 0 or stop_val <= 0:
                QtWidgets.QMessageBox.warning(
                    self, "Validation Error",
                    "Start and stop amplitudes must be positive."
                )
                return None

            if steps_val < 2:
                QtWidgets.QMessageBox.warning(
                    self, "Validation Error",
                    "Number of steps must be at least 2."
                )
                return None

            return np.linspace(start_val, stop_val, steps_val).tolist()

    # ── Validation helpers ─────────────────────────────────────────────

    def _validate_normalized_values(self):
        """
        Validates the entered normalized amplitude values after editing is finished.
        Shows a warning dialog if values are outside typical ranges.
        """
        amp_text = self.amp_edit.text().strip()
        if not amp_text:
            return

        warnings_list = []
        normalized_values = self._parse_amplitude_values(amp_text)
        for norm_val in normalized_values:
            if norm_val > 1.0:
                warnings_list.append(f"Warning: Normalized amplitude {norm_val:.6f} > 1.0 (maximum)")
            elif norm_val < 1e-4:
                warnings_list.append(f"Warning: Normalized amplitude {norm_val:.6f} < 1e-4 (minimum recommended)")
        
        if warnings_list:
            self._show_warning_dialog("Normalized Amplitude Warning", warnings_list)

    def _validate_dbm_values(self):
        """
        Validates the entered dBm values after editing is finished.
        Shows a warning dialog if values exceed DAC scale or are out of range.
        """
        dbm_text = self.dbm_edit.text().strip()
        if not dbm_text:
            return

        dac_scale = self._get_selected_dac_scale()
        if dac_scale is None:
            self._show_warning_dialog(
                "DAC Scale Unknown",
                ["Cannot validate dBm values without a known DAC scale."]
            )
            return

        warnings_list = []
        dbm_values = self._parse_dbm_values(dbm_text)
        for dbm_val in dbm_values:
            if dbm_val > dac_scale:
                warnings_list.append(f"Warning: {dbm_val:.2f} dBm > {dac_scale:+.2f} dBm (DAC maximum)")
            norm_val = UnitConverter.dbm_to_normalize(dbm_val, dac_scale)
            if norm_val > 1.0:
                warnings_list.append(
                    f"Warning: {dbm_val:.2f} dBm results in normalized amplitude > 1.0 ({norm_val:.6f})"
                )
            elif norm_val < 1e-4:
                warnings_list.append(
                    f"Warning: {dbm_val:.2f} dBm results in normalized amplitude < 1e-4 ({norm_val:.6f})"
                )
        
        if warnings_list:
            self._show_warning_dialog("dBm Amplitude Warning", warnings_list)

    def _show_warning_dialog(self, title: str, warnings_list: list[str]):
        """Displays a warning message box with a list of warnings."""
        QtWidgets.QMessageBox.warning(self, title, "\n".join(warnings_list))

    # ── Parsing helpers ────────────────────────────────────────────────

    def _parse_numeric_values(self, text: str) -> list[float]:
        """
        Parses a comma-separated string of numeric values.
        Each part can be an expression evaluatable by ``eval()``.
        Invalid parts are silently skipped.
        """
        values = []
        for part in text.split(','):
            part = part.strip()
            if part:
                try:
                    values.append(float(eval(part)))
                except (ValueError, SyntaxError, NameError, TypeError):
                    continue
        return values
    
    def _parse_amplitude_values(self, amp_text: str) -> list[float]:
        """Parse comma-separated amplitude values."""
        return self._parse_numeric_values(amp_text)
        
    def _parse_dbm_values(self, dbm_text: str) -> list[float]:
        """Parse comma-separated dBm values."""
        return self._parse_numeric_values(dbm_text)

    # ── DAC scale / module helpers ─────────────────────────────────────

    def _update_dac_scale_info(self):
        """
        Enables or disables all dBm input fields based on whether a DAC scale
        is known for any selected module.  Also refreshes the dBm display for
        all currently visible fields.

        The DAC scale is used internally for unit conversion; it is not shown
        in the UI.
        """
        selected_modules = self._get_selected_modules()
        has_known_scale = any(
            self.dac_scales.get(m) is not None for m in selected_modules
        )

        # Enable/disable all dBm fields
        for field in (self.dbm_edit, self.start_dbm_edit, self.stop_dbm_edit):
            field.setEnabled(has_known_scale)
            if not has_known_scale:
                field.clear()

        if has_known_scale:
            self._update_dbm_from_normalized()
            self._update_start_dbm_from_normalized()
            self._update_stop_dbm_from_normalized()
    
    def _get_selected_modules(self) -> list[int]:
        """
        Placeholder — subclasses must override this to return the currently
        selected module IDs.
        """
        return []
        
    def _get_selected_dac_scale(self) -> float | None:
        """
        Returns the DAC scale (dBm) for the first selected module that has a
        known scale, or None.
        """
        for module_idx in self._get_selected_modules():
            dac_scale = self.dac_scales.get(module_idx)
            if dac_scale is not None:
                return dac_scale
        return None

    # ── Single-amplitude bidirectional sync ───────────────────────────

    def _update_dbm_from_normalized(self, validate: bool = True):
        """Update the single-amplitude dBm field from the normalized field."""
        if self.currently_updating or not self.dbm_edit.isEnabled():
            return
        self.currently_updating = True
        try:
            amp_text = self.amp_edit.text().strip()
            if not amp_text:
                self.dbm_edit.setText("")
                return

            dac_scale = self._get_selected_dac_scale()
            if dac_scale is None:
                self.dbm_edit.setEnabled(False)
                self.dbm_edit.clear()
                return

            normalized_values = self._parse_amplitude_values(amp_text)
            dbm_values = []
            warnings_list = []
            
            for norm_val in normalized_values:
                if validate:
                    if norm_val > 1.0:
                        warnings_list.append(f"Warning: Normalized amplitude {norm_val:.6f} > 1.0 (maximum)")
                    elif norm_val < 1e-4:
                        warnings_list.append(f"Warning: Normalized amplitude {norm_val:.6f} < 1e-4 (minimum recommended)")
                dbm_values.append(f"{UnitConverter.normalize_to_dbm(norm_val, dac_scale):.2f}")
            
            self.dbm_edit.setText(", ".join(dbm_values))
            
            if validate and warnings_list and not self.amp_edit.hasFocus():
                self._show_warning_dialog("Normalized Amplitude Warning", warnings_list)
        finally:
            self.currently_updating = False
    
    def _update_normalized_from_dbm(self, validate: bool = True):
        """Update the single-amplitude normalized field from the dBm field."""
        if self.currently_updating or not self.dbm_edit.isEnabled():
            return
        self.currently_updating = True
        try:
            dbm_text = self.dbm_edit.text().strip()
            if not dbm_text:
                self.amp_edit.setText("")
                return

            dac_scale = self._get_selected_dac_scale()
            if dac_scale is None:
                return

            dbm_values = self._parse_dbm_values(dbm_text)
            normalized_values = []
            warnings_list = []
            
            for dbm_val in dbm_values:
                if validate and dbm_val > dac_scale:
                    warnings_list.append(f"Warning: {dbm_val:.2f} dBm > {dac_scale:+.2f} dBm (DAC maximum)")
                norm_val = UnitConverter.dbm_to_normalize(dbm_val, dac_scale)
                if validate:
                    if norm_val > 1.0:
                        warnings_list.append(
                            f"Warning: {dbm_val:.2f} dBm results in normalized amplitude > 1.0 ({norm_val:.6f})"
                        )
                    elif norm_val < 1e-4:
                        warnings_list.append(
                            f"Warning: {dbm_val:.2f} dBm results in normalized amplitude < 1e-4 ({norm_val:.6f})"
                        )
                normalized_values.append(f"{norm_val:.6f}")
            
            self.amp_edit.setText(", ".join(normalized_values))

            if validate and warnings_list and not self.dbm_edit.hasFocus():
                self._show_warning_dialog("dBm Amplitude Warning", warnings_list)
        finally:
            self.currently_updating = False

    # ── Sweep start bidirectional sync ────────────────────────────────

    def _update_start_dbm_from_normalized(self, validate: bool = True):
        """Update the sweep start dBm field from the normalized field."""
        if self.currently_updating or not self.start_dbm_edit.isEnabled():
            return
        self.currently_updating = True
        try:
            text = self.start_amp_edit.text().strip()
            if not text:
                self.start_dbm_edit.setText("")
                return
            dac_scale = self._get_selected_dac_scale()
            if dac_scale is None:
                self.start_dbm_edit.clear()
                return
            try:
                norm_val = float(eval(text))
                self.start_dbm_edit.setText(
                    f"{UnitConverter.normalize_to_dbm(norm_val, dac_scale):.2f}"
                )
            except Exception:
                pass
        finally:
            self.currently_updating = False

    def _update_start_normalized_from_dbm(self, validate: bool = True):
        """Update the sweep start normalized field from the dBm field."""
        if self.currently_updating or not self.start_dbm_edit.isEnabled():
            return
        self.currently_updating = True
        try:
            text = self.start_dbm_edit.text().strip()
            if not text:
                self.start_amp_edit.setText("")
                return
            dac_scale = self._get_selected_dac_scale()
            if dac_scale is None:
                return
            try:
                dbm_val = float(eval(text))
                norm_val = UnitConverter.dbm_to_normalize(dbm_val, dac_scale)
                self.start_amp_edit.setText(f"{norm_val:.6f}")
            except Exception:
                pass
        finally:
            self.currently_updating = False

    # ── Sweep stop bidirectional sync ─────────────────────────────────

    def _update_stop_dbm_from_normalized(self, validate: bool = True):
        """Update the sweep stop dBm field from the normalized field."""
        if self.currently_updating or not self.stop_dbm_edit.isEnabled():
            return
        self.currently_updating = True
        try:
            text = self.stop_amp_edit.text().strip()
            if not text:
                self.stop_dbm_edit.setText("")
                return
            dac_scale = self._get_selected_dac_scale()
            if dac_scale is None:
                self.stop_dbm_edit.clear()
                return
            try:
                norm_val = float(eval(text))
                self.stop_dbm_edit.setText(
                    f"{UnitConverter.normalize_to_dbm(norm_val, dac_scale):.2f}"
                )
            except Exception:
                pass
        finally:
            self.currently_updating = False

    def _update_stop_normalized_from_dbm(self, validate: bool = True):
        """Update the sweep stop normalized field from the dBm field."""
        if self.currently_updating or not self.stop_dbm_edit.isEnabled():
            return
        self.currently_updating = True
        try:
            text = self.stop_dbm_edit.text().strip()
            if not text:
                self.stop_amp_edit.setText("")
                return
            dac_scale = self._get_selected_dac_scale()
            if dac_scale is None:
                return
            try:
                dbm_val = float(eval(text))
                norm_val = UnitConverter.dbm_to_normalize(dbm_val, dac_scale)
                self.stop_amp_edit.setText(f"{norm_val:.6f}")
            except Exception:
                pass
        finally:
            self.currently_updating = False

    # ── Measurement name helpers (shared by all subclass dialogs) ──────

    def _update_name_preview(self):
        """Update the live filename preview label from the two name fields."""
        base = self.base_name_edit.text().strip()
        suffix = self.custom_suffix_edit.text().strip()
        if base:
            full = f"{base}_{suffix}.pkl" if suffix else f"{base}.pkl"
        else:
            full = f"{suffix}.pkl" if suffix else "(no name)"
        self._name_preview_label.setText(full)

    def _get_measurement_name(self) -> str:
        """Return the combined measurement name from the two dialog fields.

        Combines base name and optional custom suffix with an underscore
        separator.  Strips both values and falls back to a fresh timestamp
        if the base field is empty.

        Returns:
            The measurement name string (without .pkl extension).
        """
        base = self.base_name_edit.text().strip()
        suffix = self.custom_suffix_edit.text().strip()
        if not base:
            base = datetime.datetime.now().strftime("netanal_%H%M%S")
        return f"{base}_{suffix}" if suffix else base
