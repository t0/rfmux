"""Dialog classes for Periscope."""

# Imports from within the 'periscope' subpackage
from .utils import (
    QtWidgets, QtCore, QRegularExpression, QRegularExpressionValidator,
    QDoubleValidator, QIntValidator,
    DEFAULT_AMPLITUDE, DEFAULT_MIN_FREQ, DEFAULT_MAX_FREQ, DEFAULT_CABLE_LENGTH,
    DEFAULT_NPOINTS, DEFAULT_NSAMPLES, DEFAULT_MAX_CHANNELS, DEFAULT_MAX_SPAN,
    UnitConverter, traceback # Added traceback
)
from .tasks import DACScaleFetcher # For fetching DAC scales

class NetworkAnalysisDialogBase(QtWidgets.QDialog):
    """Base class for network analysis dialogs with shared functionality."""
    def __init__(self, parent=None, params=None, modules=None, dac_scales=None):
        super().__init__(parent)
        self.params = params or {}
        self.modules = modules or [1, 2, 3, 4] # Default or passed-in modules
        self.dac_scales = dac_scales or {m: None for m in self.modules}
        self.currently_updating = False
        
    def setup_amplitude_group(self, layout):
        amp_group = QtWidgets.QGroupBox()
        amp_layout = QtWidgets.QFormLayout(amp_group)
        amps_list = self.params.get('amps', [self.params.get('amp', DEFAULT_AMPLITUDE)]) # Renamed amps
        amp_str = ','.join(map(str, amps_list)) if amps_list else str(DEFAULT_AMPLITUDE)
        
        self.amp_edit = QtWidgets.QLineEdit(amp_str)
        self.amp_edit.setToolTip("Enter a single value or comma-separated list (e.g., 0.001,0.01,0.1)")
        amp_layout.addRow("Normalized Amplitude:", self.amp_edit)
        
        self.dbm_edit = QtWidgets.QLineEdit()
        self.dbm_edit.setToolTip("Enter a single value or comma-separated list in dBm (e.g., -30,-20,-10)")
        amp_layout.addRow("Power (dBm):", self.dbm_edit)
        
        self.dac_scale_info = QtWidgets.QLabel("Fetching DAC scales...")
        self.dac_scale_info.setWordWrap(True)
        amp_layout.addRow("DAC Scale (dBm):", self.dac_scale_info)
        
        self.amp_edit.textChanged.connect(self._update_dbm_from_normalized_no_validate)
        self.dbm_edit.textChanged.connect(self._update_normalized_from_dbm_no_validate)
        self.amp_edit.editingFinished.connect(self._validate_normalized_values)
        self.dbm_edit.editingFinished.connect(self._validate_dbm_values)
        
        layout.addRow("Amplitude Settings:", amp_group)
        return amp_group
        
    def _update_dbm_from_normalized_no_validate(self):
        if self.currently_updating or not self.dbm_edit.isEnabled(): return
        self.currently_updating = True
        try:
            amp_text = self.amp_edit.text().strip()
            if not amp_text: self.dbm_edit.setText(""); return
            dac_scale = self._get_selected_dac_scale()
            if dac_scale is None:
                self.dbm_edit.setEnabled(False); self.dbm_edit.setToolTip("Unable to query DAC scale"); self.dbm_edit.clear(); return
            normalized_values = self._parse_amplitude_values(amp_text)
            dbm_values = [f"{UnitConverter.normalize_to_dbm(norm, dac_scale):.2f}" for norm in normalized_values]
            self.dbm_edit.setText(", ".join(dbm_values))
        finally: self.currently_updating = False

    def _update_normalized_from_dbm_no_validate(self):
        if self.currently_updating or not self.dbm_edit.isEnabled(): return
        self.currently_updating = True
        try:
            dbm_text = self.dbm_edit.text().strip()
            if not dbm_text: self.amp_edit.setText(""); return
            dac_scale = self._get_selected_dac_scale()
            if dac_scale is None:
                self.dbm_edit.setEnabled(False); self.dbm_edit.setToolTip("Unable to query DAC scale"); self.dbm_edit.clear(); return
            dbm_values = self._parse_dbm_values(dbm_text)
            normalized_values = [f"{UnitConverter.dbm_to_normalize(dbm, dac_scale):.6f}" for dbm in dbm_values]
            self.amp_edit.setText(", ".join(normalized_values))
        finally: self.currently_updating = False

    def _validate_normalized_values(self):
        amp_text = self.amp_edit.text().strip(); warnings_list = [] # Renamed warnings
        if not amp_text: return
        dac_scale = self._get_selected_dac_scale();
        if dac_scale is None: return
        normalized_values = self._parse_amplitude_values(amp_text)
        for norm_val in normalized_values: # Renamed norm
            if norm_val > 1.0: warnings_list.append(f"Warning: Normalized amplitude {norm_val:.6f} > 1.0 (maximum)")
            elif norm_val < 1e-4: warnings_list.append(f"Warning: Normalized amplitude {norm_val:.6f} < 1e-4 (minimum recommended)")
        if warnings_list: self._show_warning_dialog("Amplitude Warning", warnings_list)

    def _validate_dbm_values(self):
        dbm_text = self.dbm_edit.text().strip(); warnings_list = [] # Renamed warnings
        if not dbm_text: return
        dac_scale = self._get_selected_dac_scale()
        if dac_scale is None: return
        dbm_values = self._parse_dbm_values(dbm_text)
        for dbm_val in dbm_values: # Renamed dbm
            if dbm_val > dac_scale: warnings_list.append(f"Warning: {dbm_val:.2f} dBm > {dac_scale:+.2f} dBm (DAC max)")
            norm_val = UnitConverter.dbm_to_normalize(dbm_val, dac_scale) # Renamed norm
            if norm_val > 1.0: warnings_list.append(f"Warning: {dbm_val:.2f} dBm gives normalized amplitude > 1.0")
            elif norm_val < 1e-4: warnings_list.append(f"Warning: {dbm_val:.2f} dBm gives normalized amplitude < 1e-4")
        if warnings_list: self._show_warning_dialog("Amplitude Warning", warnings_list)

    def _show_warning_dialog(self, title, warnings_list): # Renamed warnings
        QtWidgets.QMessageBox.warning(self, title, "\n".join(warnings_list))
            
    def _parse_amplitude_values(self, amp_text):
        values = [] # Renamed normalized_values
        for part in amp_text.split(','):
            part = part.strip()
            if part:
                try: values.append(float(eval(part)))
                except (ValueError, SyntaxError, NameError): continue
        return values
        
    def _parse_dbm_values(self, dbm_text):
        values = [] # Renamed dbm_values
        for part in dbm_text.split(','):
            part = part.strip()
            if part:
                try: values.append(float(eval(part)))
                except (ValueError, SyntaxError, NameError): continue
        return values

    def _update_dac_scale_info(self):
        selected_modules = self._get_selected_modules(); has_known_scale = False; scales_text_list = [] # Renamed scales
        for m_idx in selected_modules: # Renamed m
            dac_scale = self.dac_scales.get(m_idx)
            if dac_scale is not None: has_known_scale = True; scales_text_list.append(f"Module {m_idx}: {dac_scale:+.2f} dBm")
            else: scales_text_list.append(f"Module {m_idx}: Unknown")
        text_to_display = "\n".join(scales_text_list) if selected_modules else "Unknown (no modules selected)" # Renamed text
        self.dac_scale_info.setText(text_to_display)
        if has_known_scale:
            self.dbm_edit.setEnabled(True); self.dbm_edit.setToolTip("Enter dBm values (e.g., -30,-20,-10)")
            self._update_dbm_from_normalized() 
        else:
            self.dbm_edit.setEnabled(False); self.dbm_edit.setToolTip("DAC scale unknown - dBm input disabled"); self.dbm_edit.clear()
    
    def _get_selected_modules(self): return [] # To be overridden
        
    def _get_selected_dac_scale(self):
        selected_modules = self._get_selected_modules()
        if not selected_modules: return None
        for module_idx in selected_modules: # Renamed module
            dac_scale = self.dac_scales.get(module_idx)
            if dac_scale is not None: return dac_scale
        return None
    
    def _update_dbm_from_normalized(self): # Combined with _no_validate version
        if self.currently_updating or not self.dbm_edit.isEnabled(): return
        self.currently_updating = True
        try:
            amp_text = self.amp_edit.text().strip(); warnings_list = [] # Renamed warnings
            if not amp_text: self.dbm_edit.setText(""); return
            dac_scale = self._get_selected_dac_scale()
            if dac_scale is None:
                self.dbm_edit.setEnabled(False); self.dbm_edit.setToolTip("Unable to query DAC scale"); self.dbm_edit.clear(); return
            normalized_values = self._parse_amplitude_values(amp_text); dbm_values = []
            for norm_val in normalized_values: # Renamed norm
                if norm_val > 1.0: warnings_list.append(f"Warning: Normalized amplitude {norm_val:.6f} > 1.0")
                elif norm_val < 1e-4: warnings_list.append(f"Warning: Normalized amplitude {norm_val:.6f} < 1e-4")
                dbm_values.append(f"{UnitConverter.normalize_to_dbm(norm_val, dac_scale):.2f}")
            self.dbm_edit.setText(", ".join(dbm_values))
            if warnings_list and not self.amp_edit.hasFocus(): self._show_warning_dialog("Amplitude Warning", warnings_list) # Show only if not actively editing
        finally: self.currently_updating = False
    
    def _update_normalized_from_dbm(self): # Combined with _no_validate version
        if self.currently_updating or not self.dbm_edit.isEnabled(): return
        self.currently_updating = True
        try:
            dbm_text = self.dbm_edit.text().strip(); warnings_list = [] # Renamed warnings
            if not dbm_text: self.amp_edit.setText(""); return
            dac_scale = self._get_selected_dac_scale()
            if dac_scale is None:
                self.dbm_edit.setEnabled(False); self.dbm_edit.setToolTip("Unable to query DAC scale"); self.dbm_edit.clear(); return
            dbm_values = self._parse_dbm_values(dbm_text); normalized_values = []
            for dbm_val in dbm_values: # Renamed dbm
                if dbm_val > dac_scale: warnings_list.append(f"Warning: {dbm_val:.2f} dBm > {dac_scale:.2f} dBm (DAC max)")
                norm_val = UnitConverter.dbm_to_normalize(dbm_val, dac_scale) # Renamed norm
                if norm_val > 1.0: warnings_list.append(f"Warning: {dbm_val:.2f} dBm gives normalized > 1.0")
                elif norm_val < 1e-4: warnings_list.append(f"Warning: {dbm_val:.2f} dBm gives normalized < 1e-4")
                normalized_values.append(f"{norm_val:.6f}")
            self.amp_edit.setText(", ".join(normalized_values))
            if warnings_list and not self.dbm_edit.hasFocus(): self._show_warning_dialog("Amplitude Warning", warnings_list) # Show only if not actively editing
        finally: self.currently_updating = False

class NetworkAnalysisDialog(NetworkAnalysisDialogBase):
    def __init__(self, parent=None, modules=None, dac_scales=None):
        super().__init__(parent, None, modules, dac_scales)
        self.setWindowTitle("Network Analysis Configuration"); self.setModal(False); self._setup_ui()
        
    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        param_group = QtWidgets.QGroupBox("Analysis Parameters"); param_layout = QtWidgets.QFormLayout(param_group)
        self.module_entry = QtWidgets.QLineEdit("All")
        self.module_entry.setToolTip("Modules (e.g., '1,2,5', '1-4', 'All')")
        self.module_entry.textChanged.connect(self._update_dac_scale_info)
        param_layout.addRow("Modules:", self.module_entry)
        self.fmin_edit = QtWidgets.QLineEdit(str(DEFAULT_MIN_FREQ / 1e6))
        self.fmax_edit = QtWidgets.QLineEdit(str(DEFAULT_MAX_FREQ / 1e6))
        param_layout.addRow("Min Frequency (MHz):", self.fmin_edit)
        param_layout.addRow("Max Frequency (MHz):", self.fmax_edit)
        self.cable_length_edit = QtWidgets.QLineEdit(str(DEFAULT_CABLE_LENGTH))
        param_layout.addRow("Cable Length (m):", self.cable_length_edit)        
        self.setup_amplitude_group(param_layout)
        self.points_edit = QtWidgets.QLineEdit(str(DEFAULT_NPOINTS))
        param_layout.addRow("Number of Points:", self.points_edit)
        self.samples_edit = QtWidgets.QLineEdit(str(DEFAULT_NSAMPLES))
        param_layout.addRow("Samples to Average:", self.samples_edit)
        self.max_chans_edit = QtWidgets.QLineEdit(str(DEFAULT_MAX_CHANNELS))
        param_layout.addRow("Max Channels:", self.max_chans_edit)
        self.max_span_edit = QtWidgets.QLineEdit(str(DEFAULT_MAX_SPAN / 1e6))
        param_layout.addRow("Max Span (MHz):", self.max_span_edit)
        self.clear_channels_cb = QtWidgets.QCheckBox("Clear all channels first"); self.clear_channels_cb.setChecked(True)
        param_layout.addRow("", self.clear_channels_cb); layout.addWidget(param_group)
        btn_layout = QtWidgets.QHBoxLayout(); self.start_btn = QtWidgets.QPushButton("Start Analysis")
        self.cancel_btn = QtWidgets.QPushButton("Cancel"); btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.cancel_btn); layout.addLayout(btn_layout)
        self.start_btn.clicked.connect(self.accept); self.cancel_btn.clicked.connect(self.reject)
        self._update_dbm_from_normalized(); self.setMinimumSize(500, 600)
        
    def _get_selected_modules(self):
        module_text = self.module_entry.text().strip(); selected_modules = []
        if module_text.lower() == 'all': selected_modules = list(range(1, 9))
        else:
            for part in module_text.split(','):
                part = part.strip()
                if '-' in part:
                    try: start, end = map(int, part.split('-')); selected_modules.extend(range(start, end + 1))
                    except ValueError: continue
                elif part:
                    try: selected_modules.append(int(part))
                    except ValueError: continue
        return selected_modules

    def get_parameters(self):
        try:
            module_text = self.module_entry.text().strip(); selected_module_param = None # Renamed
            if module_text.lower() != 'all':
                parsed_modules = self._get_selected_modules() # Renamed
                if parsed_modules: selected_module_param = parsed_modules
            amp_text = self.amp_edit.text().strip()
            amps_list = self._parse_amplitude_values(amp_text) or [DEFAULT_AMPLITUDE] # Renamed
            params_dict = {'amps': amps_list, 'module': selected_module_param, # Renamed
                           'fmin': float(eval(self.fmin_edit.text())) * 1e6, 
                           'fmax': float(eval(self.fmax_edit.text())) * 1e6,
                           'cable_length': float(self.cable_length_edit.text()),
                           'npoints': int(self.points_edit.text()), 'nsamps': int(self.samples_edit.text()),
                           'max_chans': int(self.max_chans_edit.text()), 
                           'max_span': float(eval(self.max_span_edit.text())) * 1e6,
                           'clear_channels': self.clear_channels_cb.isChecked()}
            return params_dict
        except Exception as e:
            traceback.print_exc(); QtWidgets.QMessageBox.critical(self, "Error", f"Invalid parameter: {str(e)}"); return None

class NetworkAnalysisParamsDialog(NetworkAnalysisDialogBase):
    def __init__(self, parent=None, params=None):
        super().__init__(parent, params); self.setWindowTitle("Edit Network Analysis Parameters"); self.setModal(True); self._setup_ui()
        if parent and hasattr(parent, 'parent') and parent.parent() is not None:
            parent_main = parent.parent()
            if hasattr(parent_main, 'crs') and parent_main.crs is not None: self._fetch_dac_scales(parent_main.crs)
        
    def _fetch_dac_scales(self, crs_obj): # Renamed crs
        self.fetcher = DACScaleFetcher(crs_obj) # DACScaleFetcher from .tasks
        self.fetcher.dac_scales_ready.connect(self._on_dac_scales_ready)
        self.fetcher.start()
    
    def _on_dac_scales_ready(self, scales_dict): # Renamed scales
        self.dac_scales = scales_dict; self._update_dac_scale_info(); self._update_dbm_from_normalized()
    
    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self); form = QtWidgets.QFormLayout()
        fmin_mhz = str(self.params.get('fmin', DEFAULT_MIN_FREQ) / 1e6)
        fmax_mhz = str(self.params.get('fmax', DEFAULT_MAX_FREQ) / 1e6)
        self.fmin_edit = QtWidgets.QLineEdit(fmin_mhz); self.fmax_edit = QtWidgets.QLineEdit(fmax_mhz)
        form.addRow("Min Frequency (MHz):", self.fmin_edit); form.addRow("Max Frequency (MHz):", self.fmax_edit)
        self.setup_amplitude_group(form)
        self.points_edit = QtWidgets.QLineEdit(str(self.params.get('npoints', DEFAULT_NPOINTS)))
        form.addRow("Number of Points:", self.points_edit)
        self.samples_edit = QtWidgets.QLineEdit(str(self.params.get('nsamps', DEFAULT_NSAMPLES)))
        form.addRow("Samples to Average:", self.samples_edit)
        self.max_chans_edit = QtWidgets.QLineEdit(str(self.params.get('max_chans', DEFAULT_MAX_CHANNELS)))
        form.addRow("Max Channels:", self.max_chans_edit)
        max_span_mhz = str(self.params.get('max_span', DEFAULT_MAX_SPAN) / 1e6)
        self.max_span_edit = QtWidgets.QLineEdit(max_span_mhz); form.addRow("Max Span (MHz):", self.max_span_edit)
        self.clear_channels_cb = QtWidgets.QCheckBox("Clear all channels first")
        self.clear_channels_cb.setChecked(self.params.get('clear_channels', True)); form.addRow("", self.clear_channels_cb)
        layout.addLayout(form); btn_layout = QtWidgets.QHBoxLayout()
        self.ok_btn = QtWidgets.QPushButton("OK"); self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.ok_btn); btn_layout.addWidget(self.cancel_btn); layout.addLayout(btn_layout)
        self.ok_btn.clicked.connect(self.accept); self.cancel_btn.clicked.connect(self.reject)
        self._update_dbm_from_normalized(); self.setMinimumSize(500, 600)

    def _get_selected_modules(self):
        selected_module_param = self.params.get('module') # Renamed
        if selected_module_param is None: return list(range(1, 9))
        return selected_module_param if isinstance(selected_module_param, list) else [selected_module_param]
    
    def get_parameters(self):
        try:
            amp_text = self.amp_edit.text().strip()
            amps_list = self._parse_amplitude_values(amp_text) or [DEFAULT_AMPLITUDE] # Renamed
            params_dict = self.params.copy() # Renamed
            params_dict.update({'amps': amps_list, 'amp': amps_list[0],
                                'fmin': float(eval(self.fmin_edit.text())) * 1e6,
                                'fmax': float(eval(self.fmax_edit.text())) * 1e6,
                                'npoints': int(self.points_edit.text()), 'nsamps': int(self.samples_edit.text()),
                                'max_chans': int(self.max_chans_edit.text()),
                                'max_span': float(eval(self.max_span_edit.text())) * 1e6,
                                'clear_channels': self.clear_channels_cb.isChecked()})
            return params_dict
        except Exception as e: QtWidgets.QMessageBox.critical(self, "Error", f"Invalid parameter: {str(e)}"); return None

class InitializeCRSDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, crs_obj=None):
        super().__init__(parent); self.crs = crs_obj 
        self.setWindowTitle("Initialize CRS Board"); self.setModal(True); layout = QtWidgets.QVBoxLayout(self)
        irig_group = QtWidgets.QGroupBox("IRIG Time Source"); irig_layout = QtWidgets.QVBoxLayout(irig_group)
        self.rb_backplane = QtWidgets.QRadioButton("BACKPLANE"); self.rb_test = QtWidgets.QRadioButton("TEST")
        self.rb_sma = QtWidgets.QRadioButton("SMA"); self.rb_test.setChecked(True) 
        irig_layout.addWidget(self.rb_backplane); irig_layout.addWidget(self.rb_test); irig_layout.addWidget(self.rb_sma)
        layout.addWidget(irig_group)
        self.cb_clear_channels = QtWidgets.QCheckBox("Clear all channels on this module"); self.cb_clear_channels.setChecked(True)
        layout.addWidget(self.cb_clear_channels); btn_layout = QtWidgets.QHBoxLayout()
        self.ok_btn = QtWidgets.QPushButton("OK"); self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addStretch(); btn_layout.addWidget(self.ok_btn); btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout); self.ok_btn.clicked.connect(self.accept); self.cancel_btn.clicked.connect(self.reject)

    def get_selected_irig_source(self):
        if self.crs is None: return None 
        if self.rb_backplane.isChecked(): return self.crs.TIMESTAMP_PORT.BACKPLANE
        if self.rb_test.isChecked(): return self.crs.TIMESTAMP_PORT.TEST
        if self.rb_sma.isChecked(): return self.crs.TIMESTAMP_PORT.SMA
        return None
    def get_clear_channels_state(self) -> bool: return self.cb_clear_channels.isChecked()

class FindResonancesDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent); self.setWindowTitle("Find Resonances Parameters"); self.setModal(True); self._setup_ui()
    def _setup_ui(self):
        layout = QtWidgets.QFormLayout(self)
        self.expected_resonances_edit = QtWidgets.QLineEdit(); self.expected_resonances_edit.setPlaceholderText("Optional (e.g., 10)")
        regex_int_or_empty = QRegularExpression("^$|^[1-9][0-9]*$")
        self.expected_resonances_edit.setValidator(QRegularExpressionValidator(regex_int_or_empty, self))
        layout.addRow("Expected Resonances:", self.expected_resonances_edit)
        self.min_dip_depth_db_edit = QtWidgets.QLineEdit(str(1.0))
        self.min_dip_depth_db_edit.setValidator(QDoubleValidator(0.01, 100.0, 2, self))
        layout.addRow("Min Dip Depth (dB):", self.min_dip_depth_db_edit)
        self.min_Q_edit = QtWidgets.QLineEdit(str(1e4)); self.min_Q_edit.setValidator(QDoubleValidator(1.0, 1e9, 0, self))
        layout.addRow("Min Q:", self.min_Q_edit)
        self.max_Q_edit = QtWidgets.QLineEdit(str(1e7)); self.max_Q_edit.setValidator(QDoubleValidator(1.0, 1e9, 0, self))
        layout.addRow("Max Q:", self.max_Q_edit)
        self.min_resonance_separation_mhz_edit = QtWidgets.QLineEdit(str(0.1))
        self.min_resonance_separation_mhz_edit.setValidator(QDoubleValidator(0.001, 1000.0, 3, self))
        layout.addRow("Min Separation (MHz):", self.min_resonance_separation_mhz_edit)
        self.data_exponent_edit = QtWidgets.QLineEdit(str(2.0)); self.data_exponent_edit.setValidator(QDoubleValidator(0.1, 10.0, 2, self))
        layout.addRow("Data Exponent:", self.data_exponent_edit)
        self.button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept); self.button_box.rejected.connect(self.reject); layout.addRow(self.button_box)

    def get_parameters(self) -> dict | None:
        params_dict = {} # Renamed params
        try:
            expected_text = self.expected_resonances_edit.text().strip()
            params_dict['expected_resonances'] = int(expected_text) if expected_text else None
            params_dict['min_dip_depth_db'] = float(self.min_dip_depth_db_edit.text())
            params_dict['min_Q'] = float(self.min_Q_edit.text()); params_dict['max_Q'] = float(self.max_Q_edit.text())
            params_dict['min_resonance_separation_hz'] = float(self.min_resonance_separation_mhz_edit.text()) * 1e6
            params_dict['data_exponent'] = float(self.data_exponent_edit.text())
            if params_dict['min_Q'] >= params_dict['max_Q']: QtWidgets.QMessageBox.warning(self, "Validation Error", "Min Q must be less than Max Q."); return None
            if params_dict['min_dip_depth_db'] <=0: QtWidgets.QMessageBox.warning(self, "Validation Error", "Min Dip Depth must be positive."); return None
            return params_dict
        except ValueError as e: QtWidgets.QMessageBox.critical(self, "Input Error", f"Invalid input: {str(e)}"); return None

class MultisweepDialog(NetworkAnalysisDialogBase):
    def __init__(self, parent=None, resonance_frequencies: list[float] | None = None, dac_scales=None, current_module=None, initial_params=None):
        super().__init__(parent, params=initial_params, dac_scales=dac_scales)
        self.resonance_frequencies = resonance_frequencies or []; self.current_module = current_module
        self.setWindowTitle("Multisweep Configuration"); self.setModal(True); self._setup_ui()
        if parent and hasattr(parent, 'parent') and parent.parent() is not None:
            main_periscope_window = parent.parent()
            if hasattr(main_periscope_window, 'crs') and main_periscope_window.crs is not None:
                if not self.dac_scales and hasattr(self, '_fetch_dac_scales_for_dialog'):
                     self._fetch_dac_scales_for_dialog(main_periscope_window.crs)
                elif self.dac_scales: self._update_dac_scale_info(); self._update_dbm_from_normalized()

    def _fetch_dac_scales_for_dialog(self, crs_obj): # Renamed crs
        self.fetcher = DACScaleFetcher(crs_obj) # DACScaleFetcher from .tasks
        self.fetcher.dac_scales_ready.connect(self._on_dac_scales_ready_dialog)
        self.fetcher.start()

    def _on_dac_scales_ready_dialog(self, scales_dict): # Renamed scales
        self.dac_scales = scales_dict; self._update_dac_scale_info(); self._update_dbm_from_normalized()

    def _get_selected_modules(self): return [self.current_module] if self.current_module is not None else []

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        res_info_group = QtWidgets.QGroupBox("Target Resonances"); res_info_layout = QtWidgets.QVBoxLayout(res_info_group)
        num_resonances = len(self.resonance_frequencies)
        res_label_text = f"Number of resonances to sweep: {num_resonances}"
        if num_resonances > 0:
            res_freq_mhz_str = ", ".join([f"{f / 1e6:.3f}" for f in self.resonance_frequencies[:5]])
            if num_resonances > 5: res_freq_mhz_str += ", ..."
            res_label_text += f"\nFrequencies (MHz): {res_freq_mhz_str}"
        self.resonances_info_label = QtWidgets.QLabel(res_label_text); self.resonances_info_label.setWordWrap(True)
        res_info_layout.addWidget(self.resonances_info_label); layout.addWidget(res_info_group)
        param_group = QtWidgets.QGroupBox("Sweep Parameters"); param_form_layout = QtWidgets.QFormLayout(param_group)
        default_span_khz = self.params.get('span_hz', 100000.0) / 1e3
        self.span_khz_edit = QtWidgets.QLineEdit(str(default_span_khz))
        self.span_khz_edit.setValidator(QDoubleValidator(0.1, 10000.0, 2, self))
        param_form_layout.addRow("Span per Resonance (kHz):", self.span_khz_edit)
        default_npoints = self.params.get('npoints_per_sweep', 101)
        self.npoints_edit = QtWidgets.QLineEdit(str(default_npoints)); self.npoints_edit.setValidator(QIntValidator(2, 10000, self))
        param_form_layout.addRow("Number of Points per Sweep:", self.npoints_edit)
        default_nsamps = self.params.get('nsamps', DEFAULT_NSAMPLES) # DEFAULT_NSAMPLES from .utils
        self.nsamps_edit = QtWidgets.QLineEdit(str(default_nsamps)); self.nsamps_edit.setValidator(QIntValidator(1, 10000, self))
        param_form_layout.addRow("Samples to Average (nsamps):", self.nsamps_edit)
        self.setup_amplitude_group(param_form_layout)
        default_perform_fits = self.params.get('perform_fits', True)
        self.perform_fits_cb = QtWidgets.QCheckBox("Perform fits after sweep"); self.perform_fits_cb.setChecked(default_perform_fits)
        param_form_layout.addRow("", self.perform_fits_cb)
        default_recalc_cf = self.params.get('recalculate_center_frequencies', True)
        self.recalculate_cf_cb = QtWidgets.QCheckBox("Recalculate Center Frequencies (S21 min)")
        self.recalculate_cf_cb.setChecked(default_recalc_cf)
        self.recalculate_cf_cb.setToolTip("If checked, S21 min will be used as center for results/plotting.")
        param_form_layout.addRow("", self.recalculate_cf_cb); layout.addWidget(param_group)
        self.button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept); self.button_box.rejected.connect(self.reject); layout.addWidget(self.button_box)
        if self.dac_scales: self._update_dbm_from_normalized()
        self.setMinimumWidth(500)

    def get_parameters(self) -> dict | None:
        params_dict = {} # Renamed params
        try:
            amp_text = self.amp_edit.text().strip()
            amps_list = self._parse_amplitude_values(amp_text) or [self.params.get('amp', DEFAULT_AMPLITUDE)] # Renamed
            params_dict['amps'] = amps_list
            params_dict['span_hz'] = float(self.span_khz_edit.text()) * 1e3
            params_dict['npoints_per_sweep'] = int(self.npoints_edit.text())
            params_dict['nsamps'] = int(self.nsamps_edit.text())
            params_dict['perform_fits'] = self.perform_fits_cb.isChecked()
            params_dict['recalculate_center_frequencies'] = self.recalculate_cf_cb.isChecked()
            params_dict['resonance_frequencies'] = self.resonance_frequencies; params_dict['module'] = self.current_module
            if params_dict['span_hz'] <= 0: QtWidgets.QMessageBox.warning(self, "Validation Error", "Span must be positive."); return None
            if params_dict['npoints_per_sweep'] < 2: QtWidgets.QMessageBox.warning(self, "Validation Error", "Points must be >= 2."); return None
            if params_dict['nsamps'] < 1: QtWidgets.QMessageBox.warning(self, "Validation Error", "Samples must be >= 1."); return None
            return params_dict
        except ValueError as e: QtWidgets.QMessageBox.critical(self, "Input Error", f"Invalid input: {str(e)}"); return None
        except Exception as e: QtWidgets.QMessageBox.critical(self, "Error", f"Could not parse parameters: {str(e)}"); return None
