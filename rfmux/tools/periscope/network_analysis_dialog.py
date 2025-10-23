"""Network analysis parameter dialogs."""

from .utils import (
    QtWidgets, QtCore, QRegularExpression, QRegularExpressionValidator,
    QDoubleValidator, QIntValidator,
    DEFAULT_AMPLITUDE, DEFAULT_MIN_FREQ, DEFAULT_MAX_FREQ, DEFAULT_CABLE_LENGTH,
    DEFAULT_NPOINTS, DEFAULT_NSAMPLES, DEFAULT_MAX_CHANNELS, DEFAULT_MAX_SPAN,
    UnitConverter, traceback
)
import pickle
from .tasks import DACScaleFetcher
from .network_analysis_base import NetworkAnalysisDialogBase


def load_network_analysis_payload(parent: QtWidgets.QWidget, file_path: str | None = None):

    # If no filename was passed, fall back to showing a dialog (blocking version)
    if file_path is None:
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.Option.DontUseNativeDialog
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent,
            "Load Network Analysis Parameters",
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
        QtWidgets.QMessageBox.critical(
            parent,
            "Load Failed",
            f"Could not read '{file_path}':\n{exc}",
        )
        return None

    if (
        isinstance(payload, dict)
        and isinstance(payload.get("parameters"), dict)
        and isinstance(payload.get("modules"), dict)
    ):
        return payload

    QtWidgets.QMessageBox.warning(
        parent,
        "Invalid File",
        "The selected file does not contain network-analysis parameters.",
    )
    return None



class NetworkAnalysisDialog(NetworkAnalysisDialogBase):
    """
    Dialog for configuring and initiating a new Network Analysis.
    It allows users to specify parameters like frequency range, amplitude,
    modules to scan, and other analysis settings.
    """
    def __init__(self, parent: QtWidgets.QWidget = None, modules: list[int] = None,
                 dac_scales: dict[int, float] = None):
        """
        Initializes the Network Analysis configuration dialog.

        Args:
            parent: The parent widget.
            modules: List of available module numbers.
            dac_scales: Pre-fetched DAC scales for the modules.
        """
        super().__init__(parent, params=None, modules=modules, dac_scales=dac_scales)
        self.setWindowTitle("Network Analysis Configuration")
        self.setModal(False) # Modeless dialog
        self._setup_ui()
        self.load_data_available = False
        self._load_data = {}
        
    def _setup_ui(self):
        """Sets up the user interface elements for the dialog."""
        layout = QtWidgets.QVBoxLayout(self)

        self.import_button = QtWidgets.QPushButton("Import Data")
        self.import_button.clicked.connect(self._load_netanal_data)
        layout.addWidget(self.import_button, alignment=QtCore.Qt.AlignLeft)
        
        param_group = QtWidgets.QGroupBox("Analysis Parameters")
        param_layout = QtWidgets.QFormLayout(param_group)
        
        self.module_entry = QtWidgets.QLineEdit("All")
        self.module_entry.setToolTip("Specify modules to analyze (e.g., '1,2,5', '1-4', or 'All' for modules 1-8).")
        self.module_entry.textChanged.connect(self._update_dac_scale_info) # Update DAC info when module selection changes
        param_layout.addRow("Modules:", self.module_entry)
        
        self.fmin_edit = QtWidgets.QLineEdit(str(DEFAULT_MIN_FREQ / 1e6)) # Default in MHz
        self.fmax_edit = QtWidgets.QLineEdit(str(DEFAULT_MAX_FREQ / 1e6)) # Default in MHz
        param_layout.addRow("Min Frequency (MHz):", self.fmin_edit)
        param_layout.addRow("Max Frequency (MHz):", self.fmax_edit)

        self.cable_length_edit = QtWidgets.QLineEdit(str(DEFAULT_CABLE_LENGTH)) # Default cable length
        param_layout.addRow("Cable Length (m):", self.cable_length_edit)
        
        self.setup_amplitude_group(param_layout) # Add shared amplitude settings
        
        self.points_edit = QtWidgets.QLineEdit(str(DEFAULT_NPOINTS))
        param_layout.addRow("Number of Points:", self.points_edit)
        
        self.samples_edit = QtWidgets.QLineEdit(str(DEFAULT_NSAMPLES))
        param_layout.addRow("Samples to Average:", self.samples_edit)
        
        self.max_chans_edit = QtWidgets.QLineEdit(str(DEFAULT_MAX_CHANNELS))
        param_layout.addRow("Max Channels:", self.max_chans_edit)
        
        self.max_span_edit = QtWidgets.QLineEdit(str(DEFAULT_MAX_SPAN / 1e6)) # Default in MHz
        param_layout.addRow("Max Span (MHz):", self.max_span_edit)
        
        self.clear_channels_cb = QtWidgets.QCheckBox("Clear all channels first")
        self.clear_channels_cb.setChecked(True) # Default to clearing channels
        param_layout.addRow("", self.clear_channels_cb)
        
        layout.addWidget(param_group)
        
        # Buttons for starting or canceling the analysis
        btn_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start Analysis")
        self.load_btn = QtWidgets.QPushButton("Load Analysis")
        self.load_btn.setEnabled(False) ### Will enable once file is available.
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        
        self.start_btn.clicked.connect(self.accept) # Connect to QDialog's accept slot

        self.load_btn.clicked.connect(self._load_data_avail) 
        
        self.cancel_btn.clicked.connect(self.reject) # Connect to QDialog's reject slot
        
        self._update_dbm_from_normalized() # Initial update of dBm field based on default amplitude
        self.setMinimumSize(500, 600) # Set a reasonable minimum size
        
    def _get_selected_modules(self) -> list[int]:
        """
        Parses the module entry text to determine the list of selected modules.
        Supports "All", comma-separated values, and ranges (e.g., "1-4").

        Returns:
            A list of integer module IDs. Returns modules 1-8 if "All" is specified.
        """
        module_text = self.module_entry.text().strip()
        selected_modules = []
        if module_text.lower() == 'all':
            selected_modules = list(range(1, 9)) # "All" implies modules 1 through 8
        else:
            for part in module_text.split(','):
                part = part.strip()
                if '-' in part: # Handle ranges like "1-4"
                    try:
                        start, end = map(int, part.split('-'))
                        if start <= end: # Ensure valid range
                            selected_modules.extend(range(start, end + 1))
                    except ValueError:
                        continue # Skip malformed range parts
                elif part: # Handle single numbers
                    try:
                        selected_modules.append(int(part))
                    except ValueError:
                        continue # Skip non-integer parts
        # Remove duplicates and sort, though current logic might not produce duplicates if input is clean.
        return sorted(list(set(selected_modules)))

    
    def _load_data_avail(self):
        ''' Use loaded file parameters, accept the dialog'''
        self.load_data_available = True
        self.accept()


    def _load_netanal_data(self):
        """
        Asynchronously trigger a non-blocking QFileDialog using the main thread.
        Using QTimer avoids interfering with existing threads.
        """
        QtCore.QTimer.singleShot(0, self._open_file_dialog_async)


    def _open_file_dialog_async(self):
        """Open a non-blocking file dialog for selecting a Network Analysis parameter file."""
        if not hasattr(self, "_file_dialog") or self._file_dialog is None:
            self._file_dialog = QtWidgets.QFileDialog(self, "Load Network Analysis Parameters")
            self._file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
    
            self._file_dialog.setNameFilters([
                "Pickle Files (*.pkl *.pickle)",
                "All Files (*)",
            ])
    
            # Force non-native + non-blocking behavior
            self._file_dialog.setOptions(
                QtWidgets.QFileDialog.Option.DontUseNativeDialog
                | QtWidgets.QFileDialog.Option.ReadOnly
            )
            self._file_dialog.setModal(False)
    
            # Connect signals
            self._file_dialog.fileSelected.connect(self._on_file_selected)
            self._file_dialog.rejected.connect(self._on_file_dialog_closed)
    
        # Show the dialog async
        self._file_dialog.open()


    @QtCore.pyqtSlot(str)
    def _on_file_selected(self, path: str):
        """Handle file selection, load bias data, and populate the UI fields with file contents."""
        payload = load_network_analysis_payload(self, file_path=path)
        if payload is None:
            return
    
        self.load_btn.setEnabled(True)
        self._load_data = payload.copy()
    
        params = payload["parameters"]
    
        module_value = params.get("module")
        if module_value is None:
            module_text = "All"
        elif isinstance(module_value, (list, tuple, set)):
            module_text = ", ".join(str(int(m)) for m in module_value)
        else:
            module_text = str(module_value)
        self.module_entry.setText(module_text)
    
        amps = params.get("amps") or ([params["amp"]] if "amp" in params else None)
        if amps:
            try:
                amp_text = ", ".join(f"{float(amp):g}" for amp in amps)
            except (TypeError, ValueError):
                amp_text = ", ".join(str(amp) for amp in amps)
            self.amp_edit.setText(amp_text)
    
        def set_if_present(key, widget, formatter):
            if key in params and params[key] is not None:
                try:
                    widget.setText(formatter(params[key]))
                except Exception:
                    widget.setText(str(params[key]))
    
        set_if_present("fmin", self.fmin_edit, lambda v: f"{float(v) / 1e6:g}")
        set_if_present("fmax", self.fmax_edit, lambda v: f"{float(v) / 1e6:g}")
        set_if_present("cable_length", self.cable_length_edit, lambda v: f"{float(v):g}")
        set_if_present("npoints", self.points_edit, lambda v: str(int(float(v))))
        set_if_present("nsamps", self.samples_edit, lambda v: str(int(float(v))))
        set_if_present("max_chans", self.max_chans_edit, lambda v: str(int(float(v))))
        set_if_present("max_span", self.max_span_edit, lambda v: f"{float(v) / 1e6:g}")
    
        if "clear_channels" in params:
            self.clear_channels_cb.setChecked(bool(params["clear_channels"]))
    
        self._update_dac_scale_info()
        self._update_dbm_from_normalized()


    @QtCore.pyqtSlot()
    def _on_file_dialog_closed(self):
        """Handle the event when the file dialog is closed without selection."""
        pass
        
    def get_parameters(self) -> dict | None:
        """
        Retrieves and validates the network analysis parameters from the UI fields.

        Returns:
            A dictionary of parameters if valid, otherwise None.
            Shows an error message on invalid input.
        """
        try:
            if self.load_data_available:
                return self._load_data
            else:
                module_text = self.module_entry.text().strip()
                selected_module_param = None # Parameter for 'module' key
                if module_text.lower() != 'all':
                    parsed_modules = self._get_selected_modules()
                    if parsed_modules:
                        selected_module_param = parsed_modules
                
                amp_text = self.amp_edit.text().strip()
                # Use parsed amplitude values, or default if input is empty
                amps_list = self._parse_amplitude_values(amp_text) or [DEFAULT_AMPLITUDE]
                
                # Construct parameters dictionary
                # Using eval for frequency and span allows expressions, but ensure inputs are numbers.
                # Consider replacing eval with direct float conversion if expressions aren't strictly needed.
                params_dict = {
                    'amps': amps_list,
                    'module': selected_module_param, # Can be None (interpreted as all by backend) or list of modules
                    'fmin': float(eval(self.fmin_edit.text())) * 1e6,  # Convert MHz to Hz
                    'fmax': float(eval(self.fmax_edit.text())) * 1e6,  # Convert MHz to Hz
                    'cable_length': float(self.cable_length_edit.text()),
                    'npoints': int(self.points_edit.text()),
                    'nsamps': int(self.samples_edit.text()),
                    'max_chans': int(self.max_chans_edit.text()),
                    'max_span': float(eval(self.max_span_edit.text())) * 1e6, # Convert MHz to Hz
                    'clear_channels': self.clear_channels_cb.isChecked()
                }
                # Basic validation for frequency range
                if params_dict['fmin'] >= params_dict['fmax']:
                    QtWidgets.QMessageBox.warning(self, "Input Error", "Min Frequency must be less than Max Frequency.")
                    return None
                return params_dict
        except Exception as e:
            traceback.print_exc() # Log the full traceback for debugging
            QtWidgets.QMessageBox.critical(self, "Error Parsing Parameters", f"Invalid parameter input: {str(e)}")
            return None


class NetworkAnalysisParamsDialog(NetworkAnalysisDialogBase):
    """
    Dialog for editing existing Network Analysis parameters.
    This dialog is typically modal and pre-filled with current analysis parameters.
    It fetches DAC scales asynchronously if a CRS object is available from its parent.
    """
    def __init__(self, parent: QtWidgets.QWidget = None, params: dict = None):
        """
        Initializes the dialog for editing network analysis parameters.

        Args:
            parent: The parent widget.
            params: Dictionary of existing parameters to populate the fields.
        """
        super().__init__(parent, params=params) # Pass params to base class
        self.setWindowTitle("Edit Network Analysis Parameters")
        self.setModal(True) # Modal dialog
        self._setup_ui()

        # Attempt to fetch DAC scales if CRS is available from the main window hierarchy
        if parent and hasattr(parent, 'parent') and parent.parent() is not None:
            # Assuming parent.parent() is the main Periscope window
            main_periscope_window = parent.parent() 
            if hasattr(main_periscope_window, 'crs') and main_periscope_window.crs is not None:
                self._fetch_dac_scales(main_periscope_window.crs)
        
    def _fetch_dac_scales(self, crs_obj):
        """
        Initiates asynchronous fetching of DAC scales using DACScaleFetcher.

        Args:
            crs_obj: The CRS (Control and Readout System) object to query for scales.
        """
        self.fetcher = DACScaleFetcher(crs_obj)
        self.fetcher.dac_scales_ready.connect(self._on_dac_scales_ready)
        self.fetcher.start() # Start the QThread for fetching
    
    @QtCore.pyqtSlot(dict)
    def _on_dac_scales_ready(self, scales_dict: dict[int, float]):
        """
        Slot to handle the reception of fetched DAC scales.
        Updates the internal DAC scales and refreshes relevant UI elements.

        Args:
            scales_dict: Dictionary mapping module ID to DAC scale (dBm).
        """
        self.dac_scales = scales_dict
        self._update_dac_scale_info() # Update the DAC scale display label
        self._update_dbm_from_normalized() # Recalculate dBm based on new scales
    
    def _setup_ui(self):
        """Sets up the user interface elements for the dialog."""
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        
        # Populate fields with existing parameters or defaults
        fmin_mhz = str(self.params.get('fmin', DEFAULT_MIN_FREQ) / 1e6)
        fmax_mhz = str(self.params.get('fmax', DEFAULT_MAX_FREQ) / 1e6)
        self.fmin_edit = QtWidgets.QLineEdit(fmin_mhz)
        self.fmax_edit = QtWidgets.QLineEdit(fmax_mhz)
        form.addRow("Min Frequency (MHz):", self.fmin_edit)
        form.addRow("Max Frequency (MHz):", self.fmax_edit)
        
        self.setup_amplitude_group(form) # Add shared amplitude settings
        
        self.points_edit = QtWidgets.QLineEdit(str(self.params.get('npoints', DEFAULT_NPOINTS)))
        form.addRow("Number of Points:", self.points_edit)
        
        self.samples_edit = QtWidgets.QLineEdit(str(self.params.get('nsamps', DEFAULT_NSAMPLES)))
        form.addRow("Samples to Average:", self.samples_edit)
        
        self.max_chans_edit = QtWidgets.QLineEdit(str(self.params.get('max_chans', DEFAULT_MAX_CHANNELS)))
        form.addRow("Max Channels:", self.max_chans_edit)
        
        max_span_mhz = str(self.params.get('max_span', DEFAULT_MAX_SPAN) / 1e6)
        self.max_span_edit = QtWidgets.QLineEdit(max_span_mhz)
        form.addRow("Max Span (MHz):", self.max_span_edit)
        
        self.clear_channels_cb = QtWidgets.QCheckBox("Clear all channels first")
        self.clear_channels_cb.setChecked(self.params.get('clear_channels', True))
        form.addRow("", self.clear_channels_cb)
        
        layout.addLayout(form)
        
        # OK and Cancel buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        
        # Initial update of dBm field, especially if DAC scales were passed in constructor
        # or if _fetch_dac_scales is not called (e.g., no CRS object).
        self._update_dac_scale_info() # Call this first to set up dac_scale_info label correctly
        self._update_dbm_from_normalized() 
        self.setMinimumSize(500, 600)

    def _get_selected_modules(self) -> list[int]:
        """
        Determines the modules relevant for DAC scale display in this dialog.
        Uses the 'module' parameter passed during initialization. If 'module'
        is None or not specified, it defaults to all modules (1-8).

        Returns:
            A list of integer module IDs.
        """
        selected_module_param = self.params.get('module')
        if selected_module_param is None:
            # If no specific module(s) defined in params, assume all for DAC display purposes
            return list(range(1, 9)) 
        # Ensure it's a list, even if a single int was passed in params
        return selected_module_param if isinstance(selected_module_param, list) else [selected_module_param]
    
    def get_parameters(self) -> dict | None:
        """
        Retrieves and validates the edited network analysis parameters.

        Returns:
            A dictionary of parameters if valid, otherwise None.
            Shows an error message on invalid input.
        """
        try:
            amp_text = self.amp_edit.text().strip()
            amps_list = self._parse_amplitude_values(amp_text) or [self.params.get('amp', DEFAULT_AMPLITUDE)]
            
            # Start with a copy of existing params and update with UI values
            params_dict = self.params.copy() 
            params_dict.update({
                'amps': amps_list,
                'amp': amps_list[0] if amps_list else DEFAULT_AMPLITUDE, # Update single 'amp' for compatibility
                'fmin': float(eval(self.fmin_edit.text())) * 1e6,
                'fmax': float(eval(self.fmax_edit.text())) * 1e6,
                'npoints': int(self.points_edit.text()),
                'nsamps': int(self.samples_edit.text()),
                'max_chans': int(self.max_chans_edit.text()),
                'max_span': float(eval(self.max_span_edit.text())) * 1e6,
                'clear_channels': self.clear_channels_cb.isChecked()
            })
            # Basic validation for frequency range
            if params_dict['fmin'] >= params_dict['fmax']:
                QtWidgets.QMessageBox.warning(self, "Input Error", "Min Frequency must be less than Max Frequency.")
                return None
            return params_dict
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error Parsing Parameters", f"Invalid parameter input: {str(e)}")
