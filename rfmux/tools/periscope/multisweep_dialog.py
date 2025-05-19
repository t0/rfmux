"""Dialog for multisweep settings."""

from .utils import (
    QtWidgets, QtCore, QRegularExpression, QRegularExpressionValidator,
    QDoubleValidator, QIntValidator,
    DEFAULT_AMPLITUDE, DEFAULT_MIN_FREQ, DEFAULT_MAX_FREQ, DEFAULT_CABLE_LENGTH,
    DEFAULT_NPOINTS, DEFAULT_NSAMPLES, DEFAULT_MAX_CHANNELS, DEFAULT_MAX_SPAN,
    UnitConverter, traceback
)
from .network_analysis_base import NetworkAnalysisDialogBase

class MultisweepDialog(NetworkAnalysisDialogBase):
    """
    Dialog for configuring a Multisweep operation.
    Inherits from NetworkAnalysisDialogBase for amplitude settings and DAC scale handling.
    Allows specifying parameters for sweeping multiple pre-identified resonances.
    """
    def __init__(self, parent: QtWidgets.QWidget = None, 
                 resonance_frequencies: list[float] | None = None, 
                 dac_scales: dict[int, float] = None, 
                 current_module: int | None = None, 
                 initial_params: dict | None = None):
        """
        Initializes the Multisweep configuration dialog.

        Args:
            parent: The parent widget.
            resonance_frequencies: List of resonance frequencies (in Hz) to target.
            dac_scales: Pre-fetched DAC scales.
            current_module: The module ID on which the multisweep will be performed.
            initial_params: Dictionary of initial parameters to populate fields.
        """
        super().__init__(parent, params=initial_params, dac_scales=dac_scales)
        self.resonance_frequencies = resonance_frequencies or []
        self.current_module = current_module # Store the current module for DAC scale and params

        self.setWindowTitle("Multisweep Configuration")
        self.setModal(True)
        self._setup_ui()

        # Asynchronously fetch DAC scales if not provided and CRS is available
        # This is similar to NetworkAnalysisParamsDialog logic
        if parent and hasattr(parent, 'parent') and parent.parent() is not None:
            main_periscope_window = parent.parent()
            if hasattr(main_periscope_window, 'crs') and main_periscope_window.crs is not None:
                # Only fetch if dac_scales weren't passed in and we have a method to do so
                if not self.dac_scales and hasattr(self, '_fetch_dac_scales_for_dialog'):
                     self._fetch_dac_scales_for_dialog(main_periscope_window.crs)
                elif self.dac_scales: # If scales were provided, update UI
                    self._update_dac_scale_info()
                    self._update_dbm_from_normalized()

    def _fetch_dac_scales_for_dialog(self, crs_obj):
        """
        Initiates asynchronous fetching of DAC scales for this dialog.
        This method is specific to MultisweepDialog if its DAC fetching needs
        to be handled differently or if it's called from a different context.
        Currently, it's similar to the one in NetworkAnalysisParamsDialog.

        Args:
            crs_obj: The CRS object to query.
        """
        self.fetcher = DACScaleFetcher(crs_obj)
        self.fetcher.dac_scales_ready.connect(self._on_dac_scales_ready_dialog)
        self.fetcher.start()

    @QtCore.pyqtSlot(dict)
    def _on_dac_scales_ready_dialog(self, scales_dict: dict[int, float]):
        """
        Slot to handle received DAC scales specifically for this dialog instance.
        Updates DAC scales and refreshes relevant UI parts.

        Args:
            scales_dict: Dictionary of module ID to DAC scale (dBm).
        """
        self.dac_scales = scales_dict
        self._update_dac_scale_info()
        self._update_dbm_from_normalized()

    def _get_selected_modules(self) -> list[int]:
        """
        Returns the module relevant for this multisweep dialog.
        For multisweep, it's typically a single, pre-determined module.

        Returns:
            A list containing the current_module ID if set, otherwise an empty list.
        """
        return [self.current_module] if self.current_module is not None else []

    def _setup_ui(self):
        """Sets up the user interface elements for the Multisweep dialog."""
        layout = QtWidgets.QVBoxLayout(self)

        # Display information about target resonances
        res_info_group = QtWidgets.QGroupBox("Target Resonances")
        res_info_layout = QtWidgets.QVBoxLayout(res_info_group)
        num_resonances = len(self.resonance_frequencies)
        res_label_text = f"Number of resonances to sweep: {num_resonances}"
        if num_resonances > 0:
            # Show first few resonance frequencies for quick reference
            res_freq_mhz_str = ", ".join([f"{f / 1e6:.3f}" for f in self.resonance_frequencies[:5]])
            if num_resonances > 5:
                res_freq_mhz_str += ", ..." # Indicate more frequencies exist
            res_label_text += f"\nFrequencies (MHz): {res_freq_mhz_str}"
        self.resonances_info_label = QtWidgets.QLabel(res_label_text)
        self.resonances_info_label.setWordWrap(True)
        res_info_layout.addWidget(self.resonances_info_label)
        layout.addWidget(res_info_group)

        # Sweep parameters group
        param_group = QtWidgets.QGroupBox("Sweep Parameters")
        param_form_layout = QtWidgets.QFormLayout(param_group)

        # Span per resonance (kHz)
        default_span_khz = self.params.get('span_hz', 100000.0) / 1e3 # Default 100 kHz
        self.span_khz_edit = QtWidgets.QLineEdit(str(default_span_khz))
        self.span_khz_edit.setValidator(QDoubleValidator(0.1, 10000.0, 2, self)) # Min 0.1 kHz, Max 10 MHz
        param_form_layout.addRow("Span per Resonance (kHz):", self.span_khz_edit)

        # Number of points per sweep
        default_npoints = self.params.get('npoints_per_sweep', 101) # Default 101 points
        self.npoints_edit = QtWidgets.QLineEdit(str(default_npoints))
        self.npoints_edit.setValidator(QIntValidator(2, 10000, self)) # Min 2 points
        param_form_layout.addRow("Number of Points per Sweep:", self.npoints_edit)

        # Samples to average (nsamps)
        default_nsamps = self.params.get('nsamps', DEFAULT_NSAMPLES)
        self.nsamps_edit = QtWidgets.QLineEdit(str(default_nsamps))
        self.nsamps_edit.setValidator(QIntValidator(1, 10000, self)) # Min 1 sample
        param_form_layout.addRow("Samples to Average (nsamps):", self.nsamps_edit)

        self.setup_amplitude_group(param_form_layout) # Shared amplitude settings

        # Option to perform fits
        default_perform_fits = self.params.get('perform_fits', True)
        self.perform_fits_cb = QtWidgets.QCheckBox("Perform fits after sweep")
        self.perform_fits_cb.setChecked(default_perform_fits)
        param_form_layout.addRow("", self.perform_fits_cb)

        # Option to recalculate center frequencies
        default_recalc_cf = self.params.get('recalculate_center_frequencies', True)
        self.recalculate_cf_cb = QtWidgets.QCheckBox("Recalculate Center Frequencies (S21 min)")
        self.recalculate_cf_cb.setChecked(default_recalc_cf)
        self.recalculate_cf_cb.setToolTip(
            "If checked, the minimum of S21 magnitude will be used as the center "
            "frequency for subsequent results display and plotting, potentially "
            "overriding the initially provided resonance frequency for that sweep."
        )
        param_form_layout.addRow("", self.recalculate_cf_cb)
        layout.addWidget(param_group)

        # Standard OK and Cancel buttons
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        # Initial update of dBm field if DAC scales are already known
        if self.dac_scales: # Check if dac_scales were passed or fetched synchronously before UI setup
            self._update_dac_scale_info() # Ensure info label is also up-to-date
            self._update_dbm_from_normalized()
        
        self.setMinimumWidth(500) # Ensure dialog is wide enough

    def get_parameters(self) -> dict | None:
        """
        Retrieves and validates the parameters for the multisweep operation.

        Returns:
            A dictionary of parameters if valid, otherwise None.
            Shows an error message on invalid input or validation failure.
        """
        params_dict = {}
        try:
            amp_text = self.amp_edit.text().strip()
            # Use parsed amplitude values, or default from initial params or global default
            default_amp_val = self.params.get('amp', DEFAULT_AMPLITUDE) if self.params else DEFAULT_AMPLITUDE
            amps_list = self._parse_amplitude_values(amp_text) or [default_amp_val]
            
            params_dict['amps'] = amps_list
            params_dict['span_hz'] = float(self.span_khz_edit.text()) * 1e3 # Convert kHz to Hz
            params_dict['npoints_per_sweep'] = int(self.npoints_edit.text())
            params_dict['nsamps'] = int(self.nsamps_edit.text())
            params_dict['perform_fits'] = self.perform_fits_cb.isChecked()
            params_dict['recalculate_center_frequencies'] = self.recalculate_cf_cb.isChecked()
            
            # Include the essential context for the multisweep
            params_dict['resonance_frequencies'] = self.resonance_frequencies
            params_dict['module'] = self.current_module

            # Basic validation
            if params_dict['span_hz'] <= 0:
                QtWidgets.QMessageBox.warning(self, "Validation Error", "Span must be positive.")
                return None
            if params_dict['npoints_per_sweep'] < 2:
                QtWidgets.QMessageBox.warning(self, "Validation Error", "Number of points per sweep must be at least 2.")
                return None
            if params_dict['nsamps'] < 1:
                QtWidgets.QMessageBox.warning(self, "Validation Error", "Samples to average must be at least 1.")
                return None
            if not self.resonance_frequencies:
                 QtWidgets.QMessageBox.warning(self, "Configuration Error", "No target resonances specified for multisweep.")
                 return None


            return params_dict
        except ValueError as e: # Handles errors from float() or int() conversion
            QtWidgets.QMessageBox.critical(self, "Input Error", f"Invalid numerical input: {str(e)}")
            return None
        except Exception as e: # Catch any other unexpected errors
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not parse parameters: {str(e)}")
            return None
