"""Dialog for multisweep settings."""

from .utils import (
    QtWidgets, QtCore, QtGui, QRegularExpression, QRegularExpressionValidator,
    QDoubleValidator, QIntValidator,
    DEFAULT_AMPLITUDE, DEFAULT_MIN_FREQ, DEFAULT_MAX_FREQ, DEFAULT_CABLE_LENGTH,
    DEFAULT_NPOINTS, DEFAULT_NSAMPLES, DEFAULT_MAX_CHANNELS, DEFAULT_MAX_SPAN,
    MULTISWEEP_DEFAULT_AMPLITUDE, MULTISWEEP_DEFAULT_SPAN_HZ, MULTISWEEP_DEFAULT_NPOINTS, 
    MULTISWEEP_DEFAULT_NSAMPLES, DEFAULT_AMP_START, DEFAULT_AMP_STOP, DEFAULT_AMP_ITERATIONS,
    UnitConverter, traceback
)
from .network_analysis_base import NetworkAnalysisDialogBase
from .tasks import DACScaleFetcher # Import DACScaleFetcher from tasks.py
from . import settings  # Import settings module for persistence
import pickle
import numpy as np
from PyQt6.QtCore import Qt

def load_multisweep_payload(parent: QtWidgets.QWidget, file_path: str | None = None):
    """
    Loads a multisweep payload from a pickle file.

    If file_path is None, it prompts for a file using a blocking dialog (fallback).
    Otherwise, it loads directly from file_path.
    """
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
        and isinstance(payload.get("initial_parameters"), dict)
        and isinstance(payload.get("results_by_iteration"), dict)
    ):
        return payload

    QtWidgets.QMessageBox.warning(
        parent,
        "Invalid File",
        "The selected file does not contain Multisweep parameters.",
    )
    return None


class MultisweepDialog(NetworkAnalysisDialogBase):
    """
    Dialog for configuring a Multisweep operation.
    Inherits from NetworkAnalysisDialogBase for amplitude settings and DAC scale handling.
    Allows specifying parameters for sweeping multiple pre-identified resonances.
    """
    def __init__(self, parent: QtWidgets.QWidget = None, 
                 section_center_frequencies: list[float] | None = None, 
                 dac_scales: dict[int, float] = None, 
                 current_module: int | None = None, 
                 initial_params: dict | None = None, load_multisweep = False, fit_frequencies: list[float] = None):
        """
        Initializes the Multisweep configuration dialog.

        Args:
            parent: The parent widget.
            section_center_frequencies: List of center frequencies (in Hz) for each sweep section.
            dac_scales: Pre-fetched DAC scales.
            current_module: The module ID on which the multisweep will be performed.
            initial_params: Dictionary of initial parameters to populate fields.
        """
        # Load saved defaults if no initial params provided
        merged_params = {}
        if initial_params is None or not initial_params:
            # No initial params, load from saved defaults
            merged_params = settings.get_multisweep_defaults()
        else:
            # Start with saved defaults, then override with initial_params
            merged_params = settings.get_multisweep_defaults()
            merged_params.update(initial_params)
        
        super().__init__(parent, params=merged_params, dac_scales=dac_scales)
        self.section_center_frequencies = section_center_frequencies or []
        self.current_module = current_module # Store the current module for DAC scale and params
        self.load_multisweep = load_multisweep
        self.fit_frequencies = fit_frequencies

        self.use_data_from_file = False
        self._load_data = {}
        self.section_count = 0

        self.setWindowTitle("Multisweep Configuration")
        self.setModal(True)
        self.use_raw_frequencies = True

        
        # if self.load_multisweep:
        #     self._setup_load_ui()
        # else:
        self._setup_ui()
            
        # Asynchronously fetch DAC scales if not provided and CRS is available
        # Note: We fetch scales but don't update UI since new dialog doesn't have dBm conversion widgets
        if parent and hasattr(parent, 'parent') and parent.parent() is not None:
            main_periscope_window = parent.parent()
            if hasattr(main_periscope_window, 'crs') and main_periscope_window.crs is not None:
                # Only fetch if dac_scales weren't passed in and we have a method to do so
                if not self.dac_scales and hasattr(self, '_fetch_dac_scales_for_dialog'):
                    self._fetch_dac_scales_for_dialog(main_periscope_window.crs)



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
        # TODO: Reinstate UI update functionality once amplitude conversion widgets are added
        # self._update_dac_scale_info()
        # self._update_dbm_from_normalized()

    def _get_selected_modules(self) -> list[int]:
        """
        Returns the module relevant for this multisweep dialog.
        For multisweep, it's typically a single, pre-determined module.

        Returns:
            A list containing the current_module ID if set, otherwise an empty list.
        """
        return [self.current_module] if self.current_module is not None else []


    def _update_section_count(self, text):
        """Update label with section count based on QLineEdit content."""
        text = text.strip()
        if not text:
            self.sections_info_label.setText("No data. Enter manually if desired.")
            self.start_btn.setEnabled(False)
            self.load_btn.setEnabled(False)
            return
        self.start_btn.setEnabled(True)
        # Split on commas, ignore empty pieces
        parts = [p.strip() for p in text.split(",") if p.strip()]
        self.section_count = len(parts)
        self.sections_info_label.setText(f"Loaded {self.section_count} section(s).")
    
    def _setup_ui(self):
        """Sets up the user interface elements for the Multisweep dialog."""
        layout = QtWidgets.QVBoxLayout(self)

        # Display information about target resonances
        if self.load_multisweep:
            self.import_button = QtWidgets.QPushButton("Import Sweep File")
            self.import_button.clicked.connect(self._import_file)
            layout.addWidget(self.import_button)
    
            # --- Sweep Sections ---
            section_info_group = QtWidgets.QGroupBox("Sweep sections")
            section_info_layout = QtWidgets.QVBoxLayout(section_info_group)
            
            section_label_layout = QtWidgets.QHBoxLayout()
            self.sections_info_label = QtWidgets.QLabel("No file loaded. Enter manually if desired.")
            self.sections_info_label.setWordWrap(True)
            section_label_layout.addWidget(self.sections_info_label, stretch=1)
            
            self.section_freq_combo = QtWidgets.QComboBox()
            self.section_freq_combo.setEnabled(False)
            self.section_freq_combo.addItems(["Use multisweep central frequency", "Use resonance fit frequency"])
            self.section_freq_combo.setToolTip("Select which frequency type to use.")
            self.section_freq_combo.currentIndexChanged.connect(self._scroll_section)
            section_label_layout.addWidget(self.section_freq_combo)
            
            section_info_layout.addLayout(section_label_layout)
    
            # Manual input fallback (comma-separated sweep central frequencies in MHz)
            self.sections_edit = QtWidgets.QLineEdit()
            self.sections_edit.setPlaceholderText("Enter sweep central frequencies (MHz, comma separated)")
            self.sections_edit.textChanged.connect(self._update_section_count)
            section_info_layout.addWidget(self.sections_edit)
            layout.addWidget(section_info_group)

        elif self.fit_frequencies is not None:
            section_info_group = QtWidgets.QGroupBox("Sweep sections")
            section_info_layout = QtWidgets.QVBoxLayout(section_info_group)
            
            section_label_layout = QtWidgets.QHBoxLayout()
            self.sections_info_label = QtWidgets.QLabel("Central frequencies for re-run")
            self.sections_info_label.setWordWrap(True)
            section_label_layout.addWidget(self.sections_info_label, stretch=1)
            
            self.section_freq_combo = QtWidgets.QComboBox()
            self.section_freq_combo.addItems(["Use previous multisweep central frequencies", "Use resonant frequencies from fit"])
            self.section_freq_combo.setToolTip("Select what to use as the central frequency of this sweep")
            self.section_freq_combo.currentIndexChanged.connect(self._scroll_rerun_section)
            self.section_freq_combo.setCurrentIndex(0)
            section_label_layout.addWidget(self.section_freq_combo)
            
            section_info_layout.addLayout(section_label_layout)
    
            # Default input fallback (comma-separated sweep central frequencies in MHz)
            self.sections_edit = QtWidgets.QLineEdit()
            section_freq_rerun = ", ".join([f"{f / 1e6:.9f}" for f in self.section_center_frequencies])
            self.sections_edit.setText(section_freq_rerun)
            self.sections_edit.textChanged.connect(self._update_section_count)
            section_info_layout.addWidget(self.sections_edit)
            layout.addWidget(section_info_group)
            
        else:
            section_info_group = QtWidgets.QGroupBox("Sweep sections")
            section_info_layout = QtWidgets.QVBoxLayout(section_info_group)
            num_sections = len(self.section_center_frequencies)
            section_label_text = f"Number of sections to sweep: {num_sections}"
            if num_sections > 0:
                # Show first few sweep central frequencies for quick reference
                section_freq_mhz_str = ", ".join([f"{f / 1e6:.3f}" for f in self.section_center_frequencies[:5]])
                if num_sections > 5:
                    section_freq_mhz_str += ", ..."  # Indicate more frequencies exist
                section_label_text += f"\nFrequencies (MHz): {section_freq_mhz_str}"
    
            
            self.sections_info_label = QtWidgets.QLabel(section_label_text)
            self.sections_info_label.setWordWrap(True)
            section_info_layout.addWidget(self.sections_info_label)
            layout.addWidget(section_info_group)

        # Sweep parameters group
        param_group = QtWidgets.QGroupBox("Sweep Parameters")
        param_form_layout = QtWidgets.QFormLayout(param_group)

        # Span per section (kHz)
        default_span_khz = self.params.get('span_hz', MULTISWEEP_DEFAULT_SPAN_HZ) / 1e3
        self.span_khz_edit = QtWidgets.QLineEdit(str(default_span_khz))
        self.span_khz_edit.setValidator(QDoubleValidator(0.1, 10000.0, 2, self)) # Min 0.1 kHz, Max 10 MHz
        param_form_layout.addRow("Span per section (kHz):", self.span_khz_edit)

        # Number of points per sweep
        default_npoints = self.params.get('npoints_per_sweep', MULTISWEEP_DEFAULT_NPOINTS)
        self.npoints_edit = QtWidgets.QLineEdit(str(default_npoints))
        self.npoints_edit.setValidator(QIntValidator(2, 10000, self)) # Min 2 points
        param_form_layout.addRow("Number of points per sweep:", self.npoints_edit)

        # Samples to average (nsamps)
        default_nsamps = self.params.get('nsamps', MULTISWEEP_DEFAULT_NSAMPLES)
        self.nsamps_edit = QtWidgets.QLineEdit(str(default_nsamps))
        self.nsamps_edit.setValidator(QIntValidator(1, 10000, self)) # Min 1 sample
        param_form_layout.addRow("Samples to average per point (nsamps):", self.nsamps_edit)

        # --- NEW AMPLITUDE UI ---
        # Box 1: Base Amplitude Parameters
        base_amp_group = QtWidgets.QGroupBox("Base Amplitude Parameters")
        base_amp_layout = QtWidgets.QFormLayout(base_amp_group)
        
        self.global_amp_edit = QtWidgets.QLineEdit()
        self.global_amp_edit.setPlaceholderText("e.g., 0.1")
        self.global_amp_edit.setValidator(QDoubleValidator(0.0001, 10.0, 4, self))
        self.global_amp_edit.setToolTip(
            "Single amplitude value applied to all frequency sections.\n"
            "Provide EITHER this OR the amplitude array below (not both)."
        )
        base_amp_layout.addRow("Global amplitude:", self.global_amp_edit)
        
        self.amp_array_edit = QtWidgets.QLineEdit()
        self.amp_array_edit.setPlaceholderText("e.g., 0.1, 0.15, 0.12 (comma-separated)")
        self.amp_array_edit.setToolTip(
            "Comma-separated amplitude values, one per frequency section.\n"
            "Must match the number of sweep sections.\n"
            "Provide EITHER this OR the global amplitude above (not both)."
        )
        base_amp_layout.addRow("Amplitude array:", self.amp_array_edit)
        
        base_amp_note = QtWidgets.QLabel(
            "Note: You must provide exactly ONE of the above fields (not both, not neither)."
        )
        base_amp_note.setWordWrap(True)
        base_amp_note.setStyleSheet("font-style: italic; color: #666;")
        base_amp_layout.addRow(base_amp_note)
        
        param_form_layout.addRow(base_amp_group)
        
        # Box 2: Amplitude Iteration Options
        iteration_group = QtWidgets.QGroupBox("Amplitude Iteration Options")
        iteration_layout = QtWidgets.QVBoxLayout(iteration_group)
        
        # Number of steps field
        steps_layout = QtWidgets.QFormLayout()
        self.num_steps_edit = QtWidgets.QLineEdit("1")
        self.num_steps_edit.setValidator(QIntValidator(1, 100, self))
        self.num_steps_edit.setToolTip("Number of amplitude iterations to perform")
        steps_layout.addRow("Number of steps:", self.num_steps_edit)
        iteration_layout.addLayout(steps_layout)
        
        # Radio buttons for iteration mode
        self.single_iteration_radio = QtWidgets.QRadioButton("Single iteration (no sweep)")
        self.single_iteration_radio.setChecked(True)
        self.single_iteration_radio.setToolTip("Perform one measurement with the base amplitude")
        iteration_layout.addWidget(self.single_iteration_radio)
        
        self.uniform_sweep_radio = QtWidgets.QRadioButton("Uniform amplitude sweep")
        self.uniform_sweep_radio.setToolTip(
            "Sweep amplitude uniformly from start to stop.\n"
            "All sections get the same amplitude at each iteration."
        )
        iteration_layout.addWidget(self.uniform_sweep_radio)
        
        # Uniform sweep controls
        self.uniform_controls = QtWidgets.QWidget()
        uniform_layout = QtWidgets.QFormLayout(self.uniform_controls)
        uniform_layout.setContentsMargins(20, 0, 0, 0)  # Indent
        
        self.uniform_start_edit = QtWidgets.QLineEdit()
        self.uniform_start_edit.setPlaceholderText("e.g., 0.05")
        self.uniform_start_edit.setValidator(QDoubleValidator(0.0001, 10.0, 4, self))
        uniform_layout.addRow("Start amplitude:", self.uniform_start_edit)
        
        self.uniform_stop_edit = QtWidgets.QLineEdit()
        self.uniform_stop_edit.setPlaceholderText("e.g., 0.2")
        self.uniform_stop_edit.setValidator(QDoubleValidator(0.0001, 10.0, 4, self))
        uniform_layout.addRow("Stop amplitude:", self.uniform_stop_edit)
        
        self.uniform_controls.setVisible(False)
        iteration_layout.addWidget(self.uniform_controls)
        
        self.scaling_radio = QtWidgets.QRadioButton("Multiplicative scaling")
        self.scaling_radio.setToolTip(
            "Scale the base amplitude array by factors from start to stop.\n"
            "Preserves relative structure of amplitude array."
        )
        iteration_layout.addWidget(self.scaling_radio)
        
        # Scaling controls
        self.scaling_controls = QtWidgets.QWidget()
        scaling_layout = QtWidgets.QFormLayout(self.scaling_controls)
        scaling_layout.setContentsMargins(20, 0, 0, 0)  # Indent
        
        self.scale_start_edit = QtWidgets.QLineEdit()
        self.scale_start_edit.setPlaceholderText("e.g., 0.5")
        self.scale_start_edit.setValidator(QDoubleValidator(0.001, 100.0, 3, self))
        scaling_layout.addRow("Start factor:", self.scale_start_edit)
        
        self.scale_stop_edit = QtWidgets.QLineEdit()
        self.scale_stop_edit.setPlaceholderText("e.g., 2.0")
        self.scale_stop_edit.setValidator(QDoubleValidator(0.001, 100.0, 3, self))
        scaling_layout.addRow("Stop factor:", self.scale_stop_edit)
        
        self.scaling_controls.setVisible(False)
        iteration_layout.addWidget(self.scaling_controls)
        
        param_form_layout.addRow(iteration_group)
        
        # Connect radio buttons to show/hide controls
        self.single_iteration_radio.toggled.connect(self._on_iteration_mode_changed)
        self.uniform_sweep_radio.toggled.connect(self._on_iteration_mode_changed)
        self.scaling_radio.toggled.connect(self._on_iteration_mode_changed)
        
        # Add Clear button for amplitude settings
        clear_amp_layout = QtWidgets.QHBoxLayout()
        clear_amp_layout.addStretch()  # Push button to the right
        self.clear_amp_btn = QtWidgets.QPushButton("Clear Amplitude Settings")
        self.clear_amp_btn.setToolTip("Reset all amplitude fields to their default empty state")
        self.clear_amp_btn.clicked.connect(self._clear_amplitude_fields)
        clear_amp_layout.addWidget(self.clear_amp_btn)
        param_form_layout.addRow(clear_amp_layout)
        
        # Option to recalculate center frequencies
        self.recalc_cf_combo = QtWidgets.QComboBox()
        self.recalc_cf_combo.addItems(["max-dIQ","min-S21","None"])
        
        # Set initial value for bias_frequency_method
        default_recalc_setting = self.params.get('bias_frequency_method', "max-diq")

        if default_recalc_setting == "min-s21":
            self.recalc_cf_combo.setCurrentText("min-S21")
        elif default_recalc_setting == "max-diq":
            self.recalc_cf_combo.setCurrentText("max-dIQ")
        else: # Covers None or any other unexpected string
            self.recalc_cf_combo.setCurrentText("None")
            
        self.recalc_cf_combo.setToolTip(
            "Determines how/if center frequencies are recalculated for biasing:\n"
            "- max-dIQ [Often Optimal]: Recalculates to max IQ velocity |d(I+jQ)/df|. Finds where the IQ trajectory moves fastest.\n"
            "- min-S21: Recalculates to min |S21|.\n"
            "- None: No recalculation. Use original center frequency.\n"
        )
        param_form_layout.addRow("Bias Frequency Method:", self.recalc_cf_combo)
        
        # Option to rotate saved data
        self.rotate_saved_data_checkbox = QtWidgets.QCheckBox("Rotate Saved Data")
        self.rotate_saved_data_checkbox.setChecked(self.params.get('rotate_saved_data', False)) # Default to False for df calibration
        self.rotate_saved_data_checkbox.setToolTip(
            "Whether to rotate sweep data based on TOD analysis.\n"
            "When checked and bias frequency method is not None:\n"
            "- For min-S21: Rotates to minimize the I component of the TOD's mean.\n"
            "- For max-dIQ: Rotates to align the principal component of the TOD with the I-axis.\n"
            "Note: Should be unchecked when using df calibration to ensure consistency."
        )
        param_form_layout.addRow(self.rotate_saved_data_checkbox)
        
        # Sweep direction selection
        self.sweep_direction_combo = QtWidgets.QComboBox()
        self.sweep_direction_combo.addItems(["Upward", "Downward", "Both"])
        
        # Set initial value for sweep_direction
        default_direction = self.params.get('sweep_direction', 'upward')
        if default_direction == "upward":
            self.sweep_direction_combo.setCurrentText("Upward")
        elif default_direction == "downward":
            self.sweep_direction_combo.setCurrentText("Downward")
        elif default_direction == "both":
            self.sweep_direction_combo.setCurrentText("Both")
        else: # Default to Upward for any unexpected value
            self.sweep_direction_combo.setCurrentText("Upward")
            
        self.sweep_direction_combo.setToolTip(
            "Direction of frequency sweep:\n"
            "- Upward: Sweep from lower to higher frequencies.\n"
            "- Downward: Sweep from higher to lower frequencies.\n"
            "- Both: Perform both sweep directions sequentially."
        )
        param_form_layout.addRow("Sweep Direction:", self.sweep_direction_combo)
        
        # Fitting options
        self.apply_skewed_fit_checkbox = QtWidgets.QCheckBox("Apply Skewed Fit")
        self.apply_skewed_fit_checkbox.setChecked(self.params.get('apply_skewed_fit', True)) # Default to True
        param_form_layout.addRow(self.apply_skewed_fit_checkbox)

        self.apply_nonlinear_fit_checkbox = QtWidgets.QCheckBox("Apply Nonlinear Fit")
        self.apply_nonlinear_fit_checkbox.setChecked(self.params.get('apply_nonlinear_fit', True)) # Default to True
        param_form_layout.addRow(self.apply_nonlinear_fit_checkbox)
        
        layout.addWidget(param_group)



        if self.load_multisweep:
            btn_layout = QtWidgets.QHBoxLayout()
            self.start_btn = QtWidgets.QPushButton("Start Multisweep")
            self.start_btn.setDefault(True)  # Make this the default button (highlighted, triggered by Enter)
            self.start_btn.setEnabled(False)
            self.load_btn = QtWidgets.QPushButton("Load Multisweep")
            self.load_btn.setEnabled(False) ### Will enable once file is available.
            self.cancel_btn = QtWidgets.QPushButton("Cancel")
            btn_layout.addWidget(self.start_btn)
            btn_layout.addWidget(self.load_btn)
            btn_layout.addWidget(self.cancel_btn)
            layout.addLayout(btn_layout)
        
            self.start_btn.clicked.connect(self.accept) # Connect to QDialog's accept slot
    
            self.load_btn.clicked.connect(self._load_data_avail) 
            
            self.cancel_btn.clicked.connect(self.reject) # Connect to QDialog's reject slot

        else:
            btn_layout = QtWidgets.QHBoxLayout()
            self.start_btn = QtWidgets.QPushButton("Start Multisweep")
            self.start_btn.setDefault(True)  # Make this the default button (highlighted, triggered by Enter)
            self.load_btn = QtWidgets.QPushButton("Load Multisweep")
            self.load_btn.hide()
            self.cancel_btn = QtWidgets.QPushButton("Cancel")
            btn_layout.addWidget(self.start_btn)
            btn_layout.addWidget(self.cancel_btn)
            layout.addLayout(btn_layout)
        
            self.start_btn.clicked.connect(self.accept) # Connect to QDialog's accept slot            
            self.cancel_btn.clicked.connect(self.reject) # Connect to QDialog's reject slot
            
        # Create keyboard shortcuts for Enter/Return keys to trigger "Start Multisweep"
        # This works regardless of which widget has focus
        self.enter_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Return), self)
        self.enter_shortcut.activated.connect(self.accept)
        
        # Also handle numpad Enter
        self.numpad_enter_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Enter), self)
        self.numpad_enter_shortcut.activated.connect(self.accept)
        
        self.setMinimumWidth(500) # Ensure dialog is wide enough
        
        # Populate amplitude fields from stored parameters (if available)
        self._populate_amplitude_fields_from_params()

    def _populate_amplitude_fields_from_params(self):
        """
        Populate amplitude-related fields from self.params if available.
        Uses metadata (base_amplitude_mode, base_amplitude_values) if present,
        otherwise detects the mode from amp_arrays structure.
        """
        amp_arrays = self.params.get('amp_arrays')
        
        if not amp_arrays or not isinstance(amp_arrays, list):
            # No amplitude data to populate, or wrong format
            return
        
        if len(amp_arrays) == 0:
            return
        
        num_iterations = len(amp_arrays)
        
        # Set number of steps
        self.num_steps_edit.setText(str(num_iterations))
        
        # Get the first amplitude array as reference
        first_array = amp_arrays[0]
        if not first_array:
            return
        
        # Check for metadata about how base amplitude was originally specified
        base_amp_mode = self.params.get('base_amplitude_mode')
        base_amp_values = self.params.get('base_amplitude_values')
        
        # If we have metadata, use it to populate the correct field
        if base_amp_mode and base_amp_values is not None:
            if base_amp_mode == 'global':
                self.global_amp_edit.setText(f"{base_amp_values:.4g}")
            elif base_amp_mode == 'array':
                if isinstance(base_amp_values, list):
                    amp_text = ", ".join(f"{amp:.4g}" for amp in base_amp_values)
                    self.amp_array_edit.setText(amp_text)
        else:
            # No metadata - fall back to inference from amp_arrays structure
            # Check if this is a global amplitude (all values in array are the same)
            is_global_amp = len(set(first_array)) == 1
            
            if is_global_amp:
                self.global_amp_edit.setText(f"{first_array[0]:.4g}")
            else:
                amp_text = ", ".join(f"{amp:.4g}" for amp in first_array)
                self.amp_array_edit.setText(amp_text)
        
        # Now handle iteration settings
        if num_iterations == 1:
            # Single iteration mode
            self.single_iteration_radio.setChecked(True)
        
        elif num_iterations > 1:
            # Multiple iterations - detect pattern
            # Extract the representative amplitude from each iteration (first value)
            iteration_amps = [arr[0] if arr else 0.0 for arr in amp_arrays]
            
            # Check if all arrays have the same structure (global vs array)
            all_global = all(len(set(arr)) == 1 for arr in amp_arrays if arr)
            all_array_same_structure = len(set(len(arr) for arr in amp_arrays)) == 1
            
            if all_global:
                # All iterations use global amplitude - check for uniform or scaling
                # Check for uniform progression (linear)
                if self._is_uniform_progression(iteration_amps):
                    self.uniform_sweep_radio.setChecked(True)
                    self.uniform_start_edit.setText(f"{iteration_amps[0]:.4g}")
                    self.uniform_stop_edit.setText(f"{iteration_amps[-1]:.4g}")
                
                # Check for scaling progression (multiplicative)
                elif self._is_scaling_progression(amp_arrays, first_array):
                    self.scaling_radio.setChecked(True)
                    # Calculate scale factors
                    base_amp = first_array[0]
                    start_factor = iteration_amps[0] / base_amp if base_amp != 0 else 1.0
                    stop_factor = iteration_amps[-1] / base_amp if base_amp != 0 else 1.0
                    self.scale_start_edit.setText(f"{start_factor:.4g}")
                    self.scale_stop_edit.setText(f"{stop_factor:.4g}")
                
                else:
                    # Unknown pattern, just keep as single iteration
                    self.single_iteration_radio.setChecked(True)
            
            elif all_array_same_structure:
                # Amplitude arrays with consistent structure
                # Check if this follows a scaling pattern
                if self._is_scaling_progression(amp_arrays, first_array):
                    self.scaling_radio.setChecked(True)
                    # Calculate scale factors from first element
                    base_amp = first_array[0]
                    start_factor = iteration_amps[0] / base_amp if base_amp != 0 else 1.0
                    stop_factor = iteration_amps[-1] / base_amp if base_amp != 0 else 1.0
                    self.scale_start_edit.setText(f"{start_factor:.4g}")
                    self.scale_stop_edit.setText(f"{stop_factor:.4g}")
                else:
                    # No clear pattern, keep as single iteration
                    self.single_iteration_radio.setChecked(True)
    
    def _is_uniform_progression(self, values):
        """Check if values follow a uniform (linear) progression."""
        if len(values) < 2:
            return False
        
        # Calculate differences
        diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
        
        # Check if all differences are approximately equal (within 1% tolerance)
        if len(diffs) == 0:
            return False
        
        avg_diff = sum(diffs) / len(diffs)
        if avg_diff == 0:
            return all(d == 0 for d in diffs)
        
        tolerance = 0.01 * abs(avg_diff)
        return all(abs(d - avg_diff) < tolerance for d in diffs)
    
    def _is_scaling_progression(self, amp_arrays, base_array):
        """Check if amp_arrays follow a multiplicative scaling pattern relative to base_array."""
        if len(amp_arrays) < 2 or not base_array or base_array[0] == 0:
            return False
        
        # Calculate scale factors for each iteration
        scale_factors = []
        for arr in amp_arrays:
            if not arr or len(arr) != len(base_array):
                return False
            # Use first element to calculate scale factor
            factor = arr[0] / base_array[0] if base_array[0] != 0 else 1.0
            scale_factors.append(factor)
        
        # Check if scale factors are uniformly distributed
        if self._is_uniform_progression(scale_factors):
            # Also verify that all elements in each array are scaled by the same factor
            for i, arr in enumerate(amp_arrays):
                expected_factor = scale_factors[i]
                for j, amp in enumerate(arr):
                    expected_val = base_array[j] * expected_factor
                    if abs(amp - expected_val) > 0.01 * abs(expected_val):
                        return False
            return True
        
        return False
    
    def _clear_amplitude_fields(self):
        """Clear all amplitude-related fields to their default empty/initial state."""
        # Clear base amplitude fields
        self.global_amp_edit.clear()
        self.amp_array_edit.clear()
        
        # Reset number of steps to 1
        self.num_steps_edit.setText("1")
        
        # Clear uniform sweep fields
        self.uniform_start_edit.clear()
        self.uniform_stop_edit.clear()
        
        # Clear scaling fields
        self.scale_start_edit.clear()
        self.scale_stop_edit.clear()
        
        # Reset to single iteration mode
        self.single_iteration_radio.setChecked(True)

    def _load_data_avail(self):
        """Mark that data should be loaded from file and accept the dialog."""
        self.use_data_from_file = True
        self.accept()

    def _scroll_section(self):
        """Handle section selection changes and update displayed frequencies accordingly. Choice between fit or sweep frequencies"""
        selected = self.section_freq_combo.currentText().lower()
        self.sections_edit.clear()

        if "fit" in selected:
            self.load_btn.setEnabled(False)
            self.use_raw_frequencies = False
            freqs = self._get_frequencies(self._load_data, self.use_raw_frequencies)
        else:
            self.use_raw_frequencies = True
            self.load_btn.setEnabled(True)
            freqs = self._get_frequencies(self._load_data, self.use_raw_frequencies)

        self.sections_edit.setText(",".join([f"{f/1e6:.9f}" for f in freqs]))
        self.sections_info_label.setText(f"Loaded {len(freqs)} sections from file.")

    def _scroll_rerun_section(self):
        """Refresh section display when rerunning choice between fit or sweep frequencies."""
        selected = self.section_freq_combo.currentText().lower()
        self.sections_edit.clear()

        if "fit" in selected:
            freqs = self.fit_frequencies
        else:
            freqs = self.section_center_frequencies

        self.sections_edit.setText(",".join([f"{f/1e6:.9f}" for f in freqs]))
        self.sections_info_label.setText(f"Loaded {len(freqs)} sections from file.")
        
    
    def _import_file(self):
        """
        Trigger non-blocking async file dialog instead of blocking getOpenFileName.
        """
        QtCore.QTimer.singleShot(0, self._open_file_dialog_async)

    def _open_file_dialog_async(self):
        """Open a non-blocking file dialog to select a multisweep parameter file."""
        if not hasattr(self, "_file_dialog") or self._file_dialog is None:
            self._file_dialog = QtWidgets.QFileDialog(self, "Load Multisweep Parameters")
            self._file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
            self._file_dialog.setNameFilters([
                "Pickle Files (*.pkl *.pickle)",
                "All Files (*)",
            ])
            self._file_dialog.setOptions(
                QtWidgets.QFileDialog.Option.DontUseNativeDialog
                | QtWidgets.QFileDialog.Option.ReadOnly
            )
            self._file_dialog.setModal(False)
            self._file_dialog.fileSelected.connect(self._on_file_selected)
            self._file_dialog.rejected.connect(self._on_file_dialog_closed)

        self._file_dialog.open()


    @QtCore.pyqtSlot(str)
    def _on_file_selected(self, path: str):
        """Load selected multisweep data file, extract parameters, and populate the UI fields."""
        payload = load_multisweep_payload(self, file_path=path)
        if payload is None:
            return
    
        self.load_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self._load_data = payload.copy()
        self.section_freq_combo.setEnabled(True)
        
        try:
            params = payload['initial_parameters']
    
            freqs = self._get_frequencies(payload, self.use_raw_frequencies)
            
            self.sections_edit.setText(",".join([f"{f/1e6:.9f}" for f in freqs]))
            self.sections_info_label.setText(f"Loaded {len(freqs)} sections from file.")
        
            # Span per section (Hz -> kHz)
            span_khz = params['span_hz'] / 1e3
            self.span_khz_edit.setText(str(span_khz))
        
            self.npoints_edit.setText(str(params['npoints_per_sweep']))
            self.nsamps_edit.setText(str(params['nsamps']))
        
            idx = self.recalc_cf_combo.findText(params['bias_frequency_method'], Qt.MatchFlag.MatchFixedString)
            if idx >= 0:
                self.recalc_cf_combo.setCurrentIndex(idx)
        
            self.rotate_saved_data_checkbox.setChecked(params['rotate_saved_data'])
        
            idx = self.sweep_direction_combo.findText(params['sweep_direction'].capitalize(),
                                                      Qt.MatchFlag.MatchFixedString)
            if idx >= 0:
                self.sweep_direction_combo.setCurrentIndex(idx)
        
            self.apply_skewed_fit_checkbox.setChecked(params['apply_skewed_fit'])
            self.apply_nonlinear_fit_checkbox.setChecked(params['apply_nonlinear_fit'])
        
            # Load iteration amplitudes (for amplitude iterations)
            amps = params.get("amps") or ([params["amp"]] if "amp" in params else None)
            if amps:
                try:
                    amp_text = ", ".join(f"{float(amp):g}" for amp in amps)
                except (TypeError, ValueError):
                    amp_text = ", ".join(str(amp) for amp in amps)
                self.amp_edit.setText(amp_text)
            
            # Load per-section amplitudes if they exist
            section_amps = params.get("section_amplitudes")
            if section_amps:
                try:
                    section_amp_text = ", ".join(f"{float(amp):g}" for amp in section_amps)
                    self.section_amps_edit.setText(section_amp_text)
                except (TypeError, ValueError):
                    # If section_amplitudes exist but can't be parsed, try to extract from first iteration
                    pass
            elif payload.get('results_by_iteration'):
                # Extract per-section amplitudes from first iteration's data
                try:
                    first_iter = payload['results_by_iteration'][0]
                    section_amps_from_results = []
                    for idx in sorted(first_iter['data'].keys()):
                        if isinstance(idx, (int, np.integer)):
                            amp_val = first_iter['data'][idx].get('sweep_amplitude')
                            if amp_val is not None:
                                section_amps_from_results.append(amp_val)
                    if section_amps_from_results:
                        section_amp_text = ", ".join(f"{float(amp):g}" for amp in section_amps_from_results)
                        self.section_amps_edit.setText(section_amp_text)
                except Exception as e:
                    print(f"Warning: Could not extract per-section amplitudes from results: {e}")
        except KeyError as e:
            missing = e.args[0]
            msg = (
                f"Key '{missing}' is missing in the payload.\n"
                "Default value will be used where possible."
            )
            QtWidgets.QMessageBox.warning(self, "Missing Key", msg)
    
    
    def _get_frequencies(self, payload, raw_section_centers = True):
        """Extract section center frequencies from payload, optionally using fitted or sweep data."""
        
        params = payload['initial_parameters']
        freqs = params['resonance_frequencies']  # Legacy key name for backward compatibility
        
        if raw_section_centers:
            return freqs

        else:
            ref_freqs = []
    
            if params['apply_skewed_fit']:
                for i in range(len(freqs)):
                    ref_freqs.append(payload['results_by_iteration'][0]['data'][i+1]['fit_params']['fr'])
                ref_freqs.sort()
                return ref_freqs
    
            if params['apply_nonlinear_fit']:
                for i in range(len(freqs)):
                    ref_freqs.append(payload['results_by_iteration'][0]['data'][i+1]['nonlinear_fit_params']['fr'])
                ref_freqs.sort()
                return ref_freqs
    
            for i in range(len(freqs)):
                ref_freqs.append(payload['results_by_iteration'][0]['data'][i+1]['bias_frequency'])
            ref_freqs.sort()
                
            return ref_freqs
        
    
    def _on_iteration_mode_changed(self):
        """Handle changes to iteration mode radio buttons."""
        self.uniform_controls.setVisible(self.uniform_sweep_radio.isChecked())
        self.scaling_controls.setVisible(self.scaling_radio.isChecked())
    
    @QtCore.pyqtSlot()
    def _on_file_dialog_closed(self):
        """Handle closure of the file dialog without file selection."""
        pass  # Optional: keep or clear dialog

    
    def get_parameters(self) -> dict | None:
        """
        Retrieves and validates the parameters for the multisweep operation.

        Returns:
            A dictionary of parameters if valid, otherwise None.
            Shows an error message on invalid input or validation failure.
        """
        params_dict = {}
        try:
            if self.use_data_from_file:
                return self._load_data
            
            # Parse basic sweep parameters
            params_dict['span_hz'] = float(self.span_khz_edit.text()) * 1e3  # Convert kHz to Hz
            params_dict['npoints_per_sweep'] = int(self.npoints_edit.text())
            params_dict['nsamps'] = int(self.nsamps_edit.text())
            
            # Get center frequencies
            if self.load_multisweep:
                params_dict['resonance_frequencies'] = []
                freqs = self.sections_edit.text().split(',')
                for f in freqs:
                    params_dict['resonance_frequencies'].append(np.float64(f) * 1e6)
            else:
                params_dict['resonance_frequencies'] = self.section_center_frequencies
            
            num_sections = len(params_dict['resonance_frequencies'])
            
            # STEP 1: Parse base amplitude
            global_amp_text = self.global_amp_edit.text().strip()
            amp_array_text = self.amp_array_edit.text().strip()
            
            if global_amp_text and amp_array_text:
                QtWidgets.QMessageBox.warning(
                    self, 
                    "Validation Error", 
                    "Provide either global amplitude OR amplitude array, not both."
                )
                return None
            elif not global_amp_text and not amp_array_text:
                QtWidgets.QMessageBox.warning(
                    self, 
                    "Validation Error", 
                    "Must provide either global amplitude or amplitude array."
                )
                return None
            
            if global_amp_text:
                base_amp = float(global_amp_text)
                if base_amp <= 0:
                    QtWidgets.QMessageBox.warning(
                        self, 
                        "Validation Error", 
                        "Global amplitude must be positive."
                    )
                    return None
                base_amp_array = [base_amp] * num_sections
                # Store metadata about how base amplitude was specified
                params_dict['base_amplitude_mode'] = 'global'
                params_dict['base_amplitude_values'] = base_amp
            else:
                # Parse amplitude array
                base_amp_array = [float(x.strip()) for x in amp_array_text.split(',')]
                if len(base_amp_array) != num_sections:
                    QtWidgets.QMessageBox.warning(
                        self, 
                        "Validation Error", 
                        f"Amplitude array length ({len(base_amp_array)}) must match "
                        f"number of sweep sections ({num_sections})."
                    )
                    return None
                # Validate all amplitudes are positive
                for i, amp in enumerate(base_amp_array):
                    if amp <= 0:
                        QtWidgets.QMessageBox.warning(
                            self, 
                            "Validation Error", 
                            f"All amplitudes must be positive (section {i+1} has {amp})."
                        )
                        return None
                # Store metadata about how base amplitude was specified
                params_dict['base_amplitude_mode'] = 'array'
                params_dict['base_amplitude_values'] = base_amp_array.copy()
            
            # STEP 2: Handle iterations
            num_steps = int(self.num_steps_edit.text())
            if num_steps < 1:
                QtWidgets.QMessageBox.warning(
                    self, 
                    "Validation Error", 
                    "Number of steps must be at least 1."
                )
                return None
            
            if self.single_iteration_radio.isChecked() or num_steps == 1:
                # Single iteration mode
                params_dict['amp_arrays'] = [base_amp_array]
            
            elif self.uniform_sweep_radio.isChecked():
                # Uniform sweep mode
                start_amp_text = self.uniform_start_edit.text().strip()
                stop_amp_text = self.uniform_stop_edit.text().strip()
                
                if not start_amp_text or not stop_amp_text:
                    QtWidgets.QMessageBox.warning(
                        self, 
                        "Validation Error", 
                        "Must provide both start and stop amplitudes for uniform sweep."
                    )
                    return None
                
                start_amp = float(start_amp_text)
                stop_amp = float(stop_amp_text)
                
                if start_amp <= 0 or stop_amp <= 0:
                    QtWidgets.QMessageBox.warning(
                        self, 
                        "Validation Error", 
                        "Start and stop amplitudes must be positive."
                    )
                    return None
                
                # Generate uniform amplitude values
                amp_values = np.linspace(start_amp, stop_amp, num_steps)
                params_dict['amp_arrays'] = [[amp] * num_sections for amp in amp_values]
            
            elif self.scaling_radio.isChecked():
                # Multiplicative scaling mode
                start_factor_text = self.scale_start_edit.text().strip()
                stop_factor_text = self.scale_stop_edit.text().strip()
                
                if not start_factor_text or not stop_factor_text:
                    QtWidgets.QMessageBox.warning(
                        self, 
                        "Validation Error", 
                        "Must provide both start and stop factors for multiplicative scaling."
                    )
                    return None
                
                start_factor = float(start_factor_text)
                stop_factor = float(stop_factor_text)
                
                if start_factor <= 0 or stop_factor <= 0:
                    QtWidgets.QMessageBox.warning(
                        self, 
                        "Validation Error", 
                        "Start and stop factors must be positive."
                    )
                    return None
                
                # Generate scale factors
                factors = np.linspace(start_factor, stop_factor, num_steps)
                params_dict['amp_arrays'] = [
                    [amp * factor for amp in base_amp_array] 
                    for factor in factors
                ]
            
            else:
                QtWidgets.QMessageBox.warning(
                    self, 
                    "Validation Error", 
                    "Must select an iteration mode."
                )
                return None
            
            # Parse other parameters
            recalc_method_text = self.recalc_cf_combo.currentText()
            if recalc_method_text == "None":
                params_dict['bias_frequency_method'] = None
            elif recalc_method_text == "min-S21":
                params_dict['bias_frequency_method'] = "min-s21"
            elif recalc_method_text == "max-dIQ":
                params_dict['bias_frequency_method'] = "max-diq"
            else:
                params_dict['bias_frequency_method'] = None
            
            params_dict['rotate_saved_data'] = self.rotate_saved_data_checkbox.isChecked()
            
            sweep_direction_text = self.sweep_direction_combo.currentText()
            if sweep_direction_text == "Upward":
                params_dict['sweep_direction'] = "upward"
            elif sweep_direction_text == "Downward":
                params_dict['sweep_direction'] = "downward"
            elif sweep_direction_text == "Both":
                params_dict['sweep_direction'] = "both"
            else:
                params_dict['sweep_direction'] = "upward"
            
            params_dict['module'] = self.current_module
            params_dict['apply_skewed_fit'] = self.apply_skewed_fit_checkbox.isChecked()
            params_dict['apply_nonlinear_fit'] = self.apply_nonlinear_fit_checkbox.isChecked()
            
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
            if num_sections < 1:
                QtWidgets.QMessageBox.warning(self, "Configuration Error", "No target sweep sections specified for multisweep.")
                return None
            
            # Save these parameters as defaults for future sessions
            # (only if not loading from file)
            if not self.use_data_from_file:
                try:
                    settings.set_multisweep_defaults(params_dict)
                except Exception as e:
                    # Don't fail the dialog if settings save fails
                    print(f"Warning: Could not save multisweep defaults: {e}")
            
            return params_dict
            
        except ValueError as e:
            QtWidgets.QMessageBox.critical(self, "Input Error", f"Invalid numerical input: {str(e)}")
            return None
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not parse parameters: {str(e)}")
            return None
