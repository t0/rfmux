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
import datetime
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
        and (isinstance(payload.get("results_by_detector"), dict) or isinstance(payload.get("results_by_iteration"), dict))
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
                 initial_params: dict | None = None, load_multisweep = False, fit_frequencies: list[float] = None,
                 bias_frequencies: list[float] | None = None,
                 netanal_mode: bool = False):
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
        self.bias_frequencies = bias_frequencies or []
        self.netanal_mode = netanal_mode

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
        """Update label with section count based on QLineEdit content.

        The Start button enable/disable is delegated entirely to
        ``_validate_amplitude_live`` so that both the section count *and*
        the amplitude configuration are considered together.  Only the Load
        button is managed here (it requires a file to have been loaded and is
        unaffected by the amplitude settings).
        """
        text = text.strip()
        if not text:
            self.sections_info_label.setText("No data. Enter manually if desired.")
            if hasattr(self, 'load_btn'):
                self.load_btn.setEnabled(False)
            self.section_count = 0
        else:
            # Split on commas, ignore empty pieces
            parts = [p.strip() for p in text.split(",") if p.strip()]
            self.section_count = len(parts)
            self.sections_info_label.setText(f"Loaded {self.section_count} section(s).")

        # Re-run live amplitude validation so it can update the Start button
        # state to reflect the new section count (e.g. for array-length checks).
        # Guard against being called before the amplitude UI is fully set up.
        if hasattr(self, '_amp_status_label'):
            self._validate_amplitude_live()
        elif hasattr(self, 'start_btn'):
            # Fallback during early construction: basic enable/disable
            self.start_btn.setEnabled(self.section_count > 0)
    
    def _setup_ui(self):
        """Sets up the user interface elements for the Multisweep dialog."""
        layout = QtWidgets.QVBoxLayout(self)

        # ── Measurement Name ──────────────────────────────────────────────────
        name_group = QtWidgets.QGroupBox("Measurement Name")
        name_form = QtWidgets.QFormLayout(name_group)

        # Box 1: pre-populated base name (timestamp + meas type), editable
        default_base = datetime.datetime.now().strftime("multisweep_%H%M%S")
        self.base_name_edit = QtWidgets.QLineEdit(default_base)
        self.base_name_edit.setToolTip(
            "Base filename — pre-filled with a timestamp and measurement type.\n"
            "You may edit it freely.  The .pkl extension is added automatically."
        )
        name_form.addRow("Base name:", self.base_name_edit)

        # Box 2: optional user suffix — pre-populated from the previous run if available
        self.custom_suffix_edit = QtWidgets.QLineEdit(self.params.get('measurement_custom_suffix', ''))
        self.custom_suffix_edit.setPlaceholderText("e.g. cold_dark, tile3, run2")
        self.custom_suffix_edit.setToolTip(
            "Optional suffix appended after the base name with an underscore separator.\n"
            "Leave blank to use the base name only."
        )
        name_form.addRow("Custom suffix:", self.custom_suffix_edit)

        # Live preview label
        self._name_preview_label = QtWidgets.QLabel()
        self._name_preview_label.setStyleSheet("font-style: italic; color: #555;")
        name_form.addRow("→  filename:", self._name_preview_label)

        # Connect both fields to update the preview
        self.base_name_edit.textChanged.connect(self._update_name_preview)
        self.custom_suffix_edit.textChanged.connect(self._update_name_preview)
        self._update_name_preview()  # populate on open

        layout.addWidget(name_group)

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
            
            # Preserve a copy of the original sweep-center frequencies for the
            # "Use previous multisweep central frequencies" option.  section_center_frequencies
            # may be mutated by _scroll_rerun_section when the user switches options.
            self._original_section_center_frequencies = list(self.section_center_frequencies)

            self.section_freq_combo = QtWidgets.QComboBox()
            self.section_freq_combo.addItem("Use previous multisweep central frequencies")
            # Only add the fit-frequency option when real fit results exist
            if self.fit_frequencies:
                self.section_freq_combo.addItem("Use resonant frequencies from fit")
            # Add "Bias frequencies" option if bias frequencies are available
            if self.bias_frequencies:
                self.section_freq_combo.addItem("Use bias frequencies")
            self.section_freq_combo.setToolTip("Select what to use as the central frequency of this sweep")
            # Block signals during setup so the slot doesn't fire before sections_edit exists
            self.section_freq_combo.blockSignals(True)
            # Default to "Bias frequencies" if they exist, otherwise "previous multisweep central frequencies"
            if self.bias_frequencies:
                self.section_freq_combo.setCurrentIndex(self.section_freq_combo.count() - 1)
                self.section_center_frequencies = list(self.bias_frequencies)
            else:
                self.section_freq_combo.setCurrentIndex(0)
            self.section_freq_combo.blockSignals(False)
            section_label_layout.addWidget(self.section_freq_combo)
            
            section_info_layout.addLayout(section_label_layout)
    
            # Display field showing the selected central frequencies (MHz).
            # 6 decimal places in MHz = 1 Hz precision.
            self.sections_edit = QtWidgets.QLineEdit()
            section_freq_rerun = ", ".join([f"{f / 1e6:.6f}" for f in self.section_center_frequencies])
            self.sections_edit.setText(section_freq_rerun)
            self.sections_edit.textChanged.connect(self._update_section_count)
            section_info_layout.addWidget(self.sections_edit)
            # Connect the signal only after sections_edit exists so the slot is safe to call
            self.section_freq_combo.currentIndexChanged.connect(self._scroll_rerun_section)
            layout.addWidget(section_info_group)
            
        elif self.netanal_mode:
            # ── netanal mode: editable field with optional Find Resonances source ──
            section_info_group = QtWidgets.QGroupBox("Sweep sections")
            section_info_layout = QtWidgets.QVBoxLayout(section_info_group)

            section_label_layout = QtWidgets.QHBoxLayout()
            self.sections_info_label = QtWidgets.QLabel("Central frequencies")
            self.sections_info_label.setWordWrap(True)
            section_label_layout.addWidget(self.sections_info_label, stretch=1)

            self.section_freq_combo = QtWidgets.QComboBox()
            if self.section_center_frequencies:
                self.section_freq_combo.addItem("Approx locations from Find Resonances")
            self.section_freq_combo.addItem("Custom")
            self.section_freq_combo.setToolTip(
                "Select what to use as the central frequency of each sweep section.\n"
                "'Approx locations from Find Resonances' pre-fills with the resonances\n"
                "found by the Find Resonances algorithm (editable).\n"
                "'Custom' lets you type a comma-separated list of frequencies in MHz."
            )
            # Default to first item (Find Resonances if available, else Custom)
            self.section_freq_combo.blockSignals(True)
            self.section_freq_combo.setCurrentIndex(0)
            self.section_freq_combo.blockSignals(False)
            section_label_layout.addWidget(self.section_freq_combo)

            section_info_layout.addLayout(section_label_layout)

            # Editable frequency field.  Pre-populated when Find Resonances has run.
            # setText is called BEFORE connecting textChanged so _update_section_count
            # doesn't fire before start_btn / load_btn are created.
            self.sections_edit = QtWidgets.QLineEdit()
            self.sections_edit.setPlaceholderText("Enter sweep central frequencies (MHz, comma separated)")
            if self.section_center_frequencies:
                section_freq_str = ", ".join([f"{f / 1e6:.6f}" for f in self.section_center_frequencies])
                self.sections_edit.setText(section_freq_str)
            section_info_layout.addWidget(self.sections_edit)

            # Connect signals AFTER setText so they only fire on user edits
            self.sections_edit.textChanged.connect(self._update_section_count)
            self.section_freq_combo.currentIndexChanged.connect(self._scroll_netanal_section)

            layout.addWidget(section_info_group)

        else:
            # ── Legacy / fallback: static label (non-editable) ──────────────────
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

        # Amplitude Iteration Options (single group — base amp fields live inside)
        iteration_group = QtWidgets.QGroupBox("Amplitude Settings")
        iteration_layout = QtWidgets.QVBoxLayout(iteration_group)

        # Radio buttons for iteration mode
        self.single_iteration_radio = QtWidgets.QRadioButton("Single iteration (no sweep)")
        self.single_iteration_radio.setChecked(True)
        self.single_iteration_radio.setToolTip("Perform one measurement at a fixed amplitude")
        iteration_layout.addWidget(self.single_iteration_radio)

        # ── Base amplitude controls (shown only for Single iteration) ────────
        self.single_amp_controls = QtWidgets.QWidget()
        single_amp_layout = QtWidgets.QFormLayout(self.single_amp_controls)
        single_amp_layout.setContentsMargins(20, 0, 0, 0)  # Indent

        self.global_amp_edit = QtWidgets.QLineEdit()
        self.global_amp_edit.setPlaceholderText("e.g., 0.1")
        self.global_amp_edit.setValidator(QDoubleValidator(0.0001, 10.0, 4, self))
        self.global_amp_edit.setToolTip(
            "Single amplitude value applied to all frequency sections.\n"
            "Provide EITHER this OR the amplitude array below (not both)."
        )
        single_amp_layout.addRow("Global amplitude:", self.global_amp_edit)

        self.amp_array_edit = QtWidgets.QLineEdit()
        self.amp_array_edit.setPlaceholderText("e.g., 0.1, 0.15, 0.12 (comma-separated)")
        self.amp_array_edit.setToolTip(
            "Comma-separated amplitude values, one per frequency section.\n"
            "Must match the number of sweep sections.\n"
            "Provide EITHER this OR the global amplitude above (not both)."
        )
        single_amp_layout.addRow("Amplitude array:", self.amp_array_edit)


        self.single_amp_controls.setVisible(True)  # visible by default (single is checked)
        iteration_layout.addWidget(self.single_amp_controls)
        
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
            "Scale each resonator's probe amplitude by factors from start to stop.\n"
            "The per-resonator bias_amplitude stored in the resonator registry\n"
            "(res_info_dict) is always used as the base — no manual amplitude\n"
            "entry is required.  A registry is built automatically after the\n"
            "first single-amplitude sweep."
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

        # Number of steps field — placed after the radio buttons so it sits
        # logically between the mode selector and the mode-specific controls.
        steps_layout = QtWidgets.QFormLayout()
        self.num_steps_edit = QtWidgets.QLineEdit("1")
        self.num_steps_edit.setValidator(QIntValidator(1, 100, self))
        self.num_steps_edit.setToolTip("Number of amplitude iterations to perform.\nFixed at 1 when Single iteration is selected.")
        steps_layout.addRow("Number of steps:", self.num_steps_edit)
        iteration_layout.addLayout(steps_layout)

        # Tracks the last user-entered number of steps so it can be restored
        # when the user switches away from Single iteration back to a sweep mode.
        self._saved_num_steps = "5"

        # ── Amplitude validation status label ─────────────────────────────────
        # Shown at the bottom of the Amplitude Settings group; updated in real-
        # time by _validate_amplitude_live() whenever any relevant field changes.
        self._amp_status_label = QtWidgets.QLabel("")
        self._amp_status_label.setWordWrap(True)
        self._amp_status_label.setMinimumHeight(40)
        iteration_layout.addWidget(self._amp_status_label)

        # Connect all amplitude input fields so the live validator fires on
        # every keystroke (textChanged) regardless of which mode is active.
        for _field in (
            self.global_amp_edit, self.amp_array_edit,
            self.uniform_start_edit, self.uniform_stop_edit,
            self.scale_start_edit, self.scale_stop_edit,
            self.num_steps_edit,
        ):
            _field.textChanged.connect(self._validate_amplitude_live)

        # Apply initial state (single iteration is checked by default)
        self._on_iteration_mode_changed()
        
        # Add Clear button for amplitude settings
        clear_amp_layout = QtWidgets.QHBoxLayout()
        clear_amp_layout.addStretch()  # Push button to the right
        self.clear_amp_btn = QtWidgets.QPushButton("Clear Amplitude Settings")
        self.clear_amp_btn.setToolTip("Reset all amplitude fields to their default empty state")
        self.clear_amp_btn.clicked.connect(self._clear_amplitude_fields)
        clear_amp_layout.addWidget(self.clear_amp_btn)
        param_form_layout.addRow(clear_amp_layout)
        
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

        self.apply_find_bias_checkbox = QtWidgets.QCheckBox("Find bias")
        self.apply_find_bias_checkbox.setChecked(self.params.get('run_find_bias', False))
        self.apply_find_bias_checkbox.setToolTip(
            "Automatically run Find Bias on the collected multisweep data\n"
            "once the sweep is complete (equivalent to pressing the\n"
            "'Find Bias' button on the results panel)."
        )
        param_form_layout.addRow(self.apply_find_bias_checkbox)
        
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
            # In netanal_mode with no pre-populated frequencies (Find Resonances not run),
            # disable the button until the user types in some custom frequencies.
            if self.netanal_mode and not self.section_center_frequencies:
                self.start_btn.setEnabled(False)
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

        # Initial update of dBm field if DAC scales are already known
        # TODO reinstate when this functionality added back in 
        # if self.dac_scales: # Check if dac_scales were passed or fetched synchronously before UI setup
        #     self._update_dac_scale_info() # Ensure info label is also up-to-date
        #     self._update_dbm_from_normalized()

        
        self.setMinimumWidth(500) # Ensure dialog is wide enough
        self.resize(500, 870)     # Ensure dialog is tall enough for name group + sweep controls

        # Place focus in the custom suffix field so the user is encouraged to type
        # their annotation immediately without having to click past the base name.
        self.custom_suffix_edit.setFocus()

        # Populate amplitude fields from stored parameters (if available)
        self._populate_amplitude_fields_from_params()

    def _populate_amplitude_fields_from_params(self):
        """
        Populate amplitude-related fields from self.params if available.
        Uses saved metadata for robust reconstruction.
        """
        # Populate base amplitude fields
        base_amp_mode = self.params.get('base_amplitude_mode')
        base_amp_values = self.params.get('base_amplitude_values')
        
        if base_amp_mode == 'global' and base_amp_values is not None:
            self.global_amp_edit.setText(f"{base_amp_values:.4g}")
        elif base_amp_mode == 'array' and base_amp_values is not None:
            if isinstance(base_amp_values, list):
                amp_text = ", ".join(f"{amp:.4g}" for amp in base_amp_values)
                self.amp_array_edit.setText(amp_text)
        
        # Populate iteration settings using saved metadata.
        # Seed _saved_num_steps with the loaded value so that
        # _on_iteration_mode_changed (triggered by setting the radio below)
        # restores the correct count when a sweep mode is active.
        num_steps = self.params.get('num_steps', 1)
        iteration_mode = self.params.get('iteration_mode', 'single')
        if iteration_mode != 'single' and num_steps > 1:
            self._saved_num_steps = str(num_steps)

        if iteration_mode == 'single':
            self.single_iteration_radio.setChecked(True)
            # _on_iteration_mode_changed will lock the field to "1"
        
        elif iteration_mode == 'uniform':
            self.uniform_sweep_radio.setChecked(True)
            # _on_iteration_mode_changed restores _saved_num_steps into the field
            uniform_start = self.params.get('uniform_start_amplitude')
            uniform_stop = self.params.get('uniform_stop_amplitude')
            if uniform_start is not None:
                self.uniform_start_edit.setText(f"{uniform_start:.4g}")
            if uniform_stop is not None:
                self.uniform_stop_edit.setText(f"{uniform_stop:.4g}")
        
        elif iteration_mode == 'scaling':
            self.scaling_radio.setChecked(True)
            # _on_iteration_mode_changed restores _saved_num_steps into the field
            scale_start = self.params.get('scale_start_factor')
            scale_stop = self.params.get('scale_stop_factor')
            if scale_start is not None:
                self.scale_start_edit.setText(f"{scale_start:.4g}")
            if scale_stop is not None:
                self.scale_stop_edit.setText(f"{scale_stop:.4g}")
    
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
        """Refresh section display and update section_center_frequencies when the user
        changes the frequency source combo in re-run mode."""
        selected = self.section_freq_combo.currentText().lower()
        self.sections_edit.clear()

        if "fit" in selected:
            freqs = self.fit_frequencies or []
        elif "bias" in selected:
            freqs = self.bias_frequencies or []
        else:
            # "previous multisweep central frequencies" — use the original sweep centers
            # stored in _original_section_center_frequencies (set during _setup_ui)
            freqs = self._original_section_center_frequencies

        # Update section_center_frequencies so get_parameters() picks up the right list
        self.section_center_frequencies = list(freqs)

        self.sections_edit.setText(", ".join([f"{f/1e6:.6f}" for f in freqs]))
        n = len(freqs)
        self.sections_info_label.setText(f"{n} section(s) selected.")

    def _scroll_netanal_section(self):
        """Handle frequency source combo changes in netanal mode.

        When the user selects "Approx locations from Find Resonances" the editable
        field is pre-populated with the resonance frequencies found during the
        network analysis.  Selecting "Custom" clears the field so the user can
        type their own comma-separated MHz values.
        """
        selected = self.section_freq_combo.currentText()
        if "Find Resonances" in selected:
            # Restore the Find Resonances frequencies
            if self.section_center_frequencies:
                section_freq_str = ", ".join([f"{f / 1e6:.6f}" for f in self.section_center_frequencies])
                self.sections_edit.setText(section_freq_str)
                n = len(self.section_center_frequencies)
                self.sections_info_label.setText(f"{n} section(s) from Find Resonances.")
            else:
                self.sections_edit.clear()
                self.sections_info_label.setText("No Find Resonances data available.")
        else:  # "Custom"
            self.sections_edit.clear()
            self.sections_info_label.setText("Enter frequencies manually (MHz, comma separated).")
        
    
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
        
            idx = self.sweep_direction_combo.findText(params['sweep_direction'].capitalize(),
                                                      Qt.MatchFlag.MatchFixedString)
            if idx >= 0:
                self.sweep_direction_combo.setCurrentIndex(idx)
        
            self.apply_skewed_fit_checkbox.setChecked(params['apply_skewed_fit'])
            self.apply_nonlinear_fit_checkbox.setChecked(params['apply_nonlinear_fit'])
            if 'run_find_bias' in params:
                self.apply_find_bias_checkbox.setChecked(params['run_find_bias'])
        
            # Load iteration amplitudes (for amplitude iterations)
            amps = params.get("amps") or ([params["amp"]] if "amp" in params else None)
            if amps:
                try:
                    amp_text = ", ".join(f"{float(amp):g}" for amp in amps)
                except (TypeError, ValueError):
                    amp_text = ", ".join(str(amp) for amp in amps)
                self.global_amp_edit.setText(amp_text)
            
            # Load per-section amplitudes if they exist
            section_amps = params.get("section_amplitudes")
            if section_amps:
                try:
                    section_amp_text = ", ".join(f"{float(amp):g}" for amp in section_amps)
                    self.amp_array_edit.setText(section_amp_text)
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
                        self.amp_array_edit.setText(section_amp_text)
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
        # Support both new key name and legacy key name for backward compatibility
        freqs = params.get('sweep_center_frequencies') or params.get('resonance_frequencies', [])
        
        if raw_section_centers:
            return freqs

        else:
            ref_freqs = []
    
            # Extract fit frequencies from either new or old format
            if 'results_by_detector' in payload:
                # New detector-based format
                for det_idx in sorted(payload['results_by_detector'].keys()):
                    amp_dir_dict = payload['results_by_detector'][det_idx]
                    if not amp_dir_dict:
                        continue
                    entry = next(iter(amp_dir_dict.values()))
                    if params['apply_skewed_fit'] and entry.get('skewed_fit_success') and entry.get('fit_params'):
                        ref_freqs.append(entry['fit_params']['fr'])
                    elif params['apply_nonlinear_fit'] and entry.get('nonlinear_fit_success') and entry.get('nonlinear_fit_params'):
                        ref_freqs.append(entry['nonlinear_fit_params']['fr'])
                    else:
                        ref_freqs.append(entry.get('bias_frequency', entry.get('original_center_frequency')))
            elif 'results_by_iteration' in payload:
                # Old iteration-based format (backward compatibility)
                if params['apply_skewed_fit']:
                    for i in range(len(freqs)):
                        ref_freqs.append(payload['results_by_iteration'][0]['data'][i+1]['fit_params']['fr'])
                elif params['apply_nonlinear_fit']:
                    for i in range(len(freqs)):
                        ref_freqs.append(payload['results_by_iteration'][0]['data'][i+1]['nonlinear_fit_params']['fr'])
                else:
                    for i in range(len(freqs)):
                        ref_freqs.append(payload['results_by_iteration'][0]['data'][i+1]['bias_frequency'])

            ref_freqs.sort()
            return ref_freqs
        
    
    def _on_iteration_mode_changed(self):
        """Handle changes to iteration mode radio buttons.

        - Single iteration: amplitude input fields are shown; steps locked to 1.
        - Uniform sweep: start/stop amplitude fields shown; steps editable.
        - Multiplicative scaling: factor fields + info label shown; steps editable.
          The per-resonator ``bias_amplitude`` from ``res_info_dict`` is always
          used as the base (no manual amplitude entry needed).
        """
        is_single = self.single_iteration_radio.isChecked()
        is_scaling = self.scaling_radio.isChecked()

        if is_single:
            # Save the current value only if it is a real user value (not "1" forced
            # by a previous switch to single-iteration mode).
            current = self.num_steps_edit.text().strip()
            if current and current != "1":
                self._saved_num_steps = current
            self.num_steps_edit.setText("1")
            self.num_steps_edit.setEnabled(False)
        else:
            self.num_steps_edit.setEnabled(True)
            # Restore saved value when switching to a sweep mode
            self.num_steps_edit.setText(self._saved_num_steps)

        # Amplitude input fields only visible for single iteration
        if hasattr(self, 'single_amp_controls'):
            self.single_amp_controls.setVisible(is_single)

        self.uniform_controls.setVisible(self.uniform_sweep_radio.isChecked())
        self.scaling_controls.setVisible(is_scaling)
        # Re-run live validation so the status label and Start button reflect
        # the newly selected mode immediately (even before the user types).
        self._validate_amplitude_live()

    def _validate_amplitude_live(self):
        """Update the amplitude status label and Start button state in real-time
        as the user edits amplitude fields or switches iteration modes.

        Covers all three iteration modes:

        * **Single iteration** — requires exactly one of *Global amplitude* or
          *Amplitude array* to be filled.
        * **Uniform sweep** — requires both *Start amplitude* and *Stop amplitude*.
        * **Multiplicative scaling** — requires both scale factors **and** a
          pre-existing resonator registry (``res_info_dict`` in ``self.params``).

        The method is a no-op until ``_amp_status_label`` exists (i.e. it
        returns immediately if called before the full UI has been built).
        """
        if not hasattr(self, '_amp_status_label'):
            return

        # ── Determine the current number of sweep sections ────────────────
        if hasattr(self, 'sections_edit'):
            parts = [p.strip() for p in self.sections_edit.text().split(',') if p.strip()]
            num_sections = len(parts)
            sections_ok = num_sections > 0
        else:
            num_sections = len(self.section_center_frequencies)
            sections_ok = True  # sections fixed at construction time

        # Number of steps (used in the ✓ informational message)
        try:
            num_steps = int(self.num_steps_edit.text())
        except (ValueError, AttributeError):
            num_steps = 1

        # ── Per-mode validation ───────────────────────────────────────────
        amp_ok = False
        msg    = ""

        if self.single_iteration_radio.isChecked():
            global_text = self.global_amp_edit.text().strip()
            array_text  = self.amp_array_edit.text().strip()

            if global_text and array_text:
                msg = ("✗ Fill in only one field — either Global amplitude"
                       " or Amplitude array, not both.")

            elif not global_text and not array_text:
                msg = ("⚠ An amplitude is required — enter a value in"
                       " Global amplitude (one value for all sections)"
                       " or Amplitude array (one value per section, comma-separated).")

            elif global_text:
                try:
                    val = float(global_text)
                    if val <= 0:
                        msg = "✗ Global amplitude must be a positive number."
                    else:
                        msg = f"✓ Using global amplitude {val:g} for all sections."
                        amp_ok = True
                except ValueError:
                    msg = "✗ Global amplitude must be a valid number."

            else:  # array_text only
                try:
                    vals = [float(x.strip()) for x in array_text.split(',') if x.strip()]
                    if any(v <= 0 for v in vals):
                        msg = "✗ All amplitude values must be positive."
                    elif num_sections > 0 and len(vals) != num_sections:
                        msg = (f"⚠ Amplitude array has {len(vals)} value(s)"
                               f" but there are {num_sections} sweep section(s).")
                    else:
                        msg = f"✓ Using per-section amplitude array ({len(vals)} value(s))."
                        amp_ok = True
                except ValueError:
                    msg = "✗ Amplitude array must contain valid numbers, comma-separated."

        elif self.uniform_sweep_radio.isChecked():
            start_text = self.uniform_start_edit.text().strip()
            stop_text  = self.uniform_stop_edit.text().strip()

            if not start_text and not stop_text:
                msg = "⚠ Enter a start and stop amplitude for the uniform sweep."
            elif not start_text:
                msg = "⚠ Start amplitude is required."
            elif not stop_text:
                msg = "⚠ Stop amplitude is required."
            else:
                try:
                    start_val = float(start_text)
                    stop_val  = float(stop_text)
                    if start_val <= 0 or stop_val <= 0:
                        msg = "✗ Start and stop amplitudes must be positive."
                    else:
                        msg = (f"✓ Sweeping {start_val:g} → {stop_val:g}"
                               f" over {num_steps} step(s).")
                        amp_ok = True
                except ValueError:
                    msg = "✗ Start and stop amplitudes must be valid numbers."

        elif self.scaling_radio.isChecked():
            res_info   = self.params.get('res_info_dict')
            start_text = self.scale_start_edit.text().strip()
            stop_text  = self.scale_stop_edit.text().strip()

            if not res_info:
                msg = ("⚠ No resonator registry found. Run a single-amplitude"
                       " sweep first to build one, then re-open this dialog.")
            elif not start_text and not stop_text:
                msg = "⚠ Enter a start and stop scale factor."
            elif not start_text:
                msg = "⚠ Start factor is required."
            elif not stop_text:
                msg = "⚠ Stop factor is required."
            else:
                try:
                    start_val = float(start_text)
                    stop_val  = float(stop_text)
                    if start_val <= 0 or stop_val <= 0:
                        msg = "✗ Scale factors must be positive."
                    else:
                        msg = (f"✓ Scaling bias amplitudes by {start_val:g}×"
                               f" → {stop_val:g}× over {num_steps} step(s).")
                        amp_ok = True
                except ValueError:
                    msg = "✗ Scale factors must be valid numbers."

        # ── Update status label ───────────────────────────────────────────
        self._amp_status_label.setText(msg)
        if amp_ok:
            self._amp_status_label.setStyleSheet("color: #155724; font-weight: bold;")
        elif msg.startswith("✗"):
            self._amp_status_label.setStyleSheet("color: #721c24; font-weight: bold;")
        else:
            self._amp_status_label.setStyleSheet("color: #856404; font-weight: bold;")

        # ── Update Start button ───────────────────────────────────────────
        if hasattr(self, 'start_btn'):
            can_start = amp_ok and sections_ok
            self.start_btn.setEnabled(can_start)
            if not can_start:
                tip_parts = []
                if not amp_ok and msg:
                    # Strip the leading symbol for the tooltip
                    tip_parts.append(msg.lstrip("⚠✗ "))
                if not sections_ok:
                    tip_parts.append("No sweep sections specified.")
                self.start_btn.setToolTip("\n".join(tip_parts))
            else:
                self.start_btn.setToolTip("")

    @QtCore.pyqtSlot()
    def _on_file_dialog_closed(self):
        """Handle closure of the file dialog without file selection."""
        pass  # Optional: keep or clear dialog

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
            base = datetime.datetime.now().strftime("multisweep_%H%M%S")
        return f"{base}_{suffix}" if suffix else base

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
            if self.load_multisweep or self.netanal_mode:
                # Parse from the editable text field (comma-separated MHz values)
                params_dict['sweep_center_frequencies'] = []
                for f in self.sections_edit.text().split(','):
                    f = f.strip()
                    if f:
                        params_dict['sweep_center_frequencies'].append(np.float64(f) * 1e6)
            else:
                params_dict['sweep_center_frequencies'] = self.section_center_frequencies
            
            num_sections = len(params_dict['sweep_center_frequencies'])

            # STEP 1: Parse base amplitude (single-iteration mode only).
            # For uniform sweep, amp_arrays are built entirely from the start/stop
            # fields in Step 2 — global_amp / amp_array are hidden and not used.
            # For multiplicative scaling, the base always comes from res_info_dict
            # (also handled in Step 2), so neither field is consulted here.
            if self.single_iteration_radio.isChecked():
                global_amp_text = self.global_amp_edit.text().strip()
                amp_array_text = self.amp_array_edit.text().strip()

                if global_amp_text and amp_array_text:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Amplitude Error",
                        "Both amplitude fields are filled.\n\n"
                        "Clear one: use 'Global amplitude' for a single value across all "
                        "sections, or 'Amplitude array' for per-section values."
                    )
                    return None
                elif not global_amp_text and not amp_array_text:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Amplitude Error",
                        "No amplitude specified.\n\n"
                        "Enter a value in 'Global amplitude' (one value applied to all "
                        "sections) or 'Amplitude array' (one value per section, "
                        "comma-separated)."
                    )
                    return None

                if global_amp_text:
                    base_amp = float(global_amp_text)
                    if base_amp <= 0:
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Amplitude Error",
                            "Global amplitude must be a positive number."
                        )
                        return None
                    base_amp_array = [base_amp] * num_sections
                    params_dict['base_amplitude_mode'] = 'global'
                    params_dict['base_amplitude_values'] = base_amp
                else:
                    base_amp_array = [float(x.strip()) for x in amp_array_text.split(',')]
                    if len(base_amp_array) != num_sections:
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Amplitude Error",
                            f"Amplitude array has {len(base_amp_array)} value(s) but there "
                            f"are {num_sections} sweep section(s).\n\n"
                            "Provide exactly one amplitude value per section."
                        )
                        return None
                    for i, amp in enumerate(base_amp_array):
                        if amp <= 0:
                            QtWidgets.QMessageBox.warning(
                                self,
                                "Amplitude Error",
                                f"All amplitude values must be positive "
                                f"(section {i+1} has value {amp})."
                            )
                            return None
                    params_dict['base_amplitude_mode'] = 'array'
                    params_dict['base_amplitude_values'] = base_amp_array.copy()
            else:
                # Uniform sweep and scaling: base_amp_array is derived in Step 2
                base_amp_array = None
            
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
                params_dict['iteration_mode'] = 'single'
            
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
                
                # Store iteration metadata
                params_dict['iteration_mode'] = 'uniform'
                params_dict['uniform_start_amplitude'] = start_amp
                params_dict['uniform_stop_amplitude'] = stop_amp
            
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

                # Derive per-resonator base amplitudes from the resonator registry
                # (in key order, which matches the amp ordering used by
                # MultisweepTask / crs.multisweep Option B).
                res_info = self.params.get('res_info_dict')
                if not res_info:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Validation Error",
                        "Multiplicative scaling requires a resonator registry (res_info_dict).\n"
                        "Please run a single-amplitude sweep first to build one."
                    )
                    return None
                try:
                    base_amp_array = [
                        float(info['bias_amplitude'])
                        for info in res_info.values()
                    ]
                except (KeyError, TypeError, ValueError) as exc:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Validation Error",
                        f"Could not read bias_amplitude from the resonator registry: {exc}"
                    )
                    return None
                if any(a <= 0 for a in base_amp_array):
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Validation Error",
                        "One or more bias_amplitude values in the resonator registry "
                        "are non-positive.  Please check res_info_dict."
                    )
                    return None
                params_dict['base_amplitude_mode'] = 'res_info_dict'
                params_dict['base_amplitude_values'] = list(base_amp_array)

                # Generate scale factors and produce per-iteration amplitude arrays
                factors = np.linspace(start_factor, stop_factor, num_steps)
                params_dict['amp_arrays'] = [
                    [amp * factor for amp in base_amp_array]
                    for factor in factors
                ]

                # Store iteration metadata
                params_dict['iteration_mode'] = 'scaling'
                params_dict['scale_start_factor'] = start_factor
                params_dict['scale_stop_factor'] = stop_factor
            
            else:
                QtWidgets.QMessageBox.warning(
                    self, 
                    "Validation Error", 
                    "Must select an iteration mode."
                )
                return None
            
            # Store number of steps for all modes
            params_dict['num_steps'] = num_steps
            
            # Parse other parameters
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
            params_dict['run_find_bias'] = self.apply_find_bias_checkbox.isChecked()
            
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
            
            # Capture the user-specified measurement name (not persisted as a default)
            params_dict['measurement_name'] = self._get_measurement_name()
            # Store the raw suffix separately so the re-run dialog can restore it
            params_dict['measurement_custom_suffix'] = self.custom_suffix_edit.text().strip()

            # Save these parameters as defaults for future sessions
            # (only if not loading from file)
            if not self.use_data_from_file:
                try:
                    settings.set_multisweep_defaults(params_dict)
                except Exception as e:
                    # Don't fail the dialog if settings save fails
                    print(f"Warning: Could not save multisweep defaults: {e}")
            
            return params_dict
            
        except ValueError as e: # Handles errors from float() or int() conversion

            QtWidgets.QMessageBox.critical(self, "Input Error", f"Invalid numerical input: {str(e)}")
            return None
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not parse parameters: {str(e)}")
            return None
