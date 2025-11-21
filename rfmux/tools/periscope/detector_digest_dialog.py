"""Window for displaying a detailed digest of a single detector resonance."""

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets, QtGui

from .utils import (
    ClickableViewBox, UnitConverter, LINE_WIDTH,
    IQ_COLORS, SCATTER_COLORS, DISTINCT_PLOT_COLORS, COLORMAP_CHOICES,
    UPWARD_SWEEP_STYLE, DOWNWARD_SWEEP_STYLE, FITTING_COLORS
)
from rfmux.core.transferfunctions import convert_roc_to_volts, convert_roc_to_dbm, exp_bin_noise_data, PFB_SAMPLING_FREQ

class DetectorDigestWindow(QtWidgets.QMainWindow):
    """
    A window that displays a detailed "digest" of a single detector resonance,
    including three plots: Sweep (vs freq.), Sweep (IQ plane), Bias amplitude optimization,
    and a panel for fitting results.
    """
    def __init__(self, parent: QtWidgets.QWidget = None,
                 resonance_data_for_digest: dict = None, # {amp_raw_direction_key: {'data': sweep_data_dict, 'actual_cf_hz': float, 'direction': str, 'amplitude': float}}
                 detector_id: int = -1,
                 resonance_frequency_ghz: float = 0.0,
                 dac_scales: dict = None, # {module_id: scale_dbm}
                 zoom_box_mode: bool = True,
                 target_module: int = None,
                 normalize_plot3: bool = False, 
                 dark_mode: bool = False,
                 all_detectors_data: dict = None,  
                 initial_detector_idx: int = None,
                 noise_data = None,
                 spectrum_data = None,
                 debug_noise_data = None,
                 debug_phase_data = None,
                 debug = False): 
        super().__init__(parent)
        self.resonance_data_for_digest = resonance_data_for_digest or {} 
        self.detector_id = detector_id
        self.resonance_frequency_ghz_title = resonance_frequency_ghz 
        self.dac_scales = dac_scales or {}
        self.zoom_box_mode = zoom_box_mode
        self.target_module = target_module
        self.normalize_plot3 = normalize_plot3
        self.dark_mode = dark_mode 
        self.parent_window = parent

        self.current_detector = self.detector_id

        self.noise_tab_avail = False

        self.reference = "absolute"
        
        # Navigation support
        self.all_detectors_data = all_detectors_data or {}
        self.detector_indices = sorted(self.all_detectors_data.keys()) if self.all_detectors_data else []
        self.current_detector_index_in_list = 0  # Index in detector_indices list
        
        # If we have multiple detectors, find the initial one
        if self.detector_indices and initial_detector_idx is not None:
            try:
                self.current_detector_index_in_list = self.detector_indices.index(initial_detector_idx)
            except ValueError:
                self.current_detector_index_in_list = 0

        self.active_amplitude_raw_key = None # Stores the key like "amp_val:direction"
        self.active_sweep_info = None 
        self.active_sweep_data = None 
        self.current_plot_offset_hz = None 
        
        self.noise_data = noise_data
        self.noise_i_data = None
        self.noise_q_data = None

        #### Noise spectrum Tab #####
        self.tabs = None
        self.digest_page = None
        self.mean_subtract_enabled = False
        self.exp_binning_enabled = False

        self.spectrum_data = spectrum_data
        self.fast_freq = PFB_SAMPLING_FREQ/2 #### 2.44 MSS = 1.22 MHz nyquist frequency
        self.slow_freq = 0

        self.show_fast_tod = False



        ### Data products for plotting ####
        if self.spectrum_data:
            self.noise_tab_avail = True
            self.single_psd_i = self.spectrum_data['single_psd_i'][self.detector_id - 1]
            self.single_psd_q = self.spectrum_data['single_psd_q'][self.detector_id - 1]
            self.tod_i = self.spectrum_data['I'][self.detector_id - 1]
            self.tod_q = self.spectrum_data['Q'][self.detector_id - 1]
            self.reference = self.spectrum_data['reference']
            self.tone_amp = self.spectrum_data['amplitudes_dbm'][self.detector_id - 1]
            self.slow_freq = self.spectrum_data['slow_freq_hz']
            if self.spectrum_data['pfb_enabled']:
                self.show_fast_tod = True
            
            if self.show_fast_tod:
                self.pfb_psd_i = self.spectrum_data['pfb_psd_i'][self.detector_id - 1]
                self.pfb_psd_q = self.spectrum_data['pfb_psd_q'][self.detector_id - 1]
                self.pfb_tod_i = self.spectrum_data['pfb_I'][self.detector_id - 1]
                self.pfb_tod_q = self.spectrum_data['pfb_Q'][self.detector_id - 1]
                self.pfb_freq_iq = self.spectrum_data['pfb_freq_iq'][self.detector_id - 1]
                self.pfb_freq_dsb = self.spectrum_data['pfb_freq_dsb'][self.detector_id - 1]
                self.pfb_dual_psd = self.spectrum_data['pfb_dual_psd'][self.detector_id - 1]
        else:
            self.noise_tab_avail = False ### You can't access noise tab if no noise data was collected
        
        #### Extra debugging step ####
        self.debug = debug

        if self.debug:
            self.full_debug = debug_noise_data
            self.debug_noise = self.full_debug[self.detector_id]
            self.phase_debug = debug_phase_data
        
        if self.noise_data is not None:
            self.noise_i_data = self.noise_data.i[self.detector_id-1]
            self.noise_q_data = self.noise_data.q[self.detector_id-1]


        if self.resonance_data_for_digest:
            try:
                sorted_keys = sorted(
                    self.resonance_data_for_digest.keys(),
                    key=lambda k: float(k.split(":")[0]) 
                )
                if sorted_keys:
                    self.active_amplitude_raw_key = sorted_keys[0]
                    self.active_sweep_info = self.resonance_data_for_digest[self.active_amplitude_raw_key]
                    self.active_sweep_data = self.active_sweep_info['data']
                    self.current_plot_offset_hz = self.active_sweep_info['actual_cf_hz']
            except (ValueError, IndexError, KeyError) as e: # Added KeyError
                print(f"Error selecting default sweep: {e}. Keys: {list(self.resonance_data_for_digest.keys())}")
                if self.resonance_data_for_digest: 
                    try: # More robust fallback
                        first_key = next(iter(self.resonance_data_for_digest))
                        self.active_amplitude_raw_key = first_key
                        self.active_sweep_info = self.resonance_data_for_digest[self.active_amplitude_raw_key]
                        self.active_sweep_data = self.active_sweep_info['data']
                        self.current_plot_offset_hz = self.active_sweep_info.get('actual_cf_hz', self.resonance_frequency_ghz_title * 1e9)
                    except Exception as fallback_e:
                         print(f"Critical error in fallback default sweep selection: {fallback_e}")


        if self.current_plot_offset_hz is None: 
             self.current_plot_offset_hz = self.resonance_frequency_ghz_title * 1e9

        self._setup_ui()
        self._update_plots()

        self.setWindowTitle(f"Detector Digest: Detector {self.detector_id}  ({self.resonance_frequency_ghz_title*1e3:.6f} MHz)")
        self.setWindowFlags(QtCore.Qt.WindowType.Window)
        self.resize(1200, 800) # Increased size to accommodate tables properly

    
    def _setup_ui(self):
        """Sets up the UI layout with plots, fitting, and noise information panel."""
        # Root central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
    
        outer_layout = QtWidgets.QVBoxLayout(central_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
    
        # Tabs
        self.tabs = QtWidgets.QTabWidget()
        outer_layout.addWidget(self.tabs)
    
        # ---- Tab 1: existing Digest/Fit UI ----
        self.digest_page = self._build_digest_tab()
        self.tabs.addTab(self.digest_page, "Fit")
    
        # ---- Tab 2: Noise UI ----
        self.noise_tab = self._build_noise_tab()
        self.tabs.addTab(self.noise_tab, "Noise")
        noise_tab_index = self.tabs.indexOf(self.noise_tab)
        self.tabs.setTabEnabled(noise_tab_index, self.noise_tab_avail)
    
        # --- Optional: connect tab change ---
        self.tabs.currentChanged.connect(self._on_tab_changed if hasattr(self, "_on_tab_changed") else lambda _: None)
    
        # --- Apply theme + finish ---
        self.apply_theme(self.dark_mode)
        self._update_plots()
    
        # Window configuration
        self.setWindowTitle(f"Detector Digest: Detector {self.detector_id}  ({self.resonance_frequency_ghz_title*1e3:.6f} MHz)")
        self.setWindowFlags(QtCore.Qt.WindowType.Window)
        self.resize(1200, 800)


    def _on_tab_changed(self, index: int):
        """Handle switching between tabs (Fit and Noise)."""
        if not hasattr(self, "tabs"):
            return
    
        current_tab_name = self.tabs.tabText(index)
    
        # --- When switched to the Fit (Digest) tab ---
        if current_tab_name == "Fit":
            return
    
        # --- When switched to the Noise tab ---
        if current_tab_name == "Noise":
            # Update title and count to match the current detector
            if hasattr(self, "title_label_noise"):
                title_text = f"Detector {self.detector_id} ({self.resonance_frequency_ghz_title * 1e3:.6f} MHz)"
                self.title_label_noise.setText(title_text)
    
            if hasattr(self, "detector_count_label_noise") and self.detector_indices:
                count_text = f"({self.current_detector_index_in_list + 1} of {len(self.detector_indices)})"
                self.detector_count_label_noise.setText(count_text)


            multiple_detectors = len(self.detector_indices) > 1
            if hasattr(self, "prev_button_noise"):
                self.prev_button_noise.setEnabled(multiple_detectors)
            if hasattr(self, "next_button_noise"):
                self.next_button_noise.setEnabled(multiple_detectors)
    
            # If you later add noise plotting logic, you can enable this:
            if hasattr(self, "_update_noise_plots"):
                self._update_noise_plots()
    
            # For now, do nothing else — this is just a layout switch.
            return

    
    def _build_digest_tab(self):
        """Sets up the UI layout with plots and fitting information panel."""
        page = QtWidgets.QWidget()
        
        # Main vertical layout for the entire window content
        outer_layout = QtWidgets.QVBoxLayout(page)
        outer_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for cleaner look
        
        # Set the background color of the dialog based on dark mode
        bg_color_hex = "#1C1C1C" if self.dark_mode else "#FFFFFF" # Hex for stylesheet
        page.setStyleSheet(f"QWidget {{ background-color: {bg_color_hex}; }}") # Apply to QWidget base
        
        # Create main splitter for resizable panes
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        
        # Create top widget for navigation and plots
        top_widget = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top_widget)
        top_layout.setContentsMargins(5, 5, 5, 0)  # Small margins for the top section
        
        # Create navigation header with title and buttons
        nav_widget = QtWidgets.QWidget()
        nav_layout = QtWidgets.QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        
        title_color_str = "white" if self.dark_mode else "black"
        
        # Previous button
        self.prev_button = QtWidgets.QPushButton("◀ Previous")
        self.prev_button.clicked.connect(self._navigate_previous)
        self.prev_button.setEnabled(len(self.detector_indices) > 1)
        nav_layout.addWidget(self.prev_button)
        
        # Title in the center
        title_text = f"Detector {self.detector_id} ({self.resonance_frequency_ghz_title*1e3:.6f} MHz)"
        self.title_label = QtWidgets.QLabel(title_text)
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        font = self.title_label.font()
        font.setPointSize(font.pointSize() + 2); font.setBold(False) 
        self.title_label.setFont(font)
        self.title_label.setStyleSheet(f"QLabel {{ margin-bottom: 10px; color: {title_color_str}; background-color: transparent; }}")
        nav_layout.addWidget(self.title_label, 1)  # Stretch factor 1 to center
        
        # Next button
        self.next_button = QtWidgets.QPushButton("Next ▶")
        self.next_button.clicked.connect(self._navigate_next)
        self.next_button.setEnabled(len(self.detector_indices) > 1)
        nav_layout.addWidget(self.next_button)

        self.refresh_noise_button = QtWidgets.QPushButton("Check Noise")
        self.refresh_noise_button.setStyleSheet("background-color: #ffcccc; color: black;")
        self.refresh_noise_button.clicked.connect(self._refresh_noise_samps)
        self.refresh_noise_button.setToolTip("Captures 100 I,Q points for each detector and over-plots them on the I,Q plot in the detector digest windows. Used to conveniently re-assess detector state.")
        self.refresh_noise_button.setEnabled(len(self.detector_indices) > 1)
        nav_layout.addWidget(self.refresh_noise_button)
        
        # Add detector count label
        detector_count_text = ""
        if self.detector_indices:
            detector_count_text = f"({self.current_detector_index_in_list + 1} of {len(self.detector_indices)})"
        self.detector_count_label = QtWidgets.QLabel(detector_count_text)
        self.detector_count_label.setStyleSheet(f"QLabel {{ color: {title_color_str}; background-color: transparent; }}")
        nav_layout.addWidget(self.detector_count_label)
        
        # Add trace navigation hint
        trace_hint_text = "↑↓ to switch traces" if self.resonance_data_for_digest and len(self.resonance_data_for_digest) > 1 else ""
        self.trace_hint_label = QtWidgets.QLabel(trace_hint_text)
        self.trace_hint_label.setStyleSheet(f"QLabel {{ color: {title_color_str}; background-color: transparent; font-size: 10pt; }}")
        nav_layout.addWidget(self.trace_hint_label)
        
        top_layout.addWidget(nav_widget)
        
        plots_layout = QtWidgets.QHBoxLayout()
        plot_bg_color, plot_pen_color = ("k", "w") if self.dark_mode else ("w", "k")

        # Plot 1: Sweep (vs freq.)
        vb1 = ClickableViewBox(); vb1.parent_window = self
        self.plot1_sweep_vs_freq = pg.PlotWidget(viewBox=vb1, name="SweepVsFreq")
        self.plot1_sweep_vs_freq.setLabel('left', "Amplitude", units="V") 
        self.plot1_sweep_vs_freq.setLabel('bottom', "f - f_bias", units="Hz") 
        self.plot1_sweep_vs_freq.showGrid(x=True, y=True, alpha=0.3)
        self.plot1_legend = self.plot1_sweep_vs_freq.addLegend(offset=(30,10), labelTextColor=plot_pen_color)
        # Set smaller font for legend
        self.plot1_legend.setLabelTextSize('8pt')
        plots_layout.addWidget(self.plot1_sweep_vs_freq)

        # Plot 2: Sweep (IQ plane)
        vb2 = ClickableViewBox(); vb2.parent_window = self
        self.plot2_iq_plane = pg.PlotWidget(viewBox=vb2, name="SweepIQPlane")
        self.plot2_iq_plane.setLabel('left', "Q", units="V") 
        self.plot2_iq_plane.setLabel('bottom', "I", units="V") 
        self.plot2_iq_plane.setAspectLocked(True)
        self.plot2_iq_plane.showGrid(x=True, y=True, alpha=0.3)
        self.plot2_legend = self.plot2_iq_plane.addLegend(offset=(30,10), labelTextColor=plot_pen_color)
        # Set smaller font for legend
        self.plot2_legend.setLabelTextSize('8pt')
        plots_layout.addWidget(self.plot2_iq_plane)

        # Plot 3: Bias amplitude optimization
        vb3 = ClickableViewBox(); vb3.parent_window = self
        self.plot3_bias_opt = pg.PlotWidget(viewBox=vb3, name="BiasOpt")
        self.plot3_bias_opt.setLabel('left', "|S21|", units="dB")
        self.plot3_bias_opt.setLabel('bottom', "f - f_bias", units="Hz") 
        self.plot3_bias_opt.showGrid(x=True, y=True, alpha=0.3)
        self.plot3_legend = self.plot3_bias_opt.addLegend(offset=(30,10), labelTextColor=plot_pen_color)
        # Set smaller font for legend
        self.plot3_legend.setLabelTextSize('8pt')
        plots_layout.addWidget(self.plot3_bias_opt)
        vb3.doubleClickedEvent.connect(self._handle_plot3_double_click)
        
        top_layout.addLayout(plots_layout)  # Add plots to top widget

        # Add top widget to splitter
        self.main_splitter.addWidget(top_widget)
        
        # Fitting Information Panel
        self.fitting_info_group = QtWidgets.QGroupBox("Fitting Results")
        self.fitting_info_group.setStyleSheet(f"QGroupBox {{ color: {title_color_str}; border: 1px solid {title_color_str}; margin-top: 0.5em;}} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }}")
        fitting_info_main_layout = QtWidgets.QHBoxLayout(self.fitting_info_group)

        # Column 1: Skewed Fit
        skewed_fit_group = QtWidgets.QGroupBox("Skewed Lorentzian Fit")
        skewed_fit_group.setStyleSheet(f"QGroupBox {{ color: {title_color_str}; border: none;}}")
        skewed_fit_layout = QtWidgets.QVBoxLayout(skewed_fit_group)
        
        # Create table for skewed fit
        self.skewed_table = QtWidgets.QTableWidget()
        self.skewed_table.setColumnCount(3)
        self.skewed_table.setHorizontalHeaderLabels(["Parameter", "Value", "Description"])
        self.skewed_table.horizontalHeader().setStretchLastSection(True)
        self.skewed_table.verticalHeader().setVisible(False)
        self.skewed_table.setAlternatingRowColors(True)
        self.skewed_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.skewed_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        
        # Set smaller font for table
        table_font = self.skewed_table.font()
        table_font.setPointSize(table_font.pointSize())
        self.skewed_table.setFont(table_font)
        self.skewed_table.horizontalHeader().setFont(table_font)
        self.skewed_table.verticalHeader().setDefaultSectionSize(20)  # Smaller row height
        
        # Set column widths
        self.skewed_table.setColumnWidth(0, 80)
        self.skewed_table.setColumnWidth(1, 100)
        
        # Initialize rows for skewed fit
        self._init_skewed_table_rows()
        
        skewed_fit_layout.addWidget(self.skewed_table)
        fitting_info_main_layout.addWidget(skewed_fit_group)

        # Column 2: Nonlinear Fit
        nl_fit_group = QtWidgets.QGroupBox("Nonlinear Fit")
        nl_fit_group.setStyleSheet(f"QGroupBox {{ color: {title_color_str}; border: none;}}")
        nl_fit_layout = QtWidgets.QVBoxLayout(nl_fit_group)
        
        # Create table for nonlinear fit
        self.nl_table = QtWidgets.QTableWidget()
        self.nl_table.setColumnCount(3)
        self.nl_table.setHorizontalHeaderLabels(["Parameter", "Value", "Description"])
        self.nl_table.horizontalHeader().setStretchLastSection(True)
        self.nl_table.verticalHeader().setVisible(False)
        self.nl_table.setAlternatingRowColors(True)
        self.nl_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.nl_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        
        # Set smaller font for table
        table_font = self.nl_table.font()
        table_font.setPointSize(table_font.pointSize())
        self.nl_table.setFont(table_font)
        self.nl_table.horizontalHeader().setFont(table_font)
        self.nl_table.verticalHeader().setDefaultSectionSize(20)  # Smaller row height
        
        # Set column widths
        self.nl_table.setColumnWidth(0, 80)
        self.nl_table.setColumnWidth(1, 100)
        
        # Initialize rows for nonlinear fit
        self._init_nl_table_rows()
        
        nl_fit_layout.addWidget(self.nl_table)
        fitting_info_main_layout.addWidget(nl_fit_group)
        
        # Add fitting info group to splitter
        self.main_splitter.addWidget(self.fitting_info_group)
        
        # Configure splitter
        self.main_splitter.setSizes([600, 400])  # 60% plots, 40% tables
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(5)
        
        # Add splitter to main layout
        outer_layout.addWidget(self.main_splitter)

        self._apply_zoom_box_mode_to_all()
        self.apply_theme(self.dark_mode)

        return page


    def _build_noise_tab(self):
        """Builds the Noise tab."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(5, 5, 5, 5)

        bg_color_hex = "#1C1C1C" if self.dark_mode else "#FFFFFF" # Hex for stylesheet
        page.setStyleSheet(f"QWidget {{ background-color: {bg_color_hex}; }}") # Apply to QWidget base
    
        title_color_str = "white" if self.dark_mode else "black"
        plot_bg_color, plot_pen_color = ("k", "w") if self.dark_mode else ("w", "k")
    
        # --- Navigation Bar (same as Digest tab) ---
        nav_widget = QtWidgets.QWidget()
        nav_layout = QtWidgets.QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 0, 0, 0)

        #### Mean enabled ######
        checkbox_widget = QtWidgets.QWidget()
        checkbox_layout = QtWidgets.QHBoxLayout(checkbox_widget)
        checkbox_layout.setContentsMargins(5, 0, 5, 0)
        
        self.mean_subtract_checkbox = QtWidgets.QCheckBox("Mean Subtracted")
        self.mean_subtract_checkbox.setToolTip("If checked, TOD data will have its mean subtracted before plotting.")
        self.mean_subtract_checkbox.stateChanged.connect(self._toggle_mean_subtraction)

        self.exp_binning_checkbox = QtWidgets.QCheckBox("Exponential Binning")
        self.exp_binning_checkbox.setToolTip("If checked, noise spectrum will use exponential binning.")
        self.exp_binning_checkbox.toggled.connect(self._toggle_exp_binning)
        
        checkbox_layout.addStretch()
        checkbox_layout.addWidget(self.mean_subtract_checkbox)
        checkbox_layout.addSpacing(20)
        checkbox_layout.addWidget(self.exp_binning_checkbox)
        checkbox_layout.addStretch()
        
        layout.addWidget(checkbox_widget)

        ##### Buttons #####
    
        self.prev_button_noise = QtWidgets.QPushButton("◀ Previous")
        self.prev_button_noise.clicked.connect(self._navigate_previous)
        self.prev_button_noise.setEnabled(len(self.detector_indices) > 1)
        nav_layout.addWidget(self.prev_button_noise)
    
        title_text = f"Detector {self.detector_id} ({self.resonance_frequency_ghz_title*1e3:.6f} MHz)"
        self.title_label_noise = QtWidgets.QLabel(title_text)
        self.title_label_noise.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        font = self.title_label_noise.font()
        font.setPointSize(font.pointSize() + 2)
        self.title_label_noise.setFont(font)
        self.title_label_noise.setStyleSheet(f"QLabel {{ color: {title_color_str}; background-color: transparent; }}")
        nav_layout.addWidget(self.title_label_noise, 1)
    
        self.next_button_noise = QtWidgets.QPushButton("Next ▶")
        self.next_button_noise.clicked.connect(self._navigate_next)
        self.next_button_noise.setEnabled(len(self.detector_indices) > 1)
        nav_layout.addWidget(self.next_button_noise)
    
        detector_count_text = f"({self.current_detector_index_in_list + 1} of {len(self.detector_indices)})" if self.detector_indices else ""
        self.detector_count_label_noise = QtWidgets.QLabel(detector_count_text)
        self.detector_count_label_noise.setStyleSheet(f"QLabel {{ color: {title_color_str}; background-color: transparent; }}")
        nav_layout.addWidget(self.detector_count_label_noise)
    
        layout.addWidget(nav_widget)
    
        # --- Splitter with Conditional Fast TOD Plot ---
        splitter_main = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        
        # --- Top Splitter (Horizontal) ---
        splitter_top = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        
        # Plot colors
        plot_bg_color, plot_pen_color = ("k", "w") if self.dark_mode else ("w", "k")
        
        # --- Top Left Plot (Time vs Magnitude) ---
        vb_time = ClickableViewBox(); vb_time.parent_window = self
        self.plot_time_vs_mag = pg.PlotWidget(viewBox=vb_time, name="TimeVsAmplitude")
        self.plot_time_vs_mag.setBackground(plot_bg_color)
        self.plot_time_vs_mag.setLabel('left', "Amplitude", units="V")
        self.plot_time_vs_mag.setLabel('bottom', "Time", units="s")
        self.plot_time_vs_mag.showGrid(x=True, y=True, alpha=0.3)
        self.plot_time_vs_mag.setTitle("TOD", color=plot_pen_color)
        self.plot_time_vs_mag.addLegend(offset=(30, 10), labelTextColor=plot_pen_color)
        splitter_top.addWidget(self.plot_time_vs_mag)

        
        # --- Conditionally Add Fast TOD Plot ---
        if self.show_fast_tod:  # your condition flag
            vb_fast_tod = ClickableViewBox(); vb_fast_tod.parent_window = self
            self.plot_fast_tod = pg.PlotWidget(viewBox=vb_fast_tod, name="FastTOD")
            self.plot_fast_tod.setBackground(plot_bg_color)
            self.plot_fast_tod.setLabel('left', "Amplitude", units="V")
            self.plot_fast_tod.setLabel('bottom', "Time", units="s")
            self.plot_fast_tod.showGrid(x=True, y=True, alpha=0.3)
            self.plot_fast_tod.setTitle("Fast TOD", color=plot_pen_color)
            self.plot_fast_tod.addLegend(offset=(30, 10), labelTextColor=plot_pen_color)
            splitter_top.addWidget(self.plot_fast_tod)
        
        # --- Bottom Plot (Noise Spectrum) ---
        vb_spec = ClickableViewBox(); vb_spec.parent_window = self
        self.plot_noise_spectrum = pg.PlotWidget(viewBox=vb_spec, name="NoiseSpectrum")
        self.plot_noise_spectrum.setBackground(plot_bg_color)
        if self.reference == "relative":
            self.plot_noise_spectrum.setLabel('left', "Amplitude", units="dBc/Hz")
        else:
            self.plot_noise_spectrum.setLabel('left', "Amplitude", units="dBm/Hz")
        self.plot_noise_spectrum.setLabel('bottom', "Frequency", units="Hz")
        self.plot_noise_spectrum.showGrid(x=True, y=True, alpha=0.3)
        self.plot_noise_spectrum.setTitle("Noise Spectrum", color=plot_pen_color)
        self.plot_noise_spectrum.setYRange(-180, 0)
        self.plot_noise_spectrum.addLegend(offset=(30, 10), labelTextColor=plot_pen_color)
        
        # --- Combine Splitters ---
        splitter_main.addWidget(splitter_top)
        splitter_main.addWidget(self.plot_noise_spectrum)
        splitter_main.setSizes([400, 400])
        
        layout.addWidget(splitter_main)
        self.apply_theme(self.dark_mode)

    
        return page

    def _refresh_noise_samps(self):
        self.current_detector = self.detector_id
        self.refresh_noise_button.setEnabled(False)
        self.noise_data = self.parent()._take_noise_samps()
        self.noise_i_data = self.noise_data.i[self.current_detector - 1]
        self.noise_q_data = self.noise_data.q[self.current_detector - 1]
        self.refresh_noise_button.setEnabled(True)
        self._update_plots()
    
    
    def _init_skewed_table_rows(self):
        """Initialize the rows for the skewed fit table."""
        params = [
            ("Status", "", "Fit convergence status"),
            ("fr", "MHz", "Resonance frequency"),
            ("Qr", "", "Total quality factor (typical: 1e3-1e6)"),
            ("Qc", "", "Coupling quality factor (typical: 1e3-1e7)"),
            ("Qi", "", "Internal quality factor (typical: 1e4-1e7)"),
            ("Bifurcation", "", "Bifurcation status (from nonlinear fit)")
        ]
        
        self.skewed_table.setRowCount(len(params))
        for i, (param, unit, desc) in enumerate(params):
            # Parameter name
            self.skewed_table.setItem(i, 0, QtWidgets.QTableWidgetItem(f"{param} ({unit})" if unit else param))
            # Value (will be updated later)
            self.skewed_table.setItem(i, 1, QtWidgets.QTableWidgetItem("N/A"))
            # Description
            self.skewed_table.setItem(i, 2, QtWidgets.QTableWidgetItem(desc))
    
    def _init_nl_table_rows(self):
        """Initialize the rows for the nonlinear fit table."""
        params = [
            ("Status", "", "Fit convergence status"),
            ("fr_nl", "MHz", "Nonlinear resonance frequency"),
            ("Qr_nl", "", "Total quality factor (typical: 1e3-1e6)"),
            ("Qc_nl", "", "Coupling quality factor (typical: 1e3-1e7)"),
            ("Qi_nl", "", "Internal quality factor (typical: 1e4-1e7)"),
            ("a", "", "Nonlinearity parameter (bifurcation at ~0.77)"),
            ("φ", "deg", "Impedance mismatch phase"),
            ("I0", "", "Complex gain offset (real part)"),
            ("Q0", "", "Complex gain offset (imaginary part)"),
            ("Bifurcation", "", "Bifurcation status")
        ]
        
        self.nl_table.setRowCount(len(params))
        for i, (param, unit, desc) in enumerate(params):
            # Parameter name
            self.nl_table.setItem(i, 0, QtWidgets.QTableWidgetItem(f"{param} ({unit})" if unit else param))
            # Value (will be updated later)
            value_item = QtWidgets.QTableWidgetItem("N/A")
            self.nl_table.setItem(i, 1, value_item)
            # Description
            self.nl_table.setItem(i, 2, QtWidgets.QTableWidgetItem(desc))

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """Handle keyboard navigation between detectors and traces."""
        if event.key() == QtCore.Qt.Key.Key_Left:
            self._navigate_previous()
        elif event.key() == QtCore.Qt.Key.Key_Right:
            self._navigate_next()
        elif event.key() == QtCore.Qt.Key.Key_Up:
            self._navigate_previous_trace()
        elif event.key() == QtCore.Qt.Key.Key_Down:
            self._navigate_next_trace()
        else:
            super().keyPressEvent(event)
            
    def _toggle_mean_subtraction(self, state):
        """Toggles mean subtraction for TOD data and updates plots."""
        self.mean_subtract_enabled = (state == QtCore.Qt.CheckState.Checked.value)
        self._update_noise_plots()

    def _toggle_exp_binning(self, checked: bool):
        """Handles checkbox toggle to enable/disable exponential binning."""
        self.exp_binning_enabled = checked
        self._update_noise_plots()
    
    def _navigate_previous(self):
        """Navigate to the previous detector."""
        if not self.detector_indices or len(self.detector_indices) <= 1:
            return
        
        # Move to previous detector with wraparound
        self.current_detector_index_in_list = (self.current_detector_index_in_list - 1) % len(self.detector_indices)
        self._switch_to_detector(self.detector_indices[self.current_detector_index_in_list])
    
    def _navigate_next(self):
        """Navigate to the next detector."""
        if not self.detector_indices or len(self.detector_indices) <= 1:
            return
        
        # Move to next detector with wraparound
        self.current_detector_index_in_list = (self.current_detector_index_in_list + 1) % len(self.detector_indices)
        self._switch_to_detector(self.detector_indices[self.current_detector_index_in_list])
    
    def _navigate_previous_trace(self):
        """Navigate to the previous amplitude trace."""
        if not self.resonance_data_for_digest or len(self.resonance_data_for_digest) <= 1:
            return
        
        # Get sorted list of amplitude keys
        sorted_keys = sorted(
            self.resonance_data_for_digest.keys(),
            key=lambda k: float(k.split(":")[0])
        )
        
        # Find current index
        try:
            current_idx = sorted_keys.index(self.active_amplitude_raw_key)
            # Move to previous with wraparound
            new_idx = (current_idx - 1) % len(sorted_keys)
            self._switch_to_amplitude_trace(sorted_keys[new_idx])
        except ValueError:
            # If current key not found, select first
            if sorted_keys:
                self._switch_to_amplitude_trace(sorted_keys[0])
    
    def _navigate_next_trace(self):
        """Navigate to the next amplitude trace."""
        if not self.resonance_data_for_digest or len(self.resonance_data_for_digest) <= 1:
            return
        
        # Get sorted list of amplitude keys
        sorted_keys = sorted(
            self.resonance_data_for_digest.keys(),
            key=lambda k: float(k.split(":")[0])
        )
        
        # Find current index
        try:
            current_idx = sorted_keys.index(self.active_amplitude_raw_key)
            # Move to next with wraparound
            new_idx = (current_idx + 1) % len(sorted_keys)
            self._switch_to_amplitude_trace(sorted_keys[new_idx])
        except ValueError:
            # If current key not found, select first
            if sorted_keys:
                self._switch_to_amplitude_trace(sorted_keys[0])
    
    def _switch_to_amplitude_trace(self, amplitude_key: str):
        """Switch to a different amplitude trace."""
        if amplitude_key not in self.resonance_data_for_digest:
            return
        
        self.active_amplitude_raw_key = amplitude_key
        self.active_sweep_info = self.resonance_data_for_digest[amplitude_key]
        self.active_sweep_data = self.active_sweep_info['data']
        self.current_plot_offset_hz = self.active_sweep_info['actual_cf_hz']
        
        # Update trace hint label with current position
        if hasattr(self, 'trace_hint_label'):
            sorted_keys = sorted(
                self.resonance_data_for_digest.keys(),
                key=lambda k: float(k.split(":")[0])
            )
            try:
                current_idx = sorted_keys.index(amplitude_key)
                trace_hint_text = f"Trace {current_idx + 1}/{len(sorted_keys)} (↑↓ to switch)"
                self.trace_hint_label.setText(trace_hint_text)
            except ValueError:
                pass
        
        self._update_plots()
    
    def _switch_to_detector(self, new_detector_id: int):
        """Switch to display a different detector."""
        if new_detector_id not in self.all_detectors_data:
            return
        
        # Update detector-specific data
        detector_data = self.all_detectors_data[new_detector_id]
        self.detector_id = new_detector_id
        self.current_detector = self.detector_id
        self.resonance_data_for_digest = detector_data['resonance_data']
        self.resonance_frequency_ghz_title = detector_data['conceptual_freq_hz'] / 1e9
        
        # Reset active sweep selection
        self.active_amplitude_raw_key = None
        self.active_sweep_info = None
        self.active_sweep_data = None
        self.current_plot_offset_hz = None
        self.noise_i_data = None
        self.noise_q_data = None
        self.fast_freq = 1.22e6


        if self.spectrum_data:
            self.noise_tab_avail = True
            self.single_psd_i = self.spectrum_data['single_psd_i'][self.detector_id - 1] 
            self.single_psd_q = self.spectrum_data['single_psd_q'][self.detector_id - 1]
            self.tod_i = self.spectrum_data['I'][self.detector_id - 1]
            self.tod_q = self.spectrum_data['Q'][self.detector_id - 1]
            self.reference = self.spectrum_data['reference']
            self.tone_amp = self.spectrum_data['amplitudes_dbm'][self.detector_id - 1]
            self.slow_freq = self.spectrum_data['slow_freq_hz']
            if self.spectrum_data['pfb_enabled']:
                self.show_fast_tod = True
            
            if self.show_fast_tod:
                self.pfb_psd_i = self.spectrum_data['pfb_psd_i'][self.detector_id - 1]
                self.pfb_psd_q = self.spectrum_data['pfb_psd_q'][self.detector_id - 1]
                self.pfb_tod_i = self.spectrum_data['pfb_I'][self.detector_id - 1]
                self.pfb_tod_q = self.spectrum_data['pfb_Q'][self.detector_id - 1]
                self.pfb_freq_iq = self.spectrum_data['pfb_freq_iq'][self.detector_id - 1]
                self.pfb_freq_dsb = self.spectrum_data['pfb_freq_dsb'][self.detector_id - 1]
                self.pfb_dual_psd = self.spectrum_data['pfb_dual_psd'][self.detector_id - 1]
        if self.debug:
            self.debug_noise = self.full_debug[self.detector_id]
        # print("For detector", self.detector_id, "the noise is", self.debug_noise)
        
        if self.noise_data is not None:
            self.noise_i_data = self.noise_data.i[self.detector_id-1]
            self.noise_q_data = self.noise_data.q[self.detector_id-1]
        
        # Select the first sweep for this detector
        if self.resonance_data_for_digest:
            try:
                sorted_keys = sorted(
                    self.resonance_data_for_digest.keys(),
                    key=lambda k: float(k.split(":")[0])
                )
                if sorted_keys:
                    self.active_amplitude_raw_key = sorted_keys[0]
                    self.active_sweep_info = self.resonance_data_for_digest[self.active_amplitude_raw_key]
                    self.active_sweep_data = self.active_sweep_info['data']
                    self.current_plot_offset_hz = self.active_sweep_info['actual_cf_hz']
            except (ValueError, IndexError, KeyError) as e:
                print(f"Error selecting default sweep: {e}")
                if self.resonance_data_for_digest:
                    try:
                        first_key = next(iter(self.resonance_data_for_digest))
                        self.active_amplitude_raw_key = first_key
                        self.active_sweep_info = self.resonance_data_for_digest[self.active_amplitude_raw_key]
                        self.active_sweep_data = self.active_sweep_info['data']
                        self.current_plot_offset_hz = self.active_sweep_info.get('actual_cf_hz', self.resonance_frequency_ghz_title * 1e9)
                    except Exception as fallback_e:
                        print(f"Critical error in fallback default sweep selection: {fallback_e}")
        
        if self.current_plot_offset_hz is None:
            self.current_plot_offset_hz = self.resonance_frequency_ghz_title * 1e9
        
        # Update UI elements
        title_text = f"Detector {self.detector_id} ({self.resonance_frequency_ghz_title*1e3:.6f} MHz)"
        self.title_label.setText(title_text)
        self.setWindowTitle(f"Detector Digest: Detector {self.detector_id}  ({self.resonance_frequency_ghz_title*1e3:.6f} MHz)")
        
        # Update detector count label
        if hasattr(self, 'detector_count_label'):
            detector_count_text = f"({self.current_detector_index_in_list + 1} of {len(self.detector_indices)})"
            self.detector_count_label.setText(detector_count_text)
        
        if hasattr(self, "title_label_noise"):
            title_text = f"Detector {self.detector_id} ({self.resonance_frequency_ghz_title * 1e3:.6f} MHz)"
            self.title_label_noise.setText(title_text)

        if hasattr(self, "detector_count_label_noise") and self.detector_indices:
            count_text = f"({self.current_detector_index_in_list + 1} of {len(self.detector_indices)})"
            self.detector_count_label_noise.setText(count_text)
        
        # Redraw plots with new data
        
        self._update_plots()
        if self.spectrum_data:
            self._update_noise_plots()

    def _apply_zoom_box_mode_to_all(self):
        """Applies the current zoom_box_mode state to all plot viewboxes."""
        for plot_widget in [self.plot1_sweep_vs_freq, self.plot2_iq_plane, self.plot3_bias_opt]:
            if plot_widget and isinstance(plot_widget.getViewBox(), ClickableViewBox):
                plot_widget.getViewBox().enableZoomBoxMode(self.zoom_box_mode)

    def _clear_plots(self):
        """Clears all plot items and legends, and resets fitting info tables."""
        for plot_widget, legend_widget in [
            (self.plot1_sweep_vs_freq, self.plot1_legend),
            (self.plot2_iq_plane, self.plot2_legend),
            (self.plot3_bias_opt, self.plot3_legend)
        ]:
            if plot_widget:
                for item in plot_widget.listDataItems(): plot_widget.removeItem(item)
            if legend_widget: legend_widget.clear()
        
        # Reset table values to N/A
        if hasattr(self, 'skewed_table'):
            for row in range(self.skewed_table.rowCount()):
                if self.skewed_table.item(row, 1):
                    self.skewed_table.item(row, 1).setText("N/A")
        if hasattr(self, 'nl_table'):
            for row in range(self.nl_table.rowCount()):
                if self.nl_table.item(row, 1):
                    self.nl_table.item(row, 1).setText("N/A")


    def _get_relative_timestamps(self, ts, num_samples):
        SS_PER_SECOND = 156250000
        first = ts[0].h * 3600 + ts[0].m * 60 + ts[0].s + ts[0].ss/SS_PER_SECOND
        last = ts[-1].h * 3600 + ts[-1].m * 60 + ts[-1].s + ts[-1].ss/SS_PER_SECOND
        total_time = last - first
        timeseries = list(np.linspace(0, total_time, num_samples))
        return timeseries
    
    
    def _update_noise_plots(self):
        self.apply_theme(self.dark_mode)
    
        if not self.spectrum_data:
            self.plot_time_vs_mag.clear()
            self.plot_noise_spectrum.clear()
            return
    
        # try:
        exp_bins = 100
        self.plot_time_vs_mag.clear()
        self.plot_noise_spectrum.clear()
        if self.show_fast_tod:
            self.plot_fast_tod.clear()

        ###### First plot ######
        ts = self._get_relative_timestamps(self.spectrum_data['ts'], len(self.tod_i))
        if self.reference == "relative":
            tod_i_volts = convert_roc_to_volts(np.array(self.tod_i))
            tod_q_volts = convert_roc_to_volts(np.array(self.tod_q))
        else:
            tod_i_volts = np.array(self.tod_i)
            tod_q_volts = np.array(self.tod_q) ##### it is already converted to volts in spectrum_from_slow_tod_function for "absolute"

        if self.mean_subtract_enabled:
            tod_i_volts = tod_i_volts - np.mean(tod_i_volts)
            tod_q_volts = tod_q_volts - np.mean(tod_q_volts)

        complex_volts = tod_i_volts + 1j * tod_q_volts
        mag_volts = np.abs(complex_volts)

        self.plot_time_vs_mag.plot(ts, tod_i_volts, pen=pg.mkPen(IQ_COLORS["I"], width=LINE_WIDTH), name="I")
        self.plot_time_vs_mag.plot(ts, tod_q_volts, pen=pg.mkPen(IQ_COLORS["Q"], width=LINE_WIDTH), name="Q")
        self.plot_time_vs_mag.autoRange()

        if self.show_fast_tod:
            pfb_ts = self.spectrum_data['pfb_ts']

            tod_pfb_i_volts = convert_roc_to_volts(np.array(self.pfb_tod_i))
            tod_pfb_q_volts = convert_roc_to_volts(np.array(self.pfb_tod_q))

            if self.mean_subtract_enabled:
                tod_pfb_i_volts = tod_pfb_i_volts - np.mean(tod_pfb_i_volts)
                tod_pfb_q_volts = tod_pfb_q_volts - np.mean(tod_pfb_q_volts)


            self.plot_fast_tod.plot(pfb_ts, tod_pfb_i_volts, pen=pg.mkPen(IQ_COLORS["I"], width=LINE_WIDTH), name="PFB I")
            self.plot_fast_tod.plot(pfb_ts, tod_pfb_q_volts, pen=pg.mkPen(IQ_COLORS["Q"], width=LINE_WIDTH), name="PFB Q")
            self.plot_fast_tod.autoRange()

        ###### Second plot ########
        frequencies = self.spectrum_data['freq_iq']
        if self.mean_subtract_enabled: ### Removing DC bin
            psd_i = self.single_psd_i[3:]
            psd_q = self.single_psd_q[3:]
            freq = frequencies[3:]
        else:
            psd_i = self.single_psd_i[3:]
            psd_q = self.single_psd_q[3:]
            freq = frequencies[3:]


        #### pfb_data ####

        if self.show_fast_tod:
            overlap = self.spectrum_data['overlap'] #### Overlapping points with the slow spectrum, just for plotting
            if overlap < 2: ### Incase the calculation doesn't pick up any overlapping samples and also to remove the dc bin
                overlap = 2
            pfb_psd_i = self.pfb_psd_i[overlap:]
            pfb_psd_q = self.pfb_psd_q[overlap:]
            pfb_freq = self.pfb_freq_iq[overlap:]

            ##### For the magnitude part ####
            freq_dsb = np.array(self.pfb_freq_dsb)
            psd_dsb = np.array(self.pfb_dual_psd)
            min_f_abs = abs(np.min(freq_dsb))
            max_f_abs = abs(np.max(freq_dsb))
            
            if max_f_abs >= min_f_abs:
                # Keep positive side
                mask = freq_dsb >= 0
                freq_sel = freq_dsb[mask]
                psd_sel  = psd_dsb[mask]
                h_label = "(I-Q)/2"
            else:
                # Keep negative side, convert to positive
                mask = freq_dsb <= 0
                freq_sel = np.abs(freq_dsb[mask])
                psd_sel  = psd_dsb[mask]
                h_label = "(I+Q)/2"
            
            # Sort ascending
            sort_idx = np.argsort(freq_sel)
            freq_sel = freq_sel[sort_idx]
            psd_sel  = psd_sel[sort_idx]

            freq_sel = freq_sel[2:] ### Removing DC bin info
            psd_sel = psd_sel[2:]

            
        else:
            pfb_freq = []
            pfb_psd_i = []
            pfb_psd_q = []
            psd_sel = []
            freq_sel = []


        full_freq = list(freq) + list(pfb_freq)
        full_psd_i = list(psd_i) + list(pfb_psd_i)
        full_psd_q = list(psd_q) + list(pfb_psd_q)
        full_lin_comb = list(psd_sel)
        freq_lin_comb = list(freq_sel)

        
        log_freqs = np.log10(np.clip(full_freq, 1e-12, None))
        
        amplitude = self.tone_amp #### already in dbm
        slow_freq = self.slow_freq
        fast_freq = self.fast_freq/1e6
        
        # Create a text box to show the values
        summary_text = (
            f"<span style='font-size:12pt; color:green;'>"
            f"Tone Amp: {amplitude:.3f} dBm <br>"
            f"Slow Bandwidth: {slow_freq:.3f} Hz <br>"
            f"Fast Bandwidth: {fast_freq:.2f} MHz </span>"
        )
        
        self.summary_label = pg.TextItem(html=summary_text, anchor=(0, 0), border='w', fill=(0, 0, 0, 150))
        self.plot_noise_spectrum.addItem(self.summary_label)
        self.summary_label.setPos(np.max(log_freqs) * 0.8, np.max(psd_i) * 0.9)
        
        if self.exp_binning_enabled:

            f_bin_i, psd_i_bin = exp_bin_noise_data(freq, psd_i, exp_bins) #### fixing bins to 50
            f_bin_q, psd_q_bin = exp_bin_noise_data(freq, psd_q, exp_bins)

            
            curve_i = self.plot_noise_spectrum.plot(f_bin_i, psd_i_bin,
                                                    pen=pg.mkPen(IQ_COLORS["I"], width=LINE_WIDTH), name="I")
            curve_q = self.plot_noise_spectrum.plot(f_bin_q, psd_q_bin,
                                                    pen=pg.mkPen(IQ_COLORS["Q"], width=LINE_WIDTH), name="Q")
    
            if self.show_fast_tod:
                # --- New PFB curves ---
                pfb_f_bin_i, pfb_psd_i_bin = exp_bin_noise_data(pfb_freq, pfb_psd_i, exp_bins) #### fixing bins to 100
                pfb_f_bin_q, pfb_psd_q_bin = exp_bin_noise_data(pfb_freq, pfb_psd_q, exp_bins)
                pfb_f_bin_mag, pfb_psd_mag_bin = exp_bin_noise_data(freq_sel, psd_sel, exp_bins)

                
                curve_pfb_i = self.plot_noise_spectrum.plot(pfb_f_bin_i, pfb_psd_i_bin,
                                                            pen=pg.mkPen(IQ_COLORS["I"], width=LINE_WIDTH, style=QtCore.Qt.DashLine),
                                                            name="PFB I")
                curve_pfb_q = self.plot_noise_spectrum.plot(pfb_f_bin_q, pfb_psd_q_bin,
                                                            pen=pg.mkPen(IQ_COLORS["Q"], width=LINE_WIDTH, style=QtCore.Qt.DashLine),
                                                            name="PFB Q")
                curve_pfb_mag = self.plot_noise_spectrum.plot(pfb_f_bin_mag, pfb_psd_mag_bin,
                                                              pen=pg.mkPen(color = (255, 0, 0, 100), width=LINE_WIDTH, style=QtCore.Qt.DashLine),
                                                              name="wideband linear combination (I,Q)")
            
            self.plot_noise_spectrum.setLogMode(x=True, y=False)
            self.plot_noise_spectrum.autoRange()

        else:
            curve_i = self.plot_noise_spectrum.plot(freq, psd_i,
                                                    pen=pg.mkPen(IQ_COLORS["I"], width=LINE_WIDTH), name="I")
            curve_q = self.plot_noise_spectrum.plot(freq, psd_q,
                                                    pen=pg.mkPen(IQ_COLORS["Q"], width=LINE_WIDTH), name="Q")

    
            if self.show_fast_tod:
                # --- New PFB curves ---
                curve_pfb_i = self.plot_noise_spectrum.plot(pfb_freq, pfb_psd_i,
                                                            pen=pg.mkPen(IQ_COLORS["I"], width=LINE_WIDTH, style=QtCore.Qt.DashLine),
                                                            name="PFB I")
                curve_pfb_q = self.plot_noise_spectrum.plot(pfb_freq, pfb_psd_q,
                                                            pen=pg.mkPen(IQ_COLORS["Q"], width=LINE_WIDTH, style=QtCore.Qt.DashLine),
                                                            name="PFB Q")

                curve_pfb_mag = self.plot_noise_spectrum.plot(freq_sel, psd_sel,
                                                              pen=pg.mkPen(color = (255, 0, 0, 100), width=LINE_WIDTH, style=QtCore.Qt.DashLine),
                                                              name="wideband linear combination (I,Q)")
            
            self.plot_noise_spectrum.setLogMode(x=True, y=False)
            self.plot_noise_spectrum.autoRange()

        # --- Hover label setup ---
        vb = self.plot_noise_spectrum.getViewBox()
        self.hover_label = pg.TextItem("", anchor=(0, 1), color='w')
        self.plot_noise_spectrum.addItem(self.hover_label)
        self.hover_label.hide()


        # --- Mouse move handler ---
        def on_mouse_move(evt):
            pos = evt[0]  # current mouse position in scene coordinates
            if self.plot_noise_spectrum.sceneBoundingRect().contains(pos):
                mouse_point = vb.mapSceneToView(pos)
                
                log_x = mouse_point.x()
                x = 10 ** log_x

                summary_rect = self.summary_label.boundingRect()
                summary_rect = self.summary_label.mapRectToScene(summary_rect)
                if summary_rect.contains(pos):
                    # Mouse is over text box — disable hover update
                    return
                
                if self.exp_binning_enabled:
                    # Use the binned frequency and PSD data
                    freq_i = np.log10(np.clip(np.concatenate((f_bin_i, pfb_f_bin_i if self.show_fast_tod else [])), 1e-12, None))
                    psd_i_vals = np.concatenate((psd_i_bin, pfb_psd_i_bin if self.show_fast_tod else []))
                    psd_q_vals = np.concatenate((psd_q_bin, pfb_psd_q_bin if self.show_fast_tod else []))
                    psd_dual_vals = pfb_psd_mag_bin
                    freq_l = pfb_f_bin_mag
                else:
                    # Use the original data
                    freq_i = log_freqs
                    psd_i_vals = full_psd_i
                    psd_q_vals = full_psd_q
                    psd_dual_vals = full_lin_comb
                    freq_l = freq_lin_comb


                if self.show_fast_tod:
                    if x > max(freq_l):
                        self.hover_label.hide()
                        return

                        
                if self.show_fast_tod:
                    if log_x < max(freq_i):
                        y_i = np.interp(log_x, freq_i, psd_i_vals)
                        y_q = np.interp(log_x, freq_i, psd_q_vals)
                        y_d = np.interp(log_x, freq_l, psd_dual_vals)
                        self.hover_label.setHtml(
                            f"<span style='color:{IQ_COLORS['I']}'>I: {y_i:.3f}</span><br>"
                            f"<span style='color:{IQ_COLORS['Q']}'>Q: {y_q:.3f}</span><br>"
                            f"<span style='color:red'>{h_label}: {y_d:.3f}</span><br>"
                            f"<span style='color:yellow'>Freq: {x:.3f}</span>"
                        )
                        self.hover_label.setPos(log_x * 0.9, max(y_i, y_q, y_d)*0.9)
                    else:
                        y_d = np.interp(log_x, freq_l, psd_dual_vals)
                        self.hover_label.setHtml(
                            f"<span style='color:red'>{h_label}: {y_d:.3f}</span><br>"
                            f"<span style='color:yellow'> Freq: {x:.3f}</span>"
                        )
                        self.hover_label.setPos(log_x * 0.9, y_d * 0.9)
                else:
                    if log_x < max(freq_i):      
                        y_i = np.interp(log_x, freq_i, psd_i_vals)
                        y_q = np.interp(log_x, freq_i, psd_q_vals)
                        self.hover_label.setHtml(
                            f"<span style='color:{IQ_COLORS['I']}'>I: {y_i:.3f}</span><br>"
                            f"<span style='color:{IQ_COLORS['Q']}'>Q: {y_q:.3f}</span><br>"
                            f"<span style='color:yellow'>Freq: {x:.3f}</span>"
                        )
                        self.hover_label.setPos(log_x * 0.9, max(y_i, y_q)*0.9)
                        
                self.hover_label.show() 
            else:
                self.hover_label.hide()

        self.proxy = pg.SignalProxy(self.plot_noise_spectrum.scene().sigMouseMoved,
                                    rateLimit=60, slot=on_mouse_move)
        
        # except Exception as e:
        #     print("Error updating noise plots:", e)      
        
    def _update_plots(self):
        """Populates all three plots with data and updates fitting information panel."""
        self._clear_plots() # Clear previous data and fitting info
        if not self.active_sweep_data or not self.resonance_data_for_digest or self.active_sweep_info is None:
            # If no active sweep data, ensure fitting panel also shows "N/A" or "Not Applied"
            self._update_fitting_info_panel() # Call to set labels to default state
            return

        active_power_dbm_str = "N/A"; bifurcation_indicator = ""
        active_amp_val_for_conversion = None

        if self.active_amplitude_raw_key is not None:
            amp_part_str = self.active_amplitude_raw_key.split(":")[0]
            try: active_amp_val_for_conversion = float(amp_part_str)
            except ValueError: 
                print(f"Warning: Could not parse amplitude from key '{self.active_amplitude_raw_key}'")
                pass

            if active_amp_val_for_conversion is not None:
                dac_scale = self.dac_scales.get(self.target_module)
                if dac_scale is not None:
                    try:
                        active_power_dbm = UnitConverter.normalize_to_dbm(active_amp_val_for_conversion, dac_scale)
                        active_power_dbm_str = f"{active_power_dbm:.2f} dBm"
                    except Exception: pass
        
        if self.active_sweep_data.get('is_bifurcated', False): bifurcation_indicator = " (bifurcated)"
        
        plot_pen_color = "w" if self.dark_mode else "k"
        self.plot1_sweep_vs_freq.setTitle(f"Sweep (Probe Power {active_power_dbm_str}{bifurcation_indicator})", color=plot_pen_color)
        self.plot2_iq_plane.setTitle(f"IQ (Probe Power {active_power_dbm_str}{bifurcation_indicator})", color=plot_pen_color)

        if self.current_plot_offset_hz is not None:
            freqs_hz_active = self.active_sweep_data.get('frequencies')
            iq_complex_active = self.active_sweep_data.get('iq_complex')

            if freqs_hz_active is not None and iq_complex_active is not None and len(freqs_hz_active) > 0:
                s21_mag_volts = convert_roc_to_volts(np.abs(iq_complex_active))
                s21_i_volts = convert_roc_to_volts(iq_complex_active.real)
                s21_q_volts = convert_roc_to_volts(iq_complex_active.imag)
                x_axis_hz_offset = freqs_hz_active - self.current_plot_offset_hz

                # Plot raw data
                # Fix magnitude color - ensure it's a valid color object
                if IQ_COLORS.get("MAGNITUDE") is None:
                    raw_mag_color = pg.mkColor(plot_pen_color)  # Ensure it's a valid color object
                else:
                    raw_mag_color = IQ_COLORS["MAGNITUDE"]
                self.plot1_sweep_vs_freq.plot(x_axis_hz_offset, s21_mag_volts, pen=pg.mkPen(raw_mag_color, width=LINE_WIDTH), name="|I+jQ| (Raw)")
                self.plot1_sweep_vs_freq.plot(x_axis_hz_offset, s21_i_volts, pen=pg.mkPen(IQ_COLORS["I"], style=QtCore.Qt.PenStyle.DashLine, width=LINE_WIDTH), name="I (Raw)")
                self.plot1_sweep_vs_freq.plot(x_axis_hz_offset, s21_q_volts, pen=pg.mkPen(IQ_COLORS["Q"], width=LINE_WIDTH), name="Q (Raw)")
                self.plot2_iq_plane.plot(s21_i_volts, s21_q_volts, pen=None, symbol='o', symbolBrush=SCATTER_COLORS["DEFAULT"], symbolPen=SCATTER_COLORS["DEFAULT"], symbolSize=5, name="Sweep IQ (Raw)")

                # Skewed Fit Overlay (magnitude only)
                if self.active_sweep_data.get('skewed_fit_success'):
                    skewed_model_mag = self.active_sweep_data.get('skewed_model_mag')
                    if skewed_model_mag is not None and len(skewed_model_mag) == len(x_axis_hz_offset):
                        # The skewed model is normalized, we need to denormalize it
                        # The normalization factor used in fitting is abs(s21_iq[-1])
                        # We can calculate this from the raw data
                        normalization_factor = 1.0
                        if len(iq_complex_active) > 0:
                            # Use the last point as the normalization reference (off-resonance baseline)
                            normalization_factor = np.abs(iq_complex_active[-1])
                            if normalization_factor < 1e-15:
                                # Fallback if last point is too small
                                normalization_factor = np.mean(np.abs(iq_complex_active[-10:]))
                        
                        # Apply the normalization factor to get back to physical scale
                        skewed_model_mag_physical = skewed_model_mag * normalization_factor
                        
                        # Convert from counts to volts
                        skewed_mag_volts = convert_roc_to_volts(skewed_model_mag_physical)
                        self.plot1_sweep_vs_freq.plot(x_axis_hz_offset, skewed_mag_volts, pen=pg.mkPen(FITTING_COLORS["SKEWED"], width=LINE_WIDTH, style=QtCore.Qt.PenStyle.DotLine), name="Skewed Fit Mag")
                        # No IQ plane plot for skewed fit since it's magnitude-only
                
                # Nonlinear Fit Overlay
                if self.active_sweep_data.get('nonlinear_fit_success'):
                    nl_model_iq = self.active_sweep_data.get('nonlinear_model_iq') 
                    if nl_model_iq is not None and len(nl_model_iq) == len(x_axis_hz_offset):
                        # The model is generated for gain-corrected data, so we need to re-apply the gain
                        # to match the physical reference frame of the displayed data
                        gain_complex = self.active_sweep_data.get('gain_complex')
                        if gain_complex is not None:
                            # Apply the complex gain (both magnitude and phase)
                            nl_model_iq_physical = nl_model_iq * gain_complex
                        else:
                            # No gain information available, use model as-is
                            nl_model_iq_physical = nl_model_iq
                        
                        # Convert from counts to volts
                        nl_mag_volts = convert_roc_to_volts(np.abs(nl_model_iq_physical))
                        nl_i_volts = convert_roc_to_volts(nl_model_iq_physical.real)
                        nl_q_volts = convert_roc_to_volts(nl_model_iq_physical.imag)
                        self.plot1_sweep_vs_freq.plot(x_axis_hz_offset, nl_mag_volts, pen=pg.mkPen(FITTING_COLORS["NONLINEAR"], width=LINE_WIDTH, style=QtCore.Qt.PenStyle.DashDotLine), name="Nonlinear Fit Mag")
                        self.plot2_iq_plane.plot(nl_i_volts, nl_q_volts, pen=pg.mkPen(FITTING_COLORS["NONLINEAR"], width=LINE_WIDTH, style=QtCore.Qt.PenStyle.DashDotLine), name="Nonlinear Fit IQ")
                self.plot1_sweep_vs_freq.autoRange()

            rotation_tod_iq = self.active_sweep_data.get('rotation_tod')

            if rotation_tod_iq is not None and rotation_tod_iq.size > 0:
                tod_i_volts = convert_roc_to_volts(rotation_tod_iq.real)
                tod_q_volts = convert_roc_to_volts(rotation_tod_iq.imag)

                mean_phase_file = np.median(np.arctan(tod_q_volts/tod_i_volts))
                mean_mag_file = np.mean(np.sqrt(tod_i_volts**2 + tod_q_volts**2))
                # print("Median phase of the rotation data in file is", np.degrees(mean_phase_file), "degrees")
                # print("Mean magnitude of the rotation data in file is", mean_mag_file)

                noise_color = 'w' if self.dark_mode else 'k' 
                self.plot2_iq_plane.plot(tod_i_volts, tod_q_volts, pen=None, symbol='o', symbolBrush=noise_color, symbolPen=noise_color, symbolSize=3, name="Noise at f_bias")
                
            if self.debug:
                test_colors = ['orange', 'y']
                test_labels = ['Initial Noise', 'Refined Noise']
                for idx, (key, noise) in enumerate(self.debug_noise.items()):
                    noise_i = np.array(self.debug_noise[key].real)
                    noise_q = np.array(self.debug_noise[key].imag)
                    
                    noise_i_v = convert_roc_to_volts(noise_i)
                    noise_q_v = convert_roc_to_volts(noise_q)
    
                    test_color = test_colors[idx % len(test_colors)]
                    test_label = test_labels[idx % len(test_labels)]
    
                    self.plot2_iq_plane.plot(
                        noise_i_v,
                        noise_q_v,
                        pen=None,
                        symbol='o',
                        symbolBrush=test_color,
                        symbolPen=test_color,
                        symbolSize=3,
                        name=test_label
                    )
            
            if self.noise_i_data is not None and self.noise_q_data is not None:
                rotation_noise_i = np.array(self.noise_i_data)
                rotation_noise_q = np.array(self.noise_q_data)
                
                noise_i_volts = convert_roc_to_volts(rotation_noise_i)
                noise_q_volts = convert_roc_to_volts(rotation_noise_q)

                mean_phase_noise = np.median(np.arctan(noise_q_volts/noise_i_volts))
                mean_mag_noise = np.mean(np.sqrt(noise_i_volts**2 + noise_q_volts**2))
                
                # print("Median phase of the noise data collected is", np.degrees(mean_phase_noise), "degrees\n")
                # print("Mean magnitude of the noise data", mean_mag_noise)

                self.plot2_iq_plane.plot(
                    noise_i_volts,
                    noise_q_volts,
                    pen=None,
                    symbol='o',
                    symbolBrush='r',
                    symbolPen='r',
                    symbolSize=3,
                    name="Data after Bias sampling"
                )
                
            self.plot2_iq_plane.autoRange()

            
        
        # --- Plot 3: Bias amplitude optimization (remains largely the same logic) ---
        if self.current_plot_offset_hz is not None:
            data_by_amplitude = {} 
            for key, sweep_info_entry in self.resonance_data_for_digest.items():
                amp_part_str = key.split(":")[0]
                try: amp_val_float = float(amp_part_str)
                except ValueError: continue # Skip if key format is unexpected
                
                direction = sweep_info_entry.get('direction', 'upward') # Default if not present
                
                if amp_val_float not in data_by_amplitude: data_by_amplitude[amp_val_float] = []
                data_by_amplitude[amp_val_float].append({
                    'sweep_info': sweep_info_entry, 
                    'direction': direction, 
                    'key': key # Original key "amp:direction"
                })
            
            sorted_amplitudes_float = sorted(data_by_amplitude.keys())
            num_unique_amps = len(sorted_amplitudes_float)
            
            for amp_idx, amp_val_float in enumerate(sorted_amplitudes_float):
                color = DISTINCT_PLOT_COLORS[amp_idx % len(DISTINCT_PLOT_COLORS)] if num_unique_amps <= 3 else pg.colormap.get(COLORMAP_CHOICES["AMPLITUDE_SWEEP"]).map( (0.3 + (amp_idx / max(1, num_unique_amps - 1)) * 0.7) if self.dark_mode else (amp_idx / max(1, num_unique_amps - 1)) * 0.9 )
                for data_entry in data_by_amplitude[amp_val_float]:
                    sweep_data = data_entry['sweep_info']['data']
                    direction = data_entry['direction']
                    freqs_hz, iq_complex = sweep_data.get('frequencies'), sweep_data.get('iq_complex')
                    if freqs_hz is None or iq_complex is None or len(freqs_hz) == 0: continue
                    s21_mag_db = convert_roc_to_dbm(np.abs(iq_complex))
                    if self.normalize_plot3 and len(s21_mag_db) > 0:
                        ref_val = s21_mag_db[0]
                        if np.isfinite(ref_val): s21_mag_db -= ref_val
                    x_axis_hz_offset = freqs_hz - self.current_plot_offset_hz
                    is_bifurcated = sweep_data.get('is_bifurcated', False)
                    line_style = DOWNWARD_SWEEP_STYLE if direction == "downward" else UPWARD_SWEEP_STYLE
                    # Create a fresh color for each line to avoid alpha bleed-through
                    if is_bifurcated:
                        # For bifurcated sweeps, create a new color with reduced alpha
                        base_color = pg.mkColor(color)
                        r, g, b, _ = base_color.getRgb()
                        pen_color_plot3 = pg.mkColor(r, g, b, 128)
                    else:
                        # For non-bifurcated sweeps, use full alpha
                        pen_color_plot3 = pg.mkColor(color)
                    current_pen = pg.mkPen(pen_color_plot3, width=LINE_WIDTH, style=line_style)
                    dac_scale = self.dac_scales.get(self.target_module)
                    legend_name = f"{amp_val_float:.2e} Norm"
                    if dac_scale:
                        try: legend_name = f"{UnitConverter.normalize_to_dbm(amp_val_float, dac_scale):.2f} dBm"
                        except: pass
                    legend_name += " (Down)" if direction == "downward" else " (Up)"
                    if is_bifurcated: legend_name += " (bifurcated)"
                    self.plot3_bias_opt.plot(x_axis_hz_offset, s21_mag_db, pen=current_pen, name=legend_name)
        self.plot3_bias_opt.autoRange()
        self._update_fitting_info_panel()



    def _update_fitting_info_panel(self):
        """Updates the fitting information panel based on self.active_sweep_data."""
        if not self.active_sweep_data or not hasattr(self, 'skewed_table') or not hasattr(self, 'nl_table'):
            # Reset all table values to N/A
            if hasattr(self, 'skewed_table'):
                for row in range(self.skewed_table.rowCount()):
                    if self.skewed_table.item(row, 1):
                        self.skewed_table.item(row, 1).setText("N/A")
            if hasattr(self, 'nl_table'):
                for row in range(self.nl_table.rowCount()):
                    if self.nl_table.item(row, 1):
                        self.nl_table.item(row, 1).setText("N/A")
            # Hide fitting panel if no data
            self.fitting_info_group.setVisible(False)
            return

        # Skewed Fit - Update table rows
        skewed_applied = self.active_sweep_data.get('skewed_fit_applied', False)
        skewed_success = self.active_sweep_data.get('skewed_fit_success', False)
        fit_params = self.active_sweep_data.get('fit_params', {})

        # Row indices for skewed table
        SKEWED_STATUS_ROW = 0
        SKEWED_FR_ROW = 1
        SKEWED_QR_ROW = 2
        SKEWED_QC_ROW = 3
        SKEWED_QI_ROW = 4
        SKEWED_BIFURCATION_ROW = 5

        if skewed_applied:
            self.skewed_table.item(SKEWED_STATUS_ROW, 1).setText("Success" if skewed_success else "Failed")
            if skewed_success and fit_params:
                self.skewed_table.item(SKEWED_FR_ROW, 1).setText(f"{fit_params.get('fr', 0) / 1e6:.6f}")
                self.skewed_table.item(SKEWED_QR_ROW, 1).setText(f"{fit_params.get('Qr', 0):,.1f}")
                self.skewed_table.item(SKEWED_QC_ROW, 1).setText(f"{fit_params.get('Qc', 0):,.1f}")
                self.skewed_table.item(SKEWED_QI_ROW, 1).setText(f"{fit_params.get('Qi', 0):,.1f}")
            else:
                for row in [SKEWED_FR_ROW, SKEWED_QR_ROW, SKEWED_QC_ROW, SKEWED_QI_ROW]:
                    self.skewed_table.item(row, 1).setText("N/A")
        else: 
            self.skewed_table.item(SKEWED_STATUS_ROW, 1).setText("Not Applied")
            for row in [SKEWED_FR_ROW, SKEWED_QR_ROW, SKEWED_QC_ROW, SKEWED_QI_ROW]:
                self.skewed_table.item(row, 1).setText("-")
        
        # Update bifurcation status in skewed table using the existing is_bifurcated field
        # This is the same field used for the plot legends
        is_bifurcated = self.active_sweep_data.get('is_bifurcated', False)
        self.skewed_table.item(SKEWED_BIFURCATION_ROW, 1).setText("Yes" if is_bifurcated else "No")

        # Nonlinear Fit - Update table rows
        nl_applied = self.active_sweep_data.get('nonlinear_fit_applied', False)
        nl_success = self.active_sweep_data.get('nonlinear_fit_success', False)
        nl_params_dict = self.active_sweep_data.get('nonlinear_fit_params', self.active_sweep_data)

        # Row indices for nonlinear table
        NL_STATUS_ROW = 0
        NL_FR_ROW = 1
        NL_QR_ROW = 2
        NL_QC_ROW = 3
        NL_QI_ROW = 4
        NL_A_ROW = 5
        NL_PHI_ROW = 6
        NL_I0_ROW = 7
        NL_Q0_ROW = 8
        NL_BIFURCATION_ROW = 9

        if nl_applied:
            self.nl_table.item(NL_STATUS_ROW, 1).setText("Success" if nl_success else "Failed")
            if nl_success:
                nl_fr_val = nl_params_dict.get('fr_nl', nl_params_dict.get('fr', 0)) 
                self.nl_table.item(NL_FR_ROW, 1).setText(f"{nl_fr_val / 1e6:.6f}")
                self.nl_table.item(NL_QR_ROW, 1).setText(f"{nl_params_dict.get('Qr_nl', nl_params_dict.get('Qr',0)):,.1f}")
                self.nl_table.item(NL_QC_ROW, 1).setText(f"{nl_params_dict.get('Qc_nl', nl_params_dict.get('Qc',0)):,.1f}")
                self.nl_table.item(NL_QI_ROW, 1).setText(f"{nl_params_dict.get('Qi_nl', nl_params_dict.get('Qi',0)):,.1f}")
                
                # Nonlinearity parameter 'a' with special formatting and color
                a_value = nl_params_dict.get('a', 0)
                a_item = self.nl_table.item(NL_A_ROW, 1)
                a_item.setText(f"{a_value:.3e}")
                
                # Bifurcation threshold is 4*sqrt(3)/9 ≈ 0.77
                bifurcation_threshold = 4 * np.sqrt(3) / 9
                if a_value > bifurcation_threshold:
                    a_item.setForeground(QtGui.QBrush(QtGui.QColor("red")))
                    self.nl_table.item(NL_BIFURCATION_ROW, 1).setText("Yes")
                else:
                    # Reset to default color (based on dark mode)
                    default_color = QtGui.QColor("white" if self.dark_mode else "black")
                    a_item.setForeground(QtGui.QBrush(default_color))
                    self.nl_table.item(NL_BIFURCATION_ROW, 1).setText("No")
                
                self.nl_table.item(NL_PHI_ROW, 1).setText(f"{np.degrees(nl_params_dict.get('phi', 0)):.2f}")
                self.nl_table.item(NL_I0_ROW, 1).setText(f"{nl_params_dict.get('i0', 0):.3e}")
                self.nl_table.item(NL_Q0_ROW, 1).setText(f"{nl_params_dict.get('q0', 0):.3e}")
            else:
                for row in range(NL_FR_ROW, NL_BIFURCATION_ROW + 1):
                    self.nl_table.item(row, 1).setText("N/A")
        else: 
            self.nl_table.item(NL_STATUS_ROW, 1).setText("Not Applied")
            for row in range(NL_FR_ROW, NL_BIFURCATION_ROW + 1):
                self.nl_table.item(row, 1).setText("-")

        # Show/hide fitting panel based on whether any fits were applied
        any_fits_applied = skewed_applied or nl_applied
        self.fitting_info_group.setVisible(any_fits_applied)

    @QtCore.pyqtSlot(object)
    def _handle_plot3_double_click(self, ev):
        """Handles double-click on Plot 3 to change the active sweep."""
        view_box = self.plot3_bias_opt.getViewBox()
        if not view_box or not self.resonance_data_for_digest or self.current_plot_offset_hz is None: return

        mouse_point = view_box.mapSceneToView(ev.scenePos())
        x_coord_on_plot, y_coord_on_plot = mouse_point.x(), mouse_point.y()
        min_dist_sq, closest_amp_key = np.inf, None

        # Sort keys to ensure consistent behavior if multiple points are equidistant
        # Sorting by the float value of the amplitude part of the key
        sorted_keys = sorted(
            self.resonance_data_for_digest.keys(),
            key=lambda k: float(k.split(":")[0]) # Sort by amplitude part
        )

        for amp_key in sorted_keys: 
            sweep_info = self.resonance_data_for_digest[amp_key]
            sweep_data = sweep_info['data']
            freqs_hz, iq_complex = sweep_data.get('frequencies'), sweep_data.get('iq_complex')

            if freqs_hz is not None and iq_complex is not None and len(freqs_hz) > 0:
                s21_mag_db_curve = convert_roc_to_dbm(np.abs(iq_complex))
                if self.normalize_plot3 and len(s21_mag_db_curve) > 0:
                    ref_val = s21_mag_db_curve[0]
                    if np.isfinite(ref_val): s21_mag_db_curve -= ref_val
                
                x_axis_for_this_curve = freqs_hz - self.current_plot_offset_hz
                if not (x_axis_for_this_curve.min() <= x_coord_on_plot <= x_axis_for_this_curve.max()):
                    continue
                
                idx = np.abs(x_axis_for_this_curve - x_coord_on_plot).argmin()
                dist_sq = (x_axis_for_this_curve[idx] - x_coord_on_plot)**2 + (s21_mag_db_curve[idx] - y_coord_on_plot)**2
                if dist_sq < min_dist_sq:
                    min_dist_sq, closest_amp_key = dist_sq, amp_key
        
        if closest_amp_key is not None:
            self.active_amplitude_raw_key = closest_amp_key 
            self.active_sweep_info = self.resonance_data_for_digest[self.active_amplitude_raw_key]
            self.active_sweep_data = self.active_sweep_info['data']
            self.current_plot_offset_hz = self.active_sweep_info['actual_cf_hz']
            self._update_plots() 
            ev.accept()
            
    def apply_theme(self, dark_mode: bool):
        """Apply the dark/light theme to all plots and UI elements in this window."""
        self.dark_mode = dark_mode
        central_widget = self.centralWidget()
        if central_widget:
            bg_color_hex = "#1C1C1C" if dark_mode else "#FFFFFF"
            # Apply to central widget and ensure children inherit or are styled explicitly
            central_widget.setStyleSheet(f"QWidget {{ background-color: {bg_color_hex}; }}")
        
        title_color_str = "white" if dark_mode else "black"
        plot_bg_color, plot_pen_color = ("k", "w") if dark_mode else ("w", "k")
    
        # ----- Titles -----
        if hasattr(self, 'title_label'):
            self.title_label.setStyleSheet(
                f"QLabel {{ margin-bottom: 10px; color: {title_color_str}; background-color: transparent; }}"
            )
        if hasattr(self, 'title_label_noise'):
            self.title_label_noise.setStyleSheet(
                f"QLabel {{ margin-bottom: 10px; color: {title_color_str}; background-color: transparent; }}"
            )
    
        # ----- Plots (Digest tab) -----
        plot_widgets_legends = [
            (self.plot1_sweep_vs_freq, self.plot1_legend, "Sweep"),
            (self.plot2_iq_plane, self.plot2_legend, "IQ"),
            (self.plot3_bias_opt, self.plot3_legend, "Bias amplitude optimization"),
        ]
        for plot_widget, legend_widget, default_title_text in plot_widgets_legends:
            if plot_widget:
                plot_widget.setBackground(plot_bg_color)
                plot_item = plot_widget.getPlotItem()
                if plot_item:
                    if plot_widget == self.plot3_bias_opt:
                        current_title = (
                            plot_item.titleLabel.text
                            if plot_item.titleLabel and plot_item.titleLabel.text
                            else default_title_text
                        )
                        plot_item.setTitle(current_title, color=plot_pen_color)
                    for axis_name in ("left", "bottom", "right", "top"):
                        ax = plot_item.getAxis(axis_name)
                        if ax:
                            ax.setPen(plot_pen_color)
                            ax.setTextPen(plot_pen_color)
                if legend_widget:
                    try:
                        legend_widget.setLabelTextColor(plot_pen_color)
                    except Exception as e:
                        print(f"Error updating legend color for {default_title_text}: {e}")
    
        # ----- Plots (Noise tab) -----
        if hasattr(self, 'plot_time_vs_mag'):
            self.plot_time_vs_mag.setBackground(plot_bg_color)
            plot_item = self.plot_time_vs_mag.getPlotItem()
            plot_item.setTitle("TOD", color=plot_pen_color)
            for ax in ("left", "bottom"):
                axis = plot_item.getAxis(ax)
                axis.setPen(plot_pen_color)
                axis.setTextPen(plot_pen_color)
        if hasattr(self, 'plot_noise_spectrum'):
            self.plot_noise_spectrum.setBackground(plot_bg_color)
            plot_item = self.plot_noise_spectrum.getPlotItem()
            plot_item.setTitle("Noise Spectrum", color=plot_pen_color)
            for ax in ("left", "bottom"):
                axis = plot_item.getAxis(ax)
                axis.setPen(plot_pen_color)
                axis.setTextPen(plot_pen_color)
    
        # ----- Fitting info panel -----
        if hasattr(self, 'fitting_info_group'):
            self.fitting_info_group.setStyleSheet(
                f"QGroupBox {{ color: {title_color_str}; border: 1px solid {title_color_str}; margin-top: 0.5em;}} "
                f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }}"
            )
            sub_group_style = (
                f"QGroupBox {{ color: {title_color_str}; border: none; background-color: transparent; }}"
            )
            fitting_label_color_style = (
                f"QLabel {{ color: {title_color_str}; background-color: transparent; }}"
            )
            fitting_info_main_layout = self.fitting_info_group.layout()
            if fitting_info_main_layout:
                for i in range(fitting_info_main_layout.count()):
                    item = fitting_info_main_layout.itemAt(i)
                    if item and item.widget() and isinstance(item.widget(), QtWidgets.QGroupBox):
                        sub_group_box = item.widget()
                        sub_group_box.setStyleSheet(sub_group_style)
                        form_layout = sub_group_box.layout()
                        if (
                            form_layout
                            and isinstance(form_layout, QtWidgets.QFormLayout)
                        ):
                            for row in range(form_layout.rowCount()):
                                label_widget = form_layout.itemAt(
                                    row, QtWidgets.QFormLayout.ItemRole.LabelRole
                                ).widget()
                                if label_widget:
                                    label_widget.setStyleSheet(fitting_label_color_style)
                                field_widget = form_layout.itemAt(
                                    row, QtWidgets.QFormLayout.ItemRole.FieldRole
                                ).widget()
                                if field_widget:
                                    field_widget.setStyleSheet(fitting_label_color_style)
    
        # ----- Button styles -----
        if dark_mode:
            button_style = """
                QPushButton {
                    background-color: #3C3C3C;
                    color: white;
                    border: 1px solid #555555;
                    padding: 5px 10px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #4C4C4C;
                }
                QPushButton:pressed {
                    background-color: #2C2C2C;
                }
                QPushButton:disabled {
                    background-color: #1C1C1C;
                    color: #666666;
                }
            """
        else:
            button_style = """
                QPushButton {
                    background-color: #F0F0F0;
                    color: black;
                    border: 1px solid #CCCCCC;
                    padding: 5px 10px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #E0E0E0;
                }
                QPushButton:pressed {
                    background-color: #D0D0D0;
                }
                QPushButton:disabled {
                    background-color: #F8F8F8;
                    color: #999999;
                }
            """
    
        # Digest tab buttons
        if hasattr(self, 'prev_button'):
            self.prev_button.setStyleSheet(button_style)
        if hasattr(self, 'next_button'):
            self.next_button.setStyleSheet(button_style)
    
        # Noise tab buttons ✅
        if hasattr(self, 'prev_button_noise'):
            self.prev_button_noise.setStyleSheet(button_style)
        if hasattr(self, 'next_button_noise'):
            self.next_button_noise.setStyleSheet(button_style)
        if hasattr(self, 'refresh_noise_button'):
            self.refresh_noise_button.setStyleSheet(button_style)

        if hasattr(self, 'mean_subtract_checkbox'):
            if self.dark_mode:
                self.mean_subtract_checkbox.setStyleSheet("""
                    QCheckBox {
                        color: white;
                    }
                    QCheckBox::indicator {
                        width: 16px;
                        height: 16px;
                    }
                    QCheckBox::indicator:unchecked {
                        border: 1px solid white;
                        background-color: transparent;
                    }
                    QCheckBox::indicator:checked {
                        border: 1px solid white;
                        background-color: white;
                    }
                """)
            else:
                self.mean_subtract_checkbox.setStyleSheet("""
                    QCheckBox {
                        color: black;
                    }
                    QCheckBox::indicator {
                        width: 16px;
                        height: 16px;
                    }
                    QCheckBox::indicator:unchecked {
                        border: 1px solid black;
                        background-color: transparent;
                    }
                    QCheckBox::indicator:checked {
                        border: 1px solid black;
                        background-color: black;
                    }
                """)

        if hasattr(self, 'exp_binning_checkbox'):
            if self.dark_mode:
                self.exp_binning_checkbox.setStyleSheet("""
                    QCheckBox {
                        color: white;
                    }
                    QCheckBox::indicator {
                        width: 16px;
                        height: 16px;
                    }
                    QCheckBox::indicator:unchecked {
                        border: 1px solid white;
                        background-color: transparent;
                    }
                    QCheckBox::indicator:checked {
                        border: 1px solid white;
                        background-color: white;
                    }
                """)
            else:
                self.exp_binning_checkbox.setStyleSheet("""
                    QCheckBox {
                        color: black;
                    }
                    QCheckBox::indicator {
                        width: 16px;
                        height: 16px;
                    }
                    QCheckBox::indicator:unchecked {
                        border: 1px solid black;
                        background-color: transparent;
                    }
                    QCheckBox::indicator:checked {
                        border: 1px solid black;
                        background-color: black;
                    }
                """)
    
        # ----- Splitter -----
        if hasattr(self, 'main_splitter'):
            splitter_style = f"""
            QSplitter::handle {{
                background-color: {'#555555' if dark_mode else '#CCCCCC'};
            }}
            QSplitter::handle:hover {{
                background-color: {'#777777' if dark_mode else '#AAAAAA'};
            }}
            """
            self.main_splitter.setStyleSheet(splitter_style)
    
        # ----- Labels -----
        if hasattr(self, 'detector_count_label'):
            self.detector_count_label.setStyleSheet(
                f"QLabel {{ color: {title_color_str}; background-color: transparent; }}"
            )
        if hasattr(self, 'trace_hint_label'):
            self.trace_hint_label.setStyleSheet(
                f"QLabel {{ color: {title_color_str}; background-color: transparent; font-size: 10pt; }}"
            )
    
        # Noise tab labels ✅
        if hasattr(self, 'detector_count_label_noise'):
            self.detector_count_label_noise.setStyleSheet(
                f"QLabel {{ color: {title_color_str}; background-color: transparent; }}"
            )
    
        if hasattr(self, 'tabs') and isinstance(self.tabs, QtWidgets.QTabWidget):
            if dark_mode:
                self.tabs.setStyleSheet("""
                    QTabWidget::pane {
                        border: 1px solid #555555;
                        background: #1C1C1C;
                    }
                    QTabBar::tab {
                        background: #2C2C2C;
                        color: white;
                        padding: 8px 16px;
                        border: 1px solid #444444;
                        border-top-left-radius: 4px;
                        border-top-right-radius: 4px;
                    }
                    QTabBar::tab:selected {
                        background: #3C3C3C;
                        color: white;
                        border: 1px solid #777777;
                    }
                    QTabBar::tab:hover {
                        background: #4C4C4C;
                    }
                """)
            else:
                self.tabs.setStyleSheet("""
                    QTabWidget::pane {
                        border: 1px solid #CCCCCC;
                        background: #FFFFFF;
                    }
                    QTabBar::tab {
                        background: #F0F0F0;
                        color: black;
                        padding: 8px 16px;
                        border: 1px solid #CCCCCC;
                        border-top-left-radius: 4px;
                        border-top-right-radius: 4px;
                    }
                    QTabBar::tab:selected {
                        background: #FFFFFF;
                        color: black;
                        border: 1px solid #888888;
                    }
                    QTabBar::tab:hover {
                        background: #E8E8E8;
                    }
                """)
        
        # ----- Update plots to reflect new theme -----
        self._update_plots()

