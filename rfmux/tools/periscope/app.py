#!/usr/bin/env -S uv run
# The shebang line above allows this script to be executed directly,
# using 'uv run' for optimized execution if 'uv' (a fast Python installer and runner)
# is available in the environment.
"""
Periscope – Core Application Logic
==================================

This module defines the main `Periscope` class, which is the central
`QMainWindow` for the real-time multi-pane data viewer and network
analysis tool. It integrates UI components, data processing tasks,
and hardware interaction.

The implementation is organized by delegating responsibilities to helper modules:
  - `utils.py`: Contains constants, utility functions, and base classes.
  - `tasks.py`: Manages background worker threads for data processing (IQ, PSD, etc.)
                and hardware interactions (network analysis, CRS initialization).
  - `ui.py`: Defines various dialogs and custom window classes used by Periscope.
             (This was formerly periscope_ui.py).

The `Periscope` class itself handles:
  - Initialization of the main application window and its components.
  - Parsing of channel configurations.
  - Management of data buffers.
  - Coordination of UI updates based on incoming data and user interactions.
  - Launching and managing worker threads for computationally intensive tasks.
  - Display settings and theme management.
  - Integration of network analysis and multisweep functionalities.
  - Optional embedded iPython console for interactive sessions.
"""

# --- Subpackage Imports ---
# Wildcard imports are used here for convenience to bring in numerous
# utility functions, constants, task-related classes, and UI components
# from their respective modules within the 'periscope' subpackage.
# While explicit imports are generally preferred, this approach is used
# here due to the large number of entities being imported.
# It is assumed that critical components (e.g., QtWidgets, QIcon, specific task
# and UI classes) are correctly exported and made available by these modules.

from .utils import *  # Provides: constants (DEFAULT_BUFFER_SIZE, ICON_PATH, etc.),
                       # helper functions (parse_channels_multich, mode_title, etc.),
                       # Qt components (QtWidgets, QtCore, QFont, QIntValidator, etc.),
                       # plotting tools (pyqtgraph as pg, ClickableViewBox),
                       # and other utilities (Circular, np, math, time, traceback, warnings).

from .tasks import *  # Provides: worker thread classes (UDPReceiver, IQTask, PSDTask,
                       # NetworkAnalysisTask, CRSInitializeTask, MultisweepTask, etc.)
                       # and their associated signal classes (IQSignals, PSDSignals, etc.).

from .ui import *     # Provides: dialog classes (NetworkAnalysisDialog, InitializeCRSDialog, etc.)
                       # and window classes (NetworkAnalysisWindow, MultisweepWindow).
                       # (Assumes periscope_ui.py has been refactored into ui.py and exports these).
from .app_runtime import PeriscopeRuntime
from PyQt6 import sip  # For checking if Qt C++ objects have been deleted
from .mock_configuration_dialog import MockConfigurationDialog
from .dock_manager import PeriscopeDockManager
from .main_plot_panel import MainPlotPanel
from .session_manager import SessionManager
from .session_browser_panel import SessionBrowserPanel
from .session_startup_dialog import UnifiedStartupDialog
from rfmux.core.transferfunctions import convert_roc_to_volts
from rfmux.mock import config as mc
import datetime



class DummyReceiver:
    """A dummy receiver for Offline mode that provides the expected interface."""
    def __init__(self):
        self.queue = queue.Queue()
    def start(self): pass
    def stop(self): pass
    def wait(self): pass
    def get_dropped_packets(self): return 0
    def get_received_packets(self): return 0

class Periscope(QtWidgets.QMainWindow, PeriscopeRuntime):
    """
    Multi‑pane PyQt application for real-time data visualization of:
      - Time-domain waveforms (TOD)
      - IQ (density or scatter)
      - FFT
      - Single-sideband PSD (SSB)
      - Dual-sideband PSD (DSB)
      - Network Analysis (amplitude and phase vs frequency)

    Additional Features
    -------------------
    - Multi-channel grouping via '&'.
    - An "Auto Scale" checkbox for IQ/FFT/SSB/DSB (not TOD).
    - Global toggles to hide/show I, Q, and Mag lines for TOD, FFT, and SSB.
    - A "Help" button to display usage and interaction details.
    - Network Analysis functionality with real-time updates.

    Parameters
    ----------
    host : str
        The multicast/UDP host address for receiving packets.
    module : int
        The module index (1-based) used to filter incoming packets.
    chan_str : str, optional
        A comma-separated list of channels, possibly using '&' to group channels
        in one row. Defaults to "1".
    buf_size : int, optional
        Size of each ring buffer for storing incoming data (default is 5000).
    refresh_ms : int, optional
        GUI refresh interval in milliseconds (default 33).
    dot_px : int, optional
        Dot diameter in pixels for IQ density mode (default is 1).
    crs : CRS, optional
        CRS object for hardware communication (needed for network analysis).
    """

    def __init__(
        self,
        host: str,
        module: int,
        chan_str: str = "1",
        buf_size: int = DEFAULT_BUFFER_SIZE, # Constant from .utils
        refresh_ms: int = DEFAULT_REFRESH_MS, # Constant from .utils
        dot_px: int = DENSITY_DOT_SIZE,       # Constant from .utils
        crs=None, # CRS object, type hint likely from .utils or a core module
        skip_startup_dialog: bool = False,  # Skip dialog if already handled by launcher
    ):
        """
        Initializes the Periscope main window.

        Sets up data sources, buffers, UI elements, worker threads, and timers.
        """
        super().__init__()

        # --- Core Parameters ---
        self.host: str = host                    # UDP host for data stream
        self.module: int = module                  # Module index to filter
        self.N: int = buf_size                   # Ring buffer size per channel
        self.refresh_ms: int = refresh_ms          # GUI refresh interval (ms)
        self.dot_px: int = max(1, int(dot_px))   # Dot size for IQ density plots (pixels)
        self.crs = crs                          # CRS hardware object (optional, for netanal)

        # Parse channel string (e.g., "1&2,3") into a list of channel groups.
        # parse_channels_multich is imported from .utils.
        self.channel_list: list[list[int]] = parse_channels_multich(chan_str)
        
        # Check if we're in mock mode
        self.is_mock_mode = self.host in ["127.0.0.1", "localhost", "::1"]
        self.mock_config = None  # Store current mock configuration

        # --- State Variables ---
        self.paused: bool = False               # Flag to pause/resume data processing
        self.start_time: float | None = None    # Timestamp of the first processed packet
        self.frame_cnt: int = 0                 # Counter for GUI frames rendered
        self.pkt_cnt: int = 0                   # Counter for processed UDP packets
        self.t_last: float = time.time()        # Timestamp of the last performance update (time from .utils)
        self.prev_receive = 0
        self.prev_drop = 0

        # Decimation stage, dynamically updated based on inferred sample rate.
        # This is used for PSD calculations.
        self.dec_stage: int = 6
        self.last_dec_update: float = 0.0       # Timestamp of last decimation stage update

        # --- Display Settings ---
        self.dark_mode: bool = True             # UI theme (dark/light)
        self.real_units: bool = False           # Display data in real units (V, dBm) vs. counts
        self.unit_mode: str = "counts"          # Current unit mode: "counts", "real", or "df"
        self.psd_absolute: bool = True          # PSD y-axis scale (absolute dBm/Hz vs. relative dBc/Hz)
        self.auto_scale_plots: bool = True      # Enable/disable auto-ranging for plots (except TOD)
        self.show_i: bool = True                # Visibility of 'I' component traces
        self.show_q: bool = True                # Visibility of 'Q' component traces
        self.show_m: bool = True                # Visibility of 'Magnitude' component traces
        self.zoom_box_mode: bool = True         # Default mouse mode for plots (zoom vs pan)


        self.test_noise_samples = {}           ##### debugging purposes 
        self.noise_count = 0                   ##### debugging purposes 
        self.phase_shifts = []                 ##### debugging purposes 
        
        # --- df Calibration Storage ---
        # Stores calibration factors for frequency shift/dissipation conversion
        # Structure: {module: {detector_idx: complex_calibration_factor}}
        self.df_calibrations: Dict[int, Dict[int, complex]] = {}

        # --- Initialization Steps ---
        # Initialize structures for tracking background worker threads.
        self._init_workers()

        # Create colormap for IQ density plots (pg is pyqtgraph, from .utils).
        self._init_colormap()

        # Start the UDP packet receiver thread (UDPReceiver from .tasks).
        self._init_receiver()

        # Initialize a QThreadPool for managing concurrent tasks (QThreadPool from .utils).
        # Used for network analysis, PSD calculations, etc.
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(4)  # Example: Allow up to 4 concurrent network analyses.

        # Initialize the DockManager for managing analysis windows as dockable panels
        self.dock_manager = PeriscopeDockManager(self)

        # Create UI widgets first (before MainPlotPanel needs them)
        self._create_ui_widgets(chan_str)
        
        # Construct the main user interface.
        self._build_ui(chan_str) # Pass chan_str for initial display in QLineEdit
        
        # Create the main plot panel in its dock
        self._add_plot_container(None)  # Layout not needed, using dock
        
        # Add session browser dock (after Main dock exists so splitDockWidget works)
        self._add_session_browser_dock()
        
        # Create the Window menu for dock management
        self._create_window_menu()

        # Initialize data buffers for each channel (Circular buffer from .utils).
        self._init_buffers()

        # Build the plot layout based on initial settings.
        self._build_layout()

        # Start the timer for periodic GUI updates (QtCore from .utils).
        self._start_timer()
        
        # Set initial window size (wider and taller for better visibility)
        self.resize(900, 450)
        
        # Show session startup dialog (unless already handled by launcher)
        self._skip_startup_dialog = skip_startup_dialog
        if not skip_startup_dialog:
            QtCore.QTimer.singleShot(100, self._show_session_startup_dialog)

    def _init_workers(self):
        """
        Initialize dictionaries and signal objects for tracking worker threads.

        This includes workers for IQ plots, PSD calculations, network analysis,
        CRS initialization, and multisweep tasks.
        Signal objects (e.g., IQSignals, PSDSignals from .tasks) are used for
        thread-safe communication between worker threads and the main GUI thread.
        """
        # IQ plot worker tracking: maps row index to a boolean indicating if a worker is active.
        # IQSignals (from .tasks) handles signals for IQ task completion.
        self.iq_workers: Dict[int, bool] = {}  # Tracks active IQ workers per plot row
        self.iq_signals = IQSignals()           # Signal object for IQ tasks
        self.iq_signals.done.connect(self._iq_done) # Connect completion signal to handler

        # PSD concurrency tracking: nested dictionary [row_index][psd_type_char][channel_id] -> bool
        # 'S' for Single-Sideband, 'D' for Dual-Sideband.
        self.psd_workers: Dict[int, Dict[str, Dict[int, bool]]] = {}
        for row_i, group in enumerate(self.channel_list): # For each row of plots
            self.psd_workers[row_i] = {"S": {}, "D": {}}    # Initialize SSB and DSB dicts for the row
            for c_id in group: # For each channel in the group for that row (c_id is channel ID)
                self.psd_workers[row_i]["S"][c_id] = False # Initialize SSB worker flag for this channel
                self.psd_workers[row_i]["D"][c_id] = False # Initialize DSB worker flag for this channel

        self.psd_signals = PSDSignals() # Signal object for PSD tasks (from .tasks)
        self.psd_signals.done.connect(self._psd_done) # Connect completion signal to handler

        # Network Analysis (NetAnal) signals and tracking.
        # NetworkAnalysisSignals (from .tasks) - signals are routed directly to panels via check_connection()
        self.netanal_signals = NetworkAnalysisSignals()
        # Only keep the error signal connected globally as a fallback
        self.netanal_signals.error.connect(self._netanal_error)
        
        # Structures for managing multiple Network Analysis windows and tasks.
        # NetworkAnalysisTask is from .tasks.
        self.netanal_windows: Dict[str, Dict] = {} # Stores window instances and related data by a unique window_id
        self.netanal_window_count: int = 0         # Counter to generate unique window_ids
        self.netanal_tasks: Dict[str, NetworkAnalysisTask] = {} # Stores active NetAnal tasks

        # CRS (Control and Readout System) Initialization signals.
        # CRSInitializeSignals (from .tasks) handles success/error signals for CRS initialization.
        self.crs_init_signals = CRSInitializeSignals()
        self.crs_init_signals.success.connect(self._crs_init_success)
        self.crs_init_signals.error.connect(self._crs_init_error)
        
        # Multisweep analysis signals and tracking.
        # MultisweepSignals and MultisweepTask are from .tasks.
        self.multisweep_signals = MultisweepSignals()
        self.multisweep_windows: Dict[str, Dict] = {} # Stores multisweep window instances
        self.multisweep_window_count: int = 0        # Counter for unique multisweep window_ids
        self.multisweep_tasks: Dict[str, MultisweepTask] = {} # Stores active Multisweep tasks

        # Attributes for the optional embedded iPython console.
        # QTCONSOLE_AVAILABLE is a boolean constant from .utils.
        self.kernel_manager = None      # Manages the iPython kernel
        self.jupyter_widget = None      # The Qt widget for the console
        self.console_dock_widget = None # Dock widget to host the console

    def _create_ui_widgets(self, chan_str: str):
        """
        Create all UI widgets (but don't add them to layouts yet).
        
        These widgets will be added to layouts by MainPlotPanel.
        This method must be called before creating MainPlotPanel.
        """
        # Create toolbar widgets
        self.e_ch = QtWidgets.QLineEdit(chan_str)
        self.e_ch.setMaximumWidth(40)
        self.e_ch.setToolTip("Enter comma-separated channels or use '&' to group in one row (e.g., 1&2,3,4&5&6).")
        self.e_ch.returnPressed.connect(self._update_channels)

        self.e_buf = QtWidgets.QLineEdit(str(self.N))
        self.e_buf.setValidator(QIntValidator(10, 1_000_000, self))
        self.e_buf.setMaximumWidth(120)
        self.e_buf.setToolTip("Size of the ring buffer for each channel (history/FFT depth).")
        self.e_buf.editingFinished.connect(self._change_buffer)

        self.b_pause = QtWidgets.QPushButton("Pause", clicked=self._toggle_pause)
        self.b_pause.setToolTip("Pause or resume real-time data acquisition and display.")

        # Unit radio buttons
        self.unit_group = QtWidgets.QButtonGroup()
        self.rb_counts = QtWidgets.QRadioButton("Counts")
        self.rb_real_units = QtWidgets.QRadioButton("Real Units")
        self.rb_df_units = QtWidgets.QRadioButton("df Units")
        
        self.rb_counts.setToolTip("Display raw ADC counts")
        self.rb_real_units.setToolTip("Display in voltage/power units (V, dBm/Hz)")
        self.rb_df_units.setToolTip("Display frequency shift (Hz) and dissipation")
        
        self.unit_group.addButton(self.rb_counts, 0)
        self.unit_group.addButton(self.rb_real_units, 1)
        self.unit_group.addButton(self.rb_df_units, 2)
        
        if self.real_units:
            self.rb_real_units.setChecked(True)
        else:
            self.rb_counts.setChecked(True)
        
        self.unit_group.buttonClicked.connect(self._unit_mode_changed)

        # Plot type checkboxes
        self.cb_time = QtWidgets.QCheckBox("TOD", checked=True)
        self.cb_iq = QtWidgets.QCheckBox("IQ", checked=False)
        self.cb_fft = QtWidgets.QCheckBox("FFT", checked=self.is_mock_mode)
        self.cb_ssb = QtWidgets.QCheckBox("Single Sideband PSD", checked=not self.is_mock_mode)
        self.cb_dsb = QtWidgets.QCheckBox("Dual Sideband PSD", checked=False)
        
        for cb_plot_type in (self.cb_time, self.cb_iq, self.cb_fft):
            cb_plot_type.toggled.connect(self._build_layout)
        
        self.cb_ssb.toggled.connect(self._handle_psd_toggle)
        self.cb_dsb.toggled.connect(self._handle_psd_toggle)

        # CRS control buttons
        self.btn_init_crs = QtWidgets.QPushButton("Initialize CRS Board")
        self.btn_init_crs.setToolTip("Open a dialog to initialize the CRS board (e.g., set IRIG source).")
        self.btn_init_crs.clicked.connect(self._show_initialize_crs_dialog)
        if self.crs is None: # Disable if no CRS object is available
            self.btn_init_crs.setEnabled(False)
            self.btn_init_crs.setToolTip("CRS object not available - cannot initialize board.")

        self.btn_netanal = QtWidgets.QPushButton("Network Analyzer")
        self.btn_netanal.setToolTip("Open the network analysis configuration window to perform sweeps.")
        self.btn_netanal.clicked.connect(self._show_netanal_dialog)
        if self.crs is None:
            # In Offline Mode, allow opening dialog to load data
            if self.host == "OFFLINE":
                self.btn_netanal.setToolTip("Network Analyzer (Offline Mode - Loading Only)")
            else:
                self.btn_netanal.setEnabled(False)
                self.btn_netanal.setToolTip("CRS object not available - network analysis disabled.")

        self.btn_load_multi = QtWidgets.QPushButton("Load Multisweep")
        self.btn_load_multi.setToolTip("Load Multisweep directly from main window.")
        self.btn_load_multi.clicked.connect(self._load_multisweep_dialog)
        if self.crs is None and self.host != "OFFLINE":
            self.btn_load_multi.setEnabled(False)
            self.btn_load_multi.setToolTip("CRS object not available - load multisweep disabled.")

        self.btn_load_bias = QtWidgets.QPushButton("Load Bias")
        self.btn_load_bias.setToolTip("Bias KIDS directly from the main window.")
        self.btn_load_bias.clicked.connect(self.handle_bias_from_file)
        if self.crs is None and self.host != "OFFLINE":
            self.btn_load_bias.setEnabled(False)
            self.btn_load_bias.setToolTip("CRS object not available - load Bias disabled.")

        self.btn_toggle_cfg = QtWidgets.QPushButton("Show Configuration")
        self.btn_toggle_cfg.setCheckable(True)
        self.btn_toggle_cfg.toggled.connect(self._toggle_config)

        # Mock mode specific buttons
        if self.is_mock_mode:
            self.btn_reconfigure_mock = QtWidgets.QPushButton("Reconfigure Simulated KIDs")
            self.btn_reconfigure_mock.setToolTip("Reconfigure the simulated KID resonator parameters")
            self.btn_reconfigure_mock.clicked.connect(self._show_mock_config_dialog)
            
            self.btn_qp_pulses = QtWidgets.QPushButton("QP Pulses: Off")
            self.btn_qp_pulses.setToolTip("Toggle quasiparticle pulses in mock mode\nCycles through: Off → Periodic → Random → Off")
            self.btn_qp_pulses.clicked.connect(self._toggle_qp_pulses)
            self.qp_pulse_mode = 'none'

        # Create configuration panel (will be added to MainPlotPanel's layout)
        self.ctrl_panel = QtWidgets.QGroupBox("Configuration")
        self.ctrl_panel.setVisible(False)
        config_panel_layout = QtWidgets.QHBoxLayout(self.ctrl_panel)
        
        config_panel_layout.addWidget(self._create_show_curves_group())
        config_panel_layout.addWidget(self._create_iq_mode_group())
        config_panel_layout.addWidget(self._create_psd_mode_group())
        config_panel_layout.addWidget(self._create_display_group())

    def _init_colormap(self):
        """
        Initialize the colormap used for IQ density plots.

        Uses pyqtgraph's 'turbo' colormap and prepares a lookup table (LUT)
        with an alpha channel for transparency.
        `pg` (pyqtgraph) and `np` (numpy) are imported from .utils.
        """
        cmap = pg.colormap.get("turbo")  # Get the 'turbo' colormap object
        lut_rgb = cmap.getLookupTable(0.0, 1.0, 255)  # Get RGB values for 255 levels

        # Create the final LUT:
        # - Start with a fully transparent black for value 0 (np.zeros).
        # - Append the RGB values with full opacity (255 alpha).
        self.lut = np.vstack([
            np.zeros((1, 4), np.uint8),  # [0,0,0,0] for transparent background
            np.hstack([lut_rgb, 255 * np.ones((255, 1), np.uint8)]) # Add alpha channel
        ])

    def _init_receiver(self):
        """
        Initialize and start the UDP packet receiver thread.

        The `UDPReceiver` class (from .tasks) handles receiving packets
        from the specified host and module in a separate thread.
        """
        if self.host == "OFFLINE":
            self.receiver = DummyReceiver()
        else:
            self.receiver = UDPReceiver(self.host, self.module) # Instantiate the receiver
            self.receiver.start()  # Start the receiver thread

    def _start_timer(self):
        """
        Start the periodic QTimer for GUI updates.

        The timer triggers the `_update_gui` method at regular intervals
        defined by `self.refresh_ms`.
        `QtCore` is imported from .utils.
        """
        self.timer = QtCore.QTimer(singleShot=False)  # Create a non-single-shot timer
        self.timer.timeout.connect(self._update_gui)   # Connect timeout signal to GUI update method
        self.timer.start(self.refresh_ms)             # Start the timer
        # Set window title with CRS serial/module info
        if self.host == "OFFLINE":
            title = "Periscope - Offline Mode"
        elif self.crs is not None and hasattr(self.crs, 'serial'):
            serial = str(self.crs.serial)
            if serial == "0000":
                title = f"Periscope - Mock CRS, Module {self.module}"
            else:
                title = f"Periscope - CRS{serial}, Module {self.module}"
        elif self.host in ["127.0.0.1", "localhost", "::1"]:
            title = f"Periscope - Mock CRS, Module {self.module}"
        else:
            title = f"Periscope - {self.host}, Module {self.module}"
        self.setWindowTitle(title)

    # ───────────────────────── UI Construction ─────────────────────────
    # The methods in this section are responsible for building the various
    # components of the Periscope application's user interface.

    def _build_ui(self, chan_str: str):
        """
        Create and configure all top-level widgets and the main layout.

        This method orchestrates the construction of the entire UI by calling
        helper methods to add specific sections like the status bar.
        The toolbar, config panel, and plots are now in the MainPlotPanel dock.

        Args:
            chan_str (str): The initial channel string (passed to MainPlotPanel)
        """
        # Initialize session manager
        self.session_manager = SessionManager(self)
        
        # Add status bar
        self._add_status_bar()
        
        # Add interactive console dock
        self._add_interactive_console_dock()
        
        # Add session menu
        self._create_session_menu()
        
        # Add view menu (contains Dark Mode)
        self._create_view_menu()

        # Contains jupyter notebook and 
        self._create_jupyter_menu()

        # Add the help menu 
        self._create_help_menu()

        self.menuBar().setNativeMenuBar(False)
        
        # Note: Session browser dock will be added after Main dock is created
        
        # Connect session manager to status bar
        self.session_manager.session_started.connect(self._update_session_status)
        self.session_manager.session_ended.connect(self._update_session_status)
        self.session_manager.file_exported.connect(self._on_file_exported_status)

    def _create_show_curves_group(self) -> QtWidgets.QGroupBox:
        """
        Create the 'Show Curves' group box for the configuration panel.

        Contains checkboxes to toggle visibility of I, Q, and Magnitude traces.

        Returns:
            QtWidgets.QGroupBox: The configured group box.
        """
        # `QtWidgets` is from .utils.
        group_box = QtWidgets.QGroupBox("Show Curves")
        layout = QtWidgets.QHBoxLayout(group_box)

        self.cb_show_i = QtWidgets.QCheckBox("I", checked=True)
        self.cb_show_q = QtWidgets.QCheckBox("Q", checked=True)
        self.cb_show_m = QtWidgets.QCheckBox("Magnitude", checked=True)
        # Connect toggled signals to update plot visibility
        self.cb_show_i.toggled.connect(self._toggle_iqmag)
        self.cb_show_q.toggled.connect(self._toggle_iqmag)
        self.cb_show_m.toggled.connect(self._toggle_iqmag)
        layout.addWidget(self.cb_show_i)
        layout.addWidget(self.cb_show_q)
        layout.addWidget(self.cb_show_m)
        return group_box

    def _create_iq_mode_group(self) -> QtWidgets.QGroupBox:
        """
        Create the 'IQ Mode' group box for the configuration panel.

        Contains radio buttons to switch IQ plots between density and scatter modes.

        Returns:
            QtWidgets.QGroupBox: The configured group box.
        """
        # `QtWidgets` is from .utils.
        group_box = QtWidgets.QGroupBox("IQ Mode")
        layout = QtWidgets.QHBoxLayout(group_box)

        self.rb_density = QtWidgets.QRadioButton("Density", checked=True)
        self.rb_density.setToolTip("Display IQ data as a 2D histogram (density plot).")
        self.rb_scatter = QtWidgets.QRadioButton("Scatter")
        self.rb_scatter.setToolTip("Display IQ data as a scatter plot (up to 1,000 points, CPU intensive).")

        # Group radio buttons for exclusive selection
        rb_button_group = QtWidgets.QButtonGroup(group_box)
        rb_button_group.addButton(self.rb_density)
        rb_button_group.addButton(self.rb_scatter)

        # Connect toggled signals to rebuild plot layout
        for rb_iq_mode in (self.rb_density, self.rb_scatter):
            rb_iq_mode.toggled.connect(self._build_layout)
        layout.addWidget(self.rb_density)
        layout.addWidget(self.rb_scatter)
        return group_box

    def _create_psd_mode_group(self) -> QtWidgets.QGroupBox:
        """
        Create the 'PSD Mode' group box for the configuration panel.

        Contains controls for PSD scale (absolute/relative), Welch segments, and binning options.

        Returns:
            QtWidgets.QGroupBox: The configured group box.
        """
        # `QtWidgets` is from .utils.
        group_box = QtWidgets.QGroupBox("PSD Mode")
        grid_layout = QtWidgets.QGridLayout(group_box) # Use a grid for better alignment

        self.lbl_psd_scale = QtWidgets.QLabel("PSD Scale:")
        self.rb_psd_abs = QtWidgets.QRadioButton("Absolute (dBm)", checked=True) # Default
        self.rb_psd_rel = QtWidgets.QRadioButton("Relative (dBc)")
        # Connect toggled signals to update PSD reference and rebuild layout
        for rb_psd_scale in (self.rb_psd_abs, self.rb_psd_rel):
            rb_psd_scale.toggled.connect(self._psd_ref_changed)

        self.spin_segments = QtWidgets.QSpinBox() # Spinbox for Welch segments
        self.spin_segments.setRange(1, 256)      # Number of segments for averaging
        self.spin_segments.setValue(1)           # Default to 1 (no averaging)
        self.spin_segments.setMaximumWidth(80)
        self.spin_segments.setToolTip("Number of segments for Welch PSD averaging (more segments = smoother floor, less resolution).")

        # Binning controls
        self.cb_exp_binning = QtWidgets.QCheckBox("Exponential bins")
        self.cb_exp_binning.setChecked(False)  # Default to off
        self.cb_exp_binning.setToolTip("Apply exponential binning to PSD plots. This is a visual aid and is not statistically rigorous.")
        
        self.spin_bins = QtWidgets.QSpinBox()  # Spinbox for number of bins
        self.spin_bins.setRange(10, 5000)      # Reasonable range for bins
        self.spin_bins.setValue(100)          # Default to 100 bins
        self.spin_bins.setMaximumWidth(80)
        self.spin_bins.setToolTip("Number of bins for exponential binning.")
        self.spin_bins.setEnabled(False)       # Disabled by default
        
        # Enable/disable bins spinner based on checkbox
        self.cb_exp_binning.toggled.connect(self.spin_bins.setEnabled)

        # Add widgets to the grid layout
        grid_layout.addWidget(self.lbl_psd_scale, 0, 0) # Row 0, Col 0
        grid_layout.addWidget(self.rb_psd_abs, 0, 1)    # Row 0, Col 1
        grid_layout.addWidget(self.rb_psd_rel, 0, 2)    # Row 0, Col 2
        grid_layout.addWidget(QtWidgets.QLabel("Segments:"), 1, 0) # Row 1, Col 0
        grid_layout.addWidget(self.spin_segments, 1, 1) # Row 1, Col 1
        
        # Binning row
        grid_layout.addWidget(QtWidgets.QLabel("Binning:"), 2, 0) # Row 2, Col 0
        grid_layout.addWidget(self.cb_exp_binning, 2, 1)          # Row 2, Col 1
        grid_layout.addWidget(self.spin_bins, 2, 2)               # Row 2, Col 2
        
        return group_box

    def _create_display_group(self) -> QtWidgets.QGroupBox:
        """
        Create the 'General Display' group box for the configuration panel.

        Contains checkboxes for zoom box mode and auto-scaling plots.
        Note: Dark mode is now in the View menu.

        Returns:
            QtWidgets.QGroupBox: The configured group box.
        """
        # `QtWidgets` is from .utils.
        group_box = QtWidgets.QGroupBox("General Display")
        layout = QtWidgets.QHBoxLayout(group_box)

        # Checkbox for zoom box mode vs. pan mode on plots
        self.cb_zoom_box = QtWidgets.QCheckBox("Zoom Box Mode", checked=True)
        self.cb_zoom_box.setToolTip("Enable: Left-click drag creates a zoom box.\nDisable: Left-click drag pans the plot.")
        self.cb_zoom_box.toggled.connect(self._toggle_zoom_box_mode)
        layout.addWidget(self.cb_zoom_box)

        # Checkbox for auto-scaling plots (excluding TOD)
        self.cb_auto_scale = QtWidgets.QCheckBox("Auto Scale", checked=self.auto_scale_plots)
        self.cb_auto_scale.setToolTip("Enable/disable auto-ranging for IQ, FFT, SSB, and DSB plots. Can improve display performance when disabled.")
        self.cb_auto_scale.toggled.connect(self._toggle_auto_scale)
        layout.addWidget(self.cb_auto_scale)
        return group_box

    def _add_plot_container(self, layout: QtWidgets.QVBoxLayout):
        """
        Add the main plot panel as a dockable widget.

        Creates a MainPlotPanel, wraps it in a dock, and sets up references
        so that PeriscopeRuntime can access self.grid and self.container.
        """
        # Create the MainPlotPanel
        self.main_plot_panel = MainPlotPanel(self)
        
        # Set up references for backward compatibility with PeriscopeRuntime
        self.container = self.main_plot_panel.container
        self.grid = self.main_plot_panel.get_grid()
        
        # Wrap in a dock widget
        main_dock = self.dock_manager.create_dock(
            self.main_plot_panel,
            "Main",
            "main_plots",
            area=QtCore.Qt.DockWidgetArea.LeftDockWidgetArea  # Main plots on left/center
        )
        
        # Mark main dock as protected (hide instead of close when X is clicked)
        self.dock_manager.protect_dock("main_plots")
        
        # Show the main dock immediately, unless in offline mode (no data streaming)
        if self.host != "OFFLINE":
            main_dock.show()
            main_dock.raise_()
        else:
            # In offline mode, hide main dock since there's no data streaming
            main_dock.hide()

    def _add_status_bar(self):
        """
        Add a status bar to the main window.

        Used to display performance statistics (FPS, PPS, simulation speed in mock mode).
        """
        # `QtWidgets` is from .utils.
        self.setStatusBar(QtWidgets.QStatusBar()) # Create and set a new status bar

        self.fps_label = QtWidgets.QLabel()
        self.pps_label = QtWidgets.QLabel()
        
        # Add simulation speed label for mock mode
        if self.is_mock_mode:
            self.sim_speed_label = QtWidgets.QLabel()
            self.sim_speed_label.setToolTip("Simulation speed relative to real-time\n>1.0x = faster than real-time\n<1.0x = slower than real-time")
        
        self.packet_loss_label = QtWidgets.QLabel()
        self.dropped_label = QtWidgets.QLabel()
        self.info_text = QtWidgets.QLabel()

        self.default_packet_loss_color = self.packet_loss_label.palette().color(QtGui.QPalette.WindowText).name()

        # Add them to the status bar
        self.statusBar().addWidget(self.fps_label)
        self.statusBar().addWidget(self.pps_label)
        
        # Add simulation speed label if in mock mode
        if self.is_mock_mode:
            self.statusBar().addWidget(self.sim_speed_label)
        
        self.statusBar().addWidget(self.packet_loss_label)
        self.statusBar().addWidget(self.dropped_label)
        self.statusBar().addPermanentWidget(self.info_text)
        

    def _show_help(self):
        """
        Display the help dialog with usage instructions and tips.
        """
        # `QtWidgets`, `QtCore` are from .utils.
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Periscope Help")
        help_text = (
            "# Periscope Help\n\n"
            
            "## Basic Usage\n"
            "- **Channel Specification:** Specify channels using comma-separated values\n"
            "  - Use `&` to group multiple channels in one row\n"
            "  - Example: `3&5,7` displays channels 3 and 5 in one row, channel 7 in another row\n"
            "- **Buffer Size:** Controls the amount of data stored for each channel, affecting FFT resolution\n"
            "- **Pause/Resume:** Temporarily stop data acquisition while examining plots or settings\n\n"
            
            "## Plot Types\n"
            "- **TOD (Time-Domain):** Raw time series data showing I, Q, and Magnitude components\n"
            "- **IQ:** Complex plane visualization (density or scatter plot)\n"
            "- **FFT:** Raw frequency spectrum\n"
            "- **Single Sideband PSD:** Power spectrum with proper normalization for single-sideband analysis\n"
            "- **Dual Sideband PSD:** Power spectrum optimized for dual-sideband analysis\n\n"
            
            "## Network Analysis Features\n"
            "1. Click **Network Analyzer** to configure and launch a frequency sweep\n"
            "2. Set frequency range, sweep resolution, and amplitude parameters\n"
            "3. View amplitude and phase response vs. frequency\n"
            "4. Use **Find Resonances** to automatically identify resonance frequencies\n"
            "5. Use **Unwrap Cable Delay** to compensate for cable length effects\n"
            "6. Click **Take Multisweep** to perform detailed analysis around resonances\n"
            "7. Export data in various formats using the **Export Data** button\n\n"
            
            "## Multisweep Analysis\n"
            "- Provides high-resolution frequency sweeps around identified resonances\n"
            "- Allows amplitude-dependent characterization of resonators\n"
            "- Extracts detector parameters like Q-factor, resonant frequency, and more\n\n"
            
            "## Display Options\n"
            "- **Show Configuration:** Toggle to access advanced display settings\n"
            "- **Show Curves:** Select which components (I, Q, Magnitude) to display\n"
            "- **Real Units:** Toggle between raw counts and calibrated units (V, dBm)\n"
            "- **IQ Mode:** Choose between density (2D histogram) and scatter display\n"
            "- **PSD Mode:** Select absolute (dBm/Hz) or relative (dBc/Hz) scaling\n"
            "- **Auto Scale:** Enable/disable automatic y-axis scaling\n"
            "- **Zoom Box Mode:** When enabled, left-click drag creates a zoom box; when disabled, pans the plot\n"
            "- **Dark Mode:** Toggle between light and dark UI themes\n\n"
            
            "## Interactive Features\n"
            "- **Mouse Operations:**\n"
            "  - Left-click drag: Zoom into region (when Zoom Box enabled) or pan (when disabled)\n"
            "  - Mouse wheel: Zoom in/out around cursor position\n"
            "  - Right-click: Access context menu with plot controls\n"
            "  - Double-click: Show point coordinates\n"
            "- **Interactive Session:** Open an embedded iPython console for direct data access\n"
            "- **Initialize CRS:** Configure the CRS board settings (IRIG source, etc.)\n\n"

            "## Programmatic Usage\n"
            "**From IPython/Jupyter:**\n"
            "```python\n"
            ">>> crs.raise_periscope(module=2, channels=\"3&5\")\n"
            "```\n"
            "- In non-blocking mode, you can still interact with your session concurrently\n\n"
            
            "## Command-Line Usage\n"
            "```bash\n"
            "$ cd rfmux\n"
            "$ pip install . # Installs the commandline tool\n"
            " periscope <hostname> [options]\n"
            "```\n"
            "Options:\n"
            "- `--module <num>`: Specify module number (default: 1)\n"
            "- `--channels <spec>`: Channel specification (default: \"1\")\n"
            "- `--buffer <size>`: Buffer size (default: 5000)\n"
            "- `--refresh <ms>`: GUI refresh rate in ms (default: 33)\n"
            "- `--dot-px <size>`: Dot size for IQ density display (default: 1)\n"
            "\n"
            "## Packet Loss - Networking settings\n"
            "Packet loss can occur in the networking between the CRS and your computer, but can also occur onboard your DAQ"
            "computer if the OS UDP receive buffer is too small to handle the incoming data rate, or if the CPU is too burdened"
            "by other tasks to drain the existing buffer in time."
            "\n"
            "\n"
            "On macOS and Windows, its recommended, to run with decimation 2 or higher. To avoid dropped packets at decimation <= 4"
            " use the following commands."
            "\n"
            "\n"
            "For decimation 4. This will give you 1024 channels\n"
            "```\n"
            "     crs.set_decimation(4, module=<current_module>, short=False)\n"
            "```\n"
            "For decimation 2 and 3. This will give you 128 channels\n"
            "```\n"
            "    crs.set_decimation(2 or 3, module=<current_module>, short=True)\n"
            "```\n"
            "\n"
            "\n"
            "The steps below wil help improve the packet loss and get the maximum performance, by increasing the UDP buffer size."
            "\n"
            "- **MacOS:**\n"
            "  Run the following commands to increase the default buffer\n"
            "```\n"
            "     sudo sysctl -w kern.ipc.maxsockbuf=16777216\n"
            "     sudo sysctl -w net.inet.udp.recvspace=16777216\n"
            "```\n"
            " - **Linux:**\n"
            "   Run the following command to increase the default buffer\n"
            "```\n"
            "     sudo sysctl -w net.core.rmem_max=67108864\n"
            "```\n"
            "- **Windows:** \n\n"
            "   Please consult the README.Windows.md available on rfmux repo on how to increase the buffer size and additional resources."
            "\n"
            "\n"
            "If you are still seeing dropped packets due to UDP overflows in machines with limited UDP buffer sizes (such as OSX and Windows), it can help to reduce the periscope plotting buffer size, or number of plots.\n"            
        )
        help_dialog = QtWidgets.QDialog(self)
        help_dialog.setWindowTitle("Periscope Help")
        layout = QtWidgets.QVBoxLayout(help_dialog)
        scroll_area = QtWidgets.QScrollArea(help_dialog)
        scroll_area.setWidgetResizable(True)
        help_label = QtWidgets.QLabel(scroll_area)
        help_label.setTextFormat(QtCore.Qt.TextFormat.MarkdownText)
        help_label.setWordWrap(True)
        help_label.setText(help_text)
        help_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        help_label.setOpenExternalLinks(True)
        
        help_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse |QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard | QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse)
        
        scroll_area.setWidget(help_label)
        layout.addWidget(scroll_area)
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(help_dialog.accept)
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)
        help_dialog.setLayout(layout)
        help_dialog.resize(700, 500) 
        help_dialog.exec()

    def _toggle_zoom_box_mode(self, enable: bool):
        """
        Toggle the mouse mode for plot interaction between zoom box and pan.

        Applies the mode to all main plots and plots within any active
        NetworkAnalysisWindow instances.

        Args:
            enable (bool): True to enable zoom box mode, False for pan mode.
        """
        if not hasattr(self, "plots"): return
        self.zoom_box_mode = enable
        for rowPlots in self.plots:
            for mode, plot_widget in rowPlots.items():
                viewbox = plot_widget.getViewBox()
                # Check if viewbox is valid before accessing (may be deleted during layout rebuild)
                if isinstance(viewbox, ClickableViewBox) and not sip.isdeleted(viewbox): # ClickableViewBox from .utils, sip from PyQt6
                    viewbox.enableZoomBoxMode(enable)
        for window_id, window_data in self.netanal_windows.items():
            window = window_data.get('window')
            if window: # window is NetworkAnalysisWindow from .ui
                for module_idx in window.plots: # module_idx is int key
                    for plot_type in ['amp_plot', 'phase_plot']:
                        viewbox = window.plots[module_idx][plot_type].getViewBox()
                        if isinstance(viewbox, ClickableViewBox):
                            viewbox.enableZoomBoxMode(enable)
                if hasattr(window, 'zoom_box_cb'):
                    window.zoom_box_cb.setChecked(enable)

    def _show_initialize_crs_dialog(self):
        """
        Display a dialog for initializing the CRS (Control and Readout System) board.

        Allows the user to select an IRIG timing source and optionally clear
        existing channel configurations on the CRS. If a CRS object is not
        available or critical attributes are missing, an error message is shown.
        If the user confirms the dialog, a `CRSInitializeTask` is started.
        """
        if self.crs is None:
            QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available for initialization.")
            return
        if not hasattr(self.crs, 'TIMESTAMP_PORT') or \
           not hasattr(self.crs.TIMESTAMP_PORT, 'BACKPLANE') or \
           not hasattr(self.crs.TIMESTAMP_PORT, 'TEST') or \
           not hasattr(self.crs.TIMESTAMP_PORT, 'SMA'):
            QtWidgets.QMessageBox.critical(self, "Error", 
                                           "CRS.TIMESTAMP_PORT enum not available or incomplete. Cannot initialize.")
            return
        # InitializeCRSDialog from .ui (which imports from .dialogs)
        dialog = InitializeCRSDialog(self, self.crs)
        if dialog.exec():
            irig_source = dialog.get_selected_irig_source()
            clear_channels = dialog.get_clear_channels_state()
            if irig_source is None:
                QtWidgets.QMessageBox.warning(self, "Selection Error", "No IRIG source selected.")
                return
            # CRSInitializeTask from .tasks
            task = CRSInitializeTask(self.crs, self.module, irig_source, clear_channels, self.crs_init_signals)
            self.pool.start(task)

    def _fetch_dac_scales_for_dialog(self, dialog) -> None:
        """
        Fetch DAC scales asynchronously and connect results to dialog update slots.
        
        This helper method encapsulates the common pattern of creating a DACScaleFetcher,
        connecting its signals to update both the dialog and the Periscope instance's
        dac_scales attribute.
        
        IMPORTANT: Stores the fetcher as an instance variable to prevent garbage collection
        while the thread is still running.
        
        Args:
            dialog: A dialog instance with dac_scales dict, _update_dac_scale_info(),
                    and _update_dbm_from_normalized() methods.
        """
        if self.crs is None:
            return
        
        # Store as instance variable to prevent garbage collection while thread runs
        self._active_dac_fetcher = DACScaleFetcher(self.crs)
        
        # Clean up reference when thread completes
        self._active_dac_fetcher.finished.connect(
            lambda: setattr(self, '_active_dac_fetcher', None)
        )
        
        # Connect signals to dialog updates
        self._active_dac_fetcher.dac_scales_ready.connect(lambda scales: dialog.dac_scales.update(scales))
        self._active_dac_fetcher.dac_scales_ready.connect(dialog._update_dac_scale_info)
        self._active_dac_fetcher.dac_scales_ready.connect(dialog._update_dbm_from_normalized)
        self._active_dac_fetcher.dac_scales_ready.connect(lambda scales: setattr(self, 'dac_scales', scales))
        
        self._active_dac_fetcher.start()

    def _show_netanal_dialog(self):
        """
        Display the Network Analysis configuration dialog.

        This dialog allows users to set parameters for a network analysis sweep,
        such as frequency range, number of points, and amplitude.
        It fetches current DAC scales asynchronously to provide accurate power level
        estimations. If the user confirms the dialog with valid parameters,
        a new network analysis process is started via `_start_network_analysis`.
        """
        if self.crs is None and self.host != "OFFLINE":
            QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available")
            return
            
        default_dac_scales = {m: -0.5 for m in range(1, 9)}
        # NetworkAnalysisDialog from .ui (which imports from .dialogs)
        dialog = NetworkAnalysisDialog(self, modules=list(range(1, 9)), dac_scales=default_dac_scales)
        dialog.module_entry.setText(str(self.module))
        
        # Fetch DAC scales if CRS is available
        self._fetch_dac_scales_for_dialog(dialog)
        if self.crs is None:
            self.dac_scales = default_dac_scales.copy()
            
        if dialog.exec():
            self.dac_scales = dialog.dac_scales.copy()
            params = dialog.get_parameters()
            if params:
                if "modules" in params.keys():
                    self._load_network_analysis(params)
                else:
                    if self.crs is None:
                        QtWidgets.QMessageBox.warning(self, "Offline Mode", "Cannot start new network analysis without CRS hardware. Loading data only.")
                        return
                    self._start_network_analysis(params)

    def _start_network_analysis(self, params: dict):
        """
        Initialize and start a new network analysis process.

        This method creates a new `NetworkAnalysisPanel` wrapped in a QDockWidget 
        and a `NetworkAnalysisTask` to perform the sweep in a background thread.
        It handles single or multiple module sweeps and iterates through specified
        amplitudes if provided.

        Args:
            params (dict): A dictionary of parameters for the network analysis,
                           typically obtained from `NetworkAnalysisDialog`.
                           Expected keys include 'module' (int or list of ints),
                           'amps' (list of floats), 'amp' (float, fallback if 'amps'
                           is not present), and other sweep-specific settings.
        """
        try:
            if self.crs is None:
                QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available")
                return
            selected_module_param = params.get('module')
            if selected_module_param is None:
                modules_to_run = list(range(1, 9))
            elif isinstance(selected_module_param, list):
                modules_to_run = selected_module_param
            else:
                modules_to_run = [selected_module_param]
            if not hasattr(self, 'dac_scales'):
                QtWidgets.QMessageBox.critical(self, "Error", 
                    "DAC scales are not available. Please run the network analysis configuration again.")
                return
            
            # Create unique ID for this analysis
            window_id = f"netanal_{self.netanal_window_count}"
            self.netanal_window_count += 1
            
            # Create panel
            window_signals = NetworkAnalysisSignals()
            dac_scales_local = self.dac_scales.copy()
            panel = NetworkAnalysisPanel(self, modules_to_run, dac_scales_local, dark_mode=self.dark_mode)
            panel.set_params(params)
            
            # Wrap panel in dock
            dock_title = f"Network Analysis #{self.netanal_window_count}"
            dock = self.dock_manager.create_dock(panel, dock_title, window_id)
            
            # Store panel reference
            self.netanal_windows[window_id] = {
                'window': panel,  # Keep 'window' key for compatibility
                'dock': dock,
                'signals': window_signals,
                'amplitude_queues': {},
                'current_amp_index': {}
            }
            
            # Connect signals (same as before)
            window_signals.progress.connect(
                lambda mod, prog: panel.update_progress(mod, prog),
                QtCore.Qt.ConnectionType.QueuedConnection)
            window_signals.data_update.connect(
                lambda mod, freqs, amps, phases: panel.update_data(mod, freqs, amps, phases),
                QtCore.Qt.ConnectionType.QueuedConnection)
            window_signals.data_update_with_amp.connect(
                lambda mod, freqs, amps, phases, amp_val: panel.update_data_with_amp(mod, freqs, amps, phases, amp_val),
                QtCore.Qt.ConnectionType.QueuedConnection)
            window_signals.completed.connect(
                lambda mod: self._handle_analysis_completed(mod, window_id),
                QtCore.Qt.ConnectionType.QueuedConnection)
            window_signals.error.connect(
                lambda error_msg: QtWidgets.QMessageBox.critical(panel, "Network Analysis Error", error_msg),
                QtCore.Qt.ConnectionType.QueuedConnection)
            
            # Connect data_ready signal for session auto-export
            # Use default arg to capture panel reference for filename storage
            panel.data_ready.connect(
                lambda data, p=panel: self._handle_netanal_data_ready(modules_to_run, data, panel=p)
            )
            
            amplitudes = params.get('amps', [params.get('amp', DEFAULT_AMPLITUDE)])
            window_data = self.netanal_windows[window_id]
            window_data['amplitude_queues'] = {mod: list(amplitudes) for mod in modules_to_run}
            window_data['current_amp_index'] = {mod: 0 for mod in modules_to_run}
            for mod_iter in modules_to_run:
                panel.update_amplitude_progress(mod_iter, 1, len(amplitudes), amplitudes[0])
                self._start_next_amplitude_task(mod_iter, params, window_id)
            
            # Tabify with Main dock by default
            main_dock = self.dock_manager.get_dock("main_plots")
            if main_dock:
                self.tabifyDockWidget(main_dock, dock)
            
            # Show the dock and activate it
            dock.show()
            dock.raise_()
        except Exception as e:
            print(f"Error in _start_network_analysis: {e}")
            traceback.print_exc() # traceback from .utils
            raise


    def _load_network_analysis(self, params: dict):
        """
        Load network analysis data from file and display in a docked panel.

        Args:
            params (dict): Loaded network analysis data with 'parameters' and 'modules' keys
        """
        try:
            # Allow loading without CRS in offline mode
            if self.crs is None and self.host != "OFFLINE":
                QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available")
                return
                
            selected_module_param = params['parameters'].get('module')
            if selected_module_param is None:
                modules_to_run = list(range(1, 9))
            elif isinstance(selected_module_param, list):
                modules_to_run = selected_module_param
            else:
                modules_to_run = [selected_module_param]
            
            # Restore DAC scales from loaded data (using existing 'dac_scales_used' key)
            self.dac_scales = params['dac_scales_used']
            
            # Create unique ID for this analysis
            window_id = f"netanal_{self.netanal_window_count}"
            self.netanal_window_count += 1
            
            # Create panel
            dac_scales_local = self.dac_scales.copy()
            window_signals = NetworkAnalysisSignals()
            panel = NetworkAnalysisPanel(self, modules_to_run, dac_scales_local, dark_mode=self.dark_mode, is_loaded_data=True)
            panel._hide_progress_bars()
            panel.set_params(params['parameters'])
            
            # Wrap panel in dock
            dock_title = f"Network Analysis #{self.netanal_window_count} (Loaded)"
            dock = self.dock_manager.create_dock(panel, dock_title, window_id)
            
            # Store panel reference
            self.netanal_windows[window_id] = {'window': panel, 'dock': dock, 'signals': window_signals}
            
            # Load data into panel
            amplitudes = params['parameters'].get('amps')
            for mod in modules_to_run:
                for i in range(len(amplitudes)):
                    freqs = np.array(params['modules'][mod][i]['frequency']['values'])
                    amps = np.array(params['modules'][mod][i]['magnitude']['counts']['raw'])
                    phases = np.array(params['modules'][mod][i]['phase']['values'])
                    
                    panel.update_data(mod, freqs, amps, phases)
                    panel.update_data_with_amp(mod, freqs, amps, phases, amplitudes[i])
                
                r_freq = params['modules'][mod]['resonances_hz']
                panel._use_loaded_resonances(mod, r_freq)
            
            # Tabify with Main dock by default
            main_dock = self.dock_manager.get_dock("main_plots")
            if main_dock:
                self.tabifyDockWidget(main_dock, dock)
            
            # Show the dock and activate it
            dock.show()
            dock.raise_()
        except Exception as e:
            print(f"Error in _load_network_analysis: {e}")
            traceback.print_exc()

    def _handle_analysis_completed(self, module_param: int, window_id: str): # Renamed module
        """
        Handle the completion of a network analysis sweep for a specific module.

        This method is called when a `NetworkAnalysisTask` signals completion.
        It updates the corresponding `NetworkAnalysisWindow` to mark the module's
        analysis as complete. If there are more amplitudes to sweep for this
        module, it starts the next `NetworkAnalysisTask`.

        Args:
            module_param (int): The module index for which the analysis completed.
            window_id (str): The unique identifier of the `NetworkAnalysisWindow`
                             associated with this analysis.
        """
        try:
            if window_id not in self.netanal_windows: return
            window_data = self.netanal_windows[window_id]
            window = window_data['window']
            window.complete_analysis(module_param)
            for task_key in list(self.netanal_tasks.keys()):
                if task_key.startswith(f"{window_id}_{module_param}_"):
                    self.netanal_tasks.pop(task_key, None)
            if module_param in window_data['amplitude_queues'] and window_data['amplitude_queues'][module_param]:
                window_data['current_amp_index'][module_param] += 1
                total_amps = len(window.original_params.get('amps', []))
                next_amp = window_data['amplitude_queues'][module_param][0]
                window.update_amplitude_progress(
                    module_param, 
                    window_data['current_amp_index'][module_param] + 1,
                    total_amps,
                    next_amp
                )
                if module_param in window.progress_bars:
                    window.progress_bars[module_param].setValue(0)
                    if window.progress_group:
                        window.progress_group.setVisible(True)
                self._start_next_amplitude_task(module_param, window.original_params, window_id)
        except Exception as e:
            print(f"Error in _handle_analysis_completed: {e}")
            traceback.print_exc()

    def check_connection(self, window_data, window_id):
        
        window_instance = window_data["window"]
        window_signals = window_data["signals"]
        
        count = window_signals.receivers(window_signals.progress)
        if count == 0:
            window_signals.progress.connect(
                lambda mod, prog: window_instance.update_progress(mod, prog), # mod, prog to avoid conflict
                QtCore.Qt.ConnectionType.QueuedConnection)
            window_signals.data_update.connect(
                lambda mod, freqs, amps, phases: window_instance.update_data(mod, freqs, amps, phases),
                QtCore.Qt.ConnectionType.QueuedConnection)
            window_signals.data_update_with_amp.connect(
                lambda mod, freqs, amps, phases, amp_val:  # amp_val to avoid conflict
                window_instance.update_data_with_amp(mod, freqs, amps, phases, amp_val),
                QtCore.Qt.ConnectionType.QueuedConnection)
            window_signals.completed.connect(
                lambda mod: self._handle_analysis_completed(mod, window_id),
                QtCore.Qt.ConnectionType.QueuedConnection)
            window_signals.error.connect(
                lambda error_msg: QtWidgets.QMessageBox.critical(window_instance, "Network Analysis Error", error_msg),
                QtCore.Qt.ConnectionType.QueuedConnection)
        else:
            return
            
        
    
    
    def _start_next_amplitude_task(self, module_param: int, params: dict, window_id: str): # Renamed module
        """
        Start the next network analysis task for a given module and amplitude.

        This is called iteratively when sweeping through multiple amplitudes.
        It retrieves the next amplitude from the queue for the specified module
        and window, then creates and starts a new `NetworkAnalysisTask`.

        Args:
            module_param (int): The module index for which to start the task.
            params (dict): The base parameters for the network analysis sweep.
                           The 'amplitude' for this specific task will be taken
                           from the queue.
            window_id (str): The unique identifier of the `NetworkAnalysisWindow`
                             associated with this analysis.
        """
        try:
            if window_id not in self.netanal_windows: return
            window_data = self.netanal_windows[window_id]
            self.check_connection(window_data, window_id)
            signals = window_data['signals']
            if module_param not in window_data['amplitude_queues'] or not window_data['amplitude_queues'][module_param]:
                return
            amplitude = window_data['amplitude_queues'][module_param].pop(0)
            task_params = params.copy()
            task_params['module'] = module_param
            module_specific_cable_length = params.get('module_cable_lengths', {}).get(module_param)
            if module_specific_cable_length is not None:
                task_params['cable_length'] = module_specific_cable_length
            task_key = f"{window_id}_{module_param}_amp_{amplitude}"
            # NetworkAnalysisTask from .tasks
            task = NetworkAnalysisTask(
                self.crs, module_param, task_params, signals, amplitude=amplitude
            )
            self.netanal_tasks[task_key] = task
            task.start()  # Start the QThread directly since NetworkAnalysisTask is now a QThread
        except Exception as e:
            print(f"Error in _start_next_amplitude_task: {e}")
            traceback.print_exc()

    def _rerun_network_analysis(self, params: dict, source_panel=None):
        """
        Re-run a network analysis for an existing NetworkAnalysisPanel.

        This method is called from a NetworkAnalysisPanel's _edit_parameters method.
        It stops any ongoing tasks for that window, clears its data and plots, 
        updates its parameters, and then restarts the analysis sequence.

        Args:
            params (dict): The new or updated parameters for the network analysis.
            source_panel: The NetworkAnalysisPanel requesting the re-run (optional, auto-detected if None)
        """
        try:
            if self.crs is None: QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available"); return
            
            # Find the panel that's calling this
            if source_panel is None:
                # Try to find it from params (fallback)
                for w_id, w_data in self.netanal_windows.items():
                    if w_data['window'].current_params == params:
                        source_panel = w_data['window']
                        break
            
            # Find window_id for this panel
            window_id = None
            for w_id, w_data in self.netanal_windows.items():
                if w_data['window'] == source_panel: 
                    window_id = w_id
                    break
            
            if not window_id: 
                print("No window_id found for panel")
                return
            window_data = self.netanal_windows[window_id]
            window = window_data['window']
            window.data.clear(); window.raw_data.clear()
            for mod, pbar in window.progress_bars.items(): 
                pbar.setValue(0) # Renamed module
            window.clear_plots(); window.set_params(params)
            selected_module_param = params.get('module') # Renamed
            if selected_module_param is None: modules_to_run = list(range(1, 9))
            elif isinstance(selected_module_param, list): modules_to_run = selected_module_param
            else: modules_to_run = [selected_module_param]
            for task_key in list(self.netanal_tasks.keys()):
                if task_key.startswith(f"{window_id}_"):
                    task = self.netanal_tasks.pop(task_key); task.stop()
            amplitudes = params.get('amps', [params.get('amp', DEFAULT_AMPLITUDE)]) # DEFAULT_AMPLITUDE from .utils
            window_data['amplitude_queues'] = {mod: list(amplitudes) for mod in modules_to_run}
            window_data['current_amp_index'] = {mod: 0 for mod in modules_to_run}
            if window.progress_group: window.progress_group.setVisible(True)
            for mod_iter in modules_to_run: # Renamed
                window.update_amplitude_progress(mod_iter, 1, len(amplitudes), amplitudes[0])
                self._start_next_amplitude_task(mod_iter, params, window_id)
        except Exception as e:
            print(f"Error in _rerun_network_analysis: {e}")
            traceback.print_exc()
        
    def _load_multisweep_dialog(self) -> None:
        """
        Show the dialog to configure and run multisweep analysis.
        """
        # Try to get active module
        default_dac_scales = {m: -0.5 for m in range(1, 9)}
        netanal_dialog = NetworkAnalysisDialog(self, modules=list(range(1, 9)), dac_scales=default_dac_scales)
        netanal_dialog.module_entry.setText(str(self.module))
        
        # Fetch DAC scales if CRS is available
        self._fetch_dac_scales_for_dialog(netanal_dialog)
        if self.crs is None:
            self.dac_scales = default_dac_scales.copy()
        
        active_module = self.module

    
        # Try to get resonances (if available)
        resonances = []
        if hasattr(self, "resonance_freqs") and active_module is not None:
            resonances = self.resonance_freqs.get(active_module, [])
    
        
        # --- Launch dialog even if no resonances yet ---
        dialog = MultisweepDialog( parent=netanal_dialog, 
                                   resonance_frequencies=resonances,   # may be []
                                   dac_scales=netanal_dialog.dac_scales,   # may be {}
                                   current_module=active_module,       # may be None
                                   initial_params=None ,                # nothing prefilled
                                   load_multisweep = True
                                 )

        
        if dialog.exec():
            params = dialog.get_parameters()
            if params:
                if "results_by_iteration" in params.keys():
                    self._load_multisweep_analysis(params)
                else:
                    if self.crs is None:
                        QtWidgets.QMessageBox.warning(self, "Offline Mode", "Cannot start new multisweep analysis without CRS hardware. Loading data only.")
                        return
                    self._start_multisweep_analysis(params)
    
    
    def handle_bias_from_file(self) -> None:
        """Slot for the 'Load Bias…' button in the main application window."""
        if self.crs is None and self.host != "OFFLINE":
            QtWidgets.QMessageBox.warning(self, "CRS Not Available", "Connect to a CRS before loading bias data.")
            return

        default_dac_scales = {m: -0.5 for m in range(1, 9)}
        netanal_dialog = NetworkAnalysisDialog(self, modules=list(range(1, 9)), dac_scales=default_dac_scales)
        netanal_dialog.module_entry.setText(str(self.module))
        
        # Fetch DAC scales if CRS is available
        self._fetch_dac_scales_for_dialog(netanal_dialog)
        if self.crs is None:
            self.dac_scales = default_dac_scales.copy()
        
        from .bias_kids_dialog import BiasKidsDialog
        dialog = BiasKidsDialog(self, self.module, True)

        if dialog.exec():
            params = dialog.get_load_param()
            if params:
                if "bias_kids_output" in params.keys():
                    self._set_and_plot_bias(params)
                else:
                    self._set_bias(params)

    def _set_bias(self, params):

        # span_hz = params.get("span_hz")
        bias_freqs = params.get("bias_frequencies")
        amplitudes = params.get("amplitudes")
        phases = params.get("phases")
        module = params.get("module")

        channels = np.arange(1, len(bias_freqs)+1).tolist()

        if module is None:
            QtWidgets.QMessageBox.critical(self, "Missing Module", "The file does not specify which module was biased.")
            return

        if bias_freqs and self.crs is not None:
            # nco_freq = ((min(bias_freqs) - span_hz / 2 + (max(bias_freqs) + span_hz / 2)) / 2
            nco_freq = (min(bias_freqs)  + max(bias_freqs)) / 2
            crs = self.crs
            asyncio.run(crs.set_nco_frequency(nco_freq, module=module)) #### Setting up the nco frequency ######
    
        if self.crs is not None:
            asyncio.run(self.apply_bias_output(self.crs, module, amplitudes, bias_freqs, channels, phases))
        else:
            print("[Offline] Skipping hardware bias application")
        
    async def apply_bias_output(self, crs, module: int, amplitudes: list, bias_freqs : list,
                                channels : list, phases : list) -> None:
    
        BASE_BAND_STEP_HZ = 298.0232238769531 #### Taken from bias_kids.py
        if not bias_freqs:
            return
        nco_freq = await crs.get_nco_frequency(module=module)
        async with crs.tuber_context() as ctx:
            for i in range(len(amplitudes)):

                quantized_bias = round(bias_freqs[i] / BASE_BAND_STEP_HZ) * BASE_BAND_STEP_HZ

                ctx.set_frequency(quantized_bias - nco_freq, channel=channels[i], module=module)
                
                ctx.set_amplitude(float(amplitudes[i]), channel=channels[i], module=module)
                
                ctx.set_phase(float(phases[i]), units=crs.UNITS.DEGREES, target=crs.TARGET.ADC, channel=channels[i], module=module)
            await ctx()

        print(f"[Bias] Bias applied for {len(bias_freqs)} frequencies")
    


    def _set_and_plot_bias(self, load_params):
        """
        Load bias data from file, apply bias to hardware, and display in a docked panel.
        
        Uses the unified _create_multisweep_panel_from_loaded_data helper for panel creation.
        """
        active_module = self.module 

        try:
            # Check if module in file matches active module
            params = load_params['initial_parameters']
            target_module = params.get('module')

            if active_module != target_module:
                QtWidgets.QMessageBox.warning(self, "Module Mismatch", 
                    "The module in file doesn't match the active module. The value will be changed.")
                # Update the module in params to use the active module
                load_params['initial_parameters']['module'] = active_module
                target_module = active_module

            if target_module is None: 
                QtWidgets.QMessageBox.critical(self, "Error", "Target module not specified for Bias.")
                return

            # Extract bias data for hardware application BEFORE creating panel
            bias_output = load_params.get('bias_kids_output')
            if not bias_output:
                QtWidgets.QMessageBox.critical(self, "Error", "No bias_kids_output in loaded file.")
                return
            
            bias_freqs = []
            amplitudes = []
            phases = []
            channels = []
            data_rod = {}
            
            for det_idx, det_data in bias_output.items():
                channel = int(det_data.get("bias_channel", det_idx))
                channels.append(channel)
                bias_freq = det_data.get("bias_frequency") or det_data.get("original_center_frequency")
                bias_freqs.append(bias_freq)
                amplitude = det_data.get("sweep_amplitude")
                amplitudes.append(amplitude)
                phase = det_data.get("optimal_phase_degrees", 0)
                phases.append(phase)
                if "rotation_tod" in det_data:
                    data_rod[channel] = det_data["rotation_tod"]

            # Apply bias to hardware if CRS is available
            if self.crs is not None:
                # Set NCO frequency to center of bias frequency range before applying bias
                if bias_freqs:
                    nco_freq = (min(bias_freqs) + max(bias_freqs)) / 2
                    asyncio.run(self.crs.set_nco_frequency(nco_freq, module=target_module))
                
                asyncio.run(self.apply_bias_output(self.crs, target_module, amplitudes, bias_freqs, channels, phases))
                    
                if data_rod:
                    asyncio.run(self.adjust_phase(target_module, channels, data_rod))
                    #print(f"[Bias] Refining the rotation")
                    self.noise_count = self.noise_count + 1
                    asyncio.run(self.adjust_phase(target_module, channels, data_rod, True))
            else:
                print("[Offline] Skipping hardware bias application and phase adjustment")

            # Use the unified helper method to create the panel and dock
            panel, dock, window_id, target_module = self._create_multisweep_panel_from_loaded_data(
                load_params, source_type="bias"
            )
            
            if panel is None:
                return  # Error already displayed by helper

            # Load noise data if present
            if load_params.get('noise_data') is not None:
                noise_data = load_params['noise_data']
                panel._get_spectrum(noise_data, use_loaded_noise=True)
            else:
                print("[Bias] There is no noise data in the file")
            
        except Exception as e:
            error_msg = f"Error displaying results: {type(e).__name__}: {str(e)}"
            print(error_msg, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            QtWidgets.QMessageBox.critical(self, "Bias Error", error_msg)



    async def adjust_phase(self, module, channels, data_rod, refine=False):
        if self.crs is None: 
            QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available for Bias.") 
            return
        else:
            crs = self.crs

        for channel in channels:
            samples = await self.collecting_samples_chan(crs, module, channel)
            
            phase_shift = self.calculate_shift(data_rod[channel], samples.i, samples.q, refine)
            
            self.phase_shifts.append(phase_shift)
            
            init_phase = await crs.get_phase(crs.UNITS.DEGREES, crs.TARGET.ADC, channel = channel, module = module)
            
            mod_phase = phase_shift + init_phase 
            
            await crs.set_phase(mod_phase, crs.UNITS.DEGREES, crs.TARGET.ADC, channel = channel, module = module)
            
            phase_after_change = await crs.get_phase(crs.UNITS.DEGREES, crs.TARGET.ADC, channel = channel, module = module)
            
            print(f"[Bias] Phase shift implemented of {phase_after_change} degrees for channel {channel}")            
            
    def calculate_shift(self, file_samples, noise_i, noise_q, refine):
        i_val_file = convert_roc_to_volts(file_samples.real)
        q_val_file = convert_roc_to_volts(file_samples.imag)
        phase_file = np.degrees(np.median(np.arctan(q_val_file/i_val_file)))

        
        i_val_noise = convert_roc_to_volts(np.array(noise_i))
        q_val_noise = convert_roc_to_volts(np.array(noise_q))
        phase_noise = np.degrees(np.median(np.arctan(q_val_noise/i_val_noise)))

        phase_shift =  phase_noise - phase_file

        q_noise_m = np.median(q_val_noise)
        i_noise_m = np.median(i_val_noise)

        q_file_m = np.median(q_val_file)
        i_file_m = np.median(i_val_file)


        if ((q_noise_m/q_file_m) < 0) and ((i_noise_m/i_file_m) < 0): ### incase there are in opposite quadrants
            print(f"[Bias] Opposite quadrant shifting by 180")
            phase_shift = phase_shift + 180

        return phase_shift
        
    async def collecting_samples_chan(self, crs, module, channel, total=100):
        samples = await crs.get_samples(total, average=False, channel=channel, module=module)

        if channel not in self.test_noise_samples:
            self.test_noise_samples[channel] = {}

        self.test_noise_samples[channel][self.noise_count] = np.array(samples.i) + np.array(samples.q) * 1j
        return samples

    def get_test_noise(self):
        return self.test_noise_samples

    def get_phase_shift(self):
        return self.phase_shifts
            
    
    def _netanal_error(self, error_msg: str):
        """Slot for network analysis error signals. Displays a critical message box."""
        QtWidgets.QMessageBox.critical(self, "Network Analysis Error", error_msg)
    
    def _handle_netanal_data_ready(self, modules: list, data: dict, panel=None):
        """
        Handle data_ready signal from NetworkAnalysisPanel for session auto-export.

        Args:
            modules: List of module IDs that were analyzed
            data: Full export data dictionary (may contain '_filename_override' key)
            panel: Optional reference to the NetworkAnalysisPanel for storing export filename
        """
        if not self.session_manager.is_active or not self.session_manager.auto_export_enabled:
            return

        # Create identifier from module list
        if len(modules) == 1:
            identifier = f"module{modules[0]}"
        else:
            identifier = f"modules_{'_'.join(map(str, modules))}"

        # Extract filename override if present (for overwriting previous export)
        filename_override = data.pop('_filename_override', None)

        # Export via session manager
        exported_path = self.session_manager.export_data(
            'netanal', identifier, data, filename_override=filename_override
        )
        
        # Store the exported filename on the panel for future overwrites
        if exported_path and panel and hasattr(panel, '_last_export_filename'):
            panel._last_export_filename = exported_path.name
        
        action = "updated" if filename_override else "exported"
        print(f"[Session] Auto-{action} network analysis: {identifier}")

    def _crs_init_success(self, message: str):
        """Slot for CRS initialization success signals. Displays an information message box."""
        QtWidgets.QMessageBox.information(self, "CRS Initialization Success", message)

    def _crs_init_error(self, error_msg: str):
        """Slot for CRS initialization error signals. Displays a critical message box."""
        QtWidgets.QMessageBox.critical(self, "CRS Initialization Error", error_msg)

    def _toggle_config(self, visible: bool):
        """Toggle the visibility of the configuration panel."""
        self.ctrl_panel.setVisible(visible)
        self.btn_toggle_cfg.setText("Hide Configuration" if visible else "Show Configuration")

    def _toggle_auto_scale(self, checked: bool):
        """
        Toggle auto-scaling for plots (excluding Time-Domain plots).

        Args:
            checked (bool): True to enable auto-scaling, False to disable.
        """
        self.auto_scale_plots = checked
        if hasattr(self, "plots"):
            for rowPlots in self.plots:
                for mode_key, pw in rowPlots.items(): # Renamed mode
                    if mode_key != "T": pw.enableAutoRange(pg.ViewBox.XYAxes, checked)

    def _toggle_iqmag(self):
        """
        Toggle the visibility of I, Q, and Magnitude traces in plots.

        Updates the `show_i`, `show_q`, and `show_m` attributes based on
        checkbox states and applies visibility changes to relevant plot curves.
        """
        self.show_i = self.cb_show_i.isChecked()
        self.show_q = self.cb_show_q.isChecked()
        self.show_m = self.cb_show_m.isChecked()
        if not hasattr(self, "curves"): return
        for rowCurves in self.curves:
            for mode_key in ("T", "F", "S"): # Renamed mode
                if mode_key in rowCurves:
                    subdict = rowCurves[mode_key]
                    for ch, cset in subdict.items():
                        if "I" in cset: cset["I"].setVisible(self.show_i)
                        if "Q" in cset: cset["Q"].setVisible(self.show_q)
                        if "Mag" in cset: cset["Mag"].setVisible(self.show_m)
    
    def _unit_mode_changed(self, button: QtWidgets.QRadioButton):
        """
        Handle unit mode changes from radio buttons.
        
        Updates the unit mode and real_units flag, checks for df calibration availability,
        and rebuilds the layout with appropriate labels.
        
        Args:
            button: The radio button that was clicked
        """
        button_id = self.unit_group.id(button)
        
        if button_id == 0:  # Counts
            self.unit_mode = "counts"
            self.real_units = False
        elif button_id == 1:  # Real Units
            self.unit_mode = "real"
            self.real_units = True
        elif button_id == 2:  # df Units
            self.unit_mode = "df"
            self.real_units = False  # df units are different from standard real units
            
            # Check if calibration data is available
            if not self.df_calibrations.get(self.module):
                QtWidgets.QMessageBox.warning(
                    self,
                    "df Calibration Not Available",
                    "df calibration data is not available for this module.\n\n"
                    "To use df units:\n"
                    "1. Run a multisweep analysis\n"
                    "2. Click 'Bias KIDs' in the multisweep window\n"
                    "3. The calibration data will be loaded automatically"
                )
                # Reset to counts mode
                self.rb_counts.setChecked(True)
                self.unit_mode = "counts"
                self.real_units = False
                return
        
        # Rebuild layout to update axis labels
        self._build_layout()
    
    def _handle_df_calibration_ready(self, module: int, df_calibrations: Dict[int, complex]):
        """
        Handle the df_calibration_ready signal from MultisweepWindow.
        
        Stores the calibration data for the specified module.
        
        Args:
            module: Module number
            df_calibrations: Dictionary mapping detector indices (1-based) to complex calibration factors
        """
        # Store calibration data for this module
        self.df_calibrations[module] = df_calibrations
        
        # Log to console instead of showing popup (already shown by MultisweepWindow)
        num_calibrated = len(df_calibrations)
        print(f"[Periscope] df calibration loaded for {num_calibrated} detectors on module {module}")

    def _handle_psd_toggle(self, checked: bool):
        """
        Handle toggling of PSD checkboxes, showing warning in mock mode.
        
        In mock mode, PSDs have inaccurate high frequency correction because
        the simulated data doesn't include the actual CIC filter response.
        
        Args:
            checked (bool): Whether the checkbox was checked
        """
        # Show warning only when enabling PSD in mock mode
        if self.is_mock_mode and checked:
            sender = self.sender()
            psd_type = "Single Sideband" if sender == self.cb_ssb else "Dual Sideband"
            
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Mock Mode PSD Warning")
            msg.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msg.setText(
                f"Warning: {psd_type} PSD in Mock Mode\n\n"
                "In mock mode, the high frequency correction for PSD plots will be inaccurate.\n\n"
                "The simulated data does not include the actual CIC decimation filter response, "
                "so the droop correction applied to PSD plots may not be representative of real hardware behavior.\n\n"
                "For accurate PSD analysis, please connect to a real CRS board."
            )
            msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            msg.exec()
        
        # Always rebuild layout after toggle
        self._build_layout()

    def _show_mock_config_dialog(self):
        """
        Show the mock configuration dialog for adjusting simulated KID parameters.
        
        Only available when running in mock mode (connected to localhost).
        """
        if not self.is_mock_mode:
            return
            
        # Get current configuration if available
        current_config = self.mock_config or self._get_current_mock_config()
        
        # Show dialog
        dialog = MockConfigurationDialog(self, current_config)
        if dialog.exec():
            # Get new configuration
            new_config = dialog.get_configuration()
            self.mock_config = new_config
            
            # Apply configuration to the mock CRS
            self._apply_mock_configuration(new_config)
            
    def _get_current_mock_config(self) -> dict:
        """
        Get the current mock configuration from the unified SoT (mock_config).
        
        Returns:
            dict: Current configuration values compatible with the dialog and MockCRS
        """
        try:
            return mc.defaults()
        except Exception as e:
            print(f"Error getting current mock config: {e}")
            return mc.defaults()
            
    def _apply_mock_configuration(self, config: dict):
        """
        Apply the user's configuration to the mock CRS system.
        
        This sends the configuration to the server-side MockCRS to regenerate
        resonators with the new parameters.
        
        Args:
            config: Dictionary of configuration values
        """
        try:
            if self.crs is not None and hasattr(self.crs, 'generate_resonators'):
                import asyncio
                
                # Use the existing event loop or create new one
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                try:
                    # Apply configuration to server
                    future = asyncio.ensure_future(self.crs.generate_resonators(config))
                    resonator_count = loop.run_until_complete(future)

                    # If QP pulses are currently active, re-apply the same mode with updated parameters
                    try:
                        if self.crs is not None and hasattr(self.crs, "set_pulse_mode") and getattr(self, "qp_pulse_mode", "none") in ("periodic", "random"):
                            try:
                                cfg = mc.apply_overrides(self.mock_config) if self.mock_config else mc.defaults()
                            except Exception:
                                cfg = mc.defaults()
                            if self.qp_pulse_mode == "periodic":
                                loop.run_until_complete(self.crs.set_pulse_mode(
                                    "periodic",
                                    period=cfg.get("pulse_period", 10.0),
                                    tau_rise=cfg.get("pulse_tau_rise", 1e-6),
                                    tau_decay=cfg.get("pulse_tau_decay", 1e-1),
                                    amplitude=cfg.get("pulse_amplitude", 2.0),
                                    resonators=cfg.get("pulse_resonators", "all"),
                                ))
                                print("[Periscope] Re-applied periodic QP pulses with updated parameters")
                            elif self.qp_pulse_mode == "random":
                                loop.run_until_complete(self.crs.set_pulse_mode(
                                    "random",
                                    probability=cfg.get("pulse_probability", 0.001),
                                    tau_rise=cfg.get("pulse_tau_rise", 1e-6),
                                    tau_decay=cfg.get("pulse_tau_decay", 1e-1),
                                    amplitude=cfg.get("pulse_amplitude", 2.0),
                                    resonators=cfg.get("pulse_resonators", "all"),
                                    # Random amplitude distribution
                                    random_amp_mode=cfg.get("pulse_random_amp_mode", "fixed"),
                                    random_amp_min=cfg.get("pulse_random_amp_min", 1.5),
                                    random_amp_max=cfg.get("pulse_random_amp_max", 3.0),
                                    random_amp_logmean=cfg.get("pulse_random_amp_logmean", 0.7),
                                    random_amp_logsigma=cfg.get("pulse_random_amp_logsigma", 0.3),
                                ))
                                print("[Periscope] Re-applied random QP pulses with updated parameters")
                    except Exception as e2:
                        print(f"[Periscope] Warning: failed to re-apply QP pulse mode after reconfigure: {e2}")

                    print(f"Regenerated {resonator_count} resonators with new parameters")
                    QtWidgets.QMessageBox.information(self, "Configuration Applied", 
                                                   f"Mock KID parameters have been updated.\n"
                                                   f"Generated {resonator_count} resonators.")
                except Exception as e:
                    import traceback
                    print(f"Error regenerating resonators: {e}")
                    traceback.print_exc()
                    QtWidgets.QMessageBox.critical(self, "Configuration Error", 
                                                 f"Failed to regenerate resonators:\n{str(e)}\n\n"
                                                 f"Details:\n{traceback.format_exc()}")
            else:
                QtWidgets.QMessageBox.warning(self, "Configuration Warning", 
                                            "CRS object not available or doesn't support mock configuration.")
                
        except Exception as e:
            import traceback
            print(f"Error applying mock configuration: {e}")
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Configuration Error", 
                                          f"Failed to apply mock configuration:\n{str(e)}\n\n"
                                          f"Details:\n{traceback.format_exc()}")

    def _toggle_qp_pulses(self):
        """
        Toggle quasiparticle pulse modes in mock mode.
        
        Cycles through: Off → Periodic → Random → Off
        """
        if not self.is_mock_mode or self.crs is None:
            return
            
        # Check if the CRS has pulse control methods
        if not hasattr(self.crs, 'set_pulse_mode'):
            QtWidgets.QMessageBox.warning(
                self, 
                "Pulse System Not Available", 
                "The mock CRS does not support pulse functionality.\n"
                "Please ensure you're using the latest version of the mock system."
            )
            return
        
        # Capture reference to avoid closure issues
        crs = self.crs
        
        # Run the async pulse mode setting in a separate thread
        import asyncio
        import threading
        
        def run_async_pulse_mode():
            try:
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Use current mock configuration (dialog) or defaults
                try:
                    cfg = mc.apply_overrides(self.mock_config) if self.mock_config else mc.defaults()
                except Exception:
                    cfg = mc.defaults()
                
                # Cycle through pulse modes
                if self.qp_pulse_mode == 'none':
                    # Switch to periodic mode
                    self.qp_pulse_mode = 'periodic'
                    
                    # Configure periodic pulses using unified config
                    loop.run_until_complete(crs.set_pulse_mode(
                        'periodic',
                        period=cfg.get('pulse_period', 10.0),
                        tau_rise=cfg.get('pulse_tau_rise', 1e-6),
                        tau_decay=cfg.get('pulse_tau_decay', 1e-1),
                        amplitude=cfg.get('pulse_amplitude', 2.0),
                        resonators=cfg.get('pulse_resonators', 'all')
                    ))
                    # Sync SoT
                    try:
                        if self.mock_config is None:
                            self.mock_config = mc.defaults()
                        self.mock_config['pulse_mode'] = 'periodic'
                    except Exception:
                        pass
                    print(f"[Periscope] Enabled periodic QP pulses (period={cfg.get('pulse_period', 10.0)}s)")
                    
                elif self.qp_pulse_mode == 'periodic':
                    # Switch to random mode
                    self.qp_pulse_mode = 'random'
                    
                    # Configure random pulses using unified config
                    loop.run_until_complete(crs.set_pulse_mode(
                        'random',
                        probability=cfg.get('pulse_probability', 0.001),
                        tau_rise=cfg.get('pulse_tau_rise', 1e-6),
                        tau_decay=cfg.get('pulse_tau_decay', 1e-1),
                        amplitude=cfg.get('pulse_amplitude', 2.0),
                        resonators=cfg.get('pulse_resonators', 'all'),
                        # Random amplitude distribution (random mode)
                        random_amp_mode=cfg.get('pulse_random_amp_mode', 'fixed'),
                        random_amp_min=cfg.get('pulse_random_amp_min', 1.5),
                        random_amp_max=cfg.get('pulse_random_amp_max', 3.0),
                        random_amp_logmean=cfg.get('pulse_random_amp_logmean', 0.7),
                        random_amp_logsigma=cfg.get('pulse_random_amp_logsigma', 0.3),
                    ))
                    # Sync SoT
                    try:
                        if self.mock_config is None:
                            self.mock_config = mc.defaults()
                        self.mock_config['pulse_mode'] = 'random'
                    except Exception:
                        pass
                    print(f"[Periscope] Enabled random QP pulses (prob={cfg.get('pulse_probability', 0.001)}/s)")
                    
                elif self.qp_pulse_mode == 'random':
                    # Switch back to off
                    self.qp_pulse_mode = 'none'
                    
                    # Disable pulses
                    loop.run_until_complete(crs.set_pulse_mode('none'))
                    try:
                        if self.mock_config is None:
                            self.mock_config = mc.defaults()
                        self.mock_config['pulse_mode'] = 'none'
                    except Exception:
                        pass
                    print("[Periscope] Disabled QP pulses")
                
                # Update UI on main thread
                QtCore.QMetaObject.invokeMethod(
                    self, "_update_pulse_button_ui", 
                    QtCore.Qt.ConnectionType.QueuedConnection
                )
                
            except Exception as e:
                print(f"[Periscope] Error setting pulse mode: {e}")
                # Show error on main thread
                QtCore.QMetaObject.invokeMethod(
                    self, "_show_pulse_error", 
                    QtCore.Qt.ConnectionType.QueuedConnection,
                    QtCore.Q_ARG(str, str(e))
                )
            finally:
                loop.close()
        
        # Start the async operation in a separate thread
        thread = threading.Thread(target=run_async_pulse_mode, daemon=True)
        thread.start()
    
    @QtCore.pyqtSlot()
    def _update_pulse_button_ui(self):
        """Update the pulse button UI after async operation completes."""
        # Update button text based on current mode
        if self.qp_pulse_mode == 'periodic':
            self.btn_qp_pulses.setText("QP Pulses: Periodic")
        elif self.qp_pulse_mode == 'random':
            self.btn_qp_pulses.setText("QP Pulses: Random")
        else:
            self.btn_qp_pulses.setText("QP Pulses: Off")
        
        # Update tooltip to show current state
        # Compose tooltip including active parameters
        try:
            cfg = mc.apply_overrides(self.mock_config) if self.mock_config else mc.defaults()
        except Exception:
            cfg = mc.defaults()
        extra = ""
        if self.qp_pulse_mode == 'periodic':
            extra = f"\nPeriod={cfg.get('pulse_period', 10.0)} s, tau_rise={cfg.get('pulse_tau_rise', 1e-6)} s, tau_decay={cfg.get('pulse_tau_decay', 1e-1)} s, amp={cfg.get('pulse_amplitude', 2.0)}, res={cfg.get('pulse_resonators', 'all')}"
        elif self.qp_pulse_mode == 'random':
            ram = cfg.get('pulse_random_amp_mode', 'fixed')
            if ram == 'uniform':
                amin = cfg.get('pulse_random_amp_min', 1.5)
                amax = cfg.get('pulse_random_amp_max', 3.0)
                extra = (
                    f"\nProb={cfg.get('pulse_probability', 0.001)}/s, "
                    f"tau_rise={cfg.get('pulse_tau_rise', 1e-6)} s, "
                    f"tau_decay={cfg.get('pulse_tau_decay', 1e-1)} s, "
                    f"ampMode=uniform[{amin},{amax}], "
                    f"res={cfg.get('pulse_resonators', 'all')}"
                )
            elif ram == 'lognormal':
                mu = cfg.get('pulse_random_amp_logmean', 0.7)
                sigma = cfg.get('pulse_random_amp_logsigma', 0.3)
                extra = (
                    f"\nProb={cfg.get('pulse_probability', 0.001)}/s, "
                    f"tau_rise={cfg.get('pulse_tau_rise', 1e-6)} s, "
                    f"tau_decay={cfg.get('pulse_tau_decay', 1e-1)} s, "
                    f"ampMode=lognormal[μ={mu},σ={sigma}], "
                    f"res={cfg.get('pulse_resonators', 'all')}"
                )
            else:
                extra = (
                    f"\nProb={cfg.get('pulse_probability', 0.001)}/s, "
                    f"tau_rise={cfg.get('pulse_tau_rise', 1e-6)} s, "
                    f"tau_decay={cfg.get('pulse_tau_decay', 1e-1)} s, "
                    f"amp=fixed({cfg.get('pulse_amplitude', 2.0)}), "
                    f"res={cfg.get('pulse_resonators', 'all')}"
                )
        tooltip_text = (
            f"Toggle quasiparticle pulses in mock mode\n"
            f"Current: {self.qp_pulse_mode.title()}\n"
            f"Cycles through: Off → Periodic → Random → Off{extra}"
        )
        self.btn_qp_pulses.setToolTip(tooltip_text)
    
    @QtCore.pyqtSlot(str)
    def _show_pulse_error(self, error_msg):
        """Show pulse control error on main thread."""
        QtWidgets.QMessageBox.critical(
            self,
            "Pulse Control Error",
            f"Failed to set pulse mode.\n\nError: {error_msg}"
        )
    
    
    def _create_help_menu(self):
        help_menu = self.menuBar().addMenu("&Help")
        help_action = QtGui.QAction("&Periscope Help", self)
        help_action.setToolTip("Open Help Dialog")
        help_action.triggered.connect(self._show_help)
        help_menu.addAction(help_action)
        

    def _create_view_menu(self):
        """
        Create the View menu for display settings like dark mode.
        """
        view_menu = self.menuBar().addMenu("&View")
        
        # Dark Mode toggle action
        self.dark_mode_action = QtGui.QAction("&Dark Mode", self)
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.setChecked(self.dark_mode)
        self.dark_mode_action.setToolTip("Switch between dark and light UI themes")
        self.dark_mode_action.triggered.connect(self._toggle_dark_mode)
        self.dark_mode_action.triggered.connect(self._update_console_style)
        view_menu.addAction(self.dark_mode_action)


    def _create_jupyter_menu(self):
        """
        Create the Jupyter menu for display settings like jupyter notebook.
        """

        jupyter_menu = self.menuBar().addMenu("&Jupyter")

        self.interactive_session_action = QtGui.QAction("Interactive iPython &Session", self)
        self.interactive_session_action.setToolTip("Toggle an embedded iPython interactive session.")
        self.interactive_session_action.triggered.connect(self._toggle_interactive_session)
        if not QTCONSOLE_AVAILABLE or self.crs is None:
            self.interactive_session_action.setEnabled(False)
            if not QTCONSOLE_AVAILABLE:
                self.interactive_session_action.setToolTip("Interactive session disabled: qtconsole/ipykernel not installed.")
            else:
                self.interactive_session_action.setToolTip("Interactive session disabled: CRS object not available.")
        jupyter_menu.addAction(self.interactive_session_action)
        
        # Jupyter Notebook panel
        notebook_action = QtGui.QAction("📓 &Jupyter Notebook", self)
        notebook_action.setShortcut("Ctrl+Shift+J")
        notebook_action.setToolTip("Open an embedded Jupyter notebook for interactive analysis")
        notebook_action.triggered.connect(lambda: self._toggle_notebook_panel())
        jupyter_menu.addAction(notebook_action)
        

    def _create_window_menu(self):
        """
        Create the Window menu for managing dockable analysis panels.
        
        Provides options to organize, tile, and manage all open dock widgets.
        """
        # Create Window menu
        window_menu = self.menuBar().addMenu("&Window")
        
        # Tile Horizontally action (QAction is in QtGui in PyQt6)
        tile_h_action = QtGui.QAction("Tile &Horizontally", self)
        tile_h_action.setToolTip("Arrange all dock widgets side by side")
        tile_h_action.triggered.connect(self._tile_docks_horizontally)
        window_menu.addAction(tile_h_action)
        
        # Tile Vertically action
        tile_v_action = QtGui.QAction("Tile &Vertically", self)
        tile_v_action.setToolTip("Arrange all dock widgets stacked vertically")
        tile_v_action.triggered.connect(self._tile_docks_vertically)
        window_menu.addAction(tile_v_action)
        
        window_menu.addSeparator()
        
        # Tabify All action
        tabify_action = QtGui.QAction("&Tabify All Panels", self)
        tabify_action.setToolTip("Group all dock widgets together as tabs")
        tabify_action.triggered.connect(self._tabify_all_docks)
        window_menu.addAction(tabify_action)
        
        # Float All action
        float_action = QtGui.QAction("&Float All Panels", self)
        float_action.setToolTip("Undock all panels into separate windows")
        float_action.triggered.connect(self._float_all_docks)
        window_menu.addAction(float_action)
        
        # Dock All action
        dock_action = QtGui.QAction("&Dock All Panels", self)
        dock_action.setToolTip("Dock all floating panels back into the main window")
        dock_action.triggered.connect(self._dock_all_docks)
        window_menu.addAction(dock_action)
        
        window_menu.addSeparator()
        
        # Close All Analysis Windows action
        close_all_action = QtGui.QAction("&Close All Analysis Panels", self)
        close_all_action.setToolTip("Close all network analysis, multisweep, and detector digest panels")
        close_all_action.triggered.connect(self._close_all_analysis_panels)
        window_menu.addAction(close_all_action)
        
        window_menu.addSeparator()
        
        # List Panels action (dynamic submenu)
        self.list_panels_menu = window_menu.addMenu("&Show/Hide Panels")
        self.list_panels_menu.aboutToShow.connect(self._update_panels_list_menu)
    
    def _tile_docks_horizontally(self):
        """Tile all dock widgets horizontally."""
        dock_ids = self.dock_manager.get_all_dock_ids()
        if len(dock_ids) >= 2:
            self.dock_manager.tile_docks_horizontally(dock_ids)
    
    def _tile_docks_vertically(self):
        """Tile all dock widgets vertically."""
        dock_ids = self.dock_manager.get_all_dock_ids()
        if len(dock_ids) >= 2:
            self.dock_manager.tile_docks_vertically(dock_ids)
    
    def _tabify_all_docks(self):
        """Tabify all dock widgets together."""
        dock_ids = self.dock_manager.get_all_dock_ids()
        if len(dock_ids) >= 2:
            self.dock_manager.tabify_docks(dock_ids)
    
    def _float_all_docks(self):
        """Float all dock widgets."""
        self.dock_manager.float_all_docks()
    
    def _dock_all_docks(self):
        """Dock all floating widgets."""
        self.dock_manager.dock_all_docks()
    
    def _close_all_analysis_panels(self):
        """Close all analysis-related dock panels."""
        self.dock_manager.close_all_docks()
    
    def _update_panels_list_menu(self):
        """Update the Show/Hide Panels submenu with current dock widgets."""
        self.list_panels_menu.clear()
        
        dock_ids = self.dock_manager.get_all_dock_ids()
        
        if not dock_ids:
            no_panels_action = QtGui.QAction("(No panels open)", self)
            no_panels_action.setEnabled(False)
            self.list_panels_menu.addAction(no_panels_action)
            return
        
        # Add an action for each dock widget
        for dock_id in dock_ids:
            dock = self.dock_manager.get_dock(dock_id)
            if dock:
                action = QtGui.QAction(dock.windowTitle(), self)
                action.setCheckable(True)
                action.setChecked(dock.isVisible())
                
                # Connect to toggle visibility
                action.triggered.connect(
                    lambda checked, d=dock: d.setVisible(checked)
                )
                
                self.list_panels_menu.addAction(action)

    # ─────────────────────────────────────────────────────────────────
    # Session Management Methods
    # ─────────────────────────────────────────────────────────────────
    
    def _create_session_menu(self):
        """
        Create the Session menu in the menu bar.
        
        Provides options to start/load sessions, toggle auto-export,
        manually export data, and end sessions.
        """
        session_menu = self.menuBar().addMenu("&Session")
        
        # Start New Session
        start_action = QtGui.QAction("&Start New Session...", self)
        start_action.setShortcut("Ctrl+Shift+N")
        start_action.setToolTip("Start a new session to auto-export analysis data")
        start_action.triggered.connect(self._start_new_session)
        session_menu.addAction(start_action)
        
        # Load Session
        load_action = QtGui.QAction("&Load Session...", self)
        load_action.setShortcut("Ctrl+Shift+O")
        load_action.setToolTip("Load an existing session folder")
        load_action.triggered.connect(self._load_session)
        session_menu.addAction(load_action)
        
        session_menu.addSeparator()
        
        # Auto-Export Toggle
        self.auto_export_action = QtGui.QAction("&Auto-Export Enabled", self)
        self.auto_export_action.setCheckable(True)
        self.auto_export_action.setChecked(True)
        self.auto_export_action.setToolTip("Toggle automatic export of analysis data when session is active")
        self.auto_export_action.triggered.connect(self._toggle_auto_export)
        session_menu.addAction(self.auto_export_action)
        
        session_menu.addSeparator()
        
        # End Session
        end_action = QtGui.QAction("&End Session", self)
        end_action.setToolTip("End the current session")
        end_action.triggered.connect(self._end_session)
        session_menu.addAction(end_action)
    
    def _add_session_browser_dock(self):
        """
        Add the session browser panel as a collapsible dock on the left side.
        Collapsing reduces the dock to a thin rail with a '+' button.
        Expanding restores the previous width and full content.
        """
    
        # ---- Constants ----
        COLLAPSED_WIDTH = 28
        EXPANDED_MIN_WIDTH = 200
        EXPANDED_MAX_WIDTH = 400
        DEFAULT_EXPANDED_WIDTH = 280
    
        # ---- Create the session browser panel ----
        self.session_browser = SessionBrowserPanel(self.session_manager, self)
        self.session_browser.apply_theme(self.dark_mode)
        self.session_browser.file_load_requested.connect(self._load_session_file)
    
        # ---- Create dock widget ----
        dock = QtWidgets.QDockWidget("Session Files", self)
        dock.setObjectName("session_browser_dock")
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable)
    
        # ---- Expanded widget ----
        dock._expanded_widget = self.session_browser
    
        # ---- Collapsed placeholder widget (thin rail with '+') ----
        collapsed_widget = QtWidgets.QWidget()
        collapsed_layout = QtWidgets.QVBoxLayout(collapsed_widget)
        collapsed_layout.setContentsMargins(0, 0, 0, 0)
        collapsed_layout.setSpacing(0)
    
        expand_btn = QtWidgets.QToolButton()
        expand_btn.setText("+")
        expand_btn.setToolTip("Show Session Files")
        expand_btn.setAutoRaise(True)
        expand_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        expand_btn.setStyleSheet("""
            QToolButton {
                font-size: 18px;
                font-weight: bold;
            }
        """)
    
        collapsed_layout.addStretch(1)
        collapsed_layout.addWidget(
            expand_btn,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )
        collapsed_layout.addStretch(1)
    
        dock._collapsed_widget = collapsed_widget
    
        # ---- Internal state ----
        dock._collapsed = False
        dock._expanded_width = DEFAULT_EXPANDED_WIDTH
    
        # ---- Collapse / Expand logic ----
        def collapse_dock():
            dock._expanded_width = dock.width()
            dock.setTitleBarWidget(None)
            dock.setWindowTitle("")
            dock.setWidget(dock._collapsed_widget)
            dock.setMinimumWidth(COLLAPSED_WIDTH)
            dock.setMaximumWidth(COLLAPSED_WIDTH)
            dock._collapsed = True
    
        def expand_dock():
            dock.setWidget(dock._expanded_widget)
            dock.setTitleBarWidget(title_bar)
            dock.setWindowTitle("Session Files")
            dock.setMinimumWidth(EXPANDED_MIN_WIDTH)
            dock.setMaximumWidth(EXPANDED_MAX_WIDTH)
            dock._collapsed = False
    
        expand_btn.clicked.connect(expand_dock)
    
        # ---- Custom title bar ----
        title_bar = QtWidgets.QWidget()
        title_layout = QtWidgets.QHBoxLayout(title_bar)
        title_layout.setContentsMargins(6, 2, 6, 2)
    
        title_label = QtWidgets.QLabel(dock.windowTitle())
        title_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignVCenter |
            QtCore.Qt.AlignmentFlag.AlignLeft
        )
    
        toggle_btn = QtWidgets.QToolButton()
        toggle_btn.setAutoRaise(True)
    
        toggle_btn.setIcon(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_TitleBarMinButton
            )
        )
        toggle_btn.setToolTip("Collapse Session Files")
    
        def toggle_collapse():
            if dock._collapsed:
                expand_dock()
            else:
                collapse_dock()
    
        toggle_btn.clicked.connect(toggle_collapse)
    
        title_layout.addWidget(title_label)
        title_layout.addStretch(1)
        title_layout.addWidget(toggle_btn)
        dock.setTitleBarWidget(title_bar)
    
        # ---- Initial widget and size ----
        dock.setWidget(dock._expanded_widget)
        dock.setMinimumWidth(EXPANDED_MIN_WIDTH)
        dock.setMaximumWidth(EXPANDED_MAX_WIDTH)
        dock.resize(DEFAULT_EXPANDED_WIDTH, dock.height())
        # update_toggle_button()
    
        # ---- Add to main window ----
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
            dock
        )
    
        # ---- Split with main dock if present ----
        main_dock = self.dock_manager.get_dock("main_plots")
        if main_dock:
            self.splitDockWidget(
                dock,
                main_dock,
                QtCore.Qt.Orientation.Horizontal
            )
    
        # ---- Store reference ----
        self.session_browser_dock = dock
    
        # ---- Start expanded (not collapsed) ----
        # Session browser is visible by default to help users see their session files
    
    def _start_new_session(self):
        """
        Start a new session with folder selection dialog.
        
        Shows a folder selection dialog, then lets the user customize
        the session folder name before creating it.
        """
        # Show folder selection dialog
        # Use Qt dialog (not native) to prevent hanging on some systems
        base_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Session Location",
            "",  # Start in current directory
            QtWidgets.QFileDialog.Option.ShowDirsOnly | QtWidgets.QFileDialog.Option.DontUseNativeDialog
        )
        
        if not base_path:
            return
        
        # Generate default folder name with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"session_{timestamp}"
        
        # Let user rename
        folder_name, ok = QtWidgets.QInputDialog.getText(
            self, 
            "Session Name",
            "Enter session folder name:",
            QtWidgets.QLineEdit.EchoMode.Normal,
            default_name
        )
        
        if ok and folder_name:
            try:
                self.session_manager.start_session(base_path, folder_name)
                
                # Show the session browser dock
                if hasattr(self, 'session_browser_dock'):
                    self.session_browser_dock.show()
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Session Error",
                    f"Failed to start session:\n{str(e)}"
                )
    
    def _load_session(self):
        """
        Load an existing session folder.
        
        Shows a folder selection dialog to choose an existing session folder.
        """
        # Use Qt dialog (not native) to prevent hanging on some systems
        session_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Session Folder",
            "",
            QtWidgets.QFileDialog.Option.ShowDirsOnly | QtWidgets.QFileDialog.Option.DontUseNativeDialog
        )
        
        if session_path:
            success = self.session_manager.load_session(session_path)
            
            if success:
                # Show the session browser dock
                if hasattr(self, 'session_browser_dock'):
                    self.session_browser_dock.show()
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Load Session Failed",
                    f"Could not load session from:\n{session_path}"
                )
    
    def _show_session_startup_dialog(self):
        """
        Show the unified startup dialog to configure connection and session.
        
        This is shown automatically when Periscope launches, unless a session
        has already been configured (e.g., from command-line launch).
        """
        # Skip dialog if session is already active or configured
        # (e.g., when launched via command-line with startup dialog)
        if self.session_manager.is_active:
            return
        
        dialog = UnifiedStartupDialog(self)
        if dialog.exec():
            config = dialog.get_configuration()
            
            # Handle connection mode
            connection_mode = config['connection_mode']
            session_mode = config['session_mode']
            
            # For now, just handle session management
            # TODO: In the future, implement offline mode and use the IP/module from config
            
            if session_mode == UnifiedStartupDialog.SESS_NEW:
                # Start new session with provided path and folder name
                try:
                    self.session_manager.start_session(
                        config['session_path'], 
                        config['session_folder_name']
                    )
                    # Show the session browser dock
                    if hasattr(self, 'session_browser_dock'):
                        self.session_browser_dock.show()
                except Exception as e:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Session Error",
                        f"Failed to start session:\n{str(e)}"
                    )
            elif session_mode == UnifiedStartupDialog.SESS_LOAD:
                # Load existing session
                success = self.session_manager.load_session(config['session_path'])
                if success:
                    # Show the session browser dock
                    if hasattr(self, 'session_browser_dock'):
                        self.session_browser_dock.show()
                else:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Load Session Failed",
                        f"Could not load session from:\n{config['session_path']}"
                    )
            elif session_mode == UnifiedStartupDialog.SESS_NONE:
                # Continue without session - disable auto-export
                self.session_manager.auto_export_enabled = False
                self.auto_export_action.setChecked(False)
                print("[Session] Continuing without session (auto-export disabled)")
            
            # TODO: Handle connection mode (CONN_OFFLINE, etc.)
            if connection_mode == UnifiedStartupDialog.CONN_OFFLINE:
                print("[Connection] Offline mode selected - feature not yet implemented")
                # Future: Initialize in offline mode, disable CRS-dependent features
    
    def _toggle_auto_export(self, checked: bool):
        """
        Toggle auto-export on/off.
        
        Args:
            checked: True to enable auto-export, False to disable
        """
        self.session_manager.auto_export_enabled = checked
        
        status = "enabled" if checked else "disabled"
        print(f"[Session] Auto-export {status}")
    
    def _end_session(self):
        """
        End the current session with confirmation.
        """
        if not self.session_manager.is_active:
            QtWidgets.QMessageBox.information(
                self,
                "No Active Session",
                "There is no active session to end."
            )
            return
        
        # Confirm with user
        reply = QtWidgets.QMessageBox.question(
            self,
            "End Session",
            f"End the current session?\n\n"
            f"Session: {self.session_manager.session_name}\n"
            f"Files exported: {self.session_manager.export_count}",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No
        )
        
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.session_manager.end_session()
    
    def _load_session_file(self, file_path: str):
        """
        Load a session file into a new analysis panel.

        This is called when the user double-clicks a file in the session browser.

        Args:
            file_path: Path to the file to load (.pkl or .ipynb)
        """
        from pathlib import Path
        
        # Handle notebook files specially
        if file_path.endswith('.ipynb'):
            self._open_notebook_file(file_path)
            return
        
        # Identify file type from filename
        file_type = self.session_manager.identify_file_type(file_path)
        
        if file_type is None:
            # Unknown file type - open with system default application
            self._open_file_with_system_default(file_path)
            return
        
        # Load the data
        data = self.session_manager.load_file(file_path)
        
        if data is None:
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load file:\n{file_path}"
            )
            return
        
        # Create appropriate panel based on file type
        try:
            if file_type == 'netanal':
                self._load_netanal_from_session(data, file_path)
            elif file_type == 'multisweep':
                self._load_multisweep_from_session(data, file_path)
            elif file_type == 'bias':
                self._load_bias_from_session(data, file_path)
            elif file_type == 'noise':
                self._load_noise_from_session(data, file_path)
            else:
                QtWidgets.QMessageBox.information(
                    self,
                    "File Loaded",
                    f"Loaded {file_type} data from session.\n"
                    f"(Panel creation for this type not yet implemented)"
                )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                f"Error creating panel for {file_type} data:\n{str(e)}"
            )
            traceback.print_exc()
    
    def _load_netanal_from_session(self, data: dict, file_path: str):
        """Load network analysis data from session file into a new panel."""
        # Check if data has the expected structure
        if 'parameters' not in data and 'modules' not in data:
            # Try to wrap it in expected format
            QtWidgets.QMessageBox.information(
                self,
                "Network Analysis Loaded",
                f"Loaded network analysis data.\n"
                f"File: {file_path}\n\n"
                "(Direct panel display not yet implemented for this format)"
            )
            return
        
        # Use existing load mechanism
        self._load_network_analysis(data)
    
    def _load_multisweep_from_session(self, data: dict, file_path: str):
        """Load multisweep data from session file into a new panel."""
        if 'results_by_iteration' not in data:
            QtWidgets.QMessageBox.information(
                self,
                "Multisweep Loaded",
                f"Loaded multisweep data.\n"
                f"File: {file_path}\n\n"
                "(Direct panel display not yet implemented for this format)"
            )
            return
        
        # Use existing load mechanism
        self._load_multisweep_analysis(data)
    
    def _load_bias_from_session(self, data: dict, file_path: str):
        """
        Load bias data from session file - show dialog with options.
        
        When double-clicking a bias file, the user gets to choose whether to:
        - Set bias (apply to hardware only)
        - Set + Plot bias (apply to hardware and create visualization panel)
        """
        if 'results_by_iteration' not in data:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Bias File",
                f"File does not contain multisweep data:\n{file_path}\n\n"
                "Cannot load this bias file."
            )
            return
        
        # Show the same dialog that the "Load Bias" button shows
        from .bias_kids_dialog import BiasKidsDialog
        dialog = BiasKidsDialog(self, self.module, load_bias=True, loaded_data=data)
        
        if dialog.exec():
            params = dialog.get_load_param()
            if params:
                if "bias_kids_output" in params:
                    # User chose "Set + Plot Bias"
                    self._set_and_plot_bias(params)
                else:
                    # User chose "Set Bias" (hardware only)
                    self._set_bias(params)
    
    def _load_noise_from_session(self, data: dict, file_path: str):
        """
        Load noise spectrum data from session file.
        
        Noise files contain complete multisweep data plus noise spectrum data.
        Creates a MultisweepPanel, opens the DetectorDigestPanel (fit), and 
        opens a separate NoiseSpectrumPanel for the noise visualization.
        """
        if 'results_by_iteration' not in data:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Noise File",
                f"File does not contain multisweep data:\n{file_path}\n\n"
                "Cannot load this noise file."
            )
            return
        
        # Use the unified helper to create the multisweep panel
        panel, dock, window_id, target_module = self._create_multisweep_panel_from_loaded_data(
            data, source_type="noise"
        )
        
        if panel is None:
            return  # Error already displayed by helper
        
        # Auto-launch detector digest panel (fit panel) by simulating a double-click
        # This is the same logic used in _load_multisweep_analysis
        iteration_params = data.get('results_by_iteration', [])
        if iteration_params and len(iteration_params) > 0:
            first_iteration_data = iteration_params[0].get('data', {})
            if first_iteration_data:
                # Get any detector's frequency to simulate a click location
                first_detector_id = sorted(first_iteration_data.keys())[0]
                first_detector_data = first_iteration_data[first_detector_id]
                click_freq = first_detector_data.get('bias_frequency', 
                                                    first_detector_data.get('original_center_frequency'))
                
                if click_freq and hasattr(panel, '_handle_multisweep_plot_double_click') and panel.combined_mag_plot:
                    # Create a fake event at the detector's frequency
                    class FakeEvent:
                        def __init__(self, x, y):
                            self._scene_pos = QtCore.QPointF(x, y)
                        def scenePos(self):
                            return self._scene_pos
                        def accept(self):
                            pass
                    
                    # Map the frequency to view coordinates (x position)
                    view_box = panel.combined_mag_plot.getViewBox()
                    if view_box:
                        view_point = QtCore.QPointF(click_freq, 0)
                        scene_point = view_box.mapViewToScene(view_point)
                        fake_event = FakeEvent(scene_point.x(), scene_point.y())
                        panel._handle_multisweep_plot_double_click(fake_event)
        
        # Load noise data if present and open the NoiseSpectrumPanel
        if data.get('noise_data') is not None:
            noise_data = data['noise_data']
            panel._get_spectrum(noise_data, use_loaded_noise=True)
        else:
            print("[Noise] File loaded but no noise_data found - only fit panel shown")
        
        # Re-raise the multisweep dock to keep focus on it
        if dock:
            dock.raise_()
    
    def _open_file_with_system_default(self, file_path: str):
        """
        Open a file with the system's default application.
        
        Args:
            file_path: Path to the file to open
        """
        import subprocess
        import sys
        from pathlib import Path
        
        if not Path(file_path).exists():
            QtWidgets.QMessageBox.warning(
                self,
                "File Not Found",
                f"Cannot open file (not found):\n{file_path}"
            )
            return
        
        try:
            if sys.platform == 'darwin':  # macOS
                subprocess.run(['open', file_path], check=True)
            elif sys.platform == 'win32':  # Windows
                subprocess.run(['start', '', file_path], shell=True, check=True)
            else:  # Linux
                subprocess.run(['xdg-open', file_path], check=True)
            
            print(f"[SessionBrowser] Opened file with system default: {Path(file_path).name}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Open Error",
                f"Failed to open file:\n{file_path}\n\nError: {str(e)}"
            )
            print(f"[SessionBrowser] Error opening file: {e}")
    
    def _open_notebook_file(self, file_path: str):
        """
        Open a notebook file in the embedded Jupyter panel.
        
        If the notebook panel doesn't exist yet, creates it and starts the server.
        Once the server is ready, navigates to the specified notebook.
        
        Args:
            file_path: Full path to the .ipynb file
        """
        from pathlib import Path
        
        notebook_path = Path(file_path)
        if not notebook_path.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "File Not Found",
                f"Notebook file not found:\n{file_path}"
            )
            return
        
        # Check if notebook panel already exists
        notebook_dock = self.dock_manager.get_dock("notebook_panel")
        
        if notebook_dock is not None:
            # Panel exists - just open the notebook
            panel = notebook_dock.widget()
            if panel and hasattr(panel, 'open_notebook'):
                panel.open_notebook(file_path)
                notebook_dock.show()
                notebook_dock.raise_()
        else:
            # Need to create the panel first
            # Use the notebook's parent directory as the notebook_dir
            notebook_dir = str(notebook_path.parent)
            self._toggle_notebook_panel(notebook_dir=notebook_dir, open_file=file_path)
    
    @QtCore.pyqtSlot()
    @QtCore.pyqtSlot(str)
    def _update_session_status(self, session_path: str | None = None):
        """
        Update the status bar with session information.
        
        Args:
            session_path: Path to session (provided on session start)
        """
        if self.session_manager.is_active:
            name = self.session_manager.session_name or "Active"
            count = self.session_manager.export_count
            self.info_text.setText(f"📁 Session: {name} | {count} files")
            self.info_text.setToolTip(f"Session path: {self.session_manager.session_path}")
        else:
            self.info_text.setText("")
            self.info_text.setToolTip("")
    
    @QtCore.pyqtSlot(str, str)
    def _on_file_exported_status(self, file_path: str, data_type: str):
        """
        Update status bar when a file is exported.
        
        Args:
            file_path: Path to the exported file
            data_type: Type of data exported
        """
        self._update_session_status()
        
        # Brief flash message in status bar
        from pathlib import Path
        filename = Path(file_path).name
        self.statusBar().showMessage(f"Exported: {filename}", 3000)  # Show for 3 seconds
