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
from .mock_configuration_dialog import MockConfigurationDialog

# Note: The original commented-out lines for specific UI class imports
# (e.g., NetworkAnalysisDialog) have been removed, as these are expected
# to be covered by 'from .ui import *'.


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

        # Construct the main user interface.
        self._build_ui(chan_str) # Pass chan_str for initial display in QLineEdit

        # Initialize data buffers for each channel (Circular buffer from .utils).
        self._init_buffers()

        # Build the plot layout based on initial settings.
        self._build_layout()

        # Start the timer for periodic GUI updates (QtCore from .utils).
        self._start_timer()

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
        # NetworkAnalysisSignals (from .tasks) handles signals for NetAnal task progress, data updates, completion, and errors.
        self.netanal_signals = NetworkAnalysisSignals()
        self.netanal_signals.progress.connect(self._netanal_progress)
        self.netanal_signals.data_update.connect(self._netanal_data_update)
        self.netanal_signals.data_update_with_amp.connect(self._netanal_data_update_with_amp)
        self.netanal_signals.completed.connect(self._netanal_completed)
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
        self.btn_interactive_session = None # Button to toggle the console

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
        self.setWindowTitle("Periscope")              # Set the main window title

    # ───────────────────────── UI Construction ─────────────────────────
    # The methods in this section are responsible for building the various
    # components of the Periscope application's user interface.

    def _build_ui(self, chan_str: str):
        """
        Create and configure all top-level widgets and the main layout.

        This method orchestrates the construction of the entire UI by calling
        helper methods to add specific sections like the title, toolbar,
        configuration panel, plot container, status bar, and interactive console.

        Args:
            chan_str (str): The initial channel string, used to populate the
                            channel input field in the toolbar.
        """
        # `QtWidgets` is imported from .utils.
        central_widget = QtWidgets.QWidget() # Main content area for the QMainWindow
        self.setCentralWidget(central_widget)
        main_vbox_layout = QtWidgets.QVBoxLayout(central_widget) # Main vertical layout

        # Add UI components sequentially
        self._add_title(main_vbox_layout)
        self._add_toolbar(main_vbox_layout, chan_str)
        self._add_config_panel(main_vbox_layout)
        self._add_plot_container(main_vbox_layout)
        self._add_status_bar()
        self._add_interactive_console_dock() # Adds dock for iPython console (if available)

    def _add_title(self, layout: QtWidgets.QVBoxLayout):
        """
        Add the main application title label to the given layout.

        Displays the connected CRS host and module number.
        For mock mode, displays "EMULATED CRS BOARD" instead of the IP address.
        """
        # `QtWidgets` and `QtCore` are from .utils.
        # Check if we're in mock mode (localhost/127.0.0.1)
        if self.host in ["127.0.0.1", "localhost", "::1"]:
            display_host = "EMULATED CRS BOARD"
        else:
            display_host = self.host
        
        title_label = QtWidgets.QLabel(f"CRS: {display_host}    Module: {self.module}")
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter) # Center the title
        font_title = title_label.font() # Get current font to modify size
        font_title.setPointSize(16)    # Set a larger font size for the title
        title_label.setFont(font_title)
        layout.addWidget(title_label) # Add the label to the provided layout

    def _add_toolbar(self, layout: QtWidgets.QVBoxLayout, chan_str: str):
        """
        Add the main toolbar to the UI.

        The toolbar contains controls for channel selection, buffer size,
        pausing, toggling real units, selecting plot types, and accessing
        help, network analysis, and CRS initialization.

        Args:
            layout (QtWidgets.QVBoxLayout): The parent layout to add the toolbar to.
            chan_str (str): The initial channel string for the channel input field.
        """
        # `QtWidgets`, `QIntValidator` are from .utils.
        toolbar_widget = QtWidgets.QWidget() # Container widget for the toolbar
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar_widget) # Horizontal layout for toolbar items

        # Button to show/hide the configuration panel
        self.btn_toggle_cfg = QtWidgets.QPushButton("Show Configuration")
        self.btn_toggle_cfg.setCheckable(True)
        self.btn_toggle_cfg.toggled.connect(self._toggle_config)

        # Input field for channel selection string
        self.e_ch = QtWidgets.QLineEdit(chan_str)
        self.e_ch.setToolTip("Enter comma-separated channels or use '&' to group in one row (e.g., 1&2,3,4&5&6).")
        self.e_ch.returnPressed.connect(self._update_channels) # Update channels when Enter is pressed

        # Input field for buffer size
        self.e_buf = QtWidgets.QLineEdit(str(self.N))
        self.e_buf.setValidator(QIntValidator(10, 1_000_000, self)) # Validate input as integer (10 to 1M)
        self.e_buf.setMaximumWidth(80) # Limit width
        self.e_buf.setToolTip("Size of the ring buffer for each channel (history/FFT depth).")
        self.e_buf.editingFinished.connect(self._change_buffer) # Update buffer size when editing is done

        # Pause/Resume button
        self.b_pause = QtWidgets.QPushButton("Pause", clicked=self._toggle_pause)
        self.b_pause.setToolTip("Pause or resume real-time data acquisition and display.")

        # Radio buttons for unit selection
        self.unit_group = QtWidgets.QButtonGroup()
        self.rb_counts = QtWidgets.QRadioButton("Counts")
        self.rb_real_units = QtWidgets.QRadioButton("Real Units")
        self.rb_df_units = QtWidgets.QRadioButton("df Units")
        
        self.rb_counts.setToolTip("Display raw ADC counts")
        self.rb_real_units.setToolTip("Display in voltage/power units (V, dBm/Hz)")
        self.rb_df_units.setToolTip("Display frequency shift (Hz) and dissipation")
        
        # Add to button group for exclusive selection
        self.unit_group.addButton(self.rb_counts, 0)
        self.unit_group.addButton(self.rb_real_units, 1)
        self.unit_group.addButton(self.rb_df_units, 2)
        
        # Set initial state based on real_units flag
        if self.real_units:
            self.rb_real_units.setChecked(True)
        else:
            self.rb_counts.setChecked(True)
        
        # Connect signal
        self.unit_group.buttonClicked.connect(self._unit_mode_changed)

        # Checkboxes to select which plot types are displayed
        # For mock mode, default to TOD and FFT (not PSDs)
        self.cb_time = QtWidgets.QCheckBox("TOD", checked=True) # Time-domain
        self.cb_iq = QtWidgets.QCheckBox("IQ", checked=False)    # IQ plane
        self.cb_fft = QtWidgets.QCheckBox("FFT", checked=self.is_mock_mode)   # Raw FFT - on by default in mock mode
        self.cb_ssb = QtWidgets.QCheckBox("Single Sideband PSD", checked=not self.is_mock_mode) # SSB PSD - off in mock mode
        self.cb_dsb = QtWidgets.QCheckBox("Dual Sideband PSD", checked=False)  # DSB PSD
        
        # Connect toggled signal of each plot type checkbox
        # For PSD checkboxes in mock mode, show warning
        for cb_plot_type in (self.cb_time, self.cb_iq, self.cb_fft):
            cb_plot_type.toggled.connect(self._build_layout)
        
        # Special handling for PSD checkboxes in mock mode
        self.cb_ssb.toggled.connect(self._handle_psd_toggle)
        self.cb_dsb.toggled.connect(self._handle_psd_toggle)

        # Button to open CRS (Control and Readout System) initialization dialog
        self.btn_init_crs = QtWidgets.QPushButton("Initialize CRS Board")
        self.btn_init_crs.setToolTip("Open a dialog to initialize the CRS board (e.g., set IRIG source).")
        self.btn_init_crs.clicked.connect(self._show_initialize_crs_dialog)
        if self.crs is None: # Disable if no CRS object is available
            self.btn_init_crs.setEnabled(False)
            self.btn_init_crs.setToolTip("CRS object not available - cannot initialize board.")

        # Button to open Network Analyzer configuration dialog
        self.btn_netanal = QtWidgets.QPushButton("Network Analyzer")
        self.btn_netanal.setToolTip("Open the network analysis configuration window to perform sweeps.")
        self.btn_netanal.clicked.connect(self._show_netanal_dialog)
        if self.crs is None: # Disable if no CRS object is available
            self.btn_netanal.setEnabled(False)
            self.btn_netanal.setToolTip("CRS object not available - network analysis disabled.")

        # Help button
        self.btn_help = QtWidgets.QPushButton("Help")
        self.btn_help.setToolTip("Show usage instructions, interaction details, and examples.")
        self.btn_help.clicked.connect(self._show_help)

        # Add widgets to the toolbar layout
        toolbar_layout.addWidget(QtWidgets.QLabel("Channels:"))
        toolbar_layout.addWidget(self.e_ch)
        toolbar_layout.addSpacing(20)
        toolbar_layout.addWidget(QtWidgets.QLabel("Buffer:"))
        toolbar_layout.addWidget(self.e_buf)
        toolbar_layout.addWidget(self.b_pause)
        toolbar_layout.addSpacing(30)
        
        # Add unit radio buttons
        toolbar_layout.addWidget(QtWidgets.QLabel("Units:"))
        toolbar_layout.addWidget(self.rb_counts)
        toolbar_layout.addWidget(self.rb_real_units)
        toolbar_layout.addWidget(self.rb_df_units)
        toolbar_layout.addSpacing(30)
        for cb_plot_type in (self.cb_time, self.cb_iq, self.cb_fft, self.cb_ssb, self.cb_dsb):
            toolbar_layout.addWidget(cb_plot_type)
        toolbar_layout.addStretch(1) # Add stretch to push items to the left
        
        # Add mock reconfigure button if in mock mode
        if self.is_mock_mode:
            self.btn_reconfigure_mock = QtWidgets.QPushButton("Reconfigure Simulated KIDs")
            self.btn_reconfigure_mock.setToolTip("Reconfigure the simulated KID resonator parameters")
            self.btn_reconfigure_mock.clicked.connect(self._show_mock_config_dialog)
            toolbar_layout.addWidget(self.btn_reconfigure_mock)
        
        layout.addWidget(toolbar_widget) # Add the toolbar widget to the main vertical layout

    def _add_config_panel(self, layout: QtWidgets.QVBoxLayout):
        """
        Add the collapsible configuration panel and its associated action buttons.

        The action buttons (Interactive Session, Initialize CRS, Network Analyzer,
        Show/Hide Configuration, Help) are placed above the configuration panel itself.
        The configuration panel contains groups for 'Show Curves', 'IQ Mode',
        'PSD Mode', and 'General Display' settings.

        Args:
            layout (QtWidgets.QVBoxLayout): The parent layout to add the panel to.
        """
        # `QtWidgets`, `QTCONSOLE_AVAILABLE` are from .utils.

        # --- Action Buttons Row (above the collapsible config panel) ---
        action_buttons_widget = QtWidgets.QWidget()
        action_buttons_layout = QtWidgets.QHBoxLayout(action_buttons_widget)
        action_buttons_layout.setContentsMargins(0,0,0,0) # No margins for this layout
        action_buttons_layout.addStretch(1) # Push buttons to the right

        # Button to toggle embedded iPython session
        self.btn_interactive_session = QtWidgets.QPushButton("Interactive Session")
        self.btn_interactive_session.setToolTip("Toggle an embedded iPython interactive session.")
        self.btn_interactive_session.clicked.connect(self._toggle_interactive_session)
        if not QTCONSOLE_AVAILABLE or self.crs is None: # QTCONSOLE_AVAILABLE from .utils
            self.btn_interactive_session.setEnabled(False)
            if not QTCONSOLE_AVAILABLE:
                self.btn_interactive_session.setToolTip("Interactive session disabled: qtconsole/ipykernel not installed.")
            else:
                self.btn_interactive_session.setToolTip("Interactive session disabled: CRS object not available.")
        action_buttons_layout.addWidget(self.btn_interactive_session)

        action_buttons_layout.addWidget(self.btn_init_crs)
        action_buttons_layout.addWidget(self.btn_netanal) 
        action_buttons_layout.addWidget(self.btn_toggle_cfg)
        action_buttons_layout.addWidget(self.btn_help)    
        layout.addWidget(action_buttons_widget)

        # --- Collapsible Configuration Panel ---
        # This panel contains more detailed settings and is initially hidden.
        self.ctrl_panel = QtWidgets.QGroupBox("Configuration")
        self.ctrl_panel.setVisible(False) # Initially hidden
        config_panel_layout = QtWidgets.QHBoxLayout(self.ctrl_panel) # Horizontal layout for groups within the panel

        # Add various settings groups to the configuration panel
        config_panel_layout.addWidget(self._create_show_curves_group())
        config_panel_layout.addWidget(self._create_iq_mode_group())
        config_panel_layout.addWidget(self._create_psd_mode_group())
        config_panel_layout.addWidget(self._create_display_group())
        layout.addWidget(self.ctrl_panel) # Add the panel to the main vertical layout

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

        Contains checkboxes for zoom box mode, dark mode, and auto-scaling plots.

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

        # Checkbox for dark mode theme
        self.cb_dark = QtWidgets.QCheckBox("Dark Mode", checked=self.dark_mode)
        self.cb_dark.setToolTip("Switch between dark and light UI themes.")
        self.cb_dark.toggled.connect(self._toggle_dark_mode) # Rebuilds layout
        self.cb_dark.toggled.connect(self._update_console_style) # Updates console style if active
        layout.addWidget(self.cb_dark)

        # Checkbox for auto-scaling plots (excluding TOD)
        self.cb_auto_scale = QtWidgets.QCheckBox("Auto Scale", checked=self.auto_scale_plots)
        self.cb_auto_scale.setToolTip("Enable/disable auto-ranging for IQ, FFT, SSB, and DSB plots. Can improve display performance when disabled.")
        self.cb_auto_scale.toggled.connect(self._toggle_auto_scale)
        layout.addWidget(self.cb_auto_scale)
        return group_box

    def _add_plot_container(self, layout: QtWidgets.QVBoxLayout):
        """
        Add the main container widget for plots to the given layout.

        This container will hold the grid layout where individual plots are placed.
        """
        # `QtWidgets` is from .utils.
        self.container = QtWidgets.QWidget() # The main widget that will contain the plot grid
        layout.addWidget(self.container)    # Add it to the parent layout
        self.grid = QtWidgets.QGridLayout(self.container) # Grid layout for arranging plots

    def _add_status_bar(self):
        """
        Add a status bar to the main window.

        Used to display performance statistics (FPS, PPS).
        """
        # `QtWidgets` is from .utils.
        self.setStatusBar(QtWidgets.QStatusBar()) # Create and set a new status bar

        self.fps_label = QtWidgets.QLabel()
        self.pps_label = QtWidgets.QLabel()
        self.packet_loss_label = QtWidgets.QLabel()
        self.dropped_label = QtWidgets.QLabel()
        self.info_text = QtWidgets.QLabel()

        self.default_packet_loss_color = self.packet_loss_label.palette().color(QtGui.QPalette.WindowText).name()

        # Add them to the status bar
        self.statusBar().addWidget(self.fps_label)
        self.statusBar().addWidget(self.pps_label)
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
            "by other tasks to drain the existing buffer in time. The recommended UDP receive buffer sizes and commands to set are below.\n"
            "- **MacOS:**\n"
            "  Run the following commands to increase the default buffer\n"
            "```\n"
            "     sudo sysctl -w kern.ipc.maxsockbuf=16777216\n"
            "     sudo sysctl -w net.inet.udp.recvspace=16777216\n"
            "```\n"
            " - **Linux:**\n"
            "   Run the following command to increase the default buffer\n"
            "```\n"
            "     sudo net.core.rmem_max = 67108864\n"
            "```\n"
            "- **Windows:** \n\n"
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
                if isinstance(viewbox, ClickableViewBox): # ClickableViewBox from .utils
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

    def _show_netanal_dialog(self):
        """
        Display the Network Analysis configuration dialog.

        This dialog allows users to set parameters for a network analysis sweep,
        such as frequency range, number of points, and amplitude.
        It fetches current DAC scales asynchronously to provide accurate power level
        estimations. If the user confirms the dialog with valid parameters,
        a new network analysis process is started via `_start_network_analysis`.
        """
        if self.crs is None:
            QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available")
            return
        default_dac_scales = {m: -0.5 for m in range(1, 9)}
        # NetworkAnalysisDialog from .ui (which imports from .dialogs)
        dialog = NetworkAnalysisDialog(self, modules=list(range(1, 9)), dac_scales=default_dac_scales)
        dialog.module_entry.setText(str(self.module))
        # DACScaleFetcher from .tasks
        fetcher = DACScaleFetcher(self.crs)
        fetcher.dac_scales_ready.connect(lambda scales: dialog.dac_scales.update(scales))
        fetcher.dac_scales_ready.connect(dialog._update_dac_scale_info)
        fetcher.dac_scales_ready.connect(dialog._update_dbm_from_normalized)
        fetcher.dac_scales_ready.connect(lambda scales: setattr(self, 'dac_scales', scales))
        fetcher.start()
        if dialog.exec():
            self.dac_scales = dialog.dac_scales.copy()
            params = dialog.get_parameters()
            if params:
                self._start_network_analysis(params)

    def _start_network_analysis(self, params: dict):
        """
        Initialize and start a new network analysis process.

        This method creates a new `NetworkAnalysisWindow` to display the results
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
            selected_module_param = params.get('module') # Renamed to avoid conflict
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
            window_id = f"window_{self.netanal_window_count}"
            self.netanal_window_count += 1
            window_signals = NetworkAnalysisSignals() # From .tasks
            dac_scales_local = self.dac_scales.copy() # Renamed
            # NetworkAnalysisWindow from .ui
            window_instance = NetworkAnalysisWindow(self, modules_to_run, dac_scales_local, dark_mode=self.dark_mode) # Renamed
            window_instance.set_params(params)
            window_instance.window_id = window_id
            self.netanal_windows[window_id] = {
                'window': window_instance,
                'signals': window_signals,
                'amplitude_queues': {},
                'current_amp_index': {}
            }
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
            amplitudes = params.get('amps', [params.get('amp', DEFAULT_AMPLITUDE)]) # DEFAULT_AMPLITUDE from .utils
            window_data = self.netanal_windows[window_id]
            window_data['amplitude_queues'] = {mod: list(amplitudes) for mod in modules_to_run}
            window_data['current_amp_index'] = {mod: 0 for mod in modules_to_run}
            for mod_iter in modules_to_run: # Renamed
                window_instance.update_amplitude_progress(mod_iter, 1, len(amplitudes), amplitudes[0])
                self._start_next_amplitude_task(mod_iter, params, window_id)
            window_instance.show()
        except Exception as e:
            print(f"Error in _start_network_analysis: {e}")
            traceback.print_exc() # traceback from .utils

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

    def _rerun_network_analysis(self, params: dict):
        """
        Re-run a network analysis for an existing NetworkAnalysisWindow.

        This method is typically triggered by a signal from a
        `NetworkAnalysisWindow` instance (e.g., when the user clicks a "Re-run"
        button within that window). It stops any ongoing tasks for that window,
        clears its data and plots, updates its parameters, and then restarts
        the analysis sequence.

        Args:
            params (dict): The new or updated parameters for the network analysis.
        """
        try:
            if self.crs is None: QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available"); return
            sender_widget = self.sender() # Renamed
            source_window = None
            window_id = None
            if sender_widget and hasattr(sender_widget, 'window'): source_window = sender_widget.window()
            for w_id, w_data in self.netanal_windows.items():
                if w_data['window'] == source_window: window_id = w_id; break
            if not window_id: return
            window_data = self.netanal_windows[window_id]
            window = window_data['window']
            window.data.clear(); window.raw_data.clear()
            for mod, pbar in window.progress_bars.items(): pbar.setValue(0) # Renamed module
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
    
    def _netanal_progress(self, module_param: int, progress: float):
        """Slot for network analysis progress signals. Currently a placeholder."""
        pass # Renamed

    def _netanal_data_update(self, module_param: int, freqs: np.ndarray, amps: np.ndarray, phases: np.ndarray):
        """Slot for network analysis data update signals. Currently a placeholder."""
        pass # Renamed

    def _netanal_data_update_with_amp(self, module_param: int, freqs: np.ndarray, amps: np.ndarray, phases: np.ndarray, amplitude: float):
        """Slot for network analysis data update signals that include amplitude. Currently a placeholder."""
        pass # Renamed

    def _netanal_completed(self, module_param: int):
        """Slot for network analysis completion signals. Currently a placeholder."""
        pass # Renamed

    def _netanal_error(self, error_msg: str):
        """Slot for network analysis error signals. Displays a critical message box."""
        QtWidgets.QMessageBox.critical(self, "Network Analysis Error", error_msg)

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
        Get the current mock configuration from the mock_constants module.
        
        Returns:
            dict: Current configuration values
        """
        try:
            import rfmux.core.mock_constants as mc
            return {
                'kinetic_inductance_fraction': mc.DEFAULT_KINETIC_INDUCTANCE_FRACTION,
                'kinetic_inductance_variation': mc.KINETIC_INDUCTANCE_VARIATION,
                'frequency_shift_power_law': mc.FREQUENCY_SHIFT_POWER_LAW,
                'frequency_shift_magnitude': mc.FREQUENCY_SHIFT_MAGNITUDE,
                'power_normalization': mc.POWER_NORMALIZATION,
                'enable_bifurcation': mc.ENABLE_BIFURCATION,
                'bifurcation_iterations': mc.BIFURCATION_ITERATIONS,
                'bifurcation_convergence_tolerance': mc.BIFURCATION_CONVERGENCE_TOLERANCE,
                'bifurcation_damping_factor': mc.BIFURCATION_DAMPING_FACTOR,
                'saturation_power': mc.SATURATION_POWER,
                'saturation_sharpness': mc.SATURATION_SHARPNESS,
                'q_min': mc.DEFAULT_Q_MIN,
                'q_max': mc.DEFAULT_Q_MAX,
                'q_variation': mc.Q_VARIATION,
                'coupling_min': mc.DEFAULT_COUPLING_MIN,
                'coupling_max': mc.DEFAULT_COUPLING_MAX,
                'freq_start': mc.DEFAULT_FREQ_START,
                'freq_end': mc.DEFAULT_FREQ_END,
                'num_resonances': mc.DEFAULT_NUM_RESONANCES,
                'base_noise_level': mc.BASE_NOISE_LEVEL,
                'amplitude_noise_coupling': mc.AMPLITUDE_NOISE_COUPLING,
                'udp_noise_level': mc.UDP_NOISE_LEVEL
            }
        except Exception as e:
            print(f"Error getting current mock config: {e}")
            # Return defaults if unable to read current values
            return {}
            
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
