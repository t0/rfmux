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

# Note: The original commented-out lines for specific UI class imports
# (e.g., NetworkAnalysisDialog) have been removed, as these are expected
# to be covered by 'from .ui import *'.


class Periscope(QtWidgets.QMainWindow):
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

        # --- State Variables ---
        self.paused: bool = False               # Flag to pause/resume data processing
        self.start_time: float | None = None    # Timestamp of the first processed packet
        self.frame_cnt: int = 0                 # Counter for GUI frames rendered
        self.pkt_cnt: int = 0                   # Counter for processed UDP packets
        self.t_last: float = time.time()        # Timestamp of the last performance update (time from .utils)

        # Decimation stage, dynamically updated based on inferred sample rate.
        # This is used for PSD calculations.
        self.dec_stage: int = 6
        self.last_dec_update: float = 0.0       # Timestamp of last decimation stage update

        # --- Display Settings ---
        self.dark_mode: bool = True             # UI theme (dark/light)
        self.real_units: bool = False           # Display data in real units (V, dBm) vs. counts
        self.psd_absolute: bool = True          # PSD y-axis scale (absolute dBm/Hz vs. relative dBc/Hz)
        self.auto_scale_plots: bool = True      # Enable/disable auto-ranging for plots (except TOD)
        self.show_i: bool = True                # Visibility of 'I' component traces
        self.show_q: bool = True                # Visibility of 'Q' component traces
        self.show_m: bool = True                # Visibility of 'Magnitude' component traces
        self.zoom_box_mode: bool = True         # Default mouse mode for plots (zoom vs pan)

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
        """
        # `QtWidgets` and `QtCore` are from .utils.
        title_label = QtWidgets.QLabel(f"CRS: {self.host}    Module: {self.module}")
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

        # Checkbox to toggle between raw counts and real units (Volts, dBm/Hz)
        self.cb_real = QtWidgets.QCheckBox("Real Units", checked=self.real_units)
        self.cb_real.setToolTip("Toggle display units between raw 'counts' and real-world voltage/power units.")
        self.cb_real.toggled.connect(self._toggle_real_units)

        # Checkboxes to select which plot types are displayed
        self.cb_time = QtWidgets.QCheckBox("TOD", checked=True) # Time-domain
        self.cb_iq = QtWidgets.QCheckBox("IQ", checked=False)    # IQ plane
        self.cb_fft = QtWidgets.QCheckBox("FFT", checked=False)   # Raw FFT
        self.cb_ssb = QtWidgets.QCheckBox("Single Sideband PSD", checked=True) # SSB PSD
        self.cb_dsb = QtWidgets.QCheckBox("Dual Sideband PSD", checked=False)  # DSB PSD
        # Connect toggled signal of each plot type checkbox to rebuild the layout
        for cb_plot_type in (self.cb_time, self.cb_iq, self.cb_fft, self.cb_ssb, self.cb_dsb):
            cb_plot_type.toggled.connect(self._build_layout)

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
        toolbar_layout.addWidget(self.cb_real)
        toolbar_layout.addSpacing(30)
        for cb_plot_type in (self.cb_time, self.cb_iq, self.cb_fft, self.cb_ssb, self.cb_dsb):
            toolbar_layout.addWidget(cb_plot_type)
        toolbar_layout.addStretch(1) # Add stretch to push items to the left
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

        Contains controls for PSD scale (absolute/relative) and Welch segments.

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

        # Add widgets to the grid layout
        grid_layout.addWidget(self.lbl_psd_scale, 0, 0) # Row 0, Col 0
        grid_layout.addWidget(self.rb_psd_abs, 0, 1)    # Row 0, Col 1
        grid_layout.addWidget(self.rb_psd_rel, 0, 2)    # Row 0, Col 2
        grid_layout.addWidget(QtWidgets.QLabel("Segments:"), 1, 0) # Row 1, Col 0
        grid_layout.addWidget(self.spin_segments, 1, 1) # Row 1, Col 1 (spans 1 column)
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

    def _show_help(self):
        """
        Display the help dialog with usage instructions and tips.
        """
        # `QtWidgets`, `QtCore` are from .utils.
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Periscope Help")
        help_text = (
            "**Usage:**\n"
            "  - Multi-channel grouping: use '&' to display multiple channels in one row.\n"
            "    e.g., \"3&5\" for channels 3 and 5 in one row, \"3&5,7\" for that row plus a row with channel 7.\n\n"
            # ... (rest of help text remains the same)
            "**IPython / Jupyter:** invoke directly from CRS object\n"
            "  - `>>> crs.raise_periscope(module=2, channels=\"3&5\")`\n"
            "  - If in a non-blocking mode, you can still interact with your session concurrently.\n\n"
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
            window_instance = NetworkAnalysisWindow(self, modules_to_run, dac_scales_local) # Renamed
            window_instance.set_params(params)
            window_instance.window_id = window_id
            self.netanal_windows[window_id] = {
                'window': window_instance,
                'signals': window_signals,
                'amplitude_queues': {},
                'current_amp_index': {}
            }
            window_signals.progress.connect(
                lambda mod, prog: window_instance.update_progress(mod, prog)) # mod, prog to avoid conflict
            window_signals.data_update.connect(
                lambda mod, freqs, amps, phases: window_instance.update_data(mod, freqs, amps, phases))
            window_signals.data_update_with_amp.connect(
                lambda mod, freqs, amps, phases, amp_val:  # amp_val to avoid conflict
                window_instance.update_data_with_amp(mod, freqs, amps, phases, amp_val))
            window_signals.completed.connect(
                lambda mod: self._handle_analysis_completed(mod, window_id))
            window_signals.error.connect(
                lambda error_msg: QtWidgets.QMessageBox.critical(window_instance, "Network Analysis Error", error_msg))
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
            self.pool.start(task)
        except Exception as e:
            print(f"Error in _start_next_amplitude_task: {e}")
            traceback.print_exc()
    
    def _rerun_network_analysis(self, params: dict):
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
    
    def _netanal_progress(self, module_param: int, progress: float): pass # Renamed
    def _netanal_data_update(self, module_param: int, freqs: np.ndarray, amps: np.ndarray, phases: np.ndarray): pass # Renamed
    def _netanal_data_update_with_amp(self, module_param: int, freqs: np.ndarray, amps: np.ndarray, phases: np.ndarray, amplitude: float): pass # Renamed
    def _netanal_completed(self, module_param: int): pass # Renamed
    def _netanal_error(self, error_msg: str): QtWidgets.QMessageBox.critical(self, "Network Analysis Error", error_msg)
    def _crs_init_success(self, message: str): QtWidgets.QMessageBox.information(self, "CRS Initialization Success", message)
    def _crs_init_error(self, error_msg: str): QtWidgets.QMessageBox.critical(self, "CRS Initialization Error", error_msg)

    def _toggle_config(self, visible: bool):
        self.ctrl_panel.setVisible(visible)
        self.btn_toggle_cfg.setText("Hide Configuration" if visible else "Show Configuration")

    def _toggle_auto_scale(self, checked: bool):
        self.auto_scale_plots = checked
        if hasattr(self, "plots"):
            for rowPlots in self.plots:
                for mode_key, pw in rowPlots.items(): # Renamed mode
                    if mode_key != "T": pw.enableAutoRange(pg.ViewBox.XYAxes, checked)

    def _toggle_iqmag(self):
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

    def _init_buffers(self):
        unique_chs = set()
        for group in self.channel_list:
            for c_val in group: unique_chs.add(c_val) # Renamed c
        self.all_chs = sorted(unique_chs)
        self.buf = {}
        self.tbuf = {}
        for ch_val in self.all_chs: # Renamed ch
            self.buf[ch_val] = {k: Circular(self.N) for k in ("I", "Q", "M")} # Circular from .utils
            self.tbuf[ch_val] = Circular(self.N)

    def _build_layout(self):
        self._clear_current_layout()
        modes_active = self._get_active_modes() # Renamed modes
        self.plots = []
        self.curves = []
        font = QFont() # QFont from .utils
        font.setPointSize(UI_FONT_SIZE) # UI_FONT_SIZE from .utils
        single_colors = self._get_single_channel_colors()
        channel_families = self._get_channel_color_families()
        for row_i, group in enumerate(self.channel_list):
            rowPlots, rowCurves = self._create_row_plots_and_curves(row_i, group, modes_active, font, single_colors, channel_families)
            self.plots.append(rowPlots)
            self.curves.append(rowCurves)
        self._restore_auto_range_settings()
        self._toggle_iqmag()
        if hasattr(self, "zoom_box_mode"): self._toggle_zoom_box_mode(self.zoom_box_mode)

    def _clear_current_layout(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget_item = item.widget() # Renamed w
            if widget_item: widget_item.deleteLater()

    def _get_active_modes(self):
        modes_list = [] # Renamed modes
        if self.cb_time.isChecked(): modes_list.append("T")
        if self.cb_iq.isChecked(): modes_list.append("IQ")
        if self.cb_fft.isChecked(): modes_list.append("F")
        if self.cb_ssb.isChecked(): modes_list.append("S")
        if self.cb_dsb.isChecked(): modes_list.append("D")
        return modes_list

    def _get_single_channel_colors(self):
        return {"I": "#1f77b4", "Q": "#ff7f0e", "Mag": "#2ca02c", "DSB": "#bcbd22"}

    def _get_channel_color_families(self):
        return [
            ("#1f77b4", "#4a8cc5", "#90bce0"), ("#ff7f0e", "#ffa64d", "#ffd2a3"),
            ("#2ca02c", "#63c063", "#a1d9a1"), ("#d62728", "#eb6a6b", "#f2aeae"),
            ("#9467bd", "#ae8ecc", "#d3c4e3"), ("#8c564b", "#b58e87", "#d9c3bf"),
            ("#e377c2", "#f0a8dc", "#f7d2ee"), ("#7f7f7f", "#aaaaaa", "#d3d3d3"),
            ("#bcbd22", "#cfd342", "#e2e795"), ("#17becf", "#51d2de", "#9ae8f2"),
        ]

    def _create_row_plots_and_curves(self, row_i, group, modes_active, font, single_colors, channel_families): # Renamed modes
        rowPlots = {}; rowCurves = {}
        row_title = "Ch " + ("&".join(map(str, group)) if len(group) > 1 else str(group[0]))
        for col, mode_key in enumerate(modes_active): # Renamed mode
            vb = ClickableViewBox() # From .utils
            pw = pg.PlotWidget(viewBox=vb, title=f"{mode_title(mode_key)} – {row_title}") # mode_title from .utils
            self._configure_plot_auto_range(pw, mode_key)
            self._configure_plot_axes(pw, mode_key)
            self._apply_plot_theme(pw)
            pw.showGrid(x=True, y=True, alpha=0.3)
            self.grid.addWidget(pw, row_i, col)
            rowPlots[mode_key] = pw
            self._configure_plot_fonts(pw, font)
            if mode_key == "IQ":
                rowCurves["IQ"] = self._create_iq_plot_item(pw)
            else:
                legend = pw.addLegend(offset=(30, 10))
                if len(group) == 1:
                    ch_val = group[0] # Renamed ch
                    rowCurves[mode_key] = self._create_single_channel_curves(pw, mode_key, ch_val, single_colors, legend)
                else:
                    rowCurves[mode_key] = self._create_multi_channel_curves(pw, mode_key, group, channel_families, legend)
                self._make_legend_clickable(legend)
        return rowPlots, rowCurves

    def _configure_plot_auto_range(self, pw, mode_key): # Renamed mode
        if mode_key == "T": pw.enableAutoRange(pg.ViewBox.XYAxes, True)
        else: pw.enableAutoRange(pg.ViewBox.XYAxes, self.auto_scale_plots)

    def _configure_plot_axes(self, pw, mode_key): # Renamed mode
        # convert_roc_to_volts from .utils
        if mode_key == "T":
            pw.setLabel("left", "Amplitude", units="V" if self.real_units else "Counts")
        elif mode_key == "IQ":
            pw.getViewBox().setAspectLocked(True)
            pw.setLabel("bottom", "I", units="V" if self.real_units else "Counts")
            pw.setLabel("left",   "Q", units="V" if self.real_units else "Counts")
        elif mode_key == "F":
            pw.setLogMode(x=True, y=True)
            pw.setLabel("bottom", "Freq", units="Hz")
            pw.setLabel("left", "Amplitude", units="V" if self.real_units else "Counts")
        elif mode_key == "S":
            pw.setLogMode(x=True, y=not self.real_units)
            pw.setLabel("bottom", "Freq", units="Hz")
            lbl = "dBm/Hz" if self.psd_absolute else "dBc/Hz"
            pw.setLabel("left", f"PSD ({lbl})" if self.real_units else "PSD (Counts²/Hz)")
        else:  # "D"
            pw.setLogMode(x=False, y=not self.real_units)
            pw.setLabel("bottom", "Freq", units="Hz")
            lbl = "dBm/Hz" if self.psd_absolute else "dBc/Hz"
            pw.setLabel("left", f"PSD ({lbl})" if self.real_units else "PSD (Counts²/Hz)")

    def _configure_plot_fonts(self, pw, font):
        pi = pw.getPlotItem()
        for axis_name in ("left", "bottom", "right", "top"):
            axis = pi.getAxis(axis_name)
            if axis: axis.setTickFont(font)
            if axis and axis.label: axis.label.setFont(font) # Check axis.label
        pi.titleLabel.setFont(font)

    def _create_iq_plot_item(self, pw):
        # SCATTER_SIZE from .utils
        if self.rb_scatter.isChecked():
            sp = pg.ScatterPlotItem(pen=None, size=SCATTER_SIZE)
            pw.addItem(sp); return {"mode": "scatter", "item": sp}
        else:
            img = pg.ImageItem(axisOrder="row-major")
            img.setLookupTable(self.lut); pw.addItem(img)
            return {"mode": "density", "item": img}

    def _create_single_channel_curves(self, pw, mode_key, ch_val, single_colors, legend): # Renamed mode, ch
        # LINE_WIDTH from .utils
        if mode_key == "T":
            cI = pw.plot(pen=pg.mkPen(single_colors["I"],   width=LINE_WIDTH), name="I")
            cQ = pw.plot(pen=pg.mkPen(single_colors["Q"],   width=LINE_WIDTH), name="Q")
            cM = pw.plot(pen=pg.mkPen(single_colors["Mag"], width=LINE_WIDTH), name="Mag")
            self._fade_hidden_entries(legend, ("I", "Q"))
            return {ch_val: {"I": cI, "Q": cQ, "Mag": cM}}
        elif mode_key == "F":
            cI = pw.plot(pen=pg.mkPen(single_colors["I"],   width=LINE_WIDTH), name="I"); cI.setFftMode(True)
            cQ = pw.plot(pen=pg.mkPen(single_colors["Q"],   width=LINE_WIDTH), name="Q"); cQ.setFftMode(True)
            cM = pw.plot(pen=pg.mkPen(single_colors["Mag"], width=LINE_WIDTH), name="Mag"); cM.setFftMode(True)
            self._fade_hidden_entries(legend, ("I", "Q"))
            return {ch_val: {"I": cI, "Q": cQ, "Mag": cM}}
        elif mode_key == "S":
            cI = pw.plot(pen=pg.mkPen(single_colors["I"],   width=LINE_WIDTH), name="I")
            cQ = pw.plot(pen=pg.mkPen(single_colors["Q"],   width=LINE_WIDTH), name="Q")
            cM = pw.plot(pen=pg.mkPen(single_colors["Mag"], width=LINE_WIDTH), name="Mag")
            return {ch_val: {"I": cI, "Q": cQ, "Mag": cM}}
        else:  # "D"
            cD = pw.plot(pen=pg.mkPen(single_colors["DSB"], width=LINE_WIDTH), name="Complex DSB")
            return {ch_val: {"Cmplx": cD}}

    def _create_multi_channel_curves(self, pw, mode_key, group, channel_families, legend): # Renamed mode
        mode_dict = {}
        for i, ch_val in enumerate(group): # Renamed ch
            (colI, colQ, colM) = channel_families[i % len(channel_families)]
            if mode_key == "T":
                cI = pw.plot(pen=pg.mkPen(colI, width=LINE_WIDTH), name=f"ch{ch_val}-I")
                cQ = pw.plot(pen=pg.mkPen(colQ, width=LINE_WIDTH), name=f"ch{ch_val}-Q")
                cM = pw.plot(pen=pg.mkPen(colM, width=LINE_WIDTH), name=f"ch{ch_val}-Mag")
                mode_dict[ch_val] = {"I": cI, "Q": cQ, "Mag": cM}
            elif mode_key == "F":
                cI = pw.plot(pen=pg.mkPen(colI, width=LINE_WIDTH), name=f"ch{ch_val}-I"); cI.setFftMode(True)
                cQ = pw.plot(pen=pg.mkPen(colQ, width=LINE_WIDTH), name=f"ch{ch_val}-Q"); cQ.setFftMode(True)
                cM = pw.plot(pen=pg.mkPen(colM, width=LINE_WIDTH), name=f"ch{ch_val}-Mag"); cM.setFftMode(True)
                mode_dict[ch_val] = {"I": cI, "Q": cQ, "Mag": cM}
            elif mode_key == "S":
                cI = pw.plot(pen=pg.mkPen(colI, width=LINE_WIDTH), name=f"ch{ch_val}-I")
                cQ = pw.plot(pen=pg.mkPen(colQ, width=LINE_WIDTH), name=f"ch{ch_val}-Q")
                cM = pw.plot(pen=pg.mkPen(colM, width=LINE_WIDTH), name=f"ch{ch_val}-Mag")
                mode_dict[ch_val] = {"I": cI, "Q": cQ, "Mag": cM}
            else:  # "D"
                cD = pw.plot(pen=pg.mkPen(colI, width=LINE_WIDTH), name=f"ch{ch_val}-DSB")
                mode_dict[ch_val] = {"Cmplx": cD}
        return mode_dict

    def _restore_auto_range_settings(self):
        self.auto_scale_plots = True
        self.cb_auto_scale.setChecked(True)
        for rowPlots in self.plots:
            for mode_key, pw in rowPlots.items(): # Renamed mode
                if mode_key != "T": pw.enableAutoRange(pg.ViewBox.XYAxes, True)

    def _apply_plot_theme(self, pw: pg.PlotWidget):
        bg_color, pen_color = ("k", "w") if self.dark_mode else ("w", "k")
        pw.setBackground(bg_color)
        for axis_name in ("left", "bottom", "right", "top"):
            ax = pw.getPlotItem().getAxis(axis_name)
            if ax: ax.setPen(pen_color); ax.setTextPen(pen_color)

    @staticmethod
    def _fade_hidden_entries(legend, hide_labels):
        for sample, label in legend.items:
            txt = label.labelItem.toPlainText() if hasattr(label, "labelItem") else ""
            if txt in hide_labels: sample.setOpacity(0.3); label.setOpacity(0.3)

    @staticmethod
    def _make_legend_clickable(legend):
        for sample, label in legend.items:
            curve = sample.item
            def toggle(evt, c=curve, s=sample, l=label):
                vis = not c.isVisible(); c.setVisible(vis)
                op = 1.0 if vis else 0.3; s.setOpacity(op); l.setOpacity(op)
            label.mousePressEvent = toggle; sample.mousePressEvent = toggle

    def _toggle_dark_mode(self, checked: bool):
        self.dark_mode = checked; self._build_layout()

    def _toggle_real_units(self, checked: bool):
        self.real_units = checked
        if checked:
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Real Units On")
            msg.setText(
                "Global conversion to real units (V, dBm) is approximate.\n"
                "All PSD plots are droop-corrected for the CIC1 and CIC2 decimation filters; the 'Raw FFT' is calculated from the raw TOD and not droop-corrected."
            )
            msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok); msg.exec()
        self._build_layout()

    def _psd_ref_changed(self):
        self.psd_absolute = self.rb_psd_abs.isChecked(); self._build_layout()

    def _update_channels(self):
        # parse_channels_multich from .utils
        new_parsed = parse_channels_multich(self.e_ch.text())
        if new_parsed != self.channel_list:
            self.channel_list = new_parsed; self.iq_workers.clear(); self.psd_workers.clear()
            for row_i, group in enumerate(self.channel_list):
                self.psd_workers[row_i] = {"S": {}, "D": {}}
                for c_val in group: # Renamed c
                    self.psd_workers[row_i]["S"][c_val] = False
                    self.psd_workers[row_i]["D"][c_val] = False
            self._init_buffers(); self._build_layout()

    def _change_buffer(self):
        try: n_val = int(self.e_buf.text()) # Renamed n
        except ValueError: return
        if n_val != self.N: self.N = n_val; self._init_buffers()

    def _toggle_pause(self):
        self.paused = not self.paused
        self.b_pause.setText("Resume" if self.paused else "Pause")

    def _update_gui(self):
        if self.paused: self._discard_packets(); return
        self._process_incoming_packets()
        self.frame_cnt += 1; now = time.time()
        if (now - self.last_dec_update) > 1.0:
            self._update_dec_stage(); self.last_dec_update = now
        self._update_plot_data()
        self._update_performance_stats(now)

    def _discard_packets(self):
        while not self.receiver.queue.empty(): self.receiver.queue.get()

    def _process_incoming_packets(self):
        while not self.receiver.queue.empty():
            pkt = self.receiver.queue.get(); self.pkt_cnt += 1
            t_rel = self._calculate_relative_timestamp(pkt)
            self._update_buffers(pkt, t_rel)

    def _calculate_relative_timestamp(self, pkt):
        # streamer from .utils
        ts = pkt.ts
        if ts.recent:
            ts.ss += int(0.02 * streamer.SS_PER_SECOND); ts.renormalize()
            t_now = ts.h * 3600 + ts.m * 60 + ts.s + ts.ss / streamer.SS_PER_SECOND
            if self.start_time is None: self.start_time = t_now
            return t_now - self.start_time
        return None

    def _update_buffers(self, pkt, t_rel):
        # math from .utils
        for ch_val in self.all_chs: # Renamed ch
            Ival = pkt.s[2 * (ch_val - 1)] / 256.0
            Qval = pkt.s[2 * (ch_val - 1) + 1] / 256.0
            self.buf[ch_val]["I"].add(Ival); self.buf[ch_val]["Q"].add(Qval)
            self.buf[ch_val]["M"].add(math.hypot(Ival, Qval)); self.tbuf[ch_val].add(t_rel)

    def _update_plot_data(self):
        for row_i, group in enumerate(self.channel_list):
            rowCurves = self.curves[row_i]
            for ch_val in group: self._update_channel_plot_data(ch_val, rowCurves) # Renamed ch
            if "IQ" in rowCurves and not self.iq_workers.get(row_i, False):
                self._dispatch_iq_task(row_i, group, rowCurves)
            self._dispatch_psd_tasks(row_i, group)

    def _update_channel_plot_data(self, ch_val, rowCurves): # Renamed ch
        rawI = self.buf[ch_val]["I"].data(); rawQ = self.buf[ch_val]["Q"].data()
        rawM = self.buf[ch_val]["M"].data(); tarr = self.tbuf[ch_val].data()
        # convert_roc_to_volts from .utils
        I_data, Q_data, M_data = (convert_roc_to_volts(d) for d in (rawI, rawQ, rawM)) if self.real_units else (rawI, rawQ, rawM) # Renamed I,Q,M
        if "T" in rowCurves and ch_val in rowCurves["T"]:
            cset = rowCurves["T"][ch_val]
            if cset["I"].isVisible(): cset["I"].setData(tarr, I_data)
            if cset["Q"].isVisible(): cset["Q"].setData(tarr, Q_data)
            if cset["Mag"].isVisible(): cset["Mag"].setData(tarr, M_data)
        if "F" in rowCurves and ch_val in rowCurves["F"]:
            cset = rowCurves["F"][ch_val]
            if cset["I"].isVisible(): cset["I"].setData(tarr, I_data, fftMode=True)
            if cset["Q"].isVisible(): cset["Q"].setData(tarr, Q_data, fftMode=True)
            if cset["Mag"].isVisible(): cset["Mag"].setData(tarr, M_data, fftMode=True)

    def _dispatch_iq_task(self, row_i, group, rowCurves):
        # IQTask from .tasks
        mode_key = rowCurves["IQ"]["mode"] # Renamed mode
        if len(group) == 1:
            c_val = group[0]; rawI = self.buf[c_val]["I"].data(); rawQ = self.buf[c_val]["Q"].data() # Renamed c
            self.iq_workers[row_i] = True
            task = IQTask(row_i, c_val, rawI, rawQ, self.dot_px, mode_key, self.iq_signals)
            self.pool.start(task)
        else:
            concatI = np.concatenate([self.buf[ch]["I"].data() for ch in group])
            concatQ = np.concatenate([self.buf[ch]["Q"].data() for ch in group])
            big_size = concatI.size
            if big_size > 50000:
                stride = max(1, big_size // 50000)
                concatI = concatI[::stride]; concatQ = concatQ[::stride]
            if concatI.size > 1:
                self.iq_workers[row_i] = True
                task = IQTask(row_i, 0, concatI, concatQ, self.dot_px, mode_key, self.iq_signals)
                self.pool.start(task)

    def _dispatch_psd_tasks(self, row_i, group):
        # PSDTask from .tasks, convert_roc_to_volts from .utils
        if "S" in self.curves[row_i]:
            for ch_val in group: # Renamed ch
                if not self.psd_workers[row_i]["S"][ch_val]:
                    rawI = self.buf[ch_val]["I"].data(); rawQ = self.buf[ch_val]["Q"].data()
                    I_data, Q_data = (convert_roc_to_volts(d) for d in (rawI, rawQ)) if self.real_units else (rawI, rawQ) # Renamed I,Q
                    self.psd_workers[row_i]["S"][ch_val] = True
                    task = PSDTask(row_i, ch_val, I_data, Q_data, "SSB", self.dec_stage, self.real_units, self.psd_absolute, self.spin_segments.value(), self.psd_signals)
                    self.pool.start(task)
        if "D" in self.curves[row_i]:
            for ch_val in group: # Renamed ch
                if not self.psd_workers[row_i]["D"][ch_val]:
                    rawI = self.buf[ch_val]["I"].data(); rawQ = self.buf[ch_val]["Q"].data()
                    I_data, Q_data = (convert_roc_to_volts(d) for d in (rawI, rawQ)) if self.real_units else (rawI, rawQ) # Renamed I,Q
                    self.psd_workers[row_i]["D"][ch_val] = True
                    task = PSDTask(row_i, ch_val, I_data, Q_data, "DSB", self.dec_stage, self.real_units, self.psd_absolute, self.spin_segments.value(), self.psd_signals)
                    self.pool.start(task)

    def _update_performance_stats(self, now):
        if (now - self.t_last) >= 1.0:
            fps = self.frame_cnt / (now - self.t_last)
            pps = self.pkt_cnt / (now - self.t_last)
            self.statusBar().showMessage(f"FPS {fps:.1f} | Packets/s {pps:.1f}")
            self.frame_cnt = 0; self.pkt_cnt = 0; self.t_last = now

    def _update_dec_stage(self):
        # infer_dec_stage from .utils
        if not self.channel_list or not self.channel_list[0]: return
        ch_val = self.channel_list[0][0] # Renamed ch
        tarr = self.tbuf[ch_val].data()
        if len(tarr) < 2: return
        dt = (tarr[-1] - tarr[0]) / max(1, (len(tarr) - 1))
        fs = 1.0 / dt if dt > 0 else 1.0
        self.dec_stage = infer_dec_stage(fs)

    @QtCore.pyqtSlot(int, str, object)
    def _iq_done(self, row: int, task_mode: str, payload):
        self.iq_workers[row] = False
        if row >= len(self.curves) or "IQ" not in self.curves[row]: return
        pane = self.curves[row]["IQ"]
        if pane["mode"] != task_mode: return
        item = pane["item"]
        if task_mode == "density": self._update_density_image(item, payload)
        else: self._update_scatter_plot(item, payload)

    def _update_density_image(self, item, payload):
        # convert_roc_to_volts from .utils
        hist, (Imin, Imax, Qmin, Qmax) = payload
        if self.real_units:
            Imin, Imax = convert_roc_to_volts(np.array([Imin, Imax], dtype=float))
            Qmin, Qmax = convert_roc_to_volts(np.array([Qmin, Qmax], dtype=float))
        item.setImage(hist, levels=(0, 255), autoLevels=False)
        item.setRect(QtCore.QRectF(float(Imin), float(Qmin), float(Imax - Imin), float(Qmax - Qmin)))

    def _update_scatter_plot(self, item, payload):
        # convert_roc_to_volts, SCATTER_SIZE from .utils
        xs, ys, colors = payload
        if self.real_units: xs = convert_roc_to_volts(xs); ys = convert_roc_to_volts(ys)
        item.setData(xs, ys, brush=colors, pen=None, size=SCATTER_SIZE)

    @QtCore.pyqtSlot(int, str, int, object)
    def _psd_done(self, row: int, psd_mode: str, ch_val: int, payload): # Renamed ch
        if row not in self.psd_workers: return
        key = psd_mode[0]
        if key not in self.psd_workers[row] or ch_val not in self.psd_workers[row][key]: return
        self.psd_workers[row][key][ch_val] = False
        if row >= len(self.curves): return
        if psd_mode == "SSB": self._update_ssb_curves(row, ch_val, payload)
        else: self._update_dsb_curve(row, ch_val, payload)

    def _update_ssb_curves(self, row, ch_val, payload): # Renamed ch
        if "S" not in self.curves[row]: return
        sdict = self.curves[row]["S"]
        if ch_val not in sdict: return
        freq_i_data, psd_i_data, psd_q_data, psd_m_data, _, _, _ = payload
        freq_i = np.asarray(freq_i_data, dtype=float); psd_i = np.asarray(psd_i_data, dtype=float)
        psd_q = np.asarray(psd_q_data, dtype=float); psd_m = np.asarray(psd_m_data, dtype=float)
        if sdict[ch_val]["I"].isVisible(): sdict[ch_val]["I"].setData(freq_i, psd_i)
        if sdict[ch_val]["Q"].isVisible(): sdict[ch_val]["Q"].setData(freq_i, psd_q)
        if sdict[ch_val]["Mag"].isVisible(): sdict[ch_val]["Mag"].setData(freq_i, psd_m)

    def _update_dsb_curve(self, row, ch_val, payload): # Renamed ch
        if "D" not in self.curves[row]: return
        ddict = self.curves[row]["D"]
        if ch_val not in ddict: return
        freq_dsb_data, psd_dsb_data = payload
        freq_dsb = np.asarray(freq_dsb_data, dtype=float); psd_dsb = np.asarray(psd_dsb_data, dtype=float)
        if ddict[ch_val]["Cmplx"].isVisible(): ddict[ch_val]["Cmplx"].setData(freq_dsb, psd_dsb)

    def closeEvent(self, event):
        self.timer.stop(); self.receiver.stop(); self.receiver.wait()
        for task_key in list(self.netanal_tasks.keys()):
            task = self.netanal_tasks[task_key]; task.stop(); self.netanal_tasks.pop(task_key, None)
        for task_key in list(self.multisweep_tasks.keys()):
            task = self.multisweep_tasks[task_key]; task.stop(); self.multisweep_tasks.pop(task_key, None)
        if self.kernel_manager and self.kernel_manager.has_kernel:
            try: self.kernel_manager.shutdown_kernel()
            except Exception as e: warnings.warn(f"Error shutting down iPython kernel: {e}", RuntimeWarning) # warnings from .utils
        super().closeEvent(event); event.accept()

    def _add_interactive_console_dock(self):
        # QTCONSOLE_AVAILABLE, Qt from .utils
        if not QTCONSOLE_AVAILABLE: return
        self.console_dock_widget = QtWidgets.QDockWidget("Interactive iPython Session", self)
        self.console_dock_widget.setObjectName("InteractiveSessionDock")
        self.console_dock_widget.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.console_dock_widget.setVisible(False)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.console_dock_widget)

    def _toggle_interactive_session(self):
        # QTCONSOLE_AVAILABLE, QtInProcessKernelManager, RichJupyterWidget, rfmux, load_awaitless_extension from .utils
        # traceback from .utils
        if not QTCONSOLE_AVAILABLE or self.crs is None: return
        if self.console_dock_widget is None: return
        if self.kernel_manager is None:
            try:
                self.kernel_manager = QtInProcessKernelManager()
                self.kernel_manager.start_kernel()
                kernel = self.kernel_manager.kernel
                kernel.shell.push({'crs': self.crs, 'rfmux': rfmux, 'periscope': self})
                self.jupyter_widget = RichJupyterWidget()
                self.jupyter_widget.kernel_client = self.kernel_manager.client()
                self.jupyter_widget.kernel_client.start_channels()
                try: load_awaitless_extension(ipython=kernel.shell)
                except Exception as e_awaitless:
                    warnings.warn(f"Could not load awaitless extension: {e_awaitless}", RuntimeWarning)
                    traceback.print_exc()
                self.console_dock_widget.setWidget(self.jupyter_widget)
                self._update_console_style(self.dark_mode)
                style_sheet = (".in-prompt { color: #00FF00 !important; } .out-prompt { color: #00DD00 !important; } body { background-color: #1C1C1C; color: #DDDDDD; }" 
                               if self.dark_mode else 
                               ".in-prompt { color: #008800 !important; } .out-prompt { color: #006600 !important; } body { background-color: #FFFFFF; color: #000000; }")
                self.jupyter_widget._control.document().setDefaultStyleSheet(style_sheet)
                self.jupyter_widget.setFocus()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error Initializing Console", f"Could not initialize iPython console: {e}")
                traceback.print_exc()
                if self.kernel_manager and self.kernel_manager.has_kernel: self.kernel_manager.shutdown_kernel()
                self.kernel_manager = None; self.jupyter_widget = None
                self.console_dock_widget.setVisible(False); return
        is_visible = self.console_dock_widget.isVisible()
        self.console_dock_widget.setVisible(not is_visible)
        if not is_visible and self.jupyter_widget: self.jupyter_widget.setFocus()

    def _start_multisweep_analysis(self, params: dict):
        # MultisweepWindow from .ui, MultisweepTask from .tasks, sys, traceback from .utils
        try:
            if self.crs is None: QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available for multisweep."); return
            window_id = f"multisweep_window_{self.multisweep_window_count}"; self.multisweep_window_count += 1
            target_module = params.get('module')
            if target_module is None: QtWidgets.QMessageBox.critical(self, "Error", "Target module not specified for multisweep."); return
            dac_scales_for_window = self.dac_scales if hasattr(self, 'dac_scales') else {}
            window = MultisweepWindow(parent=self, target_module=target_module, initial_params=params.copy(), dac_scales=dac_scales_for_window)
            self.multisweep_windows[window_id] = {'window': window, 'params': params.copy()}
            try:
                self.multisweep_signals.progress.disconnect()
                self.multisweep_signals.intermediate_data_update.disconnect()
                self.multisweep_signals.data_update.disconnect()
                self.multisweep_signals.completed_amplitude.disconnect()
                self.multisweep_signals.all_completed.disconnect()
                self.multisweep_signals.error.disconnect()
            except TypeError: pass
            self.multisweep_signals.progress.connect(window.update_progress)
            self.multisweep_signals.intermediate_data_update.connect(window.update_intermediate_data)
            self.multisweep_signals.data_update.connect(window.update_data)
            self.multisweep_signals.completed_amplitude.connect(window.completed_amplitude_sweep)
            self.multisweep_signals.all_completed.connect(window.all_sweeps_completed)
            self.multisweep_signals.error.connect(window.handle_error)
            task = MultisweepTask(crs=self.crs, resonance_frequencies=params['resonance_frequencies'], params=params, signals=self.multisweep_signals)
            task_key = f"{window_id}_module_{target_module}"
            self.multisweep_tasks[task_key] = task; self.pool.start(task)
            window.show()
        except Exception as e:
            error_msg = f"Error starting multisweep analysis: {type(e).__name__}: {str(e)}"
            print(error_msg, file=sys.stderr); traceback.print_exc(file=sys.stderr)
            QtWidgets.QMessageBox.critical(self, "Multisweep Error", error_msg)

    def _start_multisweep_analysis_for_window(self, window_instance: MultisweepWindow, params: dict): # MultisweepWindow from .ui
        window_id = None
        for w_id, data in self.multisweep_windows.items():
            if data['window'] == window_instance: window_id = w_id; break
        if not window_id: QtWidgets.QMessageBox.critical(window_instance, "Error", "Could not find associated window to re-run multisweep."); return
        target_module = params.get('module')
        if target_module is None: QtWidgets.QMessageBox.critical(window_instance, "Error", "Target module not specified for multisweep re-run."); return
        old_task_key = f"{window_id}_module_{target_module}"
        if old_task_key in self.multisweep_tasks:
            old_task = self.multisweep_tasks.pop(old_task_key); old_task.stop()
        self.multisweep_windows[window_id]['params'] = params.copy()
        # MultisweepTask from .tasks
        task = MultisweepTask(crs=self.crs, resonance_frequencies=params['resonance_frequencies'], params=params, signals=self.multisweep_signals)
        self.multisweep_tasks[old_task_key] = task; self.pool.start(task)

    def stop_multisweep_task_for_window(self, window_instance: MultisweepWindow): # MultisweepWindow from .ui
        window_id = None; target_module = None
        for w_id, data in list(self.multisweep_windows.items()):
            if data['window'] == window_instance:
                window_id = w_id; target_module = data['params'].get('module'); break
        if window_id and target_module:
            task_key = f"{window_id}_module_{target_module}"
            if task_key in self.multisweep_tasks:
                task = self.multisweep_tasks.pop(task_key); task.stop()
            self.multisweep_windows.pop(window_id, None)

    def _update_console_style(self, dark_mode_enabled: bool):
        # QTCONSOLE_AVAILABLE from .utils
        if self.jupyter_widget and QTCONSOLE_AVAILABLE:
            self.jupyter_widget.syntax_style = 'monokai' if dark_mode_enabled else 'default'
            style_sheet = ("QWidget { background-color: #1C1C1C; color: #DDDDDD; } .in-prompt { color: #00FF00 !important; } .out-prompt { color: #00DD00 !important; } QPlainTextEdit { background-color: #1C1C1C; color: #DDDDDD; }"
                           if dark_mode_enabled else
                           "QWidget { background-color: #FFFFFF; color: #000000; } .in-prompt { color: #008800 !important; } .out-prompt { color: #006600 !important; } QPlainTextEdit { background-color: #FFFFFF; color: #000000; }")
            self.jupyter_widget.setStyleSheet(style_sheet)
            self.jupyter_widget.update(); self.jupyter_widget.repaint()
