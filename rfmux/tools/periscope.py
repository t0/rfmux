#!/usr/bin/env -S uv run
"""
Periscope – Real‑Time Multi‑Pane Viewer
=======================================

This module now delegates most of its implementation to helper modules
for clarity. The original functionality remains unchanged.
"""

from .periscope_utils import *  # constants and helper functions
from .periscope_tasks import *  # worker threads
from .periscope_ui import *     # dialog and window classes

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
        chan_str="1",
        buf_size=DEFAULT_BUFFER_SIZE,
        refresh_ms=DEFAULT_REFRESH_MS,
        dot_px=DENSITY_DOT_SIZE,
        crs=None,
    ):
        super().__init__()
        self.host = host
        self.module = module
        self.N = buf_size
        self.refresh_ms = refresh_ms
        self.dot_px = max(1, int(dot_px))
        self.crs = crs

        # Parse multi-channel format
        self.channel_list = parse_channels_multich(chan_str)

        # State variables
        self.paused = False
        self.start_time = None
        self.frame_cnt = 0
        self.pkt_cnt = 0
        self.t_last = time.time()

        # Single decimation stage, updated from first channel's sample rate
        self.dec_stage = 6
        self.last_dec_update = 0.0

        # Display settings
        self.dark_mode = True
        self.real_units = False
        self.psd_absolute = True
        self.auto_scale_plots = True
        self.show_i = True
        self.show_q = True
        self.show_m = True

        # Initialize worker tracking
        self._init_workers()

        # Create color map
        self._init_colormap()

        # Start UDP receiver
        self._init_receiver()

        # Initialize thread pool
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(4)  # Allow multiple network analyses

        # Build UI
        self._build_ui(chan_str)
        self._init_buffers()
        self._build_layout()

        # Start the GUI update timer
        self._start_timer()

    def _init_workers(self):
        """Initialize worker tracking structures."""
        # IQ concurrency tracking
        self.iq_workers: Dict[int, bool] = {}
        self.iq_signals = IQSignals()
        self.iq_signals.done.connect(self._iq_done)

        # PSD concurrency tracking, per row -> "S"/"D" -> channel -> bool
        self.psd_workers: Dict[int, Dict[str, Dict[int, bool]]] = {}
        for row_i, group in enumerate(self.channel_list):
            self.psd_workers[row_i] = {"S": {}, "D": {}}
            for c in group:
                self.psd_workers[row_i]["S"][c] = False
                self.psd_workers[row_i]["D"][c] = False

        self.psd_signals = PSDSignals()
        self.psd_signals.done.connect(self._psd_done)

        # Network analysis tracking
        self.netanal_signals = NetworkAnalysisSignals()
        self.netanal_signals.progress.connect(self._netanal_progress)
        self.netanal_signals.data_update.connect(self._netanal_data_update)
        self.netanal_signals.data_update_with_amp.connect(self._netanal_data_update_with_amp)
        self.netanal_signals.completed.connect(self._netanal_completed)
        self.netanal_signals.error.connect(self._netanal_error)

        # CRS Initialization signals
        self.crs_init_signals = CRSInitializeSignals()
        self.crs_init_signals.success.connect(self._crs_init_success)
        self.crs_init_signals.error.connect(self._crs_init_error)
        
        # Network analysis tracking
        self.netanal_windows = {}
        self.netanal_window_count = 0
        self.netanal_tasks = {}

        # Multisweep analysis tracking
        self.multisweep_signals = MultisweepSignals()
        # Connections will be made dynamically when a MultisweepWindow is created.
        self.multisweep_windows = {} # Dictionary of multisweep windows indexed by unique ID
        self.multisweep_window_count = 0 # Counter for multisweep window IDs
        self.multisweep_tasks = {} # Tasks dictionary for multisweep

        # Embedded iPython console attributes
        self.kernel_manager = None
        self.jupyter_widget = None
        self.console_dock_widget = None
        self.btn_interactive_session = None

    def _init_colormap(self):
        """Initialize the colormap for IQ density plots."""
        cmap = pg.colormap.get("turbo")
        lut_rgb = cmap.getLookupTable(0.0, 1.0, 255)
        self.lut = np.vstack([
            np.zeros((1, 4), np.uint8),
            np.hstack([lut_rgb, 255 * np.ones((255, 1), np.uint8)])
        ])

    def _init_receiver(self):
        """Initialize the UDP receiver."""
        self.receiver = UDPReceiver(self.host, self.module)
        self.receiver.start()

    def _start_timer(self):
        """Start the periodic GUI update timer."""
        self.timer = QtCore.QTimer(singleShot=False)
        self.timer.timeout.connect(self._update_gui)
        self.timer.start(self.refresh_ms)
        self.setWindowTitle("Periscope")

    # ───────────────────────── UI Construction ─────────────────────────
    def _build_ui(self, chan_str: str):
        """
        Create and configure all top-level widgets and layouts.
        
        Parameters
        ----------
        chan_str : str
            The user-supplied channel specification string.
        """
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_vbox = QtWidgets.QVBoxLayout(central)

        self._add_title(main_vbox)
        self._add_toolbar(main_vbox, chan_str)
        self._add_config_panel(main_vbox)
        self._add_plot_container(main_vbox)
        self._add_status_bar()
        self._add_interactive_console_dock() # Add dock for console

    def _add_title(self, layout):
        """Add the title to the layout."""
        title = QtWidgets.QLabel(f"CRS: {self.host}    Module: {self.module}")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        ft = title.font()
        ft.setPointSize(16)
        title.setFont(ft)
        layout.addWidget(title)

    def _add_toolbar(self, layout, chan_str):
        """Add the toolbar to the layout."""
        top_bar = QtWidgets.QWidget()
        top_h = QtWidgets.QHBoxLayout(top_bar)

        # Config toggle button
        self.btn_toggle_cfg = QtWidgets.QPushButton("Show Configuration")
        self.btn_toggle_cfg.setCheckable(True)
        self.btn_toggle_cfg.toggled.connect(self._toggle_config)

        # Channel input
        self.e_ch = QtWidgets.QLineEdit(chan_str)
        self.e_ch.setToolTip("Enter comma-separated channels or use '&' to group in one row.")
        self.e_ch.returnPressed.connect(self._update_channels)

        # Buffer size input
        self.e_buf = QtWidgets.QLineEdit(str(self.N))
        self.e_buf.setValidator(QIntValidator(10, 1_000_000, self))
        self.e_buf.setMaximumWidth(80)
        self.e_buf.setToolTip("Size of the ring buffer for each channel.")
        self.e_buf.editingFinished.connect(self._change_buffer)

        # Pause button
        self.b_pause = QtWidgets.QPushButton("Pause", clicked=self._toggle_pause)
        self.b_pause.setToolTip("Pause/resume real-time data.")

        # Real units checkbox
        self.cb_real = QtWidgets.QCheckBox("Real Units", checked=self.real_units)
        self.cb_real.setToolTip("Toggle raw 'counts' vs. real-voltage/dBm units.")
        self.cb_real.toggled.connect(self._toggle_real_units)

        # Mode checkboxes
        self.cb_time = QtWidgets.QCheckBox("TOD", checked=True)
        self.cb_iq = QtWidgets.QCheckBox("IQ", checked=False)
        self.cb_fft = QtWidgets.QCheckBox("FFT", checked=False)
        self.cb_ssb = QtWidgets.QCheckBox("Single Sideband PSD", checked=True)
        self.cb_dsb = QtWidgets.QCheckBox("Dual Sideband PSD", checked=False)
        for cb in (self.cb_time, self.cb_iq, self.cb_fft, self.cb_ssb, self.cb_dsb):
            cb.toggled.connect(self._build_layout)

        # Initialize CRS Board button
        self.btn_init_crs = QtWidgets.QPushButton("Initialize CRS Board")
        self.btn_init_crs.setToolTip("Open dialog to initialize CRS board.")
        self.btn_init_crs.clicked.connect(self._show_initialize_crs_dialog)
        if self.crs is None:
            self.btn_init_crs.setEnabled(False)
            self.btn_init_crs.setToolTip("CRS object not available - cannot initialize.")

        # Network Analysis button
        self.btn_netanal = QtWidgets.QPushButton("Network Analyzer")
        self.btn_netanal.setToolTip("Open network analysis configuration window.")
        self.btn_netanal.clicked.connect(self._show_netanal_dialog)
        if self.crs is None:
            self.btn_netanal.setEnabled(False)
            self.btn_netanal.setToolTip("CRS object not available - cannot run network analysis.")

        # Help button
        self.btn_help = QtWidgets.QPushButton("Help")
        self.btn_help.setToolTip("Show usage, interaction details, and examples.")
        self.btn_help.clicked.connect(self._show_help)

        # Add widgets to toolbar
        top_h.addWidget(QtWidgets.QLabel("Channels:"))
        top_h.addWidget(self.e_ch)
        top_h.addSpacing(20)
        top_h.addWidget(QtWidgets.QLabel("Buffer:"))
        top_h.addWidget(self.e_buf)
        top_h.addWidget(self.b_pause)
        top_h.addSpacing(30)
        top_h.addWidget(self.cb_real)
        top_h.addSpacing(30)
        for cb in (self.cb_time, self.cb_iq, self.cb_fft, self.cb_ssb, self.cb_dsb):
            top_h.addWidget(cb)
        top_h.addStretch(1)
        # self.btn_init_crs, self.btn_netanal and self.btn_help will be moved to the config_header_layout

        layout.addWidget(top_bar)

    def _add_config_panel(self, layout):
        """Add the configuration panel and the row with config/netanal/help buttons."""
        
        # Create a new HBox for the buttons above the config panel
        action_buttons_widget = QtWidgets.QWidget()
        action_buttons_layout = QtWidgets.QHBoxLayout(action_buttons_widget)
        action_buttons_layout.setContentsMargins(0,0,0,0) # Remove margins for tighter packing
        action_buttons_layout.addStretch(1)
        
        # Interactive Session Button
        self.btn_interactive_session = QtWidgets.QPushButton("Interactive Session")
        self.btn_interactive_session.setToolTip("Toggle an embedded iPython interactive session.")
        self.btn_interactive_session.clicked.connect(self._toggle_interactive_session)
        if not QTCONSOLE_AVAILABLE or self.crs is None:
            self.btn_interactive_session.setEnabled(False)
            if not QTCONSOLE_AVAILABLE:
                self.btn_interactive_session.setToolTip("Interactive session disabled: qtconsole/ipykernel not installed.")
            else:
                self.btn_interactive_session.setToolTip("Interactive session disabled: CRS object not available.")
        action_buttons_layout.addWidget(self.btn_interactive_session)

        action_buttons_layout.addWidget(self.btn_init_crs)
        action_buttons_layout.addWidget(self.btn_netanal) # Add Network Analyzer button
        action_buttons_layout.addWidget(self.btn_toggle_cfg)
        action_buttons_layout.addWidget(self.btn_help)    # Add Help button
        
        layout.addWidget(action_buttons_widget)

        self.ctrl_panel = QtWidgets.QGroupBox("Configuration")
        self.ctrl_panel.setVisible(False)
        cfg_hbox = QtWidgets.QHBoxLayout(self.ctrl_panel)

        # Show Curves group
        cfg_hbox.addWidget(self._create_show_curves_group())

        # IQ Mode group
        cfg_hbox.addWidget(self._create_iq_mode_group())

        # PSD Mode group
        cfg_hbox.addWidget(self._create_psd_mode_group())

        # General Display group
        cfg_hbox.addWidget(self._create_display_group())

        layout.addWidget(self.ctrl_panel)

    def _create_show_curves_group(self):
        """Create the Show Curves configuration group."""
        show_curves_g = QtWidgets.QGroupBox("Show Curves")
        show_curves_h = QtWidgets.QHBoxLayout(show_curves_g)
        
        self.cb_show_i = QtWidgets.QCheckBox("I", checked=True)
        self.cb_show_q = QtWidgets.QCheckBox("Q", checked=True)
        self.cb_show_m = QtWidgets.QCheckBox("Magnitude", checked=True)
        
        self.cb_show_i.toggled.connect(self._toggle_iqmag)
        self.cb_show_q.toggled.connect(self._toggle_iqmag)
        self.cb_show_m.toggled.connect(self._toggle_iqmag)
        
        show_curves_h.addWidget(self.cb_show_i)
        show_curves_h.addWidget(self.cb_show_q)
        show_curves_h.addWidget(self.cb_show_m)
        
        return show_curves_g

    def _create_iq_mode_group(self):
        """Create the IQ Mode configuration group."""
        iq_g = QtWidgets.QGroupBox("IQ Mode")
        iq_h = QtWidgets.QHBoxLayout(iq_g)
        
        self.rb_density = QtWidgets.QRadioButton("Density", checked=True)
        self.rb_density.setToolTip("2D histogram of I/Q values.")
        self.rb_scatter = QtWidgets.QRadioButton("Scatter")
        self.rb_scatter.setToolTip("Scatter of up to 1,000 I/Q points. CPU intensive.")
        
        rb_group = QtWidgets.QButtonGroup(iq_g)
        rb_group.addButton(self.rb_density)
        rb_group.addButton(self.rb_scatter)
        
        for rb in (self.rb_density, self.rb_scatter):
            rb.toggled.connect(self._build_layout)
            
        iq_h.addWidget(self.rb_density)
        iq_h.addWidget(self.rb_scatter)
        
        return iq_g

    def _create_psd_mode_group(self):
        """Create the PSD Mode configuration group."""
        psd_g = QtWidgets.QGroupBox("PSD Mode")
        psd_grid = QtWidgets.QGridLayout(psd_g)
        
        self.lbl_psd_scale = QtWidgets.QLabel("PSD Scale:")
        self.rb_psd_abs = QtWidgets.QRadioButton("Absolute (dBm)", checked=True)
        self.rb_psd_rel = QtWidgets.QRadioButton("Relative (dBc)")
        
        for rb in (self.rb_psd_abs, self.rb_psd_rel):
            rb.toggled.connect(self._psd_ref_changed)
        
        self.spin_segments = QtWidgets.QSpinBox()
        self.spin_segments.setRange(1, 256)
        self.spin_segments.setValue(1)
        self.spin_segments.setMaximumWidth(80)
        self.spin_segments.setToolTip("Number of segments for Welch PSD averaging.")
        
        psd_grid.addWidget(self.lbl_psd_scale, 0, 0)
        psd_grid.addWidget(self.rb_psd_abs, 0, 1)
        psd_grid.addWidget(self.rb_psd_rel, 0, 2)
        psd_grid.addWidget(QtWidgets.QLabel("Segments:"), 1, 0)
        psd_grid.addWidget(self.spin_segments, 1, 1)
        
        return psd_g

    def _create_display_group(self):
        """Create the General Display configuration group."""
        disp_g = QtWidgets.QGroupBox("General Display")
        disp_h = QtWidgets.QHBoxLayout(disp_g)

        # Add Zoom Box Mode checkbox
        self.cb_zoom_box = QtWidgets.QCheckBox("Zoom Box Mode", checked=True)
        self.cb_zoom_box.setToolTip("When enabled, left-click drag creates a zoom box. When disabled, left-click drag pans.")
        self.cb_zoom_box.toggled.connect(self._toggle_zoom_box_mode)
        disp_h.addWidget(self.cb_zoom_box)

        # Dark Mode Toggle
        self.cb_dark = QtWidgets.QCheckBox("Dark Mode", checked=self.dark_mode)
        self.cb_dark.setToolTip("Switch between dark/light themes.")
        self.cb_dark.toggled.connect(self._toggle_dark_mode)
        self.cb_dark.toggled.connect(self._update_console_style) # Connect to console style update
        disp_h.addWidget(self.cb_dark)

        # Autoscale button
        self.cb_auto_scale = QtWidgets.QCheckBox("Auto Scale", checked=self.auto_scale_plots)
        self.cb_auto_scale.setToolTip("Enable/disable auto-range for IQ/FFT/SSB/DSB. Can improve display performance.")
        self.cb_auto_scale.toggled.connect(self._toggle_auto_scale)
        disp_h.addWidget(self.cb_auto_scale)
        
        return disp_g

    def _add_plot_container(self, layout):
        """Add the plot container to the layout."""
        self.container = QtWidgets.QWidget()
        layout.addWidget(self.container)
        self.grid = QtWidgets.QGridLayout(self.container)

    def _add_status_bar(self):
        """Add the status bar."""
        self.setStatusBar(QtWidgets.QStatusBar())

    def _show_help(self):
        """
        Show a dialog containing usage instructions, interaction details,
        and example commands.
        """
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Periscope Help")
        help_text = (
            "**Usage:**\n"
            "  - Multi-channel grouping: use '&' to display multiple channels in one row.\n"
            "    e.g., \"3&5\" for channels 3 and 5 in one row, \"3&5,7\" for that row plus a row with channel 7.\n\n"
            "**Standard PyQtGraph Interactions:**\n"
            "  - Pan: Left-click and drag (when Zoom Box Mode is disabled).\n"
            "  - Zoom Box: Left-click and drag to create a selection rectangle (when enabled).\n"
            "  - Zoom: Right-click/drag or mouse-wheel in most plots.\n"
            "  - Axis scaling (log vs lin) and auto-zooming: Individually configurable by right-clicking any plot.\n"
            "  - Double-click within a plot: Show the coordinates of the clicked position.\n"
            "  - Export plot to CSV, Image, Vector Graphics, or interactive Matplotlib window: Right-click -> Export\n"
            "  - See PyQtGraph docs for more.\n\n"
            "**Network Analysis:**\n"
            "  - Click the 'Network Analysis' button to open the configuration dialog.\n"
            "  - Configure parameters like frequency range, amplitude, and number of points.\n"
            "  - Multiple amplitudes: Enter comma-separated values to run analysis at multiple power levels.\n"
            "  - Unit conversion: Configure amplitudes in normalized values or dBm power.\n"
            "  - Cable length: Compensate for phase due to cables by estimating cable lengths.\n"
            "  - Analysis runs in a separate window with real-time updates.\n"
            "  - Use the Unit selector to view data in Counts, Volts, or dBm, or normalized versions to compare relative responses.\n"
            "  - Export data from Network Analysis with the 'Export Data' button. Pickle and CSV exports available.\n\n"
            "**Interactive Session (Embedded Console):**\n"
            "  - Click the 'Interactive Session' button to toggle a drop-down iPython console.\n"
            "  - Requires `qtconsole` and `ipykernel` (`pip install qtconsole ipykernel`).\n"
            "  - The console has access to:\n"
            "    - `crs`: The current CRS object used by Periscope.\n"
            "    - `rfmux`: The rfmux module.\n"
            "    - `periscope`: The Periscope application instance itself.\n"
            "  - 'Awaitless' mode is enabled, so `await` is often not needed for async CRS calls.\n"
            "  - The console theme (dark/light) syncs with Periscope's Dark Mode setting.\n"
            "  - The console panel can be resized by dragging its top border.\n\n"
            "**Performance tips for higher FPS:**\n"
            "  - Disable auto-scaling in the configuration panel\n"
            "  - Use the density mode for IQ, smaller buffers, or enable only a subset of I,Q,M.\n"
            "  - Periscope is limited by the single-thread renderer. Launching multiple instances can improve performance for viewing many channels.\n\n"
            "**Command-Line Examples:**\n"
            "  - `$ periscope rfmux0022.local --module 2 --channels \"3&5,7\"`\n\n"
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
        help_label.setOpenExternalLinks(True) # Allow opening links if any in markdown
        
        scroll_area.setWidget(help_label)
        layout.addWidget(scroll_area)
        
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(help_dialog.accept)
        
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

        help_dialog.setLayout(layout)
        # Set a reasonable default size for the help dialog, making it large enough
        # for typical content but allowing scrollbars for smaller screens or very long text.
        help_dialog.resize(700, 500) 
        help_dialog.exec()

    def _toggle_zoom_box_mode(self, enable: bool):
        """
        Enable or disable zoom box mode for all plot viewboxes.

        Parameters
        ----------
        enable : bool
            If True, enables rectangle selection zooming. If False, returns to pan mode.
        """
        if not hasattr(self, "plots"):
            return
        self.zoom_box_mode = enable  # Store the state            

        for rowPlots in self.plots:
            for mode, plot_widget in rowPlots.items():
                viewbox = plot_widget.getViewBox()
                if isinstance(viewbox, ClickableViewBox):
                    viewbox.enableZoomBoxMode(enable)
        
        # Also update any network analysis plots if they exist
        for window_id, window_data in self.netanal_windows.items():
            window = window_data.get('window')
            if window:
                for module in window.plots:
                    for plot_type in ['amp_plot', 'phase_plot']:
                        viewbox = window.plots[module][plot_type].getViewBox()
                        if isinstance(viewbox, ClickableViewBox):
                            viewbox.enableZoomBoxMode(enable)

                # Also update the Network Analysis window's zoom box checkbox if it exists
                if hasattr(window, 'zoom_box_cb'):
                    window.zoom_box_cb.setChecked(enable)
        
    def _show_initialize_crs_dialog(self):
        """Show the CRS initialization dialog."""
        if self.crs is None:
            QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available for initialization.")
            return

        # Check if TIMESTAMP_PORT is available
        if not hasattr(self.crs, 'TIMESTAMP_PORT') or \
           not hasattr(self.crs.TIMESTAMP_PORT, 'BACKPLANE') or \
           not hasattr(self.crs.TIMESTAMP_PORT, 'TEST') or \
           not hasattr(self.crs.TIMESTAMP_PORT, 'SMA'):
            QtWidgets.QMessageBox.critical(self, "Error", 
                                           "CRS.TIMESTAMP_PORT enum not available or incomplete. Cannot initialize.")
            return

        dialog = InitializeCRSDialog(self, self.crs)
        if dialog.exec():
            irig_source = dialog.get_selected_irig_source()
            clear_channels = dialog.get_clear_channels_state()
            
            if irig_source is None:
                QtWidgets.QMessageBox.warning(self, "Selection Error", "No IRIG source selected.")
                return

            # Prepare and run the initialization task
            task = CRSInitializeTask(self.crs, self.module, irig_source, clear_channels, self.crs_init_signals)
            self.pool.start(task)


    def _show_netanal_dialog(self):
        """Show the network analysis configuration dialog."""
        if self.crs is None:
            QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available")
            return
        
        # Create a dialog with default DAC scales
        default_dac_scales = {m: -0.5 for m in range(1, 9)}
        dialog = NetworkAnalysisDialog(self, modules=list(range(1, 9)), dac_scales=default_dac_scales)
        dialog.module_entry.setText(str(self.module))
        
        # Start fetching DAC scales in background
        fetcher = DACScaleFetcher(self.crs)
        fetcher.dac_scales_ready.connect(lambda scales: dialog.dac_scales.update(scales))
        fetcher.dac_scales_ready.connect(dialog._update_dac_scale_info)
        fetcher.dac_scales_ready.connect(dialog._update_dbm_from_normalized)
        
        # Store DAC scales in our main class when they're ready
        fetcher.dac_scales_ready.connect(lambda scales: setattr(self, 'dac_scales', scales))
        
        fetcher.start()
        
        # Show dialog
        if dialog.exec():
            # Store the dialog's DAC scales in our class when dialog is accepted
            self.dac_scales = dialog.dac_scales.copy()
            
            params = dialog.get_parameters()
            if params:
                self._start_network_analysis(params)

    def _start_network_analysis(self, params: dict):
        """Start network analysis on selected modules."""
        try:
            if self.crs is None:
                QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available")
                return
                    
            # Determine which modules to run
            selected_module = params.get('module')
            if selected_module is None:
                modules = list(range(1, 9))  # All modules
            elif isinstance(selected_module, list):
                modules = selected_module
            else:
                modules = [selected_module]

            # Verify DAC scales are available
            if not hasattr(self, 'dac_scales'):
                QtWidgets.QMessageBox.critical(self, "Error", 
                    "DAC scales are not available. Please run the network analysis configuration again.")
                return
            
            # Create a unique window ID
            window_id = f"window_{self.netanal_window_count}"
            self.netanal_window_count += 1
            
            # Create window-specific signal handlers
            window_signals = NetworkAnalysisSignals()
            
            # Use actual DAC scales, no defaults
            dac_scales = self.dac_scales.copy()
            
            # Create a new window with DAC scales
            window = NetworkAnalysisWindow(self, modules, dac_scales)
            window.set_params(params)
            window.window_id = window_id  # Attach ID to window
            
            # Store window in dictionary
            self.netanal_windows[window_id] = {
                'window': window,
                'signals': window_signals,
                'amplitude_queues': {},
                'current_amp_index': {}
            }
            
            # Connect signals for this specific window
            window_signals.progress.connect(
                lambda module, progress: window.update_progress(module, progress))
            window_signals.data_update.connect(
                lambda module, freqs, amps, phases: window.update_data(module, freqs, amps, phases))
            window_signals.data_update_with_amp.connect(
                lambda module, freqs, amps, phases, amplitude: 
                window.update_data_with_amp(module, freqs, amps, phases, amplitude))
            window_signals.completed.connect(
                lambda module: self._handle_analysis_completed(module, window_id))
            window_signals.error.connect(
                lambda error_msg: QtWidgets.QMessageBox.critical(window, "Network Analysis Error", error_msg))
            
            # Get amplitudes from params
            amplitudes = params.get('amps', [params.get('amp', DEFAULT_AMPLITUDE)])
            
            # Set up amplitude queues for this window
            window_data = self.netanal_windows[window_id]
            window_data['amplitude_queues'] = {module: list(amplitudes) for module in modules}
            window_data['current_amp_index'] = {module: 0 for module in modules}
            
            # Update progress displays
            for module in modules:
                window.update_amplitude_progress(module, 1, len(amplitudes), amplitudes[0])
                    
                # Start the first amplitude task
                self._start_next_amplitude_task(module, params, window_id)
            
            # Show the window
            window.show()
        except Exception as e:
            print(f"Error in _start_network_analysis: {e}")
            traceback.print_exc()
    
    def _handle_analysis_completed(self, module: int, window_id: str):
        """Handle completion of a network analysis task for a specific window."""
        try:
            if window_id not in self.netanal_windows:
                return  # Window was closed
                
            window_data = self.netanal_windows[window_id]
            window = window_data['window']
            
            # Update window
            window.complete_analysis(module)
            
            # Clean up tasks
            for task_key in list(self.netanal_tasks.keys()):
                if task_key.startswith(f"{window_id}_{module}_"):
                    self.netanal_tasks.pop(task_key, None)
            
            # Check for more amplitudes
            if module in window_data['amplitude_queues'] and window_data['amplitude_queues'][module]:
                # Update index
                window_data['current_amp_index'][module] += 1
                
                # Update display
                total_amps = len(window.original_params.get('amps', []))
                next_amp = window_data['amplitude_queues'][module][0]
                window.update_amplitude_progress(
                    module, 
                    window_data['current_amp_index'][module] + 1,
                    total_amps,
                    next_amp
                )
                
                # Reset progress bar
                if module in window.progress_bars:
                    window.progress_bars[module].setValue(0)
                    if window.progress_group:
                        window.progress_group.setVisible(True)
                
                # Start next task
                self._start_next_amplitude_task(module, window.original_params, window_id)
        except Exception as e:
            print(f"Error in _handle_analysis_completed: {e}")
            traceback.print_exc()    

    def _start_next_amplitude_task(self, module: int, params: dict, window_id: str):
        """Start the next amplitude task for a module in a specific window."""
        try:
            if window_id not in self.netanal_windows:
                return  # Window was closed
                
            window_data = self.netanal_windows[window_id]
            window = window_data['window']
            signals = window_data['signals']
            
            if module not in window_data['amplitude_queues'] or not window_data['amplitude_queues'][module]:
                return  # No more amplitudes to process
            
            # Get the next amplitude
            amplitude = window_data['amplitude_queues'][module][0]
            window_data['amplitude_queues'][module].pop(0)  # Remove it from the queue
            
            # Create task parameters
            task_params = params.copy()
            task_params['module'] = module
            
            # Get specific cable length for this module if provided (Requirement 2)
            module_specific_cable_length = params.get('module_cable_lengths', {}).get(module)
            if module_specific_cable_length is not None:
                task_params['cable_length'] = module_specific_cable_length
            # else, it will use the global 'cable_length' if present in params, or default in task
            
            # Create a unique task key
            task_key = f"{window_id}_{module}_amp_{amplitude}"
            
            # Create and start the task
            # Pass cable_length_override to the task if it was specifically found for the module
            task = NetworkAnalysisTask(
                self.crs, 
                module, 
                task_params, # This now contains the potentially module-specific cable_length
                signals, 
                amplitude=amplitude
            )
            self.netanal_tasks[task_key] = task
            self.pool.start(task)
        except Exception as e:
            print(f"Error in _start_next_amplitude_task: {e}")
            traceback.print_exc()
    
    def _rerun_network_analysis(self, params: dict):
        """Rerun network analysis in the window that triggered this call."""
        try:
            if self.crs is None:
                QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available")
                return
            
            # Find which window triggered this
            sender = self.sender()
            source_window = None
            window_id = None
            
            if sender and hasattr(sender, 'window'):
                source_window = sender.window()
            
            # Find matching window in our dictionary
            for w_id, w_data in self.netanal_windows.items():
                if w_data['window'] == source_window:
                    window_id = w_id
                    break
                    
            if not window_id:
                # Could not determine which window called us
                return
                
            window_data = self.netanal_windows[window_id]
            window = window_data['window']
            signals = window_data['signals']
            
            # Update window
            window.data.clear()
            window.raw_data.clear()
            for module, pbar in window.progress_bars.items():
                pbar.setValue(0)
            window.clear_plots()
            window.set_params(params)
            
            # Determine modules
            selected_module = params.get('module')
            if selected_module is None:
                modules = list(range(1, 9))
            elif isinstance(selected_module, list):
                modules = selected_module
            else:
                modules = [selected_module]
            
            # Stop existing tasks for this window
            for task_key in list(self.netanal_tasks.keys()):
                if task_key.startswith(f"{window_id}_"):
                    task = self.netanal_tasks.pop(task_key)
                    task.stop()
            
            # Get amplitudes
            amplitudes = params.get('amps', [params.get('amp', DEFAULT_AMPLITUDE)])
            
            # Reset amplitude queues
            window_data['amplitude_queues'] = {module: list(amplitudes) for module in modules}
            window_data['current_amp_index'] = {module: 0 for module in modules}
            
            # Make progress visible
            if window.progress_group:
                window.progress_group.setVisible(True)
            
            # Start new tasks
            for module in modules:
                window.update_amplitude_progress(module, 1, len(amplitudes), amplitudes[0])
                self._start_next_amplitude_task(module, params, window_id)
        except Exception as e:
            print(f"Error in _rerun_network_analysis: {e}")
            traceback.print_exc()
    
    def _netanal_progress(self, module: int, progress: float):
        """Handle network analysis progress updates."""
        # This function is not used directly, as we route signals to specific windows
        pass
    
    def _netanal_data_update(self, module: int, freqs: np.ndarray, amps: np.ndarray, phases: np.ndarray):
        """Handle network analysis data updates."""
        # This function is not used directly, as we route signals to specific windows
        pass
    
    def _netanal_data_update_with_amp(self, module: int, freqs: np.ndarray, amps: np.ndarray, phases: np.ndarray, amplitude: float):
        """Handle network analysis data updates with amplitude information."""
        # This function is not used directly, as we route signals to specific windows
        pass
    
    def _netanal_completed(self, module: int):
        """Handle network analysis completion."""
        # This function is not used directly, as we route signals to specific windows
        pass
    
    def _netanal_error(self, error_msg: str):
        """Handle network analysis errors."""
        QtWidgets.QMessageBox.critical(self, "Network Analysis Error", error_msg)

    def _crs_init_success(self, message: str):
        """Handle successful CRS initialization."""
        QtWidgets.QMessageBox.information(self, "CRS Initialization Success", message)

    def _crs_init_error(self, error_msg: str):
        """Handle CRS initialization errors."""
        QtWidgets.QMessageBox.critical(self, "CRS Initialization Error", error_msg)

    def _toggle_config(self, visible: bool):
        """
        Show or hide the advanced configuration panel.

        Parameters
        ----------
        visible : bool
            True to show, False to hide.
        """
        self.ctrl_panel.setVisible(visible)
        self.btn_toggle_cfg.setText("Hide Configuration" if visible else "Show Configuration")

    def _toggle_auto_scale(self, checked: bool):
        """
        Enable or disable auto-ranging for non-TOD plots.

        Parameters
        ----------
        checked : bool
            If True, IQ/FFT/SSB/DSB will auto-range. If False, they remain fixed.
        """
        self.auto_scale_plots = checked
        if hasattr(self, "plots"):
            for rowPlots in self.plots:
                for mode, pw in rowPlots.items():
                    if mode != "T":
                        pw.enableAutoRange(pg.ViewBox.XYAxes, checked)

    def _toggle_iqmag(self):
        """
        Globally hide or show I/Q/M lines in TOD, FFT, and SSB (DSB unaffected).
        """
        self.show_i = self.cb_show_i.isChecked()
        self.show_q = self.cb_show_q.isChecked()
        self.show_m = self.cb_show_m.isChecked()

        if not hasattr(self, "curves"):
            return
        for rowCurves in self.curves:
            for mode in ("T", "F", "S"):
                if mode in rowCurves:
                    subdict = rowCurves[mode]
                    for ch, cset in subdict.items():
                        if "I" in cset:
                            cset["I"].setVisible(self.show_i)
                        if "Q" in cset:
                            cset["Q"].setVisible(self.show_q)
                        if "Mag" in cset:
                            cset["Mag"].setVisible(self.show_m)

    def _init_buffers(self):
        """
        Recreate ring buffers for all unique channels referenced in self.channel_list.
        """
        unique_chs = set()
        for group in self.channel_list:
            for c in group:
                unique_chs.add(c)
        self.all_chs = sorted(unique_chs)
        self.buf = {}
        self.tbuf = {}
        for ch in self.all_chs:
            self.buf[ch] = {k: Circular(self.N) for k in ("I", "Q", "M")}
            self.tbuf[ch] = Circular(self.N)

    def _build_layout(self):
        """
        Construct the layout grid for each row in self.channel_list and each enabled mode.
        """
        self._clear_current_layout()
        modes = self._get_active_modes()
        self.plots = []
        self.curves = []

        # Create font for labels
        font = QFont()
        font.setPointSize(UI_FONT_SIZE)

        # Define color configurations
        single_colors = self._get_single_channel_colors()
        channel_families = self._get_channel_color_families()

        # Build the layout for each row
        for row_i, group in enumerate(self.channel_list):
            rowPlots, rowCurves = self._create_row_plots_and_curves(row_i, group, modes, font, single_colors, channel_families)
            self.plots.append(rowPlots)
            self.curves.append(rowCurves)

        # Re-enable auto-range after building
        self._restore_auto_range_settings()

        # Apply "Show Curves" toggles
        self._toggle_iqmag()

        # Apply zoom box mode to all plots
        if hasattr(self, "zoom_box_mode"):
            self._toggle_zoom_box_mode(self.zoom_box_mode)

    def _clear_current_layout(self):
        """Clear the current layout."""
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _get_active_modes(self):
        """Get the list of active visualization modes."""
        modes = []
        if self.cb_time.isChecked():
            modes.append("T")
        if self.cb_iq.isChecked():
            modes.append("IQ")
        if self.cb_fft.isChecked():
            modes.append("F")
        if self.cb_ssb.isChecked():
            modes.append("S")
        if self.cb_dsb.isChecked():
            modes.append("D")
        return modes

    def _get_single_channel_colors(self):
        """Get color definitions for single-channel plots."""
        return {
            "I": "#1f77b4",
            "Q": "#ff7f0e",
            "Mag": "#2ca02c",
            "DSB": "#bcbd22",
        }

    def _get_channel_color_families(self):
        """Get color families for multi-channel plots."""
        return [
            ("#1f77b4", "#4a8cc5", "#90bce0"),
            ("#ff7f0e", "#ffa64d", "#ffd2a3"),
            ("#2ca02c", "#63c063", "#a1d9a1"),
            ("#d62728", "#eb6a6b", "#f2aeae"),
            ("#9467bd", "#ae8ecc", "#d3c4e3"),
            ("#8c564b", "#b58e87", "#d9c3bf"),
            ("#e377c2", "#f0a8dc", "#f7d2ee"),
            ("#7f7f7f", "#aaaaaa", "#d3d3d3"),
            ("#bcbd22", "#cfd342", "#e2e795"),
            ("#17becf", "#51d2de", "#9ae8f2"),
        ]

    def _create_row_plots_and_curves(self, row_i, group, modes, font, single_colors, channel_families):
        """Create plots and curves for a single row."""
        rowPlots = {}
        rowCurves = {}

        row_title = "Ch " + ("&".join(map(str, group)) if len(group) > 1 else str(group[0]))

        for col, mode in enumerate(modes):
            vb = ClickableViewBox()
            pw = pg.PlotWidget(viewBox=vb, title=f"{mode_title(mode)} – {row_title}")

            # Configure auto-range based on mode
            self._configure_plot_auto_range(pw, mode)
            
            # Set up plot labels and axes
            self._configure_plot_axes(pw, mode)

            # Apply theme and grid
            self._apply_plot_theme(pw)
            pw.showGrid(x=True, y=True, alpha=0.3)
            self.grid.addWidget(pw, row_i, col)
            rowPlots[mode] = pw

            # Apply font to axes
            self._configure_plot_fonts(pw, font)

            # Create curves or image items based on mode
            if mode == "IQ":
                rowCurves["IQ"] = self._create_iq_plot_item(pw)
            else:
                # Add legend
                legend = pw.addLegend(offset=(30, 10))
                
                # Create curves based on channel count
                if len(group) == 1:
                    ch = group[0]
                    rowCurves[mode] = self._create_single_channel_curves(pw, mode, ch, single_colors, legend)
                else:
                    rowCurves[mode] = self._create_multi_channel_curves(pw, mode, group, channel_families, legend)
                
                # Make legend entries clickable
                self._make_legend_clickable(legend)

        return rowPlots, rowCurves

    def _configure_plot_auto_range(self, pw, mode):
        """Configure plot auto-range behavior based on mode."""
        if mode == "T":
            pw.enableAutoRange(pg.ViewBox.XYAxes, True)
        else:
            pw.enableAutoRange(pg.ViewBox.XYAxes, self.auto_scale_plots)

    def _configure_plot_axes(self, pw, mode):
        """Configure plot axes based on mode."""
        if mode == "T":
            if not self.real_units:
                pw.setLabel("left", "Amplitude", units="Counts")
            else:
                pw.setLabel("left", "Amplitude", units="V")
        elif mode == "IQ":
            pw.getViewBox().setAspectLocked(True)
            if not self.real_units:
                pw.setLabel("bottom", "I", units="Counts")
                pw.setLabel("left",   "Q", units="Counts")
            else:
                pw.setLabel("bottom", "I", units="V")
                pw.setLabel("left",   "Q", units="V")
        elif mode == "F":
            pw.setLogMode(x=True, y=True)
            pw.setLabel("bottom", "Freq", units="Hz")
            if not self.real_units:
                pw.setLabel("left", "Amplitude", units="Counts")
            else:
                pw.setLabel("left", "Amplitude", units="V")
        elif mode == "S":
            pw.setLogMode(x=True, y=not self.real_units)
            pw.setLabel("bottom", "Freq", units="Hz")
            if not self.real_units:
                pw.setLabel("left", "PSD (Counts²/Hz)")
            else:
                lbl = "dBm/Hz" if self.psd_absolute else "dBc/Hz"
                pw.setLabel("left", f"PSD ({lbl})")
        else:  # "D"
            pw.setLogMode(x=False, y=not self.real_units)
            pw.setLabel("bottom", "Freq", units="Hz")
            if not self.real_units:
                pw.setLabel("left", "PSD (Counts²/Hz)")
            else:
                lbl = "dBm/Hz" if self.psd_absolute else "dBc/Hz"
                pw.setLabel("left", f"PSD ({lbl})")

    def _configure_plot_fonts(self, pw, font):
        """Configure plot font settings."""
        pi = pw.getPlotItem()
        for axis_name in ("left", "bottom", "right", "top"):
            axis = pi.getAxis(axis_name)
            if axis:
                axis.setTickFont(font)
                if axis.label:
                    axis.label.setFont(font)
        pi.titleLabel.setFont(font)

    def _create_iq_plot_item(self, pw):
        """Create an IQ plot item based on selected mode."""
        if self.rb_scatter.isChecked():
            sp = pg.ScatterPlotItem(pen=None, size=SCATTER_SIZE)
            pw.addItem(sp)
            return {"mode": "scatter", "item": sp}
        else:
            img = pg.ImageItem(axisOrder="row-major")
            img.setLookupTable(self.lut)
            pw.addItem(img)
            return {"mode": "density", "item": img}

    def _create_single_channel_curves(self, pw, mode, ch, single_colors, legend):
        """Create curves for a single-channel plot."""
        if mode == "T":
            cI = pw.plot(pen=pg.mkPen(single_colors["I"],   width=LINE_WIDTH), name="I")
            cQ = pw.plot(pen=pg.mkPen(single_colors["Q"],   width=LINE_WIDTH), name="Q")
            cM = pw.plot(pen=pg.mkPen(single_colors["Mag"], width=LINE_WIDTH), name="Mag")
            self._fade_hidden_entries(legend, ("I", "Q"))
            return {ch: {"I": cI, "Q": cQ, "Mag": cM}}
        elif mode == "F":
            cI = pw.plot(pen=pg.mkPen(single_colors["I"],   width=LINE_WIDTH), name="I")
            cI.setFftMode(True)
            cQ = pw.plot(pen=pg.mkPen(single_colors["Q"],   width=LINE_WIDTH), name="Q")
            cQ.setFftMode(True)
            cM = pw.plot(pen=pg.mkPen(single_colors["Mag"], width=LINE_WIDTH), name="Mag")
            cM.setFftMode(True)
            self._fade_hidden_entries(legend, ("I", "Q"))
            return {ch: {"I": cI, "Q": cQ, "Mag": cM}}
        elif mode == "S":
            cI = pw.plot(pen=pg.mkPen(single_colors["I"],   width=LINE_WIDTH), name="I")
            cQ = pw.plot(pen=pg.mkPen(single_colors["Q"],   width=LINE_WIDTH), name="Q")
            cM = pw.plot(pen=pg.mkPen(single_colors["Mag"], width=LINE_WIDTH), name="Mag")
            return {ch: {"I": cI, "Q": cQ, "Mag": cM}}
        else:  # "D"
            cD = pw.plot(pen=pg.mkPen(single_colors["DSB"], width=LINE_WIDTH), name="Complex DSB")
            return {ch: {"Cmplx": cD}}

    def _create_multi_channel_curves(self, pw, mode, group, channel_families, legend):
        """Create curves for a multi-channel plot."""
        mode_dict = {}
        for i, ch in enumerate(group):
            (colI, colQ, colM) = channel_families[i % len(channel_families)]
            
            if mode == "T":
                cI = pw.plot(pen=pg.mkPen(colI, width=LINE_WIDTH), name=f"ch{ch}-I")
                cQ = pw.plot(pen=pg.mkPen(colQ, width=LINE_WIDTH), name=f"ch{ch}-Q")
                cM = pw.plot(pen=pg.mkPen(colM, width=LINE_WIDTH), name=f"ch{ch}-Mag")
                mode_dict[ch] = {"I": cI, "Q": cQ, "Mag": cM}
            elif mode == "F":
                cI = pw.plot(pen=pg.mkPen(colI, width=LINE_WIDTH), name=f"ch{ch}-I")
                cI.setFftMode(True)
                cQ = pw.plot(pen=pg.mkPen(colQ, width=LINE_WIDTH), name=f"ch{ch}-Q")
                cQ.setFftMode(True)
                cM = pw.plot(pen=pg.mkPen(colM, width=LINE_WIDTH), name=f"ch{ch}-Mag")
                cM.setFftMode(True)
                mode_dict[ch] = {"I": cI, "Q": cQ, "Mag": cM}
            elif mode == "S":
                cI = pw.plot(pen=pg.mkPen(colI, width=LINE_WIDTH), name=f"ch{ch}-I")
                cQ = pw.plot(pen=pg.mkPen(colQ, width=LINE_WIDTH), name=f"ch{ch}-Q")
                cM = pw.plot(pen=pg.mkPen(colM, width=LINE_WIDTH), name=f"ch{ch}-Mag")
                mode_dict[ch] = {"I": cI, "Q": cQ, "Mag": cM}
            else:  # "D"
                cD = pw.plot(pen=pg.mkPen(colI, width=LINE_WIDTH), name=f"ch{ch}-DSB")
                mode_dict[ch] = {"Cmplx": cD}
                
        return mode_dict

    def _restore_auto_range_settings(self):
        """Restore auto-range settings after building layout."""
        self.auto_scale_plots = True
        self.cb_auto_scale.setChecked(True)
        for rowPlots in self.plots:
            for mode, pw in rowPlots.items():
                if mode != "T":
                    pw.enableAutoRange(pg.ViewBox.XYAxes, True)

    def _apply_plot_theme(self, pw: pg.PlotWidget):
        """
        Configure the plot widget's background and axis color
        based on self.dark_mode.

        Parameters
        ----------
        pw : pg.PlotWidget
            The plot widget to style.
        """
        if self.dark_mode:
            pw.setBackground("k")
            for axis_name in ("left", "bottom", "right", "top"):
                ax = pw.getPlotItem().getAxis(axis_name)
                if ax:
                    ax.setPen("w")
                    ax.setTextPen("w")
        else:
            pw.setBackground("w")
            for axis_name in ("left", "bottom", "right", "top"):
                ax = pw.getPlotItem().getAxis(axis_name)
                if ax:
                    ax.setPen("k")
                    ax.setTextPen("k")

    @staticmethod
    def _fade_hidden_entries(legend, hide_labels):
        """
        Fade out (gray) specific legend entries to indicate
        they are typically less interesting (like I and Q).
        """
        for sample, label in legend.items:
            txt = label.labelItem.toPlainText() if hasattr(label, "labelItem") else ""
            if txt in hide_labels:
                sample.setOpacity(0.3)
                label.setOpacity(0.3)

    @staticmethod
    def _make_legend_clickable(legend):
        """
        Make each legend entry clickable to toggle the associated curve's visibility.

        Parameters
        ----------
        legend : pg.LegendItem
            The legend container with (sample, label) items.
        """
        for sample, label in legend.items:
            curve = sample.item

            def toggle(evt, c=curve, s=sample, l=label):
                vis = not c.isVisible()
                c.setVisible(vis)
                op = 1.0 if vis else 0.3
                s.setOpacity(op)
                l.setOpacity(op)

            label.mousePressEvent = toggle
            sample.mousePressEvent = toggle

    def _toggle_dark_mode(self, checked: bool):
        """
        Switch the entire UI between dark or light color schemes
        and rebuild the plot layout.

        Parameters
        ----------
        checked : bool
            True if dark mode, False for light mode.
        """
        self.dark_mode = checked
        self._build_layout()

    def _toggle_real_units(self, checked: bool):
        """
        Toggle between raw counts and real-units (volts/dBm).

        Parameters
        ----------
        checked : bool
            True if real units, False if raw counts.
        """
        self.real_units = checked
        if checked:
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Real Units On")
            msg.setText(
                "Global conversion to real units (V, dBm) is approximate.\n"
                "All PSD plots are droop-corrected for the CIC1 and CIC2 decimation filters; the 'Raw FFT' is calculated from the raw TOD and not droop-corrected."
            )
            msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            msg.exec()
        self._build_layout()

    def _psd_ref_changed(self):
        """
        Switch between absolute (dBm) and relative (dBc) scaling for PSD plots
        when Real Units is enabled.
        """
        self.psd_absolute = self.rb_psd_abs.isChecked()
        self._build_layout()

    def _update_channels(self):
        """
        Parse the channel specification string from self.e_ch, supporting '&'
        to group multiple channels in one row. Re-init buffers/layout if changed.
        """
        new_parsed = parse_channels_multich(self.e_ch.text())
        if new_parsed != self.channel_list:
            self.channel_list = new_parsed
            self.iq_workers.clear()
            self.psd_workers.clear()
            for row_i, group in enumerate(self.channel_list):
                self.psd_workers[row_i] = {"S": {}, "D": {}}
                for c in group:
                    self.psd_workers[row_i]["S"][c] = False
                    self.psd_workers[row_i]["D"][c] = False
            self._init_buffers()
            self._build_layout()

    def _change_buffer(self):
        """
        Update ring buffer size from self.e_buf if it differs from the current size,
        then re-init buffers.
        """
        try:
            n = int(self.e_buf.text())
        except ValueError:
            return
        if n != self.N:
            self.N = n
            self._init_buffers()

    def _toggle_pause(self):
        """
        Pause or resume real-time data updates. When paused, new packets
        are discarded to avoid stale accumulation.
        """
        self.paused = not self.paused
        self.b_pause.setText("Resume" if self.paused else "Pause")

    # ───────────────────────── Main GUI Update ─────────────────────────
    def _update_gui(self):
        """
        Periodic GUI update method (via QTimer):
          - Reads UDP packets into ring buffers
          - Updates decimation stage
          - Spawns background tasks (IQ, PSD)
          - Updates displayed lines/images
          - Logs FPS and PPS to status bar
        """
        if self.paused:
            self._discard_packets()
            return

        # Collect new packets and update buffers
        self._process_incoming_packets()

        # Update frame counter and time tracking
        self.frame_cnt += 1
        now = time.time()

        # Recompute dec_stage once per second
        if (now - self.last_dec_update) > 1.0:
            self._update_dec_stage()
            self.last_dec_update = now

        # Update plot data for each row
        self._update_plot_data()

        # Update FPS / PPS display
        self._update_performance_stats(now)

    def _discard_packets(self):
        """Discard all pending packets while paused."""
        while not self.receiver.queue.empty():
            self.receiver.queue.get()

    def _process_incoming_packets(self):
        """Process incoming data packets and update buffers."""
        while not self.receiver.queue.empty():
            pkt = self.receiver.queue.get()
            self.pkt_cnt += 1

            # Calculate relative timestamp
            t_rel = self._calculate_relative_timestamp(pkt)

            # Update ring buffers for all channels
            self._update_buffers(pkt, t_rel)

    def _calculate_relative_timestamp(self, pkt):
        """Calculate relative timestamp from packet timestamp."""
        ts = pkt.ts
        if ts.recent:
            ts.ss += int(0.02 * streamer.SS_PER_SECOND)
            ts.renormalize()
            t_now = ts.h * 3600 + ts.m * 60 + ts.s + ts.ss / streamer.SS_PER_SECOND
            if self.start_time is None:
                self.start_time = t_now
            t_rel = t_now - self.start_time
        else:
            t_rel = None
        return t_rel

    def _update_buffers(self, pkt, t_rel):
        """Update ring buffers with packet data."""
        for ch in self.all_chs:
            Ival = pkt.s[2 * (ch - 1)] / 256.0
            Qval = pkt.s[2 * (ch - 1) + 1] / 256.0
            self.buf[ch]["I"].add(Ival)
            self.buf[ch]["Q"].add(Qval)
            self.buf[ch]["M"].add(math.hypot(Ival, Qval))
            self.tbuf[ch].add(t_rel)

    def _update_plot_data(self):
        """Update plot data for all rows and channels."""
        for row_i, group in enumerate(self.channel_list):
            rowCurves = self.curves[row_i]
            
            # Update time-domain and FFT data for each channel
            for ch in group:
                self._update_channel_plot_data(ch, rowCurves)
            
            # Dispatch IQ computation tasks
            if "IQ" in rowCurves and not self.iq_workers.get(row_i, False):
                self._dispatch_iq_task(row_i, group, rowCurves)
            
            # Dispatch PSD computation tasks
            self._dispatch_psd_tasks(row_i, group)

    def _update_channel_plot_data(self, ch, rowCurves):
        """Update plot data for a specific channel."""
        # Grab ring buffer data
        rawI = self.buf[ch]["I"].data()
        rawQ = self.buf[ch]["Q"].data()
        rawM = self.buf[ch]["M"].data()
        tarr = self.tbuf[ch].data()

        # Apply unit conversion if enabled
        if self.real_units:
            I = convert_roc_to_volts(rawI)
            Q = convert_roc_to_volts(rawQ)
            M = convert_roc_to_volts(rawM)
        else:
            I, Q, M = rawI, rawQ, rawM

        # Update time-domain plots
        if "T" in rowCurves and ch in rowCurves["T"]:
            cset = rowCurves["T"][ch]
            if cset["I"].isVisible():
                cset["I"].setData(tarr, I)
            if cset["Q"].isVisible():
                cset["Q"].setData(tarr, Q)
            if cset["Mag"].isVisible():
                cset["Mag"].setData(tarr, M)

        # Update FFT plots
        if "F" in rowCurves and ch in rowCurves["F"]:
            cset = rowCurves["F"][ch]
            if cset["I"].isVisible():
                cset["I"].setData(tarr, I, fftMode=True)
            if cset["Q"].isVisible():
                cset["Q"].setData(tarr, Q, fftMode=True)
            if cset["Mag"].isVisible():
                cset["Mag"].setData(tarr, M, fftMode=True)

    def _dispatch_iq_task(self, row_i, group, rowCurves):
        """Dispatch an IQ computation task for a row."""
        mode = rowCurves["IQ"]["mode"]
        
        if len(group) == 1:
            # Single channel case
            c = group[0]
            rawI = self.buf[c]["I"].data()
            rawQ = self.buf[c]["Q"].data()
            self.iq_workers[row_i] = True
            task = IQTask(row_i, c, rawI, rawQ, self.dot_px, mode, self.iq_signals)
            self.pool.start(task)
        else:
            # Multi-channel case - combine data
            concatI = np.concatenate([self.buf[ch]["I"].data() for ch in group])
            concatQ = np.concatenate([self.buf[ch]["Q"].data() for ch in group])
            
            # Limit data size to avoid excessive processing
            big_size = concatI.size
            if big_size > 50000:
                stride = max(1, big_size // 50000)
                concatI = concatI[::stride]
                concatQ = concatQ[::stride]
            
            if concatI.size > 1:
                self.iq_workers[row_i] = True
                task = IQTask(row_i, 0, concatI, concatQ, self.dot_px, mode, self.iq_signals)
                self.pool.start(task)

    def _dispatch_psd_tasks(self, row_i, group):
        """Dispatch PSD computation tasks for a row."""
        # Single-sideband PSD tasks
        if "S" in self.curves[row_i]:
            for ch in group:
                if not self.psd_workers[row_i]["S"][ch]:
                    rawI = self.buf[ch]["I"].data()
                    rawQ = self.buf[ch]["Q"].data()
                    
                    # Apply voltage conversion if real units are enabled
                    if self.real_units:
                        I = convert_roc_to_volts(rawI)
                        Q = convert_roc_to_volts(rawQ)
                    else:
                        I, Q = rawI, rawQ
                    
                    self.psd_workers[row_i]["S"][ch] = True
                    task = PSDTask(
                        row=row_i,
                        ch=ch,
                        I=I,  # Use the converted values
                        Q=Q,  # Use the converted values
                        mode="SSB",
                        dec_stage=self.dec_stage,
                        real_units=self.real_units,
                        psd_absolute=self.psd_absolute,
                        segments=self.spin_segments.value(),
                        signals=self.psd_signals,
                    )
                    self.pool.start(task)

        # Dual-sideband PSD tasks
        if "D" in self.curves[row_i]:
            for ch in group:
                if not self.psd_workers[row_i]["D"][ch]:
                    rawI = self.buf[ch]["I"].data()
                    rawQ = self.buf[ch]["Q"].data()
                    
                    # Apply voltage conversion if real units are enabled
                    if self.real_units:
                        I = convert_roc_to_volts(rawI)
                        Q = convert_roc_to_volts(rawQ)
                    else:
                        I, Q = rawI, rawQ
                    
                    self.psd_workers[row_i]["D"][ch] = True
                    task = PSDTask(
                        row=row_i,
                        ch=ch,
                        I=I,  # Use the converted values
                        Q=Q,  # Use the converted values
                        mode="DSB",
                        dec_stage=self.dec_stage,
                        real_units=self.real_units,
                        psd_absolute=self.psd_absolute,
                        segments=self.spin_segments.value(),
                        signals=self.psd_signals,
                    )
                    self.pool.start(task)

    def _update_performance_stats(self, now):
        """Update FPS and packets-per-second display."""
        if (now - self.t_last) >= 1.0:
            fps = self.frame_cnt / (now - self.t_last)
            pps = self.pkt_cnt / (now - self.t_last)
            self.statusBar().showMessage(f"FPS {fps:.1f} | Packets/s {pps:.1f}")
            self.frame_cnt = 0
            self.pkt_cnt = 0
            self.t_last = now

    def _update_dec_stage(self):
        """
        Update the global decimation stage by measuring the sample rate
        from the first row's first channel.
        """
        if not self.channel_list:
            return
        first_group = self.channel_list[0]
        if not first_group:
            return
        ch = first_group[0]
        tarr = self.tbuf[ch].data()
        if len(tarr) < 2:
            return
        dt = (tarr[-1] - tarr[0]) / max(1, (len(tarr) - 1))
        fs = 1.0 / dt if dt > 0 else 1.0
        self.dec_stage = infer_dec_stage(fs)

    # ───────────────────────── IQ & PSD Slots ─────────────────────────
    @QtCore.pyqtSlot(int, str, object)
    def _iq_done(self, row: int, task_mode: str, payload):
        """
        Slot called when an off-thread IQTask finishes.

        Parameters
        ----------
        row : int
            Row index within self.channel_list.
        task_mode : {"density", "scatter"}
            The IQ plot mode for which data was computed.
        payload : object
            The result data. For density: (hist, (Imin,Imax,Qmin,Qmax)).
            For scatter: (xs, ys, colors).
        """
        self.iq_workers[row] = False
        if row >= len(self.curves) or "IQ" not in self.curves[row]:
            return
        pane = self.curves[row]["IQ"]
        if pane["mode"] != task_mode:
            return

        item = pane["item"]
        if task_mode == "density":
            self._update_density_image(item, payload)
        else:  # scatter
            self._update_scatter_plot(item, payload)

    def _update_density_image(self, item, payload):
        """Update a density image with new data."""
        hist, (Imin, Imax, Qmin, Qmax) = payload
        if self.real_units:
            Imin, Imax = convert_roc_to_volts(np.array([Imin, Imax], dtype=float))
            Qmin, Qmax = convert_roc_to_volts(np.array([Qmin, Qmax], dtype=float))
        item.setImage(hist, levels=(0, 255), autoLevels=False)
        item.setRect(
            QtCore.QRectF(
                float(Imin),
                float(Qmin),
                float(Imax - Imin),
                float(Qmax - Qmin),
            )
        )

    def _update_scatter_plot(self, item, payload):
        """Update a scatter plot with new data."""
        xs, ys, colors = payload
        if self.real_units:
            xs = convert_roc_to_volts(xs)
            ys = convert_roc_to_volts(ys)
        item.setData(xs, ys, brush=colors, pen=None, size=SCATTER_SIZE)

    @QtCore.pyqtSlot(int, str, int, object)
    def _psd_done(self, row: int, psd_mode: str, ch: int, payload):
        """
        Slot called when an off-thread PSDTask finishes for a particular channel.

        Parameters
        ----------
        row : int
            Row index in self.channel_list.
        psd_mode : {"SSB", "DSB"}
            The PSD mode that was computed.
        ch : int
            Which channel (within that row) was computed.
        payload : object
            For "SSB": (freq_i, psd_i, psd_q, psd_m, freq_m, psd_m, dec_stage_float).
            For "DSB": (freq_dsb, psd_dsb).
        """
        if row not in self.psd_workers:
            return
        key = psd_mode[0]  # 'S' or 'D'
        if key not in self.psd_workers[row]:
            return
        if ch not in self.psd_workers[row][key]:
            return

        self.psd_workers[row][key][ch] = False
        if row >= len(self.curves):
            return

        if psd_mode == "SSB":
            self._update_ssb_curves(row, ch, payload)
        else:  # DSB
            self._update_dsb_curve(row, ch, payload)

    def _update_ssb_curves(self, row, ch, payload):
        """Update single-sideband PSD curves."""
        if "S" not in self.curves[row]:
            return
        sdict = self.curves[row]["S"]
        if ch not in sdict:
            return
        
        freq_i_data, psd_i_data, psd_q_data, psd_m_data, _, _, _ = payload
        
        # Ensure data is numpy array of floats
        freq_i = np.asarray(freq_i_data, dtype=float)
        psd_i = np.asarray(psd_i_data, dtype=float)
        psd_q = np.asarray(psd_q_data, dtype=float)
        psd_m = np.asarray(psd_m_data, dtype=float)
        
        if sdict[ch]["I"].isVisible():
            sdict[ch]["I"].setData(freq_i, psd_i)
        if sdict[ch]["Q"].isVisible():
            sdict[ch]["Q"].setData(freq_i, psd_q)
        if sdict[ch]["Mag"].isVisible():
            sdict[ch]["Mag"].setData(freq_i, psd_m)

    def _update_dsb_curve(self, row, ch, payload):
        """Update dual-sideband PSD curve."""
        if "D" not in self.curves[row]:
            return
        ddict = self.curves[row]["D"]
        if ch not in ddict:
            return
        
        freq_dsb_data, psd_dsb_data = payload
        
        # Ensure data is numpy array of floats
        freq_dsb = np.asarray(freq_dsb_data, dtype=float)
        psd_dsb = np.asarray(psd_dsb_data, dtype=float)
        
        if ddict[ch]["Cmplx"].isVisible():
            ddict[ch]["Cmplx"].setData(freq_dsb, psd_dsb)

    def closeEvent(self, event):
        """
        Cleanly shut down the background receiver and stop the timer before closing.

        Parameters
        ----------
        event : QCloseEvent
            The close event instance.
        """
        self.timer.stop()
        self.receiver.stop()
        self.receiver.wait()
        
        # Stop any running network analysis tasks
        for task_key in list(self.netanal_tasks.keys()):
            task = self.netanal_tasks[task_key]
            task.stop()
            self.netanal_tasks.pop(task_key, None)
        
        # Stop any running multisweep tasks
        for task_key in list(self.multisweep_tasks.keys()):
            task = self.multisweep_tasks[task_key]
            task.stop()
            self.multisweep_tasks.pop(task_key, None)

        # Shutdown iPython kernel if it exists
        if self.kernel_manager and self.kernel_manager.has_kernel:
            try:
                self.kernel_manager.shutdown_kernel()
            except Exception as e:
                warnings.warn(f"Error shutting down iPython kernel: {e}", RuntimeWarning)
        
        super().closeEvent(event)
        event.accept()

    def _add_interactive_console_dock(self):
        """Create and add the QDockWidget for the iPython console."""
        if not QTCONSOLE_AVAILABLE:
            return

        self.console_dock_widget = QtWidgets.QDockWidget("Interactive iPython Session", self)
        self.console_dock_widget.setObjectName("InteractiveSessionDock")
        self.console_dock_widget.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.console_dock_widget.setVisible(False) # Initially hidden
        # We don't set a widget here yet; it will be created lazily.
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.console_dock_widget)

    def _toggle_interactive_session(self):
        """Toggle the visibility of the interactive iPython console."""
        if not QTCONSOLE_AVAILABLE or self.crs is None:
            # Should not happen if button is disabled, but good to check
            return

        if self.console_dock_widget is None: # Should have been created by _add_interactive_console_dock
            return

        if self.kernel_manager is None:
            # First time: Initialize kernel and widget
            try:
                self.kernel_manager = QtInProcessKernelManager()
                self.kernel_manager.start_kernel()
                
                kernel = self.kernel_manager.kernel
                # Push relevant variables to the iPython shell
                # Note: rfmux module is already imported at the top of the file
                kernel.shell.push({
                    'crs': self.crs, 
                    'rfmux': rfmux, 
                    'periscope': self  # Provide access to the Periscope app instance
                })

                self.jupyter_widget = RichJupyterWidget()
                self.jupyter_widget.kernel_client = self.kernel_manager.client()
                self.jupyter_widget.kernel_client.start_channels() # Start communication channels

                # Load awaitless extension
                try:
                    load_awaitless_extension(ipython=kernel.shell)
                except Exception as e_awaitless:
                    warnings.warn(f"Could not load awaitless extension: {e_awaitless}", RuntimeWarning)
                    traceback.print_exc()

                self.console_dock_widget.setWidget(self.jupyter_widget)
                self._update_console_style(self.dark_mode) # Set initial theme

                # Be forceful about the color overrides
                if self.dark_mode:
                    # For dark mode
                    self.jupyter_widget._control.document().setDefaultStyleSheet("""
                        .in-prompt { color: #00FF00 !important; }
                        .out-prompt { color: #00DD00 !important; }
                        body { background-color: #1C1C1C; color: #DDDDDD; }
                    """)
                else:
                    # For light mode
                    self.jupyter_widget._control.document().setDefaultStyleSheet("""
                        .in-prompt { color: #008800 !important; }
                        .out-prompt { color: #006600 !important; }
                        body { background-color: #FFFFFF; color: #000000; }
                    """)
                # Give focus to the console when it's first shown
                self.jupyter_widget.setFocus()

            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error Initializing Console",
                                               f"Could not initialize iPython console: {e}")
                traceback.print_exc()
                if self.kernel_manager and self.kernel_manager.has_kernel:
                    self.kernel_manager.shutdown_kernel()
                self.kernel_manager = None
                self.jupyter_widget = None
                self.console_dock_widget.setVisible(False) # Ensure it's hidden on error
                return
        
        # Toggle visibility
        is_visible = self.console_dock_widget.isVisible()
        self.console_dock_widget.setVisible(not is_visible)
        if not is_visible and self.jupyter_widget: # If showing
            self.jupyter_widget.setFocus()

    def _start_multisweep_analysis(self, params: dict):
        """Start multisweep analysis for the given parameters."""
        try:
            if self.crs is None:
                QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available for multisweep.")
                return

            window_id = f"multisweep_window_{self.multisweep_window_count}"
            self.multisweep_window_count += 1

            target_module = params.get('module')
            if target_module is None:
                QtWidgets.QMessageBox.critical(self, "Error", "Target module not specified for multisweep.")
                return

            # Use actual DAC scales if available, otherwise empty dict
            dac_scales_for_window = self.dac_scales if hasattr(self, 'dac_scales') else {}

            window = MultisweepWindow(
                parent=self,
                target_module=target_module,
                initial_params=params.copy(),
                dac_scales=dac_scales_for_window
            )
            # window.window_id = window_id # Store ID if needed for direct access

            # Create window-specific signals instance for multisweep
            # This is important if we want each window to have its own signal handling context,
            # but for now, we can use the global self.multisweep_signals and connect
            # methods from the specific window instance.
            
            # Store window and prepare for task
            self.multisweep_windows[window_id] = {
                'window': window,
                'params': params.copy() 
                # 'signals': specific_multisweep_signals # If we used per-window signals
            }

            # Connect signals from the global multisweep_signals to the new window's methods
            # Disconnect old connections first if any, to avoid multiple calls to old windows
            try:
                self.multisweep_signals.progress.disconnect()
                self.multisweep_signals.intermediate_data_update.disconnect()
                self.multisweep_signals.data_update.disconnect()
                self.multisweep_signals.completed_amplitude.disconnect()
                self.multisweep_signals.all_completed.disconnect()
                self.multisweep_signals.error.disconnect()
            except TypeError: # Thrown if no connections exist
                pass

            self.multisweep_signals.progress.connect(window.update_progress)
            self.multisweep_signals.intermediate_data_update.connect(window.update_intermediate_data)
            self.multisweep_signals.data_update.connect(window.update_data)
            self.multisweep_signals.completed_amplitude.connect(window.completed_amplitude_sweep)
            self.multisweep_signals.all_completed.connect(window.all_sweeps_completed)
            self.multisweep_signals.error.connect(window.handle_error)
            
            task = MultisweepTask(
                crs=self.crs,
                resonance_frequencies=params['resonance_frequencies'],
                params=params, # This includes module, amps, span_hz, npoints_per_sweep, perform_fits
                signals=self.multisweep_signals # Use the global signals object
            )
            
            task_key = f"{window_id}_module_{target_module}" # Simplified task key
            self.multisweep_tasks[task_key] = task
            self.pool.start(task)
            
            window.show()

        except Exception as e:
            error_msg = f"Error starting multisweep analysis: {type(e).__name__}: {str(e)}"
            print(error_msg, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            QtWidgets.QMessageBox.critical(self, "Multisweep Error", error_msg)

    def _start_multisweep_analysis_for_window(self, window_instance: MultisweepWindow, params: dict):
        """Re-run multisweep for an existing MultisweepWindow."""
        window_id = None
        for w_id, data in self.multisweep_windows.items():
            if data['window'] == window_instance:
                window_id = w_id
                break
        
        if not window_id:
            QtWidgets.QMessageBox.critical(window_instance, "Error", "Could not find associated window to re-run multisweep.")
            return

        target_module = params.get('module')
        if target_module is None:
            QtWidgets.QMessageBox.critical(window_instance, "Error", "Target module not specified for multisweep re-run.")
            return

        # Stop existing task for this window/module if any
        old_task_key = f"{window_id}_module_{target_module}"
        if old_task_key in self.multisweep_tasks:
            old_task = self.multisweep_tasks.pop(old_task_key)
            old_task.stop()

        # Update stored params for the window
        self.multisweep_windows[window_id]['params'] = params.copy()

        task = MultisweepTask(
            crs=self.crs,
            resonance_frequencies=params['resonance_frequencies'],
            params=params,
            signals=self.multisweep_signals # Still use global signals, window methods are connected
        )
        self.multisweep_tasks[old_task_key] = task # Reuse task key
        self.pool.start(task)


    def stop_multisweep_task_for_window(self, window_instance: MultisweepWindow):
        """Stop the multisweep task associated with a given window instance."""
        window_id = None
        target_module = None
        for w_id, data in list(self.multisweep_windows.items()): # Iterate over a copy for safe removal
            if data['window'] == window_instance:
                window_id = w_id
                target_module = data['params'].get('module')
                break
        
        if window_id and target_module:
            task_key = f"{window_id}_module_{target_module}"
            if task_key in self.multisweep_tasks:
                task = self.multisweep_tasks.pop(task_key)
                task.stop()
            # Remove window from tracking
            self.multisweep_windows.pop(window_id, None)


    def _update_console_style(self, dark_mode_enabled: bool):
        """Update the iPython console style based on dark mode."""
        if self.jupyter_widget and QTCONSOLE_AVAILABLE:
            if dark_mode_enabled:
                # Set the syntax highlighting style
                self.jupyter_widget.syntax_style = 'monokai'
                
                # Custom stylesheet with green prompts
                self.jupyter_widget.setStyleSheet("""
                    QWidget { background-color: #1C1C1C; color: #DDDDDD; }
                    
                    /* Change input prompt to bright green */
                    .in-prompt { color: #00FF00 !important; }
                    
                    /* Change output prompt to a lighter green */
                    .out-prompt { color: #00DD00 !important; }
                    
                    /* Make sure these styles are applied to the internal console widget */
                    QPlainTextEdit { background-color: #1C1C1C; color: #DDDDDD; }
                """)
            else:
                # Light mode
                self.jupyter_widget.syntax_style = 'default'
                
                # Custom stylesheet with explicit white background for light mode
                self.jupyter_widget.setStyleSheet("""
                    /* Set explicit background and text colors */
                    QWidget { background-color: #FFFFFF; color: #000000; }
                    
                    /* Green prompts for light mode */
                    .in-prompt { color: #008800 !important; }
                    .out-prompt { color: #006600 !important; }
                    
                    /* Ensure the text editor has correct background */
                    QPlainTextEdit { background-color: #FFFFFF; color: #000000; }
                """)
                
            # Force a complete refresh
            self.jupyter_widget.update()
            self.jupyter_widget.repaint()

def main():
    """
    Entry point for command-line usage. Supports multi-channel grouping via '&',
    auto-scaling, and global I/Q/M toggles. Launches Periscope in blocking mode.
    """
    ap = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent("""\
        Periscope — real-time CRS packet visualizer with network analysis.

        Connects to a UDP/multicast stream, filters a single module, and drives a
        PyQt6 GUI with up to six linked views per channel group:

          • Time-domain waveform (TOD)
          • IQ plane (density or scatter)
          • Raw FFT
          • Single-sideband PSD (SSB)  – CIC-corrected
          • Dual-sideband PSD (DSB)    – CIC-corrected
          • Network Analysis — amplitude and phase vs frequency

        Key options
        -----------
          • Comma-separated channel list with '&' to overlay channels on one row,
            e.g. "3&5,7" → two rows: {3,5} and {7}
          • -n / --num-samples   Ring-buffer length per channel (history / FFT depth)
          • -f / --fps           Maximum GUI refresh rate (frames s⁻¹) [typically system-limited anyway]
          • -d / --density-dot   Dot size in IQ-density mode (pixels) [not typically adjusted]
          • --enable-netanal     Create CRS object to enable network analysis

        Advanced features
        -----------------
          • Real-unit conversion (counts → V, dBm/Hz, dBc/Hz) with CIC droop compensation in the PSDs
          • Welch segmentation to average PSD noise floor
          • Network analysis with real-time amplitude and phase measurements

        Example
        -------
          $ periscope rfmux0022.local --module 2 --channels "3&5,7" --enable-netanal

        Run with -h / --help for the full option list.
    """))

    ap.add_argument("hostname")
    ap.add_argument("-m", "--module", type=int, default=1)
    ap.add_argument("-c", "--channels", default="1")
    ap.add_argument("-n", "--num-samples", type=int, default=DEFAULT_BUFFER_SIZE)
    ap.add_argument("-f", "--fps", type=float, default=30.0)
    ap.add_argument("-d", "--density-dot", type=int, default=DENSITY_DOT_SIZE)
    args = ap.parse_args()

    if args.fps <= 0:
        ap.error("FPS must be positive.")
    if args.fps > 30:
        warnings.warn("FPS>30 might be unnecessary", RuntimeWarning)
    if args.density_dot < 1:
        ap.error("Density-dot size must be ≥1 pixel.")

    refresh_ms = int(round(1000.0 / args.fps))
    app = QtWidgets.QApplication(sys.argv[:1])
    app_icon = QIcon(ICON_PATH)
    app.setWindowIcon(app_icon)

    # --- Global Exception Hook ---
    def global_exception_hook(exctype, value, tb):
        # Ensure traceback is imported if not already
        import traceback as tb_module
        print("Unhandled exception in Periscope:", file=sys.stderr)
        tb_module.print_exception(exctype, value, tb, file=sys.stderr)
        # Optionally, re-raise or call sys.__excepthook__ if you want Python's default handling too
        # For a GUI app, often just logging is preferred over crashing.
        # sys.__excepthook__(exctype, value, tb)

    sys.excepthook = global_exception_hook
    # --- End Global Exception Hook ---

    # Bind only the main GUI (this) thread to a single CPU to reduce scheduling jitter
    pin_current_thread_to_core()

    # Create CRS object so we can tell it what to do
    crs = None
    try:
        # Extract serial number from hostname
        hostname = args.hostname
        if "rfmux" in hostname and ".local" in hostname:
            serial = hostname.replace("rfmux", "").replace(".local", "")
        else:
            # Default to hostname as serial
            serial = hostname
        
        # Create and resolve CRS in a synchronous way
        s = load_session(f'!HardwareMap [ !CRS {{ serial: "{serial}" }} ]')
        crs = s.query(CRS).one()
        
        # Resolve the CRS object
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(crs.resolve())
        
    except Exception as e:
        warnings.warn(f"Failed to create CRS object: {str(e)}\nNetwork analysis will be disabled.")
        crs = None

    viewer = Periscope(
        host=args.hostname,
        module=args.module,
        chan_str=args.channels,
        buf_size=args.num_samples,
        refresh_ms=refresh_ms,
        dot_px=args.density_dot,
        crs=crs  # Pass the CRS object
    )
    # Also set icon for main window
    viewer.setWindowIcon(app_icon)
    viewer.show()
    sys.exit(app.exec())

@macro(CRS, register=True)
async def raise_periscope(
    crs: CRS,
    *,
    module: int = 1,
    channels: str = "1",
    buf_size: int = DEFAULT_BUFFER_SIZE,
    fps: float = 30.0,
    density_dot: int = DENSITY_DOT_SIZE,
    blocking: bool | None = None,
):
    """
    Programmatic entry point for embedding or interactive usage of Periscope.

    This function either blocks (runs the Qt event loop) or returns control
    immediately, depending on whether a Qt event loop is already running or the
    'blocking' parameter is explicitly set.

    Parameters
    ----------
    crs : CRS object
        CRS object for hardware communication (needed for network analysis).
    module : int, optional
        Module number to filter data (default is 1).
    channels : str, optional
        A comma-separated list of channel indices, possibly with '&' to group
        multiple channels on one row (default "1").
    buf_size : int, optional
        Ring buffer size for each channel (default 5000).
    fps : float, optional
        Frames per second (default 30.0). Determines GUI update rate.
    density_dot : int, optional
        Dot dilation in pixels for IQ density mode (default 1).
    blocking : bool or None, optional
        If True, runs the Qt event loop until exit. If False, returns control
        immediately. If None, infers from the environment.

    Returns
    -------
    Periscope or (Periscope, QApplication)
        The Periscope instance, or (instance, QApplication) if non-blocking.
    """
    ip = get_ipython()
    qt_loop = is_qt_event_loop_running()

    if ip and not qt_loop:
        ip.run_line_magic("gui", "qt")
        qt_loop = True

    if blocking is None:
        blocking = not qt_loop

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app_icon = QIcon(ICON_PATH)
    app.setWindowIcon(app_icon)

    # --- Global Exception Hook (also for library use) ---
    # It's good practice to set this up if Periscope might be run programmatically
    # where the main() function isn't the entry point.
    # However, be cautious if the host application (e.g., Jupyter) has its own hook.
    if not hasattr(sys, '_periscope_excepthook_installed'): # Avoid multiple installs
        def global_exception_hook_lib(exctype, value, tb):
            import traceback as tb_module
            print("Unhandled exception in Periscope (library mode):", file=sys.stderr)
            tb_module.print_exception(exctype, value, tb, file=sys.stderr)
        
        # Check if we are in an environment that might already have a complex hook (like IPython)
        # A more robust check might be needed for various environments.
        if get_ipython() is None: # Simple check: if not in IPython, assume it's safe to set.
            sys.excepthook = global_exception_hook_lib
            sys._periscope_excepthook_installed = True # Mark as installed
    # --- End Global Exception Hook ---

    # Bind only the main GUI (this) thread to a single CPU to reduce scheduling jitter
    pin_current_thread_to_core()

    refresh_ms = int(round(1000.0 / fps))

    viewer = Periscope(
        host=crs.tuber_hostname,
        module=module,
        chan_str=channels,
        buf_size=buf_size,
        refresh_ms=refresh_ms,
        dot_px=density_dot,
        crs=crs,  # Pass the CRS object for network analysis
    )
    viewer.setWindowIcon(app_icon)
    viewer.show()

    if blocking:
        if is_running_inside_ipython():
            app.exec()
        else:
            sys.exit(app.exec())
        return viewer
    else:
        return viewer, app


if __name__ == "__main__":
    main()
