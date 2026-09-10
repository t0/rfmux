"""Panel class for network analysis results (dockable)."""

# Imports from within the 'periscope' subpackage
from .utils import *
from .layouts import FlowLayout, grouped, labelled
# from .tasks import * # Not directly used by this class, dialogs will import what they need.

# Dialogs are now imported from .dialogs within the same package
from .dialogs import NetworkAnalysisParamsDialog
from .find_resonances_settings_panel import FindResonancesSettingsPanel
from .tasks import FindResonancesSignals, FindResonancesTask
from ...tuning import record_search, store
from .network_analysis_export import NetworkAnalysisExportMixin

class NetworkAnalysisPanel(QtWidgets.QWidget, NetworkAnalysisExportMixin, ScreenshotMixin):
    """
    Dockable panel for displaying network analysis results with real units support.

    This panel can be wrapped in a QDockWidget for tabbed/floating display within
    the main Periscope window. All functionality from the original NetworkAnalysisWindow
    is preserved.
    
    Signals:
        analysis_finished: Emitted once every module's sweep is in, so the
            session can save the measurement.
    """

    analysis_finished = QtCore.pyqtSignal()
    
    def __init__(self, parent=None, modules=None, dac_scales=None, dark_mode=False, is_loaded_data=False):
        super().__init__(parent)
        self.modules = modules or []
        # module -> the trace take_netanal measured
        self.netanal_traces = {}
        # Every module's output under its module identifier, the shape
        # take_netanal returns and store saves. One file per panel.
        self.netanal_container = {}
        self.unit_mode = "dbm"  # Default to dBm instead of counts
        self.normalize_magnitudes = False  # Add this flag to track normalization state
        self.first_setup = True  # Flag to track initial setup
        self.zoom_box_mode = True  # Default to zoom box mode ON
        self.plots = {}  # Initialize plots dictionary early
        self.original_params = {}  # Initial parameters
        self.current_params = {}   # Most recently used parameters
        self.dac_scales = dac_scales or {}  # Store DAC scales
        # module -> its ResonanceSearch: the accepted candidates are what a
        # multisweep will be run on, the rejected ones say what was passed
        # over and why, and double-clicking moves a resonance between them.
        self.resonance_searches = {}
        self.add_subtract_mode = False
        self.module_cable_lengths = {} # For Requirement 2
        self.dark_mode = dark_mode  # Store dark mode setting
        self.is_loaded_data = is_loaded_data  # Track if this is from loaded data

        # The finder's thresholds, set once and kept: one window per panel,
        # non-modal, so a search is a button press and not a form to fill in.
        self.find_resonances_settings = FindResonancesSettingsPanel(self)
        self._find_res_task = None

        # Setup the UI components
        self._setup_ui()
        self._status_timer = QtCore.QTimer(self)
        self._status_timer.setSingleShot(True)
        # The label's own slot, not a lambda over self: Qt drops a connection
        # to a destroyed receiver, where a closure would keep this panel's
        # Python wrapper alive and fire into a deleted widget.
        self._status_timer.timeout.connect(self.status_label.clear)
        # Set initial size only on creation
        self.resize(1000, 800)

    def _setup_ui(self):
        """Set up the user interface for the panel."""
        # Create main layout for the panel (no central widget needed for QWidget)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # No margins for cleaner docking
        
        # Create toolbar
        self._setup_toolbar(layout)
        
        # Create progress bars
        self._setup_progress_bars(layout)
        
        # Create plot area
        self._setup_plot_area(layout)

    def _setup_toolbar(self, layout):
        """Set up the toolbars with controls."""
        # Toolbar 1: Global Controls (Top Row)
        # Use QWidget container instead of QToolBar for compatibility with QWidget base class
        toolbar_global = QtWidgets.QWidget()
        # Both rows wrap as the panel narrows.
        toolbar_global_layout = FlowLayout(toolbar_global)

        # Save button
        save_btn = QtWidgets.QPushButton("💾")
        save_btn.setToolTip("Save this network analysis to the session folder")
        save_btn.clicked.connect(self._save_netanal_action)
        toolbar_global_layout.addWidget(save_btn)

        # Edit Other Parameters button (renamed to Re-run Analysis)
        edit_params_btn = QtWidgets.QPushButton("Re-run analysis")
        edit_params_btn.clicked.connect(self._edit_parameters)
        toolbar_global_layout.addWidget(edit_params_btn)

        # Show/Hide resonances checkbox
        self.show_resonances_cb = QtWidgets.QCheckBox("Show Resonances")
        self.show_resonances_cb.setChecked(True)
        self.show_resonances_cb.toggled.connect(self._toggle_resonances_visible)
        toolbar_global_layout.addWidget(self.show_resonances_cb)

        # Add/Subtract mode
        self.edit_resonances_cb = QtWidgets.QCheckBox("Add/Subtract Resonances")
        self.edit_resonances_cb.setToolTip(
            "When enabled, double-click adds a resonance;\n"
            "Shift + double-click removes the nearest resonance."
        )
        self.edit_resonances_cb.toggled.connect(self._toggle_resonance_edit_mode)
        toolbar_global_layout.addWidget(self.edit_resonances_cb)

        # Normalize Magnitudes checkbox
        self.normalize_checkbox = QtWidgets.QCheckBox("Normalize Magnitudes")
        self.normalize_checkbox.setChecked(False)
        self.normalize_checkbox.setToolTip("Normalize all magnitude curves to their first data point")
        self.normalize_checkbox.toggled.connect(self._toggle_normalization)
        toolbar_global_layout.addWidget(self.normalize_checkbox)


        # Add unit controls
        self._setup_unit_controls(toolbar_global_layout)

        # Add zoom box mode checkbox
        self._setup_zoom_box_control(toolbar_global_layout)

        # Screenshot button
        screenshot_btn = QtWidgets.QPushButton("📷")
        screenshot_btn.setToolTip("Export a screenshot of this panel to the session folder (or choose location)")
        screenshot_btn.clicked.connect(self._export_screenshot)
        toolbar_global_layout.addWidget(screenshot_btn)
        
        layout.addWidget(toolbar_global)

        # Toolbar 2: Module-Specific Controls (Bottom Row)
        toolbar_module = QtWidgets.QWidget()
        toolbar_module_layout = FlowLayout(toolbar_module)

        # Cable length control
        self.cable_length_spin = QtWidgets.QDoubleSpinBox()
        self.cable_length_spin.setRange(0.0, 1000.0)
        self.cable_length_spin.setValue(DEFAULT_CABLE_LENGTH)
        self.cable_length_spin.setSingleStep(0.05)
        self.cable_length_spin.valueChanged.connect(self._on_cable_length_changed)
        toolbar_module_layout.addWidget(
            labelled("Cable Length (m):", self.cable_length_spin))

        # Unwrap Cable Delay button
        unwrap_button = QtWidgets.QPushButton("Unwrap Cable Delay")
        unwrap_button.setToolTip("Fit phase slope, calculate cable length, and apply compensation for the active module.")
        unwrap_button.clicked.connect(self._unwrap_cable_delay_action)
        toolbar_module_layout.addWidget(unwrap_button)

        # Find Resonances, and the settings it runs with
        self.find_res_btn = QtWidgets.QPushButton("Find Resonances")
        self.find_res_btn.setToolTip("Search the active module's trace for resonance dips.")
        self.find_res_btn.clicked.connect(self._find_resonances_action)
        find_res_settings_btn = QtWidgets.QPushButton("⚙")
        find_res_settings_btn.setToolTip(
            "Thresholds for Find Resonances. They stay set between searches "
            "and across sessions.")
        find_res_settings_btn.clicked.connect(self._show_find_resonances_settings)
        toolbar_module_layout.addWidget(
            grouped(self.find_res_btn, find_res_settings_btn))

        # Take Multisweep button
        self.take_multisweep_btn = QtWidgets.QPushButton("Take Multisweep")
        self.take_multisweep_btn.setToolTip("Perform a multisweep using identified resonance frequencies for the active module.")
        self.take_multisweep_btn.clicked.connect(self._show_multisweep_dialog)
        self.take_multisweep_btn.setEnabled(False) # Initially disabled
        toolbar_module_layout.addWidget(self.take_multisweep_btn)

        # Where routine outcomes go. A dialog is for a failure needing action.
        self.status_label = QtWidgets.QLabel("")
        toolbar_module_layout.addWidget(self.status_label)

        layout.addWidget(toolbar_module)

    def _setup_unit_controls(self, toolbar_layout):
        """Set up the unit selection controls and add them to the specified toolbar layout."""
        unit_group = QtWidgets.QWidget()
        unit_layout = QtWidgets.QHBoxLayout(unit_group)
        unit_layout.setContentsMargins(0, 0, 0, 0)
        unit_layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.rb_counts = QtWidgets.QRadioButton("Counts")
        self.rb_dbm = QtWidgets.QRadioButton("dBm")
        self.rb_volts = QtWidgets.QRadioButton("Volts")
        self.rb_dbm.setChecked(True)
        
        unit_layout.addWidget(QtWidgets.QLabel("Units:"))
        unit_layout.addWidget(self.rb_counts)
        unit_layout.addWidget(self.rb_dbm)
        unit_layout.addWidget(self.rb_volts)
        
        # Connect signals
        self.rb_counts.toggled.connect(lambda: self._update_unit_mode("counts"))
        self.rb_dbm.toggled.connect(lambda: self._update_unit_mode("dbm"))
        self.rb_volts.toggled.connect(lambda: self._update_unit_mode("volts"))
        
        # Set fixed size policy to make alignment more predictable
        unit_group.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, 
                                QtWidgets.QSizePolicy.Policy.Preferred)
        
        # Add the unit controls to the layout
        toolbar_layout.addWidget(unit_group)
        
    def _setup_zoom_box_control(self, toolbar_layout):
        """Set up the zoom box mode control."""
        zoom_box_cb = QtWidgets.QCheckBox("Zoom Box Mode")
        zoom_box_cb.setChecked(self.zoom_box_mode)
        zoom_box_cb.setToolTip("When enabled, left-click drag creates a zoom box. When disabled, left-click drag pans.")
        zoom_box_cb.toggled.connect(self._toggle_zoom_box)
        
        # Store reference to the checkbox
        self.zoom_box_cb = zoom_box_cb
        
        toolbar_layout.addWidget(zoom_box_cb)

    def _setup_progress_bars(self, layout):
        """Set up progress bars for each module."""
        self.progress_group = None
        if self.modules:
            self.progress_group = QtWidgets.QGroupBox("Analysis Progress")
            progress_layout = QtWidgets.QVBoxLayout(self.progress_group)
            
            self.progress_bars = {}
            for module in self.modules:
                hlayout = QtWidgets.QHBoxLayout()
                label = QtWidgets.QLabel(f"Module {module}:")
                pbar = QtWidgets.QProgressBar()
                pbar.setRange(0, 100)
                pbar.setValue(0)
                hlayout.addWidget(label)
                hlayout.addWidget(pbar)

                progress_layout.addLayout(hlayout)
                self.progress_bars[module] = pbar

            layout.addWidget(self.progress_group)
        else:
            self.progress_bars = {}

    def _hide_progress_bars(self):
        """Hide the entire Analysis Progress group."""
        if self.progress_group:
            self.progress_group.hide()

    def _show_progress_bars(self, reset=False):
        """Show the Analysis Progress group again.
           If reset=True, reset the progress bars to zero.
        """
        if self.progress_group:
            self.progress_group.show()
            if reset:
                for module, pbar in self.progress_bars.items():
                    pbar.setValue(0)  # reset progress
    
    def _setup_plot_area(self, layout):
        """Set up the plot area with tabs for each module."""
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.currentChanged.connect(self._on_active_module_changed) # For Requirement 2
        layout.addWidget(self.tabs)
        
        self.plots = {}
        # Initialize amp_plot and phase_plot to None or a default PlotWidget
        # to ensure they are bound before setXLink is called.
        last_amp_plot: Optional[pg.PlotWidget] = None
        last_phase_plot: Optional[pg.PlotWidget] = None

        for module in self.modules:
            tab = QtWidgets.QWidget()
            tab_layout = QtWidgets.QVBoxLayout(tab)

            # Create amplitude and phase plots with ClickableViewBox
            vb_amp = ClickableViewBox()
            vb_amp.parent_window = self
            vb_amp.module_id = module
            vb_amp.plot_role = 'amp'
            amp_plot = pg.PlotWidget(viewBox=vb_amp, title=f"Module {module} - Magnitude")
            plot_item_amp = amp_plot.getPlotItem()
            if plot_item_amp:
                self._update_amplitude_labels(amp_plot) # amp_plot is PlotWidget, _update_amplitude_labels expects PlotWidget
                plot_item_amp.setLabel('bottom', 'Frequency', units='Hz')
                plot_item_amp.showGrid(x=True, y=True, alpha=0.3)

            vb_phase = ClickableViewBox()
            vb_phase.parent_window = self
            vb_phase.module_id = module
            vb_phase.plot_role = 'phase'
            phase_plot = pg.PlotWidget(viewBox=vb_phase, title=f"Module {module} - Phase")
            plot_item_phase = phase_plot.getPlotItem()
            if plot_item_phase:
                plot_item_phase.setLabel('left', 'Phase', units='deg')
                plot_item_phase.setLabel('bottom', 'Frequency', units='Hz')
                plot_item_phase.showGrid(x=True, y=True, alpha=0.3)
            
            # Legends carrying the probe power, in the panel's text colour
            bg_color, pen_color = ("k", "w") if self.dark_mode else ("w", "k")
            amp_legend = plot_item_amp.addLegend(offset=(30, 10), labelTextColor=pen_color) if plot_item_amp else None
            phase_legend = plot_item_phase.addLegend(offset=(30, 10), labelTextColor=pen_color) if plot_item_phase else None

            # Create curves with periscope color scheme - but don't add data yet
            amp_curve = plot_item_amp.plot([], [], pen=pg.mkPen(TABLEAU10_COLORS[1], width=LINE_WIDTH)) if plot_item_amp else None
            phase_curve = plot_item_phase.plot([], [], pen=pg.mkPen(TABLEAU10_COLORS[0], width=LINE_WIDTH)) if plot_item_phase else None

            # The dips a search threw out, hoverable for the reason why.
            rejected_markers = pg.ScatterPlotItem(
                symbol='x', size=9, pen=pg.mkPen(RESONANCE_LINE_COLOR),
                brush=None, hoverable=True,
                tip=lambda x, y, data: str(data))
            if plot_item_amp:
                plot_item_amp.addItem(rejected_markers)

            tab_layout.addWidget(amp_plot)
            tab_layout.addWidget(phase_plot)
            self.tabs.addTab(tab, f"Module {module}")
            
            self.plots[module] = {
                'amp_plot': amp_plot, # amp_plot is PlotWidget here
                'phase_plot': phase_plot, # phase_plot is PlotWidget here
                'amp_curve': amp_curve,
                'phase_curve': phase_curve,
                'amp_legend': amp_legend,
                'phase_legend': phase_legend,
                'resonance_lines_mag': [], # For storing magnitude resonance lines
                'resonance_lines_phase': [], # For storing phase resonance lines
                'rejected_markers': rejected_markers,
            }
            last_amp_plot = amp_plot
            last_phase_plot = phase_plot
            
        # Apply zoom box mode
        self._apply_zoom_box_mode()

        # Link the x-axis of the last created amplitude and phase plots for synchronized zooming
        if last_phase_plot and last_amp_plot:
            last_phase_plot.setXLink(last_amp_plot)
        
        # Apply initial theme based on dark_mode setting
        if last_amp_plot: self._apply_theme_to_plot(last_amp_plot)
        if last_phase_plot: self._apply_theme_to_plot(last_phase_plot)
        
    def _apply_theme_to_plot(self, plot_widget):
        """Apply the current theme to a specific plot widget."""
        bg_color, pen_color = ("k", "w") if self.dark_mode else ("w", "k")
        plot_widget.setBackground(bg_color)
        
        # Update plot title color using a more direct approach
        plot_item = plot_widget.getPlotItem()
        if plot_item:
            # Set the title explicitly with the color parameter
            title_text = plot_item.titleLabel.text if plot_item.titleLabel else ""
            plot_item.setTitle(title_text, color=pen_color)
            
        # Update axes colors
        for axis_name in ("left", "bottom", "right", "top"):
            ax = plot_widget.getPlotItem().getAxis(axis_name)
            if ax:
                ax.setPen(pen_color)
                ax.setTextPen(pen_color)

    def clear_plots(self):
        """Clear all plots, curves, and legends."""
        for module_id_iter in self.plots: 
            plot_info = self.plots[module_id_iter]

            plot_info['amp_legend'].clear()
            plot_info['phase_legend'].clear()

            plot_info['amp_curve'].setData([], [])
            plot_info['phase_curve'].setData([], [])
            plot_info['rejected_markers'].setData([], [])

            self._update_multisweep_button_state(module_id_iter) 


    def _toggle_normalization(self, checked):
        """Toggle normalization of magnitude plots."""
        self.normalize_magnitudes = checked
        
        for module_id in self.plots:
            self._redraw_magnitudes(module_id)


    def _toggle_zoom_box(self, enable):
        """Toggle zoom box mode for all plots."""
        self.zoom_box_mode = enable
        self._apply_zoom_box_mode()
        
    def _apply_zoom_box_mode(self):
        """Apply the current zoom box mode setting to all plots."""
        for module in self.plots:
            for plot_type in ['amp_plot', 'phase_plot']:
                viewbox = self.plots[module][plot_type].getViewBox()
                if isinstance(viewbox, ClickableViewBox):
                    viewbox.enableZoomBoxMode(self.zoom_box_mode)

    def _toggle_resonances_visible(self, checked: bool):
        """Show or hide every resonance marker, kept and rejected alike."""
        for plot_info in self.plots.values():
            for line in plot_info['resonance_lines_mag']:
                line.setVisible(checked)
            for line in plot_info['resonance_lines_phase']:
                line.setVisible(checked)
            plot_info['rejected_markers'].setVisible(checked)

    def _toggle_resonance_edit_mode(self, checked: bool):
        """Enable or disable double-click add/subtract mode."""
        self.add_subtract_mode = checked

    def _clear_resonance_lines(self, module: int) -> None:
        """Take every kept-resonance line off one module's plots."""
        plot_info = self.plots[module]
        for lines, plot in (('resonance_lines_mag', 'amp_plot'),
                            ('resonance_lines_phase', 'phase_plot')):
            item = plot_info[plot].getPlotItem()
            for line in plot_info[lines]:
                item.removeItem(line)
            plot_info[lines] = []

    def _add_resonance_line(self, module: int, freq_hz: float,
                            tooltip: str = "") -> None:
        """One kept resonance, marked on both of a module's plots."""
        plot_info = self.plots[module]
        pen = pg.mkPen(RESONANCE_LINE_COLOR, style=QtCore.Qt.PenStyle.DashLine)
        for lines, plot in (('resonance_lines_mag', 'amp_plot'),
                            ('resonance_lines_phase', 'phase_plot')):
            line = pg.InfiniteLine(pos=freq_hz, angle=90, movable=False, pen=pen)
            if tooltip:
                line.setToolTip(tooltip)
            plot_info[plot].addItem(line)
            plot_info[lines].append(line)

    def _add_resonance(self, module: int, freq_hz: float):
        """Accept a resonance by hand: a dip the finder rejected, or a new one."""
        self._edit_search(module, lambda search: search.accept(freq_hz))

    def _remove_resonance(self, module: int, freq_hz: float):
        """Reject the nearest resonance by hand. It stays in the search."""
        self._edit_search(module, lambda search: search.reject(freq_hz))

    def _edit_search(self, module: int, edit) -> None:
        """Run one by-hand edit on a module's search, then redraw and re-save.

        The search is the record, so an edit changes the netanal block and the
        file that holds it is out of date by exactly that much -- the same move
        Find Resonances makes, and only for a netanal that has a file.
        """
        search = self.resonance_searches.get(module)
        block = self._module_block(module)
        if module not in self.plots or search is None or block is None:
            return
        try:
            edit(search)
        except ValueError as e:
            self._show_status(str(e), ok=False)
            return

        # save=False: the panel writes the whole container below, and only when
        # there is already a file to overwrite.
        record_search(block, search, save=False)
        self.draw_search(module, search)
        message = f"Module {module}: {len(search.candidates)} resonances"
        if store.saved_path(self.netanal_container):
            try:
                message += f" -- saved to {self.save_netanal().name}"
            except Exception as e:                      # noqa: BLE001 - reported
                traceback.print_exc()
                self._show_status(f"{message}, but the save failed: {e}", ok=False)
                return
        self._show_status(message)

    def _update_unit_mode(self, mode):
        """Update unit mode and redraw only amplitude plots."""
        if mode != self.unit_mode:
            self.unit_mode = mode
            
            for module_id in self.plots:
                self._redraw_magnitudes(module_id)

            self._update_legends_for_unit_mode()

    def _update_legends_for_unit_mode(self):
        """Label each module's trace with the power it was probed at.

        A partial sweep carries no ``sweep_amplitude`` -- take_netanal writes it
        with the finished trace -- so the label arrives when the sweep does.
        """
        for module, plot_info in self.plots.items():
            plot_info['amp_legend'].clear()
            plot_info['phase_legend'].clear()

            amplitude = self.netanal_traces.get(module, {}).get('sweep_amplitude')
            if amplitude is None:
                continue

            # Check DAC scale availability for physical-unit modes
            if self.unit_mode in ("dbm", "volts") and module not in self.dac_scales:
                unit_name = "dBm" if self.unit_mode == "dbm" else "Volts"
                print(f"Warning: No DAC scale available for module {module}, cannot display accurate probe power in {unit_name}.")
                self.rb_counts.setChecked(True)
                return

            label = UnitConverter.format_probe_label(
                amplitude, self.unit_mode, self.dac_scales.get(module))
            plot_info['amp_legend'].addItem(plot_info['amp_curve'], label)
            plot_info['phase_legend'].addItem(plot_info['phase_curve'], label)
    
    def _update_amplitude_labels(self, plot):
        """Update plot labels based on current unit mode and normalization state."""
        if self.normalize_magnitudes:
            if self.unit_mode == "dbm":
                plot.setLabel('left', 'Normalized Power', units='dB') 
            else:
                plot.setLabel('left', 'Normalized Magnitude', units='')
        else:
            if self.unit_mode == "counts":
                plot.setLabel('left', 'Magnitude', units='Counts')
            elif self.unit_mode == "dbm":
                plot.setLabel('left', 'Power', units='dBm')
            elif self.unit_mode == "volts":
                plot.setLabel('left', 'Magnitude', units='V')

    def _redraw_all_plots(self):
        """Redraw all plots with current unit mode."""
        for module_id in self.plots:
            self._redraw_magnitudes(module_id)
        self._update_legends_for_unit_mode()

    def _show_find_resonances_settings(self):
        """Raise the finder's settings window; it outlives any one search."""
        self.find_resonances_settings.show()
        self.find_resonances_settings.raise_()
        self.find_resonances_settings.activateWindow()

    def _find_resonances_action(self):
        """Search the active module's trace, off the GUI thread."""
        module = self._active_module()
        if module is None:
            self._show_status("Select a module tab to search.", ok=False)
            return
        block = self._module_block(module)
        if block is None:
            self._show_status(f"Module {module} has not been swept yet.", ok=False)
            return

        self.find_res_btn.setEnabled(False)
        self._show_status(f"Searching module {module}...")
        signals = FindResonancesSignals()
        signals.completed.connect(self._on_search_completed)
        signals.error.connect(self._on_search_error)
        # Held so the thread is not collected while it runs.
        self._find_res_task = FindResonancesTask(
            module, block, self.find_resonances_settings.get_parameters(), signals)
        self._find_res_task.start()

    def _on_search_completed(self, module: int, search):
        self.find_res_btn.setEnabled(True)
        self.draw_search(module, search)

        found, rejected = len(search.candidates), len(search.rejected)
        message = f"Module {module}: {found} resonances"
        if rejected:
            message += f", {rejected} rejected"
        # The search went into the netanal block, so a file that exists is now
        # out of date by exactly this. Re-saving overwrites it; a panel that
        # has never been saved is left to the Save button.
        if store.saved_path(self.netanal_container):
            try:
                message += f" -- saved to {self.save_netanal().name}"
            except Exception as e:                      # noqa: BLE001 - reported
                traceback.print_exc()
                self._show_status(f"{message}, but the save failed: {e}", ok=False)
                return
        self._show_status(message, ok=bool(found))

    def _on_search_error(self, module: int, message: str):
        self.find_res_btn.setEnabled(True)
        self._show_status(f"Module {module}: {message}", ok=False)

    def draw_search(self, module: int, search) -> None:
        """Show what one search found: dips kept, and dips thrown out.

        Also the entry point for a netanal loaded from a file, whose search
        came back out of the trace it was saved in.
        """
        if module not in self.plots:
            return
        self.resonance_searches[module] = search

        self._clear_resonance_lines(module)
        for candidate in search.candidates:
            self._add_resonance_line(
                module, candidate.frequency_hz,
                tooltip=(f"{candidate.frequency_hz / 1e6:.6f} MHz\n"
                         f"{candidate.depth_db:.2f} dB deep\n"
                         f"Q ~ {candidate.q_estimate:.0f}"))
        self._place_rejected(module)
        self._update_resonance_title(module)
        self._toggle_resonances_visible(self.show_resonances_cb.isChecked())
        self._update_multisweep_button_state(module)

    def _place_rejected(self, module: int) -> None:
        """Mark the dips the search threw out, on the magnitude curve.

        Crosses rather than the full-height lines the kept ones get: a search
        can reject many more candidates than it keeps, and the reason each one
        went is in its tooltip.

        The y positions are read off the trace in the panel's current units, so
        a units change re-places them.
        """
        plot_info = self.plots[module]
        markers = plot_info['rejected_markers']
        search = self.resonance_searches.get(module)
        trace = self.netanal_traces.get(module)
        if search is None or trace is None or not search.rejected:
            markers.setData([], [])
            return

        # Ascending, so np.interp works on a downward netanal too.
        order = np.argsort(trace['frequencies'])
        freqs = np.asarray(trace['frequencies'])[order]
        magnitudes = self._magnitude(np.asarray(trace['iq_counts']))[order]

        x = np.array([c.frequency_hz for c in search.rejected])
        markers.setData(
            x, np.interp(x, freqs, magnitudes),
            data=[c.rejected_because for c in search.rejected])

    def _update_resonance_title(self, module: int) -> None:
        """Put the count of kept resonances in the magnitude plot's title."""
        if module not in self.plots:
            return
        search = self.resonance_searches.get(module)
        count = len(search.candidates) if search else 0
        title = f"Module {module} - Magnitude"
        if count:
            title += f" - {count} resonances"
        _, pen_color = ("k", "w") if self.dark_mode else ("w", "k")
        self.plots[module]['amp_plot'].getPlotItem().setTitle(title, color=pen_color)

    def _active_module(self) -> Optional[int]:
        """The module whose tab is showing, or None if none is."""
        index = self.tabs.currentIndex()
        if index < 0:
            return None
        try:
            return int(self.tabs.tabText(index).split(" ")[1])
        except (IndexError, ValueError):
            return None

    def _module_block(self, module: int) -> Optional[dict]:
        """One module's output out of the container, as the finder wants it."""
        for block in self.netanal_container.values():
            if isinstance(block, dict) and block.get('module') == module:
                return block
        return None

    def _show_status(self, message: str, *, ok: bool = True) -> None:
        """Say what happened in the toolbar, and stop saying it after a while."""
        self.status_label.setText(message)
        colour = TABLEAU10_COLORS[2] if ok else TABLEAU10_COLORS[3]
        self.status_label.setStyleSheet(f"color: {colour};")
        self._status_timer.start(STATUS_MESSAGE_MS)

    def _get_periscope_parent(self):
        """
        Get the Periscope parent instance by walking up the parent hierarchy.
        
        Returns:
            The Periscope instance or None if not found
        """
        return find_parent_with_attr(self, 'netanal_windows')
    
    def _check_all_complete(self):
        """
        Check if all progress bars are at 100% and hide the progress group 
        when all analyses are complete.
        """
        if not self.progress_group:
            return

        if all(pbar.value() == 100 for pbar in self.progress_bars.values()):
            self.progress_group.setVisible(False)

    def _edit_parameters(self):
        """Open dialog to edit parameters. Re-runs analysis using per-module cable lengths."""
        params_for_dialog = self.current_params.copy()
        params_for_dialog.pop('module_cable_lengths', None)
        params_for_dialog.pop('cable_length', None) 

        dialog = NetworkAnalysisParamsDialog(self, params_for_dialog)
        if dialog.exec():
            updated_general_params = dialog.get_parameters() 
            
            if updated_general_params:
                # Reset the loaded data flag since we're now generating fresh data
                self.is_loaded_data = False
                
                # Update the dock title to remove "(Loaded)" suffix
                periscope = self._get_periscope_parent()
                if periscope:
                    my_dock = periscope.dock_manager.find_dock_for_widget(self)
                    if my_dock:
                        # Get current title and remove " (Loaded)" if present
                        current_title = my_dock.windowTitle()
                        new_title = current_title.replace(" (Loaded)", "")
                        my_dock.setWindowTitle(new_title)
                
                params_for_rerun = updated_general_params.copy()
                params_for_rerun['module_cable_lengths'] = self.module_cable_lengths.copy()
                params_for_rerun.pop('cable_length', None)
                self.current_params = params_for_rerun.copy()
                
                if self.progress_group:
                    self.progress_group.setVisible(True)
                
                parent_widget = self._get_periscope_parent()
                if parent_widget and hasattr(parent_widget, '_rerun_network_analysis'):
                    parent_widget._rerun_network_analysis(self.current_params) # type: ignore

    def _rerun_analysis(self):
        """Re-run the analysis with potentially updated parameters."""
        parent_widget = self._get_periscope_parent()
        if parent_widget and hasattr(parent_widget, '_rerun_network_analysis'):
            params = self.current_params.copy()
            params['module_cable_lengths'] = self.module_cable_lengths.copy()
            params.pop('cable_length', None)

            if self.progress_group:
                self.progress_group.setVisible(True)
            parent_widget._rerun_network_analysis(params) # type: ignore
    
    def set_params(self, params):
        """Set parameters for analysis."""
        self.original_params = params.copy()  
        self.current_params = params.copy()   

        default_cable_length_for_all = params.get('cable_length', DEFAULT_CABLE_LENGTH)
        for mod_id in self.modules: 
            self.module_cable_lengths[mod_id] = params.get('module_cable_lengths', {}).get(mod_id, default_cable_length_for_all)

        if self.tabs.count() > 0:
            self._on_active_module_changed(self.tabs.currentIndex())
        elif self.modules: 
            first_module_id = self.modules[0]
            initial_cable_length = self.module_cable_lengths.get(first_module_id, DEFAULT_CABLE_LENGTH)
            self.cable_length_spin.blockSignals(True)
            self.cable_length_spin.setValue(initial_cable_length)
            self.cable_length_spin.blockSignals(False)
        
        if not hasattr(self, 'plots') or not self.plots:
            return
            
        fmin = params.get('fmin', DEFAULT_MIN_FREQ)
        fmax = params.get('fmax', DEFAULT_MAX_FREQ)
        for module in self.plots:
            self.plots[module]['amp_plot'].setXRange(fmin, fmax)
            self.plots[module]['phase_plot'].setXRange(fmin, fmax)
            self.plots[module]['amp_plot'].enableAutoRange(pg.ViewBox.XAxis, False)
            self.plots[module]['phase_plot'].enableAutoRange(pg.ViewBox.XAxis, False)
            self.plots[module]['amp_plot'].enableAutoRange(pg.ViewBox.YAxis, True)
            self.plots[module]['phase_plot'].enableAutoRange(pg.ViewBox.YAxis, True)
    
    def update_data(self, module: int, trace: dict):
        """Show a module's netanal, partial or finished."""
        self.netanal_traces[module] = trace

        if module in self.plots:
            freqs, iq = trace['frequencies'], trace['iq_counts']
            plot_info = self.plots[module]
            plot_info['amp_curve'].setData(freqs, self._magnitude(iq))
            plot_info['phase_curve'].setData(freqs, np.degrees(np.angle(iq)))
            if trace.get('sweep_amplitude') is not None:
                self._update_legends_for_unit_mode()
        self._update_multisweep_button_state(module)

    def _magnitude(self, iq: np.ndarray) -> np.ndarray:
        """|S21| of a measured sweep, in the units the panel is showing."""
        magnitude = np.abs(iq)
        return UnitConverter.convert_amplitude(
            magnitude, iq, self.unit_mode, normalize=self.normalize_magnitudes)

    def _redraw_magnitudes(self, module_id: int):
        """Redraw one module's magnitude curve, after a units change."""
        plot_info = self.plots[module_id]
        self._update_amplitude_labels(plot_info['amp_plot'])
        trace = self.netanal_traces.get(module_id)
        if trace is not None:
            plot_info['amp_curve'].setData(
                trace['frequencies'], self._magnitude(trace['iq_counts']))
            self._place_rejected(module_id)
        plot_info['amp_plot'].autoRange()

    def update_progress(self, module: int, progress: float):
        """Update the progress bar for a specific module."""
        if module in self.progress_bars:
            self.progress_bars[module].setValue(int(progress))
    
    def complete_analysis(self, module: int, container: dict):
        """Take a module's finished measurement, and say when they are all in."""
        # A union keyed by module identifier: each module's task returns a
        # container of its own, and a re-measured module replaces its block.
        self.netanal_container.update(container)
        if module in self.progress_bars:
            self.progress_bars[module].setValue(100)
            self._check_all_complete()

        if self._all_modules_complete():
            self.analysis_finished.emit()
    
    def _all_modules_complete(self) -> bool:
        """Check if all modules have completed analysis."""
        if not self.progress_bars:
            return False
        return all(pbar.value() == 100 for pbar in self.progress_bars.values())
            
    def apply_theme(self, dark_mode: bool):
        """Apply the dark/light theme to all plots in this window."""
        self.dark_mode = dark_mode
        
        # Apply theme to all plots
        for module in self.plots:
            plot_info = self.plots[module]
            bg_color, pen_color = ("k", "w") if dark_mode else ("w", "k")
            
            # Apply to amplitude plot
            amp_plot = plot_info['amp_plot']
            amp_plot.setBackground(bg_color)
            
            # Update plot title color - use more direct approach
            amp_plot_item = amp_plot.getPlotItem()
            if amp_plot_item and hasattr(amp_plot_item, 'titleLabel'):
                title_text = amp_plot_item.titleLabel.text if amp_plot_item.titleLabel.text else f"Module {module} - Magnitude"
                amp_plot_item.setTitle(title_text, color=pen_color)
                
            # Update axes colors
            for axis_name in ("left", "bottom", "right", "top"):
                ax = amp_plot_item.getAxis(axis_name) if amp_plot_item else None
                if ax: 
                    ax.setPen(pen_color)
                    ax.setTextPen(pen_color)
            
            # Update legend text color for amplitude plot using the proper API
            amp_legend = plot_info['amp_legend']
            if amp_legend:
                try:
                    amp_legend.setLabelTextColor(pen_color)
                    amp_legend.update()
                except Exception as e:
                    print(f"Error updating amp_legend colors: {e}")
            
            # Apply to phase plot
            phase_plot = plot_info['phase_plot']
            phase_plot.setBackground(bg_color)
            
            # Update plot title color - use more direct approach
            phase_plot_item = phase_plot.getPlotItem()
            if phase_plot_item and hasattr(phase_plot_item, 'titleLabel'):
                title_text = phase_plot_item.titleLabel.text if phase_plot_item.titleLabel.text else f"Module {module} - Phase"
                phase_plot_item.setTitle(title_text, color=pen_color)
                
            # Update axes colors
            for axis_name in ("left", "bottom", "right", "top"):
                ax = phase_plot_item.getAxis(axis_name) if phase_plot_item else None
                if ax:
                    ax.setPen(pen_color)
                    ax.setTextPen(pen_color)
            
            # Update legend text color for phase plot using the proper API
            phase_legend = plot_info['phase_legend']
            if phase_legend:
                try:
                    phase_legend.setLabelTextColor(pen_color)
                    # No need to call phase_legend.update() here, setLabelTextColor should suffice
                except Exception as e:
                    print(f"Error updating phase_legend colors: {e}")
            
            # Redraw plots to ensure all legend items are updated correctly
            self._redraw_all_plots()

            # The count in the magnitude title is drawn in the text colour
            self._update_resonance_title(module)
