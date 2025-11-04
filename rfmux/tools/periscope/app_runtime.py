"""Runtime mixin for the Periscope application."""

from .utils import *
from .tasks import *
from .ui import *
import asyncio

class PeriscopeRuntime:
    """Mixin providing runtime methods for :class:`Periscope`."""
    def _init_buffers(self):
        """
        Initialize or re-initialize data buffers for all configured channels.

        Creates `Circular` buffers for I, Q, Magnitude, and timestamps for
        each unique channel specified in `self.channel_list`.
        The buffer size is determined by `self.N`.
        """
        unique_chs = set()
        for group in self.channel_list:
            for c_val in group: unique_chs.add(c_val) # Renamed c
        self.all_chs = sorted(unique_chs)
        self.buf = {}
        self.tbuf = {}
        for ch_val in self.all_chs: # Renamed ch
            self.buf[ch_val] = {k: Circular(self.N) for k in ("I", "Q", "M")} # Circular from .utils
            self.tbuf[ch_val] = Circular(self.N)
        
        # Initialize simulation speed tracking for mock mode
        if self.is_mock_mode:
            self._init_sim_speed_tracking()

    def _build_layout(self):
        """
        Reconstruct the entire plot layout based on current settings.

        Clears the existing layout, determines active plot modes, and then
        creates and configures plots and curves for each channel group.
        Restores auto-range settings and applies I/Q/Mag visibility.
        """
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
        """Clear all widgets from the main plot grid layout."""
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget_item = item.widget() # Renamed w
            if widget_item: widget_item.deleteLater()

    def _get_active_modes(self) -> list[str]:
        """
        Get a list of currently active plot modes based on checkbox states.

        Returns:
            list[str]: A list of characters representing active modes (e.g., ['T', 'S']).
        """
        modes_list = [] # Renamed modes
        if self.cb_time.isChecked(): modes_list.append("T")
        if self.cb_iq.isChecked(): modes_list.append("IQ")
        if self.cb_fft.isChecked(): modes_list.append("F")
        if self.cb_ssb.isChecked(): modes_list.append("S")
        if self.cb_dsb.isChecked(): modes_list.append("D")
        return modes_list

    def _get_single_channel_colors(self) -> dict[str, str]:
        """
        Get a dictionary of default colors for single-channel plot components.

        Returns:
            dict[str, str]: Colors for 'I', 'Q', 'Mag', and 'DSB' components.
        """
        return {"I": "#1f77b4", "Q": "#ff7f0e", "Mag": "#2ca02c", "DSB": "#bcbd22"}

    def _get_channel_color_families(self) -> list[tuple[str, str, str]]:
        """
        Get a list of color families for multi-channel plots.

        Each family is a tuple of three related colors (e.g., for I, Q, Mag).

        Returns:
            list[tuple[str, str, str]]: A list of color family tuples.
        """
        return [
            ("#1f77b4", "#4a8cc5", "#90bce0"), ("#ff7f0e", "#ffa64d", "#ffd2a3"),
            ("#2ca02c", "#63c063", "#a1d9a1"), ("#d62728", "#eb6a6b", "#f2aeae"),
            ("#9467bd", "#ae8ecc", "#d3c4e3"), ("#8c564b", "#b58e87", "#d9c3bf"),
            ("#e377c2", "#f0a8dc", "#f7d2ee"), ("#7f7f7f", "#aaaaaa", "#d3d3d3"),
            ("#bcbd22", "#cfd342", "#e2e795"), ("#17becf", "#51d2de", "#9ae8f2"),
        ]

    def _create_row_plots_and_curves(self, row_i: int, group: list[int], modes_active: list[str], font: QFont,
                                        single_colors: dict[str, str], channel_families: list[tuple[str, str, str]]
                                        ) -> tuple[dict, dict]:
        """
        Create all plots and their corresponding curve items for a single row in the layout.

        Args:
            row_i (int): The index of the current row.
            group (list[int]): List of channel IDs in this row.
            modes_active (list[str]): List of active plot modes (e.g., "T", "IQ").
            font (QFont): Font to use for plot labels and titles.
            single_colors (dict[str, str]): Colors for single-channel plots.
            channel_families (list[tuple[str, str, str]]): Color families for multi-channel plots.

        Returns:
            tuple[dict, dict]: A tuple containing (rowPlots, rowCurves).
                                `rowPlots` maps mode key to PlotWidget.
                                `rowCurves` maps mode key to curve data structures.
        """
        rowPlots = {}; rowCurves = {}
        row_title = "Ch " + ("&".join(map(str, group)) if len(group) > 1 else str(group[0]))
        for col, mode_key in enumerate(modes_active): # Renamed mode
            vb = ClickableViewBox() # From .utils
            pw = pg.PlotWidget(viewBox=vb, title=f"{mode_title(mode_key)} – {row_title}") # mode_title from .utils
            self._configure_plot_auto_range(pw, mode_key)
            self._configure_plot_axes(pw, mode_key, group)
            self._apply_plot_theme(pw)
            pw.showGrid(x=True, y=True, alpha=0.3)
            self.grid.addWidget(pw, row_i, col)
            rowPlots[mode_key] = pw
            self._configure_plot_fonts(pw, font)
            if mode_key == "IQ":
                rowCurves["IQ"] = self._create_iq_plot_item(pw)
            else:
                # Define colors based on dark mode
                bg_color, pen_color = ("k", "w") if self.dark_mode else ("w", "k")
                legend = pw.addLegend(offset=(30, 10), labelTextColor=pen_color)
                if len(group) == 1:
                    ch_val = group[0] # Renamed ch
                    rowCurves[mode_key] = self._create_single_channel_curves(pw, mode_key, ch_val, single_colors, legend)
                else:
                    rowCurves[mode_key] = self._create_multi_channel_curves(pw, mode_key, group, channel_families, legend)
                self._make_legend_clickable(legend)
        return rowPlots, rowCurves

    def _configure_plot_auto_range(self, pw: pg.PlotWidget, mode_key: str):
        """Configure auto-ranging for a given plot based on its mode."""
        if mode_key == "T": pw.enableAutoRange(pg.ViewBox.XYAxes, True)
        else: pw.enableAutoRange(pg.ViewBox.XYAxes, self.auto_scale_plots)

    def _configure_plot_axes(self, pw: pg.PlotWidget, mode_key: str, group: list[int]):
        """Configure axis labels and scales for a given plot based on its mode and calibration availability."""
        # Check if all channels in the group have df calibration
        all_have_calibration = False
        if self.unit_mode == "df" and hasattr(self, 'df_calibrations') and self.module in self.df_calibrations:
            all_have_calibration = all(
                ch in self.df_calibrations[self.module] 
                for ch in group
            )
        
        # Determine axis labels based on unit mode and calibration availability
        if mode_key == "T":
            if self.unit_mode == "df" and all_have_calibration:
                pw.setLabel("left", "Freq. Shift / Dissipation", units="Hz / unitless")
            else:
                pw.setLabel("left", "Amplitude", units="V" if self.real_units else "Counts")
        elif mode_key == "IQ":
            pw.getViewBox().setAspectLocked(True)
            if self.unit_mode == "df" and all_have_calibration:
                pw.setLabel("bottom", "Freq. Shift", units="Hz")
                pw.setLabel("left", "Dissipation", units="unitless")
            else:
                pw.setLabel("bottom", "I", units="V" if self.real_units else "Counts")
                pw.setLabel("left",   "Q", units="V" if self.real_units else "Counts")
        elif mode_key == "F":
            pw.setLogMode(x=True, y=True)
            pw.setLabel("bottom", "Freq", units="Hz")
            if self.unit_mode == "df" and all_have_calibration:
                pw.setLabel("left", "Amplitude", units="Hz or unitless")
            else:
                pw.setLabel("left", "Amplitude", units="V" if self.real_units else "Counts")
        elif mode_key == "S":
            pw.setLogMode(x=True, y=not self.real_units)
            pw.setLabel("bottom", "Freq", units="Hz")
            if self.unit_mode == "df" and all_have_calibration:
                # When in df units, PSDs become ASDs (Amplitude Spectral Densities)
                pw.setLabel("left", "ASD (Hz/√Hz or 1/√Hz)")
            else:
                lbl = "dBm/Hz" if self.psd_absolute else "dBc/Hz"
                pw.setLabel("left", f"PSD ({lbl})" if self.real_units else "PSD (Counts²/Hz)")
        else:  # "D"
            pw.setLogMode(x=False, y=not self.real_units)
            pw.setLabel("bottom", "Freq", units="Hz")
            if self.unit_mode == "df" and all_have_calibration:
                # When in df units, PSDs become ASDs (Amplitude Spectral Densities)
                pw.setLabel("left", "ASD (Hz/√Hz or 1/√Hz)")
            else:
                lbl = "dBm/Hz" if self.psd_absolute else "dBc/Hz"
                pw.setLabel("left", f"PSD ({lbl})" if self.real_units else "PSD (Counts²/Hz)")

    def _configure_plot_fonts(self, pw: pg.PlotWidget, font: QFont):
        """Apply the standard font to plot titles and axis labels."""
        pi = pw.getPlotItem()
        for axis_name in ("left", "bottom", "right", "top"):
            axis = pi.getAxis(axis_name)
            if axis: axis.setTickFont(font)
            if axis and axis.label: axis.label.setFont(font) # Check axis.label
        pi.titleLabel.setFont(font)

    def _create_iq_plot_item(self, pw: pg.PlotWidget) -> dict:
        """
        Create and return the appropriate item for an IQ plot (scatter or density).

        Args:
            pw (pg.PlotWidget): The plot widget to add the item to.

        Returns:
            dict: A dictionary containing the 'mode' ('scatter' or 'density')
                    and the 'item' (ScatterPlotItem or ImageItem).
        """
        # SCATTER_SIZE from .utils
        if self.rb_scatter.isChecked():
            sp = pg.ScatterPlotItem(pen=None, size=SCATTER_SIZE)
            pw.addItem(sp); return {"mode": "scatter", "item": sp}
        else:
            img = pg.ImageItem(axisOrder="row-major")
            img.setLookupTable(self.lut); pw.addItem(img)
            return {"mode": "density", "item": img}

    def _create_single_channel_curves(self, pw: pg.PlotWidget, mode_key: str, ch_val: int,
                                        single_colors: dict[str, str], legend: pg.LegendItem
                                        ) -> dict[int, dict[str, pg.PlotDataItem]]:
        """
        Create plot curves for a single channel in a given plot mode.

        Args:
            pw (pg.PlotWidget): The plot widget to add curves to.
            mode_key (str): The plot mode (e.g., "T", "F", "S", "D").
            ch_val (int): The channel ID.
            single_colors (dict[str, str]): Colors for I, Q, Mag, DSB.
            legend (pg.LegendItem): The legend to add curve entries to.

        Returns:
            dict: A dictionary mapping the channel ID to its curve items.
        """
        # LINE_WIDTH from .utils
        # Determine legend names based on unit mode AND calibration availability
        has_df_calibration = (
            self.unit_mode == "df" and 
            hasattr(self, 'df_calibrations') and 
            self.module in self.df_calibrations and
            ch_val in self.df_calibrations[self.module]
        )
        
        if has_df_calibration:
            i_name = "Freq Shift"
            q_name = "Dissipation"
        else:
            i_name = "I"
            q_name = "Q"
            
        if mode_key == "T":
            cI = pw.plot(pen=pg.mkPen(single_colors["I"],   width=LINE_WIDTH), name=i_name)
            cQ = pw.plot(pen=pg.mkPen(single_colors["Q"],   width=LINE_WIDTH), name=q_name)
            # Only create magnitude curve if not in df units with calibration
            if has_df_calibration:
                cM = None  # No magnitude in df units
                self._fade_hidden_entries(legend, (i_name, q_name))
                return {ch_val: {"I": cI, "Q": cQ}}
            else:
                cM = pw.plot(pen=pg.mkPen(single_colors["Mag"], width=LINE_WIDTH), name="Mag")
                self._fade_hidden_entries(legend, (i_name, q_name))
                return {ch_val: {"I": cI, "Q": cQ, "Mag": cM}}
        elif mode_key == "F":
            cI = pw.plot(pen=pg.mkPen(single_colors["I"],   width=LINE_WIDTH), name=i_name); cI.setFftMode(True)
            cQ = pw.plot(pen=pg.mkPen(single_colors["Q"],   width=LINE_WIDTH), name=q_name); cQ.setFftMode(True)
            # Only create magnitude curve if not in df units with calibration
            if has_df_calibration:
                self._fade_hidden_entries(legend, (i_name, q_name))
                return {ch_val: {"I": cI, "Q": cQ}}
            else:
                cM = pw.plot(pen=pg.mkPen(single_colors["Mag"], width=LINE_WIDTH), name="Mag"); cM.setFftMode(True)
                self._fade_hidden_entries(legend, (i_name, q_name))
                return {ch_val: {"I": cI, "Q": cQ, "Mag": cM}}
        elif mode_key == "S":
            cI = pw.plot(pen=pg.mkPen(single_colors["I"],   width=LINE_WIDTH), name=i_name)
            cQ = pw.plot(pen=pg.mkPen(single_colors["Q"],   width=LINE_WIDTH), name=q_name)
            # Only create magnitude curve if not in df units with calibration
            if has_df_calibration:
                return {ch_val: {"I": cI, "Q": cQ}}
            else:
                cM = pw.plot(pen=pg.mkPen(single_colors["Mag"], width=LINE_WIDTH), name="Mag")
                return {ch_val: {"I": cI, "Q": cQ, "Mag": cM}}
        else:  # "D"
            cD = pw.plot(pen=pg.mkPen(single_colors["DSB"], width=LINE_WIDTH), name="Complex DSB")
            return {ch_val: {"Cmplx": cD}}

    def _create_multi_channel_curves(self, pw: pg.PlotWidget, mode_key: str, group: list[int],
                                        channel_families: list[tuple[str, str, str]], legend: pg.LegendItem
                                        ) -> dict[int, dict[str, pg.PlotDataItem]]:
        """
        Create plot curves for multiple channels in a group for a given plot mode.

        Args:
            pw (pg.PlotWidget): The plot widget to add curves to.
            mode_key (str): The plot mode (e.g., "T", "F", "S", "D").
            group (list[int]): List of channel IDs in this group.
            channel_families (list[tuple[str, str, str]]): Color families for curves.
            legend (pg.LegendItem): The legend to add curve entries to.

        Returns:
            dict: A dictionary mapping each channel ID in the group to its curve items.
        """
        mode_dict = {}
        for i, ch_val in enumerate(group): # Renamed ch
            # Check calibration availability per channel
            has_df_calibration = (
                self.unit_mode == "df" and 
                hasattr(self, 'df_calibrations') and 
                self.module in self.df_calibrations and
                ch_val in self.df_calibrations[self.module]
            )
            
            # Set suffixes based on calibration availability
            if has_df_calibration:
                i_suffix = "-FreqShift"
                q_suffix = "-Dissipation"
            else:
                i_suffix = "-I"
                q_suffix = "-Q"
                
            (colI, colQ, colM) = channel_families[i % len(channel_families)]
            if mode_key == "T":
                cI = pw.plot(pen=pg.mkPen(colI, width=LINE_WIDTH), name=f"ch{ch_val}{i_suffix}")
                cQ = pw.plot(pen=pg.mkPen(colQ, width=LINE_WIDTH), name=f"ch{ch_val}{q_suffix}")
                # Only create magnitude curve if not in df units with calibration
                if has_df_calibration:
                    mode_dict[ch_val] = {"I": cI, "Q": cQ}
                else:
                    cM = pw.plot(pen=pg.mkPen(colM, width=LINE_WIDTH), name=f"ch{ch_val}-Mag")
                    mode_dict[ch_val] = {"I": cI, "Q": cQ, "Mag": cM}
            elif mode_key == "F":
                cI = pw.plot(pen=pg.mkPen(colI, width=LINE_WIDTH), name=f"ch{ch_val}{i_suffix}"); cI.setFftMode(True)
                cQ = pw.plot(pen=pg.mkPen(colQ, width=LINE_WIDTH), name=f"ch{ch_val}{q_suffix}"); cQ.setFftMode(True)
                # Only create magnitude curve if not in df units with calibration
                if has_df_calibration:
                    mode_dict[ch_val] = {"I": cI, "Q": cQ}
                else:
                    cM = pw.plot(pen=pg.mkPen(colM, width=LINE_WIDTH), name=f"ch{ch_val}-Mag"); cM.setFftMode(True)
                    mode_dict[ch_val] = {"I": cI, "Q": cQ, "Mag": cM}
            elif mode_key == "S":
                cI = pw.plot(pen=pg.mkPen(colI, width=LINE_WIDTH), name=f"ch{ch_val}{i_suffix}")
                cQ = pw.plot(pen=pg.mkPen(colQ, width=LINE_WIDTH), name=f"ch{ch_val}{q_suffix}")
                # Only create magnitude curve if not in df units with calibration
                if has_df_calibration:
                    mode_dict[ch_val] = {"I": cI, "Q": cQ}
                else:
                    cM = pw.plot(pen=pg.mkPen(colM, width=LINE_WIDTH), name=f"ch{ch_val}-Mag")
                    mode_dict[ch_val] = {"I": cI, "Q": cQ, "Mag": cM}
            else:  # "D"
                cD = pw.plot(pen=pg.mkPen(colI, width=LINE_WIDTH), name=f"ch{ch_val}-DSB")
                mode_dict[ch_val] = {"Cmplx": cD}
        return mode_dict

    def _restore_auto_range_settings(self):
        """Restore auto-range settings for all plots, typically after a layout rebuild."""
        self.auto_scale_plots = True
        self.cb_auto_scale.setChecked(True)
        for rowPlots in self.plots:
            for mode_key, pw in rowPlots.items(): # Renamed mode
                if mode_key != "T": pw.enableAutoRange(pg.ViewBox.XYAxes, True)

    def _apply_plot_theme(self, pw: pg.PlotWidget):
        """Apply the current theme (dark/light) to a plot widget."""
        bg_color, pen_color = ("k", "w") if self.dark_mode else ("w", "k")
        pw.setBackground(bg_color)
        
        # Update plot title color directly
        plot_item = pw.getPlotItem()
        if plot_item and plot_item.titleLabel:
            current_title = plot_item.titleLabel.text
            plot_item.setTitle(current_title, color=pen_color)
            
        # Update axis colors
        for axis_name in ("left", "bottom", "right", "top"):
            ax = pw.getPlotItem().getAxis(axis_name)
            if ax: ax.setPen(pen_color); ax.setTextPen(pen_color)
            
        # Update legend text color directly using the proper API
        legend = pw.getPlotItem().legend
        if legend:
            try:
                # Use the proper API method to update all label colors at once
                legend.setLabelTextColor(pen_color)
            except Exception as e:
                print(f"Error updating legend colors: {e}")

    @staticmethod
    def _fade_hidden_entries(legend: pg.LegendItem, hide_labels: tuple[str, ...]):
        """Fade legend entries corresponding to initially hidden curves."""
        for sample, label in legend.items:
            txt = label.labelItem.toPlainText() if hasattr(label, "labelItem") else ""
            if txt in hide_labels: sample.setOpacity(0.3); label.setOpacity(0.3)

    @staticmethod
    def _make_legend_clickable(legend: pg.LegendItem):
        """Make legend items clickable to toggle curve visibility."""
        for sample, label in legend.items:
            curve = sample.item
            def toggle(evt, c=curve, s=sample, l=label):
                vis = not c.isVisible(); c.setVisible(vis)
                op = 1.0 if vis else 0.3; s.setOpacity(op); l.setOpacity(op)
            label.mousePressEvent = toggle; sample.mousePressEvent = toggle

    def _toggle_dark_mode(self, checked: bool):
        """Toggle dark mode theme and rebuild the layout."""
        self.dark_mode = checked; self._build_layout()
        self._update_dark_mode_in_child_windows()
        
    def _update_dark_mode_in_child_windows(self):
        """Propagate dark mode setting to all open child windows."""
        # Update NetworkAnalysis windows
        if hasattr(self, 'netanal_windows'):
            for window_data in self.netanal_windows.values():
                if 'window' in window_data and hasattr(window_data['window'], 'apply_theme'):
                    window_data['window'].apply_theme(self.dark_mode)
        
        # Update Multisweep windows
        if hasattr(self, 'multisweep_windows'):
            for window_data in self.multisweep_windows.values():
                if 'window' in window_data and hasattr(window_data['window'], 'apply_theme'):
                    window_data['window'].apply_theme(self.dark_mode)

    def _toggle_real_units(self, checked: bool):
        """
        Toggle between displaying data in raw counts or real units (V, dBm/Hz).
        Rebuilds the layout and shows an informational message if real units are enabled.
        """
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
        """Handle change in PSD reference (absolute/relative) and rebuild layout."""
        self.psd_absolute = self.rb_psd_abs.isChecked(); self._build_layout()

    def _update_channels(self):
        """
        Update channel configuration based on user input in the channel string field.
        Re-initializes buffers and rebuilds the layout if changes are detected.
        """
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
        """Update buffer size based on user input and re-initialize buffers."""
        try: n_val = int(self.e_buf.text()) # Renamed n
        except ValueError: return
        if n_val != self.N: self.N = n_val; self._init_buffers()

    def _toggle_pause(self):
        """Toggle the pause state for data acquisition and display."""
        self.paused = not self.paused
        self.b_pause.setText("Resume" if self.paused else "Pause")

    def _update_gui(self):
        """Main GUI update loop, called periodically by a QTimer."""
        if self.paused: self._discard_packets(); return
        self._process_incoming_packets()
        self.frame_cnt += 1; now = time.time()
        if (now - self.last_dec_update) > 1.0: # Update decimation stage approx once per second
            self._update_dec_stage(); self.last_dec_update = now
        self._update_plot_data()
        self._update_performance_stats(now)

    def _discard_packets(self):
        """Discard all packets currently in the receiver queue (when paused)."""
        while not self.receiver.queue.empty(): self.receiver.queue.get()

    def _process_incoming_packets(self):
        """Process all packets currently in the receiver queue."""
        while not self.receiver.queue.empty():
            (seq, pkt) = self.receiver.queue.get(); self.pkt_cnt += 1
            # Store the actual decimation stage from the packet
            if hasattr(pkt, 'fir_stage'):
                self.actual_dec_stage = pkt.fir_stage
            t_rel = self._calculate_relative_timestamp(pkt)
            self._update_buffers(pkt, t_rel)
            
            # Track simulation time for speed calculation (mock mode only)
            if self.is_mock_mode and t_rel is not None:
                self._update_sim_time_tracking(t_rel)

    def _calculate_relative_timestamp(self, pkt) -> float | None:
        """
        Calculate a relative timestamp for a packet.

        If the packet's timestamp is recent, it's adjusted slightly and
        converted to seconds relative to the first packet's timestamp.

        Args:
            pkt: The incoming packet object.

        Returns:
            float | None: Relative timestamp in seconds, or None if not recent.
        """
        # streamer from .utils
        ts = pkt.ts
        if ts.recent:
            # Apply a small offset to ensure timestamps are strictly increasing for plotting
            ts.ss += int(0.02 * streamer.SS_PER_SECOND); ts.renormalize()
            t_now = ts.h * 3600 + ts.m * 60 + ts.s + ts.ss / streamer.SS_PER_SECOND
            if self.start_time is None: self.start_time = t_now
            return t_now - self.start_time
        return None

    def _update_buffers(self, pkt, t_rel: float | None):
        """
        Update data buffers with I, Q, and Magnitude values from a packet.

        Args:
            pkt: The incoming packet object.
            t_rel (float | None): The relative timestamp for this packet.
        """
        # math from .utils
        for ch_val in self.all_chs: # Renamed ch
            if len(pkt.s)/2 <= ch_val-1:
                continue # don't plot channels that aren't streamed

            Ival = pkt.s[2 * (ch_val - 1)] / 256.0  # Assuming 8-bit ADC data
            Qval = pkt.s[2 * (ch_val - 1) + 1] / 256.0
            self.buf[ch_val]["I"].add(Ival); self.buf[ch_val]["Q"].add(Qval)
            self.buf[ch_val]["M"].add(math.hypot(Ival, Qval)); self.tbuf[ch_val].add(t_rel)

    def _update_plot_data(self):
        """Update all active plots with new data from buffers and dispatch worker tasks."""
        for row_i, group in enumerate(self.channel_list):
            rowCurves = self.curves[row_i]
            for ch_val in group: self._update_channel_plot_data(ch_val, rowCurves) # Renamed ch
            # Dispatch IQ task if an IQ plot exists for this row and no worker is active
            if "IQ" in rowCurves and not self.iq_workers.get(row_i, False):
                self._dispatch_iq_task(row_i, group, rowCurves)
            # Dispatch PSD tasks (SSB and/or DSB)
            self._dispatch_psd_tasks(row_i, group)

    def _update_channel_plot_data(self, ch_val: int, rowCurves: dict):
        """
        Update Time-Domain (TOD) and FFT plots for a specific channel.

        Args:
            ch_val (int): The channel ID to update.
            rowCurves (dict): The dictionary of curve items for the current row.
        """
        rawI = self.buf[ch_val]["I"].data(); rawQ = self.buf[ch_val]["Q"].data()
        rawM = self.buf[ch_val]["M"].data(); tarr = self.tbuf[ch_val].data()
        
        # Apply appropriate unit conversion based on unit_mode
        if self.unit_mode == "df" and hasattr(self, 'df_calibrations') and self.module in self.df_calibrations:
            # Check if calibration data exists for this channel
            df_cal = self.df_calibrations[self.module].get(ch_val)
            if df_cal is not None:
                # Convert to volts first
                I_volts = convert_roc_to_volts(rawI)
                Q_volts = convert_roc_to_volts(rawQ)
                # Apply df calibration: multiply complex IQ by calibration factor
                iq_volts = I_volts + 1j * Q_volts
                df_complex = iq_volts * df_cal
                # Extract frequency shift (real) and dissipation (imaginary)
                I_data = df_complex.real  # Frequency shift in Hz
                Q_data = df_complex.imag  # Dissipation (unitless)
                # Magnitude in df space
                M_data = np.abs(df_complex)
            else:
                # No calibration for this channel, fall back to counts
                I_data, Q_data, M_data = rawI, rawQ, rawM
        elif self.real_units:
            # Standard voltage conversion
            I_data, Q_data, M_data = (convert_roc_to_volts(d) for d in (rawI, rawQ, rawM))
        else:
            # Raw counts
            I_data, Q_data, M_data = rawI, rawQ, rawM
        
        # Update Time-Domain (TOD) plots
        if "T" in rowCurves and ch_val in rowCurves["T"]:
            cset = rowCurves["T"][ch_val]
            if cset["I"].isVisible(): cset["I"].setData(tarr, I_data)
            if cset["Q"].isVisible(): cset["Q"].setData(tarr, Q_data)
            if "Mag" in cset and cset["Mag"].isVisible(): cset["Mag"].setData(tarr, M_data)
        
        # Update FFT plots
        if "F" in rowCurves and ch_val in rowCurves["F"]:
            cset = rowCurves["F"][ch_val]
            if cset["I"].isVisible(): cset["I"].setData(tarr, I_data, fftMode=True)
            if cset["Q"].isVisible(): cset["Q"].setData(tarr, Q_data, fftMode=True)
            if "Mag" in cset and cset["Mag"].isVisible(): cset["Mag"].setData(tarr, M_data, fftMode=True)

    def _dispatch_iq_task(self, row_i: int, group: list[int], rowCurves: dict):
        """
        Dispatch an IQ plot worker task (density or scatter).

        Args:
            row_i (int): The row index of the IQ plot.
            group (list[int]): List of channel IDs for this plot.
            rowCurves (dict): Curve items for the current row.
        """
        # IQTask from .tasks
        mode_key = rowCurves["IQ"]["mode"] # Renamed mode
        if len(group) == 1: # Single channel IQ plot
            c_val = group[0]
            rawI = self.buf[c_val]["I"].data()
            rawQ = self.buf[c_val]["Q"].data()
            
            # Apply unit conversion (same logic as in _update_channel_plot_data)
            if self.unit_mode == "df" and hasattr(self, 'df_calibrations') and self.module in self.df_calibrations:
                df_cal = self.df_calibrations[self.module].get(c_val)
                if df_cal is not None:
                    # Convert to volts first
                    I_volts = convert_roc_to_volts(rawI)
                    Q_volts = convert_roc_to_volts(rawQ)
                    # Apply df calibration
                    iq_volts = I_volts + 1j * Q_volts
                    df_complex = iq_volts * df_cal
                    # Extract frequency shift (real) and dissipation (imaginary)
                    I_data = df_complex.real  # Frequency shift in Hz
                    Q_data = df_complex.imag  # Dissipation (unitless)
                else:
                    # No calibration for this channel
                    I_data, Q_data = rawI, rawQ
            elif self.real_units:
                # Standard voltage conversion
                I_data = convert_roc_to_volts(rawI)
                Q_data = convert_roc_to_volts(rawQ)
            else:
                # Raw counts
                I_data, Q_data = rawI, rawQ
            
            self.iq_workers[row_i] = True
            task = IQTask(row_i, c_val, I_data, Q_data, self.dot_px, mode_key, self.iq_signals)
            self.pool.start(task)
        else: # Multi-channel IQ plot (concatenate converted data)
            all_I_data = []
            all_Q_data = []
            
            for ch in group:
                rawI = self.buf[ch]["I"].data()
                rawQ = self.buf[ch]["Q"].data()
                
                # Apply unit conversion for each channel
                if self.unit_mode == "df" and hasattr(self, 'df_calibrations') and self.module in self.df_calibrations:
                    df_cal = self.df_calibrations[self.module].get(ch)
                    if df_cal is not None:
                        # Convert to volts first
                        I_volts = convert_roc_to_volts(rawI)
                        Q_volts = convert_roc_to_volts(rawQ)
                        # Apply df calibration
                        iq_volts = I_volts + 1j * Q_volts
                        df_complex = iq_volts * df_cal
                        # Extract frequency shift and dissipation
                        I_data = df_complex.real
                        Q_data = df_complex.imag
                    else:
                        I_data, Q_data = rawI, rawQ
                elif self.real_units:
                    I_data = convert_roc_to_volts(rawI)
                    Q_data = convert_roc_to_volts(rawQ)
                else:
                    I_data, Q_data = rawI, rawQ
                    
                all_I_data.append(I_data)
                all_Q_data.append(Q_data)
            
            concatI = np.concatenate(all_I_data)
            concatQ = np.concatenate(all_Q_data)
            
            big_size = concatI.size
            if big_size > 50000: # Downsample if too many points for performance
                stride = max(1, big_size // 50000)
                concatI = concatI[::stride]; concatQ = concatQ[::stride]
            if concatI.size > 1:
                self.iq_workers[row_i] = True
                task = IQTask(row_i, 0, concatI, concatQ, self.dot_px, mode_key, self.iq_signals) # Channel ID 0 for multi-channel
                self.pool.start(task)

    def _dispatch_psd_tasks(self, row_i: int, group: list[int]):
        """
        Dispatch PSD calculation worker tasks (SSB and/or DSB) for channels in a group.

        Args:
            row_i (int): The row index of the PSD plots.
            group (list[int]): List of channel IDs for this plot row.
        """
        # PSDTask from .tasks, convert_roc_to_volts from .utils
        # Dispatch Single-Sideband (SSB) PSD tasks
        if "S" in self.curves[row_i]:
            for ch_val in group: # Renamed ch
                if not self.psd_workers[row_i]["S"].get(ch_val, False): # Check if worker already active
                    rawI = self.buf[ch_val]["I"].data(); rawQ = self.buf[ch_val]["Q"].data()
                    
                    # Apply unit conversion (same logic as in _update_channel_plot_data)
                    if self.unit_mode == "df" and hasattr(self, 'df_calibrations') and self.module in self.df_calibrations:
                        df_cal = self.df_calibrations[self.module].get(ch_val)
                        if df_cal is not None:
                            # Convert to volts first
                            I_volts = convert_roc_to_volts(rawI)
                            Q_volts = convert_roc_to_volts(rawQ)
                            # Apply df calibration
                            iq_volts = I_volts + 1j * Q_volts
                            df_complex = iq_volts * df_cal
                            # Extract frequency shift (real) and dissipation (imaginary)
                            I_data = df_complex.real  # Frequency shift in Hz
                            Q_data = df_complex.imag  # Dissipation (unitless)
                        else:
                            # No calibration for this channel
                            I_data, Q_data = (convert_roc_to_volts(d) for d in (rawI, rawQ)) if self.real_units else (rawI, rawQ)
                    elif self.real_units:
                        # Standard voltage conversion
                        I_data, Q_data = (convert_roc_to_volts(d) for d in (rawI, rawQ))
                    else:
                        # Raw counts
                        I_data, Q_data = rawI, rawQ
                    
                    self.psd_workers[row_i]["S"][ch_val] = True
                    task = PSDTask(row_i, ch_val, I_data, Q_data, "SSB", self.dec_stage, self.real_units, self.psd_absolute, self.spin_segments.value(), self.psd_signals, self.cb_exp_binning.isChecked(), self.spin_bins.value())
                    self.pool.start(task)
        
        # Dispatch Dual-Sideband (DSB) PSD tasks
        if "D" in self.curves[row_i]:
            for ch_val in group: # Renamed ch
                if not self.psd_workers[row_i]["D"].get(ch_val, False): # Check if worker already active
                    rawI = self.buf[ch_val]["I"].data(); rawQ = self.buf[ch_val]["Q"].data()
                    
                    # Apply unit conversion (same logic as in _update_channel_plot_data)
                    if self.unit_mode == "df" and hasattr(self, 'df_calibrations') and self.module in self.df_calibrations:
                        df_cal = self.df_calibrations[self.module].get(ch_val)
                        if df_cal is not None:
                            # Convert to volts first
                            I_volts = convert_roc_to_volts(rawI)
                            Q_volts = convert_roc_to_volts(rawQ)
                            # Apply df calibration
                            iq_volts = I_volts + 1j * Q_volts
                            df_complex = iq_volts * df_cal
                            # Extract frequency shift (real) and dissipation (imaginary)
                            I_data = df_complex.real  # Frequency shift in Hz
                            Q_data = df_complex.imag  # Dissipation (unitless)
                        else:
                            # No calibration for this channel
                            I_data, Q_data = (convert_roc_to_volts(d) for d in (rawI, rawQ)) if self.real_units else (rawI, rawQ)
                    elif self.real_units:
                        # Standard voltage conversion
                        I_data, Q_data = (convert_roc_to_volts(d) for d in (rawI, rawQ))
                    else:
                        # Raw counts
                        I_data, Q_data = rawI, rawQ
                    
                    self.psd_workers[row_i]["D"][ch_val] = True
                    task = PSDTask(row_i, ch_val, I_data, Q_data, "DSB", self.dec_stage, self.real_units, self.psd_absolute, self.spin_segments.value(), self.psd_signals, self.cb_exp_binning.isChecked(), self.spin_bins.value())
                    self.pool.start(task)

    def _update_performance_stats(self, now: float):
        """Update FPS and PPS display in the status bar approximately once per second."""
        if (now - self.t_last) >= 1.0:
            #### Getting packet counts ###
            dropped = self.receiver.get_dropped_packets()
            received = self.receiver.get_received_packets()
            
            ##### Percent calculation #####
            drop_lastsec = dropped - self.prev_drop
            receive_lastsec = received - self.prev_receive
            # Avoid division by zero when no packets received
            total_lastsec = drop_lastsec + receive_lastsec
            percent = (drop_lastsec / total_lastsec * 100) if total_lastsec > 0 else 0.0
            
            #### Per second metrics #####
            fps = self.frame_cnt / (now - self.t_last)
            pps = self.pkt_cnt / (now - self.t_last)

            #### Update labels ####
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.pps_label.setText(f"Packets/s: {pps:.1f}")
            
            # Calculate and display simulation speed for mock mode
            if self.is_mock_mode and hasattr(self, 'sim_speed_label'):
                sim_speed = self._calculate_simulation_speed()
                if sim_speed is not None:
                    # Format the display
                    if sim_speed < 0.01:
                        speed_text = f"Sim Speed: {sim_speed:.3f}x real-time"
                    else:
                        speed_text = f"Sim Speed: {sim_speed:.2f}x real-time"
                    self.sim_speed_label.setText(speed_text)
                    
                    # Color code based on speed
                    if sim_speed < 0.1:
                        self.sim_speed_label.setStyleSheet("color: orange;")  # Very slow
                    else:
                        self.sim_speed_label.setStyleSheet("")  # Normal (near real-time)
                    # elif sim_speed < 0.9:
                    #     self.sim_speed_label.setStyleSheet("color: orange;")  # Slow
                    # elif sim_speed > 1.1:
                    #     self.sim_speed_label.setStyleSheet("color: green;")  # Fast
                    # else:
                    #     

            # Color packet loss red if > 1%
            if percent > 1:
                color = "red"
                self.packet_loss_label.setStyleSheet(f"color: {color};")
                self.info_text.setText("PACKET LOSS HIGH - CONSULT HELP FOR NETWORKING SUGGESTIONS")
            else: 
                self.packet_loss_label.setStyleSheet(f"color: {self.default_packet_loss_color};")
                self.info_text.clear()
                
            self.packet_loss_label.setText(f"Packet Loss: {percent:.1f}%")

            self.dropped_label.setText(f"Dropped: {dropped}")
            
            #### Showing on status bar ####
            # self.statusBar().showMessage(f"FPS {fps:.1f} | Packets/s {pps:.1f} | Packet Loss : {percent_x}% | Dropped : {dropped}") 
            
            #### Initializing ####
            self.frame_cnt = 0; self.pkt_cnt = 0; self.t_last = now
            self.prev_drop = dropped
            self.prev_receive = received
            

    def _update_dec_stage(self):
        """
        Update the decimation stage from the actual packet data.
        Falls back to inferring from data rate if packet doesn't have fir_stage.
        This is used for accurate PSD frequency axis scaling.
        """
        # Use actual decimation stage from packets if available
        if hasattr(self, 'actual_dec_stage'):
            self.dec_stage = self.actual_dec_stage
            return
            
        # Fall back to inferring from data rate (legacy behavior)
        # infer_dec_stage from .utils
        if not self.channel_list or not self.channel_list[0]: return
        ch_val = self.channel_list[0][0] # Renamed ch
        tarr = self.tbuf[ch_val].data()
        if len(tarr) < 2: return # Need at least two points to calculate dt
        dt = (tarr[-1] - tarr[0]) / max(1, (len(tarr) - 1)) # Average time step
        fs = 1.0 / dt if dt > 0 else 1.0 # Sample rate
        self.dec_stage = infer_dec_stage(fs)

    @QtCore.pyqtSlot(int, str, object)
    def _iq_done(self, row: int, task_mode: str, payload):
        """
        Slot for handling completion of an IQ worker task.
        Updates the corresponding IQ plot (density or scatter).

        Args:
            row (int): Row index of the completed IQ task.
            task_mode (str): 'density' or 'scatter'.
            payload: Data returned by the IQTask (e.g., histogram, scatter points).
        """
        self.iq_workers[row] = False # Mark worker as no longer active
        if row >= len(self.curves) or "IQ" not in self.curves[row]: return # Plot might have been removed
        pane = self.curves[row]["IQ"]
        if pane["mode"] != task_mode: return # Mode might have changed while task was running
        item = pane["item"]
        if task_mode == "density": self._update_density_image(item, payload)
        else: self._update_scatter_plot(item, payload)

    def _update_density_image(self, item: pg.ImageItem, payload):
        """
        Update an IQ density plot (ImageItem) with new histogram data.

        Args:
            item (pg.ImageItem): The ImageItem to update.
            payload: Tuple containing (histogram, (Imin, Imax, Qmin, Qmax)).
        """
        # convert_roc_to_volts from .utils
        hist, (Imin, Imax, Qmin, Qmax) = payload
        if self.real_units: # Convert bounds if real units are active
            Imin, Imax = convert_roc_to_volts(np.array([Imin, Imax], dtype=float))
            Qmin, Qmax = convert_roc_to_volts(np.array([Qmin, Qmax], dtype=float))
        item.setImage(hist, levels=(0, 255), autoLevels=False) # Update image data
        item.setRect(QtCore.QRectF(float(Imin), float(Qmin), float(Imax - Imin), float(Qmax - Qmin))) # Set image bounds

    def _update_scatter_plot(self, item: pg.ScatterPlotItem, payload):
        """
        Update an IQ scatter plot (ScatterPlotItem) with new points.

        Args:
            item (pg.ScatterPlotItem): The ScatterPlotItem to update.
            payload: Tuple containing (xs, ys, colors) for scatter points.
        """
        # convert_roc_to_volts, SCATTER_SIZE from .utils
        xs, ys, colors = payload
        if self.real_units: xs = convert_roc_to_volts(xs); ys = convert_roc_to_volts(ys) # Convert points if real units
        item.setData(xs, ys, brush=colors, pen=None, size=SCATTER_SIZE) # Update scatter plot data

    @QtCore.pyqtSlot(int, str, int, object)
    def _psd_done(self, row: int, psd_mode: str, ch_val: int, payload): # Renamed ch
        """
        Slot for handling completion of a PSD worker task.
        Updates the corresponding SSB or DSB plot curves.

        Args:
            row (int): Row index of the completed PSD task.
            psd_mode (str): 'SSB' or 'DSB'.
            ch_val (int): Channel ID for which PSD was calculated.
            payload: Data returned by the PSDTask (e.g., frequencies, PSD values).
        """
        if row not in self.psd_workers: return # Row might no longer exist
        key = psd_mode[0] # 'S' or 'D'
        if key not in self.psd_workers[row] or ch_val not in self.psd_workers[row][key]: return # Channel/mode might no longer exist
        
        self.psd_workers[row][key][ch_val] = False # Mark worker as no longer active
        if row >= len(self.curves): return # Plot might have been removed
        
        if psd_mode == "SSB": self._update_ssb_curves(row, ch_val, payload)
        else: self._update_dsb_curve(row, ch_val, payload)

    def _update_ssb_curves(self, row: int, ch_val: int, payload):
        """
        Update Single-Sideband (SSB) PSD plot curves.

        Args:
            row (int): Row index of the plot.
            ch_val (int): Channel ID.
            payload: Tuple containing (freq_i, psd_i, psd_q, psd_m, ...).
        """
        if "S" not in self.curves[row]: return # SSB plot might not exist for this row
        sdict = self.curves[row]["S"]
        if ch_val not in sdict: return # Channel might not exist in this plot
        
        freq_i_data, psd_i_data, psd_q_data, psd_m_data, _, _, _ = payload
        freq_i = np.asarray(freq_i_data, dtype=float); psd_i = np.asarray(psd_i_data, dtype=float)
        psd_q = np.asarray(psd_q_data, dtype=float); psd_m = np.asarray(psd_m_data, dtype=float)
        
        if sdict[ch_val]["I"].isVisible(): sdict[ch_val]["I"].setData(freq_i, psd_i)
        if sdict[ch_val]["Q"].isVisible(): sdict[ch_val]["Q"].setData(freq_i, psd_q)
        if "Mag" in sdict[ch_val] and sdict[ch_val]["Mag"].isVisible(): sdict[ch_val]["Mag"].setData(freq_i, psd_m)

    def _update_dsb_curve(self, row: int, ch_val: int, payload):
        """
        Update Dual-Sideband (DSB) PSD plot curve.

        Args:
            row (int): Row index of the plot.
            ch_val (int): Channel ID.
            payload: Tuple containing (freq_dsb, psd_dsb).
        """
        if "D" not in self.curves[row]: return # DSB plot might not exist for this row
        ddict = self.curves[row]["D"]
        if ch_val not in ddict: return # Channel might not exist in this plot
        
        freq_dsb_data, psd_dsb_data = payload
        freq_dsb = np.asarray(freq_dsb_data, dtype=float); psd_dsb = np.asarray(psd_dsb_data, dtype=float)
        if ddict[ch_val]["Cmplx"].isVisible(): ddict[ch_val]["Cmplx"].setData(freq_dsb, psd_dsb)

    def closeEvent(self, event: QtCore.QEvent):
        """Handle the main window close event. Stops timers and worker threads."""
        self.timer.stop(); self.receiver.stop(); self.receiver.wait()
        # Stop any active network analysis tasks (QThread needs proper termination)
        for task_key in list(self.netanal_tasks.keys()):
            task = self.netanal_tasks[task_key]
            task.stop()  # Request interruption
            task.wait(2000)  # Wait up to 2 seconds for thread to finish
            self.netanal_tasks.pop(task_key, None)
        # Stop any active multisweep tasks (QThread now, needs proper termination)
        for task_key in list(self.multisweep_tasks.keys()):
            task = self.multisweep_tasks[task_key]
            task.stop()  # Request interruption
            task.wait(2000)  # Wait up to 2 seconds for thread to finish
            self.multisweep_tasks.pop(task_key, None)
        # Shutdown iPython kernel if active
        if self.kernel_manager and self.kernel_manager.has_kernel:
            try: self.kernel_manager.shutdown_kernel()
            except Exception as e: warnings.warn(f"Error shutting down iPython kernel: {e}", RuntimeWarning) # warnings from .utils
        # The super().closeEvent() call should be handled by the class that inherits this mixin
        # and also inherits from a QWidget (e.g., Periscope class itself).
        event.accept()

    def _add_interactive_console_dock(self):
        """Add the dock widget for the embedded iPython console (if qtconsole is available)."""
        # QTCONSOLE_AVAILABLE, Qt from .utils
        if not QTCONSOLE_AVAILABLE: return
        self.console_dock_widget = QtWidgets.QDockWidget("Interactive iPython Session", self)
        self.console_dock_widget.setObjectName("InteractiveSessionDock")
        self.console_dock_widget.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.console_dock_widget.setVisible(False) # Initially hidden
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.console_dock_widget)

    def _toggle_interactive_session(self):
        """
        Toggle the visibility and initialization of the embedded iPython console.
        Initializes the kernel and Jupyter widget on first toggle if not already done.
        """
        # QTCONSOLE_AVAILABLE, QtInProcessKernelManager, RichJupyterWidget, rfmux, load_awaitless_extension from .utils
        # traceback from .utils
        if not QTCONSOLE_AVAILABLE or self.crs is None: return # Dependencies not met
        if self.console_dock_widget is None: return # Dock widget not created

        if self.kernel_manager is None: # First time opening the console
            try:
                self.kernel_manager = QtInProcessKernelManager()
                self.kernel_manager.start_kernel()
                kernel = self.kernel_manager.kernel
                # Push relevant objects into the kernel's namespace
                kernel.shell.push({'crs': self.crs, 'rfmux': rfmux, 'periscope': self})
                
                self.jupyter_widget = RichJupyterWidget()
                self.jupyter_widget.kernel_client = self.kernel_manager.client()
                self.jupyter_widget.kernel_client.start_channels()
                
                try: # Attempt to load awaitless extension for better async interaction
                    load_awaitless_extension(ipython=kernel.shell)
                except Exception as e_awaitless:
                    warnings.warn(f"Could not load awaitless extension: {e_awaitless}", RuntimeWarning)
                    traceback.print_exc()
                
                self.console_dock_widget.setWidget(self.jupyter_widget)
                self._update_console_style(self.dark_mode) # Apply initial style
                # Set a more specific stylesheet for prompts and background
                style_sheet = (".in-prompt { color: #00FF00 !important; } .out-prompt { color: #00DD00 !important; } body { background-color: #1C1C1C; color: #DDDDDD; }" 
                                if self.dark_mode else 
                                ".in-prompt { color: #008800 !important; } .out-prompt { color: #006600 !important; } body { background-color: #FFFFFF; color: #000000; }")
                self.jupyter_widget._control.document().setDefaultStyleSheet(style_sheet)
                self.jupyter_widget.setFocus()
            except Exception as e: # Handle errors during console initialization
                QtWidgets.QMessageBox.critical(self, "Error Initializing Console", f"Could not initialize iPython console: {e}")
                traceback.print_exc()
                if self.kernel_manager and self.kernel_manager.has_kernel: self.kernel_manager.shutdown_kernel()
                self.kernel_manager = None; self.jupyter_widget = None
                self.console_dock_widget.setVisible(False); return
        
        # Toggle visibility of the console dock widget
        is_visible = self.console_dock_widget.isVisible()
        self.console_dock_widget.setVisible(not is_visible)
        if not is_visible and self.jupyter_widget: self.jupyter_widget.setFocus() # Focus on console when shown

    def _start_multisweep_analysis(self, params: dict):
        """
        Start a new multisweep analysis.

        Creates a `MultisweepWindow` and a `MultisweepTask`.
        Connects signals for progress, data updates, and completion.

        Args:
            params (dict): Parameters for the multisweep analysis, typically
                            from a configuration dialog.
        """
        # MultisweepWindow from .ui, MultisweepTask from .tasks, sys, traceback from .utils
        try:
            if self.crs is None: QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available for multisweep."); return
            window_id = f"multisweep_window_{self.multisweep_window_count}"; self.multisweep_window_count += 1
            target_module = params.get('module')
            if target_module is None: QtWidgets.QMessageBox.critical(self, "Error", "Target module not specified for multisweep."); return
            
            dac_scales_for_window = self.dac_scales if hasattr(self, 'dac_scales') else {}
            window = MultisweepWindow(parent=self, target_module=target_module, initial_params=params.copy(), 
                                     dac_scales=dac_scales_for_window, dark_mode=self.dark_mode)
            self.multisweep_windows[window_id] = {'window': window, 'params': params.copy()}
            
            # Connect df_calibration_ready signal if the method exists
            if hasattr(window, 'df_calibration_ready') and hasattr(self, '_handle_df_calibration_ready'):
                window.df_calibration_ready.connect(self._handle_df_calibration_ready)
            
            # Disconnect any previous signal connections to avoid multiple calls
            try:
                self.multisweep_signals.progress.disconnect()
                self.multisweep_signals.data_update.disconnect()
                self.multisweep_signals.completed_iteration.disconnect()
                self.multisweep_signals.all_completed.disconnect()
                self.multisweep_signals.error.disconnect()
            except TypeError: 
                pass # Raised if signals were not previously connected

            # Connect signals from the MultisweepTask to the new window's slots
            self.multisweep_signals.progress.connect(window.update_progress,
                                                   QtCore.Qt.ConnectionType.QueuedConnection)
            self.multisweep_signals.starting_iteration.connect(window.handle_starting_iteration,
                                                             QtCore.Qt.ConnectionType.QueuedConnection)
            self.multisweep_signals.data_update.connect(window.update_data,
                                                      QtCore.Qt.ConnectionType.QueuedConnection)
            self.multisweep_signals.completed_iteration.connect(
                lambda module, iteration, amplitude, direction: window.completed_amplitude_sweep(module, amplitude),
                QtCore.Qt.ConnectionType.QueuedConnection)
            self.multisweep_signals.all_completed.connect(window.all_sweeps_completed,
                                                        QtCore.Qt.ConnectionType.QueuedConnection)
            self.multisweep_signals.error.connect(window.handle_error,
                                                QtCore.Qt.ConnectionType.QueuedConnection)
            self.multisweep_signals.fitting_progress.connect(window.handle_fitting_progress,
                                                            QtCore.Qt.ConnectionType.QueuedConnection)
            
            # Pass the window instance to the task (now starts automatically since it's a QThread)
            task = MultisweepTask(crs=self.crs, params=params, signals=self.multisweep_signals, window=window)
            task_key = f"{window_id}_module_{target_module}"
            self.multisweep_tasks[task_key] = task
            task.start()  # Start the QThread directly
            window.show()
        except Exception as e:
            error_msg = f"Error starting multisweep analysis: {type(e).__name__}: {str(e)}"
            print(error_msg, file=sys.stderr); traceback.print_exc(file=sys.stderr)
            QtWidgets.QMessageBox.critical(self, "Multisweep Error", error_msg)
        
    
    def _load_multisweep_analysis(self, load_params: dict):
        """
        Start a new multisweep analysis.

        Creates a `MultisweepWindow` and a `MultisweepTask`.
        Connects signals for progress, data updates, and completion.

        Args:
            params (dict): Parameters for the multisweep analysis, typically
                            from a configuration dialog.
        """
        # MultisweepWindow from .ui, MultisweepTask from .tasks, sys, traceback from .utils
        try:
            if self.crs is None: QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available for multisweep. Make sure your board is correctly setup."); return
            window_id = f"multisweep_window_{self.multisweep_window_count}"; self.multisweep_window_count += 1
            params = load_params['initial_parameters']
            target_module = params.get('module')
            if target_module is None: QtWidgets.QMessageBox.critical(self, "Error", "Target module not specified for multisweep. Please check your multisweep file."); return
            
            if hasattr(self, 'dac_scales'): 
                dac_scales_for_window = self.dac_scales
            else:
                QtWidgets.QMessageBox.critical(self, "Error", "Unable to compute dac scales for the board.")
                return

            dac_scale_for_mod = load_params['dac_scales_used'][target_module]
            dac_scale_for_board = dac_scales_for_window[target_module]

            if dac_scale_for_mod != dac_scale_for_board:
                QtWidgets.QMessageBox.warning(self, "Warning", f"Mismatch in Dac scales File Value : {dac_scale_for_mod}, Board Value : {dac_scale_for_board}. Exact data won't be reproduced.")
                
            window = MultisweepWindow(parent=self, target_module=target_module, initial_params=params.copy(), 
                                     dac_scales=dac_scales_for_window, dark_mode=self.dark_mode)
            self.multisweep_windows[window_id] = {'window': window, 'params': params.copy()}
            
            # Connect df_calibration_ready signal if the method exists
            if hasattr(window, 'df_calibration_ready') and hasattr(self, '_handle_df_calibration_ready'):
                window.df_calibration_ready.connect(self._handle_df_calibration_ready)

            window._hide_progress_bars()
            iteration_params = load_params['results_by_iteration']

            span_hz = params['span_hz']
            reso_frequencies = params['resonance_frequencies']

            nco_freq = ((min(reso_frequencies)-span_hz/2) + (max(reso_frequencies)+span_hz/2))/2

            crs = self.crs
            asyncio.run(crs.set_nco_frequency(nco_freq, module=target_module)) #### Setting up the nco frequency ######

            for i in range(len(iteration_params)):
                amplitude = iteration_params[i]['amplitude']
                direction = iteration_params[i]['direction']
                data = iteration_params[i]['data']
                window.update_data(target_module, i, amplitude, direction, data, None)
            window.show()
        except Exception as e:
            error_msg = f"Error starting multisweep analysis: {type(e).__name__}: {str(e)}"
            print(error_msg, file=sys.stderr); traceback.print_exc(file=sys.stderr)
            QtWidgets.QMessageBox.critical(self, "Multisweep Error", error_msg)

    def _start_multisweep_analysis_for_window(self, window_instance: 'MultisweepWindow', params: dict):
        """
        Re-run a multisweep analysis for an existing MultisweepWindow.

        Stops any existing task for the window, updates parameters, and starts a new task.

        Args:
            window_instance (MultisweepWindow): The window instance to re-run the analysis for.
            params (dict): The new parameters for the multisweep analysis.
        """
        # MultisweepWindow from .ui, MultisweepTask from .tasks
        window_id = None
        for w_id, data in self.multisweep_windows.items():
            if data['window'] == window_instance: window_id = w_id; break
        if not window_id: QtWidgets.QMessageBox.critical(window_instance, "Error", "Could not find associated window to re-run multisweep."); return
        
        target_module = params.get('module')
        if target_module is None: QtWidgets.QMessageBox.critical(window_instance, "Error", "Target module not specified for multisweep re-run."); return
        
        old_task_key = f"{window_id}_module_{target_module}"
        if old_task_key in self.multisweep_tasks: # Stop and remove old task if it exists
            old_task = self.multisweep_tasks.pop(old_task_key); old_task.stop()
            
        self.multisweep_windows[window_id]['params'] = params.copy() # Update stored params
        # Pass the window_instance to the task (now starts automatically since it's a QThread)
        
        ### This reconnects to signal ####
        try:
            self.multisweep_signals.progress.disconnect()
            self.multisweep_signals.data_update.disconnect()
            self.multisweep_signals.completed_iteration.disconnect()
            self.multisweep_signals.all_completed.disconnect()
            self.multisweep_signals.error.disconnect()
        except TypeError: 
            pass # Raised if signals were not previously connected

        # Connect signals from the MultisweepTask to the new window's slots
        self.multisweep_signals.progress.connect(window_instance.update_progress,
                                               QtCore.Qt.ConnectionType.QueuedConnection)
        self.multisweep_signals.starting_iteration.connect(window_instance.handle_starting_iteration,
                                                         QtCore.Qt.ConnectionType.QueuedConnection)
        self.multisweep_signals.data_update.connect(window_instance.update_data,
                                                  QtCore.Qt.ConnectionType.QueuedConnection)
        self.multisweep_signals.completed_iteration.connect(
            lambda module, iteration, amplitude, direction: window_instance.completed_amplitude_sweep(module, amplitude),
            QtCore.Qt.ConnectionType.QueuedConnection)
        self.multisweep_signals.all_completed.connect(window_instance.all_sweeps_completed,
                                                    QtCore.Qt.ConnectionType.QueuedConnection)
        self.multisweep_signals.error.connect(window_instance.handle_error,
                                            QtCore.Qt.ConnectionType.QueuedConnection)
        self.multisweep_signals.fitting_progress.connect(window_instance.handle_fitting_progress,
                                                        QtCore.Qt.ConnectionType.QueuedConnection)

        
        task = MultisweepTask(crs=self.crs, params=params, signals=self.multisweep_signals, window=window_instance)
        self.multisweep_tasks[old_task_key] = task
        task.start()  # Start the QThread directly

    def stop_multisweep_task_for_window(self, window_instance: 'MultisweepWindow'):
        """
        Stop an active multisweep task associated with a specific window.

        Args:
            window_instance (MultisweepWindow): The window whose task should be stopped.
        """
        # MultisweepWindow from .ui
        window_id = None; target_module = None
        for w_id, data in list(self.multisweep_windows.items()): # Iterate over a copy for safe removal
            if data['window'] == window_instance:
                window_id = w_id; target_module = data['params'].get('module'); break
        
        if window_id and target_module:
            task_key = f"{window_id}_module_{target_module}"
            if task_key in self.multisweep_tasks:
                task = self.multisweep_tasks.pop(task_key)
                task.stop()  # Request interruption
                task.wait(2000)  # Wait up to 2 seconds for thread to finish
            self.multisweep_windows.pop(window_id, None) # Remove window tracking

    def _update_console_style(self, dark_mode_enabled: bool):
        """
        Update the style of the embedded iPython console based on the theme.

        Args:
            dark_mode_enabled (bool): True if dark mode is active, False otherwise.
        """
        # QTCONSOLE_AVAILABLE from .utils
        if self.jupyter_widget and QTCONSOLE_AVAILABLE:
            self.jupyter_widget.syntax_style = 'monokai' if dark_mode_enabled else 'default'
            # Define stylesheets for dark and light modes
            style_sheet = ("QWidget { background-color: #1C1C1C; color: #DDDDDD; } .in-prompt { color: #00FF00 !important; } .out-prompt { color: #00DD00 !important; } QPlainTextEdit { background-color: #1C1C1C; color: #DDDDDD; }"
                            if dark_mode_enabled else
                            "QWidget { background-color: #FFFFFF; color: #000000; } .in-prompt { color: #008800 !important; } .out-prompt { color: #006600 !important; } QPlainTextEdit { background-color: #FFFFFF; color: #000000; }")
            self.jupyter_widget.setStyleSheet(style_sheet)
            self.jupyter_widget.update(); self.jupyter_widget.repaint() # Force repaint

    def _init_sim_speed_tracking(self):
        """Initialize simulation speed tracking for mock mode."""
        self.sim_time_history = []  # List of (real_time, sim_time) tuples
        self.sim_speed_update_counter = 0
        self.last_sim_speed_update_time = time.time()
        
    def _update_sim_time_tracking(self, sim_time: float):
        """
        Track simulation time for speed calculation.
        
        Args:
            sim_time: Current simulation time in seconds
        """
        # Update every 100 packets as requested
        self.sim_speed_update_counter += 1
        if self.sim_speed_update_counter >= 100:
            real_time = time.time()
            self.sim_time_history.append((real_time, sim_time))
            
            # Keep only last 10 samples for rolling average
            if len(self.sim_time_history) > 10:
                self.sim_time_history.pop(0)
            
            self.sim_speed_update_counter = 0
    
    def _calculate_simulation_speed(self) -> float | None:
        """
        Calculate the simulation speed relative to real-time.
        
        Returns:
            float: Speed factor (1.0 = real-time, >1.0 = faster, <1.0 = slower)
            None: If not enough data to calculate
        """
        if len(self.sim_time_history) < 2:
            return None
        
        # Calculate speed from first to last sample
        first_real, first_sim = self.sim_time_history[0]
        last_real, last_sim = self.sim_time_history[-1]
        
        real_elapsed = last_real - first_real
        sim_elapsed = last_sim - first_sim
        
        # Avoid division by zero
        if real_elapsed <= 0 or sim_elapsed <= 0:
            return None
        
        # sim_speed = delta_sim_time / delta_real_time
        return sim_elapsed / real_elapsed
