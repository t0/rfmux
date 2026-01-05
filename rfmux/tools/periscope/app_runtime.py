"""Runtime mixin for the Periscope application."""

from .utils import *
from .tasks import *
from .ui import *
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, AsyncMock
from contextlib import contextmanager
import inspect
import ast
from .extract_params import ParamKeyExtractor
from PyQt6 import sip
from PyQt6.QtCore import QUrl
import numpy as np

class PeriscopeRuntime:
    """Mixin providing runtime methods for :class:`Periscope`."""
    
    def _convert_iq_data(self, rawI: np.ndarray, rawQ: np.ndarray, ch_val: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert raw I/Q data based on current unit mode and calibration availability.
        
        This method consolidates the unit conversion logic that appears 5 times in the codebase.
        It fixes a bug where 3 locations didn't fall back to real_units when df calibration
        was missing (they returned raw counts instead).
        
        Behavior:
        - df mode with calibration: apply df calibration to get freq shift/dissipation
        - df mode without calibration: fall back to real_units check (BUG FIX)
        - real_units mode: convert to volts
        - counts mode: return raw counts
        
        Args:
            rawI: Raw I component data
            rawQ: Raw Q component data  
            ch_val: Channel ID (for calibration lookup)
            
        Returns:
            Tuple of (I_data, Q_data) converted according to unit_mode and calibration
        """
        if self.unit_mode == "df" and hasattr(self, 'df_calibrations') and self.module in self.df_calibrations:
            df_cal = self.df_calibrations[self.module].get(ch_val)
            if df_cal is not None:
                # Apply df calibration
                iq_volts = convert_roc_to_volts(rawI) + 1j * convert_roc_to_volts(rawQ)
                df_complex = iq_volts * df_cal
                return df_complex.real, df_complex.imag  # Frequency shift (Hz), Dissipation (unitless)
            else:
                # No calibration - fall back to real_units check (matches PSD task behavior)
                return (convert_roc_to_volts(rawI), convert_roc_to_volts(rawQ)) if self.real_units else (rawI, rawQ)
        elif self.real_units:
            return convert_roc_to_volts(rawI), convert_roc_to_volts(rawQ)
        else:
            return rawI, rawQ
    
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
        """Propagate dark mode setting to all open child windows and panels."""
        # Update NetworkAnalysis panels (in docks)
        if hasattr(self, 'netanal_windows'):
            for window_data in self.netanal_windows.values():
                panel = window_data.get('window')  # 'window' key contains panel instance
                if panel and hasattr(panel, 'apply_theme'):
                    panel.apply_theme(self.dark_mode)
        
        # Update Multisweep panels (in docks)
        if hasattr(self, 'multisweep_windows'):
            for window_data in self.multisweep_windows.values():
                panel = window_data.get('window')  # 'window' key contains panel instance
                if panel and hasattr(panel, 'apply_theme'):
                    panel.apply_theme(self.dark_mode)

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
            try:
                seq, pkt = self.receiver.queue.get(block=False)
                self.pkt_cnt += 1
                if hasattr(pkt, 'fir_stage'):
                    self.actual_dec_stage = pkt.fir_stage
                t_rel = self._calculate_relative_timestamp(pkt)
                self._update_buffers(pkt, t_rel)
                
                # Track simulation time for speed calculation (mock mode only)
                if self.is_mock_mode and t_rel is not None:
                    self._update_sim_time_tracking(t_rel)
            except:
                self.receiver.queue.get_nowait()  # pop the bad element
                continue

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
        # Convert 24-bit datapath to 16-bit ADC scale
        samples = pkt.samples / 256

        for ch_val in self.all_chs: # Renamed ch
            if len(samples) <= ch_val-1:
                continue # don't plot channels that aren't streamed

            sample = samples[ch_val-1]

            self.buf[ch_val]["I"].add(sample.real)
            self.buf[ch_val]["Q"].add(sample.imag)
            self.buf[ch_val]["M"].add(np.abs(sample))
            self.tbuf[ch_val].add(t_rel)

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
        
        # Use extracted helper method for I/Q conversion
        I_data, Q_data = self._convert_iq_data(rawI, rawQ, ch_val)
        
        # Handle magnitude separately (df calibration affects it differently)
        if self.unit_mode == "df" and hasattr(self, 'df_calibrations') and self.module in self.df_calibrations:
            df_cal = self.df_calibrations[self.module].get(ch_val)
            if df_cal is not None:
                # Magnitude in df space
                M_data = np.abs((convert_roc_to_volts(rawI) + 1j * convert_roc_to_volts(rawQ)) * df_cal)
            else:
                # No calibration - use appropriate units
                M_data = convert_roc_to_volts(rawM) if self.real_units else rawM
        elif self.real_units:
            M_data = convert_roc_to_volts(rawM)
        else:
            M_data = rawM
        
        # Update Time-Domain (TOD) plots
        if "T" in rowCurves and ch_val in rowCurves["T"]:
            cset = rowCurves["T"][ch_val]
            try:
                if not sip.isdeleted(cset["I"]) and cset["I"].isVisible(): 
                    cset["I"].setData(tarr, I_data)
                if not sip.isdeleted(cset["Q"]) and cset["Q"].isVisible(): 
                    cset["Q"].setData(tarr, Q_data)
                if "Mag" in cset and not sip.isdeleted(cset["Mag"]) and cset["Mag"].isVisible(): 
                    cset["Mag"].setData(tarr, M_data)
            except RuntimeError:
                # Qt object was deleted, skip this update
                pass
        
        # Update FFT plots
        if "F" in rowCurves and ch_val in rowCurves["F"]:
            cset = rowCurves["F"][ch_val]
            try:
                if not sip.isdeleted(cset["I"]) and cset["I"].isVisible(): 
                    cset["I"].setData(tarr[1:], I_data[1:], fftMode=True)
                if not sip.isdeleted(cset["Q"]) and cset["Q"].isVisible(): 
                    cset["Q"].setData(tarr[1:], Q_data[1:], fftMode=True)
                if "Mag" in cset and not sip.isdeleted(cset["Mag"]) and cset["Mag"].isVisible(): 
                    cset["Mag"].setData(tarr[1:], M_data[1:], fftMode=True)
            except RuntimeError:
                # Qt object was deleted, skip this update
                pass

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
            
            # Use extracted helper method (fixes bug where real_units wasn't checked as fallback)
            I_data, Q_data = self._convert_iq_data(rawI, rawQ, c_val)
            
            self.iq_workers[row_i] = True
            task = IQTask(row_i, c_val, I_data, Q_data, self.dot_px, mode_key, self.iq_signals)
            self.pool.start(task)
        else: # Multi-channel IQ plot (concatenate converted data)
            all_I_data = []
            all_Q_data = []
            
            for ch in group:
                rawI = self.buf[ch]["I"].data()
                rawQ = self.buf[ch]["Q"].data()
                
                # Use extracted helper method (fixes bug where real_units wasn't checked as fallback)
                I_data, Q_data = self._convert_iq_data(rawI, rawQ, ch)
                    
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
                    
                    # Use extracted helper method (consolidation - this already had correct logic)
                    I_data, Q_data = self._convert_iq_data(rawI, rawQ, ch_val)
                    
                    self.psd_workers[row_i]["S"][ch_val] = True
                    task = PSDTask(row_i, ch_val, I_data, Q_data, "SSB", self.dec_stage, self.real_units, self.psd_absolute, self.spin_segments.value(), self.psd_signals, self.cb_exp_binning.isChecked(), self.spin_bins.value())
                    self.pool.start(task)
        
        # Dispatch Dual-Sideband (DSB) PSD tasks
        if "D" in self.curves[row_i]:
            for ch_val in group: # Renamed ch
                if not self.psd_workers[row_i]["D"].get(ch_val, False): # Check if worker already active
                    rawI = self.buf[ch_val]["I"].data(); rawQ = self.buf[ch_val]["Q"].data()
                    
                    # Use extracted helper method (consolidation - this already had correct logic)
                    I_data, Q_data = self._convert_iq_data(rawI, rawQ, ch_val)
                    
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
            # Check for zero denominator to avoid division by zero
            total_packets = drop_lastsec + receive_lastsec
            if total_packets > 0:
                percent = (drop_lastsec / total_packets) * 100
            else:
                percent = 0.0  # No packets = no loss
            
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
        # Shutdown Jupyter notebook server if running
        if hasattr(self, 'notebook_dock') and self.notebook_dock is not None:
            if not sip.isdeleted(self.notebook_dock):
                panel = self.notebook_dock.widget()
                if panel and hasattr(panel, 'shutdown'):
                    panel.shutdown()
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

        Creates a `MultisweepPanel` wrapped in a QDockWidget and a `MultisweepTask`.
        Connects signals for progress, data updates, and completion.

        Args:
            params (dict): Parameters for the multisweep analysis, typically
                            from a configuration dialog.
        """
        # MultisweepPanel from .ui, MultisweepTask from .tasks, sys, traceback from .utils
        try:
            if self.crs is None: QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available for multisweep."); return
            window_id = f"multisweep_{self.multisweep_window_count}"; self.multisweep_window_count += 1
            target_module = params.get('module')
            if target_module is None: QtWidgets.QMessageBox.critical(self, "Error", "Target module not specified for multisweep."); return
            
            # Create panel
            dac_scales_for_panel = self.dac_scales if hasattr(self, 'dac_scales') else {}
            panel = MultisweepPanel(parent=self, target_module=target_module, initial_params=params.copy(), 
                                   dac_scales=dac_scales_for_panel, dark_mode=self.dark_mode)
            
            # Wrap in dock
            dock_title = f"Multisweep #{self.multisweep_window_count}"
            dock = self.dock_manager.create_dock(panel, dock_title, window_id)
            
            # Store panel reference
            self.multisweep_windows[window_id] = {'window': panel, 'dock': dock, 'params': params.copy()}
            
            # Connect df_calibration_ready signal if the method exists
            if hasattr(panel, 'df_calibration_ready') and hasattr(self, '_handle_df_calibration_ready'):
                panel.df_calibration_ready.connect(self._handle_df_calibration_ready)
            
            # Connect data_ready signal for session auto-export
            if hasattr(panel, 'data_ready') and hasattr(self, 'session_manager'):
                panel.data_ready.connect(self.session_manager.handle_data_ready)
            
            # Disconnect any previous signal connections to avoid multiple calls
            try:
                self.multisweep_signals.progress.disconnect()
                self.multisweep_signals.data_update.disconnect()
                self.multisweep_signals.completed_iteration.disconnect()
                self.multisweep_signals.all_completed.disconnect()
                self.multisweep_signals.error.disconnect()
            except TypeError: 
                pass # Raised if signals were not previously connected

            # Connect signals from the MultisweepTask to the new panel's slots
            self.multisweep_signals.progress.connect(panel.update_progress,
                                                   QtCore.Qt.ConnectionType.QueuedConnection)
            self.multisweep_signals.starting_iteration.connect(panel.handle_starting_iteration,
                                                             QtCore.Qt.ConnectionType.QueuedConnection)
            self.multisweep_signals.data_update.connect(panel.update_data,
                                                      QtCore.Qt.ConnectionType.QueuedConnection)
            self.multisweep_signals.completed_iteration.connect(
                lambda module, iteration, amplitude, direction: panel.completed_amplitude_sweep(module, amplitude),
                QtCore.Qt.ConnectionType.QueuedConnection)
            self.multisweep_signals.all_completed.connect(panel.all_sweeps_completed,
                                                        QtCore.Qt.ConnectionType.QueuedConnection)
            self.multisweep_signals.error.connect(panel.handle_error,
                                                QtCore.Qt.ConnectionType.QueuedConnection)
            self.multisweep_signals.fitting_progress.connect(panel.handle_fitting_progress,
                                                            QtCore.Qt.ConnectionType.QueuedConnection)
            
            # Create and start the task
            task = MultisweepTask(crs=self.crs, params=params, signals=self.multisweep_signals, window=panel)
            task_key = f"{window_id}_module_{target_module}"
            self.multisweep_tasks[task_key] = task
            task.start()  # Start the QThread directly
            
            # Tabify with Main dock by default
            main_dock = self.dock_manager.get_dock("main_plots")
            if main_dock:
                self.tabifyDockWidget(main_dock, dock)
            
            # Show the dock and activate it
            dock.show()
            dock.raise_()
        except Exception as e:
            error_msg = f"Error starting multisweep analysis: {type(e).__name__}: {str(e)}"
            print(error_msg, file=sys.stderr); traceback.print_exc(file=sys.stderr)
            QtWidgets.QMessageBox.critical(self, "Multisweep Error", error_msg)
            raise
    
    def fetch_dac_scales_blocking(self) -> dict:
        dac_scales = None
    
        fetcher = DACScaleFetcher(self.crs)
        loop = QtCore.QEventLoop()
    
        def on_ready(scales):
            nonlocal dac_scales
            dac_scales = scales
            loop.quit()
    
        fetcher.dac_scales_ready.connect(on_ready)
        fetcher.finished.connect(fetcher.deleteLater)
    
        fetcher.start()
        loop.exec_()   # waits until on_ready() calls loop.quit()
    
        return dac_scales    
    
    def _create_multisweep_panel_from_loaded_data(self, load_params: dict, source_type: str = "multisweep") -> tuple:
        """
        Create and display a MultisweepPanel from loaded data.
        
        This unified helper method is used by both _load_multisweep_analysis and 
        _set_and_plot_bias to eliminate code duplication.
        
        Args:
            load_params: Loaded data dictionary containing 'initial_parameters', 
                        'results_by_iteration', 'dac_scales_used', etc.
            source_type: "multisweep", "bias", or "noise" - affects naming and panel behavior
            
        Returns:
            tuple: (panel, dock, window_id, target_module) or (None, None, None, None) on error
        """
        
        try:
            # Allow loading without CRS in offline mode
            if self.crs is None and self.host != "OFFLINE": 
                QtWidgets.QMessageBox.critical(self, "Error", "CRS object not available for multisweep. Make sure your board is correctly setup.")
                return None, None, None, None
                
            window_id = f"multisweep_window_{self.multisweep_window_count}"
            self.multisweep_window_count += 1
            
            params = load_params['initial_parameters']
            target_module = params.get('module')
            if target_module is None: 
                QtWidgets.QMessageBox.critical(self, "Error", "Target module not specified. Please check your file.")
                return None, None, None, None

            try: 
                dac_scales_for_panel = self.fetch_dac_scales_blocking() #### Gets the dac scale directly from the board #####
            except:
                QtWidgets.QMessageBox.critical(self, "Error", "Unable to compute dac scales for the board.")
                return

            dac_scale_for_mod = load_params['dac_scales_used'][target_module]
            dac_scale_for_board = dac_scales_for_panel[target_module]

            if dac_scale_for_mod != dac_scale_for_board:
                QtWidgets.QMessageBox.warning(self, "Warning", f"Mismatch in Dac scales File Value : {dac_scale_for_mod}, Board Value : {dac_scale_for_board}. Exact data won't be reproduced.")
            
            # Check if noise data exists in the loaded file
            has_noise_data = 'noise_data' in load_params and load_params['noise_data'] is not None
            
            # For bias source type, also check for bias_kids_output
            has_bias_data = 'bias_kids_output' in load_params and load_params['bias_kids_output'] is not None
            loaded_bias_flag = has_noise_data or (source_type == "bias" and has_bias_data)
                
            # Create panel
            panel = MultisweepPanel(parent=self, target_module=target_module, initial_params=params.copy(), 
                                   dac_scales=dac_scales_for_panel, dark_mode=self.dark_mode, 
                                   loaded_bias=loaded_bias_flag, is_loaded_data=True)
            
            # Load noise spectrum data if it exists
            if has_noise_data:
                panel.spectrum_noise_data = load_params['noise_data']
                panel.noise_spectrum_btn.setEnabled(True)
            
            # MultisweepPanel dock is always named "Multisweep" regardless of source type
            # The source_type affects panel behavior, not the dock title
            dock_title = f"Multisweep #{self.multisweep_window_count} (Loaded)"
            
            # Wrap in dock
            dock = self.dock_manager.create_dock(panel, dock_title, window_id)
            
            self.multisweep_windows[window_id] = {'window': panel, 'dock': dock, 'params': params.copy()}
            
            # Connect df_calibration_ready signal if the method exists
            if hasattr(panel, 'df_calibration_ready') and hasattr(self, '_handle_df_calibration_ready'):
                panel.df_calibration_ready.connect(self._handle_df_calibration_ready)
            
            # Connect data_ready signal for session auto-export
            if hasattr(panel, 'data_ready') and hasattr(self, 'session_manager'):
                panel.data_ready.connect(self.session_manager.handle_data_ready)

            panel._hide_progress_bars()
            
            # Set NCO frequency based on resonance frequencies
            iteration_params = load_params.get('results_by_iteration', [])
            reso_frequencies = params.get('resonance_frequencies', [])
            
            if reso_frequencies:
                span_hz = params.get('span_hz', 0)
                nco_freq = ((min(reso_frequencies) - span_hz/2) + (max(reso_frequencies) + span_hz/2)) / 2

                # Only set NCO frequency if CRS is available (skip in offline mode)
                if self.crs is not None:
                    asyncio.run(self.crs.set_nco_frequency(nco_freq, module=target_module))
                else:
                    print(f"[Offline] Skipping NCO frequency setup (would set to {nco_freq/1e9:.6f} GHz)")

            # Load iteration data into panel
            for i in range(len(iteration_params)):
                amplitude = iteration_params[i]['amplitude']
                direction = iteration_params[i]['direction']
                data = iteration_params[i]['data']
                panel.update_data(target_module, i, amplitude, direction, data, None)
            
            # Extract and load df_calibrations if bias_kids_output exists
            if has_bias_data:
                bias_output = load_params['bias_kids_output']
                df_calibrations = {}
                for det_idx, det_data in bias_output.items():
                    if 'df_calibration' in det_data:
                        df_calibrations[det_idx] = det_data['df_calibration']
                
                # Load calibrations into main window
                if df_calibrations and hasattr(self, '_handle_df_calibration_ready'):
                    self._handle_df_calibration_ready(target_module, df_calibrations)
                    print(f"[Session] Loaded df calibrations for {len(df_calibrations)} detectors from session file")
            
            # Tabify with Main dock by default
            main_dock = self.dock_manager.get_dock("main_plots")
            if main_dock:
                self.tabifyDockWidget(main_dock, dock)
            
            # Show the dock and activate it
            dock.show()
            dock.raise_()
            
            return panel, dock, window_id, target_module
            
        except Exception as e:
            error_msg = f"Error creating multisweep panel: {type(e).__name__}: {str(e)}"
            print(error_msg, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            QtWidgets.QMessageBox.critical(self, "Error", error_msg)
            return None, None, None, None

    def _load_multisweep_analysis(self, load_params: dict):
        """
        Load multisweep analysis data from file and display in a docked panel.

        Args:
            load_params (dict): Loaded data dictionary from file.
        """
        # Use the unified helper method
        panel, dock, window_id, target_module = self._create_multisweep_panel_from_loaded_data(
            load_params, source_type="multisweep"
        )
        
        if panel is None:
            return  # Error already displayed by helper
        
        # Auto-launch detector digest panel by programmatically triggering the existing
        # double-click handler which already has all the logic for creating digest panels
        iteration_params = load_params.get('results_by_iteration', [])
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
                        
                        # Re-raise the multisweep dock to keep focus on it
                        multisweep_dock = self.dock_manager.find_dock_for_widget(panel)
                        if multisweep_dock:
                            multisweep_dock.raise_()

    def _start_multisweep_analysis_for_window(self, window_instance: 'MultisweepPanel', params: dict):
        """
        Re-run a multisweep analysis for an existing MultisweepPanel.

        Stops any existing task for the panel, updates parameters, and starts a new task.

        Args:
            window_instance (MultisweepPanel): The panel instance to re-run the analysis for.
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

        # Connect data_ready signal for session auto-export
        if hasattr(window_instance, 'data_ready') and hasattr(self, 'session_manager'):
            window_instance.data_ready.connect(self.session_manager.handle_data_ready)
        
        task = MultisweepTask(crs=self.crs, params=params, signals=self.multisweep_signals, window=window_instance)
        self.multisweep_tasks[old_task_key] = task
        task.start()  # Start the QThread directly

    def stop_multisweep_task_for_window(self, window_instance: 'MultisweepPanel'):
        """
        Stop an active multisweep task associated with a specific panel.

        Args:
            window_instance (MultisweepPanel): The panel whose task should be stopped.
        """
        # MultisweepPanel from .ui
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

    def _toggle_notebook_panel(self, notebook_dir: str | None = None, open_file: str | None = None):
        """
        Toggle the Jupyter notebook panel.
        
        Creates a new NotebookPanel on first use, or shows/hides the existing
        dock widget on subsequent calls. The notebook directory is always set to the
        active session folder. If no session is active, the user must start one first.
        
        The panel launches Jupyter Lab in the system browser (not embedded).
        Uses a singleton server pattern - only one Jupyter server is ever running.
        
        Args:
            notebook_dir: Optional directory for notebooks. If None, uses session
                         folder (if active). Requires an active session.
            open_file: Optional path to a notebook file to open after panel starts.
        """
        try:
            from .notebook_panel import NotebookPanel, JupyterServerManager
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Import Error",
                f"Failed to import notebook_panel:\n{e}"
            )
            return
        
        # Check if notebook dock already exists and is still valid
        if hasattr(self, 'notebook_dock') and self.notebook_dock is not None:
            # Check if the dock widget still exists in Qt
            if not sip.isdeleted(self.notebook_dock):
                # Panel already exists and is valid
                panel = self.notebook_dock.widget()
                
                # If a specific file is requested, open it
                if open_file and panel and hasattr(panel, 'open_notebook'):
                    panel.open_notebook(open_file)
                
                # Show and raise the dock
                self.notebook_dock.setVisible(True)
                self.notebook_dock.raise_()
                return
            else:
                # Dock was deleted - clear reference but keep server
                self.notebook_dock = None
                # Fall through to recreate the panel
        
        # Determine notebook directory - requires an active session
        if notebook_dir is None:
            if hasattr(self, 'session_manager') and self.session_manager.is_active:
                notebook_dir = str(self.session_manager.session_path)
            else:
                # No active session - prompt user to start one
                msg = QtWidgets.QMessageBox(self)
                msg.setWindowTitle("Session Required")
                msg.setText("No session is active.")
                msg.setInformativeText(
                    "Jupyter notebooks require an active session so that notebooks "
                    "are saved alongside your analysis data.\n\n"
                    "Please start or load a session first using:\n"
                    "Session → New Session or Session → Load Session"
                )
                msg.setIcon(QtWidgets.QMessageBox.Icon.Information)
                msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
                msg.exec()
                return
        
        # Initialize singleton server if it doesn't exist
        if not hasattr(self, 'jupyter_server') or self.jupyter_server is None:
            self.jupyter_server = JupyterServerManager(self)
        
        # Check if server is already running
        server_running = (self.jupyter_server.process is not None and 
                         self.jupyter_server.process.poll() is None)
        
        # Create panel with the session directory, using the singleton server
        # Skip initial notebook creation if we're opening a specific file
        panel = NotebookPanel(
            notebook_dir=notebook_dir, 
            parent=self, 
            server=self.jupyter_server,
            skip_initial_notebook=(open_file is not None)
        )
        
        # Create dock
        self.notebook_dock = self.dock_manager.create_dock(
            panel, "Jupyter Notebook", "notebook_panel"
        )
        
        # Mark this dock as protected (hide instead of close when X is clicked)
        self.dock_manager.protect_dock("notebook_panel")
        
        # Make dock non-closeable via the standard close button (extra safety)
        self.notebook_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            # Note: No DockWidgetClosable flag - prevents closing via X button on title bar
        )
        
        # Set up handler for opening specific files when requested
        def handle_server_ready(url):
            if open_file:
                # If a specific file was requested, open it with a delay
                QtCore.QTimer.singleShot(2000, lambda: panel.open_notebook(open_file))
            # Note: Notebook panel automatically creates and opens an initial notebook
            # via _create_and_open_initial_notebook() - no need to duplicate that here
        
        # Only connect signal and start server if not already running
        if not server_running:
            self.jupyter_server.server_ready.connect(handle_server_ready)
            # Start the Jupyter server
            panel.start()
        else:
            # Server already running - update UI and open file if requested
            panel._on_server_ready(self.jupyter_server.url or "")
            if open_file:
                QtCore.QTimer.singleShot(500, lambda: panel.open_notebook(open_file))
        
        # Tabify with main dock by default
        main_dock = self.dock_manager.get_dock("main_plots")
        if main_dock:
            self.tabifyDockWidget(main_dock, self.notebook_dock)
        
        # Show the dock and activate it
        self.notebook_dock.show()
        self.notebook_dock.raise_()

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
        """Initialize simulation speed tracking for mock mode.
        
        Uses a rolling window approach that samples at regular wall-clock intervals
        (not based on packet processing) to avoid false readings when packets queue up.
        """
        self._sim_speed_history = []  # List of (wall_clock_time, sim_time) tuples
        self._sim_speed_latest_sim_time = None  # Latest simulation time seen
        self._sim_speed_last_sample_time = None  # Wall clock time of last sample
        self._sim_speed_sample_interval = 1.0  # Sample every 1 second of wall clock time
        self._sim_speed_window_size = 10  # Keep 10 samples (10 second rolling window)
        
    def _update_sim_time_tracking(self, sim_time: float):
        """
        Track simulation time for speed calculation.
        
        Samples are taken at regular wall-clock intervals (not based on packet 
        processing timing) to ensure accurate speed measurement even when the 
        GUI is slow and packets queue up.
        
        Args:
            sim_time: Current simulation time in seconds (from packet timestamp)
        """
        # Always update the latest sim time
        self._sim_speed_latest_sim_time = sim_time
        
        current_wall_clock = time.time()
        
        # Take a sample at regular wall-clock intervals
        if self._sim_speed_last_sample_time is None:
            # First sample
            self._sim_speed_history.append((current_wall_clock, sim_time))
            self._sim_speed_last_sample_time = current_wall_clock
        elif (current_wall_clock - self._sim_speed_last_sample_time) >= self._sim_speed_sample_interval:
            # Time for a new sample
            self._sim_speed_history.append((current_wall_clock, sim_time))
            self._sim_speed_last_sample_time = current_wall_clock
            
            # Keep only the last N samples for rolling window
            if len(self._sim_speed_history) > self._sim_speed_window_size:
                self._sim_speed_history.pop(0)
    
    def _calculate_simulation_speed(self) -> float | None:
        """
        Calculate the simulation speed relative to real-time using a rolling window.
        
        Compares simulation time elapsed vs wall clock time elapsed over the
        sample window. This is immune to packet queuing effects because samples
        are taken at wall-clock intervals, not packet processing intervals.
        
        Returns:
            float: Speed factor (1.0 = real-time, >1.0 = faster, <1.0 = slower)
            None: If not enough data to calculate
        """
        if len(self._sim_speed_history) < 2:
            return None
        
        # Calculate speed from oldest to newest sample in the window
        first_wall_clock, first_sim = self._sim_speed_history[0]
        last_wall_clock, last_sim = self._sim_speed_history[-1]
        
        wall_clock_elapsed = last_wall_clock - first_wall_clock
        sim_elapsed = last_sim - first_sim
        
        # Avoid division by zero
        if wall_clock_elapsed <= 0 or sim_elapsed <= 0:
            return None

        # sim_speed = delta_sim_time / delta_real_time
        return sim_elapsed / wall_clock_elapsed


    def test_dialog_params(self):
        """
        Validate all dialog parameter dictionaries used by the UI mock context.
        """
        def _assert_param_keys(expected_dict: dict, module_path: str, class_name: str):
            extractor = ParamKeyExtractor(module_path, class_name)
            actual_keys = extractor.extract()
            expected_keys = set(expected_dict.keys())
    
            if actual_keys != expected_keys:
                missing = expected_keys - actual_keys
                unexpected = actual_keys - expected_keys
                raise AssertionError(
                    f"Mock parameter keys for {class_name} do not match test dialog. "
                    f"Missing from actual: {sorted(missing)} Actual Key is: {sorted(unexpected)}"
                )
        
        self.netanal_params = {
            "amps": [DEFAULT_AMPLITUDE],
            "module": None,
            "fmin": DEFAULT_MIN_FREQ,
            "fmax": DEFAULT_MAX_FREQ,
            "cable_length": DEFAULT_CABLE_LENGTH,
            "npoints": DEFAULT_NPOINTS,
            "nsamps": DEFAULT_NSAMPLES,
            "max_chans": DEFAULT_MAX_CHANNELS,
            "max_span": DEFAULT_MAX_SPAN,
            "clear_channels": True,
        }
        _assert_param_keys(
            self.netanal_params,
            "rfmux.tools.periscope.network_analysis_dialog",
            "NetworkAnalysisDialog",
        )
    
        self.find_params = {
            "expected_resonances": DEFAULT_EXPECTED_RESONANCES,
            "min_dip_depth_db": DEFAULT_MIN_DIP_DEPTH_DB,
            "min_Q": DEFAULT_MIN_Q,
            "max_Q": DEFAULT_MAX_Q,
            "min_resonance_separation_hz": DEFAULT_MIN_RESONANCE_SEPARATION_HZ,
            "data_exponent": DEFAULT_DATA_EXPONENT,
        }
        _assert_param_keys(
            self.find_params,
            "rfmux.tools.periscope.find_resonances_dialog",
            "FindResonancesDialog",
        )
    
        self.multisweep_params = {
            "amps": [MULTISWEEP_DEFAULT_AMPLITUDE],
            "amp": MULTISWEEP_DEFAULT_AMPLITUDE,
            "span_hz": MULTISWEEP_DEFAULT_SPAN_HZ,
            "npoints_per_sweep": MULTISWEEP_DEFAULT_NPOINTS,
            "nsamps": MULTISWEEP_DEFAULT_NSAMPLES,
            "bias_frequency_method": None,
            "rotate_saved_data": False,
            "sweep_direction": "upward",
            "resonance_frequencies": {self.module: [90e6, 91e6]},
            "module": self.module,
            "apply_skewed_fit": False,
            "apply_nonlinear_fit": False,
        }
        _assert_param_keys(
            self.multisweep_params,
            "rfmux.tools.periscope.multisweep_dialog",
            "MultisweepDialog",
        )
    
        self.bias_params = {
            "nonlinear_threshold": 0.77,
            "fallback_to_lowest": True,
            "optimize_phase": True,
            "num_phase_samples": 300,
            "phase_step": 5,
            "bandpass_params": {
                "apply_bandpass": True,
                "lowcut": 5.0,
                "highcut": 20.0,
                "fs": 597.0,
            },
            "apply_bandpass": True,
            "lowcut": 5.0,
            "highcut": 20.0,
            "fs": 597.0,
        }
        _assert_param_keys(
            self.bias_params,
            "rfmux.tools.periscope.bias_kids_dialog",
            "BiasKidsDialog",
        )
    
        self.noise_params = {
            "num_samples": 10000,
            "spectrum_limit": 0.9,
            "num_segments": 10,
            "decimation": 6,
            "reference": "relative",
            "effective_highest_freq": 10.0,
            "time_taken": 1.0,
            "freq_resolution": 0.1,
            "pfb_enabled": False,
            "overlap": 2,
            "pfb_samples": 210000,
            "pfb_time": 0.41
        }
        _assert_param_keys(
            self.noise_params,
            "rfmux.tools.periscope.noise_spectrum_dialog",
            "NoiseSpectrumDialog",
        )

        
    @contextmanager
    def _ui_mock_context(self):
        """
        Mock all dialogs, windows, tasks, and signals used by Periscope UI helpers.

        This context manager replaces Qt dialogs, background tasks, and signal
        classes with lightweight :class:`unittest.mock.MagicMock` instances so UI
        entry points can be invoked without spinning up threads or opening
        windows. It is intended for quick smoke-testing of control flow.
        """

        self.test_dialog_params()
        
        from importlib import import_module

        periscope_app = import_module("rfmux.tools.periscope.app")
        utils_mod = import_module("rfmux.tools.periscope.utils")

        default_resonances = [90e6, 91e6]
    
        if len(default_resonances) < 2:
            default_resonances = list(default_resonances) + [default_resonances[0] + 1e6]
    
        init_params = {
            "irig_source": getattr(self.crs.TIMESTAMP_PORT, "TEST", "TEST"),
            "clear_channels": True,
        }

        fake_init_dialog = MagicMock()
        fake_init_dialog.exec.return_value = True
        fake_init_dialog.get_parameters.return_value = init_params
        fake_init_dialog.get_selected_irig_source.return_value = init_params["irig_source"]
        fake_init_dialog.get_clear_channels_state.return_value = init_params["clear_channels"]
        fake_init_dialog.module_entry = MagicMock()
        fake_init_dialog.module_entry.setText = MagicMock()
        fake_init_dialog.dac_scales = {}

        fake_netanal_dialog = MagicMock()
        fake_netanal_dialog.exec.return_value = True
        fake_netanal_dialog.get_parameters.return_value = self.netanal_params
        fake_netanal_dialog.module_entry = MagicMock()
        fake_netanal_dialog.module_entry.setText = MagicMock()
        fake_netanal_dialog.dac_scales = {}

        fake_find_dialog = MagicMock()
        fake_find_dialog.exec.return_value = True
        fake_find_dialog.get_parameters.return_value = self.find_params

        fake_bias_dialog = MagicMock()
        fake_bias_dialog.exec.return_value = True
        fake_bias_dialog.get_parameters.return_value = self.bias_params

        fake_noise_dialog = MagicMock()
        fake_noise_dialog.exec.return_value = True
        fake_noise_dialog.get_parameters.return_value = self.noise_params

        fake_mock_config_dialog = MagicMock()
        fake_mock_config_dialog.exec.return_value = True
        fake_mock_config_dialog.get_configuration.return_value = {"mock": True}

        fake_signals = MagicMock()
        fake_signals.receivers.return_value = 0
        for sig_name in (
            "progress",
            "data_update",
            "data_update_with_amp",
            "completed",
            "error",
        ):
            signal = MagicMock()
            signal.connect = MagicMock()
            setattr(fake_signals, sig_name, signal)

        fake_bias_signals = MagicMock()
        fake_bias_signals.progress.connect = MagicMock()
        fake_bias_signals.error.connect = MagicMock()

        fetcher_signal = MagicMock()
        fetcher_signal.connect = MagicMock()
        fake_fetcher = MagicMock()
        fake_fetcher.start = MagicMock()
        fake_fetcher.dac_scales_ready = fetcher_signal

        # Dock and panel scaffolding
        def _mock_create_dock(_self, widget, title, dock_id=None):
            dock = MagicMock()
            dock.widget.return_value = widget
            dock.windowTitle.return_value = title
            return dock

        MockDockCreate = MagicMock(side_effect=_mock_create_dock)
        MockDockGet = MagicMock(return_value=None)

        ### Use panel classes for mocked flows ###
        from rfmux.tools.periscope.network_analysis_panel import NetworkAnalysisPanel
        MockNAWindow = NetworkAnalysisPanel

        from rfmux.tools.periscope.multisweep_panel import MultisweepPanel
        MockMultiWindow = MultisweepPanel
        
        MockInitCRS = MagicMock(return_value=fake_init_dialog)
        MockNetAnal = MagicMock(return_value=fake_netanal_dialog)
        MockFindRes = MagicMock(return_value=fake_find_dialog)
        MockMulti = MagicMock(return_value=MagicMock(exec=MagicMock(return_value=True), get_parameters=MagicMock(return_value=self.multisweep_params)))
        MockBiasDialog = MagicMock(return_value=fake_bias_dialog)
        MockNoiseDialog = MagicMock(return_value=fake_noise_dialog)
        MockConfigDialog = MagicMock(return_value=fake_mock_config_dialog)
        MockCRSInitTask = MagicMock(return_value=MagicMock(start=MagicMock()))
        MockFetcher = MagicMock(return_value=fake_fetcher)
        MockNASignals = MagicMock(return_value=fake_signals)
        MockNATask = MagicMock(return_value=MagicMock(start=MagicMock()))
        MockMultiTask = MagicMock(return_value=MagicMock(start=MagicMock()))
        MockBiasTask = MagicMock(return_value=MagicMock(start=MagicMock()))
        MockBiasSignals = MagicMock(return_value=fake_bias_signals)

        qt_suppression_patchers = [
            patch.object(
                utils_mod.QtWidgets.QDialog,
                "exec",
                MagicMock(return_value=utils_mod.QtWidgets.QDialog.Accepted),
                create=True,
            ),
            patch.object(utils_mod.QtWidgets.QDialog, "show", MagicMock(), create=True),
            patch.object(utils_mod.QtWidgets.QWidget, "show", MagicMock(), create=True),
            patch.object(
                utils_mod.QtWidgets.QMainWindow, "show", MagicMock(), create=True
            ),
            patch.object(utils_mod.QtWidgets.QMessageBox, "information", MagicMock(), create=True),
            patch.object(utils_mod.QtWidgets.QMessageBox, "warning", MagicMock(), create=True),
            patch.object(utils_mod.QtWidgets.QMessageBox, "critical", MagicMock(), create=True),
            patch.object(utils_mod.QtWidgets.QMessageBox, "question", MagicMock(), create=True),
            patch.object(utils_mod.QtWidgets.QMainWindow, "tabifyDockWidget", MagicMock(), create=True),
            patch.object(utils_mod.QtWidgets.QMainWindow, "addDockWidget", MagicMock(), create=True),
        ]

        module_patchers = [
            patch(
                "rfmux.tools.periscope.initialize_crs_dialog.InitializeCRSDialog",
                MockInitCRS,
                create=True,
            ),
            patch(
                "rfmux.tools.periscope.network_analysis_dialog.NetworkAnalysisDialog",
                MockNetAnal,
                create=True,
            ),
            patch(
                "rfmux.tools.periscope.find_resonances_dialog.FindResonancesDialog",
                MockFindRes,
                create=True,
            ),
            patch(
                "rfmux.tools.periscope.multisweep_dialog.MultisweepDialog",
                MockMulti,
                create=True,
            ),
            patch(
                "rfmux.tools.periscope.bias_kids_dialog.BiasKidsDialog",
                MockBiasDialog,
                create=True,
            ),
            patch(
                "rfmux.tools.periscope.noise_spectrum_dialog.NoiseSpectrumDialog",
                MockNoiseDialog,
                create=True,
            ),
            patch(
                "rfmux.tools.periscope.mock_configuration_dialog.MockConfigurationDialog",
                MockConfigDialog,
                create=True,
            ),
            patch("rfmux.tools.periscope.tasks.CRSInitializeTask", MockCRSInitTask, create=True),
            patch("rfmux.tools.periscope.tasks.DACScaleFetcher", MockFetcher, create=True),
            patch(
                "rfmux.tools.periscope.tasks.NetworkAnalysisSignals",
                MockNASignals,
                create=True,
            ),
            patch("rfmux.tools.periscope.tasks.NetworkAnalysisTask", MockNATask, create=True),
            patch("rfmux.tools.periscope.tasks.MultisweepTask", MockMultiTask, create=True),
            patch("rfmux.tools.periscope.tasks.BiasKidsTask", MockBiasTask, create=True),
            patch("rfmux.tools.periscope.tasks.BiasKidsSignals", MockBiasSignals, create=True),
            patch(
                "rfmux.tools.periscope.dock_manager.PeriscopeDockManager.create_dock",
                MockDockCreate,
                create=True,
            ),
            patch(
                "rfmux.tools.periscope.dock_manager.PeriscopeDockManager.get_dock",
                MockDockGet,
                create=True,
            ),
        ]


        app_patchers = [
            patch.object(periscope_app, "InitializeCRSDialog", MockInitCRS, create=True),
            patch.object(periscope_app, "NetworkAnalysisDialog", MockNetAnal, create=True),
            patch.object(periscope_app, "FindResonancesDialog", MockFindRes, create=True),
            patch.object(periscope_app, "MultisweepDialog", MockMulti, create=True),
            patch.object(periscope_app, "BiasKidsDialog", MockBiasDialog, create=True),
            patch.object(periscope_app, "NoiseSpectrumDialog", MockNoiseDialog, create=True),
            patch.object(periscope_app, "MockConfigurationDialog", MockConfigDialog, create=True),
            patch.object(periscope_app.Periscope, "_apply_mock_configuration", MagicMock()),
            patch.object(periscope_app, "CRSInitializeTask", MockCRSInitTask, create=True),
            patch.object(periscope_app, "DACScaleFetcher", MockFetcher, create=True),
            patch.object(periscope_app, "NetworkAnalysisSignals", MockNASignals, create=True),
            patch.object(periscope_app, "NetworkAnalysisTask", MockNATask, create=True),
            patch.object(periscope_app, "MultisweepTask", MockMultiTask, create=True),
            patch.object(periscope_app, "BiasKidsTask", MockBiasTask, create=True),
            patch.object(periscope_app, "BiasKidsSignals", MockBiasSignals, create=True),
            patch.object(periscope_app, "NetworkAnalysisPanel", MockNAWindow, create=True),
            patch.object(periscope_app, "MultisweepPanel", MockMultiWindow, create=True),
            patch.object(periscope_app, "PeriscopeDockManager", MagicMock(), create=True),
        ]

        from rfmux.tools.periscope.detector_digest_panel import DetectorDigestPanel
        
        patchers_for_digest = [
            patch.object(DetectorDigestPanel, "_setup_ui", MagicMock()),
            patch.object(DetectorDigestPanel, "_update_plots", MagicMock()),
            patch.object(DetectorDigestPanel, "apply_theme", MagicMock()),
            patch.object(DetectorDigestPanel, "resize", MagicMock()),
            patch.object(DetectorDigestPanel, "show", MagicMock()),
        ]

        from rfmux.tools.periscope.noise_spectrum_panel import NoiseSpectrumPanel

        patchers_for_noise_spectrum = [
            patch.object(NoiseSpectrumPanel, "_setup_ui", MagicMock()),
            patch.object(NoiseSpectrumPanel, "_update_noise_plots", MagicMock()),
            patch.object(NoiseSpectrumPanel, "apply_theme", MagicMock()),
            patch.object(NoiseSpectrumPanel, "resize", MagicMock()),
            patch.object(NoiseSpectrumPanel, "show", MagicMock()),
        ]

        patchers = qt_suppression_patchers + module_patchers  + app_patchers  + patchers_for_digest + patchers_for_noise_spectrum
        try:
            for patcher in patchers:
                patcher.start()
            yield
        finally:
            for patcher in reversed(patchers):
                patcher.stop()

    
    def run_ui_mock_smoke_test(self):
        """
        Execute key UI flows with all dialogs and tasks mocked out.

        This helper initializes required attributes with safe defaults and then
        exercises the dialog- and window-opening methods within
        :meth:`_ui_mock_context` so no real Qt widgets or threads are spawned.
        """

        self.crs = getattr(self, "crs", None) or MagicMock()
        self.crs.generate_resonators = AsyncMock(return_value=None)
        self.crs.set_pulse_mode = AsyncMock(return_value=None)
        if not hasattr(self.crs, "TIMESTAMP_PORT"):
            self.crs.TIMESTAMP_PORT = SimpleNamespace(
                BACKPLANE="BACKPLANE", TEST="TEST", SMA="SMA"
            )

        self.pool = getattr(self, "pool", None) or MagicMock()
        if not hasattr(self.pool, "start"):
            self.pool.start = MagicMock()

            
        self.module = getattr(self, "module", 1)
        self.netanal_window_count = getattr(self, "netanal_window_count", 0)
        self.netanal_windows = getattr(self, "netanal_windows", {})
        self.netanal_tasks = getattr(self, "netanal_tasks", {})
        self.multisweep_window_count = getattr(self, "multisweep_window_count", 0)
        self.multisweep_windows = getattr(self, "multisweep_windows", {})
        self.multisweep_tasks = getattr(self, "multisweep_tasks", {})
        self.raw_data = getattr(self, "raw_data", {self.module: {"default": MagicMock()}})
        self.resonance_freqs = getattr(
            self, "resonance_freqs", {self.module: [90e6, 91e6]}
        )
        self.dac_scales = getattr(self, "dac_scales", {self.module: -0.5})
        self.dark_mode = getattr(self, "dark_mode", False)
        self.channel_list = getattr(self, "channel_list", [[self.module]])
        self.tabs = getattr(self, "tabs", MagicMock())
        self.tabs.currentIndex.return_value = 0
        self.tabs.tabText.return_value = f"Module {self.module}"
        self.is_mock_mode = getattr(self, "is_mock_mode", True)
        self.mock_config = getattr(self, "mock_config", {"mock": True})
        self.qp_pulse_mode = getattr(self, "qp_pulse_mode", "none")
        self.results_by_iteration = {0: {"amplitude": 0.1,
                                         "direction": "up",
                                         "data": {
                                             1: {
                                                 "bias_frequency": 90e6,                     # MUST BE FLOAT
                                                 "original_center_frequency": 90e6,          # MUST BE FLOAT
                                                 "sweep_amplitudes": [0.1, 0.2],
                                                 "some_data": [1, 2, 3]
                                             }}}}

        if not hasattr(self, "crs_init_signals"):
            self.crs_init_signals = MagicMock()
        for sig_name in ("success", "error"):
            if not hasattr(self.crs_init_signals, sig_name):
                signal = MagicMock()
                signal.connect = MagicMock()
                setattr(self.crs_init_signals, sig_name, signal)

        if not hasattr(self, "multisweep_signals"):
            self.multisweep_signals = MagicMock()
        for sig_name in (
            "progress",
            "starting_iteration",
            "data_update",
            "completed_iteration",
            "all_completed",
            "error",
            "fitting_progress",
        ):
            if not hasattr(self.multisweep_signals, sig_name):
                setattr(self.multisweep_signals, sig_name, MagicMock())

        with self._ui_mock_context():
            print(">>> Testing Mock Configuration Dialog")
            if hasattr(self, "_show_mock_config_dialog"):
                self._show_mock_config_dialog()

            print(">>> Testing Initialize CRS Dialog")
            self._show_initialize_crs_dialog()

            print(">>> Testing Network Analysis Dialog")
            self._show_netanal_dialog()

            print(">>> Testing Network Analysis Window Logic")
            self._start_network_analysis(self.netanal_params)

            netanal_window = None
            if getattr(self, "netanal_windows", None):
                netanal_window = next(iter(self.netanal_windows.values())).get("window")
                netanal_window._run_and_plot_resonances = MagicMock()

            if netanal_window:
                netanal_window.raw_data = self.raw_data
                
                print(">>> Testing Find Resonances Dialog")
                reso_diag = netanal_window._show_find_resonances_dialog()

                netanal_window.resonance_freqs = self.resonance_freqs
                print(">>> Testing Multisweep Dialog")
                netanal_window._show_multisweep_dialog()

            print(">>> Testing Multisweep Window")
            self._start_multisweep_analysis(self.multisweep_params)

            multisweep_window = None
            if getattr(self, "multisweep_windows", None):
                multisweep_window = next(iter(self.multisweep_windows.values())).get("window")
                multisweep_window.results_by_iteration = self.results_by_iteration
                multisweep_window._get_spectrum = MagicMock()

            if multisweep_window:
                print(">>> Testing Bias KIDs Dialog")
                multisweep_window._bias_kids()

                print(">>> Testing Noise Spectrum Dialog")
                multisweep_window._open_noise_spectrum_dialog()

                print(">>> Testing Detector Digest Window from double click")

                # Create a fake event with scenePos() attribute
                class FakeClickEvent:
                    def __init__(self, x):
                        self._x = x
                    def scenePos(self):
                        return QtCore.QPointF(self._x, 0)
                    def accept(self):
                        pass
    
                fake_event = FakeClickEvent(90e6)

                self.test_noise_samples = [0]
                self.phase_shifts = [0]
    
                multisweep_window._handle_multisweep_plot_double_click(fake_event)

                print(">>> Testing Detector Digest Panel")
    
                before = len(multisweep_window.detector_digest_windows)
                
                multisweep_window._open_detector_digest_for_index(1)
                
                after = len(multisweep_window.detector_digest_windows)
                assert after == before + 1
                
                digest_panel = multisweep_window.detector_digest_windows[-1]
                assert isinstance(digest_panel, rfmux.tools.periscope.detector_digest_panel.DetectorDigestPanel)
                assert digest_panel.detector_id == 1
                
                print(">>> Testing Noise Spectrum Panel")
                
                # --- Noise Spectrum Panel ---
                multisweep_window.spectrum_noise_data = MagicMock()
                before = len(multisweep_window.noise_spectrum_windows)
                
                multisweep_window._open_noise_spectrum_panel(1)
                
                after = len(multisweep_window.noise_spectrum_windows)
                assert after == before + 1
                
                noise_panel = multisweep_window.noise_spectrum_windows[-1]
                assert isinstance(noise_panel, rfmux.tools.periscope.noise_spectrum_panel.NoiseSpectrumPanel)
                assert noise_panel.detector_id == 1

        print("\n✓ ALL dialog / window functions executed successfully (mocked)\n")
