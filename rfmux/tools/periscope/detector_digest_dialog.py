"""Window for displaying a detailed digest of a single detector resonance."""

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from .utils import (
    ClickableViewBox, UnitConverter, LINE_WIDTH,
    IQ_COLORS, SCATTER_COLORS, DISTINCT_PLOT_COLORS, COLORMAP_CHOICES
)
from rfmux.core.transferfunctions import convert_roc_to_volts, convert_roc_to_dbm # For direct use if needed

class DetectorDigestWindow(QtWidgets.QMainWindow):
    """
    A window that displays a detailed "digest" of a single detector resonance,
    including three plots: Sweep (vs freq.), Sweep (IQ plane), and Bias amplitude optimization.
    """
    def __init__(self, parent: QtWidgets.QWidget = None,
                 resonance_data_for_digest: dict = None, # {amp_raw: sweep_data_dict}
                 detector_id: int = -1,
                 resonance_frequency_ghz: float = 0.0,
                 dac_scales: dict = None, # {module_id: scale_dbm}
                 zoom_box_mode: bool = True,
                 target_module: int = None,
                 normalize_plot3: bool = False, # Added for Plot 3 normalization
                 dark_mode: bool = False): # Added dark mode parameter
        super().__init__(parent)
        self.resonance_data_for_digest = resonance_data_for_digest or {} # {amp_raw: {'data': sweep_data_dict, 'actual_cf_hz': float}}
        self.detector_id = detector_id
        self.resonance_frequency_ghz_title = resonance_frequency_ghz # For window title (conceptual base)
        # self.f_active_bias_hz is now dynamic, see self.current_plot_offset_hz
        self.dac_scales = dac_scales or {}
        self.zoom_box_mode = zoom_box_mode
        self.target_module = target_module
        self.normalize_plot3 = normalize_plot3
        self.dark_mode = dark_mode # Store dark mode setting
        
        # Store reference to parent to ensure proper behavior
        self.parent_window = parent

        self.active_amplitude_raw = None
        self.active_sweep_info = None # Will store {'data': data_dict, 'actual_cf_hz': float}
        self.active_sweep_data = None # Convenience for self.active_sweep_info['data']
        self.current_plot_offset_hz = None # This will be the f_bias for x-axes, from active_sweep_info['actual_cf_hz']

        if self.resonance_data_for_digest:
            # Default to lowest power/amplitude sweep
            self.active_amplitude_raw = min(self.resonance_data_for_digest.keys())
            self.active_sweep_info = self.resonance_data_for_digest[self.active_amplitude_raw]
            self.active_sweep_data = self.active_sweep_info['data']
            self.current_plot_offset_hz = self.active_sweep_info['actual_cf_hz']
        else: # Should not happen if MultisweepWindow sends valid data
            self.current_plot_offset_hz = self.resonance_frequency_ghz_title * 1e9


        self._setup_ui()
        self._update_plots()

        self.setWindowTitle(f"Detector Digest: Detector {self.detector_id+1}  ({self.resonance_frequency_ghz_title*1e3:.6f} MHz)")
        # Set window flags to ensure proper behavior as a standalone window
        self.setWindowFlags(QtCore.Qt.WindowType.Window)
        self.resize(1200, 450) # Adjusted initial size

    def _setup_ui(self):
        """Sets up the UI layout with three plots."""
        # Create a central widget for the QMainWindow
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Create a vertical layout for title and plots
        outer_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Set the background color of the dialog based on dark mode
        bg_color = "#1C1C1C" if self.dark_mode else "#FFFFFF"
        central_widget.setStyleSheet(f"background-color: {bg_color};")
        
        # Add a title label at the top
        title_text = f"Detector {self.detector_id+1} ({self.resonance_frequency_ghz_title:.6f} GHz)"
        title_label = QtWidgets.QLabel(title_text)
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        font = title_label.font()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(False)
        title_label.setFont(font)
        title_label.setStyleSheet("margin-bottom: 10px;")
        outer_layout.addWidget(title_label)
        
        # Create horizontal layout for the three plots
        plots_layout = QtWidgets.QHBoxLayout()
        outer_layout.addLayout(plots_layout)

        # Define colors based on dark mode
        bg_color, pen_color = ("k", "w") if self.dark_mode else ("w", "k")

        # Plot 1: Sweep (vs freq.)
        vb1 = ClickableViewBox() # Use ClickableViewBox for consistency, though no specific click action here
        vb1.parent_window = self # For zoom_box_mode
        self.plot1_sweep_vs_freq = pg.PlotWidget(viewBox=vb1, name="SweepVsFreq")
        # Title will be set dynamically in _update_plots
        self.plot1_sweep_vs_freq.setLabel('left', "Amplitude", units="V") # pyqtgraph will show mV
        self.plot1_sweep_vs_freq.setLabel('bottom', "f - f_bias", units="Hz") # pyqtgraph will show kHz
        self.plot1_sweep_vs_freq.showGrid(x=True, y=True, alpha=0.3)
        self.plot1_legend = self.plot1_sweep_vs_freq.addLegend(offset=(30,10), labelTextColor=pen_color)
        plots_layout.addWidget(self.plot1_sweep_vs_freq)

        # Plot 2: Sweep (IQ plane)
        vb2 = ClickableViewBox()
        vb2.parent_window = self
        self.plot2_iq_plane = pg.PlotWidget(viewBox=vb2, name="SweepIQPlane")
        # Title will be set dynamically in _update_plots
        self.plot2_iq_plane.setLabel('left', "Q", units="V") # pyqtgraph will show mV
        self.plot2_iq_plane.setLabel('bottom', "I", units="V") # pyqtgraph will show mV
        self.plot2_iq_plane.setAspectLocked(True)
        self.plot2_iq_plane.showGrid(x=True, y=True, alpha=0.3)
        self.plot2_legend = self.plot2_iq_plane.addLegend(offset=(30,10), labelTextColor=pen_color)
        plots_layout.addWidget(self.plot2_iq_plane)

        # Plot 3: Bias amplitude optimization
        vb3 = ClickableViewBox() # Double-click interaction needed here
        vb3.parent_window = self
        self.plot3_bias_opt = pg.PlotWidget(viewBox=vb3, name="BiasOpt")
        self.plot3_bias_opt.setLabel('left', "|S21|", units="dB")
        self.plot3_bias_opt.setLabel('bottom', "f - f_bias", units="Hz") # pyqtgraph will show kHz
        self.plot3_bias_opt.showGrid(x=True, y=True, alpha=0.3)
        self.plot3_legend = self.plot3_bias_opt.addLegend(offset=(30,10), labelTextColor=pen_color)
        plots_layout.addWidget(self.plot3_bias_opt)
        
        # Connect double-click for Plot 3
        vb3.doubleClickedEvent.connect(self._handle_plot3_double_click)

        # Apply zoom box mode
        self._apply_zoom_box_mode_to_all()
        
        # Apply initial theme to plots
        
        # Apply to Plot 1: Sweep vs freq
        self.plot1_sweep_vs_freq.setBackground(bg_color)
        plot_item = self.plot1_sweep_vs_freq.getPlotItem()
        # Initialize with correct colored title
        active_power_dbm_str = "N/A"
        if self.active_amplitude_raw is not None and self.target_module in self.dac_scales:
            try:
                active_power_dbm = UnitConverter.normalize_to_dbm(self.active_amplitude_raw, self.dac_scales[self.target_module])
                active_power_dbm_str = f"{active_power_dbm:.2f} dBm"
            except Exception:
                pass  # Keep "N/A" or raw value if conversion fails
        plot_item.setTitle(f"Sweep (Probe Power {active_power_dbm_str})", color=pen_color)
        for axis_name in ("left", "bottom", "right", "top"):
            ax = plot_item.getAxis(axis_name)
            if ax:
                ax.setPen(pen_color)
                ax.setTextPen(pen_color)
        
        # Apply to Plot 2: IQ plane
        self.plot2_iq_plane.setBackground(bg_color)
        plot_item = self.plot2_iq_plane.getPlotItem()
        # Initialize with correct colored title
        plot_item.setTitle(f"IQ (Probe Power {active_power_dbm_str})", color=pen_color)
        for axis_name in ("left", "bottom", "right", "top"):
            ax = plot_item.getAxis(axis_name)
            if ax:
                ax.setPen(pen_color)
                ax.setTextPen(pen_color)
        
        # Apply to Plot 3: Bias amplitude optimization
        self.plot3_bias_opt.setBackground(bg_color)
        plot_item = self.plot3_bias_opt.getPlotItem()
        # Initialize with correct colored title
        plot_item.setTitle("Bias amplitude optimization", color=pen_color)
        for axis_name in ("left", "bottom", "right", "top"):
            ax = plot_item.getAxis(axis_name)
            if ax:
                ax.setPen(pen_color)
                ax.setTextPen(pen_color)

    def _apply_zoom_box_mode_to_all(self):
        for plot_widget in [self.plot1_sweep_vs_freq, self.plot2_iq_plane, self.plot3_bias_opt]:
            if plot_widget and isinstance(plot_widget.getViewBox(), ClickableViewBox):
                plot_widget.getViewBox().enableZoomBoxMode(self.zoom_box_mode)

    def _clear_plots(self):
        for plot_widget, legend_widget in [
            (self.plot1_sweep_vs_freq, self.plot1_legend),
            (self.plot2_iq_plane, self.plot2_legend),
            (self.plot3_bias_opt, self.plot3_legend)
        ]:
            if plot_widget:
                for item in plot_widget.listDataItems(): plot_widget.removeItem(item)
            if legend_widget: legend_widget.clear()

    def _update_plots(self):
        """Populates all three plots with data."""
        self._clear_plots()
        if not self.active_sweep_data or not self.resonance_data_for_digest:
            return

        active_power_dbm_str = "N/A"
        if self.active_amplitude_raw is not None:
            dac_scale = self.dac_scales.get(self.target_module)
            if dac_scale is not None:
                try:
                    active_power_dbm = UnitConverter.normalize_to_dbm(self.active_amplitude_raw, dac_scale)
                    active_power_dbm_str = f"{active_power_dbm:.2f} dBm"
                except Exception:
                    pass # Keep "N/A" or raw value if conversion fails

        self.plot1_sweep_vs_freq.setTitle(f"Sweep (Probe Power {active_power_dbm_str})")
        self.plot2_iq_plane.setTitle(f"IQ (Probe Power {active_power_dbm_str})")

        # --- Plot 1: Sweep (vs freq.) ---
        if self.active_sweep_data and self.current_plot_offset_hz is not None:
            freqs_hz_active = self.active_sweep_data.get('frequencies')
            iq_complex_active = self.active_sweep_data.get('iq_complex')

            if freqs_hz_active is not None and iq_complex_active is not None:
                s21_mag_volts = convert_roc_to_volts(np.abs(iq_complex_active))
                s21_i_volts = convert_roc_to_volts(iq_complex_active.real)
                s21_q_volts = convert_roc_to_volts(iq_complex_active.imag)
                
                x_axis_hz_offset = freqs_hz_active - self.current_plot_offset_hz

                # Use white for dark mode, black for light mode for the magnitude curve
                mag_color = 'w' if self.dark_mode else 'k'
                # Store the theme-dependent color in IQ_COLORS
                IQ_COLORS["MAGNITUDE"] = mag_color
                self.plot1_sweep_vs_freq.plot(x_axis_hz_offset, s21_mag_volts, pen=pg.mkPen(IQ_COLORS["MAGNITUDE"], width=LINE_WIDTH), name="|I + iQ|")
                self.plot1_sweep_vs_freq.plot(x_axis_hz_offset, s21_i_volts, pen=pg.mkPen(IQ_COLORS["I"], style=QtCore.Qt.PenStyle.DashLine, width=LINE_WIDTH), name="I")
                self.plot1_sweep_vs_freq.plot(x_axis_hz_offset, s21_q_volts, pen=pg.mkPen(IQ_COLORS["Q"], width=LINE_WIDTH), name="Q")
                self.plot1_sweep_vs_freq.autoRange()

        # --- Plot 2: Sweep (IQ plane) ---
        if self.active_sweep_data: # Re-check as it might be cleared if no data
            iq_complex_active = self.active_sweep_data.get('iq_complex')
            if iq_complex_active is not None:
                s21_i_volts = convert_roc_to_volts(iq_complex_active.real)
                s21_q_volts = convert_roc_to_volts(iq_complex_active.imag)
                # Use SCATTER_COLORS for consistency rather than IQ_COLORS
                self.plot2_iq_plane.plot(s21_i_volts, s21_q_volts, pen=None, symbol='o', 
                                        symbolBrush=SCATTER_COLORS["DEFAULT"], symbolPen=SCATTER_COLORS["DEFAULT"], 
                                        symbolSize=5, name="Sweep IQ")

            # Plot rotation_tod if available
            rotation_tod_iq = self.active_sweep_data.get('rotation_tod')
            if rotation_tod_iq is not None and rotation_tod_iq.size > 0:
                tod_i_volts = convert_roc_to_volts(rotation_tod_iq.real)
                tod_q_volts = convert_roc_to_volts(rotation_tod_iq.imag)
                # Use white for dark mode, black for light mode for the noise points
                noise_color = 'w' if self.dark_mode else 'k'
                # Store the theme-dependent color in SCATTER_COLORS
                SCATTER_COLORS["NOISE"] = noise_color
                # Use consistent styling with both brush and pen for scatter points
                self.plot2_iq_plane.plot(tod_i_volts, tod_q_volts, pen=None, symbol='o', 
                                        symbolBrush=SCATTER_COLORS["NOISE"], symbolPen=SCATTER_COLORS["NOISE"], 
                                        symbolSize=3, name="Noise at f_bias")
            
            self.plot2_iq_plane.autoRange()
        
        # --- Plot 3: Bias amplitude optimization ---
        if self.current_plot_offset_hz is not None:
            num_amps = len(self.resonance_data_for_digest)

            # Use centralized DISTINCT_PLOT_COLORS for <= 4 lines
            
            # Sort amplitudes for consistent coloring
            sorted_amp_keys = sorted(self.resonance_data_for_digest.keys())

            for idx, amp_raw in enumerate(sorted_amp_keys):
                sweep_info = self.resonance_data_for_digest[amp_raw]
                sweep_data = sweep_info['data']
                # actual_cf_hz_this_sweep = sweep_info['actual_cf_hz'] # Not used for plotting if x-axis is relative to active sweep's CF

                freqs_hz = sweep_data.get('frequencies')
                iq_complex = sweep_data.get('iq_complex')

                if freqs_hz is not None and iq_complex is not None:
                    s21_mag_db = convert_roc_to_dbm(np.abs(iq_complex))
                    
                    if self.normalize_plot3 and len(s21_mag_db) > 0:
                        ref_val = s21_mag_db[0]
                        if np.isfinite(ref_val):
                            s21_mag_db = s21_mag_db - ref_val
                    
                    x_axis_hz_offset = freqs_hz - self.current_plot_offset_hz
                    
                    current_pen = None # Initialize pen variable
                    if num_amps <= 4:
                        color_str = DISTINCT_PLOT_COLORS[idx % len(DISTINCT_PLOT_COLORS)]
                        current_pen = pg.mkPen(color_str, width=LINE_WIDTH)
                    else: # num_amps > 4
                        cmap = pg.colormap.get(COLORMAP_CHOICES["AMPLITUDE_SWEEP"])
                        # Adjust colormap: map [0,1] to [0.3, 1] if dark mode
                        norm_idx = idx / max(1, num_amps - 1)
                        if self.dark_mode:
                            color_val_in_map = 0.3 + norm_idx * 0.7
                        else:
                            color_val_in_map = norm_idx * 0.9
                        color_from_map = cmap.map(color_val_in_map) 
                        current_pen = pg.mkPen(color_from_map, width=LINE_WIDTH)
                
                    # Legend name: Bias amplitude [dBm]
                    dac_scale_for_module = self.dac_scales.get(self.target_module) # Use target_module
                    legend_name = f"{amp_raw:.2e} Norm" # Fallback
                    if dac_scale_for_module is not None:
                        try:
                            dbm_val = UnitConverter.normalize_to_dbm(amp_raw, dac_scale_for_module)
                            legend_name = f"{dbm_val:.2f} dBm"
                        except Exception: # Catch potential math errors with bad inputs
                            pass # Stick to fallback

                    if current_pen: # Ensure pen is set before plotting
                        self.plot3_bias_opt.plot(x_axis_hz_offset, s21_mag_db, pen=current_pen, name=legend_name)
        self.plot3_bias_opt.autoRange()

    @QtCore.pyqtSlot(object)
    def _handle_plot3_double_click(self, ev):
        """Handles double-click on Plot 3 to change the active sweep."""
        view_box = self.plot3_bias_opt.getViewBox()
        if not view_box or not self.resonance_data_for_digest or self.current_plot_offset_hz is None:
            return

        mouse_point = view_box.mapSceneToView(ev.scenePos())
        # x_coord_hz_offset is relative to the current_plot_offset_hz
        x_coord_on_plot = mouse_point.x() 
        y_coord_on_plot = mouse_point.y()

        min_dist_sq = float('inf')
        closest_amp_raw = None

        sorted_amp_keys = sorted(self.resonance_data_for_digest.keys())

        for amp_raw in sorted_amp_keys:
            sweep_info = self.resonance_data_for_digest[amp_raw]
            sweep_data = sweep_info['data']
            
            freqs_hz = sweep_data.get('frequencies')
            iq_complex = sweep_data.get('iq_complex')

            if freqs_hz is not None and iq_complex is not None:
                s21_mag_db_curve = convert_roc_to_dbm(np.abs(iq_complex))
                if self.normalize_plot3 and len(s21_mag_db_curve) > 0: # Apply same normalization for distance calc
                    ref_val = s21_mag_db_curve[0]
                    if np.isfinite(ref_val):
                        s21_mag_db_curve = s21_mag_db_curve - ref_val

                # X-axis for this specific curve, relative to current_plot_offset_hz
                x_axis_for_this_curve = freqs_hz - self.current_plot_offset_hz
                
                if not (x_axis_for_this_curve.min() <= x_coord_on_plot <= x_axis_for_this_curve.max()):
                    continue
                
                idx = np.abs(x_axis_for_this_curve - x_coord_on_plot).argmin()
                dist_sq = (x_axis_for_this_curve[idx] - x_coord_on_plot)**2 + (s21_mag_db_curve[idx] - y_coord_on_plot)**2

                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_amp_raw = amp_raw
        
        if closest_amp_raw is not None:
            self.active_amplitude_raw = closest_amp_raw
            self.active_sweep_info = self.resonance_data_for_digest[self.active_amplitude_raw]
            self.active_sweep_data = self.active_sweep_info['data']
            self.current_plot_offset_hz = self.active_sweep_info['actual_cf_hz'] # Update the offset
            
            self._update_plots() # Redraw all plots with new active sweep and new x-axis offset
            ev.accept() # Event handled
            
    def apply_theme(self, dark_mode: bool):
        """Apply the dark/light theme to all plots in this window."""
        self.dark_mode = dark_mode
        
        # Set the background color of the central widget
        central_widget = self.centralWidget()
        if central_widget:
            bg_color_hex = "#1C1C1C" if dark_mode else "#FFFFFF"
            central_widget.setStyleSheet(f"background-color: {bg_color_hex};")
        
        # Apply theme to all plots
        bg_color, pen_color = ("k", "w") if dark_mode else ("w", "k")
        
        try:
            # Apply to Plot 1: Sweep vs freq
            if self.plot1_sweep_vs_freq:
                self.plot1_sweep_vs_freq.setBackground(bg_color)
                # Update plot title color - use a more direct approach
                plot_item = self.plot1_sweep_vs_freq.getPlotItem()
                if plot_item:
                    # Set the title explicitly with the color parameter
                    title_text = plot_item.titleLabel.text if plot_item.titleLabel else "Sweep"
                    plot_item.setTitle(title_text, color=pen_color)
                # Update axes colors
                for axis_name in ("left", "bottom", "right", "top"):
                    ax = plot_item.getAxis(axis_name) if plot_item else None
                    if ax:
                        ax.setPen(pen_color)
                        ax.setTextPen(pen_color)
                        
                # Update legend text color using the proper API
                if self.plot1_legend:
                    try:
                        self.plot1_legend.setLabelTextColor(pen_color)
                    except Exception as e:
                        print(f"Error updating plot1_legend colors: {e}")
            
            # Apply to Plot 2: IQ plane
            if self.plot2_iq_plane:
                self.plot2_iq_plane.setBackground(bg_color)
                # Update plot title color - use a more direct approach
                plot_item = self.plot2_iq_plane.getPlotItem()
                if plot_item:
                    # Set the title explicitly with the color parameter
                    title_text = plot_item.titleLabel.text if plot_item.titleLabel else "IQ"
                    plot_item.setTitle(title_text, color=pen_color)
                # Update axes colors
                for axis_name in ("left", "bottom", "right", "top"):
                    ax = plot_item.getAxis(axis_name) if plot_item else None
                    if ax:
                        ax.setPen(pen_color)
                        ax.setTextPen(pen_color)
                        
                # Update legend text color using the proper API
                if self.plot2_legend:
                    try:
                        self.plot2_legend.setLabelTextColor(pen_color)
                    except Exception as e:
                        print(f"Error updating plot2_legend colors: {e}")
            
            # Apply to Plot 3: Bias amplitude optimization
            if self.plot3_bias_opt:
                self.plot3_bias_opt.setBackground(bg_color)
                # Update plot title color - use a more direct approach
                plot_item = self.plot3_bias_opt.getPlotItem()
                if plot_item:
                    # Set the title explicitly with the color parameter
                    title_text = plot_item.titleLabel.text if plot_item.titleLabel else "Bias amplitude optimization"
                    plot_item.setTitle(title_text, color=pen_color)
                # Update axes colors
                for axis_name in ("left", "bottom", "right", "top"):
                    ax = plot_item.getAxis(axis_name) if plot_item else None
                    if ax:
                        ax.setPen(pen_color)
                        ax.setTextPen(pen_color)
                        
                # Update legend text color using the proper API
                if self.plot3_legend:
                    try:
                        self.plot3_legend.setLabelTextColor(pen_color)
                    except Exception as e:
                        print(f"Error updating plot3_legend colors: {e}")
            
            # Redraw all plots to update curve colors based on dark mode setting
            self._update_plots()
        except Exception as e:
            print(f"Error applying theme: {e}")
