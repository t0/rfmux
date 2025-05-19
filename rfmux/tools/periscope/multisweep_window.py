"""Window for displaying multisweep analysis results.""" # Simplified docstring to match others
import datetime
import pickle
import numpy as np
from PyQt6 import QtCore, QtWidgets # Keep direct Qt imports if used directly and not via utils.*
from PyQt6.QtCore import Qt
import pyqtgraph as pg

# Imports from within the 'periscope' subpackage
from .utils import LINE_WIDTH, UnitConverter, ClickableViewBox, QtWidgets, QtCore, pg # Ensure these are available

class MultisweepWindow(QtWidgets.QMainWindow):
    """
    Window for displaying multisweep analysis results.
    Plots combined magnitude and phase for all resonances for each amplitude.
    """
    def __init__(self, parent=None, target_module=None, initial_params=None, dac_scales=None):
        super().__init__(parent)
        self.target_module = target_module
        self.initial_params = initial_params or {} 
        self.dac_scales = dac_scales or {}
        
        self.setWindowTitle(f"Multisweep Results - Module {self.target_module}")
        self.results_by_amplitude = {} 
        self.current_amplitude_being_processed = None
        self.unit_mode = "dbm"
        self.normalize_magnitudes = False
        self.zoom_box_mode = True 

        self.combined_mag_plot = None
        self.combined_phase_plot = None
        self.mag_legend = None
        self.phase_legend = None
        self.curves_mag = {} 
        self.curves_phase = {} 
        
        self.active_module_for_dac = self.target_module 

        self.show_cf_lines_cb = None
        self.cf_lines_mag = {} 
        self.cf_lines_phase = {}

        self._setup_ui()
        self.resize(1200, 800)

    def _setup_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        self._setup_toolbar(main_layout)
        self._setup_plot_area(main_layout)
        self._setup_status_bar()


    def _setup_toolbar(self, layout):
        toolbar = QtWidgets.QToolBar("Multisweep Controls")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.export_btn = QtWidgets.QPushButton("Export Data")
        self.export_btn.clicked.connect(self._export_data)
        toolbar.addWidget(self.export_btn)

        self.rerun_btn = QtWidgets.QPushButton("Re-run Multisweep")
        self.rerun_btn.clicked.connect(self._rerun_multisweep)
        toolbar.addWidget(self.rerun_btn)
        
        toolbar.addSeparator()

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True) 
        toolbar.addWidget(QtWidgets.QLabel("Current Sweep Progress:"))
        toolbar.addWidget(self.progress_bar)
        
        self.current_amp_label = QtWidgets.QLabel("Current Amplitude: N/A")
        toolbar.addWidget(self.current_amp_label)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self.normalize_checkbox = QtWidgets.QCheckBox("Normalize Magnitudes")
        self.normalize_checkbox.setChecked(self.normalize_magnitudes)
        self.normalize_checkbox.toggled.connect(self._toggle_normalization)
        toolbar.addWidget(self.normalize_checkbox)

        self.show_cf_lines_cb = QtWidgets.QCheckBox("Show Center Frequencies")
        self.show_cf_lines_cb.setChecked(False) 
        self.show_cf_lines_cb.toggled.connect(self._toggle_cf_lines_visibility)
        toolbar.addWidget(self.show_cf_lines_cb)

        self._setup_unit_controls(toolbar)
        self._setup_zoom_box_control(toolbar)

    def _setup_unit_controls(self, toolbar):
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
        
        self.rb_counts.toggled.connect(lambda: self._update_unit_mode("counts"))
        self.rb_dbm.toggled.connect(lambda: self._update_unit_mode("dbm"))
        self.rb_volts.toggled.connect(lambda: self._update_unit_mode("volts"))
        
        unit_group.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        toolbar.addSeparator()
        toolbar.addWidget(unit_group)

    def _setup_zoom_box_control(self, toolbar):
        self.zoom_box_cb = QtWidgets.QCheckBox("Zoom Box Mode")
        self.zoom_box_cb.setChecked(self.zoom_box_mode)
        self.zoom_box_cb.toggled.connect(self._toggle_zoom_box_mode)
        toolbar.addWidget(self.zoom_box_cb)

    def _setup_plot_area(self, layout):
        plot_container = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_container)

        vb_mag = ClickableViewBox() 
        vb_mag.parent_window = self 
        self.combined_mag_plot = pg.PlotWidget(viewBox=vb_mag, title="Combined S21 Magnitude (All Resonances)")
        self.combined_mag_plot.setLabel('bottom', 'Frequency', units='Hz')
        self._update_mag_plot_label() 
        self.combined_mag_plot.showGrid(x=True, y=True, alpha=0.3)
        self.mag_legend = self.combined_mag_plot.addLegend(offset=(30,10))
        plot_layout.addWidget(self.combined_mag_plot)

        vb_phase = ClickableViewBox()
        vb_phase.parent_window = self
        self.combined_phase_plot = pg.PlotWidget(viewBox=vb_phase, title="Combined S21 Phase (All Resonances)")
        self.combined_phase_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.combined_phase_plot.setLabel('left', 'Phase', units='deg')
        self.combined_phase_plot.showGrid(x=True, y=True, alpha=0.3)
        self.phase_legend = self.combined_phase_plot.addLegend(offset=(30,10))
        plot_layout.addWidget(self.combined_phase_plot)
        
        layout.addWidget(plot_container)
        
        self.combined_phase_plot.setXLink(self.combined_mag_plot)
        self._apply_zoom_box_mode()


    def _setup_status_bar(self):
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

    def _toggle_normalization(self, checked):
        self.normalize_magnitudes = checked
        self._update_mag_plot_label()
        self._redraw_plots()

    def _update_unit_mode(self, mode):
        if self.unit_mode != mode:
            self.unit_mode = mode
            self._update_mag_plot_label()
            self._redraw_plots()
            
    def _update_mag_plot_label(self):
        if not self.combined_mag_plot: return
        if self.normalize_magnitudes:
            label = "Normalized Magnitude"
            units = "dB" if self.unit_mode == "dbm" else ""
        else:
            if self.unit_mode == "counts": label, units = "Magnitude", "Counts"
            elif self.unit_mode == "dbm": label, units = "Power", "dBm"
            elif self.unit_mode == "volts": label, units = "Magnitude", "V"
            else: label, units = "Magnitude", ""
        self.combined_mag_plot.setLabel('left', label, units=units)

    def _toggle_zoom_box_mode(self, enable):
        self.zoom_box_mode = enable
        self._apply_zoom_box_mode()

    def _apply_zoom_box_mode(self):
        if self.combined_mag_plot and isinstance(self.combined_mag_plot.getViewBox(), ClickableViewBox):
            self.combined_mag_plot.getViewBox().enableZoomBoxMode(self.zoom_box_mode)
        if self.combined_phase_plot and isinstance(self.combined_phase_plot.getViewBox(), ClickableViewBox):
            self.combined_phase_plot.getViewBox().enableZoomBoxMode(self.zoom_box_mode)

    def update_progress(self, module, progress_percentage):
        if module == self.target_module:
            self.progress_bar.setValue(int(progress_percentage))

    def update_intermediate_data(self, module, amplitude, intermediate_results):
        if module != self.target_module: return
        pass

    def update_data(self, module, amplitude, final_results_for_amplitude):
        if module != self.target_module: return
        
        self.current_amplitude_being_processed = amplitude
        self.current_amp_label.setText(f"Processing Amp: {amplitude:.4f}")
        self.results_by_amplitude[amplitude] = final_results_for_amplitude
        self._redraw_plots()

    def _redraw_plots(self):
        if not self.combined_mag_plot or not self.combined_phase_plot: return

        if self.mag_legend: self.mag_legend.clear()
        if self.phase_legend: self.phase_legend.clear()

        for item in self.combined_mag_plot.listDataItems(): self.combined_mag_plot.removeItem(item)
        for item in self.combined_phase_plot.listDataItems(): self.combined_phase_plot.removeItem(item)
        
        self.curves_mag.clear(); self.curves_phase.clear()

        for amp_val_lines in self.cf_lines_mag.values():
            for line in amp_val_lines: self.combined_mag_plot.removeItem(line)
        self.cf_lines_mag.clear()
        for amp_val_lines in self.cf_lines_phase.values():
            for line in amp_val_lines: self.combined_phase_plot.removeItem(line)
        self.cf_lines_phase.clear()

        num_amps = len(self.results_by_amplitude)
        if num_amps == 0: return
        
        channel_families = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        viridis_cmap = pg.colormap.get("viridis")
        sorted_amplitudes = sorted(self.results_by_amplitude.keys())
        legend_items_mag = {}; legend_items_phase = {}

        for amp_idx, amp_val in enumerate(sorted_amplitudes):
            amp_results = self.results_by_amplitude[amp_val]
            color = channel_families[amp_idx % len(channel_families)] if num_amps <= 5 else viridis_cmap.map(amp_idx / max(1, num_amps - 1))
            pen = pg.mkPen(color, width=LINE_WIDTH)
            
            legend_name_amp = ""
            if self.unit_mode == "dbm":
                dac_scale = self.dac_scales.get(self.active_module_for_dac)
                legend_name_amp = f"Probe: {UnitConverter.normalize_to_dbm(amp_val, dac_scale):.2f} dBm" if dac_scale is not None else f"Probe: {amp_val:.3e} (Norm)"
            elif self.unit_mode == "volts":
                dac_scale = self.dac_scales.get(self.active_module_for_dac)
                if dac_scale is not None:
                    dbm_value = UnitConverter.normalize_to_dbm(amp_val, dac_scale)
                    power_watts = 10**((dbm_value - 30)/10); resistance = 50.0
                    voltage_rms = np.sqrt(power_watts * resistance); voltage_peak_uv = voltage_rms * np.sqrt(2) * 1e6
                    legend_name_amp = f"Probe: {voltage_peak_uv:.1f} uVpk"
                else: legend_name_amp = f"Probe: {amp_val:.3e} (Norm)"
            else: legend_name_amp = f"Probe: {amp_val:.3e} Norm"

            if amp_val not in legend_items_mag:
                dummy_mag_curve_for_legend = pg.PlotDataItem(pen=pen)
                self.mag_legend.addItem(dummy_mag_curve_for_legend, legend_name_amp)
                legend_items_mag[amp_val] = dummy_mag_curve_for_legend
            if amp_val not in legend_items_phase:
                dummy_phase_curve_for_legend = pg.PlotDataItem(pen=pen)
                self.phase_legend.addItem(dummy_phase_curve_for_legend, legend_name_amp)
                legend_items_phase[amp_val] = dummy_phase_curve_for_legend

            for cf, data in amp_results.items():
                freqs_hz = data.get('frequencies'); iq_complex = data.get('iq_complex')
                if freqs_hz is None or iq_complex is None or len(freqs_hz) == 0: continue
                s21_mag_raw = np.abs(iq_complex)
                s21_mag_processed = UnitConverter.convert_amplitude(s21_mag_raw, iq_complex, self.unit_mode, normalize=self.normalize_magnitudes)
                phase_deg = data.get('phase_degrees', np.degrees(np.angle(iq_complex)))
                
                mag_curve = self.combined_mag_plot.plot(pen=pen); mag_curve.setData(freqs_hz, s21_mag_processed)
                if amp_val not in self.curves_mag: self.curves_mag[amp_val] = {}
                self.curves_mag[amp_val][cf] = mag_curve
                phase_curve = self.combined_phase_plot.plot(pen=pen); phase_curve.setData(freqs_hz, phase_deg)
                if amp_val not in self.curves_phase: self.curves_phase[amp_val] = {}
                self.curves_phase[amp_val][cf] = phase_curve

                if self.show_cf_lines_cb and self.show_cf_lines_cb.isChecked():
                    cf_line_pen = pg.mkPen(color, style=QtCore.Qt.PenStyle.DashLine, width=LINE_WIDTH/2)
                    mag_cf_line = pg.InfiniteLine(pos=cf, angle=90, pen=cf_line_pen, movable=False)
                    self.combined_mag_plot.addItem(mag_cf_line); self.cf_lines_mag.setdefault(amp_val, []).append(mag_cf_line)
                    phase_cf_line = pg.InfiniteLine(pos=cf, angle=90, pen=cf_line_pen, movable=False)
                    self.combined_phase_plot.addItem(phase_cf_line); self.cf_lines_phase.setdefault(amp_val, []).append(phase_cf_line)
        
        self.combined_mag_plot.autoRange(); self.combined_phase_plot.autoRange()

    def completed_amplitude_sweep(self, module, amplitude):
        if module == self.target_module:
            self.statusBar.showMessage(f"Completed sweep for amplitude: {amplitude:.4f}")
            self.progress_bar.setValue(100)

    def all_sweeps_completed(self):
        self.statusBar.showMessage("All multisweep amplitudes completed.")
        self.progress_bar.setVisible(False)
        self.current_amp_label.setText("All Amplitudes Processed")

    def handle_error(self, module, amplitude, error_msg):
        if module == self.target_module or module == -1:
            amp_str = f"for amplitude {amplitude:.4f}" if amplitude != -1 else "general"
            QtWidgets.QMessageBox.critical(self, "Multisweep Error", f"Error {amp_str} on Module {self.target_module}:\n{error_msg}")
            self.statusBar.showMessage(f"Error during multisweep {amp_str}.")
            self.progress_bar.setVisible(False)

    def _export_data(self):
        if not self.results_by_amplitude: QtWidgets.QMessageBox.warning(self, "No Data", "No data to export."); return
        dialog = QtWidgets.QFileDialog(self); dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilters(["Pickle Files (*.pkl)", "All Files (*)"]); dialog.setDefaultSuffix("pkl")
        if dialog.exec():
            filename = dialog.selectedFiles()[0]
            try:
                export_content = {'timestamp': datetime.datetime.now().isoformat(), 'target_module': self.target_module,
                                  'initial_parameters': self.initial_params, 'dac_scales_used': self.dac_scales,
                                  'results_by_amplitude': self.results_by_amplitude}
                with open(filename, 'wb') as f: pickle.dump(export_content, f)
                QtWidgets.QMessageBox.information(self, "Export Complete", f"Data exported to {filename}")
            except Exception as e: QtWidgets.QMessageBox.critical(self, "Export Error", f"Error exporting: {str(e)}")

    def _rerun_multisweep(self):
        # Assuming MultisweepDialog is imported correctly from .dialogs
        from .dialogs import MultisweepDialog 

        if not self.initial_params.get('resonance_frequencies'):
            QtWidgets.QMessageBox.warning(self, "Cannot Re-run", "Initial resonance frequencies not available."); return
        
        current_recalc_cf_state = self.initial_params.get('recalculate_center_frequencies', True)
        current_perform_fits_state = self.initial_params.get('perform_fits', True)
        dialog_params = self.initial_params.copy()
        dialog_params['recalculate_center_frequencies'] = current_recalc_cf_state
        dialog_params['perform_fits'] = current_perform_fits_state
        
        dialog = MultisweepDialog(parent=self, resonance_frequencies=self.initial_params['resonance_frequencies'],
                                  dac_scales=self.dac_scales, current_module=self.target_module, initial_params=dialog_params)
        if dialog.exec():
            new_params = dialog.get_parameters()
            if new_params:
                self.initial_params.update(new_params) 
                self.results_by_amplitude.clear(); self._redraw_plots()
                self.progress_bar.setValue(0); self.progress_bar.setVisible(True)
                self.current_amp_label.setText("Current Amplitude: N/A"); self.statusBar.showMessage("Starting new multisweep...")
                if hasattr(self.parent(), '_start_multisweep_analysis_for_window'):
                    self.parent()._start_multisweep_analysis_for_window(self, new_params)
                else: QtWidgets.QMessageBox.warning(self, "Error", "Cannot trigger re-run. Parent linkage missing.")

    def closeEvent(self, event):
        if hasattr(self.parent(), 'stop_multisweep_task_for_window'):
            self.parent().stop_multisweep_task_for_window(self)
        super().closeEvent(event)

    def _toggle_cf_lines_visibility(self, checked):
        self._redraw_plots()
