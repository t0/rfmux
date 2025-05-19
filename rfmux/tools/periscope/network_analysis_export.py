"""Export and cable-delay utilities for NetworkAnalysisWindow."""

from .utils import *
from .dialogs import MultisweepDialog

class NetworkAnalysisExportMixin:
    """Mixin providing export and cable-delay logic."""
        def _export_data(self):
            """Export the collected data with all unit conversions and metadata."""
            if not self.data:
                QtWidgets.QMessageBox.warning(self, "No Data", "No data to export yet.")
                return
            
            dialog = QtWidgets.QFileDialog(self)
            dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
            dialog.setNameFilters(["Pickle Files (*.pkl)", "CSV Files (*.csv)", "All Files (*)"])
            dialog.setDefaultSuffix("pkl")
            
            if dialog.exec():
                filename = dialog.selectedFiles()[0]
                
                try:
                    if filename.endswith('.pkl'): self._export_to_pickle(filename)
                    elif filename.endswith('.csv'): self._export_to_csv(filename)
                    else: self._export_to_pickle(filename)
                    QtWidgets.QMessageBox.information(self, "Export Complete", f"Data exported to {filename}")
                except Exception as e:
                    traceback.print_exc() 
                    QtWidgets.QMessageBox.critical(self, "Export Error", f"Error exporting data: {str(e)}")
        
        def _export_to_pickle(self, filename):
            """Export data to a pickle file."""
            export_data = {'timestamp': datetime.datetime.now().isoformat(),
                           'parameters': self.current_params.copy() if hasattr(self, 'current_params') else {},
                           'modules': {}}
            
            for module, data_dict in self.raw_data.items():
                export_data['modules'][module] = {}; meas_idx = 0
                for key, data_tuple in data_dict.items():
                    amplitude, freqs, amps, phases, iq_data = self._extract_data_for_export(key, data_tuple)
                    counts = amps; volts = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="volts")
                    dbm = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="dbm")
                    counts_norm = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="counts", normalize=True)
                    volts_norm = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="volts", normalize=True)
                    dbm_norm = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="dbm", normalize=True)
                    
                    export_data['modules'][module][meas_idx] = {
                        'sweep_amplitude': amplitude,
                        'frequency': {'values': freqs.tolist(), 'unit': 'Hz'},
                        'magnitude': {'counts': {'raw': counts.tolist(), 'normalized': counts_norm.tolist(), 'unit': 'counts'},
                                      'volts': {'raw': volts.tolist(), 'normalized': volts_norm.tolist(), 'unit': 'V'},
                                      'dbm': {'raw': dbm.tolist(), 'normalized': dbm_norm.tolist(), 'unit': 'dBm'}},
                        'phase': {'values': phases.tolist(), 'unit': 'degrees'},
                        'complex': {'real': iq_data.real.tolist(), 'imag': iq_data.imag.tolist()}}
                    meas_idx += 1
                export_data['modules'][module]['resonances_hz'] = self.resonance_freqs.get(module, [])
            
            with open(filename, 'wb') as f: pickle.dump(export_data, f)
        
        def _export_to_csv(self, filename):
            """Export data to CSV files."""
            base, ext = os.path.splitext(filename)
            meta_filename = f"{base}_metadata{ext}"
            with open(meta_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Parameter', 'Value']); writer.writerow(['Export Date', datetime.datetime.now().isoformat()])
                if hasattr(self, 'current_params'):
                    writer.writerow(['', '']); writer.writerow(['Measurement Parameters', ''])
                    for param, value in self.current_params.items():
                        if param in ['fmin', 'fmax', 'max_span'] and isinstance(value, (int, float)):
                            writer.writerow([param, f"{value/1e6} MHz"])
                        else: writer.writerow([param, value])
                if self.resonance_freqs:
                    writer.writerow(['', '']); writer.writerow(['Resonances (Hz)', ''])
                    for module, freqs in self.resonance_freqs.items():
                        writer.writerow([f'Module {module}', ','.join(map(str, freqs))])
            
            for module, data_dict in self.raw_data.items():
                idx = 0 
                for key, data_tuple in data_dict.items():
                    amplitude, freqs, amps, phases, iq_data = self._extract_data_for_export(key, data_tuple)
                    for unit_mode in ["counts", "volts", "dbm"]:
                        converted_amps = UnitConverter.convert_amplitude(amps, iq_data, unit_mode=unit_mode)
                        unit_label = "dBm" if unit_mode == "dbm" else ("V" if unit_mode == "volts" else unit_mode)
                        csv_filename = f"{base}_module{module}_idx{idx}_{unit_mode}{ext}"
                        with open(csv_filename, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(['# Amplitude:', f"{amplitude}" if 'amplitude' in locals() else "Unknown"])
                            header = ['Frequency (Hz)', f'Power ({unit_label})' if unit_mode == "dbm" else f'Amplitude ({unit_label})', 'Phase (deg)']
                            writer.writerow(header)
                            for freq, amp, phase in zip(freqs, converted_amps, phases):
                                writer.writerow([freq, amp, phase])
                    idx += 1
        
        def _extract_data_for_export(self, key, data_tuple):
            """Extract and prepare data for export from a data tuple."""
            amplitude = DEFAULT_AMPLITUDE 
            if key != 'default':
                if len(data_tuple) >= 5: freqs, amps, phases, iq_data, amplitude = data_tuple
                else:
                    freqs, amps, phases, iq_data = data_tuple
                    try: amplitude = float(key.split('_')[-1])
                    except (ValueError, IndexError): pass
            else: freqs, amps, phases, iq_data = data_tuple
            return amplitude, freqs, amps, phases, iq_data
    
        def _unwrap_cable_delay_action(self):
            """
            Fits the phase data of the first curve in the active module's plot,
            calculates the corresponding cable length, updates the phase curves,
            and adjusts the cable length spinner.
            """
            if not self.raw_data: QtWidgets.QMessageBox.information(self, "No Data", "No data to process."); return
            current_tab_index = self.tabs.currentIndex()
            if current_tab_index < 0: QtWidgets.QMessageBox.warning(self, "No Module", "Select a module tab."); return
            active_module_text = self.tabs.tabText(current_tab_index)
            try: active_module = int(active_module_text.split(" ")[1])
            except (IndexError, ValueError): QtWidgets.QMessageBox.critical(self, "Error", f"Invalid module tab: {active_module_text}"); return
            if active_module not in self.raw_data or not self.raw_data[active_module]:
                QtWidgets.QMessageBox.information(self, "No Data", f"No data for Module {active_module}."); return
            module_data_dict = self.raw_data[active_module]; target_key = None
            if 'amps' in self.original_params and self.original_params['amps']:
                first_amplitude_setting = self.original_params['amps'][0]
                potential_key = f"{active_module}_{first_amplitude_setting}"
                if potential_key in module_data_dict: target_key = potential_key
            if target_key is None and 'default' in module_data_dict: target_key = 'default'
            if target_key is None:
                if module_data_dict: target_key = next(iter(module_data_dict))
                else: QtWidgets.QMessageBox.information(self, "No Data", f"No sweep data for Module {active_module}."); return
            data_tuple = module_data_dict[target_key]
            if len(data_tuple) == 5: freqs_active, _, phases_displayed_active_deg, _, _ = data_tuple
            elif len(data_tuple) == 4: freqs_active, _, phases_displayed_active_deg, _ = data_tuple
            else: QtWidgets.QMessageBox.critical(self, "Error", "Unexpected data format."); return
            if len(freqs_active) == 0: QtWidgets.QMessageBox.information(self, "No Data", "Selected sweep has no frequency data."); return
            L_old_physical = self.current_params.get('cable_length', DEFAULT_CABLE_LENGTH)
            try:
                tau_additional = fit_cable_delay(freqs_active, phases_displayed_active_deg)
                L_new_physical = calculate_new_cable_length(L_old_physical, tau_additional)
            except Exception as e: QtWidgets.QMessageBox.critical(self, "Calc Error", f"Cable delay calc error: {str(e)}"); traceback.print_exc(); return
            if active_module in self.plots:
                plot_info = self.plots[active_module]
                for amp_key_iter, curve_item in plot_info['phase_curves'].items():
                    raw_data_key_for_curve = f"{active_module}_{amp_key_iter}"
                    if raw_data_key_for_curve in module_data_dict:
                        data_tuple_curve = module_data_dict[raw_data_key_for_curve]
                        if len(data_tuple_curve) == 5: freqs_curve, _, phases_deg_current_display_curve, _, _ = data_tuple_curve
                        elif len(data_tuple_curve) == 4: freqs_curve, _, phases_deg_current_display_curve, _ = data_tuple_curve
                        else: continue
                        if len(freqs_curve) > 0:
                            new_phases_deg_for_curve = recalculate_displayed_phase(freqs_curve, phases_deg_current_display_curve, L_old_physical, L_new_physical)
                            if len(new_phases_deg_for_curve) > 0:
                                first_point_phase = new_phases_deg_for_curve[0]
                                new_phases_deg_for_curve = new_phases_deg_for_curve - first_point_phase
                            new_phases_deg_for_curve = ((new_phases_deg_for_curve + 180) % 360) - 180
                            curve_item.setData(freqs_curve, new_phases_deg_for_curve)
                if not plot_info['phase_curves'] and 'default' in module_data_dict:
                    main_curve_item = plot_info['phase_curve']
                    data_tuple_main = module_data_dict['default']
                    if len(data_tuple_main) == 4:
                        freqs_main, _, phases_deg_current_display_main, _ = data_tuple_main
                        if len(freqs_main) > 0:
                            new_phases_deg_for_main = recalculate_displayed_phase(freqs_main, phases_deg_current_display_main, L_old_physical, L_new_physical)
                            if len(new_phases_deg_for_main) > 0:
                                first_point_phase = new_phases_deg_for_main[0]
                                new_phases_deg_for_main = new_phases_deg_for_main - first_point_phase
                            new_phases_deg_for_main = ((new_phases_deg_for_main + 180) % 360) - 180
                            main_curve_item.setData(freqs_main, new_phases_deg_for_main)
                plot_info['phase_plot'].enableAutoRange(pg.ViewBox.YAxis, True)
            self.module_cable_lengths[active_module] = L_new_physical
            self.cable_length_spin.blockSignals(True); self.cable_length_spin.setValue(L_new_physical); self.cable_length_spin.blockSignals(False)
    
        def _on_active_module_changed(self, index: int):
            """Update UI elements when the active module tab changes."""
            if index < 0 or not self.modules or index >= len(self.modules): self._update_multisweep_button_state(None); return
            active_module_id = self.modules[index]
            if active_module_id in self.module_cable_lengths:
                self.cable_length_spin.blockSignals(True); self.cable_length_spin.setValue(self.module_cable_lengths[active_module_id]); self.cable_length_spin.blockSignals(False)
            else:
                self.cable_length_spin.blockSignals(True); self.cable_length_spin.setValue(self.current_params.get('cable_length', DEFAULT_CABLE_LENGTH)); self.cable_length_spin.blockSignals(False)
            self._update_multisweep_button_state(active_module_id)
    
        def _on_cable_length_changed(self, new_length: float):
            """Handle changes to the cable length spinner."""
            current_tab_index = self.tabs.currentIndex()
            if current_tab_index < 0 or not self.modules or current_tab_index >= len(self.modules): return
            active_module_id = self.modules[current_tab_index]
            self.module_cable_lengths[active_module_id] = new_length
            self._update_multisweep_button_state(active_module_id)
    
        def _update_multisweep_button_state(self, module_id: int | None = None):
            """Enable or disable the Take Multisweep button based on found resonances for the given module."""
            if not hasattr(self, 'take_multisweep_btn'): return
            if module_id is None:
                current_tab_index = self.tabs.currentIndex()
                if current_tab_index < 0: self.take_multisweep_btn.setEnabled(False); return
                active_module_text = self.tabs.tabText(current_tab_index)
                try: module_id = int(active_module_text.split(" ")[1])
                except (IndexError, ValueError): self.take_multisweep_btn.setEnabled(False); return
            has_resonances = bool(self.resonance_freqs.get(module_id))
            self.take_multisweep_btn.setEnabled(has_resonances)
    
        def _show_multisweep_dialog(self):
            """Show the dialog to configure and run multisweep."""
            current_tab_index = self.tabs.currentIndex()
            if current_tab_index < 0: QtWidgets.QMessageBox.warning(self, "No Module", "Select a module tab."); return
            active_module_text = self.tabs.tabText(current_tab_index)
            try: active_module = int(active_module_text.split(" ")[1])
            except (IndexError, ValueError): QtWidgets.QMessageBox.critical(self, "Error", f"Invalid module tab: {active_module_text}"); return
            resonances = self.resonance_freqs.get(active_module, [])
            if not resonances: QtWidgets.QMessageBox.information(self, "No Resonances", f"No resonances for Module {active_module}. Run 'Find Resonances'."); return
            dac_scales_for_dialog = {}
            if hasattr(self.parent(), 'dac_scales'): dac_scales_for_dialog = self.parent().dac_scales
            elif hasattr(self, 'dac_scales'): dac_scales_for_dialog = self.dac_scales
            
            dialog = MultisweepDialog(parent=self, resonance_frequencies=resonances, dac_scales=dac_scales_for_dialog, current_module=active_module)
            if dialog.exec():
                params = dialog.get_parameters()
                if params and hasattr(self.parent(), '_start_multisweep_analysis'):
                    self.parent()._start_multisweep_analysis(params)
                elif not hasattr(self.parent(), '_start_multisweep_analysis'):
                     QtWidgets.QMessageBox.critical(self, "Error", "Cannot start multisweep: Parent integration missing.")
