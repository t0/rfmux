"""
Export and cable-delay utilities for NetworkAnalysisWindow.

This module defines the NetworkAnalysisExportMixin class, which provides functionality for:
1. Exporting network analysis data to various file formats (pickle, CSV)
2. Managing cable delays and cable length adjustments
3. Handling resonance-related UI updates
4. Configuring and launching multisweep analysis

The mixin is designed to be included in the NetworkAnalysisWindow class to add these
capabilities while maintaining separation of concerns.
"""

from __future__ import annotations
from typing import Tuple, Optional, Union

from .utils import *
from .dialogs import MultisweepDialog


class NetworkAnalysisExportMixin:
    """
    Mixin providing export and cable-delay logic for NetworkAnalysisWindow.
    
    This mixin is designed to be included in the NetworkAnalysisWindow class to add
    capabilities for exporting data and managing cable delay configuration. It assumes
    the host class provides various properties and UI elements related to network analysis.
    
    Requirements from the host class:
    - netanal_traces: module -> probe amplitude -> the trace take_netanal measured
    - current_params: Dictionary of current analysis parameters
    - resonance_freqs: Dictionary of resonance frequencies per module
    - plots: Dictionary of plot information per module
    - module_cable_lengths: Dictionary of cable lengths per module
    - cable_length_spin: QDoubleSpinBox for cable length adjustment
    - tabs: QTabWidget containing module tabs
    - modules: List of module identifiers
    - take_multisweep_btn: QPushButton for taking multisweep
    """

    #
    # 1. Data Export Methods
    #

    def _export_data(self) -> None:
        """
        Export the collected data with all unit conversions and metadata.
        
        This method initiates a non-blocking file dialog to export network analysis data
        to either pickle or CSV format. Before showing the dialog, it ensures GUI responsiveness
        by pausing any live updates and disabling plot updates.
        """
        # Thread marshalling - ensure we're on the main GUI thread
        if QtCore.QThread.currentThread() != QtWidgets.QApplication.instance().thread():
            QtCore.QMetaObject.invokeMethod(
                self, 
                "_export_data", 
                QtCore.Qt.ConnectionType.QueuedConnection
            )
            return
            
        if not self.netanal_traces:
            QtWidgets.QMessageBox.warning(self, "No Data", "No data to export yet.")
            return
        
        # 1. Pause any live updates - pause the parent's timer if it exists
        self._timer_was_active = False
        if hasattr(self.parent(), 'timer') and self.parent().timer.isActive():
            self.parent().timer.stop()
            self._timer_was_active = True
            
        # 2. Disable updates on graphics views if they exist
        if hasattr(self, 'plots'):
            for module_plots in self.plots.values():
                for plot_type in ['mag_plot', 'phase_plot']:
                    if plot_type in module_plots and hasattr(module_plots[plot_type], 'setUpdatesEnabled'):
                        module_plots[plot_type].setUpdatesEnabled(False)
        
        # 3. Create a non-blocking file dialog
        dlg = QtWidgets.QFileDialog(self, "Export Data")
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setNameFilters(["Pickle Files (*.pkl)", "CSV Files (*.csv)", "All Files (*)"])
        dlg.setDefaultSuffix("pkl")
        
        # 4. Connect signals for handling dialog completion
        dlg.fileSelected.connect(self._handle_export_file_selected)
        dlg.finished.connect(self._resume_updates_after_export_dialog)
        
        # 5. Show the dialog non-modally
        dlg.open()  # Returns immediately, doesn't block
    
    def _resume_updates_after_export_dialog(self, result: int) -> None:
        """
        Resume updates after export dialog closes, regardless of whether a file was selected.
        
        Args:
            result: The dialog result code (unused but required for signal connection)
        """
        # Re-enable updates on graphics views
        if hasattr(self, 'plots'):
            for module_plots in self.plots.values():
                for plot_type in ['mag_plot', 'phase_plot']:
                    if plot_type in module_plots and hasattr(module_plots[plot_type], 'setUpdatesEnabled'):
                        module_plots[plot_type].setUpdatesEnabled(True)
        
        # Restart the timer if it was active
        if hasattr(self, '_timer_was_active') and self._timer_was_active and hasattr(self.parent(), 'timer'):
            self.parent().timer.start()
    
    def _handle_export_file_selected(self, filename: str) -> None:
        """
        Handle the file selection from the non-blocking dialog.
        
        Args:
            filename: The path to the file selected by the user
        """
        if not filename:
            return
            
        try:
            if filename.endswith('.pkl'):
                self._export_to_pickle(filename)
            elif filename.endswith('.csv'):
                self._export_to_csv(filename)
            else:
                self._export_to_pickle(filename)
            
            QtWidgets.QMessageBox.information(
                self, 
                "Export Complete", 
                f"Data exported to {filename}"
            )
        except Exception as e:
            traceback.print_exc() 
            QtWidgets.QMessageBox.critical(
                self, 
                "Export Error", 
                f"Error exporting data: {str(e)}"
            )
    
    def build_export_dict(self) -> dict:
        """
        Build the export data dictionary with comprehensive metadata.
        
        This method creates a hierarchical dictionary structure containing all measurement
        data, parameters, and module information. Can be used for both file export and
        session auto-export.
        
        Returns:
            Dictionary containing all export data
        """
        export_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'parameters': self.current_params.copy() if hasattr(self, 'current_params') else {},
            'dac_scales_used': self.dac_scales.copy() if hasattr(self, 'dac_scales') else {},
            'modules': {}
        }
        
        for module, traces in self.netanal_traces.items():
            export_data['modules'][module] = {}
            meas_idx = 0

            for amplitude, trace in traces.items():
                freqs, iq_data = trace['frequencies'], trace['iq_counts']
                magnitude = np.abs(iq_data)

                def in_units(unit_mode, normalize=False):
                    return UnitConverter.convert_amplitude(
                        magnitude, iq_data, unit_mode=unit_mode, normalize=normalize
                    ).tolist()

                export_data['modules'][module][meas_idx] = {
                    'sweep_amplitude': amplitude,
                    'frequency': {'values': freqs.tolist(), 'unit': 'Hz'},
                    'magnitude': {
                        'counts': {'raw': in_units("counts"),
                                   'normalized': in_units("counts", True),
                                   'unit': 'counts'},
                        'volts': {'raw': in_units("volts"),
                                  'normalized': in_units("volts", True),
                                  'unit': 'V'},
                        'dbm': {'raw': in_units("dbm"),
                                'normalized': in_units("dbm", True),
                                'unit': 'dBm'},
                    },
                    'phase': {'values': np.degrees(np.angle(iq_data)).tolist(),
                              'unit': 'degrees'},
                    'complex': {'real': iq_data.real.tolist(),
                                'imag': iq_data.imag.tolist()},
                }
                meas_idx += 1
            
            # Include resonance frequencies for the module
            export_data['modules'][module]['resonances_hz'] = self.resonance_freqs.get(module, [])
        
        return export_data
    
    def _export_to_pickle(self, filename: str) -> None:
        """
        Export data to a pickle file with comprehensive metadata.
        
        Uses build_export_dict() to create the data structure, then saves to file.
        
        Args:
            filename: The path to the pickle file to create
        """
        export_data = self.build_export_dict()
        
        # Write the data to file
        with open(filename, 'wb') as f:
            pickle.dump(export_data, f)
    
    def _export_to_csv(self, filename: str) -> None:
        """
        Export data to CSV files, creating multiple files as needed.
        
        This method creates:
        1. A metadata CSV file with measurement parameters
        2. Multiple CSV files (one per module/amplitude) containing measurement data
        
        Args:
            filename: The path to use as the base filename for the CSV files
        """
        # Create metadata file
        base, ext = os.path.splitext(filename)
        meta_filename = f"{base}_metadata{ext}"
        
        # Write metadata
        with open(meta_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Parameter', 'Value'])
            writer.writerow(['Export Date', datetime.datetime.now().isoformat()])
            
            if hasattr(self, 'current_params'):
                writer.writerow(['', ''])
                writer.writerow(['Measurement Parameters', ''])
                
                for param, value in self.current_params.items():
                    if param in ['fmin', 'fmax', 'max_span'] and isinstance(value, (int, float)):
                        writer.writerow([param, f"{value/1e6} MHz"])
                    else:
                        writer.writerow([param, value])
            
            if self.resonance_freqs:
                writer.writerow(['', ''])
                writer.writerow(['Resonances (Hz)', ''])
                
                for module, freqs in self.resonance_freqs.items():
                    writer.writerow([f'Module {module}', ','.join(map(str, freqs))])
        
        # Write data files - one per module/sweep/unit
        for module, traces in self.netanal_traces.items():
            idx = 0
            for amplitude, trace in traces.items():
                freqs, iq_data = trace['frequencies'], trace['iq_counts']
                magnitude = np.abs(iq_data)
                phases = np.degrees(np.angle(iq_data))

                # Export data in different unit modes
                for unit_mode in ["counts", "volts", "dbm"]:
                    converted_amps = UnitConverter.convert_amplitude(magnitude, iq_data, unit_mode=unit_mode)
                    unit_label = "dBm" if unit_mode == "dbm" else ("V" if unit_mode == "volts" else unit_mode)
                    
                    # Create CSV filename for this specific data
                    csv_filename = f"{base}_module{module}_idx{idx}_{unit_mode}{ext}"
                    
                    with open(csv_filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['# Amplitude:', f"{amplitude}"])
                        
                        # Determine column header based on unit mode
                        header = [
                            'Frequency (Hz)', 
                            f'Power ({unit_label})' if unit_mode == "dbm" else f'Amplitude ({unit_label})', 
                            'Phase (deg)'
                        ]
                        writer.writerow(header)
                        
                        # Write data rows
                        for freq, amp, phase in zip(freqs, converted_amps, phases):
                            writer.writerow([freq, amp, phase])
                
                idx += 1
    
    def _unwrap_cable_delay_action(self) -> None:
        """
        Fit the phase data of the first curve in the active module's plot,
        calculate the corresponding cable length, update the phase curves,
        and adjust the cable length spinner.
        
        This method:
        1. Gets the active module and its data
        2. Fits the cable delay using phase information
        3. Calculates a new cable length
        4. Updates the phase plots with adjusted phase values
        5. Updates the cable length spinner
        """
        # Get the active module
        active_module = self._get_active_module()
        if active_module is None:
            return

        block = self._sweep_for_cable_delay(active_module)
        if block is None:
            return
        freqs_active = trace['frequencies']
        phases_displayed_active_deg = np.degrees(np.angle(trace['iq_counts']))

        # Calculate new cable length
        L_old_physical, L_new_physical = self._calculate_cable_length(active_module, freqs_active, phases_displayed_active_deg)
        if L_new_physical is None:
            return
            
        # Update phase plots
        self._update_phase_plots(active_module, L_old_physical, L_new_physical)
        
        # Update cable length values
        self.module_cable_lengths[active_module] = L_new_physical
        self.cable_length_spin.blockSignals(True)
        self.cable_length_spin.setValue(L_new_physical)
        self.cable_length_spin.blockSignals(False)

    def _get_active_module(self) -> Optional[int]:
        """
        Get the active module from the current tab.
        
        Returns:
            The active module identifier or None if no module is selected
        """
        current_tab_index = self.tabs.currentIndex()
        if current_tab_index < 0:
            QtWidgets.QMessageBox.warning(self, "No Module", "Select a module tab.")
            return None
            
        active_module_text = self.tabs.tabText(current_tab_index)
        try:
            active_module = int(active_module_text.split(" ")[1])
            return active_module
        except (IndexError, ValueError):
            QtWidgets.QMessageBox.critical(
                self, 
                "Error", 
                f"Invalid module tab: {active_module_text}"
            )
            return None

    def _sweep_for_cable_delay(self, active_module: int) -> Optional[dict]:
        """The sweep the delay fit runs on: the weakest probe that was taken.

        The cable's phase slope is the same at every probe power, and the
        weakest sweep is the one least distorted by the resonators.
        """
        traces = self.netanal_traces.get(active_module)
        if not traces:
            QtWidgets.QMessageBox.information(
                self, "No Data", f"No data for Module {active_module}.")
            return None
        for amplitude in self.original_params.get('amps', []):
            if amplitude in traces:
                return traces[amplitude]
        return traces[min(traces)]

    def _calculate_cable_length(
        self, 
        active_module: int, 
        freqs_active: np.ndarray, 
        phases_displayed_active_deg: np.ndarray
    ) -> Tuple[float, Optional[float]]:
        """
        Calculate a new cable length based on phase data.
        
        Args:
            active_module: The module identifier
            freqs_active: Array of frequencies
            phases_displayed_active_deg: Array of phase values in degrees
            
        Returns:
            Tuple of (old_length, new_length) or (old_length, None) if calculation fails
        """
        if len(freqs_active) == 0:
            QtWidgets.QMessageBox.information(
                self, 
                "No Data", 
                "Selected sweep has no frequency data."
            )
            return 0.0, None
            
        L_old_physical = self.current_params.get('cable_length', DEFAULT_CABLE_LENGTH)
        
        try:
            tau_additional = fit_cable_delay(freqs_active, phases_displayed_active_deg)
            L_new_physical = calculate_new_cable_length(L_old_physical, tau_additional)
            return L_old_physical, L_new_physical
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, 
                "Calc Error", 
                f"Cable delay calc error: {str(e)}"
            )
            traceback.print_exc()
            return L_old_physical, None

    def _update_phase_plots(self, active_module: int, L_old_physical: float, L_new_physical: float) -> None:
        """
        Update phase plots with the new cable length.
        
        Args:
            active_module: The module identifier
            L_old_physical: The old cable length in meters
            L_new_physical: The new cable length in meters
        """
        if active_module not in self.plots:
            return

        plot_info = self.plots[active_module]
        traces = self.netanal_traces.get(active_module, {})

        for amplitude, curve_item in plot_info['phase_curves'].items():
            trace = traces.get(amplitude)
            if trace is None or len(trace['frequencies']) == 0:
                continue
            freqs = trace['frequencies']
            phases = recalculate_displayed_phase(
                freqs, np.degrees(np.angle(trace['iq_counts'])),
                L_old_physical, L_new_physical,
            )
            if len(phases) > 0:
                phases = phases - phases[0]
            curve_item.setData(freqs, ((phases + 180) % 360) - 180)

        plot_info['phase_plot'].enableAutoRange(pg.ViewBox.YAxis, True)

    def _on_cable_length_changed(self, new_length: float) -> None:
        """
        Handle changes to the cable length spinner.
        
        Args:
            new_length: The new cable length value in meters
        """
        current_tab_index = self.tabs.currentIndex()
        if current_tab_index < 0 or not self.modules or current_tab_index >= len(self.modules):
            return
            
        active_module_id = self.modules[current_tab_index]
        self.module_cable_lengths[active_module_id] = new_length
        self._update_multisweep_button_state(active_module_id)

    #
    # 3. Module Tab and UI Management Methods
    #

    def _on_active_module_changed(self, index: int) -> None:
        """
        Update UI elements when the active module tab changes.
        
        Args:
            index: The index of the newly selected tab
        """
        if index < 0 or not self.modules or index >= len(self.modules):
            self._update_multisweep_button_state(None)
            return
            
        active_module_id = self.modules[index]
        
        # Update cable length spinner
        if active_module_id in self.module_cable_lengths:
            self.cable_length_spin.blockSignals(True)
            self.cable_length_spin.setValue(self.module_cable_lengths[active_module_id])
            self.cable_length_spin.blockSignals(False)
        else:
            self.cable_length_spin.blockSignals(True)
            self.cable_length_spin.setValue(self.current_params.get('cable_length', DEFAULT_CABLE_LENGTH))
            self.cable_length_spin.blockSignals(False)
            
        self._update_multisweep_button_state(active_module_id)

    def _update_multisweep_button_state(self, module_id: Optional[int] = None) -> None:
        """
        Enable or disable the Take Multisweep button based on found resonances for the given module.
        
        Args:
            module_id: The module identifier or None to use the currently active module
        """
        if not hasattr(self, 'take_multisweep_btn'):
            return
            
        # Get module_id from current tab if not provided
        if module_id is None:
            current_tab_index = self.tabs.currentIndex()
            if current_tab_index < 0:
                self.take_multisweep_btn.setEnabled(False)
                return
                
            active_module_text = self.tabs.tabText(current_tab_index)
            try:
                module_id = int(active_module_text.split(" ")[1])
            except (IndexError, ValueError):
                self.take_multisweep_btn.setEnabled(False)
                return
        
        # Enable button if module has resonances
        has_resonances = bool(self.resonance_freqs.get(module_id))
        self.take_multisweep_btn.setEnabled(has_resonances)

    #
    # 4. Multisweep Dialog Management
    #

    def _show_multisweep_dialog(self) -> None:
        """
        Show the dialog to configure and run multisweep analysis.
        
        This method:
        1. Gets the active module and its resonance frequencies
        2. Sets up the multisweep dialog with appropriate parameters
        3. Launches the multisweep analysis if the user accepts the dialog
        """
        # Get active module
        active_module = self._get_active_module()
        if active_module is None:
            return
            
        # Check if the module has resonances
        resonances = self.resonance_freqs.get(active_module, [])
        if not resonances:
            QtWidgets.QMessageBox.information(
                self, 
                "No Resonances", 
                f"No resonances for Module {active_module}. Run 'Find Resonances'."
            )
            return
        
        # Walk up parent hierarchy to find Periscope instance
        # (panel may be wrapped in QDockWidget, so parent() might not be Periscope directly)
        periscope_parent = find_parent_with_attr(self, 'dac_scales')
        
        # Get DAC scales from the Periscope instance
        dac_scales_for_dialog = {}
        if periscope_parent and hasattr(periscope_parent, 'dac_scales'):
            dac_scales_for_dialog = periscope_parent.dac_scales
        elif hasattr(self, 'dac_scales'):
            dac_scales_for_dialog = self.dac_scales
        
        # Create and show the dialog
        dialog = MultisweepDialog(
            parent=self, 
            section_center_frequencies=resonances, 
            dac_scales=dac_scales_for_dialog, 
            current_module=active_module
        )
        
        # Process dialog result
        if dialog.exec():
            params = dialog.get_parameters()
            if not params:
                return
                
            # Find Periscope parent (walk up hierarchy if needed)
            parent = find_parent_with_attr(self, '_start_multisweep_analysis')
            
            if parent:
                try:
                    parent._start_multisweep_analysis(params)
                except Exception as e:
                    error_msg = f"Error starting multisweep: {type(e).__name__}: {str(e)}"
                    print(error_msg, file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    QtWidgets.QMessageBox.critical(self, "Multisweep Error", error_msg)
            else:
                error_msg = "Cannot start multisweep: Parent integration missing (could not find Periscope parent)"
                print(f"ERROR: {error_msg}", file=sys.stderr)
                traceback.print_stack(file=sys.stderr)
                QtWidgets.QMessageBox.critical(self, "Error", error_msg)
