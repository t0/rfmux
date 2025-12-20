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
from typing import Any, Dict, List, Tuple, Optional, Union, cast

from .utils import *
from .dialogs import MultisweepDialog
from .tasks import SetCableLengthTask, SetCableLengthSignals


class NetworkAnalysisExportMixin:
    """
    Mixin providing export and cable-delay logic for NetworkAnalysisWindow.
    
    This mixin is designed to be included in the NetworkAnalysisWindow class to add
    capabilities for exporting data and managing cable delay configuration. It assumes
    the host class provides various properties and UI elements related to network analysis.
    
    Requirements from the host class:
    - data: Dictionary of data to export
    - raw_data: Dictionary of raw data including IQ information
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
            
        if not self.data:
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
        
        for module, data_dict in self.raw_data.items():
            export_data['modules'][module] = {}
            meas_idx = 0
            
            for key, data_tuple in data_dict.items():
                # Extract and convert data for export
                amplitude, freqs, amps, phases, iq_data = self._extract_data_for_export(key, data_tuple)
                
                # Perform unit conversions
                counts = amps
                volts = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="volts")
                dbm = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="dbm")
                counts_norm = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="counts", normalize=True)
                volts_norm = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="volts", normalize=True)
                dbm_norm = UnitConverter.convert_amplitude(amps, iq_data, unit_mode="dbm", normalize=True)
                
                # Store converted data
                export_data['modules'][module][meas_idx] = {
                    'sweep_amplitude': amplitude,
                    'frequency': {'values': freqs.tolist(), 'unit': 'Hz'},
                    'magnitude': {
                        'counts': {
                            'raw': counts.tolist(), 
                            'normalized': counts_norm.tolist(), 
                            'unit': 'counts'
                        },
                        'volts': {
                            'raw': volts.tolist(), 
                            'normalized': volts_norm.tolist(), 
                            'unit': 'V'
                        },
                        'dbm': {
                            'raw': dbm.tolist(), 
                            'normalized': dbm_norm.tolist(), 
                            'unit': 'dBm'
                        }
                    },
                    'phase': {'values': phases.tolist(), 'unit': 'degrees'},
                    'complex': {'real': iq_data.real.tolist(), 'imag': iq_data.imag.tolist()}
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
        for module, data_dict in self.raw_data.items():
            idx = 0
            for key, data_tuple in data_dict.items():
                amplitude, freqs, amps, phases, iq_data = self._extract_data_for_export(key, data_tuple)
                
                # Export data in different unit modes
                for unit_mode in ["counts", "volts", "dbm"]:
                    converted_amps = UnitConverter.convert_amplitude(amps, iq_data, unit_mode=unit_mode)
                    unit_label = "dBm" if unit_mode == "dbm" else ("V" if unit_mode == "volts" else unit_mode)
                    
                    # Create CSV filename for this specific data
                    csv_filename = f"{base}_module{module}_idx{idx}_{unit_mode}{ext}"
                    
                    with open(csv_filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['# Amplitude:', f"{amplitude}" if 'amplitude' in locals() else "Unknown"])
                        
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
    
    def _extract_data_for_export(
        self, 
        key: str, 
        data_tuple: Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], 
                         Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract and prepare data for export from a data tuple.
        
        Args:
            key: The key identifying the data within the raw_data dictionary
            data_tuple: Tuple containing frequency, amplitude, phase, and IQ data,
                       optionally with amplitude value
        
        Returns:
            Tuple containing (amplitude, frequencies, amplitudes, phases, iq_data)
        """
        # Default amplitude if not available in the data
        amplitude = DEFAULT_AMPLITUDE 
        
        if key != 'default':
            # Extract data from the tuple, handling different tuple formats
            if len(data_tuple) >= 5:
                freqs, amps, phases, iq_data, amplitude = data_tuple
            else:
                freqs, amps, phases, iq_data = data_tuple
                # Try to extract amplitude from the key
                try:
                    amplitude = float(key.split('_')[-1])
                except (ValueError, IndexError):
                    pass
        else:
            # Default data format
            freqs, amps, phases, iq_data = data_tuple
            
        return amplitude, freqs, amps, phases, iq_data

    #
    # 2. Cable Delay Management Methods
    #

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
        6. Sets the cable length on the CRS hardware
        """
        # Check if there's data to process
        if not self.raw_data:
            QtWidgets.QMessageBox.information(self, "No Data", "No data to process.")
            return
            
        # Get the active module
        active_module = self._get_active_module()
        if active_module is None:
            return
            
        # Get data for the active module
        data_tuple = self._get_module_data_for_cable_delay(active_module)
        if data_tuple is None:
            return
            
        # Extract frequency and phase data
        freqs_active, phases_displayed_active_deg = self._extract_freq_and_phase(data_tuple)
        if freqs_active is None or phases_displayed_active_deg is None:
            return
            
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

        # Set cable length on CRS hardware
        self._set_cable_length_on_crs(active_module, L_new_physical)

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

    def _get_module_data_for_cable_delay(self, active_module: int) -> Optional[Tuple]:
        """
        Get the appropriate data tuple for cable delay calculation.
        
        Args:
            active_module: The module identifier
            
        Returns:
            The data tuple to use for cable delay calculation or None if no suitable data is found
        """
        if active_module not in self.raw_data or not self.raw_data[active_module]:
            QtWidgets.QMessageBox.information(
                self, 
                "No Data", 
                f"No data for Module {active_module}."
            )
            return None
            
        module_data_dict = self.raw_data[active_module]
        target_key = None
        
        # Try to get data for the first amplitude setting
        if 'amps' in self.original_params and self.original_params['amps']:
            first_amplitude_setting = self.original_params['amps'][0]
            potential_key = f"{active_module}_{first_amplitude_setting}"
            if potential_key in module_data_dict:
                target_key = potential_key
                
        # Fallback to default if available
        if target_key is None and 'default' in module_data_dict:
            target_key = 'default'
            
        # Last resort: take the first key available
        if target_key is None:
            if module_data_dict:
                target_key = next(iter(module_data_dict))
            else:
                QtWidgets.QMessageBox.information(
                    self, 
                    "No Data", 
                    f"No sweep data for Module {active_module}."
                )
                return None
                
        return module_data_dict[target_key]

    def _extract_freq_and_phase(self, data_tuple: Tuple) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract frequency and phase data from a data tuple.
        
        Args:
            data_tuple: The data tuple containing measurement results
            
        Returns:
            Tuple of (frequencies, phases) arrays or (None, None) if data format is unexpected
        """
        if len(data_tuple) == 5:
            freqs_active, _, phases_displayed_active_deg, _, _ = data_tuple
            return freqs_active, phases_displayed_active_deg
        elif len(data_tuple) == 4:
            freqs_active, _, phases_displayed_active_deg, _ = data_tuple
            return freqs_active, phases_displayed_active_deg
        else:
            QtWidgets.QMessageBox.critical(self, "Error", "Unexpected data format.")
            return None, None

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
        module_data_dict = self.raw_data[active_module]
        
        # Update amplitude sweep curves if they exist
        for amp_key_iter, curve_item in plot_info['phase_curves'].items():
            raw_data_key_for_curve = f"{active_module}_{amp_key_iter}"
            
            if raw_data_key_for_curve in module_data_dict:
                self._update_individual_phase_curve(
                    module_data_dict[raw_data_key_for_curve],
                    curve_item,
                    L_old_physical,
                    L_new_physical
                )
        
        # Update default curve if no amplitude sweep curves exist
        if not plot_info['phase_curves'] and 'default' in module_data_dict:
            main_curve_item = plot_info['phase_curve']
            data_tuple_main = module_data_dict['default']
            
            if len(data_tuple_main) == 4:
                self._update_individual_phase_curve(
                    data_tuple_main,
                    main_curve_item,
                    L_old_physical,
                    L_new_physical
                )
        
        # Enable autorange for phase plot Y axis
        plot_info['phase_plot'].enableAutoRange(pg.ViewBox.YAxis, True)

    def _update_individual_phase_curve(
        self, 
        data_tuple: Tuple, 
        curve_item: pg.PlotDataItem,
        L_old_physical: float, 
        L_new_physical: float
    ) -> None:
        """
        Update an individual phase curve with recalculated phase values.
        
        Args:
            data_tuple: The data tuple containing frequency and phase values
            curve_item: The plot curve item to update
            L_old_physical: The old cable length in meters
            L_new_physical: The new cable length in meters
        """
        if len(data_tuple) >= 4:  # Ensure we have at least 4 elements
            if len(data_tuple) == 5:
                freqs_curve, _, phases_deg_current_display_curve, _, _ = data_tuple
            else:  # len == 4
                freqs_curve, _, phases_deg_current_display_curve, _ = data_tuple
                
            if len(freqs_curve) > 0:
                # Recalculate phase with new cable length
                new_phases_deg_for_curve = recalculate_displayed_phase(
                    freqs_curve, 
                    phases_deg_current_display_curve, 
                    L_old_physical, 
                    L_new_physical
                )
                
                if len(new_phases_deg_for_curve) > 0:
                    # Normalize the phase to start from zero
                    first_point_phase = new_phases_deg_for_curve[0]
                    new_phases_deg_for_curve = new_phases_deg_for_curve - first_point_phase
                
                # Wrap phase to [-180, 180] range
                new_phases_deg_for_curve = ((new_phases_deg_for_curve + 180) % 360) - 180
                
                # Update the curve
                curve_item.setData(freqs_curve, new_phases_deg_for_curve)

    def _set_cable_length_on_crs(self, active_module: int, new_length: float) -> None:
        """
        Set the cable length on the CRS hardware.
        
        Args:
            active_module: The module identifier
            new_length: The new cable length in meters
        """
        # Ensure the main application has the crs object and thread pool
        main_app = self.window()
        if hasattr(main_app, 'crs') and main_app.crs is not None and \
           hasattr(main_app, 'pool') and main_app.pool is not None:
            
            # Ensure signals for this task are initialized
            if not hasattr(self, 'set_cable_length_signals'):
                self.set_cable_length_signals = SetCableLengthSignals()

            # Create and start the task
            set_length_task = SetCableLengthTask(
                crs=main_app.crs,
                module_id=active_module,
                length=new_length,
                signals=self.set_cable_length_signals 
            )
            main_app.pool.start(set_length_task)
        else:
            QtWidgets.QMessageBox.warning(
                self, 
                "CRS Error", 
                "Could not send set_cable_length command: CRS or thread pool not available from parent."
            )

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
            resonance_frequencies=resonances, 
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
