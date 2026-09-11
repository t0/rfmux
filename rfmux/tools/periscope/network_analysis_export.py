"""
Save and cable-delay utilities for NetworkAnalysisWindow.

This module defines the NetworkAnalysisExportMixin class, which provides functionality for:
1. Saving the measurement through rfmux.tuning.store
2. Managing cable delays and cable length adjustments
3. Handling resonance-related UI updates
4. Configuring and launching multisweep analysis

The mixin is designed to be included in the NetworkAnalysisWindow class to add these
capabilities while maintaining separation of concerns.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Union

from .utils import *
from .dialogs import MultisweepDialog
from ...tuning import store


class NetworkAnalysisExportMixin:
    """
    Mixin providing save and cable-delay logic for NetworkAnalysisWindow.
    
    This mixin is designed to be included in the NetworkAnalysisWindow class to add
    capabilities for saving data and managing cable delay configuration. It assumes
    the host class provides various properties and UI elements related to network analysis.
    
    Requirements from the host class:
    - netanal_traces: module -> the trace take_netanal measured
    - netanal_container: the modules' outputs, keyed by module identifier
    - current_params: Dictionary of current analysis parameters
    - resonance_searches: module -> the ResonanceSearch that names its resonances
    - plots: Dictionary of plot information per module
    - module_cable_lengths: Dictionary of cable lengths per module
    - cable_length_spin: QDoubleSpinBox for cable length adjustment
    - tabs: QTabWidget containing module tabs
    - modules: List of module identifiers
    - take_multisweep_btn: QPushButton for taking multisweep
    """

    #
    # 1. Saving
    #

    def save_netanal(self) -> Optional[Path]:
        """Write the measurement through ``store``, and return where it went.

        One file for however many modules the panel ran, keyed by module
        identifier the way a driver's return is, so it opens in a notebook with
        ``store.load``. Saving the same panel twice overwrites the same file:
        the container carries the path it was written to.
        """
        if not self.netanal_container:
            return None
        return store.save(self.netanal_container, "netanal",
                          label=self.current_params.get("label"))

    def _save_netanal_action(self) -> None:
        """The Save button: write the file, say where, and dialog only on failure."""
        try:
            path = self.save_netanal()
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self, "Save Error", f"Could not save this network analysis:\n{e}")
            return
        print(f"[Netanal] Saved {path}" if path
              else "[Netanal] Nothing measured yet, so nothing to save.")

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

        trace = self._sweep_for_cable_delay(active_module)
        if trace is None:
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
        """The module's sweep, which the delay fit runs on."""
        trace = self.netanal_traces.get(active_module)
        if trace is None:
            QtWidgets.QMessageBox.information(
                self, "No Data", f"No data for Module {active_module}.")
        return trace

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
        trace = self.netanal_traces.get(active_module)

        if trace is not None and len(trace['frequencies']) > 0:
            freqs = trace['frequencies']
            phases = recalculate_displayed_phase(
                freqs, np.degrees(np.angle(trace['iq_counts'])),
                L_old_physical, L_new_physical,
            )
            if len(phases) > 0:
                phases = phases - phases[0]
            plot_info['phase_curve'].setData(freqs, ((phases + 180) % 360) - 180)

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
        
        search = self.resonance_searches.get(module_id)
        self.take_multisweep_btn.setEnabled(bool(search and search.candidates))

    #
    # 4. Multisweep Dialog Management
    #

    def _show_multisweep_dialog(self) -> None:
        """
        Show the dialog to configure and run multisweep analysis.
        
        This method:
        1. Names the active module's resonances into a ResonatorCatalog
        2. Sets up the multisweep dialog with appropriate parameters
        3. Launches the multisweep analysis if the user accepts the dialog
        """
        # Get active module
        active_module = self._get_active_module()
        if active_module is None:
            return
            
        # A multisweep measures a ResonatorCatalog, so the search's accepted
        # candidates are named here, at the amplitude the netanal probed them
        # at. The catalog is not a file of its own: multisweep records the one
        # it swept in its own output, and the netanal holds the search it came
        # from. The frequency list the dialog shows is read back off it.
        search = self.resonance_searches.get(active_module)
        amplitude = self.netanal_traces.get(active_module, {}).get('sweep_amplitude')
        if not (search and search.candidates) or amplitude is None:
            self._show_status(
                f"Module {active_module}: run Find Resonances first.", ok=False)
            return
        catalog = search.to_catalog(module=active_module, amplitude=float(amplitude))
        
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
            catalog=catalog,
            dac_scales=dac_scales_for_dialog,
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
