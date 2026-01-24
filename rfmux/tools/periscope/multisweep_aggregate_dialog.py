"""Dialog for configuring multisweep aggregate plot parameters."""

from typing import Optional
from .utils import (
    QtWidgets, QtCore, QIntValidator
)


class MultisweepAggregateDialog(QtWidgets.QDialog):
    """
    Dialog for configuring parameters for aggregate multisweep visualization.
    
    Allows users to choose:
    - Which plot type to create (Sweep Plots or Parameter Histograms)
    - Batch size for sweep plots
    - Amplitude selection for histograms
    - Number of bins for histograms
    - Plot type for sweeps (Magnitude or IQ)
    """
    
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, 
                 multisweep_panel=None):
        """
        Initialize the aggregate plot configuration dialog.
        
        Args:
            parent: The parent widget
            multisweep_panel: Reference to MultisweepPanel to extract amplitude info
        """
        super().__init__(parent)
        self.setWindowTitle("Aggregate Plot Configuration")
        self.setModal(True)
        self.multisweep_panel = multisweep_panel
        
        # Extract available amplitudes from multisweep panel
        self.available_amplitudes = []
        if multisweep_panel:
            self._extract_amplitudes()
        
        self._setup_ui()
        self._update_controls()
        
    def _extract_amplitudes(self):
        """Extract list of available amplitudes from multisweep data."""
        if not self.multisweep_panel or not hasattr(self.multisweep_panel, 'results_by_iteration'):
            return
        
        # Get unique amplitudes from iterations
        amplitudes_set = set()
        for iteration_data in self.multisweep_panel.results_by_iteration.values():
            amp = iteration_data.get('amplitude')
            if amp is not None:
                amplitudes_set.add(amp)
        
        self.available_amplitudes = sorted(amplitudes_set)
        
    def _setup_ui(self):
        """Set up the user interface elements for the dialog."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Panel type selection
        type_group = QtWidgets.QGroupBox("Panel Type")
        type_layout = QtWidgets.QVBoxLayout(type_group)
        
        self.sweep_radio = QtWidgets.QRadioButton("Sweep Plots (multi-panel grid)")
        self.sweep_radio.setChecked(True)
        self.sweep_radio.toggled.connect(self._update_controls)
        type_layout.addWidget(self.sweep_radio)
        
        self.histogram_radio = QtWidgets.QRadioButton("Parameter Histograms (statistics)")
        self.histogram_radio.toggled.connect(self._update_controls)
        type_layout.addWidget(self.histogram_radio)
        
        layout.addWidget(type_group)
        
        # Sweep plots options
        self.sweep_group = QtWidgets.QGroupBox("Sweep Plot Options")
        sweep_layout = QtWidgets.QFormLayout(self.sweep_group)
        
        # Plot type (Magnitude or IQ)
        self.plot_type_combo = QtWidgets.QComboBox()
        self.plot_type_combo.addItems(["S21 Magnitude", "IQ Circles"])
        sweep_layout.addRow("Plot Type:", self.plot_type_combo)
        
        # Batch size
        self.batch_size_spin = QtWidgets.QSpinBox()
        self.batch_size_spin.setRange(10, 200)
        self.batch_size_spin.setValue(50)
        self.batch_size_spin.setSingleStep(10)
        self.batch_size_spin.setToolTip("Number of resonators to display per batch")
        sweep_layout.addRow("Batch Size:", self.batch_size_spin)
        
        layout.addWidget(self.sweep_group)
        
        # Histogram options
        self.histogram_group = QtWidgets.QGroupBox("Histogram Options")
        histogram_layout = QtWidgets.QFormLayout(self.histogram_group)
        
        # Amplitude selector
        self.amplitude_combo = QtWidgets.QComboBox()
        if self.available_amplitudes:
            for amp in self.available_amplitudes:
                self.amplitude_combo.addItem(f"{amp:.6f}")
            # Default to highest amplitude (last in sorted list)
            self.amplitude_combo.setCurrentIndex(len(self.available_amplitudes) - 1)
        else:
            self.amplitude_combo.addItem("No data available")
            self.amplitude_combo.setEnabled(False)
        histogram_layout.addRow("Amplitude:", self.amplitude_combo)
        
        # Number of bins
        self.nbins_spin = QtWidgets.QSpinBox()
        self.nbins_spin.setRange(10, 100)
        self.nbins_spin.setValue(30)
        self.nbins_spin.setSingleStep(5)
        self.nbins_spin.setToolTip("Number of bins for Q factor histograms")
        histogram_layout.addRow("Number of Bins:", self.nbins_spin)
        
        layout.addWidget(self.histogram_group)
        
        # Standard OK and Cancel buttons
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | 
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        
    def _update_controls(self):
        """Enable/disable controls based on selected panel type."""
        is_sweep = self.sweep_radio.isChecked()
        self.sweep_group.setEnabled(is_sweep)
        self.histogram_group.setEnabled(not is_sweep)
        
    def get_parameters(self) -> dict | None:
        """
        Retrieve and validate the parameters for aggregate plotting.
        
        Returns:
            A dictionary of parameters if valid, otherwise None.
        """
        try:
            params = {}
            
            # Determine panel type
            if self.sweep_radio.isChecked():
                params['panel_type'] = 'sweep'
                params['plot_type'] = 'magnitude' if self.plot_type_combo.currentText() == "S21 Magnitude" else 'iq'
                params['batch_size'] = self.batch_size_spin.value()
            else:
                params['panel_type'] = 'histogram'
                
                # Get selected amplitude index
                amp_idx = self.amplitude_combo.currentIndex()
                if amp_idx < 0 or amp_idx >= len(self.available_amplitudes):
                    QtWidgets.QMessageBox.warning(
                        self, "Invalid Selection",
                        "Please select a valid amplitude."
                    )
                    return None
                
                params['amplitude_idx'] = amp_idx
                params['amplitude'] = self.available_amplitudes[amp_idx]
                params['nbins'] = self.nbins_spin.value()
            
            return params
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error",
                f"Could not parse parameters: {str(e)}"
            )
            return None
