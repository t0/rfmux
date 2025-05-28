"""Dialog to initialize CRS."""

from .utils import (
    QtWidgets, QtCore, QRegularExpression, QRegularExpressionValidator,
    QDoubleValidator, QIntValidator,
    DEFAULT_AMPLITUDE, DEFAULT_MIN_FREQ, DEFAULT_MAX_FREQ, DEFAULT_CABLE_LENGTH,
    DEFAULT_NPOINTS, DEFAULT_NSAMPLES, DEFAULT_MAX_CHANNELS, DEFAULT_MAX_SPAN,
    UnitConverter, traceback
)

class InitializeCRSDialog(QtWidgets.QDialog):
    """
    Dialog for initializing a CRS (Control and Readout System) board.
    Allows selection of the IRIG time source and an option to clear existing channels.
    """
    def __init__(self, parent: QtWidgets.QWidget = None, crs_obj=None):
        """
        Initializes the CRS initialization dialog.

        Args:
            parent: The parent widget.
            crs_obj: The CRS object, used to access timestamp port enums.
                     This is optional but required for `get_selected_irig_source`
                     to return meaningful values.
        """
        super().__init__(parent)
        self.crs = crs_obj # Store the CRS object
        self.setWindowTitle("Initialize CRS Board")
        self.setModal(True) # Modal dialog

        layout = QtWidgets.QVBoxLayout(self)

        # IRIG Time Source selection
        irig_group = QtWidgets.QGroupBox("IRIG Time Source")
        irig_layout = QtWidgets.QVBoxLayout(irig_group)
        self.rb_backplane = QtWidgets.QRadioButton("BACKPLANE")
        self.rb_test = QtWidgets.QRadioButton("TEST")
        self.rb_sma = QtWidgets.QRadioButton("SMA")
        self.rb_test.setChecked(True) # Default to TEST
        irig_layout.addWidget(self.rb_backplane)
        irig_layout.addWidget(self.rb_test)
        irig_layout.addWidget(self.rb_sma)
        layout.addWidget(irig_group)

        # Option to clear channels
        self.cb_clear_channels = QtWidgets.QCheckBox("Clear all channels on this module")
        self.cb_clear_channels.setChecked(True) # Default to clearing channels
        layout.addWidget(self.cb_clear_channels)

        # OK and Cancel buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addStretch() # Push buttons to the right
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def get_selected_irig_source(self):
        """
        Gets the selected IRIG time source based on the radio button state.

        Returns:
            The corresponding CRS timestamp port enum value if `crs_obj` was provided
            and a selection is made, otherwise None.
        """
        if self.crs is None:
            # Log a warning or handle this case if crs_obj is critical for operation
            print("Warning: CRS object not provided to InitializeCRSDialog. Cannot determine IRIG source enum.")
            return None 
        if self.rb_backplane.isChecked():
            return self.crs.TIMESTAMP_PORT.BACKPLANE
        if self.rb_test.isChecked():
            return self.crs.TIMESTAMP_PORT.TEST
        if self.rb_sma.isChecked():
            return self.crs.TIMESTAMP_PORT.SMA
        return None # Should not happen if one is always checked

    def get_clear_channels_state(self) -> bool:
        """
        Gets the state of the 'Clear all channels' checkbox.

        Returns:
            True if the checkbox is checked, False otherwise.
        """
        return self.cb_clear_channels.isChecked()

