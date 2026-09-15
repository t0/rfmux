"""Configure a new network analysis or edit its parameters for a rerun.

New-analysis mode also offers Import and Load for existing data.
"""

from .utils import (
    QtWidgets, QtCore, DEFAULT_AMPLITUDE, DEFAULT_MIN_FREQ, DEFAULT_MAX_FREQ,
    DEFAULT_NPOINTS, DEFAULT_NSAMPLES, DEFAULT_MAX_CHANNELS, DEFAULT_MAX_SPAN,
    SWEEP_DIRECTIONS, DEFAULT_SWEEP_DIRECTION, traceback
)
from ...tuning import store
from ...tuning.find_resonances import netanal_trace
from .field_memory import remember_fields


def load_network_analysis_container(parent: QtWidgets.QWidget, file_path: str | None = None):
    """Read a netanal file, or say why it is not one.

    A netanal file holds what take_netanal returned: one output block per
    module, keyed by module identifier. ``netanal_trace`` is what says so —
    it refuses a sweep result and anything else that is not a netanal, with
    a message naming what it got.
    """
    if file_path is None:
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.Option.DontUseNativeDialog
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent,
            "Load Network Analysis",
            "",
            "Pickle Files (*.pkl *.pickle);;All Files (*)",
            options=options,
        )

    if not file_path:
        return None

    try:
        container = store.load(file_path)
        for block in container.values():
            netanal_trace(block)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(
            parent,
            "Load Failed",
            f"Could not read '{file_path}' as a network analysis:\n{exc}",
        )
        return None

    return container


class NetworkAnalysisDialog(QtWidgets.QDialog):
    """Configure one ``crs.take_netanal`` call.

    A netanal measures the band once, at one amplitude, in one direction, so
    that is what the dialog asks for. The module is the session's and the
    dialog says which rather than asking.
    """

    def __init__(self, parent: QtWidgets.QWidget = None, *, params: dict = None,
                 module: int | None = None, dac_scales: dict[int, float] = None,
                 editing: bool = False):
        """
        Args:
            params: a previous call's arguments, to seed the fields.
            module: this session's module, which is the one measured.
            dac_scales: module to full scale in dBm, for the label beside the
                amplitude. The caller's to fetch; the dialog only shows it.
            editing: open on *params* to re-run an existing analysis, rather
                than on a new one with Import and Load.
        """
        super().__init__(parent)
        self.params = dict(params or {})
        self.module = module
        self.dac_scales = dict(dac_scales or {})
        self.editing = editing

        self.load_data_available = False
        self.loaded_container = {}

        self.setWindowTitle("Edit Network Analysis Parameters" if editing
                            else "Network Analysis Configuration")
        self.setModal(editing)
        self._setup_ui()
        self._update_dac_scale_info()
        # An editing dialog opens on what the panel measured with; those win
        # over what was typed into the last one, and are still remembered.
        remember_fields(self, restore=not editing)

    # ── the UI ───────────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        if not self.editing:
            self.import_button = QtWidgets.QPushButton("Import Data")
            self.import_button.clicked.connect(self._load_netanal_data)
            layout.addWidget(self.import_button,
                             alignment=QtCore.Qt.AlignmentFlag.AlignLeft)

        param_group = QtWidgets.QGroupBox("Analysis Parameters")
        form = QtWidgets.QFormLayout(param_group)

        self.label_edit = QtWidgets.QLineEdit(self.params.get("label") or "")
        self.label_edit.setToolTip(
            "Your name for this measurement. It goes on the end of the "
            "filename: netanal_YYYYMMDD_HHMMSS_<name>.pkl")
        form.addRow("Measurement Name:", self.label_edit)

        # One Periscope is one module, so this says which rather than asking.
        form.addRow("Module:", QtWidgets.QLabel(str(self.module)))

        self.fmin_edit = self._number("fmin", DEFAULT_MIN_FREQ, 1e6)
        self.fmax_edit = self._number("fmax", DEFAULT_MAX_FREQ, 1e6)
        form.addRow("Min Frequency (MHz):", self.fmin_edit)
        form.addRow("Max Frequency (MHz):", self.fmax_edit)

        self.amp_edit = self._number("amp", DEFAULT_AMPLITUDE)
        self.amp_edit.setToolTip(
            "One normalized amplitude, per tone. Expressions like '1/1000' "
            "are allowed.")
        form.addRow("Normalized Amplitude:", self.amp_edit)

        self.dac_scale_info = QtWidgets.QLabel()
        self.dac_scale_info.setWordWrap(True)
        form.addRow("DAC full scale (dBm):", self.dac_scale_info)

        self.points_edit = self._number("npoints", DEFAULT_NPOINTS)
        form.addRow("Number of Points:", self.points_edit)

        self.samples_edit = self._number("nsamps", DEFAULT_NSAMPLES)
        form.addRow("Samples to Average:", self.samples_edit)

        self.max_chans_edit = self._number("max_chans", DEFAULT_MAX_CHANNELS)
        form.addRow("Max Channels:", self.max_chans_edit)

        self.max_span_edit = self._number("max_span", DEFAULT_MAX_SPAN, 1e6)
        form.addRow("Max Span (MHz):", self.max_span_edit)

        self.direction_combo = QtWidgets.QComboBox()
        self.direction_combo.addItems(SWEEP_DIRECTIONS)
        self.direction_combo.setCurrentText(
            self.params.get("sweep_direction", DEFAULT_SWEEP_DIRECTION))
        self.direction_combo.setToolTip(
            "Which way through the band. The trace comes back in the order it "
            "was measured, so a downward netanal has descending frequencies.")
        form.addRow("Sweep direction:", self.direction_combo)

        layout.addWidget(param_group)

        row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("OK" if self.editing
                                               else "Start Analysis")
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self.accept)
        row.addWidget(self.start_btn)

        if not self.editing:
            self.load_btn = QtWidgets.QPushButton("Load Analysis")
            self.load_btn.setEnabled(False)  # Until a file has been imported.
            self.load_btn.setAutoDefault(False)
            self.load_btn.clicked.connect(self._load_data_avail)
            row.addWidget(self.load_btn)

        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setAutoDefault(False)
        self.cancel_btn.clicked.connect(self.reject)
        row.addWidget(self.cancel_btn)
        layout.addLayout(row)

        self.setMinimumWidth(420)

    def _number(self, key: str, default, scale: float = 1.0) -> QtWidgets.QLineEdit:
        """A field seeded from *params*, in units *scale* of the argument's."""
        value = self.params.get(key)
        value = default if value is None else value
        return QtWidgets.QLineEdit(f"{float(value) / scale:g}")

    def _update_dac_scale_info(self):
        """What full scale is on this session's module, as the board reports it."""
        scale = self.dac_scales.get(self.module)
        self.dac_scale_info.setText(
            f"{scale:+.2f} dBm" if scale is not None else "Unknown")

    # ── importing a saved analysis ───────────────────────────────────────────

    def _load_data_avail(self):
        """Use the loaded file rather than measuring, and accept."""
        self.load_data_available = True
        self.accept()

    def _load_netanal_data(self):
        """
        Asynchronously trigger a non-blocking QFileDialog using the main thread.
        Using QTimer avoids interfering with existing threads.
        """
        QtCore.QTimer.singleShot(0, self._open_file_dialog_async)

    def _open_file_dialog_async(self):
        """Open a non-blocking file dialog for selecting a Network Analysis parameter file."""
        if not hasattr(self, "_file_dialog") or self._file_dialog is None:
            self._file_dialog = QtWidgets.QFileDialog(self, "Load Network Analysis Parameters")
            self._file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)

            self._file_dialog.setNameFilters([
                "Pickle Files (*.pkl *.pickle)",
                "All Files (*)",
            ])

            # Force non-native + non-blocking behavior
            self._file_dialog.setOptions(
                QtWidgets.QFileDialog.Option.DontUseNativeDialog
                | QtWidgets.QFileDialog.Option.ReadOnly
            )
            self._file_dialog.setModal(False)

            # Connect signals
            self._file_dialog.fileSelected.connect(self._on_file_selected)
            self._file_dialog.rejected.connect(self._on_file_dialog_closed)

        # Show the dialog async
        self._file_dialog.open()

    @QtCore.pyqtSlot(str)
    def _on_file_selected(self, path: str):
        """Read a netanal file and fill the fields in with how it was measured."""
        container = load_network_analysis_container(self, file_path=path)
        if container is None:
            return

        self.load_btn.setEnabled(True)
        self.loaded_container = container

        blocks = list(container.values())
        params = blocks[0]["call_params"]

        label = blocks[0].get(store.METADATA_KEY, {}).get("label")
        self.label_edit.setText(label or "")

        def set_if_present(key, widget, formatter):
            if key in params and params[key] is not None:
                try:
                    widget.setText(formatter(params[key]))
                except Exception:
                    widget.setText(str(params[key]))

        set_if_present("amp", self.amp_edit, lambda v: f"{float(v):g}")
        set_if_present("fmin", self.fmin_edit, lambda v: f"{float(v) / 1e6:g}")
        set_if_present("fmax", self.fmax_edit, lambda v: f"{float(v) / 1e6:g}")
        set_if_present("npoints", self.points_edit, lambda v: str(int(float(v))))
        set_if_present("nsamps", self.samples_edit, lambda v: str(int(float(v))))
        set_if_present("max_chans", self.max_chans_edit, lambda v: str(int(float(v))))
        set_if_present("max_span", self.max_span_edit, lambda v: f"{float(v) / 1e6:g}")
        if params.get("sweep_direction") in SWEEP_DIRECTIONS:
            self.direction_combo.setCurrentText(params["sweep_direction"])

        self._update_dac_scale_info()

    @QtCore.pyqtSlot()
    def _on_file_dialog_closed(self):
        """Handle the event when the file dialog is closed without selection."""
        pass

    # ── what the driver is called with ───────────────────────────────────────

    def get_parameters(self) -> dict | None:
        """``take_netanal``'s arguments, or None with a message about the input.

        Fields are evaluated rather than merely parsed, so '1/1000' and '2.4e9'
        both read as numbers.
        """
        try:
            params_dict = dict(self.params)
            params_dict.update({
                'amp': float(eval(self.amp_edit.text())),
                'fmin': float(eval(self.fmin_edit.text())) * 1e6,  # MHz to Hz
                'fmax': float(eval(self.fmax_edit.text())) * 1e6,  # MHz to Hz
                'npoints': int(self.points_edit.text()),
                'nsamps': int(self.samples_edit.text()),
                'max_chans': int(self.max_chans_edit.text()),
                'max_span': float(eval(self.max_span_edit.text())) * 1e6,
                'sweep_direction': self.direction_combo.currentText(),
                'label': self.label_edit.text().strip() or None,
            })
        except Exception as e:
            traceback.print_exc()  # Log the full traceback for debugging
            QtWidgets.QMessageBox.critical(
                self, "Error Parsing Parameters", f"Invalid parameter input: {str(e)}")
            return None

        if params_dict['fmin'] >= params_dict['fmax']:
            QtWidgets.QMessageBox.warning(
                self, "Input Error", "Min Frequency must be less than Max Frequency.")
            return None
        if not 0 < params_dict['amp'] <= 1.0:
            QtWidgets.QMessageBox.warning(
                self, "Input Error",
                "Normalized amplitude must be greater than 0 and at most 1.0.")
            return None
        return params_dict
