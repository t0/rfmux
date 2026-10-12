"""Control mode's widgets: one channel's editable tone fields and the
NCO banner.  Both mirror board values delivered by ToneControlTask and
send an edit on Enter or on leaving the field."""

from typing import Optional

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal

from rfmux.core.transferfunctions import (
    BALUN_LOSS_DB, convert_amplitude_to_dbm, convert_dbm_to_amplitude)

from .utils import dac_scale_output_limits, dac_scale_tooltip


class BoardEdit(QtWidgets.QLineEdit):
    """A field mirroring one board value.  Enter sends the text, and so
    does moving to another widget after typing in it; Esc puts the
    board's value back.  A refresh leaves the field alone while it has
    focus."""

    committed = pyqtSignal(str)
    # Focus lost by the user's own move: a click, Tab, or Enter
    # (clearFocus).  A context menu, another window or a dialog taking
    # focus must not send a half-typed value.
    SEND_REASONS = (Qt.FocusReason.MouseFocusReason,
                    Qt.FocusReason.TabFocusReason,
                    Qt.FocusReason.BacktabFocusReason,
                    Qt.FocusReason.OtherFocusReason)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._board = ""
        self._edited = False
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.setMaximumWidth(96)
        self.textEdited.connect(lambda _text: setattr(self, "_edited", True))
        self.returnPressed.connect(self.clearFocus)

    def show_board(self, text: str) -> None:
        self._board = text
        if not self.hasFocus():
            self.setText(text)

    def reject(self) -> None:
        """The sent text was not accepted: show the board's value."""
        self.setText(self._board)

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        if event.reason() not in self.SEND_REASONS:
            return
        edited, self._edited = self._edited, False
        if edited:
            self.committed.emit(self.text())
        else:
            self.setText(self._board)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.setText(self._board)
            self._edited = False
            self.clearFocus()
            return
        super().keyPressEvent(event)


class ToneFields(QtWidgets.QFrame):
    """One channel's frequency (kHz from the NCO, the actual frequency
    under it), amplitude (dBm against the module's DAC scale at the output,
    the normalized value under it) and DAC and ADC phases (degrees)."""

    # channel, {field: value in board units}
    write = pyqtSignal(int, dict)
    invalid = pyqtSignal(str)

    def __init__(self, channel: int, parent=None):
        super().__init__(parent)
        self.channel = channel
        self._dac_scale: Optional[float] = None
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(6, 4, 6, 4)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)

        self.title = QtWidgets.QLabel(f"<b>Ch {channel}</b>")
        self.state = QtWidgets.QLabel("")
        self.state.setAlignment(Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self.title, 0, 0)
        grid.addWidget(self.state, 0, 1, 1, 2)

        self.edits = {}
        row = 1
        # (field, caption, unit, tooltip, read-only line under it)
        for field, caption, unit, tip, under in (
                ("frequency", "Frequency", "kHz",
                 "Offset from the NCO in kHz", ("actual", "NCO + offset")),
                ("amplitude", "Amplitude", "dBm",
                 "Tone power in dBm against the module's labelled DAC "
                 "scale; 0, blank or 'off' turns the tone off",
                 ("normalized", "Normalized DAC amplitude, as the board holds it")),
                ("dac_phase", "DAC phase", "°",
                 "Carrier (DAC) phase in degrees", None),
                ("adc_phase", "ADC phase", "°",
                 "Demodulator (ADC) phase in degrees", None)):
            edit = BoardEdit()
            edit.setToolTip(tip)
            edit.committed.connect(
                lambda text, f=field: self._commit(f, text))
            self.edits[field] = edit
            grid.addWidget(QtWidgets.QLabel(caption), row, 0)
            grid.addWidget(edit, row, 1)
            grid.addWidget(QtWidgets.QLabel(unit), row, 2)
            row += 1
            if under:
                label = QtWidgets.QLabel("")
                label.setAlignment(Qt.AlignmentFlag.AlignRight)
                label.setToolTip(under[1])
                setattr(self, under[0], label)
                grid.addWidget(label, row, 1, 1, 2)
                row += 1
        self.edits["amplitude"].setPlaceholderText("off")
        grid.setColumnStretch(1, 1)

    def show_values(self, tone: dict, nco: Optional[float],
                    dac_scale: float) -> None:
        self._dac_scale = dac_scale
        frequency = tone.get("frequency")
        amplitude = tone.get("amplitude")
        self.edits["frequency"].show_board(
            "" if frequency is None else f"{frequency / 1e3:.3f}")
        for field in ("dac_phase", "adc_phase"):
            phase = tone.get(field)
            self.edits[field].show_board(
                "" if phase is None else f"{phase:.2f}")
        if amplitude is None or amplitude <= 0:
            self.edits["amplitude"].show_board("")
            self.state.setText("no tone (amplitude 0)")
        else:
            self.edits["amplitude"].show_board(
                f"{convert_amplitude_to_dbm(amplitude, dac_scale):.2f}")
            self.state.setText("tone on")
        self.normalized.setText(
            "" if amplitude is None else f"= {amplitude:.6f} normalized")
        if frequency is None or nco is None:
            self.actual.setText("")
        else:
            self.actual.setText(f"= {(nco + frequency) / 1e6:.6f} MHz")

    def _commit(self, field: str, text: str) -> None:
        try:
            value = self._to_board_units(field, text)
        except ValueError as exc:
            self.edits[field].reject()
            self.invalid.emit(f"Ch {self.channel} {field}: {exc}")
            return
        self.write.emit(self.channel, {field: value})

    def _to_board_units(self, field: str, text: str) -> float:
        # The board checks ranges; a rejected value comes back as an
        # error and the re-read restores the field.
        if field == "frequency":
            return float(text) * 1e3
        if field == "amplitude":
            if text.strip().lower() in ("", "off"):
                return 0.0
            if self._dac_scale is None:
                raise ValueError("no DAC scale read from the board yet")
            return convert_dbm_to_amplitude(float(text), self._dac_scale)
        return float(text)


class NcoBanner(QtWidgets.QFrame):
    """Module, editable NCO in MHz, the editable DAC full scale at the output
    (the setting less BALUN_LOSS_DB) that the dBm fields use, and the
    refresh cadence."""

    nco_committed = pyqtSignal(float)  # Hz
    dac_scale_committed = pyqtSignal(float)  # setting in dBm, for set_dac_scale

    def __init__(self, module: int, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setToolTip(
            "Every channel's actual frequency is NCO + offset. Changing "
            "the NCO moves all of them; the offsets stay as they are.")
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(16)
        self.module_label = QtWidgets.QLabel(f"<b>Module {module}</b>")
        self.nco_edit = BoardEdit()
        self.nco_edit.setMaximumWidth(120)
        self.nco_edit.committed.connect(self._commit)
        self.dac_edit = BoardEdit()
        self.dac_edit.setMaximumWidth(80)
        self.dac_edit.setToolTip(dac_scale_tooltip())
        self.dac_edit.committed.connect(self._commit_dac_scale)
        row.addWidget(self.module_label)
        row.addWidget(QtWidgets.QLabel("NCO"))
        row.addWidget(self.nco_edit)
        row.addWidget(QtWidgets.QLabel("MHz"))
        row.addWidget(QtWidgets.QLabel("DAC full scale at output"))
        row.addWidget(self.dac_edit)
        row.addWidget(QtWidgets.QLabel("dBm"))
        row.addStretch(1)
        row.addWidget(QtWidgets.QLabel("refreshed every 1 s"))

    def show_values(self, nco: Optional[float], dac_scale: float) -> None:
        self.nco_edit.show_board("" if nco is None else f"{nco / 1e6:.6f}")
        self.dac_edit.show_board("" if dac_scale is None else f"{dac_scale:.2f}")

    def _commit(self, text: str) -> None:
        try:
            mhz = float(text)
        except ValueError:
            self.nco_edit.reject()
            return
        self.nco_committed.emit(mhz * 1e6)

    def _commit_dac_scale(self, text: str) -> None:
        """Send the setting for a typed DAC full scale at the output, if the
        board accepts it."""
        low, high = dac_scale_output_limits()
        try:
            output = float(text)
        except ValueError:
            self.dac_edit.reject()
            return
        if not low <= output <= high:
            self.dac_edit.reject()
            return
        self.dac_scale_committed.emit(output + BALUN_LOSS_DB)
