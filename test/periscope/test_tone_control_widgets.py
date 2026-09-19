"""Control mode's fields: what they show for a board value, what they
send for an edit, and when a refresh must leave them alone."""

from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")

from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402
from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402

from rfmux.core.transferfunctions import (  # noqa: E402
    convert_amplitude_to_dbm, convert_dbm_to_amplitude)
from rfmux.tools.periscope.app import Periscope  # noqa: E402
from rfmux.tools.periscope.tone_control_widgets import (  # noqa: E402
    NcoBanner, ToneColumn, ToneFields)

NCO, DAC = 500e6, -0.5
TONE = {"frequency": 1.25e6, "amplitude": 0.01, "dac_phase": 30.0,
        "adc_phase": -5.0}


@pytest.fixture
def fields(qt_app):
    host = QtWidgets.QWidget()
    box = QtWidgets.QVBoxLayout(host)
    w = ToneFields(1)
    box.addWidget(w)
    # Somewhere else for focus to go.
    host.other = QtWidgets.QLineEdit()
    box.addWidget(host.other)
    host.show()
    host.activateWindow()
    host.other.setFocus()
    qt_app.processEvents()
    w.show_values(TONE, NCO, DAC)
    w.sent = []
    w.write.connect(lambda ch, f: w.sent.append((ch, f)))
    w.errors = []
    w.invalid.connect(w.errors.append)
    yield w
    host.close()


def _edit(edit, text, qt_app):
    edit.setFocus()
    qt_app.processEvents()
    assert edit.hasFocus()
    edit.selectAll()
    QTest.keyClicks(edit, text)


def test_shows_offset_actual_frequency_and_dbm_against_the_scale(fields):
    assert fields.edits["frequency"].text() == "1250.000"
    assert fields.actual.text() == "= 501.250000 MHz"
    assert fields.edits["amplitude"].text() == (
        f"{convert_amplitude_to_dbm(0.01, DAC):.2f}")
    assert fields.normalized.text() == "= 0.010000 normalized"
    assert fields.edits["dac_phase"].text() == "30.00"
    assert fields.edits["adc_phase"].text() == "-5.00"
    assert fields.state.text() == "tone on"


def test_amplitude_zero_is_the_no_tone_state(fields):
    fields.show_values({"frequency": 0.0, "amplitude": 0.0,
                        "dac_phase": 0.0, "adc_phase": 0.0}, NCO, DAC)
    assert fields.edits["amplitude"].text() == ""
    assert fields.normalized.text() == "= 0.000000 normalized"
    assert fields.edits["frequency"].text() == "0.000"
    assert fields.state.text() == "no tone (amplitude 0)"


def test_enter_sends_the_edit_in_board_units(fields, qt_app):
    _edit(fields.edits["frequency"], "-2450", qt_app)
    QTest.keyClick(fields.edits["frequency"], Qt.Key.Key_Return)
    qt_app.processEvents()
    assert fields.sent == [(1, {"frequency": -2.45e6})]


def test_leaving_a_changed_field_sends_it_too(fields, qt_app):
    _edit(fields.edits["amplitude"], "-38", qt_app)
    fields.parent().other.setFocus()
    qt_app.processEvents()
    assert fields.sent == [
        (1, {"amplitude": convert_dbm_to_amplitude(-38.0, DAC)})]


def test_each_phase_field_writes_its_own_target(fields, qt_app):
    _edit(fields.edits["adc_phase"], "12", qt_app)
    QTest.keyClick(fields.edits["adc_phase"], Qt.Key.Key_Return)
    assert fields.sent == [(1, {"adc_phase": 12.0})]


def test_leaving_an_unchanged_field_sends_nothing(fields, qt_app):
    _edit(fields.edits["dac_phase"], "30.00", qt_app)
    fields.parent().other.setFocus()
    qt_app.processEvents()
    assert fields.sent == []


def test_leaving_an_untouched_field_after_a_refresh_sends_nothing(
        fields, qt_app):
    """Another writer moved the tone while the field had focus: leaving
    it must not send the old value back."""
    fields.edits["frequency"].setFocus()
    qt_app.processEvents()
    fields.show_values({**TONE, "frequency": 2e6}, NCO, DAC)
    fields.parent().other.setFocus()
    qt_app.processEvents()
    assert fields.sent == []
    assert fields.edits["frequency"].text() == "2000.000"


@pytest.mark.parametrize("reason", [
    Qt.FocusReason.PopupFocusReason,          # the field's context menu
    Qt.FocusReason.ActiveWindowFocusReason,   # alt-tab, a dialog opening
])
def test_focus_taken_from_the_user_does_not_send(fields, qt_app, reason):
    """Delivered as Qt does; the half-typed value stays for the user."""
    _edit(fields.edits["dac_phase"], "9", qt_app)
    edit = fields.edits["dac_phase"]
    QtWidgets.QApplication.sendEvent(
        edit, QtGui.QFocusEvent(QtCore.QEvent.Type.FocusOut, reason))
    qt_app.processEvents()
    assert fields.sent == []
    assert edit.text() == "9"


def test_a_dbm_entry_before_any_read_is_reported_not_raised(qt_app):
    w = ToneFields(1)
    errors = []
    w.invalid.connect(errors.append)
    w._commit("amplitude", "-40")
    assert errors and "DAC scale" in errors[0]


def test_escape_discards_the_edit(fields, qt_app):
    _edit(fields.edits["dac_phase"], "99", qt_app)
    QTest.keyClick(fields.edits["dac_phase"], Qt.Key.Key_Escape)
    qt_app.processEvents()
    assert fields.sent == []
    assert fields.edits["dac_phase"].text() == "30.00"


def test_refresh_leaves_a_focused_field_alone(fields, qt_app):
    _edit(fields.edits["frequency"], "12", qt_app)
    fields.show_values({**TONE, "frequency": 2e6}, NCO, DAC)
    assert fields.edits["frequency"].text() == "12"
    assert fields.edits["dac_phase"].text() == "30.00"
    # Once the edit is sent, the board's answer is what shows.
    QTest.keyClick(fields.edits["frequency"], Qt.Key.Key_Return)
    fields.show_values({**TONE, "frequency": 12e3}, NCO, DAC)
    assert fields.edits["frequency"].text() == "12.000"


def test_off_sends_amplitude_zero(fields, qt_app):
    _edit(fields.edits["amplitude"], "off", qt_app)
    QTest.keyClick(fields.edits["amplitude"], Qt.Key.Key_Return)
    assert fields.sent == [(1, {"amplitude": 0.0})]


def test_a_non_number_is_reported_and_reverted(fields, qt_app):
    _edit(fields.edits["frequency"], "1.2.3", qt_app)
    QTest.keyClick(fields.edits["frequency"], Qt.Key.Key_Return)
    assert fields.sent == []
    assert fields.errors and "Ch 1 frequency" in fields.errors[0]
    assert fields.edits["frequency"].text() == "1250.000"


def test_banner_sends_the_nco_in_hz(qt_app):
    banner = NcoBanner(1)
    banner.show()
    banner.activateWindow()
    qt_app.processEvents()
    banner.nco_edit.clearFocus()
    qt_app.processEvents()
    banner.show_values(NCO, DAC)
    assert banner.nco_edit.text() == "500.000000"
    assert banner.dac_label.text() == "DAC scale (labelled) -0.50 dBm"
    sent = []
    banner.nco_committed.connect(sent.append)
    banner.nco_edit.setFocus()
    qt_app.processEvents()
    banner.nco_edit.selectAll()
    QTest.keyClicks(banner.nco_edit, "501.5")
    QTest.keyClick(banner.nco_edit, Qt.Key.Key_Return)
    assert sent == [501.5e6]
    banner.close()


def test_layout_puts_a_column_of_fields_after_the_plots(qt_app):
    p = Periscope.__new__(Periscope)
    QtWidgets.QMainWindow.__init__(p)
    p.channel_list = [[1, 2], [3]]
    host = QtWidgets.QWidget()
    p.grid = QtWidgets.QGridLayout(host)
    p.cb_control = QtWidgets.QCheckBox(checked=True)
    p._tone_control_task = SimpleNamespace(write=lambda ch, f: None)

    p._add_tone_columns(2)

    column = p.grid.itemAtPosition(0, 2).widget()
    assert isinstance(column, ToneColumn)
    assert [f.channel for f in column.fields] == [1, 2]
    assert [f.channel for f in p.grid.itemAtPosition(1, 2).widget().fields] == [3]
    assert set(p.tone_fields) == {1, 2, 3}
    # The plots keep the width.
    assert p.grid.columnStretch(2) == 0
    assert p.grid.columnStretch(0) == 1 and p.grid.columnStretch(1) == 1


def test_a_rebuild_with_fewer_plots_frees_the_old_columns(qt_app):
    """A column the grid once had keeps its stretch: after three plot
    modes go to one, the vacated columns must not take width."""
    p = Periscope.__new__(Periscope)
    QtWidgets.QMainWindow.__init__(p)
    p.channel_list = [[1]]
    host = QtWidgets.QWidget()
    p.grid = QtWidgets.QGridLayout(host)
    p.cb_control = QtWidgets.QCheckBox(checked=False)
    p._add_tone_columns(3)
    p._add_tone_columns(1)
    assert [p.grid.columnStretch(c) for c in range(4)] == [1, 0, 0, 0]


def test_layout_adds_nothing_with_control_off(qt_app):
    p = Periscope.__new__(Periscope)
    QtWidgets.QMainWindow.__init__(p)
    p.channel_list = [[1]]
    host = QtWidgets.QWidget()
    p.grid = QtWidgets.QGridLayout(host)
    p.cb_control = QtWidgets.QCheckBox(checked=False)
    p._add_tone_columns(2)
    assert p.grid.count() == 0 and p.tone_fields == {}
