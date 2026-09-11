"""What the dialogs remember between sessions.

A measurement is usually run many times with nearly the same settings, so a
dialog that opens on its defaults every time is a dialog the operator retypes.
These pin the round trip -- accept a dialog, build another, get the same
values back -- and the two cases where remembering would be wrong: a field
whose value was just read off the board, and a dialog the caller opened on
values of its own.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from rfmux.tools.periscope import settings  # noqa: E402
from rfmux.tools.periscope.field_memory import (  # noqa: E402
    fields_of,
    remember_fields,
)


class _Dialog(QtWidgets.QDialog):
    """A dialog holding one widget of every kind the helper handles."""

    def __init__(self, **kwargs):
        super().__init__()
        form = QtWidgets.QFormLayout(self)
        self.name_edit = QtWidgets.QLineEdit("default")
        self.count_spin = QtWidgets.QSpinBox()
        self.count_spin.setRange(0, 100)
        self.gain_spin = QtWidgets.QDoubleSpinBox()
        self.gain_spin.setRange(0.0, 10.0)
        self.enabled_check = QtWidgets.QCheckBox()
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["slow", "fast"])
        self.notes_edit = QtWidgets.QPlainTextEdit()
        self.readout_label = QtWidgets.QLabel("derived")
        for widget in (self.name_edit, self.count_spin, self.gain_spin,
                       self.enabled_check, self.mode_combo, self.notes_edit,
                       self.readout_label):
            form.addRow(widget)
        # A control this board does not have: the dialog takes it away.
        self.unavailable_check = QtWidgets.QCheckBox()
        form.addRow(self.unavailable_check)
        self.unavailable_check.setHidden(kwargs.pop("hide_unavailable", False))
        remember_fields(self, **kwargs)


@pytest.fixture
def dialogs(qt_app):
    """Builds dialogs, and closes them however the test ends."""
    made = []

    def build(**kwargs):
        made.append(_Dialog(**kwargs))
        return made[-1]

    yield build
    for widget in made:
        widget.close()


def test_every_kind_of_field_comes_back(dialogs):
    first = dialogs()
    first.name_edit.setText("cooldown 3")
    first.count_spin.setValue(17)
    first.gain_spin.setValue(2.5)
    first.enabled_check.setChecked(True)
    first.mode_combo.setCurrentIndex(1)
    first.notes_edit.setPlainText("two lines\nof notes")
    first.accept()

    second = dialogs()
    assert second.name_edit.text() == "cooldown 3"
    assert second.count_spin.value() == 17
    assert second.gain_spin.value() == 2.5
    assert second.enabled_check.isChecked()
    assert second.mode_combo.currentText() == "fast"
    assert second.notes_edit.toPlainText() == "two lines\nof notes"


def test_a_cancelled_dialog_changes_nothing(dialogs):
    first = dialogs()
    first.name_edit.setText("kept")
    first.accept()

    second = dialogs()
    second.name_edit.setText("discarded")
    second.reject()

    assert dialogs().name_edit.text() == "kept"


def test_labels_are_not_fields(dialogs):
    assert "readout_label" not in fields_of(dialogs())


def test_a_skipped_field_is_neither_saved_nor_restored(dialogs):
    first = dialogs(skip=("count_spin",))
    first.count_spin.setValue(9)
    first.name_edit.setText("remembered")
    first.accept()

    assert "count_spin" not in settings.get_dialog_fields("_Dialog")
    second = dialogs(skip=("count_spin",))
    assert second.count_spin.value() == 0
    assert second.name_edit.text() == "remembered"


def test_restore_off_still_saves(dialogs):
    first = dialogs()
    first.name_edit.setText("from last time")
    first.accept()

    seeded = dialogs(restore=False)
    assert seeded.name_edit.text() == "default"
    seeded.name_edit.setText("from the caller")
    seeded.accept()

    assert dialogs().name_edit.text() == "from the caller"


def test_a_choice_the_combo_no_longer_offers_is_ignored(dialogs):
    first = dialogs()
    first.mode_combo.addItem("pfb")
    first.mode_combo.setCurrentIndex(2)
    first.accept()

    second = dialogs()  # built without the extra item
    assert second.mode_combo.currentText() == "slow"


def test_a_field_the_dialog_took_away_keeps_what_it_last_was(dialogs):
    """A hidden control is not this dialog's to set, or to forget."""
    first = dialogs()
    first.unavailable_check.setChecked(True)
    first.accept()

    without = dialogs(hide_unavailable=True)
    assert not without.unavailable_check.isChecked()
    without.accept()

    assert dialogs().unavailable_check.isChecked()
