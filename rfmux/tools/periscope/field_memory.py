"""Remembering what the user typed, dialog by dialog.

A dialog calls :func:`remember_fields` once, at the end of its ``__init__``
and after its widgets exist. The fields it holds as attributes come back the
next time it opens, and are saved again whenever it is accepted.

Fields whose value comes from the board or from the caller belong in ``skip``:
restoring a stale copy over a value that was just read is worse than not
remembering it at all.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from PyQt6 import QtWidgets

from . import settings


def _read(widget: QtWidgets.QWidget) -> Optional[Any]:
    """The widget's value as something JSON can hold, or None if it has none."""
    if isinstance(widget, QtWidgets.QComboBox):
        return widget.currentText()
    if isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
        return widget.value()
    if isinstance(widget, QtWidgets.QLineEdit):
        return widget.text()
    if isinstance(widget, (QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit)):
        return widget.toPlainText()
    if isinstance(widget, QtWidgets.QAbstractButton) and widget.isCheckable():
        return widget.isChecked()
    return None


def _write(widget: QtWidgets.QWidget, value: Any) -> None:
    """Put a saved value back, ignoring one the widget can no longer hold."""
    if isinstance(widget, QtWidgets.QComboBox):
        index = widget.findText(str(value))
        if index >= 0:
            widget.setCurrentIndex(index)
        elif widget.isEditable():
            widget.setEditText(str(value))
    elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
        widget.setValue(type(widget.value())(value))
    elif isinstance(widget, QtWidgets.QLineEdit):
        widget.setText(str(value))
    elif isinstance(widget, (QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit)):
        widget.setPlainText(str(value))
    elif isinstance(widget, QtWidgets.QAbstractButton) and widget.isCheckable():
        widget.setChecked(bool(value))


def fields_of(dialog: QtWidgets.QWidget,
              skip: Iterable[str] = ()) -> dict[str, QtWidgets.QWidget]:
    """The input widgets the dialog holds as attributes, by attribute name.

    The attribute name is the key the value is stored under: it is stable
    across a layout being rearranged, and it reads as the field's name in the
    settings file.

    A field the dialog has hidden outright is not one of its fields: it is a
    control this board or this mode does not have, and its value is the
    constructor's rather than the user's. A field merely sitting in a
    collapsed group is not hidden in this sense and does count.
    """
    skipped = set(skip)
    return {
        name: widget
        for name, widget in vars(dialog).items()
        if name not in skipped
        and isinstance(widget, QtWidgets.QWidget)
        and not widget.isHidden()
        and _read(widget) is not None
    }


def restore_fields(dialog: QtWidgets.QWidget, *, name: str = "",
                   skip: Iterable[str] = ()) -> None:
    """Put back what this dialog was last accepted with."""
    saved = settings.get_dialog_fields(name or type(dialog).__name__)
    if not saved:
        return
    for attr, widget in fields_of(dialog, skip).items():
        if attr in saved:
            try:
                _write(widget, saved[attr])
            except (TypeError, ValueError):
                pass


def save_fields(dialog: QtWidgets.QWidget, *, name: str = "",
                skip: Iterable[str] = ()) -> None:
    """Remember what this dialog currently holds.

    Merged rather than replaced, so a field this dialog did not offer -- a
    PFB option on a board that has none, an input only one of its two modes
    shows -- keeps what it was last actually set to.
    """
    name = name or type(dialog).__name__
    saved = settings.get_dialog_fields(name)
    saved.update({attr: _read(widget)
                  for attr, widget in fields_of(dialog, skip).items()})
    settings.set_dialog_fields(name, saved)


def remember_fields(dialog: QtWidgets.QDialog, *, name: str = "",
                    skip: Iterable[str] = (), restore: bool = True) -> None:
    """Restore this dialog's fields now, and save them again on OK.

    Saving hangs off ``accepted`` rather than ``accept()`` so that a dialog
    overriding ``accept()`` for its own validation still saves, and a
    cancelled dialog still does not.

    ``restore=False`` for a dialog opened on values the caller supplied: those
    win over what was typed last time, but are still what gets remembered.
    """
    if restore:
        restore_fields(dialog, name=name, skip=skip)
    dialog.accepted.connect(
        lambda: save_fields(dialog, name=name, skip=skip))
