"""Layout pieces for panels that must fit a laptop screen.

A toolbar of twenty controls in one row forces a window wider than a
1080p display; a label carrying a file path forces it wider still.
``FlowLayout`` wraps its items into as many rows as the width needs,
``labelled`` keeps a caption with its control so the two wrap
together, and ``ElidedLabel`` shows what fits of a long text and the
rest on hover.
"""
from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets


class FlowLayout(QtWidgets.QLayout):
    """Items laid left to right, wrapping into new rows as the width
    requires; the height follows from the width."""

    def __init__(self, parent=None, margin: int = 5,
                 h_spacing: int = 6, v_spacing: int = 4) -> None:
        super().__init__(parent)
        self._items: list = []
        self._h = h_spacing
        self._v = v_spacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._place(QtCore.QRect(0, 0, width, 0), dry=True)

    def setGeometry(self, rect: QtCore.QRect) -> None:
        super().setGeometry(rect)
        self._place(rect, dry=False)

    def sizeHint(self) -> QtCore.QSize:
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:
        # The widest item, not the sum: a window may be narrower than
        # the row, and the rows wrap to suit.
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _place(self, rect: QtCore.QRect, dry: bool) -> int:
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y = area.x(), area.y()
        row_h = 0
        for item in self._items:
            hint = item.sizeHint()
            if x + hint.width() > area.right() + 1 and row_h > 0:
                x = area.x()
                y += row_h + self._v
                row_h = 0
            if not dry:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x += hint.width() + self._h
            row_h = max(row_h, hint.height())
        return y + row_h - rect.y() + m.bottom()


def grouped(*widgets: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """*widgets* in one row, as one item that wraps together."""
    box = QtWidgets.QWidget()
    h = QtWidgets.QHBoxLayout(box)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(4)
    for w in widgets:
        h.addWidget(w)
    return box


def labelled(text: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """*widget* with a caption to its left."""
    return grouped(QtWidgets.QLabel(text), widget)


class ElidedLabel(QtWidgets.QLabel):
    """A label that never asks for more width than it is given: the
    text is elided in the middle to fit."""

    def __init__(self, text: str = "", parent=None, max_width: int = 320):
        super().__init__(text, parent)
        self._full = text
        self.setMaximumWidth(max_width)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                           QtWidgets.QSizePolicy.Policy.Preferred)

    def setText(self, text: str) -> None:
        self._full = text
        super().setText(text)
        self.updateGeometry()

    def minimumSizeHint(self) -> QtCore.QSize:
        hint = super().minimumSizeHint()
        return QtCore.QSize(min(hint.width(), 60), hint.height())

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        metrics = painter.fontMetrics()
        rect = self.contentsRect()
        text = metrics.elidedText(self._full,
                                  QtCore.Qt.TextElideMode.ElideMiddle,
                                  rect.width())
        painter.drawText(rect, int(self.alignment()), text)
