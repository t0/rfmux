"""Shared horizontal colorbar widget for multisweep grid plots."""

import numpy as np
from PyQt6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

from .utils import UnitConverter, COLORMAP_CHOICES


class AmplitudeColorBar(QtWidgets.QWidget):
    """Amplitude gradient with unit-aware endpoints and optional direction key."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(46)
        self._min_amp = 0.0
        self._max_amp = 1.0
        self._min_label = ""
        self._max_label = ""
        self._direction_note = ""
        self._dark_mode = False
        self._cmap = pg.colormap.get(COLORMAP_CHOICES.get("AMPLITUDE_SWEEP", "inferno"))
        self.hide()  # hidden until explicitly shown

    # ------------------------------------------------------------------
    def update_range(self, min_amp: float, max_amp: float,
                     dac_scale, unit_mode: str,
                     dark_mode: bool, has_downward: bool):
        """Recompute endpoint labels and trigger a repaint.

        Args:
            min_amp: Lowest normalised amplitude in the sweep set.
            max_amp: Highest normalised amplitude in the sweep set.
            dac_scale: DAC full-scale in dBm (or *None* for raw labels).
            unit_mode: ``"dbm"``, ``"volts"``, or ``"counts"``.
            dark_mode: Current theme flag.
            has_downward: Whether the sweep set includes downward sweeps
                         (adds a direction legend note below the bar).
        """
        self._min_amp = min_amp
        self._max_amp = max_amp
        self._dark_mode = dark_mode
        self._min_label = UnitConverter.format_probe_label(min_amp, unit_mode, dac_scale)
        self._max_label = UnitConverter.format_probe_label(max_amp, unit_mode, dac_scale)
        self._direction_note = "solid = Upward sweep, dotted = Downward sweep" if has_downward else ""
        self.update()  # trigger repaint

    # ------------------------------------------------------------------
    def paintEvent(self, event):  # noqa: N802  (Qt naming convention)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Colours
        text_color = QtGui.QColor("white") if self._dark_mode else QtGui.QColor("black")
        bg_color = QtGui.QColor("#1C1C1C") if self._dark_mode else QtGui.QColor("#FFFFFF")
        painter.fillRect(self.rect(), bg_color)

        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        metrics = painter.fontMetrics()

        # Reserve enough space for decimal endpoint labels.
        margin = 8
        bar_top = 4
        bar_height = 14
        label_y = bar_top + bar_height + 12

        # --- Gradient bar ---
        bar_left = margin + metrics.horizontalAdvance(self._min_label) + 6
        bar_right = w - margin - metrics.horizontalAdvance(self._max_label) - 6
        bar_width = max(bar_right - bar_left, 10)

        if self._cmap is not None:
            for x in range(int(bar_width)):
                t = x / max(bar_width - 1, 1)
                # Apply same dark/light mode mapping as create_amplitude_color_map
                if self._dark_mode:
                    map_val = 0.3 + t * 0.7
                else:
                    map_val = t * 0.75
                rgba = self._cmap.map(map_val)
                if isinstance(rgba, np.ndarray):
                    c = QtGui.QColor(int(rgba[0]), int(rgba[1]), int(rgba[2]))
                else:
                    c = QtGui.QColor(rgba)
                painter.setPen(c)
                painter.drawLine(int(bar_left + x), bar_top,
                                 int(bar_left + x), bar_top + bar_height)

        # Border around bar
        painter.setPen(QtGui.QPen(text_color, 1))
        painter.drawRect(int(bar_left), bar_top, int(bar_width), bar_height)

        # --- Labels ---
        painter.setPen(text_color)

        # Min label (left of bar)
        painter.drawText(margin, bar_top, int(bar_left - margin - 2), bar_height,
                         QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
                         self._min_label)
        # Max label (right of bar)
        painter.drawText(int(bar_right + 2), bar_top, int(w - bar_right - margin), bar_height,
                         QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                         self._max_label)

        # Direction note (below bar, centered)
        if self._direction_note:
            font.setPointSize(7)
            painter.setFont(font)
            painter.drawText(0, label_y - 4, w, 12,
                             QtCore.Qt.AlignmentFlag.AlignCenter, self._direction_note)

        painter.end()
