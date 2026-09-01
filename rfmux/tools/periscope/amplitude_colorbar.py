"""Shared horizontal colorbar widget for multisweep grid plots."""

import numpy as np
from PyQt6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

from .utils import UnitConverter, COLORMAP_CHOICES


class AmplitudeColorBar(QtWidgets.QWidget):
    """A thin horizontal colorbar that maps amplitude → color.

    Shown above the sweep grid when there are too many sweeps for per-plot
    legends (>5).  Uses the same ``inferno`` colormap and dark/light mode
    mapping as the curves in the grid plots.

    The widget is ~30 px tall and stretches to full width.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(62)
        self._min_amp = 0.0
        self._max_amp = 1.0
        self._min_label = ""
        self._max_label = ""
        self._min_norm_label = ""
        self._max_norm_label = ""
        self._show_norm_row = False
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

        # Secondary row: show normalised amplitude when the primary label is in
        # physical units (dBm / volts with a known dac_scale).
        self._show_norm_row = unit_mode in ("dbm", "volts") and dac_scale is not None
        if self._show_norm_row:
            self._min_norm_label = UnitConverter.format_probe_label(min_amp, "counts", None)
            self._max_norm_label = UnitConverter.format_probe_label(max_amp, "counts", None)
        else:
            self._min_norm_label = ""
            self._max_norm_label = ""

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

        # Layout constants
        margin = 8
        bar_top = 4
        bar_height = 14
        norm_label_top = bar_top + bar_height + 4   # top of the secondary (norm) label row
        norm_label_height = 12
        direction_note_top = norm_label_top + norm_label_height + 2

        # --- Gradient bar ---
        bar_left = margin + 60   # room for min label
        bar_right = w - margin - 60  # room for max label
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
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(text_color)

        # Min label (left of bar)
        painter.drawText(margin, bar_top, int(bar_left - margin - 2), bar_height,
                         QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
                         self._min_label)
        # Max label (right of bar)
        painter.drawText(int(bar_right + 2), bar_top, int(w - bar_right - margin), bar_height,
                         QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                         self._max_label)

        # Secondary row: normalised amplitude labels (only when primary is physical)
        if self._show_norm_row:
            font.setPointSize(7)
            painter.setFont(font)
            painter.setPen(text_color)
            painter.drawText(margin, norm_label_top, int(bar_left - margin - 2), norm_label_height,
                             QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
                             self._min_norm_label)
            painter.drawText(int(bar_right + 2), norm_label_top, int(w - bar_right - margin), norm_label_height,
                             QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                             self._max_norm_label)

        # Direction note (below bar, centered)
        if self._direction_note:
            font.setPointSize(7)
            painter.setFont(font)
            painter.drawText(0, direction_note_top, w, 12,
                             QtCore.Qt.AlignmentFlag.AlignCenter, self._direction_note)

        painter.end()
