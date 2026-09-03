"""
Every panel fits a laptop screen.

Toolbars are flow layouts: they wrap into rows as the panel narrows,
so no panel asks for more width than its widest single control.
"""

import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets  # noqa: E402

from rfmux.tools.periscope.layouts import FlowLayout  # noqa: E402
from rfmux.tools.periscope.multisweep_panel import MultisweepPanel  # noqa: E402
from rfmux.tools.periscope.network_analysis_panel import (  # noqa: E402
    NetworkAnalysisPanel)


def _flow_layouts(widget):
    return [w.layout() for w in widget.findChildren(QtWidgets.QWidget)
            if isinstance(w.layout(), FlowLayout)]


@pytest.mark.parametrize("make", [
    lambda: MultisweepPanel(dark_mode=False),
    lambda: NetworkAnalysisPanel(modules=[1], dark_mode=False),
])
def test_panel_toolbars_wrap_and_fit(qt_app, make):
    panel = make()
    flows = _flow_layouts(panel)
    assert flows, "the toolbar should be a flow layout"
    for flow in flows:
        assert flow.heightForWidth(600) >= flow.heightForWidth(1900)
    assert any(f.heightForWidth(600) > f.heightForWidth(1900) for f in flows)
    assert panel.minimumSizeHint().width() < 600
    panel.resize(1000, 640)
    qt_app.processEvents()
    assert panel.width() == 1000


def test_flow_layout_wraps_by_width(qt_app):
    host = QtWidgets.QWidget()
    flow = FlowLayout(host, margin=0, h_spacing=0, v_spacing=0)
    for _ in range(5):
        b = QtWidgets.QPushButton("x")
        b.setFixedSize(100, 20)
        flow.addWidget(b)
    assert flow.heightForWidth(500) == 20
    assert flow.heightForWidth(250) == 60
    assert flow.heightForWidth(100) == 100
    assert flow.minimumSize().width() == 100
