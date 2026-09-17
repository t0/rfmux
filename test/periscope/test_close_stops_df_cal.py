"""Closing the window cancels a startup df-calibration cell still running,
so a long sweep cannot outlive the window."""

from types import SimpleNamespace

import pytest


pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from rfmux.tools.periscope.app import Periscope  # noqa: E402


def test_close_cancels_a_running_calibration(qt_app):
    p = Periscope.__new__(Periscope)
    QtWidgets.QMainWindow.__init__(p)
    p.timer = SimpleNamespace(stop=lambda: None)
    p.receiver = SimpleNamespace(stop=lambda: None, wait=lambda: None)
    p.netanal_tasks = {}
    p.multisweep_tasks = {}
    p.kernel_manager = None
    calls = []
    p._df_cal_future = SimpleNamespace(done=lambda: False,
                                       cancel=lambda: calls.append("cancel"))

    # Through Qt's dispatch, which reaches the mixin's closeEvent even
    # though QMainWindow precedes it in the MRO.
    p.show()
    assert p.close()

    assert calls == ["cancel"]
