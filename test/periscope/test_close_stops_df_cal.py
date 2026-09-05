"""Closing the window asks the startup df-calibration worker to stop
before waiting on it, so a long sweep cannot outlive the window."""

from types import SimpleNamespace

import pytest


pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from rfmux.tools.periscope.app import Periscope  # noqa: E402


def test_close_requests_interruption_before_waiting(qt_app):
    p = Periscope.__new__(Periscope)
    QtWidgets.QMainWindow.__init__(p)
    p.timer = SimpleNamespace(stop=lambda: None)
    p.receiver = SimpleNamespace(stop=lambda: None, wait=lambda: None)
    p.netanal_tasks = {}
    p.multisweep_tasks = {}
    p.kernel_manager = None
    calls = []
    p._df_cal_task = SimpleNamespace(
        isRunning=lambda: True,
        requestInterruption=lambda: calls.append("interrupt"),
        wait=lambda ms: calls.append("wait"))

    # Through Qt's dispatch, which reaches the mixin's closeEvent even
    # though QMainWindow precedes it in the MRO.
    p.show()
    assert p.close()

    assert calls == ["interrupt", "wait"]
