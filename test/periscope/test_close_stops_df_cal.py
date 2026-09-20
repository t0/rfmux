"""Closing the window asks the startup df-calibration worker to stop
before waiting on it, so a long sweep cannot outlive the window."""

from types import SimpleNamespace

import pytest


pytest.importorskip("PyQt6")

from test.qt_helpers import bare_periscope  # noqa: E402


def test_close_requests_interruption_before_waiting(qt_app):
    calls = []
    p = bare_periscope(
        timer=SimpleNamespace(stop=lambda: None),
        receiver=SimpleNamespace(stop=lambda: None, wait=lambda: None),
        netanal_tasks={}, kernel_manager=None,
        _df_cal_task=SimpleNamespace(
            isRunning=lambda: True,
            requestInterruption=lambda: calls.append("interrupt"),
            wait=lambda ms: calls.append("wait")))

    # Through Qt's dispatch, which reaches the mixin's closeEvent even
    # though QMainWindow precedes it in the MRO.
    p.show()
    assert p.close()

    assert calls == ["interrupt", "wait"]
