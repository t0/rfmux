"""The startup df-calibration worker: runs the measurement coroutine off
the GUI thread and hands the result to the window through a signal."""
import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope.tasks import (  # noqa: E402
    DfCalibrationSignals, DfCalibrationTask)


def _spin_until(qt_app, pred, timeout_ms=5000):
    from PyQt6 import QtCore
    deadline = QtCore.QDeadlineTimer(timeout_ms)
    while not pred() and not deadline.hasExpired():
        qt_app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
    return pred()


def test_result_arrives_by_signal(qt_app):
    async def measure():
        return {1: 3.0e6 + 1j, 2: 2.0e6}
    got = []
    signals = DfCalibrationSignals()
    signals.completed.connect(lambda m, c: got.append((m, c)))
    task = DfCalibrationTask(measure, 1, signals)
    task.start()
    assert _spin_until(qt_app, lambda: bool(got))
    task.wait(2000)
    assert got == [(1, {1: 3.0e6 + 1j, 2: 2.0e6})]


def test_failure_is_reported_not_raised(qt_app):
    async def measure():
        raise RuntimeError("no tones")
    errors, done = [], []
    signals = DfCalibrationSignals()
    signals.error.connect(errors.append)
    signals.completed.connect(lambda m, c: done.append(c))
    task = DfCalibrationTask(measure, 1, signals)
    task.start()
    assert _spin_until(qt_app, lambda: bool(errors))
    task.wait(2000)
    assert errors == ["no tones"] and not done
