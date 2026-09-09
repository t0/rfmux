"""What Periscope hands a NetworkAnalysisTask.

One netanal is one module at one probe amplitude: ``take_netanal`` takes a
scalar ``amp`` and returns one trace tagged with it.  The dialogs still offer
a list -- they are rewritten later in the port -- so the resolution to a single
amplitude happens here, in the one line that starts the task, and this is the
test that says so.
"""

import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope import app as app_module  # noqa: E402
from rfmux.tools.periscope.app import Periscope  # noqa: E402


class _RecordingTask:
    """Stands in for NetworkAnalysisTask: records its params, starts nothing."""

    started = []

    def __init__(self, crs, module, params, signals):
        self.module, self.params = module, params

    def start(self):
        _RecordingTask.started.append(self)


@pytest.fixture
def periscope_and_tasks(monkeypatch):
    """``(periscope, started)``: enough Periscope to start a netanal task."""
    _RecordingTask.started = []
    monkeypatch.setattr(app_module, "NetworkAnalysisTask", _RecordingTask)

    p = Periscope.__new__(Periscope)
    p.crs = None
    p.netanal_tasks = {}
    p.netanal_windows = {"na-1": {"window": None, "signals": None}}
    monkeypatch.setattr(Periscope, "check_connection",
                        lambda self, window_data, window_id: None)
    return p, _RecordingTask.started


def test_the_task_gets_one_amplitude(periscope_and_tasks):
    """A dialog's list of amplitudes reaches the task as the single ``amp``
    take_netanal takes."""
    periscope, started = periscope_and_tasks

    periscope._start_netanal_task(2, {"amps": [0.004, 0.01], "npoints": 100}, "na-1")

    assert [task.params["amp"] for task in started] == [0.004]
    assert started[0].params["module"] == 2


def test_a_single_amp_is_passed_through(periscope_and_tasks):
    """``amp`` on its own is what the task is given, with no list in sight."""
    periscope, started = periscope_and_tasks

    periscope._start_netanal_task(1, {"amp": 0.002}, "na-1")

    assert started[0].params["amp"] == 0.002


def test_one_task_per_module_under_its_own_key(periscope_and_tasks):
    """Each module's sweep is its own task, so one module finishing does not
    evict another's."""
    periscope, started = periscope_and_tasks

    for module in (1, 2):
        periscope._start_netanal_task(module, {"amp": 0.001}, "na-1")

    assert sorted(periscope.netanal_tasks) == ["na-1_1", "na-1_2"]
    assert [task.module for task in started] == [1, 2]
