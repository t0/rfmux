"""What Periscope hands a NetworkAnalysisTask, and what it takes to start one.

One netanal is one module at one probe amplitude: ``take_netanal`` takes a
scalar ``amp`` and returns one trace tagged with it.  The dialogs still offer
a list -- they are rewritten later in the port -- so the resolution to a single
amplitude happens here, in the one line that starts the task, and this is the
test that says so.

What is *not* needed to start one is a DAC scale, which only decides whether
the legend can say dBm.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

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


class _StubDockManager:
    """The dock manager's part in opening a panel: it hands back a dock."""

    def create_dock(self, panel, title, window_id):
        dock = QtWidgets.QDockWidget(title)
        dock.setWidget(panel)
        return dock

    def get_dock(self, name):
        return None


def test_a_sweep_starts_without_a_dac_scale(periscope_and_tasks, qt_app, monkeypatch):
    """A DAC scale is what turns the legend into dBm. Not having one is not a
    reason to refuse the measurement, and refusing it with a modal dialog
    deadlocks a run with nobody there to dismiss it."""
    periscope, started = periscope_and_tasks
    QtWidgets.QMainWindow.__init__(periscope)   # the C++ side, for parenting
    periscope.crs = object()                   # only tested against None
    periscope.dark_mode = False
    periscope.netanal_window_count = 0
    periscope.netanal_windows = {}
    periscope.dock_manager = _StubDockManager()
    assert not hasattr(periscope, "dac_scales")

    refused = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical",
                        lambda *args, **kwargs: refused.append(args[-1]))

    periscope._start_network_analysis({"amp": 0.001, "module": [2], "npoints": 100})

    assert refused == []
    assert [task.module for task in started] == [2]
    panel = periscope.netanal_windows["netanal_0"]["window"]
    assert panel.dac_scales == {}


def test_one_task_per_module_under_its_own_key(periscope_and_tasks):
    """Each module's sweep is its own task, so one module finishing does not
    evict another's."""
    periscope, started = periscope_and_tasks

    for module in (1, 2):
        periscope._start_netanal_task(module, {"amp": 0.001}, "na-1")

    assert sorted(periscope.netanal_tasks) == ["na-1_1", "na-1_2"]
    assert [task.module for task in started] == [1, 2]
