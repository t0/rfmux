"""What Periscope hands a NetworkAnalysisTask, and what it takes to start one.

One netanal is one module, at one probe amplitude, in one direction, which is
what the dialog asks for and what ``take_netanal`` takes. The parameters reach
the driver as they were entered; the module is not among them, because one
Periscope controls one.

What is *not* needed to start one is a DAC scale, which only decides whether
the legend can say dBm.
"""

import pytest

pytest.importorskip("PyQt6")

import numpy as np  # noqa: E402
from PyQt6 import QtWidgets  # noqa: E402

from rfmux.tools.periscope import app as app_module  # noqa: E402
from rfmux.tools.periscope.app import Periscope  # noqa: E402
from rfmux.tools.periscope.tasks import (  # noqa: E402
    NetworkAnalysisSignals, NetworkAnalysisTask,
)


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


def test_the_dialogs_parameters_reach_the_task(periscope_and_tasks):
    """What the dialog returned is what the task is given, with the module it
    is being started for written in."""
    periscope, started = periscope_and_tasks

    periscope._start_netanal_task(
        2, {"amp": 0.002, "sweep_direction": "downward", "npoints": 100}, "na-1")

    assert started[0].params["amp"] == 0.002
    assert started[0].params["sweep_direction"] == "downward"
    assert started[0].params["module"] == 2


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
    periscope.module = 1
    periscope.dock_manager = _StubDockManager()
    assert not hasattr(periscope, "dac_scales")

    refused = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical",
                        lambda *args, **kwargs: refused.append(args[-1]))

    periscope._start_network_analysis({"amp": 0.001, "npoints": 100})

    assert refused == []
    assert [task.module for task in started] == [1]
    panel = periscope.netanal_windows["netanal_0"]["window"]
    assert panel.dac_scales == {}


def test_the_session_module_is_what_gets_swept(periscope_and_tasks, qt_app, monkeypatch):
    """One Periscope controls one module, so a module in the parameters is not
    a thing the sweep can be sent to. The session's module is what runs."""
    periscope, started = periscope_and_tasks
    QtWidgets.QMainWindow.__init__(periscope)
    periscope.crs = object()
    periscope.dark_mode = False
    periscope.netanal_window_count = 0
    periscope.netanal_windows = {}
    periscope.module = 1
    periscope.dock_manager = _StubDockManager()
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical",
                        lambda *args, **kwargs: None)

    periscope._start_network_analysis({"amp": 0.001, "module": 2, "npoints": 100})

    assert [task.module for task in started] == [1]
    assert sorted(periscope.netanal_tasks) == ["netanal_0_1"]


class _Module:
    """One module of a board, for the identifier a container is keyed by."""

    def __init__(self, module):
        self.module = module

    def index(self):
        return f"crs0000_rmod{self.module}"


class _RecordingCRS:
    """Enough board to record one ``take_netanal`` call and answer it."""

    def __init__(self):
        self.calls = []
        self.module = {m: _Module(m) for m in range(1, 9)}

    async def take_netanal(self, **kwargs):
        self.calls.append(kwargs)
        return {self.module[kwargs["module"]].index(): {
            "results": {"frequencies": np.array([]), "iq_counts": np.array([])}}}


def _driver_call(params):
    """The arguments one NetworkAnalysisTask calls ``take_netanal`` with.

    Waited on rather than spun on: the task runs an asyncio loop of its own in
    its own thread and needs nothing from the GUI one.
    """
    crs = _RecordingCRS()
    signals = NetworkAnalysisSignals()   # held: the thread emits on it
    task = NetworkAnalysisTask(crs=crs, module=1, params=params, signals=signals)
    task.start()
    assert task.wait(30_000), "task never finished"
    return crs.calls[0]


def test_the_direction_reaches_the_driver(qt_app):
    """A netanal measured downward is asked for downward."""
    assert _driver_call({"amp": 0.001, "sweep_direction": "downward"}
                        )["sweep_direction"] == "downward"


def test_a_netanal_is_measured_upward_unless_asked_otherwise(qt_app):
    """Parameters from before the dialog offered a direction still measure the
    way ``take_netanal`` does on its own."""
    assert _driver_call({"amp": 0.001})["sweep_direction"] == "upward"
