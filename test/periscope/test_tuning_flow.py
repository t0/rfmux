"""Periscope's tuning flow, driven against the standard simulated array.

The tasks are the real tasks, the drivers are the real drivers, and the board is
``rfmux.mock.standard_array`` served over RPC -- nothing on the data path is
mocked, so a break in Periscope's calls into ``rfmux.tuning`` shows up here as a
failure rather than as a mock that happily accepts anything.

Periscope on this branch still calls the pre-library drivers, so the two steps
that do are marked ``xfail(strict=True)``: they say what the port owes, they
keep the suite green until it is delivered, and they turn into a failure the
moment a stage makes them pass, which is the reminder to drop the marker.
Stage 1 of ``periscope_port_roadmap.md`` clears the netanal one, stage 2 the
multisweep one, and each later stage adds its step here.

The array is served over RPC alone -- no UDP -- so this runs in the quick tier.
"""

import asyncio
import inspect
import threading

import sqlalchemy.orm

import pytest

pytest.importorskip("PyQt6")

from test.qt_helpers import spin, spin_until  # noqa: E402

from rfmux.core.hardware_map import warm_for_threads  # noqa: E402
from rfmux.mock.standard_array import standard_array  # noqa: E402
from rfmux.tools.periscope.tasks import (  # noqa: E402
    MultisweepSignals,
    MultisweepTask,
    NetworkAnalysisSignals,
    NetworkAnalysisTask,
)
from rfmux.tools.periscope.multisweep_panel import MultisweepPanel  # noqa: E402

#: The array's band, so a netanal over it is one NCO setting and a second of work.
FMIN, FMAX = 1.00e9, 1.10e9


@pytest.fixture(scope="module")
def board():
    """``(loop, crs, catalog)``: the standard array, biased by the simulator.

    One array per module, as in ``test/tuning/conftest.py``: a second
    ``load_session`` in the same process detaches the first board's objects.
    """
    loop = asyncio.new_event_loop()
    crs, catalog = loop.run_until_complete(standard_array())
    warm_for_threads(crs)     # Periscope.__init__ does this when it takes a board
    yield loop, crs, catalog
    loop.close()


def _in_a_worker_thread(call):
    """Run *call* on its own thread with its own loop, as a task does.

    Anything awaitable it returns is driven there too, so the board is touched
    only from that thread.
    """
    result = {}

    def worker():
        own_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(own_loop)
        try:
            value = call()
            if inspect.isawaitable(value):
                value = own_loop.run_until_complete(value)
            result["value"] = value
        except Exception as exc:                      # noqa: BLE001 - reported
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            own_loop.close()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=120)
    return result


def test_a_worker_thread_can_drive_a_warmed_board(board):
    """Every Periscope measurement runs on a QThread, and the drivers name
    their output block from the hardware map (``crs.module[m].index()``). The
    map is one in-memory SQLite database that only its own thread may read, so
    Periscope warms those attributes when it takes the board (the fixture does
    the same); without that, the first netanal on a worker raises
    ProgrammingError, which the next test pins."""
    _, crs, catalog = board

    result = _in_a_worker_thread(lambda: crs.take_netanal(
        amp=0.001, fmin=FMIN, fmax=FMIN + 20e6, npoints=60, nsamps=10,
        module=catalog.module, save=False))

    assert "error" not in result, result["error"]
    assert list(result["value"]) == [crs.module[catalog.module].index()]


def test_an_unwarmed_attribute_is_what_would_break(board):
    """The failure the warm-up prevents, on an attribute expired the way a
    commit expires one: the lazy load from another thread reaches for a
    connection that belongs to this one. If SQLAlchemy or the map ever stops
    working this way, this test says so and the warm-up can go."""
    _, crs, catalog = board
    module = crs.module[catalog.module]
    sqlalchemy.orm.object_session(module).expire(module)   # as a commit would

    try:
        result = _in_a_worker_thread(module.index)
        assert "ProgrammingError" in result.get("error", ""), result
    finally:
        warm_for_threads(crs)     # leave the board as the other tests expect


@pytest.mark.xfail(strict=True, reason="stage 1: the task reads the pre-library "
                                       "netanal shape (tasks.py:479)")
def test_network_analysis_task_finishes_without_error(board, qt_app):
    """A netanal through the real task reaches its completion signal."""
    _, crs, catalog = board
    signals = NetworkAnalysisSignals()
    completed, errors = [], []
    signals.completed.connect(completed.append)
    signals.error.connect(errors.append)

    task = NetworkAnalysisTask(
        crs=crs, module=catalog.module, signals=signals, amplitude=0.001,
        params={"fmin": FMIN, "fmax": FMAX, "npoints": 400, "nsamps": 10,
                "cable_length": 0.0, "clear_channels": True},
    )
    task.start()
    assert spin_until(qt_app, task.isFinished, timeout=180), "task never finished"
    spin(qt_app)          # the signals are queued to this thread; deliver them

    assert errors == []
    assert completed == [catalog.module]


@pytest.mark.xfail(strict=True, reason="stage 2: the task passes "
                                       "bias_frequency_method and rotate_saved_data "
                                       "to crs.multisweep (tasks.py:706)")
def test_multisweep_task_finishes_without_error(board, qt_app):
    """A multisweep through the real task, into a real panel, reaches
    ``all_completed`` with nothing on the error signal."""
    _, crs, catalog = board
    frequencies = [catalog[name].bias.frequency_hz for name in catalog.names()]
    params = {
        "module": catalog.module,
        "amps": [catalog[catalog.names()[0]].bias.amplitude],
        "span_hz": 100e3,
        "npoints_per_sweep": 21,
        "nsamps": 10,
        "sweep_direction": "upward",
        "resonance_frequencies": frequencies,
    }
    # A real panel, because the task reads its frequency bookkeeping, but the
    # panel's own slots stay unwired: handle_error opens a modal QMessageBox,
    # which offscreen never returns. Stage 2 replaces that with a status line;
    # test_multisweep_signals_per_task covers the panel wiring meanwhile.
    panel = MultisweepPanel(target_module=catalog.module, initial_params=params,
                            dac_scales={catalog.module: -0.5})

    signals = MultisweepSignals()
    finished, errors = [], []
    signals.all_completed.connect(lambda: finished.append(True))
    signals.error.connect(lambda module, amp, message: errors.append(message))

    task = MultisweepTask(crs=crs, params=params, signals=signals, window=panel)
    task.start()
    assert spin_until(qt_app, task.isFinished, timeout=180), "task never finished"
    spin(qt_app)          # the signals are queued to this thread; deliver them

    assert errors == []
    assert finished == [True]
