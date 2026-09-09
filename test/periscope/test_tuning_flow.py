"""Periscope's tuning flow, driven against the standard simulated array.

The tasks are the real tasks, the drivers are the real drivers, and the board is
``rfmux.mock.standard_array`` served over RPC -- nothing on the data path is
mocked, so a break in Periscope's calls into ``rfmux.tuning`` shows up here as a
failure rather than as a mock that happily accepts anything.

Periscope's multisweep still calls the pre-library driver, so the step that does
is marked ``xfail(strict=True)``: it says what the port owes, it
keep the suite green until it is delivered, and they turn into a failure the
moment a stage makes them pass, which is the reminder to drop the marker.
Stage 2 of ``periscope_port_roadmap.md`` clears the multisweep one, and each
later stage adds its step here.

The array is served over RPC alone -- no UDP -- so this runs in the quick tier.
"""

import asyncio
import inspect
import threading

import numpy as np
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
from rfmux.tools.periscope.network_analysis_panel import (  # noqa: E402
    NetworkAnalysisPanel,
)

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


def _run_netanal(crs, module, qt_app, amplitude=0.001, npoints=400):
    """``(errors, completed, updates)`` from one netanal through the real task."""
    signals = NetworkAnalysisSignals()
    completed, errors, updates = [], [], []
    signals.completed.connect(completed.append)
    signals.error.connect(errors.append)
    signals.data_update.connect(lambda mod, trace: updates.append((mod, trace)))

    task = NetworkAnalysisTask(
        crs=crs, module=module, signals=signals,
        params={"amp": amplitude, "fmin": FMIN, "fmax": FMAX,
                "npoints": npoints, "nsamps": 10},
    )
    task.start()
    assert spin_until(qt_app, task.isFinished, timeout=180), "task never finished"
    spin(qt_app)          # the signals are queued to this thread; deliver them
    return errors, completed, updates


def test_network_analysis_task_finishes_without_error(board, qt_app):
    """A netanal through the real task reaches its completion signal."""
    _, crs, catalog = board
    errors, completed, _ = _run_netanal(crs, catalog.module, qt_app)

    assert errors == []
    assert completed == [catalog.module]


def test_network_analysis_task_emits_the_measured_trace(board, qt_app):
    """What the task hands the panel is the driver's trace, not a shape of its
    own: the same keys, and IQ rather than a magnitude and a phase derived from
    it. The last update is the finished sweep."""
    _, crs, catalog = board
    errors, _, updates = _run_netanal(crs, catalog.module, qt_app, npoints=200)

    assert errors == []
    assert updates, "no data reached the panel"

    module, trace = updates[-1]
    assert module == catalog.module
    assert set(trace) >= {"frequencies", "iq_counts"}
    assert trace["sweep_amplitude"] == 0.001
    assert np.iscomplexobj(trace["iq_counts"])
    assert len(trace["frequencies"]) == len(trace["iq_counts"]) == 200

    # Sorted ascending, as an upward sweep's trace is.
    assert np.all(np.diff(trace["frequencies"]) > 0)


def test_network_analysis_panel_holds_the_trace(board, qt_app):
    """The panel stores what it was handed and draws magnitude from the
    measured IQ."""
    _, crs, catalog = board
    panel = NetworkAnalysisPanel(modules=[catalog.module])

    errors, _, updates = _run_netanal(crs, catalog.module, qt_app, npoints=200)
    assert errors == []
    for module, trace in updates:
        panel.update_data(module, trace)

    stored = panel.netanal_traces[catalog.module]
    assert stored is updates[-1][1]

    curve = panel.plots[catalog.module]["amp_curve"]
    drawn_freqs, drawn_magnitude = curve.getData()
    assert np.allclose(drawn_freqs, stored["frequencies"])
    assert np.allclose(drawn_magnitude, np.abs(stored["iq_counts"]))


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


def _panel_with_a_sweep(crs, catalog, qt_app, amplitude=0.004, npoints=60):
    """A netanal panel holding one measured sweep of the standard array."""
    panel = NetworkAnalysisPanel(modules=[catalog.module])
    panel.current_params = {"amp": amplitude}
    panel.dac_scales = {catalog.module: -0.5}

    errors, _, updates = _run_netanal(
        crs, catalog.module, qt_app, amplitude=amplitude, npoints=npoints)
    assert errors == []
    panel.update_data(*updates[-1])
    return panel


def test_export_holds_the_measured_sweep(board, qt_app):
    """A netanal is one sweep at one probe amplitude, and the export says which:
    the sweep carries the amplitude it was taken at, and its magnitude is the
    measured IQ's, not a shape rebuilt beside it."""
    _, crs, catalog = board
    panel = _panel_with_a_sweep(crs, catalog, qt_app)

    sweep = panel.build_export_dict()["modules"][catalog.module]["sweep"]

    assert sweep["sweep_amplitude"] == 0.004
    measured = panel.netanal_traces[catalog.module]["iq_counts"]
    assert np.allclose(sweep["magnitude"]["counts"]["raw"], np.abs(measured))


def test_unwrapping_cable_delay_redraws_the_measured_phase(board, qt_app):
    """Unwrap Cable Delay fits a delay off the module's sweep and adjusts the
    phase it draws. It reads the trace it was handed; nothing else is in scope
    for it to reach for."""
    _, crs, catalog = board
    panel = _panel_with_a_sweep(crs, catalog, qt_app)
    before = panel.plots[catalog.module]["phase_curve"].getData()[1].copy()

    panel._unwrap_cable_delay_action()

    assert catalog.module in panel.module_cable_lengths
    after = panel.plots[catalog.module]["phase_curve"].getData()[1]
    assert not np.allclose(before, after)
