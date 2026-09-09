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
from pathlib import Path

import numpy as np
import sqlalchemy.orm

import pytest

pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from test.qt_helpers import spin, spin_until  # noqa: E402

from rfmux.core.hardware_map import warm_for_threads  # noqa: E402
from rfmux.mock.standard_array import standard_array  # noqa: E402
from rfmux.tuning import store  # noqa: E402
from rfmux.tuning.find_resonances import (  # noqa: E402
    find_resonances_in_netanal,
    netanal_trace,
)
from rfmux.tools.periscope.app import Periscope  # noqa: E402
from rfmux.tools.periscope.network_analysis_dialog import (  # noqa: E402
    NetworkAnalysisDialog,
)
from rfmux.tools.periscope.session_manager import SessionManager  # noqa: E402
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


@pytest.fixture
def output_directory(tmp_path):
    """Where ``store`` writes during a test, as the session folder is in Periscope."""
    store.set_output_directory(tmp_path)
    yield tmp_path
    store.set_output_directory(None)


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
    signals.completed.connect(
        lambda mod, container: completed.append((mod, container)))
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
    """A netanal through the real task reaches its completion signal, carrying
    the container the driver returned: what gets saved, keyed by module
    identifier."""
    _, crs, catalog = board
    errors, completed, _ = _run_netanal(crs, catalog.module, qt_app)

    assert errors == []
    assert len(completed) == 1
    module, container = completed[0]
    assert module == catalog.module
    assert list(container) == [crs.module[catalog.module].index()]
    assert netanal_trace(container[crs.module[catalog.module].index()])


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

    errors, completed, updates = _run_netanal(
        crs, catalog.module, qt_app, amplitude=amplitude, npoints=npoints)
    assert errors == []
    panel.update_data(*updates[-1])
    panel.complete_analysis(*completed[0])
    return panel


def test_a_saved_netanal_is_the_measurement_a_notebook_reads(board, qt_app, output_directory):
    """Save writes the container through ``store``, under store's own name, and
    what comes back is the measured sweep -- the same file a notebook writes and
    ``store.load`` opens, not a payload of Periscope's own."""
    _, crs, catalog = board
    panel = _panel_with_a_sweep(crs, catalog, qt_app)
    panel.current_params["label"] = "flow test"

    path = panel.save_netanal()

    assert path.parent == output_directory
    assert path.name.startswith("netanal_")
    assert path.name.endswith("_flow_test.pkl")

    reloaded = store.load(path)
    trace = netanal_trace(reloaded[crs.module[catalog.module].index()])
    measured = panel.netanal_traces[catalog.module]
    assert trace["sweep_amplitude"] == 0.004
    assert np.array_equal(trace["frequencies"], measured["frequencies"])
    assert np.array_equal(trace["iq_counts"], measured["iq_counts"])


def test_saving_the_same_netanal_again_writes_the_same_file(board, qt_app, output_directory):
    """An analysis run over a measurement updates the file the measurement is
    in, rather than leaving a near-copy beside it: the container remembers where
    it was written. This is what Find Resonances re-saving relies on."""
    _, crs, catalog = board
    panel = _panel_with_a_sweep(crs, catalog, qt_app)

    first = panel.save_netanal()
    again = panel.save_netanal()

    assert again == first
    assert sorted(p.name for p in output_directory.glob("*.pkl")) == [first.name]


def _periscope_with(session_manager=None):
    """Just enough Periscope to run the file handlers, without a window."""
    periscope = Periscope.__new__(Periscope)
    # The C++ side only: the panels are parented to it, and Periscope.__init__
    # wants a board, a layout and a session.
    QtWidgets.QMainWindow.__init__(periscope)
    periscope.crs = None
    periscope.host = "OFFLINE"
    periscope.dark_mode = False
    periscope.netanal_window_count = 0
    periscope.netanal_windows = {}
    periscope.session_manager = session_manager
    periscope.dock_manager = _StubDockManager()
    return periscope


class _StubDockManager:
    """The dock manager's part in loading a file: it hands back a dock."""

    def create_dock(self, panel, title, window_id):
        dock = QtWidgets.QDockWidget(title)
        dock.setWidget(panel)
        return dock

    def get_dock(self, name):
        return None


def test_a_finished_netanal_lands_in_the_session_folder(board, qt_app, tmp_path):
    """One measurement, one file, in the session folder, and the session knows
    it is there so the browser lists it. The session manager no longer writes
    the file: store does, into the directory the session set."""
    _, crs, catalog = board
    panel = _panel_with_a_sweep(crs, catalog, qt_app)
    panel.current_params["label"] = "in a session"

    manager = SessionManager()
    manager.start_session(str(tmp_path), "session_under_test")
    registered = []
    manager.file_exported.connect(
        lambda path, data_type: registered.append((path, data_type)))
    try:
        _periscope_with(manager)._save_netanal_to_session(panel, [catalog.module])
    finally:
        manager.end_session()

    written = list(Path(manager.session_path or tmp_path / "session_under_test").glob("*.pkl"))
    assert len(written) == 1
    assert written[0].name.endswith("_in_a_session.pkl")
    assert registered == [(str(written[0]), "netanal")]


def test_a_saved_netanal_loads_back_into_a_panel(board, qt_app, output_directory):
    """Loading is store.load and the blocks the driver wrote: the panel draws
    the measured trace and the resonances the search recorded, with nothing
    rebuilt on the way in."""
    _, crs, catalog = board
    panel = _panel_with_a_sweep(crs, catalog, qt_app)
    module_id = crs.module[catalog.module].index()
    search = find_resonances_in_netanal(
        panel.netanal_container[module_id], min_dip_depth_db=0.5)
    path = panel.save_netanal()

    periscope = _periscope_with()
    periscope._load_network_analysis(store.load(path))

    loaded = periscope.netanal_windows["netanal_0"]["window"]
    measured = panel.netanal_traces[catalog.module]
    drawn_freqs, _ = loaded.plots[catalog.module]["amp_curve"].getData()
    assert np.array_equal(drawn_freqs, measured["frequencies"])
    assert np.array_equal(
        loaded.netanal_traces[catalog.module]["iq_counts"], measured["iq_counts"])
    assert loaded.resonance_freqs[catalog.module] == pytest.approx(
        list(search.resonance_frequencies_hz))


def test_the_measurement_name_becomes_the_files_label(qt_app):
    """What the user types as the measurement name is store's ``label``, which
    is what goes on the end of the filename."""
    dialog = NetworkAnalysisDialog(modules=[1], dac_scales={1: -0.5})
    dialog.label_edit.setText("cold plate 2")

    assert dialog.get_parameters()["label"] == "cold plate 2"

    dialog.label_edit.clear()
    assert dialog.get_parameters()["label"] is None


def test_importing_a_netanal_fills_the_dialog_in(board, qt_app, output_directory):
    """Import reads the file with ``store.load`` and fills the fields in from
    what the driver recorded about the sweep -- the module it ran on, the
    amplitude it probed at, the name it was saved under."""
    _, crs, catalog = board
    panel = _panel_with_a_sweep(crs, catalog, qt_app)
    panel.current_params["label"] = "an import"
    path = panel.save_netanal()

    dialog = NetworkAnalysisDialog(modules=[catalog.module], dac_scales={catalog.module: -0.5})
    dialog._on_file_selected(str(path))

    module_id = crs.module[catalog.module].index()
    assert np.array_equal(
        netanal_trace(dialog.loaded_container[module_id])["iq_counts"],
        panel.netanal_traces[catalog.module]["iq_counts"])
    assert dialog.module_entry.text() == str(catalog.module)
    assert float(dialog.amp_edit.text()) == 0.004
    assert dialog.label_edit.text() == "an import"
    assert dialog.load_btn.isEnabled()


def test_the_session_browser_knows_what_periscope_wrote(board, qt_app, output_directory):
    """A file written through ``store`` says what measurement it holds, so the
    session browser opens it in the right panel without looking at its shape."""
    _, crs, catalog = board
    panel = _panel_with_a_sweep(crs, catalog, qt_app)

    path = panel.save_netanal()

    assert SessionManager().identify_file_type(str(path)) == "netanal"


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
