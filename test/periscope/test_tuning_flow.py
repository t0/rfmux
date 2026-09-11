"""Periscope's tuning flow, driven against the standard simulated array.

The tasks are the real tasks, the drivers are the real drivers, and the board is
``rfmux.mock.standard_array`` served over RPC -- nothing on the data path is
mocked, so a break in Periscope's calls into ``rfmux.tuning`` shows up here as a
failure rather than as a mock that happily accepts anything.

A step the port has not reached yet is marked ``xfail(strict=True)``: it says
what is owed, it keeps the suite green until that is delivered, and it turns
into a failure the moment a stage makes it pass, which is the reminder to drop
the marker. Each stage of ``periscope_port_roadmap.md`` adds its step here.

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
from rfmux.core.resonators import on_grid  # noqa: E402
from rfmux.mock.standard_array import standard_array  # noqa: E402
from rfmux.core.transferfunctions import convert_roc_to_dbm  # noqa: E402
from rfmux.tuning import (  # noqa: E402
    AmplitudeSchedule, collect_amplitude_iterations_for, store)
from rfmux.tuning.find_resonances import (  # noqa: E402
    ResonanceSearch,
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
from rfmux.tools.periscope.multisweep_dialog import MultisweepDialog  # noqa: E402
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


def _multisweep_params(catalog, **overrides):
    """One sweep of the standard array, small enough to be quick."""
    params = {
        "module": catalog.module,
        "catalog": catalog,
        "span_hz": 100e3,
        "npoints_per_sweep": 21,
        "nsamps": 10,
        "sweep_direction": "upward",
    }
    params.update(overrides)
    return params


def _run_multisweep(crs, catalog, qt_app, **overrides):
    """Drive the real task to completion; return what each signal carried."""
    params = _multisweep_params(catalog, **overrides)
    panel = MultisweepPanel(target_module=catalog.module, initial_params=params,
                            dac_scales={catalog.module: -0.5})
    signals = MultisweepSignals()
    panel.connect_task_signals(signals)

    errors, completed, records, partials = [], [], [], []
    signals.error.connect(errors.append)
    signals.completed.connect(lambda module, container: completed.append((module, container)))
    signals.sweep_completed.connect(records.append)
    signals.partial_data.connect(
        lambda module, partial, step, direction: partials.append((step, direction, partial)))

    task = MultisweepTask(crs=crs, params=params, signals=signals)
    task.start()
    assert spin_until(qt_app, task.isFinished, timeout=180), "task never finished"
    spin(qt_app)          # the signals are queued to this thread; deliver them
    return panel, errors, completed, records, partials


def test_multisweep_task_finishes_without_error(board, qt_app):
    """A multisweep through the real task reaches ``completed`` with the
    driver's own container and nothing on the error signal."""
    _, crs, catalog = board
    _, errors, completed, _, _ = _run_multisweep(crs, catalog, qt_app)

    assert errors == []
    assert len(completed) == 1
    module, container = completed[0]
    assert module == catalog.module
    assert container[crs.module[catalog.module].index()]["measurement"] == "multisweep"


def test_the_whole_schedule_is_one_call(board, qt_app):
    """Five amplitude steps in two directions are ten sweeps of one
    measurement, not ten measurements: one container, one sweep record each."""
    _, crs, catalog = board
    schedule = AmplitudeSchedule.multiplicative(0.5, 8, 5)
    _, errors, completed, records, _ = _run_multisweep(
        crs, catalog, qt_app, amp=schedule,
        sweep_direction=("upward", "downward"))

    assert errors == []
    assert len(completed) == 1
    results = completed[0][1][crs.module[catalog.module].index()]["results"]
    assert sorted(results) == [0, 1, 2, 3, 4]
    assert all(sorted(step) == ["downward", "upward"] for step in results.values())

    assert [(r["step"], r["direction"]) for r in records] == [
        (step, direction)
        for step in range(5)
        for direction in ("upward", "downward")
    ]
    assert [r["completed"] for r in records] == list(range(1, 11))


def test_live_points_say_which_sweep_they_belong_to(board, qt_app):
    """``data_callback`` is re-emitted with its two coordinates, so a panel
    drawing live knows which step and direction the points are from -- and
    with entries in the block's own keys, so it can draw them."""
    _, crs, catalog = board
    _, errors, _, _, partials = _run_multisweep(
        crs, catalog, qt_app, sweep_direction=("upward", "downward"))

    assert errors == []
    assert ({(step, direction) for step, direction, _ in partials}
            == {(0, "upward"), (0, "downward")})
    _, _, partial = partials[-1]
    entry = partial[next(iter(partial))]
    assert len(entry["frequencies"]) == len(entry["iq_counts"])


def test_the_panel_holds_the_measurement_and_the_array_it_swept(board, qt_app):
    """On completion the panel keeps the driver's block and the catalog out of
    it -- not a restructuring of either."""
    _, crs, catalog = board
    panel, errors, completed, _, _ = _run_multisweep(crs, catalog, qt_app)

    assert errors == []
    assert panel.module_sweeps is completed[0][1][crs.module[catalog.module].index()]
    assert sorted(panel.catalog.names()) == sorted(catalog.names())


def test_a_cancelled_sweep_hands_over_nothing(board, qt_app):
    """Stop mid-call and the panel is not given a half-measurement to hold."""
    _, crs, catalog = board
    params = _multisweep_params(
        catalog, amp=AmplitudeSchedule.multiplicative(0.5, 8, 5),
        sweep_direction=("upward", "downward"))
    signals = MultisweepSignals()
    completed, records = [], []
    signals.completed.connect(lambda module, container: completed.append(container))
    signals.sweep_completed.connect(records.append)

    task = MultisweepTask(crs=crs, params=params, signals=signals)
    task.start()
    assert spin_until(qt_app, lambda: bool(records), timeout=180), "no sweep ran"
    task.stop()
    assert spin_until(qt_app, task.isFinished, timeout=180), "cancel was not answered"
    spin(qt_app)

    assert completed == []
    assert len(records) < 10        # it stopped short of the whole schedule


def test_the_worker_sweeps_a_copy_of_the_catalog(board, qt_app):
    """The panel is free to adopt a different catalog while a sweep runs."""
    _, crs, catalog = board
    params = _multisweep_params(catalog)
    task = MultisweepTask(crs=crs, params=params, signals=MultisweepSignals())

    assert task.catalog is not catalog
    assert task.module == catalog.module


def test_a_finished_sweep_puts_its_progress_report_away(board, qt_app):
    """The progress group reports a sweep in flight; a sweep that has landed
    is reported by the plots."""
    _, crs, catalog = board
    panel, errors, _, _, _ = _run_multisweep(crs, catalog, qt_app)

    assert errors == []
    assert panel.progress_group.isVisibleTo(panel) is False


def test_a_finished_sweep_says_it_is_ready_to_be_saved(board, qt_app):
    """``sweep_finished`` fires once the panel holds the block, which is what
    the session writes the file on."""
    _, crs, catalog = board
    params = _multisweep_params(catalog)
    panel = MultisweepPanel(target_module=catalog.module, initial_params=params,
                            dac_scales={catalog.module: -0.5})
    signals = MultisweepSignals()
    panel.connect_task_signals(signals)
    finished = []
    panel.sweep_finished.connect(lambda: finished.append(panel.module_sweeps))

    task = MultisweepTask(crs=crs, params=params, signals=signals)
    task.start()
    assert spin_until(qt_app, task.isFinished, timeout=180), "task never finished"
    spin(qt_app)

    assert len(finished) == 1
    assert finished[0] is panel.module_sweeps


def test_a_saved_multisweep_is_the_measurement_a_notebook_reads(board, qt_app,
                                                                output_directory):
    """Save writes the container through ``store``, under store's own name, and
    it comes back as the sweeps the driver returned."""
    _, crs, catalog = board
    panel, errors, _, _, _ = _run_multisweep(crs, catalog, qt_app)
    assert errors == []
    panel.initial_params["label"] = "a saved sweep"

    path = panel.save_multisweep()

    assert path.name.startswith("multisweep_")
    assert path.name.endswith("_a_saved_sweep.pkl")
    reloaded = store.load(path)
    block = reloaded[crs.module[catalog.module].index()]
    assert block["measurement"] == "multisweep"
    name = panel._selected_names()[0]
    assert np.array_equal(
        collect_amplitude_iterations_for(block, name)[0]["upward"]["iq_counts"],
        collect_amplitude_iterations_for(panel.module_sweeps, name)[0]["upward"]["iq_counts"])


def test_saving_the_same_multisweep_again_writes_the_same_file(board, qt_app,
                                                               output_directory):
    """The container carries where it was written, so a second Save overwrites
    rather than leaving a near-copy beside it."""
    _, crs, catalog = board
    panel, errors, _, _, _ = _run_multisweep(crs, catalog, qt_app)
    assert errors == []

    first = panel.save_multisweep()
    again = panel.save_multisweep()

    assert first == again
    assert len(list(Path(output_directory).glob("multisweep_*.pkl"))) == 1


def test_a_finished_multisweep_lands_in_the_session_folder(board, qt_app, tmp_path):
    """One measurement, one file, in the session folder, and the session knows
    it is there so the browser lists it."""
    _, crs, catalog = board
    panel, errors, _, _, _ = _run_multisweep(crs, catalog, qt_app)
    assert errors == []
    panel.initial_params["label"] = "in a session"

    manager = SessionManager()
    manager.start_session(str(tmp_path), "session_under_test")
    registered = []
    manager.file_exported.connect(
        lambda path, data_type: registered.append((path, data_type)))
    try:
        _periscope_with(manager)._save_multisweep_to_session(panel, catalog.module)
    finally:
        manager.end_session()

    written = list(Path(manager.session_path or tmp_path / "session_under_test").glob("*.pkl"))
    assert len(written) == 1
    assert written[0].name.endswith("_in_a_session.pkl")
    assert registered == [(str(written[0]), "multisweep")]


def _grid_widgets(panel, tab_idx=0):
    """The subplot widgets the grid is showing, in the order it drew them."""
    panel.plot_tabs.setCurrentIndex(tab_idx)
    panel._redraw_plots()
    grid = panel.mag_sweeps_grid if tab_idx == 0 else panel.iq_sweeps_grid
    return [grid.itemAt(i).widget() for i in range(grid.count())]


def _grid_curves(panel, tab_idx=0):
    """The curves on each subplot, in the order they were plotted."""
    return [w.getPlotItem().listDataItems() for w in _grid_widgets(panel, tab_idx)]


def test_the_grid_draws_a_curve_for_every_sweep_of_every_resonator(board, qt_app):
    """Two amplitude steps in two directions are four traces per subplot, and
    a subplot per resonator up to the batch size."""
    _, crs, catalog = board
    panel, errors, _, _, _ = _run_multisweep(
        crs, catalog, qt_app,
        amp=AmplitudeSchedule.multiplicative(0.5, 2.0, 2),
        sweep_direction=("upward", "downward"))
    assert errors == []

    curves = _grid_curves(panel)
    assert len(curves) == min(len(catalog.names()), panel.batch_size)
    assert all(len(subplot) == 4 for subplot in curves)


def test_a_curve_is_the_entry_it_was_read_from(board, qt_app):
    """The grid draws the sweep the driver wrote: its own frequencies, offset
    from its own centre, and the magnitude of its own IQ."""
    _, crs, catalog = board
    panel, errors, _, _, _ = _run_multisweep(crs, catalog, qt_app)
    assert errors == []
    panel.normalize_traces = False

    name = panel._selected_names()[0]
    sweep = collect_amplitude_iterations_for(panel.module_sweeps, name)[0]["upward"]
    x, y = _grid_curves(panel)[0][0].getData()

    assert np.allclose(
        x, (sweep["frequencies"] - sweep["original_center_frequency"]) / 1e3)
    assert np.allclose(y, convert_roc_to_dbm(np.abs(sweep["iq_counts"])))


def test_the_iq_grid_draws_the_loop_the_entry_carries(board, qt_app):
    """IQ in volts is the entry's counts on one constant scale, which is what
    the entry's own ``iq_volts`` holds."""
    _, crs, catalog = board
    panel, errors, _, _, _ = _run_multisweep(crs, catalog, qt_app)
    assert errors == []
    panel.normalize_traces = False

    name = panel._selected_names()[0]
    sweep = collect_amplitude_iterations_for(panel.module_sweeps, name)[0]["upward"]
    i_vals, q_vals = _grid_curves(panel, tab_idx=1)[0][0].getData()

    assert np.allclose(i_vals, np.real(sweep["iq_volts"]))
    assert np.allclose(q_vals, np.imag(sweep["iq_volts"]))


def test_a_live_sweep_draws_the_points_measured_so_far(board, qt_app):
    """A partial sweep is drawn on the same grid, as far as it has got."""
    _, crs, catalog = board
    schedule = AmplitudeSchedule.multiplicative(0.5, 2.0, 2)
    _, errors, _, _, partials = _run_multisweep(crs, catalog, qt_app, amp=schedule)
    assert errors == []

    step, direction, partial = partials[0]
    live = MultisweepPanel(target_module=catalog.module,
                           initial_params=_multisweep_params(catalog, amp=schedule),
                           dac_scales={catalog.module: -0.5})
    live.add_partial_sweep(catalog.module, partial, step, direction)

    name = next(n for n in live._selected_names() if n in partial)
    x, _y = _grid_curves(live)[0][0].getData()
    assert len(x) == len(partial[name]["frequencies"])


def test_a_running_sweep_is_drawn_over_the_one_before_it(board, qt_app):
    """Re-running in a panel that already holds a measurement draws the points
    arriving now, not the sweep it is replacing."""
    _, crs, catalog = board
    panel, errors, _, _, partials = _run_multisweep(crs, catalog, qt_app)
    assert errors == []
    finished_length = len(_grid_curves(panel)[0][0].getData()[0])

    step, direction, partial = partials[0]
    panel.add_partial_sweep(catalog.module, partial, step, direction)

    name = next(n for n in panel._selected_names() if n in partial)
    x, _y = _grid_curves(panel)[0][0].getData()
    assert len(x) == len(partial[name]["frequencies"]) < finished_length


def test_a_live_sweep_knows_the_drive_the_finished_one_records(board, qt_app):
    """A sweep still being measured carries no ``sweep_amplitude``, so the
    panel takes it from the schedule -- the number the driver will write into
    the entry, which is what keeps a live trace's colour when it finishes."""
    _, crs, catalog = board
    schedule = AmplitudeSchedule.multiplicative(0.5, 2.0, 2)
    panel, errors, _, _, partials = _run_multisweep(
        crs, catalog, qt_app, amp=schedule)
    assert errors == []

    live = MultisweepPanel(target_module=catalog.module,
                           initial_params=_multisweep_params(catalog, amp=schedule),
                           dac_scales={catalog.module: -0.5})
    step, direction, partial = partials[0]
    name = next(iter(partial))

    assert "sweep_amplitude" not in partial[name]
    finished = collect_amplitude_iterations_for(panel.module_sweeps, name)[step][direction]
    assert live._amplitude_of(step, name, partial[name]) == finished["sweep_amplitude"]
    # ...and both panels grade colour over the same amplitudes, so that one
    # number lands on the same colour before and after the sweep finishes.
    assert live._amplitudes_drawn() == panel._amplitudes_drawn()


def test_changing_units_redraws_and_leaves_the_measurement_alone(board, qt_app):
    """Units, normalization and batching are how the panel is looking at the
    block, never a change to it."""
    _, crs, catalog = board
    panel, errors, _, _, _ = _run_multisweep(crs, catalog, qt_app)
    assert errors == []

    block = panel.module_sweeps
    name = panel._selected_names()[0]
    sweep = collect_amplitude_iterations_for(block, name)[0]["upward"]
    before = sweep["iq_counts"].copy()

    panel.normalize_traces = False
    in_dbm = _grid_curves(panel)[0][0].getData()[1].copy()
    panel.unit_mode = "counts"
    in_counts = _grid_curves(panel)[0][0].getData()[1]

    assert not np.allclose(in_dbm, in_counts)
    assert np.allclose(in_counts, np.abs(before))
    assert panel.module_sweeps is block
    assert np.array_equal(sweep["iq_counts"], before)


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


def _search_on(panel, module, qt_app):
    """Press Find Resonances and wait for the real task to come back."""
    panel._find_resonances_action()
    assert spin_until(qt_app, lambda: panel._find_res_task.isFinished(),
                      timeout=60), "the search never finished"
    spin(qt_app)          # the signals are queued to this thread; deliver them
    return panel.resonance_searches.get(module)


def test_find_resonances_finds_the_array_through_the_real_task(board, qt_app):
    """The button runs ``find_resonances_in_netanal`` on the module's own
    netanal output, off the GUI thread, and the panel marks what came back.

    Seven of the standard array's eight, because the eighth sits above the
    band this netanal sweeps.
    """
    _, crs, catalog = board
    panel = _panel_with_a_sweep(crs, catalog, qt_app, amplitude=0.001, npoints=2000)

    search = _search_on(panel, catalog.module, qt_app)

    assert len(search.candidates) == 7
    assert panel.resonance_searches[catalog.module] is search
    # One dashed line per kept resonance, on both plots.
    plot_info = panel.plots[catalog.module]
    assert len(plot_info["resonance_lines_mag"]) == 7
    assert len(plot_info["resonance_lines_phase"]) == 7
    assert "7 resonances" in plot_info["amp_plot"].getPlotItem().titleLabel.text


#: A collision cut wider than the array's own 6-20 MHz spacing, so the finder
#: treats every resonator as colliding with its neighbour and rejects it. The
#: point is the rejections, which are the only thing a display test can show.
EVERYTHING_COLLIDES_KHZ = 20_000.0


def test_the_settings_panel_is_what_the_search_runs_with(board, qt_app):
    """The thresholds are the settings panel's, not a dialog's: change one
    between searches and the next search obeys it, with nothing to fill in."""
    _, crs, catalog = board
    panel = _panel_with_a_sweep(crs, catalog, qt_app, amplitude=0.001, npoints=2000)

    panel.find_resonances_settings.min_separation_spin.setValue(
        EVERYTHING_COLLIDES_KHZ)
    search = _search_on(panel, catalog.module, qt_app)

    assert len(search.candidates) == 1
    assert len(search.rejected) == 6
    assert all("collided" in c.rejected_because for c in search.rejected)


def test_rejected_candidates_are_drawn_with_their_reason(board, qt_app):
    """A finder that returns fewer resonances than the array has is hard to
    debug; the panel draws what was thrown out, and the reason is on it."""
    _, crs, catalog = board
    panel = _panel_with_a_sweep(crs, catalog, qt_app, amplitude=0.001, npoints=2000)

    panel.find_resonances_settings.min_separation_spin.setValue(
        EVERYTHING_COLLIDES_KHZ)
    search = _search_on(panel, catalog.module, qt_app)

    markers = panel.plots[catalog.module]["rejected_markers"]
    x, _ = markers.getData()
    assert len(x) == len(search.rejected)
    assert sorted(x) == pytest.approx(
        sorted(c.frequency_hz for c in search.rejected))
    assert [point.data() for point in markers.points()] == [
        c.rejected_because for c in search.rejected]

    # And the tooltip pyqtgraph builds on hover is that reason, unadorned.
    reason = search.rejected[0].rejected_because
    assert markers.opts["tip"](x=0.0, y=0.0, data=reason) == reason


def test_a_search_updates_the_file_the_netanal_is_in(board, qt_app, output_directory):
    """The search goes into the netanal block, so the file that holds the
    measurement now holds the search too -- the same file, not a second one
    beside it, and a notebook reads it back with ``ResonanceSearch``."""
    _, crs, catalog = board
    panel = _panel_with_a_sweep(crs, catalog, qt_app, amplitude=0.001, npoints=2000)
    path = panel.save_netanal()

    search = _search_on(panel, catalog.module, qt_app)

    assert sorted(p.name for p in output_directory.glob("*.pkl")) == [path.name]
    trace = netanal_trace(store.load(path)[crs.module[catalog.module].index()])
    assert ResonanceSearch.from_dict(
        trace["resonance_search"]).resonance_frequencies_hz == pytest.approx(
            list(search.resonance_frequencies_hz))


def test_a_search_on_an_unsaved_netanal_writes_no_file(board, qt_app, output_directory):
    """Saving stays the Save button's job and the session's: a search on a
    panel that has never been saved leaves the folder alone."""
    _, crs, catalog = board
    panel = _panel_with_a_sweep(crs, catalog, qt_app, amplitude=0.001, npoints=2000)

    _search_on(panel, catalog.module, qt_app)

    assert list(output_directory.glob("*.pkl")) == []


#: Between two of the array's resonances, so accepting here is the operator
#: marking a place the finder had no candidate for rather than one of its hits.
BETWEEN_RESONANCES_HZ = 1.05e9


def _searched_panel(crs, catalog, qt_app):
    """A netanal panel whose resonances have been found, ready to hand over."""
    panel = _panel_with_a_sweep(crs, catalog, qt_app, amplitude=0.001, npoints=2000)
    _search_on(panel, catalog.module, qt_app)
    return panel


def _block_search(panel, module):
    """The search as the netanal block holds it -- what a file would carry."""
    return ResonanceSearch.from_dict(
        netanal_trace(panel._module_block(module))["resonance_search"])


def test_removing_a_resonance_rejects_it_rather_than_deleting_it(board, qt_app):
    """Double-clicking a resonance off the plot is the last rejection pass, and
    it works like the automatic ones: the candidate keeps everything the finder
    measured, gains a reason, and stays in the search as a record of the
    decision. Nothing found is thrown away."""
    _, crs, catalog = board
    panel = _searched_panel(crs, catalog, qt_app)
    search = panel.resonance_searches[catalog.module]
    found = len(search.candidates)
    dropped = sorted(search.resonance_frequencies_hz)[0]

    panel._remove_resonance(catalog.module, dropped)

    assert len(search.candidates) == found - 1
    assert dropped not in search.resonance_frequencies_hz
    gone, = [c for c in search.rejected if c.frequency_hz == dropped]
    assert gone.rejected_because == ResonanceSearch.BY_HAND
    assert f"{found - 1} resonances" in (
        panel.plots[catalog.module]["amp_plot"].getPlotItem().titleLabel.text)


def test_accepting_a_rejected_resonance_puts_it_back_as_found(board, qt_app):
    """The other direction, so a double-click is undoable: a candidate that was
    rejected -- by hand or by a threshold -- comes back with the depth, width
    and Q the finder measured, not as a fresh guess."""
    _, crs, catalog = board
    panel = _searched_panel(crs, catalog, qt_app)
    search = panel.resonance_searches[catalog.module]
    dropped = sorted(search.resonance_frequencies_hz)[0]
    panel._remove_resonance(catalog.module, dropped)
    rejected, = [c for c in search.rejected if c.frequency_hz == dropped]

    panel._add_resonance(catalog.module, dropped)

    restored, = [c for c in search.candidates if c.frequency_hz == dropped]
    assert restored.accepted
    assert (restored.depth_db, restored.width_hz, restored.q_estimate) == (
        rejected.depth_db, rejected.width_hz, rejected.q_estimate)
    assert dropped not in [c.frequency_hz for c in search.rejected]


def test_adding_a_resonance_puts_a_tone_where_it_was_asked_for(board, qt_app):
    """Double-clicking somewhere the finder had no candidate accepts that
    frequency exactly, claiming nothing about a dip being there: it is a place
    the operator wants a tone, and what it is for is reaching the catalog."""
    _, crs, catalog = board
    panel = _searched_panel(crs, catalog, qt_app)
    search = panel.resonance_searches[catalog.module]
    found = len(search.candidates)

    panel._add_resonance(catalog.module, BETWEEN_RESONANCES_HZ)

    assert len(search.candidates) == found + 1
    added, = [c for c in search.candidates
              if c.frequency_hz == BETWEEN_RESONANCES_HZ]
    assert added.accepted
    assert np.isnan([added.depth_db, added.width_hz, added.q_estimate]).all()
    # It reaches the catalog, quantized there and only there.
    array = search.to_catalog(module=catalog.module, amplitude=0.001)
    assert on_grid(BETWEEN_RESONANCES_HZ) in [
        array[n].bias.frequency_hz for n in array.names()]


def test_a_hand_added_resonance_says_so_on_its_marker(board, qt_app):
    """Its tooltip cannot report a depth or a Q, because nothing measured one.
    Saying which resonance it is instead is also the only thing that tells an
    operator apart from a found one."""
    _, crs, catalog = board
    panel = _searched_panel(crs, catalog, qt_app)

    panel._add_resonance(catalog.module, BETWEEN_RESONANCES_HZ)

    lines = panel.plots[catalog.module]["resonance_lines_mag"]
    tooltips = [line.toolTip() for line in lines]
    assert sum("added by hand" in tip for tip in tooltips) == 1
    assert sum("dB deep" in tip for tip in tooltips) == len(lines) - 1


def test_an_edit_by_hand_updates_the_file_the_search_is_in(board, qt_app, output_directory):
    """A search lives in the netanal it searched, so an edit to it leaves that
    file out of date by exactly that much: the same file is rewritten, and no
    second file appears beside it -- least of all a catalog, which multisweep
    records in its own output."""
    _, crs, catalog = board
    panel = _searched_panel(crs, catalog, qt_app)
    path = panel.save_netanal()
    dropped = sorted(panel.resonance_searches[catalog.module]
                     .resonance_frequencies_hz)[0]

    panel._remove_resonance(catalog.module, dropped)

    assert sorted(p.name for p in output_directory.glob("*.pkl")) == [path.name]
    stored = ResonanceSearch.from_dict(
        netanal_trace(store.load(path)[crs.module[catalog.module].index()])
        ["resonance_search"])
    assert dropped not in stored.resonance_frequencies_hz
    assert dropped in [c.frequency_hz for c in stored.rejected]


def test_an_edit_on_an_unsaved_netanal_writes_no_file(board, qt_app, output_directory):
    """Saving stays the Save button's job and the session's, as it is for the
    search itself."""
    _, crs, catalog = board
    panel = _searched_panel(crs, catalog, qt_app)
    dropped = sorted(panel.resonance_searches[catalog.module]
                     .resonance_frequencies_hz)[0]

    panel._remove_resonance(catalog.module, dropped)

    assert list(output_directory.glob("*.pkl")) == []
    assert dropped in [c.frequency_hz
                       for c in _block_search(panel, catalog.module).rejected]


def test_take_multisweep_names_the_accepted_candidates(board, qt_app):
    """What crosses from netanal to multisweep is a ``ResonatorCatalog`` built
    with ``to_catalog``: the search's accepted dips, named, channelled in
    frequency order, at the amplitude the netanal probed them at. It is not
    written anywhere of its own -- multisweep records the catalog it swept."""
    _, crs, catalog = board
    panel = _searched_panel(crs, catalog, qt_app)
    search = panel.resonance_searches[catalog.module]
    panel._remove_resonance(catalog.module,
                            sorted(search.resonance_frequencies_hz)[0])

    array = search.to_catalog(module=catalog.module, amplitude=0.001)

    assert array.module == catalog.module
    assert len(array) == len(search.candidates)
    names = array.names()
    assert [array[n].bias.frequency_hz for n in names] == pytest.approx(
        [on_grid(f) for f in sorted(search.resonance_frequencies_hz)])
    assert {array[n].bias.amplitude for n in names} == {0.001}
    assert [array[n].channel for n in names] == list(range(1, len(names) + 1))


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
    periscope.multisweep_window_count = 0
    periscope.multisweep_windows = {}
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
    assert list(loaded.resonance_searches[catalog.module]
                .resonance_frequencies_hz) == pytest.approx(
        list(search.resonance_frequencies_hz))


def test_a_saved_multisweep_loads_back_into_a_panel(board, qt_app, output_directory):
    """A file Periscope wrote opens as the measurement it holds: the driver's
    block, the catalog it swept, and the amplitudes the schedule walked."""
    _, crs, catalog = board
    panel, errors, _, _, _ = _run_multisweep(
        crs, catalog, qt_app, amp=AmplitudeSchedule.multiplicative(0.5, 2.0, 2))
    assert errors == []
    path = panel.save_multisweep()

    periscope = _periscope_with()
    periscope._load_multisweep_analysis(store.load(path))

    loaded = periscope.multisweep_windows[
        next(iter(periscope.multisweep_windows))]["window"]
    name = panel._selected_names()[0]
    assert loaded.catalog.names() == panel.catalog.names()
    assert loaded._amplitudes_drawn() == panel._amplitudes_drawn()
    assert np.array_equal(
        collect_amplitude_iterations_for(loaded.module_sweeps, name)[0]["upward"]["iq_counts"],
        collect_amplitude_iterations_for(panel.module_sweeps, name)[0]["upward"]["iq_counts"])


def test_a_loaded_multisweep_draws_without_a_board(board, qt_app, output_directory):
    """Loading is reading and drawing. It touches no hardware -- the panel is
    built with no CRS at all here, which is what a review of a saved sweep is."""
    _, crs, catalog = board
    panel, errors, _, _, _ = _run_multisweep(crs, catalog, qt_app)
    assert errors == []
    path = panel.save_multisweep()

    periscope = _periscope_with()
    assert periscope.crs is None
    periscope._load_multisweep_analysis(store.load(path))

    loaded = periscope.multisweep_windows[
        next(iter(periscope.multisweep_windows))]["window"]
    curves = _grid_curves(loaded)
    assert curves and all(len(subplot) == 1 for subplot in curves)


def test_a_multisweep_file_fills_the_dialog_in(board, qt_app, output_directory):
    """Import reads the file with ``store.load`` and fills the fields in from
    what the driver recorded about the sweep."""
    _, crs, catalog = board
    panel, errors, _, _, _ = _run_multisweep(
        crs, catalog, qt_app, amp=AmplitudeSchedule.multiplicative(0.5, 2.0, 2))
    assert errors == []
    path = panel.save_multisweep()

    dialog = MultisweepDialog(dac_scales={catalog.module: -0.5},
                              current_module=catalog.module,
                              load_multisweep=True)
    dialog._on_file_selected(str(path))

    module_id = crs.module[catalog.module].index()
    assert np.array_equal(
        dialog.loaded_container[module_id]["results"][0]["upward"][
            panel._selected_names()[0]]["iq_counts"],
        collect_amplitude_iterations_for(
            panel.module_sweeps, panel._selected_names()[0])[0]["upward"]["iq_counts"])
    assert float(dialog.span_khz_edit.text()) == 100.0
    assert int(dialog.npoints_edit.text()) == 21
    assert int(dialog.nsamps_edit.text()) == 10
    assert dialog.load_btn.isEnabled()
    assert dialog.catalog.names() == panel.catalog.names()
    assert dialog.schedule() == AmplitudeSchedule.multiplicative(0.5, 2.0, 2)


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
