"""The same measurement, made through Periscope and made headlessly.

Periscope is a caller of ``rfmux.tuning`` like a notebook is, so a sweep it
takes and a sweep a script takes with the same arguments should differ in two
things and nothing else: who wrote the file, and when. Anything else that
differs is a shape Periscope grew on the way, which is what this catches --
about the artefact a user actually keeps, not about what is on screen.

The array here is the standard one with its noise turned off, so the two runs
compare value for value; one array per test module, as everywhere else.
"""

import asyncio

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from test.qt_helpers import spin, spin_until  # noqa: E402

from rfmux.core.hardware_map import warm_for_threads  # noqa: E402
from rfmux.core.transferfunctions import BASE_FREQUENCY  # noqa: E402
from rfmux.mock.standard_array import standard_array  # noqa: E402
from rfmux.tuning import store  # noqa: E402
from rfmux.tools.periscope.multisweep_panel import MultisweepPanel  # noqa: E402
from rfmux.tools.periscope.network_analysis_panel import NetworkAnalysisPanel  # noqa: E402
from rfmux.tools.periscope.tasks import (  # noqa: E402
    MultisweepSignals, MultisweepTask,
    NetworkAnalysisSignals, NetworkAnalysisTask,
)

#: The simulator's noise, off. A KID drifts and a readout is noisy, which is
#: what the standard array is for; two runs of the same sweep only agree value
#: for value with that turned off.
QUIET = {
    "nqp_noise_enabled": False,
    "tls_noise_enabled": False,
    "udp_noise_level": 0.0,
}

#: The array's band, so a netanal over it is one NCO setting.
FMIN, FMAX = 1.00e9, 1.10e9

#: Written by whichever run made the file, and expected to differ.
PROVENANCE = {"path", "created", "created_by", "label", "updated"}


@pytest.fixture(scope="module")
def quiet_board():
    """``(loop, crs, catalog)``: the standard array with its noise off."""
    loop = asyncio.new_event_loop()
    crs, catalog = loop.run_until_complete(standard_array(QUIET))
    warm_for_threads(crs)
    yield loop, crs, catalog
    loop.close()


@pytest.fixture
def output_directory(tmp_path):
    """Where ``store`` writes, as the session folder is in Periscope."""
    store.set_output_directory(tmp_path)
    yield tmp_path
    store.set_output_directory(None)
    store.set_created_by(None)


def _differences(a, b, path="", *, values=True) -> list[str]:
    """Every place two loaded measurements disagree, named by where it is.

    With *values* false, measured arrays are compared by shape and dtype only.
    That is the half that catches a re-packaging -- a key renamed, an array
    turned into a list, a quantity computed twice -- and it is all that can be
    asserted about a measurement the driver does not repeat exactly.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: keys {sorted(set(a) ^ set(b))}"]
        return [d for key in a
                for d in _differences(a[key], b[key], f"{path}/{key}", values=values)]
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if np.shape(a) != np.shape(b):
            return [f"{path}: shapes {np.shape(a)} and {np.shape(b)}"]
        if np.asarray(a).dtype != np.asarray(b).dtype:
            return [f"{path}: dtypes {np.asarray(a).dtype} and {np.asarray(b).dtype}"]
        if values and not np.array_equal(a, b):
            return [f"{path}: values differ"]
        return []
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: lengths {len(a)} and {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b))
                for d in _differences(x, y, f"{path}[{i}]", values=values)]
    return [] if a == b else [f"{path}: {a!r} and {b!r}"]


def _without_provenance(container):
    """The measurement, with the fields that say who wrote it and when removed."""
    stripped = {}
    for module_id, block in container.items():
        block = dict(block)
        metadata = {k: v for k, v in block.get("file_metadata", {}).items()
                    if k not in PROVENANCE}
        block["file_metadata"] = metadata
        stripped[module_id] = block
    return stripped


# ── multisweep ───────────────────────────────────────────────────────────────

MULTISWEEP = dict(span_hz=100e3, npoints_per_sweep=21, nsamps=10,
                  sweep_direction="upward")


def _multisweep_through_periscope(crs, catalog, qt_app):
    """One sweep the way the GUI takes it: the real task, the panel's own save."""
    params = {"module": catalog.module, "catalog": catalog, **MULTISWEEP}
    panel = MultisweepPanel(target_module=catalog.module, initial_params=params,
                            dac_scales={catalog.module: -0.5})
    signals = MultisweepSignals()
    panel.connect_task_signals(signals)
    errors = []
    signals.error.connect(errors.append)

    task = MultisweepTask(crs=crs, params=params, signals=signals)
    task.start()
    assert spin_until(qt_app, task.isFinished, timeout=180), "task never finished"
    spin(qt_app)
    assert errors == []
    return panel.save_multisweep()


def test_a_multisweep_through_periscope_is_the_file_a_script_writes(
        quiet_board, qt_app, output_directory):
    """Same array, same arguments, two callers: one measurement."""
    loop, crs, catalog = quiet_board

    store.set_created_by("script")
    headless = loop.run_until_complete(
        crs.multisweep(catalog=catalog, save=False, **MULTISWEEP))
    from_script = store.save(headless, "multisweep", label="from a script")

    store.set_created_by("periscope")
    from_periscope = _multisweep_through_periscope(crs, catalog, qt_app)

    assert from_script != from_periscope
    assert _differences(_without_provenance(store.load(from_script)),
                        _without_provenance(store.load(from_periscope))) == []


def test_the_two_multisweep_files_say_who_wrote_them(
        quiet_board, qt_app, output_directory):
    """The two fields that are meant to differ, and do."""
    loop, crs, catalog = quiet_board

    store.set_created_by("script")
    headless = loop.run_until_complete(
        crs.multisweep(catalog=catalog, save=False, **MULTISWEEP))
    script_metadata = store.load(
        store.save(headless, "multisweep"))[list(headless)[0]]["file_metadata"]

    store.set_created_by("periscope")
    periscope_metadata = store.load(
        _multisweep_through_periscope(crs, catalog, qt_app)
    )[list(headless)[0]]["file_metadata"]

    assert script_metadata["created_by"] == "script"
    assert periscope_metadata["created_by"] == "periscope"
    assert script_metadata["path"] != periscope_metadata["path"]


# ── network analysis ─────────────────────────────────────────────────────────

NETANAL = dict(amp=0.004, fmin=FMIN, fmax=FMAX, npoints=200, nsamps=10)


def _netanal_through_periscope(crs, module, qt_app):
    """One network analysis the way the GUI takes it."""
    panel = NetworkAnalysisPanel(modules=[module])
    panel.current_params = {"amp": NETANAL["amp"]}
    signals = NetworkAnalysisSignals()
    errors, completed = [], []
    signals.error.connect(errors.append)
    signals.completed.connect(lambda mod, container: completed.append((mod, container)))

    params = {"module": module, **NETANAL}
    task = NetworkAnalysisTask(crs=crs, module=module, params=params, signals=signals)
    task.start()
    assert spin_until(qt_app, task.isFinished, timeout=180), "task never finished"
    spin(qt_app)
    assert errors == []
    panel.complete_analysis(*completed[0])
    return panel.save_netanal()


def test_a_netanal_through_periscope_is_the_file_a_script_writes(
        quiet_board, qt_app, output_directory):
    """The same again for the measurement the flow starts from -- keys, dtypes,
    shapes and the arguments it was taken with.

    Not the values: two ``take_netanal`` calls on this simulator with its noise
    off do not agree either, scattering by up to ~100 Hz in tone frequency and
    ~5e-4 relative in IQ, so a value comparison here would be measuring the
    driver's repeatability rather than Periscope's fidelity. The test below
    holds that spread against the driver itself, which is where it belongs.
    """
    loop, crs, catalog = quiet_board

    store.set_created_by("script")
    headless = loop.run_until_complete(
        crs.take_netanal(module=catalog.module, save=False, **NETANAL))
    from_script = store.save(headless, "netanal", label="from a script")

    store.set_created_by("periscope")
    from_periscope = _netanal_through_periscope(crs, catalog.module, qt_app)

    assert from_script != from_periscope
    assert _differences(_without_provenance(store.load(from_script)),
                        _without_provenance(store.load(from_periscope)),
                        values=False) == []


def test_a_periscope_netanal_measures_where_a_script_one_does(quiet_board, qt_app,
                                                              output_directory):
    """The two sweeps agree to within what the simulator's own scatter allows.

    The bounds are physical rather than statistical: one tone-grid step for the
    frequencies, since a comb laid out differently -- the ``max_chans`` drift
    this test found -- moves tones by half a spacing, and 1% for the magnitudes,
    which any unit or conversion error clears by orders of magnitude. Both sit
    an order above the ~100 Hz and ~5e-4 the simulator scatters by, so this
    says Periscope measured the same thing without pinning noise.
    """
    loop, crs, catalog = quiet_board

    container = loop.run_until_complete(
        crs.take_netanal(module=catalog.module, save=False, **NETANAL))
    theirs = next(iter(container.values()))["results"]

    store.set_created_by("periscope")
    from_periscope = _netanal_through_periscope(crs, catalog.module, qt_app)
    ours = next(iter(store.load(from_periscope).values()))["results"]

    assert np.max(np.abs(ours["frequencies"] - theirs["frequencies"])) < BASE_FREQUENCY
    assert np.max(np.abs(ours["iq_counts"] - theirs["iq_counts"])) < (
        0.01 * np.max(np.abs(theirs["iq_counts"])))
