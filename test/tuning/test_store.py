"""Behaviour of the output folder.

Pure — no board, no driver, no GUI. Payload dicts in, files in a tmp_path out.

The emphasis is on the things that lose data if they are wrong: a re-save goes
back over the file it came from rather than beside it, one module written back
into a container does not take the other modules down with it, two measurements
finishing in the same second do not land on top of each other, and a payload
that cannot be stamped is refused rather than written somewhere nobody can find
it again.

The rest is the contract the notebooks rely on: what the folder is called, what
the file is called, where ``file_metadata`` ends up in each of the shapes a
measurement comes back in, and that stamping one does not break the predicates
the tuning layer uses to tell those shapes apart.
"""

import datetime
import pickle

import numpy as np
import pytest

from rfmux import config
from rfmux.tuning import store
from rfmux.tuning.find_resonances import (
    ResonanceSearch,
    find_resonances_in_netanal,
    netanal_trace,
)
from rfmux.tuning.sweep_results import _is_container

pytestmark = pytest.mark.portable


@pytest.fixture(autouse=True)
def output_dir(tmp_path, monkeypatch):
    """Every test writes into its own tmp_path, with autosave on.

    The suite-wide fixture in test/conftest.py turns autosave *off*, which is
    right everywhere except here: this file is the one place that is supposed to
    be writing files.
    """
    monkeypatch.setenv("RFMUX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RFMUX_AUTOSAVE", "1")
    store.set_output_directory(None)
    store.set_autosave(None)
    store.set_created_by(None)
    yield tmp_path
    store.set_output_directory(None)
    store.set_autosave(None)
    store.set_created_by(None)


def a_module_output(module=2, module_id=None):
    """One module's worth of packed sweep, thin but the right shape."""
    return {
        "schema_version": 3,
        "module": module,
        "call_params": {"span_hz": 1e5},
        "results": {0: {"upward": {"R0001": {"frequencies": np.arange(3.0)}}}},
    }


def a_container(*modules):
    return {f"crs0042_rmod{m}": a_module_output(m) for m in modules}


def a_netanal_output(module=1, npoints=4, iq=None):
    """One module's netanal output, packed the way take_netanal packs it."""
    frequencies = np.linspace(1e9, 2e9, npoints)
    if iq is None:
        iq = np.ones(npoints, dtype=complex)
    return {
        "schema_version": 4,
        "measurement": "netanal",
        "module": module,
        "call_params": {"amp": 1e-3, "fmin": 1e9, "fmax": 2e9},
        "results": {
            0: {
                "upward": {
                    "frequencies": frequencies,
                    "iq_counts": iq,
                    "iq_volts": iq * 1e-7,
                    "sweep_amplitude": 1e-3,
                    "sweep_direction": "upward",
                }
            }
        },
    }


def a_netanal(*modules, npoints=4, iq=None):
    """The container take_netanal returns."""
    return {
        f"crs0042_rmod{m}": a_netanal_output(m, npoints, iq)
        for m in (modules or (1,))
    }


# ─── Where files go ───────────────────────────────────────────────────────────


def test_session_folder_is_named_for_today(output_dir):
    folder = store.session_directory()
    assert folder.parent == output_dir
    assert folder.name == f"ipy_session_{datetime.date.today():%Y%m%d}"
    assert folder.is_dir()


def test_asking_where_output_goes_does_not_create_anything(output_dir):
    """output_directory() is a question, not a request."""
    assert store.output_directory() == output_dir
    assert not list(output_dir.iterdir())
    assert store.session_directory(create=False).exists() is False


def test_filename_carries_type_date_time_and_label(output_dir):
    path = store.save(a_container(2), "multisweep", label="cooldown3")
    stem = path.stem
    assert stem.startswith("multisweep_")
    assert stem.endswith("_cooldown3")
    # multisweep_YYYYMMDD_HHMMSS_cooldown3
    _, date, time, label = stem.split("_")
    assert datetime.datetime.strptime(f"{date}_{time}", "%Y%m%d_%H%M%S")
    assert label == "cooldown3"


def test_a_label_with_spaces_or_slashes_stays_one_path_component(output_dir):
    path = store.save(a_container(2), "multisweep", label="cool down 3/b")
    assert path.name.endswith("_cool_down_3_b.pkl")
    assert path.parent == store.session_directory()


def test_no_label_means_no_trailing_underscore(output_dir):
    path = store.save(a_container(2), "multisweep")
    assert path.stem.count("_") == 2  # type, date, time — nothing after


def test_same_second_does_not_overwrite(output_dir, monkeypatch):
    """Two sweeps can finish in the same second; the loser keeps its data."""
    frozen = datetime.datetime(2026, 9, 4, 14, 22, 31)
    monkeypatch.setattr(store, "_now", lambda: frozen)

    first = store.save(a_container(2), "multisweep")
    second = store.save(a_container(3), "multisweep")

    assert first != second
    assert second.name == first.stem + "_1.pkl"
    assert store.load(first)["crs0042_rmod2"]["module"] == 2
    assert store.load(second)["crs0042_rmod3"]["module"] == 3


# ─── The file_metadata block ─────────────────────────────────────────────────


def test_container_is_stamped_inside_each_module(output_dir):
    sweeps = a_container(1, 2)
    path = store.save(sweeps, "multisweep", label="both")

    for module_id, module_output in sweeps.items():
        metadata = module_output[store.METADATA_KEY]
        assert metadata["measurement_type"] == "multisweep"
        assert metadata["path"] == str(path)
        assert metadata["label"] == "both"
        assert metadata["module"] == module_output["module"]
        assert metadata["file_version"] == store.FILE_VERSION


def test_one_call_is_one_file_however_many_modules(output_dir):
    store.save(a_container(1, 2, 3), "multisweep")
    assert len(list(store.session_directory().glob("*.pkl"))) == 1


def test_stamping_a_container_leaves_it_a_container(output_dir):
    """The tuning layer dispatches on this; a top-level stamp would break it."""
    sweeps = a_container(1, 2)
    store.save(sweeps, "multisweep")
    assert _is_container(sweeps)


def test_a_netanal_is_stamped_per_module_from_itself(output_dir):
    """Nobody has to tell save() what module a netanal is any more, either."""
    netanal = a_netanal(1, 2)
    store.save(netanal, "netanal")
    assert [e[store.METADATA_KEY]["module"] for e in netanal.values()] == [1, 2]


def test_stamping_a_netanal_leaves_it_a_container(output_dir):
    netanal = a_netanal(1, 2)
    store.save(netanal, "netanal")
    assert _is_container(netanal)


def test_a_module_output_takes_its_module_from_itself(output_dir):
    """Nobody has to tell save() what module one module's sweep is."""
    module_output = a_module_output(module=7)
    store.save(module_output, "multisweep")
    assert module_output[store.METADATA_KEY]["module"] == 7


def test_created_by_can_be_declared(output_dir):
    store.set_created_by("periscope")
    sweeps = a_container(2)
    store.save(sweeps, "multisweep")
    assert sweeps["crs0042_rmod2"][store.METADATA_KEY]["created_by"] == "periscope"


def test_an_unstampable_payload_is_refused(output_dir):
    """A file nobody can trace back is worse than no file."""
    with pytest.raises(TypeError, match="to_dict"):
        store.save(object(), "multisweep")
    assert not list(output_dir.rglob("*.pkl"))


def test_a_list_of_results_is_refused_and_says_what_to_do(output_dir):
    """Several modules used to come back as a list, and are a keyed dict now.
    The error is the migration note for anyone whose script still builds one."""
    with pytest.raises(TypeError, match="keyed by module"):
        store.save([a_module_output(1), a_module_output(2)], "multisweep")
    assert not list(output_dir.rglob("*.pkl"))


# ─── Re-saving ────────────────────────────────────────────────────────────────


def test_resaving_goes_back_over_the_same_file(output_dir):
    sweeps = a_container(2)
    first = store.save(sweeps, "multisweep", label="cooldown3")

    sweeps["crs0042_rmod2"]["results"][0]["upward"]["R0001"]["fits"] = {"skewed": {}}
    again = store.save(sweeps, "multisweep")

    assert again == first
    assert len(list(store.session_directory().glob("*.pkl"))) == 1
    assert "fits" in store.load(first)["crs0042_rmod2"]["results"][0]["upward"]["R0001"]


def test_resaving_keeps_created_and_records_updated(output_dir, monkeypatch):
    """The file says when the measurement was taken, not when it was last fitted."""
    monkeypatch.setattr(store, "_now", lambda: datetime.datetime(2026, 9, 4, 9, 0, 0))
    sweeps = a_container(2)
    store.save(sweeps, "multisweep")

    monkeypatch.setattr(store, "_now", lambda: datetime.datetime(2026, 9, 7, 17, 30, 0))
    store.save(sweeps, "multisweep")

    metadata = sweeps["crs0042_rmod2"][store.METADATA_KEY]
    assert metadata["created"] == "2026-09-04T09:00:00"
    assert metadata["updated"] == "2026-09-07T17:30:00"


def test_new_forces_a_second_file(output_dir):
    sweeps = a_container(2)
    first = store.save(sweeps, "multisweep")
    second = store.save(sweeps, "multisweep", new=True)

    assert second != first
    assert first.exists() and second.exists()


def test_resaving_one_module_keeps_the_others(output_dir):
    """fit_sweeps works on sweeps[module_id]; the file holds every module.

    Writing one module straight over that path would silently drop the rest.
    """
    sweeps = a_container(1, 2)
    path = store.save(sweeps, "multisweep")

    module_output = sweeps["crs0042_rmod1"]
    module_output["results"][0]["upward"]["R0001"]["fits"] = {"skewed": {}}
    store.save(module_output, "multisweep")

    on_disk = store.load(path)
    assert set(on_disk) == {"crs0042_rmod1", "crs0042_rmod2"}
    assert _is_container(on_disk)
    assert "fits" in on_disk["crs0042_rmod1"]["results"][0]["upward"]["R0001"]


def test_a_moved_file_is_saved_back_to_where_it_now_is(output_dir):
    """Files get copied off the acquisition machine before anyone fits them."""
    sweeps = a_container(2)
    original = store.save(sweeps, "multisweep")

    moved = output_dir / "elsewhere.pkl"
    moved.write_bytes(original.read_bytes())
    original.unlink()

    loaded = store.load(moved)
    assert loaded["crs0042_rmod2"][store.METADATA_KEY]["path"] == str(moved)

    store.save(loaded, "multisweep")
    assert moved.exists() and not original.exists()


def test_measurement_type_is_only_optional_once_it_is_known(output_dir):
    sweeps = a_container(2)
    with pytest.raises(ValueError, match="measurement_type"):
        store.save(sweeps)

    store.save(sweeps, "multisweep")
    assert store.save(sweeps).name.startswith("multisweep_")


def test_saved_path_reports_where_it_went(output_dir):
    sweeps = a_container(2)
    assert store.saved_path(sweeps) is None
    path = store.save(sweeps, "multisweep")
    assert store.saved_path(sweeps) == path


# ─── Files without rfmux ──────────────────────────────────────────────────────


def test_the_file_is_builtins_and_ndarrays(output_dir):
    """pickle.load on a machine with no rfmux has to work."""
    sweeps = a_container(2)
    path = store.save(sweeps, "multisweep")

    with path.open("rb") as f:
        raw = pickle.load(f)

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                assert type(k).__module__ == "builtins", k
                walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)
        else:
            assert type(node).__module__ in ("builtins", "numpy"), type(node)

    walk(raw)


# ─── Autosave, and turning it off ────────────────────────────────────────────


def test_maybe_save_honours_an_explicit_false(output_dir):
    assert store.maybe_save(a_container(2), "multisweep", save=False) is None
    assert not list(output_dir.rglob("*.pkl"))


def test_maybe_save_defers_to_the_setting(output_dir):
    store.set_autosave(False)
    assert store.maybe_save(a_container(2), "multisweep") is None

    store.set_autosave(True)
    assert store.maybe_save(a_container(2), "multisweep") is not None


def test_an_explicit_true_beats_the_setting(output_dir):
    store.set_autosave(False)
    assert store.maybe_save(a_container(2), "multisweep", save=True) is not None


def test_a_failed_autosave_warns_and_keeps_the_data(output_dir, monkeypatch):
    """A twenty-minute sweep is not thrown away over a read-only folder."""
    def explode(*a, **k):
        raise OSError("read-only file system")

    monkeypatch.setattr(store, "_save", explode)
    sweeps = a_container(2)

    with pytest.warns(UserWarning, match="still in hand"):
        assert store.maybe_save(sweeps, "multisweep") is None

    assert sweeps["crs0042_rmod2"]["module"] == 2


def test_an_explicit_save_raises_instead_of_warning(output_dir, monkeypatch):
    monkeypatch.setattr(store, "session_directory", lambda **kw: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        store.save(a_container(2), "multisweep")


# ─── Resolution order ────────────────────────────────────────────────────────


def test_the_setter_beats_the_environment(output_dir, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    store.set_output_directory(elsewhere)
    assert store.output_directory() == elsewhere

    store.set_output_directory(None)
    assert store.output_directory() == output_dir


def test_the_environment_beats_the_config_file(output_dir, tmp_path, monkeypatch):
    written = tmp_path / "rfmux.yaml"
    written.write_text(f"store:\n  directory: {tmp_path / 'from_config'}\n")
    monkeypatch.setenv("RFMUX_CONFIG", str(written))
    config.reload()

    assert store.output_directory() == output_dir

    monkeypatch.delenv("RFMUX_DATA_DIR")
    assert store.output_directory() == tmp_path / "from_config"


def test_the_config_file_can_turn_autosave_off(tmp_path, monkeypatch):
    written = tmp_path / "rfmux.yaml"
    written.write_text("store:\n  autosave: false\n")
    monkeypatch.setenv("RFMUX_CONFIG", str(written))
    monkeypatch.delenv("RFMUX_AUTOSAVE")
    config.reload()

    assert store.autosave_enabled() is False


def test_autosave_is_on_when_nobody_has_said_otherwise(tmp_path, monkeypatch):
    monkeypatch.setenv("RFMUX_CONFIG", str(tmp_path / "empty.yaml"))
    (tmp_path / "empty.yaml").write_text("{}\n")
    monkeypatch.delenv("RFMUX_AUTOSAVE")
    config.reload()

    assert store.autosave_enabled() is True


def test_a_config_pointing_nowhere_is_an_error_not_a_shrug(tmp_path, monkeypatch):
    monkeypatch.setenv("RFMUX_CONFIG", str(tmp_path / "absent.yaml"))
    config.reload()
    with pytest.raises(FileNotFoundError, match="RFMUX_CONFIG"):
        config.path()


def test_a_malformed_config_names_the_file(tmp_path, monkeypatch):
    written = tmp_path / "rfmux.yaml"
    written.write_text("store:\n  directory: [unclosed\n")
    monkeypatch.setenv("RFMUX_CONFIG", str(written))
    config.reload()
    with pytest.raises(ValueError, match=str(written)):
        config.get("store.directory")


def test_the_shipped_template_parses_and_documents_the_defaults(tmp_path, monkeypatch):
    """The template is what users copy; it should say what the code does."""
    copy = config.init(tmp_path / "rfmux.yaml")
    monkeypatch.setenv("RFMUX_CONFIG", str(copy))
    config.reload()

    assert config.get("store.autosave") is True
    assert config.get("store.directory") == store.DEFAULT_DIRECTORY


def test_init_refuses_to_clobber(tmp_path):
    config.init(tmp_path / "rfmux.yaml")
    with pytest.raises(FileExistsError):
        config.init(tmp_path / "rfmux.yaml")
    config.init(tmp_path / "rfmux.yaml", force=True)


# ─── The analyses that save ──────────────────────────────────────────────────


def a_searchable_netanal(module=2):
    """A netanal with a resonance in it, so a search finds something."""
    frequencies = np.linspace(0.999e9, 1.001e9, 801)
    iq = 1 - 0.9 / (1 + 2j * 1e4 * (frequencies - 1e9) / 1e9)
    netanal = a_netanal(module, npoints=801, iq=iq)
    netanal[f"crs0042_rmod{module}"]["results"][0]["upward"]["frequencies"] = frequencies
    return netanal


def a_searchable_module_netanal(module=2):
    """One module's searchable output — what the finder takes."""
    return a_searchable_netanal(module)[f"crs0042_rmod{module}"]


def stored_search(module_netanal) -> ResonanceSearch:
    """The search out of a netanal: an index and a from_dict, as a caller does it."""
    return ResonanceSearch.from_dict(
        netanal_trace(module_netanal)["resonance_search"]
    )


def test_find_resonances_saves_a_netanal_with_the_search_in_it(output_dir):
    """A netanal that has never been saved gets a file of its own, label and all."""
    search = find_resonances_in_netanal(
        a_searchable_module_netanal(), save=True, label="cold"
    )

    # A netanal file, not a find_resonances one: the search went into the
    # netanal, so the netanal is what there is to save.
    path = next(store.session_directory().glob("netanal_*.pkl"))
    assert path.stem.endswith("_cold")
    assert not list(store.session_directory().glob("find_resonances_*.pkl"))

    restored = stored_search(store.load(path))
    assert len(restored) == len(search)
    assert restored.resonance_frequencies_hz == pytest.approx(
        search.resonance_frequencies_hz
    )


def test_searching_a_saved_netanal_updates_the_file_it_came_from(output_dir):
    """The point of the change: an annotation, not a second file to keep paired."""
    netanal = a_searchable_netanal()
    path = store.save(netanal, "netanal", label="cold")

    find_resonances_in_netanal(netanal["crs0042_rmod2"], save=True)

    assert [p.name for p in store.session_directory().glob("*.pkl")] == [path.name]
    reloaded = store.load(path)
    assert len(stored_search(reloaded["crs0042_rmod2"])) >= 1
    # The measurement is still all there beside the search.
    assert reloaded["crs0042_rmod2"]["results"][0]["upward"]["iq_counts"].size == 801


def test_searching_one_module_leaves_the_others_in_the_file(output_dir):
    """One module's output re-saved splices itself back into the container."""
    netanal = a_searchable_netanal(2) | a_netanal(3, npoints=64)
    path = store.save(netanal, "netanal")

    find_resonances_in_netanal(netanal["crs0042_rmod2"], save=True)

    reloaded = store.load(path)
    assert set(reloaded) == {"crs0042_rmod2", "crs0042_rmod3"}
    assert len(stored_search(reloaded["crs0042_rmod2"])) >= 1
    assert "resonance_search" not in netanal_trace(reloaded["crs0042_rmod3"])


def test_the_module_label_a_search_derives_never_names_a_file(output_dir):
    """It is for warnings. A file holding eight modules is not "module 2"."""
    search = find_resonances_in_netanal(a_searchable_module_netanal(), save=True)

    assert search.label == "module 2"
    path = next(store.session_directory().glob("netanal_*.pkl"))
    assert path.stem.count("_") == 2  # netanal_YYYYMMDD_HHMMSS, no label
    assert store.load(path)[store.METADATA_KEY].get("label") is None


def test_find_resonances_does_not_save_when_told_not_to(output_dir):
    find_resonances_in_netanal(a_netanal(npoints=64)["crs0042_rmod1"], save=False)
    assert not list(output_dir.rglob("*.pkl"))


def test_a_saved_netanal_still_reads_as_one_trace(output_dir):
    """The file_metadata a save adds must not stop the trace being found."""
    netanal = a_searchable_netanal()
    store.save(netanal, "netanal")

    search = find_resonances_in_netanal(netanal["crs0042_rmod2"], save=False)
    assert len(search) >= 1
