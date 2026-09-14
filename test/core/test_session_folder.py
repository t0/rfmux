"""The session folder Periscope and the recorder share: its metadata
file, the export listing, and the newest export by that listing."""

import datetime

from rfmux.core import session_folder as sf


def test_a_new_session_has_its_metadata_and_an_existing_one_keeps_it(tmp_path):
    folder = sf.open_session(base=tmp_path)
    assert folder.parent == tmp_path and folder.name.startswith("session_")
    assert sf.is_session(folder)
    meta = sf.load_metadata(folder)
    assert meta["folder_name"] == folder.name
    assert meta["base_path"] == str(tmp_path.resolve())
    assert meta["exports"] == [] and meta["screenshots"] == []
    sf.save_metadata(folder, dict(meta, created="then"))
    assert sf.open_session(folder) == folder
    assert sf.load_metadata(folder)["created"] == "then"
    assert sf.load_metadata(tmp_path) == {}


def test_names_carry_the_stamp_and_the_export_the_time_alone():
    now = datetime.datetime(2026, 9, 12, 15, 43, 31)
    assert sf.folder_name(now) == "session_20260912_154331"
    assert sf.export_filename("bias", "module 2/a", now=now) == \
        "bias_module_2_a_154331.pkl"
    assert sf.export_filename("pulse", "modules2+3", ".h5", now=now) == \
        "pulse_modules2+3_154331.h5"


def test_the_newest_export_is_by_the_listing_not_the_file(tmp_path):
    folder = sf.open_session(base=tmp_path)
    for name, stamp in (("bias_module2_120000.pkl", "2026-09-09T12:00:00"),
                        ("bias_module2_100000.pkl", "2026-09-09T10:00:00"),
                        ("bias_module1_110000.pkl", "2026-09-09T11:00:00")):
        (folder / name).write_bytes(b"x")
        sf.register_export(folder, name, "bias", name.split("_")[1], stamp)
    (folder / "bias_module2_130000.pkl").write_bytes(b"x")   # not listed
    assert [e["filename"] for e in sf.exports(folder, "bias", "module2")] == \
        ["bias_module2_100000.pkl", "bias_module2_120000.pkl"]
    assert sf.latest_export(folder, "bias", "module2") == \
        folder / "bias_module2_120000.pkl"
    assert sf.latest_export(folder, "bias", "module3") is None
    # A listed file that is gone is not an export.
    (folder / "bias_module2_120000.pkl").unlink()
    assert sf.latest_export(folder, "bias", "module2") == \
        folder / "bias_module2_100000.pkl"


def test_the_newest_session_under_a_base_is_by_name(tmp_path):
    for name in ("session_20260901_090000", "session_20260910_154331"):
        sf.open_session(tmp_path / name)
    (tmp_path / "session_20260911_000000").mkdir()       # no metadata
    assert sf.newest_session(tmp_path) == tmp_path / "session_20260910_154331"
    assert sf.newest_session(tmp_path / "none") is None
