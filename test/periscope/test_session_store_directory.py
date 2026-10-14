"""Test that Periscope saves measurements directly in the active session folder."""

import pytest

pytest.importorskip("PyQt6")

from pathlib import Path  # noqa: E402

from rfmux.tools.periscope.session_manager import SessionManager  # noqa: E402
from rfmux.tuning import store  # noqa: E402


@pytest.fixture(autouse=True)
def restore_output_directory():
    yield
    store.set_output_directory(None)


def test_the_session_folder_is_where_the_library_saves(tmp_path, qt_app):
    manager = SessionManager()
    session = manager.start_session(str(tmp_path), "session_under_test")

    assert store.session_directory() == Path(session)

    manager.end_session()
    assert store.session_directory(create=False) != Path(session)


def test_loading_a_session_moves_the_output_there(tmp_path, qt_app):
    existing = tmp_path / "an_old_session"
    existing.mkdir()
    manager = SessionManager()

    assert manager.load_session(str(existing))
    assert store.session_directory() == existing



def test_an_export_listed_behind_its_back_survives_the_next_save(tmp_path, qt_app):
    """crs.apply_bias lists its bias_catalog in the folder itself; the
    session manager's own next listing must not drop it."""
    from rfmux.core import session_folder
    manager = SessionManager()
    session = Path(manager.start_session(str(tmp_path), "session_under_test"))
    session_folder.register_export(session, "bias_catalog_1.pkl",
                                   "bias_catalog", "module1")

    manager.register_external_file(str(session / "pulse.h5"), "pulse", "module1")

    listed = {e["filename"]
              for e in session_folder.load_metadata(session)["exports"]}
    assert {"bias_catalog_1.pkl", "pulse.h5"} <= listed
    manager.end_session()
