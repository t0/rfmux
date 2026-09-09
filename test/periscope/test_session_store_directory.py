"""A Periscope session is where the library writes.

``store`` is what saves a measurement, in Periscope as in a notebook, so a
session folder has to be its output directory for as long as the session lasts
-- flat, with none of the dated ``ipy_session_`` folder a notebook gets.
"""

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
