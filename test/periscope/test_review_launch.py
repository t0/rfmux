"""``periscope --review FILE`` opens offline in the file's folder,
loaded as a session when it is one."""

import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope.__main__ import review_session  # noqa: E402
from rfmux.tools.periscope.session_startup_dialog import (  # noqa: E402
    UnifiedStartupDialog as Dialog)


def test_review_opens_the_files_folder_as_its_session(tmp_path):
    pulse = tmp_path / "pulse.h5"
    pulse.touch()
    assert review_session(pulse) == {
        "mode": Dialog.SESS_NONE, "path": str(tmp_path), "folder_name": None}
    (tmp_path / "session_metadata.json").write_text("{}")
    assert review_session(str(pulse))["mode"] == Dialog.SESS_LOAD
