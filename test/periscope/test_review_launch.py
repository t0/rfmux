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


def test_offline_review_reaches_no_board(monkeypatch):
    """A file under review names the board OFFLINE: nothing is loaded
    or connected to, so no warning about a host called "offline"."""
    from rfmux.tools.periscope import __main__ as m
    monkeypatch.setattr(m, "load_session",
                        lambda *a, **k: pytest.fail("a board was looked up"))
    assert m._resolve_board("OFFLINE") is None


@pytest.mark.parametrize("board, spec", [
    ("rfmux0156.local", 'serial: "0156"'), ("0156", 'serial: "0156"'),
    ("10.0.0.5", 'hostname: "10.0.0.5"')])
def test_a_board_argument_names_a_serial_or_a_host(monkeypatch, board, spec):
    from types import SimpleNamespace
    from rfmux.tools.periscope import __main__ as m
    seen = []

    async def resolved():
        pass
    crs = SimpleNamespace(resolve=resolved)
    monkeypatch.setattr(m, "load_session", lambda hwm: seen.append(hwm) or
                        SimpleNamespace(query=lambda cls: SimpleNamespace(
                            one=lambda: crs)))
    assert m._resolve_board(board) is crs
    assert seen == [f"!HardwareMap [ !CRS {{ {spec} }} ]"]
