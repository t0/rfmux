"""The command-line viewer releases owned resources before Python exits."""

from types import SimpleNamespace

import pytest


pytest.importorskip("PyQt6")

from rfmux.tools.periscope.__main__ import (  # noqa: E402
    _run_cli_application,
    _shutdown_cli_resources,
)


def test_cli_application_always_releases_resources_after_qt_returns():
    calls = []
    app = SimpleNamespace(exec=lambda: calls.append("exec") or 17)
    viewer = SimpleNamespace(close=lambda: calls.append("viewer"))
    session = SimpleNamespace(close=lambda: calls.append("session"))
    loop = SimpleNamespace(
        is_closed=lambda: False,
        close=lambda: calls.append("loop"),
    )

    assert _run_cli_application(app, viewer, session, loop) == 17
    assert calls == ["exec", "viewer", "session", "loop"]


def test_cli_shutdown_closes_window_session_then_event_loop():
    calls = []
    viewer = SimpleNamespace(close=lambda: calls.append("viewer"))
    session = SimpleNamespace(close=lambda: calls.append("session"))
    loop = SimpleNamespace(
        is_closed=lambda: False,
        close=lambda: calls.append("loop"),
    )

    _shutdown_cli_resources(viewer, session, loop)

    assert calls == ["viewer", "session", "loop"]


def test_cli_shutdown_accepts_failed_optional_setup():
    calls = []
    viewer = SimpleNamespace(close=lambda: calls.append("viewer"))

    _shutdown_cli_resources(viewer, None, None)

    assert calls == ["viewer"]
