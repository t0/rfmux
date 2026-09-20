"""The managed Jupyter server survives a busy port.

Periscope asks Jupyter for port 8888. If that port is taken Jupyter moves
to another one, and Periscope must open the URL Jupyter actually chose
rather than the one it asked for.
"""

import http.server
import json
import shutil
import socket
import threading
import urllib.request

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("jupyterlab")

from PyQt6 import QtCore  # noqa: E402

from rfmux.tools.periscope.notebook_panel import JupyterServerManager  # noqa: E402


needs_jupyter = pytest.mark.skipif(
    shutil.which("jupyter") is None, reason="jupyter not on PATH")


def _hold(family: int, host: str) -> tuple[socket.socket, int]:
    """Listen on an OS-chosen port so Jupyter cannot have it."""
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.bind((host, 0))
    sock.listen(1)
    return sock, sock.getsockname()[1]


def _run_until_ready(qt_app, manager, notebook_dir, port, timeout_s=90) -> dict:
    outcome = {}
    loop = QtCore.QEventLoop()
    manager.server_ready.connect(
        lambda url: (outcome.__setitem__("ready", url), loop.quit()))
    manager.server_error.connect(
        lambda err: (outcome.__setitem__("error", err), loop.quit()))
    QtCore.QTimer.singleShot(timeout_s * 1000, loop.quit)
    manager.start(str(notebook_dir), port=port)
    loop.exec()
    return outcome


def _assert_moved_and_reachable(manager, outcome, held_port):
    assert "ready" in outcome, outcome
    assert manager.port != held_port
    assert manager.url == f"{manager.base_url}lab?token={manager.token}"
    with urllib.request.urlopen(
            f"{manager.base_url}api/status?token={manager.token}", timeout=5) as resp:
        assert resp.status == 200


@needs_jupyter
def test_url_follows_the_port_jupyter_took(qt_app, tmp_path):
    # Only the IPv6 side of localhost is busy: a plain IPv4 bind test says
    # the port is free, but Jupyter binds both sides and moves on. The URL
    # must come from where Jupyter landed, not from the pre-check.
    try:
        sock, port = _hold(socket.AF_INET6, "::1")
    except OSError:
        pytest.skip("no IPv6 loopback")
    manager = JupyterServerManager()
    try:
        outcome = _run_until_ready(qt_app, manager, tmp_path, port)
        _assert_moved_and_reachable(manager, outcome, port)
    finally:
        manager.stop()
        sock.close()


class _Server(http.server.BaseHTTPRequestHandler):
    status = 200

    def do_GET(self):
        self.send_response(self.status)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


@pytest.fixture
def http_status():
    """Serve every GET with the status the test sets; yields the base URL."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _Server)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield _Server, f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()


def test_probe_needs_an_authenticated_200(http_status):
    handler, base_url = http_status
    manager = JupyterServerManager()
    manager.token = "abc"
    handler.status = 403  # what Jupyter answers to a wrong token
    assert manager._responds(base_url) is False
    handler.status = 200
    assert manager._responds(base_url) is True


def test_info_file_is_found_by_token_not_pid(tmp_path, monkeypatch):
    # On Windows the server runs in a grandchild, so the pid in the file
    # name is not the one Periscope holds; the token is what identifies it.
    monkeypatch.setenv("JUPYTER_RUNTIME_DIR", str(tmp_path))
    manager = JupyterServerManager()
    manager.token = "abc"
    (tmp_path / "jpserver-1.json").write_text(
        json.dumps({"url": "http://localhost:8888/", "token": "someone-else"}))
    assert manager._server_info() is None
    (tmp_path / "jpserver-999.json").write_text(
        json.dumps({"url": "http://localhost:8889/", "token": "abc"}))
    assert manager._server_info()["url"] == "http://localhost:8889/"


def test_half_written_info_file_reads_as_not_there(tmp_path, monkeypatch):
    monkeypatch.setenv("JUPYTER_RUNTIME_DIR", str(tmp_path))
    manager = JupyterServerManager()
    manager.token = "abc"
    path = tmp_path / "jpserver-4242.json"

    assert manager._server_info() is None  # not written yet
    path.write_text("{\n  \"token\": \"ab")
    assert manager._server_info() is None  # Jupyter is mid-write
    path.write_text(json.dumps({"url": "http://localhost:8889/", "token": "abc"}))
    assert manager._server_info()["url"] == "http://localhost:8889/"
