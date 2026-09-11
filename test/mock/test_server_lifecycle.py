"""Mock sessions own their servers, including failed and interrupted runs."""
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

import rfmux
from rfmux.mock import server
from rfmux.mock.crs import ServerMockCRS
from rfmux.mock.standard_array import SESSION
from rfmux.mock.udp_streamer import MockCRSStreamer, MockUDPManager


def test_repeated_sessions_create_one_server_each():
    baseline = set(server._server_processes)
    sessions = []
    try:
        for count in range(1, 4):
            sessions.append(rfmux.load_session(SESSION))
            assert len(set(server._server_processes) - baseline) == count
    finally:
        for session in sessions:
            session.close()


def test_session_close_reaps_only_its_server():
    first = rfmux.load_session(SESSION)
    first_process = server._server_processes[-1]
    second = rfmux.load_session(SESSION)
    second_process = server._server_processes[-1]
    try:
        first.close()
        first.close()
        assert not first_process.is_alive()
        assert second_process.is_alive()
    finally:
        first.close()
        second.close()
    assert not second_process.is_alive()


def test_shutdown_can_be_used_again_after_creating_another_session():
    for _ in range(2):
        session = rfmux.load_session(SESSION)
        process = server._server_processes[-1]
        try:
            server._shutdown_all_servers()
            assert not process.is_alive()
        finally:
            session.close()


@pytest.mark.skipif(server.mp_ctx.get_start_method() != "fork",
                    reason="injects a server failure across fork")
def test_server_startup_failure_is_reaped(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("injected startup failure")

    baseline = set(server._server_processes)
    monkeypatch.setattr(server, "ServerMockCRS", fail)
    with pytest.raises(RuntimeError, match="server failed to start"):
        rfmux.load_session(SESSION)
    assert set(server._server_processes) == baseline


@pytest.mark.skipif(server.mp_ctx.get_start_method() != "fork",
                    reason="observes child cleanup across fork")
def test_session_close_calls_board_stream_cleanup(monkeypatch):
    stopped = server.mp_ctx.Event()

    async def stop(self):
        stopped.set()

    monkeypatch.setattr(ServerMockCRS, "stop_udp_streaming", stop)
    session = rfmux.load_session(SESSION)
    session.close()
    assert stopped.is_set()


def test_streamer_preserves_signal_handlers(monkeypatch):
    from rfmux.mock import udp_streamer

    monkeypatch.setattr(udp_streamer, "_cleanup_registered", False)
    signals = (signal.SIGINT, signal.SIGTERM)
    before = [signal.getsignal(sig) for sig in signals]
    streamer = MockCRSStreamer(ServerMockCRS("0000"))
    try:
        assert [signal.getsignal(sig) for sig in signals] == before
    finally:
        streamer.emergency_stop()
        for sig, handler in zip(signals, before):
            signal.signal(sig, handler)


def test_timed_out_streamer_remains_visible_and_can_be_reaped():
    class StuckStreamer:
        ident = 1
        alive = True

        def stop(self):
            pass

        def join(self, timeout):
            pass

        def emergency_stop(self):
            pass

        def is_alive(self):
            return self.alive

    manager = MockUDPManager(None)
    streamer = StuckStreamer()
    manager._streamer = streamer
    manager._streaming_active = True
    assert asyncio.run(manager.stop_udp_streaming()) is False
    assert manager.get_udp_streaming_status()["thread_alive"] is True
    assert asyncio.run(manager.start_udp_streaming(host="127.0.0.1")) is False
    streamer.alive = False
    assert asyncio.run(manager.stop_udp_streaming()) is True
    assert manager.get_udp_streaming_status()["thread_alive"] is False


_PARENT = '''
import json
import multiprocessing
from pathlib import Path
import sys

def main():
    multiprocessing.set_start_method(sys.argv[3])
    import rfmux
    from rfmux.mock.server import _server_processes
    from rfmux.mock.standard_array import SESSION
    if sys.argv[2] == "killed_busy":
        import threading
        import time
        import urllib.request
        from rfmux.mock.crs import ServerMockCRS
        entered = multiprocessing.Event()

        async def stall(self):
            entered.set()
            time.sleep(60)

        ServerMockCRS.get_udp_streaming_status = stall
    session = rfmux.load_session(SESSION)
    process = _server_processes[-1]
    if sys.argv[2] == "killed_busy":
        hostname = session.query(rfmux.CRS).one().hostname

        def block_server():
            request = urllib.request.Request(
                "http://" + hostname + "/tuber",
                data=b'{"method":"get_udp_streaming_status"}')
            urllib.request.urlopen(request, timeout=30).close()

        threading.Thread(target=block_server, daemon=True).start()
        assert entered.wait(10), "RPC did not enter blocking call"
    ready = Path(sys.argv[1])
    temporary = ready.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pid": process.pid}))
    temporary.replace(ready)
    sys.stdin.readline()
    if sys.argv[2] == "failure":
        raise RuntimeError("injected parent failure")

if __name__ == "__main__":
    main()
'''


def _running(pid):
    status = Path(f"/proc/{pid}/stat")
    try:
        return status.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


@pytest.mark.skipif(sys.platform != "linux", reason="uses /proc to check orphans")
@pytest.mark.parametrize("method", ["fork", "spawn"])
@pytest.mark.parametrize("ending", ["normal", "failure", "killed"])
def test_server_exits_when_parent_exits(tmp_path, method, ending):
    script = tmp_path / "parent.py"
    script.write_text(_PARENT)
    ready = tmp_path / "ready.json"
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()))
    parent = subprocess.Popen(
        [sys.executable, str(script), str(ready), ending, method],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, env=env,
    )
    pid = None
    try:
        deadline = time.monotonic() + 40
        while not ready.exists():
            if parent.poll() is not None:
                pytest.fail(parent.communicate()[0])
            assert time.monotonic() < deadline, "parent failed to become ready"
            time.sleep(0.05)
        pid = json.loads(ready.read_text())["pid"]
        assert _running(pid)
        if ending.startswith("killed"):
            parent.kill()
        output, _ = parent.communicate(input="\n", timeout=15)
        assert parent.returncode == {"normal": 0, "failure": 1,
                                     "killed": -signal.SIGKILL,
                                     "killed_busy": -signal.SIGKILL}[ending], output
        deadline = time.monotonic() + 8
        while _running(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _running(pid), f"orphaned mock server {pid}: {output}"
    finally:
        if parent.poll() is None:
            parent.kill()
        if pid is not None and _running(pid):
            os.kill(pid, signal.SIGKILL)
        parent.communicate(timeout=10)


@pytest.mark.parametrize("builder", ["standard_array", "create_mock_crs"])
def test_failed_board_setup_closes_session(monkeypatch, builder):
    from unittest.mock import AsyncMock
    from rfmux.mock.crs import ClientMockCRS
    from rfmux.mock.helpers import create_mock_crs
    from rfmux.mock.standard_array import standard_array

    baseline = set(server._server_processes)
    monkeypatch.setattr(ClientMockCRS, "resolve", AsyncMock(
        side_effect=RuntimeError("injected resolve failure")))
    build = {"standard_array": standard_array,
             "create_mock_crs": create_mock_crs}[builder]
    with pytest.raises(Exception, match="injected resolve failure"):
        asyncio.run(build())
    assert set(server._server_processes) == baseline


@pytest.mark.parametrize("failure", ["setup", "assertion"])
def test_pytest_module_cleanup_reaps_failed_tests(tmp_path, failure):
    (tmp_path / "test_a.py").write_text('''
import pytest
import rfmux
from rfmux.mock.standard_array import SESSION

@pytest.fixture(scope="module")
def board():
    session = rfmux.load_session(SESSION)
    if FAILURE == "setup":
        raise RuntimeError("injected fixture failure")
    return session

def test_failure(board):
    assert False, "injected assertion failure"
'''.replace('FAILURE', repr(failure)))
    (tmp_path / "test_b.py").write_text('''
import multiprocessing

def test_no_leftover_servers():
    assert not multiprocessing.active_children()
''')
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path),
         "-p", "test.conftest", "-p", "no:cacheprovider", "-q"],
        env=dict(os.environ, PYTHONPATH=str(Path.cwd())),
        capture_output=True, text=True, timeout=60,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "1 passed" in output, output
    assert ("1 error" if failure == "setup" else "1 failed") in output, output


@pytest.mark.skipif(sys.platform != "linux", reason="uses /proc to check orphans")
def test_busy_server_exits_when_parent_is_killed(tmp_path):
    test_server_exits_when_parent_exits(tmp_path, "fork", "killed_busy")
