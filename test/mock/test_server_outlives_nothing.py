"""A mock server exits when the process that started it dies without
shutting it down: a crashed or killed client leaves nothing holding
memory and ports."""
import os
import subprocess
import sys
import time

import pytest

CLIENT = '''
import os, signal
import rfmux
from rfmux.mock import server

rfmux.load_session("""
!HardwareMap
- !flavour "rfmux.mock"
- !CRS { serial: "0000", hostname: "127.0.0.1" }
""")
print(server._server_processes[-1].pid, flush=True)
os.kill(os.getpid(), signal.SIGKILL)      # no atexit, no daemon cleanup
'''


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(sys.platform == "win32",
                    reason="a Windows child keeps its parent's pid on record")
def test_a_server_exits_when_its_client_is_killed(tmp_path):
    # To a file: a server left behind would hold a pipe open.
    out = tmp_path / "client.out"
    with open(out, "w") as f:
        subprocess.run([sys.executable, "-c", CLIENT], stdout=f,
                       stderr=subprocess.DEVNULL, timeout=120)
    server_pid = int(next(line for line in out.read_text().splitlines()
                          if line.strip().isdigit()))
    deadline = time.monotonic() + 15.0
    while _alive(server_pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    try:
        assert not _alive(server_pid), "the server outlived its client"
    finally:
        if _alive(server_pid):
            os.kill(server_pid, 9)
