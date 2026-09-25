"""The first mock server on a host serves at MOCK_PORT, so a client
finds it untold; a second takes an ephemeral port and still serves.
The port tests use a port of their own: a process's hardware map keeps
every board loaded into it, and a new mock map binds them all again."""

import socket

import pytest

from rfmux.mock import server


@pytest.fixture
def port(monkeypatch):
    """A port free for this test stands in for MOCK_PORT, so a mock
    already running on the machine (Periscope's, another test's) leaves
    the contract to be checked every time.  Taken below every system's
    ephemeral range, where no outgoing connection can take it before
    the mock binds it."""
    import random
    for free in random.Random().sample(range(20000, 30000), 50):
        probe = socket.socket()
        try:
            probe.bind(("localhost", free))
            probe.listen(1)
        except OSError:
            continue
        finally:
            probe.close()
        monkeypatch.setattr(server, "MOCK_PORT", free)
        return free
    pytest.skip("no free port in 20000-30000")


def test_a_free_mock_port_is_taken_and_found(port):
    s = server._listening_socket()
    try:
        assert s.getsockname()[1] == port
        assert server.running_mock() == f"127.0.0.1:{port}"
    finally:
        s.close()
    assert server.running_mock() is None


def test_a_held_mock_port_leaves_the_next_socket_on_another(port):
    """Held by another mock server: the next listens elsewhere rather
    than share it."""
    first = server._listening_socket()
    try:
        second = server._listening_socket()
        try:
            assert second.getsockname()[1] != port
        finally:
            second.close()
    finally:
        first.close()


#: Loads a map of two mock boards, checks each answers at its address,
#: and prints the addresses.
TWO_BOARDS = '''
import socket
import rfmux
session = rfmux.load_session("""
!HardwareMap
- !flavour "rfmux.mock"
- !CRS { serial: "0781" }
- !CRS { serial: "0782" }
""")
hosts = sorted(c.hostname for c in session.query(rfmux.CRS))
for host in hosts:
    h, _, p = host.rpartition(":")
    socket.create_connection((h, int(p)), timeout=2).close()
print("HOSTS", *hosts)
'''


def test_two_boards_of_one_map_are_served_at_two_ports():
    """Both sockets are bound before the server listens on either: each
    must still get a port of its own, and the map must load rather than
    wait on a server that failed to listen.  In a process of its own,
    with a deadline: the hardware map's database belongs to the thread
    and process that open it, and a hung load must fail, not hang."""
    import subprocess
    import sys
    try:
        out = subprocess.run([sys.executable, "-c", TWO_BOARDS],
                             capture_output=True, text=True, timeout=90).stdout
    except subprocess.TimeoutExpired:
        raise AssertionError("the map never loaded: a server failed to listen")
    hosts = next(line.split()[1:] for line in out.splitlines()
                 if line.startswith("HOSTS"))
    assert len(hosts) == 2 and len(set(hosts)) == 2
