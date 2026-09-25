"""The first mock server on a host serves at MOCK_PORT, so a client
finds it untold; a second takes an ephemeral port and still serves."""

import socket

import rfmux
from rfmux.mock import server


def _mock(serial: str):
    session = rfmux.load_session(f"""
!HardwareMap
- !flavour "rfmux.mock"
- !CRS {{ serial: "{serial}" }}
""")
    return session.query(rfmux.CRS).one()


def test_the_first_mock_takes_the_port_and_is_found_there():
    if server.running_mock() is not None:
        # Another test's server holds the port for this session: then
        # this one is the second, on a port of its own.
        crs = _mock("0771")
        assert crs.hostname != f"127.0.0.1:{server.MOCK_PORT}"
        return
    crs = _mock("0770")
    assert crs.hostname == f"127.0.0.1:{server.MOCK_PORT}"
    assert server.running_mock() == crs.hostname


def test_a_held_port_leaves_the_mock_on_another(tmp_path):
    holder = socket.socket()
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        holder.bind(("localhost", server.MOCK_PORT))
        holder.listen(1)
    except OSError:
        holder = None                      # a mock already has it
    try:
        crs = _mock("0772")
        assert crs.hostname != f"127.0.0.1:{server.MOCK_PORT}"
        assert crs.hostname.startswith("127.0.0.1:")
    finally:
        if holder is not None:
            holder.close()


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
