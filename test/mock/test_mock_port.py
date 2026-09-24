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
