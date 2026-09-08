import socket

import pytest


@pytest.fixture
def free_port():
    """A UDP port nothing is using: bound and released, so it is known-good."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port
