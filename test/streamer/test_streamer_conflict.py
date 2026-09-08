"""find_streamer_conflict() must see both halves of a busy streamer port.

Two processes streaming to one port do not collide — SO_REUSEPORT means the
kernel hands each reader a share of the datagrams and nothing raises. The
symptom is a trace holding samples from two unrelated detectors, which is why
this is detected up front rather than debugged later.

Neither probe sees everything on its own, so both are tested:

    state                        plain bind    short read
    nothing running              free          silent
    something streaming          free          PACKETS
    something receiving          IN USE        (not reached)
"""

import socket
import threading
import time

import pytest

from rfmux.streamer import find_streamer_conflict

pytestmark = pytest.mark.portable


@pytest.fixture
def free_port():
    """A port nothing is using.

    Bind-and-release, so the number is known-good rather than hoped-for.
    Something else could claim it in the gap; nothing on the machine wants an
    arbitrary ephemeral UDP port, so that has to be raced for deliberately.
    """
    finder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    finder.bind(("", 0))
    port = finder.getsockname()[1]
    finder.close()
    return port


@pytest.fixture
def sender(free_port):
    """Stream datagrams at the port until the test is done with it.

    ~2 ms apart, close enough to the 1.9 ms of the real slow stream at its
    slowest decimation that the probe's timeout is being tested against a
    realistic gap and not an artificially dense one.
    """
    stop = threading.Event()

    def pump():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        while not stop.is_set():
            sock.sendto(b"x" * 64, ("127.0.0.1", free_port))
            time.sleep(0.002)
        sock.close()

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    time.sleep(0.05)  # let the first few land
    yield free_port
    stop.set()
    thread.join(timeout=1.0)


def test_quiet_port_is_no_conflict(free_port):
    assert find_streamer_conflict(port=free_port) is None


def test_a_receiver_is_detected(free_port):
    """Periscope plotting a mock stream: someone holds the port."""
    holder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    holder.bind(("", free_port))
    try:
        conflict = find_streamer_conflict(port=free_port)
    finally:
        holder.close()

    assert conflict is not None
    assert "receiving" in conflict
    assert str(free_port) in conflict


def test_a_sender_with_no_receiver_is_detected(sender):
    """A mock server outliving the kernel that made it: nobody is listening,
    but packets are still on the wire. Only the read probe sees this."""
    conflict = find_streamer_conflict(port=sender)

    assert conflict is not None
    assert "arriving" in conflict


def test_the_read_probe_is_bounded_when_nothing_is_sending(free_port):
    """The timeout is the whole cost of the check on an idle machine."""
    started = time.perf_counter()
    find_streamer_conflict(port=free_port, timeout=0.05)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.5, f"idle probe took {elapsed*1000:.0f} ms"


@pytest.mark.skipif(
    not hasattr(socket, "SO_REUSEPORT"),
    reason="SO_REUSEPORT is POSIX-only; Windows has no equivalent")
def test_the_probe_refuses_reuseport(free_port):
    """The bind probe deliberately sets no socket options.

    The streamer's own readers set SO_REUSEPORT. A probe that did the same
    would bind cheerfully alongside one and report the port free — the exact
    condition it exists to find. Asserted rather than commented because the
    two sockets look identical at the call site.
    """
    reader = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    reader.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    reader.bind(("", free_port))

    permissive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    permissive.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    try:
        permissive.bind(("", free_port))
        reuseport_would_miss_it = True
    except OSError:
        reuseport_would_miss_it = False
    finally:
        permissive.close()

    try:
        conflict = find_streamer_conflict(port=free_port)
    finally:
        reader.close()

    assert reuseport_would_miss_it, \
        "SO_REUSEPORT no longer shares ports; this probe's design assumes it"
    assert conflict is not None, "the plain bind missed a SO_REUSEPORT reader"
