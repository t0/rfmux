"""A second receiver starves the mock's stream, but not a board's.

The two transports differ, and the difference decides whether warning
about a competing receiver is a service or a false alarm:

  unicast to loopback  -- the kernel picks ONE socket per datagram by
                          hash of the sender's 4-tuple, so a second
                          receiver takes the whole stream, not a share
  multicast to a group -- every joined socket gets its own copy, so
                          multiple readers are fine and expected

The mock sends unicast (rfmux/mock/udp_streamer.py does
``sendto((self.host, port))`` with host 127.0.0.1) whatever its
"(multicast)" log line says. A real board multicasts.
"""

import socket
import struct

import pytest

from rfmux.streamer import (MULTICAST_GROUP, find_competing_receiver,
                            find_streamer_conflict)


@pytest.fixture
def free_port():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _reuse_socket(port, join_group=False):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    s.bind(("", port))
    if join_group:
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                     socket.inet_aton("127.0.0.1"))
        mreq = struct.pack("4s4s", socket.inet_aton(MULTICAST_GROUP),
                           socket.inet_aton("127.0.0.1"))
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    s.setblocking(False)
    return s


@pytest.mark.portable
def test_quiet_when_the_port_is_free(free_port):
    assert find_competing_receiver(port=free_port) is None


@pytest.mark.portable
def test_reports_a_receiver_that_would_take_our_packets(free_port):
    holder = _reuse_socket(free_port)
    try:
        message = find_competing_receiver(port=free_port)
        assert message, "a competing receiver went unreported"
        assert str(free_port) in message
    finally:
        holder.close()


@pytest.mark.portable
def test_silent_for_a_real_board(free_port):
    """A board multicasts, so other readers cost nothing -- do not warn."""
    holder = _reuse_socket(free_port)
    try:
        assert find_competing_receiver("rfmux0156.local",
                                       port=free_port) is None, \
            "warned about a second reader on hardware, where every socket " \
            "joined to the group gets its own copy anyway"
    finally:
        holder.close()


@pytest.mark.portable
def test_the_probe_sees_past_so_reuseport(free_port):
    """The holder sets SO_REUSEPORT; a probe that did too would miss it."""
    holder = _reuse_socket(free_port)
    try:
        assert find_competing_receiver(port=free_port) is not None
    finally:
        holder.close()
    assert find_competing_receiver(port=free_port) is None


@pytest.mark.portable
def test_conflict_check_still_sees_a_receiver(free_port):
    """find_streamer_conflict shares the probe; keep its half working."""
    holder = _reuse_socket(free_port)
    try:
        conflict = find_streamer_conflict(port=free_port)
        assert conflict and "receiving" in conflict
    finally:
        holder.close()


def _drain(s):
    n = 0
    while True:
        try:
            s.recv(2048)
            n += 1
        except BlockingIOError:
            return n


@pytest.mark.portable
def test_unicast_starves_the_second_listener(free_port):
    """Why the warning exists at all -- this is the mock's transport."""
    a, b = _reuse_socket(free_port), _reuse_socket(free_port)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx.bind(("127.0.0.1", 0))
    try:
        for _ in range(200):
            tx.sendto(b"x" * 64, ("127.0.0.1", free_port))
        got = sorted((_drain(a), _drain(b)))
        assert got[0] == 0, \
            f"expected one listener starved, got {got} -- if unicast now " \
            f"shares, find_competing_receiver need not warn"
        assert got[1] > 0
    finally:
        for s in (a, b, tx):
            s.close()


@pytest.mark.portable
def test_multicast_feeds_every_listener(free_port):
    """Why the warning is scoped to loopback -- this is a board."""
    a = _reuse_socket(free_port, join_group=True)
    b = _reuse_socket(free_port, join_group=True)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
    tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                  socket.inet_aton("127.0.0.1"))
    try:
        for _ in range(200):
            tx.sendto(b"x" * 64, (MULTICAST_GROUP, free_port))
        na, nb = _drain(a), _drain(b)
        assert na > 0 and nb > 0, \
            f"multicast did not reach both listeners ({na}, {nb}); the " \
            f"loopback-only scoping of find_competing_receiver assumes it does"
    finally:
        for s in (a, b, tx):
            s.close()
