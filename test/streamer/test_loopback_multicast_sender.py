"""One sender configuration, shared by the multicast check and the mock."""

import socket

import pytest

from rfmux.streamer import loopback_multicast_sender

pytestmark = pytest.mark.portable


def test_the_sender_cannot_leave_this_host():
    """TTL 0 is the containment: the mock shares a group with real boards."""
    sock = loopback_multicast_sender()
    try:
        assert sock.getsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL) == 0
    finally:
        sock.close()


def test_the_sender_is_pinned_to_loopback():
    """0.0.0.0 would let the kernel route the group onto the real NIC."""
    sock = loopback_multicast_sender()
    try:
        raw = sock.getsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, 4)
        assert socket.inet_ntoa(raw) == "127.0.0.1"
    finally:
        sock.close()
