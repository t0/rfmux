"""A module nobody is streaming must not look like a dead stream.

Every counter ``UDPReceiver`` exposes reads through ``self.queue``, which
is only set once a queue's module matches.  When it never matches, the
receiver is working perfectly and reports zero packets, zero loss and no
error -- so Periscope draws nothing and says nothing about why.  That is
how a viewer left on module 2 (the startup dialog restores the last-used
module) reads a mock that only ever streams module 1.
"""

import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope.tasks import UDPReceiver


class _Queue:
    """Stand-in for the C++ queue; only get_stats is ever reached here."""

    class _Stats:
        packets_received = 4321
        packets_dropped = 0
        packets_missing = 0
        sequence_gaps = 0

    def get_stats(self):
        return self._Stats()


def _stub_queues(modules):
    return type(
        "Stub", (), {"get_all_queues":
                     lambda self: [(0, m - 1, _Queue()) for m in modules]}
    )()


def _receiver_watching(module, streaming):
    """A UDPReceiver watching *module* while *streaming* modules arrive.

    Built without __init__ on purpose: these tests exercise queue
    discovery, and the real constructor binds UDP 9876.  A silent extra
    listener on the streaming port is the one thing that must never be
    left lying around in this test suite.
    """
    rx = UDPReceiver.__new__(UDPReceiver)
    rx.module_id = module
    rx.module_idx = module - 1
    rx.queue = None
    rx.serial = None
    rx._module_mismatch = None
    rx.packets_received = 0
    rx.packets_dropped = 0
    rx.receiver = _stub_queues(streaming)
    return rx


def test_matching_module_adopts_the_queue():
    rx = _receiver_watching(module=1, streaming=[1])
    rx._discover_queue()
    assert rx.queue is not None
    assert rx.get_module_mismatch() is None


def test_unmatched_module_is_reported():
    rx = _receiver_watching(module=2, streaming=[1])
    rx._discover_queue()
    assert rx.queue is None
    msg = rx.get_module_mismatch()
    assert msg, "a module that never matches must say so"
    # Both numbers, because the fix is to change one of them.
    assert "2" in msg and "1" in msg


def test_nothing_streaming_yet_is_not_a_mismatch():
    """Startup, a board not yet streaming: silence here is normal."""
    rx = _receiver_watching(module=2, streaming=[])
    rx._discover_queue()
    assert rx.get_module_mismatch() is None


def test_message_is_stable_across_repeated_discovery():
    """run() re-discovers every batch; it must not spam or flap."""
    rx = _receiver_watching(module=2, streaming=[1])
    rx._discover_queue()
    first = rx.get_module_mismatch()
    for _ in range(5):
        rx._discover_queue()
    assert rx.get_module_mismatch() == first


def test_mismatch_clears_once_our_module_appears():
    rx = _receiver_watching(module=2, streaming=[1])
    rx._discover_queue()
    assert rx.get_module_mismatch()
    rx.receiver = _stub_queues([2])
    rx._discover_queue()
    assert rx.queue is not None
    assert rx.get_module_mismatch() is None


def test_counters_are_flat_zero_during_a_mismatch():
    """The reason the message exists: the numbers cannot show this."""
    rx = _receiver_watching(module=2, streaming=[1])
    rx._discover_queue()
    assert rx.get_received_packets() == 0
    assert rx.get_dropped_packets() == 0
    assert rx.get_missing_packets() == 0
    assert rx.get_module_mismatch(), \
        "zero packets and zero loss is indistinguishable from a dead " \
        "stream unless the receiver says which module it is watching"
