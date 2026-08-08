"""
Tests for Periscope's UDP queue selection (issue #98).

The receiver socket is bound to the streamer port on all interfaces, so
packets from any CRS multicasting on the network can reach it. Queue
selection must therefore be pinned to the serial we expect: previously the
first queue matching the module index won, and a burst of packets from
another board at startup could permanently capture the selection — the UI
then reported 0 packets/s while the desired stream accumulated unread.
"""

import pytest

from rfmux.tools.periscope.tasks import select_queue

Q1, Q2 = object(), object()


def test_prefers_expected_serial_over_arrival_order():
    # Interloper serial 24 arrived (and parsed) first; our board is serial 0
    queues = [(24, 0, Q1), (0, 0, Q2)]
    assert select_queue(queues, module_idx=0, expected_serial=0) == (0, Q2)


def test_does_not_latch_onto_foreign_serial():
    # Only a foreign serial present: wait rather than show someone else's data
    queues = [(24, 0, Q1)]
    assert select_queue(queues, module_idx=0, expected_serial=0) is None


def test_module_filter_still_applies():
    # Right serial, wrong module: keep waiting
    queues = [(0, 1, Q1)]
    assert select_queue(queues, module_idx=0, expected_serial=0) is None
    assert select_queue(queues, module_idx=1, expected_serial=0) == (0, Q1)


def test_no_expected_serial_keeps_first_match_behaviour():
    # Hardware launched by hostname (serial unknown): previous behaviour
    queues = [(24, 0, Q1), (0, 0, Q2)]
    assert select_queue(queues, module_idx=0, expected_serial=None) == (24, Q1)


def test_empty_queues():
    assert select_queue([], module_idx=0, expected_serial=0) is None
