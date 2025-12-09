#!/usr/bin/env -S PYTHONPATH=.. pytest-3 -v -W error::UserWarning

"""
Tests for the C++ packet receiver library.

This test script can be invoked in multiple ways:

- By itself, without hardware (basic API tests only):

      ./test_packets.py

- With a live CRS board (includes streaming tests):

      ./test_packets.py --serial=0024

- As part of the complete test suite:

      ~/rfmux$ ./test.sh
"""

import pytest
import socket
import struct
import time
import numpy as np
from contextlib import closing

from rfmux.streamer import (
    ReadoutPacketReceiver,
    PFBPacketReceiver,
    get_multicast_socket,
    STREAMER_PORT,
    PFB_STREAMER_PORT,
    MULTICAST_GROUP,
    READOUT_PACKET_MAGIC,
    PFB_PACKET_MAGIC,
    LONG_PACKET_SIZE,
    SHORT_PACKET_SIZE,
)


def test_readout_receiver_creation():
    """Test ReadoutPacketReceiver instantiation"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as sock:
        receiver = ReadoutPacketReceiver(sock, reorder_window=256)
        assert receiver is not None
        assert receiver.sockfd > 0

        stats = receiver.get_stats()

        assert stats.total_packets_received == 0
        assert stats.total_bytes_received == 0
        assert stats.invalid_packets == 0
        assert stats.wrong_magic == 0


def test_receiver_get_queue():
    """Test that queues can be created for specific modules."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as sock:
        receiver = ReadoutPacketReceiver(sock)

        # Get queue for a specific serial/module combination
        queue = receiver.get_queue(serial=1234, module=1)
        assert queue is not None
        assert queue.empty()
        assert queue.max_size() == 1024  # Default queue size

        stats = queue.get_stats()
        assert stats.packets_received == 0
        assert stats.packets_dropped == 0
        assert stats.sequence_gaps == 0
        assert stats.last_seq == 0


def test_receiver_get_all_queues():
    """Test that we can retrieve all active queues."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as sock:
        receiver = ReadoutPacketReceiver(sock)

        # Initially no queues
        queues = receiver.get_all_queues()
        assert len(queues) == 0

        # Create some queues
        queue1 = receiver.get_queue(serial=1234, module=1)
        queue2 = receiver.get_queue(serial=1234, module=2)

        # Now we should see them
        queues = receiver.get_all_queues()
        assert len(queues) == 2


@pytest.mark.asyncio
async def test_receive_readout_packets(crs):
    """Test receiving actual readout packets from a CRS board."""
    # Get the hostname for multicast subscription
    hostname = crs.tuber_hostname

    # Create socket and receiver (small reorder window for testing)
    with closing(get_multicast_socket(hostname, STREAMER_PORT)) as sock:
        receiver = ReadoutPacketReceiver(sock, reorder_window=128)

        # Get the CRS serial number
        serial = int(crs.serial)

        # Get queue for module 1
        queue = receiver.get_queue(serial=serial, module=1)

        # Trigger some packets by doing a network analysis or similar
        # (In a real test, you'd want to ensure streaming is active)

        # Try to receive some packets (with timeout)
        timeout_ms = 5000  # 5 second timeout
        start_time = time.time()
        packets_received = 0

        while time.time() - start_time < timeout_ms / 1000.0:
            # Receive batch
            n = receiver.receive_batch(batch_size=256, timeout_ms=100)
            if n > 0:
                packets_received += n
                break

        if packets_received > 0:
            # Check receiver stats
            stats = receiver.get_stats()
            assert stats.total_packets_received > 0
            assert stats.total_bytes_received > 0

            # Try to pop a packet from the queue
            packet = queue.try_pop()
            if packet:
                typed_pkt = packet.to_python()
                assert typed_pkt.serial == serial
                assert typed_pkt.module == 1


@pytest.mark.asyncio
async def test_packet_sequence_ordering(crs):
    """Test that packets are properly reordered by sequence number."""
    hostname = crs.tuber_hostname
    serial = int(crs.serial)

    with closing(get_multicast_socket(hostname, STREAMER_PORT)) as sock:
        receiver = ReadoutPacketReceiver(sock, reorder_window=128)
        queue = receiver.get_queue(serial=serial, module=1)

        # Receive a batch of packets
        for _ in range(10):
            receiver.receive_batch(batch_size=256, timeout_ms=100)

        # Check that sequence numbers are monotonic (allowing for wraps)
        prev_seq = None
        while not queue.empty():
            packet = queue.try_pop()
            if packet:
                typed_pkt = packet.to_python()
                if prev_seq is not None:
                    # Check for monotonic increase (within reasonable bounds)
                    seq_diff = (typed_pkt.seq - prev_seq) & 0xFFFFFFFF
                    # Should be a small positive difference or wrapped
                    assert seq_diff < 0x80000000, "Sequence numbers not ordered"
                prev_seq = typed_pkt.seq


@pytest.mark.asyncio
async def test_multiple_module_queues(crs):
    """Test receiving packets from multiple modules simultaneously."""
    hostname = crs.tuber_hostname
    serial = int(crs.serial)

    with closing(get_multicast_socket(hostname, STREAMER_PORT)) as sock:
        receiver = ReadoutPacketReceiver(sock, reorder_window=128)

        total_received = 0
        for _ in range(1000):
            n = receiver.receive_batch(batch_size=256, timeout_ms=None)
            total_received += n

        # Get all active queues (created by incoming packets)
        all_queues = receiver.get_all_queues()
        for pkt_serial, module, queue in all_queues:
            q_stats = queue.get_stats()

        # We should have at least one active queue
        assert (
            len(all_queues) > 0
        ), f"No queues created. Total received: {total_received}"

        # Check that the serial number matches what we expect
        queue_serials = [pkt_serial for pkt_serial, _, _ in all_queues]
        assert serial in queue_serials, f"Expected serial {serial}, got {queue_serials}"


@pytest.mark.asyncio
async def test_queue_overflow_handling(crs):
    """Test that queue overflow is handled properly."""
    hostname = crs.tuber_hostname
    serial = int(crs.serial)

    with closing(get_multicast_socket(hostname, STREAMER_PORT)) as sock:
        receiver = ReadoutPacketReceiver(sock, reorder_window=128)
        queue = receiver.get_queue(serial=serial, module=1)

        # Receive many packets without popping
        for _ in range(100):
            receiver.receive_batch(batch_size=256, timeout_ms=10)

        # Check queue stats for drops if it filled up
        stats = queue.get_stats()
        if stats.packets_received > queue.max_size():
            # We should have dropped some packets
            assert stats.packets_dropped > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
