#!/usr/bin/env -S pytest -m veth_fastrxd -sv
"""Test fastrxd on a veth pair, fed version-0 channel-stream packets.

These tests are auto-skipped by normal pytest discovery, because they depend on
a particular environment that's configured outside python.

Invoke this via

    rfmux$ test/streamer/test_fastrxd_veth

...which wraps this Python script in a veth setup / teardown environment.
"""

import contextlib
import numpy as np
import os
import pytest
import re
import socket
import struct
import sys
import threading
import time
import types

from test.fastrx_bytes import PACKET_MAGIC, SPP, record

try:
    from rfmux import fastrx
except ImportError:
    fastrx = None

#: Bytes of UDP payload in a K=1 packet: the 86-byte header and one pipe's
#: block.  test.fastrx_bytes.record() pads to the on-disk stride; the wire
#: does not.
WIRE_BYTES = 86 + SPP * 2 * 2

#: Header byte offsets, from struct fastrx_packet_header (packet.h).
VERSION_OFF = 11
TAG_OFF = 12


# --------------------------------------------------------------------------
# Packets
# --------------------------------------------------------------------------

def pattern(seq: int) -> np.ndarray:
    """The (SPP, 2) int16 payload of packet *seq*: every channel distinct
    from its neighbours and from the other packets', Q distinct from I, and
    nothing zero, so a sample landing one slot off is caught."""
    ch = np.arange(SPP, dtype=np.int64)
    i = (seq * 31 + ch) % 30000 - 15000
    return np.stack([i, -i - 1], axis=1).astype(np.int16)


def wire_packet(seq: int, *, module: int = 1, version: int = 0, tag: int = 0,
                serial: int = 42, sample_trunc: int = 2, ts=None) -> bytes:
    """One K=1 packet's UDP payload, built independently of any writer."""
    rec = record(SPP, seq, module=module, serial=serial,
                 sample_trunc=sample_trunc, ts=ts, iq={1: pattern(seq)})
    pkt = bytearray(rec[:WIRE_BYTES])
    pkt[VERSION_OFF] = version
    struct.pack_into("<H", pkt, TAG_OFF, tag)
    return bytes(pkt)


# --------------------------------------------------------------------------
# The runtime, as test_fastrxd_veth left it
# --------------------------------------------------------------------------

@pytest.fixture(scope="class")
def runtime(request):
    """Runtime configuration, plus setup/teardown verification"""

    tx = os.environ.get("FASTRX_TEST_TX", "fxveth1")
    socket_path = os.environ.get("FASTRX_TEST_SOCKET", "/run/fastrxd/fxveth0")
    log_path = os.environ.get("FASTRX_TEST_LOG", socket_path + ".log")

    # Skip unless pytest explicitly invoked with "-m veth_fastrxd"
    expr = request.config.getoption("markexpr", default="") or ""
    if not re.search(r"(?<!not )veth_fastrxd\b", expr):
        pytest.skip("veth_fastrxd marker absent - skipping test")

    def answering() -> bool:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.connect(socket_path)
            return True
        except OSError:
            return False
        finally:
            s.close()

    def tail(n: int = 40) -> str:
        try:
            with open(log_path) as f:
                return "".join(f.readlines()[-n:])
        except OSError as e:
            return f"(no daemon log: {e})"

    if sys.platform != "linux":
        pytest.fail("fastrxd is Linux-only")
    if fastrx is None:
        pytest.fail("this rfmux build does not include fastrx")
    if not os.path.exists(f"/sys/class/net/{tx}"):
        pytest.fail(f"no interface {tx}")
    if not os.path.exists(socket_path):
        pytest.fail(f"no fastrxd socket at {socket_path}")
    if not os.access(socket_path, os.R_OK | os.W_OK):
        pytest.fail(f"{socket_path} is not ours: run test_fastrxd_veth as the user "
                    "who runs pytest")
    if not answering():
        pytest.fail(f"nothing answers on {socket_path}; is fastrxd running?")
    with open(f"/sys/class/net/{tx}/ifindex") as f:
        tx_ifindex = int(f.read())

    yield types.SimpleNamespace(socket_path=socket_path, tx=tx, tx_ifindex=tx_ifindex, tail=tail)

    if not answering():
        pytest.fail("fastrxd stopped answering during this class:\n" + tail(),
                    pytrace=False)


# --------------------------------------------------------------------------
# The transmitter
# --------------------------------------------------------------------------

@pytest.fixture
def streaming(runtime):
    """A transmitter on the rig's transmit end: ``with streaming(make_packet):``
    sends make_packet(seq) for consecutive seq in a background thread until
    the block exits.

    The socket sends to the multicast group the way the board does: checksum
    0, TTL 1, one named interface.  It is self-clocking: packets cross the
    veth pair through a 256-entry ring drained in softirq, and veth counts
    a packet the full ring refused on the sender's tx_dropped, synchronously
    with the sendto (the device is noqueue, so nothing is deferred).  The
    thread watches that counter and retransmits a refused packet after
    yielding to the softirq, so it runs at exactly the rate the receive side
    drains -- the ring held full, the daemon fed the burstiest traffic the
    rig can make -- and no packet is lost before the daemon ever sees it."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # IP_MULTICAST_IF by interface index (struct ip_mreqn), so the
    # interface's address need not be known here.
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                    struct.pack("4s4si", b"\0" * 4, b"\0" * 4, runtime.tx_ifindex))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
    # Fidelity only: nothing on the receive side checks the UDP checksum,
    # but the board sends 0, and so does this.
    sock.setsockopt(socket.SOL_SOCKET, 11, 1)  # SO_NO_CHECK

    # Opened once; sysfs regenerates the value on every read from offset 0.
    tx_dropped = open(f"/sys/class/net/{runtime.tx}/statistics/tx_dropped")

    def dropped() -> int:
        tx_dropped.seek(0)
        return int(tx_dropped.read())

    @contextlib.contextmanager
    def stream(make_packet, *, first_seq: int = 1000):
        stop = threading.Event()
        dst = (fastrx.MULTICAST_GROUP, fastrx.PORT)

        def run():
            seq = first_seq
            while not stop.is_set():
                pkt = make_packet(seq)
                while True:
                    before = dropped()
                    sock.sendto(pkt, dst)
                    if dropped() == before:
                        break
                    # The ring was full: a syscall does not yield, so let
                    # ksoftirqd drain it now rather than at the next tick.
                    os.sched_yield()
                seq += 1

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(5)

    try:
        yield stream
    finally:
        tx_dropped.close()
        sock.close()


def capture(runtime, n, *, module=1, timeout=10.0):
    with fastrx.PacketCapture(socket=runtime.socket_path) as c:
        d = c.capture(n, SPP, module, timeout)
        # Frame accounting between daemon and client: a frame released
        # twice, or one left referenced, is a bug whichever test it
        # happens in.
        assert c.double_releases == 0, runtime.tail()
        assert c.stranded_frames == 0, runtime.tail()
        return d


@pytest.mark.veth_fastrxd
class TestVersion0OnePipeline:
    """Version-0 packets carrying one pipeline (K=1, 128 channels) through
    the real daemon on the veth runtime: what gets through exactly as sent, and
    what is kept out, and where."""

    def test_samples_arrive_in_order(self, runtime, streaming):
        n = 64
        with streaming(wire_packet):
            d = capture(runtime, n)

        assert d["complete"], runtime.tail()
        assert d["restarts"] == 0
        assert d["dropouts"] == 0
        assert d["pipe_snapshot"] == 0b1
        assert d["modules_seen"] == 0b1
        assert d["module"] == 1
        assert d["i"].shape == d["q"].shape == (n, SPP)

        seq = d["seq"]
        assert np.array_equal(seq, np.arange(seq[0], seq[0] + n, dtype=np.uint32))
        for row, s in enumerate(seq):
            want = pattern(int(s))
            assert np.array_equal(d["i"][row], want[:, 0]), f"I of seq {s}"
            assert np.array_equal(d["q"][row], want[:, 1]), f"Q of seq {s}"

    def test_recording_round_trip(self, runtime, streaming, tmp_path):
        """PacketWriter keeps the wire header verbatim and the payload in
        channel order; both readers of the file agree with what was sent."""
        from rfmux.pulse_capture.recording_file import RecordingFile

        n = 32
        ts = (2026, 289, 14, 15, 16, 777, 0, 51316)
        path = tmp_path / "k1.fastrx"
        with streaming(lambda seq: wire_packet(seq, tag=0x1234, serial=99,
                                                      sample_trunc=0, ts=ts)):
            with fastrx.PacketWriter(path, channels=SPP, n_packets=n,
                                     socket=runtime.socket_path) as w:
                assert w.wait(10), runtime.tail()
            assert w.overruns == 0
            assert w.dropouts == 0

        for Reader in (fastrx.PacketFile, RecordingFile):
            with Reader(os.fspath(path)) as f:
                assert len(f) == n
                assert f.channels == SPP

                h = f.headers()
                assert np.all(h["magic"] == PACKET_MAGIC)
                assert np.all(h["version"] == fastrx.PACKET_VERSION)
                assert np.all(h["pipe_snapshot"] == 0b1)
                assert np.all(h["sample_trunc"] == 0)
                assert np.all(h["module"] == 0)           # 0-indexed on the wire
                assert np.all(h["tag"] == 0x1234)
                assert np.all(h["serial"] == 99)
                assert np.all(h["samples_per_packet"] == SPP)

                seq = f.seq()
                assert np.array_equal(seq, np.arange(seq[0], seq[0] + n, dtype=np.uint32))

                t = f.ts()
                for name, value in zip(("y", "d", "h", "m", "s", "ss", "c", "sbs"), ts):
                    assert np.all(t[name] == value), name

                iq = f.iq()
                assert iq.shape == (n, SPP, 2)
                for row, s in enumerate(seq):
                    assert np.array_equal(iq[row], pattern(int(s))), f"seq {s}"

    def test_other_modules_reach_the_filter_but_not_the_capture(self, runtime, streaming):
        """A valid packet from another module is dispatched (the client
        counts it in modules_seen) and then dropped by the client's module
        filter."""
        with streaming(lambda seq: wire_packet(seq, module=2)):
            d = capture(runtime, 8, module=1, timeout=2.0)
        assert not d["complete"]
        assert len(d["seq"]) == 0
        assert d["modules_seen"] == 0b10

    def test_unknown_version_never_reaches_a_client(self, runtime, streaming):
        """fastrxd validates the version byte: anything but 0 is not ours,
        and is returned to the NIC without a client hearing of it.  Relaxing
        this (to accept a later wire version) is a deliberate change to this
        test."""
        with streaming(lambda seq: wire_packet(seq, version=1)):
            d = capture(runtime, 8, module=1, timeout=2.0)
        assert not d["complete"]
        assert len(d["seq"]) == 0
        assert d["modules_seen"] == 0

    def test_short_payload_never_reaches_a_client(self, runtime, streaming):
        """A header promising 128 samples over a payload that holds fewer is
        dropped at validation, not handed on as a short block."""
        with streaming(lambda seq: wire_packet(seq)[:86 + 100 * 4]):
            d = capture(runtime, 8, module=1, timeout=2.0)
        assert not d["complete"]
        assert len(d["seq"]) == 0
        assert d["modules_seen"] == 0

    def test_the_stream_resumes_after_junk(self, runtime, streaming):
        """Rejected traffic does not wedge the daemon: a valid stream started
        after it captures normally."""
        with streaming(lambda seq: wire_packet(seq, version=7)):
            time.sleep(0.2)
        with streaming(wire_packet, first_seq=5000):
            d = capture(runtime, 16)
        assert d["complete"], runtime.tail()
        assert d["restarts"] == 0
        assert d["seq"][0] >= 5000
