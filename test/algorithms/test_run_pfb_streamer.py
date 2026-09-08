"""py_run_pfb_streamer against a loopback PFB stream: ``time_run`` is
sample time from the packet timestamps, and stale timestamps end the
capture with an error rather than a capture that never ends."""
import asyncio
import contextlib
import socket
import threading
import time

import pytest

from rfmux import streamer
from rfmux.algorithms.measurement import py_run_pfb_streamer as runner
from rfmux.core.transferfunctions import PFB_SAMPLING_FREQ
from test.packet_helpers import pfb_datagram

PACKET_SAMPLES = 1000


class _Board:
    tuber_hostname = "127.0.0.1"

    async def set_pfb_streamer(self, channel, module):
        return None

    async def get_pfb_streamer(self, module):
        return [1]

    async def get_nco_frequency(self, module=1):
        return 0.0

    async def get_frequency(self, channel, module=1):
        return 100e6


def _pump(send, port, stop, recent):
    k = 0
    deadline = time.monotonic() + 8.0
    while not stop.is_set() and time.monotonic() < deadline:
        send.sendto(pfb_datagram(0, t_s=43200.0 + k * PACKET_SAMPLES / PFB_SAMPLING_FREQ,
                                 seq=k, recent=recent, values=(1000 + 0j,),
                                 num_samples=PACKET_SAMPLES), ("127.0.0.1", port))
        k += 1
        time.sleep(0.001)


@contextlib.contextmanager
def _stream(monkeypatch, recent):
    recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
    recv.bind(("127.0.0.1", 0))
    port = recv.getsockname()[1]
    send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    @contextlib.contextmanager
    def fake_socket(host, port=None, **kw):
        yield recv
    monkeypatch.setattr(streamer, "get_multicast_socket", fake_socket)
    monkeypatch.setattr(streamer, "STREAMER_TIMEOUT", 1.0)
    stop = threading.Event()
    pump = threading.Thread(target=_pump, args=(send, port, stop, recent))
    pump.start()
    try:
        yield
    finally:
        stop.set()
        pump.join()
        recv.close()
        send.close()


def _run(time_run):
    return asyncio.run(asyncio.wait_for(
        runner.py_run_pfb_streamer.__wrapped__(
            _Board(), channel=[1], module=1, time_run=time_run, nsegments=1),
        timeout=5.0))


def test_capture_ends_on_timestamped_sample_time(monkeypatch):
    time_run = 0.002
    with _stream(monkeypatch, recent=True):
        result = _run(time_run)
    wanted = time_run * PFB_SAMPLING_FREQ
    assert wanted <= len(result.i[0]) < wanted + PACKET_SAMPLES


def test_stale_timestamps_raise_and_name_the_timestamp_port(monkeypatch):
    """A board with no timestamp source stamps every packet stale, so
    sample time cannot be measured: the capture stops with an error
    that says what to set, as py_get_samples does."""
    monkeypatch.setattr(runner, "STALE_TIMESTAMP_GRACE_S", 0.2)
    with _stream(monkeypatch, recent=False):
        with pytest.raises(RuntimeError, match="set_timestamp_port"):
            _run(0.002)
