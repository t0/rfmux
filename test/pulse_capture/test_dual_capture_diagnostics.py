"""A dual capture that cannot work says so, instead of counting quietly.

Both diagnostics come from the same live debugging session against real
hardware: a both-mode capture with the PFB streamer off dies silently at
the socket timeout, and one with it on can starve the slow stream so
badly (0.8 s of sample time over 300 s of wall time, measured) that
noise training never finishes and the trigger freeze never lifts.
Either way the user watches counters that can never produce a pair.
"""

import asyncio

import numpy as np
import pytest

from rfmux import streamer
from rfmux.pulse_capture.sources import run_pfb_source


class _NeverFed:
    channels = [1]

    def feed_block(self, ch, i_vals, q_vals, timestamps):
        raise AssertionError("a silent socket must feed nothing")


def test_silent_pfb_socket_is_an_error(monkeypatch):
    """Zero packets ever is a configuration problem, not an empty capture.

    Returning 0.0 made run_dual_source treat it as one side finishing,
    which stopped the other side too: the capture ended as though the
    user had pressed stop, with nothing naming the silent socket.
    """
    monkeypatch.setattr(streamer, "STREAMER_TIMEOUT", 0.2)
    with pytest.raises(TimeoutError, match="fast streamer is not sending"):
        asyncio.run(run_pfb_source(_NeverFed(), "127.0.0.1", [1]))


def test_a_stream_that_stops_mid_capture_still_returns(monkeypatch):
    """The error is for silence from the START; a stream that dies later
    keeps today's behaviour -- return what was covered."""
    monkeypatch.setattr(streamer, "STREAMER_TIMEOUT", 0.2)

    sent = {"n": 0}
    real_wait_for = asyncio.wait_for

    async def one_packet_then_silence(coro, timeout):
        if sent["n"] == 0:
            sent["n"] += 1
            coro.close()
            pkt = streamer.PFBPacket()
            pkt.magic = streamer.PFB_PACKET_MAGIC
            pkt.num_samples = 100
            return bytes(pkt)
        return await real_wait_for(coro, timeout)

    class _Sink:
        channels = [1]
        fed = 0

        def feed_block(self, ch, i_vals, q_vals, timestamps):
            _Sink.fed += 1

    monkeypatch.setattr(asyncio, "wait_for", one_packet_then_silence)
    covered = asyncio.run(run_pfb_source(_Sink(), "127.0.0.1", [1]))
    assert covered > 0.0
    assert _Sink.fed == 1


class _Signals:
    def __init__(self, sink):
        class _E:
            def emit(_self, msg):
                sink.append(msg)
        self.error = _E()


def _watchdog_task(states):
    """A PulseCaptureTask stripped to what the watchdog touches."""
    from types import SimpleNamespace

    from rfmux.tools.periscope.pulse_capture_task import PulseCaptureTask

    t = PulseCaptureTask.__new__(PulseCaptureTask)
    errors = []
    t.signals = _Signals(errors)
    t.session = SimpleNamespace(state=states)
    return t, errors


def test_watchdog_names_the_stream_still_training(monkeypatch):
    async def fast_sleep(_s):
        pass
    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    t, errors = _watchdog_task({"slow": "estimating", "fast": "capturing"})
    ticks = {"n": 0}

    def stop():
        ticks["n"] += 1
        return ticks["n"] > 2

    asyncio.run(t._dual_watchdog(stop))
    assert errors, "a stuck stream produced no warning"
    assert "slow stream" in errors[0]
    assert "no stream may trigger" in errors[0]


def test_watchdog_is_silent_once_both_capture(monkeypatch):
    async def fast_sleep(_s):
        pass
    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    t, errors = _watchdog_task({"slow": "capturing", "fast": "capturing"})
    asyncio.run(t._dual_watchdog(lambda: False))  # returns on its own
    assert errors == []
