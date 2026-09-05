"""
What ``crs.trigger_capture`` hands back, without a board or a socket:
the fields of PulseCaptureResult that only the dual path fills in.
"""

import asyncio

import pytest

from rfmux.algorithms.measurement import trigger_capture as tc
from rfmux.core.transferfunctions import decimation_to_sampling
from rfmux.pulse_capture.capture_session import (
    DualPulseCaptureSession,
    PulseCaptureConfig,
)


def test_dual_result_reports_the_slow_time_offset(monkeypatch):
    """The slow pulses' Time arrays are shifted by the session; the
    result carries the shift so a caller can undo it against raw
    packet timestamps, as the file's reader can."""
    async def no_stream(*args, **kwargs):
        return 0.0, 0.0
    monkeypatch.setattr(tc, "run_dual_source", no_stream)

    slow_rate = decimation_to_sampling(6)
    config = PulseCaptureConfig()
    result = tc.PulseCaptureResult(
        streamer_mode="both", config=config, channels=[1], module=1)
    asyncio.run(tc._run_dual(result, "127.0.0.1", [1], 1, slow_rate,
                             0.0, None, None, False))

    expected = DualPulseCaptureSession(
        channels=[1], slow_rate=slow_rate, config=config).slow_time_offset_s
    assert result.slow_time_offset_s == expected
    assert result.slow_time_offset_s != 0.0


class _StreamerBoard:
    """A board whose PFB streamer carries *streamed*; nothing else answers."""
    tuber_hostname = "127.0.0.1"

    def __init__(self, streamed):
        self.streamed = streamed
        self.configured = []

    async def get_decimation(self):
        return 6

    async def get_pfb_streamer(self, module):
        return self.streamed

    async def set_pfb_streamer(self, channel, module):
        self.configured.append(channel)


def test_fast_capture_refuses_channels_the_streamer_does_not_carry():
    """The capture reads what the board streams and never configures it:
    a streamer carrying channel 2 cannot feed a capture of channel 1."""
    board = _StreamerBoard([2])
    with pytest.raises(ValueError, match="configure_streamer"):
        asyncio.run(tc.trigger_capture.__wrapped__(
            board, channel=[1], module=1, streamer_mode="fast", time_run=0.01))
    assert board.configured == []
