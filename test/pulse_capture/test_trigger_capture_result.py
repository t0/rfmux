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
