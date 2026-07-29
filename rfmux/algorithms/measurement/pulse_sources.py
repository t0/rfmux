"""
Async packet sources that feed a PulseCaptureSession from the streamers.

These are the headless counterparts of Periscope's slow-stream tap: an
asyncio receive loop per stream that parses packets and calls
:meth:`PulseCaptureSession.feed_sample`.  The Periscope pulse-capture
task and the reference demo scripts share these functions — the GUI
adds only Qt signal plumbing on top.

Usage (fast/PFB capture, headless)::

    await crs.configure_streamer(6, pfb_channels=[1, 2])
    session = PulseCaptureSession(channels=[1, 2], streamer_mode="fast",
                                  sample_rate=PFB_SAMPLE_RATE, ...)
    session.start()
    await run_pfb_source(session, crs.tuber_hostname, [1, 2],
                         duration_s=10.0)
    session.stop()
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, List, Optional

import numpy as np

from ... import streamer
from .streamer_config import PFB_SAMPLE_RATE
from .trigger_capture import _ts_to_seconds


def _flush(sock) -> None:
    """Discard datagrams buffered before the capture started."""
    while True:
        try:
            sock.recv(65536)
        except BlockingIOError:
            return


async def run_slow_source(
    session,
    host: str,
    module: int = 1,
    *,
    duration_s: Optional[float] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> float:
    """Feed *session* from the slow readout stream until stopped.

    Parameters
    ----------
    session : PulseCaptureSession
        Started session; its ``channels`` are extracted per packet.
    host : str
        CRS hostname (multicast interface selector).
    module : int
        1-indexed module to accept packets from.
    duration_s : float, optional
        Stop after this much *sample time* (from packet timestamps).
        None = run until ``should_stop`` returns True.
    should_stop : callable, optional
        Polled once per packet; return True to stop.

    Returns the sample time covered (seconds).
    """
    loop = asyncio.get_running_loop()
    channels = list(session.channels)
    prev_ts: Optional[float] = None
    elapsed = 0.0

    with streamer.get_multicast_socket(
            host, port=streamer.STREAMER_PORT) as sock:
        sock.setblocking(False)
        _flush(sock)
        while True:
            if should_stop is not None and should_stop():
                break
            try:
                data = await asyncio.wait_for(
                    loop.sock_recv(sock, streamer.LONG_PACKET_SIZE),
                    streamer.STREAMER_TIMEOUT)
            except asyncio.TimeoutError:
                break
            pkt = streamer.ReadoutPacket(data)
            if pkt.module != module - 1:
                continue
            raw = np.array(pkt) / 256.0
            ts = _ts_to_seconds(pkt.ts)
            for ch in channels:
                if len(pkt) <= ch - 1:
                    continue
                s = raw[ch - 1]
                session.feed_sample(ch, float(s.real), float(s.imag), ts)
            if ts is not None:
                # Monotone accumulation, clamped per packet: immune to
                # timestamp discontinuities (decimation changes, clock
                # wrap at the day boundary) that would blow up a plain
                # last-minus-first difference.
                if prev_ts is not None:
                    delta = ts - prev_ts
                    if 0.0 < delta < 5.0:
                        elapsed += delta
                prev_ts = ts
                if duration_s is not None and elapsed >= duration_s:
                    break
    return elapsed


async def run_pfb_source(
    session,
    host: str,
    channels: List[int],
    *,
    duration_s: Optional[float] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    sample_rate: float = PFB_SAMPLE_RATE,
) -> float:
    """Feed *session* from the fast/PFB stream until stopped.

    ``channels`` must match the order given to ``set_pfb_streamer`` —
    PFB packets interleave the streamed channels round-robin in that
    order.  Each sample gets a timestamp extrapolated from the packet
    timestamp at the PFB rate.

    Returns the sample time covered (seconds).
    """
    loop = asyncio.get_running_loop()
    n_groups = max(1, len(channels))
    elapsed = 0.0

    with streamer.get_multicast_socket(
            host, port=streamer.PFB_STREAMER_PORT) as sock:
        sock.setblocking(False)
        _flush(sock)
        while True:
            if should_stop is not None and should_stop():
                break
            try:
                data = await asyncio.wait_for(
                    loop.sock_recv(sock, streamer.PFB_PACKET_SIZE),
                    streamer.STREAMER_TIMEOUT)
            except asyncio.TimeoutError:
                break
            pkt = streamer.PFBPacket(data)
            # Match the slow stream's 16-bit ADC scale: np.array(pkt)
            # applies the packetizer /256; the second /256 brings the
            # 24-bit datapath down to ADC counts (same convention as
            # the slow readout path).  Without it, fast samples sit
            # exactly 256x above slow samples for the same signal.
            raw = np.array(pkt) / 256.0
            ts = _ts_to_seconds(pkt.ts)
            time_samples = pkt.num_samples // n_groups
            for si in range(time_samples):
                t = (ts + si / sample_rate) if ts is not None else None
                for slot, ch in enumerate(channels):
                    fi = si * n_groups + slot
                    if fi < len(raw):
                        v = raw[fi]
                        session.feed_sample(ch, float(v.real),
                                            float(v.imag), t)
            # Exact per-packet span — independent of timestamp
            # discontinuities across rate changes.
            elapsed += time_samples / sample_rate
            if duration_s is not None and elapsed >= duration_s:
                break
    return elapsed
