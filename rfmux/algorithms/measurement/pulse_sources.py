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

:func:`run_dual_source` drives both streams at once for a
:class:`~rfmux.algorithms.measurement.pulse_capture_dual.DualPulseCaptureSession`,
which is what live cross-stream pulse matching needs.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, List, Optional, Tuple

import numpy as np

from ... import streamer
from .streamer_config import PFB_SAMPLE_RATE


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
            ts = streamer.ts_to_seconds(pkt.ts)
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
            ts = streamer.ts_to_seconds(pkt.ts)
            time_samples = pkt.num_samples // n_groups
            # A whole packet per channel per call.  The packet
            # interleaves the streamed channels round-robin, so one
            # channel is a strided slice; feeding those arrays straight
            # through is what lets the detector absorb quiet stretches
            # with numpy rather than 1.22 million Python calls a second.
            # NaN where the packet has no usable timestamp — the session
            # drops and counts those, exactly as the per-sample path did
            # with None.
            t0 = ts if ts is not None else float("nan")
            times = t0 + np.arange(time_samples) / sample_rate
            for slot, ch in enumerate(channels):
                v = raw[slot:time_samples * n_groups:n_groups]
                if v.shape[0] == 0:
                    continue
                n_ok = min(v.shape[0], times.shape[0])
                feed = getattr(session, "feed_block", None)
                if feed is None:
                    for si in range(n_ok):
                        session.feed_sample(ch, float(v[si].real),
                                            float(v[si].imag),
                                            None if ts is None
                                            else float(times[si]))
                else:
                    feed(ch, v[:n_ok].real, v[:n_ok].imag, times[:n_ok])
            # Exact per-packet span — independent of timestamp
            # discontinuities across rate changes.
            elapsed += time_samples / sample_rate
            if duration_s is not None and elapsed >= duration_s:
                break
    return elapsed


async def run_dual_source(
    session,
    host: str,
    channels: List[int],
    *,
    module: int = 1,
    duration_s: Optional[float] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    slow_source: Optional[Callable[[Callable[[], bool]],
                                   Awaitable[float]]] = None,
) -> Tuple[float, float]:
    """Feed both streams of a :class:`DualPulseCaptureSession` at once.

    The two streams live on different ports (9876 / 9877), so they run
    as concurrent tasks against their own sockets.  Whichever side
    finishes first stops the other: the pair is only useful while both
    are live, and a matcher fed by one stream alone just accumulates
    one-sided pulses.

    The caller enables and tears down the PFB streamer, exactly as for
    :func:`run_pfb_source` — do it in a ``try``/``finally`` so a failed
    capture still leaves the fast streamer off::

        await crs.configure_streamer(dec, modules=[1],
                                     pfb_channels=[1, 2], pfb_module=1)
        try:
            await run_dual_source(session, host, [1, 2], duration_s=2.0)
        finally:
            await crs.configure_streamer(dec, modules=[1], pfb_channels=[])

    Parameters
    ----------
    session : DualPulseCaptureSession
        Started session; fed through its ``slow_feed``/``fast_feed``
        facades so stream time advances the matcher.
    slow_source : callable, optional
        Override for the slow side, called with a stop predicate and
        awaited.  Periscope passes its tap pump here: the mock streamer
        sends unicast, and with ``SO_REUSEPORT`` the kernel hands each
        datagram to exactly one socket, so a second listener alongside
        Periscope's own receiver would silently starve.  Headless there
        is no competing receiver and the default socket source is right.

    Returns ``(slow_elapsed, fast_elapsed)`` in seconds of sample time.
    """
    finished = False

    def _stop() -> bool:
        return finished or (should_stop is not None and should_stop())

    async def _slow() -> float:
        nonlocal finished
        try:
            if slow_source is not None:
                return await slow_source(_stop)
            return await run_slow_source(
                session.slow_feed, host, module=module,
                duration_s=duration_s, should_stop=_stop)
        finally:
            finished = True

    async def _fast() -> float:
        nonlocal finished
        try:
            return await run_pfb_source(
                session.fast_feed, host, channels,
                duration_s=duration_s, should_stop=_stop)
        finally:
            finished = True

    slow_elapsed, fast_elapsed = await asyncio.gather(_slow(), _fast())
    return slow_elapsed, fast_elapsed
