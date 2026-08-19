"""
Async packet sources that feed a PulseCaptureSession from the streamers.

An asyncio receive loop per stream that parses packets and feeds a
session in blocks.  The Periscope pulse-capture task and the reference
demo scripts share this code — the GUI adds only Qt signal plumbing on
top.

Periscope cannot use :func:`run_slow_source` itself (its own receiver
already owns the socket, and with ``SO_REUSEPORT`` a second listener
would starve), so it feeds :class:`SlowBlockAccumulator` from its GUI
tap instead.  That class is the shared part: both routes turn packets
into per-channel blocks identically, and neither can quietly become
slower than the other.

Usage (fast/PFB capture, headless)::

    await crs.configure_streamer(6, pfb_channels=[1, 2])
    session = PulseCaptureSession(channels=[1, 2], streamer_mode="fast",
                                  sample_rate=streamer.PFB_SAMPLE_RATE, ...)
    session.start()
    await run_pfb_source(session, crs.tuber_hostname, [1, 2],
                         duration_s=10.0)
    session.stop()

:func:`run_dual_source` drives both streams at once for a
:class:`~rfmux.pulse_capture.session.DualPulseCaptureSession`,
which is what live cross-stream pulse matching needs.
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, List, Optional, Tuple

import numpy as np

from .. import streamer


def columns_for_width(channels, width: int):
    """``(channels, index array)`` for a packet carrying `width` channels.

    Channels past the packet width are dropped rather than skipped one
    at a time: a short packet carries 128, and column 200 of it does not
    exist however the channel is biased.
    """
    kept = tuple(c for c in channels if 0 < c <= width)
    return kept, np.fromiter((c - 1 for c in kept), dtype=np.intp,
                             count=len(kept))


class SlowBlockAccumulator:
    """Turns slow-stream packets into per-channel blocks.

    A slow packet carries one sample of *each* channel, so a block has
    to be built ACROSS packets -- unlike a PFB packet, which is already
    a block of one channel.  That difference is the whole reason this
    class exists.

    Feeding sample-at-a-time costs ~2.3 us of interpreter time per
    sample (see :meth:`PulseCapture.process_sample`), which at 200
    channels caps ingest near 1.8k packets/s -- decimation stage 5.
    Going through blocks lifts that to ~65k packets/s, past stage 0.

    Both the headless socket source and Periscope's GUI tap drive this,
    so the two cannot drift apart: the GUI is a caller, not a second
    implementation.

    ``feed`` is called as ``feed(channel, I, Q, timestamps)`` -- pass
    ``session.feed_block``, or a stream-specific variant such as
    ``DualPulseCaptureSession.feed_slow_block``.
    """

    #: 256 packets is 6.7 ms at decimation stage 0 but 430 ms at stage
    #: 6, so a wall-clock cap keeps latency bounded at low rates.
    DEFAULT_MAX_PACKETS = 256
    DEFAULT_MAX_AGE_S = 0.05

    def __init__(self, feed, *, max_packets: int = DEFAULT_MAX_PACKETS,
                 max_age_s: float = DEFAULT_MAX_AGE_S):
        self._feed = feed
        self.max_packets = max_packets
        self.max_age_s = max_age_s
        self.channels: Optional[Tuple[int, ...]] = None
        self._values: List[np.ndarray] = []
        self._stamps: List[float] = []
        self._opened = 0.0

    def add(self, channels, values, timestamp) -> None:
        """Buffer one packet's worth of samples.

        A change of channel set flushes what came before, so columns
        never straddle two different layouts.
        """
        channels = tuple(channels)
        if channels != self.channels:
            self.flush()
            self.channels = channels
        if not self._values:
            self._opened = time.monotonic()
        self._values.append(values)
        # No usable timestamp becomes NaN; feed_block drops and counts
        # those exactly as feed_sample did.
        self._stamps.append(float("nan") if timestamp is None
                            else float(timestamp))

    @property
    def ready(self) -> bool:
        return bool(self._values) and (
            len(self._values) >= self.max_packets
            or time.monotonic() - self._opened >= self.max_age_s)

    def add_and_flush_if_ready(self, channels, values, timestamp) -> None:
        """The whole per-packet duty of a caller, in one call."""
        self.add(channels, values, timestamp)
        if self.ready:
            self.flush()

    def flush(self) -> None:
        """Hand everything buffered to the session, one block per channel."""
        if not self._values:
            return
        values = np.stack(self._values)          # (packets, channels)
        stamps = np.asarray(self._stamps, dtype=np.float64)
        self._values = []
        self._stamps = []
        for column, channel in enumerate(self.channels):
            samples = values[:, column]
            self._feed(channel, samples.real, samples.imag, stamps)


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
    requested = list(session.channels)
    blocks = SlowBlockAccumulator(session.feed_block)
    columns: Optional[Tuple[Tuple[int, ...], np.ndarray]] = None
    width = -1
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
            if len(pkt) != width:
                # Recomputed only when the packet mode changes, not per
                # packet -- that would undo the point of blocking.
                width = len(pkt)
                columns = columns_for_width(requested, width)
            if columns[0]:
                blocks.add_and_flush_if_ready(columns[0], raw[columns[1]],
                                              ts)
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
        blocks.flush()   # don't strand the tail of the capture
    return elapsed


async def run_pfb_source(
    session,
    host: str,
    channels: List[int],
    *,
    duration_s: Optional[float] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    sample_rate: float = streamer.PFB_SAMPLE_RATE,
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
                # Every session and facade exposes feed_block.  This
                # used to fall back to a per-sample loop when it did
                # not, which is how both-mode quietly ran the fast
                # stream on the path this function exists to avoid.
                session.feed_block(ch, v[:n_ok].real, v[:n_ok].imag,
                                   times[:n_ok])
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
