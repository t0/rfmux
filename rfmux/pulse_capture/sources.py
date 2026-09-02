"""
Async packet sources that feed a PulseCaptureSession from the streamers.

An asyncio receive loop per stream that parses packets and feeds a
session in blocks.  The Periscope pulse-capture task and the reference
demo scripts share this code — the GUI adds only Qt signal plumbing on
top.

Periscope drives :class:`SlowIngest` from its GUI tap rather than
calling :func:`run_slow_source`, because its process already holds
every slow packet and a second socket would cost kernel copies and
another drain thread in a GUI that is GIL-bound at stage 0.  Only the
transport differs: blocking, sample-time accounting and the duration
stop all live in :class:`SlowIngest`, so the GUI is a caller rather
than a second implementation.

Usage (fast/PFB capture, headless)::

    await crs.configure_streamer(6, pfb_channels=[1, 2])
    capture_session = PulseCaptureSession(channels=[1, 2], streamer_mode="fast",
                                  sample_rate=PFB_SAMPLING_FREQ, ...)
    capture_session.start()
    await run_pfb_source(capture_session, crs.tuber_hostname, [1, 2],
                         duration_s=10.0)
    capture_session.stop()

:func:`run_dual_source` drives both streams at once for a
:class:`~rfmux.pulse_capture.capture_session.DualPulseCaptureSession`,
which is what live cross-stream pulse matching needs.
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, List, Optional, Tuple

import numpy as np

from .. import streamer
from ..core.transferfunctions import PFB_SAMPLING_FREQ


def columns_for_width(channels, width: int):
    """``(channels, index array)`` for a packet carrying `width` channels.

    Channels past the packet width are dropped rather than skipped one
    at a time: a short packet carries 128, and column 200 of it does not
    exist however the channel is biased.
    """
    kept = tuple(c for c in channels if 0 < c <= width)
    return kept, np.fromiter((c - 1 for c in kept), dtype=np.intp,
                             count=len(kept))


class SlowIngest:
    """Turns slow-stream samples into a fed session.

    Everything a caller must do per packet: build blocks, keep sample
    time, and decide when a requested duration is covered.  It has no
    opinion about where the samples came from, which is the point --
    :func:`run_slow_source` drives it from a socket, and Periscope
    drives it from the receiver it already has.

    The transports differ deliberately.  Periscope's process already
    holds every packet, and a second receive path there costs kernel
    copies and a drain thread in a GUI that is GIL-bound at stage 0.
    The *interpretation* is what must not differ, so it lives here once.

    Blocks, rather than samples, because feeding sample-at-a-time costs
    ~2.3 us of interpreter time per sample (see
    :meth:`PulseCapture.process_sample`), which at 200 channels caps
    ingest near 1.8k packets/s -- decimation stage 5.  Going through
    blocks lifts that to ~65k packets/s, past stage 0.

    A slow packet carries one sample of *each* channel, so a block has
    to be built ACROSS packets -- unlike a PFB packet, which is already
    a block of one channel.

    ``feed`` is called as ``feed(channel, I, Q, timestamps)`` -- pass
    ``capture_session.feed_block``, or a stream-specific variant such as
    ``DualPulseCaptureSession.feed_slow_block``.
    """

    #: 256 packets is 6.7 ms at decimation stage 0 but 430 ms at stage
    #: 6, so a wall-clock cap keeps latency bounded at low rates.
    DEFAULT_MAX_PACKETS = 256
    DEFAULT_MAX_AGE_S = 0.05

    #: Timestamp steps outside this are treated as discontinuities
    #: rather than elapsed time.  See :meth:`advance`.
    MAX_PLAUSIBLE_STEP_S = 5.0

    def __init__(self, feed, *, duration_s: Optional[float] = None,
                 max_packets: int = DEFAULT_MAX_PACKETS,
                 max_age_s: float = DEFAULT_MAX_AGE_S):
        self._feed = feed
        self.duration_s = duration_s
        self.max_packets = max_packets
        self.max_age_s = max_age_s
        self.channels: Optional[Tuple[int, ...]] = None
        self.elapsed = 0.0
        self._values: List[np.ndarray] = []
        self._stamps: List[float] = []
        self._opened = 0.0
        self._prev_ts: Optional[float] = None

    # ── sample time ───────────────────────────────────────────────

    def advance(self, timestamp) -> None:
        """Accumulate sample time from a packet timestamp.

        Monotone and clamped per packet, so it is immune to the
        discontinuities a plain last-minus-first would turn into a
        nonsense duration: a decimation change restarts the clock, and
        the day boundary wraps it to zero.  Both are invisible in mock
        runs -- simulated captures are short and never cross midnight --
        which is exactly why this is a plain method with its own tests
        rather than something only a live socket can reach.

        Call this for packets that carry none of the wanted channels
        too: they are still time passing.  :meth:`add` calls it for you.
        """
        if timestamp is None:
            return
        if self._prev_ts is not None:
            delta = timestamp - self._prev_ts
            if 0.0 < delta < self.MAX_PLAUSIBLE_STEP_S:
                self.elapsed += delta
        self._prev_ts = timestamp

    @property
    def complete(self) -> bool:
        """True once ``duration_s`` of sample time has been covered."""
        return (self.duration_s is not None
                and self.elapsed >= self.duration_s)

    # ── blocking ──────────────────────────────────────────────────

    def add(self, channels, values, timestamp) -> None:
        """The whole per-packet duty: buffer, flush if ready, keep time.

        A change of channel set flushes what came before, so columns
        never straddle two different layouts.
        """
        self.advance(timestamp)
        channels = tuple(channels)
        if not channels:
            return
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
        if self.ready:
            self.flush()

    @property
    def ready(self) -> bool:
        return bool(self._values) and (
            len(self._values) >= self.max_packets
            or time.monotonic() - self._opened >= self.max_age_s)

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


#: Datagrams drained per event-loop wake.  One per wake ties a stream's
#: throughput to how often the loop schedules it, and two streams on one
#: loop with millisecond packet processing starve the slower one.  A
#: bounded batch per wake makes throughput a function of the data rate
#: instead; the caps bound how long one wake holds the loop.
_SLOW_DRAIN_CAP = 256
_PFB_DRAIN_CAP = 64
#: PFB packets processed between turns of the loop.  While the engine is
#: capturing, every sample of a packet walks through process_sample, so
#: a full drain would hold the loop for ~150 ms; eight packets bound it
#: to ~20 ms, and the extra no-op yields cost nothing.
_PFB_YIELD_EVERY = 8


def _drain(sock, first: bytes, size: int, cap: int) -> list:
    """*first* plus whatever else the socket already holds, up to *cap*."""
    batch = [first]
    while len(batch) < cap:
        try:
            batch.append(sock.recv(size))
        except BlockingIOError:
            break
    return batch


async def run_slow_source(
    capture_session,
    host: str,
    module: int = 1,
    *,
    duration_s: Optional[float] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> float:
    """Feed *capture_session* from the slow readout stream until stopped.

    Parameters
    ----------
    capture_session : PulseCaptureSession
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
    requested = list(capture_session.channels)
    ingest = SlowIngest(capture_session.feed_block, duration_s=duration_s)
    columns: Optional[Tuple[Tuple[int, ...], np.ndarray]] = None
    width = -1

    with streamer.get_multicast_socket(
            host, port=streamer.STREAMER_PORT) as sock:
        sock.setblocking(False)
        _flush(sock)
        done = False
        while not done:
            if should_stop is not None and should_stop():
                break
            try:
                data = await asyncio.wait_for(
                    loop.sock_recv(sock, streamer.LONG_PACKET_SIZE),
                    streamer.STREAMER_TIMEOUT)
            except asyncio.TimeoutError:
                break
            for data in _drain(sock, data, streamer.LONG_PACKET_SIZE,
                               _SLOW_DRAIN_CAP):
                pkt = streamer.ReadoutPacket(data)
                if pkt.module != module - 1:
                    continue
                raw = np.array(pkt) / 256.0
                ts = streamer.ts_to_seconds(pkt.ts)
                if len(pkt) != width:
                    # Recomputed only when the packet mode changes, not
                    # per packet -- that would undo the point of
                    # blocking.
                    width = len(pkt)
                    columns = columns_for_width(requested, width)
                if columns[0]:
                    ingest.add(columns[0], raw[columns[1]], ts)
                else:
                    # No wanted channel in this packet, but still time
                    # passing -- the duration must not stall on it.
                    ingest.advance(ts)
                if ingest.complete:
                    done = True
                    break
            # An explicit turn for whatever shares the loop -- in a dual
            # capture, the other stream.
            await asyncio.sleep(0)
        ingest.flush()   # don't strand the tail of the capture
    return ingest.elapsed


async def run_pfb_source(
    capture_session,
    host: str,
    channels: List[int],
    *,
    duration_s: Optional[float] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    sample_rate: float = PFB_SAMPLING_FREQ,
) -> float:
    """Feed *capture_session* from the fast/PFB stream until stopped.

    ``channels`` must match the order given to ``set_pfb_streamer`` —
    PFB packets interleave the streamed channels round-robin in that
    order.  Each sample gets a timestamp extrapolated from the packet
    timestamp at the PFB rate.

    Returns the sample time covered (seconds).

    Raises
    ------
    TimeoutError
        If no packet ever arrives: the fast streamer is off or on the
        wrong module, and returning 0.0 would end a dual capture as
        though it had been stopped.
    """
    loop = asyncio.get_running_loop()
    n_groups = max(1, len(channels))
    elapsed = 0.0
    got_any = False

    with streamer.get_multicast_socket(
            host, port=streamer.PFB_STREAMER_PORT) as sock:
        sock.setblocking(False)
        _flush(sock)
        done = False
        while not done:
            if should_stop is not None and should_stop():
                break
            try:
                data = await asyncio.wait_for(
                    loop.sock_recv(sock, streamer.PFB_PACKET_SIZE),
                    streamer.STREAMER_TIMEOUT)
            except asyncio.TimeoutError:
                if not got_any:
                    raise TimeoutError(
                        f"No PFB packets in {streamer.STREAMER_TIMEOUT} s — "
                        "the fast streamer is not sending. Check that "
                        "set_pfb_streamer enabled the right channels and "
                        "module (get_pfb_streamer says what the board "
                        "thinks), and that nothing tore it down.")
                break
            got_any = True
            for k, data in enumerate(_drain(sock, data,
                                            streamer.PFB_PACKET_SIZE,
                                            _PFB_DRAIN_CAP)):
                if k and k % _PFB_YIELD_EVERY == 0:
                    await asyncio.sleep(0)
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
                # channel is a strided slice; feeding those arrays
                # straight through is what lets the engine absorb quiet
                # stretches with numpy rather than 1.22 million Python
                # calls a second.  NaN where the packet has no usable
                # timestamp — the session drops and counts those,
                # exactly as the per-sample path did with None.
                t0 = ts if ts is not None else float("nan")
                times = t0 + np.arange(time_samples) / sample_rate
                for slot, ch in enumerate(channels):
                    v = raw[slot:time_samples * n_groups:n_groups]
                    if v.shape[0] == 0:
                        continue
                    n_ok = min(v.shape[0], times.shape[0])
                    # Every session and facade exposes feed_block, and
                    # there is deliberately no per-sample fallback: it
                    # would put the fast stream on the path this
                    # function exists to avoid.
                    capture_session.feed_block(ch, v[:n_ok].real,
                                               v[:n_ok].imag,
                                               times[:n_ok])
                # Exact per-packet span — independent of timestamp
                # discontinuities across rate changes.
                elapsed += time_samples / sample_rate
                if duration_s is not None and elapsed >= duration_s:
                    done = True
                    break
            # An explicit turn for whatever shares the loop -- in a dual
            # capture, the slow stream.  The drain cap above bounds how
            # long this coroutine held it.
            await asyncio.sleep(0)
    return elapsed


async def run_dual_source(
    capture_session,
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
            await run_dual_source(capture_session, host, [1, 2], duration_s=2.0)
        finally:
            await crs.configure_streamer(dec, modules=[1], pfb_channels=[])

    Parameters
    ----------
    capture_session : DualPulseCaptureSession
        Started session; fed through its ``slow_feed``/``fast_feed``
        facades so stream time advances the matcher.
    slow_source : callable, optional
        Override for the slow side, called with a stop predicate and
        awaited.  Periscope passes its tap pump here because its
        process already holds every slow packet: a second socket would
        cost kernel copies and another drain thread in a GUI that is
        GIL-bound at stage 0.  Both routes drive :class:`SlowIngest`,
        so only the transport differs.  Headless there is nothing to
        reuse and the default socket source is right.

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
                capture_session.slow_feed, host, module=module,
                duration_s=duration_s, should_stop=_stop)
        finally:
            finished = True

    async def _fast() -> float:
        nonlocal finished
        try:
            return await run_pfb_source(
                capture_session.fast_feed, host, channels,
                duration_s=duration_s, should_stop=_stop)
        finally:
            finished = True

    slow_elapsed, fast_elapsed = await asyncio.gather(_slow(), _fast())
    return slow_elapsed, fast_elapsed
