"""
Async packet sources that feed a PulseCaptureSession from the streamers.

An asyncio receive loop per stream that parses packets and feeds a
session in blocks.  The Periscope pulse-capture task and the reference
demo scripts share this code — the GUI adds only Qt signal plumbing on
top.

Periscope drives :class:`SlowIngest` from its GUI tap rather than
calling :func:`run_slow_source` (see :class:`SlowIngest` for why).
Only the transport differs: blocking, sample-time accounting and the
duration stop all live in :class:`SlowIngest`, so the GUI is a caller
rather than a second implementation.

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
import socket
import struct
import sys
import threading
import time
from typing import Awaitable, Callable, List, Optional, Tuple

import numpy as np

from numba import njit

from .. import streamer
from ..core.transferfunctions import PFB_SAMPLING_FREQ


@njit(cache=True)
def _advance_block(stamps, prev, elapsed, max_step):
    """SlowIngest.advance over a block of stamps, packet by packet:
    (prev, elapsed) after the last.  NaN stamps are no timestamp."""
    for t in stamps:
        if t != t:
            continue
        if prev == prev:
            delta = t - prev
            if 0.0 < delta < max_step:
                elapsed += delta
            elif -max_step < delta <= 0.0:
                continue
        prev = t
    return prev, elapsed


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
    #: Packets kept back at a size- or age-triggered flush.  Datagrams
    #: arrive slightly out of order at high rates; a block is fed in
    #: timestamp order, and the newest few wait for any straggler that
    #: belongs before them.  An explicit flush releases them.
    HOLD_BACK = 4

    def __init__(self, feed, *, duration_s: Optional[float] = None,
                 max_packets: int = DEFAULT_MAX_PACKETS,
                 max_age_s: float = DEFAULT_MAX_AGE_S):
        self._feed = feed
        self.duration_s = duration_s
        self.max_packets = max_packets
        self.max_age_s = max_age_s
        self.channels: Optional[Tuple[int, ...]] = None
        self.elapsed = 0.0
        self._values: List[np.ndarray] = []     # rows, or (rows, channels) blocks
        self._stamps: List[float] = []
        self._rows = 0
        self._opened = 0.0
        self._prev_ts: Optional[float] = None

    # ── sample time ───────────────────────────────────────────────

    def advance(self, timestamp) -> None:
        """Accumulate sample time from a packet timestamp.

        Monotone and clamped per packet, so it is immune to the
        discontinuities a plain last-minus-first would turn into a
        nonsense duration: a decimation change restarts the clock, and
        the day boundary wraps it to zero.  Call this for packets that
        carry none of the wanted channels too: they are still time
        passing.  :meth:`add` calls it for you.
        """
        self._advance(np.array([np.nan if timestamp is None
                                else float(timestamp)]))

    def _advance(self, stamps: np.ndarray) -> None:
        prev = float("nan") if self._prev_ts is None else self._prev_ts
        prev, self.elapsed = _advance_block(
            stamps, prev, self.elapsed, self.MAX_PLAUSIBLE_STEP_S)
        if prev == prev:
            self._prev_ts = prev

    @property
    def complete(self) -> bool:
        """True once ``duration_s`` of sample time has been covered."""
        return (self.duration_s is not None
                and self.elapsed >= self.duration_s)

    # ── blocking ──────────────────────────────────────────────────

    def add(self, channels, values, timestamp) -> None:
        """The whole per-packet duty: buffer, flush if ready, keep time.
        A change of channel set flushes what came before, so columns
        never straddle two different layouts.  No usable timestamp
        becomes NaN; feed_block drops and counts those."""
        self.add_block(channels, np.asarray(values)[None, :],
                       [np.nan if timestamp is None else timestamp])

    def add_block(self, channels, values, stamps) -> None:
        """:meth:`add` for a block: *values* (packets, channels), one
        stamp per row, NaN where a packet had no usable timestamp."""
        values = np.asarray(values)
        stamps = np.asarray(stamps, dtype=np.float64)
        if stamps.size:
            self._advance(stamps)
        channels = tuple(channels)
        if not channels or values.shape[0] == 0:
            return
        if channels != self.channels:
            self.flush()
            self.channels = channels
        if not self._values:
            self._opened = time.monotonic()
        self._values.append(values)
        self._stamps.extend(stamps.tolist())
        self._rows += values.shape[0]
        if self.ready:
            self.flush(final=False)

    @property
    def ready(self) -> bool:
        return bool(self._values) and (
            self._rows >= self.max_packets
            or time.monotonic() - self._opened >= self.max_age_s)

    def flush(self, final: bool = True) -> None:
        """Hand what is buffered to the session in time order, one block
        per channel.  Unless *final*, the newest HOLD_BACK packets stay
        buffered for stragglers."""
        if not self._values:
            return
        order = np.argsort(self._stamps, kind="stable")   # NaN sorts last
        keep = 0 if final else min(self.HOLD_BACK, len(order) - 1)
        if keep:
            order, held = order[:-keep], order[-keep:]
        rows = np.concatenate([np.atleast_2d(v) for v in self._values])
        values = rows[order]                              # (packets, channels)
        stamps = np.asarray(self._stamps, dtype=np.float64)[order]
        if keep:
            self._values = [rows[i] for i in held]
            self._stamps = [self._stamps[i] for i in held]
            self._rows = int(keep)
            self._opened = time.monotonic()
        else:
            self._values = []
            self._stamps = []
            self._rows = 0
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
#: instead; the cap bounds how long one wake holds the loop.
_SLOW_DRAIN_CAP = 256
#: The most the fast stream is let fall behind the slow one, in stream
#: time, where the session measures the two clocks.  Past this the
#: source discards packets by their stamps until the fast clock is
#: within half the bound, and counts them: a bounded lag with the loss
#: in the open, rather than a lag that grows to the socket buffer and
#: leaves every pair without its fast window.  The kernel's own account
#: of the queue (SO_MEMINFO) is shown but not acted on: UDP releases
#: receive memory in steps of a quarter of the buffer, so it reads
#: high by up to that much.
_PFB_MAX_LAG_S = 0.25
#: The C++ PFB receiver: packets held to put arrivals in order (6.5 ms
#: at four channels; the stream arrives a few packets out of order),
#: how many more before the excess is released, the queue's bound
#: (its oldest go first when it fills, though the lag bound acts long
#: before), and packets per pop, which is the block the engine gets.
_PFB_REORDER_WINDOW = 64
_PFB_FLUSH_EVERY = 16
_PFB_QUEUE_MAX = 8192
_PFB_POP_MAX = 256
#: Bytes per second of a four-channel PFB stream, for the socket's
#: receive buffer in seconds: 1000 samples per packet, so four
#: channels at 2.44 MHz are 78 MB/s.  The socket asks for the largest
#: buffer the host allows; the kernel charges memory only for packets
#: actually queued.
_PFB_BYTES_PER_SECOND = (4 * PFB_SAMPLING_FREQ / streamer.PFBPACKET_NSAMP_MAX
                         * streamer.PFB_PACKET_SIZE)
#: How long the PFB receive thread waits on a quiet socket before it
#: looks at the stop flag again.
_PFB_RECEIVE_TIMEOUT_S = 0.05


def _rcvbuf_request() -> Optional[int]:
    """The largest receive buffer this host allows, or None where that
    is not readable and the socket helper's own ladder applies."""
    try:
        with open("/proc/sys/net/core/rmem_max") as f:
            return int(f.read())
    except (OSError, ValueError):
        return None


def _rcvq_bytes(sock) -> Optional[int]:
    """Bytes queued in the socket's receive buffer, or None where the
    kernel does not say (SO_MEMINFO is Linux)."""
    try:
        raw = sock.getsockopt(socket.SOL_SOCKET, 55, 36)   # SO_MEMINFO
        return int.from_bytes(raw[:4], sys.byteorder)      # rmem_alloc
    except (OSError, AttributeError, TypeError, ValueError):
        return None


def _pfb_buffer_seconds(sock) -> float:
    held = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    if sys.platform == "linux":
        held //= 2                     # Linux reports double the request
    return held / _PFB_BYTES_PER_SECOND


def _set_receive_timeout(sock, seconds: float) -> None:
    """Bound every receive on a blocking *sock* to *seconds*.

    SO_RCVTIMEO, not ``settimeout``: the latter makes the fd
    non-blocking, on which recvmmsg returns EAGAIN at once (its own
    timeout is consulted only between datagrams), so a quiet stream
    would spin the receive thread.  The non-Linux receiver waits in
    select() with its own timeout, so ``settimeout`` serves there.
    """
    if sys.platform != "linux":
        sock.settimeout(seconds)
        return
    sock.setblocking(True)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO,
                    struct.pack("ll", int(seconds),
                                int(seconds % 1 * 1_000_000)))


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
        Polled once per receive batch (up to _SLOW_DRAIN_CAP packets);
        return True to stop.

    Returns the sample time covered (seconds).

    Raises ``ValueError`` when the packets are too narrow to carry a
    requested channel: a short packet carries 128, and a capture that
    waited on channel 200 of one would train forever.
    """
    origin_set = False
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
                if ts is not None and not origin_set:
                    set_origin = getattr(capture_session, "set_time_origin", None)
                    if set_origin is not None:
                        set_origin(streamer.ts_day_epoch(pkt.ts))
                    origin_set = True
                if len(pkt) != width:
                    # Recomputed only when the packet mode changes, not
                    # per packet -- that would undo the point of
                    # blocking.
                    width = len(pkt)
                    columns = columns_for_width(requested, width)
                    missing = [c for c in requested if c not in columns[0]]
                    if missing:
                        mode = ("; short-packet mode"
                                if width == streamer.SHORT_PACKET_CHANNELS
                                else "")
                        raise ValueError(
                            f"Channels {missing} are beyond the slow "
                            f"packet width ({width} channels{mode}); the "
                            "stream cannot carry them.")
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


async def pfb_streamer_mismatch(crs, module: int, channels) -> Optional[str]:
    """Why the PFB streamer as configured cannot feed a capture of
    *channels* on *module*, or None.  A capture never configures the
    streamer; it reads what the board streams, and every captured
    channel must be among the streamed ones.  The board reports one
    channel as an integer and several as a list."""
    raw = await crs.get_pfb_streamer(module=module)
    active = raw
    if isinstance(active, dict):
        active = active.get("channel", active.get("channels"))
    if isinstance(active, (int, float)):
        active = [active]
    try:
        active = [int(c) for c in (active or [])]
    except TypeError:
        return (f"get_pfb_streamer(module={module}) returned {raw!r}, "
                "which this capture cannot read as a channel list.")
    if set(channels) <= set(active):
        return None
    have = f"streaming channels {active}" if active else "off"
    return (f"The PFB streamer on module {module} is {have}; this "
            f"capture needs channels {list(channels)}.  Set it with "
            "configure_streamer (Streamer Configuration in Periscope), "
            "then start again.")


async def run_pfb_source(
    capture_session,
    host: str,
    channels: List[int],
    *,
    module: Optional[int] = None,
    duration_s: Optional[float] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    sample_rate: float = PFB_SAMPLING_FREQ,
    max_lag_s: float = _PFB_MAX_LAG_S,
) -> float:
    """Feed *capture_session* from the fast/PFB stream until stopped.

    Every module's PFB streamer sends to the same port; with *module*
    given, packets from any other module are ignored.  A packet names
    the channel in each of its slots, so *channels* may be any subset
    of what the streamer carries.  The C++ receiver reads, validates
    and orders the packets on its own thread; this coroutine pops them
    demuxed, up to _PFB_POP_MAX at a time, and each pop is one engine
    call per channel.

    Each sample gets a timestamp extrapolated from the packet timestamp
    at the PFB rate.

    Returns the sample time covered (seconds).

    Raises
    ------
    TimeoutError
        If no packet ever arrives: the fast streamer is off or on the
        wrong module, and returning 0.0 would end a dual capture as
        though it had been stopped.
    """
    elapsed = 0.0
    source = getattr(capture_session, "source", None)
    set_origin = getattr(capture_session, "set_time_origin", None)
    stream_lag = getattr(capture_session, "stream_lag", None)
    origin_set = False
    last_ts = None                 # newest stamp fed, the fast clock here
    packet_s = streamer.PFBPACKET_NSAMP_MAX / sample_rate   # per packet

    def feed_batch(batch) -> bool:
        """One popped batch to the session, a block per channel; True
        once duration_s is covered."""
        nonlocal elapsed, origin_set, last_ts, packet_s
        (samples, seconds, recent, _seq, _mode, slots, num_samples,
         year, yday) = batch
        groups = samples.shape[0]
        time_samples = num_samples // groups
        packet_s = time_samples / sample_rate
        # Slot fields are 0-indexed on the wire, like the module field.
        slots = tuple(int(sl) + 1 for sl in slots[:groups])
        if recent.any():
            last_ts = float(seconds[recent][-1])
            if not origin_set and set_origin is not None:
                set_origin(streamer.day_epoch(year, yday))
                origin_set = True
        # Per-sample times from each packet's stamp; NaN where a packet
        # has no usable one, which the session drops and counts.
        times = (seconds[:, None]
                 + np.arange(time_samples) / sample_rate).ravel()
        # Match the slow stream's 16-bit ADC scale: the packetizer /256
        # is already out; the second /256 brings the 24-bit datapath
        # down to ADC counts, the slow readout path's convention.
        samples = samples / 256.0
        for ch in channels:
            if ch in slots:
                row = samples[slots.index(ch)]
                capture_session.feed_block(ch, row.real, row.imag, times)
        # Exact per-packet span -- independent of timestamp
        # discontinuities across rate changes.
        elapsed += seconds.shape[0] * time_samples / sample_rate
        return duration_s is not None and elapsed >= duration_s

    with streamer.get_multicast_socket(
            host, port=streamer.PFB_STREAMER_PORT,
            buffer_size=_rcvbuf_request()) as sock:
        held_s = _pfb_buffer_seconds(sock)
        sock.setblocking(False)
        _flush(sock)
        # The receiver reads, validates, and orders packets on its own
        # thread without the interpreter lock; this coroutine pops them
        # demuxed, a batch at a time.  The receive timeout is what lets
        # the thread notice a stop on a quiet stream.
        _set_receive_timeout(sock, _PFB_RECEIVE_TIMEOUT_S)
        receiver = streamer.PFBPacketReceiver(
            sock, reorder_window=_PFB_REORDER_WINDOW,
            queue_max_size=_PFB_QUEUE_MAX, flush_threshold=_PFB_FLUSH_EVERY)
        stop_rx = threading.Event()
        rx_error: List[BaseException] = []   # the coroutine raises it

        def receive() -> None:
            while not stop_rx.is_set():
                try:
                    receiver.receive_batch(
                        batch_size=2048,
                        timeout_ms=int(_PFB_RECEIVE_TIMEOUT_S * 1000))
                except Exception as e:
                    if not stop_rx.is_set():
                        rx_error.append(e)
                    return

        rx = threading.Thread(target=receive, name="pfb-receive",
                              daemon=True)
        rx.start()

        def find_queue():
            for _serial, mod, q in receiver.get_all_queues():
                if module is None or mod == module - 1:
                    return q
            return None

        queue = None
        got_any = False
        t_start = time.monotonic()
        last_batch_t = t_start
        flushed = 0
        backlog_peak = 0.0
        # Busy fraction: time spent feeding batches over wall time, per
        # second.  Near 1 the capture thread is saturated.
        busy = 0.0
        window_t = time.monotonic()
        if source is not None:
            source.update(buffer_s=held_s, backlog_s=0.0,
                          backlog_peak_s=0.0, lost_packets=0,
                          flushed_packets=0, busy=0.0)
        done = False
        stopped = False          # the loop ended by stop or duration
        try:
            while not done:
                if should_stop is not None and should_stop():
                    break
                if rx_error:
                    # A dead receiver would otherwise read as a stream
                    # gone quiet, and end only at the streamer timeout.
                    raise rx_error[0]
                if queue is None:
                    queue = find_queue()
                if not got_any and (queue is None or queue.empty()):
                    if time.monotonic() - t_start > streamer.STREAMER_TIMEOUT:
                        raise TimeoutError(
                            f"No PFB packets in {streamer.STREAMER_TIMEOUT} s"
                            " — the fast streamer is not sending. Check "
                            "that set_pfb_streamer enabled the right "
                            "channels and module (get_pfb_streamer says "
                            "what the board thinks), and that nothing "
                            "tore it down.")
                    await asyncio.sleep(0.01)
                    continue
                if (stream_lag is not None and last_ts is not None
                        and max_lag_s > 0):
                    lag = stream_lag()
                    if lag is not None and lag > max_lag_s:
                        flushed += queue.drop_pfb_before(
                            last_ts + lag - max_lag_s / 2)
                if source is not None:
                    # Lost: never arrived, dropped by the full queue, or
                    # discarded here for the lag bound; the last also
                    # on its own.
                    stats = queue.get_stats()
                    source["lost_packets"] = (stats.packets_missing
                                              + stats.packets_dropped
                                              + flushed)
                    source["flushed_packets"] = flushed
                    backlog = queue.size() * packet_s
                    queued = _rcvq_bytes(sock)
                    if queued is not None:
                        backlog += queued / _PFB_BYTES_PER_SECOND
                    backlog_peak = max(backlog_peak, backlog)
                    source["backlog_s"] = backlog
                    source["backlog_peak_s"] = backlog_peak
                    now = time.monotonic()
                    if now - window_t >= 1.0:
                        source["busy"] = min(1.0, busy / (now - window_t))
                        busy = 0.0
                        window_t = now
                batch = queue.pop_pfb_batch(_PFB_POP_MAX)
                if batch is None:
                    # A stream quiet for the streamer timeout has ended.
                    if time.monotonic() - last_batch_t > streamer.STREAMER_TIMEOUT:
                        break
                    await asyncio.sleep(0.002)
                    continue
                got_any = True
                batch_t = last_batch_t = time.monotonic()
                done = feed_batch(batch)
                busy += time.monotonic() - batch_t
                # An explicit turn for whatever shares the loop -- in a
                # dual capture, the slow stream.
                await asyncio.sleep(0)
            stopped = True
        finally:
            stop_rx.set()
            rx.join(1.0)
            # What the receiver's reorder window still held belongs to
            # the capture.
            receiver.flush_all()
            if queue is None:
                queue = find_queue()
            # Not after a failure: feeding the tail would only fail again.
            if queue is not None and stopped and not done:
                while True:
                    batch = queue.pop_pfb_batch(_PFB_POP_MAX)
                    if batch is None:
                        break
                    if feed_batch(batch):
                        break
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
        awaited.  Periscope passes its tap pump here (see
        :class:`SlowIngest` for why); both routes drive
        :class:`SlowIngest`, so only the transport differs.  Headless
        there is nothing to reuse and the default socket source is
        right.

    Returns ``(slow_elapsed, fast_elapsed)`` in seconds of sample time.

    A failure on one side cancels the other before it propagates, so
    neither stream outlives the capture it was feeding.
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
                capture_session.fast_feed, host, channels, module=module,
                duration_s=duration_s, should_stop=_stop)
        finally:
            finished = True

    sides = (asyncio.ensure_future(_slow()), asyncio.ensure_future(_fast()))
    try:
        slow_elapsed, fast_elapsed = await asyncio.gather(*sides)
    except BaseException:
        # gather propagates the first failure and leaves the other side
        # running; it would feed a session the caller has stopped.
        for side in sides:
            side.cancel()
        await asyncio.gather(*sides, return_exceptions=True)
        raise
    return slow_elapsed, fast_elapsed
