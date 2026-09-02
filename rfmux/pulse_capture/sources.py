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
import socket
import sys
import time
import warnings
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
            elif -self.MAX_PLAUSIBLE_STEP_S < delta <= 0.0:
                # A straggler from before the newest time seen: it
                # counts for nothing, and the newest time stands.
                return
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
        self._rows += 1
        # No usable timestamp becomes NaN; feed_block drops and counts
        # those exactly as feed_sample did.
        self._stamps.append(float("nan") if timestamp is None
                            else float(timestamp))
        if self.ready:
            self.flush(final=False)

    def add_block(self, channels, values, stamps) -> None:
        """:meth:`add` for a block: *values* (packets, channels), one
        stamp per row, NaN where a packet had no usable timestamp."""
        values = np.asarray(values)
        stamps = np.asarray(stamps, dtype=np.float64)
        if stamps.size:
            prev = float("nan") if self._prev_ts is None else self._prev_ts
            prev, self.elapsed = _advance_block(
                stamps, prev, self.elapsed, self.MAX_PLAUSIBLE_STEP_S)
            if prev == prev:
                self._prev_ts = prev
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
#: instead; the caps bound how long one wake holds the loop.
_SLOW_DRAIN_CAP = 256
_PFB_DRAIN_CAP = 64
#: PFB packets processed between turns of the loop.  While the engine is
#: capturing, every sample of a packet walks through process_sample, so
#: a full drain would hold the loop for ~150 ms; eight packets bound it
#: to ~20 ms, and the extra no-op yields cost nothing.
_PFB_YIELD_EVERY = 8
#: PFB packets gathered per channel before one engine call.  Sixteen
#: is 6.5 ms of stream.
_PFB_FEED_PACKETS = 16
# What the PFB socket's receive buffer holds, in seconds of a
# four-channel stream: 1000 samples per packet, so four channels at
# 2.44 MHz are 78 MB/s.  The socket asks for the largest buffer the
# host allows -- the kernel charges memory only for packets actually
# queued -- and warns below this much: a stall on the capture thread
# longer than the buffer loses packets, where a longer buffer is lag
# the source drains.
_PFB_BUFFER_WARN_S = 1.0
# The most the fast stream is let fall behind the slow one, in stream
# time, where the session measures the two clocks.  Past this the
# source reads and discards packets, looking only at their stamps,
# until the fast clock is within half the bound, and counts them: a
# bounded lag with the loss in the open, rather than a lag that grows
# to the socket buffer and leaves every pair without its fast window.
# The kernel's own account of the queue (SO_MEMINFO) is shown but not
# acted on: UDP releases receive memory in steps of a quarter of the
# buffer, so it reads high by up to that much.
_PFB_MAX_LAG_S = 0.25
_PFB_BYTES_PER_SECOND = 4 * PFB_SAMPLING_FREQ / 1000 * streamer.PFB_PACKET_SIZE


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


def _discard_until(sock, size: int, t_target: float, module,
                   limit: int = 1 << 20):
    """Read and drop datagrams stamped before *t_target*, at most
    *limit* of them.  Returns (dropped, the first kept datagram or
    None): the first one at or past the target is handed back so it
    is not lost with the rest."""
    dropped = 0
    while dropped < limit:
        try:
            data = sock.recv(size)
        except BlockingIOError:
            return dropped, None
        pkt = streamer.PFBPacket(data)
        if module is not None and pkt.module != module - 1:
            continue
        ts = streamer.ts_to_seconds(pkt.ts)
        if ts is not None and ts >= t_target:
            return dropped, data
        dropped += 1
    return dropped, None


def _pfb_buffer_seconds(sock) -> float:
    held = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    if sys.platform == "linux":
        held //= 2                     # Linux reports double the request
    return held / _PFB_BYTES_PER_SECOND
#: Channel groups a packet interleaves, by its mode field.
_PFB_GROUPS = {0: 1, 1: 2, 2: 4}


class _SeqReorder:
    """Puts datagrams back in sequence order across a short window.

    Datagrams arrive slightly out of order at high rates.  A packet is
    released once every earlier one has been, or once the stream has
    run *window* packets past it, which is a real loss.
    """

    def __init__(self, window: int = 16):
        self.window = window
        self._pending = {}
        self._next = None
        self._highest = None
        self.lost = 0            # sequence numbers given up on

    def push(self, seq: int, item) -> None:
        self._pending[seq] = item
        self._highest = seq if self._highest is None else max(self._highest, seq)

    def ready(self) -> list:
        out = []
        while self._pending:
            if self._next is None:
                self._next = min(self._pending)
            if self._next in self._pending:
                out.append(self._pending.pop(self._next))
                self._next += 1
            elif self._highest - self._next > self.window:
                skip_to = min(self._pending)
                self.lost += skip_to - self._next
                self._next = skip_to
            else:
                break
        return out

    def flush(self) -> list:
        out = [self._pending[k] for k in sorted(self._pending)]
        self._pending.clear()
        return out


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
    of what the streamer carries.  Packets are gathered into blocks of
    _PFB_FEED_PACKETS per channel before the engine sees them: the
    engine's cost per call is fixed, and one call per packet per
    channel is most of a core at four channels before any pulse.

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
    elapsed = 0.0
    got_any = False
    reorder = _SeqReorder()
    pending = {ch: [] for ch in channels}     # (I, Q, T) per packet
    pending_packets = 0
    source = getattr(capture_session, "source", None)
    set_origin = getattr(capture_session, "set_time_origin", None)
    stream_lag = getattr(capture_session, "stream_lag", None)
    origin_set = False
    last_ts = None                 # newest stamp fed, the fast clock here

    def flush_pending() -> None:
        nonlocal pending_packets
        for ch, parts in pending.items():
            if parts:
                capture_session.feed_block(
                    ch, np.concatenate([p[0] for p in parts]),
                    np.concatenate([p[1] for p in parts]),
                    np.concatenate([p[2] for p in parts]))
                parts.clear()
        pending_packets = 0

    def feed_packet(pkt) -> bool:
        """Gather one packet's samples; True once duration_s is covered."""
        nonlocal elapsed, pending_packets
        n_groups = _PFB_GROUPS.get(pkt.mode, 4)
        # Slot fields are 0-indexed on the wire, like the module field.
        slots = tuple(s + 1 for s in
                      (pkt.slot1, pkt.slot2, pkt.slot3, pkt.slot4)[:n_groups])
        # Match the slow stream's 16-bit ADC scale: np.array(pkt)
        # applies the packetizer /256; the second /256 brings the
        # 24-bit datapath down to ADC counts (same convention as the
        # slow readout path).  Without it, fast samples sit exactly
        # 256x above slow samples for the same signal.
        raw = np.array(pkt) / 256.0
        ts = streamer.ts_to_seconds(pkt.ts)
        nonlocal origin_set, last_ts
        if ts is not None:
            last_ts = ts
        if not origin_set and ts is not None and set_origin is not None:
            set_origin(streamer.ts_day_epoch(pkt.ts))
            origin_set = True
        time_samples = pkt.num_samples // n_groups
        # A whole packet per channel per call.  The packet interleaves
        # the streamed channels round-robin, so one channel is a
        # strided slice; feeding those arrays straight through is what
        # lets the engine absorb quiet stretches with numpy rather than
        # 1.22 million Python calls a second.  NaN where the packet has
        # no usable timestamp -- the session drops and counts those,
        # exactly as the per-sample path did with None.
        t0 = ts if ts is not None else float("nan")
        times = t0 + np.arange(time_samples) / sample_rate
        for ch in channels:
            if ch not in slots:
                continue
            v = raw[slots.index(ch):time_samples * n_groups:n_groups]
            n_ok = min(v.shape[0], times.shape[0])
            if n_ok:
                pending[ch].append((v[:n_ok].real, v[:n_ok].imag,
                                    times[:n_ok]))
        pending_packets += 1
        if pending_packets >= _PFB_FEED_PACKETS:
            flush_pending()
        # Exact per-packet span -- independent of timestamp
        # discontinuities across rate changes.
        elapsed += time_samples / sample_rate
        return duration_s is not None and elapsed >= duration_s

    with streamer.get_multicast_socket(
            host, port=streamer.PFB_STREAMER_PORT,
            buffer_size=_rcvbuf_request()) as sock:
        held_s = _pfb_buffer_seconds(sock)
        if held_s < _PFB_BUFFER_WARN_S:
            warnings.warn(
                f"PFB socket buffer holds {held_s:.2f} s of a four-channel "
                f"stream; a stall longer than that loses packets. Raise it "
                f"with: sudo sysctl -w net.core.rmem_max=268435456")
        sock.setblocking(False)
        _flush(sock)
        backlog_peak = 0.0
        flushed = 0
        carried = None             # a datagram a discard handed back
        # Busy fraction: time from a batch's arrival to the end of its
        # processing, over wall time, per second.  Near 1 the capture
        # thread is saturated; well under 1 while the backlog grows,
        # something else holds the interpreter.
        busy = 0.0
        window_t = time.monotonic()
        if source is not None:
            source.update(buffer_s=held_s, backlog_s=0.0,
                          backlog_peak_s=0.0, lost_packets=0,
                          flushed_packets=0, busy=0.0)
        done = False
        while not done:
            if should_stop is not None and should_stop():
                break
            if (stream_lag is not None and last_ts is not None
                    and max_lag_s > 0):
                lag = stream_lag()
                if lag is not None and lag > max_lag_s:
                    n, carried = _discard_until(
                        sock, streamer.PFB_PACKET_SIZE,
                        last_ts + lag - max_lag_s / 2, module)
                    flushed += n
            queued = _rcvq_bytes(sock)
            if queued is not None and source is not None:
                backlog = queued / _PFB_BYTES_PER_SECOND
                backlog_peak = max(backlog_peak, backlog)
                source["backlog_s"] = backlog
                source["backlog_peak_s"] = backlog_peak
            if source is not None:
                source["lost_packets"] = reorder.lost
                source["flushed_packets"] = flushed
                now = time.monotonic()
                if now - window_t >= 1.0:
                    source["busy"] = min(1.0, busy / (now - window_t))
                    busy = 0.0
                    window_t = now
            try:
                if carried is not None:
                    data, carried = carried, None
                else:
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
            batch_t = time.monotonic()
            for k, data in enumerate(_drain(sock, data,
                                            streamer.PFB_PACKET_SIZE,
                                            _PFB_DRAIN_CAP)):
                if k and k % _PFB_YIELD_EVERY == 0:
                    await asyncio.sleep(0)
                pkt = streamer.PFBPacket(data)
                if module is not None and pkt.module != module - 1:
                    continue
                reorder.push(pkt.seq, pkt)
                for pkt in reorder.ready():
                    if feed_packet(pkt):
                        done = True
                        break
                if done:
                    break
            busy += time.monotonic() - batch_t
            # An explicit turn for whatever shares the loop -- in a dual
            # capture, the slow stream.  The drain cap above bounds how
            # long this coroutine held it.
            await asyncio.sleep(0)
        for pkt in reorder.flush():
            feed_packet(pkt)
        flush_pending()
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
                capture_session.fast_feed, host, channels, module=module,
                duration_s=duration_s, should_stop=_stop)
        finally:
            finished = True

    slow_elapsed, fast_elapsed = await asyncio.gather(_slow(), _fast())
    return slow_elapsed, fast_elapsed
