"""fastrx: zero-copy consumers for the fastrxd channel-stream daemon.

A separate capture path from rfmux.streamer's socket-based receiver: this one
reads packets straight out of the NIC via AF_XDP, through a daemon
(fastrxd, run via "rfmux fastrxd") that a client attaches to over a Unix
socket. The compiled extension (_fastrx) is built alongside _receiver from
rfmux/streamer/ (see rfmux/streamer/CMakeLists.txt).

The API is synchronous; every blocking call releases the GIL.
"""

import os
from dataclasses import dataclass

import numpy as np

from .streamer import SS_PER_SECOND
from .streamer._fastrx import (
    ABI_VERSION,
    MAX_CLIENTS,
    MAX_SAMPLES,
    NUM_PIPELINES,
    SOCKET_DIR,
)
from .streamer._fastrx import PacketCapture as _PacketCapture
from .streamer._fastrx import PacketFile as _PacketFile
from .streamer._fastrx import PacketWriter as _PacketWriter

__all__ = [
    "PacketCapture",
    "PacketWriter",
    "PacketFile",
    "Window",
    "channel_location",
    "get_samples",
    "resolve_socket",
    "NUM_PIPELINES",
    "MAX_SAMPLES",
    "ABI_VERSION",
    "MAX_CLIENTS",
    "SOCKET_DIR",
]


def resolve_socket(interface: str | None = None,
                   socket: str | None = None) -> str:
    """Turn an interface name (or nothing at all) into a fastrxd socket path.

    "socket" is for a daemon started with --socket-path, whose path cannot be
    derived from anything.
    """

    if interface is not None and socket is not None:
        raise ValueError("give either interface= or socket=, not both")
    if socket is not None:
        return socket

    if interface is None:
        # Nothing specified. One socket means there is no choice to make, so
        # requiring a name would be ceremony; two or more is a real ambiguity,
        # and picking one would silently attach to the wrong NIC -- which
        # presents as a transmitter that is not sending rather than a mistake.
        try:
            found = sorted(os.listdir(SOCKET_DIR))
        except OSError:
            found = []
        if len(found) != 1:
            raise ValueError(
                f"no fastrxd sockets in {SOCKET_DIR}; is fastrxd running?"
                if not found else
                "several fastrxd instances are running; name one with "
                f"interface=: {', '.join(found)}"
            )
        interface = found[0]

    if "/" in interface:
        raise ValueError(f"not an interface name: {interface!r}")

    return os.path.join(SOCKET_DIR, interface)


class PacketCapture(_PacketCapture):
    """One fastrxd client slot, with its own hot and cold thread.

    The extension takes a socket path; this adds the convenience of naming the
    interface instead, or of naming nothing when there is only one fastrxd.

        PacketCapture() # when only one fastrxd is running
        PacketCapture(interface="enp9s0f0np0")
        PacketCapture(socket="/tmp/fastrxd.sock")

    The pipeline is chosen per capture() call, not here.
    """

    def __init__(
        self,
        *,
        interface: str | None = None,
        socket: str | None = None,
    ):
        super().__init__(resolve_socket(interface, socket))


class PacketWriter(_PacketWriter):
    """Records the stream to disk (for readback using PacketFile).

        with PacketWriter("run.fastrx", pipes=[1], n_packets=1_000_000) as w:
            w.wait()
        # stop() (via __exit__) flushes, finalizes and closes.

    n_packets (defaults to None) bounds the recording.

    pipes selects which pipelines to record (1-indexed) and is required.

    If the disk falls behind, records are dropped and counted in .overruns
    rather than ever blocking the packet path.

    A recorded pipe that goes missing mid-stream (transmitter
    reconfiguration) is zero-filled rather than dropped, counted in
    .dropouts; each record's pipe_snapshot (PacketFile.headers()) says
    which blocks are real.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        pipes: list[int],
        n_packets: int | None = None,
        interface: str | None = None,
        socket: str | None = None,
        ring_mb: int = 256,
        queue_depth: int = 32,
    ):
        mask = 0
        pipes = list(pipes)
        if not pipes:
            raise ValueError("pipes must name at least one pipeline")
        for p in pipes:
            if not 1 <= p <= NUM_PIPELINES:
                raise ValueError(
                    f"pipe must be in 1..{NUM_PIPELINES}, got {p}")
            mask |= 1 << (p - 1)

        if n_packets is not None and n_packets <= 0:
            raise ValueError(
                f"n_packets must be positive (or None), got {n_packets}")

        super().__init__(
            resolve_socket(interface, socket),
            os.fspath(path),
            pipe_mask=mask,
            n_packets=n_packets or 0,
            ring_bytes=ring_mb << 20,
            queue_depth=queue_depth,
        )


#: Channels per pipeline block: channel c (1-indexed, as everywhere in
#: rfmux) is column (c-1) % MAX_SAMPLES of pipe (c-1) // MAX_SAMPLES + 1,
#: the parser's channel order.
CHANNELS_PER_PIPE = MAX_SAMPLES

#: Scale from the truncated int16 on the wire to the ADC counts the 1G
#: paths report (the /256 packetizer gain taken out), by sample_trunc:
#: HIGH keeps bits 23:8 and is exact; MID and LOW keep lower windows, so
#: they are exact only while the signal stays inside them.
COUNTS_PER_LSB = {0: 1.0 / 256, 1: 1.0 / 16, 2: 1.0}

_DAY_S = 86400.0
#: Records probed past an undisciplined stamp before giving up on it.
_PROBE = 64
_RECENT = 0x80000000


def channel_location(channel: int) -> tuple[int, int]:
    """(pipe, column) of a 1-indexed channel."""
    top = NUM_PIPELINES * CHANNELS_PER_PIPE
    if not 1 <= channel <= top:
        raise ValueError(f"channel must be in 1..{top}, got {channel}")
    return (channel - 1) // CHANNELS_PER_PIPE + 1, (channel - 1) % CHANNELS_PER_PIPE


def _seconds_of_day(ts) -> np.ndarray:
    """Seconds of day of IRIG stamps (a structured array or one element),
    NaN where the stamp is not disciplined."""
    ts = np.asarray(ts)
    t = (ts["h"].astype(np.float64) * 3600.0 + ts["m"] * 60.0 + ts["s"]
         + ts["ss"] / SS_PER_SECOND)
    return np.where(ts["c"] & _RECENT, t, np.nan)


@dataclass
class Window:
    """One channel's samples over a time window of a recording.

    ``times`` are seconds of day (see :meth:`PacketFile.seconds`), NaN
    where a record's stamp is not disciplined; ``samples`` are complex
    ADC counts (``PacketFile.counts_per_lsb`` applied).  ``seq_gaps``
    counts sequence discontinuities inside the window and ``dropouts``
    the records whose pipe the transmitter was not sending (zero-filled
    by the writer)."""
    channel: int
    start: int
    stop: int
    times: np.ndarray
    samples: np.ndarray
    seq_gaps: int
    dropouts: int


class PacketFile(_PacketFile):
    """A recording, with a time index over its IRIG stamps.

    The extension maps the file and hands back strided views; nothing
    here reads more of it than the records asked for.  Every record is
    one sample of each channel in its pipes, stamped by the board, so a
    stamp is a sample time with no first-or-last-in-packet ambiguity.

    Time is seconds of day, the axis pulse-capture files and parser
    dirfiles use.  A recording that crosses midnight is unwrapped: any
    stamp more than half a day before the first one is taken as the next
    day, and queries are read the same way.
    """

    def __init__(self, path: str | os.PathLike):
        super().__init__(os.fspath(path))
        self._ts = self.ts()
        self._seq = self.seq()
        n = self.num_packets
        hdr0 = self.headers()[0] if n else None
        #: fastrx_trunc_t of the recording (from the first record).
        self.sample_trunc = int(hdr0["sample_trunc"]) if n else 2
        #: Wire module field of the first record, as sent.
        self.module = int(hdr0["module"]) if n else None
        self.counts_per_lsb = COUNTS_PER_LSB[self.sample_trunc]
        #: Seconds of day of the first disciplined stamp; None when the
        #: recording has none near its start, in which case there is no
        #: time axis to index by.
        self.t_first = None
        t, _ = self._second_from(0)
        self.t_first = None if t != t else float(t)

    # ── time ──────────────────────────────────────────────────────

    def _unwrap(self, t):
        """Seconds of day onto the recording's monotone axis."""
        if self.t_first is None:
            return t
        return np.where(t < self.t_first - _DAY_S / 2, t + _DAY_S, t)

    def seconds(self, start: int = 0, stop: int | None = None) -> np.ndarray:
        """Seconds of day of records ``start:stop``, NaN where the stamp
        is not disciplined.  Touches only those records."""
        t = _seconds_of_day(self._ts[start:stop])
        return self._unwrap(t)

    def _second_from(self, i: int):
        """(seconds, index) of the first disciplined stamp at or after
        record *i* within the probe distance; (NaN, i) if none."""
        for j in range(i, min(i + _PROBE, self.num_packets)):
            t = _seconds_of_day(self._ts[j])
            if t == t:
                return float(self._unwrap(t)), j
        return float("nan"), i

    def index_at(self, t: float, side: str = "left") -> int:
        """Record index for seconds-of-day *t*: the first record stamped
        at or after *t* (``side="left"``) or after it (``"right"``).
        A bisect, so a query costs about log2(records) page touches
        rather than a read of the file.  A record with no usable stamp
        is placed with the next disciplined one (so it can open a
        window, NaN-timed); a stretch of them longer than the probe
        distance is treated as beyond the end."""
        if self.t_first is None:
            raise ValueError("recording has no disciplined timestamp to "
                             "index by")
        t = float(self._unwrap(np.float64(t)))
        lo, hi = 0, self.num_packets
        while lo < hi:
            mid = (lo + hi) // 2
            tm, j = self._second_from(mid)
            if tm != tm:                       # nothing usable ahead of mid
                hi = mid
            elif tm < t or (side == "right" and tm == t):
                lo = j + 1
            else:
                hi = mid
        return lo

    # ── samples ───────────────────────────────────────────────────

    def channel(self, channel: int, start: int = 0,
                stop: int | None = None) -> np.ndarray:
        """One channel's samples over records ``start:stop`` as complex
        ADC counts."""
        pipe, col = channel_location(channel)
        iq = self.pipe_iq(pipe)[start:stop, col, :]
        z = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)
        return z * np.float32(self.counts_per_lsb)

    def window(self, t0: float, t1: float, channel: int) -> Window:
        """*channel* over seconds-of-day ``[t0, t1]``."""
        start = self.index_at(t0)
        stop = self.index_at(t1, side="right")
        pipe, _ = channel_location(channel)
        seq = self._seq[start:stop].astype(np.int64)
        snap = self.headers()[start:stop]["pipe_snapshot"]
        return Window(
            channel=channel, start=start, stop=stop,
            times=self.seconds(start, stop),
            samples=self.channel(channel, start, stop),
            seq_gaps=int(np.count_nonzero(np.diff(seq) != 1)) if seq.size else 0,
            dropouts=int(np.count_nonzero((snap & (1 << (pipe - 1))) == 0)),
        )


def get_samples(n_packets: int, pipe: int = 1, timeout: float = 5.0, **kwargs):
    """Grab the next `n_packets` from one pipe, then tear everything down.

    A short-lived PacketCapture, for callers who want one grab and no lifetime
    to manage.  Everything expensive is per-connection rather than per-packet,
    so code doing this repeatedly should keep a PacketCapture and call
    capture() on it instead:

        c = fastrx.PacketCapture(interface="enp9s0f0np0")
        while True:
            d = c.capture(1024)
    """

    if n_packets <= 0:
        raise ValueError(f"n_packets must be positive, got {n_packets}")
    with PacketCapture(**kwargs) as c:
        return c.capture(n_packets, pipe, timeout)
