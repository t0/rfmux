"""A fastrx recording read, and written, with numpy alone.

``rfmux.fastrx.PacketFile`` reads a recording through the fastrx
extension, which builds on Linux only.  A recording is analysed wherever
the analysis happens, so :class:`RecordingFile` reads the same file with
a memory map and a structured dtype, and offers the same five accessors.
:class:`~rfmux.pulse_capture.overlay.Recording` opens a path with the
extension where it is built and with this elsewhere.

The layout is ``rfmux/streamer/include/fastrx.h``: a header block of
``HEADER_BYTES``, then records of one stride each, a record being the
86-byte wire header followed by the module's first ``channels`` I/Q
pairs as int16.  :func:`write_recording` writes that layout from arrays,
for a recording with known content: a test, or a demonstration without
a board.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..streamer import SS_PER_SECOND, TIMESTAMP_RECENT

FILE_MAGIC = 0x58464843
FILE_VERSION = 2
HEADER_BYTES = 4096
PACKET_MAGIC = 0x4348414E
MAX_CHANNELS = 1024
#: Channels per pipeline block on the wire.
CHANNELS_PER_PIPE = 128

FILE_HEADER_DTYPE = np.dtype([("magic", "<u4"), ("version", "<u4"),
                              ("record_stride", "<u4"), ("channels", "<u2"),
                              ("num_records", "<u8")])

TS_DTYPE = np.dtype([(name, "<u4") for name in
                     ("y", "d", "h", "m", "s", "ss", "c", "sbs")])

PACKET_HEADER_DTYPE = np.dtype({
    "names": ["magic", "seq", "pipe_snapshot", "sample_trunc", "module",
              "version", "tag", "serial", "samples_per_packet", "ts"],
    "formats": ["<u4", "<u4", "u1", "u1", "u1", "u1", "<u2", "<u2", "<u2",
                TS_DTYPE],
    "offsets": [0, 4, 8, 9, 10, 11, 12, 14, 16, 24],
    "itemsize": 86})


def record_stride(channels: int) -> int:
    """Bytes per record: the wire header and *channels* I/Q pairs,
    padded to a multiple of eight."""
    return (PACKET_HEADER_DTYPE.itemsize + channels * 4 + 7) & ~7


def _record_dtype(channels: int, stride: int) -> np.dtype:
    return np.dtype({"names": ["header", "iq"],
                     "formats": [PACKET_HEADER_DTYPE, ("<i2", (channels, 2))],
                     "offsets": [0, PACKET_HEADER_DTYPE.itemsize],
                     "itemsize": stride})


class RecordingFile:
    """A recording mapped read-only; the accessors return views."""

    def __init__(self, path):
        self.path = Path(path)
        size = self.path.stat().st_size
        if size < HEADER_BYTES:
            raise ValueError(f"{self.path}: too short to be a recording")
        head = np.fromfile(self.path, dtype=FILE_HEADER_DTYPE, count=1)[0]
        if head["magic"] != FILE_MAGIC:
            raise ValueError(f"{self.path}: bad magic, not a fastrx "
                             "recording")
        if head["version"] != FILE_VERSION:
            raise ValueError(f"{self.path}: format version {head['version']} "
                             f"(this reads {FILE_VERSION})")
        self.channels = int(head["channels"])
        self.record_stride = int(head["record_stride"])
        if not 1 <= self.channels <= MAX_CHANNELS:
            raise ValueError(f"{self.path}: {self.channels} channels "
                             f"(a recording holds 1..{MAX_CHANNELS})")
        if (self.record_stride < record_stride(self.channels)
                or self.record_stride % 8):
            raise ValueError(f"{self.path}: record stride {self.record_stride} "
                             f"cannot hold {self.channels} channels")
        # The file may end in padding past the last record, so its size
        # bounds the count rather than giving it.
        cap = (size - HEADER_BYTES) // self.record_stride
        count = int(head["num_records"])
        if count > cap:
            raise ValueError(f"{self.path}: header claims {count} records but "
                             f"the file holds at most {cap}")
        self._records = np.memmap(
            self.path, mode="r", offset=HEADER_BYTES, shape=(cap,),
            dtype=_record_dtype(self.channels, self.record_stride)) if cap \
            else np.empty(0, _record_dtype(self.channels, self.record_stride))
        if count == 0 and cap:
            # The count is written when a recording closes cleanly; zero
            # with data means the writer died.  Every record leads with
            # the packet magic: count those.
            ok = self._records["header"]["magic"] == PACKET_MAGIC
            count = cap if ok.all() else int(np.argmin(ok))
        self.num_packets = count
        self._records = self._records[:count]

    def __len__(self) -> int:
        return self.num_packets

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self._records = self._records[:0].copy()      # let the map go

    def headers(self) -> np.ndarray:
        return self._records["header"]

    def ts(self) -> np.ndarray:
        return self._records["header"]["ts"]

    def seq(self) -> np.ndarray:
        return self._records["header"]["seq"]

    def iq(self) -> np.ndarray:
        """int16, shape (records, channels, 2): I and Q as sent."""
        return self._records["iq"]


def write_recording(path, seconds, iq, *, module=1, sample_trunc=0,
                    seq=None, pipe_snapshot=None, year: int = 26,
                    day: int = 245, serial: int = 0) -> Path:
    """Write a recording of *iq* (int16, shape (records, channels, 2)),
    one record per entry of *seconds* (seconds of day, IRIG-disciplined;
    NaN for a stamp that is not).

    *module* (1-indexed), *seq* and *pipe_snapshot* are one value or one
    per record.  By default the sequence counts from zero and every pipe
    the channels span is present: pass a *seq* with a jump for a gap, or
    clear a *pipe_snapshot* bit for a drop-out.
    """
    iq = np.asarray(iq, dtype="<i2")
    seconds = np.asarray(seconds, dtype=np.float64)
    n, channels = iq.shape[0], iq.shape[1]
    if iq.shape != (len(seconds), channels, 2):
        raise ValueError("iq is (records, channels, 2), one record per stamp")
    stride = record_stride(channels)
    pipes = -(-channels // CHANNELS_PER_PIPE)
    records = np.zeros(n, dtype=_record_dtype(channels, stride))
    h = records["header"]
    h["magic"] = PACKET_MAGIC
    h["seq"] = np.arange(n) if seq is None else seq
    h["pipe_snapshot"] = (1 << pipes) - 1 if pipe_snapshot is None \
        else pipe_snapshot
    h["sample_trunc"] = sample_trunc
    h["module"] = np.asarray(module) - 1          # the wire counts from 0
    h["serial"] = serial
    h["samples_per_packet"] = pipes * CHANNELS_PER_PIPE
    good = np.isfinite(seconds)
    whole = np.where(good, seconds, 0.0).astype(np.int64)
    ts = h["ts"]
    ts["y"], ts["d"] = year, day
    ts["h"], ts["m"], ts["s"] = whole // 3600, (whole // 60) % 60, whole % 60
    ts["ss"] = np.round((np.where(good, seconds, 0.0) - whole)
                        * SS_PER_SECOND)
    ts["c"] = np.where(good, TIMESTAMP_RECENT, 0)
    ts["sbs"] = whole
    records["iq"] = iq
    head = np.zeros(1, dtype=FILE_HEADER_DTYPE)
    head["magic"], head["version"] = FILE_MAGIC, FILE_VERSION
    head["record_stride"], head["channels"] = stride, channels
    head["num_records"] = n
    path = Path(path)
    with open(path, "wb") as f:
        f.write(head.tobytes().ljust(HEADER_BYTES, b"\0"))
        f.write(records.tobytes())
    return path
