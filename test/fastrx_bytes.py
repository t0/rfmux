"""A fastrx recording built byte by byte, independently of any writer,
so the tests that read one pin the on-disk format itself (see
rfmux/streamer/include/fastrx.h).  No extension, no daemon, no NIC."""

import struct

import numpy as np

FILE_MAGIC = 0x58464843
FILE_VERSION = 2
HEADER_BYTES = 4096
PACKET_MAGIC = 0x4348414E
SPP = 128            # SAMPLES_PER_PIPELINE: one pipe's block on the wire
MAX_CHANNELS = 1024  # all 8 pipes


def n_pipes(channels: int) -> int:
    """Pipes the module's first `channels` channels span: 1..n_pipes."""
    return -(-channels // SPP)


def stride_for(channels: int) -> int:
    return (86 + channels * 2 * 2 + 7) & ~7


def file_header(channels: int, num_records: int, stride: int | None = None,
                *, magic=FILE_MAGIC, version=FILE_VERSION) -> bytes:
    if stride is None:
        stride = stride_for(channels)
    h = struct.pack(
        "<IIIHQ",
        magic, version,
        stride, channels, num_records,
    )
    return h.ljust(HEADER_BYTES, b"\0")


RECENT = 0x80000000  # irigb_timestamp.c bit 31: the stamp is disciplined


def seconds_ts(seconds: float, *, y=26, d=245):
    """A (y, d, h, m, s, ss, c, sbs) stamp for *seconds* of day."""
    whole = int(seconds)
    ss = int(round((seconds - whole) * 156_250_000))
    return (y, d, whole // 3600, (whole // 60) % 60, whole % 60, ss, 0, whole)


def record(channels: int, seq: int, *, snapshot=None, serial=42, ts=None,
           recent=False, sample_trunc=2, module=1, iq=None) -> bytes:
    """One record: wire header plus the module's first `channels` I/Q pairs.

    Pipe p's samples are filled with a value derived from (seq, p), so a
    misplaced stride or block offset shows up as wrong data, not just wrong
    shape; *iq* ({pipe: (SPP, 2) int16}) overrides that per pipe.  A pipe
    the layout needs but absent from snapshot is zero-filled, as the writer
    does during a pipeline drop-out.  *ts* is (y, d, h, m, s, ss, c, sbs);
    *recent* sets the disciplined bit in c.  *module* is 1-indexed; the
    wire carries it from 0.

    The wire header advertises full blocks for every pipe in snapshot: the
    writer copies it verbatim and truncates only the payload.
    """
    if snapshot is None:
        snapshot = (1 << n_pipes(channels)) - 1
    if ts is None:
        ts = (2026, 238, 12, 34, 56, 1000 + seq, 0, 0)
    ts = list(ts)
    if recent:
        ts[6] |= RECENT
    hdr = struct.pack(
        "<IIBBBBHHH6x8I30x",
        PACKET_MAGIC, seq,
        snapshot, sample_trunc, module - 1, 0,   # pipe_snapshot, sample_trunc, module, version
        0, serial,                      # tag, serial
        bin(snapshot).count("1") * SPP,
        *ts,
    )
    assert len(hdr) == 86

    blocks = b""
    remaining = channels
    for p in range(n_pipes(channels)):
        n = min(remaining, SPP)
        remaining -= n
        if not snapshot & (1 << p):
            blocks += b"\0" * (n * 2 * 2)
            continue
        if iq is not None and (p + 1) in iq:
            blocks += np.ascontiguousarray(iq[p + 1][:n], dtype=np.int16).tobytes()
            continue
        value = np.int64(100 * (p + 1) + seq)
        samples = np.empty(2 * n, dtype=np.int16)
        samples[0::2] = value.astype(np.int16)       # I, wrapping
        samples[1::2] = (-value).astype(np.int16)    # Q
        blocks += samples.tobytes()
    assert remaining == 0

    rec = hdr + blocks
    return rec.ljust(stride_for(channels), b"\0")


def write(tmp_path, chunks, name="capture.fastrx"):
    path = tmp_path / name
    path.write_bytes(b"".join(chunks))
    return str(path)
