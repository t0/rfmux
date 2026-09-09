"""Byte-level builders for fastrx recording files (rfmux/streamer/include/
fastrx.h), independent of PacketWriter, so tests pin the on-disk format
itself.  No daemon, no NIC."""

import struct

import numpy as np

FILE_MAGIC = 0x58464843
FILE_VERSION = 1
HEADER_BYTES = 4096
PACKET_MAGIC = 0x4348414E
SPP = 128  # SAMPLES_PER_PIPELINE
BLOCK = SPP * 2 * 2  # bytes per pipe block
RECENT = 0x80000000  # irigb_timestamp.c bit 31: the stamp is disciplined


def stride_for(mask: int) -> int:
    return (86 + bin(mask).count("1") * BLOCK + 7) & ~7


def file_header(mask: int, num_records: int, stride: int | None = None,
                *, magic=FILE_MAGIC, version=FILE_VERSION) -> bytes:
    if stride is None:
        stride = stride_for(mask)
    h = struct.pack(
        "<IIIHHQ",
        magic, version,
        stride, SPP, mask, num_records,
    )
    return h.ljust(HEADER_BYTES, b"\0")


def record(mask: int, seq: int, *, snapshot=None, serial=42,
           ts=None, recent=False, sample_trunc=2,
           iq=None) -> bytes:
    """One record: wire header plus one I/Q block per pipe in mask.

    Each pipe's samples are filled with a value derived from (seq, pipe),
    so a misplaced stride or block rank shows up as wrong data, not just
    wrong shape; *iq* ({pipe: (SPP, 2) int16}) overrides that per pipe.
    A pipe in mask but absent from snapshot is zero-filled, as the
    writer does during a pipeline drop-out.

    *ts* is (y, d, h, m, s, ss, c, sbs); the default counts seq into the
    sub-second field.  *recent* sets the disciplined bit in c.
    """
    if snapshot is None:
        snapshot = mask
    if ts is None:
        ts = (2026, 238, 12, 34, 56, 1000 + seq, 0, 0)
    ts = list(ts)
    if recent:
        ts[6] |= RECENT
    hdr = struct.pack(
        "<IIBBBBHHH6x8I30x",
        PACKET_MAGIC, seq,
        snapshot, sample_trunc, 1, 0,   # pipe_snapshot, sample_trunc, module, version
        0, serial,                      # tag, serial
        bin(snapshot).count("1") * SPP,
        *ts,
    )
    assert len(hdr) == 86

    blocks = b""
    for p in range(8):
        if not mask & (1 << p):
            continue
        if not snapshot & (1 << p):
            blocks += b"\0" * BLOCK
            continue
        if iq is not None and (p + 1) in iq:
            block = np.ascontiguousarray(iq[p + 1], dtype=np.int16)
            assert block.shape == (SPP, 2)
            blocks += block.tobytes()
            continue
        value = 100 * (p + 1) + seq
        samples = np.empty(2 * SPP, dtype=np.int16)
        samples[0::2] = value       # I
        samples[1::2] = -value      # Q
        blocks += samples.tobytes()

    rec = hdr + blocks
    return rec.ljust(stride_for(mask), b"\0")


def write(tmp_path, chunks, name="capture.fastrx"):
    path = tmp_path / name
    path.write_bytes(b"".join(chunks))
    return str(path)


def seconds_ts(seconds: float, *, y=26, d=245):
    """A (y, d, h, m, s, ss, c, sbs) stamp for *seconds* of day, ss at
    the 156.25 MHz IRIG count."""
    whole = int(seconds)
    ss = int(round((seconds - whole) * 156_250_000))
    return (y, d, whole // 3600, (whole // 60) % 60, whole % 60, ss, 0, whole)
