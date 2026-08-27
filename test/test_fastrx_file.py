"""PacketFile: the offline half of the fastrx recording format.

These tests build recording files byte-by-byte, independently of PacketWriter,
so they pin the on-disk format itself (see rfmux/streamer/include/fastrx.h)
rather than merely whatever the writer happens to emit. No daemon, no NIC.
"""

import struct

import numpy as np
import pytest

fastrx = pytest.importorskip(
    "rfmux.fastrx", reason="this rfmux build does not include fastrx"
)

FILE_MAGIC = 0x58464843
FILE_VERSION = 1
HEADER_BYTES = 4096
PACKET_MAGIC = 0x4348414E
SPP = 128  # SAMPLES_PER_PIPELINE
BLOCK = SPP * 2 * 2  # bytes per pipe block


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


def record(mask: int, seq: int, *, snapshot=None, serial=42) -> bytes:
    """One record: wire header plus one I/Q block per pipe in mask.

    Each pipe's samples are filled with a value derived from (seq, pipe), so a
    misplaced stride or block rank shows up as wrong data, not just wrong
    shape. A pipe in mask but absent from snapshot is zero-filled, as the
    writer does during a pipeline drop-out.
    """
    if snapshot is None:
        snapshot = mask
    ts = (2026, 238, 12, 34, 56, 1000 + seq, 0, 0)
    hdr = struct.pack(
        "<IIBBBBHHH6x8I30x",
        PACKET_MAGIC, seq,
        snapshot, 2, 1, 0,   # pipe_snapshot, sample_trunc, module, version
        0, serial,           # tag, serial
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
        value = 100 * (p + 1) + seq
        iq = np.empty(2 * SPP, dtype=np.int16)
        iq[0::2] = value       # I
        iq[1::2] = -value      # Q
        blocks += iq.tobytes()

    rec = hdr + blocks
    return rec.ljust(stride_for(mask), b"\0")


def write(tmp_path, chunks, name="capture.fastrx"):
    path = tmp_path / name
    path.write_bytes(b"".join(chunks))
    return str(path)


def test_round_trip(tmp_path):
    mask = 0b101  # pipes 1 and 3
    n = 5
    path = write(tmp_path,
                 [file_header(mask, n)] + [record(mask, 100 + i) for i in range(n)])

    with fastrx.PacketFile(path) as f:
        assert len(f) == n
        assert f.num_packets == n
        assert f.pipes == [1, 3]
        assert f.pipe_mask == mask
        assert f.n_pipes == 2
        assert f.samples_per_pipe == SPP
        assert f.record_stride == stride_for(mask)

        assert np.array_equal(f.seq(), np.arange(100, 100 + n, dtype=np.uint32))

        # Stream metadata is not promoted to object level: it comes from the
        # wire headers, which every record carries verbatim.
        hdrs = f.headers()
        assert hdrs.shape == (n,)
        assert hdrs[0]["serial"] == 42
        assert hdrs[0]["sample_trunc"] == 2
        assert np.all(hdrs["magic"] == PACKET_MAGIC)
        assert np.all(hdrs["pipe_snapshot"] == mask)
        assert np.array_equal(hdrs["seq"], f.seq())

        for pipe in (1, 3):
            iq = f.pipe_iq(pipe)
            assert iq.shape == (n, SPP, 2)
            for i in range(n):
                value = 100 * pipe + 100 + i
                assert np.all(iq[i, :, 0] == value), (pipe, i)
                assert np.all(iq[i, :, 1] == -value), (pipe, i)

        ts = f.ts()
        assert ts.shape == (n,)
        assert np.all(ts["y"] == 2026)
        assert np.array_equal(ts["ss"], np.arange(1100, 1100 + n, dtype=np.uint32))

        # Pipe 2 exists on the wire generally, but not in this file.
        with pytest.raises(ValueError, match="not recorded"):
            f.pipe_iq(2)
        with pytest.raises(ValueError, match="pipe must be"):
            f.pipe_iq(0)


def test_dropout_records_are_zero_extended(tmp_path):
    # A recorded pipe absent from a packet's snapshot is zero-filled rather
    # than dropped: the record layout stays fixed, the timeline stays
    # contiguous, and pipe_snapshot says which blocks are real.
    mask = 0b11
    path = write(tmp_path, [
        file_header(mask, 3),
        record(mask, 0),
        record(mask, 1, snapshot=0b01),  # pipe 2 dropped out
        record(mask, 2),
    ])

    with fastrx.PacketFile(path) as f:
        assert f.num_packets == 3
        assert np.array_equal(f.seq(), [0, 1, 2])  # no gap

        iq2 = f.pipe_iq(2)
        assert np.all(iq2[0] != 0)
        assert np.all(iq2[1] == 0)                 # the zero-extended block
        assert np.all(iq2[2] != 0)

        # Real zeros are distinguished from fill by the wire snapshot.
        snap = f.headers()["pipe_snapshot"]
        assert list(snap & 0b10) == [0b10, 0, 0b10]

        # Pipe 1 was present throughout and is untouched by the drop-out.
        assert np.all(f.pipe_iq(1)[1, :, 0] == 100 + 1)


def test_odirect_tail_padding_is_not_data(tmp_path):
    # The writer pads the byte stream to a 4 KiB boundary; the count in the
    # header, not the file size, says where the records end.
    mask = 0b1
    n = 3
    body = [file_header(mask, n)] + [record(mask, i) for i in range(n)]
    total = sum(len(c) for c in body)
    body.append(b"\0" * (-total % 4096))
    path = write(tmp_path, body)

    with fastrx.PacketFile(path) as f:
        assert f.num_packets == n


def test_crash_recovery_scans_record_magics(tmp_path):
    # num_records == 0 with data present: the writer died before the final
    # header rewrite. The reader counts leading records with a valid magic
    # and refuses to guess past the first bad one (an out-of-order O_DIRECT
    # chunk that never landed reads as zeros).
    mask = 0b11
    stride = stride_for(mask)
    path = write(tmp_path,
                 [file_header(mask, 0)]
                 + [record(mask, i) for i in range(4)]
                 + [b"\0" * (2 * stride)]          # the hole
                 + [record(mask, 10)])             # landed beyond it
    with fastrx.PacketFile(path) as f:
        assert f.num_packets == 4


def test_empty_recording(tmp_path):
    # A recording that never saw a packet: geometry is still declared in the
    # header (it is fixed at construction, never inferred), just no records.
    path = write(tmp_path, [file_header(0b1, 0)])
    with fastrx.PacketFile(path) as f:
        assert f.num_packets == 0
        assert f.pipes == [1]
        assert len(f.seq()) == 0


def test_rejects_empty_pipe_mask(tmp_path):
    path = write(tmp_path, [file_header(0, 0, stride=0)])
    with pytest.raises(RuntimeError, match="no pipes"):
        fastrx.PacketFile(path)


def test_rejects_bad_magic(tmp_path):
    path = write(tmp_path, [file_header(0b1, 1, magic=0xDEADBEEF), record(0b1, 0)])
    with pytest.raises(RuntimeError, match="magic"):
        fastrx.PacketFile(path)


def test_rejects_unknown_version(tmp_path):
    path = write(tmp_path, [file_header(0b1, 1, version=99), record(0b1, 0)])
    with pytest.raises(RuntimeError, match="version 99"):
        fastrx.PacketFile(path)


def test_rejects_truncated_file(tmp_path):
    path = write(tmp_path, [file_header(0b1, 1)[:100]])
    with pytest.raises(RuntimeError, match="too short"):
        fastrx.PacketFile(path)


def test_rejects_overclaimed_count(tmp_path):
    # Header says more records than the bytes can hold.
    path = write(tmp_path, [file_header(0b1, 10), record(0b1, 0)])
    with pytest.raises(RuntimeError, match="at most"):
        fastrx.PacketFile(path)
