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
           recent=False, sample_trunc=2, module=2, iq=None) -> bytes:
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


def test_round_trip(tmp_path):
    channels = 2 * SPP  # pipes 1 and 2, whole
    n = 5
    path = write(tmp_path,
                 [file_header(channels, n)] + [record(channels, 100 + i) for i in range(n)])

    with fastrx.PacketFile(path) as f:
        assert len(f) == n
        assert f.num_packets == n
        assert f.channels == channels
        assert f.record_stride == stride_for(channels)

        assert np.array_equal(f.seq(), np.arange(100, 100 + n, dtype=np.uint32))

        # Stream metadata is not promoted to object level: it comes from the
        # wire headers, which every record carries verbatim.
        hdrs = f.headers()
        assert hdrs.shape == (n,)
        assert hdrs[0]["serial"] == 42
        assert hdrs[0]["sample_trunc"] == 2
        assert np.all(hdrs["magic"] == PACKET_MAGIC)
        assert np.all(hdrs["pipe_snapshot"] == 0b11)
        assert np.array_equal(hdrs["seq"], f.seq())

        iq = f.iq()
        assert iq.shape == (n, channels, 2)
        for pipe in (1, 2):
            lo = (pipe - 1) * SPP
            for i in range(n):
                value = 100 * pipe + 100 + i
                assert np.all(iq[i, lo:lo + SPP, 0] == value), (pipe, i)
                assert np.all(iq[i, lo:lo + SPP, 1] == -value), (pipe, i)

        ts = f.ts()
        assert ts.shape == (n,)
        assert np.all(ts["y"] == 2026)
        assert np.array_equal(ts["ss"], np.arange(1100, 1100 + n, dtype=np.uint32))


@pytest.mark.parametrize("channels", [1, 16, SPP, SPP + 1, 200, 2 * SPP - 1,
                                      7 * SPP + 3, MAX_CHANNELS])
def test_channels_are_a_prefix_of_the_module(tmp_path, channels):
    """channels counts module channels from 1: the pipes it spans are 1..k,
    every block whole except the last, which holds the remainder, and iq()
    sees the payload as one (n, channels, 2) array with pipe p's block at
    columns (p-1)*SPP onwards."""
    n = 3
    path = write(tmp_path,
                 [file_header(channels, n)]
                 + [record(channels, 500 + i) for i in range(n)])
    k = n_pipes(channels)

    with fastrx.PacketFile(path) as f:
        assert len(f) == n
        assert f.channels == channels
        assert f.record_stride == stride_for(channels)
        # Rounded to 8 bytes, so dropping one channel (4 bytes) may not
        # shrink it; it never grows.
        assert f.record_stride <= stride_for(MAX_CHANNELS)

        # The wire header is untouched by the truncation: it still says how
        # many samples the packet carried, not how many were kept.
        assert np.all(f.headers()["samples_per_packet"] == k * SPP)

        iq = f.iq()
        assert iq.shape == (n, channels, 2)

        offset = 0
        for pipe in range(1, k + 1):
            expect = min(SPP, channels - (pipe - 1) * SPP)
            block = iq[:, offset:offset + expect, :]
            for i in range(n):
                value = 100 * pipe + 500 + i
                assert np.all(block[i, :, 0] == value), (pipe, i)
                assert np.all(block[i, :, 1] == -value), (pipe, i)
            offset += expect
        assert offset == channels


def test_rejects_bad_channels(tmp_path):
    for channels in (0, MAX_CHANNELS + 1):
        path = write(tmp_path, [file_header(channels, 0, stride=stride_for(1))],
                     name=f"c{channels}.fastrx")
        with pytest.raises(RuntimeError, match="channels"):
            fastrx.PacketFile(path)


def test_dropout_records_are_zero_extended(tmp_path):
    # A pipe the layout needs but absent from a packet's snapshot is
    # zero-filled rather than dropped: the record layout stays fixed, the
    # timeline stays contiguous, and pipe_snapshot says which blocks are real.
    channels = 2 * SPP
    path = write(tmp_path, [
        file_header(channels, 3),
        record(channels, 0),
        record(channels, 1, snapshot=0b01),  # pipe 2 dropped out
        record(channels, 2),
    ])

    with fastrx.PacketFile(path) as f:
        assert f.num_packets == 3
        assert np.array_equal(f.seq(), [0, 1, 2])  # no gap

        iq = f.iq()
        pipe2 = iq[:, SPP:2 * SPP, :]
        assert np.all(pipe2[0] != 0)
        assert np.all(pipe2[1] == 0)               # the zero-extended block
        assert np.all(pipe2[2] != 0)

        # Real zeros are distinguished from fill by the wire snapshot.
        snap = f.headers()["pipe_snapshot"]
        assert list(snap & 0b10) == [0b10, 0, 0b10]

        # Pipe 1 was present throughout and is untouched by the drop-out.
        assert np.all(iq[1, :SPP, 0] == 100 + 1)


def test_odirect_tail_padding_is_not_data(tmp_path):
    # The writer pads the byte stream to a 4 KiB boundary; the count in the
    # header, not the file size, says where the records end.
    channels = SPP
    n = 3
    body = [file_header(channels, n)] + [record(channels, i) for i in range(n)]
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
    channels = 2 * SPP
    stride = stride_for(channels)
    path = write(tmp_path,
                 [file_header(channels, 0)]
                 + [record(channels, i) for i in range(4)]
                 + [b"\0" * (2 * stride)]          # the hole
                 + [record(channels, 10)])         # landed beyond it
    with fastrx.PacketFile(path) as f:
        assert f.num_packets == 4


def test_empty_recording(tmp_path):
    # A recording that never saw a packet: geometry is still declared in the
    # header (it is fixed at construction, never inferred), just no records.
    path = write(tmp_path, [file_header(SPP, 0)])
    with fastrx.PacketFile(path) as f:
        assert f.num_packets == 0
        assert f.channels == SPP
        assert len(f.seq()) == 0


def test_rejects_bad_magic(tmp_path):
    path = write(tmp_path, [file_header(SPP, 1, magic=0xDEADBEEF), record(SPP, 0)])
    with pytest.raises(RuntimeError, match="magic"):
        fastrx.PacketFile(path)


def test_rejects_unknown_version(tmp_path):
    path = write(tmp_path, [file_header(SPP, 1, version=99), record(SPP, 0)])
    with pytest.raises(RuntimeError, match="version 99"):
        fastrx.PacketFile(path)


def test_rejects_truncated_file(tmp_path):
    path = write(tmp_path, [file_header(SPP, 1)[:100]])
    with pytest.raises(RuntimeError, match="too short"):
        fastrx.PacketFile(path)


def test_rejects_overclaimed_count(tmp_path):
    # Header says more records than the bytes can hold.
    path = write(tmp_path, [file_header(SPP, 10), record(SPP, 0)])
    with pytest.raises(RuntimeError, match="at most"):
        fastrx.PacketFile(path)


def test_daemon_helpers_read_the_socket_dir_and_name_the_binary(
        tmp_path, monkeypatch):
    monkeypatch.setattr(fastrx, "SOCKET_DIR", str(tmp_path))
    assert fastrx.running_interfaces() == []
    (tmp_path / "enp2s0f0np0").touch()
    assert fastrx.running_interfaces() == ["enp2s0f0np0"]
    cmd = fastrx.start_command("enp2s0f0np0")
    assert cmd.startswith("sudo ") and cmd.endswith("/fastrxd -i enp2s0f0np0")

