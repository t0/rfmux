"""The offline half of the fastrx recording format, read two ways:
``rfmux.fastrx.PacketFile`` through the extension, where it is built, and
``RecordingFile`` with numpy alone.

The files are built byte by byte (``test/fastrx_bytes.py``), so these pin
the on-disk format rather than whatever a writer happens to emit.
"""

import sys

import numpy as np
import pytest

from rfmux.pulse_capture import recording_file
from rfmux.pulse_capture.recording_file import RecordingFile
from test.fastrx_bytes import (  # noqa: F401  (re-exported to the overlay tests)
    MAX_CHANNELS, PACKET_MAGIC, SPP, file_header, n_pipes, record,
    seconds_ts, stride_for, write)

try:
    from rfmux import fastrx
except ImportError:                     # the extension builds on Linux only
    fastrx = None


@pytest.fixture(params=["extension", "numpy"])
def PacketFile(request):
    """The reader under test."""
    if request.param == "numpy":
        if sys.platform == "win32":
            pytest.skip("bisecting an access violation on the Windows runner")
        return RecordingFile
    if fastrx is None:
        pytest.skip("this rfmux build does not include fastrx")
    return fastrx.PacketFile


#: What a reader raises for a file it will not read.
REJECTED = (RuntimeError, ValueError)


def test_the_bound_stride_is_the_formats():
    """What the writer lays out and record_streams budgets by."""
    for channels in (1, 16, 114, SPP, MAX_CHANNELS):
        assert recording_file.record_stride(channels) == stride_for(channels)
        if fastrx is not None:
            assert fastrx.record_stride(channels) == stride_for(channels)


def test_round_trip(tmp_path, PacketFile):
    channels = 2 * SPP  # pipes 1 and 2, whole
    n = 5
    path = write(tmp_path,
                 [file_header(channels, n)] + [record(channels, 100 + i) for i in range(n)])

    with PacketFile(path) as f:
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
def test_channels_are_a_prefix_of_the_module(tmp_path, channels,
                                             PacketFile):
    """channels counts module channels from 1: the pipes it spans are 1..k,
    every block whole except the last, which holds the remainder, and iq()
    sees the payload as one (n, channels, 2) array with pipe p's block at
    columns (p-1)*SPP onwards."""
    n = 3
    path = write(tmp_path,
                 [file_header(channels, n)]
                 + [record(channels, 500 + i) for i in range(n)])
    k = n_pipes(channels)

    with PacketFile(path) as f:
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


def test_rejects_bad_channels(tmp_path, PacketFile):
    for channels in (0, MAX_CHANNELS + 1):
        path = write(tmp_path, [file_header(channels, 0, stride=stride_for(1))],
                     name=f"c{channels}.fastrx")
        with pytest.raises(REJECTED, match="channels"):
            PacketFile(path)


def test_dropout_records_are_zero_extended(tmp_path, PacketFile):
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

    with PacketFile(path) as f:
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


def test_odirect_tail_padding_is_not_data(tmp_path, PacketFile):
    # The writer pads the byte stream to a 4 KiB boundary; the count in the
    # header, not the file size, says where the records end.
    channels = SPP
    n = 3
    body = [file_header(channels, n)] + [record(channels, i) for i in range(n)]
    total = sum(len(c) for c in body)
    body.append(b"\0" * (-total % 4096))
    path = write(tmp_path, body)

    with PacketFile(path) as f:
        assert f.num_packets == n


def test_crash_recovery_scans_record_magics(tmp_path, PacketFile):
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
    with PacketFile(path) as f:
        assert f.num_packets == 4


def test_empty_recording(tmp_path, PacketFile):
    # A recording that never saw a packet: geometry is still declared in the
    # header (it is fixed at construction, never inferred), just no records.
    path = write(tmp_path, [file_header(SPP, 0)])
    with PacketFile(path) as f:
        assert f.num_packets == 0
        assert f.channels == SPP
        assert len(f.seq()) == 0


def test_rejects_bad_magic(tmp_path, PacketFile):
    path = write(tmp_path, [file_header(SPP, 1, magic=0xDEADBEEF), record(SPP, 0)])
    with pytest.raises(REJECTED, match="magic"):
        PacketFile(path)


def test_rejects_unknown_version(tmp_path, PacketFile):
    path = write(tmp_path, [file_header(SPP, 1, version=99), record(SPP, 0)])
    with pytest.raises(REJECTED, match="version 99"):
        PacketFile(path)


def test_rejects_truncated_file(tmp_path, PacketFile):
    path = write(tmp_path, [file_header(SPP, 1)[:100]])
    with pytest.raises(REJECTED, match="too short"):
        PacketFile(path)


def test_rejects_overclaimed_count(tmp_path, PacketFile):
    # Header says more records than the bytes can hold.
    path = write(tmp_path, [file_header(SPP, 10), record(SPP, 0)])
    with pytest.raises(REJECTED, match="at most"):
        PacketFile(path)


def test_a_written_recording_reads_back(tmp_path, PacketFile):
    """write_recording lays out what either reader reads: the samples,
    the stamps with their disciplined bit, the modules from 1, and the
    sequence and snapshot it was given."""
    from rfmux.pulse_capture.overlay import Recording
    n, channels = 6, 5
    iq = np.arange(n * channels * 2, dtype=np.int16).reshape(n, channels, 2)
    seconds = 43000.0 + np.arange(n) * 1e-3
    seconds[2] = np.nan
    path = recording_file.write_recording(
        tmp_path / "w.fastrx", seconds, iq, module=[1, 2, 1, 2, 1, 2],
        seq=[0, 0, 1, 1, 5, 5], pipe_snapshot=[1, 1, 1, 1, 0, 1],
        sample_trunc=1)
    with PacketFile(str(path)) as f:
        assert (f.num_packets, f.channels) == (n, channels)
        assert f.record_stride == stride_for(channels)
        assert np.array_equal(f.iq(), iq)
        assert list(f.seq()) == [0, 0, 1, 1, 5, 5]
        assert list(f.headers()["module"]) == [0, 1, 0, 1, 0, 1]
        assert list(f.headers()["pipe_snapshot"]) == [1, 1, 1, 1, 0, 1]
        rec = Recording(f)
        assert rec.sample_trunc == 1 and rec.counts_per_lsb == 16.0
        t = rec.seconds()
        assert np.isnan(t[2]) and np.allclose(t[[0, 1, 3]], seconds[[0, 1, 3]])
        w = rec.window(43000.0, 43000.006, 3, module=1)
        assert (w.seq_gaps, w.dropouts) == (1, 1)
        assert np.array_equal(w.samples, (iq[0::2, 2, 0] + 1j * iq[0::2, 2, 1]) * 16.0)


def test_daemon_helpers_read_the_socket_dir_and_name_the_binary(
        tmp_path, monkeypatch):
    if fastrx is None:
        pytest.skip("this rfmux build does not include fastrx")
    monkeypatch.setattr(fastrx, "SOCKET_DIR", str(tmp_path))
    assert fastrx.running_interfaces() == []
    (tmp_path / "enp2s0f0np0").touch()
    assert fastrx.running_interfaces() == ["enp2s0f0np0"]
    cmd = fastrx.start_command("enp2s0f0np0")
    assert cmd.startswith("sudo ") and cmd.endswith("/fastrxd -i enp2s0f0np0")

