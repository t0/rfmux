"""Recording: a time index over a fastrx file, seconds of day from the
IRIG stamps, a bisect that touches one record per step, and one
channel's samples over a window.  Files are built byte-by-byte with
the helpers in test/test_fastrx_file.py."""

import numpy as np
import pytest

pytest.importorskip(
    "rfmux.fastrx", reason="this rfmux build does not include fastrx"
)

from rfmux.core.transferfunctions import PFB_SAMPLING_FREQ
from rfmux.pulse_capture.overlay import Recording, channel_location
from test.test_fastrx_file import file_header, record, seconds_ts, write

T0 = 43000.0
DT = 1.0 / PFB_SAMPLING_FREQ


def _file(tmp_path, seqs, *, t0=T0, mask=0b11, recent=lambda s: True,
          **kw):
    """Records for *seqs*, stamped t0 + seq * DT."""
    recs = [record(mask, s, ts=seconds_ts(t0 + s * DT), recent=recent(s), **kw)
            for s in seqs]
    return write(tmp_path, [file_header(mask, len(recs))] + recs)


def test_channel_location_follows_the_parser_order():
    assert channel_location(1) == (1, 0)
    assert channel_location(128) == (1, 127)
    assert channel_location(129) == (2, 0)
    assert channel_location(200) == (2, 71)
    assert channel_location(1024) == (8, 127)
    for bad in (0, 1025):
        with pytest.raises(ValueError):
            channel_location(bad)


def test_index_and_window_over_a_gap_and_an_undisciplined_stamp(tmp_path):
    # Records 100..109 were lost (a seq gap); record 7's stamp is not
    # disciplined.
    seqs = [s for s in range(300) if not 100 <= s < 110]
    f = Recording(_file(tmp_path, seqs, recent=lambda s: s != 7))
    assert f.t_first == pytest.approx(T0)
    assert f.counts_per_lsb == 1.0            # HIGH: exact counts

    # Stamp of record k (seq s) is T0 + s*DT: the bisect lands on it.
    for k, s in ((0, 0), (50, 50), (99, 99), (100, 110), (289, 299)):
        assert f.index_at(T0 + s * DT) == k
        assert f.index_at(T0 + s * DT, side="right") == k + 1
    # Between two stamps: the next record; past the end: the count.
    assert f.index_at(T0 + 50.5 * DT) == 51
    assert f.index_at(T0 + 1.0) == len(f)
    # Record 7's stamp is unusable, so it is placed with the next
    # disciplined one: any query in (t6, t8] starts at record 7.
    assert f.index_at(T0 + 7 * DT) == 7
    assert f.index_at(T0 + 8 * DT) == 7
    assert f.index_at(T0 + 8 * DT, side="right") == 9

    w = f.window(T0 + 95 * DT, T0 + 115 * DT, channel=200)   # pipe 2, col 71
    assert (w.start, w.stop) == (95, 106)
    assert w.seq_gaps == 1 and w.dropouts == 0
    assert np.allclose(w.times[~np.isnan(w.times)],
                       T0 + np.array([95, 96, 97, 98, 99, 110, 111, 112,
                                      113, 114, 115]) * DT)
    # Helper fills pipe p with 100*p + seq, Q = -I.
    expect = 200 + np.array([95, 96, 97, 98, 99, 110, 111, 112, 113, 114, 115])
    assert np.array_equal(w.samples.real, expect)
    assert np.array_equal(w.samples.imag, -expect)

    w = f.window(T0 + 5 * DT, T0 + 9 * DT, channel=1)
    assert np.isnan(w.times[2]) and not np.isnan(w.times[[0, 1, 3, 4]]).any()


def test_a_recording_across_midnight_stays_monotone(tmp_path):
    t0 = 86400.0 - 3 * DT
    f = Recording(_file(tmp_path, range(10), t0=t0))
    t = f.seconds()
    assert np.all(np.diff(t) > 0)
    assert t[3] == pytest.approx(86400.0)
    # A query in the new day (seconds of day near zero) finds the record.
    assert f.index_at(5 * DT) == 8
    assert f.window(0.0, 4 * DT, channel=1).start == 3


def test_no_disciplined_stamp_means_no_time_axis(tmp_path):
    f = Recording(_file(tmp_path, range(10), recent=lambda s: False))
    assert f.t_first is None
    with pytest.raises(ValueError, match="no disciplined"):
        f.index_at(T0)
    assert np.isnan(f.seconds()).all()
    # Samples are still reachable by index.
    assert f.channel(1, 2, 4).shape == (2,)


def test_truncation_scales_to_counts(tmp_path):
    f = Recording(_file(tmp_path, range(4), sample_trunc=0))
    assert f.sample_trunc == 0
    assert f.channel(1, 0, 1)[0] == pytest.approx((100 + 0) / 256 * (1 - 1j))


def test_empty_recording_has_no_time_axis(tmp_path):
    f = Recording(write(tmp_path, [file_header(0b1, 0)]))
    assert f.t_first is None and f.module is None
    assert f.seconds().shape == (0,)
