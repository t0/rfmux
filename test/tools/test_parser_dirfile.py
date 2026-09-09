"""The parser's readout dirfile: the timebase reads on the PFB clock.

The board stamps the decimated stream late by its CIC group delay; the
dirfile takes it out per frame from the packet's decimation stage, and
keeps the raw stamp beside it.  Needs pygetdata (libgetdata).
"""

import numpy as np
import pytest

gd = pytest.importorskip("pygetdata")

pytestmark = pytest.mark.portable

from rfmux.core.transferfunctions import decimated_stream_delay_s
from rfmux.streamer import ReadoutPacket, SS_PER_SECOND
from rfmux.tools.parser import (BoardStats, ModuleStats,
                                setup_dirfile_for_module, write_dec_stage)


def test_timebase_takes_out_the_stage_delay_and_keeps_the_raw_stamp(tmp_path):
    path = str(tmp_path / "board")
    board = BoardStats()
    board.dirfile = gd.dirfile(path, gd.CREAT | gd.RDWR | gd.EXCL)
    mod = ModuleStats()
    setup_dirfile_for_module(board, mod, 0, [range(0, 2)])
    df, fields = board.dirfile, mod.dirfile_fields

    # Stage 6, then stage 0 with the short-packet flag (bit 3) set.
    stamps = [(6, 43000, 0), (0 | 0x8, 43000, SS_PER_SECOND // 2)]
    for frame, (stage, sbs, ss) in enumerate(stamps):
        df.putdata(fields["ts_sbs"], np.array([sbs], dtype=np.int32),
                   first_frame=frame)
        df.putdata(fields["ts_ss"], np.array([ss], dtype=np.int32),
                   first_frame=frame)
        pkt = ReadoutPacket(magic=0x5344494b, version=6, serial=42,
                            num_modules=1, flags=0, fir_stage=stage,
                            module=0, seq=frame)
        write_dec_stage(df, fields, frame, pkt)
    df.close()

    # Read back as a consumer would: a fresh read-only handle.
    df = gd.dirfile(path, gd.RDONLY)
    timebase = df.getdata("m01_timebase", gd.FLOAT64, num_frames=2)
    raw = (df.getdata("m01_ts_sbs", gd.FLOAT64, num_frames=2)
           + df.getdata("m01_ts_ss", gd.FLOAT64, num_frames=2) / SS_PER_SECOND)
    stage = df.getdata("m01_dec_stage", gd.UINT8, num_frames=2)

    assert list(stage) == [6, 0]
    assert raw == pytest.approx([43000.0, 43000.5])
    assert timebase == pytest.approx(
        raw - np.array([decimated_stream_delay_s(6),
                        decimated_stream_delay_s(0)]))
