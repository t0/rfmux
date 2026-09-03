"""
Packets carry a clock; records carry it decoded.

The packet Timestamp names a two-digit year and a day of year on top
of the seconds-of-day every other consumer uses.  The session learns
the day from the first stamped packet and each record gets the trigger
instant as seconds since 1970 and as an ISO string, in the file and in
the summary the viewer shows.
"""

import h5py
import numpy as np

from rfmux import streamer
from rfmux.pulse_capture.capture_session import (
    DualPulseCaptureSession, PulseCaptureConfig, PulseCaptureSession)
from rfmux.pulse_capture.hdf5 import PulseHDF5Reader

from test.packet_helpers import stamp
from test.pulse_capture.ingest_helpers import FS, packets as _packets


def test_the_day_decodes_from_year_and_day_of_year():
    ts = stamp(58445, y=26, d=245)           # 2026-09-02
    day = streamer.ts_day_epoch(ts)
    assert streamer.epoch_to_utc(day) == "2026-09-02T00:00:00.000000Z"
    whole = day + streamer.ts_to_seconds(ts)
    assert streamer.epoch_to_utc(whole) == "2026-09-02T16:14:05.000000Z"
    assert streamer.ts_day_epoch(stamp(0, y=26, d=245, recent=False)) is None


def test_records_carry_the_decoded_trigger_time(tmp_path):
    path = tmp_path / "t.h5"
    got = []
    s = PulseCaptureSession(channels=[1], sample_rate=FS, noise_samples=400,
                            hdf5_path=path,
                            on_pulse=lambda ch, idx, summ, data: got.append(summ))
    s.start()
    day = streamer.ts_day_epoch(stamp(0, y=26, d=245))
    s.set_time_origin(day)
    s.set_time_origin(day + 86400)                 # the first wins
    rng = np.random.default_rng(1)
    t0 = 16 * 3600 + 14 * 60 + 5.0                 # seconds of day
    for values, ts in _packets((1,), 1200, rng, pulse_starts=(800,)):
        s.feed_sample(1, float(values[0].real), float(values[0].imag), t0 + ts)
    s.stop()
    assert got, "the fixture should trigger"
    summ = got[0]
    assert abs(summ["trigger_epoch"] - (day + summ["trigger_time"])) < 1e-6
    assert summ["trigger_utc"].startswith("2026-09-02T16:14:05.")
    with h5py.File(path, "r") as f:
        meta = f["metadata"].attrs
        assert meta["time_origin_utc"] == "2026-09-02T00:00:00.000000Z"
        assert meta["time_origin_epoch"] == day
    with PulseHDF5Reader(path) as r:
        rec = r.get_pulse(1, 1)
    assert rec["trigger_utc"] == summ["trigger_utc"]


def test_without_a_day_records_have_no_calendar_time():
    got = []
    s = PulseCaptureSession(channels=[1], sample_rate=FS, noise_samples=400,
                            hdf5_path=None,
                            on_pulse=lambda ch, idx, summ, data: got.append(summ))
    s.start()
    rng = np.random.default_rng(1)
    for values, ts in _packets((1,), 1200, rng, pulse_starts=(800,)):
        s.feed_sample(1, float(values[0].real), float(values[0].imag), ts)
    s.stop()
    assert got and "trigger_utc" not in got[0]


def test_a_dual_capture_file_records_the_day(tmp_path):
    path = tmp_path / "d.h5"
    d = DualPulseCaptureSession(
        channels=[1], config=PulseCaptureConfig(), slow_rate=FS,
        fast_rate=1e5, hdf5_path=path)
    d.start()
    day = streamer.ts_day_epoch(stamp(0, y=26, d=245))
    d.set_time_origin(day)
    d.stop()
    with h5py.File(path, "r") as f:
        assert f["metadata"].attrs["time_origin_utc"] == \
            "2026-09-02T00:00:00.000000Z"
    assert d.slow.time_origin_epoch == day
    assert d.stats()["time_origin_epoch"] == day


def test_a_pair_reads_back_with_its_windows(tmp_path):
    """The live viewer fetches an evicted pair from the file it is
    writing; what comes back is what the session emitted, windows and
    union bounds included."""
    from rfmux.pulse_capture.hdf5 import DualPulseHDF5Writer
    w = DualPulseHDF5Writer(tmp_path / "p.h5", [1],
                            capture_params={"streamer_mode": "both"})
    t = np.linspace(1.0, 1.001, 11)
    pair = {"pair_idx": 1, "channel": 1, "slow_idx": 2, "fast_idx": None,
            "time_offset": None, "window": (0.9995, 1.0015),
            "slow_tod": {"Amp_I": t * 0, "Amp_Q": t * 0 + 1, "Time": t}}
    w.append_match(1, pair)
    back = w.read_match(1, 1)
    assert back["slow_idx"] == 2 and back["fast_idx"] is None
    assert back["window"] == (0.9995, 1.0015)
    assert list(back["slow_tod"]["Time"]) == list(t)
    assert "fast_tod" not in back
    assert w.read_match(1, 2) is None
    w.finalize()
