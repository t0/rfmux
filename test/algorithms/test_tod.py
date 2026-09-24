"""The time-ordered data file: the dirfile and the fastrx recording of
one event repacked into the units a capture of the same channel stores,
on the PFB clock, with the capture file's metadata beside them."""

import pathlib

import numpy as np
import pytest
import h5py

from rfmux.core.transferfunctions import (
    PFB_SAMPLING_FREQ, VOLTS_PER_ROC, decimated_stream_delay_s)
from rfmux.pulse_capture.analysis import storage_transform
from rfmux.pulse_capture.hdf5 import PulseHDF5Reader
from rfmux.algorithms.measurement.tod import merge_tod, write_tod
from rfmux.streamer import day_epoch
from test.pulse_capture.test_overlay import (
    AMP, CHANNEL, CHANNELS, FS, LATE, _capture, _dirfile, _recording_file,
    _shape)

CAL = complex(3.0e6, -4.0e6)                      # Hz per volt, rotated
TUNING = {CHANNEL: {"bias_channel": CHANNEL, "df_calibration": CAL}}


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    return _recording_file(tmp_path_factory.mktemp("fastrx"))


@pytest.fixture(scope="module")
def dirfile(tmp_path_factory):
    return _dirfile(tmp_path_factory.mktemp("dirfile"))


def _tod(tmp_path, **kw):
    kw.setdefault("tuning", TUNING)
    return write_tod(tmp_path / "tod.h5", [CHANNEL], 1, **kw)


def _trace(f, stream, channel=CHANNEL, module=None):
    where = f"tod/{stream}" + (f"/module_{module}" if module else "")
    grp = f[where]
    ch = grp[f"channel_{channel}"]
    return grp["time"][()], ch["I"][()] + 1j * ch["Q"][()]


def test_both_streams_hold_the_event_in_the_captures_units_on_one_clock(
        tmp_path, recording, dirfile):
    """A sample of either stream is the event's value at its stamp,
    times the factor a df-basis capture stores that channel with; the
    dirfile's late stamps come out corrected, so the two agree in time."""
    factor, units = storage_transform(CAL, "df")
    assert units == "Hz"
    path = _tod(tmp_path, fastrx=recording, dirfile=dirfile)
    with h5py.File(path, "r") as f:
        # The wire's int16 rounds to a count; the parser's raw counts
        # are in 1/256 and its stamp is truncated to 6.4 ns, which on
        # the decay's steepest slope is 0.005 counts.
        for stream, tol in (("fast", 1.0), ("slow", 0.01)):
            t, z = _trace(f, stream)
            ok = np.isfinite(t)
            assert ok.all() and np.all(np.diff(t) > 0)
            expected = _shape(t) * factor
            assert np.abs(z - expected).max() <= tol * abs(factor) * 1.01, stream
            # The channel stream has a record at the event; the slow
            # stream's nearest frame is later, and reads the decay there.
            peak = AMP if stream == "fast" else _shape(t).max()
            assert np.abs(z).max() == pytest.approx(peak * abs(factor), rel=2e-3)
            assert f[f"tod/{stream}/channel_{CHANNEL}"].attrs["stored_units"] == "Hz"
            assert complex(f[f"tod/{stream}/channel_{CHANNEL}/tuning"]
                           .attrs["df_calibration"]) == CAL
        assert f["tod/fast/seq"].shape == f["tod/fast/time"].shape
        assert f["tod/fast/pipe_snapshot"].shape == f["tod/fast/time"].shape


def test_the_metadata_is_the_capture_files(tmp_path, recording, dirfile):
    path = _tod(tmp_path, fastrx=recording, dirfile=dirfile)
    with PulseHDF5Reader(path) as r:
        m = r.metadata
        assert r.channels == [CHANNEL] and int(m["module"]) == 1
        assert m["trigger_basis"] == "df" and m["stored_units"] == "Hz"
        assert m["volts_per_count"] == VOLTS_PER_ROC
        assert m["sample_rate_slow"] == pytest.approx(FS)
        assert m["sample_rate_fast"] == PFB_SAMPLING_FREQ
        assert m["slow_time_offset_s"] == pytest.approx(-LATE)
        assert list(m["fast_channels"]) == [CHANNEL]
        assert m["time_origin_epoch"] == day_epoch(26, 245)
        assert m["time_origin_utc"].startswith("2026-09-02")
        assert r.stored_units(CHANNEL) == "Hz"     # read from the tod/ groups


def test_an_uncalibrated_channel_is_stored_in_volts(tmp_path, recording):
    path = _tod(tmp_path, fastrx=recording, tuning=None)
    with h5py.File(path, "r") as f:
        _, z = _trace(f, "fast")
        assert np.abs(z).max() == pytest.approx(AMP * VOLTS_PER_ROC, rel=2e-3)
        assert f[f"tod/fast/channel_{CHANNEL}"].attrs["stored_units"] == "V"
        assert f["metadata"].attrs["stored_units"] == "V"
        assert "tod/slow" not in f
        assert "sample_rate_slow" not in f["metadata"].attrs


def test_blocks_join_seamlessly(tmp_path, recording, dirfile):
    whole = _tod(tmp_path, fastrx=recording, dirfile=dirfile)
    blocked = write_tod(tmp_path / "blocked.h5", [CHANNEL], 1,
                        fastrx=recording, dirfile=dirfile, tuning=TUNING,
                        block=7)
    with h5py.File(whole, "r") as a, h5py.File(blocked, "r") as b:
        for stream in ("slow", "fast"):
            ta, za = _trace(a, stream)
            tb, zb = _trace(b, stream)
            assert np.array_equal(ta, tb) and np.array_equal(za, zb)


def test_a_run_across_modules_nests_each_modules_records(tmp_path,
                                                        tmp_path_factory):
    rec = _recording_file(tmp_path_factory.mktemp("two"), modules=(1, 2))
    keys = [(1, CHANNEL), (2, CHANNEL)]
    path = write_tod(tmp_path / "tod.h5", keys, None, fastrx=rec,
                     tuning={(1, CHANNEL): TUNING[CHANNEL]}, trigger_basis="df")
    with h5py.File(path, "r") as f:
        t1, z1 = _trace(f, "fast", module=1)
        t2, z2 = _trace(f, "fast", module=2)
        assert len(t1) == len(t2) == len(f["tod/fast/module_1/seq"])
        assert np.array_equal(t1, t2)                 # one stamp per module
        assert np.abs(z1).max() > 0 and not np.abs(z2).any()
        assert f["tod/fast/module_1/channel_200"].attrs["stored_units"] == "Hz"
        assert f["tod/fast/module_2/channel_200"].attrs["stored_units"] == "V"
        assert f["metadata"].attrs["stored_units"] == "mixed"
        assert "module" not in f["metadata"].attrs
    with PulseHDF5Reader(path) as r:
        assert r.channels == keys and r.modules == [1, 2]


def test_a_channel_the_recording_lacks_has_no_fast_group(tmp_path, recording):
    path = write_tod(tmp_path / "tod.h5", [CHANNEL, CHANNELS + 1], 1,
                     fastrx=recording)
    with h5py.File(path, "r") as f:
        assert f"tod/fast/channel_{CHANNEL}" in f
        assert f"tod/fast/channel_{CHANNELS + 1}" not in f
        assert list(f["metadata"].attrs["fast_channels"]) == [CHANNEL]
        assert list(f["metadata"].attrs["channels"]) == [CHANNEL, CHANNELS + 1]


def test_nothing_to_repack_is_refused_and_a_failure_leaves_no_file(tmp_path):
    with pytest.raises(ValueError, match="nothing to repack"):
        write_tod(tmp_path / "tod.h5", [CHANNEL], 1)
    with pytest.raises(Exception):
        write_tod(tmp_path / "tod.h5", [CHANNEL], 1, fastrx=tmp_path / "none")
    assert list(tmp_path.iterdir()) == []


def test_merge_copies_the_streams_into_the_pulse_file(tmp_path, recording,
                                                      dirfile):
    pulse = _capture(tmp_path, tuning=TUNING, trigger_basis="df")
    with PulseHDF5Reader(pulse) as r:
        before = (r.pulse_count(CHANNEL), dict(r.metadata))
    tod = _tod(tmp_path, fastrx=recording, dirfile=dirfile)
    assert merge_tod(pulse, tod) == pathlib.Path(pulse)
    with h5py.File(pulse, "r") as f, h5py.File(tod, "r") as src:
        for stream in ("slow", "fast"):
            t, z = _trace(f, stream)
            ts, zs = _trace(src, stream)
            assert np.array_equal(t, ts) and np.array_equal(z, zs)
        m = f["metadata"].attrs
        assert m["sample_rate_fast"] == PFB_SAMPLING_FREQ  # the capture lacked it
        for key, value in before[1].items():             # and kept its own
            assert np.array_equal(m[key], value), key
    with PulseHDF5Reader(pulse) as r:
        assert r.pulse_count(CHANNEL) == before[0]
        assert r.get_pulse(CHANNEL, 1)["Amp_I"].size
    assert tod.exists()
    with pytest.raises(ValueError, match="already holds tod/"):
        merge_tod(pulse, tod)
    assert not list(tmp_path.glob("*.merging"))


def test_merge_refuses_a_tod_in_other_units_than_the_pulses(tmp_path,
                                                             recording):
    """A merged file's pulses and streams compare directly, so a TOD
    written in the quadratures does not join a capture in the frequency
    basis, and the pulse file is left as it was."""
    pulse = _capture(tmp_path, tuning=TUNING, trigger_basis="df")
    with h5py.File(pulse, "r") as f:
        assert f[f"channel_{CHANNEL}"].attrs["stored_units"] == "Hz"
    tod = _tod(tmp_path, fastrx=recording, trigger_basis="iq")
    with h5py.File(tod, "r") as f:
        assert f[f"tod/fast/channel_{CHANNEL}"].attrs["stored_units"] == "V"
    with pytest.raises(ValueError, match="iq basis"):
        merge_tod(pulse, tod)
    with h5py.File(pulse, "r") as f:
        assert "tod" not in f
    assert not list(tmp_path.glob("*.merging"))
    # The same basis with a channel stored differently is refused too:
    # a df capture whose channel had no calibration stays in volts.
    (tmp_path / "v").mkdir()
    pulse_v = _capture(tmp_path / "v", tuning=None, trigger_basis="df")
    with pytest.raises(ValueError, match="stores it in V"):
        merge_tod(pulse_v, _tod(tmp_path, fastrx=recording))


def test_the_reader_finds_a_tod_files_tuning_units_and_streams(
        tmp_path, recording, dirfile):
    """Review reads a file of time-ordered data alone through the same
    reader: the tuning and units come from the tod/ channel groups, the
    streams are listed, and there are no pulses."""
    path = _tod(tmp_path, fastrx=recording, dirfile=dirfile)
    with PulseHDF5Reader(path) as r:
        assert not r.has_pulses and r.tod_streams == ["slow", "fast"]
        assert r.tuning(CHANNEL)["df_calibration"] == CAL
        assert r.df_calibration(CHANNEL) == CAL
        assert r.stored_units(CHANNEL) == "Hz"
        assert r.pulse_count(CHANNEL) == 0 and r.event_count == 0
        info = r.tod_info()
        assert info["fast"]["channels"] == [CHANNEL]
        with h5py.File(path) as f:
            assert info["fast"]["samples"] == f["tod/fast/time"].shape[0]
            assert info["slow"]["samples"] == f["tod/slow/time"].shape[0]
    # A pulse file's own groups win over the tod/ copies once merged.
    pulse = _capture(tmp_path, tuning=TUNING, trigger_basis="df")
    merge_tod(pulse, path)
    with PulseHDF5Reader(pulse) as r:
        assert r.has_pulses and r.tod_streams == ["slow", "fast"]
        assert r.pulse_count(CHANNEL) > 0


def test_merge_to_another_path_leaves_the_source(tmp_path, recording):
    pulse = _capture(tmp_path, tuning=TUNING, trigger_basis="df")
    tod = _tod(tmp_path, fastrx=recording)
    out = merge_tod(pulse, tod, tmp_path / "both.h5")
    with h5py.File(pulse, "r") as f:
        assert "tod" not in f
    with h5py.File(out, "r") as f:
        assert f"tod/fast/channel_{CHANNEL}/I" in f
