"""Viewing a time-ordered data file without reading it whole: the
overview write_tod leaves, a window found by its stamps, and a view of
at most a few hundred bins that keeps every extreme."""

import h5py
import numpy as np
import pytest

from rfmux.algorithms.measurement.tod import (index_at, tod_extent,
                                              tod_window, write_tod)
from rfmux.core.transferfunctions import PFB_SAMPLING_FREQ, VOLTS_PER_ROC
from rfmux.pulse_capture.recording_file import write_recording

N = 20_000
T0 = 43000.0
OVERVIEW = 64
CHANNELS = 3


def _recording(tmp_path, nan_at=(), modules=(1,)):
    """Channel 2 carries a spike train the overview must keep; stamps at
    the channel-stream rate, NaN (undisciplined) at *nan_at*."""
    rng = np.random.default_rng(4)
    iq = rng.integers(-200, 200, (N, CHANNELS, 2)).astype(np.int16)
    iq[::997, 1, 0] = 30000                  # one-sample spikes on I
    iq[::1009, 1, 1] = -30000                # and on Q
    seconds = T0 + np.arange(N) / PFB_SAMPLING_FREQ
    seconds[list(nan_at)] = np.nan
    module = np.resize(np.asarray(modules), N)
    return write_recording(tmp_path / "run.fastrx", seconds, iq,
                           module=module), iq, seconds


@pytest.fixture
def tod(tmp_path):
    """The file, the samples, and its stamps as stored (the wire rounds
    them to 6.4 ns); records 5000-5099 are undisciplined, longer than
    any probe distance."""
    rec, iq, _ = _recording(tmp_path, nan_at=range(5000, 5100))
    path = write_tod(tmp_path / "tod.h5", [1, 2, 3], 1, fastrx=rec,
                     trigger_basis="iq", block=1000, overview=OVERVIEW)
    with h5py.File(path, "r") as f:
        seconds = f["tod/fast/time"][()]
    return path, iq, seconds


def test_the_overview_is_the_min_and_max_of_each_bin_across_blocks(tod):
    """Blocks of 1000 records do not divide into bins of 64: a bin a
    block leaves partial is completed by the next, and the last bin is
    the short remainder."""
    path, iq, seconds = tod
    with h5py.File(path, "r") as f:
        g = f["tod/fast"]
        assert g.attrs["overview_samples"] == OVERVIEW
        ov = g["channel_2/overview"][()]
        tov = g["time_overview"][()]
    i = iq[:, 1, 0] * VOLTS_PER_ROC
    q = iq[:, 1, 1] * VOLTS_PER_ROC
    starts = np.arange(0, N, OVERVIEW)
    assert len(ov) == len(starts) == -(-N // OVERVIEW)
    np.testing.assert_allclose(ov[:, 0], np.minimum.reduceat(i, starts), rtol=1e-6)
    np.testing.assert_allclose(ov[:, 1], np.maximum.reduceat(i, starts), rtol=1e-6)
    np.testing.assert_allclose(ov[:, 2], np.minimum.reduceat(q, starts), rtol=1e-6)
    np.testing.assert_allclose(ov[:, 3], np.maximum.reduceat(q, starts), rtol=1e-6)
    np.testing.assert_allclose(tov[:, 0], np.fmin.reduceat(seconds, starts))
    np.testing.assert_allclose(tov[:, 1], np.fmax.reduceat(seconds, starts))


def test_a_wide_view_comes_from_the_overview_and_keeps_every_spike(tod):
    path, iq, _ = tod
    with h5py.File(path, "r") as f:
        t0, t1 = tod_extent(f, "fast", 2)
        view = tod_window(f, "fast", 2, t0, t1, bins=100)
    assert view["kind"] == "envelope" and view["source"] == "overview"
    assert view["samples"] == N and len(view["i_min"]) <= 100
    assert view["i_max"].max() == pytest.approx(30000 * VOLTS_PER_ROC, rel=1e-6)
    assert view["q_min"].min() == pytest.approx(-30000 * VOLTS_PER_ROC, rel=1e-6)
    # Every spike lands in a bin whose maximum shows it.
    spikes = T0 + np.arange(0, N, 997) / PFB_SAMPLING_FREQ
    for t in spikes:
        hit = (view["t_first"] <= t) & (t <= view["t_last"])
        assert view["i_max"][hit].max() > 0.9 * 30000 * VOLTS_PER_ROC


def test_a_middle_view_reduces_the_samples_and_a_narrow_one_is_raw(tod):
    path, iq, seconds = tod
    i = iq[:, 1, 0] * VOLTS_PER_ROC
    with h5py.File(path, "r") as f:
        # 3000 samples in 100 bins: fewer than 100 overview bins span it.
        mid = tod_window(f, "fast", 2, seconds[10000], seconds[12999], bins=100)
        raw = tod_window(f, "fast", 2, seconds[997 * 12 - 20],
                         seconds[997 * 12 + 20], bins=100)
    assert mid["kind"] == "envelope" and mid["source"] == "samples"
    assert mid["samples"] == 3000 and len(mid["i_min"]) == 100
    assert mid["i_max"].max() == pytest.approx(i[10000:13000].max(), rel=1e-6)
    assert mid["i_min"].min() == pytest.approx(i[10000:13000].min(), rel=1e-6)
    assert raw["kind"] == "raw" and raw["samples"] == 41
    np.testing.assert_allclose(raw["I"], i[997 * 12 - 20:997 * 12 + 21], rtol=1e-6)
    np.testing.assert_array_equal(raw["time"], seconds[997 * 12 - 20:997 * 12 + 21])


def test_a_window_is_found_past_undisciplined_stamps(tod):
    """Records 5000-5099 have no disciplined stamp: a window starting
    among them opens at the first of them, as the recording's own
    index places them, and one after them at its sample."""
    path, _, seconds = tod
    assert np.isnan(seconds[5000:5100]).all()
    with h5py.File(path, "r") as f:
        at = lambda t, side="left": index_at(f, "fast", 2, t, side)
        assert at(seconds[4999] + 1e-9) == 5000
        assert at(seconds[5100]) == 5000
        assert at(seconds[7000]) == 7000
        assert at(seconds[7000], side="right") == 7001
        assert at(T0 - 1.0) == 0
        assert at(seconds[-1] + 1.0) == N
        # A window past the undisciplined stretch is the window asked for.
        view = tod_window(f, "fast", 2, seconds[10000], seconds[12999])
        assert view["samples"] == 3000


def test_an_undisciplined_stretch_over_whole_bins_is_placed_with_the_next(
        tmp_path):
    rec, _, _ = _recording(tmp_path, nan_at=range(3000, 3330))
    path = write_tod(tmp_path / "tod.h5", [2], 1, fastrx=rec,
                     trigger_basis="iq", overview=OVERVIEW)
    with h5py.File(path, "r") as f:
        s = f["tod/fast/time"][()]
        assert index_at(f, "fast", 2, s[2999] + 1e-9) == 3000
        assert index_at(f, "fast", 2, s[3330]) == 3000
        assert index_at(f, "fast", 2, s[3331]) == 3331
        # Everywhere: the search over the whole axis, NaN taking the
        # next finite stamp.
        axis, following = s.copy(), np.inf
        for i in range(len(s) - 1, -1, -1):
            if np.isnan(s[i]):
                axis[i] = following
            else:
                following = s[i]
        for t in np.random.default_rng(2).uniform(s[0] - 1e-5, s[-1] + 1e-5, 300):
            for side in ("left", "right"):
                assert index_at(f, "fast", 2, t, side) == \
                    np.searchsorted(axis, t, side=side)


def test_a_run_across_modules_views_each_modules_records(tmp_path):
    rec, iq, seconds = _recording(tmp_path, modules=(1, 2))
    path = write_tod(tmp_path / "tod.h5", [(1, 2), (2, 2)], None,
                     fastrx=rec, trigger_basis="iq", overview=OVERVIEW)
    with h5py.File(path, "r") as f:
        assert "time_overview" in f["tod/fast/module_2"]
        t0, t1 = tod_extent(f, "fast", (2, 2))
        view = tod_window(f, "fast", (2, 2), t0, t1, bins=50)
    assert view["samples"] == N // 2
    # Module 2 holds the odd records; spikes sit on records 0, 997, ...
    odd_spikes = iq[1::2, 1, 0].max() * VOLTS_PER_ROC
    assert view["i_max"].max() == pytest.approx(odd_spikes, rel=1e-6)


def test_a_view_factor_converts_samples_exactly_and_bounds_the_overview(tod):
    """(I + jQ) * factor, as Periscope's units choice converts: the
    samples converted before they are reduced; the overview, which holds
    each stored axis's extremes, scaled exactly and, under a rotation,
    bounded so that every converted sample lies inside its bin."""
    path, iq, seconds = tod
    z = (iq[:, 1, 0] + 1j * iq[:, 1, 1]) * VOLTS_PER_ROC
    rot = 3e6 * np.exp(1j * 0.9)
    with h5py.File(path, "r") as f:
        raw = tod_window(f, "fast", 2, seconds[100], seconds[140], factor=rot)
        mid = tod_window(f, "fast", 2, seconds[10000], seconds[12999],
                         bins=100, factor=rot)
        t0, t1 = tod_extent(f, "fast", 2)
        scaled = tod_window(f, "fast", 2, t0, t1, bins=100, factor=2.0)
        plain = tod_window(f, "fast", 2, t0, t1, bins=100)
        wide = tod_window(f, "fast", 2, t0, t1, bins=100, factor=rot)
    np.testing.assert_allclose(raw["I"] + 1j * raw["Q"], z[100:141] * rot,
                               rtol=1e-5)
    conv = z[10000:13000] * rot
    assert mid["i_max"].max() == pytest.approx(conv.real.max(), rel=1e-5)
    assert mid["q_min"].min() == pytest.approx(conv.imag.min(), rel=1e-5)
    assert not scaled["bounds"]
    np.testing.assert_allclose(scaled["i_max"], 2 * plain["i_max"])
    assert wide["bounds"] and wide["source"] == "overview"
    conv = z * rot
    t = np.where(np.isnan(seconds), -np.inf, seconds)
    for k in range(len(wide["t_first"])):
        inside = (t >= wide["t_first"][k]) & (t <= wide["t_last"][k])
        assert conv.real[inside].max() <= wide["i_max"][k] * (1 + 1e-6) + 1e-9
        assert conv.real[inside].min() >= wide["i_min"][k] * (1 + 1e-6) - 1e-9
        assert conv.imag[inside].max() <= wide["q_max"][k] * (1 + 1e-6) + 1e-9
        assert conv.imag[inside].min() >= wide["q_min"][k] * (1 + 1e-6) - 1e-9


def test_the_slow_stream_has_its_overview_too(tmp_path):
    from test.pulse_capture.test_overlay import CHANNEL, _dirfile
    path = write_tod(tmp_path / "tod.h5", [CHANNEL], 1,
                     dirfile=_dirfile(tmp_path), overview=8)
    with h5py.File(path, "r") as f:
        g = f["tod/slow"]
        i = g[f"channel_{CHANNEL}/I"][()]
        ov = g[f"channel_{CHANNEL}/overview"][()]
        assert g.attrs["overview_samples"] == 8
        assert len(ov) == -(-len(i) // 8)
        assert ov[:, 1].max() == i.max()
        t0, t1 = tod_extent(f, "slow", CHANNEL)
        assert (t0, t1) == (g["time"][0], g["time"][-1])
