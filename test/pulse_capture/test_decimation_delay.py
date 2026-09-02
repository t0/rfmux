"""The decimated stream's timestamps are late by its CIC group delay;
in "both" mode the dual session pulls them back so the two streams
share the PFB clock, and the mock stamps its slow packets late the
same way.  TEMPORARY until the firmware corrects the timestamps — see
decimated_stream_delay_s in rfmux/core/transferfunctions.py.
"""
import numpy as np
import pytest

from rfmux.core.transferfunctions import (
    PFB_SAMPLING_FREQ, decimated_stream_delay_s, decimation_to_sampling,
    sampling_to_decimation)
from rfmux.pulse_capture.capture_session import (
    DualPulseCaptureSession, PulseCaptureConfig)

CIC1_S = 94.5 / PFB_SAMPLING_FREQ          # 3 stages, R=64, at the PFB rate


@pytest.mark.parametrize("dec, cic2_samples", [
    (4, 2.8125), (5, 2.90625), (6, 2.953125)])
def test_cic2_delay_is_about_2_9_output_samples(dec, cic2_samples):
    """Joshua's numbers: 3(R-1)/R output samples of CIC2, plus CIC1."""
    fs = decimation_to_sampling(dec)
    assert decimated_stream_delay_s(sampling_to_decimation(fs)) == pytest.approx(
        CIC1_S + cic2_samples / fs)


def test_dec0_is_cic1_alone():
    fs = decimation_to_sampling(0)
    assert decimated_stream_delay_s(sampling_to_decimation(fs)) == pytest.approx(CIC1_S)
    assert decimated_stream_delay_s(sampling_to_decimation(fs)) * fs == pytest.approx(94.5 / 64)


def test_rate_to_stage_round_trips_and_rounds():
    for dec in range(7):
        assert sampling_to_decimation(decimation_to_sampling(dec)) == dec
    assert sampling_to_decimation(1000.0) == 5      # nearest stage
    with pytest.raises(ValueError):
        sampling_to_decimation(0.0)


def _dual(slow_rate, **kw):
    cfg = PulseCaptureConfig(threshold_sigma=5.0, end_sigma=1.5,
                             max_pulse_ms=20.0, noise_train_ms=200.0)
    pairs = []
    d = DualPulseCaptureSession(
        channels=[1], slow_rate=slow_rate, fast_rate=2.44e6, config=cfg,
        hdf5_path=None, on_pair=pairs.append, **kw)
    d.start()
    return d, cfg, pairs


def _feed_both(d, cfg, slow_rate, slow_late_by):
    """Train both streams, then one pulse at true time T on both, with
    the slow samples stamped *slow_late_by* seconds late, as the board
    does.  Training goes first on both: triggering is held until both
    streams have trained, so a pulse fed before the other stream's
    training would be missed."""
    rng = np.random.default_rng(3)
    T0 = 43000.0
    T = T0 + 1.2
    rates = (("slow", slow_rate), ("fast", 2.44e6))
    feeds = {"slow": d.feed_slow_block, "fast": d.feed_fast_block}
    late = {"slow": slow_late_by, "fast": 0.0}
    n_tr = {}
    for stream, fs in rates:
        n = cfg.noise_samples(fs) + 50
        n_tr[stream] = n
        t = T0 + np.arange(n) / fs
        feeds[stream](1, rng.normal(0, 1, n), rng.normal(0, 1, n),
                      t + late[stream])
    for stream, fs in rates:
        t = T0 + np.arange(n_tr[stream], int(2.0 * fs)) / fs
        y = rng.normal(0, 1, len(t))
        k = int(np.argmax(t >= T))
        y[k:k + int(0.004 * fs) + 2] += 40.0
        feeds[stream](1, y, rng.normal(0, 1, len(t)), t + late[stream])
    d.stop()
    return T


@pytest.mark.parametrize("dec", [4, 6])
def test_pulse_at_one_true_time_pairs_with_no_skew(dec):
    fs = decimation_to_sampling(dec)
    late = decimated_stream_delay_s(sampling_to_decimation(fs))
    d, cfg, pairs = _dual(fs)
    T = _feed_both(d, cfg, fs, slow_late_by=late)
    assert d.slow_time_offset_s == pytest.approx(-late)
    assert pairs, "the pulse did not pair"
    # Residual skew is within one slow sample of zero: the slow trigger
    # can only land on a sample.
    assert abs(pairs[0]["trigger_offset"]) < 1.0 / fs
    assert abs(pairs[0]["slow_summary"]["trigger_time"] - T) < 1.0 / fs


def test_without_the_shift_the_slow_trigger_is_late_by_the_delay():
    """Sign pin: the board stamps slow late, so an uncorrected capture
    shows slow − fast ≈ +delay."""
    fs = decimation_to_sampling(6)
    late = decimated_stream_delay_s(sampling_to_decimation(fs))
    d, cfg, pairs = _dual(fs, slow_time_offset_s=0.0)
    _feed_both(d, cfg, fs, slow_late_by=late)
    assert d.slow_time_offset_s == 0.0
    assert pairs
    assert pairs[0]["trigger_offset"] == pytest.approx(late, abs=1.0 / fs)


def test_shift_reaches_ring_record_and_single_sample_path():
    fs = decimation_to_sampling(6)
    d, cfg, _ = _dual(fs)
    off = d.slow_time_offset_s
    rng = np.random.default_rng(1)
    for feed, rate in ((d.feed_slow_block, fs), (d.feed_fast_block, 2.44e6)):
        n = cfg.noise_samples(rate) + 10
        feed(1, rng.normal(0, 1, n), rng.normal(0, 1, n), np.arange(n) / rate)
    d.feed_slow(1, 0.0, 0.0, 100.0)
    d.feed_fast_block(1, np.zeros(4), np.zeros(4), 100.0 + np.arange(4) / 2.44e6)
    assert d.slow.pcap.buf[1]["ts"].data()[-1] == pytest.approx(100.0 + off)
    assert d.fast.pcap.buf[1]["ts"].data()[-4] == pytest.approx(100.0)
    d.feed_slow_block(1, np.zeros(3), np.zeros(3),
                      np.array([101.0, np.nan, 102.0]))
    ts = d.slow.pcap.buf[1]["ts"].data()
    # The NaN-stamped sample is dropped by the session, as before.
    assert ts[-1] == pytest.approx(102.0 + off)
    assert ts[-2] == pytest.approx(101.0 + off)
    d.stop()


def test_the_file_says_how_much_was_shifted(tmp_path):
    from rfmux.pulse_capture.hdf5 import PulseHDF5Reader
    fs = decimation_to_sampling(6)
    cfg = PulseCaptureConfig()
    for name, kw in (("shifted.h5", {}), ("raw.h5", {"slow_time_offset_s": 0.0})):
        d = DualPulseCaptureSession(channels=[1], slow_rate=fs, config=cfg,
                                    hdf5_path=str(tmp_path / name), **kw)
        d.start(); d.stop()
        meta = PulseHDF5Reader(str(tmp_path / name)).metadata
        assert meta["slow_time_offset_s"] == pytest.approx(
            -decimated_stream_delay_s(sampling_to_decimation(fs)) if not kw else 0.0)
        assert d.stats()["slow_time_offset_s"] == meta["slow_time_offset_s"]
