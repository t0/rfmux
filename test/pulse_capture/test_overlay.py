"""Overlaying a fastrx recording on a capture file by IRIG time.

The board stamps the slow stream late by its CIC delay and the channel
stream on time.  A capture written by the session has the delay taken
out, so its pulse window picks the fastrx samples of the same event;
an older file without the record of that shift gets it from its slow
rate.  The fastrx files are built byte-by-byte; their stamp spacing is
arbitrary because the index reads stamps, not a rate.
"""

import numpy as np
import pytest

pytest.importorskip(
    "rfmux.fastrx", reason="this rfmux build does not include fastrx"
)

from rfmux.core.transferfunctions import (
    VOLTS_PER_ROC, decimated_stream_delay_s, decimation_to_sampling,
    sampling_to_decimation)
from rfmux.pulse_capture.capture_session import (
    PulseCaptureConfig, PulseCaptureSession)
from rfmux.pulse_capture.hdf5 import PulseHDF5Reader, PulseHDF5Writer
from rfmux.pulse_capture.overlay import (
    Recording, correlation_lag_s, counts_to_stored, merge_fastrx,
    pulse_overlay, slow_shift_s)
from test.test_fastrx_file import file_header, record, seconds_ts, write

FS = decimation_to_sampling(6)                    # 596 Hz slow stream
LATE = decimated_stream_delay_s(sampling_to_decimation(FS))
T0 = 43000.0
T = T0 + 2.0                                      # the event, PFB clock
AMP = 4000.0                                      # counts
TAU = 0.005
CHANNELS = 256                                    # per record
CHANNEL = 200                                     # pipe 2, column 71


def _shape(t):
    return np.where(t >= T, AMP * np.exp(-(t - T) / TAU), 0.0)


def _recording_file(tmp_path, spacing=20e-6, span=(-0.01, 0.04)):
    """The event as the channel stream would carry it, stamped on time,
    on CHANNEL with the other channels quiet."""
    t = T + np.arange(span[0], span[1], spacing)
    recs = []
    for i, ti in enumerate(t):
        block = np.zeros((128, 2), dtype=np.int16)
        block[71, 0] = int(round(float(_shape(ti))))
        recs.append(record(CHANNELS, i, ts=seconds_ts(ti), recent=True,
                           sample_trunc=0, iq={2: block}))
    return write(tmp_path, [file_header(CHANNELS, len(recs))] + recs)


def _recording(tmp_path, spacing=20e-6, span=(-0.01, 0.04)):
    return Recording(_recording_file(tmp_path, spacing, span))


def _capture(tmp_path):
    """A slow-only capture of the event, stamps fed late as the board
    stamps them."""
    path = str(tmp_path / "slow.h5")
    cfg = PulseCaptureConfig(threshold_sigma=5.0, end_sigma=1.5,
                             max_pulse_ms=30.0, noise_train_ms=300.0)
    got = []
    s = PulseCaptureSession(channels=[CHANNEL], sample_rate=FS,
                            hdf5_path=path,
                            on_pulse=lambda ch, idx, summ, data: got.append(idx),
                            **cfg.session_kwargs(FS))
    s.start()
    rng = np.random.default_rng(5)
    n = int(2.5 * FS)
    t = T0 + np.arange(n) / FS
    s.feed_block(CHANNEL, _shape(t) + rng.normal(0, 1, n),
                 rng.normal(0, 1, n), t + LATE)
    s.stop()
    assert got, "the event did not trigger"
    return path


def test_the_window_of_a_corrected_capture_holds_the_event(tmp_path):
    rec = _recording(tmp_path)
    with PulseHDF5Reader(_capture(tmp_path)) as r:
        assert slow_shift_s(r) == 0.0                 # shifted as written
        ov = pulse_overlay(r, rec, CHANNEL, 1)
        assert ov.units == "V" and ov.shift_s == 0.0
        # The session took the CIC delay out: the trigger sits on the
        # event to within a slow sample.  Without the correction it
        # would be late by LATE, three samples.
        pulse = r.get_pulse(CHANNEL, 1)
        assert abs(pulse["trigger_time"] - T) < 1.0 / FS
        # The fastrx window is the pulse's window, and holds the event.
        t = ov.fastrx["times"]
        assert t[0] >= ov.pulse["times"][0] - 1e-9
        assert t[-1] <= ov.pulse["times"][-1] + 1e-9
        assert t[0] < T < t[-1]
        assert ov.seq_gaps == 0 and ov.dropouts == 0
        # In the file's units: counts times its volts_per_count.
        i = ov.fastrx["I"]
        assert i.max() == pytest.approx(AMP * VOLTS_PER_ROC, rel=1e-3)
        onset = t[np.flatnonzero(i > 0)[0]]
        assert abs(onset - T) < 25e-6
        # A slow-only file has no fast stream to correlate against.
        assert ov.fast is None and ov.lag_s is None


def test_an_older_file_is_shifted_by_its_slow_rate(tmp_path):
    """Written before the session took the delay out: raw, late stamps
    and no slow_time_offset_s attribute."""
    rec = _recording(tmp_path)
    path = str(tmp_path / "old.h5")
    t = T + LATE + np.arange(-3, 12) / FS            # late, as the board stamps
    w = PulseHDF5Writer(path, [CHANNEL], {}, {
        "streamer_mode": "slow", "sample_rate_slow": FS,
        "volts_per_count": VOLTS_PER_ROC, "stored_units": "V",
        "trigger_basis": "iq"}, stored_units={CHANNEL: "V"})
    w.append_pulse(CHANNEL, 1, {"Amp_I": _shape(t - LATE) * VOLTS_PER_ROC,
                                "Amp_Q": np.zeros_like(t), "Time": t})
    w.finalize()
    with PulseHDF5Reader(path) as r:
        assert slow_shift_s(r) == pytest.approx(-LATE)
        ov = pulse_overlay(r, rec, CHANNEL, 1)
        assert ov.shift_s == pytest.approx(-LATE)
        assert ov.pulse["times"][3] == pytest.approx(T)
        tf = ov.fastrx["times"]
        assert tf[0] == pytest.approx(ov.pulse["times"][0], abs=25e-6)
        assert tf[0] < T < tf[-1]


def test_counts_to_stored_follows_the_file(tmp_path):
    path = str(tmp_path / "u.h5")
    cal = 3000.0 * np.exp(1j * 0.7)                   # Hz per volt, rotated
    w = PulseHDF5Writer(path, [1, 2], {}, {
        "volts_per_count": 2e-6, "trigger_basis": "df"},
        df_calibrations={1: cal}, stored_units={1: "Hz", 2: "V"})
    w.finalize()
    with PulseHDF5Reader(path) as r:
        assert counts_to_stored(r, 1) == pytest.approx(cal * 2e-6)
        assert counts_to_stored(r, 2) == pytest.approx(2e-6)
    # A file from before samples were stored in physical units.
    path = str(tmp_path / "c.h5")
    PulseHDF5Writer(path, [1], {}, {}).finalize()
    with PulseHDF5Reader(path) as r:
        assert r.stored_units(1) == "counts"
        assert counts_to_stored(r, 1) == 1.0


def test_correlation_lag_reads_a_known_offset():
    dt = 1.0 / 2441406.25
    i = np.arange(400)
    pulse = np.where(i >= 100, 50.0 * np.exp(-(i - 100) / 40.0), 0.0)
    a = {"times": 10.0 + i * dt, "I": pulse + 1.0, "Q": np.zeros(400)}
    # The same feature 30 samples later in b, whose clock also reads 1 ms
    # ahead: b's stamp of the feature is 1 ms + 30 samples past a's.
    b_pulse = np.roll(pulse, 30)
    b = {"times": 10.001 + i * dt, "I": np.zeros(400),
         "Q": b_pulse * np.exp(1j * 1.2).imag + 0.5}      # rotated, offset
    assert correlation_lag_s(a, b, dt) == pytest.approx(0.001 + 30 * dt)
    assert correlation_lag_s(a, {"times": np.array([np.nan]), "I": np.zeros(1),
                                 "Q": np.zeros(1)}, dt) is None


def _dirfile(tmp_path, channel=CHANNEL):
    """The event as the parser writes it: long packets stamped late by
    the board, dec stage 6, so the timebase is corrected as written."""
    gd = pytest.importorskip("pygetdata")
    from rfmux.streamer import ReadoutPacket, Timestamp, TimestampSource
    from rfmux.tools.parser import (BoardStats, ModuleStats,
                                    setup_dirfile_for_module, write_dec_stage)
    path = str(tmp_path / "serial_0042")
    board = BoardStats()
    board.dirfile = gd.dirfile(path, gd.CREAT | gd.RDWR | gd.EXCL)
    mod = ModuleStats()
    setup_dirfile_for_module(board, mod, 0, [range(channel - 1, channel)])
    t = T + np.arange(-0.02, 0.06, 1.0 / FS)
    for frame, ti in enumerate(t):
        pkt = ReadoutPacket(magic=0x5344494b, version=5, serial=42,
                            num_modules=1, flags=0, fir_stage=6, module=0,
                            seq=frame)
        # Slice assignment sizes a constructed packet's sample buffer;
        # a scalar assignment on an empty one writes out of bounds.
        vals = np.zeros(len(pkt), dtype=complex)
        vals[channel - 1] = float(_shape(ti))
        pkt[:] = vals
        late = ti + LATE
        whole = int(late)
        pkt.ts = Timestamp(y=26, d=245, h=whole // 3600, m=(whole // 60) % 60,
                           s=whole % 60, ss=int((late - whole) * 156_250_000),
                           c=0, sbs=whole, source=TimestampSource.TEST,
                           recent=True)
        df, fields = board.dirfile, mod.dirfile_fields
        df.putdata(fields["ts_sbs"], np.array([pkt.ts.sbs], dtype=np.int32),
                   first_frame=frame)
        df.putdata(fields["ts_ss"], np.array([pkt.ts.ss], dtype=np.int32),
                   first_frame=frame)
        write_dec_stage(df, fields, frame, pkt)
        df.putdata(fields["raw"],
                   pkt.raw_samples[2 * (channel - 1):2 * channel],
                   first_frame=frame, first_sample=0)
    board.dirfile.close()
    return path


def test_the_parser_trace_joins_in_the_same_units(tmp_path):
    rec = _recording(tmp_path)
    dirfile = _dirfile(tmp_path)
    with PulseHDF5Reader(_capture(tmp_path)) as r:
        ov = pulse_overlay(r, rec, CHANNEL, 1, dirfile=dirfile)
        d = ov.dirfile
        assert d is not None
        t = d["times"]
        # The dirfile's timebase was corrected as written: its samples
        # sit inside the pulse window on the PFB clock, around the event.
        assert t[0] >= ov.pulse["times"][0] - 1e-9
        assert t[-1] <= ov.pulse["times"][-1] + 1e-9
        assert t[0] < T < t[-1]
        # In the capture's units, and stamped on the PFB clock: the
        # value at each parser sample is the event at that time.
        k = np.argmax(d["I"])
        assert d["I"][k] == pytest.approx(
            float(_shape(t[k])) * VOLTS_PER_ROC, rel=1e-3)
        # Same event, same clock: a few samples into the decay, the
        # capture's sample nearest a parser sample is within half a slow
        # sample and reads the event at its own time (the two grids are
        # not phase-aligned, so the values themselves differ).
        k = np.argmin(np.abs(t - (T + 3.0 / FS)))
        j = np.argmin(np.abs(ov.pulse["times"] - t[k]))
        assert abs(ov.pulse["times"][j] - t[k]) < 0.5 / FS
        assert ov.pulse["I"][j] == pytest.approx(
            float(_shape(ov.pulse["times"][j])) * VOLTS_PER_ROC, rel=0.02)


def test_hertz_capture_projects_the_others_onto_its_axis(tmp_path):
    """A channel stored with its df calibration is in hertz along the
    frequency direction; the recording is rotated and scaled the same
    way, so a pulse along that direction reads on I in hertz."""
    rec = _recording(tmp_path)
    cal = 2500.0 * np.exp(1j * 0.4)          # Hz per volt, its direction
    path = str(tmp_path / "hz.h5")
    cfg = PulseCaptureConfig(threshold_sigma=5.0, end_sigma=1.5,
                             max_pulse_ms=30.0, noise_train_ms=300.0)
    s = PulseCaptureSession(channels=[CHANNEL], sample_rate=FS,
                            hdf5_path=path, df_calibrations={CHANNEL: cal},
                            **cfg.session_kwargs(FS))
    s.start()
    rng = np.random.default_rng(6)
    n = int(2.5 * FS)
    t = T0 + np.arange(n) / FS
    # The event lies along the calibration's direction in counts.
    z = _shape(t) * np.exp(-1j * np.angle(cal))
    s.feed_block(CHANNEL, z.real + rng.normal(0, 1, n),
                 z.imag + rng.normal(0, 1, n), t + LATE)
    s.stop()
    with PulseHDF5Reader(path) as r:
        assert r.stored_units(CHANNEL) == "Hz"
        ov = pulse_overlay(r, rec, CHANNEL, 1)
        assert ov.units == "Hz"
        zf = ov.fastrx["I"] + 1j * ov.fastrx["Q"]
        # The recording's event was on I in counts: rotated by the
        # calibration's phase and scaled to hertz.
        peak = zf[np.argmax(np.abs(zf))]
        assert peak == pytest.approx(AMP * VOLTS_PER_ROC * cal, rel=1e-3)
        # The capture's event, along the calibration, is on I in hertz,
        # read at its own sample times.
        j = np.argmax(ov.pulse["I"])
        assert ov.pulse["I"][j] == pytest.approx(
            float(_shape(ov.pulse["times"][j])) * VOLTS_PER_ROC * abs(cal),
            rel=0.02)
        assert abs(ov.pulse["Q"][j]) < 0.05 * ov.pulse["I"][j]


def test_a_dual_file_brings_the_fast_pulse_and_its_lag(tmp_path):
    """A slow pulse of a dual file comes with its paired fast pulse, and
    the lag at which the recording matches the fast trace: both are on
    the PFB clock, so it reads zero to within a couple of samples."""
    from rfmux.core.transferfunctions import PFB_SAMPLING_FREQ
    from rfmux.pulse_capture.capture_session import DualPulseCaptureSession
    fs_fast = PFB_SAMPLING_FREQ
    rec = _recording(tmp_path, spacing=1.0 / fs_fast, span=(-0.002, 0.035))
    path = str(tmp_path / "dual.h5")
    cfg = PulseCaptureConfig(threshold_sigma=5.0, end_sigma=1.5,
                             max_pulse_ms=20.0, noise_train_ms=200.0)
    d = DualPulseCaptureSession(channels=[CHANNEL], slow_rate=FS,
                                fast_rate=fs_fast, config=cfg,
                                hdf5_path=path)
    d.start()
    rng = np.random.default_rng(7)
    # Training first on both streams (triggering waits for both), then
    # the event: slow stamped late as the board does, fast on time.
    for feed, rate, late in ((d.feed_slow_block, FS, LATE),
                             (d.feed_fast_block, fs_fast, 0.0)):
        n = cfg.noise_samples(rate) + 50
        t = T0 + np.arange(n) / rate
        feed(CHANNEL, rng.normal(0, 1, n), rng.normal(0, 1, n), t + late)
    for feed, rate, late in ((d.feed_slow_block, FS, LATE),
                             (d.feed_fast_block, fs_fast, 0.0)):
        n0 = cfg.noise_samples(rate) + 50
        t = T0 + np.arange(n0, int(2.1 * rate)) / rate
        feed(CHANNEL, _shape(t) + rng.normal(0, 1, len(t)),
             rng.normal(0, 1, len(t)), t + late)
    d.stop()
    with PulseHDF5Reader(path) as r:
        assert r.dual and r.pulse_count(CHANNEL, "slow") >= 1
        ov = pulse_overlay(r, rec, CHANNEL, 1, stream="slow")
        assert ov.fast is not None, "the slow pulse should have its pair"
        assert ov.lag_s is not None
        assert abs(ov.lag_s) < 2.0 / fs_fast
        # The fast pulse itself overlays the recording the same way.
        ovf = pulse_overlay(r, rec, CHANNEL, 1, stream="fast")
        assert ovf.fast is None and abs(ovf.lag_s) < 2.0 / fs_fast


def test_merging_a_recording_makes_a_both_mode_file_of_slow_triggered_pairs(
        tmp_path):
    """The slow side is kept as it was; every slow pulse becomes a pair
    with no fast trigger and the recording over its window, in the
    file's units, so Periscope reviews it as a both-mode capture."""
    from rfmux.core.transferfunctions import PFB_SAMPLING_FREQ
    path = _capture(tmp_path)
    with PulseHDF5Reader(path) as r:
        before = r.get_pulse(CHANNEL, 1)
        n = r.pulse_count(CHANNEL)
    fx = _recording_file(tmp_path, spacing=1.0 / PFB_SAMPLING_FREQ,
                         span=(-0.002, 0.035))

    assert merge_fastrx(path, fx) == __import__("pathlib").Path(path)

    with PulseHDF5Reader(path) as r:
        assert r.dual
        assert r.metadata["streamer_mode"] == "both"
        assert r.metadata["sample_rate_fast"] == PFB_SAMPLING_FREQ
        assert list(r.metadata["fast_channels"]) == [CHANNEL]
        assert r.pulse_count(CHANNEL, "slow") == n
        assert r.pulse_count(CHANNEL, "fast") == 0
        after = r.get_pulse(CHANNEL, 1, "slow")
        np.testing.assert_array_equal(after["Amp_I"], before["Amp_I"])
        np.testing.assert_array_equal(after["Time"], before["Time"])
        assert r.pair_count(CHANNEL) == n
        pair = r.get_match(CHANNEL, 1)
        assert pair["slow_idx"] == 1 and pair["fast_idx"] is None
        t = pair["fast_tod"]["Time"]
        assert np.all(np.isfinite(t))
        assert pair["window"][0] <= t[0] and t[-1] <= pair["window"][1]
        factor = counts_to_stored(r, CHANNEL, "slow")
        assert pair["fast_tod"]["Amp_I"].max() == \
            pytest.approx(AMP * factor.real, rel=0.02)
        assert "noise_std_I" in r.f[f"fast/channel_{CHANNEL}"].attrs


def test_merging_to_another_path_leaves_the_source_slow_only(tmp_path):
    path = _capture(tmp_path)
    fx = _recording_file(tmp_path, spacing=1e-4, span=(-0.002, 0.035))
    out = tmp_path / "both.h5"
    assert merge_fastrx(path, fx, out) == out
    with PulseHDF5Reader(path) as r:
        assert not r.dual
    with PulseHDF5Reader(out) as r:
        assert r.dual and r.pair_count(CHANNEL) == r.pulse_count(CHANNEL, "slow")


def test_a_pulse_the_recording_misses_gets_no_fast_window(tmp_path):
    """The pair is written for the slow pulse; without a window the
    recording covers there is no fast_tod, and Periscope reads the pair
    as slow-only with the fast side unavailable."""
    path = _capture(tmp_path)
    fx = _recording_file(tmp_path, spacing=1e-4, span=(0.5, 0.6))
    merge_fastrx(path, fx)
    with PulseHDF5Reader(path) as r:
        pair = r.get_match(CHANNEL, 1)
        assert pair["slow_idx"] == 1
        assert "fast_tod" not in pair


def test_the_merged_file_carries_fast_histograms_and_templates(tmp_path):
    """The fast side's histograms and templates are built from the
    recording over each pair's window, one entry per pair."""
    from rfmux.core.transferfunctions import PFB_SAMPLING_FREQ
    path = _capture(tmp_path)
    fx = _recording_file(tmp_path, spacing=1.0 / PFB_SAMPLING_FREQ,
                         span=(-0.002, 0.035))
    merge_fastrx(path, fx)
    with PulseHDF5Reader(path) as r:
        n = r.pair_count(CHANNEL)
        hist = r.get_histograms("fast")
        assert hist, "no fast histograms"
        assert int(hist[f"snr_counts_ch{CHANNEL}"].sum()) == n
        assert set(r.get_histograms("slow")) == set(hist)
        tmpl = r.get_templates("fast")
        assert tmpl and f"template_I_ch{CHANNEL}" in tmpl

