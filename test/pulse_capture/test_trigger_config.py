"""Per-channel trigger settings, and the trigger config a file carries:
a capture file records the config it ran with, a trigger config file
holds nothing else, and either loads back into a new capture."""

import json

import numpy as np
import pytest

from rfmux.pulse_capture import (
    PulseCaptureConfig, PulseCaptureSession, read_trigger_config,
    write_trigger_config)

FS = 1000.0
CHANNELS = [1, 2, 3]


def _capture(tmp_path=None, pulses=((1, 100), (2, 103)), **config_kw):
    """Noise, then a 60-sigma pulse on each of *pulses* (channel, start
    sample), then quiet.  Returns (pulses by channel, events, path)."""
    cfg = PulseCaptureConfig(max_pulse_ms=50.0, noise_train_ms=300.0,
                             trigger_samples=1, **config_kw)
    found, events = {}, []
    path = None if tmp_path is None else tmp_path / "capture.h5"
    s = PulseCaptureSession(
        channels=CHANNELS, sample_rate=FS, hdf5_path=path,
        on_pulse=lambda ch, idx, summary, data:
            found.setdefault(ch, []).append(data),
        on_event=events.append, **cfg.session_kwargs(FS))
    s.start()
    rng = np.random.default_rng(3)
    t = np.arange(300) / FS
    for ch in CHANNELS:
        s.feed_block(ch, rng.normal(0, 1, 300), rng.normal(0, 1, 300), t)
    t = (300 + np.arange(600)) / FS
    for ch in CHANNELS:
        i = np.zeros(600)
        for pch, start in pulses:
            if pch == ch:
                i[start:start + 4] = 60.0
        s.feed_block(ch, i, np.zeros(600), t)
    s.stop()
    return found, events, path


def test_a_channel_triggers_at_its_own_threshold():
    found, _, _ = _capture(per_channel={2: {"threshold_sigma": 100.0}})
    assert sorted(found) == [1]
    assert found[1][0]["threshold_sigma"] == 5.0


def test_a_pulse_records_its_channels_end_band():
    found, _, _ = _capture(per_channel={2: {"end_sigma": 0.8}})
    assert found[1][0]["end_sigma"] == 1.5
    assert found[2][0]["end_sigma"] == 0.8


def test_a_record_only_channel_is_saved_with_every_event_and_never_triggers():
    found, events, _ = _capture(pulses=((1, 100), (2, 300)),
                                per_channel={2: {"trigger": False}})
    assert sorted(found) == [1]
    (event,) = events
    assert [m["channel"] for m in event["members"]] == [1]
    assert sorted(event["dump"]) == [2]


def test_the_config_round_trips_through_json_with_either_kind_of_key():
    for per_channel in ({5: {"trigger": False}},
                        {(2, 5): {"threshold_sigma": 7.0, "end_sigma": 2.0}}):
        cfg = PulseCaptureConfig(threshold_sigma=4.0, per_channel=per_channel)
        back = PulseCaptureConfig.from_dict(json.loads(json.dumps(
            cfg.to_dict())))
        assert back == cfg


def test_a_trigger_config_file_loads_back_with_its_setup(tmp_path):
    cfg = PulseCaptureConfig(threshold_sigma=6.0,
                             per_channel={3: {"trigger": False}})
    path = write_trigger_config(tmp_path / "trigger_config.h5", cfg,
                                channels=[1, 3], module=2,
                                streamer_mode="both")
    assert read_trigger_config(path) == (
        cfg, {"channels": [1, 3], "module": 2, "streamer_mode": "both"})


def test_a_capture_file_loads_the_config_it_ran_with(tmp_path):
    kw = dict(per_channel={2: {"threshold_sigma": 100.0}})
    _, _, path = _capture(tmp_path, **kw)
    cfg, setup = read_trigger_config(path)
    assert cfg == PulseCaptureConfig(max_pulse_ms=50.0, noise_train_ms=300.0,
                                     trigger_samples=1, **kw)
    assert setup["channels"] == CHANNELS


def test_a_file_without_a_config_says_so(tmp_path):
    import h5py
    path = tmp_path / "other.h5"
    h5py.File(path, "w").close()
    with pytest.raises(ValueError, match="no trigger configuration"):
        read_trigger_config(path)


def test_a_channels_end_band_must_sit_below_its_threshold():
    cfg = PulseCaptureConfig(per_channel={4: {"threshold_sigma": 3.0,
                                              "end_sigma": 3.0}})
    errors = [m for s, m in cfg.validate() if s == "error"]
    assert errors == ["End σ of channel 4 must sit below its threshold "
                      "(3 ≥ 3)."]
