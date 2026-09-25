"""Per-channel trigger settings, and the trigger config a file carries:
a capture configured through PulseCaptureConfig records the config it
ran with, a trigger config file holds only that, and either loads back
into a new capture."""

import dataclasses
import json

import numpy as np
import pytest

from rfmux.pulse_capture import (
    ChannelNoiseStats, PulseCapture, PulseCaptureConfig, PulseCaptureSession,
    PulseTemplateAccumulator, derive_tau, pulse_summary, read_trigger_config,
    write_trigger_config)

FS = 1000.0
CHANNELS = [1, 2, 3]


def _square(k):
    """Four samples at 60 sigma."""
    return np.where((k >= 0) & (k < 4), 60.0, 0.0)


def _decay(k):
    """60 sigma decaying over 5 samples: settled well inside the hard
    stop at any end band."""
    return np.where(k >= 0, 60.0 * np.exp(-np.maximum(k, 0) / 5.0), 0.0)


def _capture(tmp_path=None, pulses=((1, 100), (2, 103)), channels=CHANNELS,
             shape=_square, **config_kw):
    """Noise, then *shape* on each of *pulses* (channel, start sample),
    then quiet, the config taken for *channels* as every caller takes
    it.  Returns (pulses by channel, events, path)."""
    cfg = PulseCaptureConfig(max_pulse_ms=50.0, noise_train_ms=300.0,
                             trigger_samples=1, **config_kw)
    found, events = {}, []
    path = None if tmp_path is None else tmp_path / "capture.h5"
    s = PulseCaptureSession(
        channels=channels, sample_rate=FS, hdf5_path=path,
        on_pulse=lambda ch, idx, summary, data:
            found.setdefault(ch, []).append(data),
        on_event=events.append,
        **cfg.for_channels(channels).session_kwargs(FS))
    s.start()
    rng = np.random.default_rng(3)
    t = np.arange(300) / FS
    for ch in channels:
        s.feed_block(ch, rng.normal(0, 1, 300), rng.normal(0, 1, 300), t)
    t = (300 + np.arange(600)) / FS
    k = np.arange(600)
    for ch in channels:
        i = np.zeros(600)
        for pch, start in pulses:
            if pch == ch:
                i += shape(k - start)
        s.feed_block(ch, i, np.zeros(600), t)
    s.stop()
    return found, events, path


# ── per-channel settings in a capture ─────────────────────────────

def test_a_channel_above_the_capture_threshold_does_not_trigger():
    found, _, _ = _capture(per_channel={2: {"threshold_sigma": 100.0}})
    assert sorted(found) == [1]


@pytest.mark.parametrize("use_walk", [True, False])
def test_a_channel_below_the_capture_threshold_triggers_on_either_path(
        monkeypatch, use_walk):
    """The compiled walk and process_sample each apply the channel's
    own threshold, not only the quiet-run pre-scan."""
    monkeypatch.setattr(PulseCapture, "use_walk", use_walk)
    found, _, _ = _capture(threshold_sigma=100.0, end_sigma=1.5,
                           per_channel={2: {"threshold_sigma": 5.0}})
    assert sorted(found) == [2]
    assert found[2][0]["threshold_sigma"] == 5.0


def test_a_tighter_end_band_keeps_the_pulse_longer():
    found, _, _ = _capture(pulses=((1, 100), (2, 100)), shape=_decay,
                           per_channel={1: {"end_sigma": 0.8},
                                        2: {"end_sigma": 3.0}})
    assert len(found[1][0]["Amp_I"]) > len(found[2][0]["Amp_I"])


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


def test_module_channel_keys_run_through_a_capture():
    found, events, _ = _capture(
        pulses=(((2, 1), 100), ((3, 1), 100)), channels=[(2, 1), (3, 1)],
        per_channel={(3, 1): {"trigger": False}})
    assert list(found) == [(2, 1)]
    assert list(events[0]["dump"]) == [(3, 1)]


def test_a_setting_for_a_channel_not_captured_changes_nothing():
    """The forms keep settings for channels taken out of the table; a
    leftover record-only one must not turn events on."""
    found, events, _ = _capture(pulses=((1, 100),),
                                per_channel={7: {"trigger": False}})
    assert list(found) == [1] and events == []
    assert PulseCaptureConfig(per_channel={7: {"trigger": False}}) \
        .for_channels(CHANNELS).buf_size(FS) == PulseCaptureConfig() \
        .buf_size(FS)


def test_the_ring_holds_an_events_span_for_a_record_only_channel():
    rate = 38146.97265625     # above the ring's floor
    plain = PulseCaptureConfig()
    record_only = PulseCaptureConfig(per_channel={2: {"trigger": False}})
    assert record_only.buf_size(rate) > plain.buf_size(rate)


# ── what a pulse's own threshold feeds ────────────────────────────

def _pulse(threshold_sigma=None):
    k = np.arange(200)
    data = {"Amp_I": _decay(k - 20), "Amp_Q": np.zeros(200),
            "Time": k / FS, "trigger_time": 20 / FS}
    if threshold_sigma is not None:
        data["threshold_sigma"] = threshold_sigma
    return data


def test_the_decay_constant_is_derived_at_the_pulses_own_threshold():
    ns = ChannelNoiseStats(std_I=1.0, std_Q=1.0)
    tau = pulse_summary(_pulse(3.0), ns, threshold_sigma=10.0)["tau_s"]
    assert tau == derive_tau(_pulse(), ns, 3.0) != derive_tau(
        _pulse(), ns, 10.0)


def test_a_template_aligns_at_the_pulses_own_threshold():
    """No sample reaches 100 sigma: at the capture's threshold the
    pulse could not be aligned at all."""
    acc = PulseTemplateAccumulator(pre_samples=10, post_samples=50,
                                   threshold_sigma=100.0)
    assert acc.add(_pulse(5.0), ChannelNoiseStats(std_I=1.0, std_Q=1.0))


# ── the config as JSON and as a file ──────────────────────────────

@pytest.mark.parametrize("per_channel", [
    {5: {"trigger": False}},
    {(2, 5): {"threshold_sigma": 7.0, "end_sigma": 2.0}}])
def test_the_config_round_trips_through_json_with_either_kind_of_key(
        per_channel):
    cfg = PulseCaptureConfig(threshold_sigma=4.0, per_channel=per_channel)
    assert PulseCaptureConfig.from_dict(
        json.loads(json.dumps(cfg.to_dict()))) == cfg


def test_from_dict_takes_a_config_that_was_never_through_json():
    cfg = PulseCaptureConfig(per_channel={(2, 5): {"trigger": False}})
    assert PulseCaptureConfig.from_dict(dataclasses.asdict(cfg)) == cfg


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


# ── validation ────────────────────────────────────────────────────

def _errors(per_channel):
    return [m for s, m in PulseCaptureConfig(
        per_channel=per_channel).validate() if s == "error"]


def test_a_channels_end_band_must_sit_below_its_threshold():
    assert any("End σ of channel 4 must sit below" in m
               for m in _errors({4: {"threshold_sigma": 3.0,
                                     "end_sigma": 3.0}}))


def test_an_unknown_channel_setting_is_refused():
    assert any("Unknown setting for channel 4" in m
               for m in _errors({4: {"thresh": 5.0}}))


def test_a_channel_sigma_must_be_positive():
    assert any("must be positive" in m
               for m in _errors({4: {"threshold_sigma": -1.0}}))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "5"])
def test_a_channel_sigma_must_be_a_finite_number(value):
    assert any("finite number" in m
               for m in _errors({4: {"threshold_sigma": value}}))


def test_a_channels_trigger_must_be_true_or_false():
    assert any("true or false" in m for m in _errors({4: {"trigger": 0}}))
