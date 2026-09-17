"""The saved window: a set span before the trigger and a set span after
the pulse settled, whatever the pulse's own length."""

import numpy as np
import pytest

from rfmux.pulse_capture.capture_session import (
    PulseCaptureConfig, PulseCaptureSession)

FS = 1000.0


def _capture(pulse_len, **session_kw):
    """(summary, data) of one clean pulse of *pulse_len* samples."""
    got = []
    s = PulseCaptureSession(
        channels=[1], sample_rate=FS, noise_samples=300, hdf5_path=None,
        threshold_sigma=5.0, end_sigma=1.5, min_end_samples=10,
        trigger_samples=1, buf_size=2000,
        on_pulse=lambda ch, idx, summ, data: got.append((summ, data)),
        **session_kw)
    s.start()
    rng = np.random.default_rng(2)
    n = 300
    s.feed_block(1, rng.normal(0, 1, n), rng.normal(0, 1, n),
                 np.arange(n) / FS)
    m = 600
    i = np.zeros(m)
    i[100:100 + pulse_len] = 60.0
    s.feed_block(1, i, np.zeros(m), (n + np.arange(m)) / FS)
    s.stop()
    assert len(got) == 1
    return got[0]


@pytest.mark.parametrize("pulse_len", [4, 120])
def test_pre_pulse_span_does_not_depend_on_the_pulse(pulse_len):
    _, data = _capture(pulse_len, pre_samples=25, post_samples=0)
    assert data["trigger_index"] == 25
    # The trigger mark sits on the pulse's first sample.
    assert data["Amp_I"][25] > 0.0 and data["Amp_I"][24] == 0.0


@pytest.mark.parametrize("pulse_len", [4, 120])
def test_post_pulse_span_follows_the_settled_sample(pulse_len):
    summ, data = _capture(pulse_len, pre_samples=25, post_samples=40)
    assert len(data["Amp_I"]) - 1 - data["settled_index"] == 40
    # The pulse is measured as before: trigger to settled.
    assert summ["duration_s"] == pytest.approx(
        data["settled_time"] - data["trigger_time"])
    assert summ["saved_end_time"] == pytest.approx(
        data["settled_time"] + 40 / FS)


def test_the_end_waits_for_the_post_pulse_span():
    """The samples have to exist before they can be saved: the capture
    is released no sooner than the span after the settled sample."""
    _, short = _capture(4, pre_samples=25, post_samples=0)
    _, long = _capture(4, pre_samples=25, post_samples=40)
    assert short["end_index"] - short["settled_index"] == 10   # the floor
    assert long["end_index"] - long["settled_index"] == 40
    assert long["end_confirm_target"] == 40


def test_without_a_post_pulse_span_the_record_ends_where_it_settled():
    _, data = _capture(4, pre_samples=25, post_samples=0)
    assert len(data["Amp_I"]) - 1 == data["settled_index"]


def test_times_reach_the_engine_as_samples_at_the_stream_rate():
    cfg = PulseCaptureConfig(pre_pulse_ms=25.0, post_pulse_ms=40.0)
    kw = cfg.session_kwargs(FS)
    assert (kw["pre_samples"], kw["post_samples"]) == (25, 40)
    kw["noise_samples"] = 200
    s = PulseCaptureSession(channels=[1], sample_rate=FS, hdf5_path=None, **kw)
    s.start()
    rng = np.random.default_rng(1)
    s.feed_block(1, rng.normal(0, 1, 200), rng.normal(0, 1, 200),
                 np.arange(200) / FS)
    assert (s.pcap.pre_samples, s.pcap.post_samples) == (25, 40)
    s.stop()
