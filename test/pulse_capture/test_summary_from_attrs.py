"""A reviewed pulse's summary has the shape a live one has."""
import math

from rfmux.pulse_capture.analysis import pulse_summary, summary_from_attrs
from rfmux.pulse_capture.detection import ChannelNoiseStats
import numpy as np


def _stored_attrs():
    """What PulseHDF5Writer keeps for one pulse: seconds, no ms twins."""
    return {"n_samples": 40, "pileup": False, "truncated": False,
            "peak_I": 3.0e-5, "peak_Q": 1.0e-5, "peak_amp": 3.0e-5,
            "snr": 12.0, "duration_s": 0.004, "timestamp": 43000.0,
            "tau_s": 0.0012, "peak_snr_I": 12.0, "peak_snr_Q": 4.0,
            "trigger_utc": "2026-09-01T12:00:00.250000Z"}


def test_milliseconds_are_derived_from_the_stored_seconds():
    s = summary_from_attrs(_stored_attrs())
    assert s["tau_ms"] == 1.2
    assert s["duration_ms"] == 4.0


def test_the_clock_mark_passes_through():
    assert summary_from_attrs(_stored_attrs())["trigger_utc"] \
        == "2026-09-01T12:00:00.250000Z"
    assert "trigger_utc" not in summary_from_attrs({"n_samples": 1})


def test_snr_and_peak_fall_back_to_the_quadrature_attrs():
    attrs = _stored_attrs()
    del attrs["snr"], attrs["peak_amp"], attrs["tau_s"]
    s = summary_from_attrs(attrs)
    assert s["snr"] == 12.0
    assert s["peak_amp"] == 3.0e-5
    assert math.isnan(s["tau_ms"])


def test_key_set_matches_a_live_summary():
    """Every key a live pulse_summary carries for a pulse with no clock
    or start/end marks is present after the round trip."""
    t = np.arange(40) / 1000.0
    data = {"Amp_I": np.r_[np.zeros(10), 8.0 * np.exp(-np.arange(30) / 5)],
            "Amp_Q": np.zeros(40), "Time": t}
    live = pulse_summary(data, ChannelNoiseStats(std_I=1.0, std_Q=1.0), 5.0)
    stored = {k: live[k] for k in ("n_samples", "pileup", "truncated",
                                   "peak_I", "peak_Q", "peak_amp", "snr",
                                   "duration_s", "timestamp", "tau_s")}
    back = summary_from_attrs(stored)
    positional = {"start_time", "saved_end_time", "trigger_time"}
    assert set(live) - positional <= set(back)
    for k in ("snr", "peak_amp", "duration_ms", "tau_ms", "n_samples"):
        assert back[k] == live[k], k
