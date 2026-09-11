"""A capture that spans modules is keyed by (module, channel) pairs.
The file nests them as module_<M>/channel_<n>; a one-module file is
unchanged (every other test in this directory pins that)."""

import numpy as np
import pytest

from rfmux.pulse_capture.accumulators import PulseHistogramSet, PulseTemplateSet
from rfmux.pulse_capture.channel_keys import (channel_group, channel_suffix,
                                              check_keys, keys_from_attr)
from rfmux.pulse_capture.detection import ChannelNoiseStats
from rfmux.pulse_capture.hdf5 import (DualPulseHDF5Writer, PulseHDF5Reader,
                                      PulseHDF5Writer)

KEYS = [(2, 5), (3, 1)]


def _pulse(n=64, peak=500.0):
    t = np.arange(n) / 596.0
    amp = np.where(t >= t[8], peak * np.exp(-(t - t[8]) / 0.01), 0.0)
    return {"Amp_I": amp, "Amp_Q": np.zeros(n), "Time": t, "pileup": False}


def test_keys_name_groups_and_datasets():
    assert channel_group(5) == "channel_5"
    assert channel_group((2, 5)) == "module_2/channel_5"
    assert channel_suffix(5) == "ch5"
    assert channel_suffix((2, 5)) == "m2ch5"
    assert check_keys([np.int64(3), 4]) == [3, 4]
    assert check_keys([[2, 5], (3, 1)]) == KEYS
    with pytest.raises(ValueError, match="all be numbers or all"):
        check_keys([1, (2, 5)])
    assert keys_from_attr(np.array([1, 2])) == [1, 2]
    assert keys_from_attr(np.array(KEYS)) == KEYS
    assert keys_from_attr([]) == []


def test_a_multi_module_file_round_trips_its_pair_keys(tmp_path):
    path = tmp_path / "pulse.h5"
    noise = {k: ChannelNoiseStats(std_I=10.0, std_Q=10.0) for k in KEYS}
    w = PulseHDF5Writer(path, KEYS, noise, {"streamer_mode": "slow"},
                        df_calibrations={(2, 5): 1 + 1j})
    w.append_pulse((2, 5), 1, _pulse())
    hist = PulseHistogramSet(threshold_sigma=5.0)
    hist.add_pulse((2, 5), _pulse(), noise[(2, 5)])
    w.update_histograms(hist.get_histogram_data())
    tmpl = PulseTemplateSet(pre_samples=4, post_samples=32, sample_rate=596.0)
    tmpl.add_pulse((3, 1), _pulse(), noise[(3, 1)])
    w.update_templates(tmpl.get_template_data())
    w.finalize()

    with PulseHDF5Reader(path) as r:
        assert r.channels == KEYS and r.multi_module and r.modules == [2, 3]
        assert "module_2/channel_5" in r.f and "module_3/channel_1" in r.f
        assert r.pulse_count((2, 5)) == 1 and r.pulse_count((3, 1)) == 0
        assert r.get_pulse((2, 5), 1)["n_samples"] == 64
        assert r.df_calibration((2, 5)) == 1 + 1j
        assert r.noise_stats((3, 1)).std_I == 10.0
        assert "snr_counts_m2ch5" in r.get_histograms()
        assert "template_I_m3ch1" in r.get_templates()


def test_a_one_module_file_reports_its_module(tmp_path):
    path = tmp_path / "pulse.h5"
    PulseHDF5Writer(path, [1, 2], {}, {"module": 3}).finalize()
    with PulseHDF5Reader(path) as r:
        assert r.channels == [1, 2] and not r.multi_module
        assert r.modules == [3]


def test_a_dual_file_nests_pairs_under_each_stream(tmp_path):
    path = tmp_path / "dual.h5"
    w = DualPulseHDF5Writer(path, KEYS, {"fast_channels": [(2, 5)]})
    w.append_pulse("fast", (2, 5), 1, _pulse())
    w.append_match((2, 5), {"channel": (2, 5), "pair_idx": 1,
                            "slow_idx": None, "fast_idx": 1,
                            "window": (0.0, 0.1)})
    w.finalize()
    with PulseHDF5Reader(path) as r:
        assert r.dual and r.channels == KEYS
        assert keys_from_attr(r.metadata["fast_channels"]) == [(2, 5)]
        assert r.pulse_count((2, 5), "fast") == 1
        assert r.pair_count((2, 5)) == 1
        assert r.get_match((2, 5), 1)["fast_idx"] == 1
        for stream in ("slow", "fast", "matched"):
            assert f"{stream}/module_3/channel_1" in r.f
