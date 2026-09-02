"""The record carries the bands each decision was made against, and the
views draw those rather than the live stats, which the rolling median
re-centres after the pulse."""
import h5py
import numpy as np
import pytest

from rfmux.pulse_capture.capture_session import (
    PulseCaptureConfig, PulseCaptureSession)
from rfmux.pulse_capture.hdf5 import _pulse_dict_from_group

FS = 596.0
THR, END = 5.0, 1.5
BAND_KEYS = ("trigger_baseline_I", "trigger_baseline_Q",
             "trigger_sigma_I", "trigger_sigma_Q", "trigger_quad",
             "end_baseline_I", "end_baseline_Q",
             "threshold_sigma", "end_sigma")


def _capture(drift_after, hdf5_path=None):
    """One shallow pulse, then the baseline drifts by *drift_after* σ."""
    got = []
    cfg = PulseCaptureConfig(threshold_sigma=THR, end_sigma=END,
                             max_pulse_ms=100.0, noise_train_ms=500.0)
    kw = cfg.session_kwargs(FS)
    kw["baseline_window"] = 600          # ~1 s rolling median
    s = PulseCaptureSession(
        channels=[1], sample_rate=FS, hdf5_path=hdf5_path,
        on_pulse=lambda ch, i, summ, d: got.append(d), **kw)
    s.start()
    rng = np.random.default_rng(7)
    n_tr = cfg.noise_samples(FS)
    n1 = n_tr + 1200
    t1 = np.arange(n1) / FS
    y1 = rng.normal(0, 1, n1)
    k0 = n_tr + 900
    y1[k0:k0 + 20] += np.linspace(0, 30, 20)         # 1.5σ per sample
    y1[k0 + 20:k0 + 80] += 30 * np.exp(-np.arange(60) / 20)
    s.feed_block(1, y1, rng.normal(0, 1, n1), 43000.0 + t1)
    n2 = 1800
    t2 = t1[-1] + (1 + np.arange(n2)) / FS
    s.feed_block(1, rng.normal(0, 1, n2) + drift_after,
                 rng.normal(0, 1, n2), 43000.0 + t2)
    s.stop()
    assert len(got) == 1
    return s, got[0]


def test_record_carries_the_band_that_fired():
    s, d = _capture(+2.0)
    for k in BAND_KEYS:
        assert k in d, k
    assert d["threshold_sigma"] == THR and d["end_sigma"] == END
    assert d["trigger_quad"] == "I"
    base, sig = d["trigger_baseline_I"], d["trigger_sigma_I"]
    amp = np.asarray(d["Amp_I"], float)
    ti = int(d["trigger_index"])
    # Against the recorded band the trigger sample is the first over.
    assert abs(amp[ti] - base) > THR * sig
    assert abs(amp[ti - 1] - base) <= THR * sig
    # The live stats have moved on since — and would hide that sample.
    live = s.noise_stats[1]
    assert abs(live.mean_I - base) > 1.5 * sig
    assert abs(amp[ti] - live.mean_I) < THR * live.std_I
    assert np.isfinite(d["end_baseline_I"]) and np.isfinite(d["end_baseline_Q"])


def test_the_bands_survive_the_file(tmp_path):
    path = tmp_path / "capture.h5"
    _, d = _capture(0.0, hdf5_path=str(path))
    with h5py.File(path, "r") as f:
        back = _pulse_dict_from_group(f["channel_1/pulse_000001"])
    for k in BAND_KEYS:
        assert back[k] == d[k], k


def _drawn(panel, plot, name_part):
    it = next(it for it in plot.getPlotItem().listDataItems()
              if it.name() and name_part in it.name())
    return np.asarray(it.xData, float), np.asarray(it.yData, float)


def _mark(plot, label_part):
    import pyqtgraph as pg
    return next(float(it.value()) for it in plot.getPlotItem().items
                if isinstance(it, pg.InfiniteLine) and it.label
                and label_part in it.label.format)


def _panel_showing(qt_app, d, ns):
    from rfmux.tools.periscope.pulse_capture_panel import PulseCapturePanel
    panel = PulseCapturePanel(dark_mode=False)
    panel.noise_stats = {1: ns}
    panel._pulse_summaries[(1, 1)] = {
        "n_samples": len(d["Amp_I"]), "duration_ms": 1.0, "peak_amp": 0.0,
        "snr": 0.0, "tau_ms": float("nan")}
    panel._get_waveform = lambda ch, idx, stream="slow": d
    panel.threshold_spin.setValue(THR)
    panel.end_spin.setValue(END)
    panel._show_pulse(1, 1)
    return panel


@pytest.mark.parametrize("drift", [+0.7, +2.0])
def test_the_trigger_mark_sits_on_the_first_drawn_crossing(qt_app, drift):
    s, d = _capture(drift)
    panel = _panel_showing(qt_app, d, s.noise_stats[1])
    try:
        x, y = _drawn(panel, panel.pulse_plot_i, "(pulse)")
        _, band = _drawn(panel, panel.pulse_plot_i, "trigger")
        first = x[y > np.nanmax(band)].min()
        assert abs(_mark(panel.pulse_plot_i, "trigger") - first) * FS < 0.5
        # ...and the band drawn is the record's, not the wandered one.
        assert np.nanmax(band) == pytest.approx(
            d["trigger_baseline_I"] + THR * d["trigger_sigma_I"])
    finally:
        panel.close()


def test_the_end_band_sits_on_the_anchor(qt_app):
    s, d = _capture(+2.0)
    panel = _panel_showing(qt_app, d, s.noise_stats[1])
    try:
        _, band = _drawn(panel, panel.pulse_plot_i, "end")
        assert np.nanmax(band) == pytest.approx(
            d["end_baseline_I"] + END * d["trigger_sigma_I"])
        assert np.nanmin(band) == pytest.approx(
            d["end_baseline_I"] - END * d["trigger_sigma_I"])
        assert "(on I)" in panel._decision_text(d)
    finally:
        panel.close()


def test_a_record_without_bands_falls_back_to_the_stats(qt_app):
    s, d = _capture(0.0)
    bare = {k: v for k, v in d.items() if k not in BAND_KEYS}
    panel = _panel_showing(qt_app, bare, s.noise_stats[1])
    try:
        _, band = _drawn(panel, panel.pulse_plot_i, "trigger")
        ns = s.noise_stats[1]
        assert np.nanmax(band) == pytest.approx(ns.mean_I + THR * ns.std_I)
    finally:
        panel.close()


def _record(t0, base, sig, n=40, fs=FS):
    t = t0 + np.arange(n) / fs
    amp = np.full(n, base) + np.r_[np.zeros(10), np.full(30, 8 * sig)]
    return {"Amp_I": amp, "Amp_Q": np.full(n, 0.0), "Time": t,
            "trigger_index": 10, "end_index": 30,
            "trigger_time": float(t[10]), "end_time": float(t[30]),
            "trigger_baseline_I": base, "trigger_baseline_Q": 0.0,
            "trigger_sigma_I": sig, "trigger_sigma_Q": sig,
            "trigger_quad": "I", "end_baseline_I": base + 0.3 * sig,
            "end_baseline_Q": 0.0, "threshold_sigma": THR, "end_sigma": END}


def test_the_pair_view_draws_each_records_band(qt_app):
    """Both streams: the band is the one on that stream's own record,
    not the live stats (which have moved on) — and each is its own."""
    from rfmux.pulse_capture.detection import ChannelNoiseStats
    from rfmux.tools.periscope.pulse_capture_panel import PulseCapturePanel
    slow = _record(43000.0, base=1.0, sig=0.1)
    fast = _record(43000.0, base=5.0, sig=0.5, fs=2.44e6)
    panel = PulseCapturePanel(dark_mode=False)
    try:
        panel._both_mode = True
        panel._reset_results([1])
        panel._noise_by_stream = {
            "slow": {1: ChannelNoiseStats(mean_I=1.7, std_I=0.1)},
            "fast": {1: ChannelNoiseStats(mean_I=9.0, std_I=0.5)}}
        panel._on_pair_matched({
            "channel": 1, "pair_idx": 1, "slow_idx": 1, "fast_idx": 1,
            "time_offset": 0.0, "slow_summary": {"snr": 10.0},
            "fast_summary": {"snr": 10.0}})
        panel._get_pair = lambda ch, i: None
        panel._get_waveform = lambda ch, idx, stream: {
            "slow": slow, "fast": fast}[stream]
        panel.threshold_spin.setValue(3.0)     # moved since the capture
        panel._show_pair(1, 1)
        _, sb = _drawn(panel, panel.pulse_plot_i, "slow ±5σ trigger")
        _, fb = _drawn(panel, panel.pulse_plot_i, "fast ±5σ trigger")
        assert np.nanmax(sb) == pytest.approx(1.0 + THR * 0.1)
        assert np.nanmax(fb) == pytest.approx(5.0 + THR * 0.5)
        _, se = _drawn(panel, panel.pulse_plot_i, "slow ±1.5σ end")
        assert np.nanmax(se) == pytest.approx(1.03 + END * 0.1)
    finally:
        panel.close()
