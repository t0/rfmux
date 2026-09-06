#!/usr/bin/env python3
"""
Regenerate the static figures embedded in the pulse-capture documentation.

The notebook (rfmux/reference-notebooks/Demos/pulse_capture.md) produces all
of its data plots by running code, so the only figures shipped as files are
diagrams that explain the detector rather than show measurements.  This
script writes them, so a figure is the output of code that can be read
and rerun rather than an opaque binary in the tree.

The anatomy figure is the engine's own output: a synthetic pulse, and a
piled-up pair, are fed through PulseCapture with the PulseCaptureConfig
defaults, and every shaded window and mark is read from the record it
saved.  Writes every copy of each figure, so the notebook and the guide
cannot drift apart.

    python docs/make_pulse_capture_figures.py
"""

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rfmux.core.transferfunctions import decimation_to_sampling  # noqa: E402
from rfmux.pulse_capture import PulseCaptureConfig  # noqa: E402
from rfmux.pulse_capture.detection import (  # noqa: E402
    ChannelNoiseStats, PulseCapture)

#: Every place each figure is embedded.  The first is the source of truth;
#: the rest are copies kept byte-identical by writing them all here.
ANATOMY_PATHS = (
    ROOT / "rfmux" / "reference-notebooks" / "Demos" / "pulse_capture_anatomy.png",
    ROOT / "docs" / "guides" / "images" / "capture-window-anatomy.png",
)

#: Decimation stage 3, 4.77 kHz: a 12 ms decay spans tens of samples.
FS = decimation_to_sampling(3)
#: Every field but the pulse length at its default.
CONFIG = PulseCaptureConfig(max_pulse_ms=50.0)
AMP_SIGMA, TAU_MS, RISE_MS = 14.0, 12.0, 1.0
SEED = 13

RED, GREEN, ORANGE, BLUE, PURPLE, TEAL = (
    "#CC3333", "#33884D", "#CC6633", "#3366CC", "#7A5AA8", "#2A8FA8")


def _noise(rng, n):
    """Unit-sigma noise smoothed over three samples, so neighboring
    samples are correlated as in a decimated stream rather than
    independent."""
    x = np.convolve(rng.normal(0.0, 1.0, n + 2), np.ones(3) / 3, "valid")
    return x / x.std()


def _pulses(t_ms, arrivals):
    """An exponential pulse of AMP_SIGMA at each arrival time."""
    x = np.zeros(t_ms.size)
    for t0 in arrivals:
        dt = t_ms - t0
        x += np.where(dt >= 0,
                      AMP_SIGMA * (1 - np.exp(-dt / RISE_MS))
                      * np.exp(-dt / TAU_MS), 0.0)
    return x


def _run_engine(t_ms, i_sigma, q_sigma):
    """Feed one channel through the engine; return the records it saved."""
    kwargs = CONFIG.session_kwargs(FS)
    for key in ("trigger_basis", "noise_samples"):
        kwargs.pop(key)
    records = []
    stats = ChannelNoiseStats(mean_I=0.0, std_I=1.0, mean_Q=0.0, std_Q=1.0)
    pcap = PulseCapture(channels=[1], noise_stats={1: stats},
                        on_pulse=lambda ch, k, rec: records.append(rec),
                        **kwargs)
    for t, i, q in zip(t_ms, i_sigma, q_sigma):
        pcap.process_sample(1, float(i), float(q), float(t))
    return records


def _bands(ax):
    ax.axhline(0, color="#888888", lw=0.9)
    for s, color, label in (
            (CONFIG.threshold_sigma, RED,
             f"threshold_sigma = {CONFIG.threshold_sigma:g}σ"),
            (CONFIG.end_sigma, GREEN,
             f"end_sigma = {CONFIG.end_sigma:g}σ")):
        ax.axhline(s, color=color, ls="--", lw=1.2, label=label)
        ax.axhline(-s, color=color, ls="--", lw=1.2)


def _one_pulse(ax, t, trace, rec):
    """Where each parameter acts on a single pulse."""
    edge_ms = CONFIG.edge_lookback_samples(FS) / FS * 1e3
    stop_ms = CONFIG.max_capture_samples(FS) / FS * 1e3
    trig, below = rec["trigger_time"], rec["below_threshold_time"]
    settled, confirmed = rec["settled_time"], rec["end_time"]
    saved = rec["Time"]

    _bands(ax)
    ax.text(69.5, CONFIG.threshold_sigma + 0.3,
            f"threshold_sigma = {CONFIG.threshold_sigma:g}σ",
            ha="right", va="bottom", fontsize=8, color=RED)
    ax.text(69.5, CONFIG.end_sigma + 0.3,
            f"end_sigma = {CONFIG.end_sigma:g}σ",
            ha="right", va="bottom", fontsize=8, color=GREEN)
    ax.plot(t, trace, color=BLUE, lw=1.1, zorder=3)
    ax.axvspan(saved[0], saved[-1], color=BLUE, alpha=0.10, zorder=0)
    ax.axvline(trig, color=ORANGE, lw=1.6, zorder=4)
    ax.axvline(below, color=RED, lw=1.2, ls=":", zorder=4)
    ax.axvline(settled, color=TEAL, lw=1.4, ls=":", zorder=4)
    ax.axvline(confirmed, color=GREEN, lw=1.2, ls="--", zorder=4)
    ax.axvline(trig + stop_ms, color=PURPLE, lw=1.4, ls="-.")

    # Edge test: the bracket the trigger looks back across.
    y = 15.6
    ax.plot([trig - edge_ms, trig], [y, y], color=ORANGE, lw=1.2)
    ax.plot([trig - edge_ms] * 2, [y - 0.3, y + 0.3], color=ORANGE, lw=1.2)
    ax.text(trig - edge_ms - 0.8, y + 0.4,
            "edge test: rose by more\nthan threshold_sigma\n"
            "jump-σ across the last\nedge_lookback samples\n"
            "(margin_fraction ×\nmax_pulse_ms)",
            ha="right", va="top", fontsize=7.5, color=ORANGE)
    ax.annotate("trigger: trigger_samples\nabove threshold_sigma,\n"
                "dated to the first",
                xy=(trig, CONFIG.threshold_sigma), xytext=(trig - 7, 6.5),
                ha="right", va="top", fontsize=7.5, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))
    ax.annotate("back below\nthreshold_sigma\n(feeds the fit-free\n"
                "decay constant)",
                xy=(below, CONFIG.threshold_sigma), xytext=(below + 4, 9.8),
                fontsize=7.5, color=RED, va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1))
    ax.annotate("settled: both axes back\ninside end_sigma\n"
                "(duration_ms ends here)",
                xy=(settled, 2.4), xytext=(settled - 4, -3.2),
                fontsize=7.5, color=TEAL, va="top", ha="right",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1))
    ax.annotate("saved window:\nmargin_fraction of it\n"
                "before the trigger,\nto the end confirmation",
                xy=(saved[0], -1.8), xytext=(-29.5, -2.4),
                ha="left", va="top", fontsize=7.5, color=BLUE,
                weight="bold",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))
    ax.annotate("end confirmed: inside\n"
                "end_sigma of the baseline\n"
                "or of the level the pulse\n"
                "rose from, counted up\n"
                "while inside and down\n"
                "while out, past\n"
                "max(min_end_samples,\n"
                "margin_fraction × time\n"
                "above threshold).\n"
                "The window ends here.",
                xy=(confirmed, 1.6), xytext=(confirmed + 1.5, 16.4),
                fontsize=7.5, color=GREEN, va="top", ha="left",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))
    ax.text(trig + stop_ms + 0.8, 14.6,
            "hard stop\n(1.2 × max_pulse_ms)\ncloses it anyway,\n"
            "flagged truncated",
            ha="left", va="top", fontsize=7.5, color=PURPLE)
    ax.set_title("Anatomy of one capture window")
    ax.set_ylim(-7.0, 17.0)


def _pileup(ax, t, trace, first, second):
    """Two pulses in one window are split on the second's rise."""
    _bands(ax)
    ax.plot(t, trace, color=BLUE, lw=1.1, zorder=3)
    for rec, color in ((first, BLUE), (second, ORANGE)):
        saved = rec["Time"]
        ax.axvspan(saved[0], saved[-1], color=color, alpha=0.10, zorder=0)
        ax.axvline(rec["trigger_time"], color=ORANGE, lw=1.6, zorder=4)
    split = second["trigger_time"]
    ax.annotate("pileup split: a fresh rise above the\n"
                "pulse's own recent level, after it was\n"
                "seen decaying (enable_pileup)",
                xy=(split, 6.0), xytext=(split + 14, 8.8),
                fontsize=7.5, color=ORANGE, va="top",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))
    ax.text((first["Time"][0] + first["Time"][-1]) / 2, -2.6,
            "first fragment:\nends at the split,\nflagged pileup",
            ha="center", va="top", fontsize=7.5, color=BLUE, weight="bold")
    ax.text(second["Time"][-1] + 1.0, -2.6,
            "second fragment: sits on the first's tail,\n"
            "flagged pileup; templates skip both,\nhistograms keep them",
            ha="left", va="top", fontsize=7.5, color=ORANGE, weight="bold")
    ax.set_title("Two pulses piled up")
    ax.set_ylim(-7.0, 17.0)


def capture_window_anatomy(path):
    rng = np.random.default_rng(SEED)
    t = np.arange(-30e-3, 95e-3, 1 / FS) * 1e3         # ms
    stop_ms = CONFIG.max_capture_samples(FS) / FS * 1e3

    single = _pulses(t, [0.0]) + _noise(rng, t.size)
    recs = _run_engine(t, single, _noise(rng, t.size))
    assert len(recs) == 1, [r["trigger_time"] for r in recs]
    assert not recs[0]["pileup"] and not recs[0]["truncated"]
    assert recs[0]["end_time"] < recs[0]["trigger_time"] + stop_ms - 1

    pair = _pulses(t, [0.0, 18.0]) + _noise(rng, t.size)
    pair_recs = _run_engine(t, pair, _noise(rng, t.size))
    assert len(pair_recs) == 2, [r["trigger_time"] for r in pair_recs]
    assert all(r["pileup"] for r in pair_recs)
    assert (pair_recs[1]["end_time"]
            < pair_recs[1]["trigger_time"] + stop_ms - 1)

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10.0, 8.6),
                                      sharex=True)
    _one_pulse(top, t, single, recs[0])
    _pileup(bottom, t, pair, *pair_recs)
    for ax in (top, bottom):
        ax.set_ylabel("deviation from baseline (σ)")
        ax.set_xlim(t[0], 70.0)
    bottom.set_xlabel("time (ms)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

    rec = recs[0]
    print(f"wrote {path}")
    print(f"  {FS:.0f} Hz, max_pulse_ms={CONFIG.max_pulse_ms:g}: "
          f"edge_lookback {CONFIG.edge_lookback_samples(FS)} samples, "
          f"trigger_samples {CONFIG.trigger_samples_for(FS)}, "
          f"hard stop {CONFIG.max_capture_samples(FS) / FS * 1e3:.0f} ms")
    print(f"  saved {rec['Time'][0]:.1f} to {rec['Time'][-1]:.1f} ms, "
          f"trigger {rec['trigger_time']:.1f}, below threshold "
          f"{rec['below_threshold_time']:.1f}, settled "
          f"{rec['settled_time']:.1f}, end confirmed "
          f"{rec['end_time']:.1f} ms")
    print(f"  pileup split at {pair_recs[1]['trigger_time']:.1f} ms")


if __name__ == "__main__":
    first, *copies = ANATOMY_PATHS
    first.parent.mkdir(parents=True, exist_ok=True)
    capture_window_anatomy(first)
    for dest in copies:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(first.read_bytes())
        print(f"copied to {dest}")
