#!/usr/bin/env python3
"""
Regenerate the static figures embedded in the pulse-capture documentation.

The notebook (rfmux/reference-notebooks/Demos/pulse_capture.md) produces all
of its data plots by running code, so the only figures shipped as files are
diagrams that explain the detector rather than show measurements.  This
script writes them, so they can be regenerated rather than being an opaque
binary in the tree -- which is how the saved window came to be drawn ending
at the wrong place and stayed that way through a policy change.

Writes every copy of each figure, so the notebook and the release note
cannot drift apart.

    python docs/make_pulse_capture_figures.py
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Every place each figure is embedded.  The first is the source of truth;
#: the rest are copies kept byte-identical by writing them all here.
ANATOMY_PATHS = (
    ROOT / "rfmux" / "reference-notebooks" / "Demos" / "pulse_capture_anatomy.png",
    ROOT / "docs" / "release-notes" / "images" / "capture-window-anatomy.png",
)

THRESH, END = 5.0, 1.0


def capture_window_anatomy(path):
    """Where each PulseCaptureConfig parameter acts on a real capture."""
    rng = np.random.default_rng(4)
    # max_pulse_ms = 50 for this illustration, so the hard stop sits at 60 ms.
    max_pulse_ms, hard_stop = 50.0, 60.0
    t = np.linspace(-20, 80, 700)               # ms
    tau, t0 = 12.0, 0.0
    pulse = np.where(t >= t0, 9.0 * np.exp(-(t - t0) / tau), 0.0)
    trace = pulse + rng.normal(0, 0.45, t.size)

    fig, ax = plt.subplots(figsize=(9.5, 4.2))

    bands = ((THRESH, "#CC3333", f"threshold_sigma = {THRESH:g}σ"),
             (END, "#33884D", f"end_sigma = {END:g}σ"))
    for s, color, label in bands:
        ax.axhline(s, color=color, ls="--", lw=1.2, label=label)
        ax.axhline(-s, color=color, ls="--", lw=1.2)
    ax.axhline(0, color="#888888", lw=0.9)

    ax.plot(t, trace, color="#3366CC", lw=1.1, zorder=3)

    # Trigger: first sample above threshold, then confirmation.
    trig = t[np.argmax(trace > THRESH)]
    # Back below threshold: one past the last sample above it.
    above = np.where((trace > THRESH) & (t >= trig))[0]
    below = t[above[-1] + 1]
    # End band: back inside end_sigma and staying there.
    back = np.where((np.abs(trace) < END) & (t > trig))[0]
    end = t[back[0]] if len(back) else t[-1]
    # The confirmation count needs a stretch of samples inside the band
    # before it calls the pulse over.  Drawn generously here.
    confirmed = end + 0.45 * (end - trig)

    # What the detector saves by default: margin_fraction of the core
    # before the trigger, and the same past the below-threshold instant,
    # floored at min_end_samples.  The end confirmation bounds the state
    # machine, not the record; save_to_end_confirmed=True extends the
    # window to it.
    core = below - trig
    pre = 0.1 * core
    tail = max(0.1 * core, 2.0)
    ax.axvspan(trig - pre, below + tail, color="#3366CC", alpha=0.10,
               zorder=0)
    ax.axvline(trig, color="#CC6633", lw=1.6, zorder=4)
    ax.axvline(below, color="#CC3333", lw=1.2, ls=":", zorder=4)
    ax.axvline(end, color="#33884D", lw=1.6, ls=":", zorder=4)
    ax.axvline(confirmed, color="#33884D", lw=1.2, ls="--", zorder=4)

    ax.text((trig - pre + below + tail) / 2, -3.3, "saved\nwindow",
            ha="center", fontsize=9, color="#3366CC", weight="bold")
    ax.annotate("trigger", xy=(trig, THRESH), xytext=(trig - 15, 8.2),
                fontsize=9, color="#CC6633",
                arrowprops=dict(arrowstyle="->", color="#CC6633", lw=1.1))
    ax.annotate("back below\nthreshold_sigma", xy=(below, THRESH),
                xytext=(below + 5, 7.6), fontsize=8, color="#CC3333",
                va="center",
                arrowprops=dict(arrowstyle="->", color="#CC3333", lw=1))
    ax.annotate("both axes back\ninside end_sigma", xy=(end, END),
                xytext=(confirmed + 3.5, 3.2), fontsize=8, color="#33884D",
                va="top",
                arrowprops=dict(arrowstyle="->", color="#33884D", lw=1))
    ax.annotate("end confirmed\n(save_to_end_confirmed=True\n"
                "extends the window to here)",
                xy=(confirmed, 5.4), xytext=(confirmed + 2.0, 5.6),
                fontsize=8, color="#33884D", va="top",
                arrowprops=dict(arrowstyle="->", color="#33884D", lw=1))

    ax.axvline(hard_stop, color="#7A5AA8", lw=1.4, ls="-.")
    ax.annotate("hard stop\n(1.2 x max_pulse_ms)\ncloses it anyway;\n"
                "truncated only if\nstill above threshold",
                xy=(hard_stop, 9.4), xytext=(hard_stop - 1.5, 9.4),
                ha="right", va="top", fontsize=8, color="#7A5AA8")

    ax.set_xlabel("time (ms)")
    ax.set_ylabel("deviation from baseline (σ)")
    ax.set_title("Anatomy of one capture window")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-6.2, 10.5)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    first, *copies = ANATOMY_PATHS
    first.parent.mkdir(parents=True, exist_ok=True)
    capture_window_anatomy(first)
    for dest in copies:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(first.read_bytes())
        print(f"copied to {dest}")
