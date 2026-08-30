#!/usr/bin/env python3
"""
Regenerate the static figures embedded in the pulse-capture notebook.

The notebook (rfmux/reference-notebooks/Demos/pulse_capture.md) produces all
of its data plots by running code, so the only figures shipped as files are
diagrams that explain the detector rather than show measurements.  This
script writes them, so they can be regenerated rather than being an opaque
binary in the tree.

    python diagnostics/make_pulse_capture_figures.py
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = (pathlib.Path(__file__).resolve().parents[1]
       / "rfmux" / "reference-notebooks" / "Demos")

THRESH, END = 5.0, 1.5


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

    # Trigger: first sample above threshold, then confirmation
    trig = t[np.argmax(trace > THRESH)]
    # End: signal back inside the end band and staying there
    back = np.where((np.abs(trace) < END) & (t > trig))[0]
    end = t[back[0]] if len(back) else t[-1]

    margin = 0.1 * (end - trig)
    # The bucket needs a stretch of samples inside the band before it calls
    # the pulse over.  Drawn generously here; in a capture it is
    # max(10 samples, margin_fraction x core).
    confirmed = end + 0.45 * (end - trig)

    # save_to_end_confirmed defaults to True, which keeps every sample the
    # state machine saw.  So the saved window runs from before the trigger
    # all the way to the confirmation: one band, drawn once.  The hatched
    # stretch is the tail that save_to_end_confirmed=False drops, and it is
    # marked rather than shaded separately so the default window does not
    # read as ending at the below-threshold instant.
    ax.axvspan(trig - margin, confirmed, color="#3366CC", alpha=0.10,
               zorder=0)
    ax.axvspan(end + margin, confirmed, facecolor="none", edgecolor="#3366CC",
               hatch="///", lw=0.0, alpha=0.55, zorder=1)
    ax.axvline(trig, color="#CC6633", lw=1.6, zorder=4)
    ax.axvline(end, color="#33884D", lw=1.6, ls=":", zorder=4)
    ax.axvline(confirmed, color="#33884D", lw=1.2, ls="--", zorder=4)

    # Labels are kept sparse on purpose: margin_fraction is 10% of an 18 ms
    # window here, far too small to annotate legibly, so the prose covers it.
    ax.text((trig - margin + confirmed) / 2, -3.3, "saved window (default)",
            ha="center", fontsize=9, color="#3366CC", weight="bold")
    ax.annotate("end confirmed",
                xy=(confirmed, 7.4), xytext=(confirmed + 2.0, 8.4),
                fontsize=8, color="#33884D", va="center",
                arrowprops=dict(arrowstyle="->", color="#33884D", lw=1))
    ax.annotate("save_to_end_confirmed=False\nstops the window here",
                xy=(end + margin, -2.6), xytext=(confirmed + 4.0, -6.5),
                fontsize=7.5, color="#3366CC", va="center",
                arrowprops=dict(arrowstyle="->", color="#3366CC", lw=1))

    ax.annotate("trigger", xy=(trig, THRESH), xytext=(trig + 7, 9.2),
                fontsize=9, color="#CC6633",
                arrowprops=dict(arrowstyle="->", color="#CC6633", lw=1.1))
    ax.annotate("both quadratures back\ninside end_sigma", xy=(end, END),
                xytext=(end + 4, 5.6), fontsize=8, color="#33884D",
                arrowprops=dict(arrowstyle="->", color="#33884D", lw=1))

    ax.axvline(hard_stop, color="#7A5AA8", lw=1.4, ls="-.")
    ax.annotate("hard stop\n(1.2 x max_pulse_ms)\ncloses it anyway,\n"
                "flagged truncated",
                xy=(hard_stop, 9.4), xytext=(hard_stop - 1.5, 9.4),
                ha="right", va="top", fontsize=8, color="#7A5AA8")

    ax.set_xlabel("time (ms)")
    ax.set_ylabel("deviation from baseline (σ)")
    ax.set_title("Anatomy of one capture window")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-7.6, 10.5)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    capture_window_anatomy(OUT / "pulse_capture_anatomy.png")
