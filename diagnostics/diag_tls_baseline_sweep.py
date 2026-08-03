#!/usr/bin/env python3
"""
Does the pulse architecture survive 1/f noise?  Measure, don't argue.

Sweeps TLS drift amplitude against the rolling-median baseline window
(the "track_ms" axis below; it was an EMA time constant before the
rolling-median rework, and the experiment is the same either way ---
how much baseline history you average over vs. the drift timescale)
and reports, for each cell:

  * false triggers    — captures with no injected pulse behind them
  * detection efficiency — injected pulses actually recovered
  * stuck fraction    — how much of the run the engine spent inside a
                        capture it could not end (the nastiest 1/f
                        failure: with a frozen baseline the end
                        condition becomes unsatisfiable)

The signal is synthetic — the real TLS generator drives the baseline,
so the wander has a genuine 1/f^alpha spectrum — which keeps the sweep
fast and free of mock-server overhead.

Usage:
  python tls_baseline_sweep.py            # default grid
  python tls_baseline_sweep.py --quick    # smaller/faster grid
"""

import argparse
import sys

import numpy as np

from rfmux.algorithms.measurement.pulse_detection import (
    ChannelNoiseStats,
    PulseCapture,
)
from rfmux.mock.tls_noise import TLSNoiseGenerator

FS = 1000.0             # samples/s
DURATION_S = 60.0
PULSE_TAU_S = 0.02      # 20 ms decay
PULSE_PERIOD_S = 2.0    # one injected pulse every 2 s
PULSE_SNR = 40.0        # peak height in sigma
THRESHOLD_SIGMA = 5.0
END_SIGMA = 1.5
SIGMA = 1.0             # white noise level (counts)


def build_stream(drift_sigma, seed, corner_hz=5.0):
    """White noise + injected pulses + TLS 1/f baseline wander.

    drift_sigma is the RMS baseline wander in units of the white sigma.
    """
    n = int(DURATION_S * FS)
    t = np.arange(n) / FS
    rng = np.random.default_rng(seed)
    sig = rng.normal(0.0, SIGMA, n)

    starts = np.arange(1.0, DURATION_S - 1.0, PULSE_PERIOD_S)
    for t0 in starts:
        m = t >= t0
        sig[m] += PULSE_SNR * SIGMA * np.exp(-(t[m] - t0) / PULSE_TAU_S)

    if drift_sigma > 0:
        gen = TLSNoiseGenerator(
            n_resonators=1, fractional_rms=1.0, alpha=1.0,
            corner_hz=corner_hz, seed=seed + 1)
        wander = gen.values_at(t)[:, 0]
        wander = wander / max(np.std(wander), 1e-30)   # normalise
        sig += drift_sigma * SIGMA * wander

    return sig, t, starts


def run_cell(drift_sigma, track_ms, seed=0, corner_hz=5.0):
    ns = {1: ChannelNoiseStats(mean_I=0.0, std_I=SIGMA,
                               mean_Q=0.0, std_Q=SIGMA)}
    track_samples = int(round(track_ms * 1e-3 * FS)) if track_ms else 0
    detected = []
    pcap = PulseCapture(
        buf_size=int(5 * FS), channels=[1], noise_stats=ns,
        threshold_sigma=THRESHOLD_SIGMA, end_sigma=END_SIGMA,
        margin_fraction=0.1, accumulate=False,
        baseline_window=track_samples,
        on_pulse=lambda ch, idx, pd: detected.append(
            float(np.nanmin(pd["Time"]))),
    )

    sig, t, starts = build_stream(drift_sigma, seed, corner_hz)
    rng_q = np.random.default_rng(seed + 77)
    stuck = 0
    for i in range(len(sig)):
        pcap.process_sample(1, float(sig[i]),
                            float(rng_q.normal(0.0, SIGMA)), float(t[i]))
        if pcap.state[1].capturing:
            stuck += 1

    # A detection counts if it starts within one tau of an injection
    det = np.array(detected)
    matched = 0
    for t0 in starts:
        if len(det) and np.min(np.abs(det - t0)) < 5 * PULSE_TAU_S:
            matched += 1
    false_trigs = max(0, len(det) - matched)
    return {
        "detected": len(det),
        "efficiency": matched / len(starts),
        "false": false_trigs,
        "stuck_frac": stuck / len(sig),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    drift = 5.0                       # baseline wander RMS, in sigma
    corners = [0.05, 0.5, 5.0] if not args.quick else [0.05, 5.0]
    tracks = [0.0, 200.0, 2000.0] if not args.quick else [0.0, 2000.0]

    print(f"Pulses: tau={PULSE_TAU_S*1e3:.0f} ms, SNR={PULSE_SNR:.0f}, "
          f"every {PULSE_PERIOD_S:.0f} s over {DURATION_S:.0f} s")
    print(f"Trigger {THRESHOLD_SIGMA}sigma / end {END_SIGMA}sigma; "
          f"baseline wander {drift:.0f} sigma RMS\n")
    print("The validity window is  pulse << tracking << drift.")
    print("A 1/f corner of f puts drift power down to ~1/f seconds, so a")
    print("high corner leaves no room between the pulse and the drift.\n")
    print(f"{'corner(Hz)':>11} {'drift ts(s)':>12} {'track(ms)':>10} "
          f"{'eff':>7} {'false':>7} {'stuck':>8}")
    print("-" * 62)

    for corner in corners:
        drift_ts = 1.0 / corner
        for tr in tracks:
            r = run_cell(drift, tr, corner_hz=corner)
            healthy = (r["efficiency"] >= 0.9 and r["false"] <= 5
                       and r["stuck_frac"] < 0.2)
            flag = "  OK" if healthy else ""
            if r["stuck_frac"] > 0.5:
                flag = "  <-- STUCK"
            elif r["false"] > 20:
                flag = "  <-- false triggers"
            print(f"{corner:11.2f} {drift_ts:12.1f} {tr:10.0f} "
                  f"{r['efficiency']:7.2f} {r['false']:7d} "
                  f"{r['stuck_frac']:8.2f}{flag}")
        print()

    print("Read the table as: tracking rescues the capture when the")
    print("drift timescale is far longer than the tracking time, which")
    print("must itself be far longer than a pulse. Where the corner is")
    print("high (drift as fast as a few pulse lengths) no tracking time")
    print("works, and a trigger-path high-pass or matched filter would")
    print("be needed instead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
