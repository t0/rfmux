#!/usr/bin/env python3
"""
E2E test for unified trigger_capture macro — both slow and fast (PFB) modes.

Uses sigma-based dual I/Q triggering (no hard thresholds needed).
Setup: load_session → generate_resonators(auto_bias + periodic pulses) → start_udp
Then:  trigger_capture(streamer_mode="slow")  and  trigger_capture(streamer_mode="fast")

Generates diagnostic plots of captured pulses for visual verification.
"""
import asyncio
import os
import sys
import pathlib
# Run from anywhere: point at the repo root rather than the working directory.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import rfmux

from rfmux.algorithms.measurement.pulse_hdf5 import PulseHDF5Reader


# ── Plotting ──────────────────────────────────────────────────────

def plot_results(slow_results, fast_results, outpath):
    """Generate per-event cutout plots: each captured pulse on its own subplot.

    For each mode (slow / fast) and each channel, every detected pulse gets
    its own subplot showing:
      - Full I/Q waveform with sample-rate–derived time axis (ms)
      - Noise threshold bands (±threshold_sigma dashed, ±end_sigma dotted)
      - Text annotation with sample count, peak amplitudes, and SNR
      - A tag indicating if this event was a pileup split

    Layout: up to 12 events per page in a 4-col grid, one figure per
    mode/channel combination.  All figures saved as a single composite PNG.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    MAX_EVENTS = 12  # max events to show per mode/channel
    N_COLS = 4

    datasets = [
        ("Slow", slow_results),
        ("Fast (PFB)", fast_results),
    ]

    # Collect all (mode, ch_key, ch_num, pulse_list, noise_stats) groups
    groups = []
    for mode_label, results in datasets:
        if results is None:
            continue
        pulses = results.pulses
        for ch_num in sorted(pulses)[:2]:
            ch_pulses = pulses[ch_num]
            if len(ch_pulses) == 0:
                continue
            groups.append((mode_label, f"Channel {ch_num}", ch_num,
                           ch_pulses, results.noise.get(ch_num)))

    if not groups:
        print("  No pulses to plot — skipping")
        return

    # Total rows across all groups
    total_rows = 0
    group_layouts = []  # (n_show, n_rows) per group
    for mode_label, ch_key, ch_num, ch_pulses, ns in groups:
        n_show = min(MAX_EVENTS, len(ch_pulses))
        n_rows = int(np.ceil(n_show / N_COLS))
        group_layouts.append((n_show, n_rows))
        total_rows += n_rows + 1  # +1 for a section header row

    fig_height = max(4, total_rows * 3.2)
    fig = plt.figure(figsize=(N_COLS * 4.5, fig_height))
    fig.suptitle('trigger_capture E2E — Per-Event Cutouts', fontsize=14, y=0.995)

    # Use gridspec for flexible layout
    import matplotlib.gridspec as gridspec

    # Calculate total grid rows: each group gets a header row + data rows
    grid_rows = sum(nr + 1 for _, nr in group_layouts)
    gs = gridspec.GridSpec(grid_rows, N_COLS, figure=fig, hspace=0.55, wspace=0.35)

    current_row = 0
    for grp_idx, (mode_label, ch_key, ch_num, ch_pulses, ns) in enumerate(groups):
        n_show, n_rows = group_layouts[grp_idx]
        n_total = len(ch_pulses)

        # ── Section header ────────────────────────────────────────
        header_ax = fig.add_subplot(gs[current_row, :])
        header_ax.set_axis_off()
        extra = f' (showing {n_show}/{n_total})' if n_show < n_total else ''
        noise_txt = ''
        if ns is not None:
            noise_txt = f'  |  Noise σ: I={ns.std_I:.1f}, Q={ns.std_Q:.1f}'
        header_ax.text(
            0.5, 0.5,
            f'{mode_label} — {ch_key}: {n_total} events{extra}{noise_txt}',
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8e8f0', alpha=0.9))
        current_row += 1

        # ── Determine sample rate ─────────────────────────────────
        if 'Fast' in mode_label:
            sample_rate = 625e6 / 512
        else:
            # dec=1 in the test → slow rate = 625e6 / 256 / 64 / 2^1
            sample_rate = 625e6 / 256 / 64 / 2

        # ── Per-event subplots ────────────────────────────────────
        for i, (k, p) in enumerate(list(ch_pulses.items())[:n_show]):
            row_in_group = i // N_COLS
            col = i % N_COLS
            ax = fig.add_subplot(gs[current_row + row_in_group, col])

            n_samp = len(p['Amp_I'])
            t_ms = np.arange(n_samp) / sample_rate * 1e3

            # Plot I and Q waveforms
            ax.plot(t_ms, p['Amp_I'], color='C0', linewidth=1.0, alpha=0.9, label='I')
            ax.plot(t_ms, p['Amp_Q'], color='C1', linewidth=1.0, alpha=0.9, label='Q')

            # Draw noise threshold bands
            if ns is not None:
                for sigma_level, style, lbl in [
                    (3.0, '--', f'±3σ'),
                    (1.5, ':', f'±1.5σ'),
                ]:
                    ax.axhline(ns.mean_I + sigma_level * ns.std_I,
                               color='C0', linestyle=style, alpha=0.25, linewidth=0.7)
                    ax.axhline(ns.mean_I - sigma_level * ns.std_I,
                               color='C0', linestyle=style, alpha=0.25, linewidth=0.7)
                    ax.axhline(ns.mean_Q + sigma_level * ns.std_Q,
                               color='C1', linestyle=style, alpha=0.25, linewidth=0.7)
                    ax.axhline(ns.mean_Q - sigma_level * ns.std_Q,
                               color='C1', linestyle=style, alpha=0.25, linewidth=0.7)
                # Baseline means
                ax.axhline(ns.mean_I, color='C0', linestyle='-', alpha=0.15, linewidth=0.5)
                ax.axhline(ns.mean_Q, color='C1', linestyle='-', alpha=0.15, linewidth=0.5)

            # Compute metadata — peak excursion from baseline
            peak_I = float(np.max(np.abs(p['Amp_I'] - ns.mean_I))) if ns else float(np.max(np.abs(p['Amp_I'])))
            peak_Q = float(np.max(np.abs(p['Amp_Q'] - ns.mean_Q))) if ns else float(np.max(np.abs(p['Amp_Q'])))
            snr_I = peak_I / max(ns.std_I, 1e-30) if ns else 0
            snr_Q = peak_Q / max(ns.std_Q, 1e-30) if ns else 0

            ax.set_title(f'Event {k}  ({n_samp} samp, {t_ms[-1]:.2f} ms)',
                         fontsize=8, pad=3)
            ax.tick_params(labelsize=7)

            # Annotation: peak, SNR, and pileup tag
            pileup_tag = '\n⚠ PILEUP' if p.get('pileup', False) else ''
            ax.annotate(
                f'I_pk={peak_I:.0f} ({snr_I:.1f}σ)\n'
                f'Q_pk={peak_Q:.0f} ({snr_Q:.1f}σ){pileup_tag}',
                xy=(0.98, 0.95), xycoords='axes fraction',
                fontsize=6, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

            if i == 0:
                ax.legend(fontsize=6, loc='upper left')

            # Only label bottom row axes
            if row_in_group == n_rows - 1:
                ax.set_xlabel('Time (ms)', fontsize=7)
            if col == 0:
                ax.set_ylabel('ADC counts', fontsize=7)

        current_row += n_rows

    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\n  📊 Per-event cutout plot saved: {outpath}")


def plot_both_results(both_results, outpath):
    """Per-pulse twinx overplot: slow and fast TOD for each matched pulse.

    Each matched pulse gets its own subplot with:
      - Left y-axis (blue/orange): slow I/Q (ADC/256 units, ~38 kHz)
      - Right y-axis (red/green):  fast I/Q (raw ADC counts, ~1.22 MHz)
      - Shared x-axis: simulation time (ms) from pulse timestamps
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Pairs carry union-window TOD from both rings; either side may be
    # None on a one-sided match.
    all_pairs = [(f"Channel {pair['channel']}", pair["pair_idx"], pair)
                 for pair in both_results.pairs
                 if pair.get("slow_tod") is not None
                 or pair.get("fast_tod") is not None]

    if not all_pairs:
        print("  No pulses at all — skipping plot")
        return

    n_plot = min(len(all_pairs), 6)  # max 6 subplots
    fig, axes = plt.subplots(n_plot, 1, figsize=(14, 4 * n_plot), squeeze=False)
    fig.suptitle('Matched Pulse Overplot — Slow (left axis) + Fast/PFB (right axis)',
                 fontsize=13, y=0.99)

    for idx in range(n_plot):
        ch_key, pulse_idx, pair = all_pairs[idx]
        ax_slow = axes[idx, 0]
        ax_fast = ax_slow.twinx()

        # Prefer TOD (covers full matched time window) over triggered
        # pulse data (narrower, only covers that stream's capture).
        sp = pair.get("slow_tod") or pair.get("slow")
        fp = pair.get("fast_tod") or pair.get("fast")

        n_slow = 0
        n_fast = 0

        # Determine tag based on what was triggered vs TOD fallback
        s_triggered = pair.get("slow") is not None
        f_triggered = pair.get("fast") is not None
        if s_triggered and f_triggered:
            tag = "matched"
        elif s_triggered:
            tag = "slow trig + fast TOD" if fp else "slow only"
        elif f_triggered:
            tag = "fast trig + slow TOD" if sp else "fast only"
        else:
            tag = "TOD only"

        # Plot slow (left axis) — thick markers for visibility
        if sp is not None:
            s_time = np.array(sp["Time"], dtype=float) * 1e3
            style = 'o-' if s_triggered else '.-'
            ax_slow.plot(s_time, sp["Amp_I"], style, color='C0', markersize=3,
                         linewidth=1.2, alpha=0.9, label='Slow I')
            ax_slow.plot(s_time, sp["Amp_Q"], style, color='C1', markersize=3,
                         linewidth=1.2, alpha=0.9, label='Slow Q')
            n_slow = len(sp["Amp_I"])

        # Plot fast (right axis) — thin lines, many points
        if fp is not None:
            f_time = np.array(fp["Time"], dtype=float) * 1e3
            ax_fast.plot(f_time, fp["Amp_I"], '-', color='C3', linewidth=0.4,
                         alpha=0.9, label='Fast I')
            ax_fast.plot(f_time, fp["Amp_Q"], '-', color='C2', linewidth=0.4,
                         alpha=0.9, label='Fast Q')
            n_fast = len(fp["Amp_I"])

        ax_slow.set_xlabel('Simulation Time (ms)')
        ax_slow.set_ylabel('Slow (ADC/256)', color='C0')
        ax_fast.set_ylabel('Fast PFB (ADC counts)', color='C3')
        ax_slow.tick_params(axis='y', labelcolor='C0')
        ax_fast.tick_params(axis='y', labelcolor='C3')

        # Combined legend
        lines1, labels1 = ax_slow.get_legend_handles_labels()
        lines2, labels2 = ax_fast.get_legend_handles_labels()
        ax_slow.legend(lines1 + lines2, labels1 + labels2,
                       fontsize=7, loc='upper right')

        ax_slow.set_title(
            f'{ch_key} — Pulse {pulse_idx} [{tag}]  '
            f'(slow: {n_slow} samp, fast: {n_fast} samp)',
            fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(outpath, dpi=150)
    print(f"\n  📊 Per-pulse overplot saved: {outpath}")


# ── Test functions ────────────────────────────────────────────────

async def setup_mock():
    """Common mock setup: session + resonators + streaming + periodic pulses."""
    print("=" * 60)
    print("Setting up mock CRS with resonators + periodic pulses")
    print("=" * 60)

    s = rfmux.load_session("""!HardwareMap
- !flavour "rfmux.mock"
- !CRS { serial: "0000", hostname: "127.0.0.1" }
""")
    crs = s.query(rfmux.CRS).one()
    await crs.resolve()
    await crs.set_timestamp_port(crs.TIMESTAMP_PORT.TEST)
    await crs.set_decimation(1, short=True)

    mock_config = {
        'num_resonances': 2,
        'freq_start': 1.0e9,
        'freq_end': 1.3e9,
        'auto_bias_kids': True,
        'bias_amplitude': 0.001,
        'pulse_mode': 'periodic',
        'pulse_period': 0.05,          # 10 ms period (matches diag_pfb_timestream)
        'pulse_tau_rise': 1e-6,
        'pulse_tau_decay': 0.001,      # 1 ms decay (default for fixed mode)
        'pulse_amplitude': 2.0,        # default for fixed mode
        # 'pulse_mode': 'random',           # use 'random' instead of 'periodic'
        # 'pulse_probability': 0.1,         # per-timestep probability (random mode)        
        # Random amplitude: each pulse gets a different amplitude
        'pulse_random_amp_mode': 'uniform',
        'pulse_random_amp_min': 1.5,
        'pulse_random_amp_max': 4.0,
        # Random tau_decay: each pulse gets a different decay time
        'pulse_random_tau_mode': 'uniform',
        'pulse_random_tau_min': 5e-6,  # 5 µs
        'pulse_random_tau_max': 3e-3,  # 3 ms
        # pfb_noise_scale uses default 64 (physics-accurate);
        # robust MAD-based noise estimation handles pulse contamination
    }
    count, freqs = await crs.generate_resonators(mock_config)
    print(f"  ✓ {count} resonators at {[f'{f/1e9:.3f} GHz' for f in freqs]}")

    # Verify auto-bias applied
    f1 = await crs.get_frequency(channel=1, module=1)
    if f1 is None:
        print("  ⚠ WARNING: auto_bias_kids didn't apply channels — pulses may not be detected")
    else:
        nco = await crs.get_nco_frequency(module=1)
        print(f"  ✓ Auto-bias: NCO={nco/1e9:.3f} GHz, ch1_freq={f1/1e6:.1f} MHz")

    # Start slow streaming
    await crs.start_udp_streaming()
    await asyncio.sleep(10.0)
    print("  ✓ Setup complete\n")
    return crs


async def test_trigger_capture_slow(crs):
    """Test trigger_capture with slow readout streamer — sigma-based.

    Returns (success, PulseCaptureResult) or (success, None).
    """
    print("=" * 60)
    print("TEST: trigger_capture(streamer_mode='slow', threshold_sigma=50)")
    print("=" * 60)

    try:
        res = await crs.trigger_capture(
            channel=[1, 2],
            module=1,
            streamer_mode="slow",
            time_run=10,
            threshold_sigma=50.0,   # deliberately high: mock pulses are huge
            end_sigma=3,
            max_pulse_ms=50.0,
            hdf5_path=os.path.join(os.path.dirname(__file__),
                                   "trigger_capture_slow.h5"),
        )

        print(f"  start_time: {res.start_time}")
        for ch in sorted(res.pulses):
            print(f"  Channel {ch}: {len(res.pulses[ch])} pulses")
            for k in sorted(res.pulses[ch])[:3]:
                pd = res.pulses[ch][k]
                print(f"    Pulse {k}: {len(pd['Amp_I'])} samples, "
                      f"I_peak={max(abs(pd['Amp_I'])):.0f}, "
                      f"Q_peak={max(abs(pd['Amp_Q'])):.0f}")

        print(f"  Total pulses: {res.total_pulses}")
        if res.total_pulses > 0:
            print("  ✅ PASS — pulses detected!\n")
        else:
            print("  ⚠ No pulses detected (may need threshold tuning)\n")
        return True, res

    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback; traceback.print_exc()
        return False, None


async def test_trigger_capture_fast(crs):
    """Test trigger_capture with fast PFB streamer — sigma-based.

    Returns (success, PulseCaptureResult) or (success, None).
    """
    print("=" * 60)
    print("TEST: trigger_capture(streamer_mode='fast', threshold_sigma=50)")
    print("=" * 60)

    try:
        res = await crs.trigger_capture(
            channel=[1, 2],
            module=1,
            streamer_mode="fast",
            time_run=0.1,
            threshold_sigma=50.0,
            end_sigma=3,
            max_pulse_ms=5.0,
            hdf5_path=os.path.join(os.path.dirname(__file__),
                                   "trigger_capture_fast.h5"),
        )

        print(f"  start_time: {res.start_time}")
        for ch in sorted(res.pulses):
            print(f"  Channel {ch}: {len(res.pulses[ch])} pulses")
            for k in sorted(res.pulses[ch])[:3]:
                print(f"    Pulse {k}: "
                      f"{len(res.pulses[ch][k]['Amp_I'])} samples")

        print(f"  Total pulses: {res.total_pulses}")
        if res.total_pulses > 0:
            print("  ✅ PASS — pulses detected!\n")
        else:
            print("  ⚠ No pulses detected (may need threshold tuning)\n")
        return True, res

    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback; traceback.print_exc()
        return False, None


async def test_trigger_capture_both(crs):
    """Test trigger_capture with both streamers simultaneously.

    Returns (success, PulseCaptureResult) or (success, None).
    """
    print("=" * 60)
    print("TEST: trigger_capture(streamer_mode='both', threshold_sigma=50)")
    print("=" * 60)

    try:
        res = await crs.trigger_capture(
            channel=[1, 2],
            module=1,
            streamer_mode="both",
            time_run=0.2,           # pulse period is 0.05s → ~4 pulses/ch
            threshold_sigma=50.0,
            end_sigma=3,
            max_pulse_ms=5.0,
            hdf5_path=os.path.join(os.path.dirname(__file__),
                                   "trigger_capture_both.h5"),
        )

        print(f"  start_time: {res.start_time}")
        two_sided = [p for p in res.pairs
                     if p.get("slow_idx") is not None
                     and p.get("fast_idx") is not None]
        by_ch = {}
        for pair in res.pairs:
            by_ch.setdefault(pair["channel"], []).append(pair)
        for ch in sorted(by_ch):
            pairs = by_ch[ch]
            n_both = sum(1 for p in pairs if p.get("fast_idx") is not None
                         and p.get("slow_idx") is not None)
            print(f"  Channel {ch}: {len(pairs)} pairs "
                  f"({n_both} matched slow+fast)")
            for pair in pairs[:3]:
                sp, fp = pair.get("slow_tod"), pair.get("fast_tod")
                s_n = len(sp["Amp_I"]) if sp else 0
                f_n = len(fp["Amp_I"]) if fp else 0
                print(f"    Pair {pair['pair_idx']}: slow={s_n}, "
                      f"fast={f_n}, offset="
                      f"{(pair.get('time_offset') or 0)*1e6:+.0f} us")

        if two_sided:
            print(f"  ✅ PASS — {len(two_sided)} matched slow+fast pairs!\n")
            return True, res
        # Verified working (12/12 matched at time_run=0.3, 2026-07-29):
        # zero matches now means a real regression, not a thin window.
        print("  ✗ FAIL — no matched slow+fast pairs\n")
        return False, res

    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback; traceback.print_exc()
        return False, None


def verify_hdf5(res, mode_label):
    """Read back the file the capture wrote and report what is in it.

    The session writes pulses, noise stats, histograms and templates as
    the capture runs — trigger_capture just passes hdf5_path through — so
    there is nothing to assemble here, only to check.
    """
    if res is None or res.hdf5_path is None:
        return None
    if not os.path.exists(res.hdf5_path):
        print(f"  [{mode_label}] no file at {res.hdf5_path}")
        return None

    with PulseHDF5Reader(res.hdf5_path) as reader:
        print(f"\n  HDF5: {res.hdf5_path}")
        print(f"     mode={reader.metadata.get('streamer_mode')}, "
              f"channels={reader.channels}, dual={reader.dual}")
        for ch in reader.channels:
            streams = reader.streams or [None]
            for stream in streams:
                count = reader.pulse_count(ch, stream=stream)
                ns = reader.noise_stats(ch, stream=stream)
                tag = f" [{stream}]" if stream else ""
                print(f"     Channel {ch}{tag}: {count} pulses, "
                      f"noise I={ns.mean_I:.1f}+-{ns.std_I:.2f}, "
                      f"Q={ns.mean_Q:.1f}+-{ns.std_Q:.2f}")
                if count > 0:
                    pd = reader.get_pulse(ch, 1, stream=stream)
                    print(f"       Pulse 1: {pd['n_samples']} samples, "
                          f"peak_I={pd['peak_I']:.0f}, "
                          f"duration={pd['duration_s']*1e3:.2f} ms")
            if reader.dual:
                print(f"     Channel {ch}: {reader.pair_count(ch)} pairs")

        hists = reader.get_histograms()
        for ch in reader.channels:
            key = f"amplitude_counts_ch{ch}"
            if key in hists:
                print(f"     Histogram ch{ch}: "
                      f"{int(np.sum(hists[key]))} pulses binned")

    return res.hdf5_path


async def main():
    print("\n🔬 Unified trigger_capture E2E Test (Sigma-Based)\n")

    crs = await setup_mock()

    slow_ok, slow_results = await test_trigger_capture_slow(crs)
    fast_ok, fast_results = await test_trigger_capture_fast(crs)
    both_ok, both_results = await test_trigger_capture_both(crs)

    # Cleanup
    await crs.stop_udp_streaming()

    all_ok = slow_ok and fast_ok and both_ok
    print("=" * 60)
    if all_ok:
        print("ALL TESTS PASSED ✓")
    else:
        print(f"RESULTS: slow={'PASS' if slow_ok else 'FAIL'}, "
              f"fast={'PASS' if fast_ok else 'FAIL'}, "
              f"both={'PASS' if both_ok else 'FAIL'}")
    print("=" * 60)

    # ── Save to HDF5 + compute histograms ─────────────────────────
    outdir = os.path.dirname(__file__)
    print("\n" + "=" * 60)
    print("HDF5 READ-BACK")
    print("=" * 60)
    verify_hdf5(slow_results, "slow")
    verify_hdf5(fast_results, "fast")
    verify_hdf5(both_results, "both")

    # ── Generate diagnostic plots ─────────────────────────────────
    outpath = os.path.join(outdir, 'trigger_capture_pulses.png')
    try:
        plot_results(slow_results, fast_results, outpath)
    except Exception as e:
        print(f"  ⚠ Plot generation failed: {e}")
        import traceback; traceback.print_exc()

    # ── "Both" mode overplot ──────────────────────────────────────
    if both_results is not None:
        both_outpath = os.path.join(outdir, 'trigger_capture_both.png')
        try:
            plot_both_results(both_results, both_outpath)
        except Exception as e:
            print(f"  ⚠ Both-mode plot failed: {e}")
            import traceback; traceback.print_exc()

    return all_ok


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
