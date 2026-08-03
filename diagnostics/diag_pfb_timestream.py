#!/usr/bin/env python3
"""
Diagnostic: Capture raw PFB and slow-stream I/Q timestreams and plot them.

This bypasses trigger_capture entirely — uses py_run_pfb_streamer and
py_get_samples to get raw time-domain data from both the fast (PFB) and
slow (readout) streamers, then saves plots so we can visually inspect
for pulse signatures and diagnose pulse detection issues.

Both captures use the same mock configuration (same resonators, same
pulse injection).  Differences in pulse visibility help isolate whether
the issue is in the physics simulation, the streamer, or the trigger
detection logic.
"""
import asyncio
import sys
import pathlib
import os
# Run from anywhere: point at the repo root rather than the working directory.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import rfmux


async def main():
    print("=" * 60)
    print("DIAGNOSTIC: Raw PFB + Slow Timestream Plot")
    print("=" * 60)

    # ── 1. Setup ──────────────────────────────────────────────────
    s = rfmux.load_session("""!HardwareMap
- !flavour "rfmux.mock"
- !CRS { serial: "0000", hostname: "127.0.0.1" }
""")
    crs = s.query(rfmux.CRS).one()
    await crs.resolve()
    await crs.set_timestamp_port(crs.TIMESTAMP_PORT.TEST)
    # dec=0 → ~38 kHz slow rate (good time resolution).  Stages below 3 only
    # support short (128-channel) packets, so short=True is mandatory here.
    await crs.set_decimation(0, short=True)

    mock_config = {
        'num_resonances': 2,
        'freq_start': 1.0e9,
        'freq_end': 1.3e9,
        'auto_bias_kids': True,
        'bias_amplitude': 0.001,
        'pulse_mode': 'periodic',
        'pulse_period': 0.01,          # 10 ms period
        'pulse_tau_rise': 1e-6,
        'pulse_tau_decay': 0.001,
        'pulse_amplitude': 2.0,
        # 'pfb_noise_scale': 4,       # Uncomment for reduced noise (easier to see pulses)
    }
    count, freqs = await crs.generate_resonators(mock_config)
    print(f"  {count} resonators generated")

    # Verify auto-bias applied
    f1 = await crs.get_frequency(channel=1, module=1)
    if f1 is None:
        print("  ⚠ WARNING: auto_bias_kids didn't apply channels — data will be noise only")
    else:
        nco = await crs.get_nco_frequency(module=1)
        print(f"  ✓ Auto-bias: NCO={nco/1e9:.3f} GHz, ch1_freq={f1/1e6:.1f} MHz")

    ps = await crs.get_pulse_status()
    print(f"  Pulse mode: {getattr(ps, 'mode', '?')}")

    # Start slow streaming (needed for both slow capture and PFB)
    await crs.start_udp_streaming()
    await asyncio.sleep(1.0)
    print("  Streaming active\n")

    # ── 2. Capture PFB data ───────────────────────────────────────
    pfb_time_run = 0.1  # 0.1s sample time → ~10 pulse periods
    print(f"  [PFB] Capturing {pfb_time_run}s of sample time (channels=[1,2])...")
    pfb_result = await crs.py_run_pfb_streamer(channel=[1, 2], module=1, time_run=pfb_time_run)

    if pfb_result is None:
        print("  FAIL: No PFB result")
        await crs.stop_udp_streaming()
        return

    pfb_i_ch0 = np.array(pfb_result.i[0])
    pfb_q_ch0 = np.array(pfb_result.q[0])
    pfb_i_ch1 = np.array(pfb_result.i[1])
    pfb_q_ch1 = np.array(pfb_result.q[1])

    pfb_rate = 625e6 / 512  # ~1.22 MHz
    pfb_t_ch0 = np.arange(len(pfb_i_ch0)) / pfb_rate * 1e3  # ms

    print(f"  [PFB] Channel 0: {len(pfb_i_ch0)} samples")
    print(f"    I: mean={pfb_i_ch0.mean():.1f}, std={pfb_i_ch0.std():.2f}")
    print(f"    Q: mean={pfb_q_ch0.mean():.1f}, std={pfb_q_ch0.std():.2f}")
    print(f"  [PFB] Channel 1: {len(pfb_i_ch1)} samples")
    print(f"    I: mean={pfb_i_ch1.mean():.1f}, std={pfb_i_ch1.std():.2f}")
    print(f"    Q: mean={pfb_q_ch1.mean():.1f}, std={pfb_q_ch1.std():.2f}")

    # PFB deviation stats
    for label, arr in [("PFB ch0 I", pfb_i_ch0), ("PFB ch0 Q", pfb_q_ch0),
                       ("PFB ch1 I", pfb_i_ch1), ("PFB ch1 Q", pfb_q_ch1)]:
        dev = np.abs(arr - arr.mean()) / max(arr.std(), 1e-30)
        print(f"    {label}: max_dev={dev.max():.1f}σ, samples>3σ={np.sum(dev > 3)}, samples>5σ={np.sum(dev > 5)}")

    # ── 3. Capture slow-stream data ──────────────────────────────
    # Use py_get_samples (UDP multicast streaming capture, same as trigger_capture)
    dec = await crs.get_decimation()
    slow_rate = 625e6 / 256 / 64 / (2**dec)
    # Capture enough samples to cover the same simulation time as the PFB capture
    slow_num_samples = max(int(pfb_time_run * slow_rate), 100)

    print(f"\n  [Slow] Capturing {slow_num_samples} samples at dec={dec} "
          f"(rate={slow_rate:.0f} Hz, ~{slow_num_samples/slow_rate:.3f}s)...")

    slow_result = await crs.py_get_samples(
        num_samples=slow_num_samples,
        channel=1,   # channel 1
        module=1,
    )

    slow_i_ch1 = np.array(slow_result.i)
    slow_q_ch1 = np.array(slow_result.q)
    slow_t_ch1 = np.arange(len(slow_i_ch1)) / slow_rate * 1e3  # ms

    # Also get channel 2
    slow_result2 = await crs.py_get_samples(
        num_samples=slow_num_samples,
        channel=2,
        module=1,
    )

    slow_i_ch2 = np.array(slow_result2.i)
    slow_q_ch2 = np.array(slow_result2.q)
    slow_t_ch2 = np.arange(len(slow_i_ch2)) / slow_rate * 1e3  # ms

    print(f"  [Slow] Channel 1: {len(slow_i_ch1)} samples")
    print(f"    I: mean={slow_i_ch1.mean():.1f}, std={slow_i_ch1.std():.2f}")
    print(f"    Q: mean={slow_q_ch1.mean():.1f}, std={slow_q_ch1.std():.2f}")
    print(f"  [Slow] Channel 2: {len(slow_i_ch2)} samples")
    print(f"    I: mean={slow_i_ch2.mean():.1f}, std={slow_i_ch2.std():.2f}")
    print(f"    Q: mean={slow_q_ch2.mean():.1f}, std={slow_q_ch2.std():.2f}")

    # Slow deviation stats
    for label, arr in [("Slow ch1 I", slow_i_ch1), ("Slow ch1 Q", slow_q_ch1),
                       ("Slow ch2 I", slow_i_ch2), ("Slow ch2 Q", slow_q_ch2)]:
        dev = np.abs(arr - arr.mean()) / max(arr.std(), 1e-30)
        print(f"    {label}: max_dev={dev.max():.1f}σ, samples>3σ={np.sum(dev > 3)}, samples>5σ={np.sum(dev > 5)}")

    # ── 4. Generate plots ─────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Raw Timestream Diagnostic — PFB (fast) + Slow (readout)', fontsize=14)

    # Row 0: PFB Channel 0
    axes[0, 0].plot(pfb_t_ch0, pfb_i_ch0, linewidth=0.3)
    axes[0, 0].set_title(f'PFB Ch 0 — I  ({len(pfb_i_ch0)} samp, std={pfb_i_ch0.std():.1f})')
    axes[0, 0].set_ylabel('ADC counts')

    axes[0, 1].plot(pfb_t_ch0[:len(pfb_q_ch0)], pfb_q_ch0, linewidth=0.3, color='orange')
    axes[0, 1].set_title(f'PFB Ch 0 — Q  (std={pfb_q_ch0.std():.1f})')

    # Row 1: PFB Channel 1
    pfb_t_ch1 = np.arange(len(pfb_i_ch1)) / pfb_rate * 1e3
    axes[1, 0].plot(pfb_t_ch1, pfb_i_ch1, linewidth=0.3)
    axes[1, 0].set_title(f'PFB Ch 1 — I  ({len(pfb_i_ch1)} samp, std={pfb_i_ch1.std():.1f})')
    axes[1, 0].set_ylabel('ADC counts')

    axes[1, 1].plot(pfb_t_ch1[:len(pfb_q_ch1)], pfb_q_ch1, linewidth=0.3, color='orange')
    axes[1, 1].set_title(f'PFB Ch 1 — Q  (std={pfb_q_ch1.std():.1f})')

    # Row 2: Slow stream Channels 1 & 2
    axes[2, 0].plot(slow_t_ch1, slow_i_ch1, linewidth=0.5, label='Ch1 I')
    axes[2, 0].plot(slow_t_ch1, slow_q_ch1, linewidth=0.5, label='Ch1 Q', alpha=0.7)
    axes[2, 0].set_title(f'Slow Ch 1 — I/Q  ({len(slow_i_ch1)} samp, rate={slow_rate:.0f} Hz)')
    axes[2, 0].set_ylabel('ADC counts')
    axes[2, 0].set_xlabel('Time (ms)')
    axes[2, 0].legend(fontsize=8)

    axes[2, 1].plot(slow_t_ch2, slow_i_ch2, linewidth=0.5, label='Ch2 I')
    axes[2, 1].plot(slow_t_ch2, slow_q_ch2, linewidth=0.5, label='Ch2 Q', alpha=0.7)
    axes[2, 1].set_title(f'Slow Ch 2 — I/Q  ({len(slow_i_ch2)} samp, rate={slow_rate:.0f} Hz)')
    axes[2, 1].set_xlabel('Time (ms)')
    axes[2, 1].legend(fontsize=8)

    # Set x-axis labels for PFB rows
    for r in range(2):
        axes[r, 0].set_xlabel('')
        axes[r, 1].set_xlabel('')
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 1].set_xlabel('Time (ms)')

    plt.tight_layout()
    outpath = os.path.join(os.path.dirname(__file__), 'pfb_timestream_diagnostic.png')
    plt.savefig(outpath, dpi=150)
    print(f"\n  Plot saved: {outpath}")

    # ── 5. Summary for noise estimation debugging ─────────────────
    print("\n" + "=" * 60)
    print("NOISE ESTIMATION DIAGNOSTIC")
    print("=" * 60)
    print("  If pulses are present, the global std includes pulse energy.")
    print("  This inflates the noise estimate and raises the trigger threshold,")
    print("  potentially hiding pulses from detection.")
    print()
    for label, arr in [("PFB ch0 I", pfb_i_ch0), ("Slow ch1 I", slow_i_ch1)]:
        median_abs_dev = np.median(np.abs(arr - np.median(arr)))
        robust_std = 1.4826 * median_abs_dev  # MAD → σ estimator
        naive_std = arr.std()
        ratio = naive_std / max(robust_std, 1e-30)
        print(f"  {label}: naive_std={naive_std:.2f}, robust_std(MAD)={robust_std:.2f}, "
              f"ratio={ratio:.2f} {'⚠ pulses inflating std!' if ratio > 1.3 else '✓'}")

    # Cleanup
    await crs.stop_udp_streaming()
    print("\n  Done.")


if __name__ == "__main__":
    asyncio.run(main())
