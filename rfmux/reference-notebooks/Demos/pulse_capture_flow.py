#!/usr/bin/env python3
"""
Headless pulse-capture flow: streamer configuration → live capture →
streaming HDF5 → histograms, with no GUI involved.

This is the same code path the Periscope "Pulse Capture" panel drives —
PulseCaptureSession fed by run_slow_source / run_pfb_source — so
anything the panel can do, this script can do from a terminal or
notebook.

Steps:
  1. CRS setup (mock: simulated resonators, auto-bias, periodic pulses)
  2. Streamer configuration with the allowable-combination math
  3. Live slow-stream capture → streaming HDF5 + running histograms
  4. Fast (PFB, ~1.22 MHz) capture of the same channels
  5. Pointers: one-shot trigger_capture macro (slow / fast / both)

Usage:
  python pulse_capture_flow.py MOCK      # simulated CRS
  python pulse_capture_flow.py 0042      # real board serial
"""

import asyncio
import sys

import numpy as np

import rfmux
from rfmux.algorithms.measurement.pulse_capture_session import (
    PulseCaptureSession,
)
from rfmux.algorithms.measurement.pulse_hdf5 import PulseHDF5Reader
from rfmux.algorithms.measurement.pulse_sources import (
    run_pfb_source,
    run_slow_source,
)
from rfmux.algorithms.measurement.streamer_config import (
    PFB_SAMPLE_RATE,
    StreamerConfig,
    describe,
    slow_sample_rate,
    validate,
)


async def main(serial: str = "MOCK") -> int:
    MODULE = 1
    CHANNELS = [1, 2]

    # Expected pulse decay time constant — sets the decimation choice:
    # we want >= 10 samples across one tau on the slow stream.
    PULSE_TAU_S = 1e-3

    CAPTURE_PARAMS = dict(
        threshold_sigma=5.0,
        end_sigma=1.5,
        enable_pileup=True,
    )
    SLOW_CAPTURE_S = 2.0     # sample time for the slow live capture
    FAST_CAPTURE_S = 0.25    # sample time for the PFB capture (~5 pulses)

    print("=" * 60)
    print("Pulse capture flow (headless)")
    print("=" * 60)

    try:
        is_mock = serial.upper() == "MOCK"

        # ── 1. CRS setup ──────────────────────────────────────────
        print("\n1. CRS setup")
        if is_mock:
            from rfmux.mock.helpers import create_mock_crs
            crs = await create_mock_crs(
                module=MODULE,
                config={
                    "num_resonances": 2,
                    "resonator_random_seed": 42,
                    "auto_bias_kids": True,
                    "bias_amplitude": 0.001,
                    "pulse_mode": "periodic",
                    "pulse_period": 0.05,
                    "pulse_tau_rise": 1e-6,
                    "pulse_tau_decay": PULSE_TAU_S,
                    "pulse_amplitude": 2.0,
                },
                verbose=False)
            host = "127.0.0.1"
            print("   ✓ MockCRS with 2 auto-biased resonators, "
                  "periodic pulses every 50 ms")
        else:
            session_hwm = rfmux.load_session(
                f'!HardwareMap [ !CRS {{ serial: "{serial}" }} ]')
            crs = session_hwm.query(rfmux.CRS).one()
            await crs.resolve()
            host = crs.tuber_hostname
            print(f"   ✓ CRS {serial} resolved at {host}")
            print("   (detectors must already be biased — see "
                  "simplified_tuning_flow.py)")

        # ── 2. Streamer configuration ─────────────────────────────
        print("\n2. Streamer configuration")
        # Pick the decimation from the physics: >=10 samples per tau.
        needed_fs = 10.0 / PULSE_TAU_S
        dec = next(d for d in range(6, -1, -1)
                   if slow_sample_rate(d) >= needed_fs)
        cfg = StreamerConfig(dec_stage=dec, short_packets=(dec < 3),
                             modules=[MODULE])
        print(f"   Pulse tau {PULSE_TAU_S*1e3:.1f} ms needs fs >= "
              f"{needed_fs:.0f} Hz → stage {dec} "
              f"({slow_sample_rate(dec):.0f} Hz)")
        for sev, msg in validate(cfg):
            print(f"   [{sev}] {msg}")
        info = await crs.configure_streamer(
            cfg.dec_stage, short=cfg.short_packets, modules=cfg.modules)
        print(f"   ✓ Applied: {info['sample_rate_hz']:.0f} Hz, "
              f"{info['channels_per_module']} ch/module, "
              f"{info['total_mbps']:.0f} Mbps")

        if is_mock:
            await crs.start_udp_streaming()
            await asyncio.sleep(2.0)

        # ── 3. Live slow capture → streaming HDF5 ─────────────────
        print("\n3. Live slow-stream capture "
              f"({SLOW_CAPTURE_S:.0f} s of sample time)")
        slow_path = "pulse_flow_slow.h5"
        session = PulseCaptureSession(
            channels=CHANNELS, module=MODULE, streamer_mode="slow",
            sample_rate=slow_sample_rate(cfg.dec_stage),
            noise_samples=1000, hdf5_path=slow_path,
            on_pulse=lambda ch, idx, s, _wf: print(
                f"   pulse ch{ch} #{idx}: {s['snr']:.1f}σ, "
                f"{s['duration_ms']:.2f} ms, τ={s['tau_ms']:.2f} ms"),
            **CAPTURE_PARAMS)
        session.start()
        covered = await run_slow_source(
            session, host, module=MODULE,
            duration_s=SLOW_CAPTURE_S)
        session.stop()
        print(f"   ✓ {session.total_pulses} pulses in {covered:.2f} s "
              f"→ {slow_path}")

        with PulseHDF5Reader(slow_path) as reader:
            hist = reader.get_histograms()
            for ch in CHANNELS:
                n = reader.pulse_count(ch)
                print(f"   Ch {ch}: {n} pulses", end="")
                key = f"tau_ms_counts_ch{ch}"
                if key in hist and np.sum(hist[key]) > 0:
                    centers = hist["tau_ms_bins"]
                    mean_tau = (np.sum(centers * hist[key])
                                / np.sum(hist[key]))
                    print(f", histogram ⟨τ⟩ ≈ {mean_tau:.2f} ms", end="")
                print()

        # ── 4. Fast (PFB) capture ─────────────────────────────────
        print(f"\n4. Fast (PFB) capture ({FAST_CAPTURE_S*1e3:.0f} ms "
              f"at {PFB_SAMPLE_RATE/1e6:.2f} MHz)")
        await crs.configure_streamer(
            cfg.dec_stage, short=cfg.short_packets, modules=[MODULE],
            pfb_channels=CHANNELS, pfb_module=MODULE)
        try:
            fast_path = "pulse_flow_fast.h5"
            fast_session = PulseCaptureSession(
                channels=CHANNELS, module=MODULE, streamer_mode="fast",
                sample_rate=PFB_SAMPLE_RATE, buf_size=200_000,
                noise_samples=50_000, hdf5_path=fast_path,
                threshold_sigma=50.0, end_sigma=3.0)
            fast_session.start()
            covered = await run_pfb_source(
                fast_session, host, CHANNELS,
                duration_s=FAST_CAPTURE_S)
            fast_session.stop()
            print(f"   ✓ {fast_session.total_pulses} pulses in "
                  f"{covered*1e3:.1f} ms → {fast_path}")
        finally:
            await crs.configure_streamer(
                cfg.dec_stage, short=cfg.short_packets,
                modules=[MODULE], pfb_channels=[])
            print("   ✓ PFB streamer disabled")

        # ── 5. Where to go next ───────────────────────────────────
        print("\n5. One-shot captures (no session management):")
        print("   start, pulses, noise = await crs.trigger_capture(")
        print("       channel=[1, 2], module=1, streamer_mode='both',")
        print("       time_run=0.3, threshold_sigma=5.0)")
        print("   → matched slow+fast pulse pairs with cross-stream TOD")

        if is_mock:
            await crs.stop_udp_streaming()
        print("\n✓ Done")
        return 0

    except Exception as e:
        import traceback
        print(f"\n❌ {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    serial = sys.argv[1] if len(sys.argv) > 1 else "MOCK"
    sys.exit(asyncio.run(main(serial=serial)))
