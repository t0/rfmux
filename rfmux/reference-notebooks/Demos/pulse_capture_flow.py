#!/usr/bin/env python3
"""
Unattended pulse-capture acquisition run.

Captures the same channels three ways — slow stream, fast (PFB) stream,
and both at once with cross-stream matching — and writes one HDF5 file
each.  No plots, no prompts: this is the batch counterpart of the
Pulse Capture panel, for cron jobs, overnight runs and smoke-testing a
board from a terminal.

For what any of these parameters mean and how the detector works, open
pulse_capture.md in this folder as a notebook (double-click it in the
Jupyter panel Periscope launches; in your own JupyterLab, right-click →
Open With → Notebook).  It is the documentation; this is the runner.

Usage:
  python pulse_capture_flow.py MOCK      # simulated CRS
  python pulse_capture_flow.py 0042      # real board serial
"""

import asyncio
import sys

import rfmux
from rfmux.algorithms.measurement.pulse_capture_dual import (
    DualPulseCaptureSession,
)
from rfmux.algorithms.measurement.pulse_capture_session import (
    PulseCaptureConfig,
    PulseCaptureSession,
)
from rfmux.algorithms.measurement.pulse_sources import (
    run_dual_source,
    run_pfb_source,
    run_slow_source,
)
from rfmux.algorithms.measurement.streamer_config import (
    PFB_SAMPLE_RATE,
    StreamerConfig,
    slow_sample_rate,
    validate,
)
from rfmux.streamer import find_streamer_conflict

# ── What to capture ───────────────────────────────────────────────
MODULE = 1
CHANNELS = [1, 2]
PULSE_TAU_S = 1e-3        # expected decay constant; sets the decimation

CONFIG = PulseCaptureConfig(
    threshold_sigma=5.0,
    end_sigma=1.5,
    min_pulse_ms=0.2,
    max_pulse_ms=50.0,
    noise_train_ms=50.0,
    enable_pileup=True,
)

SLOW_S = 2.0              # sample time per capture, seconds
FAST_S = 0.25
DUAL_S = 2.0

MOCK_CONFIG = {
    "num_resonances": 2,
    "resonator_random_seed": 42,
    "auto_bias_kids": True,
    "bias_amplitude": 0.001,
    "pulse_mode": "periodic",
    "pulse_period": 0.05,
    "pulse_tau_rise": 1e-6,
    "pulse_tau_decay": PULSE_TAU_S,
    "pulse_amplitude": 2.0,
}


async def connect(serial: str):
    """Return (crs, host, is_mock) for a serial or "MOCK"."""
    if serial.upper() == "MOCK":
        # Refuse to be the second simulation on the port. Mock streamers all
        # send to 127.0.0.1:9876, so a reader gets both interleaved and every
        # pulse count below is quietly wrong. Cheap to check, and the
        # alternative is noticing it in the numbers days later.
        conflict = find_streamer_conflict()
        if conflict:
            raise RuntimeError(
                f"refusing to start a simulation — {conflict}. "
                "Stop whatever is streaming (a Periscope in mock mode, "
                "another run of this script, a mock server left behind by a "
                "crashed process) and try again.")

        from rfmux.mock.helpers import create_mock_crs
        crs = await create_mock_crs(module=MODULE, config=MOCK_CONFIG,
                                    verbose=False)
        return crs, "127.0.0.1", True

    session = rfmux.load_session(
        f'!HardwareMap [ !CRS {{ serial: "{serial}" }} ]')
    crs = session.query(rfmux.CRS).one()
    await crs.resolve()
    return crs, crs.tuber_hostname, False


async def capture_slow(host, fs, path):
    session = PulseCaptureSession(
        channels=CHANNELS, module=MODULE, streamer_mode="slow",
        sample_rate=fs, hdf5_path=path,
        **CONFIG.session_kwargs(fs))
    session.start()
    covered = await run_slow_source(session, host, module=MODULE,
                                    duration_s=SLOW_S)
    session.stop()
    return session.total_pulses, covered


async def capture_fast(host, path):
    session = PulseCaptureSession(
        channels=CHANNELS, module=MODULE, streamer_mode="fast",
        sample_rate=PFB_SAMPLE_RATE, hdf5_path=path,
        **CONFIG.session_kwargs(PFB_SAMPLE_RATE))
    session.start()
    covered = await run_pfb_source(session, host, CHANNELS,
                                   duration_s=FAST_S)
    session.stop()
    return session.total_pulses, covered


async def capture_dual(host, fs, path):
    session = DualPulseCaptureSession(
        channels=CHANNELS, module=MODULE, slow_rate=fs,
        fast_rate=PFB_SAMPLE_RATE, config=CONFIG, hdf5_path=path)
    session.start()
    covered, _ = await run_dual_source(session, host, CHANNELS,
                                       module=MODULE, duration_s=DUAL_S)
    session.stop()
    return session.stats(), covered


async def main(serial: str = "MOCK") -> int:
    try:
        crs, host, is_mock = await connect(serial)

        # Decimation from the physics: >= 10 samples across one tau.
        needed_fs = 10.0 / PULSE_TAU_S
        dec = next(d for d in range(6, -1, -1)
                   if slow_sample_rate(d) >= needed_fs)
        cfg = StreamerConfig(dec_stage=dec, short_packets=(dec < 3),
                             modules=[MODULE])
        fs = slow_sample_rate(dec)

        for severity, message in validate(cfg):
            print(f"[{severity}] {message}")
        for severity, message in CONFIG.validate(fs):
            print(f"[{severity}] {message}")

        await crs.configure_streamer(dec, short=cfg.short_packets,
                                     modules=[MODULE])
        if is_mock:
            await crs.start_udp_streaming()
            await asyncio.sleep(2.0)

        print(f"slow stream: stage {dec}, {fs:.0f} Hz, "
              f"channels {CHANNELS} on module {MODULE}")

        n, covered = await capture_slow(host, fs, "pulse_flow_slow.h5")
        print(f"slow: {n} pulses in {covered:.2f} s → pulse_flow_slow.h5")

        # The PFB streamer stays off unless a capture needs it.
        await crs.configure_streamer(dec, short=cfg.short_packets,
                                     modules=[MODULE],
                                     pfb_channels=CHANNELS,
                                     pfb_module=MODULE)
        try:
            n, covered = await capture_fast(host, "pulse_flow_fast.h5")
            print(f"fast: {n} pulses in {covered*1e3:.0f} ms "
                  f"→ pulse_flow_fast.h5")

            stats, covered = await capture_dual(host, fs,
                                                "pulse_flow_dual.h5")
            print(f"dual: {stats['slow']['total_pulses']} slow + "
                  f"{stats['fast']['total_pulses']} fast in "
                  f"{covered:.2f} s, {stats['pairs_matched']} matched "
                  f"({stats['pairs_unmatched']} unmatched) "
                  f"→ pulse_flow_dual.h5")
        finally:
            await crs.configure_streamer(dec, short=cfg.short_packets,
                                         modules=[MODULE],
                                         pfb_channels=[])

        if is_mock:
            await crs.stop_udp_streaming()
        return 0

    except Exception as e:
        import traceback
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    serial = sys.argv[1] if len(sys.argv) > 1 else "MOCK"
    sys.exit(asyncio.run(main(serial=serial)))
