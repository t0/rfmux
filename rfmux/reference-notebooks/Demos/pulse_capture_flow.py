#!/usr/bin/env python3
"""
Pulse capture as a plain script.

Captures the same channels three ways — slow stream, fast (PFB) stream,
and both at once with cross-stream matching — and writes one HDF5 file
each.  The same sequence the Pulse Capture panel runs, with no plots and
no prompts.

Read it as a reference for writing your own capture code, or run it
against MOCK to check a change end to end.

For what the parameters mean and how the detector works, open
pulse_capture.md in this folder as a notebook (double-click it in the
Jupyter panel Periscope launches; in your own JupyterLab, right-click →
Open With → Notebook).  Steps below cite its sections.

Usage:
  python pulse_capture_flow.py MOCK      # simulated CRS
  python pulse_capture_flow.py 0042      # real board serial
"""

import asyncio
import sys

import rfmux
from rfmux.pulse_capture import (
    DualPulseCaptureSession,
    PulseCaptureConfig,
    PulseCaptureSession,
    run_dual_source,
    run_pfb_source,
    run_slow_source,
)
from rfmux.core.transferfunctions import (
    PFB_SAMPLING_FREQ,
    decimation_to_sampling,
)
from rfmux.algorithms.measurement.streamer_config import (
    StreamerConfig,
    validate,
)
from rfmux.streamer import find_streamer_conflict

# ── What to capture ───────────────────────────────────────────────
MODULE = 1
CHANNELS = [1, 2]
PULSE_TAU_S = 1e-3        # expected decay constant; sets the decimation

CONFIG = PulseCaptureConfig(
    threshold_sigma=5.0,
    end_sigma=1.0,
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
        # pulse count below is quietly wrong.
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


async def run_capture_flow(crs, host, is_mock) -> int:
    """The sequence, in order.  Steps cite pulse_capture.md sections."""

    # ── Step 1. Pick the decimation from the pulse decay constant ──
    # notebook §3.  Ten or more samples across one tau, or the decay is
    # too sparsely sampled to fit.
    needed_fs = 10.0 / PULSE_TAU_S
    dec = next(d for d in range(6, -1, -1)
               if decimation_to_sampling(d) >= needed_fs)
    fs = decimation_to_sampling(dec)
    cfg = StreamerConfig(dec_stage=dec, short_packets=(dec < 3),
                         modules=[MODULE])

    # ── Step 2. Check the configuration before spending a capture ──
    # notebook §3 and §4.  validate() reports the hardware rules and the
    # link budget; CONFIG.validate() catches inconsistent thresholds.
    for severity, message in validate(cfg):
        print(f"[{severity}] {message}")
    for severity, message in CONFIG.validate(fs):
        print(f"[{severity}] {message}")

    # ── Step 3. Configure the slow streamer ────────────────────────
    # create_mock_crs already started the simulated stream; it needs a
    # moment to settle after the rate change.
    await crs.configure_streamer(dec, short=cfg.short_packets,
                                 modules=[MODULE])
    if is_mock:
        await asyncio.sleep(2.0)
    print(f"slow stream: stage {dec}, {fs:.0f} Hz, "
          f"channels {CHANNELS} on module {MODULE}")

    # ── Step 4. Capture the slow stream ────────────────────────────
    # notebook §6.  Build a session, start it, feed it from a source,
    # stop it.  Every capture below is the same four calls.  No
    # df_calibrations are passed, so the files hold volts on the I/Q
    # axes; with them, trigger_basis defaults to "df" and hertz.
    capture_session = PulseCaptureSession(
        channels=CHANNELS, module=MODULE, streamer_mode="slow",
        sample_rate=fs, hdf5_path="pulse_flow_slow.h5",
        **CONFIG.session_kwargs(fs))
    capture_session.start()
    covered = await run_slow_source(capture_session, host, module=MODULE,
                                    duration_s=SLOW_S)
    capture_session.stop()
    print(f"slow: {capture_session.total_pulses} pulses in {covered:.2f} s "
          f"→ pulse_flow_slow.h5")

    # ── Step 5. Enable the PFB streamer ────────────────────────────
    # It stays off unless a capture needs it, and step 8 turns it back
    # off even if a capture below raises.
    await crs.configure_streamer(dec, short=cfg.short_packets,
                                 modules=[MODULE],
                                 pfb_channels=CHANNELS, pfb_module=MODULE)
    try:
        # ── Step 6. Capture the fast (PFB) stream ──────────────────
        # notebook §8.  Same four calls, at the PFB rate.
        capture_session = PulseCaptureSession(
            channels=CHANNELS, module=MODULE, streamer_mode="fast",
            sample_rate=PFB_SAMPLING_FREQ, hdf5_path="pulse_flow_fast.h5",
            **CONFIG.session_kwargs(PFB_SAMPLING_FREQ))
        capture_session.start()
        covered = await run_pfb_source(capture_session, host, CHANNELS,
                                       module=MODULE, duration_s=FAST_S)
        capture_session.stop()
        print(f"fast: {capture_session.total_pulses} pulses in "
              f"{covered*1e3:.0f} ms → pulse_flow_fast.h5")

        # ── Step 7. Capture both at once, with cross-stream matching ─
        # notebook §9.  A dual session runs two detectors and matches
        # their pulses by trigger time.
        capture_session = DualPulseCaptureSession(
            channels=CHANNELS, module=MODULE, slow_rate=fs,
            fast_rate=PFB_SAMPLING_FREQ, config=CONFIG,
            hdf5_path="pulse_flow_dual.h5")
        capture_session.start()
        covered, _ = await run_dual_source(capture_session, host, CHANNELS,
                                           module=MODULE, duration_s=DUAL_S)
        capture_session.stop()
        stats = capture_session.stats()
        print(f"dual: {stats['slow']['total_pulses']} slow + "
              f"{stats['fast']['total_pulses']} fast in {covered:.2f} s, "
              f"{stats['pairs_matched']} matched "
              f"({stats['pairs_unmatched']} unmatched) "
              f"→ pulse_flow_dual.h5")
    finally:
        # ── Step 8. Disable the PFB streamer ───────────────────────
        await crs.configure_streamer(dec, short=cfg.short_packets,
                                     modules=[MODULE], pfb_channels=[])

    if is_mock:
        await crs.stop_udp_streaming()
    return 0


async def main(serial: str = "MOCK") -> int:
    try:
        crs, host, is_mock = await connect(serial)
        return await run_capture_flow(crs, host, is_mock)
    except Exception as e:
        import traceback
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    serial = sys.argv[1] if len(sys.argv) > 1 else "MOCK"
    sys.exit(asyncio.run(main(serial=serial)))
