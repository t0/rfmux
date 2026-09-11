#!/usr/bin/env python3
"""Tune an array, apply its bias catalog, and acquire slow and PFB noise.

Usage:
    python simplified_tuning_flow.py                       # fresh mock array
    python simplified_tuning_flow.py MOCK --output /tmp/tuning
    python simplified_tuning_flow.py 0042 --module 1        # real board
    python simplified_tuning_flow.py 0042 --hostname crs0042.local
    python simplified_tuning_flow.py ATTACHED              # Periscope's CRS

Edit the measurement settings below for your real array and RF attenuation.
Hardware needs a configured clock/timestamp source and a UDP readout stream
that reaches this computer. Only a mock created here is started and stopped.
The mock PFB RPC capture returns uniform synthetic noise, not detector noise.

See simplified_tuning_flow.md for the same workflow with plots and explanations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from tuber.codecs import TuberResult

import rfmux
from rfmux.core.resonators import ResonatorCatalog
from rfmux.core.transferfunctions import (
    PFB_SAMPLING_FREQ, decimation_to_sampling,
)
from rfmux.streamer import find_streamer_conflict
from rfmux.tuning import (
    AmplitudeSchedule, BiasReport, find_bias_points,
    find_resonances_in_netanal, store,
)

# Match the notebook's compact, unbiased array and measurement settings.
MOCK_CONFIG = {
    "num_resonances": 10,
    "freq_start": 601e6,
    "freq_end": 608e6,
    "C_variation": 0.0001,
    "resonator_random_seed": 42,
    "auto_bias_kids": False,
    "pulse_mode": "none",
    "tls_noise_enabled": False,
    "nqp_noise_std_factor": 0.001,
    "T": 0.23,
}
NETANAL_PARAMS = dict(
    fmin=600e6, fmax=610e6, npoints=2_000, amp=0.001,
    nsamps=10, max_chans=1023,
)
FIND_RES_PARAMS = dict(
    min_dip_depth_db=1.0, min_Q=1e4, max_Q=1e7,
    min_separation_hz=100e3,
)
MULTISWEEP_PARAMS = dict(span_hz=100e3, npoints_per_sweep=201, nsamps=10)
SCHEDULE = AmplitudeSchedule.ramp(0.002, 0.032, 5)
BIAS_SETTINGS = dict(
    frequency_method="iq_derivative", direction="upward",
    spike_prominence_factor=0.5, noise_gate_factor=50.0,
    max_discrepancy=0.1, compare="magnitude",
)
SLOW_NOISE_PARAMS = dict(
    num_samples=1_000, channel=None, return_spectrum=True,
    scaling="psd", reference="absolute", nsegments=5, spectrum_cutoff=0.9,
)
PFB_NOISE_PARAMS = dict(
    nsamps=20_000, binlim=1e6, trim=False, nsegments=5,
    reference="absolute", reset_NCO=False,
)


async def _connect(
    serial: str, hostname: str | None = None,
) -> tuple[rfmux.CRS, bool]:
    created_mock = serial.upper() == "MOCK"
    if created_mock:
        session = rfmux.load_session('''
!HardwareMap
- !flavour "rfmux.mock"
- !CRS { serial: "0000", hostname: "127.0.0.1" }
''')
    else:
        if serial.upper() == "ATTACHED":
            hostname = hostname or os.environ.get("RFMUX_CRS_HOSTNAME")
            if not hostname:
                raise ValueError(
                    "ATTACHED needs RFMUX_CRS_HOSTNAME or --hostname.")
            serial = os.environ.get("RFMUX_CRS_SERIAL", "0000")
        address = f", hostname: {json.dumps(hostname)}" if hostname else ""
        session = rfmux.load_session(
            f'!HardwareMap [ !CRS {{ serial: {json.dumps(serial)}{address} }} ]')
    crs = session.query(rfmux.CRS).one()
    await crs.resolve()
    if created_mock:
        from rfmux.mock.config import apply_overrides

        count, _ = await crs.generate_resonators(apply_overrides(MOCK_CONFIG))
        print(f"Generated {count} unbiased mock resonators")
    return crs, created_mock


def _noise_record(
    data: TuberResult, channel_index: int | None = None,
) -> dict:
    def values(value: list) -> np.ndarray:
        return np.asarray(value if channel_index is None else value[channel_index])

    return {
        "i": values(data.i), "q": values(data.q),
        "freq_iq": np.asarray(data.spectrum.freq_iq),
        "freq_dsb": np.asarray(data.spectrum.freq_dsb),
        "psd_i": values(data.spectrum.psd_i),
        "psd_q": values(data.spectrum.psd_q),
        "psd_dual_sideband": values(data.spectrum.psd_dual_sideband),
    }


async def _acquire_noise(
    crs: rfmux.CRS, catalog: ResonatorCatalog, *, created_mock: bool,
) -> dict:
    module = catalog.module
    slow_params = dict(SLOW_NOISE_PARAMS, module=module)
    pfb_params = dict(PFB_NOISE_PARAMS, module=module)
    noise = {
        "module_id": crs.module[module].index(), "module": module,
        "catalog": catalog.to_dict(),
        "slow_params": slow_params, "pfb_params": pfb_params,
        "resonators": {},
    }
    started_mock_stream = False
    started = time.perf_counter()
    try:
        if created_mock:
            conflict = find_streamer_conflict()
            if conflict:
                raise RuntimeError(
                    f"Cannot start a second mock stream: {conflict}. "
                    "Stop the other sender/receiver or use ATTACHED "
                    "to measure the existing session.")
            started_mock_stream = await crs.start_udp_streaming()
            print("Mock PFB RPC capture is synthetic uniform noise.")

        slow_rate = decimation_to_sampling(await crs.get_decimation())
        noise["slow_sample_rate_hz"] = slow_rate
        noise["pfb_sample_rate_hz"] = PFB_SAMPLING_FREQ
        slow_data = await crs.py_get_samples(**slow_params)
        for resonator in catalog:
            noise["resonators"][resonator.name] = {
                "channel": resonator.channel,
                "slow": _noise_record(slow_data, resonator.channel - 1),
            }
        del slow_data

        for resonator in catalog:
            pfb_data = await crs.py_get_pfb_samples(
                channel=resonator.channel, **pfb_params)
            noise["resonators"][resonator.name]["pfb"] = _noise_record(pfb_data)
            print(f"Noise acquired: {resonator.name}, channel {resonator.channel}")
    finally:
        if started_mock_stream:
            await crs.stop_udp_streaming()
            print("Script's mock UDP streamer stopped")

    print(f"Slow capture: {slow_params['num_samples'] / slow_rate:.3f} s, "
          f"{slow_rate:.1f} samples/s")
    print(f"PFB capture per resonator: "
          f"{pfb_params['nsamps'] / PFB_SAMPLING_FREQ * 1e3:.2f} ms")
    print(f"Noise acquisition completed in {time.perf_counter() - started:.1f} s")
    return noise


async def run_algorithm_flow(
    crs: rfmux.CRS, module: int = 1, *, created_mock: bool = False,
) -> BiasReport:
    """Tune, apply and measure noise; return the report and save measurements."""
    module_id = crs.module[module].index()
    await crs.clear_channels(module=module)

    print(f"1. Network analysis on {module_id}", flush=True)
    netanal = await crs.take_netanal(
        module=module, **NETANAL_PARAMS, save=True, label="tuning_netanal")
    module_netanal = netanal[module_id]
    search = find_resonances_in_netanal(
        module_netanal, **FIND_RES_PARAMS, save=True)
    print(f"Saved network analysis: {store.saved_path(module_netanal)}")
    if not len(search):
        raise RuntimeError("No resonances found: inspect the band and search cuts.")
    catalog = search.to_catalog(module=module, amplitude=NETANAL_PARAMS["amp"])
    print(f"Found {len(catalog)} resonators")

    print("2. Initial multisweep at the probe amplitude", flush=True)
    initial = await crs.multisweep(
        catalog, **MULTISWEEP_PARAMS, sweep_direction="upward",
        save=True, label="tuning_probe")
    print(f"Saved initial multisweep: {store.saved_path(initial[module_id])}")

    print(f"3. Amplitude scan: {SCHEDULE.nsteps} levels, both directions", flush=True)
    sweeps = await crs.multisweep(
        catalog, **MULTISWEEP_PARAMS, amp=SCHEDULE,
        sweep_direction=("upward", "downward"),
        save=True, label="tuning_amplitudes")
    module_sweeps = sweeps[module_id]
    report = find_bias_points(
        module_sweeps, **BIAS_SETTINGS,
        amplitude_method="derivative" if created_mock else "both", save=True)
    print(f"Saved amplitude scan and bias report: {store.saved_path(module_sweeps)}")
    print(f"Bias report: {len(report.findings)} points, {len(report.flagged)} flagged")
    for finding in report.findings:
        print(f"  {finding.name}: amplitude {finding.amplitude:g}, "
              f"{finding.frequency_hz / 1e6:.6f} MHz; "
              f"{finding.flagged_because or 'bracketed operating point'}")

    # Like the notebook, apply the full catalog, including reported fallbacks.
    print("4. Applying bias catalog", flush=True)
    await crs.apply_bias(report.catalog)

    print("5. Acquiring slow-stream and PFB noise", flush=True)
    noise = await _acquire_noise(crs, report.catalog, created_mock=created_mock)
    noise_path = store.save(noise, "noise", label="tuning_noise")
    print(f"Saved noise: {noise_path}")
    return report


async def main(
    serial: str = "MOCK", *, hostname: str | None = None,
    module: int = 1, output: Path | None = None,
) -> int:
    started = time.perf_counter()
    try:
        if output is None:
            output = Path(os.environ.get(
                "RFMUX_DEMO_OUTPUT", Path(tempfile.gettempdir()) / "rfmux_tuning_flow"))
        store.set_output_directory(output)
        print(f"Target: {serial}, module {module}; results: {output}", flush=True)
        crs, created_mock = await _connect(serial, hostname)
        await run_algorithm_flow(crs, module, created_mock=created_mock)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Workflow completed in {time.perf_counter() - started:.1f} s")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("serial", nargs="?", default="MOCK",
                        help="MOCK (default), board serial, or ATTACHED")
    parser.add_argument("--hostname", help="explicit CRS hostname/address")
    parser.add_argument("--module", type=int, default=1, help="module number (default: 1)")
    parser.add_argument("--output", type=Path,
                        help="output directory (default: RFMUX_DEMO_OUTPUT or temporary folder)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(asyncio.run(main(**vars(_parse_args()))))
