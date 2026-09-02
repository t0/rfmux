"""
Streamer configuration: rates, link-budget math, validation, and apply.

One headless home for everything about configuring the CRS streamers —
the slow readout stream (decimation stage, short/long packets, which
modules are streamed) and the fast PFB stream (up to 4 channels of one
module at ~2.44 MHz).  The Periscope "Streamer Configuration" dialog is
a thin view over :func:`describe` and :func:`validate`; scripts and
notebooks use :func:`apply_streamer_config` or the registered
``crs.configure_streamer(...)`` macro directly.

Rules encoded here (sources: firmware/CHANGES, test/core/test_spotcheck.py
against real hardware, and the noise-dialog advisories):

- decimation stage 0-6; slow rate = 625 MHz / 256 / 64 / 2**stage.
- long packets (1024 ch) require stage >= 3; below that the packet rate
  exceeds the 1 GbE link and firmware refuses — short packets (128 ch)
  work at every stage.
- total streamed bandwidth must fit 1 GbE; firmware derates to ~0.8.
- the PFB streamer carries at most 4 channels of a single module and is
  mutually exclusive with ``get_pfb_samples`` while active.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ...core.hardware_map import macro
from ...core.schema import CRS
from ...core.transferfunctions import (
    PFB_SAMPLING_FREQ,
    decimation_to_sampling,
)
from ... import streamer

# Link budget (1 GbE), with the firmware's ~0.8 bandwidth derating
LINK_MBPS = 1000.0
DERATED_LINK_MBPS = 800.0


@dataclass
class StreamerConfig:
    """Desired streamer state.

    ``pfb_channels``: None = leave the PFB streamer untouched;
    [] = disable it; [ch, ...] (max 4) = stream those channels of
    ``pfb_module``.
    """
    dec_stage: int = 6
    short_packets: bool = False
    modules: Optional[List[int]] = None      # None = all active modules
    pfb_channels: Optional[List[int]] = None
    pfb_module: int = 1

    def n_modules(self, default: int = 4) -> int:
        return len(self.modules) if self.modules else default


def describe(cfg: StreamerConfig) -> Dict[str, Any]:
    """Derived quantities for a configuration (rates, widths, bandwidth)."""
    fs = decimation_to_sampling(cfg.dec_stage)
    packet_bytes = (streamer.SHORT_PACKET_SIZE if cfg.short_packets
                    else streamer.LONG_PACKET_SIZE)
    channels = (streamer.SHORT_PACKET_CHANNELS if cfg.short_packets
                else streamer.LONG_PACKET_CHANNELS)
    n_mod = cfg.n_modules()
    slow_mbps = packet_bytes * 8 * fs * n_mod / 1e6

    n_pfb = len(cfg.pfb_channels) if cfg.pfb_channels else 0
    # PFB packets carry 1000 interleaved samples in PFB_PACKET_SIZE bytes
    pfb_mbps = (streamer.PFB_PACKET_SIZE * 8 * PFB_SAMPLING_FREQ * n_pfb
                / 1000.0 / 1e6)

    return {
        "sample_rate_hz": fs,
        "nyquist_hz": fs / 2.0,
        "channels_per_module": channels,
        "packet_bytes": packet_bytes,
        "n_modules": n_mod,
        "slow_mbps": slow_mbps,
        "pfb_sample_rate_hz": PFB_SAMPLING_FREQ,
        "n_pfb_channels": n_pfb,
        "pfb_mbps": pfb_mbps,
        "total_mbps": slow_mbps + pfb_mbps,
    }


def validate(cfg: StreamerConfig) -> List[Tuple[str, str]]:
    """Check a configuration; returns [(severity, message), ...].

    Severities: ``"error"`` (firmware will refuse / cannot work),
    ``"warning"`` (likely trouble), ``"info"`` (worth knowing).
    """
    issues: List[Tuple[str, str]] = []

    if not isinstance(cfg.dec_stage, int) or not 0 <= cfg.dec_stage <= 6:
        issues.append(("error",
                       f"Decimation stage must be 0-6 (got {cfg.dec_stage})"))
        return issues  # everything else depends on a valid stage

    if not cfg.short_packets and cfg.dec_stage < 3:
        issues.append((
            "error",
            f"Stage {cfg.dec_stage} requires short packets (128 channels): "
            f"long packets exceed the 1 GbE link below stage 3."))

    if cfg.modules is not None:
        bad = [m for m in cfg.modules if not 1 <= int(m) <= 8]
        if bad:
            issues.append(("error", f"Invalid module number(s): {bad}"))
        if not cfg.modules:
            issues.append(("warning",
                           "Empty module list blanks the slow streamer "
                           "entirely (get_decimation will return None)."))

    if cfg.pfb_channels:
        if len(cfg.pfb_channels) > 4:
            issues.append(("error",
                           "The PFB streamer supports at most 4 channels "
                           f"(got {len(cfg.pfb_channels)})."))
        if any(int(c) < 1 for c in cfg.pfb_channels):
            issues.append(("error", "PFB channels are 1-indexed."))
        issues.append((
            "info",
            "While the PFB streamer is active, get_pfb_samples() is "
            "unavailable (they share the packetizer)."))

    d = describe(cfg)
    if d["total_mbps"] > LINK_MBPS:
        issues.append((
            "error",
            f"Configuration needs {d['total_mbps']:.0f} Mbps — beyond the "
            f"1 GbE link. Reduce modules, increase decimation, or use "
            f"short packets."))
    elif d["total_mbps"] > DERATED_LINK_MBPS:
        issues.append((
            "warning",
            f"{d['total_mbps']:.0f} Mbps exceeds the firmware's ~"
            f"{DERATED_LINK_MBPS:.0f} Mbps derated budget — expect "
            f"refusal or packet loss."))
    elif cfg.pfb_channels and cfg.short_packets:
        # Firmware r1.6 checks the PFB budget against long packets
        # whatever the packet format, so a short-packet configuration
        # the link carries can still be refused.
        as_long = (streamer.LONG_PACKET_CHANNELS * 8 * 8 * d["sample_rate_hz"]
                   * d["n_modules"] / 1e6)
        if as_long + d["pfb_mbps"] > DERATED_LINK_MBPS:
            issues.append((
                "warning",
                f"The firmware's PFB budget check counts the readout as "
                f"long packets ({as_long:.0f} Mbps here) and will refuse "
                f"this although the link has room. Enable the PFB "
                f"streamer at stage 3 or above."))

    if cfg.dec_stage <= 1:
        issues.append((
            "warning",
            "Stage ≤ 1: expect dropped packets on macOS/Windows; on Linux "
            "increase the UDP buffer (sysctl net.core.rmem_max)."))

    if cfg.dec_stage < 5 and cfg.n_modules() > 1:
        issues.append((
            "info",
            "Below stage 5, streaming has only been hardware-validated "
            "one module at a time — consider modules=[current]."))

    return issues


# ── Board interaction ─────────────────────────────────────────────

async def apply_streamer_config(crs, cfg: StreamerConfig) -> Dict[str, Any]:
    """Validate and apply a configuration to the board.

    Raises ValueError when validation finds errors.  Returns
    :func:`describe`'s dict for the applied configuration.
    """
    errors = [msg for sev, msg in validate(cfg) if sev == "error"]
    if errors:
        raise ValueError("Invalid streamer configuration:\n- "
                         + "\n- ".join(errors))

    # 'module' (singular) is the firmware spelling as of r1.6.0; it takes
    # None, an int, or a list.  r1.5.6 spelled it 'modules' -- see the
    # firmware/CHANGES entry for r1.6.0.
    await crs.set_decimation(cfg.dec_stage, short=cfg.short_packets,
                             module=cfg.modules)

    if cfg.pfb_channels is not None:
        if cfg.pfb_channels:
            await crs.set_pfb_streamer(channel=list(cfg.pfb_channels),
                                       module=cfg.pfb_module)
            await asyncio.sleep(0.3)  # let the fast stream settle
        else:
            await crs.set_pfb_streamer(channel=None, module=cfg.pfb_module)

    return describe(cfg)


async def read_streamer_config(crs, pfb_module: int = 1) -> Dict[str, Any]:
    """Best-effort readback of current streamer state.

    ``short_packets`` and the streamed-module list are not exposed by
    firmware RPCs — GUI callers should merge the packet-derived values
    (``fir_stage`` bit 3) they already track.
    """
    state: Dict[str, Any] = {"dec_stage": None, "pfb_channels": None,
                             "pfb_module": pfb_module}
    try:
        state["dec_stage"] = await crs.get_decimation()
    except Exception:
        pass
    try:
        state["pfb_channels"] = await crs.get_pfb_streamer(module=pfb_module)
    except Exception:
        pass
    return state


@macro(CRS, register=True)
async def configure_streamer(
    crs,
    dec_stage: int = 6,
    *,
    short: bool = False,
    modules: Optional[List[int]] = None,
    pfb_channels: Optional[List[int]] = None,
    pfb_module: int = 1,
) -> Dict[str, Any]:
    """Configure the slow (and optionally fast/PFB) streamers.

    Usage::

        info = await crs.configure_streamer(1, short=True, modules=[1])
        info = await crs.configure_streamer(6, pfb_channels=[1, 2])
        await crs.configure_streamer(6, pfb_channels=[])   # disable PFB

    Returns the derived-quantities dict (sample rate, bandwidth, ...).
    """
    cfg = StreamerConfig(dec_stage=dec_stage, short_packets=short,
                         modules=modules, pfb_channels=pfb_channels,
                         pfb_module=pfb_module)
    return await apply_streamer_config(crs, cfg)
