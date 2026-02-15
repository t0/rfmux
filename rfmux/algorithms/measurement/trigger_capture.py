"""
Unified pulse capture for slow (readout), fast (PFB), or both streamers simultaneously.

Uses sigma-based triggering: the noise statistics (mean, std) are computed
independently for both I and Q on each channel.  A pulse is detected when
**either** component deviates by more than ``threshold_sigma`` standard
deviations from its baseline mean.

**Sample-time semantics for ``time_run``:**

``time_run`` specifies the capture duration in terms of **elapsed time in the
sample domain** (derived from packet timestamps), *not* wall-clock time.  For
real hardware, sample time ≡ wall time by construction, so there is no
behavioural difference.  For mock/simulated streamers, wall time may advance
faster (or slower) than simulation time — using sample time makes captures
deterministic: ``time_run=0.1`` with ``pulse_period=0.01`` will always span
~10 pulse periods regardless of physics throughput.

Usage::

    # Slow streamer only
    start, pulses, noise = await crs.trigger_capture(
        channel=[1, 2], module=1,
        streamer_mode="slow", threshold_sigma=5.0, time_run=10.0,
    )

    # Fast PFB streamer only
    start, pulses, noise = await crs.trigger_capture(
        channel=[1, 2], module=1,
        streamer_mode="fast", threshold_sigma=5.0, time_run=0.1,
    )

    # Both simultaneously — pulse-matched, shared time axis
    start, matched, noise = await crs.trigger_capture(
        channel=[1, 2], module=1,
        streamer_mode="both", threshold_sigma=3.0, time_run=0.03,
    )
    # matched["Channel 1"][1] = {"slow": {...}, "fast": {...}}
"""

from __future__ import annotations

import asyncio
import time
import warnings
import numpy as np
from typing import Union, List, Optional, Dict

from ...core.hardware_map import macro
from ...core.schema import CRS
from ... import streamer

from .pulse_detection import (
    PulseCapture,
    ChannelNoiseStats,
    estimate_noise_stats,
)


# ── Sample-rate helpers ────────────────────────────────────────────

def _slow_sample_rate(decimation: int) -> float:
    return 625e6 / 256 / 64 / (2 ** decimation)

_PFB_SAMPLE_RATE = 625e6 / 512  # ≈1.22 MHz


# ── Timestamp → relative seconds ──────────────────────────────────

def _ts_to_seconds(ts) -> Optional[float]:
    if not ts.recent:
        return None
    return ts.h * 3600 + ts.m * 60 + ts.s + ts.ss / streamer.SS_PER_SECOND


# ── Noise estimation from live packets ────────────────────────────

async def _collect_noise_samples(
    host: str,
    port: int,
    packet_size: int,
    channels: List[int],
    n_packets: int,
    is_pfb: bool,
    n_groups: int = 1,
) -> Dict[int, np.ndarray]:
    """Collect raw complex samples for noise estimation."""
    loop = asyncio.get_running_loop()
    samples_by_ch: Dict[int, list] = {c: [] for c in channels}

    with streamer.get_multicast_socket(host, port=port) as sock:
        sock.setblocking(False)
        collected = 0
        while collected < n_packets:
            data = await asyncio.wait_for(
                loop.sock_recv(sock, packet_size),
                streamer.STREAMER_TIMEOUT,
            )
            if is_pfb:
                pkt = streamer.PFBPacket(data)
                raw = np.array(pkt.samples)
                for slot_idx, ch in enumerate(channels):
                    ch_samples = raw[slot_idx::n_groups]
                    samples_by_ch[ch].extend(ch_samples.tolist())
            else:
                pkt = streamer.ReadoutPacket(data)
                raw = np.array(pkt.samples) / 256.0
                for ch in channels:
                    samples_by_ch[ch].append(complex(raw[ch - 1]))
            collected += 1

    return {c: np.array(v, dtype=np.complex128) for c, v in samples_by_ch.items()}


# ── Baseline confirmation ─────────────────────────────────────────

async def _wait_for_baseline(
    host: str,
    port: int,
    packet_size: int,
    channels: List[int],
    module: int,
    noise_stats: Dict[int, "ChannelNoiseStats"],
    threshold_sigma: float,
    is_pfb: bool,
    n_groups: int,
    confirm_packets: int = 50,
    timeout: float = 5.0,
) -> None:
    """Receive packets until all channels are at baseline.

    Called between noise estimation and capture to ensure the first
    sample in the capture loop is at a known-good baseline, eliminating
    the need for a warmup phase in PulseCapture.
    """
    loop = asyncio.get_running_loop()
    baseline_count: Dict[int, int] = {c: 0 for c in channels}

    with streamer.get_multicast_socket(host, port=port) as sock:
        sock.setblocking(False)
        t0 = time.monotonic()

        while time.monotonic() - t0 < timeout:
            data = await asyncio.wait_for(
                loop.sock_recv(sock, packet_size),
                streamer.STREAMER_TIMEOUT,
            )

            if is_pfb:
                pkt = streamer.PFBPacket(data)
                raw = np.array(pkt.samples)
                for slot_idx, ch in enumerate(channels):
                    # Check last sample in this packet for this channel
                    ch_samples = raw[slot_idx::n_groups]
                    if len(ch_samples) > 0:
                        s = ch_samples[-1]
                        ns = noise_stats[ch]
                        dev_I = abs(s.real - ns.mean_I) / max(ns.std_I, 1e-30)
                        dev_Q = abs(s.imag - ns.mean_Q) / max(ns.std_Q, 1e-30)
                        if dev_I < threshold_sigma and dev_Q < threshold_sigma:
                            baseline_count[ch] += 1
                        else:
                            baseline_count[ch] = 0
            else:
                pkt = streamer.ReadoutPacket(data)
                if pkt.module != module - 1:
                    continue
                raw = np.array(pkt.samples) / 256.0
                for ch in channels:
                    s = raw[ch - 1]
                    ns = noise_stats[ch]
                    dev_I = abs(s.real - ns.mean_I) / max(ns.std_I, 1e-30)
                    dev_Q = abs(s.imag - ns.mean_Q) / max(ns.std_Q, 1e-30)
                    if dev_I < threshold_sigma and dev_Q < threshold_sigma:
                        baseline_count[ch] += 1
                    else:
                        baseline_count[ch] = 0

            if all(c >= confirm_packets for c in baseline_count.values()):
                return

    print(f"[trigger_capture] Warning: baseline confirmation timed out "
          f"(counts: {baseline_count})")


# ── Receive loop helper ───────────────────────────────────────────

async def _receive_stream(
    host: str,
    port: int,
    packet_size: int,
    pcap: PulseCapture,
    channels: List[int],
    module: int,
    time_run: float,
    is_pfb: bool,
    n_groups: int,
    sample_rate: float,
    label: str = "",
    shared_time_ref: Optional[Dict] = None,
) -> float:
    """Run a single-stream receive loop feeding into a PulseCapture.

    Parameters
    ----------
    shared_time_ref : dict, optional
        Mutable dict ``{"ref": float | None}`` for sharing the absolute
        timestamp reference between concurrent slow + fast receive loops.
        When provided, both loops compute ``rel_time`` from the same
        reference, making their ``Time`` arrays directly comparable.

    Returns the elapsed sample time covered.
    """
    aio_loop = asyncio.get_running_loop()
    # Use shared reference if provided, else local
    local_ref: Optional[float] = None
    elapsed_sample_time: float = 0.0

    def _get_rel_time(ts_sec: Optional[float]) -> Optional[float]:
        nonlocal local_ref
        if ts_sec is None:
            return None
        if shared_time_ref is not None:
            if shared_time_ref["ref"] is None:
                shared_time_ref["ref"] = ts_sec
            if pcap.start_time is None:
                pcap.start_time = 0.0
            return ts_sec - shared_time_ref["ref"]
        else:
            if local_ref is None:
                local_ref = ts_sec
                if pcap.start_time is None:
                    pcap.start_time = 0.0
            return ts_sec - local_ref

    with streamer.get_multicast_socket(host, port=port) as sock:
        sock.setblocking(False)
        wall_start = time.monotonic()

        _mock_mode_detected = False
        _last_progress_frac = 0.0
        _progress_step = 0.20
        _warmup_wall = 2.0

        if label:
            print(f"[trigger_capture:{label}] Capturing for {time_run}s "
                  f"on port {port}...")

        drain_limit = time_run * 1.5  # Allow 50% extra for in-progress pulses

        while True:
            # Once time_run reached, freeze new triggers and drain existing
            if elapsed_sample_time >= time_run and not pcap.freeze_triggers:
                pcap.freeze_triggers = True
            any_capturing = any(
                st.capturing for st in pcap.state.values())
            if pcap.freeze_triggers and not any_capturing:
                break
            if elapsed_sample_time >= drain_limit:
                break
            data = await asyncio.wait_for(
                aio_loop.sock_recv(sock, packet_size),
                streamer.STREAMER_TIMEOUT,
            )

            if is_pfb:
                pkt = streamer.PFBPacket(data)
                raw = np.array(pkt.samples)
                rel_time = _get_rel_time(_ts_to_seconds(pkt.ts))

                time_samples = pkt.num_samples // n_groups
                for sample_idx in range(time_samples):
                    st = (rel_time + sample_idx / sample_rate) if rel_time is not None else None
                    for slot_idx, ch in enumerate(channels):
                        flat_idx = sample_idx * n_groups + slot_idx
                        if flat_idx < len(raw):
                            s = raw[flat_idx]
                            pcap.process_sample(ch, float(s.real), float(s.imag), st)

                if rel_time is not None:
                    elapsed_sample_time = rel_time + (time_samples - 1) / sample_rate

                # Yield to event loop after PFB packet processing — the
                # 400+ process_sample() calls per packet are CPU-intensive
                # and would starve the slow stream in concurrent "both" mode.
                await asyncio.sleep(0)
            else:
                pkt = streamer.ReadoutPacket(data)
                if pkt.module != module - 1:
                    continue

                raw = np.array(pkt.samples) / 256.0
                rel_time = _get_rel_time(_ts_to_seconds(pkt.ts))

                for ch in channels:
                    s = raw[ch - 1]
                    pcap.process_sample(ch, float(s.real), float(s.imag), rel_time)

                if rel_time is not None:
                    elapsed_sample_time = rel_time

            # ── Progress estimation ───────────────────────────
            wall_elapsed = time.monotonic() - wall_start
            if (elapsed_sample_time > 0 and wall_elapsed > _warmup_wall
                    and not _mock_mode_detected):
                ratio = elapsed_sample_time / wall_elapsed
                if ratio < 0.5:
                    _mock_mode_detected = True
                    if label:
                        print(f"[trigger_capture:{label}] Mock/sim: "
                              f"{ratio*100:.1f}% real time, "
                              f"ETA ~{time_run/ratio:.0f}s")

            if _mock_mode_detected and time_run > 0 and label:
                frac = elapsed_sample_time / time_run
                if frac - _last_progress_frac >= _progress_step:
                    _last_progress_frac = frac
                    print(f"[trigger_capture:{label}] "
                          f"{frac*100:.0f}% ({elapsed_sample_time:.4f}/"
                          f"{time_run}s)")

        wall_elapsed = time.monotonic() - wall_start
        if label:
            print(f"[trigger_capture:{label}] Done: "
                  f"{elapsed_sample_time:.4f}s (wall: {wall_elapsed:.1f}s)")

    return elapsed_sample_time


# ── Pulse matching ────────────────────────────────────────────────

def _match_pulses(
    slow_pulses: Dict[str, dict],
    fast_pulses: Dict[str, dict],
    pcap_slow: Optional["PulseCapture"] = None,
    pcap_fast: Optional["PulseCapture"] = None,
) -> Dict[str, dict]:
    """Bidirectional pulse matching with cross-stream TOD extraction.

    A pulse detected in only the fast stream (but not slow) is still
    included with ``"slow": None``, and vice versa.  When the other
    stream's ``PulseCapture`` is provided, the raw TOD at the same time
    window is extracted via ``get_window_by_time()`` and included as
    ``"slow_tod"`` / ``"fast_tod"``.

    Returns
    -------
    dict[str, dict[int, dict]]
        ``{"Channel 1": {1: {"slow": ..., "fast": ...,
                             "slow_tod": ..., "fast_tod": ...}, ...}}``
    """
    matched: Dict[str, dict] = {}

    all_ch_keys = sorted(set(slow_pulses.keys()) | set(fast_pulses.keys()))

    for ch_key in all_ch_keys:
        matched[ch_key] = {}
        s_pulses = slow_pulses.get(ch_key, {})
        f_pulses = fast_pulses.get(ch_key, {})

        # Extract channel number from "Channel N" key
        try:
            ch_num = int(ch_key.split()[-1])
        except (ValueError, IndexError):
            ch_num = None

        def _mid_time(p):
            t = p["Time"]
            valid = t[t != None]  # noqa: E711
            return float(np.median(valid)) if len(valid) > 0 else None

        def _time_bounds(p, margin=0.0):
            """Return (t_start, t_end) of the pulse timestamps."""
            t = p["Time"]
            valid = t[t != None]  # noqa: E711
            if len(valid) == 0:
                return None, None
            return float(np.min(valid)) - margin, float(np.max(valid)) + margin

        # Build indexed lists with midpoint times
        slow_list = [(k, p, _mid_time(p)) for k, p in s_pulses.items()]
        fast_list = [(k, p, _mid_time(p)) for k, p in f_pulses.items()]

        used_slow = set()
        used_fast = set()
        pairs = []  # (slow_data|None, fast_data|None, sort_time)

        # Forward pass: match each slow pulse to closest fast pulse
        for si, (sk, sp, s_mid) in enumerate(slow_list):
            if s_mid is None:
                continue
            best_fi = None
            best_dist = float("inf")
            for fi, (fk, fp, f_mid) in enumerate(fast_list):
                if f_mid is None or fi in used_fast:
                    continue
                dist = abs(f_mid - s_mid)
                if dist < best_dist:
                    best_dist = dist
                    best_fi = fi

            if best_fi is not None and best_dist < 0.05:
                used_slow.add(si)
                used_fast.add(best_fi)
                pairs.append((sp, fast_list[best_fi][1], s_mid))
            else:
                used_slow.add(si)
                pairs.append((sp, None, s_mid))

        # Reverse pass: include fast-only pulses (not matched to any slow)
        for fi, (fk, fp, f_mid) in enumerate(fast_list):
            if fi not in used_fast and f_mid is not None:
                pairs.append((None, fp, f_mid))

        # Sort by time and assign indices, extracting cross-stream TOD
        pairs.sort(key=lambda x: x[2] if x[2] is not None else float("inf"))
        for idx, (sp, fp, _t) in enumerate(pairs, start=1):
            entry: Dict = {"slow": sp, "fast": fp,
                           "slow_tod": None, "fast_tod": None}

            # Cross-stream TOD extraction: use the WIDEST time window
            # from either pulse (in real time units) to query both
            # circular buffers.  This ensures the TOD covers the full
            # pulse as seen by whichever stream captured more detail.
            if ch_num is not None:
                t0, t1 = None, None
                if sp is not None:
                    s0, s1 = _time_bounds(sp)
                    if s0 is not None:
                        t0 = s0 if t0 is None else min(t0, s0)
                        t1 = s1 if t1 is None else max(t1, s1)
                if fp is not None:
                    f0, f1 = _time_bounds(fp)
                    if f0 is not None:
                        t0 = f0 if t0 is None else min(t0, f0)
                        t1 = f1 if t1 is None else max(t1, f1)
                if t0 is not None and t1 is not None:
                    if pcap_slow is not None:
                        entry["slow_tod"] = pcap_slow.get_window_by_time(
                            ch_num, t0, t1)
                    if pcap_fast is not None:
                        entry["fast_tod"] = pcap_fast.get_window_by_time(
                            ch_num, t0, t1)

            matched[ch_key][idx] = entry

    return matched


# ── Main macro ─────────────────────────────────────────────────────

@macro(CRS, register=True)
async def trigger_capture(
    crs: CRS,
    channel: Union[None, int, List[int]] = None,
    module: int = 1,
    *,
    streamer_mode: str = "slow",
    time_run: float = 10.0,
    threshold_sigma: float = 5.0,
    end_sigma: float = 1.5,
    buf_size: int = 5000,
    margin_fraction: float = 0.1,
    min_pulse_samples: int = 0,
    enable_pileup: bool = True,
    noise_packets: int = 1000,
):
    """Capture threshold-triggered pulses from slow, fast, or both streamers.

    Parameters
    ----------
    crs : CRS
        Connected CRS device handle.
    channel : int | list[int] | None
        Channel(s) to monitor.  Max 4 for fast/both modes.
    module : int
        Module index (1-based).
    streamer_mode : str
        ``"slow"``, ``"fast"`` (PFB), or ``"both"`` (simultaneous).
        In ``"both"`` mode, pulses from the two streams are matched by
        time and returned as paired dicts.
    time_run : float
        Capture duration in seconds of **sample time**.
    threshold_sigma : float
        Trigger threshold in σ.
    end_sigma : float
        End-of-pulse threshold in σ.
    buf_size : int
        Circular buffer size per channel.
    margin_fraction : float
        Fraction of pulse duration used as symmetric pre/post margin
        and adaptive end-of-pulse confirmation count.  Default 0.1.
    noise_packets : int
        Packets for noise estimation.

    Returns
    -------
    tuple
        For ``"slow"`` or ``"fast"``:
            ``(start_time, pulses, noise_stats)``
        For ``"both"``:
            ``(start_time, matched_pulses, {"slow": noise, "fast": noise})``
            where ``matched_pulses["Channel N"][k]`` =
            ``{"slow": {Amp_I, Amp_Q, Time}, "fast": {Amp_I, Amp_Q, Time}}``
    """

    # ── Validate ──────────────────────────────────────────────────
    if channel is None:
        raise ValueError("channel must be specified")
    channels = channel if isinstance(channel, list) else [channel]

    if streamer_mode in ("fast", "both") and len(channels) > 4:
        raise ValueError("PFB streamer supports max 4 channels")
    if streamer_mode not in ("slow", "fast", "both"):
        raise ValueError(f"streamer_mode must be 'slow', 'fast', or 'both'")

    # ── Resolve hostname ──────────────────────────────────────────
    host = crs.tuber_hostname
    if host and ':' in host:
        host = host.split(':')[0]
    if host in ("rfmux0000.local",):
        host = "127.0.0.1"

    dec = await crs.get_decimation()
    if dec is None:
        dec = 6
    slow_rate = _slow_sample_rate(dec)
    pfb_rate = _PFB_SAMPLE_RATE
    n_groups = len(channels)

    print(f"[trigger_capture] mode={streamer_mode}, ch={channels}, "
          f"σ={threshold_sigma}, time_run={time_run}s")

    use_pfb = streamer_mode in ("fast", "both")
    if use_pfb:
        await crs.set_pfb_streamer(channel=channels, module=module)
        await asyncio.sleep(0.3)

    try:
        # ── SINGLE-STREAM: "slow" or "fast" ──────────────────────
        if streamer_mode in ("slow", "fast"):
            is_pfb = (streamer_mode == "fast")
            rate = pfb_rate if is_pfb else slow_rate
            port = streamer.PFB_STREAMER_PORT if is_pfb else streamer.STREAMER_PORT
            pkt_sz = streamer.PFB_PACKET_SIZE if is_pfb else streamer.LONG_PACKET_SIZE
            ng = n_groups if is_pfb else 1

            print(f"[trigger_capture] Noise estimation ({noise_packets} pkts)...")
            samples = await _collect_noise_samples(
                host, port, pkt_sz, channels, noise_packets, is_pfb, ng)
            noise, _ = estimate_noise_stats(samples, channels)
            for c in channels:
                ns = noise[c]
                print(f"  Ch {c}: I={ns.mean_I:.1f}±{ns.std_I:.2f}, "
                      f"Q={ns.mean_Q:.1f}±{ns.std_Q:.2f}")

            # Wait for baseline before starting capture
            print(f"[trigger_capture] Waiting for baseline...")
            await _wait_for_baseline(
                host, port, pkt_sz, channels, module, noise,
                threshold_sigma, is_pfb, ng)

            pcap = PulseCapture(
                buf_size=buf_size, channels=channels, noise_stats=noise,
                threshold_sigma=threshold_sigma, end_sigma=end_sigma,
                sample_rate=rate, margin_fraction=margin_fraction,
                min_pulse_samples=min_pulse_samples,
                enable_pileup=enable_pileup)

            await _receive_stream(
                host, port, pkt_sz, pcap, channels, module,
                time_run, is_pfb, ng, rate, label=streamer_mode)

            total = sum(pcap.pulse_count.values())
            print(f"[trigger_capture] {total} pulses: {pcap.pulse_count}")
            return pcap.start_time, pcap.pulses, noise

        # ── CONCURRENT: "both" ────────────────────────────────────
        print(f"[trigger_capture] Concurrent slow ({slow_rate:.0f} Hz) "
              f"+ fast ({pfb_rate:.0f} Hz)")

        # Noise estimation (concurrent)
        print(f"[trigger_capture] Noise estimation...")
        s_samp, f_samp = await asyncio.gather(
            _collect_noise_samples(
                host, streamer.STREAMER_PORT, streamer.LONG_PACKET_SIZE,
                channels, noise_packets, False, 1),
            _collect_noise_samples(
                host, streamer.PFB_STREAMER_PORT, streamer.PFB_PACKET_SIZE,
                channels, noise_packets, True, n_groups),
        )
        slow_noise, _ = estimate_noise_stats(s_samp, channels)
        fast_noise, _ = estimate_noise_stats(f_samp, channels)

        for c in channels:
            print(f"  Ch {c} slow: I={slow_noise[c].mean_I:.1f}"
                  f"±{slow_noise[c].std_I:.2f}, "
                  f"Q={slow_noise[c].mean_Q:.1f}±{slow_noise[c].std_Q:.2f}")
            print(f"  Ch {c} fast: I={fast_noise[c].mean_I:.1f}"
                  f"±{fast_noise[c].std_I:.2f}, "
                  f"Q={fast_noise[c].mean_Q:.1f}±{fast_noise[c].std_Q:.2f}")

        # Wait for baseline on both streams concurrently
        print(f"[trigger_capture] Waiting for baseline...")
        await asyncio.gather(
            _wait_for_baseline(
                host, streamer.STREAMER_PORT, streamer.LONG_PACKET_SIZE,
                channels, module, slow_noise, threshold_sigma, False, 1),
            _wait_for_baseline(
                host, streamer.PFB_STREAMER_PORT, streamer.PFB_PACKET_SIZE,
                channels, module, fast_noise, threshold_sigma, True, n_groups),
        )

        pcap_slow = PulseCapture(
            buf_size=buf_size, channels=channels, noise_stats=slow_noise,
            threshold_sigma=threshold_sigma, end_sigma=end_sigma,
            sample_rate=slow_rate, margin_fraction=margin_fraction,
            min_pulse_samples=min_pulse_samples,
            enable_pileup=enable_pileup)
        pcap_fast = PulseCapture(
            buf_size=buf_size, channels=channels, noise_stats=fast_noise,
            threshold_sigma=threshold_sigma, end_sigma=end_sigma,
            sample_rate=pfb_rate, margin_fraction=margin_fraction,
            min_pulse_samples=min_pulse_samples,
            enable_pileup=enable_pileup)

        # ── Multiplexed receive: both sockets in one loop ─────────
        # Using asyncio.wait(FIRST_COMPLETED) to fairly interleave
        # slow and fast packets — prevents PFB from starving the
        # slow stream (which happens with asyncio.gather because
        # PFB packets arrive faster than they're processed).
        shared_ref: Dict[str, Optional[float]] = {"ref": None}

        def _get_rel(ts_sec):
            if ts_sec is None:
                return None
            if shared_ref["ref"] is None:
                shared_ref["ref"] = ts_sec
            return ts_sec - shared_ref["ref"]

        aio_loop = asyncio.get_running_loop()
        slow_elapsed = 0.0
        fast_elapsed = 0.0

        with (streamer.get_multicast_socket(host, port=streamer.STREAMER_PORT) as s_sock,
              streamer.get_multicast_socket(host, port=streamer.PFB_STREAMER_PORT) as f_sock):
            s_sock.setblocking(False)
            f_sock.setblocking(False)

            # Flush stale buffered packets from both sockets.
            # The slow streamer has been running since test start, so old
            # packets with outdated timestamps may be sitting in the kernel
            # buffer.  Discard them to ensure synchronized fresh data.
            for sock in (s_sock, f_sock):
                while True:
                    try:
                        sock.recv(65536)
                    except BlockingIOError:
                        break

            wall_start = time.monotonic()

            print(f"[trigger_capture:both] Capturing for {time_run}s ...")

            drain_limit = time_run * 1.5
            _last_progress_wall = 0.0

            while True:
                max_elapsed = max(slow_elapsed, fast_elapsed)
                # Once time_run reached, freeze new triggers on both pcaps
                if max_elapsed >= time_run:
                    if not pcap_slow.freeze_triggers:
                        pcap_slow.freeze_triggers = True
                    if not pcap_fast.freeze_triggers:
                        pcap_fast.freeze_triggers = True
                any_capturing = (
                    any(st.capturing for st in pcap_slow.state.values()) or
                    any(st.capturing for st in pcap_fast.state.values()))
                if pcap_slow.freeze_triggers and not any_capturing:
                    break
                if max_elapsed >= drain_limit:
                    break

                # Build recv tasks for both sockets
                tasks = {}
                tasks["slow"] = asyncio.ensure_future(
                    aio_loop.sock_recv(s_sock, streamer.LONG_PACKET_SIZE))
                tasks["fast"] = asyncio.ensure_future(
                    aio_loop.sock_recv(f_sock, streamer.PFB_PACKET_SIZE))

                done, pending = await asyncio.wait(
                    tasks.values(),
                    timeout=streamer.STREAMER_TIMEOUT,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel any pending task
                for t in pending:
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass

                if not done:
                    continue

                for completed in done:
                    data = completed.result()

                    if completed is tasks.get("slow"):
                        pkt = streamer.ReadoutPacket(data)
                        if pkt.module != module - 1:
                            continue
                        raw = np.array(pkt.samples) / 256.0
                        rel = _get_rel(_ts_to_seconds(pkt.ts))
                        if pcap_slow.start_time is None and rel is not None:
                            pcap_slow.start_time = 0.0
                        for ch in channels:
                            s = raw[ch - 1]
                            pcap_slow.process_sample(
                                ch, float(s.real), float(s.imag), rel)
                        if rel is not None:
                            slow_elapsed = rel

                    elif completed is tasks.get("fast"):
                        pkt = streamer.PFBPacket(data)
                        raw = np.array(pkt.samples)
                        rel = _get_rel(_ts_to_seconds(pkt.ts))
                        if pcap_fast.start_time is None and rel is not None:
                            pcap_fast.start_time = 0.0
                        time_samples = pkt.num_samples // n_groups
                        for si in range(time_samples):
                            st = (rel + si / pfb_rate) if rel is not None else None
                            for slot_idx, ch in enumerate(channels):
                                fi = si * n_groups + slot_idx
                                if fi < len(raw):
                                    v = raw[fi]
                                    pcap_fast.process_sample(
                                        ch, float(v.real), float(v.imag), st)
                        if rel is not None:
                            fast_elapsed = rel + (time_samples - 1) / pfb_rate

                # Progress — at most once per 5 wall seconds
                wall_el = time.monotonic() - wall_start
                if wall_el - _last_progress_wall >= 5.0:
                    _last_progress_wall = wall_el
                    print(f"[trigger_capture:both] "
                          f"slow={slow_elapsed:.4f}s "
                          f"fast={fast_elapsed:.4f}s "
                          f"(wall: {wall_el:.0f}s)")

            wall_elapsed = time.monotonic() - wall_start
            print(f"[trigger_capture:both] Done: "
                  f"slow={slow_elapsed:.4f}s fast={fast_elapsed:.4f}s "
                  f"(wall: {wall_elapsed:.1f}s)")

        s_total = sum(pcap_slow.pulse_count.values())
        f_total = sum(pcap_fast.pulse_count.values())
        print(f"[trigger_capture] slow: {s_total} pulses, "
              f"fast: {f_total} pulses")

        # Match slow and fast pulses by time, with cross-stream TOD
        matched = _match_pulses(
            pcap_slow.pulses, pcap_fast.pulses,
            pcap_slow=pcap_slow, pcap_fast=pcap_fast,
        )
        n_matched = sum(
            1 for ch in matched.values()
            for p in ch.values()
            if p.get("fast") is not None
        )
        print(f"[trigger_capture] {n_matched} pulse pairs matched")

        start_time = pcap_slow.start_time or pcap_fast.start_time
        return (
            start_time,
            matched,
            {"slow": slow_noise, "fast": fast_noise},
        )

    finally:
        if use_pfb:
            await crs.set_pfb_streamer(channel=None, module=module)
