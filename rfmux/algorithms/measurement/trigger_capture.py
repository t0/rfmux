"""
One-shot pulse capture: ``await crs.trigger_capture(...)``.

The convenient front door to the pulse-capture stack — one call, no
session lifecycle to manage, everything handed back in memory.  It is a
thin caller over the same machinery Periscope and the reference notebook
drive:

- :class:`~rfmux.algorithms.measurement.pulse_capture_session.PulseCaptureConfig`
  for the parameters, in physical units;
- :class:`~rfmux.algorithms.measurement.pulse_capture_session.PulseCaptureSession`
  (or :class:`~rfmux.algorithms.measurement.pulse_capture_session.DualPulseCaptureSession`)
  for detection, persistence, histograms and templates;
- :mod:`~rfmux.algorithms.measurement.pulse_sources` for the sockets.

Because it is a session underneath, passing ``hdf5_path=`` writes a real
capture file — pulses, histograms and trigger-aligned templates — that
Periscope can open in review mode.  For long or unattended captures,
drive a session directly instead: this macro holds every pulse in memory.

**Sample-time semantics for** ``time_run``: it is elapsed time in the
*sample* domain (from packet timestamps), not wall clock.  On real
hardware the two are identical; against the mock streamer, wall time can
run faster or slower than simulation time, so sample time is what makes
a capture reproducible.  Noise training happens first and is *not*
charged against ``time_run`` — the source runs for the training span
plus ``time_run``.

Usage::

    res = await crs.trigger_capture(
        channel=[1, 2], module=1, streamer_mode="slow",
        time_run=2.0, threshold_sigma=5.0)

    res.pulses[1][3]["Amp_I"]     # waveform for channel 1, pulse 3
    res.summaries[1][3]["snr"]    # its metrics
    res.noise[1].std_I            # per-channel noise
    res.pairs                     # matched slow/fast pairs ("both" mode)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from ...core.hardware_map import macro
from ...core.schema import CRS
from ... import streamer

from .pulse_capture_session import (
    DualPulseCaptureSession,
    PulseCaptureConfig,
    PulseCaptureSession,
)
from .pulse_detection import ChannelNoiseStats
from .pulse_sources import run_dual_source, run_pfb_source, run_slow_source
from .streamer_config import PFB_SAMPLE_RATE, slow_sample_rate

#: Default longest-pulse estimate for a one-shot capture.  Deliberately
#: shorter than PulseCaptureConfig's own default: that one is sized for
#: an open-ended run, whereas a one-shot capture is usually seconds long
#: and cannot afford a training window measured in tens of seconds.
DEFAULT_MAX_PULSE_MS = 20.0


@dataclass
class StreamResult:
    """Everything one stream produced during a capture."""

    sample_rate: float
    noise: Dict[int, ChannelNoiseStats] = field(default_factory=dict)
    #: ``{channel: {pulse_idx: {"Amp_I", "Amp_Q", "Time", ...}}}``
    pulses: Dict[int, Dict[int, dict]] = field(default_factory=dict)
    #: ``{channel: {pulse_idx: pulse_summary(...)}}`` — same keys as pulses
    summaries: Dict[int, Dict[int, dict]] = field(default_factory=dict)
    elapsed_s: float = 0.0

    @property
    def total_pulses(self) -> int:
        return sum(len(v) for v in self.pulses.values())

    def count(self, channel: int) -> int:
        return len(self.pulses.get(channel, {}))


@dataclass
class PulseCaptureResult:
    """Result of :func:`trigger_capture`.

    For single-stream captures the ``pulses`` / ``summaries`` / ``noise``
    shortcuts are the whole story.  In ``"both"`` mode they refer to the
    slow stream, and the two streams are also reachable individually as
    ``.slow`` and ``.fast`` — with ``.pairs`` holding the cross-stream
    matches, which is the reason to run both at once.
    """

    streamer_mode: str
    config: PulseCaptureConfig
    channels: List[int]
    module: int
    #: Wall-clock (``time.time()``) at which the capture started, matching
    #: the ``capture_start`` attribute in the HDF5 file.  Note the ``Time``
    #: arrays inside each pulse are in the *sample* domain
    #: (seconds-of-day from packet timestamps), not this one — use
    #: :attr:`first_pulse_time` to locate the capture on that axis.
    start_time: Optional[float] = None
    slow: Optional[StreamResult] = None
    fast: Optional[StreamResult] = None
    #: Matched slow/fast pairs; each carries ``slow_idx``/``fast_idx``,
    #: the two summaries, ``time_offset``, and the union-window TOD from
    #: both ring buffers.  Empty unless ``streamer_mode="both"``.
    pairs: List[dict] = field(default_factory=list)
    hdf5_path: Optional[Path] = None

    @property
    def primary(self) -> StreamResult:
        """The stream the shortcuts refer to (slow unless fast-only)."""
        if self.streamer_mode == "fast":
            return self.fast
        return self.slow

    @property
    def pulses(self) -> Dict[int, Dict[int, dict]]:
        return self.primary.pulses

    @property
    def summaries(self) -> Dict[int, Dict[int, dict]]:
        return self.primary.summaries

    @property
    def noise(self) -> Dict[int, ChannelNoiseStats]:
        return self.primary.noise

    @property
    def sample_rate(self) -> float:
        return self.primary.sample_rate

    @property
    def total_pulses(self) -> int:
        """Pulses across every stream — in "both" mode the two streams
        see the same events, so this is roughly twice the event count."""
        return sum(s.total_pulses for s in (self.slow, self.fast)
                   if s is not None)

    @property
    def first_pulse_time(self) -> Optional[float]:
        """Earliest pulse timestamp, on the same sample-domain axis as
        each pulse's ``Time`` array.  None if nothing triggered."""
        times = [s["timestamp"]
                 for stream in (self.slow, self.fast) if stream is not None
                 for by_idx in stream.summaries.values()
                 for s in by_idx.values()
                 if s.get("timestamp") is not None]
        return min(times) if times else None

    def __repr__(self) -> str:
        parts = [f"mode={self.streamer_mode}",
                 f"channels={self.channels}"]
        for name in ("slow", "fast"):
            s = getattr(self, name)
            if s is not None:
                parts.append(f"{name}={s.total_pulses} pulses")
        if self.pairs:
            parts.append(f"pairs={len(self.pairs)}")
        if self.hdf5_path is not None:
            parts.append(f"hdf5={self.hdf5_path.name}")
        return f"<PulseCaptureResult {', '.join(parts)}>"


def _collector(stream: StreamResult):
    """on_pulse callback that files waveforms and summaries by channel."""
    def _on_pulse(channel: int, pulse_idx: int, summary: dict,
                  waveform: dict) -> None:
        stream.pulses.setdefault(channel, {})[pulse_idx] = waveform
        stream.summaries.setdefault(channel, {})[pulse_idx] = summary
    return _on_pulse


def _training_seconds(config: PulseCaptureConfig, rate: float) -> float:
    return config.noise_samples(rate) / rate


@macro(CRS, register=True)
async def trigger_capture(
    crs: CRS,
    channel: Union[None, int, List[int]] = None,
    module: int = 1,
    *,
    streamer_mode: str = "slow",
    time_run: float = 10.0,
    config: Optional[PulseCaptureConfig] = None,
    threshold_sigma: float = 5.0,
    end_sigma: float = 1.5,
    max_pulse_ms: float = DEFAULT_MAX_PULSE_MS,
    hdf5_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> PulseCaptureResult:
    """Capture threshold-triggered pulses from the slow, fast, or both streams.

    Parameters
    ----------
    channel : int | list[int]
        Channel(s) to monitor.  Max 4 for ``"fast"``/``"both"`` — the PFB
        streamer's hard limit.
    module : int
        Module index (1-based).
    streamer_mode : str
        ``"slow"``, ``"fast"`` (PFB, ~1.22 MHz) or ``"both"``.  In
        ``"both"`` the two streams are detected independently and their
        pulses matched by trigger time into :attr:`PulseCaptureResult.pairs`.
    time_run : float
        Capture duration in seconds of **sample time**, excluding noise
        training.
    config : PulseCaptureConfig, optional
        Full detection configuration.  When omitted one is built from
        ``threshold_sigma``, ``end_sigma`` and ``max_pulse_ms``; when
        given, those three are ignored.
    max_pulse_ms : float
        Longest pulse to expect.  Sizes the ring buffer and, through it,
        the noise-training and baseline windows — so it also determines
        how much of the stream is spent training before detection starts.
    hdf5_path : str | Path, optional
        Write a capture file as well (pulses, histograms, templates).
    verbose : bool
        Print progress.  Set False for scripted use.

    Returns
    -------
    PulseCaptureResult
    """
    if channel is None:
        raise ValueError("channel must be specified")
    channels = list(channel) if isinstance(channel, list) else [channel]

    if streamer_mode not in ("slow", "fast", "both"):
        raise ValueError(
            f"streamer_mode must be 'slow', 'fast' or 'both', "
            f"not {streamer_mode!r}")
    if streamer_mode in ("fast", "both") and len(channels) > 4:
        raise ValueError(
            f"the PFB streamer carries at most 4 channels, got "
            f"{len(channels)}")

    if config is None:
        config = PulseCaptureConfig(
            threshold_sigma=threshold_sigma, end_sigma=end_sigma,
            max_pulse_ms=max_pulse_ms)

    host = streamer.resolve_host(crs.tuber_hostname)
    dec = await crs.get_decimation()
    if dec is None:
        dec = 6
    slow_rate = slow_sample_rate(dec)

    rates = ({"slow": slow_rate} if streamer_mode == "slow"
             else {"fast": PFB_SAMPLE_RATE} if streamer_mode == "fast"
             else {"slow": slow_rate, "fast": PFB_SAMPLE_RATE})

    # Validate against each rate actually in play: most of what the config
    # derives (confirmation length, buffer, training span) is rate
    # dependent, so checking a fast capture against the slow rate would
    # both miss real problems and invent absent ones.
    seen = set()
    for stream, rate in rates.items():
        for severity, message in config.validate(rate):
            if severity == "error":
                raise ValueError(
                    f"invalid capture configuration for the {stream} "
                    f"stream: {message}")
            if verbose and message not in seen:
                seen.add(message)
                print(f"[trigger_capture] [{severity}] {message}")

    # Detection only starts once the noise fit has its samples, so the
    # source has to run for the training span on top of the requested
    # capture -- otherwise time_run would silently buy less data the
    # longer the training window is.
    train_s = max(_training_seconds(config, r) for r in rates.values())
    duration_s = time_run + train_s

    if verbose:
        rate_str = ", ".join(f"{k} {v:,.0f} Hz" for k, v in rates.items())
        print(f"[trigger_capture] mode={streamer_mode}, ch={channels}, "
              f"module={module}, {rate_str}")
        print(f"[trigger_capture] {train_s*1e3:.0f} ms noise training "
              f"+ {time_run:.2f} s capture")

    hdf5_path = Path(hdf5_path) if hdf5_path is not None else None
    result = PulseCaptureResult(
        streamer_mode=streamer_mode, config=config, channels=channels,
        module=module, hdf5_path=hdf5_path)

    use_pfb = streamer_mode in ("fast", "both")
    if use_pfb:
        await crs.set_pfb_streamer(channel=channels, module=module)
        await asyncio.sleep(0.3)  # let the streamer settle before listening
    try:
        if streamer_mode == "both":
            await _run_dual(result, crs, host, channels, module,
                            slow_rate, duration_s, hdf5_path, verbose)
        else:
            await _run_single(result, crs, host, channels, module,
                              streamer_mode, slow_rate, duration_s,
                              hdf5_path, verbose)
    finally:
        if use_pfb:
            await crs.set_pfb_streamer(channel=None, module=module)

    if verbose:
        print(f"[trigger_capture] {result!r}")
    return result


async def _run_single(result, crs, host, channels, module, streamer_mode,
                      slow_rate, duration_s, hdf5_path, verbose) -> None:
    is_fast = streamer_mode == "fast"
    rate = PFB_SAMPLE_RATE if is_fast else slow_rate
    stream = StreamResult(sample_rate=rate)

    session = PulseCaptureSession(
        channels=channels, module=module, streamer_mode=streamer_mode,
        sample_rate=rate, hdf5_path=hdf5_path,
        on_pulse=_collector(stream),
        on_error=(lambda m: print(f"[trigger_capture] {m}")) if verbose
        else None,
        **result.config.session_kwargs(rate))
    session.start()
    result.start_time = time.time()
    try:
        if is_fast:
            stream.elapsed_s = await run_pfb_source(
                session, host, channels, duration_s=duration_s)
        else:
            stream.elapsed_s = await run_slow_source(
                session, host, module=module, duration_s=duration_s)
    finally:
        session.stop()

    stream.noise = dict(session.noise_stats)
    setattr(result, "fast" if is_fast else "slow", stream)

    if session.state.name == "ESTIMATING" and verbose:
        print("[trigger_capture] noise training never completed — the "
              "stream ended first; try a longer time_run or a smaller "
              "max_pulse_ms")


async def _run_dual(result, crs, host, channels, module, slow_rate,
                    duration_s, hdf5_path, verbose) -> None:
    slow = StreamResult(sample_rate=slow_rate)
    fast = StreamResult(sample_rate=PFB_SAMPLE_RATE)
    collectors = {"slow": _collector(slow), "fast": _collector(fast)}

    session = DualPulseCaptureSession(
        channels=channels, module=module, slow_rate=slow_rate,
        fast_rate=PFB_SAMPLE_RATE, config=result.config,
        hdf5_path=hdf5_path,
        on_pulse=lambda s, ch, idx, summary, wf:
            collectors[s](ch, idx, summary, wf),
        on_pair=result.pairs.append,
        on_error=(lambda m: print(f"[trigger_capture] {m}")) if verbose
        else None)
    session.start()
    result.start_time = time.time()
    try:
        slow.elapsed_s, fast.elapsed_s = await run_dual_source(
            session, host, channels, module=module, duration_s=duration_s)
    finally:
        session.stop()

    slow.noise = dict(session.slow.noise_stats)
    fast.noise = dict(session.fast.noise_stats)
    result.slow = slow
    result.fast = fast
