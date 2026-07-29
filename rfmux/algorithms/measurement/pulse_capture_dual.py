"""
Concurrent slow+fast pulse capture with incremental cross-stream matching.

:class:`DualPulseCaptureSession` composes two
:class:`~.pulse_capture_session.PulseCaptureSession` instances (slow
readout stream + fast PFB stream) with an
:class:`IncrementalPulseMatcher` and a
:class:`~.pulse_hdf5.DualPulseHDF5Writer`.  It is the streaming
counterpart of ``trigger_capture(streamer_mode="both")``: pulses seen
in both streams within the match window become pairs; pulses seen in
only one stream are emitted after a grace period with the other
stream's time-window extracted from its ring buffer (cross-stream TOD).

Feed it from any source::

    dual = DualPulseCaptureSession(channels=[1, 2], slow_rate=fs, ...)
    dual.start()
    # e.g. run_slow_source(dual.slow, ...) + run_pfb_source(dual.fast, ...)
    dual.stop()

Like everything in this package, no Qt and no sockets — the Periscope
task and headless scripts drive it identically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .pulse_capture_session import (
    CaptureState,
    PulseCaptureConfig,
    PulseCaptureSession,
)
from .streamer_config import PFB_SAMPLE_RATE

try:
    from .pulse_hdf5 import DualPulseHDF5Writer
except ImportError:  # pragma: no cover - h5py missing
    DualPulseHDF5Writer = None  # type: ignore[assignment]


# ───────────────────────── Incremental matcher ──────────────────────

@dataclass
class _Pending:
    pulse_idx: int
    mid: float
    summary: dict


class IncrementalPulseMatcher:
    """Streaming version of trigger_capture's ``_match_pulses``.

    Pulses from the two streams are paired per channel when their
    midpoint times fall within ``window_s`` (best match wins).  A pulse
    that finds no partner within ``grace_s`` of stream time is emitted
    as a one-sided pair.  ``on_pair`` receives dicts::

        {"channel", "pair_idx", "slow_idx" | None, "fast_idx" | None,
         "time_offset" | None, "slow_summary" | None,
         "fast_summary" | None}
    """

    def __init__(self, window_s: float = 0.05, grace_s: float = 0.25,
                 on_pair: Optional[Callable[[dict], None]] = None):
        self.window_s = window_s
        self.grace_s = grace_s
        self.on_pair = on_pair
        self._pending: Dict[str, Dict[int, List[_Pending]]] = {
            "slow": {}, "fast": {}}
        self._pair_counts: Dict[int, int] = {}
        self._latest: Optional[float] = None
        self.matched = 0
        self.unmatched = 0

    @staticmethod
    def _other(stream: str) -> str:
        return "fast" if stream == "slow" else "slow"

    def add(self, stream: str, channel: int, pulse_idx: int,
            summary: dict) -> None:
        mid = summary.get("timestamp", 0.0) \
            + summary.get("duration_s", 0.0) / 2.0
        if not math.isfinite(mid):
            return

        if self._latest is None or mid > self._latest:
            self._latest = mid

        # Best partner in the other stream's pending list
        others = self._pending[self._other(stream)].setdefault(channel, [])
        best_i, best_diff = -1, self.window_s
        for i, cand in enumerate(others):
            diff = abs(cand.mid - mid)
            if diff <= best_diff:
                best_i, best_diff = i, diff
        if best_i >= 0:
            partner = others.pop(best_i)
            mine = _Pending(pulse_idx, mid, summary)
            slow, fast = ((mine, partner) if stream == "slow"
                          else (partner, mine))
            self.matched += 1
            self._emit(channel, slow=slow, fast=fast)
        else:
            self._pending[stream].setdefault(channel, []).append(
                _Pending(pulse_idx, mid, summary))

        self._expire(self._latest - self.grace_s)

    def flush(self) -> None:
        """Emit every remaining pending pulse as one-sided (on stop)."""
        self._expire(float("inf"))

    # ── internals ─────────────────────────────────────────────────

    def _expire(self, cutoff: float) -> None:
        for stream in ("slow", "fast"):
            for channel, pend in self._pending[stream].items():
                keep: List[_Pending] = []
                for p in pend:
                    if p.mid < cutoff:
                        self.unmatched += 1
                        self._emit(channel,
                                   slow=p if stream == "slow" else None,
                                   fast=p if stream == "fast" else None)
                    else:
                        keep.append(p)
                self._pending[stream][channel] = keep

    def _emit(self, channel: int, slow: Optional[_Pending],
              fast: Optional[_Pending]) -> None:
        self._pair_counts[channel] = self._pair_counts.get(channel, 0) + 1
        pair = {
            "channel": channel,
            "pair_idx": self._pair_counts[channel],
            "slow_idx": slow.pulse_idx if slow else None,
            "fast_idx": fast.pulse_idx if fast else None,
            "slow_summary": slow.summary if slow else None,
            "fast_summary": fast.summary if fast else None,
            "time_offset": (slow.mid - fast.mid)
            if slow and fast else None,
        }
        if self.on_pair is not None:
            self.on_pair(pair)


# ───────────────────────── Dual session ─────────────────────────────

class DualPulseCaptureSession:
    """Slow + fast capture with live matching and one dual HDF5 file.

    Callbacks (all optional):

    - ``on_noise(stream, {ch: ChannelNoiseStats})``
    - ``on_pulse(stream, channel, pulse_idx, summary, pulse_data)``
    - ``on_pair(pair_dict)`` — pair_dict as from the matcher, plus
      ``slow_tod``/``fast_tod`` cross-stream windows on one-sided pairs
      when the other ring still covers the interval
    - ``on_stats(stats_dict)`` — merged per-stream stats + pair counts
    - ``on_histograms(stream, data)``
    - ``on_error(message)``
    """

    def __init__(
        self,
        channels: List[int],
        *,
        module: int = 1,
        slow_rate: float,
        fast_rate: float = PFB_SAMPLE_RATE,
        config: Optional[PulseCaptureConfig] = None,
        hdf5_path=None,
        df_calibrations: Optional[Dict[int, float]] = None,
        match_window_s: float = 0.05,
        match_grace_s: float = 0.25,
        on_noise: Optional[Callable] = None,
        on_pulse: Optional[Callable] = None,
        on_pair: Optional[Callable] = None,
        on_stats: Optional[Callable] = None,
        on_histograms: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ):
        self.channels = list(channels)
        self.module = module
        self.config = config or PulseCaptureConfig()
        # Parity with PulseCaptureSession (panel/task read this)
        from pathlib import Path
        self.hdf5_path = Path(hdf5_path) if hdf5_path is not None else None
        self.on_noise = on_noise
        self.on_pulse = on_pulse
        self.on_pair = on_pair
        self.on_stats = on_stats
        self.on_histograms = on_histograms
        self.on_error = on_error

        self.writer = None
        if hdf5_path is not None:
            if DualPulseHDF5Writer is None:
                self._error("h5py not available — capturing without HDF5")
            else:
                try:
                    self.writer = DualPulseHDF5Writer(
                        hdf5_path, self.channels,
                        capture_params={
                            "streamer_mode": "both",
                            "threshold_sigma": self.config.threshold_sigma,
                            "end_sigma": self.config.end_sigma,
                            "margin_fraction": self.config.margin_fraction,
                            "enable_pileup": self.config.enable_pileup,
                            "module": module,
                            "sample_rate_slow": slow_rate,
                            "sample_rate_fast": fast_rate,
                        },
                        df_calibrations=df_calibrations)
                except Exception as e:
                    self._error(f"Could not open HDF5 file "
                                f"{hdf5_path}: {e}")

        self.matcher = IncrementalPulseMatcher(
            window_s=match_window_s, grace_s=match_grace_s,
            on_pair=self._on_matcher_pair)

        self.slow = self._make_stream("slow", slow_rate)
        self.fast = self._make_stream("fast", fast_rate)

    def _make_stream(self, stream: str,
                     sample_rate: float) -> PulseCaptureSession:
        return PulseCaptureSession(
            channels=self.channels,
            module=self.module,
            streamer_mode=stream,
            sample_rate=sample_rate,
            hdf5_path=None,  # the dual writer owns the file
            on_noise=lambda ns, s=stream: self._on_stream_noise(s, ns),
            on_pulse=lambda ch, idx, summary, data, s=stream:
                self._on_stream_pulse(s, ch, idx, summary, data),
            on_stats=lambda _s, s=stream: self._emit_stats(),
            on_histograms=lambda data, s=stream:
                self._on_stream_histograms(s, data),
            on_error=self._error,
            **self.config.session_kwargs(sample_rate),
        )

    # ── Lifecycle / feeding ───────────────────────────────────────

    def start(self) -> None:
        self.slow.start()
        self.fast.start()

    def feed_slow(self, ch: int, i: float, q: float, t) -> None:
        self.slow.feed_sample(ch, i, q, t)

    def feed_fast(self, ch: int, i: float, q: float, t) -> None:
        self.fast.feed_sample(ch, i, q, t)

    def re_estimate_noise(self) -> None:
        """Freeze both streams and retrain their noise statistics."""
        for session in (self.slow, self.fast):
            if session.state is CaptureState.CAPTURING:
                session.re_estimate_noise()

    def stop(self) -> None:
        self.slow.stop()
        self.fast.stop()
        self.matcher.flush()
        if self.writer is not None:
            try:
                self.writer.finalize()
            except Exception as e:
                self._error(f"HDF5 finalize failed: {e}")

    @property
    def state(self) -> Dict[str, str]:
        return {"slow": self.slow.state.value,
                "fast": self.fast.state.value}

    def stats(self) -> Dict[str, Any]:
        return {
            "slow": self.slow.stats(),
            "fast": self.fast.stats(),
            "pairs_matched": self.matcher.matched,
            "pairs_unmatched": self.matcher.unmatched,
            "total_pulses": (self.slow.total_pulses
                             + self.fast.total_pulses),
        }

    # ── Internal wiring ───────────────────────────────────────────

    def _on_stream_noise(self, stream: str, noise_stats: dict) -> None:
        if self.writer is not None:
            try:
                self.writer.set_noise_stats(stream, noise_stats)
            except Exception as e:
                self._error(f"HDF5 noise write failed: {e}")
        self._callback(self.on_noise, stream, noise_stats)

    def _on_stream_pulse(self, stream: str, channel: int, pulse_idx: int,
                         summary: dict, pulse_data: dict) -> None:
        if self.writer is not None:
            try:
                self.writer.append_pulse(stream, channel, pulse_idx,
                                         pulse_data)
            except Exception as e:
                self._error(f"HDF5 write failed for {stream} "
                            f"ch{channel}#{pulse_idx}: {e}")
        self._callback(self.on_pulse, stream, channel, pulse_idx,
                       summary, pulse_data)
        self.matcher.add(stream, channel, pulse_idx, summary)

    def _on_stream_histograms(self, stream: str, data: dict) -> None:
        if self.writer is not None:
            try:
                self.writer.update_histograms(stream, data)
            except Exception as e:
                self._error(f"HDF5 histogram update failed: {e}")
        self._callback(self.on_histograms, stream, data)

    def _on_matcher_pair(self, pair: dict) -> None:
        # One-sided pair: pull the matching time window from the OTHER
        # stream's ring buffer while it still covers the interval.
        try:
            if pair["fast_idx"] is None and pair["slow_summary"]:
                pair["fast_tod"] = self._cross_tod(
                    self.fast, pair["channel"], pair["slow_summary"])
            elif pair["slow_idx"] is None and pair["fast_summary"]:
                pair["slow_tod"] = self._cross_tod(
                    self.slow, pair["channel"], pair["fast_summary"])
        except Exception as e:
            self._error(f"Cross-stream TOD extraction failed: {e}")

        if self.writer is not None:
            try:
                self.writer.append_match(pair["channel"], pair)
            except Exception as e:
                self._error(f"HDF5 match write failed: {e}")
        self._callback(self.on_pair, pair)
        self._emit_stats()

    @staticmethod
    def _cross_tod(session: PulseCaptureSession, channel: int,
                   summary: dict) -> Optional[dict]:
        if session.pcap is None:
            return None
        t0 = summary.get("timestamp", 0.0)
        dur = summary.get("duration_s", 0.0)
        margin = max(dur * 0.25, 1e-4)
        return session.pcap.get_window_by_time(
            channel, t0 - margin, t0 + dur + margin)

    def _emit_stats(self) -> None:
        self._callback(self.on_stats, self.stats())

    def _callback(self, cb: Optional[Callable], *args) -> None:
        if cb is None:
            return
        try:
            cb(*args)
        except Exception as e:
            self._error(f"Callback raised: {e}")

    def _error(self, message: str) -> None:
        if self.on_error is not None:
            try:
                self.on_error(message)
            except Exception:
                pass
