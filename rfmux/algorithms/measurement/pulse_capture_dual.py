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
        # Per-stream clocks: a pulse in stream S may only be declared
        # one-sided once the OTHER stream's time has provably passed it
        # — robust to stream skew (sequential feeding, stalled socket).
        self._latest: Dict[str, Optional[float]] = {
            "slow": None, "fast": None}
        self.matched = 0
        self.unmatched = 0

    @staticmethod
    def _other(stream: str) -> str:
        return "fast" if stream == "slow" else "slow"

    def advance_time(self, stream: str, now: float) -> None:
        """Advance *stream*'s clock from stream time (not just pulse
        arrivals) so the OTHER stream's one-sided pulses expire
        ~grace_s after the event — while ring buffers still cover it —
        instead of waiting for the next pulse or capture stop."""
        if not math.isfinite(now):
            return
        latest = self._latest[stream]
        if latest is None or now > latest:
            self._latest[stream] = now
            self._expire()

    def add(self, stream: str, channel: int, pulse_idx: int,
            summary: dict) -> None:
        mid = summary.get("timestamp", 0.0) \
            + summary.get("duration_s", 0.0) / 2.0
        if not math.isfinite(mid):
            return

        latest = self._latest[stream]
        if latest is None or mid > latest:
            self._latest[stream] = mid

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

        self._expire()

    def flush(self) -> None:
        """Emit every remaining pending pulse as one-sided (on stop)."""
        self._expire(force=True)

    # ── internals ─────────────────────────────────────────────────

    def _expire(self, force: bool = False) -> None:
        for stream in ("slow", "fast"):
            if force:
                cutoff = float("inf")
            else:
                other_latest = self._latest[self._other(stream)]
                if other_latest is None:
                    continue  # other stream has no time yet — can't judge
                cutoff = other_latest - self.grace_s
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


class _StreamFeed:
    """Source-compatible facade: run_slow_source/run_pfb_source call
    ``feed_sample`` and read ``channels`` — this routes them through the
    dual session so stream time drives matcher expiry."""

    def __init__(self, dual: "DualPulseCaptureSession", feed):
        self._dual = dual
        self._feed = feed

    @property
    def channels(self):
        return self._dual.channels

    def feed_sample(self, ch: int, i: float, q: float, t) -> None:
        self._feed(ch, i, q, t)


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
        on_templates: Optional[Callable] = None,
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
        self.on_templates = on_templates
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
        self._last_advance: Dict[str, float] = {}

        self.slow = self._make_stream("slow", slow_rate)
        self.fast = self._make_stream("fast", fast_rate)
        #: source-compatible facades (feed_sample + channels) that route
        #: through feed_slow/feed_fast so stream time advances the matcher
        self.slow_feed = _StreamFeed(self, self.feed_slow)
        self.fast_feed = _StreamFeed(self, self.feed_fast)

    def _make_stream(self, stream: str,
                     sample_rate: float) -> PulseCaptureSession:
        kwargs = self.config.session_kwargs(sample_rate)
        # Union-window extraction happens up to grace_s after a pulse
        # (single-trigger expiry): the ring must cover the full window
        # PLUS the grace, or the extraction races the ring.
        grace = self.matcher.grace_s
        min_buf = int((self.config.max_pulse_ms / 1e3 * 1.5
                       + grace + 0.1) * sample_rate)
        kwargs["buf_size"] = max(kwargs["buf_size"], min_buf)
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
            on_templates=lambda data, s=stream:
                self._on_stream_templates(s, data),
            on_error=self._error,
            **kwargs,
        )

    # ── Lifecycle / feeding ───────────────────────────────────────

    def start(self) -> None:
        self.slow.start()
        self.fast.start()

    def feed_slow(self, ch: int, i: float, q: float, t) -> None:
        self.slow.feed_sample(ch, i, q, t)
        self._advance_matcher("slow", t)

    def feed_fast(self, ch: int, i: float, q: float, t) -> None:
        self.fast.feed_sample(ch, i, q, t)
        self._advance_matcher("fast", t)

    def _advance_matcher(self, stream: str, t) -> None:
        # Throttled: expiry sweep at most every 20 ms of stream time
        if t is not None and t - self._last_advance.get(stream,
                                                        float("-inf")) > 0.02:
            self._last_advance[stream] = t
            self.matcher.advance_time(stream, t)

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
        self._sync_capture_start()
        self._callback(self.on_noise, stream, noise_stats)

    def _sync_capture_start(self) -> None:
        """Hold triggering until BOTH streams have finished training.

        The two streams need the same training span in sample time but
        reach it at different moments — the fast one collects its
        samples far quicker and is additionally capped, so it can
        finish seconds ahead.  Left alone it triggers into a partner
        that has no ring yet, and every pair comes out one-sided with
        "window unavailable".  Its buffers keep filling while frozen,
        so nothing is lost by waiting.
        """
        both_ready = all(s.state is CaptureState.CAPTURING
                         for s in (self.slow, self.fast))
        for s in (self.slow, self.fast):
            if s.pcap is not None:
                s.pcap.freeze_triggers = not both_ready

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

    def _on_stream_templates(self, stream: str, data: dict) -> None:
        if self.writer is not None:
            try:
                self.writer.update_templates(stream, data)
            except Exception as e:
                self._error(f"HDF5 template update failed: {e}")
        self._callback(self.on_templates, stream, data)

    def _on_matcher_pair(self, pair: dict) -> None:
        # EVERY pair carries both streams over the UNION time window
        # (widest interval spanned by the trigger(s), plus margin),
        # extracted from the ring buffers.  The per-stream triggered
        # records stay untouched — metrics (SNR, tau, histograms) are
        # computed on the physically-triggered cores only; the union
        # windows are the pair's display/analysis data.
        try:
            union = self._union_window(pair)
            if union is not None:
                t0, t1 = union
                for stream, session in (("slow", self.slow),
                                        ("fast", self.fast)):
                    if session.pcap is not None:
                        pair[f"{stream}_tod"] = \
                            session.pcap.get_window_by_time(
                                pair["channel"], t0, t1)
        except Exception as e:
            self._error(f"Union-window extraction failed: {e}")

        if self.writer is not None:
            try:
                self.writer.append_match(pair["channel"], pair)
            except Exception as e:
                self._error(f"HDF5 match write failed: {e}")
        self._callback(self.on_pair, pair)
        self._emit_stats()

    @staticmethod
    def _union_window(pair: dict) -> Optional[tuple]:
        """[t0, t1] spanning every available trigger window + 10% margin."""
        t0 = t1 = None
        for key in ("slow_summary", "fast_summary"):
            summ = pair.get(key)
            if not summ:
                continue
            s0 = summ.get("timestamp", 0.0)
            s1 = s0 + summ.get("duration_s", 0.0)
            if not (math.isfinite(s0) and math.isfinite(s1)):
                continue
            t0 = s0 if t0 is None else min(t0, s0)
            t1 = s1 if t1 is None else max(t1, s1)
        if t0 is None:
            return None
        margin = max((t1 - t0) * 0.1, 1e-4)
        return (t0 - margin, t1 + margin)

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
