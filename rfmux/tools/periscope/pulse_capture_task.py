"""
Background worker running a
:class:`~rfmux.pulse_capture.capture_session.PulseCaptureSession` for
Periscope.

Architecture: all capture logic (noise estimation, detection, HDF5,
histograms) lives in :mod:`rfmux.pulse_capture`, which knows nothing
about Qt.  This QThread only

1. feeds the session from its own thread (h5py writes must stay on
   one thread): in slow mode by draining the queue the GUI-thread tap
   fills (:meth:`PulseCaptureTask.enqueue_packet`) into
   :meth:`PulseCaptureSession.feed_block`; in fast mode by running the
   shared PFB socket source; in both mode by running the shared dual
   source with the tap as its slow side,
2. services control requests (re-estimate, waveform fetch) from the
   same queue, and
3. re-emits the session callbacks as Qt signals.

Waveforms never cross threads through signals — ``pulse_detected``
carries only the scalar summary.  The task keeps a bounded in-memory
cache of recent waveforms (:meth:`get_pulse`) so the live Pulse View
can display events without opening the HDF5 file the writer holds.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time

import numpy as np
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSignal

from ...pulse_capture.capture_session import (
    CaptureState,
    PulseCaptureSession,
)
from ...pulse_capture.sources import SlowIngest, pfb_streamer_mismatch


class PulseCaptureSignals(QtCore.QObject):
    """Signal bundle for :class:`PulseCaptureTask` (constructed by the caller)."""

    noise_estimated = pyqtSignal(dict)      # {channel: ChannelNoiseStats};
    #                                         both-mode: {stream, stats}
    noise_progress = pyqtSignal(dict)       # {collected: {ch: n}, target: N}
    pulse_detected = pyqtSignal(int, int, dict)  # channel, pulse_idx, summary
    #                                         (both-mode summary has "stream")
    pair_matched = pyqtSignal(dict)         # lean pair meta (both-mode)
    stats_updated = pyqtSignal(dict)        # PulseCaptureSession.stats()
    histograms_updated = pyqtSignal(dict)   # PulseHistogramSet.get_histogram_data()
    templates_updated = pyqtSignal(dict)    # trigger-aligned stack arrays
    waveform_ready = pyqtSignal(int, int)   # channel, pulse_idx (cache warmed)
    error = pyqtSignal(str)
    failed = pyqtSignal(str)                # the capture cannot run; finished follows
    finished = pyqtSignal()


class PulseCaptureTask(QtCore.QThread):
    """Runs a PulseCaptureSession in a worker thread.

    Parameters
    ----------
    session : PulseCaptureSession
        A fully configured (but not started) session.  The task installs
        its own signal-emitting callbacks on it.
    signals : PulseCaptureSignals
        Signal bundle constructed by the caller.
    queue_size : int
        Bounded queue capacity in tap batches, each up to
        ``SlowIngest.DEFAULT_MAX_PACKETS`` packets or
        ``SlowIngest.DEFAULT_MAX_AGE_S`` of arrivals.  On overflow the
        batch is dropped and ``error`` reports it once.
    waveform_cache : int
        Number of recent pulse waveforms kept for :meth:`get_pulse`.
    """

    def __init__(
        self,
        session: PulseCaptureSession,
        signals: PulseCaptureSignals,
        queue_size: int = 8_192,
        waveform_cache: int = 200,
        parent: Optional[QtCore.QObject] = None,
        mode: str = "slow",
        crs=None,
        host: Optional[str] = None,
        module: int = 1,
    ):
        super().__init__(parent)
        self.session = session
        self.signals = signals
        self.mode = mode
        self.crs = crs
        self.host = host
        self.module = module
        self.sample_queue: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._overflow_reported = False

        # Tap batch, filled on the GUI thread (see enqueue_packet).
        self._tap_channels = None
        self._tap_values: list = []
        self._tap_stamps: list = []
        self._tap_rows = 0
        self._tap_day = None
        self._tap_opened = 0.0

        self._cache_size = waveform_cache
        self._cache: "OrderedDict[Tuple[int, int], dict]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._stop_requested = False

        self._pair_cache: "OrderedDict[Tuple[int, int], dict]" = OrderedDict()

        # Route session callbacks through Qt signals (called in run() thread)
        session.on_stats = lambda s: self.signals.stats_updated.emit(s)
        session.on_error = lambda msg: self.signals.error.emit(msg)
        if mode == "both":
            session.on_noise = lambda stream, ns: \
                self.signals.noise_estimated.emit(
                    {"stream": stream, "stats": self._noise_snapshot(ns)})
            session.on_pulse = self._on_stream_pulse
            session.on_pair = self._on_pair
            session.on_histograms = lambda stream, d: \
                self.signals.histograms_updated.emit(
                    {"stream": stream, "data": d})
            session.on_templates = lambda stream, d: \
                self.signals.templates_updated.emit(
                    {"stream": stream, "data": d})
            for inner in (session.slow, session.fast):
                inner.on_progress = lambda p, s=inner.streamer_mode: \
                    self.signals.noise_progress.emit({**p, "stream": s})
        else:
            session.on_noise = lambda ns: \
                self.signals.noise_estimated.emit(self._noise_snapshot(ns))
            session.on_progress = lambda p: \
                self.signals.noise_progress.emit(p)
            session.on_pulse = self._on_pulse
            session.on_histograms = lambda d: \
                self.signals.histograms_updated.emit(d)
            session.on_templates = lambda d: \
                self.signals.templates_updated.emit(d)

    @staticmethod
    def _noise_snapshot(noise_stats: dict) -> dict:
        """Copies: the engine re-centres the live objects on every
        baseline refresh, and the GUI shows the estimate it was told."""
        return {c: replace(ns) for c, ns in noise_stats.items()}

    # ── GUI-thread API ────────────────────────────────────────────

    #: Rows gathered before one hand-off to the worker: a queue put per
    #: packet contends with the worker's get.  SlowIngest's gather
    #: limits, which exist for the same reason.
    _TAP_BATCH_PACKETS = SlowIngest.DEFAULT_MAX_PACKETS
    _TAP_BATCH_MAX_S = SlowIngest.DEFAULT_MAX_AGE_S

    def enqueue_packet(self, channels, values: np.ndarray, timestamp,
                       day_epoch: Optional[float] = None) -> None:
        """Tap callback, from the GUI thread once per batch: ``values``
        is (packets, channels) with one stamp per row, NaN for none, or
        one packet's row with its stamp.  Rows are gathered here and
        handed over in batches.
        """
        if day_epoch is not None and day_epoch != self._tap_day:
            self._tap_day = day_epoch
            self.flush_tap()
            self._send_control(("__day__", day_epoch))
        if channels != self._tap_channels:
            self.flush_tap()
            self._tap_channels = channels
        if not self._tap_values:
            self._tap_opened = time.monotonic()
        values = np.asarray(values)
        if values.ndim == 1:
            values = values[None, :]
            timestamp = np.array([float("nan") if timestamp is None
                                  else float(timestamp)])
        self._tap_values.append(values)
        self._tap_stamps.append(np.asarray(timestamp, dtype=np.float64))
        self._tap_rows += values.shape[0]
        if (self._tap_rows >= self._TAP_BATCH_PACKETS
                or (time.monotonic() - self._tap_opened
                    >= self._TAP_BATCH_MAX_S)):
            self.flush_tap()

    def flush_tap(self) -> None:
        """Hand over whatever is gathered, however little.

        The runtime calls this at the end of every frame: the
        batch size alone would strand the tail of a capture
        whenever the stream pauses, because the age check only
        runs when the NEXT packet arrives.
        """
        if not self._tap_values:
            return
        item = (self._tap_channels, self._tap_values, self._tap_stamps)
        self._tap_values = []
        self._tap_stamps = []
        self._tap_rows = 0
        try:
            self.sample_queue.put_nowait(item)
        except queue.Full:
            if not self._overflow_reported:
                self._overflow_reported = True
                self.signals.error.emit(
                    "Capture worker queue full; tap samples are being "
                    "dropped — the worker is not keeping up")

    def request_stop(self) -> None:
        """Ask the worker to finish; session.stop() runs in the worker."""
        # Whatever is still gathered belongs to the capture; the worker
        # drains the queue once it sees the flag, so hand it over first.
        self.flush_tap()
        self._stop_requested = True
        self.requestInterruption()

    def request_noise_reestimate(self) -> None:
        """Queue a noise re-estimation (executed in the worker thread)."""
        self._send_control(("__reestimate__",))

    def get_pulse(self, channel: int, pulse_idx: int,
                  stream: Optional[str] = None) -> Optional[dict]:
        """Waveform dict for a recent pulse, or None if evicted."""
        with self._cache_lock:
            return self._cache.get((stream, channel, pulse_idx))

    def get_pair(self, channel: int, pair_idx: int) -> Optional[dict]:
        """Full pair dict (summaries + any cross-stream TODs)."""
        with self._cache_lock:
            return self._pair_cache.get((channel, pair_idx))

    def _send_control(self, item: tuple) -> None:
        """A control item to the worker; a full queue is reported, not
        swallowed, since a dropped one is a request that never runs."""
        try:
            self.sample_queue.put_nowait(item)
        except queue.Full:
            self.signals.error.emit(
                f"Capture worker queue full; dropped {item[0]}")

    def _should_stop(self) -> bool:
        return self._stop_requested or self.isInterruptionRequested()

    def request_pair(self, channel: int, pair_idx: int) -> None:
        """Ask the worker to load an evicted pair, windows and all, from
        the live file; ``waveform_ready`` fires when it is cached."""
        self._send_control(("__fetch_pair__", channel, pair_idx))

    def request_waveform(self, channel: int, pulse_idx: int,
                         stream: Optional[str] = None) -> None:
        """Ask the worker to load an evicted waveform from the live HDF5
        file (writer's own handle, writer's thread).  ``waveform_ready``
        fires when the cache has been warmed (or the fetch failed)."""
        self._send_control(("__fetch__", channel, pulse_idx, stream))

    # ── Worker thread ─────────────────────────────────────────────

    def run(self) -> None:
        try:
            if self.mode == "fast":
                asyncio.run(self._run_fast())
            elif self.mode == "both":
                asyncio.run(self._run_both())
            else:
                self.session.start()
                asyncio.run(self._slow_tap_pump(self._should_stop,
                                                self.session.feed_block))
        except Exception as e:
            self.signals.failed.emit(f"Pulse capture worker failed: {e}")
        finally:
            try:
                self.session.stop()
            except Exception as e:
                self.signals.error.emit(f"Session stop failed: {e}")
            self.signals.finished.emit()

    async def _pfb_mismatch(self, channels) -> Optional[str]:
        return await pfb_streamer_mismatch(self.crs, self.module, channels)

    async def _start_after_streamer_check(self) -> Optional[list]:
        """The channels to capture once the fast streamer is confirmed to
        carry them, or None with ``failed`` emitted.

        The session starts only after the check: a dual session opens
        its HDF5 file on start, and a capture that cannot run must not
        replace the previous file with an empty one.
        """
        channels = list(self.session.channels)
        problem = await self._pfb_mismatch(channels)
        if problem:
            self.signals.failed.emit(problem)
            return None
        self.session.start()
        return channels

    async def _run_fast(self) -> None:
        """PFB capture: run the shared source, keep servicing control
        requests."""
        from ...pulse_capture.sources import run_pfb_source

        channels = await self._start_after_streamer_check()
        if channels is None:
            return
        stop = self._should_stop
        pump = asyncio.ensure_future(self._control_pump())
        try:
            await run_pfb_source(self.session, self.host, channels,
                                 module=self.module, should_stop=stop)
        finally:
            pump.cancel()
            try:
                await pump
            except asyncio.CancelledError:
                pass

    async def _run_both(self) -> None:
        """Concurrent slow+fast capture (DualPulseCaptureSession).

        The gather, the shared stop and the fast socket all live in
        run_dual_source — this adds only the streamer check and the
        tap-fed slow side.  The slow side comes from the Periscope
        tap because this process already holds every slow packet: a
        second socket would cost kernel copies and another drain thread
        in a GUI that is GIL-bound at stage 0.  Both routes drive
        SlowIngest, so only the transport differs.
        """
        from ...pulse_capture.sources import (
            run_dual_source,
        )

        channels = await self._start_after_streamer_check()
        if channels is None:
            return
        stop = self._should_stop
        watchdog = asyncio.ensure_future(self._dual_watchdog(stop))
        try:
            await run_dual_source(
                self.session, self.host, channels, module=self.module,
                should_stop=stop,
                slow_source=lambda stop: self._slow_tap_pump(
                    stop, self.session.feed_slow_block))
        finally:
            watchdog.cancel()

    async def _dual_watchdog(self, stop) -> None:
        """Report a dual capture that is training instead of capturing.

        Both streams have to finish noise training before either may
        trigger (freeze in _sync_capture_start), so a stream stuck in
        ESTIMATING silences the whole capture without saying why.
        """
        waited = 0.0
        while not stop():
            await asyncio.sleep(15.0)
            waited += 15.0
            states = self.session.state
            stuck = [name for name, st in states.items()
                     if st == CaptureState.ESTIMATING.value]
            if not stuck:
                return
            done = [n for n in states if n not in stuck]
            lag = " and ".join(stuck)
            self.signals.error.emit(
                f"Still training noise on the {lag} stream after "
                f"{waited:.0f} s"
                + (f" ({', '.join(done)} finished; no stream may trigger "
                   "until both have)" if done else "")
                + ". Check that the stream is being sent and that its "
                "training length fits the capture.")

    async def _slow_tap_pump(self, stop, feed) -> float:
        """Drain the tap-fed queue into *feed* (the session's slow block
        feed) until *stop*, then whatever is still queued: request_stop
        hands the gathered tail over before raising the flag, and the
        worker may be mid-item when it does.  Warns once if no slow
        samples arrive at all.

        Everything past "here is a packet" is SlowIngest, shared with
        the headless socket source, so the GUI is a caller rather than
        a second implementation.  Only the transport differs: this
        process already holds the packets.

        Returns the sample time SlowIngest accumulated, on the same
        clock the socket source uses -- so run_dual_source's
        ``slow_elapsed`` means one thing whichever route filled it.
        """
        fed = 0
        warned = False
        t_start = time.monotonic()
        ingest = SlowIngest(feed)
        while not stop():
            try:
                item = self.sample_queue.get_nowait()
            except queue.Empty:
                ingest.flush()
                self.session.flush_progress()
                if (not warned and fed == 0
                        and time.monotonic() - t_start > 5.0):
                    warned = True
                    self.signals.error.emit(
                        "No slow samples arriving via the tap after "
                        "5 s — is the slow stream running?")
                await asyncio.sleep(0.02)
                continue
            fed += self._ingest_item(ingest, item)
        while True:
            try:
                item = self.sample_queue.get_nowait()
            except queue.Empty:
                break
            self._ingest_item(ingest, item)
        ingest.flush()
        return ingest.elapsed

    def _ingest_item(self, ingest: SlowIngest, item) -> int:
        """One queue item: a control request, or tap blocks into
        *ingest*.  Returns the number of blocks fed."""
        if self._handle_control(item):
            return 0
        channels, values, stamps = item
        for block, block_stamps in zip(values, stamps):
            ingest.add_block(channels, block, block_stamps)
        return len(values)

    async def _control_pump(self) -> None:
        """Service control items (see _handle_control) while a socket
        source owns the sample flow (fast mode)."""
        while True:
            try:
                item = self.sample_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
            self._handle_control(item)

    def _handle_control(self, item) -> bool:
        """Handle a control tuple; returns True when consumed."""
        if item[0] == "__reestimate__":
            # The session owns the rule: the dual one retrains each
            # stream that is capturing, the single one refuses outside
            # CAPTURING -- and a refused request is reported, not lost.
            try:
                self.session.re_estimate_noise()
            except RuntimeError as e:
                self.signals.error.emit(str(e))
            return True
        if item[0] == "__day__":
            self.session.set_time_origin(item[1])
            return True
        if item[0] == "__fetch_pair__":
            _, ch, idx = item
            self._fetch(self._pair_cache, (ch, idx),
                        lambda w: w.read_match(ch, idx),
                        f"Pair read failed for ch{ch} pair {idx}", ch, idx)
            return True
        if item[0] == "__fetch__":
            _, ch, idx, stream = item
            self._fetch(self._cache, (stream, ch, idx),
                        lambda w: (w.read_pulse(stream, ch, idx)
                                   if self.mode == "both"
                                   else w.read_pulse(ch, idx)),
                        f"Waveform read failed for ch{ch}#{idx}", ch, idx)
            return True
        return False

    def _fetch(self, cache, key, read, what: str, ch: int, idx: int) -> None:
        """Bring an evicted item back from the live file into *cache*;
        waveform_ready fires either way so the view redraws."""
        item = None
        writer = self.session.writer
        if writer is not None:
            try:
                item = read(writer)
            except Exception as e:
                self.signals.error.emit(f"{what}: {e}")
        if item is not None:
            with self._cache_lock:
                cache[key] = item
        self.signals.waveform_ready.emit(ch, idx)

    # ── Session callback (worker thread) ──────────────────────────

    def _on_pulse(self, channel: int, pulse_idx: int,
                  summary: Dict[str, Any], pulse_data: dict) -> None:
        self._on_stream_pulse(None, channel, pulse_idx, summary, pulse_data)

    def _on_stream_pulse(self, stream: Optional[str], channel: int,
                         pulse_idx: int, summary: Dict[str, Any],
                         pulse_data: dict) -> None:
        with self._cache_lock:
            self._cache[(stream, channel, pulse_idx)] = pulse_data
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        summary = dict(summary)
        if stream is not None:
            summary["stream"] = stream
        self.signals.pulse_detected.emit(channel, pulse_idx, summary)

    def _on_pair(self, pair: dict) -> None:
        key = (pair["channel"], pair["pair_idx"])
        with self._cache_lock:
            self._pair_cache[key] = pair
            while len(self._pair_cache) > self._cache_size:
                self._pair_cache.popitem(last=False)
        lean = {k: pair.get(k) for k in
                ("channel", "pair_idx", "slow_idx", "fast_idx",
                 "time_offset", "slow_summary", "fast_summary")}
        lean["has_slow_tod"] = pair.get("slow_tod") is not None
        lean["has_fast_tod"] = pair.get("fast_tod") is not None
        self.signals.pair_matched.emit(lean)
