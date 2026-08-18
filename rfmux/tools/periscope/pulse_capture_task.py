"""
Background worker bridging the Periscope slow-stream tap to a
:class:`~rfmux.algorithms.measurement.pulse_capture_session.PulseCaptureSession`.

Architecture: all capture logic (noise estimation, detection, HDF5,
histograms) lives in the algorithms layer.  This QThread only

1. drains a thread-safe queue that the GUI-thread tap fills
   (:meth:`PulseCaptureTask.enqueue_packet`),
2. assembles those packets into per-channel blocks and feeds
   :meth:`PulseCaptureSession.feed_block` from its own thread
   (h5py writes must stay on one thread), and
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
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSignal

from ...algorithms.measurement.pulse_capture_session import (
    CaptureState,
    PulseCaptureSession,
)


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
    finished = pyqtSignal()


class PulseCaptureTask(QtCore.QThread):
    """Drains tap samples into a PulseCaptureSession in a worker thread.

    Parameters
    ----------
    session : PulseCaptureSession
        A fully configured (but not started) session.  The task installs
        its own signal-emitting callbacks on it.
    signals : PulseCaptureSignals
        Signal bundle constructed by the caller.
    queue_size : int
        Bounded queue capacity, counted in PACKETS -- one entry holds a
        whole packet's worth of channels, so this is not the sample
        count it used to be.  8k is ~0.2 s of backlog at decimation
        stage 0; overflow drops packets and counts the samples lost.
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
        self.dropped_overflow = 0

        self._cache_size = waveform_cache
        self._cache: "OrderedDict[Tuple[int, int], dict]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._stop_requested = False

        self._pair_cache: "OrderedDict[Tuple[int, int], dict]" = OrderedDict()

        # Route session callbacks through Qt signals (called in run() thread)
        if mode == "both":
            session.on_noise = lambda stream, ns: \
                self.signals.noise_estimated.emit(
                    {"stream": stream, "stats": dict(ns)})
            session.on_pulse = self._on_stream_pulse
            session.on_pair = self._on_pair
            session.on_stats = lambda s: self.signals.stats_updated.emit(s)
            session.on_histograms = lambda stream, d: \
                self.signals.histograms_updated.emit(
                    {"stream": stream, "data": d})
            session.on_templates = lambda stream, d: \
                self.signals.templates_updated.emit(
                    {"stream": stream, "data": d})
            session.on_error = lambda msg: self.signals.error.emit(msg)
            for inner in (session.slow, session.fast):
                inner.on_progress = lambda p, s=inner.streamer_mode: \
                    self.signals.noise_progress.emit({**p, "stream": s})
        else:
            session.on_noise = lambda ns: \
                self.signals.noise_estimated.emit(dict(ns))
            session.on_progress = lambda p: \
                self.signals.noise_progress.emit(p)
            session.on_pulse = self._on_pulse
            session.on_stats = lambda s: self.signals.stats_updated.emit(s)
            session.on_histograms = lambda d: \
                self.signals.histograms_updated.emit(d)
            session.on_templates = lambda d: \
                self.signals.templates_updated.emit(d)
            session.on_error = lambda msg: self.signals.error.emit(msg)

    # ── GUI-thread API ────────────────────────────────────────────

    #: Slow packets carry one sample per channel, so a block has to be
    #: built ACROSS packets rather than within one (the PFB path blocks
    #: within a packet, which is why it needs neither of these).  256
    #: packets is 6.7 ms at decimation stage 0 but 430 ms at stage 6, so
    #: a wall-clock cap bounds the added latency at low rates.
    _BLOCK_PACKETS = 256
    _BLOCK_MAX_S = 0.05

    def enqueue_packet(self, channels, values, timestamp) -> None:
        """Tap callback — called from the GUI thread once per packet.

        ``values`` holds one complex sample per entry of ``channels``.
        """
        try:
            self.sample_queue.put_nowait((channels, values, timestamp))
        except queue.Full:
            self.dropped_overflow += len(channels)

    def request_stop(self) -> None:
        """Ask the worker to finish; session.stop() runs in the worker."""
        self._stop_requested = True
        self.requestInterruption()

    def request_noise_reestimate(self) -> None:
        """Queue a noise re-estimation (executed in the worker thread)."""
        try:
            self.sample_queue.put_nowait(("__reestimate__",))
        except queue.Full:
            self.signals.error.emit("Sample queue full — could not "
                                    "request noise re-estimation")

    def get_pulse(self, channel: int, pulse_idx: int,
                  stream: Optional[str] = None) -> Optional[dict]:
        """Waveform dict for a recent pulse, or None if evicted."""
        with self._cache_lock:
            return self._cache.get((stream, channel, pulse_idx))

    def get_pair(self, channel: int, pair_idx: int) -> Optional[dict]:
        """Full pair dict (summaries + any cross-stream TODs)."""
        with self._cache_lock:
            return self._pair_cache.get((channel, pair_idx))

    def request_waveform(self, channel: int, pulse_idx: int,
                         stream: Optional[str] = None) -> None:
        """Ask the worker to load an evicted waveform from the live HDF5
        file (writer's own handle, writer's thread).  ``waveform_ready``
        fires when the cache has been warmed (or the fetch failed)."""
        try:
            self.sample_queue.put_nowait(
                ("__fetch__", channel, pulse_idx, stream))
        except queue.Full:
            self.signals.error.emit("Sample queue full — could not "
                                    "request waveform load")

    # ── Worker thread ─────────────────────────────────────────────

    def run(self) -> None:
        try:
            self.session.start()
            if self.mode == "fast":
                asyncio.run(self._run_fast())
            elif self.mode == "both":
                asyncio.run(self._run_both())
            else:
                self._run_slow_loop()
        except Exception as e:
            self.signals.error.emit(f"Pulse capture worker failed: {e}")
        finally:
            try:
                self.session.stop()
            except Exception as e:
                self.signals.error.emit(f"Session stop failed: {e}")
            self.signals.finished.emit()

    class _BlockBuilder:
        """Accumulates tap packets into per-channel blocks."""

        def __init__(self, task, feed):
            self._task = task
            self._feed = feed
            self.channels = None
            self._vals = []
            self._stamps = []
            self._t_first = 0.0

        def add(self, channels, values, timestamp) -> None:
            if channels != self.channels:
                self.flush()
                self.channels = channels
            if not self._vals:
                self._t_first = time.monotonic()
            self._vals.append(values)
            # A packet with no usable timestamp becomes NaN; feed_block
            # drops and counts those exactly as feed_sample did.
            self._stamps.append(np.nan if timestamp is None
                                else float(timestamp))

        @property
        def ready(self) -> bool:
            return bool(self._vals) and (
                len(self._vals) >= self._task._BLOCK_PACKETS
                or time.monotonic() - self._t_first
                >= self._task._BLOCK_MAX_S)

        def flush(self) -> None:
            if not self._vals:
                return
            values = np.stack(self._vals)          # (packets, channels)
            stamps = np.asarray(self._stamps, dtype=np.float64)
            self._vals = []
            self._stamps = []
            for column, channel in enumerate(self.channels):
                samples = values[:, column]
                self._feed(channel, samples.real, samples.imag, stamps)

    def _run_slow_loop(self) -> None:
        """Drain the tap-fed queue (slow mode) into per-channel blocks."""
        blocks = self._BlockBuilder(self, self.session.feed_block)
        while not (self._stop_requested or self.isInterruptionRequested()):
            try:
                item = self.sample_queue.get(timeout=0.02)
            except queue.Empty:
                blocks.flush()   # don't strand a partial block when idle
                continue
            if self._handle_control(item):
                continue
            blocks.add(*item)
            if blocks.ready:
                blocks.flush()
        blocks.flush()

    async def _run_fast(self) -> None:
        """PFB capture: configure the fast streamer, run the shared
        source, keep servicing control requests, always tear down."""
        from ...algorithms.measurement.pulse_sources import run_pfb_source

        channels = list(self.session.channels)
        await self.crs.set_pfb_streamer(channel=channels,
                                        module=self.module)
        await asyncio.sleep(0.3)  # settle (trigger_capture precedent)
        try:
            stop = (lambda: self._stop_requested
                    or self.isInterruptionRequested())
            pump = asyncio.ensure_future(self._control_pump())
            try:
                await run_pfb_source(self.session, self.host, channels,
                                     should_stop=stop)
            finally:
                pump.cancel()
                try:
                    await pump
                except asyncio.CancelledError:
                    pass
        finally:
            try:
                await self.crs.set_pfb_streamer(channel=None,
                                                module=self.module)
            except Exception as e:
                self.signals.error.emit(f"PFB teardown failed: {e}")

    async def _run_both(self) -> None:
        """Concurrent slow+fast capture (DualPulseCaptureSession).

        The gather, the shared stop and the fast socket all live in
        run_dual_source — this adds only the PFB streamer lifecycle and
        the tap-fed slow side.  The SLOW side must come from the
        Periscope tap, not a second socket: the mock streamer sends
        UNICAST, and with SO_REUSEPORT the kernel hands each datagram
        to ONE socket — Periscope's own receiver wins and a second
        listener silently starves.  (Real hardware multicasts, but the
        tap works for both and costs nothing.)
        """
        from ...algorithms.measurement.pulse_sources import (
            run_dual_source,
        )

        channels = list(self.session.channels)
        await self.crs.set_pfb_streamer(channel=channels,
                                        module=self.module)
        await asyncio.sleep(0.3)
        try:
            stop = (lambda: self._stop_requested
                    or self.isInterruptionRequested())
            await run_dual_source(
                self.session, self.host, channels, module=self.module,
                should_stop=stop, slow_source=self._slow_tap_pump)
        finally:
            try:
                await self.crs.set_pfb_streamer(channel=None,
                                                module=self.module)
            except Exception as e:
                self.signals.error.emit(f"PFB teardown failed: {e}")

    async def _slow_tap_pump(self, stop) -> float:
        """Drain tap-fed slow samples + control items into the dual
        session.  Warns once if no slow samples arrive at all.

        Returns 0.0 for run_dual_source's slow_elapsed: the tap hands
        over samples without the packet timestamps the socket sources
        accumulate sample time from, and the GUI reads its elapsed time
        off the session stats instead.
        """
        fed = 0
        warned = False
        t_start = time.monotonic()
        blocks = self._BlockBuilder(self, self.session.feed_slow_block)
        while not stop():
            try:
                item = self.sample_queue.get_nowait()
            except queue.Empty:
                blocks.flush()
                if (not warned and fed == 0
                        and time.monotonic() - t_start > 5.0):
                    warned = True
                    self.signals.error.emit(
                        "No slow samples arriving via the tap after "
                        "5 s — is the slow stream running?")
                await asyncio.sleep(0.02)
                continue
            if self._handle_control(item):
                continue
            blocks.add(*item)
            fed += 1
            if blocks.ready:
                blocks.flush()
        blocks.flush()
        return 0.0

    async def _control_pump(self) -> None:
        """Service __reestimate__/__fetch__ requests while a socket
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
            if self.session.state is CaptureState.CAPTURING:
                self.session.re_estimate_noise()
            return True
        if item[0] == "__fetch__":
            _, ch, idx, stream = (item if len(item) == 4
                                  else (*item, None))
            wf = None
            writer = self.session.writer
            if writer is not None:
                try:
                    wf = (writer.read_pulse(stream, ch, idx)
                          if self.mode == "both"
                          else writer.read_pulse(ch, idx))
                except Exception as e:
                    self.signals.error.emit(
                        f"Waveform read failed for ch{ch}#{idx}: {e}")
            if wf is not None:
                with self._cache_lock:
                    self._cache[(stream, ch, idx)] = wf
            self.signals.waveform_ready.emit(ch, idx)
            return True
        return False

    # ── Session callback (worker thread) ──────────────────────────

    def _on_pulse(self, channel: int, pulse_idx: int,
                  summary: Dict[str, Any], pulse_data: dict) -> None:
        with self._cache_lock:
            self._cache[(None, channel, pulse_idx)] = pulse_data
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        self.signals.pulse_detected.emit(channel, pulse_idx, dict(summary))

    def _on_stream_pulse(self, stream: str, channel: int, pulse_idx: int,
                         summary: Dict[str, Any],
                         pulse_data: dict) -> None:
        with self._cache_lock:
            self._cache[(stream, channel, pulse_idx)] = pulse_data
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        self.signals.pulse_detected.emit(
            channel, pulse_idx, {**summary, "stream": stream})

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
