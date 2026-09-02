"""
Background worker bridging the Periscope slow-stream tap to a
:class:`~rfmux.pulse_capture.capture_session.PulseCaptureSession`.

Architecture: all capture logic (noise estimation, detection, HDF5,
histograms) lives in :mod:`rfmux.pulse_capture`, which knows nothing
about Qt.  This QThread only

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

from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSignal

from ...pulse_capture.capture_session import (
    CaptureState,
    PulseCaptureSession,
)
from ...pulse_capture.sources import SlowIngest


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

        # Tap batch, filled on the GUI thread (see enqueue_packet).
        self._tap_channels = None
        self._tap_values: list = []
        self._tap_stamps: list = []
        self._tap_day = None
        self._tap_opened = 0.0

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

    #: Packets gathered before one hand-off to the worker.  A queue
    #: put and a matching get per PACKET is 38k of each per second at
    #: decimation stage 0, and the two threads then spend their time
    #: taking the lock from each other rather than working: measured on
    #: a board with 128 channels captured, 302,908 of 481,852 packets
    #: were lost to this queue filling.  Batched, none were.
    _TAP_BATCH_PACKETS = 256
    #: ...but 256 packets is 430 ms at stage 6, so cap the wait too.
    _TAP_BATCH_MAX_S = 0.05

    def enqueue_packet(self, channels, values, timestamp,
                       day_epoch=None) -> None:
        """Tap callback — called from the GUI thread once per packet.

        ``values`` holds one complex sample per entry of ``channels``.
        Packets are gathered here and handed over in batches.
        """
        if day_epoch is not None and day_epoch != self._tap_day:
            self._tap_day = day_epoch
            self.flush_tap()
            try:
                self.sample_queue.put_nowait(("__day__", day_epoch))
            except queue.Full:
                pass
        if channels != self._tap_channels:
            self.flush_tap()
            self._tap_channels = channels
        if not self._tap_values:
            self._tap_opened = time.monotonic()
        self._tap_values.append(values)
        self._tap_stamps.append(timestamp)
        if (len(self._tap_values) >= self._TAP_BATCH_PACKETS
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
        try:
            self.sample_queue.put_nowait(item)
        except queue.Full:
            self.dropped_overflow += len(item[1]) * len(item[0])

    def request_stop(self) -> None:
        """Ask the worker to finish; session.stop() runs in the worker."""
        # Whatever is still gathered belongs to the capture; the worker
        # is about to stop reading, so hand it over first.
        self.flush_tap()
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
            self.signals.failed.emit(f"Pulse capture worker failed: {e}")
        finally:
            try:
                self.session.stop()
            except Exception as e:
                self.signals.error.emit(f"Session stop failed: {e}")
            self.signals.finished.emit()

    def _run_slow_loop(self) -> None:
        """Drain the tap-fed queue (slow mode) into per-channel blocks.

        Everything past "here is a packet" is SlowIngest, shared with
        the headless socket source, so the GUI is a caller rather than
        a second implementation.  Only the transport differs: this
        process already holds the packets.
        """
        ingest = SlowIngest(self.session.feed_block)
        while not (self._stop_requested or self.isInterruptionRequested()):
            try:
                item = self.sample_queue.get(timeout=0.02)
            except queue.Empty:
                ingest.flush()   # don't strand a partial block when idle
                self.session.flush_progress()
                continue
            if self._handle_control(item):
                continue
            channels, values, stamps = item
            for packet, stamp in zip(values, stamps):
                ingest.add(channels, packet, stamp)
        ingest.flush()

    async def _pfb_mismatch(self, channels) -> Optional[str]:
        """Why the PFB streamer as configured cannot feed this capture,
        or None.

        The capture never configures the streamer; it reads what the
        board is streaming, and every captured channel must be among
        the streamed ones.
        """
        raw = await self.crs.get_pfb_streamer(module=self.module)
        active = raw
        if isinstance(active, dict):
            active = active.get("channel", active.get("channels"))
        if isinstance(active, (int, float)):     # one channel, as the board reports it
            active = [active]
        try:
            active = [int(c) for c in (active or [])]
        except TypeError:
            return (f"get_pfb_streamer(module={self.module}) returned "
                    f"{raw!r}, which this capture cannot read as a "
                    "channel list.")
        if set(channels) <= set(active):
            return None
        have = (f"streaming channels {active}" if active else "off")
        return (f"The PFB streamer on module {self.module} is {have}; "
                f"this capture needs channels {list(channels)}.  Set it "
                "under Streamer Configuration, then start again.")

    async def _run_fast(self) -> None:
        """PFB capture: check the fast streamer carries these channels,
        run the shared source, keep servicing control requests."""
        from ...pulse_capture.sources import run_pfb_source

        channels = list(self.session.channels)
        problem = await self._pfb_mismatch(channels)
        if problem:
            self.signals.failed.emit(problem)
            return
        stop = (lambda: self._stop_requested
                or self.isInterruptionRequested())
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

        channels = list(self.session.channels)
        problem = await self._pfb_mismatch(channels)
        if problem:
            self.signals.failed.emit(problem)
            return
        stop = (lambda: self._stop_requested
                or self.isInterruptionRequested())
        watchdog = asyncio.ensure_future(self._dual_watchdog(stop))
        try:
            await run_dual_source(
                self.session, self.host, channels, module=self.module,
                should_stop=stop, slow_source=self._slow_tap_pump)
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

    async def _slow_tap_pump(self, stop) -> float:
        """Drain tap-fed slow samples + control items into the dual
        session.  Warns once if no slow samples arrive at all.

        Returns the sample time SlowIngest accumulated, on the same
        clock the socket source uses -- so run_dual_source's
        ``slow_elapsed`` means one thing whichever route filled it.
        """
        fed = 0
        warned = False
        t_start = time.monotonic()
        ingest = SlowIngest(self.session.feed_slow_block)
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
            if self._handle_control(item):
                continue
            channels, values, stamps = item
            for packet, stamp in zip(values, stamps):
                ingest.add(channels, packet, stamp)
            fed += len(values)
        ingest.flush()
        return ingest.elapsed

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
        if item[0] == "__day__":
            self.session.set_time_origin(item[1])
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
