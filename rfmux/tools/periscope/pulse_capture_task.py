"""
Background worker bridging the Periscope slow-stream tap to a
:class:`~rfmux.algorithms.measurement.pulse_capture_session.PulseCaptureSession`.

Architecture: all capture logic (noise estimation, detection, HDF5,
histograms) lives in the algorithms layer.  This QThread only

1. drains a thread-safe queue that the GUI-thread tap fills
   (:meth:`PulseCaptureTask.enqueue`),
2. feeds :meth:`PulseCaptureSession.feed_sample` from its own thread
   (h5py writes must stay on one thread), and
3. re-emits the session callbacks as Qt signals.

Waveforms never cross threads through signals — ``pulse_detected``
carries only the scalar summary.  The task keeps a bounded in-memory
cache of recent waveforms (:meth:`get_pulse`) so the live Pulse View
can display events without opening the HDF5 file the writer holds.
"""

from __future__ import annotations

import queue
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSignal

from ...algorithms.measurement.pulse_capture_session import (
    CaptureState,
    PulseCaptureSession,
)


class PulseCaptureSignals(QtCore.QObject):
    """Signal bundle for :class:`PulseCaptureTask` (constructed by the caller)."""

    noise_estimated = pyqtSignal(dict)      # {channel: ChannelNoiseStats}
    noise_progress = pyqtSignal(dict)       # {collected: {ch: n}, target: N}
    pulse_detected = pyqtSignal(int, int, dict)  # channel, pulse_idx, summary
    stats_updated = pyqtSignal(dict)        # PulseCaptureSession.stats()
    histograms_updated = pyqtSignal(dict)   # PulseHistogramSet.get_histogram_data()
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
        Bounded sample queue capacity.  At the slow rate (~600 Hz per
        channel after periscope decimation) 100k covers minutes of
        backlog; overflow drops samples and counts them.
    waveform_cache : int
        Number of recent pulse waveforms kept for :meth:`get_pulse`.
    """

    def __init__(
        self,
        session: PulseCaptureSession,
        signals: PulseCaptureSignals,
        queue_size: int = 100_000,
        waveform_cache: int = 200,
        parent: Optional[QtCore.QObject] = None,
    ):
        super().__init__(parent)
        self.session = session
        self.signals = signals
        self.sample_queue: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self.dropped_overflow = 0

        self._cache_size = waveform_cache
        self._cache: "OrderedDict[Tuple[int, int], dict]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._stop_requested = False

        # Route session callbacks through Qt signals (called in run() thread)
        session.on_noise = lambda ns: self.signals.noise_estimated.emit(dict(ns))
        session.on_progress = lambda p: self.signals.noise_progress.emit(p)
        session.on_pulse = self._on_pulse
        session.on_stats = lambda s: self.signals.stats_updated.emit(s)
        session.on_histograms = lambda d: self.signals.histograms_updated.emit(d)
        session.on_error = lambda msg: self.signals.error.emit(msg)

    # ── GUI-thread API ────────────────────────────────────────────

    def enqueue(self, channel: int, i_val: float, q_val: float,
                timestamp) -> None:
        """Tap callback — called from the GUI thread for every sample."""
        try:
            self.sample_queue.put_nowait((channel, i_val, q_val, timestamp))
        except queue.Full:
            self.dropped_overflow += 1

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

    def get_pulse(self, channel: int, pulse_idx: int) -> Optional[dict]:
        """Waveform dict for a recent pulse, or None if evicted."""
        with self._cache_lock:
            return self._cache.get((channel, pulse_idx))

    # ── Worker thread ─────────────────────────────────────────────

    def run(self) -> None:
        try:
            self.session.start()
            while not (self._stop_requested or self.isInterruptionRequested()):
                try:
                    item = self.sample_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item[0] == "__reestimate__":
                    if self.session.state is CaptureState.CAPTURING:
                        self.session.re_estimate_noise()
                    continue
                ch, i_val, q_val, ts = item
                self.session.feed_sample(ch, i_val, q_val, ts)
        except Exception as e:
            self.signals.error.emit(f"Pulse capture worker failed: {e}")
        finally:
            try:
                self.session.stop()
            except Exception as e:
                self.signals.error.emit(f"Session stop failed: {e}")
            self.signals.finished.emit()

    # ── Session callback (worker thread) ──────────────────────────

    def _on_pulse(self, channel: int, pulse_idx: int,
                  summary: Dict[str, Any], pulse_data: dict) -> None:
        with self._cache_lock:
            self._cache[(channel, pulse_idx)] = pulse_data
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        self.signals.pulse_detected.emit(channel, pulse_idx, dict(summary))
