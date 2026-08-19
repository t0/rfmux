"""
Pulse capture: detect, record and summarise transient events in a stream.

This is infrastructure, not a measurement.  Nothing in here talks to a
board, registers a ``@macro`` or imports :class:`~rfmux.core.schema.CRS`
— it turns a stream of I/Q samples into detected pulses, running
statistics and an HDF5 file.  The operations that *use* it live in
``rfmux.algorithms.measurement`` (``crs.trigger_capture(...)``), and
Periscope's pulse-capture panel is a third caller.  That is why the
package sits beside :mod:`rfmux.streamer` and :mod:`rfmux.mock` rather
than under ``algorithms``.

The layers, bottom up::

    detection      the engine: ring buffers, noise estimation, the
                   trigger state machine (threshold, end confirmation,
                   pileup, the edge test)
    analysis       pure functions over one detected pulse: peaks, tau,
                   the scalar summary, unit scaling
    accumulators   running products over many pulses: histograms of the
                   summary scalars, trigger-aligned waveform templates
    hdf5           streaming writer (write-as-you-go, so a crash keeps
                   what it had) and the matching reader
    sources        getting samples in: async socket loops for the slow
                   and fast streams, and SlowBlockAccumulator, which
                   Periscope's GUI tap uses too so the two cannot drift
    session        the orchestrator: lifecycle, callbacks, persistence.
                   DualPulseCaptureSession runs slow+fast at once with
                   cross-stream pulse matching

Typical headless use::

    from rfmux.pulse_capture import PulseCaptureSession, run_slow_source

    session = PulseCaptureSession(channels=[1, 2], sample_rate=fs,
                                  hdf5_path="capture.h5")
    session.start()
    await run_slow_source(session, host, module=1, duration_s=10.0)
    session.stop()

See ``rfmux/reference-notebooks/Demos/pulse_capture.md`` for the full
worked example; it is executed against a MockCRS in CI, so it is a test
as well as documentation.
"""

from .detection import (
    BUFFER_SAFETY,
    HARD_STOP_RING_FRACTION,
    ChannelNoiseStats,
    Circular,
    PulseCapture,
    estimate_noise_stats,
)
from .analysis import (
    counts_to_hz_scale,
    derive_tau,
    pulse_peaks,
    pulse_summary,
)
from .accumulators import (
    HistogramAccumulator,
    PulseHistogramSet,
    PulseTemplateAccumulator,
    PulseTemplateSet,
    find_trigger_index,
)
from .session import (
    DETECTION_PARAMS,
    CaptureState,
    DualPulseCaptureSession,
    IncrementalPulseMatcher,
    PulseCaptureConfig,
    PulseCaptureSession,
)
from .sources import (
    SlowBlockAccumulator,
    columns_for_width,
    run_dual_source,
    run_pfb_source,
    run_slow_source,
)

# h5py is optional: importing the package must not require it.
try:
    from .hdf5 import DualPulseHDF5Writer, PulseHDF5Reader, PulseHDF5Writer
except ImportError:  # pragma: no cover - h5py missing
    DualPulseHDF5Writer = None  # type: ignore[assignment]
    PulseHDF5Reader = None  # type: ignore[assignment]
    PulseHDF5Writer = None  # type: ignore[assignment]

__all__ = [
    # detection
    "BUFFER_SAFETY",
    "HARD_STOP_RING_FRACTION",
    "ChannelNoiseStats",
    "Circular",
    "PulseCapture",
    "estimate_noise_stats",
    # analysis
    "counts_to_hz_scale",
    "derive_tau",
    "pulse_peaks",
    "pulse_summary",
    # accumulators
    "HistogramAccumulator",
    "PulseHistogramSet",
    "PulseTemplateAccumulator",
    "PulseTemplateSet",
    "find_trigger_index",
    # session
    "DETECTION_PARAMS",
    "CaptureState",
    "DualPulseCaptureSession",
    "IncrementalPulseMatcher",
    "PulseCaptureConfig",
    "PulseCaptureSession",
    # sources
    "SlowBlockAccumulator",
    "columns_for_width",
    "run_dual_source",
    "run_pfb_source",
    "run_slow_source",
    # hdf5 (None when h5py is unavailable)
    "DualPulseHDF5Writer",
    "PulseHDF5Reader",
    "PulseHDF5Writer",
]
