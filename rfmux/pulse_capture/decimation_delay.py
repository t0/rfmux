"""Group delay of the decimated readout stream relative to the PFB stream.

TEMPORARY FIRMWARE COMPENSATION — remove when the RTL timestamps the
decimated stream at its filter centroid.

The PFB (fast) stream's timestamps are taken as correct.  The decimated
(slow) stream reaches the network through two CIC filters that the PFB
stream does not pass, and its packet timestamps are not corrected for
their group delay, so a feature the PFB stream shows at time T appears
in the slow stream at T + delay — "late" by the same amount at every
decimation stage, in output samples:

    CIC1: N=3 stages, R=64, at the PFB rate      -> 94.5 input samples
    CIC2: N=6 stages, R=2**dec, at CIC1's rate  -> 3(R-1) input samples

An N-stage boxcar CIC of rate change R delays by N(R-1)/2 of its input
samples.  Output-referred, CIC2 contributes 3(R-1)/R samples of the slow
stream: 2.81 at dec 4, 2.91 at dec 5, 2.95 at dec 6 — about 2.9 with a
slight dependence on the stage.  CIC1's 38.7 µs is 1.5 samples at dec 0
and negligible against the sample interval above that; it is included
because the formula is exact and costs nothing.

Where it is applied: :class:`DualPulseCaptureSession` shifts every slow
timestamp by ``-decimated_stream_delay_s(slow_rate)`` before the sample
reaches the engine, the matcher or the file, so in "both" mode the two
streams share one time axis.  Single-stream captures are left as the
board stamps them.  The shift is recorded in the file's metadata as
``slow_time_offset_s`` (0 when not applied), which is how a reader tells
a corrected capture from an uncorrected one.

MockCRS streams carry no CIC delay, so in mock "both" mode the shift is
spurious and the residual stream skew reads about minus the delay
(≈ -5 ms at decimation stage 6).  The 50 ms match window absorbs it.

To remove once the firmware corrects the timestamps:
  1. delete this module;
  2. in ``DualPulseCaptureSession.__init__`` make ``slow_time_offset_s``
     default to 0.0 instead of calling :func:`decimated_stream_delay_s`
     (or drop the parameter and the shift in ``feed_slow`` /
     ``_feed_block`` altogether);
  3. keep the ``slow_time_offset_s`` metadata attribute: files written
     before the fix carry a nonzero value, and it is what says so.
The CIC parameters are the same ones ``core.transferfunctions`` uses for
the spectral CIC correction (see ``_apply_cic_correction`` there); they
describe the firmware, not this module.
"""
from __future__ import annotations

from ..core.transferfunctions import PFB_SAMPLING_FREQ

#: CIC1: PFB rate in, /64 out, three stages.
CIC1_STAGES = 3
CIC1_RATE_CHANGE = 64
#: CIC2: CIC1 rate in, /2**dec out, six stages.
CIC2_STAGES = 6
CIC1_OUTPUT_RATE = PFB_SAMPLING_FREQ / CIC1_RATE_CHANGE


def cic_group_delay_input_samples(stages: int, rate_change: int) -> float:
    """N(R-1)/2: the delay of an N-stage boxcar CIC in its input samples."""
    return stages * (rate_change - 1) / 2.0


def decimated_stream_delay_s(slow_rate: float) -> float:
    """Seconds the decimated stream lags the PFB stream at *slow_rate*.

    The CIC2 rate change is recovered from the sample rate
    (``CIC1_OUTPUT_RATE / slow_rate``, 1 at decimation stage 0), so any
    caller that knows the slow sample rate can apply this without also
    knowing the stage.
    """
    if not slow_rate or slow_rate <= 0:
        return 0.0
    r2 = max(1, int(round(CIC1_OUTPUT_RATE / slow_rate)))
    cic1 = cic_group_delay_input_samples(CIC1_STAGES, CIC1_RATE_CHANGE) \
        / PFB_SAMPLING_FREQ
    cic2 = cic_group_delay_input_samples(CIC2_STAGES, r2) / CIC1_OUTPUT_RATE
    return cic1 + cic2
