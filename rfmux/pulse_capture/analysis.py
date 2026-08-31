"""
Per-pulse derived quantities for pulse capture.

Single source of truth for the scalar metrics attached to every detected
pulse — peak amplitudes, the canonical signal-to-noise definition, window
duration, and the fit-free decay-constant estimate.  The HDF5 writer,
histogram accumulators, and any GUI display should all derive their
numbers from :func:`pulse_summary` so they can never disagree.

The canonical SNR is::

    snr = max(peak_I, peak_Q) / max(std_I, std_Q)

matching :class:`~rfmux.pulse_capture.accumulators.PulseHistogramSet`.

The decay constant is recovered without curve fitting from two
well-measured points on the falling edge — the peak and the moment the
envelope falls back through the trigger threshold::

    tau = (t_thr - t_peak) / ln(peak_snr / threshold_sigma)

Taking the amplitude ratio cancels the unknown event energy, so for a
detector with a fixed decay time every energy line collapses onto a
single tau value.  The discrete threshold-crossing sample lands slightly
below the true crossing, so this estimator runs a few percent low — it
is a live cross-check, not a precision fit.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional

from .detection import ChannelNoiseStats
from ..core.transferfunctions import VOLTS_PER_ROC


def pulse_peaks(
    pulse_data: dict,
    noise_stats: Optional[ChannelNoiseStats],
) -> Dict[str, float]:
    """Peak excursions and the canonical SNR for one pulse.

    Parameters
    ----------
    pulse_data : dict
        Pulse dict with ``Amp_I`` and ``Amp_Q`` arrays.
    noise_stats : ChannelNoiseStats or None
        Baseline statistics.  When None, peaks are absolute values and
        ``snr`` is 0.0.

    Returns
    -------
    dict
        ``peak_I``, ``peak_Q`` (excursion from baseline), ``peak_amp``
        (max of the two), ``snr`` (peak_amp / max(std_I, std_Q)).
    """
    amp_I = np.asarray(pulse_data["Amp_I"], dtype=np.float64)
    amp_Q = np.asarray(pulse_data["Amp_Q"], dtype=np.float64)

    if noise_stats is None:
        peak_I = float(np.max(np.abs(amp_I))) if len(amp_I) else 0.0
        peak_Q = float(np.max(np.abs(amp_Q))) if len(amp_Q) else 0.0
        return {
            "peak_I": peak_I,
            "peak_Q": peak_Q,
            "peak_amp": max(peak_I, peak_Q),
            "snr": 0.0,
        }

    peak_I = float(np.max(np.abs(amp_I - noise_stats.mean_I))) if len(amp_I) else 0.0
    peak_Q = float(np.max(np.abs(amp_Q - noise_stats.mean_Q))) if len(amp_Q) else 0.0
    peak_amp = max(peak_I, peak_Q)
    snr = peak_amp / max(noise_stats.std_I, noise_stats.std_Q, 1e-30)
    return {"peak_I": peak_I, "peak_Q": peak_Q, "peak_amp": peak_amp, "snr": snr}


def derive_tau(
    pulse_data: dict,
    noise_stats: Optional[ChannelNoiseStats],
    threshold_sigma: float,
    min_snr_margin: float = 1.3,
) -> float:
    """Fit-free decay-constant estimate from one pulse.

    Uses the dominant quadrature (larger peak SNR).  On the falling edge
    the envelope decays as ``A(t) = peak * exp(-(t - t_peak) / tau)``;
    the ratio between the peak and the trigger-threshold crossing gives::

        tau = (t_thr - t_peak) / ln(peak_snr / threshold_sigma)

    Parameters
    ----------
    pulse_data : dict
        Pulse dict with ``Amp_I``, ``Amp_Q``, ``Time`` arrays.
    noise_stats : ChannelNoiseStats or None
        Baseline statistics.  Required — returns NaN when None.
    threshold_sigma : float
        The trigger threshold used during capture, in sigma units.
    min_snr_margin : float
        The peak must exceed ``min_snr_margin * threshold_sigma`` for
        the log ratio to be meaningful.  Default 1.3.

    Returns
    -------
    float
        Decay constant in seconds, or NaN when it cannot be derived
        (weak pulse, no threshold crossing in the window, bad
        timestamps, or fewer than 5 valid samples).
    """
    if noise_stats is None or threshold_sigma <= 0:
        return float("nan")

    amp_I = np.asarray(pulse_data["Amp_I"], dtype=np.float64)
    amp_Q = np.asarray(pulse_data["Amp_Q"], dtype=np.float64)
    times = np.asarray(pulse_data["Time"], dtype=np.float64)
    if not (len(amp_I) == len(amp_Q) == len(times)):
        return float("nan")

    valid = np.isfinite(times)
    if np.count_nonzero(valid) < 5:
        return float("nan")
    amp_I, amp_Q, times = amp_I[valid], amp_Q[valid], times[valid]

    # Dominant quadrature: larger peak in its own sigma units.
    std_I = max(noise_stats.std_I, 1e-30)
    std_Q = max(noise_stats.std_Q, 1e-30)
    env_I = np.abs(amp_I - noise_stats.mean_I)
    env_Q = np.abs(amp_Q - noise_stats.mean_Q)
    if float(np.max(env_I)) / std_I >= float(np.max(env_Q)) / std_Q:
        env, sigma = env_I, std_I
    else:
        env, sigma = env_Q, std_Q

    pk = int(np.argmax(env))
    peak_snr = float(env[pk]) / sigma
    if peak_snr <= min_snr_margin * threshold_sigma:
        return float("nan")

    # First sample after the peak that falls back through the threshold.
    below = np.nonzero(env[pk:] < threshold_sigma * sigma)[0]
    if len(below) == 0:
        return float("nan")
    decay_t = float(times[pk + int(below[0])] - times[pk])

    denom = float(np.log(peak_snr / threshold_sigma))
    if denom <= 0 or decay_t <= 0:
        return float("nan")
    return decay_t / denom


def pulse_summary(
    pulse_data: dict,
    noise_stats: Optional[ChannelNoiseStats],
    threshold_sigma: Optional[float] = None,
) -> Dict[str, float]:
    """Complete scalar metadata for one pulse.

    The one dict every consumer (HDF5 attrs, tree rows, histograms,
    info boxes) should build from.

    Returns
    -------
    dict
        ``n_samples``, ``pileup``, ``truncated``, ``peak_I``, ``peak_Q``,
        ``peak_amp``, ``snr``, ``duration_s``, ``duration_ms``,
        ``timestamp`` (first valid time), ``tau_s``, ``tau_ms`` (NaN when
        not derivable).

        ``duration`` is the time the pulse spent above threshold
        (trigger → below-threshold), not the length of the saved
        window, which also carries the pre-trigger margin and a tail
        whose length depends on the save policy.  It falls back to the
        window span only for pileup splits and hard stops, which have
        no below-threshold instant to measure to.
    """
    peaks = pulse_peaks(pulse_data, noise_stats)

    times = np.asarray(pulse_data["Time"], dtype=np.float64)
    valid_times = times[np.isfinite(times)]
    timestamp = float(np.min(valid_times)) if len(valid_times) else 0.0

    # Duration is trigger → below-threshold, NOT the length of the saved
    # window.  The window also holds the pre-trigger margin and whatever
    # tail the save policy kept, and under save_to_end_confirmed that
    # tail runs until the end condition is confirmed: a baseline
    # property.  Measured on the mock at 19 kHz with tau=1 ms, identical
    # injected pulses gave windows spanning 3.2-17.8 ms while the
    # threshold crossings stayed inside 3.0-4.0 ms.  Deriving duration
    # from the window would put that 5.6x spread into every histogram.
    trigger_time = pulse_data.get("trigger_time")
    below_time = pulse_data.get("below_threshold_time")
    if trigger_time is not None and below_time is not None:
        duration_s = float(below_time) - float(trigger_time)
    elif len(valid_times) > 1:
        # Pileup splits and hard stops have no below-threshold instant:
        # the pulse never demonstrably ended, so the window is the only
        # evidence of how long it lasted.
        duration_s = float(np.max(valid_times) - np.min(valid_times))
    else:
        duration_s = 0.0

    if threshold_sigma is not None:
        tau_s = derive_tau(pulse_data, noise_stats, threshold_sigma)
    else:
        tau_s = float("nan")

    return {
        "n_samples": int(len(np.asarray(pulse_data["Amp_I"]))),
        "pileup": bool(pulse_data.get("pileup", False)),
        "truncated": bool(pulse_data.get("truncated", False)),
        **peaks,
        "duration_s": duration_s,
        "duration_ms": duration_s * 1e3,
        "timestamp": timestamp,
        "tau_s": tau_s,
        "tau_ms": tau_s * 1e3,
    }


def counts_to_hz_scale(df_calibration: Optional[float]) -> Optional[float]:
    """Multiplier taking pulse amplitudes from ADC counts to Δf in Hz.

    Pulse waveforms and every amplitude-like metric derived from them
    are stored in raw counts, which are only comparable across channels
    once calibrated.  ``df_calibration`` (Hz per radian, from
    :func:`~rfmux.algorithms.measurement.bias_kids.bias_kids` and saved
    into the capture file) combines with the readout's counts-to-volts
    constant into a single linear factor::

        amplitude_hz = amplitude_counts * counts_to_hz_scale(df_cal)

    Returns None when the channel is uncalibrated, which callers should
    treat as "display raw counts" rather than substituting 1.0 —
    unscaled counts mislabelled as Hz are worse than no calibration.

    ``bias_kids`` returns a *complex* calibration: its magnitude is the
    Hz-per-count scale, and its phase is the rotation from the (I, Q)
    axes to (frequency, dissipation).  A scalar amplitude only wants the
    magnitude, so take it here rather than float(), which raised
    TypeError on every genuinely calibrated channel.
    """
    if df_calibration is None:
        return None
    return abs(complex(df_calibration)) * VOLTS_PER_ROC
