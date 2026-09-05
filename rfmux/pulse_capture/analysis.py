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
import warnings
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

    # Three instants.  ``timestamp`` is the first saved sample: the
    # record start, pre-trigger margin included.  ``trigger_time`` is
    # the event, the anchor for anything aligned across streams, since
    # each stream's record start sits its own margin before it.
    # ``saved_end_time`` is the last saved sample -- not pulse_data's
    # ``end_time``, the confirmation instant, which lies past the saved
    # window when the tail is not kept.
    return {
        "n_samples": int(len(np.asarray(pulse_data["Amp_I"]))),
        "pileup": bool(pulse_data.get("pileup", False)),
        "truncated": bool(pulse_data.get("truncated", False)),
        **peaks,
        "duration_s": duration_s,
        "duration_ms": duration_s * 1e3,
        "timestamp": timestamp,
        "start_time": timestamp,
        "trigger_time": (float(trigger_time) if trigger_time is not None
                         and np.isfinite(trigger_time) else timestamp),
        "saved_end_time": (float(np.max(valid_times))
                           if len(valid_times) else timestamp),
        "tau_s": tau_s,
        "tau_ms": tau_s * 1e3,
        # The packet clock decoded to a calendar time, when the session
        # knew the day; absent otherwise.
        **{k: pulse_data[k] for k in ("trigger_epoch", "trigger_utc")
           if k in pulse_data},
    }


def storage_transform(df_calibration, trigger_basis: str = "iq"):
    """How one channel's samples are rotated and scaled for storage.

    Returns ``(factor, units)``: the complex map applied on the way in,
    once, so that everything downstream -- ring buffer, noise
    estimate, trigger, summaries, histograms, templates and the stored
    waveform -- is in the same physical units::

        first + j*second = (I + jQ) * factor

    Samples are never stored in ADC counts.  Counts are an artefact of
    the readout tied to the I/Q axes; a *count* projected onto the
    frequency axis is a linear combination of two ADC readings and means
    nothing on its own.  So a channel is stored in volts, or in hertz
    once it has been rotated into the frequency basis, and the file
    carries both conversions so any other view is recoverable.

    ``trigger_basis="df"`` uses the calibration's phase to rotate, and
    its magnitude as part of the scale.  Without a calibration there is
    nothing to rotate with, so the channel stays in the quadratures and
    in volts.  Folding the scale into the rotation coefficients keeps
    this to the same two multiplies per axis either way, which matters
    at 2.44 MHz.
    """
    cal = _calibration(df_calibration)
    if df_calibration is not None and cal is None:
        warnings.warn(
            "ignoring df_calibration: expected a number, got "
            f"{type(df_calibration).__name__}.  df_calibrations is the "
            "flat {channel: calibration} mapping, not one keyed by module.",
            stacklevel=2)
    if trigger_basis == "df" and cal:
        basis, units = "df", "Hz"
    else:
        basis, units = "iq", "V"
    return _basis_units_factor(cal, basis, units), units


def display_transform(df_calibration, stored_basis: str, stored_units: str,
                      view_basis: str, view_units: str):
    """Coefficients taking *stored* samples to what the user asked for.

    Storage and display are each a rotation and a scale, so each is one
    complex number and going between them is a single division.  No
    round trip through counts, and no second place to get the rotation
    backwards.

    The rotation comes from the basis and the scale from the units, and
    they are independent: counts in the frequency basis are still
    rotated, they are simply not scaled.  Returns ``(factor, units)``
    for :func:`apply_iq_conversion`, or ``None`` when the request
    cannot be honoured -- hertz or a rotation without a calibration --
    so callers fall back to what is stored rather than putting an
    unscaled number under the wrong label.
    """
    stored = _basis_units_factor(df_calibration, stored_basis, stored_units)
    wanted = _basis_units_factor(df_calibration, view_basis, view_units)
    if stored is None or wanted is None or stored == 0:
        return None
    return wanted / stored, view_units


def _calibration(df_calibration) -> Optional[complex]:
    """The calibration as one complex number, or ``None`` when it is
    absent or not a number at all (a mapping keyed by module where the
    flat ``{channel: calibration}`` one was expected).  Zero passes
    through; the callers decide what a zero calibration means."""
    if df_calibration is None:
        return None
    try:
        return complex(df_calibration)
    except TypeError:
        return None


def _basis_units_factor(df_calibration, basis: str, units: str):
    """One complex number for "rotate into *basis*, scale to *units*"."""
    cal = _calibration(df_calibration)
    if basis == "df":
        if cal is None or cal == 0:
            return None                 # nothing to rotate with
        rotation = cal / abs(cal)
    else:
        rotation = complex(1.0, 0.0)

    if units == "counts":
        scale = 1.0
    elif units == "V":
        scale = VOLTS_PER_ROC
    elif units == "Hz":
        if cal is None or cal == 0 or basis != "df":
            return None                 # Hz is a property of the df axis
        scale = VOLTS_PER_ROC * abs(cal)
    else:
        return None
    return rotation * scale


def window_shortfall(times, window, tolerance: float) -> tuple:
    """How much of *window* (t0, t1) the samples at *times* fail to
    cover, as (seconds missing at the start, seconds missing at the
    end); zero where the gap is within *tolerance*.  A fast window
    that falls short of the union window it was asked for was never
    in the ring: on the fast stream, packets lost."""
    t = np.asarray(times, dtype=float)
    t = t[np.isfinite(t)]
    if t.size == 0 or window is None:
        return 0.0, 0.0
    head = float(t[0]) - window[0]
    tail = window[1] - float(t[-1])
    return (head if head > tolerance else 0.0,
            tail if tail > tolerance else 0.0)


# ── Plotting many channels ──────────────────────────────────────────

def plot_groups(spec: str, channels) -> list:
    """What to draw from *channels*, one entry per series.

    Each comma-separated item of *spec* is one series: a channel on
    its own, a range ``a-b`` whose channels are combined into one, or
    ``*`` / ``all`` for every channel combined.  An empty spec is one
    series per channel.  Channels absent from *channels* are dropped,
    and a series that ends up empty is dropped with them.

    Returns ``[(label, [channels...]), ...]``.  Raises ValueError with
    the offending token, since the caller is a GUI field.
    """
    present = sorted(int(c) for c in channels)
    cleaned = "".join(spec.split())
    if not cleaned:
        return [(f"Ch{c}", [c]) for c in present]
    out = []
    for token in cleaned.split(","):
        if not token:
            continue
        if token.lower() in ("*", "all"):
            if present:
                out.append((f"All {len(present)} ch", list(present)))
            continue
        lo, sep, hi = token.partition("-")
        try:
            start = int(lo)
            stop = int(hi) if sep else start
        except ValueError:
            raise ValueError(
                f"Could not read {token!r}. Use channels like \"1,2\", "
                f"ranges like \"2-19\" to combine, or \"*\".") from None
        if stop < start:
            raise ValueError(
                f"Range {token!r} runs backwards -- write "
                f"\"{stop}-{start}\".")
        members = [c for c in present if start <= c <= stop]
        if not members:
            continue
        label = f"Ch{start}" if not sep else f"Ch{start}-{stop}"
        out.append((label, members))
    return out


def rebin_counts(src_edges, counts, dst_edges) -> np.ndarray:
    """Counts redistributed from one bin grid onto another, exactly for
    a density that is uniform within each source bin."""
    src_edges = np.asarray(src_edges, dtype=np.float64)
    cum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(
        np.asarray(counts, dtype=np.float64)))])
    cum_dst = np.interp(np.asarray(dst_edges, dtype=np.float64),
                        src_edges, cum, left=0.0, right=cum[-1])
    return np.diff(cum_dst)


def combine_histograms(edges_list, counts_list):
    """One histogram from several: summed on a shared grid, or rebinned
    onto one spanning them all when the grids differ (channels in Hz
    have their own calibration scale)."""
    edges0 = np.asarray(edges_list[0], dtype=np.float64)
    same = all(np.array_equal(np.asarray(e, dtype=np.float64), edges0)
               for e in edges_list[1:])
    if same:
        total = np.zeros(len(edges0) - 1)
        for c in counts_list:
            total += np.nan_to_num(np.asarray(c, dtype=np.float64))
        return edges0, total
    lo = min(float(np.min(e)) for e in edges_list)
    hi = max(float(np.max(e)) for e in edges_list)
    grid = np.linspace(lo, hi, len(edges0))
    total = np.zeros(len(grid) - 1)
    for e, c in zip(edges_list, counts_list):
        total += rebin_counts(e, c, grid)
    return grid, total


def combine_templates(means, resids, counts):
    """Stack several channels' templates as if their pulses had been
    stacked together: count-weighted mean, and the RMS about it
    (each channel's own spread plus its offset from the joint mean)."""
    m = np.asarray(means, dtype=np.float64)
    n = np.nan_to_num(np.asarray(counts, dtype=np.float64))
    n = np.where(np.isfinite(m), n, 0.0)
    m0 = np.nan_to_num(m)
    total = n.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(total > 0, (n * m0).sum(axis=0) / total, np.nan)
        if resids is None:
            resid = None
        else:
            r0 = np.nan_to_num(np.asarray(resids, dtype=np.float64))
            second = (n * (r0 ** 2 + m0 ** 2)).sum(axis=0) / total
            resid = np.where(total > 0,
                             np.sqrt(np.maximum(second - mean ** 2, 0.0)),
                             np.nan)
    return mean, resid, total


def summary_from_attrs(attrs: dict) -> Dict[str, float]:
    """The :func:`pulse_summary` key set rebuilt from a pulse's stored
    HDF5 attributes, so a reviewed pulse reads like a live one.

    The file keeps seconds (``duration_s``, ``tau_s``); the millisecond
    twins are derived here.  ``snr`` and ``peak_amp`` fall back to the
    per-quadrature attributes for files that lack them.  The clock
    marks (``trigger_utc``, ``trigger_epoch``, ``trigger_time``) pass
    through when present.
    """
    nan = float("nan")
    snr = float(attrs.get("snr") or max(attrs.get("peak_snr_I", 0.0),
                                       attrs.get("peak_snr_Q", 0.0)))
    peak_I = float(attrs.get("peak_I", 0.0))
    peak_Q = float(attrs.get("peak_Q", 0.0))
    duration_s = float(attrs.get("duration_s", 0.0))
    tau_s = float(attrs.get("tau_s", nan))
    return {
        "n_samples": int(attrs.get("n_samples", 0)),
        "pileup": bool(attrs.get("pileup", False)),
        "truncated": bool(attrs.get("truncated", False)),
        "peak_I": peak_I,
        "peak_Q": peak_Q,
        "peak_amp": float(attrs.get("peak_amp") or max(peak_I, peak_Q)),
        "snr": snr,
        "duration_s": duration_s,
        "duration_ms": duration_s * 1e3,
        "timestamp": float(attrs.get("timestamp", 0.0)),
        "tau_s": tau_s,
        "tau_ms": tau_s * 1e3,
        **{k: attrs[k] for k in ("trigger_time", "trigger_epoch",
                                 "trigger_utc") if k in attrs},
    }
