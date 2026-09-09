"""Overlay a fastrx recording (the 100G channel stream) on 1G data by
IRIG time: the pulses of a capture file, or a window of a parser
dirfile.

All three streams are stamped by the board's IRIG clock.  The channel
stream shares the PFB stream's tap, so their stamps agree; the
decimated (slow) stream is stamped late by its CIC group delay, which
the capture session and the parser take out where they write
(``slow_time_offset_s`` in a capture file, ``fir_stage`` beside the
dirfile's corrected timebase).  Files written before that carry raw
stamps, and :func:`slow_shift_s` supplies the shift for them.

Nothing here imports the fastrx extension: a :class:`rfmux.fastrx.PacketFile`
is passed in, so the module loads where the extension does not build.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..core.transferfunctions import (PFB_SAMPLING_FREQ,
                                      decimated_stream_delay_s,
                                      sampling_to_decimation)
from .hdf5 import PulseHDF5Reader


def slow_shift_s(reader: PulseHDF5Reader) -> float:
    """Seconds to add to a file's slow ``Time`` arrays to put them on the
    PFB clock: 0 for a file that records ``slow_time_offset_s`` (its
    times were shifted as they were written), minus the CIC delay of
    its slow rate for an older file, and 0 when the rate is unknown."""
    meta = reader.metadata
    if "slow_time_offset_s" in meta:
        return 0.0
    rate = meta.get("sample_rate_slow")
    if not rate:
        return 0.0
    return -decimated_stream_delay_s(sampling_to_decimation(float(rate)))


def counts_to_stored(reader: PulseHDF5Reader, channel: int,
                     stream: Optional[str] = None) -> complex:
    """Factor taking complex ADC counts to *channel*'s stored samples,
    from the file's own record of how they were stored: its
    ``volts_per_count``, and the df calibration's rotation and
    magnitude for a channel stored in hertz.  1 for a file that holds
    counts."""
    units = reader.stored_units(channel, stream)
    if units == "counts":
        return 1.0 + 0j
    vpc = reader.volts_per_count()
    scale = float(vpc) if vpc is not None else 1.0
    if units == "V":
        return complex(scale)
    cal = reader.df_calibration(channel, stream)
    if units == "Hz" and cal is not None and cal != 0:
        return complex(cal) * scale       # rotation and hertz per volt in one
    return complex(scale)


@dataclass
class Overlay:
    """One pulse and the same channel from a fastrx recording, on one
    time axis (seconds of day, PFB clock) and in the file's stored
    units.  ``fast`` is the paired fast-stream pulse of a dual file;
    ``lag_s`` is where the fastrx trace best matches the fast one,
    fastrx stamp minus fast stamp of the same feature."""
    channel: int
    pulse_idx: int
    stream: str
    units: str
    shift_s: float
    pulse: Dict[str, np.ndarray]
    fastrx: Dict[str, np.ndarray]
    fast: Optional[Dict[str, np.ndarray]] = None
    #: The parser dirfile's slow trace over the same window, in the
    #: same units, when one was given.
    dirfile: Optional[Dict[str, np.ndarray]] = None
    lag_s: Optional[float] = None
    seq_gaps: int = 0
    dropouts: int = 0


def _tod(pulse: Dict[str, Any], shift: float = 0.0) -> Dict[str, np.ndarray]:
    return {"times": np.asarray(pulse["Time"], dtype=np.float64) + shift,
            "I": np.asarray(pulse["Amp_I"], dtype=np.float64),
            "Q": np.asarray(pulse["Amp_Q"], dtype=np.float64)}


def correlation_lag_s(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray],
                      dt: float = 1.0 / PFB_SAMPLING_FREQ) -> Optional[float]:
    """Where trace *b* best matches trace *a*, both sampled every *dt*:
    ``b``'s stamp minus ``a``'s for the same feature.  The magnitude
    excursion from each trace's median is correlated, so a rotation
    between them does not matter.  None when either is too short."""
    ta, tb = a["times"], b["times"]
    fa, fb = np.isfinite(ta), np.isfinite(tb)
    if fa.sum() < 2 or fb.sum() < 2:
        return None
    za = a["I"] + 1j * a["Q"]
    zb = b["I"] + 1j * b["Q"]
    xa = np.abs(za - np.median(za))
    xb = np.abs(zb - np.median(zb))
    xa = xa - xa.mean()
    xb = xb - xb.mean()
    if not xa.any() or not xb.any():
        return None
    # FFT: a 2.44 MHz pulse window is tens of thousands of samples, and a
    # direct correlation of two of them is minutes per pulse.
    from scipy.signal import correlate
    c = correlate(xb, xa, mode="full", method="fft")
    s = int(np.argmax(c)) - (len(xa) - 1)   # a's sample i sits at b's i + s
    t_a0 = ta[fa][0] - dt * np.flatnonzero(fa)[0]
    t_b0 = tb[fb][0] - dt * np.flatnonzero(fb)[0]
    return float(t_b0 - t_a0 + s * dt)


def _in_units(times, z: np.ndarray, factor: complex) -> Dict[str, np.ndarray]:
    z = np.asarray(z) * factor
    return {"times": np.asarray(times, dtype=np.float64),
            "I": z.real.astype(np.float64), "Q": z.imag.astype(np.float64)}


def pulse_overlay(reader: PulseHDF5Reader, recording, channel: int,
                  pulse_idx: int, stream: str = "slow",
                  pad_s: float = 0.0, dirfile=None,
                  module: Optional[int] = None) -> Overlay:
    """Pulse *pulse_idx* of *channel* with the fastrx samples over its
    window (plus *pad_s* either side), converted to the file's stored
    units.  *recording* is a :class:`rfmux.fastrx.PacketFile`.  For a
    slow pulse of a dual file the paired fast pulse comes too, with the
    correlation lag between it and the fastrx trace.  With *dirfile* (a
    board's parser subdirfile) its slow trace over the window comes
    too, in the same units; *module* defaults to the capture's."""
    stream_key = stream if reader.dual else None
    pulse = reader.get_pulse(channel, pulse_idx, stream_key)
    if pulse is None:
        raise KeyError(f"no pulse {pulse_idx} on channel {channel} ({stream})")
    shift = slow_shift_s(reader) if stream == "slow" else 0.0
    tod = _tod(pulse, shift)
    t = tod["times"][np.isfinite(tod["times"])]
    if t.size == 0:
        raise ValueError("pulse has no usable timestamps")
    t0, t1 = float(t[0]) - pad_s, float(t[-1]) + pad_s
    factor = counts_to_stored(reader, channel, stream_key)
    w = recording.window(t0, t1, channel)
    fastrx = _in_units(w.times, w.samples, factor)

    parsed = None
    if dirfile is not None:
        if module is None:
            module = int(reader.metadata.get("module", 1))
        d = dirfile_window(dirfile, module, channel, t0, t1)
        parsed = _in_units(d["times"], d["I"] + 1j * d["Q"], factor)

    fast = None
    if reader.dual and stream == "slow":
        for pair in reader.iter_matches(channel):
            if pair["slow_idx"] == pulse_idx and pair["fast_idx"] is not None:
                fp = reader.get_pulse(channel, pair["fast_idx"], "fast")
                if fp is not None:
                    fast = _tod(fp)
                break
    ref = fast if fast is not None else (tod if stream == "fast" else None)
    lag = correlation_lag_s(ref, fastrx) if ref is not None else None

    return Overlay(channel=channel, pulse_idx=pulse_idx, stream=stream,
                   units=reader.stored_units(channel, stream_key),
                   shift_s=shift, pulse=tod, fastrx=fastrx, fast=fast,
                   dirfile=parsed, lag_s=lag, seq_gaps=w.seq_gaps,
                   dropouts=w.dropouts)


def dirfile_window(path, module: int, channel: int, t0: float,
                   t1: float) -> Dict[str, Any]:
    """*channel* of *module* over seconds-of-day ``[t0, t1]`` from a
    parser dirfile (one board's subdirfile), as complex ADC counts on
    the PFB clock.  A dirfile written with ``fir_stage`` has its
    timebase corrected already; an older one is shifted here by the
    delay of the stage inferred from its frame spacing.  Only the
    timebase is read whole; the channel is read for the window."""
    import pygetdata as gd

    df = gd.dirfile(str(path), gd.RDONLY)
    prefix = f"m{module:02d}_"
    fields = {f.decode() if isinstance(f, bytes) else f
              for f in df.field_list()}
    tb = np.asarray(df.getdata(prefix + "timebase", gd.FLOAT64),
                    dtype=np.float64)
    shift = 0.0
    if prefix + "fir_stage" not in fields and tb.size > 1:
        step = float(np.median(np.diff(tb)))
        if step > 0:
            shift = -decimated_stream_delay_s(sampling_to_decimation(1.0 / step))
    tb = tb + shift
    i0 = int(np.searchsorted(tb, t0, side="left"))
    i1 = int(np.searchsorted(tb, t1, side="right"))
    z = np.asarray(df.getdata(prefix + f"c{channel:04d}", gd.COMPLEX128,
                              first_frame=int(i0), num_frames=int(i1 - i0)))
    df.close()
    # A channel field is phase-shifted out of the raw block, so the
    # last frame can come back short: keep the samples the read covered.
    i1 = i0 + len(z)
    return {"times": tb[i0:i1], "I": z.real, "Q": z.imag,
            "shift_s": shift, "start": int(i0), "stop": int(i1)}
