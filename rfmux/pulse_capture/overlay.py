"""Overlay a fastrx recording (the 100G channel stream) on 1G data by
IRIG time: the pulses of a capture file, or a window of a parser
dirfile.

All three streams are stamped by the board's IRIG clock.  The channel
stream shares the PFB stream's tap, so their stamps agree; the
decimated (slow) stream is stamped late by its CIC group delay, which
the capture session and the parser take out where they write
(``slow_time_offset_s`` in a capture file, ``dec_stage`` beside the
dirfile's corrected timebase).  Files written before that carry raw
stamps, and :func:`slow_shift_s` supplies the shift for them.

The fastrx extension is imported only when a :class:`Recording` is
opened from a path, so this module loads where it does not build.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..core.transferfunctions import (PFB_SAMPLING_FREQ, VOLTS_PER_ROC,
                                      decimated_stream_delay_s,
                                      sampling_to_decimation)
from ..streamer import SS_PER_SECOND
from .channel_keys import channel_group, split_key
from .hdf5 import PulseHDF5Reader

# ── The recording ─────────────────────────────────────────────────

#: Channels per pipeline block on the wire: bit p of a record's
#: pipe_snapshot says whether channels p*128+1 to (p+1)*128 were sent.
CHANNELS_PER_PIPE = 128

#: Scale from the truncated int16 on the wire to the ADC counts the 1G
#: paths report, by sample_trunc.  The 24-bit sample is in counts: LOW
#: (bits 15:0) is exact while |counts| < 32768, MID (bits 19:4) is
#: counts/16 and HIGH (bits 23:8) counts/256, each dropping the bits
#: below its window.  Measured on board 0156: HIGH against the parser's
#: counts reads 256 to 268, the excess being the dropped bits.
COUNTS_PER_LSB = {0: 1.0, 1: 16.0, 2: 256.0}

_DAY_S = 86400.0
#: Records probed past an undisciplined stamp before giving up on it.
_PROBE = 64
_RECENT = 0x80000000


def _seconds_of_day(ts) -> np.ndarray:
    """Seconds of day of IRIG stamps (a structured array or one element),
    NaN where the stamp is not disciplined."""
    ts = np.asarray(ts)
    t = (ts["h"].astype(np.float64) * 3600.0 + ts["m"] * 60.0 + ts["s"]
         + ts["ss"] / SS_PER_SECOND)
    return np.where(ts["c"] & _RECENT, t, np.nan)


@dataclass
class Window:
    """One channel's samples over a time window of a recording.

    ``times`` are seconds of day (see :meth:`Recording.seconds`), NaN
    where a record's stamp is not disciplined; ``samples`` are complex
    ADC counts.  ``seq_gaps`` counts sequence discontinuities inside the
    window and ``dropouts`` the records whose pipe the transmitter was
    not sending (zero-filled by the writer).  With a ``module`` only
    its records of ``start:stop`` are kept."""
    channel: int
    start: int
    stop: int
    module: Optional[int]
    times: np.ndarray
    samples: np.ndarray
    seq_gaps: int
    dropouts: int


class Recording:
    """A fastrx recording with a time index over its IRIG stamps.

    Wraps a ``rfmux.fastrx.PacketFile`` (or opens one from a path).  The
    extension maps the file and hands back strided views; nothing here
    reads more of it than the records asked for.  Every record is one
    sample of channels 1 to :attr:`channels`, stamped by the board, so
    a stamp is a sample time with no first-or-last-in-packet ambiguity.

    Time is seconds of day, the axis pulse-capture files and parser
    dirfiles use.  A recording that crosses midnight is unwrapped: any
    stamp more than half a day before the first one is taken as the next
    day, and queries are read the same way.

    The writer records every module the channel stream carries, one
    record per module per sample, interleaved; a query with a module
    keeps that module's records, and its sequence counter is read on
    its own.
    """

    def __init__(self, source):
        if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
            from ..fastrx import PacketFile
            source = PacketFile(str(source))
        self.file = source
        self._ts = source.ts()
        self._seq = source.seq()
        n = self.num_packets
        hdr0 = source.headers()[0] if n else None
        #: fastrx_trunc_t of the recording (from the first record).
        self.sample_trunc = int(hdr0["sample_trunc"]) if n else 2
        #: Wire module field of the first record, as sent.
        self.module = int(hdr0["module"]) if n else None
        self.counts_per_lsb = COUNTS_PER_LSB[self.sample_trunc]
        #: Seconds of day of the first disciplined stamp; None when the
        #: recording has none near its start, in which case there is no
        #: time axis to index by.
        self.t_first = None
        t, _ = self._second_from(0)
        self.t_first = None if t != t else float(t)

    @property
    def num_packets(self) -> int:
        return self.file.num_packets

    @property
    def channels(self) -> int:
        """Every record holds the module's channels 1 to this."""
        return int(self.file.channels)

    def __len__(self) -> int:
        return self.num_packets

    # ── time ──────────────────────────────────────────────────────

    def _unwrap(self, t):
        """Seconds of day onto the recording's monotone axis."""
        if self.t_first is None:
            return t
        return np.where(t < self.t_first - _DAY_S / 2, t + _DAY_S, t)

    def seconds(self, start: int = 0, stop: int | None = None) -> np.ndarray:
        """Seconds of day of records ``start:stop``, NaN where the stamp
        is not disciplined.  Touches only those records."""
        return self._unwrap(_seconds_of_day(self._ts[start:stop]))

    def _second_from(self, i: int):
        """(seconds, index) of the first disciplined stamp at or after
        record *i* within the probe distance; (NaN, i) if none."""
        for j in range(i, min(i + _PROBE, self.num_packets)):
            t = _seconds_of_day(self._ts[j])
            if t == t:
                return float(self._unwrap(t)), j
        return float("nan"), i

    def index_at(self, t: float, side: str = "left") -> int:
        """Record index for seconds-of-day *t*: the first record stamped
        at or after *t* (``side="left"``) or after it (``"right"``).
        A bisect, so a query costs about log2(records) page touches
        rather than a read of the file.  A record with no usable stamp
        is placed with the next disciplined one (so it can open a
        window, NaN-timed); a stretch of them longer than the probe
        distance is treated as beyond the end."""
        if self.t_first is None:
            raise ValueError("recording has no disciplined timestamp to "
                             "index by")
        t = float(self._unwrap(np.float64(t)))
        lo, hi = 0, self.num_packets
        while lo < hi:
            mid = (lo + hi) // 2
            tm, j = self._second_from(mid)
            if tm != tm:                       # nothing usable ahead of mid
                hi = mid
            elif tm < t or (side == "right" and tm == t):
                lo = j + 1
            else:
                hi = mid
        return lo

    # ── samples ───────────────────────────────────────────────────

    def _keep(self, start: int, stop: int | None, module: Optional[int]):
        """Index of the records of ``start:stop`` that are *module*'s
        (1-indexed; the wire counts from 0), or a slice of all of them
        for None."""
        if module is None:
            return slice(None)
        hdr = self.file.headers()[start:stop]
        return np.flatnonzero(hdr["module"] == module - 1)

    def channel(self, channel: int, start: int = 0,
                stop: int | None = None,
                module: Optional[int] = None) -> np.ndarray:
        """One channel's samples over records ``start:stop`` as complex
        ADC counts; *module*'s records only when given."""
        if not 1 <= channel <= self.channels:
            raise ValueError(f"channel {channel} is not in the recording, "
                             f"which holds 1..{self.channels}")
        iq = self.file.iq()[start:stop, channel - 1, :][self._keep(start, stop, module)]
        z = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)
        return z * np.float32(self.counts_per_lsb)

    def window(self, t0: float, t1: float, channel: int,
               module: Optional[int] = None) -> Window:
        """*channel* over seconds-of-day ``[t0, t1]``; *module*'s
        records only when given."""
        start = self.index_at(t0)
        stop = self.index_at(t1, side="right")
        keep = self._keep(start, stop, module)
        pipe = (channel - 1) // CHANNELS_PER_PIPE
        seq = self._seq[start:stop].astype(np.int64)[keep]
        snap = self.file.headers()[start:stop]["pipe_snapshot"][keep]
        return Window(
            channel=channel, start=start, stop=stop, module=module,
            times=self.seconds(start, stop)[keep],
            samples=self.channel(channel, start, stop)[keep],
            seq_gaps=int(np.count_nonzero(np.diff(seq) != 1)) if seq.size else 0,
            dropouts=int(np.count_nonzero((snap & (1 << pipe)) == 0)),
        )


# ── The 1G side ───────────────────────────────────────────────────

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
    ``dirfile`` the parser's slow trace when one was given; ``lag_s``
    is where the fastrx trace best matches the fast one, fastrx stamp
    minus fast stamp of the same feature."""
    channel: int
    pulse_idx: int
    stream: str
    units: str
    shift_s: float
    pulse: Dict[str, np.ndarray]
    fastrx: Dict[str, np.ndarray]
    fast: Optional[Dict[str, np.ndarray]] = None
    dirfile: Optional[Dict[str, np.ndarray]] = None
    lag_s: Optional[float] = None
    seq_gaps: int = 0
    dropouts: int = 0


def _tod(pulse: Dict[str, Any], shift: float = 0.0) -> Dict[str, np.ndarray]:
    return {"times": np.asarray(pulse["Time"], dtype=np.float64) + shift,
            "I": np.asarray(pulse["Amp_I"], dtype=np.float64),
            "Q": np.asarray(pulse["Amp_Q"], dtype=np.float64)}


def _in_units(times, z: np.ndarray, factor: complex) -> Dict[str, np.ndarray]:
    z = np.asarray(z) * factor
    return {"times": np.asarray(times, dtype=np.float64),
            "I": z.real.astype(np.float64), "Q": z.imag.astype(np.float64)}


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


def pulse_overlay(reader: PulseHDF5Reader, recording: Recording,
                  channel, pulse_idx: int, stream: str = "slow",
                  pad_s: float = 0.0, dirfile=None,
                  module: Optional[int] = None) -> Overlay:
    """Pulse *pulse_idx* of *channel* (a channel number, or a (module,
    channel) key of a capture across modules) with the recording's
    samples over its window (plus *pad_s* either side), converted to
    the file's stored units.  For a slow pulse of a dual file the
    paired fast pulse comes too, with the correlation lag between it
    and the fastrx trace.  With *dirfile* (a board's parser subdirfile)
    its slow trace over the window comes too, in the same units;
    *module* defaults to the key's, else the capture's."""
    stream_key = stream if reader.dual else None
    key = channel
    rec_module, channel = split_key(key, reader.metadata.get("module"))
    pulse = reader.get_pulse(key, pulse_idx, stream_key)
    if pulse is None:
        raise KeyError(f"no pulse {pulse_idx} on channel {key} ({stream})")
    shift = slow_shift_s(reader) if stream == "slow" else 0.0
    tod = _tod(pulse, shift)
    t = tod["times"][np.isfinite(tod["times"])]
    if t.size == 0:
        raise ValueError("pulse has no usable timestamps")
    t0, t1 = float(t[0]) - pad_s, float(t[-1]) + pad_s
    factor = counts_to_stored(reader, key, stream_key)
    w = recording.window(t0, t1, channel, module=rec_module)
    fastrx = _in_units(w.times, w.samples, factor)

    parsed = None
    if dirfile is not None:
        if module is None:
            module = rec_module if rec_module is not None else 1
        d = dirfile_window(dirfile, module, channel, t0, t1)
        parsed = _in_units(d["times"], d["I"] + 1j * d["Q"], factor)

    fast = None
    if reader.dual and stream == "slow":
        for pair in reader.iter_matches(key):
            if pair["slow_idx"] == pulse_idx and pair["fast_idx"] is not None:
                fp = reader.get_pulse(key, pair["fast_idx"], "fast")
                if fp is not None:
                    fast = _tod(fp)
                break
    ref = fast if fast is not None else (tod if stream == "fast" else None)
    lag = correlation_lag_s(ref, fastrx) if ref is not None else None

    return Overlay(channel=key, pulse_idx=pulse_idx, stream=stream,
                   units=reader.stored_units(key, stream_key),
                   shift_s=shift, pulse=tod, fastrx=fastrx, fast=fast,
                   dirfile=parsed, lag_s=lag, seq_gaps=w.seq_gaps,
                   dropouts=w.dropouts)


def dirfile_window(path, module: int, channel: int, t0: float,
                   t1: float) -> Dict[str, Any]:
    """*channel* of *module* over seconds-of-day ``[t0, t1]`` from a
    parser dirfile (one board's subdirfile), as complex ADC counts on
    the PFB clock.  A dirfile written with ``dec_stage`` has its
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
    if prefix + "dec_stage" not in fields and tb.size > 1:
        step = float(np.median(np.diff(tb)))
        if step > 0:
            shift = -decimated_stream_delay_s(sampling_to_decimation(1.0 / step))
    tb = tb + shift
    i0 = int(np.searchsorted(tb, t0, side="left"))
    i1 = int(np.searchsorted(tb, t1, side="right"))
    z = np.asarray(df.getdata(prefix + f"c{channel:04d}", gd.COMPLEX128,
                              first_frame=i0, num_frames=i1 - i0))
    df.close()
    # A channel field is phase-shifted out of the raw block, so the
    # last frame can come back short: keep the samples the read covered.
    i1 = i0 + len(z)
    return {"times": tb[i0:i1], "I": z.real, "Q": z.imag,
            "shift_s": shift, "start": i0, "stop": i1}


# ── Merging a recording into a capture file ───────────────────────

def merge_fastrx(pulse_path, fastrx_path, out=None,
                 noise_span_s: float = 0.2) -> Path:
    """Add a fastrx recording to a slow-stream capture file as its fast
    stream, making it a dual ("both") file: every slow pulse becomes a
    slow-triggered pair carrying the recording over the pulse's union
    window, in the file's stored units, and the fast side's noise is
    estimated from the recording's first *noise_span_s* seconds.  The
    slow side is copied unchanged.  In place unless *out* is given;
    returns the path written."""
    import os

    pulse_path = Path(pulse_path)
    out = Path(out) if out is not None else pulse_path
    rec = Recording(fastrx_path)
    if rec.t_first is None:
        raise ValueError(
            f"{fastrx_path}: no disciplined timestamp to index by")
    tmp = out.with_name(out.name + ".merging")
    try:
        with PulseHDF5Reader(pulse_path) as reader:
            _merge_into(reader, rec, tmp, noise_span_s)
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return out


def _merge_into(reader: PulseHDF5Reader, rec: Recording, tmp: Path,
                noise_span_s: float) -> None:
    from .accumulators import PulseHistogramSet, PulseTemplateSet
    from .analysis import storage_transform
    from .capture_session import DualPulseCaptureSession, PulseCaptureConfig
    from .detection import estimate_noise_stats
    from .hdf5 import DualPulseHDF5Writer

    if reader.dual:
        raise ValueError(f"{reader.path}: already a dual file")
    channels = list(reader.channels)
    # A key's module in the recording: its own, or the capture's; a
    # recording carries every module that streamed.
    where = {c: split_key(c, reader.metadata.get("module")) for c in channels}
    fast_channels = [c for c in channels if where[c][1] <= rec.channels]
    slow_rate = float(reader.metadata.get("sample_rate_slow") or 0.0)
    params = {**reader.metadata, "streamer_mode": "both",
              "sample_rate_fast": PFB_SAMPLING_FREQ,
              "fast_channels": fast_channels}
    cals = {c: reader.df_calibration(c) for c in channels
            if reader.df_calibration(c) is not None}
    units = {c: reader.stored_units(c) for c in channels}
    writer = DualPulseHDF5Writer(tmp, channels, params,
                                 df_calibrations=cals or None,
                                 stored_units=units)
    try:
        meta = writer.f["metadata"]
        for key in ("capture_start", "time_origin_epoch",
                    "time_origin_utc"):
            if key in reader.metadata:
                meta.attrs[key] = reader.metadata[key]
        for c in channels:
            key = channel_group(c)
            del writer.f["slow"][key]
            reader.f.copy(reader.f[key], writer.f["slow"], name=key)
        writer.update_histograms("slow", reader.get_histograms())
        writer.update_templates("slow", reader.get_templates())

        shift = slow_shift_s(reader)
        period = 1.0 / slow_rate if slow_rate else 0.0
        noise_stop = rec.index_at(rec.t_first + noise_span_s)
        factors = {c: counts_to_stored(reader, c) for c in channels}
        noise = {}
        for c in fast_channels:
            module, number = where[c]
            z = rec.channel(number, 0, noise_stop, module).astype(np.complex128)
            noise[c] = estimate_noise_stats({c: z * factors[c]}, [c])[0][c]

        # The fast side's histograms and templates, from the same
        # accumulators a live session feeds, over each pair's window;
        # sized as the session sizes them for this rate and its noise.
        meta = reader.metadata
        thr = float(meta.get("threshold_sigma", 0.0)) or None
        vpc = reader.volts_per_count() or VOLTS_PER_ROC
        to_raw = {}
        for c in fast_channels:
            co, _ = storage_transform(reader.df_calibration(c),
                                      reader.trigger_basis())
            to_raw[c] = (co / vpc if reader.stored_units(c) == "Hz"
                         else None)
        hists = PulseHistogramSet(threshold_sigma=thr)
        hists.size_amplitude_to_noise(
            max((max(s.std_I, s.std_Q) for s in noise.values()),
                default=0.0),
            raw_sigma=max((max(s.std_I, s.std_Q) / abs(to_raw[c])
                           for c, s in noise.items() if to_raw[c]),
                          default=None))
        config = PulseCaptureConfig(
            max_pulse_ms=float(meta.get("max_pulse_ms",
                                        PulseCaptureConfig().max_pulse_ms)))
        post = max(64, min(config.buf_size(PFB_SAMPLING_FREQ) // 2, 20000))
        templates = PulseTemplateSet(
            pre_samples=max(8, post // 10), post_samples=post,
            threshold_sigma=thr, sample_rate=PFB_SAMPLING_FREQ)

        for c in channels:
            recorded = c in fast_channels
            factor = factors[c]
            for idx in range(1, reader.pulse_count(c) + 1):
                pulse = reader.get_pulse(c, idx)
                pair = {"pair_idx": idx, "channel": c, "slow_idx": idx,
                        "fast_idx": None, "time_offset": None}
                t = np.asarray(pulse["Time"], dtype=np.float64) + shift
                t = t[np.isfinite(t)]
                if t.size:
                    window = DualPulseCaptureSession._union_window(
                        {"slow_summary": {
                            "start_time": float(t[0]),
                            "saved_end_time": float(t[-1])}},
                        period)
                    pair["window"] = window
                    if recorded:
                        w = rec.window(window[0], window[1], where[c][1],
                                       module=where[c][0])
                        ok = np.isfinite(w.times)
                        # A window the recording does not cover
                        # stays absent: the pair reads "fast n/a".
                        if ok.any():
                            tod = _in_units(w.times[ok],
                                            w.samples[ok], factor)
                            pair["fast_tod"] = {"Time": tod["times"],
                                                "Amp_I": tod["I"],
                                                "Amp_Q": tod["Q"]}
                            hists.add_pulse(c, pair["fast_tod"], noise[c],
                                            to_raw=to_raw[c])
                            templates.add_pulse(c, pair["fast_tod"],
                                                noise[c])
                writer.append_match(c, pair)
        writer.set_noise_stats("fast", noise)
        if hists.total_pulses():
            writer.update_histograms("fast", hists.get_histogram_data())
            tmpl = templates.get_template_data()
            if tmpl:
                writer.update_templates("fast", tmpl)
    finally:
        writer.finalize()
