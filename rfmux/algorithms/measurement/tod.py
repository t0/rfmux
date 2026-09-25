"""The time-ordered data of a run as one HDF5 file: the parser dirfile
(the slow stream) and the fastrx recording (the channel stream)
repacked a block of records at a time into the units a pulse capture
of the same channels stores, with the same metadata and each channel's
tuning beside the samples.  A reader then needs neither pygetdata nor
the fastrx extension, and the two streams compare directly.

Layout, the names of a capture file (``metadata`` and the channel
groups as :mod:`.hdf5` writes them)::

    metadata/                  module, channels, trigger_basis,
                               stored_units, volts_per_count,
                               sample_rate_slow, sample_rate_fast,
                               slow_time_offset_s, fast_channels,
                               time_origin_epoch, time_origin_utc
    tod/slow/time              seconds of day, on the PFB clock
    tod/slow/channel_<n>/I, Q  the channel in its stored units
    tod/slow/channel_<n>/tuning/, stored_units
    tod/fast/time              seconds of day, NaN where the stamp is
                               not disciplined
    tod/fast/seq               the record's sequence number
    tod/fast/pipe_snapshot     bit p set when channels p*128+1 to
                               (p+1)*128 were sent (else zero-filled)
    tod/fast/channel_<n>/I, Q, tuning/, stored_units
    tod/<stream>/channel_<n>/overview    (bins, 4): I min, I max, Q min,
                               Q max per overview_samples samples
    tod/<stream>/time_overview (bins, 2): each bin's first and last
                               finite stamp
    tod/<stream>               attribute overview_samples (4096)

A run across modules nests ``module_<m>/`` under each stream, each
module with its own ``time`` (and ``seq``, ``pipe_snapshot``,
``time_overview``), since the recording interleaves the modules'
records.  Samples are float32: the wire carries at most 24 bits.
:func:`merge_tod` copies ``tod/`` into the run's pulse file so one file
holds the pulses and the streams they were cut from; :func:`tod_window`
reads a time window of one channel fit to draw.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import h5py
import numpy as np

from ...core.transferfunctions import (PFB_SAMPLING_FREQ, VOLTS_PER_ROC,
                                      decimated_stream_delay_s,
                                      decimation_to_sampling)
from ...streamer import TIMESTAMP_RECENT, day_epoch, epoch_to_utc
from ...pulse_capture.analysis import calibration_of, storage_transform
from ...pulse_capture.channel_keys import (ChannelKey, channel_group,
                                           check_keys, describe,
                                           keys_by_module, keys_from_attr)
from ...pulse_capture.hdf5 import _store_tuning, _store_units, write_metadata
from ...pulse_capture.overlay import _PROBE, Recording, dirfile_stage

#: Records converted and written at a time.
BLOCK = 1 << 16
#: Samples per overview bin: each channel's min and max of I and Q per
#: this many samples, so a view of the whole run reads kilobytes.
OVERVIEW = 4096
#: Bins a view is reduced to; at most twice this many samples, a view is
#: the samples themselves.
VIEW_BINS = 500
#: Chunk of a slow-stream dataset: a run's slow stream is short next to
#: its channel stream, and a chunk is allocated whole.
SLOW_CHUNK = 4096
#: Metadata a merge brings into a pulse file that lacks it.
_STREAM_FACTS = ("sample_rate_slow", "sample_rate_fast", "slow_time_offset_s",
                 "time_origin_epoch", "time_origin_utc")
#: Bytes a fast-stream record costs in the file, per channel (I and Q
#: as float32) and per record (time, seq, pipe_snapshot).
FAST_BYTES_PER_CHANNEL = 8
FAST_BYTES_PER_RECORD = 13


def tod_bytes_per_s(fast_channels: int) -> float:
    """Disk rate of the fast stream of *fast_channels* channels."""
    return float((FAST_BYTES_PER_RECORD
                  + FAST_BYTES_PER_CHANNEL * fast_channels) * PFB_SAMPLING_FREQ)


def write_tod(out, channels: Iterable[ChannelKey], module: Optional[int] = None,
              *, fastrx=None, dirfile=None, tuning: Optional[Dict] = None,
              trigger_basis: str = "df", time_origin_epoch: Optional[float] = None,
              block: int = BLOCK, overview: int = OVERVIEW) -> Path:
    """Write *out*: the time-ordered data of *channels* (numbers on
    *module*, or (module, channel) pairs) from a fastrx recording
    and/or a parser dirfile (one board's subdirfile).  Each channel is
    converted by :func:`~.analysis.storage_transform` with its row of
    *tuning* and *trigger_basis*, as a capture stores it: volts, or
    hertz for a calibrated channel in the frequency basis.  The packet
    clock's day comes from *time_origin_epoch* or the recording."""
    if fastrx is None and dirfile is None:
        raise ValueError("nothing to repack: give a recording, a dirfile "
                         "or both")
    keys = check_keys(channels)
    if not keys:
        raise ValueError("no channels")
    if module is None and not isinstance(keys[0], tuple):
        raise ValueError("channel numbers need their module; a run across "
                         "modules gives (module, channel) pairs")
    tuning = tuning or {}
    factors, units = {}, {}
    for c in keys:
        factors[c], units[c] = storage_transform(
            calibration_of(tuning.get(c)), trigger_basis)
    distinct = set(units.values())
    params = {"module": module, "trigger_basis": trigger_basis,
              "stored_units": distinct.pop() if len(distinct) == 1 else "mixed",
              "volts_per_count": VOLTS_PER_ROC}
    rec = None
    if fastrx is not None:
        rec = Recording(fastrx)
        params["sample_rate_fast"] = PFB_SAMPLING_FREQ
        params["fast_channels"] = [c for c in keys
                                   if _number(c) <= rec.channels]
        if time_origin_epoch is None:
            time_origin_epoch = _recording_day_epoch(rec)
    df = None
    if dirfile is not None:
        import pygetdata as gd
        df = gd.dirfile(str(dirfile), gd.RDONLY)
        stage, _ = dirfile_stage(df, next(iter(keys_by_module(keys, module))))
        if stage is not None:
            params["sample_rate_slow"] = decimation_to_sampling(stage)
            params["slow_time_offset_s"] = -decimated_stream_delay_s(stage)

    out = Path(out)
    tmp = out.with_name(out.name + ".writing")
    try:
        with h5py.File(tmp, "w") as f:
            write_metadata(f, keys, params)
            if time_origin_epoch is not None:
                f["metadata"].attrs["time_origin_epoch"] = float(time_origin_epoch)
                f["metadata"].attrs["time_origin_utc"] = epoch_to_utc(time_origin_epoch)
            if df is not None:
                sgrp = f.create_group("tod/slow")
                sgrp.attrs["overview_samples"] = overview
                _write_slow(sgrp, df, keys, module, factors, units, tuning,
                            block, overview)
            if rec is not None:
                sgrp = f.create_group("tod/fast")
                sgrp.attrs["overview_samples"] = overview
                _write_fast(sgrp, rec, keys, module, factors, units, tuning,
                            block, overview)
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        if df is not None:
            df.close()
    return out


def _number(key: ChannelKey) -> int:
    return key[1] if isinstance(key, tuple) else int(key)


def _recording_day_epoch(rec: Recording) -> Optional[float]:
    """Midnight of the packet clock's day, from the recording's first
    disciplined stamp; None without one near the start."""
    ts = rec.file.ts()[:_PROBE]
    ok = np.flatnonzero(ts["c"] & TIMESTAMP_RECENT)
    if not ok.size:
        return None
    first = ts[ok[0]]
    return day_epoch(int(first["y"]), int(first["d"]))


def _channel_groups(sgrp, keys: List[ChannelKey], units, tuning):
    """A group per key under stream group *sgrp*, with its ``I`` and
    ``Q`` datasets to append to, tuning and units."""
    out = {}
    for c in keys:
        grp = sgrp.create_group(channel_group(c))
        _store_tuning(grp, tuning, c)
        _store_units(grp, units, c)
        out[c] = grp
    return out


def _appendable(grp, name: str, dtype, block: int):
    return grp.create_dataset(name, shape=(0,), maxshape=(None,),
                              dtype=dtype, chunks=(block,))


def _append(ds, values: np.ndarray) -> None:
    n = ds.shape[0]
    ds.resize(n + len(values), axis=0)
    ds[n:] = values


class _Overview:
    """Each channel's min and max of I and Q per *size* samples
    (``overview``, rows of I min, I max, Q min, Q max), and the first
    and last finite stamp of each bin (``time_overview`` beside the
    ``time`` they summarise), appended as blocks arrive.  A bin that a
    block leaves partial is completed by the next; the last bin of the
    stream may be short."""

    def __init__(self, time_grp, channel_grps, size: int):
        self.size = size
        self.t_ds = time_grp.create_dataset(
            "time_overview", (0, 2), maxshape=(None, 2), dtype=np.float64,
            chunks=(1024, 2))
        self.ds = [g.create_dataset("overview", (0, 4), maxshape=(None, 4),
                                    dtype=np.float32, chunks=(1024, 4))
                   for g in channel_grps]
        self.t_tail = np.empty(0)
        self.z_tail = np.empty((0, len(channel_grps)), np.complex64)

    def feed(self, t: np.ndarray, z: np.ndarray) -> None:
        """*t* (records,) and *z* (records, channels) complex."""
        if len(self.t_tail):
            head = self.size - len(self.t_tail)
            self.t_tail = np.concatenate([self.t_tail, t[:head]])
            self.z_tail = np.concatenate([self.z_tail, z[:head]])
            t, z = t[head:], z[head:]
            if len(self.t_tail) < self.size:
                return
            self._emit(self.t_tail, self.z_tail, self.size)
            self.t_tail, self.z_tail = self.t_tail[:0], self.z_tail[:0]
        m = len(t) // self.size * self.size
        if m:
            self._emit(t[:m], z[:m], self.size)
        self.t_tail, self.z_tail = t[m:].copy(), z[m:].copy()

    def finish(self) -> None:
        if len(self.t_tail):
            self._emit(self.t_tail, self.z_tail, len(self.t_tail))

    def _emit(self, t, z, size: int) -> None:
        k = len(t) // size
        r = z.reshape(k, size, z.shape[1])
        rows = np.stack([r.real.min(1), r.real.max(1),
                         r.imag.min(1), r.imag.max(1)], axis=2)
        for j, ds in enumerate(self.ds):
            _append(ds, rows[:, j, :])
        tt = t.reshape(k, size)
        _append(self.t_ds, np.stack([np.fmin.reduce(tt, axis=1),
                                     np.fmax.reduce(tt, axis=1)], axis=1))


def _write_slow(sgrp, df, keys, module, factors, units, tuning, block,
                overview) -> None:
    import pygetdata as gd

    for m, members in keys_by_module(keys, module).items():
        prefix = f"m{m:02d}_"
        mgrp = sgrp if module is not None else sgrp.require_group(f"module_{m}")
        _, shift = dirfile_stage(df, m)
        groups = _channel_groups(sgrp, [k for _, k in members], units, tuning)
        chunk = min(block, SLOW_CHUNK)
        time_ds = _appendable(mgrp, "time", np.float64, chunk)
        iq_ds = {k: (_appendable(groups[k], "I", np.float32, chunk),
                     _appendable(groups[k], "Q", np.float32, chunk))
                 for _, k in members}
        ov = _Overview(mgrp, [groups[k] for _, k in members], overview)
        nframes = df.nframes
        for a in range(0, nframes, block):
            n = min(block, nframes - a)
            tb = np.asarray(df.getdata(prefix + "timebase", gd.FLOAT64,
                                       first_frame=a, num_frames=n),
                            dtype=np.float64) + shift
            zs = {k: (np.asarray(df.getdata(prefix + f"c{number:04d}",
                                            gd.COMPLEX128, first_frame=a,
                                            num_frames=n)) * factors[k]
                      ).astype(np.complex64)
                  for number, k in members}
            # A channel field is phase-shifted out of the raw block, so
            # the last frame can come back short: keep what every
            # channel covers.
            n = min(len(tb), *(len(z) for z in zs.values()))
            _append(time_ds, tb[:n])
            for k, z in zs.items():
                _append(iq_ds[k][0], z.real[:n])
                _append(iq_ds[k][1], z.imag[:n])
            ov.feed(tb[:n], np.stack([z[:n] for z in zs.values()], axis=1))
        ov.finish()


def _write_fast(sgrp, rec: Recording, keys, module, factors, units, tuning,
                block, overview) -> None:
    headers = rec.file.headers()
    iq_all = rec.file.iq()
    lsb = np.complex64(rec.counts_per_lsb)
    for m, members in keys_by_module(keys, module).items():
        members = [(n, k) for n, k in members if n <= rec.channels]
        if not members:
            continue
        mgrp = sgrp if module is not None else sgrp.require_group(f"module_{m}")
        groups = _channel_groups(sgrp, [k for _, k in members], units, tuning)
        time_ds = _appendable(mgrp, "time", np.float64, block)
        seq_ds = _appendable(mgrp, "seq", np.uint32, block)
        snap_ds = _appendable(mgrp, "pipe_snapshot", np.uint8, block)
        iq_ds = [(_appendable(groups[k], "I", np.float32, block),
                  _appendable(groups[k], "Q", np.float32, block))
                 for _, k in members]
        columns = np.array([n - 1 for n, _ in members])
        scale = np.array([lsb * np.complex64(factors[k]) for _, k in members],
                         dtype=np.complex64)
        ov = _Overview(mgrp, [groups[k] for _, k in members], overview)
        for a in range(0, rec.num_packets, block):
            b = min(a + block, rec.num_packets)
            hdr = headers[a:b]
            keep = np.flatnonzero(hdr["module"] == m - 1)
            if not keep.size:
                continue
            t = rec.seconds(a, b)[keep]
            _append(time_ds, t)
            _append(seq_ds, hdr["seq"][keep])
            _append(snap_ds, hdr["pipe_snapshot"][keep])
            # The block's wanted channels converted in one pass:
            # (records, channels) complex, scaled per channel.
            iq = iq_all[a:b][keep][:, columns, :].astype(np.float32)
            z = (iq[..., 0] + 1j * iq[..., 1]).astype(np.complex64) * scale
            for j, (i_ds, q_ds) in enumerate(iq_ds):
                _append(i_ds, np.ascontiguousarray(z.real[:, j]))
                _append(q_ds, np.ascontiguousarray(z.imag[:, j]))
            ov.feed(t, z)
        ov.finish()


def _check_same_units(pulse: h5py.File, tod: h5py.File, pulse_path,
                      tod_path) -> None:
    """Every channel the two files share must be stored in the same
    units: a merged file's pulses and streams compare directly or not
    at all."""
    slow = pulse["slow"] if "slow" in pulse and "fast" in pulse else pulse
    basis = (pulse["metadata"].attrs.get("trigger_basis"),
             tod["metadata"].attrs.get("trigger_basis"))
    if None not in basis and basis[0] != basis[1]:
        raise ValueError(f"{pulse_path.name} is in the {basis[0]} basis, "
                         f"{tod_path.name} in the {basis[1]} basis")
    for key in keys_from_attr(tod["metadata"].attrs.get("channels", [])):
        group = channel_group(key)
        if group not in slow:
            continue
        theirs = slow[group].attrs.get("stored_units")
        for stream in tod["tod"].values():
            if group in stream and theirs is not None \
                    and stream[group].attrs.get("stored_units") != theirs:
                raise ValueError(
                    f"{describe(key)}: {pulse_path.name} stores it in "
                    f"{theirs}, {tod_path.name} in "
                    f"{stream[group].attrs.get('stored_units')}")


def merge_tod(pulse_path, tod_path, out=None) -> Path:
    """Copy a time-ordered data file's ``tod/`` group into a pulse
    capture file of the same run, so one file holds the pulses and the
    streams; the copy runs inside the HDF5 library, a chunk at a time.
    In place unless *out* is given; returns the path written."""
    pulse_path, tod_path = Path(pulse_path), Path(tod_path)
    out = Path(out) if out is not None else pulse_path
    tmp = out.with_name(out.name + ".merging")
    try:
        shutil.copyfile(pulse_path, tmp)
        with h5py.File(tmp, "a") as dst, h5py.File(tod_path, "r") as src:
            if "tod" in dst:
                raise ValueError(f"{pulse_path}: already holds tod/")
            if "tod" not in src:
                raise ValueError(f"{tod_path}: no tod/ group; not a "
                                 "time-ordered data file")
            _check_same_units(dst, src, pulse_path, tod_path)
            src.copy(src["tod"], dst, name="tod")
            # The clock facts the capture did not record, the fast
            # rate above all; what it did record stands.
            meta = dst["metadata"].attrs
            for key in _STREAM_FACTS:
                if key not in meta and key in src["metadata"].attrs:
                    meta[key] = src["metadata"].attrs[key]
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return out


# ── Viewing ────────────────────────────────────────────────────────

def _groups(f: h5py.File, stream: str, key: ChannelKey):
    """(stream group, the group holding *key*'s ``time``, *key*'s
    channel group) in the time-ordered data of an open file."""
    sgrp = f[f"tod/{stream}"]
    tgrp = sgrp[f"module_{key[0]}"] if isinstance(key, tuple) else sgrp
    return sgrp, tgrp, sgrp[channel_group(key)]


def _backfill(t: np.ndarray, after: float = np.inf) -> np.ndarray:
    """*t* with each NaN replaced by the next finite value, *after* past
    the last one: a stamp axis with undisciplined stretches, searchable
    as a sorted one."""
    bad = ~np.isfinite(t)
    if not bad.any():
        return t
    n = len(t)
    nxt = np.where(bad, n, np.arange(n))
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    return np.append(t, after)[nxt]


def index_at(f: h5py.File, stream: str, key: ChannelKey, t: float,
             side: str = "left") -> int:
    """Index of the first sample of *key*'s stream stamped at or after
    *t* (``"left"``) or after it (``"right"``).  The bin comes from the
    time overview, then the sample from that bin's stamps: two small
    reads however long the run, and a stretch of undisciplined (NaN)
    stamps of any length is placed with the next finite stamp."""
    sgrp, tgrp, _ = _groups(f, stream, key)
    t_ds = tgrp["time"]
    n = t_ds.shape[0]
    size = int(sgrp.attrs.get("overview_samples", 0))
    if "time_overview" in tgrp and size:
        # Each bin's first stamp, a NaN bin taking the next bin's: the
        # stamp its first sample is placed with.  The sample sought
        # then lies in the bin before the first at or after t.
        # ponytail: the column is read whole per lookup, 8 bytes per
        # 4096 samples (100 kB for 20 s); bisect it on disk if hour-long
        # runs make refreshes slow.
        first = _backfill(tgrp["time_overview"][:, 0])
    else:
        size, first = max(n, 1), np.empty(0)
    k = int(np.searchsorted(first, t, side=side))
    if k == 0 and len(first):
        return 0
    a = max(k - 1, 0) * size
    seg = _backfill(t_ds[a:min(a + size, n)],
                    first[k] if k < len(first) else np.inf)
    return a + int(np.searchsorted(seg, t, side=side))


def tod_extent(f: h5py.File, stream: str, key: ChannelKey):
    """(first, last) finite stamp of *key*'s stream, from its time
    overview; (NaN, NaN) for an empty stream."""
    _, tgrp, _ = _groups(f, stream, key)
    if "time_overview" in tgrp:
        ov = tgrp["time_overview"][()]
        first, last = ov[:, 0], ov[:, 1]
    else:                                  # a file written without one
        first = last = tgrp["time"][()]
    first, last = first[np.isfinite(first)], last[np.isfinite(last)]
    return ((float(first[0]), float(last[-1])) if first.size
            else (np.nan, np.nan))


def _reduce(t_first, t_last, i_min, i_max, q_min, q_max, bins: int):
    """Groups of consecutive bins merged into at most *bins*."""
    n = len(t_first)
    starts = np.arange(0, n, max(1, -(-n // bins)))
    return {"t_first": np.fmin.reduceat(t_first, starts),
            "t_last": np.fmax.reduceat(t_last, starts),
            "i_min": np.minimum.reduceat(i_min, starts),
            "i_max": np.maximum.reduceat(i_max, starts),
            "q_min": np.minimum.reduceat(q_min, starts),
            "q_max": np.maximum.reduceat(q_max, starts)}


def _convert(i, q, factor: complex):
    """(I, Q) as (I + jQ) * *factor*, the conversion a view applies."""
    if factor == 1:
        return i, q
    z = (np.asarray(i, np.float64) + 1j * np.asarray(q, np.float64)) * factor
    return z.real, z.imag


def _bounds(lo, hi, coef: float):
    """Extremes of *coef* times a value lying in [lo, hi]."""
    a, b = coef * lo, coef * hi
    return np.minimum(a, b), np.maximum(a, b)


def tod_window(f: h5py.File, stream: str, key: ChannelKey, t0: float,
               t1: float, bins: int = VIEW_BINS,
               factor: complex = 1) -> dict:
    """*key*'s samples of *stream* stamped in ``[t0, t1]``, fit to draw:

    * ``kind="raw"``: ``time``, ``I``, ``Q``, the samples themselves,
      when there are at most twice *bins* of them;
    * ``kind="envelope"``: at most *bins* bins, each with ``t_first``
      and ``t_last`` and the min and max of I and Q (``i_min`` ...), from
      the overview when the window spans at least *bins* of its bins
      (``source="overview"``), else from the samples
      (``source="samples"``).

    ``samples`` is how many samples the window holds.  Only the
    window's slice of the file is read, and an overview read covers
    the whole run in kilobytes.

    *factor* converts the stored (I, Q) to a view, (I + jQ) * factor,
    as Periscope's units choice does.  Samples are converted before
    they are reduced, exactly.  The overview holds each stored axis's
    extremes, so a factor that turns I into Q gives each bin bounds
    that hold its converted samples (``bounds=True``): never narrower
    than the samples, possibly wider."""
    sgrp, tgrp, cgrp = _groups(f, stream, key)
    t_ds = tgrp["time"]
    a = index_at(f, stream, key, t0)
    b = index_at(f, stream, key, t1, side="right")
    n = b - a
    if n <= 2 * bins:
        i, q = _convert(cgrp["I"][a:b], cgrp["Q"][a:b], factor)
        return {"kind": "raw", "samples": n, "time": t_ds[a:b],
                "I": i, "Q": q}
    size = int(sgrp.attrs.get("overview_samples", 0))
    if size and "overview" in cgrp and n >= size * bins:
        ka, kb = a // size, -(-b // size)
        ov = cgrp["overview"][ka:kb].astype(np.float64)
        tov = tgrp["time_overview"][ka:kb]
        view = _reduce(tov[:, 0], tov[:, 1], *ov.T, bins)
        c, s = complex(factor).real, complex(factor).imag
        if s == 0:
            view["i_min"], view["i_max"] = _bounds(view["i_min"],
                                                   view["i_max"], c)
            view["q_min"], view["q_max"] = _bounds(view["q_min"],
                                                   view["q_max"], c)
        else:
            # I' = c I - s Q and Q' = s I + c Q, each term bounded.
            ci = _bounds(view["i_min"], view["i_max"], c)
            si = _bounds(view["i_min"], view["i_max"], s)
            cq = _bounds(view["q_min"], view["q_max"], c)
            sq = _bounds(view["q_min"], view["q_max"], -s)
            view["i_min"], view["i_max"] = ci[0] + sq[0], ci[1] + sq[1]
            view["q_min"], view["q_max"] = si[0] + cq[0], si[1] + cq[1]
        return {"kind": "envelope", "source": "overview", "samples": n,
                "bounds": s != 0, **view}
    t = t_ds[a:b]
    i, q = _convert(cgrp["I"][a:b], cgrp["Q"][a:b], factor)
    view = _reduce(t, t, i, i, q, q, bins)
    return {"kind": "envelope", "source": "samples", "samples": n, **view}
