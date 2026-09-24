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

A run across modules nests ``module_<m>/`` under each stream, each
module with its own ``time`` (and ``seq``, ``pipe_snapshot``), since
the recording interleaves the modules' records.  Samples are float32:
the wire carries at most 24 bits.  :func:`merge_tod` copies ``tod/``
into the run's pulse file so one file holds the pulses and the streams
they were cut from.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import h5py
import numpy as np

from ..core.transferfunctions import (PFB_SAMPLING_FREQ, VOLTS_PER_ROC,
                                      decimated_stream_delay_s,
                                      decimation_to_sampling)
from ..streamer import TIMESTAMP_RECENT, day_epoch, epoch_to_utc
from .analysis import calibration_of, storage_transform
from .channel_keys import (ChannelKey, channel_group, check_keys,
                           keys_by_module)
from .hdf5 import _store_tuning, _store_units, write_metadata
from .overlay import _PROBE, Recording, dirfile_stage

#: Records converted and written at a time.
BLOCK = 1 << 16
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
              block: int = BLOCK) -> Path:
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
                _write_slow(f.create_group("tod/slow"), df, keys, module,
                            factors, units, tuning, block)
            if rec is not None:
                _write_fast(f.create_group("tod/fast"), rec, keys, module,
                            factors, units, tuning, block)
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


def _write_slow(sgrp, df, keys, module, factors, units, tuning, block) -> None:
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


def _write_fast(sgrp, rec: Recording, keys, module, factors, units, tuning,
                block) -> None:
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
        for a in range(0, rec.num_packets, block):
            b = min(a + block, rec.num_packets)
            hdr = headers[a:b]
            keep = np.flatnonzero(hdr["module"] == m - 1)
            if not keep.size:
                continue
            _append(time_ds, rec.seconds(a, b)[keep])
            _append(seq_ds, hdr["seq"][keep])
            _append(snap_ds, hdr["pipe_snapshot"][keep])
            # The block's wanted channels converted in one pass:
            # (records, channels) complex, scaled per channel.
            iq = iq_all[a:b][keep][:, columns, :].astype(np.float32)
            z = (iq[..., 0] + 1j * iq[..., 1]).astype(np.complex64) * scale
            for j, (i_ds, q_ds) in enumerate(iq_ds):
                _append(i_ds, np.ascontiguousarray(z.real[:, j]))
                _append(q_ds, np.ascontiguousarray(z.imag[:, j]))


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
