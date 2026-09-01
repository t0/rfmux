"""
Streaming HDF5 writer and lazy reader for pulse capture data.

The writer opens an HDF5 file at construction and appends pulses
incrementally as they are detected — it never holds all pulses in
memory.  The reader loads metadata eagerly but defers waveform data
loading until explicitly requested, enabling efficient browsing of
large capture files.

Usage (write)::

    writer = PulseHDF5Writer("capture.h5", [1, 2], noise_stats, params)
    writer.append_pulse(channel=1, pulse_idx=1, pulse_data={...})
    writer.finalize()

Usage (read)::

    with PulseHDF5Reader("capture.h5") as reader:
        print(reader.channels, reader.pulse_count(1))
        pulse = reader.get_pulse(channel=1, pulse_idx=1)
        for meta in reader.iter_pulse_metadata(channel=1):
            print(meta)
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import h5py

from .detection import ChannelNoiseStats
from .analysis import pulse_summary



def _store_units(grp, stored_units, channel) -> None:
    """Record what a channel's stored samples actually are.

    Per channel rather than per file: a capture in the frequency basis
    still holds uncalibrated channels in volts, because there is nothing
    to rotate them with.  A reader that assumed one answer for the whole
    file would mislabel those.
    """
    if not isinstance(stored_units, dict):
        return
    units = stored_units.get(channel)
    if units:
        grp.attrs["stored_units"] = str(units)


def _store_df_calibration(grp, df_calibrations, channel) -> None:
    """Stamp *channel*'s df calibration onto *grp*, if there is a usable one.

    Expects the flat ``{channel: calibration}`` mapping.  Anything h5py
    cannot store as an attribute is skipped with a warning rather than
    raised: the pulses are worth more than the units they are labelled
    in, and a writer that refuses to open costs the whole capture.
    """
    if not isinstance(df_calibrations, dict) or not df_calibrations:
        return
    value = df_calibrations.get(channel)
    if value is None:
        return
    if not isinstance(value, (int, float, complex, np.number)):
        warnings.warn(
            f"ignoring df_calibration for channel {channel}: expected a "
            f"number, got {type(value).__name__}.  df_calibrations is the "
            f"flat {{channel: calibration}} mapping, not one keyed by module.",
            stacklevel=3)
        return
    grp.attrs["df_calibration"] = value


# ───────────────────────── Shared writer plumbing ───────────────────

class _PulseFileWriter:
    """What the single- and dual-stream writers do identically.

    Both hold one open file, write a ``metadata`` group of capture
    parameters, stamp noise statistics onto channel groups, replace
    histogram/template datasets wholesale on every flush, and flush
    after each write for crash safety.  Only the LAYOUT differs —
    where a channel group lives, and whether there are two sets of
    them — so that is all the subclasses carry.
    """

    #: capture_params written to ``metadata``, grouped by attribute type.
    #: Must cover everything in
    #: :data:`~.session.DETECTION_PARAMS` — a parameter
    #: missing here is dropped without complaint.
    #: test_every_detection_param_reaches_the_file pins it.
    _META = (
        (str, ("streamer_mode", "trigger_basis", "stored_units")),
        (float, ("threshold_sigma", "end_sigma", "margin_fraction",
                 "sample_rate_slow", "sample_rate_fast",
                 "volts_per_count")),
        (int, ("min_pulse_samples", "module", "trigger_samples",
               "baseline_window", "edge_lookback", "max_capture_samples", "min_end_samples")),
        (bool, ("enable_pileup", "save_to_end_confirmed")),
    )

    def __init__(self, path: str | Path, channels: List[int],
                 capture_params: Dict[str, Any]):
        self.path = Path(path)
        self._channels = list(channels)
        self._threshold_sigma = capture_params.get("threshold_sigma")
        self.f: Optional[h5py.File] = h5py.File(self.path, "w")

        meta = self.f.create_group("metadata")
        meta.attrs["capture_start"] = time.time()
        meta.attrs["format_version"] = 1
        for cast, keys in self._META:
            for k in keys:
                if k in capture_params:
                    meta.attrs[k] = cast(capture_params[k])
        meta.attrs["channels"] = channels

    # ── Shared helpers ────────────────────────────────────────────

    @staticmethod
    def _write_noise_attrs(grp, ns: ChannelNoiseStats) -> None:
        grp.attrs["noise_mean_I"] = ns.mean_I
        grp.attrs["noise_std_I"] = ns.std_I
        grp.attrs["noise_mean_Q"] = ns.mean_Q
        grp.attrs["noise_std_Q"] = ns.std_Q
        grp.attrs["noise_jump_std_I"] = ns.jump_std_I
        grp.attrs["noise_jump_std_Q"] = ns.jump_std_Q

    def _set_noise_stats(self, key_for,
                         noise_stats: Dict[int, ChannelNoiseStats]) -> None:
        """Stamp per-channel noise attrs; *key_for* maps channel → group."""
        if not self.is_open:
            return
        for ch, ns in noise_stats.items():
            key = key_for(ch)
            if key in self.f:
                self._write_noise_attrs(self.f[key], ns)
        self.f.flush()

    def _append_pulse_to(self, key: str, pulse_idx: int, pulse_data: dict,
                         noise_stats: Optional[ChannelNoiseStats]) -> None:
        if not self.is_open or key not in self.f:
            return
        grp = self.f[key]
        _write_pulse(grp, pulse_idx, pulse_data, noise_stats,
                     self._threshold_sigma)
        grp.attrs["pulse_count"] = pulse_idx
        self.f.flush()

    def _read_pulse_at(self, key: str) -> Optional[dict]:
        if not self.is_open or key not in self.f:
            return None
        return _pulse_dict_from_group(self.f[key])

    def _replace_datasets(self, group_key: str,
                          data: Dict[str, np.ndarray]) -> None:
        """Overwrite a group's datasets wholesale (histograms/templates).

        Running accumulators are rewritten in full on every flush rather
        than appended to, so the file always holds one self-consistent
        snapshot however the capture ends.
        """
        if not self.is_open:
            return
        grp = self.f.require_group(group_key)
        for key, arr in data.items():
            if key in grp:
                del grp[key]
            grp.create_dataset(key, data=np.asarray(arr))
        self.f.flush()

    # ── Lifecycle ─────────────────────────────────────────────────

    def finalize(self) -> None:
        """Write final metadata and close the HDF5 file."""
        if self.is_open:
            self.f["metadata"].attrs["capture_end"] = time.time()
            self.f.flush()
            self.f.close()
        self.f = None

    @property
    def is_open(self) -> bool:
        return self.f is not None and self.f.id.valid

    def __del__(self):
        try:
            if self.is_open:
                self.finalize()
        except Exception:
            pass


# ───────────────────────── Writer ───────────────────────────────────

class PulseHDF5Writer(_PulseFileWriter):
    """Streaming HDF5 writer — appends pulses as detected.

    Opens the file at construction and writes capture metadata and
    per-channel noise statistics.  Each call to :meth:`append_pulse`
    creates a new HDF5 group with compressed waveform datasets and
    metadata attributes.  The file is flushed after every write for
    crash safety.

    Parameters
    ----------
    path : str or Path
        Output HDF5 file path.  Parent directories must exist.
    channels : list[int]
        Channel numbers being captured.
    noise_stats : dict[int, ChannelNoiseStats]
        Per-channel noise statistics from the estimation phase.
    capture_params : dict
        Capture configuration (streamer_mode, threshold_sigma, etc.).
    df_calibrations : dict[int, float], optional
        Per-channel df calibration values (Hz per ADC count).
    """

    def __init__(
        self,
        path: str | Path,
        channels: List[int],
        noise_stats: Dict[int, ChannelNoiseStats],
        capture_params: Dict[str, Any],
        df_calibrations: Optional[Dict[int, float]] = None,
        stored_units: Optional[Dict[int, str]] = None,
    ):
        super().__init__(path, channels, capture_params)
        self._noise_stats = dict(noise_stats)

        # ── Per-channel groups ────────────────────────────────────
        for ch in channels:
            grp = self.f.create_group(f"channel_{ch}")
            self._write_noise_attrs(grp, noise_stats.get(
                ch, ChannelNoiseStats()))
            grp.attrs["pulse_count"] = 0
            _store_df_calibration(grp, df_calibrations, ch)
            _store_units(grp, stored_units, ch)

        # ── Histogram / template groups (updated periodically) ────
        self.f.create_group("histograms")
        self.f.create_group("templates")
        self.f.flush()

    # ── Public API ────────────────────────────────────────────────

    def append_pulse(
        self,
        channel: int,
        pulse_idx: int,
        pulse_data: dict,
        noise_stats: Optional[ChannelNoiseStats] = None,
    ) -> None:
        """Append a single detected pulse to the HDF5 file.

        Parameters
        ----------
        channel : int
            1-indexed channel number.
        pulse_idx : int
            Sequential pulse index (1-based, from PulseCapture).
        pulse_data : dict
            Must contain ``Amp_I``, ``Amp_Q``, ``Time`` ndarrays and
            optionally ``pileup`` bool.
        noise_stats : ChannelNoiseStats, optional
            If provided, peak amplitude and SNR are computed relative
            to the noise baseline.
        """
        if noise_stats is None:
            noise_stats = self._noise_stats.get(channel)
        self._append_pulse_to(f"channel_{channel}", pulse_idx, pulse_data,
                              noise_stats)

    def read_pulse(self, channel: int, pulse_idx: int) -> Optional[dict]:
        """Read back a previously appended pulse through the open write
        handle.

        Used for live browsing while a capture is running: the file is
        flushed after every append, and reading through the same handle
        involves no file locking.  Must be called from the same thread
        that writes (the h5py single-thread rule).
        """
        return self._read_pulse_at(f"channel_{channel}/pulse_{pulse_idx:06d}")

    def update_noise_stats(
        self, noise_stats: Dict[int, ChannelNoiseStats],
    ) -> None:
        """Refresh per-channel noise attributes after a re-estimation.

        Later pulses' derived attrs use the new statistics; the channel
        group attrs always reflect the most recent estimate.
        """
        self._noise_stats.update(noise_stats)
        self._set_noise_stats(lambda ch: f"channel_{ch}", noise_stats)

    def update_histograms(self, histogram_data: Dict[str, np.ndarray]) -> None:
        """Overwrite histogram datasets with current running histograms.

        Parameters
        ----------
        histogram_data : dict[str, ndarray]
            Flat dict of histogram arrays keyed by descriptive names
            (e.g. ``"amplitude_bins"``, ``"amplitude_counts_ch1"``).
        """
        self._replace_datasets("histograms", histogram_data)

    def update_templates(self, template_data: Dict[str, np.ndarray]) -> None:
        """Overwrite the trigger-aligned template datasets."""
        self._replace_datasets("templates", template_data)


# ───────────────────────── Dual-stream writer ───────────────────────

class DualPulseHDF5Writer(_PulseFileWriter):
    """One HDF5 file for a concurrent slow+fast ("both") capture.

    Layout::

        metadata/                     (streamer_mode="both", rates, ...)
        slow/channel_<n>/pulse_*      fast/channel_<n>/pulse_*
        matched/channel_<n>/pair_*    (slow_idx/fast_idx, -1 = one-sided;
                                       optional cross-stream TOD datasets)
        histograms/slow/  histograms/fast/
    """

    STREAMS = ("slow", "fast")

    def __init__(self, path, channels: List[int],
                 capture_params: Dict[str, Any],
                 df_calibrations: Optional[Dict[int, float]] = None,
                 stored_units: Optional[Dict[int, str]] = None):
        super().__init__(path, channels, capture_params)
        self._noise: Dict[str, Dict[int, ChannelNoiseStats]] = {
            s: {} for s in self.STREAMS}
        # Layout invariants, not capture parameters: this file holds two
        # streams however it was configured.
        self.f["metadata"].attrs["layout"] = "dual"
        self.f["metadata"].attrs["streamer_mode"] = "both"

        for stream in self.STREAMS:
            sgrp = self.f.create_group(stream)
            for ch in channels:
                grp = sgrp.create_group(f"channel_{ch}")
                grp.attrs["pulse_count"] = 0
                _store_df_calibration(grp, df_calibrations, ch)
                _store_units(grp, stored_units, ch)
            self.f.create_group(f"histograms/{stream}")

        matched = self.f.create_group("matched")
        for ch in channels:
            mgrp = matched.create_group(f"channel_{ch}")
            mgrp.attrs["pair_count"] = 0
        self.f.flush()

    def set_noise_stats(self, stream: str,
                        noise_stats: Dict[int, ChannelNoiseStats]) -> None:
        self._noise[stream].update(noise_stats)
        self._set_noise_stats(lambda ch: f"{stream}/channel_{ch}",
                              noise_stats)

    def append_pulse(self, stream: str, channel: int, pulse_idx: int,
                     pulse_data: dict) -> None:
        self._append_pulse_to(f"{stream}/channel_{channel}", pulse_idx,
                              pulse_data, self._noise[stream].get(channel))

    def append_match(self, channel: int, pair: dict) -> None:
        if not self.is_open:
            return
        key = f"matched/channel_{channel}"
        if key not in self.f:
            return
        mgrp = self.f[key]
        pair_idx = int(pair["pair_idx"])
        pg = mgrp.create_group(f"pair_{pair_idx:06d}")
        pg.attrs["slow_idx"] = (pair["slow_idx"]
                                if pair["slow_idx"] is not None else -1)
        pg.attrs["fast_idx"] = (pair["fast_idx"]
                                if pair["fast_idx"] is not None else -1)
        pg.attrs["time_offset"] = (
            float(pair["time_offset"])
            if pair.get("time_offset") is not None else float("nan"))
        for side in ("slow_tod", "fast_tod"):
            tod = pair.get(side)
            if tod:
                for name in ("Amp_I", "Amp_Q", "Time"):
                    pg.create_dataset(
                        f"{side}_{name}",
                        data=np.asarray(tod[name], dtype=np.float64),
                        compression="gzip", compression_opts=1)
        mgrp.attrs["pair_count"] = max(
            int(mgrp.attrs.get("pair_count", 0)), pair_idx)
        self.f.flush()

    def read_pulse(self, stream: str, channel: int,
                   pulse_idx: int) -> Optional[dict]:
        """Live read-back through the open write handle (writer thread)."""
        return self._read_pulse_at(
            f"{stream}/channel_{channel}/pulse_{pulse_idx:06d}")

    def update_histograms(self, stream: str,
                          histogram_data: Dict[str, np.ndarray]) -> None:
        self._replace_datasets(f"histograms/{stream}", histogram_data)

    def update_templates(self, stream: str,
                         template_data: Dict[str, np.ndarray]) -> None:
        self._replace_datasets(f"templates/{stream}", template_data)


# ───────────────────────── Reader ───────────────────────────────────

class PulseHDF5Reader:
    """Lazy reader for pulse capture HDF5 files.

    Reads capture metadata and channel info eagerly on open.  Pulse
    waveform data is loaded only when :meth:`get_pulse` is called,
    keeping memory usage low for large files.

    Supports context manager protocol::

        with PulseHDF5Reader("capture.h5") as reader:
            pulse = reader.get_pulse(1, 42)
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.f: Optional[h5py.File] = h5py.File(self.path, "r")

        # Eagerly read metadata
        meta = self.f["metadata"]
        self.metadata: Dict[str, Any] = dict(meta.attrs)
        self.channels: List[int] = list(self.metadata.get("channels", []))
        #: True for dual-layout ("both" mode) files
        self.dual: bool = "slow" in self.f and "fast" in self.f
        self.streams: List[str] = ["slow", "fast"] if self.dual else []

    def _ch_key(self, channel: int, stream: Optional[str]) -> str:
        if self.dual:
            return f"{stream or 'slow'}/channel_{channel}"
        return f"channel_{channel}"

    # ── Channel-level queries ─────────────────────────────────────

    def pulse_count(self, channel: int,
                    stream: Optional[str] = None) -> int:
        """Return the number of pulses stored for *channel*."""
        key = self._ch_key(channel, stream)
        if self.f is not None and key in self.f:
            return int(self.f[key].attrs.get("pulse_count", 0))
        return 0

    def noise_stats(self, channel: int,
                    stream: Optional[str] = None) -> ChannelNoiseStats:
        """Return the noise statistics stored for *channel*."""
        if self.f is None:
            return ChannelNoiseStats()
        grp = self.f.get(self._ch_key(channel, stream))
        if grp is None or "noise_mean_I" not in grp.attrs:
            return ChannelNoiseStats()
        return ChannelNoiseStats(
            mean_I=float(grp.attrs["noise_mean_I"]),
            std_I=float(grp.attrs["noise_std_I"]),
            mean_Q=float(grp.attrs["noise_mean_Q"]),
            std_Q=float(grp.attrs["noise_std_Q"]),
            jump_std_I=float(grp.attrs.get("noise_jump_std_I", 0.0)),
            jump_std_Q=float(grp.attrs.get("noise_jump_std_Q", 0.0)),
        )

    def volts_per_count(self) -> Optional[float]:
        """The counts-to-volts constant *this file* was written with.

        Read from the file's metadata, not from the library: it is what
        the capture was converted with, which is what makes the samples
        interpretable without knowing which version wrote them.
        """
        if self.f is None:
            return None
        meta = self.f.get("metadata")
        return None if meta is None else meta.attrs.get("volts_per_count")

    def stored_units(self, channel: int,
                     stream: Optional[str] = None) -> str:
        """Units of *channel*'s stored samples: ``"Hz"`` or ``"V"``.

        ``"counts"`` for files written before samples were stored in
        physical units, which is what those actually hold.
        """
        if self.f is None:
            return "counts"
        grp = self.f.get(self._ch_key(channel, stream))
        if grp is None:
            return "counts"
        return str(grp.attrs.get("stored_units", "counts"))

    def trigger_basis(self) -> str:
        """Basis the samples were triggered on, and are stored in.

        ``"iq"`` or ``"df"``.  Files written before this was recorded
        predate the rotation and are therefore I/Q.
        """
        if self.f is None:
            return "iq"
        meta = self.f.get("metadata")
        if meta is None:
            return "iq"
        return str(meta.attrs.get("trigger_basis", "iq"))

    def df_calibration(self, channel: int,
                       stream: Optional[str] = None) -> Optional[float]:
        """Return the df calibration for *channel*, or None."""
        if self.f is None:
            return None
        grp = self.f.get(self._ch_key(channel, stream))
        if grp is None:
            return None
        return grp.attrs.get("df_calibration")

    # ── Pulse-level queries ───────────────────────────────────────

    def get_pulse(self, channel: int, pulse_idx: int,
                  stream: Optional[str] = None) -> Optional[dict]:
        """Load a single pulse's waveform data and metadata.

        Returns a dict with keys: ``Amp_I``, ``Amp_Q``, ``Time``,
        ``pileup``, ``peak_I``, ``peak_Q``, ``peak_snr_I``,
        ``peak_snr_Q``, ``n_samples``, ``duration_s``, ``timestamp``.
        Scalars are as :func:`pulse_summary` computed them at capture
        time, so ``duration_s`` is the time above threshold rather than
        the span of the saved window.  Returns ``None`` if the pulse
        doesn't exist.
        """
        if self.f is None:
            return None
        key = f"{self._ch_key(channel, stream)}/pulse_{pulse_idx:06d}"
        if key not in self.f:
            return None
        return _pulse_dict_from_group(self.f[key])

    def get_pulse_metadata(
        self, channel: int, pulse_idx: int,
        stream: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Read only scalar attributes (no waveform data) for a pulse.

        Much faster than :meth:`get_pulse` for tree population.
        """
        if self.f is None:
            return None
        key = f"{self._ch_key(channel, stream)}/pulse_{pulse_idx:06d}"
        if key not in self.f:
            return None
        grp = self.f[key]
        return {k: _convert_attr(grp.attrs[k]) for k in grp.attrs}

    def iter_pulse_metadata(
        self, channel: int, stream: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield metadata dicts for all pulses in *channel*.

        Iterates in order from pulse 1 to pulse_count.  Each dict
        includes a ``"pulse_idx"`` key.  No waveform data is loaded.
        """
        count = self.pulse_count(channel, stream)
        for idx in range(1, count + 1):
            meta = self.get_pulse_metadata(channel, idx, stream)
            if meta is not None:
                meta["pulse_idx"] = idx
                yield meta

    # ── Matched pairs (dual files) ────────────────────────────────

    def pair_count(self, channel: int) -> int:
        key = f"matched/channel_{channel}"
        if self.f is not None and key in self.f:
            return int(self.f[key].attrs.get("pair_count", 0))
        return 0

    def get_match(self, channel: int,
                  pair_idx: int) -> Optional[Dict[str, Any]]:
        """One matched pair: indices (None = one-sided), time offset,
        and any stored cross-stream TOD windows."""
        if self.f is None:
            return None
        key = f"matched/channel_{channel}/pair_{pair_idx:06d}"
        if key not in self.f:
            return None
        pg = self.f[key]
        slow_idx = int(pg.attrs.get("slow_idx", -1))
        fast_idx = int(pg.attrs.get("fast_idx", -1))
        pair: Dict[str, Any] = {
            "pair_idx": pair_idx,
            "channel": channel,
            "slow_idx": slow_idx if slow_idx >= 0 else None,
            "fast_idx": fast_idx if fast_idx >= 0 else None,
            "time_offset": float(pg.attrs.get("time_offset",
                                              float("nan"))),
        }
        for side in ("slow_tod", "fast_tod"):
            if f"{side}_Amp_I" in pg:
                pair[side] = {
                    "Amp_I": np.array(pg[f"{side}_Amp_I"]),
                    "Amp_Q": np.array(pg[f"{side}_Amp_Q"]),
                    "Time": np.array(pg[f"{side}_Time"]),
                }
        return pair

    def iter_matches(self, channel: int) -> Iterator[Dict[str, Any]]:
        for idx in range(1, self.pair_count(channel) + 1):
            pair = self.get_match(channel, idx)
            if pair is not None:
                yield pair

    # ── Histograms ────────────────────────────────────────────────

    def get_histograms(
            self, stream: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Read all histogram datasets (per stream for dual files)."""
        if self.f is None:
            return {}
        key = (f"histograms/{stream or 'slow'}" if self.dual
               else "histograms")
        hist_grp = self.f.get(key)
        if hist_grp is None:
            return {}
        return {k: np.array(hist_grp[k]) for k in hist_grp
                if not isinstance(hist_grp[k], h5py.Group)}

    def get_templates(
            self, stream: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Read the trigger-aligned template datasets."""
        if self.f is None:
            return {}
        key = (f"templates/{stream or 'slow'}" if self.dual
               else "templates")
        grp = self.f.get(key)
        if grp is None:
            return {}
        return {k: np.array(grp[k]) for k in grp
                if not isinstance(grp[k], h5py.Group)}

    # ── Lifecycle ─────────────────────────────────────────────────

    def close(self) -> None:
        if self.f is not None and self.f.id.valid:
            self.f.close()
        self.f = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ── Helpers ───────────────────────────────────────────────────────

def _write_pulse(channel_grp, pulse_idx: int, pulse_data: dict,
                 noise_stats: Optional[ChannelNoiseStats],
                 threshold_sigma: Optional[float]) -> None:
    """Write one pulse group (datasets + derived attrs) under a channel
    group — shared by the single-stream and dual-stream writers."""
    pulse_grp = channel_grp.create_group(f"pulse_{pulse_idx:06d}")

    amp_I = np.asarray(pulse_data["Amp_I"], dtype=np.float64)
    amp_Q = np.asarray(pulse_data["Amp_Q"], dtype=np.float64)
    time_arr = np.asarray(pulse_data["Time"], dtype=np.float64)

    for name, arr in (("Amp_I", amp_I), ("Amp_Q", amp_Q),
                      ("Time", time_arr)):
        pulse_grp.create_dataset(name, data=arr, compression="gzip",
                                 compression_opts=1)

    pulse_grp.attrs["pileup"] = bool(pulse_data.get("pileup", False))
    pulse_grp.attrs["truncated"] = bool(pulse_data.get("truncated", False))
    pulse_grp.attrs["n_samples"] = len(amp_I)

    # Where the engine triggered and where the end condition confirmed
    # the end — kept so a saved capture can be reviewed against the
    # decisions that produced it, not just its samples.
    for key in ("trigger_index", "end_index", "below_threshold_index",
                "end_confirm_samples", "end_confirm_target"):
        if key in pulse_data:
            pulse_grp.attrs[key] = int(pulse_data[key])
    for key in ("trigger_time", "end_time", "below_threshold_time"):
        if key in pulse_data:
            pulse_grp.attrs[key] = float(pulse_data[key])

    # Every scalar below comes from pulse_summary(), the same call the
    # histograms, the live on_pulse callback and the GUI derive from.
    # Computing any of them a second time here is how a capture file
    # ends up disagreeing with itself: duration_s was the span of the
    # saved window while the duration_ms histogram beside it measured
    # trigger -> below-threshold, so one pulse read back as 4.72 ms or
    # 3.09 ms depending on which you asked.  Under save_to_end_confirmed
    # the window also carries however long the end confirmation took to be
    # satisfied, which is a property of the baseline, not the event.
    summary = pulse_summary(pulse_data, noise_stats, threshold_sigma)
    for key in ("peak_I", "peak_Q", "peak_amp", "snr", "duration_s",
                "timestamp", "tau_s"):
        pulse_grp.attrs[key] = float(summary[key])

    # Per-quadrature SNRs have no summary equivalent - they are for
    # reviewing a pulse, not for binning it.
    if noise_stats is not None:
        pulse_grp.attrs["peak_snr_I"] = summary["peak_I"] / max(
            noise_stats.std_I, 1e-30)
        pulse_grp.attrs["peak_snr_Q"] = summary["peak_Q"] / max(
            noise_stats.std_Q, 1e-30)
    else:
        pulse_grp.attrs["peak_snr_I"] = 0.0
        pulse_grp.attrs["peak_snr_Q"] = 0.0


def _pulse_dict_from_group(grp) -> dict:
    """Waveforms + scalar attrs for one pulse group (reader/writer shared)."""
    marks = {k: _convert_attr(grp.attrs[k])
             for k in ("trigger_index", "end_index", "below_threshold_index",
                       "end_confirm_samples", "end_confirm_target",
                       "trigger_time", "end_time", "below_threshold_time")
             if k in grp.attrs}
    return {
        **marks,
        "Amp_I": np.array(grp["Amp_I"]),
        "Amp_Q": np.array(grp["Amp_Q"]),
        "Time": np.array(grp["Time"]),
        "pileup": bool(grp.attrs.get("pileup", False)),
        "truncated": bool(grp.attrs.get("truncated", False)),
        "peak_I": float(grp.attrs.get("peak_I", 0)),
        "peak_Q": float(grp.attrs.get("peak_Q", 0)),
        "peak_snr_I": float(grp.attrs.get("peak_snr_I", 0)),
        "peak_snr_Q": float(grp.attrs.get("peak_snr_Q", 0)),
        "n_samples": int(grp.attrs.get("n_samples", 0)),
        "duration_s": float(grp.attrs.get("duration_s", 0)),
        "timestamp": float(grp.attrs.get("timestamp", 0)),
        "peak_amp": float(grp.attrs.get("peak_amp", 0)),
        "snr": float(grp.attrs.get("snr", 0)),
        "tau_s": float(grp.attrs.get("tau_s", float("nan"))),
    }


def _convert_attr(val: Any) -> Any:
    """Convert HDF5 attribute values to native Python types."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return val
