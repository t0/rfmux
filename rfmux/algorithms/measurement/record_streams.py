"""
Record the slow stream (a pulse capture and a parser dirfile) and the
channel stream (a fastrx recording) of a module, or of several feeding
one RF line, together into one session folder.  ``rfmux record`` is the
command-line front.

The board is read, and its channel streamer turned on for the modules
only when asked (``channel_streamer=True``); configure the slow
streamer first.  The parser
process is brought up first (it takes seconds to import), then the
capture starts; it spends its noise-training span before it detects
anything, so the fastrx writer starts when that span ends and runs for
the capture's duration.  The capture and the recording then cover the
same stretch, and the dirfile that stretch plus the training span.
Without the capture the recording starts as soon as the parser is up.

Products go into a Periscope session folder (rfmux.core.session_folder:
``session_YYYYMMDD_HHMMSS/`` holding ``<type>_module<M>_HHMMSS.<ext>``
and the metadata that lists them), so an existing session takes them
alongside its bias export, which is where the channels and df
calibrations come from by default::

    from rfmux.core.session_folder import open_session
    result = await record_streams(
        crs, module=2, channels=range(1, 89), duration_s=20.0,
        session=open_session("~/data/session_20260909_153654"))
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import importlib.util
import os
import pickle
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ... import streamer
from ...core.transferfunctions import (PFB_SAMPLING_FREQ,
                                       decimation_to_sampling)
from ...pulse_capture.capture_session import PulseCaptureConfig
from ...core.session_folder import (is_session, latest_export, load_metadata,
                             register_export, save_metadata)
from ...core.channels import (MAX_MODULE, format_channel_spec,
                              parse_channel_spec, parse_module_channels)
from ...pulse_capture.channel_keys import (describe, keys_by_module,
                                           pair_keys)
from .df_calibration import tuning_rows

PARSER_EXIT_S = 10.0


def fastrx_bytes_per_s(fx, channels: int) -> float:
    """Disk rate of a recording of channels 1 to *channels*: the record
    stride, as the fastrx module *fx* lays a record out, at the
    channel-stream rate."""
    return float(fx.record_stride(channels) * PFB_SAMPLING_FREQ)


def interface_speeds() -> dict:
    """{interface: negotiated Mb/s} for the host's interfaces, without
    loopback; None for one without a link or a reported speed."""
    speeds = {}
    try:
        names = sorted(n for n in os.listdir("/sys/class/net") if n != "lo")
    except OSError:
        return speeds
    for name in names:
        try:
            speed = int(Path("/sys/class/net", name, "speed").read_text())
        except (OSError, ValueError):
            speed = None
        speeds[name] = speed if speed and speed > 0 else None
    return speeds


#: The parser as a child: it says so on stderr once imported, which is
#: seconds after launch; nothing else starts before that.
PARSER_CHILD = ("import sys; from rfmux.tools import parser; "
                "print('parser up', file=sys.stderr, flush=True); "
                "sys.exit(parser.main(*sys.argv[1:]))")


# ── The session folder ─────────────────────────────────────────────

def latest_bias_export(session: Path, module: int) -> Optional[Path]:
    """The newest Bias KIDs export for *module* the session lists."""
    return latest_export(session, "bias", f"module{module}")


def biased_channels(bias_path: Path) -> Tuple[List[int], Dict[int, dict]]:
    """The channels a bias_kids export biased, and the tuning row of each
    (``tuning_rows`` of the export, with its NCO frequency)."""
    with open(bias_path, "rb") as f:
        export = pickle.load(f)
    rows = tuning_rows(export.get("bias_kids_output"),
                       export.get("nco_frequency_hz"))
    return sorted(rows), rows


def resolve_channels(modules: List[int], spec: Optional[str],
                     folder: Optional[Path], bias: Optional[Path] = None,
                     ) -> Tuple[Dict[int, List[int]], Dict[Any, dict], List[str]]:
    """What a run records, from the command line's or the dialog's
    choices: ``(wanted, tuning, notes)``, *wanted* the ``{module:
    channels}``, *tuning* the rows from each module's bias export keyed
    by channel (by (module, channel) across modules), *notes* one line
    per export read.

    *spec* is a range spec for every module of *modules* (``1-88``), a
    per-module spec that names the modules itself (``2:1-114,3:1-96``),
    or None for the biased channels of each module's newest bias export
    in *folder*; *bias* names one module's export instead of the newest.
    Raises ValueError for anything that cannot be recorded.
    """
    wanted: Dict[int, List[int]] = {}
    ranges = None
    if spec and ":" in spec:
        wanted = parse_module_channels(spec, max_module=MAX_MODULE,
                                       max_channel=MAX_CHANNEL)
        modules = list(wanted)
    elif spec:
        ranges = parse_channel_spec(spec, max_value=MAX_CHANNEL, wildcard=False)
    for m in modules:
        if not 1 <= int(m) <= MAX_MODULE:
            raise ValueError(f"module {m}: modules run 1-{MAX_MODULE}")
    if bias is not None and len(modules) > 1:
        raise ValueError("--bias names one module's export; several modules "
                         "take the session's newest export each")
    multi = len(modules) > 1
    tuning: Dict[Any, dict] = {}
    notes = []
    for module in modules:
        bias_path = (Path(bias) if bias is not None else
                     latest_bias_export(folder, module) if folder else None)
        biased, rows = biased_channels(bias_path) if bias_path else ([], {})
        chosen = wanted.get(module) or ranges or biased
        if not chosen:
            raise ValueError(f"no channels given, and no bias export for "
                             f"module {module} in the session to take them "
                             "from")
        wanted[module] = chosen
        tuning.update({(module, c) if multi else c: row
                       for c, row in rows.items()})
        if bias_path:
            notes.append(f"{bias_path.name}: {len(biased)} channels, "
                         f"{calibrated(rows)} calibrated")
    return wanted, tuning, notes


def calibrated(tuning: Dict[Any, dict]) -> int:
    """How many rows carry a df calibration."""
    return sum(1 for r in tuning.values()
               if isinstance(r, dict) and r.get("df_calibration") is not None)


def by_module(module: Optional[int], channels) -> Dict[int, List[int]]:
    """``{module: sorted channels}`` from a channel list on *module* or
    a mapping of them; modules with no channels are dropped."""
    keys = pair_keys(channels) if isinstance(channels, dict) else channels
    return {m: sorted({c for c, _ in pairs})
            for m, pairs in keys_by_module(keys, module).items()}


def modules_tag(modules: Iterable[int]) -> str:
    """``module2`` or ``modules2+3``: the module part of a product's
    name."""
    modules = list(modules)
    if len(modules) == 1:
        return f"module{modules[0]}"
    return "modules" + "+".join(str(m) for m in modules)


# ── The recording ──────────────────────────────────────────────────

@dataclass
class _Parser:
    proc: object
    ready: asyncio.Event          # set once up, or once exited
    pump: Optional[asyncio.Task]
    up: bool = False


@dataclass
class RecordResult:
    session: Path
    #: The one module recorded, or None for a run across modules, whose
    #: ``channels`` are then (module, channel) pairs.
    module: Optional[int]
    channels: List
    duration_s: float
    training_s: float
    modules: List[int] = field(default_factory=list)
    #: When the recording window opened: the end of noise training, or
    #: once the parser was up without a capture.
    started_at: Optional[float] = None
    pulse_path: Optional[Path] = None
    #: The parser's subdirfile for the board, the path the viewer takes.
    dirfile_path: Optional[Path] = None
    parser_log: Optional[Path] = None
    fastrx_path: Optional[Path] = None
    fastrx_stats: Dict[str, int] = field(default_factory=dict)
    #: The recording was merged into the pulse file as its fast stream.
    merged_fastrx: bool = False
    capture: object = None
    warnings: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        parts = [p.name for p in (self.pulse_path, self.dirfile_path,
                                  self.fastrx_path) if p is not None]
        return (f"RecordResult({self.session.name}: {', '.join(parts)}; "
                f"{len(self.warnings)} warnings)")


async def record_streams(
    crs,
    module: Optional[int],
    channels: Union[Iterable[int], Dict[int, Iterable[int]]],
    duration_s: float,
    *,
    session: Path,
    capture: bool = True,
    parser: bool = True,
    fastrx: bool = True,
    config: Optional[PulseCaptureConfig] = None,
    tuning: Optional[Dict[int, dict]] = None,
    trigger_basis: Optional[str] = None,
    parser_interface: Optional[str] = None,
    fastrx_interface: Optional[str] = None,
    fastrx_socket: Optional[str] = None,
    merge_fastrx: bool = True,
    channel_streamer: bool = False,
    sample_trunc: str = "LOW",
    verbose: bool = True,
) -> RecordResult:
    """Record the selected products of *channels* on *module*, or of
    ``{module: channels}`` across modules, for *duration_s* into
    *session*.  A run across modules is keyed by (module, channel)
    pairs in the capture, the file and *tuning*; its products are named
    ``modules2+3``.

    ``capture`` runs ``crs.trigger_capture`` on the slow stream with
    *config*, *tuning* and *trigger_basis*; ``parser`` runs
    ``rfmux parser`` as a subprocess on the board's 1G traffic
    (*parser_interface* overrides the interface it finds from the
    board's address); ``fastrx`` records channels 1 to the highest of
    *channels*
    through the running fastrxd (*fastrx_interface* or *fastrx_socket*
    name it when several run).  Each requirement is checked before
    anything starts.
    """
    wanted = by_module(module, channels)
    if not wanted:
        raise ValueError("no channels to record")
    modules = list(wanted)
    if len(modules) == 1:
        module = modules[0]
        channels = list(wanted[module])
    else:
        module = None
        channels = pair_keys(wanted)
    if duration_s <= 0:
        raise ValueError(f"duration must be positive, got {duration_s}")
    if not (capture or parser or fastrx):
        raise ValueError("nothing selected to record")
    session = Path(session)
    if not is_session(session):
        raise ValueError(f"{session} is not a session folder; open_session() makes one")
    config = config or PulseCaptureConfig()
    say = print if verbose else (lambda *a, **k: None)

    # Requirements, before the board is touched or a file written.
    if parser and importlib.util.find_spec("pygetdata") is None:
        raise RuntimeError("the parser dirfile needs pygetdata: "
                           "uv pip install -e .[dirfile]")
    fastrx_channels = 0
    fx = None
    if fastrx:
        try:
            from ... import fastrx as fx
        except ImportError as e:
            raise RuntimeError(f"fastrx is not built in this rfmux: {e}") from e
        fastrx_socket = fx.resolve_socket(fastrx_interface, fastrx_socket)
        if not Path(fastrx_socket).exists():
            raise RuntimeError(
                f"no fastrxd socket at {fastrx_socket}: is fastrxd running? "
                "Start it with: "
                + fx.start_command(Path(fastrx_socket).name))
        fastrx_channels = max(c for chs in wanted.values() for c in chs)
        if channel_streamer:
            await _enable_channel_streamer(crs, modules, fastrx_channels,
                                           fx.MAX_SAMPLES,
                                           sample_trunc, say)
        silent = await asyncio.to_thread(
            _fastrx_silent_modules, fx, fastrx_socket, modules)
        if silent:
            raise RuntimeError(
                f"no channel-stream packets from module(s) {silent}: is "
                "the channel streamer on for them?")

    result = RecordResult(session=session, module=module, channels=channels,
                          duration_s=float(duration_s), training_s=0.0,
                          modules=modules)
    if capture:
        dec = await crs.get_decimation()
        rate = decimation_to_sampling(6 if dec is None else dec)
        result.training_s = config.noise_samples(rate) / rate
    if fastrx:
        need = duration_s * fastrx_bytes_per_s(fx, fastrx_channels)
        free = shutil.disk_usage(session).free
        if free < need:
            result.warnings.append(
                f"{free / 1e9:.0f} GB free in {session} for a recording "
                f"of about {need / 1e9:.0f} GB")

    stamp = datetime.datetime.now().strftime("%H%M%S")
    host = streamer.resolve_host(crs.tuber_hostname)

    def name(kind: str, ext: str) -> Path:
        return session / f"{kind}_{modules_tag(modules)}_{stamp}{ext}"

    started = asyncio.Event()
    stop = asyncio.Event()
    trained = False

    def on_noise(*_):
        nonlocal trained
        trained = True
        started.set()

    async def run_capture():
        try:
            result.pulse_path = name("pulse", ".h5")
            result.capture = await crs.trigger_capture(
                channel=wanted if module is None else channels,
                module=module, streamer_mode="slow",
                time_run=duration_s, config=config,
                hdf5_path=result.pulse_path,
                tuning=tuning,
                trigger_basis=trigger_basis, on_noise=on_noise,
                verbose=verbose)
        finally:
            started.set()      # a capture that dies never trains

    async def run_side():
        await started.wait()
        if capture and not trained:
            result.warnings.append(
                "the capture ended before noise training completed; "
                "nothing was recorded")
            return
        result.started_at = time.time()
        writer = None
        try:
            if fastrx:
                result.fastrx_path = name("fastrx", ".fastrx")
                writer = fx.PacketWriter(
                    result.fastrx_path, channels=fastrx_channels,
                    socket=fastrx_socket)
            say(f"[record] recording for {duration_s:.1f} s")
            await _hold(duration_s, stop, writer)
        finally:
            if writer is not None:
                await asyncio.to_thread(writer.stop)
                result.fastrx_stats = {
                    "packets": writer.packets, "overruns": writer.overruns,
                    "dropouts": writer.dropouts}
                if writer.packets == 0:
                    result.warnings.append(
                        "no channel-stream packets: is the channel "
                        f"streamer on for module(s) {modules}?")

    handle = None
    tasks: List[asyncio.Task] = []
    try:
        if parser:
            result.parser_log = name("parser", ".log")
            handle = await _start_parser(
                host, parser_interface, wanted,
                name("parser", ".dirfile"), result.parser_log)
            await handle.ready.wait()
            if not handle.up:
                await handle.proc.wait()
                raise RuntimeError("the parser exited before it was up: "
                                   + _log_tail(result.parser_log))
        if not capture:
            started.set()
            trained = True
        else:
            tasks.append(asyncio.ensure_future(run_capture()))
        tasks.append(asyncio.ensure_future(run_side()))
        # A failure in one ends the other: the recording through the
        # stop event, so its cleanup runs whole; the capture by cancel,
        # which closes its file.
        try:
            done, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION)
            if any(t.exception() for t in done if not t.cancelled()):
                stop.set()
                for t in tasks[:-1]:
                    t.cancel()
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            raise
        finally:
            await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if handle is not None:
            await _stop_parser(handle, result, name("parser", ".dirfile"),
                               result.parser_log)
        if merge_fastrx and tasks and all(
                t.done() and not t.cancelled() and t.exception() is None
                for t in tasks):
            await asyncio.to_thread(_merge_recording, result)
        _record(result, config)
    # The one that failed first raised; the other was cancelled to end
    # the run, and its cancellation is not the error.
    for t in tasks:
        if not t.cancelled():
            t.result()
    say(f"[record] {result!r}")
    return result


async def _hold(duration_s: float, stop: asyncio.Event, writer) -> None:
    """The recording window: *duration_s*, or until *stop*; a writer's
    failure surfaces within half a second."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + duration_s
    while not stop.is_set():
        remaining = deadline - loop.time()
        if remaining <= 0:
            return
        if writer is not None:
            await asyncio.to_thread(writer.wait, min(remaining, 0.5))
        else:
            try:
                await asyncio.wait_for(stop.wait(), remaining)
            except asyncio.TimeoutError:
                return


def _log_tail(log: Path) -> str:
    try:
        return " | ".join(Path(log).read_text().strip().splitlines()[-3:])
    except OSError:
        return ""


async def _start_parser(host, interface, wanted, dirfile, log) -> _Parser:
    """The parser on the channels of each module of *wanted*."""
    where = ["-i", interface] if interface else ["-H", host]
    cmd = [sys.executable, "-c", PARSER_CHILD, *where, "-d", str(dirfile)]
    for module, channels in wanted.items():
        cmd += ["-c", f"{module}:{format_channel_spec(channels)}"]
    cmd.append("--drop-stats")
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE)
    handle = _Parser(proc, asyncio.Event(), None)

    async def pump():
        with open(log, "wb") as f:
            while line := await proc.stderr.readline():
                f.write(line)
                f.flush()
                if line.startswith(b"parser up"):
                    handle.up = True
                    handle.ready.set()
        handle.ready.set()

    handle.pump = asyncio.ensure_future(pump())
    return handle


async def _stop_parser(handle: _Parser, result, dirfile, log):
    proc = handle.proc
    if proc.returncode is None:
        proc.send_signal(signal.SIGINT)
        try:
            await asyncio.wait_for(proc.wait(), PARSER_EXIT_S)
        except asyncio.TimeoutError:
            proc.terminate()
            await proc.wait()
    if handle.pump is not None:
        await handle.pump
    subdirs = sorted(Path(dirfile).glob("serial_*"))
    if subdirs:
        result.dirfile_path = subdirs[0]
    else:
        tail = _log_tail(log)
        result.warnings.append(
            "the parser wrote no dirfile" + (": " + tail if tail else ""))


#: Seconds for a freshly enabled channel streamer to flow before the
#: stream is probed.
CHANNEL_STREAMER_SETTLE_S = 1.0
#: Channels a module has, the most a range spec may name.
MAX_CHANNEL = 1024
#: Seconds a fastrx probe waits for the channel stream before a run.
FASTRX_PROBE_S = 1.0


async def _enable_channel_streamer(crs, modules: List[int], channels: int,
                                   pipeline: int, sample_trunc: str,
                                   say) -> None:
    """Turn the channel streamer on for *modules*, channels 1 to
    *channels* as the recording keeps them, rounded up to whole
    pipelines of *pipeline* channels: fastrxd drops a packet whose
    pipelines are not all full.  Then let it flow before the stream is
    probed."""
    if not hasattr(crs, "set_channel_streamer"):
        raise RuntimeError("this board has no channel streamer to turn on")
    channels = -(-channels // pipeline) * pipeline
    for m in modules:
        say(f"[record] channel streamer on for module {m}: channels "
            f"1-{channels}, {sample_trunc} bits")
        await crs.set_channel_streamer(channels=channels, module=m,
                                       sample_trunc=sample_trunc)
    await asyncio.sleep(CHANNEL_STREAMER_SETTLE_S)


def _fastrx_silent_modules(fx, socket: str, modules: List[int],
                           timeout_s: float = FASTRX_PROBE_S) -> List[int]:
    """The wanted *modules* the channel stream did not carry during one
    short capture, whose ``modules_seen`` counts every module's packets
    (the window is the capture of a few thousand of one module's, or
    the whole timeout when that module is silent)."""
    with fx.PacketCapture(socket=socket) as c:
        seen = c.capture(4096, channels=1, module=modules[0],
                         timeout=timeout_s)["modules_seen"]
    return [m for m in modules if not seen & (1 << (m - 1))]


def _merge(pulse_path: Path, fastrx_path: Path) -> None:
    from ...pulse_capture.overlay import merge_fastrx
    merge_fastrx(pulse_path, fastrx_path)


def _merge_recording(result: RecordResult) -> None:
    """The fastrx recording into the pulse file as its fast stream; a
    merge that fails is a warning, the run itself having succeeded."""
    if not (result.pulse_path and result.pulse_path.exists()
            and result.fastrx_path and result.fastrx_path.exists()):
        return
    try:
        _merge(result.pulse_path, result.fastrx_path)
        result.merged_fastrx = True
    except Exception as e:
        result.warnings.append(
            f"fastrx not merged into {result.pulse_path.name}: {e}")


def pulse_summary_lines(capture) -> List[str]:
    """One line per channel that triggered, most pulses first, then the
    total; from a trigger_capture result."""
    stream = getattr(capture, "primary", None)
    if stream is None:
        return []
    rows = [(ch, len(by_idx), max(s.get("snr", 0.0) for s in by_idx.values()))
            for ch, by_idx in stream.summaries.items() if by_idx]
    rows.sort(key=lambda r: (-r[1], r[0]))
    lines = [f"{describe(ch)}: {n} pulse{'s' if n != 1 else ''}, "
             f"best {snr:.1f}\u03c3" for ch, n, snr in rows]
    lines.append(f"{sum(r[1] for r in rows)} pulses on {len(rows)} of "
                 f"{len(stream.summaries)} channels")
    return lines


def _record(result: RecordResult, config: PulseCaptureConfig) -> None:
    """The products into the session's exports, and the run into its
    recordings."""
    for kind, path in (("pulse", result.pulse_path),
                       ("parser", result.dirfile_path),
                       ("fastrx", result.fastrx_path)):
        if path is not None and path.exists():
            register_export(result.session, str(path.relative_to(result.session)),
                            kind, modules_tag(result.modules))
    metadata = load_metadata(result.session)
    metadata.setdefault("recordings", []).append({
        "timestamp": datetime.datetime.now().isoformat(),
        "module": result.module,
        "modules": result.modules,
        "channels": [list(c) if isinstance(c, tuple) else c
                     for c in result.channels],
        "duration_s": result.duration_s,
        "training_s": result.training_s,
        "started_at": result.started_at,
        "pulse": result.pulse_path.name if result.pulse_path else None,
        "dirfile": (str(result.dirfile_path.relative_to(result.session))
                    if result.dirfile_path else None),
        "fastrx": result.fastrx_path.name if result.fastrx_path else None,
        "fastrx_stats": result.fastrx_stats,
        "merged_fastrx": result.merged_fastrx,
        "capture_config": dataclasses.asdict(config),
        "warnings": result.warnings,
    })
    save_metadata(result.session, metadata)
