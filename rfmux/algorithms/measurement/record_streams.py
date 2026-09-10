"""
Record one module's slow stream (a pulse capture and a parser dirfile)
and its channel stream (a fastrx recording) together, into one session
folder.  ``rfmux record`` is the command-line front.

The board is only read: configure the streamers first.  The parser
process is brought up first (it takes seconds to import), then the
capture starts; it spends its noise-training span before it detects
anything, so the fastrx writer starts when that span ends and runs for
the capture's duration.  The capture and the recording then cover the
same stretch, and the dirfile that stretch plus the training span.
Without the capture the recording starts as soon as the parser is up.

Products follow the Periscope session convention, one folder
``session_YYYYMMDD_HHMMSS/`` holding ``<type>_module<M>_HHMMSS.<ext>``
and a ``session_metadata.json`` that lists them, so an existing
Periscope session can take them alongside its bias export, which is
where the channels and df calibrations come from by default::

    result = await record_streams(
        crs, module=2, channels=range(1, 89), duration_s=20.0,
        session=open_session("~/data/session_20260909_153654"))
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import importlib.util
import json
import pickle
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ... import streamer
from ...core.transferfunctions import decimation_to_sampling
from ...pulse_capture.capture_session import PulseCaptureConfig
from ...pulse_capture.overlay import channel_location

SESSION_FOLDER_FORMAT = "session_%Y%m%d_%H%M%S"
METADATA_FILE = "session_metadata.json"
FASTRX_BYTES_PER_PIPE_S = 1.5e9
PARSER_EXIT_S = 10.0
#: The parser as a child: it says so on stderr once imported, which is
#: seconds after launch; nothing else starts before that.
PARSER_CHILD = ("import sys; from rfmux.tools import parser; "
                "print('parser up', file=sys.stderr, flush=True); "
                "sys.exit(parser.main(*sys.argv[1:]))")


# ── The session folder ─────────────────────────────────────────────

def open_session(path: Optional[Path] = None,
                 base: Optional[Path] = None) -> Path:
    """The session folder at *path*, or a new one under *base* (the
    working directory), with its metadata file."""
    if path is None:
        stamp = datetime.datetime.now().strftime(SESSION_FOLDER_FORMAT)
        path = Path(base or ".") / stamp
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    if not (path / METADATA_FILE).exists():
        _save_metadata(path, {
            "created": datetime.datetime.now().isoformat(),
            "folder_name": path.name,
            "base_path": str(path.resolve().parent),
            "exports": [],
            "screenshots": [],
        })
    return path


def _load_metadata(session: Path) -> dict:
    try:
        with open(session / METADATA_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_metadata(session: Path, metadata: dict) -> None:
    with open(session / METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def register_export(session: Path, filename: str, data_type: str,
                    identifier: str) -> None:
    """List a file in the session's exports, as Periscope's browser
    expects them."""
    metadata = _load_metadata(session)
    metadata.setdefault("exports", []).append({
        "filename": filename,
        "data_type": data_type,
        "identifier": identifier,
        "timestamp": datetime.datetime.now().isoformat(),
    })
    _save_metadata(session, metadata)


def latest_bias_export(session: Path, module: int) -> Optional[Path]:
    """The newest bias_kids export for *module* in the session, by the
    export's own timestamp (a copied folder keeps no file times)."""
    found = []
    for path in Path(session).glob("bias_*.pkl"):
        with open(path, "rb") as f:
            export = pickle.load(f)
        if export.get("target_module") == module:
            found.append((str(export.get("timestamp", "")),
                          path.stat().st_mtime, path))
    return max(found)[2] if found else None


def biased_channels(bias_path: Path) -> Tuple[List[int], Dict[int, complex]]:
    """The channels a bias_kids export biased, and the df calibration of
    each one that has it."""
    with open(bias_path, "rb") as f:
        export = pickle.load(f)
    results = export.get("bias_kids_output") or {}
    channels = sorted(int(r["bias_channel"]) for r in results.values()
                      if r.get("bias_channel") is not None)
    calibrations = {int(r["bias_channel"]): complex(r["df_calibration"])
                    for r in results.values()
                    if r.get("bias_channel") is not None
                    and r.get("df_calibration") is not None}
    return channels, calibrations


def channel_spec(channels: Iterable[int]) -> str:
    """Channels as the parser's ``-c`` ranges: ``1-4,7``."""
    runs: List[List[int]] = []
    for c in sorted(set(int(c) for c in channels)):
        if runs and c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in runs)


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
    module: int
    channels: List[int]
    duration_s: float
    training_s: float
    #: When the recording window opened: the end of noise training, or
    #: once the parser was up without a capture.
    started_at: Optional[float] = None
    pulse_path: Optional[Path] = None
    #: The parser's subdirfile for the board, the path the viewer takes.
    dirfile_path: Optional[Path] = None
    parser_log: Optional[Path] = None
    fastrx_path: Optional[Path] = None
    fastrx_stats: Dict[str, int] = field(default_factory=dict)
    capture: object = None
    warnings: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        parts = [p.name for p in (self.pulse_path, self.dirfile_path,
                                  self.fastrx_path) if p is not None]
        return (f"RecordResult({self.session.name}: {', '.join(parts)}; "
                f"{len(self.warnings)} warnings)")


async def record_streams(
    crs,
    module: int,
    channels: Iterable[int],
    duration_s: float,
    *,
    session: Path,
    capture: bool = True,
    parser: bool = True,
    fastrx: bool = True,
    config: Optional[PulseCaptureConfig] = None,
    df_calibrations: Optional[Dict[int, complex]] = None,
    trigger_basis: Optional[str] = None,
    parser_interface: Optional[str] = None,
    fastrx_interface: Optional[str] = None,
    fastrx_socket: Optional[str] = None,
    verbose: bool = True,
) -> RecordResult:
    """Record the selected products of *module* for *duration_s* into
    *session*.

    ``capture`` runs ``crs.trigger_capture`` on the slow stream with
    *config*, *df_calibrations* and *trigger_basis*; ``parser`` runs
    ``rfmux parser`` as a subprocess on the board's 1G traffic
    (*parser_interface* overrides the interface it finds from the
    board's address); ``fastrx`` records the pipes carrying *channels*
    through the running fastrxd (*fastrx_interface* or *fastrx_socket*
    name it when several run).  Each requirement is checked before
    anything starts.
    """
    channels = sorted(set(int(c) for c in channels))
    if not channels:
        raise ValueError("no channels to record")
    if duration_s <= 0:
        raise ValueError(f"duration must be positive, got {duration_s}")
    if not (capture or parser or fastrx):
        raise ValueError("nothing selected to record")
    session = Path(session)
    if not (session / METADATA_FILE).exists():
        raise ValueError(f"{session} is not a session folder; open_session() makes one")
    config = config or PulseCaptureConfig()
    say = print if verbose else (lambda *a, **k: None)

    # Requirements, before the board is touched or a file written.
    if parser and importlib.util.find_spec("pygetdata") is None:
        raise RuntimeError("the parser dirfile needs pygetdata: "
                           "uv pip install -e .[dirfile]")
    pipes: List[int] = []
    fx = None
    if fastrx:
        try:
            from ... import fastrx as fx
        except ImportError as e:
            raise RuntimeError(f"fastrx is not built in this rfmux: {e}") from e
        fastrx_socket = fx.resolve_socket(fastrx_interface, fastrx_socket)
        if not Path(fastrx_socket).exists():
            raise RuntimeError(f"no fastrxd socket at {fastrx_socket}: "
                               "is fastrxd running?")
        pipes = sorted({channel_location(c)[0] for c in channels})

    result = RecordResult(session=session, module=module, channels=channels,
                          duration_s=float(duration_s), training_s=0.0)
    if capture:
        dec = await crs.get_decimation()
        rate = decimation_to_sampling(6 if dec is None else dec)
        result.training_s = config.noise_samples(rate) / rate
    if fastrx:
        need = duration_s * FASTRX_BYTES_PER_PIPE_S * len(pipes)
        free = shutil.disk_usage(session).free
        if free < need:
            result.warnings.append(
                f"{free / 1e9:.0f} GB free in {session} for a recording "
                f"of about {need / 1e9:.0f} GB")

    stamp = datetime.datetime.now().strftime("%H%M%S")
    host = streamer.resolve_host(crs.tuber_hostname)

    def name(kind: str, ext: str) -> Path:
        return session / f"{kind}_module{module}_{stamp}{ext}"

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
                channel=channels, module=module, streamer_mode="slow",
                time_run=duration_s, config=config,
                hdf5_path=result.pulse_path,
                df_calibrations=df_calibrations,
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
                    result.fastrx_path, pipes=pipes, socket=fastrx_socket)
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
                        f"no channel-stream packets on pipe(s) {pipes}: "
                        f"is the channel streamer on for module {module}?")

    handle = None
    tasks: List[asyncio.Task] = []
    try:
        if parser:
            result.parser_log = name("parser", ".log")
            handle = await _start_parser(
                host, parser_interface, module, channels,
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


async def _start_parser(host, interface, module, channels, dirfile, log) -> _Parser:
    where = ["-i", interface] if interface else ["-H", host]
    cmd = [sys.executable, "-c", PARSER_CHILD, *where,
           "-d", str(dirfile), "-c", f"{module}:{channel_spec(channels)}",
           "--drop-stats"]
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


def _record(result: RecordResult, config: PulseCaptureConfig) -> None:
    """The products into the session's exports, and the run into its
    recordings."""
    for kind, path in (("pulse", result.pulse_path),
                       ("parser", result.dirfile_path),
                       ("fastrx", result.fastrx_path)):
        if path is not None and path.exists():
            register_export(result.session, str(path.relative_to(result.session)),
                            kind, f"module{result.module}")
    metadata = _load_metadata(result.session)
    metadata.setdefault("recordings", []).append({
        "timestamp": datetime.datetime.now().isoformat(),
        "module": result.module,
        "channels": result.channels,
        "duration_s": result.duration_s,
        "training_s": result.training_s,
        "started_at": result.started_at,
        "pulse": result.pulse_path.name if result.pulse_path else None,
        "dirfile": (str(result.dirfile_path.relative_to(result.session))
                    if result.dirfile_path else None),
        "fastrx": result.fastrx_path.name if result.fastrx_path else None,
        "fastrx_stats": result.fastrx_stats,
        "capture_config": dataclasses.asdict(config),
        "warnings": result.warnings,
    })
    _save_metadata(result.session, metadata)
