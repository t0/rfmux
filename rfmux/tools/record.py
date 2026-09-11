#!/usr/bin/env python3
"""
rfmux record - a pulse capture, a parser dirfile and a fastrx recording
of one module, for the same stretch, into one session folder.

    rfmux record --serial 0156 --module 2 --duration 20 \\
        --session ~/data/session_20260909_153654

The board is only read; configure the streamers first.  With --session
the run joins an existing folder and takes its channels and df
calibrations from the newest bias export there; otherwise a new
session_YYYYMMDD_HHMMSS folder is made under --session-dir.
"""

import asyncio
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import click

from rfmux.algorithms.measurement.record_streams import (
    biased_channels,
    latest_bias_export,
    open_session,
    pulse_summary_lines,
    record_streams,
)
from rfmux.pulse_capture.capture_session import PulseCaptureConfig
from rfmux.core.channels import parse_channel_spec, parse_module_channels
from rfmux.pulse_capture.channel_keys import channel_arg

_DEFAULTS = PulseCaptureConfig()


async def _main(serial: str, hostname: str | None, **kw):
    """Connect, record, and for a simulated board stop its stream."""
    import rfmux
    if serial.upper() == "MOCK":
        from rfmux.mock.helpers import create_mock_crs
        crs = await create_mock_crs(
            module=kw["module"] or min(kw["channels"]), verbose=False)
        await asyncio.sleep(2.0)             # stream warm-up
        try:
            return await record_streams(crs, **kw)
        finally:
            await crs.stop_udp_streaming()
    host = f', hostname: "{hostname}"' if hostname else ""
    session = rfmux.load_session(
        f'!HardwareMap [ !CRS {{ serial: "{serial}"{host} }} ]')
    crs = session.query(rfmux.CRS).one()
    await crs.resolve()
    return await record_streams(crs, **kw)


@click.command()
@click.option("--serial", default=None,
              help="CRS serial (rfmux<NNNN>.local), or MOCK for a simulated board; "
                   "with no options at all, a dialog asks for everything")
@click.option("--hostname", default=None, help="Board address when it is not <serial>.local")
@click.option("--module", "modules", type=int, multiple=True, default=(1,),
              show_default=True,
              help="Module to record; repeat it for one RF line over several")
@click.option("--channels", default=None,
              help="Channel ranges, 1-88 or 1,5-10, on every module, or per "
                   "module as 2:1-114,3:1-96; default: the biased channels of "
                   "the session's newest bias export for each module")
@click.option("--duration", type=float, default=None,
              help="Seconds to record, after the capture's noise training")
@click.option("--session", type=click.Path(file_okay=False), default=None,
              help="An existing session folder to record into")
@click.option("--session-dir", type=click.Path(file_okay=False), default=".",
              show_default=True, help="Where a new session folder is made")
@click.option("--capture/--no-capture", default=True, show_default=True,
              help="Pulse capture of the slow stream")
@click.option("--parser/--no-parser", default=True, show_default=True,
              help="Parser dirfile of the slow stream")
@click.option("--fastrx/--no-fastrx", default=True, show_default=True,
              help="fastrx recording of the channel stream")
@click.option("--parser-interface", default=None,
              help="1G interface for the parser; default: found from the board address")
@click.option("--fastrx-interface", default=None,
              help="100G interface fastrxd runs on; needed when several run")
@click.option("--fastrx-socket", default=None, help="fastrxd socket path, if not derivable")
@click.option("--merge-fastrx/--no-merge-fastrx", default=True, show_default=True,
              help="After the run, add the fastrx recording to the pulse file as its "
                   "fast stream (a both-mode file, as Periscope reviews it)")
@click.option("--show", type=click.Choice(["periscope", "overlay", "none"]),
              default="periscope", show_default=True,
              help="After the run: Periscope in review mode on the pulse file, the "
                   "overlay viewer on the channel with the most pulses, or nothing")
@click.option("--bias", type=click.Path(dir_okay=False, exists=True), default=None,
              help="bias_kids export for the df calibrations; default: the session's newest")
@click.option("--threshold-sigma", type=float, default=_DEFAULTS.threshold_sigma, show_default=True)
@click.option("--end-sigma", type=float, default=_DEFAULTS.end_sigma, show_default=True)
@click.option("--min-pulse-ms", type=float, default=_DEFAULTS.min_pulse_ms, show_default=True)
@click.option("--max-pulse-ms", type=float, default=_DEFAULTS.max_pulse_ms, show_default=True)
@click.option("--noise-train-ms", type=float, default=_DEFAULTS.noise_train_ms, show_default=True,
              help="Noise-training span; the other recorders start when it ends")
@click.option("--trigger-basis", type=click.Choice(["df", "iq"]), default=_DEFAULTS.trigger_basis,
              show_default=True)
@click.option("-q", "--quiet", is_flag=True)
def cli(serial, hostname, modules, channels, duration, session, session_dir,
        capture, parser, fastrx, parser_interface, fastrx_interface,
        fastrx_socket, merge_fastrx, show, bias, threshold_sigma, end_sigma,
        min_pulse_ms, max_pulse_ms, noise_train_ms, trigger_basis, quiet):
    """Record the slow and channel streams of a module, or of several
    feeding one RF line, into a session."""
    if serial is None:
        ctx = click.get_current_context()
        given = [name for name in ctx.params if name != "quiet"
                 and ctx.get_parameter_source(name)
                 != click.core.ParameterSource.DEFAULT]
        if given:
            raise click.UsageError("--serial is required (with no options "
                                   "at all, a dialog asks for everything)")
        from rfmux.tools.record_dialog import RecordDialog
        options = RecordDialog.ask()
        if options is None:
            return
        _run(quiet=quiet, **options)
        return
    if duration is None:
        raise click.UsageError("--duration is required")
    config = dataclasses.replace(
        _DEFAULTS, threshold_sigma=threshold_sigma, end_sigma=end_sigma,
        min_pulse_ms=min_pulse_ms, max_pulse_ms=max_pulse_ms,
        noise_train_ms=noise_train_ms, trigger_basis=trigger_basis)
    _run(serial=serial, hostname=hostname, modules=list(modules), channels=channels,
         duration=duration, session=session, session_dir=session_dir,
         capture=capture, parser=parser, fastrx=fastrx,
         parser_interface=parser_interface, fastrx_interface=fastrx_interface,
         fastrx_socket=fastrx_socket, merge_fastrx=merge_fastrx, show=show,
         bias=bias, config=config, quiet=quiet)


def _run(*, serial, hostname, modules, channels, duration, session,
         session_dir, capture, parser, fastrx, parser_interface,
         fastrx_interface, fastrx_socket, merge_fastrx, show, bias, config,
         quiet):
    """One recording, from the command line's options or the dialog's.
    *channels* is a range spec for every module of *modules*, a
    per-module spec (which names the modules itself), or None for each
    module's newest bias export."""
    folder = open_session(Path(session) if session else None, Path(session_dir))
    wanted = {}
    ranges = None
    if channels and ":" in channels:
        wanted = parse_module_channels(channels, max_channel=1024)
        modules = list(wanted)
    elif channels:
        ranges = parse_channel_spec(channels, max_value=1024, wildcard=False)
    if bias and len(modules) > 1:
        raise click.UsageError("--bias names one module's export; several "
                               "modules take the session's newest export each")
    multi = len(modules) > 1
    calibrations = {}
    if not quiet:
        click.echo(f"[record] session {folder}")
    for module in modules:
        bias_path = Path(bias) if bias else latest_bias_export(folder, module)
        biased, cals = biased_channels(bias_path) if bias_path else ([], {})
        chosen = wanted.get(module) or ranges or biased
        if not chosen:
            raise click.UsageError(
                f"no --channels, and no bias export for module {module} in "
                "the session to take them from")
        wanted[module] = chosen
        calibrations.update({(module, c) if multi else c: cal
                             for c, cal in cals.items()})
        if not quiet:
            if bias_path:
                click.echo(f"[record] bias export {bias_path.name}: "
                           f"{len(biased)} channels, {len(cals)} calibrated")
            click.echo(f"[record] module {module}, channels "
                       f"{chosen[0]}-{chosen[-1]} ({len(chosen)}), "
                       f"{duration:.1f} s")

    try:
        result = asyncio.run(_main(
            serial, hostname, module=None if multi else modules[0],
            channels=wanted if multi else wanted[modules[0]],
            duration_s=duration, session=folder, capture=capture,
            parser=parser, fastrx=fastrx, config=config,
            df_calibrations=calibrations or None,
            parser_interface=parser_interface,
            fastrx_interface=fastrx_interface, fastrx_socket=fastrx_socket,
            merge_fastrx=merge_fastrx, verbose=not quiet))
    except (RuntimeError, ValueError) as e:
        raise click.ClickException(str(e))
    if not quiet and result.capture is not None:
        for line in pulse_summary_lines(result.capture):
            click.echo(f"[record] {line}")
    for name in ("pulse_path", "dirfile_path", "fastrx_path"):
        path = getattr(result, name)
        if path is not None:
            click.echo(f"[record] {name.split('_')[0]:7s} {path}")
    if result.merged_fastrx:
        click.echo("[record] fastrx merged into the pulse file as its fast stream")
    for w in result.warnings:
        click.echo(f"[record] warning: {w}", err=True)
    _show(result, show)
    if result.warnings:
        raise SystemExit(1)


def periscope_review_command(pulse_path) -> list:
    """Periscope in review mode on *pulse_path*, offline, in its session."""
    return [sys.executable, "-m", "rfmux.tools.periscope", "--review",
            str(pulse_path)]


def _show(result, how: str) -> None:
    """After the run, when there is a display: Periscope in review mode
    on the pulse file, or the overlay viewer on the channel with the
    most pulses.  Without one, the command to run."""
    if how == "none" or result.capture is None or result.pulse_path is None:
        return
    headless = sys.platform != "darwin" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if how == "periscope":
        cmd = periscope_review_command(result.pulse_path)
        if headless:
            click.echo("[record] no display; to review: " + " ".join(cmd[1:]))
            return
        subprocess.Popen(cmd, start_new_session=True)
        return
    if result.fastrx_path is None:
        return
    counts = {ch: len(v) for ch, v in result.capture.summaries.items()}
    if not any(counts.values()):
        return
    channel = sorted(counts, key=lambda ch: (-counts[ch], ch))[0]
    dirfile = result.dirfile_path
    if headless:
        click.echo("[record] no display; to view: rfmux fastrx overlay "
                   f"{result.pulse_path} {result.fastrx_path} --channel "
                   f"{channel_arg(channel)} --pad 5"
                   + (f" --dirfile {dirfile}" if dirfile else ""))
        return
    from rfmux.tools.fastrx import show_overlay
    show_overlay(result.pulse_path, result.fastrx_path, channel=channel,
                 pad_ms=5.0, dirfile=dirfile)


if __name__ == "__main__":
    cli()
