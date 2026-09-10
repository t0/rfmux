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
from rfmux.tools.parser import parse_ranges

_DEFAULTS = PulseCaptureConfig()


async def _main(serial: str, hostname: str | None, **kw):
    """Connect, record, and for a simulated board stop its stream."""
    import rfmux
    if serial.upper() == "MOCK":
        from rfmux.mock.helpers import create_mock_crs
        crs = await create_mock_crs(module=kw["module"], verbose=False)
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
@click.option("--serial", required=True,
              help="CRS serial (rfmux<NNNN>.local), or MOCK for a simulated board")
@click.option("--hostname", default=None, help="Board address when it is not <serial>.local")
@click.option("--module", type=int, default=1, show_default=True)
@click.option("--channels", default=None,
              help="Channel ranges, 1-88 or 1,5-10; default: the biased channels of "
                   "the session's newest bias export")
@click.option("--duration", type=float, required=True,
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
@click.option("--show/--no-show", default=True, show_default=True,
              help="After the run, open the overlay viewer on the channel with the "
                   "most pulses")
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
def cli(serial, hostname, module, channels, duration, session, session_dir,
        capture, parser, fastrx, parser_interface, fastrx_interface,
        fastrx_socket, merge_fastrx, show, bias, threshold_sigma, end_sigma,
        min_pulse_ms, max_pulse_ms, noise_train_ms, trigger_basis, quiet):
    """Record the slow and channel streams of one module into a session."""
    folder = open_session(Path(session) if session else None, Path(session_dir))
    bias_path = Path(bias) if bias else latest_bias_export(folder, module)
    biased, calibrations = biased_channels(bias_path) if bias_path else ([], {})
    if channels:
        chosen = [c + 1 for r in parse_ranges(channels, 1, 1024, "channel") for c in r]
    elif biased:
        chosen = biased
    else:
        raise click.UsageError(
            "no --channels, and no bias export in the session to take them from")
    config = dataclasses.replace(
        _DEFAULTS, threshold_sigma=threshold_sigma, end_sigma=end_sigma,
        min_pulse_ms=min_pulse_ms, max_pulse_ms=max_pulse_ms,
        noise_train_ms=noise_train_ms, trigger_basis=trigger_basis)
    if not quiet:
        click.echo(f"[record] session {folder}")
        if bias_path:
            click.echo(f"[record] bias export {bias_path.name}: "
                       f"{len(biased)} channels, {len(calibrations)} calibrated")
        click.echo(f"[record] module {module}, channels {chosen[0]}-{chosen[-1]} "
                   f"({len(chosen)}), {duration:.1f} s")

    try:
        result = asyncio.run(_main(
            serial, hostname, module=module, channels=chosen,
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
    if show:
        _show(result)
    if result.warnings:
        raise SystemExit(1)


def _show(result) -> None:
    """The overlay viewer on the channel with the most pulses, when
    there is a display; the command to run otherwise."""
    if result.capture is None or result.pulse_path is None \
            or result.fastrx_path is None:
        return
    counts = {ch: len(v) for ch, v in result.capture.summaries.items()}
    if not any(counts.values()):
        return
    channel = max(counts, key=lambda ch: (counts[ch], -ch))
    dirfile = result.dirfile_path
    if sys.platform != "darwin" and not (os.environ.get("DISPLAY")
                                         or os.environ.get("WAYLAND_DISPLAY")):
        click.echo("[record] no display; to view: rfmux fastrx overlay "
                   f"{result.pulse_path} {result.fastrx_path} --channel {channel} "
                   f"--pad 5" + (f" --dirfile {dirfile}" if dirfile else ""))
        return
    from rfmux.tools.fastrx import show_overlay
    show_overlay(result.pulse_path, result.fastrx_path, channel=channel,
                 pad_ms=5.0, dirfile=dirfile)


if __name__ == "__main__":
    cli()
