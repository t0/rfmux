"""rfmux qc - run the CRS QC test suite and write an HTML report."""

import click
import datetime
import subprocess
import sys


def _modules(ctx, param, ranges: str | None) -> set[int] | None:
    """The modules in "1,3,5-8", or None for every module"""
    if ranges is None:
        return None
    try:
        modules = set()
        for part in ranges.split(","):
            first, _, last = part.partition("-")
            modules.update(range(int(first), int(last or first) + 1))
    except ValueError:
        raise click.BadParameter(f"{ranges!r} isn't modules like 5-8 or 1,3,5-8")
    if not modules <= set(range(1, 9)):
        raise click.BadParameter(f"{ranges!r}: the modules are 1-8")
    return modules


@click.command()
@click.option("--serial", help="The board to test, e.g. 0246.")
@click.option("--output", metavar="FILE",
              help="The report; default qc_<serial>_<date>_<time>.html here. An "
                   "existing report gains this run. A name ending .db, .sqlite or "
                   ".sqlite3 means the SQLite database alone, which --input also reads.")
@click.option("--view", is_flag=True, help="Open the report in a browser when done.")
@click.option("--input", "inputs", multiple=True, metavar="FILE",
              help="Instead of testing, build a report from the results in existing "
                   "reports, with this version of rfmux (no --serial needed). Repeat it "
                   "to merge reports into --output; with one, --output defaults to FILE.")
@click.option("--step", "steps", multiple=True, metavar="NAME",
              help="Run only this step (dac_mixmodes, say); repeat it for more.")
@click.option("--modules", metavar="RANGES", callback=_modules,
              help="Run only the steps on these modules (5-8, or 1,3,5-8), "
                   "leaving out the board's.")
@click.option("-x", "exitfirst", is_flag=True, help="Stop at the first step that fails.")
@click.pass_context
def cli(ctx, serial, output, view, inputs, steps, modules, exitfirst):
    """Run the CRS QC test suite against one board and write an HTML report.

    \b
    Steps, run in this order (PLAN in rfmux/tools/qc/runner.py):
      board     communications, metadata, temperatures, voltages,
                currents. Needs no cables.
      datapath  DAC and ADC, once for each module 1-8. Needs each
                module's DAC looped back to its ADC with an SMA cable.

    It exits with status 1 if a step fails.

    \b
    The report is one file to e-mail around, and the record of every run
    into it: an SQLite database inside it, of runs, results and their
    contents. It filters by board, run (or each board's latest results),
    module and status, and shows tests against modules or boards. Its figures
    are drawn interactively by Python in the browser (loaded from the web),
    and a Python console at its end works on its data. In Python:
    rfmux.tools.qc.report_generator.load(FILE)

    \b
    Examples:
      rfmux qc --serial=0246 --view
      # re-run one step into an existing report; newer results
      # replace older ones
      rfmux qc --serial=0246 --output=FILE --step=dac_mixmodes
      rfmux qc --serial=0246 --output=FILE --modules=5-8
      rfmux qc --input=FILE --view
      rfmux qc --input=A.html --input=B.html --output=both.html
      rfmux qc --input=FILE --output=results.db  # the database alone
    """

    if inputs:
        from . import report_generator
        from .record import Record

        if output is None and len(inputs) > 1:
            raise click.ClickException("merging reports needs --output")
        output = output or inputs[0]
        record = Record()
        for input_ in inputs:
            try:
                record.merge(report_generator.read(input_))
            except (OSError, ValueError) as e:
                raise click.ClickException(f"no QC results in {input_}: {e}")
        report_generator.write(record, output)
        click.echo(f"Wrote {output}")
        passed = True

    elif serial is None:
        click.echo(ctx.get_help())
        sys.exit(1)

    else:
        import asyncio

        from . import runner

        names = {runner.step_name(e) for e in runner.PLAN if runner.reported(e)}
        if unknown := set(steps) - names:
            raise click.BadParameter(
                f"no step {', '.join(sorted(unknown))}; the steps are "
                f"{', '.join(sorted(names))}", param_hint="--step")

        output = output or f"qc_{serial}_{datetime.datetime.now():%Y%m%d_%H%M%S}.html"
        passed = asyncio.run(
            runner.run(serial, output, steps=steps, modules=modules,
                       exitfirst=exitfirst))

    if view:
        subprocess.Popen(("xdg-open", output), start_new_session=True)
    if not passed:
        sys.exit(1)
