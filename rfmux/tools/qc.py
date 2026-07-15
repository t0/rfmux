"""
rfmux qc - Run the CRS QC test suite.

This is a CLI front-end for the QC tests in test/crs_qc/, equivalent to
running test/crs_qc/run_qc.py directly. It accepts any arguments pytest
understands (try "pytest --help" for details). A few useful examples:

    -x: abort testing after the first failure
    -s: don't suppress stdout/stderr
    -v: verbose (or "-vv" for more verbose)
    -k EXPRESSION: select which tests are executed

It also understands a few QC-specific options:

    --view: launch a PDF viewer after completing the QC run
    --serial SERIAL: this is how you tell the QC script which CRS to use
    --directory DIRECTORY: collect outputs in DIRECTORY

Because the QC tests live in test/crs_qc/ (not in the installed rfmux
package), this command only works from a repository checkout.
"""

import click
import datetime
import pathlib
import pytest
import subprocess
import sys


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
    allow_interspersed_args=False,
))
@click.pass_context
def cli(ctx):
    """Run the CRS QC test suite (requires a repository checkout)"""

    # The QC tests aren't shipped in the rfmux wheel - locate them relative
    # to this file, which only works when running from a repo checkout.
    qc_path = pathlib.Path(__file__).parents[2] / "test" / "crs_qc"
    if not qc_path.is_dir():
        raise click.ClickException(
            f"QC tests not found at {qc_path} - 'rfmux qc' requires a "
            "repository checkout, not an installed wheel."
        )

    base_args = [qc_path.as_posix()] + list(ctx.args)

    if "--serial" not in base_args and not any(
        arg.startswith("--serial=") for arg in base_args
    ):
        # We require, at very least, a --serial argument.
        click.echo(sys.modules[__name__].__doc__)
        sys.exit(1)

    if "--directory" not in base_args and not any(
        arg.startswith("--directory") for arg in base_args
    ):
        # If --directory wasn't provided, invent one.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        day = datetime.datetime.now().strftime("%Y%m%d")
        results_dir = f"CRS_test_results/{day}/CRS_test_{timestamp}"
        base_args.extend(["--directory", results_dir])
    else:
        results_dir = None

    if launch_viewer := "--view" in base_args:
        # Intercept --view so we don't open several work-in-progress PDFs
        base_args.remove("--view")

    # Stage-1 tests have no particular filter requirements
    pytest.main(base_args + ["-m", "qc_stage1"])

    # Stage-2 tests require loopback cables
    pytest.main(base_args + ["-m", "qc_stage2", "--low-bank"])
    pytest.main(base_args + ["-m", "qc_stage2", "--high-bank"])

    # If called with "--view", open it. If this unexpectedly opens gimp, try
    # $ xdg-mime default org.gnome.Evince.desktop application/pdf
    if launch_viewer:
        if results_dir is None:
            # --directory was passed explicitly; recover it for the viewer.
            idx = next(
                i for i, arg in enumerate(base_args)
                if arg == "--directory" or arg.startswith("--directory=")
            )
            arg = base_args[idx]
            results_dir = (
                arg.split("=", 1)[1] if "=" in arg else base_args[idx + 1]
            )
        subprocess.Popen(
            ("xdg-open", f"{results_dir}/qc.pdf"), start_new_session=True
        )
