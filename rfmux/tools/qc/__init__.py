"""
rfmux qc - Run the CRS QC test suite.

This is a CLI front-end for the QC tests that live alongside this module.
It accepts any arguments pytest understands (try "pytest --help" for
details). A few useful examples:

    -x: abort testing after the first failure
    -s: don't suppress stdout/stderr
    -v: verbose (or "-vv" for more verbose)
    -k EXPRESSION: select which tests are executed

It also understands a few QC-specific options:

    --view: launch a PDF viewer after completing the QC run
    --serial SERIAL: this is how you tell the QC script which CRS to use
    --directory DIRECTORY: collect outputs in DIRECTORY
"""

import click
import datetime
import importlib.util
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
    """Run the CRS QC test suite"""

    # The QC suite has dependencies beyond rfmux's own (report generation,
    # regression fits). Fail early with a hint rather than mid-collection.
    # find_spec (not import) checks availability without importing - actually
    # importing pytest plugins like pytest_check here would prevent pytest
    # from applying assertion rewriting to them later.
    missing = [
        module
        for module in ("bs4", "htpy", "markdown", "markupsafe",
                       "pytest_check", "requests", "sklearn", "weasyprint")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise click.ClickException(
            f"missing QC dependencies ({', '.join(missing)}) - "
            "install with: pip install rfmux[qc]"
        )

    qc_path = pathlib.Path(__file__).parent

    # Pin pytest's configuration to the one we ship. This keeps behaviour
    # identical whether we're run from a checkout or an installed wheel, and
    # stops pytest from picking up an unrelated project's ini file or writing
    # a cache into site-packages.
    base_args = [
        qc_path.as_posix(),
        "-c", (qc_path / "pytest.ini").as_posix(),
        "-p", "no:cacheprovider",
    ] + list(ctx.args)

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
