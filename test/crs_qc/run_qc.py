#!/usr/bin/env -S uv run python3
"""
Usage: run_qc.py --serial=0033 [QC arguments...] [pytest arguments...]

This is the recommend point-of-entry for QC runs. You should invoke this script
as follows:

    $ ./run_qc.py --serial=0033

This script accepts any arguments pytest understands (try "pytest --help for
details). Here are a few useful examples:

    -x: abort testing after the first failure
    -s: don't suppress stdout/stderr
    -v: verbose (or "-vv" for more verbose)
    -k EXPRESSION: select which tests are executed

This script also understands a few QC-specific options:

    --view: launch a PDF viewer after completing the QC run
    --serial SERIAL: this is how you tell the QC script which CRS to use
    --directory DIRECTORY: collect outputs in DIRECTORY
"""

import argparse
import datetime
import pathlib
import pytest
import subprocess
import sys

if __name__ == "__main__":
    # We're going to invoke pytest and, at very least, want to point it to this
    # script's path (so it discovers the right conftest.py and QC tests.)
    qc_path = pathlib.Path(__file__).parent
    base_args = sys.argv[1:] + [qc_path.as_posix()]

    if "--serial" not in base_args and not any(
        arg.startswith("--serial=") for arg in base_args
    ):
        # We require, at very least, a --serial argument.
        print(sys.modules[__name__].__doc__)
        sys.exit(1)

    if "--directory" not in base_args and not any(
        arg.startswith("--directory") for arg in base_args
    ):
        # If --directory wasn't provided, invent one.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        day = datetime.datetime.now().strftime("%Y%m%d")
        results_dir = f"CRS_test_results/{day}/CRS_test_{timestamp}"
        base_args.extend(["--directory", results_dir])

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
        subprocess.Popen(("xdg-open", f"{results_dir}/qc.pdf"), start_new_session=True)
