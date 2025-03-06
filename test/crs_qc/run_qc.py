#!/usr/bin/env -S uv run python3

import pytest
import sys
import datetime
import pathlib
import argparse

if __name__ == "__main__":
    qc_path = pathlib.Path(__file__).parent
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    day = datetime.datetime.now().strftime("%Y%m%d")
    results_dir = f"CRS_test_results/{day}/CRS_test_{timestamp}"

    base_args = sys.argv[1:] + [
        qc_path.as_posix(),
        "--directory",
        results_dir,
    ]

    # Stage-1 tests have no particular filter requirements
    pytest.main(base_args + ["-m", "qc_stage1"])

    # Stage-2 tests require XYZ
    pytest.main(base_args + ["-m", "qc_stage2", "--low-bank"])
    pytest.main(base_args + ["-m", "qc_stage2", "--high-bank"])
