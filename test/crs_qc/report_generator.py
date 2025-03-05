#!/usr/bin/env -S python3
import argparse
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import pytest
import shelve
import socket
import sys
import warnings

from htpy import (
    body,
    h1,
    h2,
    h3,
    head,
    html,
    li,
    link,
    p,
    style,
    table,
    tbody,
    td,
    th,
    thead,
    title,
    tr,
    ul,
)
import weasyprint

from .crs_qc import render_markdown, ResultTable


def main(directory, html_filename, pdf_filename):
    # Retrieve sections from shelf
    with shelve.open((pathlib.Path(directory) / "qc").as_posix()) as qc:
        serial = qc["serial"]

        sections = sorted(qc["sections"].keys())
        elements = [qc["sections"][k] for k in sections]

        # Generate run table
        rt_runs = ResultTable(
            "Run", "Date", "Arguments", "Passed", "Failed", "Tests Executed"
        )
        num_runs = len(qc["runs"])

        for n, run in enumerate(qc["runs"]):

            outcomes = [x["outcome"] for x in run["logs"]]
            passed = list(map(lambda x: x == "passed", outcomes))
            failed = list(map(lambda x: x == "failed", outcomes))

            if all(passed):
                rt_runs.pass_(
                    f"{n+1}",
                    f'{run["date"].strftime("%Y-%m-%d %H:%M:%S")}',
                    f'{" ".join(run["args"])}',
                    sum(passed),
                    sum(failed),
                    len(outcomes),
                )
            elif any(failed):
                rt_runs.fail(
                    f"{n+1}",
                    f'{run["date"].strftime("%Y-%m-%d %H:%M:%S")}',
                    f'{" ".join(run["args"])}',
                    sum(passed),
                    sum(failed),
                    len(outcomes),
                )
            else:
                rt_runs.row(
                    f"{n+1}",
                    f'{run["date"].strftime("%Y-%m-%d %H:%M:%S")}',
                    f'{" ".join(run["args"])}',
                    sum(passed),
                    sum(failed),
                    len(outcomes),
                )

        # Generate test results for each run
        runs = []
        for n, run in enumerate(qc["runs"]):
            rt = ResultTable("Test", "Status")
            for l in run["logs"]:
                if l["outcome"] == "passed":
                    rt.pass_(l["nodeid"], "Passed")
                elif l["outcome"] == "failed":
                    rt.fail(l["nodeid"], "Failed")
                else:
                    rt.row(l["nodeid"], l["outcome"])

            r = [h3[f"Run {n+1}"], rt]
            runs.extend(r)

    # Render document preamble
    preamble = [
        render_markdown(
            f"""
            # CRS QC Test Results: Serial {serial}

            This report was assembled in the following environment:

            | Item  | Value |
            | ----  | ----- |
            | CRS Serial | {serial} |
            | Date/Time  | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
            | Hostname | {socket.gethostname()} |
            | User | {os.environ["USER"]} |
            | Python Executable | {sys.argv[0]} |
            | Python Arguments | {' '.join(sys.argv[1:])} |
            | Python Version | {sys.version} |
            | Pytest Version | {pytest.__version__} |

            This report was assembled from {num_runs} different test runs:
        """
        ),
        rt_runs,
        render_markdown(
            f"""
            Typically, newer test results replace results from older ones.

            ## Run Summaries
        """
        ),
        runs,
    ]

    # Assemble document
    doc = html[
        head[title[f"CRS QC Test Results: Board CRS{serial}"]],
        preamble,
        body[elements],
    ]
    encoded = doc.encode()

    # Save HTML if requested
    if html_filename is not None:
        with open(html_filename, "wb") as f:
            f.write(encoded)

    # Save PDF
    w = weasyprint.HTML(string=encoded, base_url=directory)
    css_path = pathlib.Path(__file__).parent / "style.css"
    w.write_pdf(pdf_filename, stylesheets=[weasyprint.CSS(css_path.as_posix())])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_report",
        description="Generate a PDF version of the QC test run",
        conflict_handler="resolve",
    )
    parser.add_argument("-d", "--directory", default=".")
    parser.add_argument("--html", default=None)  # can't override -h
    parser.add_argument("-p", "--pdf", default="qc.pdf")
    args = parser.parse_args()
    main(args.directory, args.html, args.pdf)
