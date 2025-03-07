#!/usr/bin/env -S python3
import argparse
import bs4
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
import uuid
import warnings

from htpy import (
    a,
    body,
    div,
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

try:
    # when imported from conftest.py, we need package-relative imports
    from .crs_qc import render_markdown, ResultTable
except ImportError:
    # when invoked separately, we need a straight import
    from crs_qc import render_markdown, ResultTable


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
        runs = ResultTable("Run", "Test", "Status")
        for n, run in enumerate(qc["runs"]):
            for l in run["logs"]:
                if l["outcome"] == "passed":
                    runs.pass_(f"{n+1}", l["nodeid"], "Passed")
                elif l["outcome"] == "failed":
                    # Format failures as a list
                    failures = ul[(li[f] for f in l["failures"])]
                    runs.fail(f"{n+1}", l["nodeid"], failures)
                else:
                    runs.row(f"{n+1}", l["nodeid"], l["outcome"])

    # Render document preamble
    preamble = [
        render_markdown(
            f"""
            # CRS QC Test Results: Serial {serial}

            ## Environment

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

            ## Runs

            This report was assembled from {num_runs} different test runs:
        """
        ),
        rt_runs,
        render_markdown(
            f"""
            Typically, newer test results replace results from older ones.

            ## Run Details
        """
        ),
        runs,
        h2["Table of Contents"],
        div(id="toc"),
    ]

    # Assemble document
    doc = html[
        head[title[f"CRS QC Test Results: Board CRS{serial}"]],
        preamble,
        body[elements],
    ]
    encoded = doc.encode()

    # Inject TOC by finding the placeholder, and filling it with a list of all
    # subsequent headers
    soup = bs4.BeautifulSoup(encoded, "html.parser")
    container = soup.find("div", {"id": "toc"})
    headings = container.find_all_next(["h1", "h2", "h3", "h4", "h5", "h6"])

    toc_stack = [(0, [])]
    for heading in headings:
        # Add id if we don't already have one - need a link target
        if not heading.has_attr("id"):
            heading["id"] = uuid.uuid4().hex

        # pop back to the correct parent level
        level = int(heading.name[1])
        while level <= toc_stack[-1][0] and len(toc_stack) > 1:
            toc_stack.pop()

        # add this one
        toc_stack[-1][1].append(
            node := {
                "level": level,
                "id": heading["id"],
                "title": heading.get_text(),
                "children": [],
            }
        )

        toc_stack.append((level, node["children"]))

    def build_toc_ul(nodes):
        """Recursively build TOC markup"""
        if not nodes:
            return []

        parts = []
        for n in nodes:
            parts.append(
                li[
                    a(href=f'#{n["id"]}')[n["title"]],  # this node
                    build_toc_ul(n["children"]),  # children
                ]
            )

        return ul[*parts]

    toc_html = build_toc_ul(toc_stack[0][1])

    # If we produced anything, append it to the table of contents.
    # (If we didn't -- well, that was a funny empty test run)
    if toc_html:
        container.append(bs4.BeautifulSoup(toc_html.encode(), "html.parser"))

    reencoded = str(soup)

    # Save HTML if requested
    if html_filename is not None:
        with open(html_filename, "w") as f:
            f.write(reencoded)

    # Save PDF
    w = weasyprint.HTML(string=reencoded, base_url=directory)
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
