"""Repository-wide pytest options and incremental timing reports.

Options live in the root conftest so pytest parses them even when invoked
outside test/. Keep this module dependency-light: the QC suite also loads it.
"""

from collections import defaultdict
from datetime import datetime, timezone
import json
import os
import platform
import subprocess
import sys
import time
from typing import Iterator

import pytest

# Named tiers, so an invocation says what it covers instead of making the reader
# evaluate a marker expression. Every tier except "hardware" and "all" excludes
# the board tests, so they report zero skips: a bare pass/fail rather than a
# result buried under ~75 "no --serial" skips.
#
# Values are marker expressions; "" means no filtering at all.
TIERS = {
    # no CRS and no GUI — the subset tox runs on every supported Python
    "portable": "portable and not hardware",
    # the edit loop: no server, no board
    "quick": "not slow_acquisition and not hardware",
    # only the data-acquisition tests: MockCRS server + UDP over loopback
    "acquisition": "slow_acquisition and not hardware",
    # everything runnable without a board — run this before pushing
    "full": "not hardware",
    # the board tests on their own (needs --serial)
    "hardware": "hardware",
    # literally everything (needs --serial, or the hardware tier just skips)
    "all": "",
}


def pytest_addoption(parser):
    parser.addoption(
        "--no-test-timings", action="store_true",
        help="Disable the automatic test-timings JSONL log and section summary.",
    )
    # rfmux/tools/qc/conftest.py also declares --serial and tolerates this one
    # already existing. Be symmetric about it: either conftest may load first
    # depending on which directory you point pytest at.
    try:
        parser.addoption("--serial", action="store", default=None)
    except ValueError:
        pass

    parser.addoption(
        "--tier",
        choices=sorted(TIERS),
        default=None,
        help="Run a named tier instead of writing a -m expression. "
             + "; ".join(f"{k}={v or 'everything'}" for k, v in TIERS.items()),
    )
    # Consumed by the _streamer_ports_free guard in test/conftest.py. Declared
    # here for the same reason as the options above: pytest only honours
    # pytest_addoption from an initial conftest.
    parser.addoption(
        "--allow-busy-streamer-ports",
        action="store_true",
        default=False,
        help="Run the acquisition tests even if UDP 9876/9877 are already "
             "bound. SO_REUSEPORT lets the second bind succeed and then "
             "nothing raises: on the mock's unicast loopback fallback one "
             "reader is starved outright, and where multicast works both "
             "readers see both simulations interleaved. Expect spurious "
             "pulse-detection failures either way.",
    )


def pytest_configure(config):
    if not config.getoption("no_test_timings"):
        timings = _TestTimings(config)
        config.pluginmanager.register(timings, "rfmux-test-timings")
        config.add_cleanup(timings.close)
    tier = config.getoption("tier", default=None)
    if tier is None:
        return

    # -m from the command line and --tier are two ways to say the same thing,
    # and silently letting one win would misreport what ran. addopts always
    # sets markexpr, so inspect the real argv rather than the resolved value.
    if any(a == "-m" or a.startswith("-m") and len(a) > 2 or a.startswith("--markexpr")
           for a in config.invocation_params.args):
        raise pytest.UsageError(
            f"--tier={tier} and -m both select tests; use one or the other."
        )

    config.option.markexpr = TIERS[tier]


class _TestTimings:
    def __init__(self, config: pytest.Config) -> None:
        self.started = time.perf_counter()
        directory = config.rootpath / "test-timings"
        directory.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.path = directory / f"{stamp}-{os.getpid()}.jsonl"
        self.stream = self.path.open("x", encoding="utf-8")
        self.sections = defaultdict(lambda: defaultdict(float))
        self.tests = defaultdict(float)
        self.collection_seconds = None
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=config.rootpath,
                capture_output=True, text=True, timeout=2, check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            commit = None
        self._write(
            "run_start", schema_version=1,
            args=list(config.invocation_params.args),
            cwd=str(config.invocation_params.dir),
            tier=config.getoption("tier"), commit=commit,
            machine=platform.node(), platform=platform.platform(),
            python=sys.version, pytest=pytest.__version__, pid=os.getpid(),
        )

    def _write(self, event: str, **fields: object) -> None:
        self.stream.write(json.dumps({
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - self.started,
            **fields,
        }) + "\n")
        # Preserve the last started phase even if pytest is killed or hangs.
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        self._write("session_start", markexpr=session.config.option.markexpr)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_collection(self, session: pytest.Session) -> Iterator[None]:
        started = time.perf_counter()
        self._write("collection_start")
        yield
        self.collection_seconds = time.perf_counter() - started
        self._write("collection_finish", duration_seconds=self.collection_seconds,
                    selected=len(session.items))

    def pytest_collectstart(self, collector: pytest.Collector) -> None:
        self._write("collector_start", nodeid=collector.nodeid)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        self._write("collector_finish", nodeid=report.nodeid,
                    outcome=report.outcome)

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item: pytest.Item) -> None:
        self._write("phase_start", nodeid=item.nodeid, phase="setup")

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_call(self, item: pytest.Item) -> None:
        self._write("phase_start", nodeid=item.nodeid, phase="call")

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_teardown(self, item: pytest.Item) -> None:
        self._write("phase_start", nodeid=item.nodeid, phase="teardown")

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        path = report.nodeid.split("::", 1)[0]
        parts = path.split("/")
        section = "/".join(parts[:2]) if len(parts) > 2 else parts[0]
        self.sections[section][report.when] += report.duration
        self.tests[report.nodeid] += report.duration
        self._write("phase_finish", nodeid=report.nodeid, section=section,
                    phase=report.when, duration_seconds=report.duration,
                    outcome=report.outcome)

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_sessionfinish(self, session: pytest.Session,
                             exitstatus: int) -> Iterator[None]:
        self._write("session_finish_start", exitstatus=int(exitstatus))
        yield
        self._write("run_finish", exitstatus=int(session.exitstatus),
                    sections=dict(self.sections),
                    collection_seconds=self.collection_seconds)

    def pytest_terminal_summary(self, terminalreporter: object) -> None:
        terminalreporter.write_sep("-", "Test timings (seconds)")
        terminalreporter.write_line(
            f"{'Section':30} {'Setup':>9} {'Call':>9} {'Teardown':>9} {'Total':>9}")
        for section, phases in sorted(
                self.sections.items(), key=lambda pair: -sum(pair[1].values())):
            terminalreporter.write_line(
                f"{section:30} {phases.get('setup', 0):9.3f} "
                f"{phases.get('call', 0):9.3f} {phases.get('teardown', 0):9.3f} "
                f"{sum(phases.values()):9.3f}")
        if self.collection_seconds is not None:
            terminalreporter.write_line(f"Collection: {self.collection_seconds:.3f}s")
        terminalreporter.write_line(
            f"Elapsed through summary: {time.perf_counter() - self.started:.3f}s")
        terminalreporter.write_line("Slowest tests (setup + call + teardown):")
        for nodeid, duration in sorted(
                self.tests.items(), key=lambda pair: -pair[1])[:10]:
            terminalreporter.write_line(f"  {duration:9.3f}s {nodeid}")
        terminalreporter.write_line(f"Timing log: {self.path}")
