import datetime
import matplotlib
import os
import pytest
import pytest_asyncio
from . import report_generator
import rfmux
import shelve
import subprocess
import sys
import pytest_check  # imported to ensure "check" can be used as a fixture

from .crs_qc import ResultTable

matplotlib.use("Agg")  # for non-interactive use


def pytest_addoption(parser):
    # These options all end up determining which ReadoutModules are attached to the CRS
    parser.addoption("--module", action="store", default=None, type=int)
    parser.addoption("--modules", action="store", default=None, type=int)
    parser.addoption("--low-bank", action="store_true", default=None)
    parser.addoption("--high-bank", action="store_true", default=None)

    # Open a PDF viewer after the test run completes
    parser.addoption("--view", action="store_true", default=False)
    parser.addoption("--directory", default=None)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    # Validate mandatory command-line arguments
    if (serial := session.config.getoption("--serial")) is None:
        raise pytest.UsageError(
            "--serial number not provided! Can't talk to the CRS board."
        )
    session.serial = serial

    if (results_dir := session.config.getoption("--directory")) is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        day = datetime.datetime.now().strftime("%Y%m%d")
        results_dir = f"CRS_test_results/{day}/CRS_test_{session.serial}_{timestamp}"
    session.results_dir = results_dir
    os.makedirs(session.results_dir, exist_ok=True)

    # Initialize shelf file with appropriate structure
    session.shelf_filename = f"{session.results_dir}/qc"
    with shelve.open(session.shelf_filename, writeback=True) as shelf:
        shelf["serial"] = serial
        shelf.setdefault("sections", {})
        shelf.setdefault("runs", [])

        shelf["runs"].append(
            {
                # TODO: it would be nice to assemble a list of what tests ran, when
                "date": datetime.datetime.now(),
                "args": sys.argv[1:],
                "logs": [],
            }
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Capture a "pass" / "fail" result for each test, and store it in the results shelf.
    """

    outcome = yield
    report = outcome.get_result()

    if call.when == "call":
        with shelve.open(item.session.shelf_filename, writeback=True) as shelf:
            shelf["runs"][-1]["logs"].append(
                {
                    "nodeid": item.nodeid,
                    "outcome": report.outcome,
                }
            )


@pytest.fixture(scope="function")
def shelf(request):
    """
    A fixture that allows the results shelf to be accessed from within each test case.
    """
    return shelve.open(request.session.shelf_filename, writeback=True)


@pytest_asyncio.fixture(scope="function")
async def d(request):
    """
    A fixture that provides access to a Dfmux object ("d") from within a test case.
    """

    kwargs = {}

    # Depending on what combination of options (--module, --modules,
    # --low-bank, --high-bank), instantiate the CRS with the appropriate
    # ReadoutModules attached.

    module = request.config.getoption("--module")
    modules = request.config.getoption("--modules")
    low_bank = request.config.getoption("--low-bank")
    high_bank = request.config.getoption("--high-bank")

    match (low_bank, high_bank, modules, module):
        case [None, None, None, None]:
            # default constructor
            pass

        case [True, None, None, None]:
            # low banks only
            kwargs["modules"] = [rfmux.ReadoutModule(module=m + 1) for m in range(4)]

        case [None, True, None, None]:
            # high banks only
            kwargs["modules"] = [rfmux.ReadoutModule(module=m + 5) for m in range(4)]

        case [None, None, int(), None]:
            # specified bank count
            if modules < 1 or modules > 8:
                raise pytest.UsageError(
                    "--modules argument out of range! Expected 1 <= modules <= 8"
                )
            kwargs["modules"] = [
                rfmux.ReadoutModule(module=m + 1) for m in range(modules)
            ]

        case [None, None, None, int()]:
            # specific module index
            if module < 1 or module > 8:
                raise pytest.UsageError(
                    "--module argument out of range! Expected 1 <= module <= 8"
                )
            kwargs["modules"] = [rfmux.ReadoutModule(module=module)]

        case _:
            # confused
            raise pytest.UsageError(
                "Unexpected combination of --modules / --module / --low-bank / --high-bank"
            )

    # Create a HardwareMap - normally, we'd do this through a YAML session
    # file, but here, we want to control the ReadoutModules associated with it.
    hwm = rfmux.HardwareMap()
    hwm.add(rfmux.CRS(serial=request.session.serial, **kwargs))
    hwm.commit()

    d = hwm.query(rfmux.CRS).one()

    # Resolve the device to get all the methods
    await d.tuber_resolve()
    await d.set_timestamp_port(d.TIMESTAMP_PORT.TEST)

    # If we're going to be doing low or high banking things, make sure we're
    # correctly configured. If neither --low-bank or --high-bank were
    # specified, don't configure anything - any misconfiguration should be
    # caught by error checks in the on-board C++ code.
    if low_bank:
        await d.set_analog_bank(high=False)
    elif high_bank:
        await d.set_analog_bank(high=True)

    yield d


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session):
    """After the session completes, assemble (and --view, if requested) a PDF report"""
    if hasattr(session, "results_dir"):
        pdf = f"{session.results_dir}/qc.pdf"

        # Generate the report PDF
        report_generator.main(
            f"{session.results_dir}", f"{session.results_dir}/qc.html", pdf
        )

        # If called with "--view", open it. If this unexpectedly opens gimp, try
        # $ xdg-mime default org.gnome.Evince.desktop application/pdf
        if session.config.getoption("view"):
            subprocess.call(("xdg-open", pdf))
