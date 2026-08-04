"""Root conftest: command-line options for the whole repo.

``pytest_addoption`` is only honoured from an *initial* conftest — one pytest
loads before parsing arguments, which means the rootdir conftest and the
conftests along the path of the arguments you passed. It is emphatically not
honoured from ``test/conftest.py`` when the arguments do not point into
``test/``: run ``pytest --tier=quick`` from any subdirectory with the options
declared down there and you get "unrecognized arguments" rather than a test
run.

So the options live here, where every in-repo invocation sees them. Fixtures
and collection hooks stay in test/conftest.py, which has no such restriction.

Deliberately dependency-light: this module is imported for every pytest run
anywhere in the repo, including the QC suite, so it must not import rfmux.
"""

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
             "bound. They would be shared via SO_REUSEPORT and each reader "
             "would see only part of the stream, so expect spurious "
             "pulse-detection failures.",
    )


def pytest_configure(config):
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
