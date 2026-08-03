import pytest
import rfmux
import os
import pytest_asyncio

# Fixtures that can only be satisfied by a real board. Requesting one — directly
# or transitively — is what makes a test part of the hardware tier.
#
# "serial" is test_mock_vs_real.py's own gate; it compares a mock CRS against a
# live one, so even its crs_mock fixture needs a board behind it.
HARDWARE_FIXTURES = frozenset({"live_session", "crs", "serial"})

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
    parser.addoption("--serial", action="store", default=None)
    parser.addoption(
        "--tier",
        choices=sorted(TIERS),
        default=None,
        help="Run a named tier instead of writing a -m expression. "
             + "; ".join(f"{k}={v or 'everything'}" for k, v in TIERS.items()),
    )


def pytest_configure(config):
    tier = config.getoption("tier")
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


def pytest_collection_modifyitems(config, items):
    """Mark hardware tests by the fixtures they request.

    The board-dependent tests are gated by a skip inside ``live_session``,
    which makes them impossible to select: there is nothing to write after
    ``-m``. Deriving the marker from the fixture graph keeps the hardware tier
    addressable (``-m hardware``, ``-m "not hardware"``) without asking anyone
    to remember a decorator that duplicates what the fixtures already say.
    """
    for item in items:
        if HARDWARE_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.hardware)


@pytest.fixture
def live_session(pytestconfig):
    if (serial := pytestconfig.getoption("serial")) is None:
        pytest.skip(
            "Use the '--serial' argument to specify a running CRS board for this test."
        )

    return rfmux.load_session(
        f"""
        !HardwareMap
        - !CRS {{ serial: "{serial}" }}
        """
    )

@pytest_asyncio.fixture
async def crs(live_session):
    crs = live_session.query(rfmux.CRS).one()
    await crs.resolve()

    # setup: instill politeness
    await crs.set_timestamp_port(crs.TIMESTAMP_PORT.TEST)

    yield crs

    # teardown: restore politeness
    await crs.set_analog_bank(high=False)
    await crs.set_decimation(stage=6, short=False, module=[1,2,3,4])
