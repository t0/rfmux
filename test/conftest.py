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

# --serial and --tier are declared in the ROOT conftest.py, not here.
# pytest only honours pytest_addoption from an initial conftest, so options
# declared in this file are invisible whenever the arguments do not point into
# test/ — e.g. running pytest from a subdirectory.


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
