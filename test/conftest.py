# empty or noop

import pytest
import rfmux
import os
import pytest_asyncio

def pytest_addoption(parser):
    parser.addoption("--serial", action="store", default=None)


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
