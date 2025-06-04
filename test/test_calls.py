#!/usr/bin/env -S PYTHONPATH=.. pytest-3 -v

"""
Calling tests.

This test script can be invoked in three ways:

- By itself, in your PC's current Python environment, without relying on a
  running CRS board:

      ./test_calls.py

- By itself, in your PC's current Python environment, with a CRS board running
  alongside:

      ./test_calls.py --serial=0024

- As part of a complete regression test environment (exercising all test
  scripts and in a variety of different Python environments), without relying
  on a running CRS board:

      ~/rfmux$ ./test.sh
"""

import rfmux
import pytest


@pytest.mark.asyncio
async def test_macro():
    @rfmux.macro(rfmux.CRS, register=True)
    async def macro_that_returns_serial_number(d):
        return d.serial

    s = rfmux.load_session(
        """
        !HardwareMap
        - !CRS { serial: "0024" }
        """
    )
    ds = s.query(rfmux.CRS)
    assert set(await ds.macro_that_returns_serial_number()) == {"0024"}

    d = s.query(rfmux.CRS).one()
    assert d.serial == await d.macro_that_returns_serial_number()


@pytest.mark.asyncio
async def test_algorithm():
    @rfmux.algorithm(rfmux.CRS)
    async def algorithm_that_returns_serial_number(d):
        return d.serial

    s = rfmux.load_session(
        """
        !HardwareMap
        - !CRS { serial: "0024" }
        """
    )
    ds = s.query(rfmux.CRS)
    assert set(await ds.algorithm_that_returns_serial_number()) == {"0024"}


@pytest.mark.asyncio
async def test_simple_live_board_interaction(crs):
    await crs.get_frequency(crs.UNITS.HZ, channel=1, module=1)


@pytest.mark.asyncio
async def test_live_board_interaction_with_orm(live_session):
    ds = live_session.query(rfmux.CRS)
    await ds.resolve()

    d = live_session.query(rfmux.CRS).one()
    f = await d.get_frequency(d.UNITS.HZ, channel=1, module=1)
    fs = await ds.get_frequency(d.UNITS.HZ, channel=1, module=1)

    assert f == fs[0]
    assert await d.module[1].channel[1].get_frequency(d.UNITS.HZ) == f


@pytest.mark.asyncio
async def test_macro_with_arg_filler():
    @rfmux.macro(rfmux.CRS, register=True)
    async def channel_macro(self, channel, module):
        return module == 3 and channel == 4

    s = rfmux.load_session('!HardwareMap [!CRS { serial: "0024" }]')
    d = s.query(rfmux.CRS).one()

    assert await d.module[3].channel[4].channel_macro()


if __name__ == "__main__":
    pytest.main([__file__])
