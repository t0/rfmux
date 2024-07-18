#!/usr/bin/env -S PYTHONPATH=.. pytest-3 -v

"""
Calling tests.

This test script can be invoked in three ways:

- By itself, in your PC's current Python environment, without relying on a
  running CRS board:

      ./test_calls.py

- By itself, in your PC's current Python environment, with a CRS board running
  alongside: (note the CRS_SERIAL environment variable!)

      CRS_SERIAL=0024 ./test_calls.py

- As part of a complete regression test environment (exercising all test
  scripts and in a variety of different Python environments), without relying
  on a running CRS board:

      ~/rfmux$ ./test.sh
"""

import rfmux
import pytest


@pytest.mark.xfail(reason="@macro unexpectedly requires resolve()")
def test_macro():
    @rfmux.macro(rfmux.CRS)
    async def macro_that_returns_serial_number(d):
        return d.serial

    s = rfmux.load_session(
        """
        !HardwareMap
        - !CRS { serial: "0024" }
        """
    )
    ds = s.query(rfmux.CRS)
    assert set(ds.macro_that_returns_serial_number()) == {"0024"}

    d = s.query(rfmux.CRS).one()
    assert d.serial == d.macro_that_returns_serial_number()


def test_algorithm():
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
    with pytest.warns(DeprecationWarning):
        assert set(ds.algorithm_that_returns_serial_number()) == {"0024"}


def test_simple_live_board_interaction(live_session):
    d = live_session.query(rfmux.CRS).one()
    d.resolve()
    d.get_frequency(d.UNITS.HZ, channel=1, module=1)


def test_live_board_interaction_with_orm(live_session):
    ds = live_session.query(rfmux.CRS)
    ds.resolve()

    d = live_session.query(rfmux.CRS).one()
    f = d.get_frequency(d.UNITS.HZ, channel=1, module=1)
    fs = ds.get_frequency(d.UNITS.HZ, channel=1, module=1)

    assert {f} == set(fs)


if __name__ == "__main__":
    pytest.main([__file__])
