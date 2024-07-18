#!/usr/bin/env -S PYTHONPATH=.. python3

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
import textwrap


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
