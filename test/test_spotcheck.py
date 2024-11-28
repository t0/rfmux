#!/usr/bin/env -S PYTHONPATH=.. pytest-3 -v -W error::UserWarning

"""
Spot checks for API regressions.

This test script can be invoked in three ways:

- By itself, in your PC's current Python environment, without relying on a
  running CRS board:

      ./test_spotcheck.py

- By itself, in your PC's current Python environment, with a CRS board running
  alongside:

      ./test_spotcheck.py --serial=0024

- As part of a complete regression test environment (exercising all test
  scripts and in a variety of different Python environments), without relying
  on a running CRS board:

      ~/rfmux$ ./test.sh
"""

import rfmux
import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "frequency", (100e6, 2.4e9, 2.6e9, 4.9e9, 5.1e9, 7.4e9, 7.6e9, 9.9e9)
)
async def test_nco_set_get_frequencies(live_session, frequency):
    """
    Do NCO setters and getters work correctly?

    The purpose is to check that the get/set correctly manages higher Nyquist
    zones in the C API. Note that it doesn't guarantee the RFDC is actually
    doing the right thing - just API consistency.
    """

    d = live_session.query(rfmux.CRS).one()
    await d.resolve()

    await d.set_nco_frequency(frequency, d.UNITS.HZ, module=1)
    assert (await d.get_nco_frequency(d.UNITS.HZ, module=1)) == pytest.approx(
        frequency, abs=0.001
    )  # expect agreement below 1 mHz


if __name__ == "__main__":
    pytest.main([__file__])
