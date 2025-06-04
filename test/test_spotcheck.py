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
    "frequency", (100e6, 2.4e9, 2.6e9, 4.9e9, 5.1e9, 7.4e9, 7.6e9, 9.9e9))
@pytest.mark.parametrize("module", range(1, 9))
async def test_nco_set_get_frequencies(live_session, frequency, module):
    """
    Do NCO setters and getters work correctly?

    The purpose is to check that the get/set correctly manages higher Nyquist
    zones in the C API. Note that it doesn't guarantee the RFDC is actually
    doing the right thing - just API consistency.
    """

    zone = int(frequency / 2.5e9) + 1

    d = live_session.query(rfmux.CRS).one()
    await d.resolve()

    await d.set_analog_bank(high=module > 4)

    if zone <= 2:
        # Works as expected without API warning
        await d.set_nco_frequency(frequency, module=module)
    else:
        # Nyquist zones 3 and 4 currently emit a bogus warning from the RFDC
        # API
        with pytest.warns(UserWarning):
            await d.set_nco_frequency(frequency, module=module)

    assert (await d.get_nco_frequency(module=module)) == pytest.approx(
        frequency, abs=0.001
    )  # expect agreement below 1 mHz

    dac_filter = await d.get_dac_compensation_filter(module=module)

    match(zone):
        case 1:
            assert dac_filter==d.DAC_COMPENSATION_FILTER.MODE1
        case 2:
            assert dac_filter==d.DAC_COMPENSATION_FILTER.MODE2
        case 3:
            assert dac_filter==d.DAC_COMPENSATION_FILTER.MODE2
        case 4:
            assert dac_filter==d.DAC_COMPENSATION_FILTER.MODE1
        case _:
            assert False


@pytest.mark.asyncio
async def test_set_get_decimation(live_session):
    """
    Does set_decimation and get_decimation operate correctly?
    """

    d = live_session.query(rfmux.CRS).one()
    await d.resolve()

    # clearly this should fail - requesting absurd amounts of bandwidth
    with pytest.raises(rfmux.tuber.TuberRemoteError):
        await d.set_decimation(stage=0, short=False, modules=[1, 2, 3, 4])

    # we should be able to stream nothing, but can't get a value from get_decimation like this
    await d.set_decimation(stage=6, short=False, modules=[])
    assert (await d.get_decimation()) is None

    # With short packets, we can go all the way down to FIR stage 0
    # (but only with one module at a time)
    for module in range(1, 5):
        for stage in range(0, 7):
            await d.set_decimation(stage=stage, short=True, modules=[module])
            assert (await d.get_decimation()) == stage

    # With longer packets, we can only do this at FIR stage 3 and up
    for module in range(1, 5):
        for stage in range(3, 7):
            await d.set_decimation(stage=stage, short=False, modules=[module])
            assert (await d.get_decimation()) == stage

    # be polite
    await d.set_decimation(stage=6, short=False, modules=[1, 2, 3, 4])


@pytest.mark.asyncio
async def test_py_get_samples_long_and_short(live_session):
    """
    Does py_get_samples play nice with long and short packets?
    """

    d = live_session.query(rfmux.CRS).one()
    await d.resolve()

    await d.set_decimation(stage=6, short=True, modules=[1])

    # Try with "channel" specified and low
    x = await d.py_get_samples(10, channel=1, module=1)
    assert len(x.i) == 10 and len(x.q) == 10

    # Try with "channel" specified and too high
    with pytest.raises(ValueError):
        x = await d.py_get_samples(10, channel=129, module=1)

    # Now don't specify "channel" - should get an array of 128 channels
    x = await d.py_get_samples(10, module=1)
    assert len(x.i) == 128 and len(x.q) == 128

    # Back to long packets
    await d.set_decimation(stage=6, short=False, modules=[1, 2, 3, 4])

    # Try with "channel" specified and low
    x = await d.py_get_samples(10, channel=1, module=1)
    assert len(x.i) == 10 and len(x.q) == 10

    # Now don't specify "channel" - should get an array of 1024 channels
    x = await d.py_get_samples(10, module=1)
    assert len(x.i) == 1024 and len(x.q) == 1024


if __name__ == "__main__":
    pytest.main([__file__])
