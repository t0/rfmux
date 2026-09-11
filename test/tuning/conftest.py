"""Fixtures for the tuning tests.

Most of ``test/tuning`` works on synthetic traces and needs nothing from here.
The tests that drive the real flow — a sweep, a schedule, a bias — run against
the standard simulated array, ``rfmux.mock.standard_array``, built once per
test module: building it is cheap, but a second ``load_session`` in one
process detaches the first board's objects, so one module is one array.

The array is served over RPC alone. No UDP is streamed, which keeps these in
the quick tier (see test/README.md).
"""

import asyncio

import pytest

from rfmux.mock.standard_array import standard_array


@pytest.fixture(scope="module")
def standard_array_board():
    """``(loop, crs, catalog)``: the standard array, biased by the simulator.

    The loop is the one the board was resolved on; drive the board with
    ``loop.run_until_complete(...)`` so every call shares it.
    """
    loop = asyncio.new_event_loop()
    crs, catalog = loop.run_until_complete(standard_array())
    yield loop, crs, catalog
    loop.close()
