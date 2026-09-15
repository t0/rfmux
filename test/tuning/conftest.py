"""Share one standard mock array per test module, served over RPC without UDP.

A second ``load_session`` detaches the first session's board objects, so each
module uses one array.
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
