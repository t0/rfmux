#!/usr/bin/env -S pytest-3 -s
"""
Basic bring-up tests

Please see README for hints on using and extending these test cases.
"""

import pytest
import socket
import requests

from .crs_qc import render_markdown, ResultTable


# Ensure network can resolve the board. Does not use rfmux.
@pytest.mark.qc
@pytest.mark.asyncio
async def test_communications(d, request, shelf, check):
    """
    ## Basic board communications tests

    Tests the following:

    * Is the CRS board alive?
    * Can we send commands to it?
    * Can we retrieve multicast samples from it?

    If any of these tests fail, you should stop here and debug your test setup
    before proceeding with more complex tests.
    """

    rt = ResultTable("Check", "Status")
    hostname = f'rfmux{request.config.getoption("--serial")}.local'

    # 1. Can we resolve the board?
    try:
        address = socket.gethostbyname(hostname)
        rt.pass_("Name resolution", "Pass")
    except Exception as e:
        rt.fail("Name resolution", e.message)
        check.fail(e.message)

    # 2. Can we make a bare/empty HTTP request to it?
    try:
        text = requests.post(f"http://{hostname}/tuber", data="[]").text
        rt.pass_("HTTP POST", "Pass")
    except Exception as e:
        rt.fail("HTTP POST", e.message)
        check.fail(e.message)

    # 3. Can we receive multicast packets directly?
    try:
        x = await d.py_get_samples(1, module=1, channel=1)
        rt.pass_("Multicast reception", "Pass")
    except Exception as e:
        rt.fail("Multicast reception", str(e))
        check.fail(str(e))

    shelf["sections"][request.node.name] = [
        render_markdown(test_communications.__doc__),
        rt,
    ]


@pytest.mark.asyncio
async def test_metadata(d, request, shelf):
    """
    ## Metadata Retrieval

    The board includes helper functions to retrieve its own metadata. This
    function tests the structure of this metadata and also displays the results
    for record-keeping.
    """

    rt = ResultTable("Attribute", "Value")
    r = await d.get_firmware_release()
    for k in r:
        rt.row(k, getattr(r, k))

    shelf["sections"][request.node.name] = [
        render_markdown(test_metadata.__doc__),
        rt,
    ]
