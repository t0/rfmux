"""
Basic bring-up tests

Please see README for hints on using and extending these test cases.
"""

import inspect
import socket
import requests

from .crs_qc import render_markdown, ResultTable


# Ensure network can resolve the board. Does not use rfmux.
async def communications(d, report):
    """
    ## Basic board communications tests

    Tests the following:

    * Is the CRS board alive?
    * Can we send commands to it?
    * Can we retrieve multicast samples from it?

    If any of these tests fail, you should stop here and debug your test setup
    before proceeding with more complex tests.
    """
    report.append(render_markdown(inspect.getdoc(communications)))

    report.append(rt := ResultTable("Check", "Status"))
    hostname = f"rfmux{d.serial}.local"

    # 1. Can we resolve the board?
    try:
        address = socket.gethostbyname(hostname)
        rt.pass_("Name resolution", "Pass")
    except Exception as e:
        rt.fail("Name resolution", str(e))
        report.fail(f"{type(e).__name__}: {e}")

    # 2. Can we make a bare HTTP request to it?
    try:
        # Coming up after this is a py_get_samples() call. Because
        # py_get_samples uses "module=1", we sneak in a set_analog_bank(False)
        # call here to ensure we aren't inadvertently set up for high banking.
        text = requests.post(
            f"http://{hostname}/tuber",
            json={"object": "Dfmux", "method": "set_analog_bank", "args": [False]},
        ).text
        rt.pass_("HTTP POST", "Pass")
    except Exception as e:
        rt.fail("HTTP POST", str(e))
        report.fail(f"{type(e).__name__}: {e}")

    # 3. Can we receive multicast packets directly?
    try:
        x = await d.py_get_samples(1, module=1, channel=1)
        rt.pass_("Multicast reception", "Pass")
    except Exception as e:
        rt.fail("Multicast reception", str(e))
        report.fail(f"{type(e).__name__}: {e}")


async def metadata(d, report):
    """
    ## Metadata Retrieval

    The board includes helper functions to retrieve its own metadata. This
    function tests the structure of this metadata and also displays the results
    for record-keeping.
    """
    report.append(render_markdown(inspect.getdoc(metadata)))

    report.append(rt := ResultTable("Attribute", "Value"))
    r = await d.get_firmware_release()
    for k, v in vars(r).items():
        rt.row(k, v)
