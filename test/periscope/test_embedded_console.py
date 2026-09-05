"""
Regression test for the embedded (in-process) IPython console.

ipykernel 7.x broke the in-process kernel qtconsole uses (do_execute
dereferences shell_channel_thread.parent_thread, which only exists for
real ZMQ kernels) — every execute in Periscope's embedded console then
raised AttributeError + queue.Empty.  pyproject pins ipykernel<7; this
test fails if the pin is ever relaxed while the breakage persists.
"""

import os
import time

import pytest


pytest.importorskip("PyQt6")
pytest.importorskip("qtconsole")

from PyQt6 import QtWidgets  # noqa: E402



def test_inprocess_kernel_executes(qt_app):
    from qtconsole.inprocess import QtInProcessKernelManager

    km = QtInProcessKernelManager()
    km.start_kernel()
    kc = km.client()
    kc.start_channels()
    try:
        # This is the call path the embedded console uses; on broken
        # ipykernel it raises queue.Empty (no reply — handler crashed).
        kc.execute("_console_probe = 6 * 7", silent=False)

        deadline = time.monotonic() + 5.0
        ns = km.kernel.shell.user_ns
        while time.monotonic() < deadline:
            qt_app.processEvents()
            if ns.get("_console_probe") == 42:
                break
            time.sleep(0.01)
        assert ns.get("_console_probe") == 42, \
            "in-process kernel did not execute code — embedded console broken"
    finally:
        kc.stop_channels()
        km.shutdown_kernel()
