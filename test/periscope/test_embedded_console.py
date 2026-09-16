"""
The embedded (in-process) IPython console executes both plain and
``await`` cells.

Periscope's console is used almost entirely for ``await crs.<method>()``,
so a kernel that runs synchronous cells but fails on asynchronous ones
(ipykernel 7's in-process kernel, which has no shell channel thread to
consult) is broken for its purpose.
"""

import time

import pytest


pytest.importorskip("PyQt6")
pytest.importorskip("qtconsole")


def _run_cell(qt_app, code, name, timeout_s=5.0):
    """Execute *code* in a fresh in-process kernel and return user_ns[name]."""
    from rfmux.tools.periscope.console_kernel import ConsoleKernelManager

    km = ConsoleKernelManager()
    km.start_kernel()
    kc = km.client()
    kc.start_channels()
    try:
        kc.execute(code, silent=False)
        ns = km.kernel.shell.user_ns
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and name not in ns:
            qt_app.processEvents()
            time.sleep(0.01)
        return ns.get(name)
    finally:
        kc.stop_channels()
        km.shutdown_kernel()


def test_sync_cell_executes(qt_app):
    assert _run_cell(qt_app, "_probe = 6 * 7", "_probe") == 42


def test_await_cell_executes(qt_app):
    code = "async def _f():\n    return 6 * 7\n_probe = await _f()"
    assert _run_cell(qt_app, code, "_probe") == 42
