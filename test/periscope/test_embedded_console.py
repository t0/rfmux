"""The embedded IPython console: cells run on the interpreter thread and
the GUI keeps going while they do."""

import re
import threading
import time

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("qtconsole")

from PyQt6 import QtCore  # noqa: E402


@pytest.fixture
def console(qt_app):
    """(kernel manager, client, run) where run(code, name) executes a cell
    and returns user_ns[name] once it appears."""
    from rfmux.tools.periscope.console_kernel import ConsoleKernelManager, Interpreter

    interpreter = Interpreter()
    km = ConsoleKernelManager(interpreter)
    km.start_kernel()
    kc = km.client()
    kc.start_channels()
    ns = km.kernel.shell.user_ns

    def run(code, name, timeout_s=5.0):
        kc.execute(code, silent=False)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and name not in ns:
            qt_app.processEvents()
            time.sleep(0.005)
        return ns.get(name)

    yield km, kc, run
    kc.stop_channels()
    km.shutdown_kernel()
    interpreter.close()


def test_sync_cell_executes(console):
    _, _, run = console
    assert run("_probe = 6 * 7", "_probe") == 42


def test_await_cell_executes(console):
    _, _, run = console
    code = "async def _f():\n    return 6 * 7\n_probe = await _f()"
    assert run(code, "_probe") == 42


def test_cells_run_off_the_gui_thread(console):
    _, _, run = console
    assert run("import threading\n_t = threading.current_thread().name", "_t") == "interpreter"


def test_gui_event_loop_runs_during_a_cell(console):
    _, _, run = console
    ticks = []
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(10)
    run("import time\ntime.sleep(0.3)\n_done = 1", "_done")
    timer.stop()
    assert len(ticks) >= 10


def test_back_to_back_cells_both_complete(console):
    _, kc, run = console
    kc.execute("_first = 1", silent=False)
    assert run("_second = 2", "_second") == 2
    assert run("_third = _first + _second", "_third") == 3


def test_console_widget_shows_a_prompt_and_executes_typed_input(console, qt_app):
    from rfmux.tools.periscope.console_kernel import PeriscopeConsole
    km, kc, _ = console
    widget = PeriscopeConsole()
    widget.kernel_client = kc
    early = widget.run("early = 'queued before the console was ready'")

    def prompt_number(timeout_s=10.0):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            found = re.findall(r"In \[(\d+)\]:", widget._control.toPlainText())
            if found:
                return int(found[-1])
            qt_app.processEvents()
            time.sleep(0.005)
        pytest.fail("no prompt appeared: " + widget._control.toPlainText())

    # The in-process shell is a process-wide singleton, so the count carries
    # across tests; only its advance is the contract. Wait for the queued
    # cell so the advance measured below is the typed cell's alone.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not early.done():
        qt_app.processEvents()
        time.sleep(0.005)
    first = prompt_number()
    widget.execute("typed = 6 * 7")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and prompt_number() == first:
        qt_app.processEvents()
        time.sleep(0.005)
    assert prompt_number() == first + 1, widget._control.toPlainText()
    assert km.kernel.shell.user_ns["typed"] == 42
    assert early.done() and km.kernel.shell.user_ns["early"].startswith("queued")
    assert "early = 'queued before the console was ready'" in widget._control.toPlainText()


@pytest.fixture
def widget(console, qt_app):
    """(console widget, pump) with the first prompt already shown; pump(pred)
    processes Qt events until pred() holds."""
    from rfmux.tools.periscope.console_kernel import PeriscopeConsole
    _, kc, _ = console
    w = PeriscopeConsole()
    w.kernel_client = kc

    def pump(pred, timeout_s=10.0):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not pred():
            qt_app.processEvents()
            time.sleep(0.005)
        assert pred(), w._control.toPlainText()

    pump(lambda: re.search(r"In \[\d+\]:", w._control.toPlainText()))
    return w, pump


def test_run_shows_the_cell_like_typed_input_and_records_it(widget, console):
    w, pump = widget
    km, _, _ = console
    future = w.run("# from a panel\nran = 6 * 7")
    pump(future.done)
    text = w._control.toPlainText()
    assert "# from a panel" in text and "ran = 6 * 7" in text
    assert km.kernel.shell.user_ns["ran"] == 42
    assert "ran = 6 * 7" in km.kernel.shell.history_manager.input_hist_raw[-1]
    pump(lambda: re.search(r"In \[\d+\]: $", w._control.toPlainText()))


def test_run_puts_back_a_half_typed_line(widget):
    w, pump = widget
    w.input_buffer = "half typed"
    future = w.run("x = 1")
    pump(future.done)
    pump(lambda: w.input_buffer == "half typed")


def test_runs_queue_behind_a_running_cell(widget, console):
    w, pump = widget
    km, _, _ = console
    w.run("import asyncio\nawait asyncio.sleep(0.3)\nfirst = 1")
    second = w.run("second = first + 1")
    pump(second.done)
    assert km.kernel.shell.user_ns["second"] == 2


def test_cancel_interrupts_the_running_cell(widget, console):
    w, pump = widget
    km, _, _ = console
    future = w.run("import asyncio\nawait asyncio.sleep(30)\nfinished = True")
    pump(lambda: km.kernel.cell_task is not None)
    future.cancel()
    pump(future.done)
    assert future.cancelled()
    assert "finished" not in km.kernel.shell.user_ns
    pump(lambda: re.search(r"In \[\d+\]: $", w._control.toPlainText()))


def test_interrupt_kernel_stops_a_typed_await_cell(widget, console):
    w, pump = widget
    km, _, _ = console
    w.execute("import asyncio\nawait asyncio.sleep(30)\ntyped_finished = True")
    pump(lambda: km.kernel.cell_task is not None)
    km.interrupt_kernel()
    pump(lambda: not w._executing)
    assert "typed_finished" not in km.kernel.shell.user_ns


def test_on_done_calls_back_on_the_gui_thread(widget):
    from rfmux.tools.periscope.console_kernel import on_done
    w, pump = widget
    seen = []
    on_done(w.run("value = 6 * 7"),
            lambda f: seen.append((threading.current_thread(), f.exception())))
    pump(lambda: seen)
    assert seen == [(threading.main_thread(), None)]


def test_on_done_delivers_the_cells_exception(widget):
    from rfmux.tools.periscope.console_kernel import on_done
    w, pump = widget
    seen = []
    on_done(w.run("raise RuntimeError('boom')"), lambda f: seen.append(str(f.exception())))
    pump(lambda: seen)
    assert seen == ["RuntimeError: boom"]


def test_cell_output_reaches_the_console_not_the_gui_threads_output(console, qt_app):
    _, kc, run = console
    streams = []
    kc.iopub_channel.message_received.connect(
        lambda m: streams.append(m["content"]["text"]) if m["msg_type"] == "stream" else None)
    kc.execute("import time\nprint('from the cell')\ntime.sleep(0.2)\n_done = 1", silent=False)
    time.sleep(0.05)
    print("from the GUI thread")
    run("_wait = 1", "_wait")
    for _ in range(20):
        qt_app.processEvents()
        time.sleep(0.005)
    assert streams == ["from the cell\n"]
