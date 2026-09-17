"""The in-process IPython kernel behind the embedded console.

Cells execute on the Interpreter thread, so a long ``await crs.…`` in the
console leaves the Qt event loop free to keep the display moving. A panel
runs its board work by handing the console a cell, exactly as if the user
had typed it: same prompt, highlighting, output and history entry.
"""

import ast
import asyncio
import collections
import concurrent.futures
import inspect
import sys
import threading
import time
from contextlib import contextmanager

from PyQt6 import QtCore
from ipykernel.inprocess.ipkernel import InProcessKernel
from qtconsole.inprocess import QtInProcessKernelClient, QtInProcessKernelManager
from qtconsole.rich_jupyter_widget import RichJupyterWidget


class _GuiThread(QtCore.QObject):
    """Relays (future, callback) pairs to the thread that created it."""

    call = QtCore.pyqtSignal(object, object)

    def __init__(self):
        super().__init__()
        self.call.connect(self._run, QtCore.Qt.ConnectionType.QueuedConnection)

    def _run(self, future, callback):
        callback(future)


_gui_thread = None


def on_done(future: concurrent.futures.Future, callback) -> None:
    """Call callback(future) on the GUI thread once *future* completes.
    A cancelled future is not reported. First use must be from the GUI thread."""
    global _gui_thread
    if _gui_thread is None:
        _gui_thread = _GuiThread()
    future.add_done_callback(
        lambda f: None if f.cancelled() else _gui_thread.call.emit(f, callback))


class Interpreter:
    """One thread with one asyncio loop: where the session's Python runs."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._serve, name="interpreter",
                                        daemon=True)
        self._thread.start()

    def _serve(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def run(self, code: str, namespace: dict,
            filename: str = "<periscope>") -> concurrent.futures.Future:
        """Execute *code* in *namespace* on the loop; ``await`` is allowed at
        top level. Used when there is no console to hand the cell to."""
        compiled = compile(code, filename, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)

        async def _run():
            result = eval(compiled, namespace)
            if inspect.isawaitable(result):
                await result

        return self.submit(_run())

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)


class CellFuture(concurrent.futures.Future):
    """A cell a panel handed to the console: resolves with the reply content,
    fails with the cell's exception, and cancel() interrupts it."""

    def __init__(self, console, code: str):
        super().__init__()
        self.console, self.code = console, code
        self.msg_id = None
        self.saved_input = ""
        self.interrupted = False

    def cancel(self):
        if self.done():
            return False
        if self.msg_id is None:  # not yet handed to the kernel
            return super().cancel()
        self.interrupted = True
        self.console.kernel_client.interrupt(self.msg_id)
        return True

    def _finish(self, content: dict):
        if self.interrupted:
            super().cancel()
        elif content["status"] == "ok":
            self.set_result(content)
        else:
            self.set_exception(RuntimeError(
                f"{content.get('ename', 'error')}: {content.get('evalue', '')}"))


class PeriscopeConsole(RichJupyterWidget):
    """The console widget; panels run their Python through it with run()."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._queue = collections.deque()
        self._running = None

    def run(self, code: str) -> CellFuture:
        """Execute *code* as though typed, after any cell already running.
        A half-typed line is set aside and put back afterwards. A cell that
        does not parse is a bug in whatever built it, and raises here."""
        compile(code, "<periscope>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        future = CellFuture(self, code)
        self._queue.append(future)
        self._pump()
        return future

    def _ready(self) -> bool:
        # Idle, and past the startup handshake (banner, then a silent execute
        # that fetches the prompt number).
        return (self._running is None and not self._executing
                and not self._starting and not self._prompt_requested)

    def _pump(self):
        while self._queue and self._ready():
            future = self._queue.popleft()
            if future.cancelled():
                continue
            future.saved_input = self.input_buffer
            before = set(self._request_info["execute"])
            self.execute(future.code, hidden=False, interactive=False)
            (future.msg_id,) = set(self._request_info["execute"]) - before
            self._running = future

    def _handle_execute_reply(self, msg):
        super()._handle_execute_reply(msg)
        future = self._running
        if future is not None and msg["parent_header"].get("msg_id") == future.msg_id:
            self._running = None
            if future.saved_input:
                self.input_buffer = future.saved_input
            future._finish(msg["content"])
        self._pump()


class _ShellChannel:
    """Stands in for the shell channel thread a ZMQ kernel owns.

    ipykernel 7 decides whether an ``await`` cell may install its SIGINT
    handler by comparing the current thread with the shell channel's
    parent thread. The in-process kernel has no shell channel thread, so
    the comparison raises. Only that attribute is served here; cells run
    on the interpreter thread, so the handler is never installed.
    """

    def __init__(self):
        self.parent_thread = threading.current_thread()


class _ThreadStream:
    """Writes from *thread* go to the kernel's stream, all others to *other*."""

    def __init__(self, thread, kernel_stream, other):
        self._thread, self._kernel, self._other = thread, kernel_stream, other

    def _target(self):
        return self._kernel if threading.current_thread() is self._thread else self._other

    def write(self, text):
        return self._target().write(text)

    def flush(self):
        self._kernel.flush()
        self._other.flush()

    def __getattr__(self, name):
        return getattr(self._other, name)


class ConsoleKernel(InProcessKernel):
    def __init__(self, **traits):
        super().__init__(**traits)
        self.shell_channel_thread = _ShellChannel()
        # The asyncio task running the current await-ing cell, so it can be
        # cancelled: the in-process kernel has no interrupt of its own.
        self.cell_task = None
        run_cell_async = self.shell.run_cell_async

        async def tracked(*args, **kwargs):
            self.cell_task = asyncio.current_task()
            try:
                return await run_cell_async(*args, **kwargs)
            finally:
                self.cell_task = None

        self.shell.run_cell_async = tracked

    # Subshells need a real shell channel thread. With the stand-in present,
    # ipykernel would otherwise route every request through its manager.
    _supports_kernel_subshells = property(lambda self: False)

    def interrupt(self):
        """Cancel the running cell. Call on the interpreter loop."""
        if self.cell_task is not None:
            self.cell_task.cancel()

    @contextmanager
    def _redirected_io(self):
        # Only the executing thread's output belongs to the cell; the GUI
        # thread keeps printing to the terminal.
        out, err = sys.stdout, sys.stderr
        here = threading.current_thread()
        sys.stdout = _ThreadStream(here, self.stdout, out)
        sys.stderr = _ThreadStream(here, self.stderr, err)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = out, err

    def _input_request(self, prompt, ident, parent, password=False):
        # The stock implementation pumps Qt events while it waits, which
        # only the GUI thread may do. The GUI thread is free, so wait.
        self.raw_input_str = None
        sys.stdout.flush()
        sys.stderr.flush()
        msg = self.session.msg("input_request",
                               dict(prompt=prompt, password=password), parent)
        for frontend in self.frontends:
            if frontend.session.session == parent["header"]["session"]:
                frontend.stdin_channel.call_handlers(msg)
                break
        else:
            return ""
        while self.raw_input_str is None:
            time.sleep(0.01)
        return self.raw_input_str


class ConsoleKernelClient(QtInProcessKernelClient):
    """Hands each request to the interpreter loop, one at a time."""

    interpreter: Interpreter = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._one_at_a_time = asyncio.Lock()
        self._current_msg_id = None
        self._cancel_on_start = set()

    def _dispatch_to_kernel(self, msg):
        if self.kernel is None:
            raise RuntimeError("Cannot send request. No kernel exists.")
        self.interpreter.submit(self._dispatch(msg))

    async def _dispatch(self, msg):
        # The in-process socket is one queue for both directions, so every
        # touch of it happens here, on the loop, under the lock.
        async with self._one_at_a_time:
            self._current_msg_id = msg["header"]["msg_id"]
            if self._current_msg_id in self._cancel_on_start:
                self._cancel_on_start.discard(self._current_msg_id)
                asyncio.ensure_future(self._cancel_once_running(self._current_msg_id))
            stream = self.kernel.shell_stream
            self.session.send(stream, msg)
            parts = stream.recv_multipart()
            try:
                await self.kernel.dispatch_shell(parts)
            finally:
                self._current_msg_id = None
            _idents, reply = self.session.recv(stream, copy=False)
            # call_handlers_later would arm a QTimer on this thread, which has
            # no Qt event loop, so the reply would never arrive. The signal
            # emitted here is queued to the widget's thread by Qt itself.
            self.shell_channel.call_handlers(reply)

    async def _cancel_once_running(self, msg_id):
        while self._current_msg_id == msg_id and self.kernel.cell_task is None:
            await asyncio.sleep(0.01)
        if self._current_msg_id == msg_id:
            self.kernel.interrupt()

    def interrupt(self, msg_id: str):
        """Cancel the cell *msg_id* started, now or as soon as it starts."""
        def on_loop():
            if self._current_msg_id == msg_id:
                self.kernel.interrupt()
            else:
                self._cancel_on_start.add(msg_id)
        self.interpreter.loop.call_soon_threadsafe(on_loop)


class ConsoleKernelManager(QtInProcessKernelManager):
    client_class = __module__ + ".ConsoleKernelClient"

    def __init__(self, interpreter: Interpreter, **kwargs):
        super().__init__(**kwargs)
        self.interpreter = interpreter

    def start_kernel(self, **kwds):
        self.kernel = ConsoleKernel(parent=self, session=self.session)

    def interrupt_kernel(self):
        """Ctrl-C in the console: cancel the running cell."""
        self.interpreter.loop.call_soon_threadsafe(self.kernel.interrupt)

    def client(self, **kwargs):
        client = super().client(**kwargs)
        client.interpreter = self.interpreter
        return client
