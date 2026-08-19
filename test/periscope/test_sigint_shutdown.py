"""Ctrl+C must actually stop Periscope.

It used to not. The KeyboardInterrupt surfaced inside whatever Python
callback happened to be running (a GUI timer tick), where the global
excepthook printed it and swallowed it -- so the window stayed up and
the receive thread kept UDP 9876 bound. Because get_multicast_socket
sets SO_REUSEPORT the next launch bound the same port without error and
then lost the entire stream to the process the user thought they had
killed, giving a blank viewer with no explanation.

Verified before the fix: still alive and still holding the port ten
seconds after SIGINT.
"""

import os
import signal
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope import __main__ as periscope_main


@pytest.fixture
def restore_sigint():
    """Never leave a test's SIGINT handler installed in pytest."""
    previous = signal.getsignal(signal.SIGINT)
    yield
    signal.signal(signal.SIGINT, previous)


@pytest.fixture
def quit_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(periscope_main, "_quit_application",
                        lambda: calls.append(True))
    return calls


def test_keyboard_interrupt_is_not_swallowed(quit_calls):
    periscope_main.periscope_excepthook(
        KeyboardInterrupt, KeyboardInterrupt(), None)
    assert quit_calls, \
        "KeyboardInterrupt was logged and ignored; the window stays up " \
        "and its receive thread keeps the streaming port bound"


def test_other_exceptions_are_reported_without_quitting(quit_calls, capsys):
    """A bug in a callback must not take the whole GUI down."""
    try:
        raise ValueError("boom")
    except ValueError as exc:
        periscope_main.periscope_excepthook(type(exc), exc, exc.__traceback__)
    assert not quit_calls
    assert "boom" in capsys.readouterr().err


def test_first_interrupt_asks_second_one_forces(qt_app, restore_sigint,
                                                quit_calls, monkeypatch):
    forced = []
    monkeypatch.setattr(os, "_exit", lambda code: forced.append(code))

    timer = periscope_main.install_sigint_handler()
    try:
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
        assert quit_calls and not forced, "first Ctrl+C should ask politely"
        handler(signal.SIGINT, None)
        assert forced == [130], \
            "a second Ctrl+C must leave even if teardown is stuck"
    finally:
        timer.stop()


def test_a_timer_keeps_the_interpreter_reachable(qt_app, restore_sigint):
    """app.exec() is C++; without this the handler never runs."""
    timer = periscope_main.install_sigint_handler()
    try:
        assert timer.isActive()
        assert 0 < timer.interval() <= 1000
    finally:
        timer.stop()


# The subprocess below is the only test that proves the part that
# actually failed: that a Python signal handler is reached at all while
# the Qt event loop holds the main thread.
_CHILD = textwrap.dedent(
    """
    import os, sys
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6 import QtWidgets
    from rfmux.tools.periscope.__main__ import install_sigint_handler

    app = QtWidgets.QApplication([])
    window = QtWidgets.QWidget()
    window.show()
    wake = install_sigint_handler()
    print("ready", flush=True)
    sys.exit(app.exec())
    """
)


@pytest.mark.portable
def test_sigint_actually_ends_the_event_loop(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(_CHILD)

    child = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        child.send_signal(signal.SIGINT)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pytest.fail(
                "SIGINT did not end the Qt event loop: Periscope would "
                "stay up holding the streaming port, exactly the state "
                "that starved the next launch"
            )
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
