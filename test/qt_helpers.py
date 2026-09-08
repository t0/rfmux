"""Event-loop helpers shared by the GUI tests.

These live in a normal module rather than in ``conftest.py`` because
conftest is not importable by name — ``from conftest import spin`` picks
up whichever conftest is first on sys.path, which in this repo is the
root one. The ``qt_app`` fixture itself does live in
``test/conftest.py``, where pytest finds it by fixture lookup.
"""

import time


def spin(qt_app, seconds=0.05):
    """Pump the Qt event loop for a fixed spell."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qt_app.processEvents()
        time.sleep(0.005)


def spin_until(qt_app, predicate, timeout=8.0):
    """Pump the event loop until *predicate* holds; True if it did.

    Several call sites rely on the default timeout rather than passing one,
    and one passes it positionally.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qt_app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False
