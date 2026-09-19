import pytest
import rfmux
import gc
import os
import socket
import sys
import pytest_asyncio

# Fixtures that can only be satisfied by a real board. Requesting one — directly
# or transitively — is what makes a test part of the hardware tier.
#
# "serial" is test_mock_vs_real.py's own gate; it compares a mock CRS against a
# live one, so even its crs_mock fixture needs a board behind it.
HARDWARE_FIXTURES = frozenset({"live_session", "crs", "serial"})

# --serial and --tier are declared in the ROOT conftest.py, not here.
# pytest only honours pytest_addoption from an initial conftest, so options
# declared in this file are invisible whenever the arguments do not point into
# test/ — e.g. running pytest from a subdirectory.


@pytest.fixture(scope="session", autouse=True)
def _isolate_qsettings(tmp_path_factory):
    """Keep the suite out of the developer's real Periscope preferences.

    QSettings is a per-user store, not a per-process one, so anything that
    writes it during a test writes ~/.config/rfmux/periscope.conf for real.
    That is not hypothetical: SessionManager.start_session() records the
    session path, and a test calling it with tmp_path left the user's
    "last session" pointing at /tmp/pytest-of-.../session_test — which then
    turned up as the default in Periscope's session dialog, long after the
    test run was over.

    Redirect the store to a temp file for the whole session. Tests that
    genuinely exercise settings still work; they just stop being destructive.
    Skipped entirely when PyQt6 is unavailable (the portable tier).

    Patching the module's one constructor rather than calling
    QSettings.setPath(): setPath only takes effect for objects created before
    Qt has resolved the native location, which it already has by the time any
    fixture runs, so it silently does nothing here. Every read and write in the
    package goes through settings._get_settings(), so this is the whole surface.
    """
    try:
        from PyQt6.QtCore import QSettings
    except ImportError:
        yield
        return

    from rfmux.tools.periscope import settings as periscope_settings

    store = tmp_path_factory.mktemp("qsettings") / "periscope.ini"
    original = periscope_settings._get_settings
    periscope_settings._get_settings = (
        lambda: QSettings(str(store), QSettings.Format.IniFormat))
    yield
    periscope_settings._get_settings = original


def pytest_collection_modifyitems(config, items):
    """Mark hardware tests by the fixtures they request.

    The board-dependent tests are gated by a skip inside ``live_session``,
    which makes them impossible to select: there is nothing to write after
    ``-m``. Deriving the marker from the fixture graph keeps the hardware tier
    addressable (``-m hardware``, ``-m "not hardware"``) without asking anyone
    to remember a decorator that duplicates what the fixtures already say.
    """
    for item in items:
        if HARDWARE_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.hardware)


def _port_holder(port):
    """Return a description of whatever holds ``port``, or None if it is free.

    Probes with a PLAIN bind and no socket options on purpose. The streamer sets
    ``SO_REUSEPORT``, so a probe that also set it would happily bind alongside an
    existing listener and report the port as free — which is precisely the
    condition being looked for. Without it the kernel refuses with EADDRINUSE.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind(("", port))
        return None
    except OSError as exc:
        return f"{exc.strerror or exc}"
    finally:
        probe.close()


@pytest.fixture(scope="session", autouse=True)
def _streamer_ports_free(request):
    """Refuse to start the acquisition tier if the streamer ports are taken.

    The mock streamer binds fixed ports (9876 slow, 9877 PFB) with
    ``SO_REUSEPORT``, so a second listener does not collide — the bind just
    succeeds. What happens next is decided by the transport, and the mock's is
    unicast to loopback: the kernel hands each datagram to exactly ONE of the
    bound sockets, chosen by hash of the sender's 4-tuple. Measured over 20
    sender ports: 6 went to the first listener, 14 to the second, never once
    split. So one run takes the entire stream and the other sees nothing, and
    which one wins flips between launches. (A real board multicasts instead,
    and every socket joined to the group gets its own copy — that is why
    multiple readers are fine on hardware and fatal here.) Neither errors. The
    starved run reports it as a pulse-detector fault: "no matched pairs",
    thousands of phantom pulses, nonsensical elapsed times. That misdirection
    cost two full investigations on this branch, so the condition is now named
    at the point it can still be explained.

    The usual cause is a previous run that has not finished dying — its servers
    are still being reaped, or a test leaked a reader thread. A live Periscope
    listening on the same ports will do it too.

    Checked once per session rather than per test: mid-session the suite's own
    readers legitimately hold these ports, so a per-test probe would fail on the
    run's own traffic.

    Escape hatch, for when you know two readers are fine:
    ``--allow-busy-streamer-ports``.
    """
    if request.config.getoption("allow_busy_streamer_ports", default=False):
        return

    # session.items is the list that survived deselection, which is what
    # matters: read at collection time it still holds every marker-excluded
    # test, so `pytest --tier=quick` would trip a guard meant for the
    # acquisition tier.
    if not any(item.get_closest_marker("slow_acquisition")
               for item in getattr(request.session, "items", ())):
        return

    from rfmux import streamer

    busy = {}
    for port in (streamer.STREAMER_PORT, streamer.PFB_STREAMER_PORT):
        holder = _port_holder(port)
        if holder is not None:
            busy[port] = holder
    if not busy:
        return

    detail = "; ".join(f"{port} ({why})" for port, why in sorted(busy.items()))
    pytest.exit(
        "Streamer port(s) already in use: " + detail + ".\n"
        "The acquisition tests would share them via SO_REUSEPORT and read only "
        "part of the packet stream, which surfaces as bogus pulse-detection "
        "failures rather than as this clash.\n"
        "Find the holder with:  ss -ulnp | grep -E '9876|9877'\n"
        "A finished pytest run can still hold them while it reaps mock servers; "
        "wait for it to exit, or kill it.\n"
        "Override with --allow-busy-streamer-ports if you know it is safe.",
        returncode=pytest.ExitCode.USAGE_ERROR,
    )


@pytest.fixture
def live_session(pytestconfig):
    if (serial := pytestconfig.getoption("serial")) is None:
        pytest.skip(
            "Use the '--serial' argument to specify a running CRS board for this test."
        )

    return rfmux.load_session(
        f"""
        !HardwareMap
        - !CRS {{ serial: "{serial}" }}
        """
    )

@pytest_asyncio.fixture
async def crs(live_session):
    crs = live_session.query(rfmux.CRS).one()
    await crs.resolve()

    # setup: instill politeness
    await crs.set_timestamp_port(crs.TIMESTAMP_PORT.TEST)

    yield crs

    # teardown: restore politeness
    await crs.set_analog_bank(high=False)
    await crs.set_decimation(stage=6, short=False, module=[1,2,3,4])


# ── Qt: one fixture and one pair of helpers for the whole suite ──────
#
# origin/main had exactly one qt_app fixture. This branch grew twelve more,
# plus thirteen copies of the offscreen-platform line and four of the spin
# helpers, because each new GUI test file started from the last one. They
# were identical apart from scope, so they live here now.
#
# QT_QPA_PLATFORM is set at import time rather than in the fixture: conftest
# is imported before any test module, so this still precedes every
# QApplication construction, including ones made at module scope.



os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qt_app():
    """The QApplication every GUI test shares.

    importorskip lives in the BODY, not at module level: test/mock/ has
    files that are only partly Qt, and skipping the whole module would
    take their non-Qt tests with it.
    """
    QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _free_dropped_widgets_between_tests(request):
    """Free the widgets a GUI test dropped, here and nowhere else.

    A panel is a reference cycle (its slots hold it), so one a test
    dropped waits for the cyclic collector, which otherwise runs
    whenever allocation counts say so.  Inside ``processEvents`` that
    frees plot items while Qt is dispatching to their scene; on a worker
    thread it destroys widgets off the GUI thread.  Either one
    segfaults a later test.

    So the collector is off while a GUI test runs and is run once the
    test is over, on the main thread with nothing being dispatched.
    That is in ``pytest_runtest_logfinish`` below, not here: pytest
    still holds the test's fixture values when the fixtures finish, and
    a widget a fixture yielded would survive a collection run here and
    be promoted to generation 2, where nothing frees it until a full
    collection happens to run.  Everything the test made is still in
    generation 0, because the collector has not run, and
    ``gc.collect(1)`` frees it without scanning the whole interpreter.
    """
    if "qt_app" in request.fixturenames:
        gc.disable()
    yield


def pytest_runtest_logfinish(nodeid, location):
    """Free what the GUI test that just ended left behind; pytest has
    dropped its fixture values by now."""
    if not gc.isenabled():
        gc.collect(1)
        gc.enable()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """A GUI test fails when a Qt callback of its raised.

    An exception escaping a slot or a reimplemented virtual (a
    closeEvent, say) never reaches the test: PyQt prints it through
    sys.excepthook, where pytest's capture hides it, and leaves it in
    sys.last_exc, whose traceback keeps every widget the frames touched
    alive.
    """
    if "qt_app" not in item.fixturenames:
        yield
        return
    seen = []
    prev = sys.excepthook

    def record(exc_type, exc, tb):
        seen.append(exc)
        prev(exc_type, exc, tb)

    sys.excepthook = record
    outcome = yield
    sys.excepthook = prev
    sys.last_exc = sys.last_value = sys.last_traceback = None
    if seen and outcome.excinfo is None:
        outcome.force_exception(pytest.fail.Exception(
            f"a Qt callback raised {seen[0]!r}", pytrace=False))
