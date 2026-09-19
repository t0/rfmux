"""The Control worker reads the board, answers a write with the
channel's re-read, and stops when asked."""

import asyncio
import time

import pytest

pytest.importorskip("PyQt6")

import rfmux  # noqa: E402
from rfmux.tools.periscope.tasks import (  # noqa: E402
    ToneControlSignals, ToneControlTask)

SESSION = """
!HardwareMap
- !flavour "rfmux.mock"
- !CRS { serial: "0000", hostname: "127.0.0.1" }
"""


@pytest.fixture(scope="module")
def mock_crs():
    loop = asyncio.new_event_loop()
    session = rfmux.load_session(SESSION)
    crs = session.query(rfmux.CRS).one()
    loop.run_until_complete(crs.resolve())
    loop.run_until_complete(crs.set_nco_frequency(500e6, module=1))
    loop.run_until_complete(crs.set_frequency(1e6, channel=1, module=1))
    loop.run_until_complete(crs.set_amplitude(0.01, channel=1, module=1))
    yield loop, crs
    loop.close()


def _wait_for(qt_app, received, predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qt_app.processEvents()
        if any(predicate(r) for r in received):
            return
        time.sleep(0.02)
    raise AssertionError(f"nothing matched within {timeout_s} s: {received}")


def test_reads_writes_and_stops(qt_app, mock_crs):
    loop, crs = mock_crs
    signals = ToneControlSignals()
    received, errors = [], []
    signals.values_ready.connect(lambda m, r: received.append(r))
    signals.error.connect(errors.append)
    task = ToneControlTask(crs, 1, [1, 2], signals)
    task.start()
    try:
        _wait_for(qt_app, received, lambda r: set(r["channels"]) == {1, 2})
        first = received[-1]
        assert first["nco"] == 500e6
        assert first["channels"][1]["frequency"] == 1e6
        assert first["dac_scale"] is not None

        task.write(2, {"frequency": -2e6, "amplitude": 0.02})
        _wait_for(qt_app, received,
                  lambda r: r["channels"].get(2, {}).get("frequency") == -2e6)
        assert loop.run_until_complete(
            crs.get_amplitude(channel=2, module=1)) == 0.02

        task.set_nco(501e6)
        _wait_for(qt_app, received, lambda r: r["nco"] == 501e6)
    finally:
        task.stop()
        assert task.wait(3000)
    assert not task.isRunning()
    assert errors == []
