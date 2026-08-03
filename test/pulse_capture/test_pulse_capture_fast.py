"""
Fast (PFB) pulse-capture integration test.

Drives PulseCaptureTask in fast mode against a MockCRS with periodic
QP pulses: the task must configure the PFB streamer for the capture
channels, feed the session from the 1.22 MHz stream via the shared
run_pfb_source, detect pulses, write the HDF5, and tear the PFB
streamer down on stop.
"""

import asyncio
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
pytest.importorskip("h5py")

pytestmark = pytest.mark.slow_acquisition

from PyQt6 import QtWidgets  # noqa: E402

from rfmux.algorithms.measurement.pulse_capture_session import (  # noqa: E402
    CaptureState,
    PulseCaptureSession,
)
from rfmux.algorithms.measurement.pulse_hdf5 import PulseHDF5Reader  # noqa: E402
from rfmux.algorithms.measurement.streamer_config import (  # noqa: E402
    PFB_SAMPLE_RATE,
)
from rfmux.tools.periscope.pulse_capture_task import (  # noqa: E402
    PulseCaptureSignals,
    PulseCaptureTask,
)


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture(scope="module")
def mock_crs():
    from rfmux.mock.helpers import create_mock_crs
    loop = asyncio.new_event_loop()
    # auto_bias_kids parks carriers on the resonators — without it the
    # QP pulses modulate resonators no channel is tuned to, and the
    # streams carry pure noise (mirrors e2e setup_mock).
    crs = loop.run_until_complete(create_mock_crs(
        module=1,
        config={
            "num_resonances": 2,
            "resonator_random_seed": 11,
            "auto_bias_kids": True,
            "bias_amplitude": 0.001,
            "pulse_mode": "periodic",
            "pulse_period": 0.02,
            "pulse_tau_rise": 1e-6,
            "pulse_tau_decay": 1e-3,
            "pulse_amplitude": 3.0,
        },
        verbose=False))
    loop.run_until_complete(asyncio.sleep(2.0))  # stream warm-up
    yield loop, crs
    try:
        loop.run_until_complete(crs.stop_udp_streaming())
    except Exception:
        pass
    loop.close()


def _spin_until(qt_app, predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qt_app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_fast_capture_end_to_end(qt_app, mock_crs, tmp_path):
    loop, crs = mock_crs
    channels = [1, 2]
    path = tmp_path / "fast_capture.h5"

    session = PulseCaptureSession(
        channels=channels, module=1, streamer_mode="fast",
        threshold_sigma=50.0, end_sigma=3.0,
        sample_rate=PFB_SAMPLE_RATE, buf_size=200_000,
        noise_samples=50_000, hdf5_path=path,
        histogram_flush_every=2)
    signals = PulseCaptureSignals()
    events = {"pulses": [], "errors": [], "finished": []}
    signals.pulse_detected.connect(
        lambda ch, idx, s: events["pulses"].append((ch, idx, s)))
    signals.error.connect(lambda m: events["errors"].append(m))
    signals.finished.connect(lambda: events["finished"].append(True))

    task = PulseCaptureTask(session, signals, mode="fast", crs=crs,
                            host="127.0.0.1", module=1)
    task.start()

    assert _spin_until(
        qt_app, lambda: session.state is CaptureState.CAPTURING, 30), \
        f"never reached CAPTURING (state={session.state}, " \
        f"errors={events['errors']})"

    # PFB streamer was configured for our channels by the task
    assert loop.run_until_complete(
        crs.get_pfb_streamer(module=1)) == channels

    assert _spin_until(qt_app, lambda: session.total_pulses >= 2, 60), \
        f"no pulses detected (errors={events['errors']})"

    task.request_stop()
    assert _spin_until(qt_app, lambda: events["finished"], 30), \
        "task never finished"
    task.wait(5000)

    # Teardown contract: PFB streamer disabled again
    assert loop.run_until_complete(
        crs.get_pfb_streamer(module=1)) is None
    assert not events["errors"], events["errors"]

    with PulseHDF5Reader(path) as reader:
        total = sum(reader.pulse_count(c) for c in channels)
        assert total >= 2
        assert reader.metadata.get("streamer_mode") == "fast"
        assert reader.metadata.get("sample_rate_fast") == pytest.approx(
            PFB_SAMPLE_RATE)


def test_both_mode_end_to_end(qt_app, mock_crs, tmp_path):
    """Both-mode task: dual sockets, live matching, dual file, teardown."""
    from rfmux.algorithms.measurement.pulse_capture_dual import (
        DualPulseCaptureSession,
    )
    from rfmux.algorithms.measurement.pulse_capture_session import (
        PulseCaptureConfig,
    )
    from rfmux.algorithms.measurement.streamer_config import (
        slow_sample_rate,
    )

    loop, crs = mock_crs
    channels = [1, 2]
    path = tmp_path / "both_capture.h5"
    cfg = PulseCaptureConfig(threshold_sigma=5.0, end_sigma=1.5,
                             max_pulse_ms=20.0, noise_train_ms=50.0)
    dec = loop.run_until_complete(crs.get_decimation()) or 6
    dual = DualPulseCaptureSession(
        channels=channels, module=1,
        slow_rate=slow_sample_rate(dec),
        config=cfg, hdf5_path=path)

    signals = PulseCaptureSignals()
    events = {"pairs": [], "errors": [], "finished": []}
    signals.pair_matched.connect(lambda p: events["pairs"].append(p))
    signals.error.connect(lambda m: events["errors"].append(m))
    signals.finished.connect(lambda: events["finished"].append(True))

    task = PulseCaptureTask(dual, signals, mode="both", crs=crs,
                            host="127.0.0.1", module=1)

    # Production topology: the slow stream reaches the task through the
    # Periscope tap (queue), NOT a second socket — the mock's unicast
    # would be load-balanced away from it.  Emulate the tap with a
    # background thread pumping slow packets into task.enqueue.
    import threading

    from rfmux.algorithms.measurement.pulse_sources import (
        run_slow_source,
    )

    class _TapShim:
        pass

    _TapShim.channels = channels
    _TapShim.feed_sample = staticmethod(
        lambda ch, i, q, t: task.enqueue(ch, i, q, t))

    tap_stop = {"stop": False}
    tap_thread = threading.Thread(
        target=lambda: asyncio.run(run_slow_source(
            _TapShim, "127.0.0.1", module=1,
            should_stop=lambda: tap_stop["stop"])),
        daemon=True)

    task.start()
    tap_thread.start()

    try:
        assert _spin_until(
            qt_app,
            lambda: dual.slow.state is CaptureState.CAPTURING
            and dual.fast.state is CaptureState.CAPTURING, 60), \
            f"states={dual.state}, errors={events['errors']}"
    except AssertionError:
        tap_stop["stop"] = True
        raise

    assert _spin_until(
        qt_app,
        lambda: any(p["slow_idx"] and p["fast_idx"]
                    for p in events["pairs"]), 90), \
        f"no matched pairs (pairs={len(events['pairs'])}, " \
        f"stats={dual.stats()}, errors={events['errors']})"

    task.request_stop()
    tap_stop["stop"] = True
    assert _spin_until(qt_app, lambda: events["finished"], 30)
    task.wait(5000)
    tap_thread.join(timeout=10)

    assert loop.run_until_complete(
        crs.get_pfb_streamer(module=1)) is None
    matched = [p for p in events["pairs"]
               if p["slow_idx"] and p["fast_idx"]]
    assert matched, "expected at least one matched pair"

    with PulseHDF5Reader(path) as reader:
        assert reader.dual
        assert sum(reader.pair_count(c) for c in channels) >= 1
        found = False
        for c in channels:
            for pair in reader.iter_matches(c):
                if pair["slow_idx"] and pair["fast_idx"]:
                    wf = reader.get_pulse(c, pair["fast_idx"],
                                          stream="fast")
                    assert wf is not None
                    found = True
        assert found

        # NOTE on cross-stream scales: run_pfb_source applies the
        # 24-bit -> 16-bit /256 so both streams use the same digital
        # convention, but the MOCK's slow-path gain varies with the
        # decimation stage while its PFB rendering is fixed (measured
        # fast/slow baseline ratios pre-/256: ~255x at dec 1, ~51x at
        # dec 6) — so no amplitude-ratio assertion is possible here.
        # See the plan doc follow-up on mock stream-gain modeling.
