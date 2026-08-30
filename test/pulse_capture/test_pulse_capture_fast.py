"""
Fast (PFB) pulse-capture integration test.

Drives PulseCaptureTask in fast mode against a MockCRS with periodic
QP pulses: the task must configure the PFB streamer for the capture
channels, feed the session from the 2.44 MHz stream via the shared
run_pfb_source, detect pulses, write the HDF5, and tear the PFB
streamer down on stop.
"""

import asyncio

import numpy as np
import pytest

from test.qt_helpers import spin_until  # noqa: E402


pytest.importorskip("PyQt6")
pytest.importorskip("h5py")

pytestmark = pytest.mark.slow_acquisition


from rfmux.pulse_capture.capture_session import (  # noqa: E402
    CaptureState,
    PulseCaptureSession,
)
from rfmux.pulse_capture.hdf5 import PulseHDF5Reader  # noqa: E402
from rfmux.core.transferfunctions import PFB_SAMPLING_FREQ  # noqa: E402
from rfmux.tools.periscope.pulse_capture_task import (  # noqa: E402
    PulseCaptureSignals,
    PulseCaptureTask,
)



@pytest.fixture(scope="module")
def mock_crs():
    from rfmux.mock.helpers import create_mock_crs
    loop = asyncio.new_event_loop()
    # auto_bias_kids biases the detectors — without it the
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


@pytest.fixture
def stream_guard(qt_app):
    """Stops capture tasks and tap threads whether or not the test passes.

    The tasks and threads registered here hold the streamer multicast sockets —
    9876 for the slow stream, 9877 for the PFB one. Stopping them only on the
    success path, which is what these tests used to do, means a failing
    assertion leaves them running until the process exits.

    How much that costs depends on how much process lifetime is left. Running
    one test it is harmless: the interpreter exits immediately and the daemon
    thread dies with it. In a full-tier run the failure lands with tests still to
    go, and then ~20 MockCRS servers to reap at exit, so the sockets stay open
    for minutes — a leaked run was measured still holding both ports 419 s after
    its last assertion.

    That window is dangerous because ``get_multicast_socket()`` sets
    ``SO_REUSEPORT``: another run starting inside it does not fail to bind, it
    joins the same multicast group, and the two processes split the packet
    stream so both see partial data. It surfaces as a pulse-detector bug
    (``no matched pairs``, thousands of phantom pulses) rather than as a resource
    clash, and two separate investigations were lost to that before the cause was
    found. It can also corrupt the later tests of the very run that leaked.

    Register anything that owns a socket and teardown becomes unconditional::

        task = stream_guard.task(PulseCaptureTask(...))
        stream_guard.stop_flag(tap_stop)
        stream_guard.thread(tap_thread)
    """
    tasks: list = []
    flags: list = []
    threads: list = []

    class _Guard:
        def task(self, task):
            """Register a PulseCaptureTask to request_stop() + wait()."""
            tasks.append(task)
            return task

        def stop_flag(self, mapping, key="stop"):
            """Register a dict flag that a source loop polls to exit."""
            flags.append((mapping, key))
            return mapping

        def thread(self, thread):
            """Register a threading.Thread to join."""
            threads.append(thread)
            return thread

    yield _Guard()

    # Flags first: the source loops poll them, so setting them lets the reader
    # threads leave their `with get_multicast_socket(...)` blocks and close the
    # sockets. Only then is it worth waiting on anything.
    for mapping, key in flags:
        mapping[key] = True
    for task in tasks:
        try:
            task.request_stop()
        except Exception:
            pass
    for task in tasks:
        try:
            task.wait(5000)
        except Exception:
            pass
    for thread in threads:
        thread.join(timeout=10)



def test_fast_capture_end_to_end(qt_app, mock_crs, tmp_path, stream_guard):
    loop, crs = mock_crs
    channels = [1, 2]
    path = tmp_path / "fast_capture.h5"

    capture_session = PulseCaptureSession(
        channels=channels, module=1, streamer_mode="fast",
        threshold_sigma=50.0, end_sigma=3.0,
        sample_rate=PFB_SAMPLING_FREQ, buf_size=200_000,
        noise_samples=50_000, hdf5_path=path,
        histogram_flush_every=2)
    signals = PulseCaptureSignals()
    events = {"pulses": [], "errors": [], "finished": []}
    signals.pulse_detected.connect(
        lambda ch, idx, s: events["pulses"].append((ch, idx, s)))
    signals.error.connect(lambda m: events["errors"].append(m))
    signals.finished.connect(lambda: events["finished"].append(True))

    # Registered so the task stops (and releases port 9877) even if an
    # assertion below fails; see stream_guard.
    task = stream_guard.task(
        PulseCaptureTask(capture_session, signals, mode="fast", crs=crs,
                         host="127.0.0.1", module=1))
    task.start()

    assert spin_until(
        qt_app, lambda: capture_session.state is CaptureState.CAPTURING, 30), \
        f"never reached CAPTURING (state={capture_session.state}, " \
        f"errors={events['errors']})"

    # PFB streamer was configured for our channels by the task
    assert loop.run_until_complete(
        crs.get_pfb_streamer(module=1)) == channels

    assert spin_until(qt_app, lambda: capture_session.total_pulses >= 2, 60), \
        f"no pulses detected (errors={events['errors']})"

    task.request_stop()
    assert spin_until(qt_app, lambda: events["finished"], 30), \
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
            PFB_SAMPLING_FREQ)


def test_both_mode_end_to_end(qt_app, mock_crs, tmp_path, stream_guard):
    """Both-mode task: dual sockets, live matching, dual file, teardown."""
    from rfmux.pulse_capture.capture_session import (
        DualPulseCaptureSession,
    )
    from rfmux.pulse_capture.capture_session import (
        PulseCaptureConfig,
    )
    from rfmux.core.transferfunctions import decimation_to_sampling

    loop, crs = mock_crs
    channels = [1, 2]
    path = tmp_path / "both_capture.h5"
    cfg = PulseCaptureConfig(threshold_sigma=5.0, end_sigma=1.5,
                             max_pulse_ms=20.0, noise_train_ms=50.0)
    # Stage 0 is a valid decimation (the ~38 kHz rate), so test it against
    # None rather than falsiness: "or 6" would rewrite stage 0 to stage 6 and
    # build the session at 596 Hz for a stream running 64x faster.  Mirrors
    # trigger_capture.py, which does get the None check right.
    dec = loop.run_until_complete(crs.get_decimation())
    if dec is None:
        dec = 6
    dual = DualPulseCaptureSession(
        channels=channels, module=1,
        slow_rate=decimation_to_sampling(dec),
        config=cfg, hdf5_path=path)

    signals = PulseCaptureSignals()
    events = {"pairs": [], "errors": [], "finished": []}
    signals.pair_matched.connect(lambda p: events["pairs"].append(p))
    signals.error.connect(lambda m: events["errors"].append(m))
    signals.finished.connect(lambda: events["finished"].append(True))

    # Registered so the task stops (releasing port 9877) even if an assertion
    # below fails; see stream_guard.
    task = stream_guard.task(
        PulseCaptureTask(dual, signals, mode="both", crs=crs,
                         host="127.0.0.1", module=1))

    # Production topology: the slow stream reaches the task through the
    # Periscope tap (queue), NOT a second socket — the mock's unicast
    # would be load-balanced away from it.  Emulate the tap with a
    # background thread pumping slow packets into task.enqueue.
    import threading

    from rfmux.pulse_capture.sources import (
        run_slow_source,
    )

    class _TapShim:
        pass

    def _shim_feed_block(ch, i_vals, q_vals, stamps):
        # run_slow_source hands over per-channel blocks; the real tap
        # hands the task one packet at a time, so unpack back into
        # packets rather than short-circuiting the queue under test.
        for n in range(len(i_vals)):
            task.enqueue_packet(
                (ch,), np.array([complex(i_vals[n], q_vals[n])]),
                float(stamps[n]))

    _TapShim.channels = channels
    _TapShim.feed_block = staticmethod(_shim_feed_block)

    # Both registered: the flag lets run_slow_source leave its socket block,
    # the join makes sure it actually did. Without this the thread outlives a
    # failing assertion and keeps port 9876 for the life of the process.
    tap_stop = stream_guard.stop_flag({"stop": False})
    tap_thread = stream_guard.thread(threading.Thread(
        target=lambda: asyncio.run(run_slow_source(
            _TapShim, "127.0.0.1", module=1,
            should_stop=lambda: tap_stop["stop"])),
        daemon=True))

    task.start()
    tap_thread.start()

    # No per-assert cleanup here on purpose: stream_guard tears down the task
    # and tap thread however this test exits. The version that guarded only
    # this first assertion is what leaked the sockets when the second one
    # failed.
    # Budgets are generous because the mock generates both streams in one
    # process, and two PFB channels at PFB_SAMPLING_FREQ is 4.9 M complex
    # samples a second.  It does not keep up with real time on a loaded CI
    # runner, so the slow stream reaches CAPTURING late.  These are liveness
    # bounds, not performance assertions -- a healthy run gets here in
    # seconds and does not spend the budget.
    assert spin_until(
        qt_app,
        lambda: dual.slow.state is CaptureState.CAPTURING
        and dual.fast.state is CaptureState.CAPTURING, 180), \
        f"states={dual.state}, errors={events['errors']}"

    assert spin_until(
        qt_app,
        lambda: any(p["slow_idx"] and p["fast_idx"]
                    for p in events["pairs"]), 180), \
        f"no matched pairs (pairs={len(events['pairs'])}, " \
        f"stats={dual.stats()}, errors={events['errors']})"

    task.request_stop()
    tap_stop["stop"] = True
    assert spin_until(qt_app, lambda: events["finished"], 30)
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
