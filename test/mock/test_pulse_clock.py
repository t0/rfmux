"""A sample read does not stop the stream's pulses.  The stream and
get_samples both advance one pulse schedule, which ignores any time at
or before the latest it has seen; a stream that has fallen behind the
wall clock keeps pulsing after a read."""
import asyncio
import contextlib
import io
from datetime import datetime

from rfmux.mock.crs import ServerMockCRS
from rfmux.mock.udp_streamer import MockCRSStreamer


def _pulsing_mock():
    crs = ServerMockCRS("0000")
    with contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(crs.generate_resonators({
            "num_resonances": 2, "resonator_random_seed": 11,
            "pulse_mode": "periodic", "pulse_period": 0.05}))
    return crs


def test_a_read_does_not_put_the_schedule_ahead_of_a_lagging_stream():
    crs = _pulsing_mock()
    model = crs._resonator_model
    st = MockCRSStreamer(crs)
    st.running = True
    st.slow_socket = None               # build and count; do not send
    st.start_datetime = datetime(2026, 1, 1)
    crs._udp_manager._streamer = st
    crs.mock_start_time -= 600.0        # the wall clock is ten minutes on
    st.t_stream = 1.0                   # and the stream has made one second

    with contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(crs.get_samples(10, module=1))
    assert model.last_update_time <= st.t_stream

    before = len(model.pulse_events)
    with contextlib.redirect_stdout(io.StringIO()):
        st._emit_slow_block(1, 1.0, 6, 120)      # 0.2 s of stream at 596 Hz
    assert len(model.pulse_events) > before


def test_a_read_without_a_stream_follows_the_wall_clock():
    crs = _pulsing_mock()
    crs.mock_start_time -= 600.0
    with contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(crs.get_samples(10, module=1))
    assert crs._resonator_model.last_update_time >= 599.0
