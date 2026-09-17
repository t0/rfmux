"""A tone set on a module whose NCO nobody has set: the mock reports
that NCO as 0 and streams the module's packets on that basis, as after
a console `crs.set_frequency(...)` on a fresh module."""
import asyncio
import contextlib
import io
from datetime import datetime

import pytest

from rfmux.mock.crs import ServerMockCRS
from rfmux.mock.udp_streamer import MockCRSStreamer


def _unbiased_mock():
    crs = ServerMockCRS("0000")
    with contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(crs.generate_resonators({
            "num_resonances": 2, "resonator_random_seed": 11,
            "auto_bias_kids": False}))
    return crs


def test_slow_stream_survives_a_tone_with_no_nco_set():
    crs = _unbiased_mock()
    assert asyncio.run(crs.get_nco_frequency(module=1)) == 0
    asyncio.run(crs.set_frequency(1e6, channel=1, module=1))
    asyncio.run(crs.set_amplitude(0.01, channel=1, module=1))

    st = MockCRSStreamer(crs)
    st.running = True
    st.slow_socket = None               # build and count; do not send
    st.start_datetime = datetime(2026, 1, 1)
    st._emit_slow_block(1, 0.0, 6, 10)
    assert st.packets_sent == 10


@pytest.mark.parametrize("setter, value", [("set_frequency", 1e6),
                                           ("set_amplitude", 0.01)])
def test_a_missing_channel_is_named(setter, value):
    crs = _unbiased_mock()
    with pytest.raises(AssertionError, match="Channel must be an integer"):
        asyncio.run(getattr(crs, setter)(value, module=1))
