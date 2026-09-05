"""get_pfb_streamer reports what the streamer thread is emitting, and
nothing once that thread is gone."""
import asyncio
import contextlib
import io

from rfmux.mock.crs import ServerMockCRS
from rfmux.mock.udp_streamer import MockCRSStreamer


def _crs_with_pfb(channels, module=1):
    crs = ServerMockCRS("0000")
    st = MockCRSStreamer(crs)      # never started: no socket, no thread
    st.pfb_channels = list(channels)
    st.pfb_module = module
    st.pfb_enabled = True
    crs._udp_manager._streamer = st
    return crs


def test_answers_from_the_streamer_thread():
    crs = _crs_with_pfb([1, 2])
    assert asyncio.run(crs.get_pfb_streamer(module=1)) == [1, 2]
    assert asyncio.run(crs.get_pfb_streamer(module=2)) is None


def test_stopping_the_stream_stops_the_answer():
    crs = _crs_with_pfb([1, 2])
    crs._udp_manager._streaming_active = True
    with contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(crs.stop_udp_streaming())
    assert asyncio.run(crs.get_pfb_streamer(module=1)) is None
