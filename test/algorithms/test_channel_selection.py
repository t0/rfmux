"""
Tests for get_biased_channels.

Uses a MockCRS session directly rather than create_mock_crs(): these
tests only need the tuber RPC surface, not a UDP stream, which keeps
them out of the acquisition tier (and off ports 9876/9877).
"""

import asyncio

import pytest

import rfmux

SESSION = """
!HardwareMap
- !flavour "rfmux.mock"
- !CRS { serial: "0000", hostname: "127.0.0.1" }
"""

# (channel, amplitude) programmed onto module 1 by the fixture.
BIASED = ((1, 0.25), (4, 0.25), (17, 0.25), (128, 0.25))
WEAK = (3, 0.01)
EXPLICIT_ZERO = (9, 0.0)


@pytest.fixture(scope="module")
def mock_crs():
    loop = asyncio.new_event_loop()
    session = rfmux.load_session(SESSION)
    crs = session.query(rfmux.CRS).one()
    loop.run_until_complete(crs.resolve())
    for channel, amplitude in BIASED + (WEAK, EXPLICIT_ZERO):
        loop.run_until_complete(
            crs.set_amplitude(amplitude, channel=channel, module=1))
    yield loop, crs
    loop.close()


def test_reports_only_biased_channels(mock_crs):
    loop, crs = mock_crs
    got = loop.run_until_complete(
        crs.get_biased_channels(1, max_channels=128))
    assert got == [1, 3, 4, 17, 128]


def test_explicit_zero_is_not_biased(mock_crs):
    loop, crs = mock_crs
    got = loop.run_until_complete(
        crs.get_biased_channels(1, max_channels=128))
    assert EXPLICIT_ZERO[0] not in got


def test_max_channels_bounds_the_scan(mock_crs):
    # A channel the packet cannot carry must not be offered, however
    # it is biased.
    loop, crs = mock_crs
    got = loop.run_until_complete(
        crs.get_biased_channels(1, max_channels=20))
    assert got == [1, 3, 4, 17]


def test_threshold_filters_weak_tones(mock_crs):
    loop, crs = mock_crs
    got = loop.run_until_complete(
        crs.get_biased_channels(1, max_channels=128, threshold=0.1))
    assert got == [1, 4, 17, 128]
    assert WEAK[0] not in got


def test_defaults_to_long_packet_width(mock_crs):
    loop, crs = mock_crs
    got = loop.run_until_complete(crs.get_biased_channels(1))
    assert got == [1, 3, 4, 17, 128]


def test_unbiased_module_is_empty(mock_crs):
    loop, crs = mock_crs
    assert loop.run_until_complete(
        crs.get_biased_channels(2, max_channels=128)) == []


def test_zero_width_is_empty(mock_crs):
    loop, crs = mock_crs
    assert loop.run_until_complete(
        crs.get_biased_channels(1, max_channels=0)) == []
