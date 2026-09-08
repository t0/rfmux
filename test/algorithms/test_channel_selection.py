"""
Tests for get_biased_channels.

Uses a MockCRS session directly rather than create_mock_crs(): these
tests only need the tuber RPC surface, not a UDP stream, which keeps
them out of the acquisition tier (and off ports 9876/9877).
"""

import asyncio

import pytest

import rfmux
from rfmux.algorithms.measurement.channel_selection import (
    parse_channel_spec,
)

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


# ── channel spec parsing ──────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("1", [1]),
    ("1,2", [1, 2]),
    ("2-19", list(range(2, 20))),
    ("1,5-8,20", [1, 5, 6, 7, 8, 20]),
    ("5-5", [5]),                       # degenerate range is just the one
    (" 1 , 5 - 7 ", [1, 5, 6, 7]),      # whitespace anywhere
    ("3,1,2", [1, 2, 3]),               # sorted
    ("1-3,2-4", [1, 2, 3, 4]),          # overlapping ranges merge
    ("1,,2,", [1, 2]),                  # tolerate stray commas
])
def test_spec_parses(text, expected):
    assert parse_channel_spec(text) == expected


@pytest.mark.parametrize("text", ["all", "ALL", "  All  ", "*"])
def test_spec_wildcard_is_none(text):
    # None means "ask the board", not "no channels".
    assert parse_channel_spec(text) is None


@pytest.mark.parametrize("text,fragment", [
    ("", "No channels"),
    ("   ", "No channels"),
    ("abc", "'abc'"),
    ("1,abc", "'abc'"),
    ("1-", "'1-'"),
    ("2-x", "'2-x'"),
    ("0", "1-indexed"),
    ("0-4", "1-indexed"),
    ("-3", "'-3'"),                     # empty range start, not a negative
    ("19-2", "backwards"),
])
def test_spec_rejects(text, fragment):
    with pytest.raises(ValueError) as e:
        parse_channel_spec(text)
    assert fragment in str(e.value)


def test_spec_reversed_range_suggests_the_fix():
    with pytest.raises(ValueError) as e:
        parse_channel_spec("19-2")
    assert '"2-19"' in str(e.value)
