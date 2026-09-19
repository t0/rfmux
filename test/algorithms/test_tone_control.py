"""read_tones and write_tone: the batched read Periscope's Control mode
refreshes from, and the set-then-re-read it programs a field with."""

import asyncio

import pytest

import rfmux
from rfmux.algorithms.measurement.tone_control import read_tones, write_tone

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
    loop.run_until_complete(crs.set_frequency(1.25e6, channel=1, module=1))
    loop.run_until_complete(crs.set_amplitude(0.01, channel=1, module=1))
    loop.run_until_complete(crs.set_phase(
        30.0, units=crs.UNITS.DEGREES, target=crs.TARGET.DAC,
        channel=1, module=1))
    yield loop, crs
    loop.close()


def test_read_returns_nco_and_every_field_per_channel(mock_crs):
    loop, crs = mock_crs
    got = loop.run_until_complete(read_tones(crs, 1, [1, 2]))
    assert got["nco"] == 500e6
    assert got["dac_scale"] == -0.5
    assert got["channels"][1] == {
        "frequency": 1.25e6, "amplitude": 0.01, "phase": 30.0}


def test_a_channel_the_board_never_set_reads_as_none(mock_crs):
    loop, crs = mock_crs
    got = loop.run_until_complete(read_tones(crs, 1, [2]))
    assert got["channels"][2] == {
        "frequency": None, "amplitude": None, "phase": None}


def test_write_round_trips_through_the_board(mock_crs):
    loop, crs = mock_crs
    got = loop.run_until_complete(write_tone(
        crs, 1, 3, frequency=-2.45e6, amplitude=0.02, phase=45.0))
    assert got["channels"][3] == {
        "frequency": -2.45e6, "amplitude": 0.02, "phase": 45.0}
    assert loop.run_until_complete(
        crs.get_frequency(channel=3, module=1)) == -2.45e6


def test_write_touches_only_the_given_fields(mock_crs):
    loop, crs = mock_crs
    loop.run_until_complete(write_tone(crs, 1, 3, amplitude=0.02, phase=45.0))
    got = loop.run_until_complete(write_tone(crs, 1, 3, frequency=1e6))
    assert got["channels"][3]["amplitude"] == 0.02
    assert got["channels"][3]["phase"] == 45.0
