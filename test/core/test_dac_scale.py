"""The DAC scale a module's amplitudes are labelled against."""

import math

import pytest

from rfmux.core.dac_scale import DAC_SCALE_LABEL_OFFSET_DB, dac_scale_dbm
from rfmux.core.resonators import BiasPoint
from rfmux.core.transferfunctions import (
    convert_dacunits_to_dbm,
    convert_dacunits_to_volts,
    convert_volts_to_dbm,
)


class _Board:
    def __init__(self, scale=-0.5):
        self.scale = scale

    async def get_dac_scale(self, units="DBM", module=None):
        return self.scale


def test_the_label_offset_is_zero():
    """A label is the board's own number. This pins the offset because
    changing it moves every power this package reports, in files as well
    as on screen -- see the warning on rfmux.core.dac_scale."""
    assert DAC_SCALE_LABEL_OFFSET_DB == 0.0


@pytest.mark.asyncio
async def test_the_scale_is_the_board_s_less_the_offset():
    assert await dac_scale_dbm(_Board(-0.5), 1) == -0.5 - DAC_SCALE_LABEL_OFFSET_DB


@pytest.mark.asyncio
async def test_a_module_the_banking_hides_has_no_scale():
    """An answer, not a failure: a module the analog banking does not
    expose has no scale to report."""
    board = _Board()

    async def refuse(units="DBM", module=None):
        raise RuntimeError("Can't access module 1: analog banking")
    board.get_dac_scale = refuse
    assert await dac_scale_dbm(board, 1) is None


@pytest.mark.asyncio
async def test_any_other_failure_is_raised():
    board = _Board()

    async def broken(units="DBM", module=None):
        raise RuntimeError("tuber timeout")
    board.get_dac_scale = broken
    with pytest.raises(RuntimeError, match="tuber timeout"):
        await dac_scale_dbm(board, 1)


@pytest.mark.asyncio
async def test_the_scale_labels_an_amplitude_as_a_power():
    """What the scale is for: BiasPoint.power_dbm adds it to
    20*log10(amplitude), so the offset is applied once, on the way in."""
    scale = await dac_scale_dbm(_Board(0.0), 1)
    assert BiasPoint(frequency_hz=1e9, amplitude=0.1).power_dbm(scale) == \
        pytest.approx(-20.0)


# ─── the conversions every label is built from ───────────────────────────────


@pytest.mark.parametrize("amplitude", [1.0, 0.5, 0.005, 1e-4])
def test_dacunits_to_dbm_is_the_scale_plus_twenty_log_amplitude(amplitude):
    assert convert_dacunits_to_dbm(amplitude, 1.0) == pytest.approx(
        1.0 + 20.0 * math.log10(amplitude))


def test_full_scale_drives_the_scale_itself():
    """What "DAC full scale" means: amplitude 1.0 is the scale, by definition."""
    assert convert_dacunits_to_dbm(1.0, -3.25) == pytest.approx(-3.25)


def test_a_silent_tone_drives_no_power():
    assert convert_dacunits_to_dbm(0.0, 1.0) == -math.inf


def test_dacunits_to_volts_is_the_dbm_expressed_as_peak_volts():
    """The two conversions are one quantity in two units, so the volts must
    come back as the dBm through the package's volts convention."""
    volts = convert_dacunits_to_volts(0.005, 1.0)
    assert convert_volts_to_dbm(volts) == pytest.approx(
        convert_dacunits_to_dbm(0.005, 1.0))


def test_the_conversions_are_what_a_bias_point_labels_itself_with():
    """One definition: a bias point's power is the shared conversion, so a
    label on screen and a power in a file cannot drift apart."""
    bias = BiasPoint(frequency_hz=1e9, amplitude=0.017)
    assert bias.power_dbm(-0.5) == pytest.approx(
        convert_dacunits_to_dbm(0.017, -0.5))
