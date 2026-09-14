"""The DAC scale a module's amplitudes are labelled against."""

import pytest

from rfmux.core.dac_scale import DAC_SCALE_LABEL_OFFSET_DB, dac_scale_dbm
from rfmux.core.resonators import BiasPoint


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
