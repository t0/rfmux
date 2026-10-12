"""The DAC scale a module's amplitudes are measured against: the
programmed scale less the balun loss."""

import math

import pytest

from rfmux.core.dac_scale import (
    DAC_SCALE_OUTPUT_RANGE_DBM, QUANTIZATION_PEAK_FRACTION, comb_drive_issues,
    comb_peak_fraction, dac_scale_dbm)
from rfmux.core.resonators import BiasPoint
from rfmux.core.transferfunctions import (
    CREST_FACTOR,
    BALUN_LOSS_DB,
    convert_dacunits_to_dbm,
    convert_dacunits_to_volts,
    convert_volts_to_dbm,
)


class _Board:
    def __init__(self, scale=-0.5):
        self.scale = scale

    async def get_dac_scale(self, units="DBM", module=None):
        return self.scale


def test_the_balun_loss_is_pinned():
    """The balun between the DAC and the output loses 1.75 dB. Pinned
    because it moves every power this package reports, in files as well
    as on screen."""
    assert BALUN_LOSS_DB == 1.75


@pytest.mark.asyncio
async def test_the_scale_is_the_board_s_less_the_balun_loss():
    assert await dac_scale_dbm(_Board(-0.5), 1) == -0.5 - BALUN_LOSS_DB


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
    20*log10(amplitude), so the balun loss is applied once, on the way in."""
    scale = await dac_scale_dbm(_Board(0.0), 1)
    assert BiasPoint(frequency_hz=1e9, amplitude=0.1).power_dbm(scale) == \
        pytest.approx(-20.0 - BALUN_LOSS_DB)


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



# ─── the drive of a comb ─────────────────────────────────────────────────────


@pytest.mark.parametrize("n, amplitude", [(1, 0.5), (3, 0.2), (1023, 0.001)])
def test_the_peak_of_equal_tones_is_the_smaller_estimate(n, amplitude):
    """The coherent sum for a few tones, the crest factor for many."""
    expected = min(n * amplitude, CREST_FACTOR * amplitude * math.sqrt(n / 2))
    assert comb_peak_fraction([amplitude] * n) == pytest.approx(expected)


def test_a_comb_inside_the_limits_has_no_issue():
    assert comb_drive_issues([[0.01] * 100], DAC_SCALE_OUTPUT_RANGE_DBM[0]) == []


def test_a_comb_above_full_scale_can_clip_and_the_dac_scale_can_go_up():
    (kind, message), = comb_drive_issues([[0.6] * 3], DAC_SCALE_OUTPUT_RANGE_DBM[0])
    assert kind == "clipping"
    assert "Raise the DAC scale" in message


def test_at_the_highest_dac_scale_clipping_asks_for_less_amplitude():
    (kind, message), = comb_drive_issues([[0.6] * 3], DAC_SCALE_OUTPUT_RANGE_DBM[1])
    assert "Raise the DAC scale" not in message
    assert "Lower the amplitude" in message


def test_a_small_comb_risks_quantization_and_the_dac_scale_can_go_down():
    small = QUANTIZATION_PEAK_FRACTION / 10
    (kind, message), = comb_drive_issues([[small]], DAC_SCALE_OUTPUT_RANGE_DBM[1])
    assert kind == "quantization"
    assert "Lower the DAC scale" in message


def test_at_the_lowest_dac_scale_quantization_asks_for_more_amplitude():
    small = QUANTIZATION_PEAK_FRACTION / 10
    (kind, message), = comb_drive_issues([[small]], DAC_SCALE_OUTPUT_RANGE_DBM[0])
    assert "Lower the DAC scale" not in message
    assert "Raise the amplitude" in message


def test_the_output_range_is_the_data_sheet_range_less_the_balun_loss():
    assert DAC_SCALE_OUTPUT_RANGE_DBM == pytest.approx(
        (1.0 - BALUN_LOSS_DB, 4.5 - BALUN_LOSS_DB))
