"""Normalizing a trace states it against the drive it was taken at.

A sweep's magnitude is a received power, which moves with the amplitude the
tone was driven at. Dividing by the drive -- subtracting it, in dBm -- makes
it a transmission instead, so an amplitude ladder lands on one axis. What
that division needs differs by unit, and a plot that cannot do it says so on
its axis rather than drawing one thing and labelling another.
"""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from rfmux.core.transferfunctions import (  # noqa: E402
    convert_dacunits_to_dbm, convert_dacunits_to_volts, convert_roc_to_dbm,
    convert_roc_to_volts)
from rfmux.tools.periscope.utils import UnitConverter  # noqa: E402

pytestmark = pytest.mark.portable

COUNTS = np.array([1.0e5, 2.0e5, 5.0e4])
DRIVE = 0.005
DAC_SCALE = 1.0


def converted(unit_mode, **kwargs):
    return UnitConverter.convert_amplitude(COUNTS, unit_mode, **kwargs)


# ─── what each unit divides by ────────────────────────────────────────────────


def test_counts_are_divided_by_the_drive():
    """Counts per unit drive: |S21| times a gain nothing here knows, which is
    why the axis says counts rather than claiming a transmission."""
    assert np.allclose(
        converted("counts", normalize=True, drive=DRIVE, dac_scale=DAC_SCALE),
        COUNTS / DRIVE)


def test_volts_are_divided_by_the_drive_in_volts():
    """Both sides in volts, so what comes out is |S21| and carries no unit."""
    assert np.allclose(
        converted("volts", normalize=True, drive=DRIVE, dac_scale=DAC_SCALE),
        convert_roc_to_volts(COUNTS) / convert_dacunits_to_volts(DRIVE, DAC_SCALE))


def test_dbm_subtracts_the_drive_power():
    assert np.allclose(
        converted("dbm", normalize=True, drive=DRIVE, dac_scale=DAC_SCALE),
        convert_roc_to_dbm(COUNTS) - convert_dacunits_to_dbm(DRIVE, DAC_SCALE))


def test_the_dbm_and_volts_normalizations_are_one_quantity():
    """dB is 20*log10 of the dimensionless ratio, so selecting dBm and
    selecting volts differ by a log and nothing else."""
    in_db = converted("dbm", normalize=True, drive=DRIVE, dac_scale=DAC_SCALE)
    ratio = converted("volts", normalize=True, drive=DRIVE, dac_scale=DAC_SCALE)

    assert np.allclose(in_db, 20.0 * np.log10(ratio))


# ─── why it is done at all ────────────────────────────────────────────────────


def test_two_drives_land_on_one_axis():
    """The point of the feature. The same transmission measured at two
    amplitudes is two curves stacked by drive until they are normalized, and
    one curve afterwards."""
    quiet, loud = 0.002, 0.02
    # A linear device: what comes back scales with what went in.
    at_quiet, at_loud = COUNTS * quiet, COUNTS * loud

    raw = [UnitConverter.convert_amplitude(c, "dbm") for c in (at_quiet, at_loud)]
    assert not np.allclose(*raw)

    normalized = [
        UnitConverter.convert_amplitude(c, "dbm", normalize=True, drive=d,
                                        dac_scale=DAC_SCALE)
        for c, d in ((at_quiet, quiet), (at_loud, loud))]
    assert np.allclose(*normalized)


def test_normalizing_does_not_flatten_a_real_difference():
    """It divides by the drive, not by the trace's own peak: a device that
    really does transmit differently at two drives still shows it."""
    bifurcated = COUNTS * np.array([1.0, 0.5, 1.0])

    one, other = (
        UnitConverter.convert_amplitude(c, "counts", normalize=True,
                                        drive=DRIVE, dac_scale=DAC_SCALE)
        for c in (COUNTS, bifurcated))
    assert not np.allclose(one, other)


# ─── when it cannot be done ───────────────────────────────────────────────────


def test_counts_need_no_dac_scale():
    """A drive is already the divisor, so counts normalize on a module whose
    scale the board never reported."""
    assert UnitConverter.can_normalize("counts", DRIVE, None)
    assert np.allclose(
        converted("counts", normalize=True, drive=DRIVE, dac_scale=None),
        COUNTS / DRIVE)


@pytest.mark.parametrize("unit_mode", ["volts", "dbm"])
def test_volts_and_dbm_need_the_dac_scale(unit_mode):
    """A drive is a fraction of full scale; without what full scale is worth
    there is no voltage or power to divide by."""
    assert not UnitConverter.can_normalize(unit_mode, DRIVE, None)


@pytest.mark.parametrize("drive", [None, 0.0, -1.0, float("nan")])
def test_no_usable_drive_means_no_normalization(drive):
    assert not UnitConverter.can_normalize("counts", drive, DAC_SCALE)


@pytest.mark.parametrize("unit_mode", ["counts", "volts", "dbm"])
def test_a_refused_normalization_leaves_the_measurement_alone(unit_mode):
    """Not a silent half-conversion: what comes back is the plain measurement,
    which is what the axis will be labelled as, because the label asks
    can_normalize the same question."""
    assert np.allclose(
        converted(unit_mode, normalize=True, drive=None, dac_scale=None),
        converted(unit_mode))
