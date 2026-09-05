"""What a multisweep result carries, on a board whose samples are
chosen to make the sweep jump: the bifurcation flag, and nothing else
about it."""
import contextlib
import warnings

import numpy as np
import pytest

from rfmux.algorithms.measurement.fitting import identify_bifurcation
from rfmux.algorithms.measurement.multisweep import multisweep

NPOINTS = 101
CF = 1.0e9


class _Ctx:
    def __init__(self, board):
        self.board = board

    def set_frequency(self, f, channel, module):
        self.board.freq[channel] = f

    def set_amplitude(self, a, channel, module):
        pass

    def set_phase(self, p, units, target, channel, module):
        pass

    async def __call__(self):
        pass


class _Samples:
    def __init__(self, z):
        class _Mean:
            i = [z.real]
            q = [z.imag]
        self.mean = _Mean()


class _JumpingBoard:
    """S21 in counts that steps in the middle of the sweep, with a
    ripple either side so the point-to-point differences have a
    median: the shape identify_bifurcation flags."""
    class UNITS:
        DEGREES = "deg"

    class TARGET:
        ADC = "adc"

    def __init__(self):
        self.freq = {}
        self.nco = 0.0

    async def get_decimation(self):
        return 6

    async def set_nco_frequency(self, f, module):
        self.nco = f

    @contextlib.asynccontextmanager
    async def tuber_context(self):
        yield _Ctx(self)

    @staticmethod
    def s21(f):
        x = (f - CF) / 1e3
        return (2000.0 if x < 0 else 1000.0) + 3.0 * np.sin(x) + 0j

    async def get_samples(self, n, average, channel, module):
        return _Samples(self.s21(self.nco + self.freq[1]))


def _sweep(board):
    return multisweep.__wrapped__(
        board, center_frequencies=[CF], span_hz=100e3,
        npoints_per_sweep=NPOINTS, amp=0.01, nsamps=1,
        bias_frequency_method=None, module=1)


def test_the_synthetic_sweep_is_one_the_detector_flags():
    f = np.linspace(CF - 50e3, CF + 50e3, NPOINTS)
    assert identify_bifurcation(np.array([_JumpingBoard.s21(x) for x in f]))


@pytest.mark.asyncio
async def test_a_jumping_sweep_is_flagged_bifurcated():
    res = await _sweep(_JumpingBoard())
    assert res[1]["is_bifurcated"] is True


@pytest.mark.asyncio
async def test_a_bifurcated_sweep_raises_no_warning():
    # The flag is in the result; bias_kids reads it there.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = await _sweep(_JumpingBoard())
    assert res[1]["is_bifurcated"]
    assert [str(w.message) for w in caught] == []


@pytest.mark.asyncio
async def test_the_calibration_lives_on_bias_kids_result_only():
    res = await _sweep(_JumpingBoard())
    assert "df_calibration" not in res[1]
