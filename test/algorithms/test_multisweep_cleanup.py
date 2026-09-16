"""A failed sweep releases its own tones and preserves unrelated ones."""

from types import SimpleNamespace

import pytest

from rfmux.algorithms.measurement.multisweep import _measure_sweep, _SweepTarget

pytestmark = pytest.mark.portable


class Board:
    def __init__(self):
        self.amplitudes = {1: 0.3, 9: 0.2}
        self.start_amplitude = None

    def tuber_context(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def __call__(self):
        pass

    async def set_nco_frequency(self, frequency, module):
        self.start_amplitude = self.amplitudes[1]

    def set_frequency(self, frequency, channel, module):
        pass

    def set_amplitude(self, amplitude, channel, module):
        self.amplitudes[channel] = amplitude

    async def get_samples(self, *args, **kwargs):
        return SimpleNamespace(mean=SimpleNamespace(i=[1.0], q=[0.0]))


@pytest.mark.asyncio
async def test_callback_failure_leaves_owned_tones_off():
    board = Board()

    def fail(*args):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        await _measure_sweep(
            board, [_SweepTarget("R", 1, 1e9)], {"R": 0.01}, module=1,
            sweep_direction="upward", span_hz=1e5, npoints_per_sweep=3,
            nsamps=1, step=0, report_progress=None, data_callback=fail)
    assert board.amplitudes == {1: 0.0, 9: 0.2}


@pytest.mark.asyncio
async def test_first_sweep_starts_with_owned_tones_off():
    board = Board()
    await _measure_sweep(
        board, [_SweepTarget("R", 1, 1e9)], {"R": 0.01}, module=1,
        sweep_direction="upward", span_hz=1e5, npoints_per_sweep=3,
        nsamps=1, step=0, report_progress=None, data_callback=None)
    assert board.start_amplitude == 0.0
    assert board.amplitudes[9] == 0.2
