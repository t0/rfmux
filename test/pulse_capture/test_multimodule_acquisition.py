"""A slow-stream capture across two of the mock's modules, end to end:
the mock streams every module with a configured channel."""

import asyncio

import pytest

from rfmux.algorithms.measurement.trigger_capture import trigger_capture
from rfmux.pulse_capture.capture_session import PulseCaptureConfig
from rfmux.pulse_capture.hdf5 import PulseHDF5Reader

pytestmark = pytest.mark.slow_acquisition


def test_a_capture_across_modules_is_keyed_by_pairs(tmp_path):
    from rfmux.mock.helpers import create_mock_crs
    path = tmp_path / "pulse.h5"

    async def run():
        crs = await create_mock_crs(
            module=1, config={"num_resonances": 1, "resonator_random_seed": 5},
            verbose=False)
        try:
            # The mock streams a module once one of its channels is set.
            for module, channel in ((1, 1), (2, 1), (2, 2)):
                await crs.set_amplitude(0.01, channel=channel, module=module)
            await asyncio.sleep(1.0)
            return await trigger_capture.__wrapped__(
                crs, channel={1: [1], 2: [1, 2]}, time_run=0.3,
                config=PulseCaptureConfig(noise_train_ms=200.0),
                hdf5_path=path, verbose=False)
        finally:
            await crs.stop_udp_streaming()

    result = asyncio.run(run())
    keys = [(1, 1), (2, 1), (2, 2)]
    assert result.channels == keys and result.module is None
    assert set(result.slow.noise) == set(keys)
    with PulseHDF5Reader(path) as r:
        assert r.channels == keys and r.modules == [1, 2]
        assert "module_2/channel_2" in r.f
        assert "module" not in r.metadata
