"""take_noise_spectrum: argument checks before any board call, and one
acquisition from a streaming mock with per-channel results in order."""

import asyncio
import contextlib
import io

import numpy as np
import pytest

from rfmux.core.schema import CRS


@pytest.mark.parametrize("bad", [
    dict(channels=[]),
    dict(decimation=7),
    dict(num_samples=3, num_segments=2),
    dict(reference="dBc"),
    dict(spectrum_limit=0.0),
    dict(pfb_samples=1),
])
def test_rejects_arguments_before_touching_the_board(bad):
    crs = CRS(serial="0000")  # unresolved: any board call would fail differently
    kwargs = dict(channels=[1], decimation=6, num_samples=100, num_segments=2, module=1)
    kwargs.update(bad)
    with pytest.raises(ValueError, match=next(iter(bad))):
        asyncio.run(crs.take_noise_spectrum(**kwargs))


@pytest.mark.slow_acquisition
def test_streaming_mock_gives_one_entry_per_channel():
    from rfmux.mock.helpers import create_mock_crs

    loop = asyncio.new_event_loop()
    with contextlib.redirect_stdout(io.StringIO()):
        crs = loop.run_until_complete(create_mock_crs(
            module=1, verbose=False,
            config={"num_resonances": 2, "resonator_random_seed": 5, "auto_bias_kids": True}))
        result = loop.run_until_complete(crs.take_noise_spectrum(
            channels=[2, 1], decimation=6, num_samples=600, num_segments=2, module=1))
        nco = loop.run_until_complete(crs.get_nco_frequency(module=1))
        offsets = [loop.run_until_complete(crs.get_frequency(channel=c, module=1)) for c in (2, 1)]
        loop.run_until_complete(crs.stop_udp_streaming())
    loop.close()

    assert [len(result[k]) for k in ("I", "Q", "single_psd_i", "single_psd_q", "dual_psd",
                                      "amplitudes", "channel_frequencies")] == [2] * 7
    assert len(result["I"][0]) == 600 and len(result["freq_iq"]) == len(result["single_psd_i"][0])
    assert result["channel_frequencies"] == pytest.approx([nco + o for o in offsets])
    assert result["pfb_enabled"] is False and "pfb_I" not in result
