"""The launcher's mock build goes through apply_mock_config, the path a
reconfigure takes, and the count comes back as a number rather than the
server's (count, frequencies) pair."""
import asyncio

import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope import __main__ as periscope_main  # noqa: E402


class _FakeCRS:
    async def get_mock_configuration(self):
        return None   # no array yet: everything counts as changed

    async def generate_resonators(self, config):
        """The server's shape: the count and the resonance frequencies."""
        n = config["num_resonances"]
        return n, [1.0e9] * n

    async def get_build_progress(self):
        return {"stage": "generating", "done": 0, "total": 1}

    async def empirically_calibrate_df(self, module, save=None,
                                       progress_callback=None):
        from rfmux.core.resonators import (BiasPoint, Resonator,
                                           ResonatorCatalog)
        return ResonatorCatalog(
            [Resonator("R1", 1, BiasPoint(1.0e9, 0.01, dI_df=0.5, dQ_df=-0.5))],
            module=module)

    async def get_nco_frequency(self, module):
        return 1.0e9

    async def get_dac_scale(self, units, module):
        return 1.0


def _build(config):
    loop = asyncio.new_event_loop()
    try:
        return periscope_main._build_with_progress(_FakeCRS(), config, loop, 1)
    finally:
        loop.close()


def test_build_returns_the_count_and_the_calibrations(qt_app):
    count, cals = _build({"num_resonances": 40, "auto_bias_kids": True,
                          "resonator_random_seed": 7})
    assert count == 40
    assert list(cals) == [1]
    assert cals[1]["df_calibration"] == pytest.approx(1 + 1j)
    assert cals[1]["nco_frequency_hz"] == 1.0e9
