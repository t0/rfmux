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

    async def measure_df_calibrations(self, module, progress=None):
        return {1: {"df_calibration": 1 + 1j}}


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
    assert cals == {1: {"df_calibration": 1 + 1j}}
