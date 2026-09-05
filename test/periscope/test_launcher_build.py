"""The launcher's mock build goes through apply_mock_config, the path a
reconfigure takes: one place pins the seed and normalizes the config,
and the count comes back as a number rather than the server's
(count, frequencies) pair."""
import asyncio

import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope import __main__ as periscope_main  # noqa: E402


class _FakeCRS:
    def __init__(self):
        self.built = []

    async def generate_resonators(self, config):
        """The server's shape: the count and the resonance frequencies."""
        self.built.append(dict(config))
        n = config["num_resonances"]
        return n, [1.0e9] * n

    async def get_build_progress(self):
        return {"stage": "generating", "done": 0, "total": 1}

    async def measure_df_calibrations(self, module, progress=None):
        return {1: 1 + 1j}


def _build(config):
    crs = _FakeCRS()
    loop = asyncio.new_event_loop()
    try:
        result = periscope_main._build_with_progress(crs, config, loop, 1)
    finally:
        loop.close()
    return crs, result


def test_build_returns_the_count_and_the_calibrations(qt_app):
    _, (count, cals) = _build({"num_resonances": 40, "auto_bias_kids": True,
                               "resonator_random_seed": 7})
    assert count == 40
    assert cals == {1: 1 + 1j}


def test_build_pins_the_seed_it_sent(qt_app):
    config = {"num_resonances": 40, "auto_bias_kids": False,
              "resonator_random_seed": None}
    crs, _ = _build(config)
    [sent] = crs.built
    assert config["resonator_random_seed"] is not None
    assert sent["resonator_random_seed"] == config["resonator_random_seed"]


def test_build_sends_the_normalized_config(qt_app):
    """The clamps of mc.apply_overrides hold at launch as they do on a
    reconfigure (tls_alpha tops out at 2)."""
    crs, _ = _build({"num_resonances": 40, "auto_bias_kids": False,
                     "resonator_random_seed": 7, "tls_alpha": 5.0})
    [sent] = crs.built
    assert sent["tls_alpha"] == 2.0
