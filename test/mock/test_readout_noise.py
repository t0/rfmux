"""Readout noise on mock samples, including channels with a tone."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from rfmux.mock.crs import ServerMockCRS


def _board(monkeypatch: pytest.MonkeyPatch) -> ServerMockCRS:
    crs = ServerMockCRS("0000")
    crs._physics_config["scale_factor"] = 1.0
    crs._physics_config["udp_noise_level"] = 1000.0
    crs._resonator_model = SimpleNamespace(
        calculate_module_response_coupled=lambda *args, **kwargs: {1: 100.0 + 50.0j})
    monkeypatch.setattr(crs, "channels_per_module", lambda: 1)
    return crs


def test_readout_noise_reaches_biased_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crs = _board(monkeypatch)
    samples = asyncio.run(crs.get_samples(2000, channel=1))

    assert np.std(samples["i"]) == pytest.approx(1000.0, rel=0.1)
    assert np.std(samples["q"]) == pytest.approx(1000.0, rel=0.1)


def test_average_reduces_readout_noise_on_biased_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crs = _board(monkeypatch)

    async def collect() -> list[dict]:
        return [await crs.get_samples(25, average=True) for _ in range(200)]

    results = asyncio.run(collect())
    means = np.array([result["mean"]["i"][0] for result in results])
    assert np.std(means) == pytest.approx(1000.0 / np.sqrt(25), rel=0.15)
