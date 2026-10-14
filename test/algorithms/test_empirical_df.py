"""Calibrating the tones a module is already playing.

A MockCRS session over tuber RPC only: the sweep reads samples by RPC and
no stream is started, so this stays out of the acquisition tier.
"""

import numpy as np
import pytest

import rfmux
from rfmux.core.resonators import on_grid
from rfmux.core.session_folder import latest_export
from rfmux.mock.config import apply_overrides
from rfmux.tuning import load_bias_catalog, store

SESSION = """
!HardwareMap
- !flavour "rfmux.mock"
- !CRS { serial: "0000", hostname: "127.0.0.1" }
"""
MODULE = 1


@pytest.fixture
def biased_board():
    """A board whose simulator biased its own five tones."""
    session = rfmux.load_session(SESSION)
    crs = session.query(rfmux.CRS).one()
    yield crs
    session.close()


async def _tones(crs, channels):
    """``{channel: (absolute frequency, amplitude)}`` as the board plays it."""
    nco = await crs.get_nco_frequency(module=MODULE)
    return {c: (nco + await crs.get_frequency(channel=c, module=MODULE),
                await crs.get_amplitude(channel=c, module=MODULE))
            for c in channels}


@pytest.mark.asyncio
async def test_the_playing_tones_come_back_calibrated_and_where_they_were(
        biased_board, tmp_path, monkeypatch):
    crs = biased_board
    await crs.resolve()
    await crs.generate_resonators(apply_overrides(
        {"num_resonances": 5, "auto_bias_kids": True,
         "freq_start": 1.0e9, "freq_end": 1.01e9}))
    monkeypatch.setattr(store, "_output_directory", tmp_path)
    before = await _tones(crs, range(1, 6))

    catalog = await crs.empirically_calibrate_df(module=MODULE, save=True)

    assert sorted(r.channel for r in catalog) == [1, 2, 3, 4, 5]
    for r in catalog:
        assert (r.bias.frequency_hz, r.bias.amplitude) == before[r.channel]
        assert np.isfinite(abs(r.bias.df_calibration))
        assert set(r.bias.bias_sweep["fits"]) == {"skewed", "nonlinear", "circle"}
        assert r.bias.bias_power_dbm is not None
    # The sweep moves the NCO and apply_bias settles it on the tone grid; the
    # tones themselves are back where they were.
    after = await _tones(crs, range(1, 6))
    for channel, (frequency, amplitude) in before.items():
        assert after[channel][0] == pytest.approx(frequency, abs=1e-3)
        assert after[channel][1] == amplitude
    nco = await crs.get_nco_frequency(module=MODULE)
    assert nco == pytest.approx(on_grid(nco), abs=1e-3)
    applied = load_bias_catalog(
        latest_export(tmp_path, "bias_catalog", f"module{MODULE}"))
    assert applied.catalog.names() == catalog.names()


@pytest.mark.asyncio
async def test_a_module_with_no_tones_gives_an_empty_catalog(biased_board):
    crs = biased_board
    await crs.resolve()
    await crs.clear_channels(module=MODULE)
    catalog = await crs.empirically_calibrate_df(module=MODULE, save=False)
    assert len(catalog) == 0
