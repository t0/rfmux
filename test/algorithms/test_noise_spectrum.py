"""Noise measurement contracts without a board or a UDP sender."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from tuber.codecs import TuberResult

from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog
from rfmux.core.schema import CRS
from rfmux.core.transferfunctions import (
    PFB_SAMPLING_FREQ, VOLTS_PER_ROC, convert_dacunits_to_dbm,
)
from rfmux.algorithms.measurement.noise_spectrum import measure_noise
from rfmux.tuning import store
from rfmux.tuning.sweep_results import RESULTS_SCHEMA_VERSION

pytestmark = [pytest.mark.portable, pytest.mark.asyncio]

measure_noise = measure_noise.__wrapped__


class ToneContext:
    async def __aenter__(self):
        self.values = []
        return self

    async def __aexit__(self, *args):
        pass

    def get_frequency(self, *, channel, module):
        self.values.append(channel * 1e6)

    def get_amplitude(self, *, channel, module):
        self.values.append(None if channel == 7 else channel / 100)

    async def __call__(self):
        return self.values


def capture(channels=None, reference="absolute"):
    counts = np.arange(8) + 2j
    if channels is not None:
        counts = np.array([counts + c for c in range(1, channels + 1)])
    samples = counts * (VOLTS_PER_ROC if reference == "absolute" else 1)
    spectra = np.arange(4.)
    if channels is not None:
        spectra = np.array([spectra + c for c in range(1, channels + 1)])
    return TuberResult(
        i=samples.real.tolist(), q=samples.imag.tolist(),
        ts=[TuberResult(s=np.uint32(100), ss=np.uint32(i), source=1)
            for i in range(8)],
        spectrum=TuberResult(freq_iq=[0., 1., 2., 3.],
                             freq_dsb=[-2., -1., 0., 1.],
                             psd_i=spectra.tolist(), psd_q=spectra.tolist(),
                             psd_dual_sideband=spectra.tolist()))


@pytest.fixture
def board():
    return SimpleNamespace(
        modules=SimpleNamespace(module=[1, 2, 3, 4]),
        module={1: SimpleNamespace(index=lambda: "crs1234_rmod1")},
        get_analog_bank=AsyncMock(return_value=False),
        get_decimation=AsyncMock(return_value=6),
        set_decimation=AsyncMock(),
        get_dac_scale=AsyncMock(return_value=None),
        get_nco_frequency=AsyncMock(return_value=1e9),
        get_pfb_streamer=AsyncMock(return_value=None),
        tuber_context=ToneContext,
        py_get_samples=AsyncMock(side_effect=lambda **kw: capture(7, kw["reference"])),
        py_get_pfb_samples=AsyncMock(side_effect=lambda **kw: capture(None, kw["reference"])),
    )


async def measure(board, **kwargs):
    return await measure_noise(
        board, channels=[7, 2], module=1, num_samples=8, nsegments=2,
        save=False, **kwargs)


@pytest.mark.parametrize("reference", ["absolute", "relative"])
async def test_sparse_channel_mapping_units_and_schema(board, reference):
    result = await measure(board, reference=reference)
    block = result["crs1234_rmod1"]
    assert block["schema_version"] == RESULTS_SCHEMA_VERSION
    assert block["measurement"] == "noise"
    assert block["dac_scale_dbm"] is None
    records = block["results"]["resonators"]
    info = block["results"]["info"]
    assert info["iq_units"] == ("volts" if reference == "absolute" else "counts")
    assert "frequency_units" not in info
    assert list(records) == ["CH0007", "CH0002"]
    for channel in (7, 2):
        record = records[f"CH{channel:04d}"]
        np.testing.assert_allclose(record["slow_data"][f"iq_{info['iq_units']}"],
                                   (np.arange(8) + channel + 2j) *
                                   (VOLTS_PER_ROC if reference == "absolute" else 1))
        np.testing.assert_array_equal(record["slow_data"]["psd_i"],
                                      np.arange(4) + channel)
        assert "freq_iq" not in record["slow_data"]
        assert "pfb_data" not in record
        assert record["bias_frequency_hz"] == 1e9 + channel * 1e6
    assert records["CH0007"]["bias_amplitude"] is None
    assert records["CH0007"]["bias_amplitude_dbm"] is None
    assert records["CH0002"]["bias_amplitude"] == .02
    assert records["CH0002"]["bias_amplitude_dbm"] is None
    assert block["results"]["shared_slow"]["timestamps"][0] == \
        dict(s=100, ss=0, source=1)
    assert block["results"]["info"]["carrier_bin_units"] == (
        "dBm/Hz" if reference == "absolute" else "dBc")
    board.set_decimation.assert_not_awaited()
    assert hasattr(CRS, "measure_noise")


async def test_catalog_is_snapshotted_before_acquisition(board):
    catalog = ResonatorCatalog([
        Resonator(name="A", channel=7, bias=BiasPoint(1e9, .1)),
        Resonator(name="B", channel=2, bias=BiasPoint(1.1e9, .2)),
    ], module=1)
    before = catalog.to_dict()
    board.get_dac_scale.return_value = -1.5

    def acquire(**kw):
        catalog["A"].bias = BiasPoint(1e9, .8)
        return capture(7)

    board.py_get_samples.side_effect = acquire
    result = await measure_noise(board, catalog, num_samples=8,
                                       nsegments=2, save=False)
    block = next(iter(result.values()))
    assert block["call_params"]["catalog"] == before
    assert block["results"]["resonators"]["A"]["channel"] == 7
    assert block["dac_scale_dbm"] == -1.5
    record = block["results"]["resonators"]["B"]
    assert record["bias_amplitude_dbm"] == pytest.approx(
        convert_dacunits_to_dbm(record["bias_amplitude"], -1.5))


@pytest.mark.parametrize("reference", ["absolute", "relative"])
async def test_optional_pfb_spacing_and_progress(board, reference):
    progress = Mock()
    result = await measure(board, pfb_samples=8, progress_callback=progress,
                           reference=reference)
    records = next(iter(result.values()))["results"]["resonators"]
    for record in records.values():
        np.testing.assert_allclose(record["pfb_data"]["iq_volts" if reference == "absolute" else "iq_counts"],
                                   (np.arange(8) + 2j) *
                                   (VOLTS_PER_ROC if reference == "absolute" else 1))
    np.testing.assert_allclose(
        next(iter(result.values()))["results"]["shared_pfb"]["time_s"],
        np.arange(8) / PFB_SAMPLING_FREQ)
    assert [c.args[0] for c in progress.call_args_list] == [
        dict(stream="slow", channel=None, completed=1, total=3),
        dict(stream="pfb", channel=7, completed=2, total=3),
        dict(stream="pfb", channel=2, completed=3, total=3),
    ]
    assert board.py_get_pfb_samples.call_args.kwargs["reset_NCO"] is False


@pytest.mark.parametrize("overrides", [
    dict(channels=[]), dict(channels=[2, 2]), dict(channels=[0]),
    dict(channels=[True]), dict(module=1.5), dict(decimation=7),
    dict(num_samples=1), dict(nsegments=0), dict(nsegments=5),
    dict(num_samples=8.5), dict(reference="unknown"),
    dict(spectrum_cutoff=float("nan")), dict(spectrum_cutoff=0),
    dict(pfb_samples=10_000_001), dict(pfb_samples=8, pfb_nsegments=3),
    dict(pfb_nsegments=2), dict(progress_callback=42),
    dict(decimation=2, channels=[129]),
])
async def test_invalid_request_never_mutates_or_acquires(board, overrides):
    kwargs = dict(channels=[2], module=1, num_samples=8, nsegments=2,
                  decimation=3, save=False)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        await measure_noise(board, **kwargs)
    board.set_decimation.assert_not_awaited()
    board.py_get_samples.assert_not_awaited()


async def test_explicit_decimation_selects_only_requested_module(board):
    await measure(board, decimation=2)
    board.set_decimation.assert_awaited_once_with(2, short=True, module=1)


async def test_same_decimation_does_not_change_stream_configuration(board):
    await measure(board, decimation=6)
    board.set_decimation.assert_not_awaited()


async def test_active_pfb_stream_is_rejected_before_decimation_change(board):
    board.get_pfb_streamer.return_value = [2]
    with pytest.raises(ValueError, match="Disable the PFB"):
        await measure(board, pfb_samples=8, decimation=3)
    board.set_decimation.assert_not_awaited()


@pytest.mark.parametrize("failure", [RuntimeError("capture failed"), asyncio.CancelledError()])
async def test_interrupted_pfb_does_not_save_partial_measurement(board, monkeypatch, failure):
    save = Mock()
    monkeypatch.setattr(store, "maybe_save", save)
    board.py_get_pfb_samples.side_effect = failure
    with pytest.raises(type(failure)):
        await measure(board, pfb_samples=8)
    save.assert_not_called()


async def test_save_load_and_resave_use_one_file(board, tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_output_directory", tmp_path)
    result = await measure_noise(board, module=1, channels=[2],
                                       num_samples=8, nsegments=2, save=True)
    path = store.saved_path(result)
    loaded = store.load(path)
    assert store.save(loaded) == path
    assert len(list(tmp_path.glob("*.pkl"))) == 1
    block = loaded["crs1234_rmod1"]
    np.testing.assert_allclose(block["results"]["resonators"]["CH0002"]
                               ["slow_data"]["iq_volts"],
                               (np.arange(8) + 2 + 2j) * VOLTS_PER_ROC)


@pytest.mark.parametrize("reference", ["absolute", "relative"])
async def test_actual_pfb_helper_preserves_native_iq_and_spectral_values(board, reference):
    from rfmux.algorithms.measurement.py_get_pfb_samples import py_get_pfb_samples

    helper = py_get_pfb_samples.__wrapped__
    counts = np.random.default_rng(42).normal(size=64) + 30 + 5j
    board.get_frequency = AsyncMock(return_value=0.)
    board.get_pfb_samples = AsyncMock(return_value=TuberResult(
        i=counts.real.tolist(), q=counts.imag.tolist()))

    async def acquire(**kwargs):
        return await helper(board, **kwargs)

    board.py_get_pfb_samples.side_effect = acquire
    expected = await acquire(nsamps=64, channel=2, module=1, nsegments=2,
                             reference=reference, trim=False)
    result = await measure(board, pfb_samples=64, reference=reference)
    actual = result["crs1234_rmod1"]["results"]["resonators"]["CH0002"]["pfb_data"]
    np.testing.assert_array_equal(
        actual["iq_volts" if reference == "absolute" else "iq_counts"],
        np.asarray(expected.i) + 1j * np.asarray(expected.q))
    for key in ("freq_iq", "freq_dsb", "psd_i", "psd_q", "psd_dual_sideband"):
        np.testing.assert_array_equal(actual[key], getattr(expected.spectrum, key))


@pytest.mark.parametrize("kwargs", [dict(channels=[2]), dict(module=2), dict(module=True)])
async def test_catalog_rejects_conflicting_or_invalid_selection(board, kwargs):
    catalog = ResonatorCatalog([
        Resonator(name="A", channel=7, bias=BiasPoint(1e9, .1)),
    ], module=1)
    with pytest.raises(ValueError):
        await measure_noise(board, catalog, save=False, **kwargs)
    board.set_decimation.assert_not_awaited()
    board.py_get_samples.assert_not_awaited()


async def test_disabled_slow_stream_is_not_started(board):
    board.get_decimation.return_value = None
    with pytest.raises(ValueError, match="Enable the slow stream"):
        await measure(board, decimation=3)
    board.set_decimation.assert_not_awaited()


async def test_channel_absent_from_current_packet_width_is_rejected(board):
    board.py_get_samples.side_effect = lambda **kw: capture(2)
    with pytest.raises(ValueError, match="Channel 7 is absent"):
        await measure(board)


async def test_calibrated_noise_saves_display_spectra(board, tmp_path):
    from rfmux.algorithms.measurement.noise_display import noise_display_products
    catalog = ResonatorCatalog([
        Resonator(name="calibrated", channel=2,
                  bias=BiasPoint(1e9, .1, dI_df=2., dQ_df=3.)),
    ], module=1)
    result = await measure_noise(board, catalog, num_samples=8,
                                       nsegments=2, save=False)
    block = next(iter(result.values()))
    assert block["results"]["info"]["nco_frequency_hz"] == \
        await board.get_nco_frequency()
    data = block["results"]["resonators"]["calibrated"]["slow_data"]
    assert set(data["display_psds"]) == {"volts", "df"}
    path = store.save(result, "noise", directory=tmp_path)
    loaded = next(iter(store.load(path).values()))
    plotted = noise_display_products(loaded, "calibrated", units="df")
    np.testing.assert_array_equal(plotted["psd_i"], data["display_psds"]["df"]["psd_i"])


@pytest.mark.parametrize("pfb", [False, True])
async def test_noise_worker_uses_warmed_hardware_map(board, pfb):
    from concurrent.futures import ThreadPoolExecutor
    from rfmux.core.hardware_map import HardwareMap, warm_for_threads
    from rfmux.core.schema import ReadoutModule
    from rfmux.algorithms.measurement.py_get_pfb_samples import py_get_pfb_samples

    session = HardwareMap()
    crs = CRS(serial="1234", hostname="localhost", modules=[
        ReadoutModule(module=1, channels=[])])
    session.add(crs)
    session.commit()
    warm_for_threads(crs)
    for name, value in vars(board).items():
        if name not in ("module", "modules"):
            setattr(crs, name, value)
    crs.get_frequency = AsyncMock(return_value=0.)
    crs.get_pfb_samples = AsyncMock(return_value=TuberResult(
        i=np.arange(64.).tolist(), q=np.ones(64).tolist()))

    async def capture_pfb(**kwargs):
        return await py_get_pfb_samples.__wrapped__(crs, **kwargs)

    crs.py_get_pfb_samples = capture_pfb

    def measure_in_worker():
        return asyncio.run(measure_noise(
            crs, module=1, channels=[2], num_samples=8, nsegments=2,
            pfb_samples=64 if pfb else None, save=False))

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(measure_in_worker).result(timeout=10)
        record = result["crs1234_rmod1"]["results"]["resonators"]["CH0002"]
        assert len(record["slow_data"]["iq_volts"]) == 8
        if pfb:
            assert len(record["pfb_data"]["iq_volts"]) == 64
    finally:
        session.close()
