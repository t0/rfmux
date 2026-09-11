from importlib import util
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from tuber.codecs import TuberResult

from rfmux.core.resonators import ResonatorCatalog
from rfmux.tuning import BiasReport, store

DEMO_PATH = (Path(__file__).resolve().parents[2] / "rfmux" /
             "reference-notebooks" / "Demos" / "simplified_tuning_flow.py")
spec = util.spec_from_file_location("simplified_tuning_flow", DEMO_PATH)
demo = util.module_from_spec(spec)
spec.loader.exec_module(demo)


@pytest.fixture(autouse=True)
def output_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "_output_directory", tmp_path)


@pytest.fixture
def catalog() -> ResonatorCatalog:
    catalog = ResonatorCatalog.from_frequencies(
        [602e6, 605e6], module=2, amplitude=0.008, names=["ALFA", "BETA"])
    catalog["ALFA"].channel = 2
    catalog["BETA"].channel = 5
    return catalog


def samples(channel: int | None = None) -> TuberResult:
    channels = range(1, 6) if channel is None else [channel]
    values = [[float(c)] * 8 for c in channels]
    data = values if channel is None else values[0]
    return TuberResult(
        i=data, q=data,
        spectrum=TuberResult(freq_iq=list(range(8)), freq_dsb=list(range(8)),
                             psd_i=data, psd_q=data, psd_dual_sideband=data))


@pytest.fixture
def fake_crs() -> Mock:
    fake_crs = Mock()
    fake_crs.module = {2: Mock(index=Mock(return_value="crs0000_rmod2"))}
    fake_crs.clear_channels = AsyncMock()
    fake_crs.multisweep = AsyncMock()
    fake_crs.apply_bias = AsyncMock()
    fake_crs.get_decimation = AsyncMock(return_value=6)
    fake_crs.start_udp_streaming = AsyncMock(return_value=True)
    fake_crs.stop_udp_streaming = AsyncMock(return_value=True)
    fake_crs.py_get_samples = AsyncMock(return_value=samples())
    fake_crs.py_get_pfb_samples = AsyncMock(
        side_effect=lambda **kwargs: samples(kwargs["channel"]))
    return fake_crs


@pytest.mark.asyncio
async def test_no_resonances_stops_before_biasing(fake_crs: Mock) -> None:
    fake_crs.take_netanal = AsyncMock(return_value={
        "crs0000_rmod2": {
            "measurement": "netanal", "module": 2,
            "results": {"frequencies": np.linspace(600e6, 610e6, 2000),
                        "iq_counts": np.ones(2000, dtype=complex)},
        },
    })
    with pytest.raises(RuntimeError, match="No resonances found"):
        await demo.run_algorithm_flow(fake_crs, module=2)
    fake_crs.multisweep.assert_not_awaited()
    fake_crs.apply_bias.assert_not_awaited()


@pytest.mark.asyncio
async def test_noise_uses_catalog_channel_numbers(
    fake_crs: Mock, catalog: ResonatorCatalog,
) -> None:
    noise = await demo._acquire_noise(fake_crs, catalog, created_mock=False)
    for name, channel in [("ALFA", 2), ("BETA", 5)]:
        record = noise["resonators"][name]
        assert record["channel"] == channel
        for stream in ("slow", "pfb"):
            np.testing.assert_array_equal(record[stream]["i"], [channel] * 8)
    assert noise["slow_params"]["module"] == catalog.module
    assert noise["pfb_params"]["reset_NCO"] is False


@pytest.mark.asyncio
async def test_capture_failure_stops_owned_stream(
    fake_crs: Mock, catalog: ResonatorCatalog, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(demo, "find_streamer_conflict", lambda: None)
    fake_crs.py_get_pfb_samples.side_effect = RuntimeError("PFB capture failed")
    with pytest.raises(RuntimeError, match="PFB capture failed"):
        await demo._acquire_noise(fake_crs, catalog, created_mock=True)
    fake_crs.stop_udp_streaming.assert_awaited_once()


@pytest.mark.asyncio
async def test_conflict_does_not_touch_existing_stream(
    fake_crs: Mock, catalog: ResonatorCatalog, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(demo, "find_streamer_conflict", lambda: "port 9876 busy")
    with pytest.raises(RuntimeError, match="port 9876 busy"):
        await demo._acquire_noise(fake_crs, catalog, created_mock=True)
    fake_crs.start_udp_streaming.assert_not_awaited()
    fake_crs.stop_udp_streaming.assert_not_awaited()
    fake_crs.py_get_samples.assert_not_awaited()


@pytest.mark.asyncio
async def test_attached_capture_leaves_sender_running(
    fake_crs: Mock, catalog: ResonatorCatalog,
) -> None:
    await demo._acquire_noise(fake_crs, catalog, created_mock=False)
    fake_crs.start_udp_streaming.assert_not_awaited()
    fake_crs.stop_udp_streaming.assert_not_awaited()


@pytest.mark.asyncio
async def test_main_reports_connection_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(demo, "_connect",
                        AsyncMock(side_effect=RuntimeError("connection failed")))
    assert await demo.main(output=tmp_path) == 1
    assert "RuntimeError: connection failed" in capsys.readouterr().err


@pytest.mark.slow_acquisition
@pytest.mark.asyncio
async def test_mock_flow_saves_bias_and_noise_and_stops_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    connected = []
    connect = demo._connect

    async def record_connection(serial: str, hostname: str | None) -> tuple:
        result = await connect(serial, hostname)
        connected.append(result[0])
        return result

    monkeypatch.setattr(demo, "_connect", record_connection)
    assert await demo.main(output=tmp_path) == 0
    crs = connected[0]
    module_id = crs.module[1].index()
    sweeps = store.load(next(tmp_path.glob("*tuning_amplitudes.pkl")))
    report = BiasReport.from_dict(sweeps[module_id]["bias_report"])
    noise = store.load(next(tmp_path.glob("noise_*.pkl")))
    assert len(report.catalog) == 10
    assert set(noise["resonators"]) == set(report.catalog.names())
    nco = await crs.get_nco_frequency(module=1)
    for resonator in report.catalog:
        frequency = nco + await crs.get_frequency(channel=resonator.channel, module=1)
        amplitude = await crs.get_amplitude(channel=resonator.channel, module=1)
        assert abs(frequency - resonator.bias.frequency_hz) < 1.0
        assert amplitude == pytest.approx(resonator.bias.amplitude)
        record = noise["resonators"][resonator.name]
        assert record["channel"] == resonator.channel
        for stream, count in (("slow", 1000), ("pfb", 20000)):
            data = record[stream]
            assert data["i"].shape == data["q"].shape == (count,)
            assert data["freq_iq"].shape == data["psd_i"].shape == data["psd_q"].shape
            positive = data["freq_iq"] > 0
            assert np.all(np.isfinite(data["psd_i"][positive]))
            assert np.all(np.isfinite(data["psd_q"][positive]))
    status = await crs.get_udp_streaming_status()
    assert not status.active and not status.thread_alive
