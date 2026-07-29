"""
Tests for the headless streamer-configuration layer and packet sources.

Pure-math tests for describe()/validate(); MockCRS integration for
apply/read (including the modules= spelling and the relaxed stage-3
long-packet rule); live-socket tests feeding a PulseCaptureSession
through run_slow_source / run_pfb_source.
"""

import asyncio

import numpy as np
import pytest

from rfmux.algorithms.measurement.streamer_config import (
    DERATED_LINK_MBPS,
    PFB_SAMPLE_RATE,
    StreamerConfig,
    apply_streamer_config,
    describe,
    read_streamer_config,
    slow_sample_rate,
    validate,
)


def _severities(issues):
    return [sev for sev, _ in issues]


class TestDescribe:
    def test_rates(self):
        assert slow_sample_rate(0) == pytest.approx(38146.97265625)
        assert slow_sample_rate(6) == pytest.approx(596.0464477539062)
        assert PFB_SAMPLE_RATE == pytest.approx(1220703.125)

    def test_bandwidth_math(self):
        d = describe(StreamerConfig(dec_stage=6, short_packets=False,
                                    modules=[1, 2, 3, 4]))
        # 8240 B * 8 * 596.05 Hz * 4 modules
        assert d["slow_mbps"] == pytest.approx(157.2, rel=0.01)
        assert d["channels_per_module"] == 1024

        d = describe(StreamerConfig(dec_stage=0, short_packets=True,
                                    modules=[1], pfb_channels=[1, 2]))
        assert d["channels_per_module"] == 128
        # 1072 B * 8 * 38147 Hz
        assert d["slow_mbps"] == pytest.approx(327.2, rel=0.01)
        # 2 ch * 8056 B/1000 samp * 8 * 1.2207 MHz
        assert d["pfb_mbps"] == pytest.approx(157.4, rel=0.01)
        assert d["total_mbps"] == pytest.approx(484.5, rel=0.01)


class TestValidate:
    def test_long_below_stage3_is_error(self):
        issues = validate(StreamerConfig(dec_stage=0, short_packets=False))
        assert "error" in _severities(issues)

    def test_short_any_stage_ok(self):
        for stage in range(0, 7):
            issues = validate(StreamerConfig(
                dec_stage=stage, short_packets=True, modules=[1]))
            assert "error" not in _severities(issues), \
                f"short packets must be legal at stage {stage}: {issues}"

    def test_long_stage3_single_module_ok_but_four_modules_over_budget(self):
        ok = validate(StreamerConfig(dec_stage=3, short_packets=False,
                                     modules=[1]))
        assert "error" not in _severities(ok)
        over = validate(StreamerConfig(dec_stage=3, short_packets=False,
                                       modules=[1, 2, 3, 4]))
        assert any(sev == "error" and "Mbps" in msg for sev, msg in over)

    def test_low_dec_os_buffer_warning(self):
        issues = validate(StreamerConfig(dec_stage=1, short_packets=True,
                                         modules=[1]))
        assert any("UDP buffer" in msg for _, msg in issues)

    def test_pfb_rules(self):
        too_many = validate(StreamerConfig(
            dec_stage=6, pfb_channels=[1, 2, 3, 4, 5]))
        assert any(sev == "error" and "4 channels" in msg
                   for sev, msg in too_many)
        four = validate(StreamerConfig(dec_stage=6,
                                       pfb_channels=[1, 2, 3, 4]))
        assert "error" not in _severities(four)
        assert any("get_pfb_samples" in msg for _, msg in four)

    def test_bad_stage_short_circuits(self):
        issues = validate(StreamerConfig(dec_stage=9))
        assert _severities(issues) == ["error"]


@pytest.fixture(scope="module")
def mock_crs():
    from rfmux.mock.helpers import create_mock_crs
    loop = asyncio.new_event_loop()
    crs = loop.run_until_complete(create_mock_crs(
        module=1, config={"num_resonances": 1,
                          "resonator_random_seed": 7},
        verbose=False))
    yield loop, crs
    try:
        loop.run_until_complete(crs.stop_udp_streaming())
    except Exception:
        pass
    loop.close()


class TestApplyOnMock:
    def test_apply_and_read(self, mock_crs):
        loop, crs = mock_crs
        info = loop.run_until_complete(apply_streamer_config(
            crs, StreamerConfig(dec_stage=1, short_packets=True,
                                modules=[1])))
        assert info["sample_rate_hz"] == pytest.approx(slow_sample_rate(1))
        state = loop.run_until_complete(read_streamer_config(crs))
        assert state["dec_stage"] == 1

    def test_modules_spelling_accepted_by_mock(self, mock_crs):
        loop, crs = mock_crs
        loop.run_until_complete(crs.set_decimation(6, short=False,
                                                   modules=[1, 2]))

    def test_stage3_long_now_allowed(self, mock_crs):
        loop, crs = mock_crs
        loop.run_until_complete(crs.set_decimation(3, short=False,
                                                   module=[1]))

    def test_stage2_long_still_refused(self, mock_crs):
        loop, crs = mock_crs
        with pytest.raises(ValueError):
            loop.run_until_complete(apply_streamer_config(
                crs, StreamerConfig(dec_stage=2, short_packets=False)))

    def test_pfb_enable_disable_roundtrip(self, mock_crs):
        loop, crs = mock_crs
        loop.run_until_complete(apply_streamer_config(
            crs, StreamerConfig(dec_stage=6, short_packets=False,
                                modules=[1], pfb_channels=[1, 2])))
        assert loop.run_until_complete(
            crs.get_pfb_streamer(module=1)) == [1, 2]
        loop.run_until_complete(apply_streamer_config(
            crs, StreamerConfig(dec_stage=6, short_packets=False,
                                modules=[1], pfb_channels=[])))
        assert loop.run_until_complete(
            crs.get_pfb_streamer(module=1)) is None

    def test_configure_streamer_macro(self, mock_crs):
        loop, crs = mock_crs
        info = loop.run_until_complete(crs.configure_streamer(
            6, short=False, modules=[1]))
        assert info["n_modules"] == 1


class TestSources:
    def test_slow_source_feeds_session(self, mock_crs):
        from rfmux.algorithms.measurement.pulse_capture_session import (
            CaptureState, PulseCaptureSession)
        from rfmux.algorithms.measurement.pulse_sources import (
            run_slow_source)

        loop, crs = mock_crs
        loop.run_until_complete(crs.set_decimation(6, short=False,
                                                   modules=[1]))
        # Let in-flight packets from earlier decimation settings drain
        loop.run_until_complete(asyncio.sleep(0.3))
        session = PulseCaptureSession(channels=[1], noise_samples=50,
                                      hdf5_path=None)
        session.start()
        elapsed = loop.run_until_complete(run_slow_source(
            session, "127.0.0.1", module=1, duration_s=0.15))
        assert session.state is CaptureState.CAPTURING, \
            f"state={session.state}, elapsed={elapsed}"
        assert session.noise_stats
        session.stop()

    def test_pfb_source_feeds_session(self, mock_crs):
        from rfmux.algorithms.measurement.pulse_capture_session import (
            CaptureState, PulseCaptureSession)
        from rfmux.algorithms.measurement.pulse_sources import (
            run_pfb_source)

        loop, crs = mock_crs
        loop.run_until_complete(apply_streamer_config(
            crs, StreamerConfig(dec_stage=6, short_packets=False,
                                modules=[1], pfb_channels=[1, 2])))
        try:
            session = PulseCaptureSession(
                channels=[1, 2], streamer_mode="fast",
                sample_rate=PFB_SAMPLE_RATE, noise_samples=400,
                hdf5_path=None)
            session.start()
            elapsed = loop.run_until_complete(run_pfb_source(
                session, "127.0.0.1", [1, 2], duration_s=0.01))
            assert session.state is CaptureState.CAPTURING, \
                f"state={session.state}, elapsed={elapsed}"
            assert set(session.noise_stats) == {1, 2}
            session.stop()
        finally:
            loop.run_until_complete(crs.set_pfb_streamer(channel=None,
                                                         module=1))
