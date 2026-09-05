"""The mock streamer's block loop: one stream clock shared by the
physics, the slow stamps and the PFB packets, whichever modules carry
tones; PFB channel changes and module failures confined to their own
stream."""
import asyncio
from datetime import datetime

import numpy as np
import pytest

from rfmux.core.transferfunctions import (
    CIC1_DECIMATION, decimated_stream_delay_s, decimation_to_sampling)
from rfmux.mock.udp_streamer import MockCRSStreamer
from rfmux.streamer import SS_PER_SECOND


def _crs(**cfg):
    from rfmux.mock.crs import ServerMockCRS
    crs = ServerMockCRS("0000")
    asyncio.run(crs.generate_resonators({
        "num_resonances": 2, "resonator_random_seed": 11,
        "auto_bias_kids": True, "bias_amplitude": 0.001, **cfg}))
    crs._fir_stage = 6
    return crs


def _streamer(crs):
    st = MockCRSStreamer(crs)
    st.running = True
    st.slow_socket = None               # build, stamp, count; do not send
    st._init_pfb_socket = lambda: None
    st.start_datetime = datetime(2026, 1, 1)
    return st


def _seconds(ts):
    return ts.h * 3600 + ts.m * 60 + ts.s + ts.ss / SS_PER_SECOND


def _tone_on_module_2(crs):
    asyncio.run(crs.set_nco_frequency(1e9, module=2))
    asyncio.run(crs.set_frequency(1e6, channel=1, module=2))
    asyncio.run(crs.set_amplitude(0.001, channel=1, module=2))


def test_stream_time_advances_when_only_module_2_carries_a_tone(monkeypatch):
    crs = _crs(auto_bias_kids=False)
    _tone_on_module_2(crs)
    st = _streamer(crs)
    assert st._get_configured_modules() == [2]
    starts = []
    model = crs._resonator_model
    real = model.calculate_module_response_coupled

    def spy(*a, **k):
        starts.append(k["start_time"])
        return real(*a, **k)
    monkeypatch.setattr(model, "calculate_module_response_coupled", spy)

    st._run_block()
    st._run_block()
    block = MockCRSStreamer.slow_block_len(6) / decimation_to_sampling(6)
    assert starts == pytest.approx([0.0, block])


def test_slow_stamps_of_a_late_module_follow_the_stream_clock():
    """A module first toned after the stream has run is stamped at the
    stream time its samples were computed for, like module 1."""
    crs = _crs()
    st = _streamer(crs)
    for _ in range(4):
        st._run_block()
    _tone_on_module_2(crs)
    first_stamp = {}
    real = st._send_slow_packet

    def spy(module_num, dec, samples, t_frame):
        real(module_num, dec, samples, t_frame)
        first_stamp.setdefault(module_num, _seconds(crs._last_timestamp))
    st._send_slow_packet = spy

    t_block = st.t_stream
    assert t_block > 0.1
    st._run_block()
    assert first_stamp[2] == pytest.approx(
        t_block + decimated_stream_delay_s(6), abs=2e-6)
    assert first_stamp[2] == first_stamp[1]


def test_pfb_frames_leave_the_block_pulses_for_the_slow_stream():
    """With PFB enabled, a pulse a few decay times shorter than the
    block still shows on the slow stream: the PFB frames' bookkeeping
    must not discard it before the slow block is evaluated."""
    crs = _crs()
    st = _streamer(crs)
    st.enable_pfb([1], 1)
    model = crs._resonator_model
    model.set_pulse_mode("manual", tau_rise=1e-6, tau_decay=1e-3,
                         amplitude=3.0)
    model.add_pulse_event(0, st.t_stream - 1e-5, amplitude=3.0,
                          tau_decay=1e-3)
    seen = []
    real = st._send_slow_packet

    def spy(module_num, dec, samples, t_frame):
        seen.append(samples[0])
        real(module_num, dec, samples, t_frame)
    st._send_slow_packet = spy

    st._run_block()
    x = np.abs(np.array(seen))
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    assert abs(x[0] - med) > 8 * mad, "the pulse is missing from the slow block"


def test_block_pulses_trigger_on_the_pfb_grid_when_pfb_is_on():
    """The slow block is evaluated before the PFB frames, but its pulses
    are still scheduled on the 26 us sub-batch grid, not the frame grid."""
    from rfmux.mock.udp_streamer import PFB_BATCH
    from rfmux.core.transferfunctions import PFB_SAMPLING_FREQ
    crs = _crs()
    st = _streamer(crs)
    st.enable_pfb([1], 1)
    model = crs._resonator_model
    model.set_pulse_mode("periodic", period=0.0007, tau_rise=1e-6,
                         tau_decay=1e-4, amplitude=3.0)
    starts = set()
    real = st._emit_pfb_frame

    def spy(t_frame, dec):
        starts.update(p["start_time"] for p in model.pulse_events)
        real(t_frame, dec)
    st._emit_pfb_frame = spy

    st._run_block()
    sub = PFB_BATCH / PFB_SAMPLING_FREQ
    frame = 1.0 / decimation_to_sampling(6)
    assert starts, "no pulse triggered"
    assert all(abs(s / sub - round(s / sub)) < 1e-6 for s in starts)
    assert any(abs(s / frame - round(s / frame)) > 1e-6 for s in starts)


def test_pfb_channel_change_takes_effect_at_the_block_boundary():
    """enable_pfb is called from the server thread; the frame the
    streamer is cutting keeps its channel set until the next block."""
    crs = _crs()
    st = _streamer(crs)
    st.enable_pfb([1], 1)
    st._run_block()
    widths = []
    real = st._send_pfb_packet

    def spy(interleaved, t_first):
        widths.append(len(st.pfb_channels))
        real(interleaved, t_first)
    st._send_pfb_packet = spy

    st.enable_pfb([1, 2], 1)
    st._emit_pfb_frame(st.t_stream, 6)
    assert widths and set(widths) == {1}
    st._run_block()
    assert widths[-1] == 2


def test_a_module_whose_physics_fails_does_not_stop_the_others(monkeypatch, capsys):
    crs = _crs()
    _tone_on_module_2(crs)
    st = _streamer(crs)
    assert st._get_configured_modules() == [1, 2]
    model = crs._resonator_model
    real = model.calculate_module_response_coupled

    def failing(module, *a, **k):
        if module == 2:
            raise RuntimeError("module 2 physics")
        return real(module, *a, **k)
    monkeypatch.setattr(model, "calculate_module_response_coupled", failing)

    n = MockCRSStreamer.slow_block_len(6)
    seconds = st._run_block()
    assert seconds == pytest.approx(n / decimation_to_sampling(6))
    assert st.seq_counters[1] == n and st.seq_counters[2] == 0
    st._run_block()
    assert st.seq_counters[1] == 2 * n
    assert capsys.readouterr().out.count("module 2 physics") == 1


def test_a_failing_pfb_stream_does_not_hold_the_clock(monkeypatch, capsys):
    """The slow packets of a block go out before its PFB frames; a PFB
    failure must not replay the block at the same time."""
    crs = _crs()
    st = _streamer(crs)
    st.enable_pfb([1], 1)

    def boom(t_frame, dec):
        raise RuntimeError("pfb physics")
    monkeypatch.setattr(st, "_emit_pfb_frame", boom)
    stamps = []
    real = st._send_slow_packet

    def spy(module_num, dec, samples, t_frame):
        real(module_num, dec, samples, t_frame)
        stamps.append(_seconds(crs._last_timestamp))
    st._send_slow_packet = spy

    st._run_block()
    st._run_block()
    n = MockCRSStreamer.slow_block_len(6)
    assert st.t_stream == pytest.approx(2 * n / decimation_to_sampling(6))
    assert len(stamps) == 2 * n and np.all(np.diff(stamps) > 0)
    assert capsys.readouterr().out.count("pfb physics") == 1


def test_three_pfb_channels_are_rejected():
    crs = _crs()
    with pytest.raises(ValueError, match="1, 2 or 4"):
        asyncio.run(crs.set_pfb_streamer(channel=[1, 2, 3], module=1))
    assert asyncio.run(crs.get_pfb_streamer(module=1)) is None


@pytest.mark.parametrize("dec", [0, 6])
def test_pfb_noise_is_the_slow_floor_scaled_by_the_decimation_ratio(dec):
    """udp_noise_level is the slow stream's sigma at every stage; the
    PFB stream, decimated by 64 * 2**dec into it, carries white noise
    root-that-many times larger."""
    crs = _crs()
    crs._fir_stage = dec
    st = _streamer(crs)
    st.pfb_channels, st.pfb_module, st.pfb_enabled = [4], 1, True   # no tone: noise only
    samples = []
    st._send_pfb_packet = lambda interleaved, t_first: samples.append(interleaved)
    rate = decimation_to_sampling(dec)
    for k in range(20):
        st._emit_pfb_frame(k / rate, dec)
    x = np.concatenate(samples)
    sigma = crs._physics_config["udp_noise_level"]
    assert np.std(x.real) == pytest.approx(
        sigma * np.sqrt(CIC1_DECIMATION * 2 ** dec), rel=0.1)
