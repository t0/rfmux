"""The mock's slow emitter: one physics call per block of frames, one
packet per frame, sequence numbers and timestamps running as if each
frame had been emitted on its own."""
import asyncio
from datetime import datetime

import numpy as np
import pytest

from rfmux.mock.udp_streamer import MockCRSStreamer
from rfmux.streamer import SS_PER_SECOND


def _streamer(dec, n_res=2):
    from rfmux.mock.crs import ServerMockCRS
    crs = ServerMockCRS("0000")
    asyncio.run(crs.generate_resonators({
        "num_resonances": n_res, "resonator_random_seed": 11,
        "auto_bias_kids": True, "bias_amplitude": 0.001}))
    crs._fir_stage = dec
    st = MockCRSStreamer(crs)
    st.running = True
    st.slow_socket = None               # build, stamp, count; do not send
    st.start_datetime = datetime(2026, 1, 1)
    return crs, st


def _seconds(ts):
    return ts.h * 3600 + ts.m * 60 + ts.s + ts.ss / SS_PER_SECOND


@pytest.mark.parametrize("dec", [6, 0])
def test_block_length_is_about_50_ms_capped(dec):
    rate = 625e6 / 256 / 64 / 2 ** dec
    n = MockCRSStreamer.slow_block_len(dec)
    assert 1 <= n <= MockCRSStreamer.SLOW_BLOCK_MAX
    assert n == min(MockCRSStreamer.SLOW_BLOCK_MAX, round(0.05 * rate))


def test_one_physics_call_per_block_and_one_packet_per_frame(monkeypatch):
    crs, st = _streamer(6)
    model = crs._resonator_model
    calls = []
    real = model.calculate_module_response_coupled

    def counting(*a, **k):
        calls.append(k.get("num_samples"))
        return real(*a, **k)
    monkeypatch.setattr(model, "calculate_module_response_coupled", counting)

    stamps = []
    real_send = st._send_slow_packet

    def spy(module_num, dec, samples):
        real_send(module_num, dec, samples)
        stamps.append((st.seq_counters[module_num] - 1,
                       _seconds(crs._last_timestamp)))
    monkeypatch.setattr(st, "_send_slow_packet", spy)

    rate = 625e6 / 256 / 64 / 2 ** 6
    st._emit_slow_block(1, 0.0, 6, 30)
    st._emit_slow_block(1, 30 / rate, 6, 30)

    assert calls == [30, 30]
    assert st.packets_sent == 60 and st.seq_counters[1] == 60
    seqs = [s for s, _ in stamps]
    assert seqs == list(range(60))
    dts = np.diff([t for _, t in stamps])
    # One frame apart, across the block boundary too, to the microsecond
    # the datetime path stamps with.
    assert np.allclose(dts, 1 / rate, atol=2e-6)


def test_a_block_of_one_is_the_old_single_frame():
    crs, st = _streamer(6)
    st._emit_slow_packet(1, 0.0, 6)
    assert st.packets_sent == 1 and st.seq_counters[1] == 1


def test_pulses_starting_inside_a_block_reach_its_frames():
    """A 50 ms block is many decay times of a 1 ms pulse.  The
    scheduler used to prune events older than fifteen decay times from
    the span's END before the span was evaluated, so every pulse that
    started inside a block vanished; single frames never noticed."""
    crs, st = _streamer(6)
    m = crs._resonator_model
    m.set_pulse_mode("periodic", period=0.02, tau_rise=1e-6,
                     tau_decay=1e-3, amplitude=3.0, resonators="all")
    rate = 625e6 / 256 / 64 / 2 ** 6
    seen = []
    real_send = st._send_slow_packet

    def spy(module_num, dec, samples):
        seen.append(samples[0])
        real_send(module_num, dec, samples)
    st._send_slow_packet = spy
    t = 0.0
    for _ in range(20):                      # 1 s of stream
        st._emit_slow_block(1, t, 6, 30)
        t += 30 / rate
    x = np.abs(np.array(seen))
    dev = np.abs(x - np.median(x)) / (1.4826 * np.median(np.abs(x - np.median(x))))
    assert (dev > 8).sum() >= 10, "pulses should show in the block's frames"


def test_block_carries_the_signal_on_the_biased_channels():
    """Each frame in the block carries that frame's physics, not the
    block's first sample repeated: the tone amplitude on a biased
    channel is present in every packet and channels without a tone
    are noise only."""
    crs, st = _streamer(6)
    seen = []
    real_send = st._send_slow_packet

    def spy(module_num, dec, samples):
        seen.append(np.abs(samples[:4]).copy())
        real_send(module_num, dec, samples)
    st._send_slow_packet = spy
    st._emit_slow_block(1, 0.0, 6, 8)
    seen = np.array(seen)
    noise = crs._physics_config.get("udp_noise_level", 10.0)
    assert np.all(seen[:, :2] > 20 * noise)      # the two biased tones
    assert np.all(seen[:, 2:] < 10 * noise)      # unbiased: noise only
