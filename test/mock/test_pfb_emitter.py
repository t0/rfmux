"""The mock's PFB emitter: one physics call per slow frame, cut into
hardware-sized packets whose timestamps run contiguously across frames."""
import asyncio

import numpy as np
import pytest

from rfmux.core.transferfunctions import PFB_SAMPLING_FREQ
from rfmux.mock.udp_streamer import MockCRSStreamer, PFB_BATCH


def _streamer(dec, channels):
    from rfmux.mock.crs import ServerMockCRS
    crs = ServerMockCRS("0000")
    asyncio.run(crs.generate_resonators({
        "num_resonances": 2, "resonator_random_seed": 11,
        "auto_bias_kids": True, "bias_amplitude": 0.001}))
    crs._fir_stage = dec
    st = MockCRSStreamer(crs)
    st.pfb_channels = list(channels)
    st.pfb_module = 1
    st.pfb_enabled = True
    st.running = True
    st.pfb_socket = None                # build, stamp, count; do not send
    from datetime import datetime
    st.start_datetime = datetime(2026, 1, 1)
    return crs, st


@pytest.mark.parametrize("dec, channels", [(6, [1]), (6, [1, 2]), (0, [1])])
def test_one_physics_call_per_frame_and_hardware_sized_packets(dec, channels, monkeypatch):
    crs, st = _streamer(dec, channels)
    model = crs._resonator_model
    calls = {"n": 0}
    real = model.calculate_module_response_coupled

    def counting(*a, **k):
        calls["n"] += 1
        assert k.get("num_samples") == PFB_BATCH * 2 ** dec
        return real(*a, **k)
    monkeypatch.setattr(model, "calculate_module_response_coupled", counting)

    stamps = []
    real_send = st._send_pfb_packet

    def spy(interleaved, t_first):
        stamps.append((len(interleaved), t_first))
        real_send(interleaved, t_first)
    monkeypatch.setattr(st, "_send_pfb_packet", spy)

    frame_time = 1.0 / (625e6 / 256 / 64 / 2 ** dec)
    n_frames = 40
    for k in range(n_frames):
        st._emit_pfb_frame(k * frame_time, dec)

    assert calls["n"] == n_frames
    n_groups = len(channels)
    per_packet = (1000 // n_groups) * n_groups
    assert stamps, "no packet in 40 frames"
    assert {n for n, _ in stamps} == {per_packet}
    expected = n_frames * PFB_BATCH * 2 ** dec * n_groups // per_packet
    assert len(stamps) == expected
    # Contiguous: each packet is stamped one packet-span after the last.
    t = np.array([t for _, t in stamps])
    assert np.allclose(np.diff(t), (per_packet // n_groups) / PFB_SAMPLING_FREQ, rtol=1e-9)
    assert t[0] == 0.0
    assert st.pfb_packets_sent == expected


def test_pulse_triggers_stay_on_the_sub_batch_grid():
    """A periodic pulse due mid-frame starts on the 26 us sub-batch grid."""
    crs, st = _streamer(6, [1])
    model = crs._resonator_model
    model.set_pulse_mode("periodic", period=0.0007, tau_rise=1e-6,
                         tau_decay=1e-4, amplitude=3.0)
    frame_time = 1.0 / (625e6 / 256 / 64 / 2 ** 6)
    for k in range(3):
        st._emit_pfb_frame(k * frame_time, 6)
    starts = sorted({p["start_time"] for p in model.pulse_events})
    sub = PFB_BATCH / PFB_SAMPLING_FREQ
    assert starts, "no pulse triggered"
    for s in starts:
        assert abs(s / sub - round(s / sub)) < 1e-6, f"start {s} off the grid"
    # ...and not only on frame boundaries.
    assert any(abs(s / frame_time - round(s / frame_time)) > 1e-6 for s in starts)
