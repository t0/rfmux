"""Building a biased mock warms the pulse caches: the first real pulse
converges nothing new, and the warm-up leaves no trace in time or in
pulse state."""
import asyncio
from datetime import datetime

from rfmux.mock.udp_streamer import MockCRSStreamer

PULSE = {"pulse_tau_rise": 1e-6, "pulse_tau_decay": 1e-3,
         "pulse_amplitude": 2.0, "pulse_random_amp_mode": "fixed",
         "pulse_random_tau_mode": "fixed"}


def _built():
    from rfmux.mock.crs import ServerMockCRS
    crs = ServerMockCRS("0000")
    asyncio.run(crs.generate_resonators({
        "num_resonances": 3, "resonator_random_seed": 11,
        "auto_bias_kids": True, "pulse_mode": "none", **PULSE}))
    crs._fir_stage = 6
    st = MockCRSStreamer(crs)
    st.running = True
    st.slow_socket = None
    st.start_datetime = datetime(2026, 1, 1)
    return crs, st


def test_warm_up_leaves_time_and_pulse_state_alone():
    crs, _ = _built()
    m = crs._resonator_model
    assert m.last_update_time == 0
    assert m.pulse_config["mode"] == "none"
    assert m.pulse_events == [] and m.last_pulse_time == {}
    assert m._convergence_cache, "the warm-up should have left caches"


def test_first_real_pulse_converges_nothing_new():
    crs, st = _built()
    m = crs._resonator_model
    rate = 625e6 / 256 / 64 / 2 ** 6
    n = MockCRSStreamer.slow_block_len(6)
    t = 0.0
    for _ in range(3):
        st._emit_slow_block(1, t, 6, n); t += n / rate
    m.set_pulse_mode("periodic", period=0.5, tau_rise=1e-6, tau_decay=1e-3,
                     amplitude=2.0, resonators="all")
    before = m._convergence_counter
    for _ in range(6):                       # 0.3 s: the first pulse and its decay
        st._emit_slow_block(1, t, 6, n); t += n / rate
    assert m._convergence_counter == before, \
        f"{m._convergence_counter - before} convergences on the first pulse"
