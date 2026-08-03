"""
Mock stream-scale parity: slow and PFB paths must share one scale.

Real hardware is dec-invariant and stream-invariant — a fixed carrier
reads the same counts on every CIC stage and on the PFB streamer under
the uniform consumer convention (np.array(pkt)/256 for both).  This
locks the mock's model to that behavior at every decimation stage.

Carrier-only (no pulses): with QP pulses active the carrier baseline
drifts with quasiparticle density, and short pulses are legitimately
temporally unresolved on the slow stream — so stream-data amplitude
comparisons are NOT valid invariants; this model-level one is.
"""

import asyncio

import numpy as np
import pytest

pytestmark = pytest.mark.slow_acquisition  # heavy resonator generation


@pytest.fixture(scope="module")
def model():
    from rfmux.mock.crs import ServerMockCRS
    crs = ServerMockCRS("0000")
    asyncio.run(crs.generate_resonators({
        "num_resonances": 2, "resonator_random_seed": 3,
        "auto_bias_kids": True, "bias_amplitude": 0.001}))
    yield crs, crs._resonator_model


def test_carrier_parity_across_dec_stages(model):
    crs, model_obj = model
    for dec in range(0, 7):
        crs._fir_stage = dec
        slow_rate = 625e6 / 256 / 64 / 2 ** dec
        vals = [model_obj.calculate_module_response_coupled(
            1, num_samples=1, sample_rate=slow_rate,
            start_time=k / slow_rate)[1] for k in range(50)]
        slow_mean = abs(np.mean(vals))
        pfb = model_obj.calculate_module_response_coupled(
            1, num_samples=512, sample_rate=1220703.125,
            start_time=0.0, pulse_time=0.0)[1]
        pfb_mean = abs(np.mean(pfb))
        ratio = pfb_mean / slow_mean
        assert ratio == pytest.approx(1.0, rel=0.01), \
            f"dec {dec}: pfb/slow carrier ratio {ratio:.4f} — mock " \
            f"streams no longer share one scale"
