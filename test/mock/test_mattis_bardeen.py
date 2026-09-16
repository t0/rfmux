"""The Mattis-Bardeen conductivity rests on two modified Bessel
functions; their fits hold over the whole argument range the mock
reaches (hf/2kT from 0.01 at 300 mK and 100 MHz to 3.6 at 50 mK
and 7.5 GHz), so a resonator above 5 GHz is as sound as one at 1 GHz."""
import asyncio
import contextlib
import io
import warnings

import numpy as np
import pytest
from scipy.special import i0, k0

from rfmux.mr_resonator import jit_physics as jp


@pytest.mark.portable
def test_bessel_fits_match_scipy():
    x = np.concatenate([np.logspace(-3, np.log10(1.999), 40),
                        [2.0, 2.0000001, 3.7499, 3.75, 3.7501],
                        np.linspace(2.01, 30, 40)])
    assert np.allclose([jp.bessel_k0(v) for v in x], k0(x), rtol=1e-6)
    assert np.allclose([jp.bessel_i0(v) for v in x], i0(x), rtol=1e-6)


def test_resonators_above_five_gigahertz_are_superconducting():
    from rfmux.mock.crs import ServerMockCRS
    crs = ServerMockCRS("0000")
    cfg = {"num_resonances": 5, "freq_start": 5.2e9, "freq_end": 5.4e9,
           "T": 0.12, "Popt": 1e-15, "resonator_random_seed": 1,
           "auto_bias_kids": False}
    with contextlib.redirect_stdout(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        asyncio.run(crs.generate_resonators(cfg))
    m = crs._resonator_model
    assert all(k.Lk > 0 and k.R > 0 for k in m.mr_lekids)
    assert all(q > 0 for q in m.resonator_q_values)
    f = np.asarray(m.resonator_frequencies)
    assert np.all((f > 5.1e9) & (f < 5.5e9)), f
    env = m.envelope_parameters()
    assert np.all(env["kappa"] > 0)
