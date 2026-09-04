"""A multisweep with no fit selected runs no fit; bias_kids then fits
what it needs.  Uses a MockCRS session for its RPC surface only, no UDP
stream, so this stays out of the acquisition tier."""
import asyncio
import contextlib
import io

import numpy as np
import pytest

import rfmux
from rfmux.algorithms.measurement import fitting, fitting_nonlinear
from rfmux.algorithms.measurement.bias_kids import bias_kids
from rfmux.mock.config import bias_amplitude_from_dbm

SESSION = """
!HardwareMap
- !flavour "rfmux.mock"
- !CRS { serial: "0000", hostname: "127.0.0.1" }
"""
N = 3


@pytest.fixture(scope="module")
def swept():
    loop = asyncio.new_event_loop()
    session = rfmux.load_session(SESSION)
    crs = session.query(rfmux.CRS).one()
    loop.run_until_complete(crs.resolve())
    amp = bias_amplitude_from_dbm(-55.0)
    with contextlib.redirect_stdout(io.StringIO()):
        loop.run_until_complete(crs.generate_resonators(
            {"num_resonances": N, "resonator_random_seed": 5,
             "auto_bias_kids": True, "bias_amplitude": amp}))
    nco = loop.run_until_complete(crs.get_nco_frequency(module=1))
    cfs = [nco + loop.run_until_complete(crs.get_frequency(channel=ch, module=1))
           for ch in range(1, N + 1)]
    yield loop, crs, cfs, amp
    loop.close()


def test_multisweep_without_a_fit_selected_runs_no_fit(swept, monkeypatch):
    loop, crs, cfs, amp = swept
    monkeypatch.setattr(fitting_nonlinear, "fit_nonlinear_iq",
                        lambda *a, **k: pytest.fail("nonlinear fit ran"))
    monkeypatch.setattr(fitting, "fit_skewed",
                        lambda *a, **k: pytest.fail("skewed fit ran"))
    with contextlib.redirect_stdout(io.StringIO()):
        res = loop.run_until_complete(crs.multisweep(
            center_frequencies=cfs, span_hz=200e3, npoints_per_sweep=21,
            amp=amp, nsamps=1, module=1))
    assert len(res) == N
    for entry in res.values():
        assert "nonlinear_fit_params" not in entry
        assert entry["df_calibration"] is None
        assert isinstance(entry["is_bifurcated"], (bool, np.bool_))
        assert np.isfinite(entry["bias_frequency"])


def test_bias_kids_fits_the_unfitted_sweeps(swept):
    loop, crs, cfs, amp = swept
    with contextlib.redirect_stdout(io.StringIO()):
        res = loop.run_until_complete(crs.multisweep(
            center_frequencies=cfs, span_hz=200e3, npoints_per_sweep=101,
            amp=amp, nsamps=1, module=1))
        out = loop.run_until_complete(bias_kids(crs, res, module=1))
    assert len(out) == N
    for det, entry in out.items():
        assert entry["nonlinear_fit_success"]
        assert entry["bias_frequency_source"] == "nonlinear"
        assert np.isfinite(entry["df_calibration"]) and entry["df_calibration"] != 0
        # Written back onto the sweep handed in, so a second call has
        # nothing to fit.
        assert res[det]["nonlinear_fit_success"]
