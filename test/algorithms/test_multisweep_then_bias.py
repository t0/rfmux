"""A multisweep with no fit selected runs no fit; bias_kids then fits
what it needs.  A MockCRS session for its RPC surface, no UDP stream,
but a server process all the same: the acquisition tier."""
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
pytestmark = pytest.mark.slow_acquisition


@pytest.fixture(scope="module")
def swept():
    loop = asyncio.new_event_loop()
    session = rfmux.load_session(SESSION)
    crs = session.query(rfmux.CRS).one()
    loop.run_until_complete(crs.resolve())
    amp = bias_amplitude_from_dbm(-55.0)
    with contextlib.redirect_stdout(io.StringIO()):
        # No frequency noise in the physics: the tone-step check below
        # compares two reads a second or more apart, and the simulator's
        # 1/f noise is about 100 Hz rms, a third of the step.
        loop.run_until_complete(crs.generate_resonators(
            {"num_resonances": N, "resonator_random_seed": 5,
             "auto_bias_kids": True, "bias_amplitude": amp,
             "tls_noise_enabled": False, "nqp_noise_enabled": False}))
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
        assert "df_calibration" not in entry
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


def _shift_seen(loop, crs, out, step_hz):
    """What each biased detector's calibration reports when its tone
    moves *step_hz* up: the shift the samples show, in hertz."""
    from rfmux.core.transferfunctions import convert_roc_to_volts
    nco = loop.run_until_complete(crs.get_nco_frequency(module=1))
    seen = {}
    for det, entry in out.items():
        ch = entry["bias_channel"]
        f0 = loop.run_until_complete(crs.get_frequency(channel=ch, module=1))
        z = []
        for f in (f0, f0 + step_hz):
            loop.run_until_complete(crs.set_frequency(f, channel=ch, module=1))
            s = loop.run_until_complete(crs.get_samples(50, module=1))
            z.append(complex(np.mean(s.i[ch - 1]), np.mean(s.q[ch - 1])))
        loop.run_until_complete(crs.set_frequency(f0, channel=ch, module=1))
        seen[det] = convert_roc_to_volts(z[1] - z[0]) * entry["df_calibration"]
    return seen


@pytest.mark.parametrize("adc_phase", [None, 37.0, 236.0])
def test_calibration_reports_a_tone_step_whatever_the_adc_phase(swept, monkeypatch, adc_phase):
    # The simulator's RPC samples carry no noise, so the phase choice
    # itself has nothing to find here; what this pins is the bookkeeping:
    # whatever ADC phase bias_kids applies, the calibration it reports
    # still turns the samples into the right frequency shift.
    loop, crs, cfs, amp = swept
    if adc_phase is not None:
        async def fixed(crs, bias_configs, module, **kwargs):
            return {det: (adc_phase, 0.0) for det in bias_configs}
        monkeypatch.setattr("rfmux.algorithms.measurement.bias_kids."
                            "find_optimal_phases_parallel", fixed)
    with contextlib.redirect_stdout(io.StringIO()):
        res = loop.run_until_complete(crs.multisweep(
            center_frequencies=cfs, span_hz=200e3, npoints_per_sweep=101,
            amp=amp, nsamps=1, module=1))
        out = loop.run_until_complete(bias_kids(
            crs, res, module=1, optimize_phase=adc_phase is not None))
    step = 300.0
    for det, shift in _shift_seen(loop, crs, out, step).items():
        # The calibration is the inverse of the sweep's slope, so a tone
        # moved up by `step` reads as +step along df.
        assert abs(shift.real - step) < 0.25 * step, (det, shift)
        assert abs(shift.imag) < 0.5 * step, (det, shift)
        ch = out[det]["bias_channel"]
        assert loop.run_until_complete(crs.get_phase(channel=ch, module=1)) == (adc_phase or 0.0)
        assert out[det]["df_calibration_source"] == "measured"
        # The fit's version agrees to a few degrees; the measured one is
        # what a tone step reads.
        ratio = out[det]["df_calibration"] / out[det]["df_calibration_fit"]
        assert abs(np.degrees(np.angle(ratio))) < 6.0
