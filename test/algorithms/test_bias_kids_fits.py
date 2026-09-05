"""bias_kids works from the resonance fit it is told to: it fits the
sweeps that lack it, moves the bias frequency onto the fitted curve,
chooses the amplitude by the fitted nonlinearity, and calibrates there."""
import contextlib

import numpy as np
import pytest

from rfmux.algorithms.measurement import bias_kids as bk
from rfmux.algorithms.measurement.fitting_nonlinear import nonlinear_iq
from rfmux.core.transferfunctions import VOLTS_PER_ROC

FR, QR, NCO = 1.0e9, 5.0e4, 0.9e9


class _Ctx:
    def __init__(self, board):
        self.board = board
        self.log = board.log

    def set_frequency(self, f, channel, module):
        self.log.append(("frequency", channel, f))
        self.board.freq[channel] = f

    def set_amplitude(self, a, channel, module):
        self.log.append(("amplitude", channel, a))

    def set_phase(self, p, units, target, channel, module):
        self.log.append(("phase", channel, p))

    async def __call__(self):
        pass


class _Board:
    class UNITS:
        DEGREES = "deg"

    class TARGET:
        ADC = "adc"

    def __init__(self):
        self.log = []
        self.freq = {}

    async def get_nco_frequency(self, module):
        return NCO

    @contextlib.asynccontextmanager
    async def tuber_context(self):
        yield _Ctx(self)

    async def set_phase(self, *a, **k):
        pass

    samples = None   # unmoving by default: no stepped calibration, the fit's is used

    async def get_samples(self, n, channel=None, module=None, average=False):
        return self.samples if self.samples is not None else _Samples(np.zeros(n), np.zeros(n))


class _Samples:
    def __init__(self, i, q):
        self.i, self.q = [np.asarray(i)], [np.asarray(q)]


SLOPE = 0.4 - 0.3j   # counts per hertz, the board's response to a tone step


class _SlopeBoard(_Board):
    """Samples that move with the channel's frequency: a straight
    trajectory of slope SLOPE, so a stepped-tone calibration is exact."""
    async def get_samples(self, n, channel=None, module=None, average=False):
        z = SLOPE * self.freq.get(1, 0.0)
        return _Samples(np.full(n, z.real), np.full(n, z.imag))


@pytest.mark.asyncio
async def test_calibration_is_measured_where_the_detector_sits():
    board = _SlopeBoard()
    entry = _entry()
    out = await bk.bias_kids(board, {1: entry}, module=1)
    expect = 1.0 / (SLOPE * VOLTS_PER_ROC)
    assert out[1]["df_calibration_source"] == "measured"
    assert out[1]["df_calibration"] == pytest.approx(expect, rel=1e-6)
    assert "df_calibration_fit" in out[1]
    # The tone is back where it was biased.
    bias_rel = [f for kind, ch, f in board.log if kind == "frequency"][0]
    assert board.freq[1] == bias_rel


@pytest.mark.asyncio
async def test_unmoving_samples_fall_back_to_the_fit():
    board = _Board()
    board.samples = _Samples(np.full(100, 3.0), np.full(100, -2.0))
    out = await bk.bias_kids(board, {1: _entry()}, module=1)
    assert out[1]["df_calibration_source"] == "fit"
    assert np.isfinite(out[1]["df_calibration"])


@pytest.mark.asyncio
async def test_phase_puts_the_principal_axis_along_q():
    t = np.linspace(-1, 1, 200)
    board = _Board()
    # Samples along 30 degrees: the board rotates by +phase, so 60
    # degrees more puts that axis on Q.
    board.samples = _Samples(t * np.cos(np.radians(30)), t * np.sin(np.radians(30)))
    phases = await bk.find_optimal_phases_parallel(
        board, {1: {"channel": 1}}, module=1, apply_bandpass=False)
    assert phases[1][0] == pytest.approx(60.0, abs=1e-6)


@pytest.mark.asyncio
async def test_no_variation_means_no_phase():
    board = _Board()
    board.samples = _Samples(np.full(200, 3.0), np.full(200, -2.0))
    with pytest.warns(UserWarning, match="no variation"):
        phases = await bk.find_optimal_phases_parallel(
            board, {1: {"channel": 1}}, module=1, apply_bandpass=False)
    assert phases[1] == (0.0, 0.0)


def _entry(a=0.0, amplitude=0.01, n=101, span=200e3):
    """A multisweep result entry, its bias frequency on the raw grid
    two steps above the resonance, as a noisy max-diq might put it."""
    f = np.linspace(FR - span / 2, FR + span / 2, n)
    z = nonlinear_iq(f, FR, QR, 0.6, 0.1, a, 1.0, 0.2) / VOLTS_PER_ROC
    return {"frequencies": f, "iq_complex": z, "original_center_frequency": FR,
            "bias_frequency": float(f[n // 2 + 2]), "recalculation_method_applied": "max-diq",
            "sweep_amplitude": amplitude, "amplitude": amplitude, "direction": "upward",
            "is_bifurcated": False}


@pytest.mark.asyncio
async def test_fits_moves_the_bias_point_and_calibrates():
    board = _Board()
    entry = _entry()
    out = await bk.bias_kids(board, {1: entry}, module=1)
    assert entry["nonlinear_fit_success"]
    assert entry["bias_frequency_source"] == "nonlinear"
    # a = 0: the IQ trajectory is fastest on resonance, well inside the
    # 2 kHz grid step the raw choice was off by.
    assert entry["bias_frequency"] == pytest.approx(FR, abs=100.0)
    assert out[1]["bias_frequency"] == entry["bias_frequency"]
    assert np.isfinite(out[1]["df_calibration"])
    freqs = [f for kind, ch, f in board.log if kind == "frequency"]
    assert freqs and abs(freqs[0] + NCO - FR) < 300.0


@pytest.mark.asyncio
async def test_skewed_choice_uses_the_skewed_fit_and_runs_no_nonlinear_fit(monkeypatch):
    monkeypatch.setattr("rfmux.algorithms.measurement.fitting_nonlinear."
                        "fit_nonlinear_iq_multisweep",
                        lambda *a, **k: pytest.fail("nonlinear fit ran"))
    entry = _entry()
    entry["fit_params"] = {"fr": FR, "Qr": QR, "Qcre": QR / 0.6, "Qcim": 0.0}
    out = await bk.bias_kids(_Board(), {1: entry}, module=1, fit_method="skewed")
    assert entry["bias_frequency_source"] == "skewed"
    assert entry["bias_frequency"] == pytest.approx(FR, abs=100.0)
    assert np.isfinite(out[1]["df_calibration"])


@pytest.mark.asyncio
async def test_a_failed_fit_counts_as_no_fit_not_a_crash():
    # The batch fitter leaves nonlinear_fit_params = None behind when a
    # fit fails; the amplitude choice then falls back to the jump
    # detector for that entry.
    loud = _entry(a=0.2, amplitude=0.03)
    loud["nonlinear_fit_params"] = None
    loud["nonlinear_fit_success"] = False
    loud["is_bifurcated"] = True
    results = {"results_by_detector": {1: {0: _entry(a=0.2, amplitude=0.01), 1: loud}}}
    out = await bk.bias_kids(_Board(), results, module=1)
    assert out[1]["selected_amplitude"] == 0.01


@pytest.mark.asyncio
async def test_amplitude_choice_comes_from_the_fitted_nonlinearity():
    # Two amplitudes, neither sweep jumping: only the fitted nonlinearity
    # can rule the louder one out, and it has to be fitted first.
    results = {"results_by_detector": {1: {0: _entry(a=0.2, amplitude=0.01),
                                           1: _entry(a=0.9, amplitude=0.03)}}}
    out = await bk.bias_kids(_Board(), results, module=1)
    assert out[1]["selected_amplitude"] == 0.01
    assert out[1]["nonlinear_fit_params"]["a"] == pytest.approx(0.2, abs=0.05)
