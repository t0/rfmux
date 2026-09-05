"""The Noise Spectrum panel's fast timestream is converted to volts once:
py_get_pfb_samples hands back volts under the absolute reference, so the
panel converts only under the relative one, as it does for the slow
timestream."""
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from test.qt_helpers import spin  # noqa: E402

from rfmux.core.transferfunctions import VOLTS_PER_ROC  # noqa: E402
from rfmux.tools.periscope.noise_spectrum_panel import NoiseSpectrumPanel  # noqa: E402


class _Stamp:
    """The fields the panel reads off a slow-stream packet timestamp."""
    def __init__(self, seconds):
        self.h, rem = divmod(int(seconds), 3600)
        self.m, self.s = divmod(rem, 60)
        self.ss = int((seconds - int(seconds)) * 156250000)


def _spectrum_data(reference):
    n, m = 64, 32
    freqs = np.linspace(1.0, 300.0, n)
    return {
        "reference": reference,
        "slow_freq_hz": 596.0, "freq_iq": freqs, "overlap": 2,
        "single_psd_i": [np.ones(n)], "single_psd_q": [np.ones(n)],
        "I": [np.full(m, 1000.0)], "Q": [np.full(m, -500.0)],
        "amplitudes_dbm": [-55.0],
        "pfb_enabled": True,
        "pfb_psd_i": [np.ones(n)], "pfb_psd_q": [np.ones(n)],
        "pfb_I": [np.full(m, 7.0)], "pfb_Q": [np.full(m, 3.0)],
        "pfb_freq_iq": [freqs], "pfb_freq_dsb": [freqs], "pfb_dual_psd": [np.ones(n)],
        "pfb_ts": np.arange(m) / 2.44e6,
        "ts": [_Stamp(43200.0 + k / 596.0) for k in range(m)],
    }


def _fast_curves(panel):
    return {c.name(): c.getData()[1] for c in panel.plot_fast_tod.listDataItems()
            if c.name() in ("PFB I", "PFB Q")}


@pytest.mark.parametrize("reference, factor", [("absolute", 1.0), ("relative", VOLTS_PER_ROC)])
def test_fast_tod_is_converted_once(qt_app, reference, factor):
    panel = NoiseSpectrumPanel(
        detector_id=1, resonance_frequency_ghz=4.0,
        all_detectors_data={1: {"conceptual_freq_hz": 4.0e9}},
        initial_detector_idx=1, spectrum_data=_spectrum_data(reference))
    panel.mean_subtract_enabled = False
    panel._update_noise_plots()
    curves = _fast_curves(panel)
    assert curves, "the fast TOD is drawn"
    assert np.allclose(curves["PFB I"], 7.0 * factor)
    assert np.allclose(curves["PFB Q"], 3.0 * factor)
    panel.close()
    spin(qt_app)
