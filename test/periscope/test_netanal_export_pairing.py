"""A multi-amplitude network-analysis export holds one sweep per probe
amplitude, each tagged with its own amplitude, so a loader pairing by
the tag (not by position) gets every curve on the power it was taken at."""
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope.network_analysis_export import NetworkAnalysisExportMixin  # noqa: E402


class _Panel(NetworkAnalysisExportMixin):
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.dac_scales = {1: -0.5}
        self.resonance_freqs = {}


def _sweep(level):
    f = np.linspace(1e9, 1.1e9, 5)
    iq = np.full(5, level, dtype=complex)
    return f, np.abs(iq), np.zeros(5), iq


def test_export_holds_one_tagged_sweep_per_amplitude():
    amps = [0.001, 0.002, 0.004]
    raw = {1: {}}
    for k, a in enumerate(amps):
        f, mag, ph, iq = _sweep(100.0 * (k + 1))
        raw[1][f"1_{a}"] = (f, mag, ph, iq, a)
    # The display keeps a copy of the latest sweep under 'default'.
    f, mag, ph, iq = _sweep(300.0)
    raw[1]["default"] = (f, mag, ph, iq)

    out = _Panel(raw).build_export_dict()
    sweeps = [v for k, v in out["modules"][1].items() if isinstance(k, int)]
    assert len(sweeps) == len(amps)
    assert [s["sweep_amplitude"] for s in sweeps] == amps
    # Each tagged sweep is the one taken at that amplitude.
    for s, k in zip(sweeps, range(len(amps))):
        assert s["magnitude"]["counts"]["raw"][0] == pytest.approx(100.0 * (k + 1))
