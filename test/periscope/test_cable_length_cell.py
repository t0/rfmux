"""The unwrapped cable length is a plain float where it is produced, so the
value the panel stores, shows and writes into a session cell needs no
conversion downstream."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope.network_analysis_export import NetworkAnalysisExportMixin  # noqa: E402


def test_calculated_cable_length_is_a_plain_float(qt_app):
    panel = SimpleNamespace(current_params={"cable_length": 10.0})
    freqs = np.linspace(1e9, 1.1e9, 50)
    phases = np.degrees(-2 * np.pi * freqs * 5e-9)  # a 5 ns residual delay
    old, new = NetworkAnalysisExportMixin._calculate_cable_length(panel, 1, freqs, phases)
    assert type(new) is float and new != old
