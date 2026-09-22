"""PFB samples use their saved units and time axis."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from rfmux.core.transferfunctions import VOLTS_PER_ROC
from rfmux.tools.periscope.noise_spectrum_panel import NoiseSpectrumPanel
from test.tuning.test_noise_display import noise_block


def test_pfb_timestream_volts_and_elapsed_time(qt_app):
    block = noise_block()
    time = np.arange(32) / 2.44e6
    block["results"]["shared_pfb"] = dict(time_s=time)
    for record in block["results"]["resonators"].values():
        record["pfb_data"] = dict(
            iq_counts=np.full(32, (7 + 3j) / VOLTS_PER_ROC))
    panel = NoiseSpectrumPanel(block)
    panel.mean_subtract.setChecked(False)
    panel.stream_combo.setCurrentIndex(1)
    curves = panel.plots[0][0].listDataItems()
    np.testing.assert_array_equal(curves[0].getData()[0], time)
    np.testing.assert_allclose(curves[0].getData()[1], 7)
    np.testing.assert_allclose(curves[1].getData()[1], 3)
    panel.close()
