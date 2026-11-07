"""The main streamer PSD omits the carrier but preserves the noise bins."""

import numpy as np
import pytest

from rfmux.core.transferfunctions import spectrum_from_slow_tod
from rfmux.tools.periscope.tasks import PSDSignals, PSDTask


@pytest.mark.parametrize("mode", ["SSB", "DSB"])
def test_psd_viewer_omits_carrier(mode: str) -> None:
    rng = np.random.default_rng(42)
    i = 1000 + rng.normal(size=1024)
    q = 500 + rng.normal(size=1024)
    task = PSDTask(0, 1, i, q, mode, 0, False, False, 1, PSDSignals())
    received = []
    task.signals.done.connect(lambda row, mode, ch, payload: received.append(payload))
    task.run()
    spectrum = spectrum_from_slow_tod(
        i_data=i, q_data=q, dec_stage=0, scaling="psd",
        reference="counts", nperseg=1024, spectrum_cutoff=0.9,
        input_units="adc_counts",
    )
    payload = received[0]
    assert 0 not in payload[0]
    if mode == "SSB":
        np.testing.assert_array_equal(payload[0], spectrum["freq_iq"][1:])
        np.testing.assert_array_equal(payload[1], spectrum["psd_i"][1:])
        np.testing.assert_array_equal(payload[2], spectrum["psd_q"][1:])
        np.testing.assert_array_equal(
            payload[3], spectrum["psd_i"][1:] + spectrum["psd_q"][1:],
        )
    else:
        frequencies = spectrum["freq_dsb"]
        order = np.argsort(frequencies)
        order = order[frequencies[order] != 0]
        np.testing.assert_array_equal(payload[0], frequencies[order])
        np.testing.assert_array_equal(payload[1], spectrum["psd_dual_sideband"][order])
