"""A measurement that was biased arrives biased.

``find_bias_points`` leaves its report in the block, so the file carries
both the answer and the tuned catalog. Loading it must read them back:
otherwise Apply Bias on a loaded session parks tones from the untuned
catalog and publishes rows with no df calibration.
"""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog  # noqa: E402
from rfmux.tuning import AmplitudeSchedule, BiasReport, store, tuning_rows  # noqa: E402
from rfmux.tuning.sweep_results import pack_multisweep  # noqa: E402
from rfmux.tools.periscope.multisweep_panel import MultisweepPanel  # noqa: E402

MODULE_ID = "board1.2"


def _tuned_catalog(name: str):
    """The swept resonator, now carrying the IQ derivatives read off its
    sweep, so its df calibration is a number rather than None.

    Named after the sweep it came from, as ``find_bias_points``' own catalog
    is: the panel draws by name.
    """
    bias = BiasPoint(frequency_hz=1.0e9, amplitude=0.01,
                     dI_df=1e-9, dQ_df=2e-9)
    return ResonatorCatalog([Resonator(name=name, channel=1, bias=bias)],
                            module=2)


def _container():
    catalog = ResonatorCatalog.from_frequencies([1.0e9], module=2, amplitude=0.01)
    name = catalog.names()[0]
    frequencies = np.linspace(1.0e9 - 1e5, 1.0e9 + 1e5, 21)
    iq = np.ones(21, dtype=complex)
    entry = {"channel": 1, "frequencies": frequencies, "iq_counts": iq,
             "iq_volts": iq * 1e-6, "original_center_frequency": 1.0e9,
             "sweep_direction": "upward", "sweep_amplitude": 0.01}
    container = pack_multisweep(
        {0: {"upward": {name: entry}}}, module_id=MODULE_ID, module=2,
        amp_schedule=AmplitudeSchedule(), directions=["upward"], span_hz=2e5,
        npoints_per_sweep=21, nsamps=10, catalog=catalog)
    container[MODULE_ID]["bias_report"] = BiasReport(
        catalog=_tuned_catalog(name), findings=[]).to_dict()
    return container


def _panel(container):
    call_params = dict(container[MODULE_ID]["call_params"])
    call_params["catalog"] = ResonatorCatalog.from_dict(call_params["catalog"])
    call_params["amp"] = AmplitudeSchedule.from_dict(call_params["amp_schedule"])
    panel = MultisweepPanel(target_module=2, initial_params=call_params,
                            is_loaded_data=True)
    panel.show_measurement(2, container)
    return panel


def test_a_loaded_measurement_brings_its_bias_report_back(qt_app, tmp_path):
    container = store.load(store.save(_container(), "multisweep",
                                      directory=tmp_path))
    panel = _panel(container)
    try:
        assert panel.bias_report is not None
    finally:
        panel.close()


def test_the_catalog_a_loaded_measurement_applies_is_the_tuned_one(
        qt_app, tmp_path):
    container = store.load(store.save(_container(), "multisweep",
                                      directory=tmp_path))
    panel = _panel(container)
    try:
        # What Apply Bias would publish to the main window.
        rows = tuning_rows(panel.catalog)
        assert rows[1]["df_calibration"] is not None
    finally:
        panel.close()


def test_a_measurement_that_was_never_biased_has_no_report(qt_app, tmp_path):
    container = _container()
    del container[MODULE_ID]["bias_report"]
    panel = _panel(store.load(store.save(container, "multisweep",
                                         directory=tmp_path)))
    try:
        assert panel.bias_report is None
    finally:
        panel.close()
