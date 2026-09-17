"""A panel's data is its named session result: the panel derives its view
from the value, the export carries the value under that name with the
cells that built it, and a load binds the name again."""

import pickle

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from test.qt_helpers import bare_periscope, spin  # noqa: E402


def _sweep(fr, module=1):
    f = np.linspace(fr - 1e5, fr + 1e5, 5)
    iq = np.exp(1j * np.linspace(0, 1, 5))
    return {"module": module, "frequencies": f, "iq_complex": iq, "phase_degrees": np.degrees(np.angle(iq))}


def test_multisweep_panel_derives_iterations_in_the_order_swept(qt_app):
    from rfmux.tools.periscope.multisweep_panel import MultisweepPanel
    panel = MultisweepPanel(dark_mode=False, target_module=1,
                            initial_params={"resonance_frequencies": [1e9, 2e9]})
    value = {(0.001, "upward"): {1: {"original_center_frequency": 1e9, "bias_frequency": 1e9 + 10},
                                 2: {"original_center_frequency": 2e9}},
             (0.002, "upward"): {1: {"original_center_frequency": 1e9 + 10, "bias_frequency": 1e9 + 20},
                                 2: {"original_center_frequency": 2e9, "bias_frequency": 2e9 + 5}}}
    panel.set_result(value)
    assert sorted(panel.results_by_detector) == [1, 2]
    assert [panel.results_by_detector[1][i]["amplitude"] for i in (0, 1)] == [0.001, 0.002]
    assert panel.results_by_detector[2][1]["direction"] == "upward"
    assert panel.last_output_cfs_by_amp_and_conceptual_idx == {0.001: {0: 1e9 + 10, 1: 2e9},
                                                                0.002: {0: 1e9 + 20, 1: 2e9 + 5}}
    assert "amplitude" not in value[(0.001, "upward")][1], "the value is not modified"
    panel.close()
    spin(qt_app)


def test_netanal_panel_derives_a_curve_per_module_and_amplitude(qt_app):
    from rfmux.tools.periscope.network_analysis_panel import NetworkAnalysisPanel
    panel = NetworkAnalysisPanel(None, [1, 2], {1: -0.5, 2: -0.5})
    panel.set_params({"amps": [0.001, 0.002]})
    panel.set_result({0.001: [_sweep(1e9, 1), _sweep(1.5e9, 2)], 0.002: [_sweep(1e9, 1), _sweep(1.5e9, 2)]})
    assert sorted(panel.data) == [1, 2]
    # The panel keys each sweep by module and amplitude, beside the 'default' display copy.
    assert sorted(k for k in panel.data[1] if k != "default") == ["1_0.001", "1_0.002"]
    assert len(panel.raw_data[2]["2_0.002"]) >= 4  # freqs, amps, phases, iq
    panel.close()
    spin(qt_app)


def test_exports_carry_the_named_value_and_its_cells(qt_app, monkeypatch):
    from rfmux.tools.periscope.multisweep_panel import MultisweepPanel
    p = bare_periscope(monkeypatch)
    value = {(0.001, "upward"): {1: {"original_center_frequency": 1e9}}}
    p.session_namespace = lambda: {"multisweep_0": value}
    p.session_cells = {"multisweep_0": ["multisweep_0 = {}"]}
    panel = MultisweepPanel(parent=p, dark_mode=False, target_module=1,
                            initial_params={"resonance_frequencies": [1e9]})
    panel.result_name = "multisweep_0"
    panel.set_result(value)
    export = panel._prepare_export_data()
    assert export["name"] == "multisweep_0" and export["result"] is value
    assert export["cells"] == ["multisweep_0 = {}"]
    panel.close()
    spin(qt_app)


def test_load_result_reads_the_value_and_refuses_unnamed_files(tmp_path):
    from rfmux.tools.periscope.session_manager import load_result
    named, unnamed = tmp_path / "named.pkl", tmp_path / "old.pkl"
    named.write_bytes(pickle.dumps({"name": "netanal_0", "result": {0.001: [1, 2]}, "parameters": {}}))
    unnamed.write_bytes(pickle.dumps({"parameters": {}, "modules": {}}))
    assert load_result(named) == {0.001: [1, 2]}
    with pytest.raises(ValueError):
        load_result(unnamed)


def test_free_name_avoids_names_the_session_has(monkeypatch):
    p = bare_periscope(monkeypatch)
    p.session_namespace = lambda: {"netanal_0": 1, "netanal_0_2": 2}
    assert p.free_name("netanal_0") == "netanal_0_3"
    assert p.free_name("netanal_1") == "netanal_1"
