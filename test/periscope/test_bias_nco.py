"""Viewing loaded multisweep data leaves the board alone; the NCO the
sweep ran at is set only when bias_kids is started from that data."""

import asyncio
import pytest

pytest.importorskip("PyQt6")

from rfmux.algorithms.measurement.multisweep import sweep_nco_frequency  # noqa: E402
from rfmux.tools.periscope.multisweep_panel import MultisweepPanel  # noqa: E402
from rfmux.tools.periscope.tasks import BiasKidsSignals, BiasKidsTask  # noqa: E402
from rfmux.tools.periscope.utils import QtWidgets  # noqa: E402
from test.qt_helpers import bare_periscope  # noqa: E402

PARAMS = {"module": 1, "resonance_frequencies": [1.00e9, 1.10e9],
          "span_hz": 2.0e5, "amps": [0.01], "sweep_direction": "upward"}


class _Board:
    def __init__(self):
        self.calls = []

    async def get_dac_scale(self, units="DBM", module=None):
        return -0.5

    async def set_nco_frequency(self, f, module=None):
        self.calls.append(("nco", f, module))

    async def get_nco_frequency(self, module=None):
        return 1.05e9


def test_the_sweep_nco_is_the_middle_of_the_band():
    assert sweep_nco_frequency([1.00e9, 1.10e9], 2.0e5) == 1.05e9
    assert sweep_nco_frequency([1.00e9], 2.0e5) == 1.00e9
    # A placeholder for a channel with no resonance is ignored.
    nan = float("nan")
    assert sweep_nco_frequency([1.00e9, nan, 1.10e9], 2.0e5) == 1.05e9


def test_loading_a_multisweep_file_sets_no_nco(qt_app, monkeypatch):
    board = _Board()
    p = bare_periscope(monkeypatch, crs=board)
    # The fetcher reports the board's scale less 1.5 dB; the file agrees.
    load = {"initial_parameters": dict(PARAMS), "dac_scales_used": {1: -2.0},
            "results_by_detector": {}}
    panel, dock, window_id, module = p._create_multisweep_panel_from_loaded_data(load)
    try:
        assert panel is not None and module == 1
        assert board.calls == []
        assert panel._bias_nco_frequency() == 1.05e9
    finally:
        panel.close()


def test_the_task_hands_the_panel_its_rows_and_the_nco(qt_app, monkeypatch):
    """The task's run ends in the panel's slot: the module, the
    bias_kids output and the NCO read after biasing, which the panel
    reports as one tuning_ready of rows keyed by channel."""
    board = _Board()
    row = {"bias_channel": 3, "bias_frequency": 1.0e9, "df_calibration": 1 + 0j}

    async def fake_bias_kids(**kw):
        return {0: row}
    monkeypatch.setattr("rfmux.algorithms.measurement.bias_kids.bias_kids",
                        fake_bias_kids)
    said = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        lambda *a, **k: said.append(a[2]))
    panel = MultisweepPanel(dark_mode=False, target_module=1,
                            initial_params=dict(PARAMS), is_loaded_data=False)
    got = []
    panel.tuning_ready.connect(lambda m, rows: got.append((m, rows)))
    task = BiasKidsTask(board, 1, {"results_by_detector": {}},
                        BiasKidsSignals(), {})
    task.signals.completed.connect(panel._bias_kids_completed)
    try:
        task.run()
        assert got == [(1, {3: dict(row, nco_frequency_hz=1.05e9)})]
        assert panel.nco_frequency_hz == 1.05e9
        assert "biased 1 out of" in said[0]
    finally:
        panel.close()


def test_a_live_panel_sets_no_nco_before_biasing(qt_app):
    panel = MultisweepPanel(dark_mode=False, target_module=1,
                            initial_params=dict(PARAMS), is_loaded_data=False)
    try:
        assert panel._bias_nco_frequency() is None
    finally:
        panel.close()


@pytest.mark.parametrize("nco, expect", [
    (1.05e9, [("nco", 1.05e9, 1), "bias_kids"]),
    (None, ["bias_kids"]),
])
def test_bias_kids_sets_the_nco_first_when_given_one(monkeypatch, nco, expect):
    board = _Board()

    async def fake_bias_kids(**kw):
        board.calls.append("bias_kids")
        return {}

    monkeypatch.setattr("rfmux.algorithms.measurement.bias_kids.bias_kids",
                        fake_bias_kids)
    task = BiasKidsTask(board, 1, {"results_by_detector": {}},
                        BiasKidsSignals(), {}, nco_frequency_hz=nco)
    asyncio.run(task._run_bias_kids(lambda m, p: None))
    assert board.calls == expect
