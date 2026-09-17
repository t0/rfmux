"""The main window holds a module's tuning rows and converts its plots
with the df calibration of every row that has one."""

import concurrent.futures

import pytest

pytest.importorskip("PyQt6")

from test.qt_helpers import bare_periscope, spin  # noqa: E402


def test_the_rows_are_held_and_the_calibrations_derived(
        qt_app, capsys, monkeypatch):
    p = bare_periscope(monkeypatch)
    rows = {3: {"bias_channel": 3, "df_calibration": 1 + 1j,
                "bias_frequency": 1e9},
            7: {"bias_channel": 7, "df_calibration": None}}
    p._handle_tuning_ready(2, rows)
    assert p.tuning == {2: rows}
    assert p.df_calibrations == {2: {3: 1 + 1j}}
    assert "2 detectors on module 2, 1 with a df calibration" in \
        capsys.readouterr().out


def test_a_live_multisweep_window_reports_its_tuning_to_the_main_window(
        qt_app, monkeypatch):
    from rfmux.tools.periscope.tasks import MultisweepSignals

    cells = []
    p = bare_periscope(monkeypatch, crs=object())
    p.multisweep_signals, p.multisweep_tasks = MultisweepSignals(), {}
    p.run_python = lambda code, comment="": cells.append(code) or concurrent.futures.Future()
    p.session_namespace = lambda: {}
    p._start_multisweep_analysis({
        "module": 2, "resonance_frequencies": [1.0e9], "span_hz": 2.0e5,
        "amps": [0.01], "sweep_direction": "upward"})
    panel = p.multisweep_windows["multisweep_0"]["window"]
    try:
        assert len(cells) == 1 and "await crs.multisweep(center_frequencies=[1000000000.0], " \
            "span_hz=200000.0, amp=0.01, sweep_direction='upward', fit_skewed=False, " \
            "fit_nonlinear=False, module=2, " in cells[0]
        rows = {3: {"bias_channel": 3, "df_calibration": 2 + 0j}}
        panel.tuning_ready.emit(2, rows)
        assert p.tuning[2] == rows
        assert p.df_calibrations[2] == {3: 2 + 0j}
    finally:
        panel.close()
        spin(qt_app)
