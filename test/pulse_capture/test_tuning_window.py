"""A capture's tuning is browsed as a multisweep window: the pulse list
carries a Tuning item per module, live or in review, and the main
window opens the file's sweeps as the catalog they came from."""

from types import SimpleNamespace
import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("h5py")

from rfmux.core.resonators import BiasPoint, Resonator, ResonatorCatalog  # noqa: E402
from rfmux.tools.periscope.pulse_capture_panel import (  # noqa: E402
    PulseCapturePanel)
from rfmux.tools.periscope.utils import QtWidgets  # noqa: E402
from rfmux.tuning import tuning_rows  # noqa: E402
from test.pulse_capture.capture_files import capture_file  # noqa: E402
from test.qt_helpers import bare_periscope, spin  # noqa: E402


def _row(channel, f0, amp=0.01, direction="upward"):
    """One row, through the producer a capture's rows come from."""
    f = np.linspace(f0 - 1e5, f0 + 1e5, 9)
    bias = BiasPoint(frequency_hz=f0, amplitude=amp, dI_df=5e-7, dQ_df=0.0,
                     bias_sweep={"frequencies": f,
                                 "iq_volts": np.exp(1j * np.linspace(0, 1, 9)),
                                 "original_center_frequency": f0,
                                 "sweep_amplitude": amp,
                                 "sweep_direction": direction})
    catalog = ResonatorCatalog(
        [Resonator(name=f"R{channel:04d}", channel=channel, bias=bias)],
        module=2)
    return tuning_rows(catalog, nco_frequency_hz=1.0e9,
                       dac_scale_dbm=-2.0, nsamps=10)[channel]


def _file(tmp_path, channels, module, tuning):
    return capture_file(tmp_path / "tuned.h5", channels, module,
                        tuning=tuning)


def _tuning_items(panel):
    return [(panel.pulse_tree.topLevelItem(i).text(0),
             panel.pulse_tree.topLevelItem(i).data(0, 0x0100))
            for i in range(panel.pulse_tree.topLevelItemCount())
            if panel.pulse_tree.topLevelItem(i).text(0).startswith("▦ Tuning")]


@pytest.fixture
def panel(qt_app):
    p = PulseCapturePanel(dark_mode=False)
    yield p
    p.close()
    spin(qt_app)


def test_a_live_capture_lists_its_tuning(qt_app):
    # Channel 2's row has no sweep (a measured calibration only): not counted.
    panel = PulseCapturePanel(dark_mode=False, tuning={
        1: {1: _row(1, 1.0e9), 2: {"df_calibration": 1.0}, 3: _row(3, 1.1e9)}})
    try:
        panel.module_spin.setValue(1)
        panel._reset_results([1, 2, 3])
        assert _tuning_items(panel) == [("▦ Tuning (2 detectors)", ("tuning", 1))]
        tops = [panel.pulse_tree.topLevelItem(i).text(0)
                for i in range(panel.pulse_tree.topLevelItemCount())]
        assert tops[-2:] == ["▦ Tuning (2 detectors)", "▦ Metadata"]
    finally:
        panel.close()
        spin(qt_app)


def test_a_capture_without_tuning_lists_none(panel):
    panel._reset_results([1, 2])
    assert _tuning_items(panel) == []


def test_a_reviewed_file_lists_tuning_per_module(qt_app, tmp_path, panel):
    keys = [(2, 5), (3, 1), (3, 4)]
    tuning = {(2, 5): _row(5, 1.0e9), (3, 1): _row(1, 1.2e9),
              (3, 4): _row(4, 1.3e9)}
    panel.load_from_hdf5(_file(tmp_path, keys, None, tuning))
    assert _tuning_items(panel) == [
        ("▦ Tuning module 2 (1 detector)", ("tuning", 2)),
        ("▦ Tuning module 3 (2 detectors)", ("tuning", 3))]
    assert set(panel._tuning_by_module()[3]) == {1, 4}


def test_double_clicking_the_item_asks_the_main_window(qt_app, tmp_path, panel):
    path = _file(tmp_path, [4, 6], 2, {4: _row(4, 1.0e9), 6: _row(6, 1.1e9)})
    panel.load_from_hdf5(path)
    opened = []
    panel.periscope = SimpleNamespace(
        open_tuning_window=lambda rows, module, name: opened.append(
            (sorted(rows), module, name)))
    [(_, data)] = _tuning_items(panel)
    item = next(panel.pulse_tree.topLevelItem(i)
                for i in range(panel.pulse_tree.topLevelItemCount())
                if panel.pulse_tree.topLevelItem(i).data(0, 0x0100) == data)
    panel._on_tree_double_click(item, 0)
    assert opened == [([4, 6], 2, "tuned.h5")]


def test_without_a_main_window_the_item_says_so(qt_app, tmp_path, panel):
    panel.load_from_hdf5(_file(tmp_path, [4], 2, {4: _row(4, 1.0e9)}))
    panel._open_tuning_window(2)
    assert "main window" in panel.status_label.text()


def test_the_main_window_opens_the_sweeps_read_only(qt_app, monkeypatch):
    p = bare_periscope(monkeypatch)
    rows = {3: _row(3, 1.0e9), 7: _row(7, 1.1e9, amp=0.02)}
    panel = p.open_tuning_window(rows, 2, "tuned.h5")
    try:
        assert panel is not None and panel.is_loaded_data
        # The rows came back as the catalog they were built from.
        assert sorted(r.channel for r in panel.catalog) == [3, 7]
        assert panel.catalog.module == 2 and panel.target_module == 2
        biased = {r.channel: r.bias for r in panel.catalog}
        assert biased[7].amplitude == 0.02
        assert biased[3].frequency_hz == pytest.approx(1.0e9, abs=1e3)
        # And as the one sweep each of them is biased at.
        [(_, by_direction)] = panel.module_sweeps["results"].items()
        assert sorted(by_direction["upward"]) == ["R0003", "R0007"]
        # Read-only: nothing to re-run, because there is no schedule here.
        assert not panel.rerun_btn.isEnabled()
        assert panel.is_capture_tuning
        title = p.dock_manager.create_dock.call_args.args[1]
        assert title == "Tuning of tuned.h5 (module 2)"
    finally:
        panel.close()
        spin(qt_app)


def test_a_sweep_taken_downward_is_shown_as_downward(qt_app, monkeypatch):
    p = bare_periscope(monkeypatch)
    rows = {3: _row(3, 1.0e9), 7: _row(7, 1.1e9, direction="downward")}
    panel = p.open_tuning_window(rows, 2, "tuned.h5")
    try:
        [(_, by_direction)] = panel.module_sweeps["results"].items()
        assert list(by_direction["upward"]) == ["R0003"]
        assert list(by_direction["downward"]) == ["R0007"]
    finally:
        panel.close()
        spin(qt_app)


def test_rows_without_a_sweep_open_nothing(qt_app, monkeypatch):
    p = bare_periscope(monkeypatch)
    bare = {k: v for k, v in _row(1, 1.0e9).items()
            if k not in ("frequencies", "iq_volts")}
    assert p.open_tuning_window({1: bare}, 1, "x.h5") is None
    assert "no tuning row carries a sweep" in p.statusBar().currentMessage()
