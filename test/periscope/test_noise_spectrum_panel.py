"""Current noise files display one subplot per named resonator."""

from unittest.mock import AsyncMock

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from rfmux.tools.periscope.noise_spectrum_panel import NoiseSpectrumPanel
from rfmux.tools.periscope.tasks import NoiseSpectrumTask
from test.tuning.test_noise_display import noise_block
from test.qt_helpers import spin


@pytest.fixture
def panel(qt_app):
    widget = NoiseSpectrumPanel(noise_block(calibrated=True))
    yield widget
    widget.close()
    widget.deleteLater()
    spin(qt_app)


def test_named_resonators_have_timestream_and_psd_panels(panel):
    assert panel.plot_tabs.count() == 2
    assert len(panel.plots[0]) == 2
    assert "A · channel 2" in panel.plots[0][0].getPlotItem().titleLabel.text
    assert "B · channel 7" in panel.plots[0][1].getPlotItem().titleLabel.text
    panel.plot_tabs.setCurrentIndex(1)
    assert len(panel.plots[1]) == 2


def test_mean_subtraction_defaults_on_and_does_not_mutate_saved_data(panel):
    original = panel.block["results"]["resonators"]["A"]["slow_data"]["iq_counts"].copy()
    assert panel.mean_subtract.isChecked()
    curves = panel.plots[0][0].listDataItems()
    assert abs(curves[0].getData()[1].mean()) < 1e-14
    panel.mean_subtract.setChecked(False)
    assert panel.plots[0][0].listDataItems()[0].getData()[1].mean() == pytest.approx(3)
    np.testing.assert_array_equal(
        panel.block["results"]["resonators"]["A"]["slow_data"]["iq_counts"], original)


def test_psd_omits_only_zero_and_is_independent_of_mean_subtraction(panel):
    panel.plot_tabs.setCurrentIndex(1)
    curve = panel.plots[1][0].listDataItems()[0]
    x, y = curve.getOriginalDataset()
    frequencies = panel.block["results"]["shared_slow"]["freq_iq"]
    np.testing.assert_array_equal(x, frequencies[1:])
    panel.mean_subtract.setChecked(False)
    np.testing.assert_array_equal(panel.plots[1][0].listDataItems()[0].getOriginalDataset()[1], y)


def test_df_available_from_catalog_and_rotates_timestream(panel):
    panel.units_combo.setCurrentIndex(1)
    assert panel.units_combo.currentData() == "df"
    curves = panel.plots[0][0].listDataItems()
    assert [c.name() for c in curves] == ["df", "diss"]
    np.testing.assert_allclose(curves[1].getData()[1], 0, atol=1e-15)


def test_uncalibrated_files_offer_volts_only(qt_app):
    panel = NoiseSpectrumPanel(noise_block())
    assert panel.units_combo.count() == 1
    panel.close()


def test_noise_task_calls_headless_driver_and_returns_container(qt_app):
    from types import SimpleNamespace
    result = {"module1": noise_block()}
    board = SimpleNamespace(measure_noise=AsyncMock(return_value=result))
    params = dict(module=1, channels=[2, 7], nsegments=2)
    task = NoiseSpectrumTask(board, params)
    received = []
    task.completed.connect(received.append)
    task.run()
    assert received == [result]
    arguments = board.measure_noise.call_args.kwargs
    assert arguments["channels"] == [2, 7]
    assert arguments["save"] is True
    assert arguments["nsegments"] == 2
    assert callable(arguments["progress_callback"])


def test_loader_rejects_legacy_noise_payload(qt_app):
    from rfmux.tools.periscope.app import Periscope
    with pytest.raises(ValueError, match="current noise"):
        Periscope._load_noise_from_session(object(), {"noise_data": {}}, "old.pkl")


def test_pages_keep_resonator_names_and_reuse_grid_slots(panel):
    records = panel.block["results"]["resonators"]
    for index in range(21):
        records[f"extra{index}"] = dict(records["A"], channel=index + 8)
    panel.names = list(records)
    panel._redraw()
    assert panel.next_button.isEnabled()
    panel.next_button.click()
    assert "extra18" in panel.plots[0][0].getPlotItem().titleLabel.text
    assert panel.grids[0].count() == 3
    assert panel.prev_button.isEnabled()
    assert not panel.next_button.isEnabled()


def test_saved_current_container_opens_without_a_board(qt_app, tmp_path):
    from PyQt6 import QtWidgets
    from rfmux.tools.periscope.app import Periscope
    from rfmux.tools.periscope.dock_manager import PeriscopeDockManager
    from rfmux.tuning import store
    window = QtWidgets.QMainWindow()
    window.dark_mode = False
    window.dock_manager = PeriscopeDockManager(window)
    path = store.save({"module1": noise_block()}, "noise", directory=tmp_path)
    Periscope._load_noise_from_session(window, store.load(path), str(path))
    panels = window.findChildren(NoiseSpectrumPanel)
    assert len(panels) == 1
    assert panels[0].names == ["A", "B"]
    assert len(panels[0].plots[0]) == 2
    window.close()
    window.deleteLater()
    spin(qt_app)
