"""Reviewing a capture across modules: the module shows with each
channel, and only there; a one-module file reads as it always has."""

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("h5py")

from rfmux.pulse_capture.analysis import plot_groups  # noqa: E402
from rfmux.tools.periscope.pulse_capture_panel import (  # noqa: E402
    PulseCapturePanel)
from test.pulse_capture.capture_files import capture_file  # noqa: E402
from test.qt_helpers import spin  # noqa: E402

KEYS = [(2, 5), (3, 1)]


def _file(tmp_path, channels, module):
    """Two pulses on the first channel, one on the second."""
    return capture_file(
        tmp_path / "review.h5", channels, module, seconds=0.5, seed=3,
        events={key: (0.1, 0.25, 0.4)[:2 - k]
                for k, key in enumerate(channels)})


@pytest.fixture
def panel(qt_app):
    panel = PulseCapturePanel(dark_mode=False)
    yield panel
    panel.close()
    spin(qt_app)


def test_a_capture_across_modules_reviews_with_the_module_named(
        qt_app, tmp_path, panel):
    panel.load_from_hdf5(_file(tmp_path, KEYS, None))
    assert panel.channels_edit.text() == "2:5,3:1"
    assert panel.module_spin.value() == 2
    tops = [panel.pulse_tree.topLevelItem(i).text(0)
            for i in range(panel.pulse_tree.topLevelItemCount())]
    assert tops[0] == "▤ Module 2 channel 5 (2)"
    assert tops[1] == "▤ Module 3 channel 1 (1)"
    panel._show_pulse((2, 5), 1)
    assert "Pulse #000001 — Module 2 channel 5" in panel.pulse_info.text()
    assert "M2Ch5" in panel.noise_label.text()
    assert panel._label_channel() == (2, 5)


def test_a_one_module_file_reads_as_before(qt_app, tmp_path, panel):
    panel.load_from_hdf5(_file(tmp_path, [1, 5], 2))
    assert panel.channels_edit.text() == "1,5"
    assert panel.module_spin.value() == 2
    tops = [panel.pulse_tree.topLevelItem(i).text(0)
            for i in range(panel.pulse_tree.topLevelItemCount())]
    assert tops[:2] == ["▤ Channel 1 (2)", "▤ Channel 5 (1)"]
    panel._show_pulse(1, 1)
    assert "Pulse #000001 — Channel 1" in panel.pulse_info.text()


def test_the_plot_spec_may_name_a_module():
    keys = [(2, 1), (2, 5), (3, 1)]
    assert plot_groups("", keys) == [("M2Ch1", [(2, 1)]), ("M2Ch5", [(2, 5)]),
                                     ("M3Ch1", [(3, 1)])]
    assert plot_groups("1", keys) == [("Ch1", [(2, 1), (3, 1)])]
    assert plot_groups("2:1-5", keys) == [("M2Ch1-5", [(2, 1), (2, 5)])]
    assert plot_groups("3:5", keys) == []
    assert plot_groups("*", keys) == [("All 3 ch", keys)]
    with pytest.raises(ValueError, match="Could not read"):
        plot_groups("x:1", keys)
    assert plot_groups("1-2", [1, 2, 3]) == [("Ch1-2", [1, 2])]
