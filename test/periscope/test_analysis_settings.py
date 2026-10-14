"""Analysis settings are committed only by Apply."""

import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtGui

from rfmux.tools.periscope.find_resonances_settings_panel import FindResonancesSettingsPanel
from rfmux.tools.periscope.fit_settings_panel import FitSettingsPanel
from rfmux.tools.periscope.bias_settings_panel import BiasSettingsPanel


@pytest.fixture(params=[FindResonancesSettingsPanel, FitSettingsPanel, BiasSettingsPanel])
def editor(request, qt_app):
    window = request.param()
    window.show()
    yield window
    window.close()


def edit(window) -> None:
    if isinstance(window, FindResonancesSettingsPanel):
        window.min_dip_depth_spin.setValue(0.4)
    elif isinstance(window, FitSettingsPanel):
        window._model_checks["skewed"].setChecked(False)
    else:
        window.prominence_spin.setValue(0.8)


def test_edits_do_not_change_run_parameters_or_saved_preferences(editor):
    original = editor.get_parameters()
    edit(editor)
    assert editor.get_parameters() == original
    other = type(editor)()
    try:
        assert other.get_parameters() == original
    finally:
        other.close()


def test_apply_saves_edits_and_keeps_window_open(editor):
    original = editor.get_parameters()
    edit(editor)
    editor.apply_button.click()
    assert editor.isVisible()
    assert editor.get_parameters() != original
    other = type(editor)()
    try:
        assert other.get_parameters() == editor.get_parameters()
    finally:
        other.close()


@pytest.mark.parametrize("close", ["button", "window", "escape"])
def test_close_discards_edits_when_reopened(editor, close):
    original = editor._read_parameters()
    edit(editor)
    if close == "button":
        editor.close_button.click()
    elif close == "window":
        editor.close()
    else:
        event = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress,
                               QtCore.Qt.Key.Key_Escape,
                               QtCore.Qt.KeyboardModifier.NoModifier)
        QtCore.QCoreApplication.sendEvent(editor, event)
    assert not editor.isVisible()
    editor.show()
    assert editor._read_parameters() == original


def test_close_keeps_last_applied_values(editor):
    original = editor._read_parameters()
    edit(editor)
    editor.apply_button.click()
    accepted = editor._read_parameters()
    editor.set_parameters(original)
    editor.close_button.click()
    editor.show()
    assert editor._read_parameters() == accepted


@pytest.mark.parametrize("panel", [FindResonancesSettingsPanel, BiasSettingsPanel])
def test_reset_waits_for_apply(panel, qt_app):
    window = panel()
    try:
        edit(window)
        window.apply_button.click()
        accepted = window.get_parameters()
        window._reset()
        assert window.get_parameters() == accepted
        window.close()
        assert window.get_parameters() == accepted
        window._reset()
        window.apply_button.click()
        assert window.get_parameters() != accepted
    finally:
        window.close()


def test_measurement_change_does_not_apply_pending_bias_edits(qt_app):
    window = BiasSettingsPanel()
    try:
        original = window.get_parameters()["spike_prominence_factor"]
        edit(window)
        window.set_directions_swept(["downward"])
        assert window.get_parameters()["amplitude_method"] == "derivative"
        assert window.get_parameters()["spike_prominence_factor"] == original
        window._reset()
        window.apply_button.click()
        assert window.get_parameters()["amplitude_method"] == "derivative"
    finally:
        window.close()


def test_measurement_change_does_not_apply_pending_fit_edits(qt_app):
    window = FitSettingsPanel()
    try:
        window.set_amplitude_choices([("All", None), ("Step 0", 0)])
        window.set_amplitude_choice(0)
        window.apply_button.click()
        edit(window)
        window.set_amplitude_choices([("All", None)])
        assert window.get_parameters() == {
            "models": ("skewed", "nonlinear"), "amplitude_choice": None}
        window.close()
        assert window._read_parameters() == window.get_parameters()
    finally:
        window.close()
