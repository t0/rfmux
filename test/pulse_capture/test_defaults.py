"""The defaults: trigger on the frequency basis where a calibration
allows it, stop the window a margin past below-threshold, and view in
hertz when it can be drawn."""
import numpy as np
import pytest

from rfmux.pulse_capture.capture_session import (
    PulseCaptureConfig, PulseCaptureSession)


def test_config_and_session_defaults():
    for obj in (PulseCaptureConfig(),
                PulseCaptureSession(channels=[1], sample_rate=1000.0)):
        assert obj.trigger_basis == "df"
        assert obj.save_to_end_confirmed is False


@pytest.mark.parametrize("cal, units", [(None, "V"), (2.0e6 + 0.5e6j, "Hz")])
def test_df_default_rotates_only_where_a_calibration_exists(cal, units):
    s = PulseCaptureSession(
        channels=[1], sample_rate=1000.0,
        df_calibrations=({1: cal} if cal is not None else None))
    s.start()
    n = 20
    s.feed_block(1, np.zeros(n), np.zeros(n), np.arange(n) / 1000.0)
    assert s.stored_units[1] == units
    s.stop()


def test_dialog_shows_the_defaults(qt_app):
    from rfmux.tools.periscope.pulse_capture_settings_dialog import (
        PulseCaptureSettingsDialog)
    dlg = PulseCaptureSettingsDialog(config=PulseCaptureConfig(),
                                     sample_rate=1000.0, mode="slow")
    assert dlg.basis_combo.currentIndex() == 1          # df
    assert not dlg.end_confirmed_check.isChecked()
    out = dlg.get_config()
    assert out.trigger_basis == "df" and out.save_to_end_confirmed is False
    dlg.close()


def test_view_defaults_to_hertz_when_calibrated(qt_app):
    from rfmux.tools.periscope.pulse_capture_panel import (
        PulseCapturePanel, UNITS_DF, UNITS_VOLTS)
    with_cal = PulseCapturePanel(dark_mode=False,
                                 df_calibrations={1: {1: 2.0e6 + 0j}})
    without = PulseCapturePanel(dark_mode=False)
    try:
        assert with_cal.units_combo.currentText() == UNITS_DF
        assert without.units_combo.currentText() == UNITS_VOLTS
    finally:
        with_cal.close(); without.close()


def test_a_chosen_view_is_not_overridden(qt_app):
    from rfmux.tools.periscope.pulse_capture_panel import (
        PulseCapturePanel, UNITS_DF, UNITS_VOLTS)
    panel = PulseCapturePanel(dark_mode=False,
                              df_calibrations={1: {1: 2.0e6 + 0j}})
    try:
        assert panel.units_combo.currentText() == UNITS_DF
        panel.units_combo.setCurrentText(UNITS_VOLTS)       # the user's pick
        panel._apply_default_view()                          # start / open
        assert panel.units_combo.currentText() == UNITS_VOLTS
    finally:
        panel.close()


def test_a_late_calibration_switches_an_untouched_view(qt_app):
    from rfmux.tools.periscope.pulse_capture_panel import (
        PulseCapturePanel, UNITS_DF, UNITS_VOLTS)
    panel = PulseCapturePanel(dark_mode=False)
    try:
        assert panel.units_combo.currentText() == UNITS_VOLTS
        panel.df_calibrations = {1: {1: 2.0e6 + 0j}}         # measured later
        panel._apply_default_view()
        assert panel.units_combo.currentText() == UNITS_DF
    finally:
        panel.close()


def test_settings_dialog_learns_whether_df_is_available(qt_app, monkeypatch):
    """The panel tells the dialog whether the named channels have a
    calibration, so the rotated basis is only offered when it can act."""
    from rfmux.tools.periscope import pulse_capture_panel as m
    seen = {}

    class _Dialog:
        def __init__(self, parent, **kw):
            seen.update(kw)

        def exec(self):
            return 0
    monkeypatch.setattr(m, "PulseCaptureSettingsDialog", _Dialog)

    panel = m.PulseCapturePanel(dark_mode=False)
    try:
        panel.channels_edit.setText("1,2")
        panel._on_capture_settings()
        assert seen["df_available"] is False
        panel.df_calibrations = {1: {2: 2.0e6 + 0j}}
        panel._on_capture_settings()
        assert seen["df_available"] is True
    finally:
        panel.close()


def test_uncalibrated_channel_is_shown_as_volts_on_the_quadratures(qt_app):
    """With trigger_basis="df" and no calibration the engine stores the
    quadratures in volts, and the axes must say so."""
    from rfmux.tools.periscope.pulse_capture_panel import PulseCapturePanel
    panel = PulseCapturePanel(dark_mode=False)
    try:
        assert panel.capture_config.trigger_basis == "df"
        assert panel._stored_state(1) == ("iq", "V")
        assert panel._axis_names(1) == ("I (V)", "Q (V)")
        panel.df_calibrations = {1: {1: 2.0e6 + 0j}}
        assert panel._stored_state(1) == ("df", "Hz")
    finally:
        panel.close()


def test_the_tap_is_released_however_the_worker_ends(qt_app):
    from types import SimpleNamespace
    from rfmux.tools.periscope.pulse_capture_panel import PulseCapturePanel
    released = []
    runtime = SimpleNamespace(unregister_pulse_tap=lambda: released.append(True))
    panel = PulseCapturePanel(periscope=runtime, dark_mode=False)
    try:
        panel._tap_registered = True
        panel.task = None
        panel._on_task_finished()
        assert released == [True]
        assert panel._tap_registered is False
    finally:
        panel.close()


def test_a_new_capture_starts_from_the_newest_packets():
    """Registering the tap discards the receiver's backlog, so the slow
    stream's clock starts current instead of behind."""
    from types import SimpleNamespace
    from rfmux.tools.periscope.app_runtime import PeriscopeRuntime

    class _Q:
        def __init__(self, n):
            self.n = n

        def clear(self):
            self.n = 0

    rt = PeriscopeRuntime.__new__(PeriscopeRuntime)
    rt.receiver = SimpleNamespace(queue=_Q(37))
    rt.register_pulse_tap(lambda *a: None)
    assert rt.receiver.queue.n == 0
    PeriscopeRuntime.__new__(PeriscopeRuntime)._discard_packets()  # no receiver yet


def test_a_new_capture_does_not_show_the_last_runs_counts(qt_app):
    from rfmux.tools.periscope.pulse_capture_panel import PulseCapturePanel
    panel = PulseCapturePanel(dark_mode=False)
    try:
        panel._last_stats = {"total_pulses": 57, "rate_per_min": 12.0,
                             "elapsed_s": 300, "per_channel": {1: 57}}
        panel._reset_results([1])
        panel._refresh_status_line()
        assert "57" not in panel.status_label.text()
    finally:
        panel.close()


def test_pulse_tree_shows_the_clock_and_stays_resizable(qt_app):
    """Columns size to their contents as rows arrive, until the user
    drags a divider; the decoded packet clock has its own column."""
    from PyQt6 import QtWidgets
    from rfmux.tools.periscope.pulse_capture_panel import PulseCapturePanel
    panel = PulseCapturePanel(dark_mode=False)
    tree = panel.pulse_tree
    header = tree.header()
    assert tree.columnCount() == 4
    assert header.sectionResizeMode(0) == \
        QtWidgets.QHeaderView.ResizeMode.Interactive
    panel._reset_results([1], "now")
    panel._add_pulse_row(1, 1, {"n_samples": 120, "snr": 7.0,
                                "trigger_utc": "2026-09-02T16:14:05.123456Z"})
    row = panel._channel_items[1].child(0)
    assert row.text(1) == "16:14:05.123456"
    assert not panel._tree_user_sized
    header.resizeSection(1, 333)          # the user drags a divider
    assert panel._tree_user_sized
    panel._add_pulse_row(1, 2, {"n_samples": 5, "snr": 3.0})
    assert header.sectionSize(1) == 333   # and keeps the width


def test_the_panel_fits_a_laptop_screen(qt_app):
    """The toolbar wraps into rows when the panel is narrow, the file
    label shows the name with the path on hover, and none of it asks
    for more width than a 1080p display has."""
    from rfmux.tools.periscope.pulse_capture_panel import PulseCapturePanel
    panel = PulseCapturePanel(dark_mode=False)
    panel._show_path("/very/long/session/directory/that/goes/on/and/on/"
                     "session_20260902_180701/pulse_module2_180833.h5")
    assert panel.path_label.text() == "HDF5: pulse_module2_180833.h5"
    assert panel.path_label.toolTip().endswith("pulse_module2_180833.h5")
    bar = panel.btn_start.parentWidget()
    flow = bar.layout()
    assert flow.heightForWidth(700) > flow.heightForWidth(1900)
    assert panel.minimumSizeHint().width() < 900
    panel.resize(1000, 600)
    qt_app.processEvents()
    assert panel.width() == 1000
