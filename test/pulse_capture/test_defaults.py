"""The defaults: trigger on the frequency basis where a calibration
allows it, stop the window a margin past below-threshold, and view in
hertz when it can be drawn."""
import numpy as np
import pytest

from rfmux.pulse_capture.capture_session import (
    PulseCaptureConfig, PulseCaptureSession)


def test_config_defaults():
    cfg = PulseCaptureConfig()
    assert cfg.trigger_basis == "df"
    assert cfg.save_to_end_confirmed is False


def test_session_defaults_match_the_config():
    s = PulseCaptureSession(channels=[1], sample_rate=1000.0)
    assert s.trigger_basis == "df"
    assert s.save_to_end_confirmed is False


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
