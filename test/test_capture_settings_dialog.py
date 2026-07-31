"""Offscreen tests for the Pulse Capture Settings dialog."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from rfmux.algorithms.measurement.pulse_capture_session import (  # noqa: E402
    PulseCaptureConfig,
)
from rfmux.tools.periscope.pulse_capture_settings_dialog import (  # noqa: E402
    PulseCaptureSettingsDialog,
)


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _ok(dlg):
    return dlg.buttons.button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok)


def test_derived_labels_follow_rate(qt_app):
    cfg = PulseCaptureConfig(min_pulse_ms=1.0)
    slow = PulseCaptureSettingsDialog(
        config=cfg, sample_rate=19073.486328125, mode="slow")
    assert "min pulse = 19 samples" in slow.derived_label.text()
    slow.close()

    fast = PulseCaptureSettingsDialog(
        config=cfg, sample_rate=1220703.125, mode="fast")
    assert "min pulse = 1,221 samples" in fast.derived_label.text() or \
        "min pulse = 1221 samples" in fast.derived_label.text()
    fast.close()


def test_ok_gated_on_errors(qt_app):
    dlg = PulseCaptureSettingsDialog(sample_rate=19073.486328125)
    assert _ok(dlg).isEnabled()
    dlg.end_spin.setValue(dlg.threshold_spin.value())  # end >= threshold
    assert not _ok(dlg).isEnabled()
    dlg.end_spin.setValue(1.5)
    assert _ok(dlg).isEnabled()
    dlg.close()


def test_roundtrip(qt_app):
    dlg = PulseCaptureSettingsDialog(sample_rate=19073.486328125)
    dlg.threshold_spin.setValue(8.0)
    dlg.min_pulse_spin.setValue(0.5)
    dlg.max_pulse_spin.setValue(100.0)
    dlg.margin_spin.setValue(0.2)
    dlg.pileup_check.setChecked(False)

    cfg = dlg.get_config()
    assert cfg.threshold_sigma == 8.0
    assert cfg.min_pulse_ms == 0.5
    assert cfg.max_pulse_ms == 100.0
    # Training is derived from the pulse length, not entered.
    assert cfg.noise_train_ms == 0.0
    assert cfg.noise_train_span_ms() == 100.0 * cfg.NOISE_TRAIN_PULSES
    assert "20×" in dlg.noise_label.text()
    assert cfg.margin_fraction == 0.2
    assert cfg.enable_pileup is False
    dlg.close()


def test_auto_baseline_toggle(qt_app):
    """Auto is the default; the fixed spinbox is only live when it is
    off, and the derived line says which one is in force."""
    dlg = PulseCaptureSettingsDialog(sample_rate=19073.486328125)
    assert dlg.baseline_auto_check.isChecked()
    assert not dlg.baseline_spin.isEnabled()
    assert dlg.get_config().baseline_track_auto is True
    assert "baseline measured at training" in dlg.derived_label.text()

    dlg.baseline_auto_check.setChecked(False)
    dlg.baseline_spin.setValue(2000.0)
    assert dlg.baseline_spin.isEnabled()
    cfg = dlg.get_config()
    assert cfg.baseline_track_auto is False
    assert cfg.baseline_track_ms == 2000.0
    assert "baseline EMA" in dlg.derived_label.text()
    dlg.close()


def test_max_pulse_is_a_primary_control_and_drives_training(qt_app):
    """Max pulse sits in the main form, and the derived training length
    tracks it — the ratio is what matters, not any absolute duration."""
    dlg = PulseCaptureSettingsDialog(sample_rate=19073.486328125)
    # Not hidden behind Advanced.
    assert not dlg.adv_box.isChecked()
    assert dlg.max_pulse_spin.isVisible() or not dlg.isVisible()

    dlg.max_pulse_spin.setValue(10.0)
    assert dlg.get_config().noise_train_span_ms() == 200.0
    first = dlg.noise_label.text()
    dlg.max_pulse_spin.setValue(40.0)
    assert dlg.get_config().noise_train_span_ms() == 800.0
    assert dlg.noise_label.text() != first, "readout did not follow"
    dlg.close()
