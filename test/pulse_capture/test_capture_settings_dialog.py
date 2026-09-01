"""Offscreen tests for the Pulse Capture Settings dialog."""


import pytest


pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from rfmux.pulse_capture.capture_session import (  # noqa: E402
    PulseCaptureConfig,
)
from rfmux.tools.periscope.pulse_capture_settings_dialog import (  # noqa: E402
    PulseCaptureSettingsDialog,
)



def _ok(dlg):
    return dlg.buttons.button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok)


def test_derived_labels_follow_rate(qt_app):
    cfg = PulseCaptureConfig(min_pulse_ms=1.0)
    slow = PulseCaptureSettingsDialog(
        config=cfg, sample_rate=19073.486328125, mode="slow")
    assert "min pulse 19 samples" in slow.pulse_derived_label.text()
    slow.close()

    fast = PulseCaptureSettingsDialog(
        config=cfg, sample_rate=1220703.125, mode="fast")
    assert "min pulse 1,221 samples" in fast.pulse_derived_label.text() or \
        "min pulse 1221 samples" in fast.pulse_derived_label.text()
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


def test_rolling_baseline_span_is_shown(qt_app):
    """No baseline controls left to get wrong — the window is the
    training span, so the dialog only reports it."""
    dlg = PulseCaptureSettingsDialog(sample_rate=19073.486328125)
    assert not hasattr(dlg, "baseline_spin")
    assert not hasattr(dlg, "baseline_auto_check")
    assert "baseline median" in dlg.pulse_derived_label.text()
    dlg.close()


def test_derived_readouts_split_by_driving_knob(qt_app):
    """The dialog shows WHAT each primary input drives: every time
    scale under max pulse, everything statistical under threshold σ —
    including the two new derived quantities (edge lookback, hard
    stop) and the edge amplitude floor."""
    dlg = PulseCaptureSettingsDialog(sample_rate=19073.486328125)
    pulse_txt = dlg.pulse_derived_label.text()
    sigma_txt = dlg.sigma_derived_label.text()
    for piece in ("ring buffer", "hard stop", "noise training",
                  "baseline median", "edge lookback"):
        assert piece in pulse_txt, piece
    for piece in ("confirmation", "accidentals", "edge jump",
                  "amplitude floor"):
        assert piece in sigma_txt, piece
    # The floor readout follows the threshold: 5σ → ≈7.1σ.
    assert "7.1σ" in sigma_txt
    dlg.threshold_spin.setValue(10.0)
    assert "14.1σ" in dlg.sigma_derived_label.text()

    # And the time scales follow max pulse: 250 ms → 300 ms hard stop,
    # 25 ms edge lookback.
    assert "300 ms" in pulse_txt
    assert "25 ms" in pulse_txt
    dlg.max_pulse_spin.setValue(500.0)
    assert "600 ms" in dlg.pulse_derived_label.text()
    assert "50 ms" in dlg.pulse_derived_label.text()
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


def test_trigger_basis_round_trips(qt_app):
    """The basis survives the dialog, and does not disturb the rest.

    It is the one capture setting that changes what gets detected rather
    than how much of it is kept, so a dialog that silently reset it to
    the default would be worse than not offering it.
    """
    from rfmux.pulse_capture import PulseCaptureConfig

    for basis in ("iq", "df"):
        src = PulseCaptureConfig(trigger_basis=basis, threshold_sigma=7.5,
                                 max_pulse_ms=123.0, enable_pileup=False)
        out = PulseCaptureSettingsDialog(
            config=src, sample_rate=596.0).get_config()
        assert out.trigger_basis == basis
        # The other knobs come back untouched.
        assert out.threshold_sigma == pytest.approx(7.5)
        assert out.max_pulse_ms == pytest.approx(123.0)
        assert out.enable_pileup is False


def test_end_floor_is_exposed_and_round_trips(qt_app):
    """The end-confirmation floor sits under Advanced with the rest of
    the end logic, reads back into the config, and the derived readout
    says what it is in time at this rate."""
    dlg = PulseCaptureSettingsDialog(sample_rate=596.0)
    assert dlg.min_end_spin.value() == 10
    assert "end floor 10 samples" in dlg.pulse_derived_label.text()
    assert "16.8 ms" in dlg.pulse_derived_label.text()
    dlg.min_end_spin.setValue(4)
    assert dlg.get_config().min_end_samples == 4
    assert "end floor 4 samples" in dlg.pulse_derived_label.text()
    assert "bucket" in dlg.min_end_spin.toolTip()
    dlg.close()
